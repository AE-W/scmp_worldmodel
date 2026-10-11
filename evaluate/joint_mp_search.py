"""Budgeted joint MP search using IRASim's complete noise-prediction objective.

Stages: smoke, calibration search, independent noise confirmation.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate.search_recipe import (DEFAULT_RECIPE,SearchRecipe,resolve_ladder,
    default_ladder,uniform_variants,runtime_environment,runtime_settings,configure_float_precision)

_runtime_env = runtime_environment(DEFAULT_RECIPE)
for _knob in set(os.environ)-set(_runtime_env):
    del os.environ[_knob]
os.environ.update(_runtime_env)
import torch
from scmp_kernels.mp import MPConfig
from scmp_kernels.mp.config import _bucket_index

from diffusion import create_mask_diffusion
from evaluate.search_io import digest, save, runtime_sources,software_versions,hardware_versions
from evaluate.runtime_audit import RuntimePatch
from evaluate.parallel_cases import CaseWorkers
from evaluate.eval_local_n_samples import (
    ANNOT_DIR, C_ACT_SCALER, GT_LATENT_DIR, SEQUENCE_LENGTH,
    build_args, compute_actions_for_slice, load_model, parse_skip,
)
from evaluate.joint_mp_space import budget_cost, uniform
from evaluate.upstream_mp_solver import DENSE_LADDER, UPSTREAM_COMMIT, bin_records, normalized_metric, policy, solve_global
from evaluate.upstream_mp_measure import capture_calls, measure
from evaluate.mixed_runtime_smoke import check_mixed_runtime
from evaluate.task_loss_refine import DEFAULT_BUDGET_TOL, canonical, key, refine, select_on_search, threshold_uniform, within_budget
from models.sc_integration import clear_skip_blocks, get_config, reconfigure, set_skip_blocks
from models.sc_integration import sc_linear
from models.sc_integration import sc_attention

SUFFIX = {'attn.qkv': 'qkv', 'attn.proj': 'proj', 'mlp.fc1': 'mlp_fc1', 'mlp.fc2': 'mlp_fc2'}
OPS = ('qkv', 'qk', 'av', 'proj', 'mlp_fc1', 'mlp_fc2')


def make_cases(keys, diffusion, device, seed, draws,expected_shape):
    """Common random numbers with stratified timesteps over the diffusion schedule."""
    cases = []
    for key in keys:
        eid = '_'.join(key.split('_')[:-2])
        start = int(key.split('_')[-1])
        ann = json.load(open(Path(ANNOT_DIR) / (eid + '.json')))
        path = Path(GT_LATENT_DIR) / (key + '.pt')
        latent = torch.load(path, map_location=device, weights_only=False).float().unsqueeze(0)
        if tuple(latent.shape) != expected_shape:
            raise ValueError(f'Unexpected input shape: {key} {latent.shape}')
        arm = np.asarray(ann['state'])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.asarray(ann['continuous_gripper_state'])[start:start + SEQUENCE_LENGTH]
        if len(arm) != SEQUENCE_LENGTH:
            raise ValueError(f'Incomplete actions: {key}')
        actions = torch.from_numpy(compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0).to(device)
        for draw in range(draws):
            value = int.from_bytes(hashlib.sha256(f'{seed}:{key}:{draw}'.encode()).digest()[:4], 'little')
            lo, hi = draw * diffusion.num_timesteps // draws, (draw + 1) * diffusion.num_timesteps // draws
            if hi<=lo:
                raise ValueError('Draw count exceeds diffusion timesteps')
            timestep = random.Random(value).randrange(lo, hi)
            t = torch.tensor([timestep], device=device, dtype=torch.long)
            eps = torch.randn(latent[:, 1:].shape, device=device,
                              generator=torch.Generator(device=device).manual_seed(value))
            future = diffusion.q_sample(latent[:, 1:], t, noise=eps)
            x_t = torch.cat([latent[:, :1], future], dim=1)
            assert torch.equal(x_t[:, :1], latent[:, :1])
            cases.append(dict(key=key, draw=draw, seed=value, timestep=timestep,
                              weight=(hi-lo)/diffusion.num_timesteps,
                              x_t=x_t, t=t, noise=eps, actions=actions, latent=latent,
                              latent_sha256=digest(path),
                              annotation_sha256=digest(Path(ANNOT_DIR)/(eid+'.json')),
                              actions_sha256=hashlib.sha256(actions.cpu().numpy().tobytes()).hexdigest(),
                              noise_sha256=hashlib.sha256(eps.cpu().numpy().tobytes()).hexdigest()))
    return cases


class Evaluator:
    def __init__(self, cli):
        self._budget_patch = None
        self._workers = None
        self.recipe = SearchRecipe(layer_buckets=cli.layer_buckets)
        self.ladder = cli.ladder
        configure_float_precision(torch)
        self.device = torch.device('cuda:0')
        self.args = build_args(cli.config, self.recipe.video_steps)
        if self.args.learn_sigma or self.args.mask_frame_num != 1:
            raise ValueError('This objective requires epsilon prediction and one clean context frame')
        if self.args.num_frames!=SEQUENCE_LENGTH:
            raise ValueError('Bridge evaluation requires the dataset sequence length')
        self.model = load_model(self.args, self.device).eval().requires_grad_(False)
        self.diffusion = create_mask_diffusion('', learn_sigma=False)
        if not 1<=cli.draws<=cli.confirmation_draws<=self.diffusion.num_timesteps:
            raise ValueError('Draw count exceeds the diffusion schedule')
        self.expected_shape = (1,self.args.num_frames,self.model.in_channels,*self.args.latent_size)
        reconfigure(self.args.attention_mode)
        clear_skip_blocks()
        self.skip = parse_skip(cli.skip)
        for op, blocks in self.skip.items():
            set_skip_blocks(op, blocks)
        self.cfg = get_config()
        self.flags = {op: getattr(self.cfg, 'enable_' + op) for op in OPS}
        self.modules, self.address = {}, {}
        scales = torch.load(cli.scales, map_location=self.device, weights_only=False)['scales']
        for name, mod in self.model.named_modules():
            match = re.search(r'blocks\.(\d+)\.', name)
            if not isinstance(mod, torch.nn.Linear) or not match:
                continue
            for suffix, op in SUFFIX.items():
                block = int(match.group(1))
                if name.endswith(suffix) and self.flags[op] and block not in self.skip.get(op, []):
                    if mod.in_features % self.recipe.chunk or name not in scales:
                        raise ValueError(f'Missing SQ or incomplete groups: {name}')
                    mod._sc_smooth_scales = scales[name].float()
                    self.modules[name] = mod
                    self.address[name] = (op, block)
        if not self.modules:
            raise RuntimeError('No active MP linears')
        self.buckets = {n:f'{op}:l{_bucket_index(block,len(self.model.blocks),self.recipe.layer_buckets)}'
                        for n,(op,block) in self.address.items()}
        self.metric_profile = None
        # Apply the fixed experiment precision and dispatch configuration.
        sc_linear._STEP_SCHEDULE = sc_linear._UNIFORM_STOC_LEN = None
        sc_linear._GRANULARITY = 'per_row'
        sc_linear._HALVE = sc_attention._HALVE = self.recipe.halve
        sc_linear._MP_FIXED_PREC = True
        sc_linear._SC_PREC = self.recipe.precision
        sc_linear._MP_GROUP_CHUNK_D = self.recipe.chunk
        self.shapes = {n: [] for n in self.modules}

    def fp(self):
        for op in OPS:
            setattr(self.cfg, 'enable_' + op, False)

    def install(self, candidate):
        if any('group_thresholds' in d for d in candidate['per_module_fractions'].values()):
            candidate = canonical(candidate)
        if set(candidate['per_module_fractions']) != set(self.modules):
            raise ValueError('Candidate must cover exactly all enabled MP modules')
        for op, enabled in self.flags.items():
            setattr(self.cfg, 'enable_' + op, enabled)
        sc_linear._MP_CONFIG = MPConfig(stoc_len_levels=candidate['stoc_len_levels'],
                                       level_fractions=candidate['level_fractions'])
        sc_linear._MP_PER_MODULE = {
            self.address[n]: dict(fractions=d.get('level_fractions'), invert=d.get('invert', False),
                                 group_thresholds=d.get('group_thresholds'))
            for n, d in candidate['per_module_fractions'].items()}
        self.current_candidate = candidate

    def forward(self, case):
        with torch.inference_mode():
            output = self.model(case['x_t'], case['t'], actions=case['actions'], mask_frame_num=1)
        if output.shape != case['latent'].shape or not torch.isfinite(output).all():
            raise RuntimeError('Invalid noise prediction')
        return output

    def profile(self, case):
        if self._budget_patch is not None:
            raise RuntimeError('Evaluator already has a deployment-shape profile')
        self.fp()
        hooks = []
        try:
            for name, mod in self.modules.items():
                def capture(_mod, inputs, _out, name=name):
                    x = inputs[0]
                    self.shapes[name].append((x.numel() // x.shape[-1], x.shape[-1], _mod.out_features))
                hooks.append(mod.register_forward_hook(capture))
            self.forward(case)
        finally:
            for hook in hooks:
                hook.remove()
        if any(not calls for calls in self.shapes.values()):
            raise RuntimeError('Incomplete deployment-shape profile')
        original = sc_linear._mp_linear_forward
        self.kernel_dispatch = original
        reverse = {address: name for name, address in self.address.items()}
        self.actual = {name: [] for name in self.modules}

        def audited(*args, **kwargs):
            name = reverse[(kwargs['op'], kwargs['block_idx'])]
            x, w = args[:2]
            self.actual[name].append((x.shape[0], x.shape[1], w.shape[0]))
            d = self.current_candidate['per_module_fractions'][name]
            if self.metric_profile is not None:
                m = normalized_metric(x,self.recipe.chunk)
                # Profile the candidate's OWN full activation population.
                indices = torch.round(m.reshape(-1)*(self.profile_bins-1)).long()
                hist = torch.bincount(indices,minlength=self.profile_bins).double()
                self.metric_profile[self.buckets[name]] += hist.cpu().numpy()*w.shape[0]*self.recipe.chunk*self.case_weight
            if 'group_thresholds' in d:
                m = normalized_metric(x,self.recipe.chunk)
                cuts = torch.as_tensor(d['group_thresholds'],device=x.device,dtype=m.dtype)
                rung = torch.bucketize(m.contiguous(),cuts)
                lens = torch.as_tensor(self.current_candidate['stoc_len_levels'],device=x.device)
                self.actual_cost += float(lens[rung].sum()) * w.shape[0]*self.recipe.chunk
                self.actual_units += rung.numel()*w.shape[0]*self.recipe.chunk
                self.actual_level_units += torch.bincount(rung.reshape(-1).long(),minlength=len(lens)).cpu().numpy()*w.shape[0]*self.recipe.chunk
            else:
                from evaluate.group_mp_policy import group_counts
                counts = group_counts(x.shape[0]*(x.shape[1]//self.recipe.chunk), d['level_fractions'])
                self.actual_cost += sum(a*b for a,b in zip(counts,self.current_candidate['stoc_len_levels']))*w.shape[0]*self.recipe.chunk
                self.actual_units += sum(counts)*w.shape[0]*self.recipe.chunk
            return original(*args, **kwargs)

        self._budget_patch = RuntimePatch(sc_linear,'_mp_linear_forward',audited)
        self._budget_patch.__enter__()

    def close(self):
        if self._workers is not None:
            self._workers.close()
            self._workers = None
        if self._budget_patch is not None:
            self._budget_patch.close()
            self._budget_patch = None

    def score(self, candidate, cases, profile=False, profile_bins=257, serial=False):
        if self._workers is not None and not serial:
            return self._workers.score(candidate,cases,profile,profile_bins)
        self.install(candidate)
        self.profile_bins = profile_bins
        self.metric_profile = {b:np.zeros(profile_bins) for b in set(self.buckets.values())} if profile else None
        losses, costs = [], []
        level_weights = np.zeros(len(candidate['stoc_len_levels']))
        total_weight = sum(c['weight'] for c in cases)
        for case in cases:
            self.case_weight = case['weight']/total_weight
            torch.cuda.manual_seed_all(case['seed'])
            self.actual = {name: [] for name in self.modules}
            self.actual_cost = self.actual_units = 0
            self.actual_level_units = np.zeros(len(candidate['stoc_len_levels']))
            pred = self.forward(case)
            if any(Counter(self.actual[n]) != Counter(self.shapes[n]) for n in self.modules):
                raise RuntimeError('Runtime call shapes differ from budget accounting')
            # IRASim supervision: true sampled noise on the future frames.
            loss = float((pred[:, 1:] - case['noise']).square().mean())
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite objective')
            losses.append(loss)
            costs.append(self.actual_cost/self.actual_units)
            if 'group_thresholds' in next(iter(candidate['per_module_fractions'].values())):
                level_weights += self.actual_level_units/self.actual_units*self.case_weight
        weights = [c['weight'] for c in cases]
        result = dict(mean=sum(a*b for a,b in zip(losses, weights))/sum(weights), per_case=losses,
                      avg_cycles=sum(a*b for a,b in zip(costs,weights))/sum(weights), per_case_cycles=costs)
        if profile:
            result['metric_profile'] = {b:h.tolist() for b,h in self.metric_profile.items()}
            result['level_weights'] = level_weights.tolist()
        self.metric_profile = None
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['smoke', 'search', 'noise-validate'])
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='configs/evaluation/bridge/frame_ada_sc_full.yaml')
    p.add_argument('--calibration_keys', default='results/eval_calibration_keys.json')
    p.add_argument('--holdout_keys', default='results/eval_holdout_keys.json')
    p.add_argument('--test_keys', default='results/diverse_10.json')
    p.add_argument('--scales', default='results/smoothquant_scales.pt')
    p.add_argument('--skip', required=True)
    p.add_argument('--budget', type=int, default=96)
    p.add_argument('--budget_tol','--budget-tol',type=float,default=DEFAULT_BUDGET_TOL,
                   help='Absolute mean-cycle tolerance around target (upstream: 0.35)')
    p.add_argument('--bins', type=int, default=2048)
    p.add_argument('--rows_per_call', type=int, default=64)
    p.add_argument('--price_iterations', type=int, default=80)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--ladder_rounds',type=int,default=2)
    p.add_argument('--threshold_rounds',type=int,default=1)
    p.add_argument('--floor_step',type=int,default=4)
    p.add_argument('--budget_corrections',type=int,default=2)
    p.add_argument('--min_level',type=int,default=1)
    p.add_argument('--max_level',type=int,default=128)
    p.add_argument('--ladder',default=None,help='Exact ascending initial lengths; default grid is clipped to bounds and spans the budget')
    p.add_argument('--layer_buckets',type=int,default=DEFAULT_RECIPE.layer_buckets)
    p.add_argument('--refine_start',choices=['fis','raw','parent'],default='fis')
    p.add_argument('--parent_policy',default='',help='Numeric per-group policy used as the refinement parent')
    p.add_argument('--threshold_mass_step',type=float,default=.02)
    p.add_argument('--profile_bins',type=int,default=257)
    p.add_argument('--min_loss_improvement',type=float,default=0.0)
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--confirmation_draws',type=int,default=6,
                   help='Number of stratified draws per confirmation episode')
    p.add_argument('--expand_at',default='',help='Optional re-linearization trajectory policy JSON, like upstream --expand-at')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--video_seed',type=int,default=None,help='Paired generation seed; defaults to --seed')
    cli = p.parse_args()
    cli.ladder = (default_ladder(DENSE_LADDER,cli.min_level,cli.max_level,cli.budget) if cli.ladder is None
                  else resolve_ladder([int(v) for v in cli.ladder.split(',')],cli.min_level,cli.max_level,cli.budget))
    cli.video_seed = cli.seed if cli.video_seed is None else cli.video_seed
    if not 1 <= cli.draws <= cli.confirmation_draws or cli.layer_buckets<1:
        raise ValueError('Invalid budget or case count')
    if (cli.refine_start=='parent') != bool(cli.parent_policy):
        raise ValueError('Specify --parent_policy exactly when --refine_start=parent')
    within_budget(cli.budget,cli.budget,cli.budget_tol)  # validate tolerance
    if min(cli.bins,cli.rows_per_call,cli.price_iterations,cli.workers) < 1:
        raise ValueError('Invalid upstream solver parameters')
    if (min(cli.ladder_rounds,cli.threshold_rounds,cli.budget_corrections)<0 or cli.floor_step<1 or
            not 1<=cli.min_level<=cli.max_level<=128 or cli.profile_bins<2 or
            not 0<cli.threshold_mass_step<.25 or not math.isfinite(cli.min_loss_improvement) or cli.min_loss_improvement<0):
        raise ValueError('Invalid task-loss refinement parameters')
    splits = [json.load(open(f)) for f in (cli.calibration_keys, cli.holdout_keys, cli.test_keys)]
    if any(not keys or len(set(keys)) != len(keys) for keys in splits):
        raise ValueError('Empty or duplicate split keys')
    episodes = [{'_'.join(k.split('_')[:-2]) for k in keys} for keys in splits]
    if any(episodes[i] & episodes[j] for i in range(3) for j in range(i)):
        raise ValueError('Episode leakage between splits')
    out = Path(cli.out)
    if cli.stage == 'search' and (out / 'search_state.json').exists():
        raise ValueError('Use a fresh output directory; refusing to overwrite a search')
    evaluator = Evaluator(cli)
    try:
        return run_stage(cli,splits,out,evaluator)
    finally:
        evaluator.close()


def run_stage(cli,splits,out,evaluator):
    calibration = make_cases(splits[0], evaluator.diffusion, evaluator.device, cli.seed, cli.draws,evaluator.expected_shape)
    evaluator.profile(calibration[0])
    if cli.workers>1:
        evaluator._workers = CaseWorkers(cli,splits)
    base = uniform(evaluator.modules, cli.budget)
    provenance = dict(script_sha256=digest(__file__),
                      space_sha256=digest('evaluate/joint_mp_space.py'),
                      solver_sha256=digest('evaluate/upstream_mp_solver.py'),
                      measure_sha256=digest('evaluate/upstream_mp_measure.py'),
                      refine_sha256=digest('evaluate/task_loss_refine.py'),
                      joint_core_sha256=digest('evaluate/upstream_joint_core.py'),
                      io_sha256=digest('evaluate/search_io.py'),
                      fraction_accounting_sha256=digest('evaluate/group_mp_policy.py'),
                      recipe_sha256=digest('evaluate/search_recipe.py'),
                      recipe=evaluator.recipe.to_dict(),
                      runtime_settings=runtime_settings(evaluator.recipe),
                      runtime_sources=runtime_sources(),
                      software_versions=software_versions(),
                      hardware_versions=hardware_versions(torch,evaluator.device),
                      config_sha256=digest(cli.config),
                      split_sha256=[digest(f) for f in (cli.calibration_keys,cli.holdout_keys,cli.test_keys)],
                      runtime_sha256=digest('models/sc_integration/sc_linear.py'),
                      upstream_reference=UPSTREAM_COMMIT,
                      kernel_commit=subprocess.check_output(['git','-C','kernels','rev-parse','HEAD'], text=True).strip(),
                      code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                      checkpoint_path=str(evaluator.args.evaluate_checkpoint),
                      checkpoint_size=Path(evaluator.args.evaluate_checkpoint).stat().st_size,
                      checkpoint_mtime_ns=Path(evaluator.args.evaluate_checkpoint).stat().st_mtime_ns,
                      checkpoint_sha256=digest(evaluator.args.evaluate_checkpoint))
    if cli.expand_at:
        provenance['expand_at_sha256'] = digest(cli.expand_at)
    if cli.parent_policy:
        provenance['parent_policy_sha256'] = digest(cli.parent_policy)
    manifests = [{k: c[k] for k in ('key', 'draw', 'seed', 'timestep', 'weight', 'latent_sha256', 'annotation_sha256', 'actions_sha256', 'noise_sha256')}
                 for c in calibration]
    if cli.stage == 'smoke':
        mixed_check = check_mixed_runtime(evaluator.recipe,evaluator.device,evaluator.kernel_dispatch)
        evaluator.fp()
        case = calibration[0]
        manual = float((evaluator.forward(case)[:, 1:] - case['noise']).square().mean())
        # Replay the native training loss's implicit noise draw.
        torch.cuda.manual_seed_all(case['seed'])
        with torch.inference_mode():
            native = float(evaluator.diffusion.training_losses(evaluator.model, case['latent'], case['t'],
                           dict(actions=case['actions'], mask_frame_num=1))['loss'].mean())
        if not math.isclose(manual, native, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError(f'Original objective mismatch: {manual} vs {native}')
        a = evaluator.score(base, calibration,serial=True)
        parallel_baseline = evaluator.score(base,calibration)
        if any(not math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-7)
               for field in ('per_case','per_case_cycles') for x,y in zip(a[field],parallel_baseline[field])):
            raise RuntimeError('Parallel case evaluation differs from serial evaluation')
        a = evaluator.score(base, calibration[:1])
        padded,uniform_threshold_policy = uniform_variants(list(evaluator.modules),cli.budget,cli.ladder,cli.min_level,cli.max_level)
        b = evaluator.score(padded, calibration[:1])
        c = evaluator.score(uniform_threshold_policy,calibration[:1])
        if a != b or a != c or budget_cost(base, evaluator.shapes,evaluator.recipe.chunk) != cli.budget:
            raise RuntimeError('Uniform candidate differs from uniform baseline')
        evaluator.install(base)
        expected = evaluator.forward(case)
        captured = []
        with torch.enable_grad(), capture_calls(evaluator,captured,case['seed'],cli.rows_per_call):
            xt = case['x_t'].detach().clone().requires_grad_(True)
            actual = evaluator.model(xt,case['t'],actions=case['actions'],mask_frame_num=1)
            if not torch.equal(actual.detach(),expected):
                raise RuntimeError('Straight-through wrapper changed SC forward values')
            probe = torch.randn(actual[:,1:].shape,device=actual.device,
                                generator=torch.Generator(device=actual.device).manual_seed(case['seed']+1))
            torch.autograd.grad((actual[:,1:]*probe).sum(),xt)
        if ({r['name'] for r in captured} != set(evaluator.modules) or
                any('fisher' not in r or not torch.isfinite(r['fisher']).all() for r in captured)):
            raise RuntimeError('Invalid straight-through sensitivity coverage')
        save(out / 'smoke.json', dict(original_fp_loss=native, explicit_loss=manual,
                                    uniform_identity=True, ste_identity=True,
                                    mixed_runtime=mixed_check,
                                    parallel_identity=True,workers=cli.workers,
                                    sensitivity_modules=len(evaluator.modules),budget=evaluator.shapes,
                                    protocol=vars(cli),provenance=provenance,
                                    scales_sha256=digest(cli.scales),manifest=manifests))
        print('Objective, uniform, mixed dispatch and sensitivity smoke passed', flush=True)
        return
    if cli.stage == 'noise-validate':
        state = json.load(open(out / 'search_state.json'))
        if state['protocol'] != (vars(cli) | {'stage': 'search'}):
            raise ValueError('Validation arguments differ from recorded search protocol')
        if not state.get('complete') or len(state.get('refinement',[]))<1:
            raise ValueError('Noise confirmation requires a completed calibration search')
        if state['scales_sha256'] != digest(cli.scales) or state['shapes'] != json.loads(json.dumps(evaluator.shapes)):
            raise ValueError('Budget profile or SmoothQuant provenance changed')
        if state['provenance'] != provenance or state['manifest'] != manifests:
            raise ValueError('Model, implementation, calibration data or noise samples changed')
        holdout = make_cases(splits[1], evaluator.diffusion, evaluator.device, cli.seed, cli.confirmation_draws,evaluator.expected_shape)
        # Confirm the frozen search winner and uniform reference.
        selected_trial = state['trials'][state['selected_index']]
        shortlist = [state['trials'][0],selected_trial]
        records = [dict(candidate=r['candidate'], holdout=evaluator.score(r['candidate'], holdout)) for r in shortlist]
        selected = records[-1]
        save(out / 'noise_validation.json', dict(records=records, selected=selected,
             selection_split='calibration',confirmation_only=True,
             budget_ok=within_budget(selected['holdout']['avg_cycles'],cli.budget,cli.budget_tol),
             manifest=[{k:c[k] for k in manifests[0]} for c in holdout]))
        print('Independent noise confirmation complete.', flush=True)
        return
    smoke = json.load(open(out/'smoke.json'))
    if not all(smoke.get('mixed_runtime',{}).get(field) for field in
               ('boundary_identity','chunk_product_identity')):
        raise ValueError('Mixed runtime smoke required before calibration')
    if (smoke['provenance'] != provenance or smoke['manifest'] != manifests or
            smoke['scales_sha256'] != digest(cli.scales) or
            smoke['protocol'] != (vars(cli)|{'stage':'smoke'})):
        raise ValueError('Matching GPU smoke required before starting calibration')
    trials,refinement,selection_indices = [],[],[]
    protocol = vars(cli)
    evaluator.actual_cost = evaluator.actual_units = 0
    trajectory = canonical(json.load(open(cli.expand_at)),cli.min_level,cli.max_level) if cli.expand_at else base
    measurements,buckets = (evaluator._workers.measure(calibration,trajectory,cli.rows_per_call)
        if evaluator._workers is not None else measure(evaluator,calibration,trajectory,cli.rows_per_call))
    binned = {k:bin_records(v,cli.bins) for k,v in measurements.items()}
    proposals = [('uniform',base,{})]
    for currency in ('fis','raw'):
        cuts,diag = solve_global(binned,currency,cli.budget,cli.price_iterations,cli.ladder)
        proposals.append((currency,policy(buckets,cuts,cli.ladder),diag))
    if cli.parent_policy:
        proposals.append(('parent',canonical(json.load(open(cli.parent_policy)),cli.min_level,cli.max_level),{}))
    cache = {}

    def persist(complete=False):
        indices = sorted(set(selection_indices))
        retained = [trials[i] for i in indices]
        best = select_on_search(retained)
        state = dict(protocol=protocol,shapes=evaluator.shapes,manifest=manifests,
            provenance=provenance,scales_sha256=digest(cli.scales),trials=trials,
            refinement=refinement,selection_indices=indices,complete=complete)
        if complete:
            state['selected_index'] = best['index']
            save(out / 'search_policy.json',best['candidate'])
        save(out / 'search_state.json',state)

    def score_policy(candidate):
        k = key(candidate)
        if k not in cache:
            cache[k] = evaluator.score(candidate,calibration,profile=True,profile_bins=cli.profile_bins)
        return cache[k]

    for index,(currency,candidate,diag) in enumerate(proposals):
        # Use a one-rung threshold policy for the uniform reference.
        candidate = threshold_uniform(evaluator.modules,cli.budget) if currency=='uniform' else canonical(candidate,cli.min_level,cli.max_level)
        score = score_policy(candidate)
        feasible = within_budget(score['avg_cycles'],cli.budget,cli.budget_tol)
        trials.append(dict(index=index, currency=currency, candidate=candidate, solver=diag,
                           avg_cycles=score['avg_cycles'],budget_error=score['avg_cycles']-cli.budget,
                           feasible=feasible,calibration=score))
        if currency=='uniform':
            selection_indices.append(index)
        persist()
        print(f"candidate {currency}: cycles={score['avg_cycles']:.6f} "
              f"feasible={feasible} noise_loss={score['mean']:.8f}",flush=True)
    for initial in [next(r for r in trials if r['currency']==cli.refine_start)]:
        chain = initial['currency']
        def record(candidate,score,stage,iteration):
            index = len(trials)
            trials.append(dict(index=index,currency=chain+'_refine',stage=stage,round=iteration,candidate=candidate,
                               calibration=score,avg_cycles=score['avg_cycles'],
                               budget_error=score['avg_cycles']-cli.budget,
                               feasible=within_budget(score['avg_cycles'],cli.budget,cli.budget_tol)))
            persist()
            print(f"refine {chain} round {iteration+1}: K={len(candidate['stoc_len_levels'])} "
                  f"cycles={score['avg_cycles']:.6f} loss={score['mean']:.8f}",flush=True)

        final,final_score,history = refine(initial['candidate'],initial['calibration'],score_policy,evaluator.buckets,
               budget=cli.budget,ladder_rounds=cli.ladder_rounds,threshold_rounds=cli.threshold_rounds,
               floor_step=cli.floor_step,mass_step=cli.threshold_mass_step,budget_corrections=cli.budget_corrections,
               min_level=cli.min_level,max_level=cli.max_level,min_improvement=cli.min_loss_improvement,
               on_trial=record,budget_tol=cli.budget_tol)
        final_index = next(r['index'] for r in trials if key(r['candidate'])==key(final))
        selection_indices.append(final_index)
        refinement.append(dict(start=chain,final_index=final_index,history=history,
                               status='ok' if within_budget(final_score['avg_cycles'],cli.budget,cli.budget_tol) else 'budget_projection_failed',
                               loss_before=initial['calibration']['mean'],loss_after=final_score['mean']))
        persist()
    persist(complete=True)
    print('Calibration search complete; search policy saved.', flush=True)


if __name__ == '__main__':
    main()
