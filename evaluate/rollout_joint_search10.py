"""Joint ladder/threshold refinement accepted by paired PNDM latent MSE."""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

_E = _PIPE = _CASES = None


def initialize(arguments, keys):
    global _E, _PIPE, _CASES
    from evaluate.search_recipe import SearchRecipe, runtime_environment
    env = runtime_environment(SearchRecipe(layer_buckets=arguments['layer_buckets']))
    os.environ.clear()
    os.environ.update(env)
    from evaluate.joint_mp_search import Evaluator, make_cases
    from evaluate.eval_local_n_samples import make_pipe
    from diffusers.models import AutoencoderKL
    import torch
    cli = SimpleNamespace(**arguments)
    _E = Evaluator(cli)
    atexit.register(_E.close)
    cases = make_cases(keys,_E.diffusion,_E.device,cli.seed,1,_E.expected_shape)
    _CASES = {case['key']:case for case in cases}
    _E.profile(cases[0])
    vae = AutoencoderKL.from_pretrained(_E.args.vae_model_path,subfolder='vae').eval()
    _PIPE = make_pipe(_E.args,vae,_E.model,_E.recipe.video_scheduler)
    _PIPE.decode_latents = lambda latent: torch.empty(0)


def execute(task):
    candidate, identity, seed, bins = task
    import numpy as np
    import torch
    from evaluate.eval_local_n_samples import gen_one
    case = _CASES[identity]
    _E.install(candidate)
    _E.profile_bins = bins
    _E.metric_profile = {b:np.zeros(bins) for b in set(_E.buckets.values())}
    _E.actual = {name:[] for name in _E.modules}
    _E.actual_cost = _E.actual_units = 0
    _E.actual_level_units = np.zeros(len(candidate['stoc_len_levels']))
    _E.case_weight = 1.0
    os.environ['EVAL_PAIR_SEED'] = str(seed)
    _, latent = gen_one(_PIPE,case['latent'][:,:1],case['actions'],_E.args,_E.device)
    if latent.shape != case['latent'].shape:
        raise RuntimeError('Generated latent shape mismatch')
    loss = float((latent[:,1:]-case['latent'][:,1:]).square().mean())
    if not np.isfinite(loss) or _E.actual_units<=0:
        raise RuntimeError('Invalid rollout loss or missing budget trace')
    return dict(mean=loss,cost_units=_E.actual_cost,macs=_E.actual_units,
        buckets=_E.buckets,
        level_units=_E.actual_level_units.tolist(),
        metric_profile={b:h.tolist() for b,h in _E.metric_profile.items()})


def merge(results):
    import numpy as np
    if not results:
        raise ValueError('Empty rollout evaluation')
    macs = sum(r['macs'] for r in results)
    return dict(mean=sum(r['mean'] for r in results)/len(results),
        buckets=results[0]['buckets'],
        per_case=[r['mean'] for r in results],
        avg_cycles=sum(r['cost_units'] for r in results)/macs,
        level_weights=(sum(np.asarray(r['level_units']) for r in results)/macs).tolist(),
        metric_profile={b:sum(np.asarray(r['metric_profile'][b])/len(results)
                            for r in results).tolist() for b in results[0]['metric_profile']})


def main():
    from evaluate.search_io import save,digest,runtime_sources,software_versions,file_manifest
    from evaluate.task_loss_refine import canonical,key,refine,within_budget
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--confirmation-keys',help='Explicit frozen confirmation trajectory list')
    p.add_argument('--smoke-reference',help='Prior paired uniform rollout result for the identical keys and seed')
    cli=p.parse_args()
    if cli.workers<1:
        raise ValueError('Positive worker count required')
    source=Path(cli.source).resolve()
    state=json.load(open(source/'search_state.json'))
    args=state['protocol']
    out=Path(cli.out).resolve()
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    keys=json.load(open(args['calibration_keys']))
    confirmation_path=cli.confirmation_keys or args['holdout_keys']
    holdout=json.load(open(confirmation_path))
    episodes=lambda split:{'_'.join(k.split('_')[:-2]) for k in split}
    if not keys or not holdout or len(set(keys))!=len(keys) or len(set(holdout))!=len(holdout):
        raise ValueError('Empty or duplicated search/confirmation keys')
    if episodes(keys)&episodes(holdout):
        raise ValueError('Search and confirmation overlap')
    for path,expected in ((args['config'],state['provenance']['config_sha256']),
        (state['provenance']['checkpoint_path'],state['provenance']['checkpoint_sha256']),
        (args['scales'],state['scales_sha256'])):
        if digest(path)!=expected:
            raise ValueError('Input changed: '+str(path))
    save(out/'protocol.json',dict(objective='paired PNDM-50 final future-frame latent MSE',
        search_keys=keys,confirmation_keys=holdout,seed=args['video_seed'],budget=args['budget'],
        tolerance=args['budget_tol'],source=digest(source/'search_state.json'),
        runtime=runtime_sources(),search_script_sha256=digest(__file__),software=software_versions(),
        inputs=file_manifest([args['calibration_keys'],confirmation_path,args['scales']]),
        scope='exploratory; existing confirmation data previously inspected'))
    records,cache=[],{}
    def persist(complete=False,selected=None):
        save(out/'state.json',dict(complete=complete,trials=records,selected=selected))
    with ProcessPoolExecutor(max_workers=cli.workers,mp_context=multiprocessing.get_context('spawn'),
         initializer=initialize,initargs=(args,keys+holdout)) as pool:
        def score(candidate,split=keys):
            token=(key(candidate),tuple(split))
            if token not in cache:
                tasks=[(candidate,k,args['video_seed'],args['profile_bins']) for k in split]
                cache[token]=merge(list(pool.map(execute,tasks)))
            return cache[token]
        def record(candidate,result,stage,iteration):
            records.append(dict(candidate=candidate,score=result,stage=stage,iteration=iteration,
                feasible=within_budget(result['avg_cycles'],args['budget'],args['budget_tol'])))
            persist()
            print(stage,iteration,'L2',result['mean'],'cycles',result['avg_cycles'],flush=True)
        initial=[]
        for trial in state['trials'][:3]:
            candidate=canonical(trial['candidate'])
            result=score(candidate)
            if trial['currency']=='uniform' and cli.smoke_reference:
                reference=json.load(open(cli.smoke_reference))
                if 'means' in reference:
                    if reference['keys']!=keys or reference['seed']!=args['video_seed']:
                        raise ValueError('Uniform reference keys or seed differ')
                    expected=reference['means']['uniform']['latent_l2']
                    cycles=reference['avg_cycles']['uniform']
                    tolerance=5.1e-5  # Historical per-trajectory metrics were rounded to 4 decimals.
                else:
                    expected,cycles,tolerance=reference['mean'],reference['cycles'],1e-6
                if abs(result['mean']-expected)>tolerance or abs(result['avg_cycles']-cycles)>1e-6:
                    raise RuntimeError('Uniform rollout parity smoke failed')
                save(out/'smoke.json',dict(uniform_rollout_identity=True,mean=result['mean'],
                    avg_cycles=result['avg_cycles'],reference_sha256=digest(cli.smoke_reference)))
            record(candidate,result,trial['currency'],0)
            initial.append((candidate,result))
        feasible=[pair for pair in initial if within_budget(pair[1]['avg_cycles'],args['budget'],args['budget_tol'])]
        uniform=initial[0]
        if not within_budget(uniform[1]['avg_cycles'],args['budget'],args['budget_tol']):
            raise RuntimeError('Uniform budget smoke failed')
        # The upstream refinement repairs measured budget before quality moves.
        mixed=initial[1:]
        parent=min(mixed or [uniform],key=lambda pair:pair[1]['mean'])
        buckets=parent[1]['buckets']
        final,result,history=refine(parent[0],parent[1],score,buckets,budget=args['budget'],
            ladder_rounds=args['ladder_rounds'],threshold_rounds=args['threshold_rounds'],
            floor_step=args['floor_step'],mass_step=args['threshold_mass_step'],
            budget_corrections=args['budget_corrections'],min_level=args['min_level'],max_level=args['max_level'],
            budget_tol=args['budget_tol'],on_trial=record)
        if within_budget(result['avg_cycles'],args['budget'],args['budget_tol']):
            feasible.append((final,result))
        winner=min(feasible,key=lambda pair:(pair[1]['mean'],len(pair[0]['stoc_len_levels'])))
        assert winner[1]['mean']<=uniform[1]['mean']
        save(out/'policy.json',winner[0])
        persist(True,dict(candidate=winner[0],score=winner[1],history=history))
        save(out/'confirmation.json',dict(uniform=score(uniform[0],holdout),
            selected=score(winner[0],holdout),rule='frozen policy; no confirmation re-selection'))


if __name__=='__main__':
    main()
