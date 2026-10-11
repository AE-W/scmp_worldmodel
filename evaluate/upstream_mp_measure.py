"""Measure upstream-style per-group curves and Gaussian MC Fisher for IRASim.

No parameter updates. SC forward values are unchanged; FP straight-through
derivatives propagate through linear/QK/AV only during sensitivity capture.
"""
from contextlib import contextmanager
import hashlib
import math

import numpy as np
import torch
from scmp_kernels import sc_matmul
from scmp_kernels.mp.config import _bucket_index
import models.sc_integration as integration
from models.sc_integration import sc_linear, sc_mlp
from evaluate.upstream_mp_solver import normalized_metric


@contextmanager
def capture_calls(evaluator, captured, seed, rows):
    original_lin = integration.sc_linear_forward
    original_qk, original_av = integration.sc_qk_matmul, integration.sc_av_matmul
    original_mlp = sc_mlp.sc_linear_forward
    reverse = {id(mod): name for name,mod in evaluator.modules.items()}

    def linear(x, mod, *args, **kw):
        with torch.no_grad():
            value = original_lin(x, mod, *args, **kw)
        reference = torch.nn.functional.linear(x, mod.weight, mod.bias)
        y = value + (reference-reference.detach())
        name = reverse.get(id(mod))
        if name is not None:
            if not y.requires_grad:
                raise RuntimeError(f'Missing sensitivity graph at {name}')
            flat = x.detach().reshape(-1, x.shape[-1]).float()
            call = sum(r['name'] == name for r in captured)
            salt = int.from_bytes(hashlib.sha256(f'{seed}:{name}:{call}'.encode()).digest()[:4], 'little')
            idx = torch.randperm(len(flat), generator=torch.Generator().manual_seed(salt))[:min(rows,len(flat))].to(x.device)
            metric = normalized_metric(flat,evaluator.recipe.chunk)
            rec = dict(name=name, x=flat[idx].clone(), rows=idx,
                       metric=metric[idx].cpu().numpy().reshape(-1),
                       population=len(flat))
            captured.append(rec)
            def gradient(g):
                rec['fisher'] = g.detach().reshape(-1, g.shape[-1])[idx].float().square()
            y.register_hook(gradient)
        return y

    def qk(a,b,*args,**kw):
        with torch.no_grad():
            value = original_qk(a,b,*args,**kw)
        ref = a @ b.transpose(-1,-2)
        return value + (ref-ref.detach())

    def av(a,b,*args,**kw):
        with torch.no_grad():
            value = original_av(a,b,*args,**kw)
        ref = a @ b
        return value + (ref-ref.detach())

    integration.sc_linear_forward = sc_mlp.sc_linear_forward = linear
    integration.sc_qk_matmul, integration.sc_av_matmul = qk, av
    try:
        yield
    finally:
        integration.sc_linear_forward = original_lin
        sc_mlp.sc_linear_forward = original_mlp
        integration.sc_qk_matmul, integration.sc_av_matmul = original_qk, original_av


def measure(evaluator, cases, baseline, rows=64, total_weight=None):
    records = {}
    total_blocks = len(evaluator.model.blocks)
    buckets = {n: f'{op}:l{_bucket_index(block,total_blocks,evaluator.recipe.layer_buckets)}'
               for n,(op,block) in evaluator.address.items()}
    evaluator.install(baseline)
    total_weight = sum(c['weight'] for c in cases) if total_weight is None else total_weight
    for case_index, case in enumerate(cases):
        torch.cuda.manual_seed_all(case['seed'])
        captured = []
        evaluator.actual = {n: [] for n in evaluator.modules}
        evaluator.actual_cost = evaluator.actual_units = 0
        evaluator.actual_level_units = np.zeros(len(baseline['stoc_len_levels']))
        with torch.enable_grad(), capture_calls(evaluator, captured, case['seed'], rows):
            xt = case['x_t'].detach().clone().requires_grad_(True)
            pred = evaluator.model(xt, case['t'], actions=case['actions'], mask_frame_num=1)
            # MC Fisher for unit-variance Gaussian epsilon likelihood: sample
            # pseudo-target from N(pred, I). Equivalent random output VJP.
            z = torch.randn(pred[:,1:].shape, device=pred.device,
                            generator=torch.Generator(device=pred.device).manual_seed(case['seed']+1))
            loss = (pred[:,1:] * z).sum() / math.sqrt(pred[:,1:].numel())
            torch.autograd.grad(loss, xt)
        del pred, xt, loss
        if {r['name'] for r in captured} != set(evaluator.modules):
            raise RuntimeError('Sensitivity capture did not cover all modules')
        for rec in captured:
            if 'fisher' not in rec or not torch.isfinite(rec['fisher']).all():
                raise RuntimeError(f"Missing or invalid gradient at {rec['name']}")
            mod = evaluator.modules[rec['name']]
            x, w = rec.pop('x'), mod.weight.detach().float()
            fisher = rec.pop('fisher')
            R, D = x.shape
            chunk = evaluator.recipe.chunk
            chunks = D//chunk
            raw = torch.empty(R,chunks,len(evaluator.ladder), device=x.device)
            fis = torch.empty_like(raw)
            smooth = mod._sc_smooth_scales
            with torch.no_grad():
                for c in range(chunks):
                    sl = slice(c*chunk,(c+1)*chunk)
                    xc,wc = x[:,sl].contiguous(),w[:,sl].contiguous()
                    ref = xc @ wc.t()
                    for j,L in enumerate(evaluator.ladder):
                        y = sc_matmul(xc,wc, granularity='per_row', mode='bipolar',
                                      sc_prec=evaluator.recipe.precision, stoc_len=L, halve_bipolar_stoc_len=evaluator.recipe.halve,
                                      smooth_scales=smooth[sl].contiguous())
                        d2 = (y-ref).square()
                        raw[:,c,j] = d2.sum(-1)
                        fis[:,c,j] = (d2*fisher).sum(-1)
            weight = case['weight']/total_weight
            expansion = rec['population']/R * weight
            rec['sampled_rows'] = R
            rec.pop('rows')
            rec['raw'] = raw.cpu().numpy().reshape(-1,len(evaluator.ladder))*expansion
            rec['fis'] = fis.cpu().numpy().reshape(-1,len(evaluator.ladder))*expansion
            rec['mac_weight'] = chunk*w.shape[0]*weight
            if not all(np.isfinite(rec[k]).all() for k in ('raw','fis','metric')):
                raise RuntimeError('Nonfinite measured curve')
            records.setdefault(buckets[rec['name']],[]).append(rec)
            del raw,fis,x,fisher
        print(f'Group curves and MC Fisher: case {case_index+1}/{len(cases)}',flush=True)
    return records,buckets
