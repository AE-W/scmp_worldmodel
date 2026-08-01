"""Bipolar int8 SC replacement for nn.Linear forward.

Quantization granularity & stream halving are env-controlled so runs can be
A/B'd without code edits (aligned with scmp_diffusion's integration):

    SC_LINEAR_GRANULARITY  "per_tensor" (legacy default) | "per_row"
                           (kernel README: per_row is the intended mode for
                           all linear/MLP paths)
    SC_HALVE=1             uSystolic/HUB bipolar stream halving — pass
                           stoc_len=None and let the kernel run 2**(prec-1)
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from scmp_kernels import sc_matmul

from .sc_attention import _get_config

_GRANULARITY = os.environ.get("SC_LINEAR_GRANULARITY", "per_tensor")
_HALVE = os.environ.get("SC_HALVE") == "1"

# ---- mixed precision (heterogeneous stream lengths) -------------------------
# SC_MP_CONFIG='{"stoc_len_levels":[128,96,64,32],"level_fractions":[...]}'
#   rows are ranked by |x|.amax(-1) and bucketed into the levels (same policy as
#   scmp_diffusion / scmp_llm), then each bucket runs sc_matmul at its own
#   stoc_len. Per group spec, MP is only for logic<8; int8 stays uniform.
# SC_PREC=7|8 sets the quantization grid; SC_MP_FIXED_PREC=1 keeps sc_prec
#   pinned instead of deriving it per level (both variants are to be measured).
_MP_CONFIG = None
# Per-(operator, block) fractions. A single global fraction triple spends the
# SAME average budget on every operator and block, so the only mixing it does
# is within a layer by |x| magnitude — at matched budget that measured no
# better than uniform. SC_MP_PER_MODULE points at a calibration JSON whose
# "per_module_fractions" keeps the split the Lagrangian solver actually
# produced, letting budget flow across layers/operators.
_MP_PER_MODULE = None      # {(op, block_idx): [fractions]}
if os.environ.get("SC_MP_CONFIG"):
    import json as _json
    from scmp_kernels.mp import MPConfig as _MPConfig
    _spec = _json.loads(os.environ["SC_MP_CONFIG"])
    _MP_CONFIG = _MPConfig(stoc_len_levels=_spec["stoc_len_levels"],
                           level_fractions=_spec.get("level_fractions"))
if os.environ.get("SC_MP_PER_MODULE"):
    import json as _json, re as _re
    _pm = _json.load(open(os.environ["SC_MP_PER_MODULE"]))["per_module_fractions"]
    _SUFFIX_TO_OP = {"attn.qkv": "qkv", "attn.proj": "proj",
                     "mlp.fc1": "mlp_fc1", "mlp.fc2": "mlp_fc2"}
    _MP_PER_MODULE = {}
    for _name, _d in _pm.items():
        _m = _re.search(r"blocks\.(\d+)\.", _name)
        if _m is None:
            continue
        for _suf, _op in _SUFFIX_TO_OP.items():
            if _name.endswith(_suf):
                # Sensitivity calibration showed the metric-weight correlation
                # flips sign per module, so each module carries its own
                # ordering direction alongside its fractions.
                _MP_PER_MODULE[(_op, int(_m.group(1)))] = {
                    "fractions": _d.get("level_fractions"),
                    # scmp_llm's full mechanism: absolute thresholds on the
                    # per-call min-max-normalised metric, and salient input
                    # channels split out at a fixed high stream length with
                    # row dispatch applied only to the residual columns.
                    "thresholds": _d.get("thresholds"),
                    "protected": _d.get("protected") or [],
                    "protect_sl": _d.get("protect_sl"),
                    "invert": bool(_d.get("invert", False))}
                break
    print(f"sc_mp: per-module fractions for {len(_MP_PER_MODULE)} (op, block) cells",
          flush=True)
_SC_PREC = int(os.environ.get("SC_PREC", "8"))
_MP_FIXED_PREC = os.environ.get("SC_MP_FIXED_PREC") == "1"
# SC_UNIFORM_STOC_LEN: force a fixed stream length for the uniform ladder
# configs (sc_int8=128 / sc_int7=64 / sc_int6=32). Overrides the halve default.
_UNIFORM_STOC_LEN = os.environ.get("SC_UNIFORM_STOC_LEN")
_UNIFORM_STOC_LEN = int(_UNIFORM_STOC_LEN) if _UNIFORM_STOC_LEN else None
# SC_STEP_SCHEDULE: JSON list of halved cycle counts, indexed by diffusion
# step (resampled if its length differs from the sampler's step count).
_STEP_SCHEDULE = None
if os.environ.get("SC_STEP_SCHEDULE"):
    import json as _json2
    _STEP_SCHEDULE = [int(x) for x in _json2.loads(os.environ["SC_STEP_SCHEDULE"])]


def _resolve_sc_prec(stoc_len: int, default_prec: int) -> int:
    """Per-level sc_prec: pinned, or derived as ceil(log2(stoc_len))."""
    if _MP_FIXED_PREC:
        return default_prec
    import math
    return max(1, min(default_prec, int(math.ceil(math.log2(max(stoc_len, 2))))))


def _mp_linear_forward(x_flat, w, linear, orig_shape, sc_prec, out_dtype,
                       op=None, block_idx=None):
    """Mixed-precision path: bucket rows by importance, one sc_matmul per level.

    Mirrors scmp_diffusion's SCLinear MP forward — rows ranked by |x|.amax(-1),
    quantile-bucketed into MPConfig.stoc_len_levels, each bucket run at its own
    stream length, results scattered back.
    """
    from scmp_kernels.mp import classify_rows_by_metric

    in_dim = x_flat.shape[-1]
    out_features = w.shape[0]
    smooth = getattr(linear, "_sc_smooth_scales", None)

    fractions = _MP_CONFIG.level_fractions
    mod_invert = False
    thresholds = None
    prot_idx = None
    prot_sl = None
    if _MP_PER_MODULE is not None and op is not None and block_idx is not None:
        _ent = _MP_PER_MODULE.get((op, block_idx))
        if _ent is not None:
            fractions = _ent["fractions"] or fractions
            mod_invert = _ent["invert"]
            thresholds = _ent.get("thresholds")
            _pl = _ent.get("protected")
            if _pl:
                prot_idx = torch.as_tensor(sorted(set(int(i) for i in _pl)),
                                           dtype=torch.long, device=x_flat.device)
                prot_sl = int(_ent.get("protect_sl")
                              or max(_MP_CONFIG.stoc_len_levels))

    out_prot = None
    if prot_idx is not None and prot_idx.numel() > 0:
        # Salient-channel split (scmp_llm sc_common semantics): protected
        # columns run at a fixed high stream length, the row dispatch below
        # sees only the residual columns, and the partial products sum.
        mask = torch.ones(in_dim, dtype=torch.bool, device=x_flat.device)
        mask[prot_idx] = False
        rest_idx = mask.nonzero(as_tuple=True)[0]
        x_prot = x_flat.index_select(1, prot_idx).contiguous()
        w_prot = w.index_select(1, prot_idx).contiguous()
        sm_prot = (smooth.index_select(0, prot_idx).contiguous()
                   if smooth is not None else None)
        sp = _resolve_sc_prec(prot_sl, sc_prec)
        out_prot = sc_matmul(
            x_prot, w_prot, granularity=_GRANULARITY, mode="bipolar",
            sc_prec=sp, stoc_len=prot_sl,
            config=_get_config(int(prot_idx.numel()), sp),
            halve_bipolar_stoc_len=_HALVE, smooth_scales=sm_prot)
        x_flat = x_flat.index_select(1, rest_idx).contiguous()
        w = w.index_select(1, rest_idx).contiguous()
        smooth = (smooth.index_select(0, rest_idx).contiguous()
                  if smooth is not None else None)
        in_dim = int(rest_idx.numel())

    # Rank rows by the magnitude the kernel actually quantises. With
    # SmoothQuant attached, sc_matmul divides x by the per-channel scales
    # before per-row quantisation, so a row's SC error is set by
    # max_j |x_j / s_j| — not by max_j |x_j|. The scales spread ~5x (up to
    # 30x) across channels within one linear, so ranking on the raw amax
    # systematically misorders rows and hands the long streams to the wrong
    # ones; that misordering costs more than mixing gains (a wrong split is
    # WORSE than uniform, see the shuffled-bucket control in the error-grid
    # analysis). Rank on the smoothed activation instead.
    if smooth is not None:
        row_metric = (x_flat / smooth.to(x_flat.dtype)).abs().amax(dim=-1)
    else:
        row_metric = x_flat.abs().amax(dim=-1)
    # SC_MP_INVERT=1 hands the long streams to the LOW-metric rows instead.
    # Motivation: the more accurately rows are ranked by absolute matmul error
    # (smoothed amax), the worse the end-to-end quality gets — which points to
    # the PSNR-optimal allocation running in the opposite direction: per-row
    # quantisation roughly equalises relative error, so high-magnitude rows
    # carry structure that is robust to SC noise while low-magnitude detail
    # rows are the quality bottleneck.
    if os.environ.get("SC_MP_INVERT") == "1" or mod_invert:
        row_metric = -row_metric
    if thresholds is not None:
        # scmp_llm calibrated-table path: min-max normalise the metric per
        # call, then bucket against absolute descending thresholds — the
        # realised fractions adapt to each call's metric distribution.
        levels_ = _MP_CONFIG.stoc_len_levels
        m_min, m_max = row_metric.min(), row_metric.max()
        if float(m_max - m_min) < 1e-8:
            row_levels = torch.zeros(row_metric.shape[0], dtype=torch.long,
                                     device=row_metric.device)
        else:
            mn = (row_metric - m_min) / (m_max - m_min)
            th = torch.as_tensor(thresholds, dtype=mn.dtype, device=mn.device)
            row_levels = (mn.unsqueeze(1) < th.unsqueeze(0)).sum(dim=1)
        from scmp_kernels.mp.config import RowAssignment
        assignment = RowAssignment(
            row_levels=row_levels,
            level_row_indices={sl: (row_levels == i).nonzero(as_tuple=True)[0]
                               for i, sl in enumerate(levels_)})
    else:
        assignment = classify_rows_by_metric(
            row_metric, _MP_CONFIG.stoc_len_levels, fractions)

    out = torch.zeros(x_flat.shape[0], out_features,
                      device=x_flat.device, dtype=torch.float32)
    for sl, rows in assignment.level_row_indices.items():
        if len(rows) == 0 or sl == 0:      # level 0 == pruned rows, leave zeros
            continue
        sp = _resolve_sc_prec(sl, sc_prec)
        idx = rows if torch.is_tensor(rows) else torch.as_tensor(rows, device=x_flat.device)
        out[idx] = sc_matmul(
            x_flat[idx].contiguous(), w,
            granularity=_GRANULARITY,
            mode="bipolar",
            sc_prec=sp,
            stoc_len=sl,
            config=_get_config(in_dim, sp),
            halve_bipolar_stoc_len=_HALVE,
            smooth_scales=smooth,
        )
    if out_prot is not None:
        out = out + out_prot
    if linear.bias is not None:
        out = out + linear.bias.float()
    return out.reshape(*orig_shape[:-1], -1).to(out_dtype)


def sc_linear_forward(
    x: torch.Tensor,
    linear: nn.Linear,
    sc_prec: int = 8,
    stoc_len: int | None = None,
    op: str | None = None,
    block_idx: int | None = None,
) -> torch.Tensor:
    """Compute y = SC(x @ Wᵀ) + bias with per-tensor bipolar int8 SC.

    Args:
        x:      (..., in_dim) FP tensor
        linear: nn.Linear module with weight (out_dim, in_dim) and optional bias
    Returns:
        (..., out_dim) tensor in x.dtype
    """
    sc_prec = _SC_PREC if sc_prec == 8 else sc_prec  # SC_PREC env overrides default
    if _STEP_SCHEDULE is not None:
        # Per-timestep schedule: uniform within a step, varying across steps at
        # matched average. Needs no row-ordering signal — the step index is
        # exact — which is why this axis survives the rho=0.02 finding that
        # killed row-level MP.
        from .sc_controller import get_current_step
        _i, _n = get_current_step()
        stoc_len = _STEP_SCHEDULE[min(_i * len(_STEP_SCHEDULE) // max(_n, 1),
                                      len(_STEP_SCHEDULE) - 1)]
    elif _UNIFORM_STOC_LEN is not None:    # uniform ladder: fixed stream length
        stoc_len = _UNIFORM_STOC_LEN
    elif stoc_len is None and not _HALVE:
        stoc_len = 2 ** sc_prec
    # with SC_HALVE=1 keep stoc_len=None: the kernel then runs the bipolar
    # stream at 2**(sc_prec-1) (uSystolic sign-magnitude trick, lossless).

    orig_shape = x.shape
    in_dim = orig_shape[-1]
    x_flat = x.reshape(-1, in_dim).float().contiguous()
    w = linear.weight.float().contiguous()  # (out_dim, in_dim)

    if _MP_CONFIG is not None:
        return _mp_linear_forward(x_flat, w, linear, orig_shape, sc_prec, x.dtype, op=op, block_idx=block_idx)

    config = _get_config(in_dim, sc_prec)
    # SmoothQuant: calibration (evaluate/calibrate_smoothquant.py) attaches a
    # per-channel (D,) scale vector to the module; kernel rewrites the matmul
    # as (x/s) @ (w*s).T — mathematically equivalent, easier to quantize.
    smooth = getattr(linear, "_sc_smooth_scales", None)
    # granularity: legacy "per_tensor" (one (max,min) per operand matrix) or
    # "per_row" (one scale per row — kernel README's intended linear/MLP
    # mode, same as scmp_diffusion). sc_matmul computes x_flat @ w.T == x @ Wᵀ.
    y = sc_matmul(
        x_flat, w,
        granularity=_GRANULARITY,
        mode="bipolar",
        sc_prec=sc_prec,
        stoc_len=stoc_len,
        config=config,
        halve_bipolar_stoc_len=_HALVE,
        smooth_scales=smooth,
    )
    if linear.bias is not None:
        y = y + linear.bias.float()
    return y.reshape(*orig_shape[:-1], -1).to(x.dtype)
