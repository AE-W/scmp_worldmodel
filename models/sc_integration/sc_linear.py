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


def sc_linear_forward(
    x: torch.Tensor,
    linear: nn.Linear,
    sc_prec: int = 8,
    stoc_len: int | None = None,
) -> torch.Tensor:
    """Compute y = SC(x @ Wᵀ) + bias with per-tensor bipolar int8 SC.

    Args:
        x:      (..., in_dim) FP tensor
        linear: nn.Linear module with weight (out_dim, in_dim) and optional bias
    Returns:
        (..., out_dim) tensor in x.dtype
    """
    if stoc_len is None and not _HALVE:
        stoc_len = 2 ** sc_prec
    # with SC_HALVE=1 keep stoc_len=None: the kernel then runs the bipolar
    # stream at 2**(sc_prec-1) (uSystolic sign-magnitude trick, lossless).

    orig_shape = x.shape
    in_dim = orig_shape[-1]
    x_flat = x.reshape(-1, in_dim).float().contiguous()
    w = linear.weight.float().contiguous()  # (out_dim, in_dim)

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
