"""Per-tensor bipolar int8 SC replacement for nn.Linear forward."""
from __future__ import annotations

import torch
import torch.nn as nn

from scmp_kernels import sc_matmul

from .sc_attention import _get_config


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
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    orig_shape = x.shape
    in_dim = orig_shape[-1]
    x_flat = x.reshape(-1, in_dim).float().contiguous()
    w = linear.weight.float().contiguous()  # (out_dim, in_dim)

    config = _get_config(in_dim, sc_prec)
    # per_tensor: one (max, min) over the whole matrix for each operand —
    # matches the previous sc_matmul_enable_triton behavior. sc_matmul
    # computes x_flat @ w.T == x @ Wᵀ.
    y = sc_matmul(
        x_flat, w,
        granularity="per_tensor",
        mode="bipolar",
        sc_prec=sc_prec,
        stoc_len=stoc_len,
        config=config,
    )
    if linear.bias is not None:
        y = y + linear.bias.float()
    return y.reshape(*orig_shape[:-1], -1).to(x.dtype)
