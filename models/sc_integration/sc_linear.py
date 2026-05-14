"""Int8 SC replacement for nn.Linear forward."""
from __future__ import annotations

import torch
import torch.nn as nn

from sc_triton import sc_matmul_enable_triton

from .sc_attention import _get_config


def sc_linear_forward(
    x: torch.Tensor,
    linear: nn.Linear,
    sc_prec: int = 8,
    stoc_len: int | None = None,
) -> torch.Tensor:
    """Compute y = SC(x @ W^T) + bias with bipolar int8 SC.

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
    y = sc_matmul_enable_triton(
        x_flat, w,
        x_flat.max().item(), x_flat.min().item(),
        w.max().item(), w.min().item(),
        mode="bipolar",
        sc_prec=sc_prec,
        config=config,
        stoc_len=stoc_len,
    )
    if linear.bias is not None:
        y = y + linear.bias.float()
    return y.reshape(*orig_shape[:-1], -1).to(x.dtype)
