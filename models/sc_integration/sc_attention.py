"""Bipolar int8 SC attention matmuls (Q·Kᵀ and softmax·V) for IRASim.

Both ops route through the shared ``scmp_kernels.sc_matmul`` dispatcher with
``granularity="per_head"`` (one ``(max, min)`` per attention head). Only these
two matmuls are replaced; softmax, dropout, and the projections stay FP32. The
dispatcher computes ``a @ b.T`` and auto-detects the per-head batched layout
from the 3D shape, so it handles the non-square attn·V case directly — no
local kernel orchestration needed.
"""
from __future__ import annotations

import os

import torch

from scmp_kernels import sc_matmul
from scmp_kernels.sc.config_helpers import make_sobol_simple_config

_CONFIG_CACHE: dict[tuple[int, int], dict] = {}

# SC_HALVE=1: uSystolic bipolar stream halving (stoc_len=None lets the kernel
# derive 2**(sc_prec-1)). Same knob as sc_linear.py / scmp_diffusion.
_HALVE = os.environ.get("SC_HALVE") == "1"


def _get_config(contraction_dim: int, sc_prec: int) -> dict:
    """Cache one Sobol RNG/SNG config per (contraction dim, precision).

    ``contraction_dim`` is the inner (summed) dimension of the matmul: the
    head dim ``D`` for Q·Kᵀ, the key length ``Nk`` for softmax·V, and the input
    feature count for linear layers.
    """
    key = (contraction_dim, sc_prec)
    cfg = _CONFIG_CACHE.get(key)
    if cfg is None:
        cfg = make_sobol_simple_config(contraction_dim, contraction_dim, sc_prec)
        _CONFIG_CACHE[key] = cfg
    return cfg


def sc_qk_matmul(
    q_scaled: torch.Tensor,
    k: torch.Tensor,
    sc_prec: int = 8,
    stoc_len: int | None = None,
) -> torch.Tensor:
    """Bipolar int8 SC Q·Kᵀ matmul.

    Args:
        q_scaled: (B, H, N, D) — Q already multiplied by 1/sqrt(D).
        k:        (B, H, N, D)
        sc_prec:  SC bit precision (8 ⇒ int8).
        stoc_len: Stochastic stream length. Defaults to 2**sc_prec.

    Returns:
        (B, H, N, N) attention logits in q_scaled.dtype.
    """
    B, H, N, D = q_scaled.shape
    if stoc_len is None and not _HALVE:
        stoc_len = 2 ** sc_prec
    config = _get_config(D, sc_prec)

    q_flat = q_scaled.reshape(B * H, N, D).float().contiguous()
    k_flat = k.reshape(B * H, N, D).float().contiguous()

    out = sc_matmul(
        q_flat, k_flat,
        granularity="per_head",
        mode="bipolar",
        sc_prec=sc_prec,
        stoc_len=stoc_len,
        config=config,
        halve_bipolar_stoc_len=_HALVE,
    )
    return out.reshape(B, H, N, N).to(q_scaled.dtype)


def sc_av_matmul(
    attn: torch.Tensor,
    v: torch.Tensor,
    sc_prec: int = 8,
    stoc_len: int | None = None,
) -> torch.Tensor:
    """Bipolar int8 SC (softmax·V) matmul, per-head batched.

    Args:
        attn: (B, H, N, Nk)  — softmax output.
        v:    (B, H, Nk, D)  — value.
        sc_prec:  SC bit precision (8 ⇒ int8).
        stoc_len: Stochastic stream length. Defaults to 2**sc_prec.

    Returns:
        (B, H, N, D) in attn.dtype.
    """
    B, H, N, Nk = attn.shape
    _, _, _, D = v.shape
    if stoc_len is None and not _HALVE:
        stoc_len = 2 ** sc_prec
    # Inner contraction dim for attn·V is Nk (key/sequence length).
    config = _get_config(Nk, sc_prec)

    a_flat = attn.reshape(B * H, N, Nk).float().contiguous()
    # sc_matmul computes a @ b.T, so feed V transposed to (BH, D, Nk);
    # b.T then restores (BH, Nk, D) and a @ b.T == attn @ V.
    v_t_flat = v.transpose(-1, -2).reshape(B * H, D, Nk).float().contiguous()

    out = sc_matmul(
        a_flat, v_t_flat,
        granularity="per_head",
        mode="bipolar",
        sc_prec=sc_prec,
        stoc_len=stoc_len,
        config=config,
        halve_bipolar_stoc_len=_HALVE,
    )
    return out.reshape(B, H, N, D).to(attn.dtype)
