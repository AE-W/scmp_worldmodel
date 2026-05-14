"""Int8 SC QK matmul for IRASim attention.

Only the Q @ K^T step is replaced. Softmax, dropout, attn @ V, and projections
remain FP32. The kernel is imported from the shared scmp_llm SC library.
"""
from __future__ import annotations

import torch

import triton

from sc_triton import (
    sc_matmul_enable_batched_bipolar,
    sc_matmul_enable_triton,
    fused_quant_bipolar_batched_kernel,
    enable_matmul_bipolar_batched_kernel,
    _get_cached_sequences,
    _get_cached_enable_tables,
    _COMPACT_ENABLE_THRESHOLD_BYTES,
)
from config_helpers import make_sobol_simple_config

_CONFIG_CACHE: dict[tuple[int, int], dict] = {}


def _get_config(head_dim: int, sc_prec: int) -> dict:
    key = (head_dim, sc_prec)
    cfg = _CONFIG_CACHE.get(key)
    if cfg is None:
        cfg = make_sobol_simple_config(head_dim, head_dim, sc_prec)
        _CONFIG_CACHE[key] = cfg
    return cfg


def sc_qk_matmul(
    q_scaled: torch.Tensor,
    k: torch.Tensor,
    sc_prec: int = 8,
    stoc_len: int | None = None,
) -> torch.Tensor:
    """Bipolar int8 SC QK matmul.

    Args:
        q_scaled: (B, H, N, D) — Q already multiplied by 1/sqrt(D).
        k:        (B, H, N, D)
        sc_prec:  SC bit precision (8 ⇒ int8).
        stoc_len: Stochastic stream length. Defaults to 2**sc_prec.

    Returns:
        (B, H, N, N) float32 attention logits.
    """
    B, H, N, D = q_scaled.shape
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    config = _get_config(D, sc_prec)

    q_flat = q_scaled.reshape(B * H, N, D).float().contiguous()
    k_flat = k.reshape(B * H, N, D).float().contiguous()

    q_maxs = q_flat.amax(dim=(1, 2))
    q_mins = q_flat.amin(dim=(1, 2))
    k_maxs = k_flat.amax(dim=(1, 2))
    k_mins = k_flat.amin(dim=(1, 2))

    out = sc_matmul_enable_batched_bipolar(
        q_flat, k_flat,
        q_maxs, q_mins, k_maxs, k_mins,
        sc_prec, config, stoc_len=stoc_len,
    )
    return out.reshape(B, H, N, N).to(q_scaled.dtype)


def _sc_matmul_enable_batched_bipolar_nm(
    a_flat: torch.Tensor,     # (BH, N, K)
    b_flat: torch.Tensor,     # (BH, M, K)
    a_maxs: torch.Tensor,     # (BH,)
    a_mins: torch.Tensor,
    b_maxs: torch.Tensor,
    b_mins: torch.Tensor,
    sc_prec: int,
    config: dict,
    stoc_len: int,
) -> torch.Tensor:
    """BH-batched bipolar int8 SC matmul with N != M.

    Structurally identical to sc_matmul_enable_batched_bipolar but does not
    hard-code M = N (that version is QK-only). Output: (BH, N, M).
    """
    a_flat = a_flat.contiguous()
    b_flat = b_flat.contiguous()
    BH, N, K = a_flat.shape
    M = b_flat.shape[1]
    assert b_flat.shape[2] == K and b_flat.shape[0] == BH
    device = a_flat.device

    q_max = 2 ** (sc_prec - 1) - 1
    q_min = -(2 ** (sc_prec - 1))
    max_rng_val = 2 ** sc_prec
    q_max_sq = float(q_max * q_max)

    abs_max_a = torch.maximum(a_maxs.abs(), a_mins.abs()).clamp(min=1e-5)
    abs_max_b = torch.maximum(b_maxs.abs(), b_mins.abs()).clamp(min=1e-5)
    scale_a = abs_max_a / q_max
    scale_b = abs_max_b / q_max
    inv_scale_a = 1.0 / scale_a
    inv_scale_b = 1.0 / scale_b

    # Quantize a and b — note the quant kernel writes transposed (BH, K, N/M).
    boundary_a = torch.empty(BH, K, N, dtype=torch.int16, device=device)
    sign_a = torch.empty(BH, K, N, dtype=torch.int8, device=device)
    boundary_b = torch.empty(BH, K, M, dtype=torch.int16, device=device)
    sign_b = torch.empty(BH, K, M, dtype=torch.int8, device=device)

    BLOCK = 1024
    slice_a = N * K
    slice_b = M * K
    fused_quant_bipolar_batched_kernel[(triton.cdiv(slice_a, BLOCK), BH)](
        a_flat, boundary_a, sign_a, inv_scale_a,
        q_max, q_min, max_rng_val, slice_a, N, K, BLOCK,
    )
    fused_quant_bipolar_batched_kernel[(triton.cdiv(slice_b, BLOCK), BH)](
        b_flat, boundary_b, sign_b, inv_scale_b,
        q_max, q_min, max_rng_val, slice_b, M, K, BLOCK,
    )

    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, device)
    V = 2 ** sc_prec + 1
    cum_table_bytes = K * (stoc_len + 1) * V * 2
    if cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES:
        raise RuntimeError(
            "Compact enable path not supported in N!=M variant; shrink K or stoc_len."
        )
    cum_indicator, k_table = _get_cached_enable_tables(
        config, sc_prec, device, rand_seqs_a_t, rand_seqs_b_t, stoc_len)
    V_actual = cum_indicator.shape[2]
    out_scale = scale_a * scale_b  # (BH,)

    output = torch.empty(BH, N, M, dtype=torch.float32, device=device)
    if N <= 64 or M <= 64:
        BLOCK_M, BLOCK_N = 16, 16
    else:
        BLOCK_M, BLOCK_N = 32, 32
    if K >= 4 and K % 4 == 0:
        BLOCK_K = 4
    elif K % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N), BH)
    enable_matmul_bipolar_batched_kernel[grid_mm](
        cum_indicator, k_table,
        boundary_a, boundary_b,
        sign_a, sign_b,
        output, out_scale,
        N, M, K,
        stoc_len, V_actual, q_max_sq,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=nw,
    )
    return output


def sc_av_matmul(
    attn: torch.Tensor,
    v: torch.Tensor,
    sc_prec: int = 8,
    stoc_len: int | None = None,
) -> torch.Tensor:
    """Bipolar int8 SC attn @ V matmul, BH-batched (single kernel launch).

    attn: (B, H, N, Nk)  — softmax output
    v:    (B, H, Nk, D)  — value
    Returns:
        (B, H, N, D) in attn.dtype.
    """
    B, H, N, Nk = attn.shape
    _, _, _, D = v.shape
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    # Inner contraction dim for AV is Nk (seq length).
    config = _get_config(Nk, sc_prec)

    a_flat = attn.reshape(B * H, N, Nk).float().contiguous()
    v_t_flat = v.transpose(-1, -2).reshape(B * H, D, Nk).float().contiguous()

    a_maxs = a_flat.amax(dim=(1, 2))
    a_mins = a_flat.amin(dim=(1, 2))
    v_maxs = v_t_flat.amax(dim=(1, 2))
    v_mins = v_t_flat.amin(dim=(1, 2))

    out = _sc_matmul_enable_batched_bipolar_nm(
        a_flat, v_t_flat,
        a_maxs, a_mins, v_maxs, v_mins,
        sc_prec, config, stoc_len,
    )
    return out.reshape(B, H, N, D).to(attn.dtype)
