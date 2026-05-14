"""Global flags controlling which matmuls use SC in each transformer block.

Attention / SCMlp read these at forward time. The configure() call is
idempotent and driven by the attention_mode string on Attention init.

Per-block skip:
    skip_blocks maps op_name ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2")
    to a set of block indices that should stay FP even when the op is enabled
    globally. Use set_skip_blocks() / clear_skip_blocks() at eval time.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, Set


@dataclass
class SCConfig:
    enable_qkv: bool = False       # attn.qkv linear
    enable_qk: bool = False        # Q @ K^T
    enable_av: bool = False        # attn @ V
    enable_proj: bool = False      # attn.proj linear
    enable_mlp_fc1: bool = False
    enable_mlp_fc2: bool = False
    sc_prec: int = 8
    stoc_len: int = 256
    skip_blocks: Dict[str, Set[int]] = field(default_factory=dict)


_cfg = SCConfig()
_last_configured_mode = None

_PRESETS = {
    # Progressive SC replacement variants. All use int8, stoc_len=256.
    "sc_int8":                  ["qk"],                                      # v1
    "sc_int8_qk_av":            ["qk", "av"],                                # v2
    "sc_int8_qk_av_proj":       ["qk", "av", "proj"],                        # v3
    "sc_int8_qk_av_proj_fc1":   ["qk", "av", "proj", "mlp_fc1"],             # v4
    "sc_int8_full":             ["qk", "av", "qkv", "proj", "mlp_fc1", "mlp_fc2"],  # v5
}

_FIELDS = ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2")


def configure(mode: str) -> bool:
    """Set flags from an attention_mode string. Idempotent.

    First call with a given mode sets enable_* flags from the preset.
    Subsequent calls with the same mode are no-ops, so external overrides
    to enable_* / skip_blocks persist across model forwards.

    Returns True if `mode` matches a known SC preset.
    """
    global _last_configured_mode
    if mode == _last_configured_mode:
        return mode in _PRESETS
    if mode not in _PRESETS:
        return False
    for f in _FIELDS:
        setattr(_cfg, f"enable_{f}", False)
    for f in _PRESETS[mode]:
        setattr(_cfg, f"enable_{f}", True)
    _last_configured_mode = mode
    return True


def reconfigure(mode: str) -> bool:
    """Force re-apply preset flags (resets enable_*). Does NOT reset skip_blocks."""
    global _last_configured_mode
    _last_configured_mode = None
    return configure(mode)


def get_config() -> SCConfig:
    return _cfg


def is_op_enabled(op: str, block_idx: int) -> bool:
    """Return True if `op` is SC-enabled AND `block_idx` is not in its skip set."""
    if not getattr(_cfg, f"enable_{op}", False):
        return False
    skip = _cfg.skip_blocks.get(op)
    if skip is not None and block_idx in skip:
        return False
    return True


def set_skip_blocks(op: str, block_indices: Iterable[int]) -> None:
    """Replace the skip set for `op` with the given indices."""
    if op not in _FIELDS:
        raise ValueError(f"unknown op {op}; expected one of {_FIELDS}")
    _cfg.skip_blocks[op] = set(int(i) for i in block_indices)


def clear_skip_blocks(op: str = None) -> None:
    """Clear skip blocks for one op (or all ops if op is None)."""
    if op is None:
        _cfg.skip_blocks.clear()
    elif op in _cfg.skip_blocks:
        del _cfg.skip_blocks[op]
