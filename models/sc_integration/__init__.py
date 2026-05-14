"""SC (Stochastic Computing) int8 QK replacement for IRASim.

Pulls SC kernels from /home/dingqy/Bench/scmp_llm/SC/ without copying code.
"""
import os
import sys

_SC_KERNEL_ROOT = os.environ.get(
    "IRASIM_SC_KERNEL_ROOT",
    "/home/dingqy/Bench/scmp_llm/SC",
)
if _SC_KERNEL_ROOT not in sys.path:
    sys.path.insert(0, _SC_KERNEL_ROOT)

from .sc_attention import sc_qk_matmul, sc_av_matmul  # noqa: E402
from .sc_linear import sc_linear_forward  # noqa: E402
from .sc_mlp import SCMlp  # noqa: E402
from .sc_controller import (  # noqa: E402
    configure,
    reconfigure,
    get_config,
    is_op_enabled,
    set_skip_blocks,
    clear_skip_blocks,
)

__all__ = [
    "sc_qk_matmul",
    "sc_av_matmul",
    "sc_linear_forward",
    "SCMlp",
    "configure",
    "reconfigure",
    "get_config",
    "is_op_enabled",
    "set_skip_blocks",
    "clear_skip_blocks",
]
