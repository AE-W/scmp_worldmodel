"""SC (Stochastic Computing) int8 replacement ops for IRASim.

The actual SC Triton kernels live in the ``scmp_kernels`` package, vendored as
a git submodule at ``kernels/`` (repo root) and installed editable with
``pip install -e ./kernels``. This package only holds the IRASim-specific glue
(attention / linear / MLP drop-ins + the global SC controller); the kernels
themselves are shared with the other SC applications, so kernel updates
propagate here automatically by bumping the submodule.

If imports below fail with ``ModuleNotFoundError: scmp_kernels``, the submodule
is not installed — run::

    git submodule update --init --recursive
    pip install -e ./kernels
"""
from .sc_attention import sc_qk_matmul, sc_av_matmul
from .sc_linear import sc_linear_forward
from .sc_mlp import SCMlp
from .sc_controller import (
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
