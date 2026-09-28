# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE process-lifetime DeviceContext for every GP and GPC device entry.

A context per call exhausted Metal's per-process command queues on the M2 Pro
("Failed to create Metal command queue for context"): the hyperparameter
optimizer calls `gpr_lml_grad_host` once per L-BFGS step and the Laplace
classifier calls `_gpc_matvec` once per Newton step, so gp-optimize,
gp-optimize-restarts, gpc-multiclass and par-gpc-* REFUSED or returned the
same stale result for every fixture there (steward 1790601762837). Same
kernels, same launches, same order on one stream, and every entry still
synchronizes before it returns, so no bit moves. Storage is
`core/neural_context.mojo`'s `std.ffi._Global` slot, one per numeric tier."""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx

comptime _GP_CTX_NAME = "MojoGpContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoGpContextFast"


def gp_ctx() raises -> DeviceContext:
    """The GP binding's shared context, created on first use."""
    return neural_ctx[_GP_CTX_NAME]()
