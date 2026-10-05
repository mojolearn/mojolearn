# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""fix-kg1-kernel (2026-10-04, audit B13): the SEGMENTED fold of the GP
predictive variance, `sum_i v[i][t]^2` over the training axis, shared by
the device kernels and every host column.

IDENTICAL, ON by default (`-D MOJOLEARN_IDN_GP_VAR_SEG_OFF` or
`MOJOLEARN_IDN_ALL_OFF` restores the one-thread serial chain on every
column). The training axis is cut into GP_VAR_SEGS contiguous segments of
`ceil(n_train / GP_VAR_SEGS)` rows; each segment is one ascending chain
`acc = ftz(identical_mul_add(vv, vv, acc))` from +0.0, and the eight partials
are added in the FIXED tree `((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))`, each add
rounded once. The cut depends on n_train only, never on n_star or the launch,
so a test point predicted alone and in a batch gets the same bits.

On the device one block is GP_VAR_PTS test points x GP_VAR_SEGS segments
(256 threads): threads with adjacent thread_idx.x read adjacent test points,
so every load stays coalesced, and a test point's chain is eight times
shorter than the serial one. BITS CHANGE (a different bracketing of the same
squares): `gp_variance_kernel` / `gpc_latent_var_kernel`'s replacements,
`gpr_oracle.mojo`, `gpc_steps.mojo` and `gp_oracle.mojo` all fold this way
under the same gate. The sabotage copy (`checks/gp_sabotage.mojo`) still
folds serially (lane/review-fixes: docstring corrected), so its variance
stage differs from production by the fold as well as by its planted fault.

Shared memory: GP_VAR_PTS * GP_VAR_SEGS floats = 1 KiB, a static page far
below every vendor's per-block limit (the fits gate is the comptime assert
in the kernels).
"""
from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)

comptime GP_IDN_VAR_SEG = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_GP_VAR_SEG_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime GP_VAR_SEGS = 8
comptime GP_VAR_PTS = 32
comptime GP_VAR_SMEM_FLOATS = GP_VAR_SEGS * GP_VAR_PTS
#: the smallest per-block shared page any target vendor grants (Apple 32 KiB)
comptime GP_VAR_SMEM_LIMIT_BYTES = 32768


@always_inline
def gp_var_seg_len(n_train: Int) -> Int:
    """Rows per segment: ceil(n_train / GP_VAR_SEGS)."""
    return (n_train + GP_VAR_SEGS - 1) // GP_VAR_SEGS


@always_inline
def gp_var_tree8(
    p0: Float32,
    p1: Float32,
    p2: Float32,
    p3: Float32,
    p4: Float32,
    p5: Float32,
    p6: Float32,
    p7: Float32,
) -> Float32:
    """The fixed combine `((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))`."""
    var a = ftz(ftz(ftz(p0) + ftz(p1)) + ftz(ftz(p2) + ftz(p3)))
    var b = ftz(ftz(ftz(p4) + ftz(p5)) + ftz(ftz(p6) + ftz(p7)))
    return ftz(a + b)


def gp_var_seg_sumsq_host(
    v: List[Float32], n_train: Int, n_star: Int, t: Int
) -> Float32:
    """The host column's replay of the device fold for test point `t`;
    `v` is `n_train x n_star` row-major."""
    var seg_len = gp_var_seg_len(n_train)
    var p = SIMD[DType.float32, GP_VAR_SEGS](0.0)
    for s in range(GP_VAR_SEGS):
        var lo = s * seg_len
        var hi = min(lo + seg_len, n_train)
        var acc = Float32(0.0)
        for i in range(lo, hi):
            var vv = ftz(v[i * n_star + t])
            acc = ftz(identical_mul_add(vv, vv, acc))
        p[s] = acc
    return gp_var_tree8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
