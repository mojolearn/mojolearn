# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba S1 RMSNorm row fold as lanes then a fixed tree (lane nr-mamba,
2026-10-04, roadmap B11).

IDN_MAMBA_RMS_TREE (IDENTICAL, default ON; `-D MOJOLEARN_IDN_MAMBA_RMS_TREE_OFF`
or `-D MOJOLEARN_IDN_ALL_OFF` restores main's one serial chain per row): the
row's sum of squares over d_model is MAMBA_RMS_LANES strided lane chains
(lane k folds j = k, k + 32, k + 64, ... ascending, `acc = fma(x, x, acc)`
from +0.0, each step flushed), then a balanced pairwise tree over the 32 lane
partials in lane order (`p[i] = p[2i] + p[2i+1]`, five levels, each add
flushed). The lane count is a constant (never the vendor's warp width, never a
shape rule), so the bits are a function of the row alone.

BITS CHANGE for `norm.sumsq` and everything downstream of it, in Mamba-1,
Mamba-2 and Mamba-3, on every column together: the device kernel
(`modeling_mamba.mojo::mamba_rms_norm`), the generated host column (it runs the
same source) and the three oracles (`mamba_oracle`, `mamba2_oracle`,
`mamba3_oracle`), which the host bindings also run. The backward reads the
recorded sumsq, so it follows. A sum of squares from +0.0 is never -0.0, so a
+0.0 lane partial (d_model below 32) is exact. The output loop (S2-S4) is per
element and keeps its bits. Every S1 sabotage arm turns this off, so the
sabotage checks still see main's serial chain move.

No GPU import: the oracles and the host column import this module as is.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add

comptime MAMBA_RMS_LANES = 32

comptime IDN_MAMBA_RMS_TREE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_MAMBA_RMS_TREE_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not is_defined["MOJOLEARN_MAMBA_SABOTAGE_S1_FOLD_DESCENDING"]()
    and not is_defined["MOJOLEARN_BATCHINV_SABOTAGE_NORM_CHUNK_FROM_M"]()
)


@always_inline
def mamba_rms_tree(v_in: SIMD[DType.float32, MAMBA_RMS_LANES]) -> Float32:
    """The fixed pairwise tree over the lane partials, lane order."""
    var v = v_in
    comptime for lvl in range(5):
        comptime w = (MAMBA_RMS_LANES // 2) >> lvl
        comptime for i in range(w):
            v[i] = ftz(ftz(v[2 * i]) + ftz(v[2 * i + 1]))
    return v[0]


def mamba_rms_row_sumsq_list(x: List[Float32], base: Int, n: Int) -> Float32:
    """The oracles' S1 fold of `x[base : base + n]`: the lanes and tree
    under IDN_MAMBA_RMS_TREE, else main's serial ascending chain."""
    comptime if IDN_MAMBA_RMS_TREE:
        var v = SIMD[DType.float32, MAMBA_RMS_LANES](0.0)
        comptime for lane in range(MAMBA_RMS_LANES):
            var acc = Float32(0.0)
            var j = lane
            while j < n:
                var xj = ftz(x[base + j])
                acc = ftz(identical_mul_add(xj, xj, acc))
                j += MAMBA_RMS_LANES
            v[lane] = acc
        return mamba_rms_tree(v)
    else:
        var acc = Float32(0.0)
        for j in range(n):
            var xj = ftz(x[base + j])
            acc = ftz(identical_mul_add(xj, xj, acc))
        return acc


def mamba_rms_row_sumsq_host(
    x: MutPointer[Float32, MutUntrackedOrigin], base: Int, n: Int
) -> Float32:
    """`mamba_rms_row_sumsq_list` over a host pointer (the Mamba-3 oracle's
    task closures)."""
    comptime if IDN_MAMBA_RMS_TREE:
        var v = SIMD[DType.float32, MAMBA_RMS_LANES](0.0)
        comptime for lane in range(MAMBA_RMS_LANES):
            var acc = Float32(0.0)
            var j = lane
            while j < n:
                var xj = ftz(x.unsafe_load(base + j))
                acc = ftz(identical_mul_add(xj, xj, acc))
                j += MAMBA_RMS_LANES
            v[lane] = acc
        return mamba_rms_tree(v)
    else:
        var acc = Float32(0.0)
        for j in range(n):
            var xj = ftz(x.unsafe_load(base + j))
            acc = ftz(identical_mul_add(xj, xj, acc))
        return acc
