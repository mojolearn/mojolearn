# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Mamba-2 / Mamba-3 backward row folds as lanes then a fixed tree (lane
mamba-folds, 2026-10-28; idea 14 of the 2026-10-28 neural report, roadmap
B6's remaining serial folds).

`-D MOJOLEARN_IDN_MAMBA_BWD_FOLDS=1`, IDENTICAL only, default OFF. A BITS
CHANGE across versions, the same on every column (NVIDIA, AMD, Apple, and the
host column that `tools/mamba_host_gen.py` restates from this source).

WHAT WAS SERIAL. The RMSNorm backwards of the Mamba-2 gated norm
(`mamba2_gnorm_backward_kernel`, width d_inner; it also serves the Mamba-2
block norm at width d_model) and of the Mamba-3 block norm
(`mamba3_block_norm_backward_kernel`, width d_model) run ONE thread per token
row: the row dot `sum_j (dy_j * w_j) * x_j` is one serial chain over the
width, and the per-cell outputs are a second serial walk over the same width.
At B * L rows that is B * L threads, each 2 * width steps deep.

THE FOLD (v2 when the define is on). The row dot is MAMBA_RMS_LANES = 32
strided lane chains -- lane k folds j = k, k + 32, k + 64, ... ascending from
+0.0, `acc = ftz(fma(ftz(ftz(dy_j) * ftz(w_j)), ftz(x_j), acc))` (each term
spelled exactly as the v1 chain spells it) -- then the fixed pairwise tree over
the 32 lane partials in lane order (`mamba_rms_tree`, five levels, each add
flushed). That is the forward S1 fold (`mamba/checks/mamba_rms_fold.mojo`,
roadmap B11, shipped default) applied to the backward's row dot, so forward
and backward fold their rows the same way. The lane count is a profile
constant, never the vendor's warp width and never a shape rule: the bits are
a function of the row alone. The per-cell outputs then run one thread per
cell with the v1 per-cell arithmetic (only the dot they read changes).

WHAT MOVES. The row dot, so `d_x` (d_gate) of the norm and everything
upstream of it. The weight-product rows are per cell and keep their bits
(they do not read the dot), so the norm weight gradients only move through
upstream changes in later steps.

Generator-friendly: plain cell kernels, no shared memory, no atomics, no float
crossing a thread except through the global partials a later launch reads.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_mul,
    identical_mul_add,
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
)
from mamba.checks.mamba_rms_fold import MAMBA_RMS_LANES, mamba_rms_tree


comptime IDN_MAMBA_BWD_FOLDS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_MAMBA_BWD_FOLDS"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: Threads per block. Execution plan only (no float crosses a thread inside a
#: launch), so it moves no bit.
comptime MAMBA_BWD_FOLD_TPB = 128


def mamba_bwd_fold_grid(n: Int) -> Int:
    var g = (n + MAMBA_BWD_FOLD_TPB - 1) // MAMBA_BWD_FOLD_TPB
    if g < 1:
        return 1
    return g


def mamba_bwd_fold_floats(m: Int) -> Int:
    """Scratch for one row-dot fold over `m` rows: `[m, LANES]` lane partials
    then `[m]` dots, at offset `mamba_bwd_fold_dot_offset(m)`. 1 when off."""
    comptime if IDN_MAMBA_BWD_FOLDS:
        var n = m * (MAMBA_RMS_LANES + 1)
        if n < 1:
            return 1
        return n
    return 1


def mamba_bwd_fold_dot_offset(m: Int) -> Int:
    return m * MAMBA_RMS_LANES


def mamba_bwd_rowdot_lane_kernel(
    fold: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    width_in: Int32,
):
    """One `(row, lane)` per thread: lane k's chain over j = k, k + LANES, ...
    ascending from +0.0, the v1 term spelling. Writes `fold[row * LANES + k]`."""
    var m = Int(m_in)
    var width = Int(width_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * MAMBA_RMS_LANES:
        return
    var row = cell // MAMBA_RMS_LANES
    var lane = cell - row * MAMBA_RMS_LANES
    var acc = Float32(0.0)
    var j = lane
    while j < width:
        var weighted = ftz(
            identical_mul(
                ftz(dy.unsafe_load(row * width + j)), ftz(weight.unsafe_load(j))
            )
        )
        acc = ftz(identical_mul_add(weighted, ftz(x.unsafe_load(row * width + j)), acc))
        j += MAMBA_RMS_LANES
    fold.unsafe_store(cell, acc)


def mamba_bwd_rowdot_tree_kernel(
    fold: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dot_off_in: Int32,
):
    """One row per thread: the fixed pairwise tree over its lane partials,
    lane order, into `fold[dot_off + row]`."""
    var m = Int(m_in)
    var dot_off = Int(dot_off_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= m:
        return
    var v = SIMD[DType.float32, MAMBA_RMS_LANES](0.0)
    comptime for lane in range(MAMBA_RMS_LANES):
        v[lane] = fold.unsafe_load(row * MAMBA_RMS_LANES + lane)
    fold.unsafe_store(dot_off + row, mamba_rms_tree(v))
