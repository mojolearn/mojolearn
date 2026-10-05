"""The two-level FeatureFreq fit's row work on the device (lane cpu3-gbdt-a).

`fit_two_level_feature_freq_tree` (`doc_parallel_boosting.mojo`) used to
stage the stat planes, validate the weights and sum the two magnitudes in a
host loop over every row, and to estimate each leaf in a second host loop
over the leaf's rows. Both are row work on a GPU route, so both run here.

THE FOLD ORDER, the same on NVIDIA, AMD, Apple and the host column
(`gbdt/host/gbdt_oracle_feature_freq.mojo`, `_ff_two_stage_sums`):

  stage 1: `TL_GRID` blocks of `TL_BLOCK` threads, a fixed geometry that
           never depends on the device. Global thread `g = b * TL_BLOCK + t`
           folds items `g, g + TL_STRIDE, ...` ascending from +0.0; each
           block folds its threads with the halving tree
           (`halving_block_sum`, `red[t] += red[t + step]`).
  stage 2: the `TL_GRID` block partials, as `deterministic_sum_lanes_kernel`
           folds them: `+0.0 + partial[t]` per thread, then the same
           halving tree.

Every product is `identical_mul` (the contraction pin) and the one division
is `identical_div`, so the four columns agree bit for bit. The old order (a
sequential row-order chain) moved with this change on every column at once.
"""

from std.gpu import block_idx, thread_idx
from std.math import isfinite
from core.pinned_reduce import halving_block_sum
from checks.numerics import identical_div, identical_mul

comptime TL_BLOCK = 256
"""Threads per block, both stages (a power of two for the halving tree)."""
comptime TL_GRID = 256
"""Stage-1 blocks: fixed, so the fold shape is the same on every device."""
comptime TL_STRIDE = TL_BLOCK * TL_GRID
comptime TL_PREP_LANES = 4
"""Stage-1 prep lanes: weight sum, |weighted target| sum, the bad-weight
flag count and the non-finite-target flag count."""


def two_level_prep_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Int32,
    n_in: Int32,
    rows: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """Stage 1 of the prep: `rows[r] = r`, `stats[r] = weight`,
    `stats[n + r] = weight * y[r]`, and per block the four lanes of
    `partials[b * TL_PREP_LANES + lane]`. Grid `(TL_GRID, 1, 1)`, block
    `(TL_BLOCK, 1, 1)`. `w` is read only when `has_w` is non-zero."""
    var n = Int(n_in)
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var acc_w = Float32(0.0)
    var acc_g = Float32(0.0)
    var bad_w = Float32(0.0)
    var bad_t = Float32(0.0)
    var r = b * TL_BLOCK + t
    while r < n:
        var weight = Float32(1.0)
        if has_w != Int32(0):
            weight = w.unsafe_load(r)
        if weight < Float32(0.0) or not isfinite(weight):
            bad_w = Float32(1.0)
        var wt = identical_mul(weight, y.unsafe_load(r))
        if not isfinite(wt):
            bad_t = Float32(1.0)
        rows.unsafe_store(r, UInt32(r))
        stats.unsafe_store(r, weight)
        stats.unsafe_store(n + r, wt)
        acc_w = acc_w + weight
        acc_g = acc_g + (-wt if wt < Float32(0.0) else wt)
        r += TL_STRIDE
    var s_w = halving_block_sum[TL_BLOCK](acc_w)
    var s_g = halving_block_sum[TL_BLOCK](acc_g)
    var s_bw = halving_block_sum[TL_BLOCK](bad_w)
    var s_bt = halving_block_sum[TL_BLOCK](bad_t)
    if t == 0:
        partials.unsafe_store(b * TL_PREP_LANES + 0, s_w)
        partials.unsafe_store(b * TL_PREP_LANES + 1, s_g)
        partials.unsafe_store(b * TL_PREP_LANES + 2, s_bw)
        partials.unsafe_store(b * TL_PREP_LANES + 3, s_bt)


def two_level_leaf_partials_kernel(
    rows: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    leaf_tab: MutPointer[UInt32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """Stage 1 of the leaf estimate. Grid `(TL_GRID, n_leaves, 1)`, block
    `(TL_BLOCK, 1, 1)`. Leaf `L = block_idx.y` owns the final row order's
    `[leaf_tab[2L], leaf_tab[2L] + leaf_tab[2L + 1])`; block `b` writes
    `partials[(L * TL_GRID + b) * 2 + {0: target sum, 1: weight sum}]`."""
    var n = Int(n_in)
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var leaf = Int(block_idx.y)
    var off = Int(leaf_tab.unsafe_load(2 * leaf))
    var size = Int(leaf_tab.unsafe_load(2 * leaf + 1))
    var acc_t = Float32(0.0)
    var acc_w = Float32(0.0)
    var i = b * TL_BLOCK + t
    while i < size:
        var row = Int(rows.unsafe_load(off + i))
        acc_t = acc_t + stats.unsafe_load(n + row)
        acc_w = acc_w + stats.unsafe_load(row)
        i += TL_STRIDE
    var s_t = halving_block_sum[TL_BLOCK](acc_t)
    var s_w = halving_block_sum[TL_BLOCK](acc_w)
    if t == 0:
        partials.unsafe_store((leaf * TL_GRID + b) * 2, s_t)
        partials.unsafe_store((leaf * TL_GRID + b) * 2 + 1, s_w)


def two_level_leaf_values_kernel(
    partials: MutPointer[Float32, MutAnyOrigin],
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """Stage 2 of the leaf estimate and the leaf value. Grid
    `(n_leaves, 1, 1)`, block `(TL_GRID, 1, 1)` (one partial per thread).
    `dst[L] = learning_rate * total / (weight + l2)`, or 0 for a leaf with
    no weight (the old `two_level_weighted_leaf_value` statements)."""
    var t = Int(thread_idx.x)
    var leaf = Int(block_idx.x)
    var p_t = Float32(0.0) + partials.unsafe_load((leaf * TL_GRID + t) * 2)
    var p_w = Float32(0.0) + partials.unsafe_load((leaf * TL_GRID + t) * 2 + 1)
    var total = halving_block_sum[TL_GRID](p_t)
    var total_weight = halving_block_sum[TL_GRID](p_w)
    if t == 0:
        var v = Float32(0.0)
        if total_weight > Float32(0.0):
            v = identical_div(
                identical_mul(learning_rate, total), total_weight + l2_leaf_reg
            )
        dst.unsafe_store(leaf, v)
