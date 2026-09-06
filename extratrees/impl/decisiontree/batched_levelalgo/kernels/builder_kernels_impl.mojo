# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ExtraTrees builder kernels.

Ports cuML `builder_kernels_impl.cuh` at `00094f7`. Split search deliberately
uses one threshold drawn from the node-local feature range, following sklearn's
RandomSplitter, instead of cuML's quantile histogram (DEVIATION 137).
"""

from std.sys.compile import is_defined

from extratrees.checks.pcg_rng import SplitKey, key_for, uniform_float
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.decisiontree.batched_levelalgo.split import (
    Split,
    split_reduce_seed_at,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    NodeWorkItem,
    SAMPLER_UNVISITED,
)


comptime TPB_DEFAULT = 128
"""`builder_kernels_impl.cuh:33`, `static constexpr int TPB_DEFAULT = 128`."""


comptime SEARCH_ROWS_PER_THREAD = _search_rows_per_thread()
"""Rows each thread of the range and score passes folds -- DEVIATION 2020."""


def _search_rows_per_thread() -> Int:
    """DEVIATION 2020's measurement arms."""
    if is_defined["MOJOLEARN_ET_SEARCH_RPT_16"]():
        return 16
    if is_defined["MOJOLEARN_ET_SEARCH_RPT_8"]():
        return 8
    if is_defined["MOJOLEARN_ET_SEARCH_RPT_4"]():
        return 4
    if is_defined["MOJOLEARN_ET_SEARCH_RPT_2"]():
        return 2
    return 1


comptime SEARCH_SAB_RPT_TAIL_DROP = is_defined[
    "MOJOLEARN_ET_SAB_RPT_TAIL_DROP"
]()
"""DEVIATION 2020's required-RED arm (a measurement arm, never a gate; it must never reach a shipped build)."""


def partition_samples(
    dataset: Dataset,
    split: Split,
    work_item: NodeWorkItem,
    tpb: Int = TPB_DEFAULT,
):
    """Partition `row_ids` into left/right by the best split, in place."""
    var row_ids = dataset.row_ids
    var range_start = Int(work_item.instances.begin)
    var range_len = Int(work_item.instances.count)
    var col_base = Int(split.colid) * Int(dataset.m)
    var quesval = split.quesval

    var loffset = range_start
    var part = loffset + Int(split.n_left)
    var roffset = part
    var end = range_start + range_len

    var lflag = List[Int](length=tpb, fill=0)
    var rflag = List[Int](length=tpb, fill=0)
    var lidx = List[Int](length=tpb, fill=0)
    var ridx = List[Int](length=tpb, fill=0)
    var lcomp = List[Int](length=tpb, fill=0)
    var rcomp = List[Int](length=tpb, fill=0)

    var llen = 0
    var rlen = 0
    var minlen = 0

    while loffset < part and roffset < end:
        for tid in range(tpb):
            var loff = loffset + tid
            var roff = roffset + tid
            if llen == minlen:
                if loff < part:
                    lflag[tid] = 1 if dataset.data[
                        unsafe_offset = col_base + Int(row_ids[unsafe_offset=loff])
                    ] > quesval else 0
                else:
                    lflag[tid] = 0
            if rlen == minlen:
                if roff < end:
                    rflag[tid] = 1 if dataset.data[
                        unsafe_offset = col_base + Int(row_ids[unsafe_offset=roff])
                    ] <= quesval else 0
                else:
                    rflag[tid] = 0

        var lrun = 0
        var rrun = 0
        for tid in range(tpb):
            lidx[tid] = lrun
            lrun += lflag[tid]
            ridx[tid] = rrun
            rrun += rflag[tid]
        llen = lrun
        rlen = rrun

        minlen = llen if llen < rlen else rlen

        for tid in range(tpb):
            if lflag[tid] != 0:
                lcomp[lidx[tid]] = loffset + tid
            if rflag[tid] != 0:
                rcomp[ridx[tid]] = roffset + tid

        for tid in range(tpb):
            if lidx[tid] < minlen:
                lflag[tid] = 0
            if ridx[tid] < minlen:
                rflag[tid] = 0
        if llen == minlen:
            loffset += tpb
        if rlen == minlen:
            roffset += tpb

        for tid in range(minlen):
            var a = row_ids[unsafe_offset= lcomp[tid]]
            var b = row_ids[unsafe_offset= rcomp[tid]]
            row_ids[unsafe_offset= lcomp[tid]] = b
            row_ids[unsafe_offset= rcomp[tid]] = a


comptime FEATURE_THRESHOLD: Float32 = 1e-7
"""`sklearn/tree/_partitioner.pxd:13`, `cdef const float32_t FEATURE_THRESHOLD = 1e-7`."""


@fieldwise_init
struct FeatureRange(ImplicitlyCopyable, Movable):
    """The output of the range pass: one feature's extent over one node."""

    var min_value: Float32
    var max_value: Float32
    var n_missing: Int32
    """NaN count."""


def node_feature_min_max(
    dataset: Dataset, work_item: NodeWorkItem, col: Int32
) -> FeatureRange:
    """Min and max of one feature over one node's rows."""
    var begin = Int(work_item.instances.begin)
    var end = begin + Int(work_item.instances.count)
    var col_base = Int(col) * Int(dataset.m)

    var min_value = Float32(0.0)
    var max_value = Float32(0.0)
    var n_missing = 0
    var seen_non_missing = False

    for p in range(begin, end):
        var row = Int(dataset.row_ids[unsafe_offset=p])
        var v = dataset.data[unsafe_offset = col_base + row]
        if v != v:  # isnan, without importing one
            n_missing += 1
        elif not seen_non_missing:
            min_value = v
            max_value = v
            seen_non_missing = True
        elif v < min_value:
            min_value = v
        elif v > max_value:
            max_value = v

    if not seen_non_missing:
        return FeatureRange(1.0, -1.0, Int32(n_missing))
    return FeatureRange(min_value, max_value, Int32(n_missing))


def node_feature_is_constant(extent: FeatureRange, n_rows: Int32) -> Bool:
    """Whether this feature is constant over this node, sklearn's test."""
    if n_rows == extent.n_missing:
        return True
    return (
        extent.max_value <= extent.min_value + FEATURE_THRESHOLD
        and extent.n_missing == 0
    )


def draw_threshold(key: SplitKey, extent: FeatureRange) -> Float32:
    """One threshold for one (node, feature), keyed rather than streamed."""
    var gen = key.generator()
    var threshold = uniform_float(gen, extent.min_value, extent.max_value)
    if threshold == extent.max_value:
        return extent.min_value
    return threshold


def draw_threshold_raw(key: SplitKey, extent: FeatureRange) -> Float32:
    """`draw_threshold` WITHOUT sklearn's `== max` guard."""
    var gen = key.generator()
    return uniform_float(gen, extent.min_value, extent.max_value)


# ============================================================================

from std.atomic import Atomic, Ordering
from std.memory import bitcast
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv, inf
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import min as block_min
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

comptime BUILD_MODE = GLOBAL_NUMERIC_MODE
"""The repo-wide numeric mode (`checks/numerics.mojo`)."""

from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    WorkloadInfo,
)


comptime RANGE_SAB_NONE = 0
"""No sabotage."""

comptime RANGE_SAB_ROW_MAJOR = 1
"""Index the feature matrix as `row * N + col` instead of cuML's column-major `col * M + row` (`dataset.h:24`)."""

comptime RANGE_SAB_NO_ROW_IDS = 2
"""Drop the `row_ids` indirection and treat the slot index as the row id."""

comptime RANGE_SAB_BLOCK0_ONLY = 3
"""Publish only from `offset_blockid == 0`, so a large node sees just the first block's slice."""

comptime RANGE_SAB_NAN_AS_VALUE = 4
"""Skip the `v != v` test, so NaN is neither counted nor diverted."""

comptime RANGE_SAB_NO_SENTINEL = 5
"""Publish `(+inf, -inf)` for an empty cell instead of the host's `(1.0, -1.0)` sentinel."""

comptime RANGE_SAB_EMPTY_NOT_IDENTITY = 7
"""An empty block publishes `range_key(0.0)` rather than its `+inf`/`-inf` identities."""

comptime RANGE_SAB_SIGN_UNFLIPPED = 6
"""Skip `range_key`'s NEGATIVE branch: set the sign bit unconditionally instead of inverting a negative's bits."""


def range_key(v: Float32) -> UInt32:
    """A float as an unsigned key whose INTEGER order is the float's order."""
    var b = rebind[UInt32](v.to_bits())
    if (b & UInt32(0x80000000)) != 0:
        return ~b
    return b | UInt32(0x80000000)


def range_unkey(k: UInt32) -> Float32:
    """The exact inverse of `range_key`."""
    if (k & UInt32(0x80000000)) != 0:
        return bitcast[DType.float32](k ^ UInt32(0x80000000))
    return bitcast[DType.float32](~k)


comptime RANGE_KEY_MIN_SEED: UInt32 = 0xFF800000
"""`range_key(+inf)`: the largest key, so an untouched cell loses every min."""
comptime RANGE_KEY_MAX_SEED: UInt32 = 0x007FFFFF
"""`range_key(-inf)`: the smallest key, so an untouched cell loses every max."""


def node_feature_range_decode_kernel(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    minkey: MutPointer[UInt32, MutAnyOrigin],
    maxkey: MutPointer[UInt32, MutAnyOrigin],
    len: Int32,
    sabotage_in: Int32,
):
    """Keys back into the `(min, max)` floats every later pass reads."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        var lo = range_unkey(minkey[unsafe_offset=idx])
        var hi = range_unkey(maxkey[unsafe_offset=idx])
        if lo > hi and sabotage_in != RANGE_SAB_NO_SENTINEL:
            out_min[unsafe_offset=idx] = Float32(1.0)
            out_max[unsafe_offset=idx] = Float32(-1.0)
        else:
            out_min[unsafe_offset=idx] = lo
            out_max[unsafe_offset=idx] = hi
        idx += stride


def node_feature_range_kernel[
    TPB: Int
](
    out_minkey: MutPointer[UInt32, MutAnyOrigin],
    out_maxkey: MutPointer[UInt32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    n_sampled_cols_in: Int32,
    sabotage_in: Int32,
):
    """Min, max and NaN count of one feature over one node's rows, on device."""
    var wb = Int(block_idx.x)
    var fslot = Int(block_idx.y)
    var nid = Int(workload_info[unsafe_offset=wb].nodeid)
    var offset_blockid = Int(workload_info[unsafe_offset=wb].offset_blockid)
    var num_blocks = Int(workload_info[unsafe_offset=wb].num_blocks)

    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)

    var m = Int(m_in)
    var n = Int(n_in)
    var n_sampled_cols = Int(n_sampled_cols_in)
    var sabotage = Int(sabotage_in)

    var col = Int(colids[unsafe_offset = nid * n_sampled_cols + fslot])

    var end = range_start + range_len
    var stride = TPB * num_blocks
    var tid = Int(thread_idx.x) + offset_blockid * TPB

    var local_min = inf[DType.float32]()
    var local_max = -inf[DType.float32]()
    var local_missing = Int32(0)

    var col_offset = col * m
    var i = range_start + tid
    while i < end:
        var slot_row = Int(row_ids[unsafe_offset=i])
        if sabotage == RANGE_SAB_NO_ROW_IDS:
            slot_row = i
        var v: Float32
        if sabotage == RANGE_SAB_ROW_MAJOR:
            v = data[unsafe_offset = slot_row * n + col]
        else:
            v = data[unsafe_offset = col_offset + slot_row]

        if v != v and sabotage != RANGE_SAB_NAN_AS_VALUE:
            local_missing += 1
        else:
            comptime if SEARCH_ROWS_PER_THREAD > 1:
                if range_key(v) < range_key(local_min):
                    local_min = v
                if range_key(v) > range_key(local_max):
                    local_max = v
            else:
                if v < local_min:
                    local_min = v
                if v > local_max:
                    local_max = v
        i += stride
        comptime if SEARCH_SAB_RPT_TAIL_DROP:
            break

    var kmin: UInt32
    var kmax: UInt32
    comptime if BUILD_MODE == NUMERIC_IDENTICAL:
        kmin = block_min[block_size=TPB](range_key(local_min))
        barrier()
        kmax = block_max[block_size=TPB](range_key(local_max))
        barrier()
    else:
        var blk_min = block_min[block_size=TPB](local_min)
        barrier()
        var blk_max = block_max[block_size=TPB](local_max)
        barrier()
        kmin = range_key(blk_min)
        kmax = range_key(blk_max)
    var blk_missing = block_sum[block_size=TPB](local_missing)
    barrier()

    if Int(thread_idx.x) != 0:
        return
    if sabotage == RANGE_SAB_BLOCK0_ONLY and offset_blockid != 0:
        return

    var slot = nid * n_sampled_cols + fslot

    var empty_block = kmin > kmax
    if sabotage == RANGE_SAB_SIGN_UNFLIPPED and not empty_block:
        kmin = rebind[UInt32](range_unkey(kmin).to_bits()) | UInt32(0x80000000)
        kmax = rebind[UInt32](range_unkey(kmax).to_bits()) | UInt32(0x80000000)
    if sabotage == RANGE_SAB_EMPTY_NOT_IDENTITY and empty_block:
        kmin = range_key(Float32(0.0))
        kmax = range_key(Float32(0.0))
    Atomic.min(out_minkey.unsafe_offset(slot), kmin)
    Atomic.max(out_maxkey.unsafe_offset(slot), kmax)
    _ = Atomic.fetch_add(out_n_missing.unsafe_offset(slot), blk_missing)
    _ = Atomic.fetch_add(out_n_merges.unsafe_offset(slot), Int32(1))


@always_inline
def range_init_seed_at(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_minkey: MutPointer[UInt32, MutAnyOrigin],
    out_maxkey: MutPointer[UInt32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    idx: Int,
):
    """Seed ONE range cell: `node_feature_range_init_kernel`'s per-index body."""
    out_min[unsafe_offset=idx] = Float32(1.0)
    out_max[unsafe_offset=idx] = Float32(-1.0)
    out_minkey[unsafe_offset=idx] = RANGE_KEY_MIN_SEED
    out_maxkey[unsafe_offset=idx] = RANGE_KEY_MAX_SEED
    out_n_missing[unsafe_offset=idx] = Int32(0)
    out_n_merges[unsafe_offset=idx] = Int32(0)


def node_feature_range_init_kernel(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_minkey: MutPointer[UInt32, MutAnyOrigin],
    out_maxkey: MutPointer[UInt32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    len: Int32,
):
    """Seed every output cell with the EMPTY range, before any merge."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        range_init_seed_at(
            out_min,
            out_max,
            out_minkey,
            out_maxkey,
            out_n_missing,
            out_n_merges,
            idx,
        )
        idx += stride


def node_nonconstant_flag_kernel(
    out_flag: MutPointer[Int32, MutAnyOrigin],
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    n_cells: Int32,
    n_sampled_cols: Int32,
):
    """Per node: did ANY of its sampled columns vary?"""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(n_cells):
        var nid = idx // Int(n_sampled_cols)
        var extent = FeatureRange(
            out_min[unsafe_offset=idx],
            out_max[unsafe_offset=idx],
            out_n_missing[unsafe_offset=idx],
        )
        if not node_feature_is_constant(
            extent, work_items[unsafe_offset=nid].instances.count
        ):
            _ = Atomic.fetch_add(out_flag.unsafe_offset(nid), Int32(1))
        idx += stride


def feature_range_at(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    nid: Int,
    fslot: Int,
    n_sampled_cols: Int,
) -> FeatureRange:
    """Reassemble one cell of the kernel's output as a `FeatureRange`."""
    var slot = nid * n_sampled_cols + fslot
    return FeatureRange(
        out_min[unsafe_offset=slot],
        out_max[unsafe_offset=slot],
        out_n_missing[unsafe_offset=slot],
    )


@fieldwise_init
struct WorkloadPlan(Copyable, Movable):
    """What `updateWorkloadInfo` returns, as a struct because Mojo will not return their `std::pair` plus an out-parameter."""

    var info: List[WorkloadInfo]
    """`n_blocks_dimx` entries, one per threadblock along x."""

    var n_blocks_dimx: Int
    """`gridDim.x` for the range kernel."""

    var n_large_nodes: Int
    """Nodes needing more than one block."""


def build_workload_info(
    work_items: List[NodeWorkItem], tpb: Int = TPB_DEFAULT
) -> WorkloadPlan:
    """Flatten a ragged batch of nodes into a list of threadblocks."""
    var info = List[WorkloadInfo]()
    var n_large_nodes = 0
    var n_blocks_dimx = 0
    for i in range(len(work_items)):
        var count = Int(work_items[i].instances.count)
        var n_blocks_per_node = ceildiv(count, tpb)
        if n_blocks_per_node < 1:
            n_blocks_per_node = 1
        if n_blocks_per_node > 1:
            n_large_nodes += 1
        for b in range(n_blocks_per_node):
            info.append(
                WorkloadInfo(
                    Int32(i),
                    Int32(n_large_nodes - 1),
                    Int32(b),
                    Int32(n_blocks_per_node),
                )
            )
        n_blocks_dimx += n_blocks_per_node
    return WorkloadPlan(info^, n_blocks_dimx, n_large_nodes)


# ============================================================================
#     invariant than recomputing something that cannot vary.


comptime SCORE_STATUS_UNVISITED: Int32 = 0
"""The initialized value, and it is not a legal outcome."""

comptime SCORE_STATUS_SCORED: Int32 = 1
"""Reached `_splitter.pyx:691`: not constant, not refused, not rejected."""

comptime SCORE_STATUS_CONSTANT: Int32 = 2
"""`_splitter.pyx:613-618` said constant, so `:619-626` skipped it."""

comptime SCORE_STATUS_MISSING_REFUSED: Int32 = 3
"""`n_missing != 0`."""

comptime SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF: Int32 = 4
"""`_splitter.pyx:664-666`, a `continue` and NOT a redraw."""

comptime SCORE_STATUS_REGRESSION_REFUSED: Int32 = 5
"""REGRESSION ONLY, and OURS: a partial sum larger than the fixed-point slot, so the published key would wrap."""

comptime SCORE_STATUS_PURE_NODE: Int32 = 6
"""DEVIATION 216's companion (sklearn `_tree.pyx:240`): the node is PURE -- one class holds every row, Gini exactly 0 -- so it is a leaf before any candidate can win."""


comptime SCORE_MAX_ACC_DEFAULT: Int = 16
"""Default comptime bound on `n_acc` (classes, or 1 for regression)."""


comptime SCORE_MAX_ROWS_EXACT: Int = 1 << 21
"""Rows in ONE NODE above which the published `Int64` Gini numerator wraps."""


def score_row_bound_ok(row_count: Int) -> Bool:
    """Is the published classification rational exact at this node size?"""
    if row_count <= 0:
        return True
    var n = Int128(row_count)
    var worst = (n * n * n) // Int128(4)
    return worst <= (Int128(1) << Int128(63)) - Int128(1)


# ---------------------------------------------------------------------------

comptime REGRESSION_SUM_BITS: Int = 30
"""The fixed-point slot a partial label sum must fit, `2^REGRESSION_SUM_BITS`."""

comptime REGRESSION_KEY_BITS: Int = 31
"""Bits `|A| >> j` is allowed to occupy, so `num = (|A| >> j)^2 <= 2^62`."""


def classification_key_shift(row_count: Int) -> Int:
    """`s`, the NODE-UNIFORM right shift on the classification squared sums."""
    var bits = 0
    var acc = 1
    while acc < row_count:
        acc *= 2
        bits += 1
    var s = 3 * bits - 64
    return s if s > 0 else 0


def float_gain_key(gain: Float32) -> Int64:
    """The order-preserving SIGN-MAGNITUDE map of a Float32 gain onto `Int64`, the ENTROPY cell's `num` (with `den = 1`)."""
    var bits = Int64(Int(gain.to_bits()))
    var mag = bits & Int64(0x7FFFFFFF)
    if (bits & Int64(0x80000000)) != 0:
        return -mag
    return mag


def regression_key_shift(row_count: Int) -> Int:
    """`j`, the NODE-UNIFORM right shift on `|A|`."""
    var bits = 0
    var acc = 1
    while acc < row_count:
        acc *= 2
        bits += 1
    var j = bits + REGRESSION_SUM_BITS - REGRESSION_KEY_BITS
    return j if j > 0 else 0


def regression_key_bound_ok(row_count: Int, max_abs_sum: Int) -> Bool:
    """Does the published `Int64` numerator hold at this node size and this largest partial-sum magnitude?"""
    if row_count <= 0:
        return True
    var j = regression_key_shift(row_count)
    var worst = Int128(row_count) * Int128(max_abs_sum)
    var scaled = worst >> Int128(j)
    return scaled * scaled <= (Int128(1) << Int128(63)) - Int128(1)


def mse_gain_from_exact_totals(
    sum_left: Int64,
    sum_total: Int64,
    n_left: Int,
    n_right: Int,
    inv_scale: Float64,
) -> Float32:
    """cuML's `MSEObjectiveFunction::GainPerSplit` for the winning candidate, from EXACT INTEGERS, in the LABEL's own units."""
    if n_left <= 0 or n_right <= 0:
        return Float32(0.0)
    var nl = Float64(n_left)
    var nr = Float64(n_right)
    var n = nl + nr
    var sum_right = sum_total - sum_left
    var a = Float64(sum_left) * nr - Float64(sum_right) * nl
    var t = a / n * inv_scale
    return Float32(0.5 * t * t / (n * nl * nr))


comptime SCORE_SAB_NONE: Int32 = 0
"""No sabotage."""

comptime SCORE_SAB_CONSTANT_STRICT: Int32 = 1
"""Test `max < min + FEATURE_THRESHOLD` instead of `max <= min + ...` (`_splitter.pyx:616-617`)."""

comptime SCORE_SAB_NO_MAX_GUARD: Int32 = 2
"""Drop sklearn's `:653-654` `threshold == max -> min` guard, i.e."""

comptime SCORE_SAB_SIDE_INVERTED: Int32 = 3
"""Accumulate a row into the LEFT child when `value > threshold`."""

comptime SCORE_SAB_STRICT_LESS: Int32 = 4
"""`value < threshold` goes left, so a row EQUAL to the threshold goes right."""

comptime SCORE_SAB_NO_ROW_IDS: Int32 = 5
"""Drop the `row_ids` indirection and treat the slot index as the row id, for BOTH the feature value and the label."""

comptime SCORE_SAB_SCALE_X2: Int32 = 6
"""Regression: accumulate `2 * labels_q[row]`, i.e."""

comptime SCORE_SAB_FLOAT_ACCUM: Int32 = 7
"""Regression: accumulate the label sum in `Float32` -- per thread AND through the block reduction -- and truncate once at the publish, instead of in fixed point."""

comptime SCORE_SAB_BLOCK0_ONLY: Int32 = 8
"""Publish only from `offset_blockid == 0`, so a large node sees one block's slice."""

comptime SCORE_SAB_TOTAL_IS_LEFT: Int32 = 9
"""Accumulate the node TOTALS over left-going rows only, so `total - left` -- cuML's own recovery of the right child (`objectives.cuh:72-73`, DEVIATION 143) -- is zero."""

comptime SCORE_SAB_REG_NO_SHIFT: Int32 = 10
"""Regression: publish `|A|^2` with NO shift, i.e."""

comptime SCORE_SAB_REG_NO_CENTER: Int32 = 11
"""Regression: `A = sum_L * n_R` instead of `sum_L * n_R - sum_R * n_L`, i.e."""

comptime SCORE_SAB_REG_DEN_ONE: Int32 = 12
"""Regression: publish `den = 1`, so the key becomes `A'^2` and the `n_L n_R` weighting -- the difference between cuML's gain and the squared mean gap -- disappears."""

comptime SCORE_SAB_REG_NO_SLOT_GUARD: Int32 = 13
"""Regression: skip the fixed-point slot precondition of DEVIATION BLOCK 193, so a partial sum outside the slot publishes a WRAPPED numerator instead of `SCORE_STATUS_REGRESSION_REFUSED`."""


def regression_key(
    sum_left: Int64,
    sum_total: Int64,
    n_left: Int,
    n_right: Int,
    row_count: Int,
    sabotage: Int32,
    mut out_num: Int64,
    mut out_den: Int64,
) -> Bool:
    """cuML's MSE gain as an exact rational in `Int64`."""
    var sum_right = sum_total - sum_left
    if sabotage != SCORE_SAB_REG_NO_SLOT_GUARD:
        var abs_left = sum_left if sum_left >= 0 else -sum_left
        var abs_right = sum_right if sum_right >= 0 else -sum_right
        var slot = Int64(1) << Int64(REGRESSION_SUM_BITS)
        if abs_left > slot or abs_right > slot:
            out_num = Int64(0)
            out_den = Int64(0)
            return False

    var nl = Int64(n_left)
    var nr = Int64(n_right)
    var a = sum_left * nr - sum_right * nl
    if sabotage == SCORE_SAB_REG_NO_CENTER:
        a = sum_left * nr
    if a < 0:
        a = -a
    var j = regression_key_shift(row_count)
    if sabotage == SCORE_SAB_REG_NO_SHIFT:
        j = 0
    # key is invariant under negating every label. DEVIATION BLOCK 191.
    var scaled = a >> Int64(j)
    out_num = scaled * scaled
    out_den = nl * nr
    if sabotage == SCORE_SAB_REG_DEN_ONE:
        out_den = Int64(1)
    return True




def draw_threshold_device(
    key: SplitKey, extent: FeatureRange, guarded: Bool = True
) -> Float32:
    """`draw_threshold`'s draw, written so that NO backend can re-round it."""
    var gen = key.generator()
    var res = gen.next_float()
    var lo = ftz(extent.min_value)
    var hi = ftz(extent.max_value)
    var span = ftz(hi - lo)
    var threshold = ftz(res.fma(span, lo))
    if guarded and threshold == hi:
        return lo
    return threshold


@fieldwise_init
struct ScoredCandidate(Copyable, Movable):
    """One (node, feature) cell of the score pass, as one value."""

    var status: Int32
    var threshold: Float32
    var n_left: Int32
    var n_total: Int32
    var acc_left: List[Int32]
    var acc_total: List[Int32]
    var gini_num: Int64
    var gini_den: Int64

    def matches(self, other: Self) -> Bool:
        """Exact equality, with the threshold on FLOAT BIT PATTERNS."""
        if self.status != other.status:
            return False
        if self.threshold.to_bits() != other.threshold.to_bits():
            return False
        if self.n_left != other.n_left or self.n_total != other.n_total:
            return False
        if self.gini_num != other.gini_num or self.gini_den != other.gini_den:
            return False
        if len(self.acc_left) != len(other.acc_left):
            return False
        if len(self.acc_total) != len(other.acc_total):
            return False
        for k in range(len(self.acc_left)):
            if self.acc_left[k] != other.acc_left[k]:
                return False
        for k in range(len(self.acc_total)):
            if self.acc_total[k] != other.acc_total[k]:
                return False
        return True


def empty_scored_candidate(status: Int32, n_acc: Int) -> ScoredCandidate:
    """A cell nothing was accumulated into: `_empty_candidate`'s shape (`host_splitter.mojo:418-440`), which reports `threshold = 0.0` and zero accumulators for a candidate that never reached the draw."""
    return ScoredCandidate(
        status,
        Float32(0.0),
        Int32(0),
        Int32(0),
        List[Int32](length=n_acc, fill=Int32(0)),
        List[Int32](length=n_acc, fill=Int32(0)),
        Int64(0),
        Int64(0),
    )


def node_feature_score_host(
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    m: Int,
    range_start: Int,
    range_len: Int,
    col: Int,
    extent: FeatureRange,
    key: SplitKey,
    n_acc: Int,
    is_classification: Bool,
    min_samples_leaf: Int,
    device_draw: Bool = True,
) -> ScoredCandidate:
    """Steps 2, 3 and 4 of DEVIATION 137 for ONE (node, feature), sequentially."""
    if extent.n_missing != Int32(0):
        return empty_scored_candidate(SCORE_STATUS_MISSING_REFUSED, n_acc)
    if node_feature_is_constant(extent, Int32(range_len)):
        return empty_scored_candidate(SCORE_STATUS_CONSTANT, n_acc)

    var threshold: Float32
    if device_draw:
        threshold = draw_threshold_device(key, extent)
    else:
        threshold = draw_threshold(key, extent)

    var acc_left = List[Int32](length=n_acc, fill=Int32(0))
    var acc_total = List[Int32](length=n_acc, fill=Int32(0))
    var n_left = 0
    var n_total = 0
    var col_offset = col * m

    for p in range(range_start, range_start + range_len):
        var row = Int(row_ids[unsafe_offset=p])
        var v = data[unsafe_offset = col_offset + row]
        var lab = Int(labels_q[unsafe_offset=row])
        n_total += 1
        var goes_left = v <= threshold
        if is_classification:
            if lab >= 0 and lab < n_acc:
                acc_total[lab] += 1
                if goes_left:
                    acc_left[lab] += 1
        else:
            acc_total[0] += Int32(lab)
            if goes_left:
                acc_left[0] += Int32(lab)
        if goes_left:
            n_left += 1

    var n_right = n_total - n_left
    var status = SCORE_STATUS_SCORED
    if (
        n_left < min_samples_leaf
        or n_right < min_samples_leaf
        or n_left == 0
        or n_right == 0
    ):
        status = SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF
    if status == SCORE_STATUS_SCORED and is_classification and n_total > 0:
        for kc in range(n_acc):
            if Int(acc_total[kc]) == n_total:
                status = SCORE_STATUS_PURE_NODE

    var num = Int64(0)
    var den = Int64(0)
    if status == SCORE_STATUS_SCORED and is_classification:
        var sq_left = Int64(0)
        var sq_right = Int64(0)
        for k in range(n_acc):
            var lv = Int64(Int(acc_left[k]))
            var rv = Int64(Int(acc_total[k] - acc_left[k]))
            sq_left += lv * lv
            sq_right += rv * rv
        var nl = Int64(n_left)
        var nr = Int64(n_right)
        var cs = Int64(classification_key_shift(range_len))
        num = (sq_left >> cs) * nr + (sq_right >> cs) * nl
        den = nl * nr
    elif status == SCORE_STATUS_SCORED:
        if not regression_key(
            Int64(Int(acc_left[0])),
            Int64(Int(acc_total[0])),
            n_left,
            n_right,
            range_len,
            SCORE_SAB_NONE,
            num,
            den,
        ):
            status = SCORE_STATUS_REGRESSION_REFUSED

    return ScoredCandidate(
        status,
        threshold,
        Int32(n_left),
        Int32(n_total),
        acc_left^,
        acc_total^,
        num,
        den,
    )


def scored_candidate_at(
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    nid: Int,
    fslot: Int,
    n_sampled_cols: Int,
    n_acc: Int,
) -> ScoredCandidate:
    """Reassemble one cell of the kernels' output."""
    var slot = nid * n_sampled_cols + fslot
    var left = List[Int32]()
    var total = List[Int32]()
    for k in range(n_acc):
        left.append(out_acc_left[unsafe_offset = slot * n_acc + k])
        total.append(out_acc_total[unsafe_offset = slot * n_acc + k])
    return ScoredCandidate(
        out_status[unsafe_offset=slot],
        out_threshold[unsafe_offset=slot],
        out_n_left[unsafe_offset=slot],
        out_n_total[unsafe_offset=slot],
        left^,
        total^,
        out_gini_num[unsafe_offset=slot],
        out_gini_den[unsafe_offset=slot],
    )


from std.memory import stack_allocation


@always_inline
def score_init_seed_cell_at(
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    out_n_blocks: MutPointer[Int32, MutAnyOrigin],
    idx: Int,
):
    """Seed ONE score cell: `node_feature_score_init_kernel`'s per-index body over the seven per-cell arrays."""
    out_status[unsafe_offset=idx] = SCORE_STATUS_UNVISITED
    out_threshold[unsafe_offset=idx] = Float32(0.0)
    out_n_left[unsafe_offset=idx] = Int32(0)
    out_n_total[unsafe_offset=idx] = Int32(0)
    out_gini_num[unsafe_offset=idx] = Int64(0)
    out_gini_den[unsafe_offset=idx] = Int64(0)
    out_n_blocks[unsafe_offset=idx] = Int32(0)


@always_inline
def score_init_seed_acc_at(
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    idx: Int,
):
    """Seed ONE class-accumulator cell: the second loop of `node_feature_score_init_kernel`, extracted for the same DEVIATION 470 reason as `score_init_seed_cell_at` above."""
    out_acc_left[unsafe_offset=idx] = Int32(0)
    out_acc_total[unsafe_offset=idx] = Int32(0)


def node_feature_score_init_kernel(
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    out_n_blocks: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    n_cells: Int32,
    n_acc_cells: Int32,
):
    """Seed every output cell, before any accumulation."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var i = idx
    while i < Int(n_cells):
        score_init_seed_cell_at(
            out_status,
            out_threshold,
            out_n_left,
            out_n_total,
            out_gini_num,
            out_gini_den,
            out_n_blocks,
            i,
        )
        i += stride
    var j = idx
    while j < Int(n_acc_cells):
        score_init_seed_acc_at(out_acc_left, out_acc_total, j)
        j += stride


# ===========================================================================

comptime PHASE_SETUP_TPB = 256
"""The fused seeders' block width."""


def phase_setup_a_kernel(
    out_report: MutPointer[Int32, MutAnyOrigin],
    n_report: Int32,
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_minkey: MutPointer[UInt32, MutAnyOrigin],
    out_maxkey: MutPointer[UInt32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    n_range_cells: Int32,
    out_nonconst: MutPointer[Int32, MutAnyOrigin],
    n_nonconst: Int32,
):
    """DEVIATION 470, half A: the range half of the search cycle's setup -- the `d_samp_report` memset, `node_feature_range_init_kernel` and the `d_nonconst` memset -- as one launch, enqueued (with half B right behind it) after `stage_batch` and before the feature sampler."""
    var extent = Int(n_report)
    if Int(n_range_cells) > extent:
        extent = Int(n_range_cells)
    if Int(n_nonconst) > extent:
        extent = Int(n_nonconst)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < extent:
        if idx < Int(n_report):
            out_report[unsafe_offset=idx] = SAMPLER_UNVISITED
        if idx < Int(n_range_cells):
            range_init_seed_at(
                out_min,
                out_max,
                out_minkey,
                out_maxkey,
                out_n_missing,
                out_n_merges,
                idx,
            )
        if idx < Int(n_nonconst):
            out_nonconst[unsafe_offset=idx] = Int32(0)
        idx += stride


def phase_setup_b_kernel(
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    out_n_blocks: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    n_score_cells: Int32,
    n_acc_cells: Int32,
    out_mutex: MutPointer[Int32, MutAnyOrigin],
    n_mutex: Int32,
    out_r_quesval: MutPointer[Float32, MutAnyOrigin],
    out_r_colid: MutPointer[Int32, MutAnyOrigin],
    out_r_metric: MutPointer[Float32, MutAnyOrigin],
    out_r_nleft: MutPointer[Int32, MutAnyOrigin],
    out_r_num: MutPointer[Int64, MutAnyOrigin],
    out_r_den: MutPointer[Int64, MutAnyOrigin],
    out_r_valid: MutPointer[Int32, MutAnyOrigin],
    out_r_merges: MutPointer[Int32, MutAnyOrigin],
    out_r_warps: MutPointer[Int32, MutAnyOrigin],
    n_reduce_nodes: Int32,
):
    """DEVIATION 470, half B: the score/reduce half of the search cycle's setup -- `node_feature_score_init_kernel`, the `r_mx` memset and `split_reduce_init_kernel` -- as one launch, enqueued immediately after half A."""
    var extent = Int(n_score_cells)
    if Int(n_acc_cells) > extent:
        extent = Int(n_acc_cells)
    if Int(n_mutex) > extent:
        extent = Int(n_mutex)
    if Int(n_reduce_nodes) > extent:
        extent = Int(n_reduce_nodes)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < extent:
        if idx < Int(n_score_cells):
            score_init_seed_cell_at(
                out_status,
                out_threshold,
                out_n_left,
                out_n_total,
                out_gini_num,
                out_gini_den,
                out_n_blocks,
                idx,
            )
        if idx < Int(n_acc_cells):
            score_init_seed_acc_at(out_acc_left, out_acc_total, idx)
            comptime if is_defined["MOJOLEARN_ET_SAB_PHASE_SETUP"]():
                out_acc_left[unsafe_offset=idx] = Int32(idx & 1)
        if idx < Int(n_mutex):
            out_mutex[unsafe_offset=idx] = Int32(0)
        if idx < Int(n_reduce_nodes):
            split_reduce_seed_at(
                out_r_quesval,
                out_r_colid,
                out_r_metric,
                out_r_nleft,
                out_r_num,
                out_r_den,
                out_r_valid,
                out_r_merges,
                out_r_warps,
                idx,
            )
        idx += stride


def node_feature_score_kernel[
    TPB: Int, MAX_ACC: Int, CLASSIFICATION: Bool
](
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    out_n_blocks: MutPointer[Int32, MutAnyOrigin],
    in_min: MutPointer[Float32, MutAnyOrigin],
    in_max: MutPointer[Float32, MutAnyOrigin],
    in_n_missing: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_sampled_cols_in: Int32,
    n_acc_in: Int32,
    seed: UInt64,
    sabotage_in: Int32,
):
    """Steps 2, 3 and 4 of DEVIATION 137: skip, draw, and accumulate."""
    var wb = Int(block_idx.x)
    var fslot = Int(block_idx.y)
    var nid = Int(workload_info[unsafe_offset=wb].nodeid)
    var offset_blockid = Int(workload_info[unsafe_offset=wb].offset_blockid)
    var num_blocks = Int(workload_info[unsafe_offset=wb].num_blocks)
    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)
    var node_id = UInt32(Int(work_items[unsafe_offset=nid].idx))

    var m = Int(m_in)
    var n_sampled_cols = Int(n_sampled_cols_in)
    var n_acc = Int(n_acc_in)
    var sabotage = sabotage_in

    var col = Int(colids[unsafe_offset = nid * n_sampled_cols + fslot])
    var slot = nid * n_sampled_cols + fslot

    if n_acc > MAX_ACC or n_acc < 1:
        return

    var extent = FeatureRange(
        in_min[unsafe_offset=slot],
        in_max[unsafe_offset=slot],
        in_n_missing[unsafe_offset=slot],
    )
    if extent.n_missing != Int32(0):
        return
    var constant: Bool
    if sabotage == SCORE_SAB_CONSTANT_STRICT:
        constant = Int32(range_len) == extent.n_missing or (
            extent.max_value < extent.min_value + FEATURE_THRESHOLD
            and extent.n_missing == 0
        )
    else:
        constant = node_feature_is_constant(extent, Int32(range_len))
    if constant:
        return

    var key = key_for(
        seed,
        tree_ids[unsafe_offset=nid].cast[DType.uint32](),
        node_id,
        UInt32(col),
    )
    var threshold = draw_threshold_device(
        key, extent, sabotage != SCORE_SAB_NO_MAX_GUARD
    )

    var priv_left = stack_allocation[MAX_ACC, Scalar[DType.int32]]()
    var priv_total = stack_allocation[MAX_ACC, Scalar[DType.int32]]()
    for k in range(MAX_ACC):
        priv_left[unsafe_offset=k] = Int32(0)
        priv_total[unsafe_offset=k] = Int32(0)

    var col_offset = col * m
    var end = range_start + range_len
    var stride = TPB * num_blocks
    var tid = Int(thread_idx.x) + offset_blockid * TPB
    var n_left = Int32(0)
    var n_seen = Int32(0)
    var f_left = Float32(0.0)
    var f_total = Float32(0.0)

    var i = range_start + tid
    while i < end:
        var row = Int(row_ids[unsafe_offset=i])
        if sabotage == SCORE_SAB_NO_ROW_IDS:
            row = i
        var v = data[unsafe_offset = col_offset + row]
        var lab = Int(labels_q[unsafe_offset=row])
        n_seen += 1

        var goes_left: Bool
        if sabotage == SCORE_SAB_SIDE_INVERTED:
            goes_left = v > threshold
        elif sabotage == SCORE_SAB_STRICT_LESS:
            goes_left = v < threshold
        else:
            goes_left = v <= threshold

        var counts_total = True
        if sabotage == SCORE_SAB_TOTAL_IS_LEFT:
            counts_total = goes_left

        comptime if CLASSIFICATION:
            if lab >= 0 and lab < n_acc:
                if counts_total:
                    priv_total[unsafe_offset=lab] += Int32(1)
                if goes_left:
                    priv_left[unsafe_offset=lab] += Int32(1)
        else:
            var q = Int32(lab)
            if sabotage == SCORE_SAB_SCALE_X2:
                q = Int32(2 * lab)
            if sabotage == SCORE_SAB_FLOAT_ACCUM:
                if counts_total:
                    f_total += Float32(lab)
                if goes_left:
                    f_left += Float32(lab)
            else:
                if counts_total:
                    priv_total[unsafe_offset=0] += q
                if goes_left:
                    priv_left[unsafe_offset=0] += q

        if goes_left:
            n_left += 1
        i += stride
        comptime if SEARCH_SAB_RPT_TAIL_DROP:
            break

    var blk_n_left = block_sum[block_size=TPB](n_left)
    barrier()
    var blk_n_seen = block_sum[block_size=TPB](n_seen)
    barrier()

    var publishes = not (
        sabotage == SCORE_SAB_BLOCK0_ONLY and offset_blockid != 0
    )
    if Int(thread_idx.x) == 0 and publishes:
        _ = Atomic.fetch_add(out_n_left.unsafe_offset(slot), blk_n_left)
        _ = Atomic.fetch_add(out_n_total.unsafe_offset(slot), blk_n_seen)
        _ = Atomic.fetch_add(out_n_blocks.unsafe_offset(slot), Int32(1))

    comptime if not CLASSIFICATION:
        if sabotage == SCORE_SAB_FLOAT_ACCUM:
            var fl = block_sum[block_size=TPB](f_left)
            barrier()
            var ft = block_sum[block_size=TPB](f_total)
            barrier()
            if Int(thread_idx.x) == 0 and publishes:
                _ = Atomic.fetch_add(
                    out_acc_left.unsafe_offset(slot * n_acc), Int32(fl)
                )
                _ = Atomic.fetch_add(
                    out_acc_total.unsafe_offset(slot * n_acc), Int32(ft)
                )
            return

    for k in range(n_acc):
        var bl = block_sum[block_size=TPB](priv_left[unsafe_offset=k])
        barrier()
        var bt = block_sum[block_size=TPB](priv_total[unsafe_offset=k])
        barrier()
        if Int(thread_idx.x) == 0 and publishes:
            _ = Atomic.fetch_add(
                out_acc_left.unsafe_offset(slot * n_acc + k), bl
            )
            _ = Atomic.fetch_add(
                out_acc_total.unsafe_offset(slot * n_acc + k), bt
            )


def node_feature_score_finalize_kernel[
    MAX_ACC: Int, CLASSIFICATION: Bool
](
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    in_min: MutPointer[Float32, MutAnyOrigin],
    in_max: MutPointer[Float32, MutAnyOrigin],
    in_n_missing: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    n_cells_in: Int32,
    n_sampled_cols_in: Int32,
    n_acc_in: Int32,
    seed: UInt64,
    min_samples_leaf_in: Int32,
    sabotage_in: Int32,
):
    """Publish each cell's status, threshold and score."""
    var n_cells = Int(n_cells_in)
    var n_sampled_cols = Int(n_sampled_cols_in)
    var n_acc = Int(n_acc_in)
    var min_samples_leaf = Int(min_samples_leaf_in)
    var sabotage = sabotage_in
    if n_acc > MAX_ACC or n_acc < 1:
        return

    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var slot = idx
    while slot < n_cells:
        var nid = slot // n_sampled_cols
        var fslot = slot % n_sampled_cols
        var range_len = Int(work_items[unsafe_offset=nid].instances.count)
        var node_id = UInt32(Int(work_items[unsafe_offset=nid].idx))
        var col = Int(colids[unsafe_offset = nid * n_sampled_cols + fslot])

        var extent = FeatureRange(
            in_min[unsafe_offset=slot],
            in_max[unsafe_offset=slot],
            in_n_missing[unsafe_offset=slot],
        )
        if extent.n_missing != Int32(0):
            out_status[unsafe_offset=slot] = SCORE_STATUS_MISSING_REFUSED
            slot += stride
            continue
        var constant: Bool
        if sabotage == SCORE_SAB_CONSTANT_STRICT:
            constant = Int32(range_len) == extent.n_missing or (
                extent.max_value < extent.min_value + FEATURE_THRESHOLD
                and extent.n_missing == 0
            )
        else:
            constant = node_feature_is_constant(extent, Int32(range_len))
        if constant:
            out_status[unsafe_offset=slot] = SCORE_STATUS_CONSTANT
            slot += stride
            continue

        var key = key_for(
            seed,
            tree_ids[unsafe_offset=nid].cast[DType.uint32](),
            node_id,
            UInt32(col),
        )
        var threshold = draw_threshold_device(
            key, extent, sabotage != SCORE_SAB_NO_MAX_GUARD
        )
        out_threshold[unsafe_offset=slot] = threshold

        var n_left = Int(out_n_left[unsafe_offset=slot])
        var n_total = Int(out_n_total[unsafe_offset=slot])
        var n_right = n_total - n_left

        if (
            n_left < min_samples_leaf
            or n_right < min_samples_leaf
            or n_left == 0
            or n_right == 0
        ):
            out_status[unsafe_offset=slot] = (
                SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF
            )
            slot += stride
            continue

        comptime if CLASSIFICATION:
            var pure = False
            for kc in range(n_acc):
                if (
                    Int(out_acc_total[unsafe_offset = slot * n_acc + kc])
                    == n_total
                ):
                    pure = True
            if pure and n_total > 0:
                out_status[unsafe_offset=slot] = SCORE_STATUS_PURE_NODE
                slot += stride
                continue

        out_status[unsafe_offset=slot] = SCORE_STATUS_SCORED
        comptime if CLASSIFICATION:
            var sq_left = Int64(0)
            var sq_right = Int64(0)
            for k in range(n_acc):
                var lv = Int64(
                    Int(out_acc_left[unsafe_offset = slot * n_acc + k])
                )
                var rv = Int64(
                    Int(
                        out_acc_total[unsafe_offset = slot * n_acc + k]
                        - out_acc_left[unsafe_offset = slot * n_acc + k]
                    )
                )
                sq_left += lv * lv
                sq_right += rv * rv
            var nl = Int64(n_left)
            var nr = Int64(n_right)
            var cs = Int64(classification_key_shift(range_len))
            out_gini_num[unsafe_offset=slot] = (
                (sq_left >> cs) * nr + (sq_right >> cs) * nl
            )
            out_gini_den[unsafe_offset=slot] = nl * nr
        else:
            var key_num = Int64(0)
            var key_den = Int64(0)
            var key_ok = regression_key(
                Int64(Int(out_acc_left[unsafe_offset = slot * n_acc])),
                Int64(Int(out_acc_total[unsafe_offset = slot * n_acc])),
                n_left,
                n_right,
                range_len,
                sabotage,
                key_num,
                key_den,
            )
            if not key_ok:
                out_status[unsafe_offset=slot] = (
                    SCORE_STATUS_REGRESSION_REFUSED
                )
            out_gini_num[unsafe_offset=slot] = key_num
            out_gini_den[unsafe_offset=slot] = key_den
        slot += stride


# ============================================================================
# THE PARTITION AND THE LEAF PASS, ON THE DEVICE.

from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import prefix_sum as block_prefix_sum

from extratrees.impl.decisiontree.flatnode import (
    NODE_IS_LEAF,
    SparseTreeNode,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    split_not_valid,
)


# ---------------------------------------------------------------------------

comptime PARTITION_UNVISITED: Int32 = -3
"""The seed the CALLER must write into `out_n_iters` before the launch."""

comptime PARTITION_SKIPPED: Int32 = -1
"""`nodeSplitKernel`'s early return, `:100-104`: `SplitNotValid` was true, so NOTHING was partitioned and the node's `row_ids` are untouched."""

comptime PARTITION_OVERRUN: Int32 = -2
"""The derived iteration bound of DEVIATION 176 was reached."""


comptime PART_SAB_NONE: Int32 = 0
"""No sabotage."""

comptime PART_SAB_LEFT_MISFIT_GE: Int32 = 1
"""A LEFT-side misfit becomes `col[row] >= quesval` instead of `> quesval` (`:65`), so a row sitting exactly ON the threshold is dragged right."""

comptime PART_SAB_RIGHT_MISFIT_LT: Int32 = 2
"""A RIGHT-side misfit becomes `col[row] < quesval` instead of `<= quesval` (`:66`), the other half of the same boundary rule and equally invisible with distinct values."""

comptime PART_SAB_NO_SCAN: Int32 = 3
"""Compact at `thread_idx.x` instead of at the block scan's exclusive prefix (`:69-71`, `:75-77`), so flagged threads scatter into holes instead of packing down."""

comptime PART_SAB_UNPAIRED_SWAP: Int32 = 4
"""Write only the LEFT half of the swap (`:86-88`), so a row is duplicated and another is dropped."""

comptime PART_SAB_NO_ROW_IDS: Int32 = 5
"""Drop the `row_ids` indirection in the feature read: `col[row_ids[loff]]` becomes `col[loff]` (`:65-66`)."""

comptime PART_SAB_NO_VALID_GUARD: Int32 = 6
"""Partition even when `SplitNotValid` says not to (`:100-104`)."""


def partition_iteration_bound(range_len: Int, n_left: Int, tpb: Int) -> Int:
    """The proved bound on `partitionSamples`' `while`."""
    var n_right = range_len - n_left
    if n_right < 0:
        n_right = 0
    var nl = n_left
    if nl < 0:
        nl = 0
    return ceildiv(nl, tpb) + ceildiv(n_right, tpb) + 2


def node_split_kernel[
    TPB: Int
](
    row_ids: MutPointer[Int32, MutAnyOrigin],
    out_n_iters: MutPointer[Int32, MutAnyOrigin],
    out_n_swaps: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    m_in: Int32,
    min_impurity_decrease: Float32,
    min_samples_leaf_in: Int32,
    sabotage_in: Int32,
):
    """`nodeSplitKernel` (`:89-107`) and the `partitionSamples` it calls."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sabotage = sabotage_in

    var range_start = Int(work_items[unsafe_offset=b].instances.begin)
    var range_len = Int(work_items[unsafe_offset=b].instances.count)
    var quesval = splits[unsafe_offset=b].quesval
    var colid = splits[unsafe_offset=b].colid
    var best_metric = splits[unsafe_offset=b].best_metric_val
    var n_left = Int(splits[unsafe_offset=b].n_left)

    var invalid = split_not_valid(
        Split(quesval, colid, best_metric, Int32(n_left)),
        min_impurity_decrease,
        min_samples_leaf_in,
        Int32(range_len),
    )
    if sabotage == PART_SAB_NO_VALID_GUARD:
        invalid = False
    if invalid:
        if tid == 0:
            out_n_iters[unsafe_offset=b] = PARTITION_SKIPPED
            out_n_swaps[unsafe_offset=b] = Int32(0)
        return

    var smem = stack_allocation[
        2 * TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    # of turning a sabotage into an out-of-bounds write.
    smem[unsafe_offset=tid] = Int32(range_start)
    smem[unsafe_offset = TPB + tid] = Int32(range_start)
    barrier()

    var col_offset = Int(colid) * Int(m_in)
    var loffset = range_start
    var part = loffset + n_left
    var roffset = part
    var end = range_start + range_len

    var lflag = Int32(0)
    var rflag = Int32(0)
    var llen = 0
    var rlen = 0
    var minlen = 0

    var iters = 0
    var swaps = 0
    var bound = partition_iteration_bound(range_len, n_left, TPB)

    while loffset < part and roffset < end:
        if iters >= bound:
            if tid == 0:
                out_n_iters[unsafe_offset=b] = PARTITION_OVERRUN
                out_n_swaps[unsafe_offset=b] = Int32(swaps)
            return

        var loff = loffset + tid
        var roff = roffset + tid
        if llen == minlen:
            if loff < part:
                var sl = Int(row_ids[unsafe_offset=loff])
                if sabotage == PART_SAB_NO_ROW_IDS:
                    sl = loff
                var v = data[unsafe_offset = col_offset + sl]
                var misfit: Bool
                if sabotage == PART_SAB_LEFT_MISFIT_GE:
                    misfit = v >= quesval
                else:
                    misfit = v > quesval
                lflag = Int32(1) if misfit else Int32(0)
            else:
                lflag = Int32(0)
        if rlen == minlen:
            if roff < end:
                var sr = Int(row_ids[unsafe_offset=roff])
                if sabotage == PART_SAB_NO_ROW_IDS:
                    sr = roff
                var v = data[unsafe_offset = col_offset + sr]
                var misfit: Bool
                if sabotage == PART_SAB_RIGHT_MISFIT_LT:
                    misfit = v < quesval
                else:
                    misfit = v <= quesval
                rflag = Int32(1) if misfit else Int32(0)
            else:
                rflag = Int32(0)

        barrier()

        var lidx = Int(
            block_prefix_sum[block_size=TPB, exclusive=True](lflag)
        )
        barrier()
        llen = Int(block_sum[block_size=TPB](lflag))
        barrier()
        var ridx = Int(
            block_prefix_sum[block_size=TPB, exclusive=True](rflag)
        )
        barrier()
        rlen = Int(block_sum[block_size=TPB](rflag))
        barrier()

        if sabotage == PART_SAB_NO_SCAN:
            lidx = tid
            ridx = tid

        minlen = llen if llen < rlen else rlen

        if lflag != Int32(0):
            smem[unsafe_offset=lidx] = Int32(loff)
        if rflag != Int32(0):
            smem[unsafe_offset = TPB + ridx] = Int32(roff)
        barrier()

        if lidx < minlen:
            lflag = Int32(0)
        if ridx < minlen:
            rflag = Int32(0)
        if llen == minlen:
            loffset += TPB
        if rlen == minlen:
            roffset += TPB

        if tid < minlen:
            var lslot = Int(smem[unsafe_offset=tid])
            var rslot = Int(smem[unsafe_offset = TPB + tid])
            var a = row_ids[unsafe_offset=lslot]
            var bv = row_ids[unsafe_offset=rslot]
            row_ids[unsafe_offset=lslot] = bv
            if sabotage != PART_SAB_UNPAIRED_SWAP:
                row_ids[unsafe_offset=rslot] = a

        barrier()
        iters += 1
        swaps += minlen

    if tid == 0:
        out_n_iters[unsafe_offset=b] = Int32(iters)
        out_n_swaps[unsafe_offset=b] = Int32(swaps)


# ---------------------------------------------------------------------------

comptime LEAF_VISIT_NONE: Int32 = 0
"""The zero the CALLER must seed `out_visit` with."""

comptime LEAF_VISIT_INTERNAL: Int32 = 1
"""A block ran on this node and took `leafKernel`'s early return at `:403`, because the node is not a leaf."""

comptime LEAF_VISIT_PUBLISHED: Int32 = 2
"""A block ran on this node, it is a leaf, and `SetLeafVector` wrote its value."""

comptime LEAF_MAX_OUT_DEFAULT: Int = 16
"""Default comptime bound on `num_outputs`."""


comptime LEAF_SAB_NONE: Int32 = 0
"""No sabotage."""

comptime LEAF_SAB_NO_ISLEAF: Int32 = 1
"""Drop `if (!node.IsLeaf()) return;` (`:403`), so every node's slot is filled."""

comptime LEAF_SAB_NO_ROW_IDS: Int32 = 2
"""Read `labels[i]` instead of `labels[row_ids[i]]` (`:409-411`), i.e."""

comptime LEAF_SAB_STRIDE_ONE: Int32 = 3
"""Write at `node_id + c` instead of `node_id * num_outputs + c` (`:412`, `decisiontree.cuh:218`)."""

comptime LEAF_SAB_NO_NORMALIZE: Int32 = 4
"""Publish the raw accumulator instead of dividing it: skip the `/ total` of `SetLeafVector` for Gini (`objectives.cuh:97-107`) and the `/ count` for MSE (`:259-264`)."""


def leaf_values_host(
    nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    instance_ranges: MutPointer[InstanceRange, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    n_nodes: Int,
    num_outputs: Int,
    inv_scale: Float32,
    is_classification: Bool,
) -> List[Float32]:
    """THE ORACLE for `leaf_kernel`: the same pass, sequentially."""
    var out = List[Float32](length=n_nodes * num_outputs, fill=Float32(0.0))
    for node_id in range(n_nodes):
        if nodes[unsafe_offset=node_id].left_child_id != NODE_IS_LEAF:
            continue
        var begin = Int(instance_ranges[unsafe_offset=node_id].begin)
        var count = Int(instance_ranges[unsafe_offset=node_id].count)
        var acc = List[Int32](length=num_outputs, fill=Int32(0))
        var seen = 0
        for i in range(begin, begin + count):
            var row = Int(row_ids[unsafe_offset=i])
            var lab = Int(labels_q[unsafe_offset=row])
            if is_classification:
                if lab >= 0 and lab < num_outputs:
                    acc[lab] += Int32(1)
            else:
                acc[0] += Int32(lab)
            seen += 1
        var base = node_id * num_outputs
        if is_classification:
            var total = Int32(0)
            for c in range(num_outputs):
                total += acc[c]
            for c in range(num_outputs):
                out[base + c] = Float32(Int(acc[c])) / Float32(Int(total))
        else:
            for c in range(num_outputs):
                out[base + c] = ftz(
                    Float32(Int(acc[c])) / Float32(seen) * inv_scale
                )
    return out^


def leaf_kernel[
    TPB: Int, MAX_OUT: Int, CLASSIFICATION: Bool, zero_fill: Bool = False
](
    out_leaves: MutPointer[Float32, MutAnyOrigin],
    out_visit: MutPointer[Int32, MutAnyOrigin],
    nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    instance_ranges: MutPointer[InstanceRange, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    num_outputs_in: Int32,
    inv_scale: Float32,
    sabotage_in: Int32,
):
    """`leafKernel`, `builder_kernels_impl.cuh:391-417`."""
    var node_id = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var k = Int(num_outputs_in)
    var sabotage = sabotage_in

    comptime if zero_fill:
        if tid == 0:
            var zfill = Float32(0.0)
            comptime if is_defined["MOJOLEARN_ET_SAB_LEAF_ZERO_FILL"]():
                zfill = Float32(node_id & 1)
            out_visit[unsafe_offset=node_id] = LEAF_VISIT_NONE
            for z in range(k):
                out_leaves[unsafe_offset = node_id * k + z] = zfill

    if k > MAX_OUT or k < 1:
        return

    var is_leaf = (
        nodes[unsafe_offset=node_id].left_child_id == NODE_IS_LEAF
    )
    if sabotage == LEAF_SAB_NO_ISLEAF:
        is_leaf = True
    if not is_leaf:
        if tid == 0:
            out_visit[unsafe_offset=node_id] = LEAF_VISIT_INTERNAL
        return

    var begin = Int(instance_ranges[unsafe_offset=node_id].begin)
    var count = Int(instance_ranges[unsafe_offset=node_id].count)

    var priv = stack_allocation[MAX_OUT, Scalar[DType.int32]]()
    for c in range(MAX_OUT):
        priv[unsafe_offset=c] = Int32(0)
    var seen = Int32(0)

    var i = begin + tid
    while i < begin + count:
        var row = Int(row_ids[unsafe_offset=i])
        if sabotage == LEAF_SAB_NO_ROW_IDS:
            row = i
        var lab = Int(labels_q[unsafe_offset=row])
        comptime if CLASSIFICATION:
            if lab >= 0 and lab < k:
                priv[unsafe_offset=lab] += Int32(1)
        else:
            priv[unsafe_offset=0] += Int32(lab)
        seen += 1
        i += TPB

    var blk_seen = block_sum[block_size=TPB](seen)
    barrier()

    var tot = stack_allocation[MAX_OUT, Scalar[DType.int32]]()
    for c in range(MAX_OUT):
        tot[unsafe_offset=c] = Int32(0)
    for c in range(k):
        var v = block_sum[block_size=TPB](priv[unsafe_offset=c])
        barrier()
        tot[unsafe_offset=c] = v

    if tid != 0:
        return

    var base = node_id * k
    if sabotage == LEAF_SAB_STRIDE_ONE:
        base = node_id

    comptime if CLASSIFICATION:
        var total = Int32(0)
        for c in range(k):
            total += tot[unsafe_offset=c]
        for c in range(k):
            var num = Float32(Int(tot[unsafe_offset=c]))
            if sabotage == LEAF_SAB_NO_NORMALIZE:
                out_leaves[unsafe_offset = base + c] = num
            else:
                out_leaves[unsafe_offset = base + c] = num / Float32(
                    Int(total)
                )
    else:
        for c in range(k):
            var num = Float32(Int(tot[unsafe_offset=c]))
            if sabotage == LEAF_SAB_NO_NORMALIZE:
                out_leaves[unsafe_offset = base + c] = num * inv_scale
            else:
                out_leaves[unsafe_offset = base + c] = ftz(
                    num / Float32(Int(blk_seen)) * inv_scale
                )

    out_visit[unsafe_offset=node_id] = LEAF_VISIT_PUBLISHED
