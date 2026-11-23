# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The OrderedRMSE fit on the host, a SECOND spelling of
`gbdt/train.mojo::train_ordered_rmse` on the gbdt-ordered-rmse lane
(lane/cpu-training-gbdt-ordered, 2026-09-15).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
`checks/fixed_point.choose_scale`, the GPU-free host modules the device fit
itself runs on the host (`best_split`, `build_layout`, `blocks_for`), and the
symmetric oracle's restatements of the binarize, the dithered quantizer, the
halving fold and the pinned partition stats. `create_folds`,
`plan_fold_layout` and `make_fold_doc_indices` are RESTATED below, because
their modules import kernel modules.

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py
`gbdt-ordered-rmse`: 20 trees, depth 6, the default border_count 128,
learning_rate 0.03 and l2 3.0, one explicit permutation, unit weights). The
binding refuses `sample_weight` by name. All three histogram policies are
carried: OneByteFeatures, HalfByteFeatures and, since 2026-09-22,
BinaryFeatures (a feature with exactly one border; `_pw_binary_cells` and
the `pw_hb_binary_sum` writeback, the gbdt-binary-columns lane).

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build)

  1. `train_ordered_rmse` (`gbdt/train.mojo:2244-2302`): `best_split` per
     column on the calling thread, `build_layout`, the binarize.
  2. `fit_ordered_rmse` (`gbdt/methods/dynamic_boosting.mojo:159-294`):
     `create_folds` (`dynamic_boosting_folds.mojo:472-570`, no queries, min
     fold size 100, growth 2), one zero cursor per fold over
     `quality_evaluate_samples.right` permutation positions and one over
     all rows; per tree the Float64 magnitude bound and `choose_scale`, the
     fold targets of `_ordered_target_kernel` (`:44-60`).
  3. `fit_oblivious_tree_structure_traced`'s fold arm
     (`oblivious_tree_doc_parallel_structure_searcher.mojo:268-695`):
     `plan_fold_layout` (`oblivious_tree_fold_tasks.mojo:127-151`, 2N
     partitions, fold bits `IntLog2` ceil), the concatenated doc ids
     (`make_fold_doc_indices`, `:257-296`), `create_fold_based_subsets` and
     `update_subsets_stats` (`pointwise_optimization_subsets.mojo:473-520`:
     the partition offsets and sizes of the bin-sorted documents, the
     gathers, `partition_update_kernel` at 1024 threads with
     `_compute_sum` and `_block_reduce_sum`, `pointwise_scores.mojo:
     1319-1443`). Per level:
       - `compute_hist2` for OneByteFeatures under IDENTICAL
         (`pointwise_kernels.mojo:1601-1801`): the 8-bit fixed-point kernel
         (`pointwise_hist2_one_byte_templ.mojo`, `pointwise_hist2_one_byte_
         8bit.mojo`) at multiplier 1, full pass at depth 0 and the smaller
         child filed under the right slot after (`shift_part_and_bin_sums_
         ptr`); every cell is the Int32 sum of
         `hist2_quantize(stat, scale, hist2_dither(doc id))`, written as
         `Float32(Int(q)) / scale` when its magnitude exceeds 1e-20;
         `scan_pointwise_histograms_kernel` (`split_properties_helpers.
         mojo`, plain Float32 prefix); `update_pointwise_histograms_kernel`
         (the subtraction, `pointwise_kernels.mojo:529-633`).
       - `find_optimal_split_cosine_kernel[128]` (`pointwise_scores.mojo:
         952-1122`, the (estimate, test) fold pairs, `denum_sqr` from
         1e-20), `_block_argmin_and_store` (the 128-lane halving argmin),
         `pw_fold_winner_kernel` and `pw_pack_winner_kernel`
         (`pointwise_split_resolve.mojo`): the block records folded
         challenger first, the winner's score as the next level's
         `score_before`.
       - the post-tree gates (`HasSplit`, the sentinel raise) and
         `split_subsets_from_desc` (the bit OR-ed at `depth + fold_bits`
         and the stable one-bit radix sort).
  4. `ordered_estimate_and_apply` (`dynamic_boosting.mojo:104-156`) per fold
     then for the estimation cursor: `compute_bins_for_model`, the gather in
     permutation order, `partition_from_bins` (the stable counting sort),
     `_estimate_and_apply` (`doc_parallel_boosting.mojo:622-850`) with
     `make_bin_optimized_oracle` (weighted `WeightsCpu` from the one-stat
     partition reduce) and `newton_like_walker_estimate` at one iteration
     (`descent_helpers.mojo:198-238`), then `_ordered_apply_kernel`
     (`identical_mul(leaf, rate)` added onto every position of the cursor).
  5. The exported leaves `identical_mul(leaf, learning_rate)` and
     `model_text` for the oblivious float model (no losses, no bias).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to the Newton Hessian regularizer,
so every leaf of every cursor moves and every fixture's model moves.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-ordered-rmse lane is the measurement.
"""
from std.math import ceil, isfinite, log2

from checks.fixed_point import choose_scale
from checks.numerics import ftz, identical_mul, identical_mul64, identical_mul_add, identical_mul_add_simd, identical_sqrt
from gbdt.data.quantization import NAN_TREATMENT_AS_IS
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.grid_creator.binarization import (
    IDN_ORDERED_RMSE_DEVICE_GRID,
    best_split,
)
from gbdt.options.data_processing_options import NAN_MODE_FORBIDDEN
from gbdt.data.permutation import TRandom
from gbdt.host.gbdt_oracle_tree_ctr import (
    GbdtTreeCtrHost,
    TreeU32Ptr,
    _u32_ptr,
)
from gbdt.gpu_util.kernel.random_gen import (
    advance_seed_k,
    next_normal_f,
    next_poisson_f,
    next_uniform_f,
)
from core.host_parallel import host_nested_units, host_parallelize
from core.host_predict_threads import (
    HostF32Ptr,
    host_predict_chunk,
    host_predict_task_count_work,
)
from gbdt.host.gbdt_oracle import (
    GBDT_HOST_BINARY_SABOTAGE,
    GBDT_ORACLE_HOST_SABOTAGE,
    GbdtHostGrid,
    GbdtHostModel,
    _binarize_columns,
    _halving_fold_live,
    _snap_plane,
    _hist2_dither,
    GbdtI32Ptr,
    GbdtIntPtr,
    GbdtU64Ptr,
    _gather_rows,
    _identity_rows,
    _gbdt_units,
    _partition_stats_level,
    _pinned_partition_stats,
    _hist2_quantize,
    gbdt_host_grid,
    gbdt_host_model_text,
)


comptime GBDT_ORD_FLOAT32_MAX = Float32(3.4028234663852886e38)
comptime GBDT_ORD_SENTINEL = UInt32(0xFFFFFFFF)
#: `POINTWISE_WIDE_BLOCK` (`pointwise_scores.mojo:284`), the partition
#: update's threads per part.
comptime GBDT_ORD_WIDE_BLOCK = 1024
#: `POINTWISE_SCORE_BLOCK` (`pointwise_scores.mojo:278`).
comptime GBDT_ORD_SCORE_BLOCK = 128
#: `STATS_BLOCK` (`partitions_reduce.mojo`) and the pinned sm count.
comptime GBDT_ORD_STATS_BLOCK = 512
comptime GBDT_ORD_PINNED_SM = 32
#: `PW_HB_BLOCK` under IDENTICAL (`block_size_for[K_POINTWISE_HIST_2_HALF_BYTE]`:
#: the 32 KB floor over 16 floats x 4 bytes).
comptime GBDT_ORD_HB_BLOCK = 512


# ===========================================================================
# THE FOLDS (`dynamic_boosting_folds.mojo`, `oblivious_tree_fold_tasks.mojo`)
# ===========================================================================


def _int_log2_floor_ceil(v: Int) -> Int:
    """`NCB::IntLog2`, ceil (`libs/helpers/math_utils.h:14-16`)."""
    var bit = 0
    while (1 << bit) < v:
        bit += 1
    return bit


def _next_offset(line: Int, n: Int) -> Int:
    """`NextQueryOffsetForLine` without queries: `min(line + 1, DocCount)`."""
    return min(line + 1, n)


def _ordered_folds(n: Int, growth_rate: Float64, min_fold_size: Int) raises -> List[Int]:
    """`create_folds` (`dynamic_boosting_folds.mojo:472-570`) at one device,
    no queries, Ordered: returns `[est_right_0, qe_right_0, est_right_1, ...]`
    (every estimate slice starts at 0, every quality slice at the estimate's
    right)."""
    var min_est_raw = 1
    if n >= 500:
        var folds = _int_log2_floor_ceil((n + min_fold_size - 1) // min_fold_size)
        if folds >= 18:
            min_est_raw = (n + (1 << 18) - 1) // (1 << 18)
        else:
            min_est_raw = min(min_fold_size, n // 50)
    var min_estimation = _next_offset(min_est_raw, n)
    if n < 4:
        raise Error("Error: pool has just " + String(n) + " groups or docs")
    if min_estimation == 0:
        raise Error("Error: min learn size should be positive")
    var out = List[Int]()
    var test_end = _next_offset(min(Int(Float64(min_estimation) * growth_rate), n), n)
    out.append(min_estimation)
    out.append(test_end)
    var iterations = 0
    while out[len(out) - 1] < n:
        iterations += 1
        if iterations > n + 2:
            raise Error("CreateFolds did not converge")
        var right = out[len(out) - 1]
        var end = _next_offset(min(Int(Float64(right) * growth_rate), n), n)
        out.append(right)
        out.append(end)
    return out^


# ===========================================================================
# THE SUBSETS (`pointwise_optimization_subsets.mojo`)
# ===========================================================================


def _partition_update_lanes(
    values: HostF32Ptr, offset: Int, size: Int, lane_lo: Int, lane_hi: Int,
    slab: HostF32Ptr,
):
    """`_compute_sum[1024]` (`pointwise_scores.mojo:578-628`) for the lanes
    `[lane_lo, lane_hi)` of one part: lane `tid` is the Float32 chain of
    `values[offset + tid + k * 1024]`, `k` ascending, from +0.0. SERIAL BY
    NATURE inside a lane (one device thread's strided chain); the lanes are
    independent."""
    for tid in range(lane_lo, lane_hi):
        var acc = Float32(0.0)
        var i = tid
        while i < size:
            acc += values[offset + i]
            i += GBDT_ORD_WIDE_BLOCK
        slab[tid] = acc


def _partition_update_sum(
    values: HostF32Ptr, offset: Int, size: Int, mut slab: List[Float32]
) -> Float32:
    """`_compute_sum[1024]` per thread then `_block_reduce_sum[1024]`
    (`pointwise_scores.mojo:578-628`, `:1408-1443`) for one part, on the
    calling thread: the unit of `_update_subsets_stats`'s per-part split.
    `slab` is the caller's 1024-lane scratch."""
    if size <= 0:
        return Float32(0.0)
    var live = min(size, GBDT_ORD_WIDE_BLOCK)
    var p = 1
    while p < live:
        p <<= 1
    _partition_update_lanes(
        values, offset, size, 0, p, rebind[HostF32Ptr](slab.unsafe_ptr())
    )
    # lanes past `size` hold +0.0 (`_halving_fold_live`)
    return _halving_fold_live(slab, live, GBDT_ORD_WIDE_BLOCK)


def _partition_update_sum_wide(
    values: HostF32Ptr, offset: Int, size: Int, mut slab: List[Float32]
) -> Float32:
    """`_partition_update_sum` for a part that holds a large share of the
    rows: the 1024 lanes (independent strided chains, one device thread
    each) run in `_gbdt_units` contiguous lane ranges, and the halving fold
    runs after the join over the same slab. The same adds in the same
    order as the one-thread form at any task count."""
    if size <= 0:
        return Float32(0.0)
    var live = min(size, GBDT_ORD_WIDE_BLOCK)
    var p = 1
    while p < live:
        p <<= 1
    var units = _gbdt_units(p, 2 * size)
    var chunk = host_predict_chunk(p, units)
    var slab_p = rebind[HostF32Ptr](slab.unsafe_ptr())

    def lanes(unit: Int) {imm chunk, imm p, imm values, imm offset, imm size, imm slab_p}:
        _partition_update_lanes(
            values, offset, size, unit * chunk, min(p, (unit + 1) * chunk), slab_p
        )

    host_parallelize(lanes, units)
    return _halving_fold_live(slab, live, GBDT_ORD_WIDE_BLOCK)


@fieldwise_init
struct _OrdSubsets(Movable):
    var bins: List[UInt32]
    var indices: List[Int]
    var p_off: List[Int]
    var p_sz: List[Int]
    var part_stats: List[Float32]
    var g_weight: List[Float32]
    var g_target: List[Float32]


def _ord_lower_bound(bins: TreeU32Ptr, n: Int, key: Int) -> Int:
    """The first position of the sorted `bins[0:n]` whose value is at or
    above `key` (integers only)."""
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(bins[mid]) < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _ord_part_bounds_range(
    bins: TreeU32Ptr, n: Int, part_lo: Int, part_hi: Int,
    p_off: GbdtIntPtr, p_sz: GbdtIntPtr,
):
    """`update_partition_offsets_kernel` / `update_partition_sizes_kernel`
    for the parts `[part_lo, part_hi)`: a lower bound per part on the
    bin-sorted documents (one device thread a part)."""
    for b in range(part_lo, part_hi):
        var first = _ord_lower_bound(bins, n, b)
        p_off[b] = first
        p_sz[b] = _ord_lower_bound(bins, n, b + 1) - first


def _ord_part_sums_stride(
    first: Int, step: Int, part_count: Int, wide_from: Int,
    g_weight: HostF32Ptr, g_target: HostF32Ptr,
    p_off: GbdtIntPtr, p_sz: GbdtIntPtr, part_stats: HostF32Ptr,
):
    """The partition update of the parts `first, first + step, ...` whose
    size is below `wide_from` (the wide ones are summed by
    `_partition_update_sum_wide`): one 1024-lane scratch per call."""
    var slab = List[Float32](length=GBDT_ORD_WIDE_BLOCK, fill=Float32(0.0))
    var b = first
    while b < part_count:
        var sz = p_sz[b]
        if sz < wide_from:
            part_stats[3 * b] = _partition_update_sum(g_weight, p_off[b], sz, slab)
            part_stats[3 * b + 1] = _partition_update_sum(g_target, p_off[b], sz, slab)
            part_stats[3 * b + 2] = Float32(sz)
        b += step


@no_inline
def _update_subsets_stats(
    mut s: _OrdSubsets, part_count: Int, sw: List[Float32], sg: List[Float32]
):
    """`update_subsets_stats`: the offsets and sizes of the sorted bins
    (`update_partition_offsets_kernel`, `update_partition_sizes_kernel`, a
    lower bound per part), the two gathers, the partition update.

    THE UNITS (lane gbdt-host-par). Offsets and sizes: one PART (two binary
    searches on the sorted bins; integers). Gathers: one position (a move),
    in contiguous ranges. Partition update: one PART, whose two sums are
    its own 1024 strided lanes and halving fold (`partition_update_kernel`,
    one block a part); the parts go round robin over the tasks, because
    the sizes are skewed (the fold parts double, a split's children are
    uneven). A part holding a quarter of the documents or more would leave
    its task with that quarter, so such a part (at most four) is summed
    with its LANES across the tasks instead (`_partition_update_sum_wide`).
    Every sum's adds and their order are the device block's at any task
    count; every task count comes from `_gbdt_units` (the nested rule)."""
    var doc_count = len(s.bins)
    var bins_p = _u32_ptr(s.bins)
    var off_p = rebind[GbdtIntPtr](s.p_off.unsafe_ptr())
    var sz_p = rebind[GbdtIntPtr](s.p_sz.unsafe_ptr())
    var idx_p = rebind[GbdtIntPtr](s.indices.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())
    var gw_p = rebind[HostF32Ptr](s.g_weight.unsafe_ptr())
    var gt_p = rebind[HostF32Ptr](s.g_target.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](s.part_stats.unsafe_ptr())

    var b_units = _gbdt_units(part_count, 48 * part_count)
    var b_chunk = host_predict_chunk(part_count, b_units)

    def bounds(unit: Int) {imm b_chunk, imm part_count, imm doc_count, imm bins_p, imm off_p, imm sz_p}:
        _ord_part_bounds_range(
            bins_p, doc_count, unit * b_chunk,
            min(part_count, (unit + 1) * b_chunk), off_p, sz_p,
        )

    host_parallelize(bounds, b_units)

    var g_units = _gbdt_units(doc_count, 4 * doc_count)
    var g_chunk = host_predict_chunk(doc_count, g_units)

    def gather(unit: Int) {imm g_chunk, imm doc_count, imm idx_p, imm sw_p, imm sg_p, imm gw_p, imm gt_p}:
        for i in range(unit * g_chunk, min(doc_count, (unit + 1) * g_chunk)):
            var row = idx_p[i]
            gw_p[i] = sw_p[row]
            gt_p[i] = sg_p[row]

    host_parallelize(gather, g_units)

    var s_units = _gbdt_units(part_count, 2 * doc_count)
    # a part is wide when it holds a quarter of the documents or more and
    # the per-part split has more than one task to starve
    var wide_from = doc_count + 1
    if s_units > 1:
        wide_from = max((doc_count + 3) // 4, 1)
        var slab = List[Float32](length=GBDT_ORD_WIDE_BLOCK, fill=Float32(0.0))
        for b in range(part_count):  # driver-loop: over the parts, the wide ones' lanes split inside
            var sz = s.p_sz[b]
            if sz >= wide_from:
                s.part_stats[3 * b] = _partition_update_sum_wide(gw_p, s.p_off[b], sz, slab)
                s.part_stats[3 * b + 1] = _partition_update_sum_wide(gt_p, s.p_off[b], sz, slab)
                s.part_stats[3 * b + 2] = Float32(sz)

    def sums(unit: Int) {imm s_units, imm part_count, imm wide_from, imm gw_p, imm gt_p, imm off_p, imm sz_p, imm stats_p}:
        _ord_part_sums_stride(
            unit, s_units, part_count, wide_from, gw_p, gt_p, off_p, sz_p, stats_p
        )

    host_parallelize(sums, s_units)
    _ = len(s.bins)
    _ = len(s.p_off)
    _ = len(s.p_sz)
    _ = len(s.indices)
    _ = len(s.g_weight)
    _ = len(s.g_target)
    _ = len(s.part_stats)
    _ = len(sw)
    _ = len(sg)


# ===========================================================================
# THE STRUCTURE SEARCH (the fold arm)
# ===========================================================================


@fieldwise_init
struct _OrdSplit(ImplicitlyCopyable, Movable):
    var feature: Int
    var bin: Int


@fieldwise_init
struct _PwHelper(Movable):
    """One `PolicyScoreHelper` (`pointwise_scores_calcer.mojo:124-541`):
    the policy, the block's features in block order (global id, the element
    offset of its compressed-index column, its fold offset inside the block
    and its folds), the block's bin-feature count and its own histogram."""

    var policy: Int
    var gids: List[Int]
    var offsets: List[Int]
    var first: List[Int]
    var folds: List[Int]
    var hist_line: Int
    var hist: List[Float32]
    # the slots written since the histogram was last all +0.0: the next
    # tree zeroes these instead of allocating and filling a fresh plane
    var dirty: List[Int]
    var dirty_flag: List[Bool]

    def mark(mut self, slot: Int):
        if not self.dirty_flag[slot]:
            self.dirty_flag[slot] = True
            self.dirty.append(slot)


def _pw_half_byte_group(
    points: List[List[Int]],
    docs: List[Int],
    g_weight: List[Float32],
    g_target: List[Float32],
    cindex: List[UInt32],
    word_offset: Int,
    f_count: Int,
) -> List[Float32]:
    """`TPointHistHalfByte` (`pointwise_hist2_half_byte_template.mojo`) for
    one group of eight features, through its `Reduce`: returns the 256
    reduced cells `[fold * 16 + 2 * fid + stat]`.

    EVERY ADD SITS BETWEEN TWO BARRIERS and every thread runs the same point
    count, so sync window `16 * k + 2 * i + d` of every thread coincides
    (point `k`, nibble turn `i`, first or second stat `d`). Within a window
    no two threads of one inner copy write one cell: a thread's turn `i`
    names feature `((tid & 14) / 2 + i) mod 8`, and the two threads sharing
    `tid & 14` write opposite stats. So each slice cell is the Float32 sum
    of its contributions in window order, which is the loop order below.
    A zero point adds 0.0 to bin 0 and moves no bit of a cell that is never
    -0.0, so it is skipped."""
    comptime SLICE = 512
    var slices = List[Float32](length=16 * SLICE, fill=Float32(0.0))
    var k_count = len(points[0])
    for k in range(k_count):
        for sub in range(16):
            var i = sub // 2
            var d = sub % 2
            for t in range(GBDT_ORD_HB_BLOCK):
                var p = points[t][k]
                if p < 0:
                    continue
                var shift = t & 14
                var j = ((shift // 2) + i) % 8
                if j >= f_count:
                    continue
                var flag = t & 1
                var stat = flag if d == 0 else 1 - flag
                var row = docs[p]
                var ci = cindex[word_offset + row]
                var bin = Int((ci >> UInt32(28 - 4 * j)) & UInt32(15))
                var at = (t // 32) * SLICE + (t & 16) + (bin << 5) + 2 * j + stat
                var val = g_target[p] if stat == 1 else g_weight[p]
                slices[at] = slices[at] + val
    var stage1 = List[Float32](length=SLICE, fill=Float32(0.0))
    for s in range(SLICE):
        var acc = Float32(0.0)
        for w in range(16):
            acc += slices[w * SLICE + s]
        stage1[s] = acc
    var cells = List[Float32](length=256, fill=Float32(0.0))
    for tid in range(256):
        var fold2 = tid >> 4
        var e = tid & 15
        cells[tid] = stage1[32 * fold2 + e] + stage1[32 * fold2 + e + 16]
    return cells^


def _pw_binary_cells(
    runs: List[Int],
    docs: List[Int],
    g_weight: List[Float32],
    g_target: List[Float32],
    cindex: List[UInt32],
    word_offset: Int,
    nibbles: Int,
) -> List[Float32]:
    """`_pw_half_byte_group` over the live points only: the same 256
    reduced cells `[value * 16 + 2 * nibble + stat]`, the same window order
    (point `k`, turn `sub`, thread `t`). A zero point is skipped there
    already; here the loop does not visit it. The fold of the warp slices
    visits only the slices a live thread wrote: a slice no thread reaches
    is all +0.0, a cell never holds -0.0 (it starts at +0.0 and a float sum
    reaches -0.0 only from -0.0 + -0.0), so adding it moves no bit."""
    comptime SLICE = 512
    var max_live = 0
    for r in range(len(runs) // 3):
        if runs[3 * r + 2] > max_live:
            max_live = runs[3 * r + 2]
    var warps = (max_live + 31) // 32
    var slices = List[Float32](length=warps * SLICE, fill=Float32(0.0))
    for r in range(len(runs) // 3):
        var first = runs[3 * r]
        var step = runs[3 * r + 1]
        var live = runs[3 * r + 2]
        for sub in range(16):
            var i = sub // 2
            var d = sub % 2
            for t in range(live):
                var shift = t & 14
                var j = ((shift // 2) + i) % 8
                if j >= nibbles:
                    continue
                var p = first + step * t
                var flag = t & 1
                var stat = flag if d == 0 else 1 - flag
                var row = docs[p]
                var ci = cindex[word_offset + row]
                var bin = Int((ci >> UInt32(28 - 4 * j)) & UInt32(15))
                var at = (t // 32) * SLICE + (t & 16) + (bin << 5) + 2 * j + stat
                var val = g_target[p] if stat == 1 else g_weight[p]
                slices[at] = slices[at] + val
    var cells = List[Float32](length=256, fill=Float32(0.0))
    if warps == 0:
        return cells^
    for tid in range(256):
        var fold2 = tid >> 4
        var e = tid & 15
        var s0 = 32 * fold2 + e
        var a0 = Float32(0.0)
        var a1 = Float32(0.0)
        for w in range(warps):
            a0 += slices[w * SLICE + s0]
            a1 += slices[w * SLICE + s0 + 16]
        cells[tid] = a0 + a1
    return cells^


# ===========================================================================
# THE LEVEL'S SPLITS (lane gbdt-host-par): every stage of
# `_ordered_tree_structure_levels` as ordinary range functions, launched
# over `_gbdt_units` tasks (the nested rule: one task inside a member's or
# a permutation's pool task).
# ===========================================================================


def _pw_zero_dirty(mut hz: _PwHelper):
    """Every slot the helper wrote since its histogram was last all +0.0,
    back to +0.0 (the device's `reset_for_tree`). The unit is one written
    SLOT (its own cells); the flags are cleared after the join."""
    var line2 = hz.hist_line * 2
    var n = len(hz.dirty)
    var units = _gbdt_units(n, n * line2)
    var chunk = host_predict_chunk(n, units)
    var hist_p = rebind[HostF32Ptr](hz.hist.unsafe_ptr())
    var dirty_p = rebind[GbdtIntPtr](hz.dirty.unsafe_ptr())

    def slots(unit: Int) {imm chunk, imm n, imm line2, imm hist_p, imm dirty_p}:
        for k in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var base = dirty_p[k] * line2
            for c in range(base, base + line2):
                hist_p[c] = Float32(0.0)

    host_parallelize(slots, units)
    for k in range(n):  # driver-loop: over the written slots, their flags
        hz.dirty_flag[hz.dirty[k]] = False
    hz.dirty.clear()
    _ = len(hz.hist)


def _ord_initial_bins_range(
    lo: Int, hi: Int, bounds: GbdtIntPtr, fold_count: Int, bins: TreeU32Ptr
):
    for p in range(fold_count):
        for i in range(max(lo, bounds[p]), min(hi, bounds[p + 1])):
            bins[i] = UInt32(p)


def _ord_initial_bins(part_bounds: List[Int], fold_count: Int, mut bins: List[UInt32]):
    """`write_fold_based_initial_bins`: part `p` over its concatenated
    range. One statement per position (a position is in one part): a task
    owns a run of positions and writes each part's range clipped to it."""
    var n = len(bins)
    var units = _gbdt_units(n, n)
    var chunk = host_predict_chunk(n, units)
    var bounds_p = rebind[GbdtIntPtr](part_bounds.unsafe_ptr())
    var bins_p = _u32_ptr(bins)

    def rows(unit: Int) {imm chunk, imm n, imm fold_count, imm bounds_p, imm bins_p}:
        _ord_initial_bins_range(
            unit * chunk, min(n, (unit + 1) * chunk), bounds_p, fold_count, bins_p
        )

    host_parallelize(rows, units)
    _ = len(part_bounds)
    _ = len(bins)


def _ord_gather_docs(doc_ids: List[Int], indices: List[Int], mut docs: List[Int]):
    """`docs[i] = doc_ids[indices[i]]`, the level's document order (a move
    per position), the positions in `_gbdt_units` contiguous ranges."""
    var n = len(docs)
    var units = _gbdt_units(n, 2 * n)
    var chunk = host_predict_chunk(n, units)
    var ids_p = rebind[GbdtIntPtr](doc_ids.unsafe_ptr())
    var idx_p = rebind[GbdtIntPtr](indices.unsafe_ptr())
    var docs_p = rebind[GbdtIntPtr](docs.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm ids_p, imm idx_p, imm docs_p}:
        for i in range(unit * chunk, min(n, (unit + 1) * chunk)):
            docs_p[i] = ids_p[idx_p[i]]

    host_parallelize(rows, units)
    _ = len(doc_ids)
    _ = len(indices)
    _ = len(docs)


def _ord_hist_piece(
    policy: Int,
    group: Int,
    lo: Int,
    hi: Int,
    n_feat: Int,
    fixed_scale: Float32,
    cindex: TreeU32Ptr,
    docs: GbdtIntPtr,
    g_weight: HostF32Ptr,
    g_target: HostF32Ptr,
    offsets: GbdtIntPtr,
    folds: GbdtIntPtr,
    first: GbdtIntPtr,
    mut acc: List[Int32],
    mut reached: List[Bool],
    mut reached_list: List[Int],
):
    """The fixed-point histogram adds of the positions `[lo, hi)` of one
    slot into `acc` (two Int32 cells a bin-feature), the cells reached
    filed in `reached_list`. INTEGER sums: `hist2_quantize(stat, scale,
    hist2_dither(doc id))` per row, added in any order and any grouping
    (the device adds them with atomics), so a slot's rows may be cut
    anywhere and the pieces added afterwards."""
    if policy == POLICY_ONE_BYTE:
        # the 8-bit fixed-point kernel: Int32 sums, the dither keyed on the
        # document id
        for pos in range(lo, hi):
            var row = docs[pos]
            var u = _hist2_dither(row)
            var qw = _hist2_quantize(g_weight[pos], fixed_scale, u)
            var qt = _hist2_quantize(g_target[pos], fixed_scale, u)
            var f_base = 0
            while f_base < n_feat:
                var ci = cindex[offsets[f_base] + row]
                var f_count = min(group, n_feat - f_base)
                for j in range(f_count):
                    var bin = Int((ci >> UInt32(24 - 8 * j)) & UInt32(255))
                    if bin < folds[f_base + j]:
                        var at = (first[f_base + j] + bin) * 2
                        if not reached[at // 2]:
                            reached[at // 2] = True
                            reached_list.append(at)
                        acc[at] = acc[at] + qw
                        acc[at + 1] = acc[at + 1] + qt
                f_base += group
    else:
        # lane/neural-pass124: the half-byte and binary kernels in fixed
        # point (`PointHistHalfByteInt`): the Int32 sum of
        # `hist2_quantize(stat, scale, hist2_dither(doc id))` per cell, any
        # order. A half-byte feature's cell is its 4-bit value's (filed
        # when below its fold count); a binary feature's one cell takes the
        # rows whose bit `3 - (j & 3)` of nibble `j / 4` is clear
        # (`pw_hb_binary_sum_int`).
        var is_bin = policy == POLICY_BINARY
        for pos in range(lo, hi):
            var row = docs[pos]
            var u = _hist2_dither(row)
            var qw = _hist2_quantize(g_weight[pos], fixed_scale, u)
            var qt = _hist2_quantize(g_target[pos], fixed_scale, u)
            var f_base = 0
            while f_base < n_feat:
                var ci = cindex[offsets[f_base] + row]
                var f_count = min(group, n_feat - f_base)
                for j in range(f_count):
                    var at: Int
                    if is_bin:
                        var v = Int((ci >> UInt32(28 - 4 * (j // 4))) & UInt32(15))
                        var f_mask = 1 << (3 - (j & 3))
                        if (v & f_mask) != 0:
                            continue
                        if GBDT_HOST_BINARY_SABOTAGE and v == 0:
                            continue
                        at = first[f_base + j] * 2
                    else:
                        var bin = Int((ci >> UInt32(28 - 4 * j)) & UInt32(15))
                        if bin >= folds[f_base + j]:
                            continue
                        at = (first[f_base + j] + bin) * 2
                    if not reached[at // 2]:
                        reached[at // 2] = True
                        reached_list.append(at)
                    acc[at] = acc[at] + qw
                    acc[at + 1] = acc[at + 1] + qt
                f_base += group


def _ord_hist_cells(
    acc: GbdtI32Ptr, acc_base: Int, cells: Int, fixed_scale: Float32,
    dst: HostF32Ptr, dst_base: Int,
):
    """The slot's write-back from its Int32 sums: `Float32(Int(q)) / scale`
    when its magnitude exceeds 1e-20 (a cell no row reaches, or whose sum
    is 0, is not written and keeps the slot's +0.0)."""
    for c in range(cells):
        var val = Float32(Int(acc[acc_base + c])) / fixed_scale
        if abs(val) > Float32(1e-20):
            dst[dst_base + c] = val


def _ord_hist_task(
    unit: Int,
    chunk: Int,
    total: Int,
    n_seg: Int,
    line2: Int,
    policy: Int,
    group: Int,
    n_feat: Int,
    fixed_scale: Float32,
    cindex: TreeU32Ptr,
    docs: GbdtIntPtr,
    g_weight: HostF32Ptr,
    g_target: HostF32Ptr,
    offsets: GbdtIntPtr,
    folds: GbdtIntPtr,
    first: GbdtIntPtr,
    seg_slot: GbdtIntPtr,
    seg_off: GbdtIntPtr,
    seg_at: GbdtIntPtr,
    hist: HostF32Ptr,
    partial: GbdtI32Ptr,
    piece_seg: GbdtIntPtr,
):
    """One task of `_ord_hist_level`: the run `[unit * chunk, (unit + 1) *
    chunk)` of the level's rows, the computed slots laid end to end
    (`seg_at`). A slot wholly inside the run is converted and written here
    (the task owns the slot's cells). A slot the run cuts leaves its Int32
    piece in the task's plane `2 * unit` (the run's first slot) or
    `2 * unit + 1` (its last), named in `piece_seg`, for the join."""
    var lo = unit * chunk
    var hi = min(total, (unit + 1) * chunk)
    if lo >= hi:
        return
    var acc = List[Int32](length=line2, fill=Int32(0))
    var reached = List[Bool](length=line2 // 2, fill=False)
    var reached_list = List[Int](capacity=line2 // 2)
    var seg = 0
    while seg < n_seg and seg_at[seg + 1] <= lo:
        seg += 1
    var at = lo
    while at < hi and seg < n_seg:
        var seg_lo = seg_at[seg]
        var seg_hi = seg_at[seg + 1]
        var stop = min(hi, seg_hi)
        var off = seg_off[seg]
        _ord_hist_piece(
            policy, group, off + (at - seg_lo), off + (stop - seg_lo), n_feat,
            fixed_scale, cindex, docs, g_weight, g_target, offsets, folds,
            first, acc, reached, reached_list,
        )
        if at == seg_lo and stop == seg_hi:
            # the whole slot: only the reached cells are converted, then
            # cleared for the next slot
            var base = seg_slot[seg] * line2
            for k in range(len(reached_list)):
                var c0 = reached_list[k]
                for c in range(c0, c0 + 2):
                    var val = Float32(Int(acc[c])) / fixed_scale
                    if abs(val) > Float32(1e-20):
                        hist[base + c] = val
                    acc[c] = Int32(0)
                reached[c0 // 2] = False
        else:
            var which = 2 * unit
            if at != lo:
                which = 2 * unit + 1
            piece_seg[which] = seg
            var pbase = which * line2
            for k in range(len(reached_list)):
                var c0 = reached_list[k]
                partial[pbase + c0] = acc[c0]
                partial[pbase + c0 + 1] = acc[c0 + 1]
                acc[c0] = Int32(0)
                acc[c0 + 1] = Int32(0)
                reached[c0 // 2] = False
        reached_list.clear()
        at = stop
        seg += 1


def _ord_hist_level(
    mut hp: _PwHelper,
    cindex: TreeU32Ptr,
    docs: List[Int],
    g_weight: List[Float32],
    g_target: List[Float32],
    seg_slot: List[Int],
    seg_off: List[Int],
    seg_at: List[Int],
    fixed_scale: Float32,
):
    """`compute_hist2` for one helper at one level: every computed slot
    `seg_slot[k]` from its rows `[seg_off[k], seg_off[k] + size)`, the
    sizes as the running `seg_at` (one more entry than slots).

    THE UNIT IS A RUN OF ROWS. The slots are skewed (the fold parts double
    and a level computes the smaller child), so the level's rows, slot
    after slot, are cut into `_gbdt_units` equal runs. Every cell is an
    Int32 sum of per-row quantized stats, exact in any order and grouping
    (`_ord_hist_piece`), so a slot cut between tasks is the integer sum of
    its pieces, added after the join in task order and converted once with
    the expression a whole slot uses. The join's merge is serial over at
    most two planes a task (work in tasks x cells, never rows)."""
    var n_seg = len(seg_slot)
    if n_seg == 0:
        return
    var total = seg_at[n_seg]
    var n_feat = len(hp.gids)
    var line2 = hp.hist_line * 2
    var policy = hp.policy
    var group = 8 if policy == POLICY_HALF_BYTE else (
        32 if policy == POLICY_BINARY else 4
    )
    var units = _gbdt_units(total, total * (n_feat + 4))
    var chunk = host_predict_chunk(total, units)
    var planes = 0
    if units > 1:
        planes = 2 * units
    var partial = List[Int32](length=planes * line2, fill=Int32(0))
    var piece_seg = List[Int](length=2 * units, fill=-1)
    var docs_p = rebind[GbdtIntPtr](docs.unsafe_ptr())
    var gw_p = rebind[HostF32Ptr](g_weight.unsafe_ptr())
    var gt_p = rebind[HostF32Ptr](g_target.unsafe_ptr())
    var offs_p = rebind[GbdtIntPtr](hp.offsets.unsafe_ptr())
    var folds_p = rebind[GbdtIntPtr](hp.folds.unsafe_ptr())
    var first_p = rebind[GbdtIntPtr](hp.first.unsafe_ptr())
    var slot_p = rebind[GbdtIntPtr](seg_slot.unsafe_ptr())
    var off_p = rebind[GbdtIntPtr](seg_off.unsafe_ptr())
    var at_p = rebind[GbdtIntPtr](seg_at.unsafe_ptr())
    var hist_p = rebind[HostF32Ptr](hp.hist.unsafe_ptr())
    var partial_p = rebind[GbdtI32Ptr](partial.unsafe_ptr())
    var piece_p = rebind[GbdtIntPtr](piece_seg.unsafe_ptr())

    def runs(unit: Int) {imm chunk, imm total, imm n_seg, imm line2, imm policy, imm group, imm n_feat, imm fixed_scale, imm cindex, imm docs_p, imm gw_p, imm gt_p, imm offs_p, imm folds_p, imm first_p, imm slot_p, imm off_p, imm at_p, imm hist_p, imm partial_p, imm piece_p}:
        _ord_hist_task(
            unit, chunk, total, n_seg, line2, policy, group, n_feat,
            fixed_scale, cindex, docs_p, gw_p, gt_p, offs_p, folds_p, first_p,
            slot_p, off_p, at_p, hist_p, partial_p, piece_p,
        )

    host_parallelize(runs, units)
    # the cut slots: a slot's pieces are neighbours in (task, first / last)
    # order, because the runs ascend
    if units > 1:
        var cur_seg = -1
        var cur_base = 0
        for k in range(2 * units):  # driver-loop: over the cut slots' pieces, at most two a task
            var seg = piece_seg[k]
            if seg < 0:
                continue
            if seg == cur_seg:
                for c in range(line2):
                    partial[cur_base + c] = partial[cur_base + c] + partial[k * line2 + c]
            else:
                if cur_seg >= 0:
                    _ord_hist_cells(
                        partial_p, cur_base, line2, fixed_scale, hist_p,
                        seg_slot[cur_seg] * line2,
                    )
                cur_seg = seg
                cur_base = k * line2
        if cur_seg >= 0:
            _ord_hist_cells(
                partial_p, cur_base, line2, fixed_scale, hist_p,
                seg_slot[cur_seg] * line2,
            )
    _ = len(docs)
    _ = len(g_weight)
    _ = len(g_target)
    _ = len(seg_slot)
    _ = len(seg_off)
    _ = len(seg_at)
    _ = len(partial)
    _ = len(piece_seg)
    _ = len(hp.hist)
    _ = len(hp.offsets)
    _ = len(hp.folds)
    _ = len(hp.first)


def _ord_scan_range(
    u_lo: Int,
    u_hi: Int,
    n_scan: Int,
    hist_line: Int,
    hist: HostF32Ptr,
    seg_slot: GbdtIntPtr,
    scan_first: GbdtIntPtr,
    scan_folds: GbdtIntPtr,
):
    """`scan_pointwise_histograms_kernel` for the (slot, feature) units
    `[u_lo, u_hi)`: the plain Float32 prefix over a feature's bins, per
    stat. SERIAL BY NATURE inside one (slot, feature, stat): a running
    sum, each cell the one before plus its own."""
    for u in range(u_lo, u_hi):
        var seg = u // n_scan
        var k = u - seg * n_scan
        var sbase = (seg_slot[seg] * hist_line + scan_first[k]) * 2
        var n_bins = scan_folds[k]
        for st in range(2):
            var running = Float32(0.0)
            for b in range(n_bins):
                var at = sbase + b * 2 + st
                running += hist[at]
                hist[at] = running


def _ord_scan_level(
    mut hp: _PwHelper,
    seg_slot: List[Int],
    have_one_hot: Bool,
    one_hot_of: List[Bool],
):
    """The scan over the level's computed slots. A slot is first written at
    the level that computes it (leaves [ny, 2ny), leaf 0 at depth 0), so
    one whose partition is empty is still all +0.0 and is not in
    `seg_slot`. The unit is one (slot, feature): its two prefix sums are
    its own cells; a feature with one fold or less has no prefix, and a
    one-hot feature keeps its per-bin cells."""
    var n_seg = len(seg_slot)
    var scan_first = List[Int]()
    var scan_folds = List[Int]()
    for f in range(len(hp.gids)):  # driver-loop: over the helper's features, naming those scanned
        if hp.folds[f] <= 1:
            continue
        if have_one_hot and one_hot_of[hp.gids[f]]:
            continue
        scan_first.append(hp.first[f])
        scan_folds.append(hp.folds[f])
    var n_scan = len(scan_first)
    var total = n_seg * n_scan
    if total == 0:
        return
    var hist_line = hp.hist_line
    var units = _gbdt_units(total, 2 * n_seg * hist_line)
    var chunk = host_predict_chunk(total, units)
    var hist_p = rebind[HostF32Ptr](hp.hist.unsafe_ptr())
    var slot_p = rebind[GbdtIntPtr](seg_slot.unsafe_ptr())
    var first_p = rebind[GbdtIntPtr](scan_first.unsafe_ptr())
    var folds_p = rebind[GbdtIntPtr](scan_folds.unsafe_ptr())

    def scan(unit: Int) {imm chunk, imm total, imm n_scan, imm hist_line, imm hist_p, imm slot_p, imm first_p, imm folds_p}:
        _ord_scan_range(
            unit * chunk, min(total, (unit + 1) * chunk), n_scan, hist_line,
            hist_p, slot_p, first_p, folds_p,
        )

    host_parallelize(scan, units)
    _ = len(hp.hist)
    _ = len(seg_slot)
    _ = len(scan_first)
    _ = len(scan_folds)


def _ord_subtract_range(
    c_lo: Int,
    c_hi: Int,
    line2: Int,
    hist: HostF32Ptr,
    pair_l: GbdtIntPtr,
    pair_r: GbdtIntPtr,
    pair_left: GbdtIntPtr,
):
    """`update_pointwise_histograms_kernel` for the cells `[c_lo, c_hi)` of
    the level's (parent, computed child) pairs laid end to end: one
    subtraction and two stores per cell, each cell its own."""
    comptime SW = 8
    var at = c_lo
    while at < c_hi:
        var pair = at // line2
        var c = at - pair * line2
        var stop = min(line2, c + (c_hi - at))
        var lslot = pair_l[pair] * line2
        var rslot = pair_r[pair] * line2
        var is_left = pair_left[pair] != 0
        var c0 = c
        while c0 + SW <= stop:
            var calc = hist.unsafe_load[width=SW](rslot + c0)
            var comp = hist.unsafe_load[width=SW](lslot + c0) - calc
            hist.unsafe_store(lslot + c0, calc if is_left else comp)
            hist.unsafe_store(rslot + c0, comp if is_left else calc)
            c0 += SW
        for cc in range(c0, stop):
            var calc = hist[rslot + cc]
            var comp = hist[lslot + cc] - calc
            hist[lslot + cc] = calc if is_left else comp
            hist[rslot + cc] = comp if is_left else calc
        at += stop - c


def _ord_subtract_level(
    mut hp: _PwHelper,
    pair_l: List[Int],
    pair_r: List[Int],
    pair_left: List[Int],
):
    """The subtraction on a partial pass: for every (left slot, right slot)
    pair with a row, the computed child (filed under the right slot) and
    the parent less it, stored on the sides `pair_left` names. Cell by
    cell, so the pairs' cells run in `_gbdt_units` contiguous ranges."""
    var n_pairs = len(pair_l)
    var line2 = hp.hist_line * 2
    var total = n_pairs * line2
    if total == 0:
        return
    var units = _gbdt_units(total, 3 * total)
    var chunk = host_predict_chunk(total, units)
    var hist_p = rebind[HostF32Ptr](hp.hist.unsafe_ptr())
    var l_p = rebind[GbdtIntPtr](pair_l.unsafe_ptr())
    var r_p = rebind[GbdtIntPtr](pair_r.unsafe_ptr())
    var left_p = rebind[GbdtIntPtr](pair_left.unsafe_ptr())

    def cells(unit: Int) {imm chunk, imm total, imm line2, imm hist_p, imm l_p, imm r_p, imm left_p}:
        _ord_subtract_range(
            unit * chunk, min(total, (unit + 1) * chunk), line2, hist_p, l_p,
            r_p, left_p,
        )

    host_parallelize(cells, units)
    _ = len(hp.hist)
    _ = len(pair_l)
    _ = len(pair_r)
    _ = len(pair_left)


def _ord_score_blocks(
    blk_lo: Int,
    blk_hi: Int,
    blocks: Int,
    hist_line: Int,
    plain_l2: Bool,
    part_count: Int,
    l2: Float32,
    score_before: Float32,
    part_stats: HostF32Ptr,
    hist: HostF32Ptr,
    c_noisy: HostF32Ptr,
    c_gain: HostF32Ptr,
    blk_score: HostF32Ptr,
    blk_gain: HostF32Ptr,
    blk_index: GbdtIntPtr,
):
    """The score blocks `[blk_lo, blk_hi)` of one helper: each of a block's
    128 threads takes its strided candidates' strict argmin, then the
    128-lane halving argmin (`_block_argmin_and_store`); the block's
    record lands in `blk_*[blk]`. A block reads the candidates and writes
    only its own record."""
    var t_score = List[Float32](length=GBDT_ORD_SCORE_BLOCK, fill=Float32(0.0))
    var t_gain = List[Float32](length=GBDT_ORD_SCORE_BLOCK, fill=Float32(0.0))
    var t_index = List[Int](length=GBDT_ORD_SCORE_BLOCK, fill=0)
    for blk in range(blk_lo, blk_hi):
        for tid in range(GBDT_ORD_SCORE_BLOCK):
            var th_score = GBDT_ORD_FLOAT32_MAX
            var th_gain = GBDT_ORD_FLOAT32_MAX
            var th_index = 0
            var i = blk * GBDT_ORD_SCORE_BLOCK
            while i < hist_line:
                if i + tid >= hist_line:
                    break
                var b = i + tid
                if plain_l2:
                    # `TL2ScoreCalcer`: Score = 0, per leaf both sides'
                    # `(-sum * sum) / (weight + lambda)` when the weight
                    # exceeds 1e-20, no normalization
                    var l2score = Float32(0.0)
                    for leaf in range(part_count):
                        var pw = part_stats[3 * leaf]
                        var psum = part_stats[3 * leaf + 1]
                        var hb = 2 * b + hist_line * leaf * 2
                        var wl = hist[hb]
                        var sl = hist[hb + 1]
                        var wr = max(pw - wl, Float32(0.0))
                        var sr = psum - sl
                        if wl > Float32(1e-20):
                            l2score += (-sl * sl) / (wl + l2)
                        if wr > Float32(1e-20):
                            l2score += (-sr * sr) / (wr + l2)
                    l2score *= Float32(1.0)
                    var l2gain = l2score - score_before
                    l2gain *= Float32(1.0)
                    if l2gain < th_gain:
                        th_score = l2score
                        th_gain = l2gain
                        th_index = b
                    i += GBDT_ORD_SCORE_BLOCK * blocks
                    continue
                var noisy = c_noisy[b]
                var gain = c_gain[b]
                if gain < th_gain:
                    th_score = noisy
                    th_gain = gain
                    th_index = b
                i += GBDT_ORD_SCORE_BLOCK * blocks
            t_score[tid] = th_score
            t_gain[tid] = th_gain
            t_index[tid] = th_index
        var step = GBDT_ORD_SCORE_BLOCK >> 1
        while step > 0:
            for tid in range(step):
                var take = t_gain[tid] > t_gain[tid + step]
                if t_gain[tid] == t_gain[tid + step] and t_index[tid] > t_index[tid + step]:
                    take = True
                if take:
                    t_score[tid] = t_score[tid + step]
                    t_index[tid] = t_index[tid + step]
                    t_gain[tid] = t_gain[tid + step]
            step >>= 1
        blk_score[blk] = t_score[0]
        blk_gain[blk] = t_gain[0]
        blk_index[blk] = t_index[0]


def _ord_block_records(
    hp: _PwHelper,
    part_stats: List[Float32],
    blocks: Int,
    plain_l2: Bool,
    part_count: Int,
    l2: Float32,
    score_before: Float32,
    c_noisy: List[Float32],
    c_gain: List[Float32],
    mut blk_score: List[Float32],
    mut blk_gain: List[Float32],
    mut blk_index: List[Int],
):
    """Every score block's record for one helper (`_ord_score_blocks`), the
    blocks in `_gbdt_units` contiguous ranges. The caller folds the records
    in block order (`_record_less`, challenger first)."""
    var hist_line = hp.hist_line
    var work = 2 * hist_line
    if plain_l2:
        work = 8 * hist_line * part_count
    var units = _gbdt_units(blocks, work)
    var chunk = host_predict_chunk(blocks, units)
    var stats_p = rebind[HostF32Ptr](part_stats.unsafe_ptr())
    var hist_p = rebind[HostF32Ptr](hp.hist.unsafe_ptr())
    var noisy_p = rebind[HostF32Ptr](c_noisy.unsafe_ptr())
    var gain_p = rebind[HostF32Ptr](c_gain.unsafe_ptr())
    var bs_p = rebind[HostF32Ptr](blk_score.unsafe_ptr())
    var bg_p = rebind[HostF32Ptr](blk_gain.unsafe_ptr())
    var bi_p = rebind[GbdtIntPtr](blk_index.unsafe_ptr())

    def score_blocks(unit: Int) {imm chunk, imm blocks, imm hist_line, imm plain_l2, imm part_count, imm l2, imm score_before, imm stats_p, imm hist_p, imm noisy_p, imm gain_p, imm bs_p, imm bg_p, imm bi_p}:
        _ord_score_blocks(
            unit * chunk, min(blocks, (unit + 1) * chunk), blocks, hist_line,
            plain_l2, part_count, l2, score_before, stats_p, hist_p, noisy_p,
            gain_p, bs_p, bg_p, bi_p,
        )

    host_parallelize(score_blocks, units)
    _ = len(part_stats)
    _ = len(hp.hist)
    _ = len(c_noisy)
    _ = len(c_gain)
    _ = len(blk_score)
    _ = len(blk_gain)
    _ = len(blk_index)


def _ord_mark_range(
    lo: Int,
    hi: Int,
    cindex: TreeU32Ptr,
    word_offset: Int,
    docs: GbdtIntPtr,
    mask: UInt32,
    value: UInt32,
    take_bin: Bool,
    bit: UInt32,
    bins: TreeU32Ptr,
) -> Int:
    """`split_subsets_from_desc`'s bit for the positions `[lo, hi)`: OR-ed
    in where the row goes right (`feature > value`, or `feature == value`
    on the take-bin arm, the rows of the split's category). Returns the
    count of positions that stay left."""
    var zeros = 0
    for i in range(lo, hi):
        var feature_val = cindex[word_offset + docs[i]] & mask
        var right = feature_val > value
        if take_bin:
            right = feature_val == value
        if right:
            bins[i] = bins[i] | bit
        if (bins[i] & bit) == UInt32(0):
            zeros += 1
    return zeros


def _ord_place_range(
    lo: Int,
    hi: Int,
    bit: UInt32,
    bins: TreeU32Ptr,
    indices: GbdtIntPtr,
    zero_at: Int,
    one_at: Int,
    new_bins: TreeU32Ptr,
    new_indices: GbdtIntPtr,
):
    """The stable one-bit radix placement of the positions `[lo, hi)`: the
    left ones from `zero_at`, the right ones from `one_at`, both ascending."""
    var z = zero_at
    var o = one_at
    for i in range(lo, hi):
        var v = bins[i]
        if (v & bit) == UInt32(0):
            new_bins[z] = v
            new_indices[z] = indices[i]
            z += 1
        else:
            new_bins[o] = v
            new_indices[o] = indices[i]
            o += 1


def _ord_split_subsets(
    mut s: _OrdSubsets,
    cindex: TreeU32Ptr,
    word_offset: Int,
    docs: List[Int],
    mask: UInt32,
    value: UInt32,
    take_bin: Bool,
    bit: UInt32,
):
    """`split_subsets_from_desc`: the split's bit OR-ed into every
    position's bin, then the STABLE one-bit radix sort of the bins and the
    indices (integers and moves only).

    The unit is a position range: task `c` marks its positions and counts
    those that stay left; the tasks' left starts are one running sum over
    the tasks (serial by nature: a prefix sum over `tasks` counts) and a
    task's right start is the total left plus the right ones before it;
    then task `c` places its positions from its own two starts. The ranges
    ascend, so the order on each side is the serial sort's at any task
    count."""
    var n = len(s.bins)
    var units = _gbdt_units(n, 8 * n)
    var chunk = host_predict_chunk(n, units)
    var zero_at = List[Int](length=units + 1, fill=0)
    var new_bins = List[UInt32](length=n, fill=UInt32(0))
    var new_indices = List[Int](length=n, fill=0)
    var docs_p = rebind[GbdtIntPtr](docs.unsafe_ptr())
    var bins_p = _u32_ptr(s.bins)
    var idx_p = rebind[GbdtIntPtr](s.indices.unsafe_ptr())
    var zero_p = rebind[GbdtIntPtr](zero_at.unsafe_ptr())
    var nb_p = _u32_ptr(new_bins)
    var ni_p = rebind[GbdtIntPtr](new_indices.unsafe_ptr())

    def mark(unit: Int) {imm chunk, imm n, imm cindex, imm word_offset, imm docs_p, imm mask, imm value, imm take_bin, imm bit, imm bins_p, imm zero_p}:
        zero_p[unit] = _ord_mark_range(
            min(n, unit * chunk), min(n, (unit + 1) * chunk), cindex,
            word_offset, docs_p, mask, value, take_bin, bit, bins_p,
        )

    host_parallelize(mark, units)
    var running = 0
    for u in range(units):  # driver-loop: over the tasks, the running count of left positions
        var count = zero_at[u]
        zero_at[u] = running
        running += count
    var total_zeros = running

    def place(unit: Int) {imm chunk, imm n, imm bit, imm total_zeros, imm bins_p, imm idx_p, imm zero_p, imm nb_p, imm ni_p}:
        var lo = min(n, unit * chunk)
        _ord_place_range(
            lo, min(n, (unit + 1) * chunk), bit, bins_p, idx_p, zero_p[unit],
            total_zeros + lo - zero_p[unit], nb_p, ni_p,
        )

    host_parallelize(place, units)
    _ = len(docs)
    _ = len(zero_at)
    # the old bins and indices are read by the tasks through `bins_p` /
    # `idx_p` and assigned just below: named past the joins (lane
    # fx-gbdt-segv, the freed-before-read of `_split_leaves`)
    _ = len(s.bins)
    _ = len(s.indices)
    s.bins = new_bins^
    s.indices = new_indices^


def _ordered_tree_structure(
    cindex: List[UInt32],
    mut helpers: List[_PwHelper],
    max_depth: Int,
    sw: List[Float32],
    sg: List[Float32],
    doc_ids: List[Int],
    part_bounds: List[Int],
    fold_count: Int,
    fold_bits: Int,
    fixed_scale: Float32,
    l2: Float32,
    word_offset_of: List[Int],
    shift_of: List[UInt32],
    mask_of: List[UInt32],
    one_hot_of: List[Bool],
    plain_l2: Bool = False,
    score_std_dev: Float32 = Float32(0.0),
    seed: UInt64 = UInt64(0),
    plain_cosine: Bool = False,
) raises -> List[_OrdSplit]:
    """The searcher without tree CTRs: `_ordered_tree_structure_levels` over
    `cindex` with no tree-CTR state (its arguments are documented there)."""
    var no_tree_ctr = List[GbdtTreeCtrHost]()
    var no_indexes = List[List[UInt32]]()
    var no_test = List[UInt32]()
    return _ordered_tree_structure_levels(
        _u32_ptr(cindex), helpers, max_depth, sw, sg, doc_ids, part_bounds,
        fold_count, fold_bits, fixed_scale, l2, word_offset_of, shift_of,
        mask_of, one_hot_of, no_tree_ctr, 0, no_indexes, no_test,
        plain_l2, score_std_dev, seed, plain_cosine,
    )


def _ordered_tree_structure_levels(
    # the search index's words (`_u32_ptr` of the list; with tree CTRs it is
    # `tree_cindexes[tree_p]`, whose slots the level driver rewrites in
    # place)
    cindex: TreeU32Ptr,
    mut helpers: List[_PwHelper],
    max_depth: Int,
    sw: List[Float32],
    sg: List[Float32],
    doc_ids: List[Int],
    part_bounds: List[Int],
    fold_count: Int,
    fold_bits: Int,
    fixed_scale: Float32,
    l2: Float32,
    word_offset_of: List[Int],
    shift_of: List[UInt32],
    mask_of: List[UInt32],
    # one flag per layout feature, or empty for none: a ONE-HOT feature's
    # candidates are equality tests (`TBinarizedFeature::OneHotFeature`).
    # Its histogram cells are the per-bin sums as filed (the scan skips its
    # prefix sum, `scan_pointwise_histograms_kernel`'s `skipFeature`), the
    # scorer reads the cell as one side and the partition less the cell as
    # the other with the same expressions as a threshold, and the partition
    # update sends `bin == split bin` right (`pw_split_resolve`'s take-bin
    # arm).
    one_hot_of: List[Bool],
    # TREE CTRS UNDER ORDERED (the device's level driver,
    # `fit_oblivious_tree_structure_levels`): empty for none, else the
    # fit's state, the index of the search index in `tree_cindexes` (every
    # permutation's index, or the one they share) and the held-out index.
    # With it: the slots are cleared before the tree, EVERY level's
    # histograms are built from scratch over zeroed slots, an accepted
    # winner goes through `take_winner` (a won candidate is recorded as the
    # persist column it became, which is also what later levels' repeat
    # test compares) and the next level's candidates are minted after the
    # partition.
    mut tree_ctr: List[GbdtTreeCtrHost],
    tree_p: Int,
    mut tree_cindexes: List[List[UInt32]],
    mut tree_test_cindex: List[UInt32],
    plain_l2: Bool = False,
    # the per-tree noise (`fit_oblivious_tree_structure`'s `score_std_dev`
    # and `seed`): a `TRandom(seed)` draw per LEVEL is the level's
    # `global_seed`, and each candidate's noise is the normal of
    # `advance_seed_k(global_seed + feature, 4)`, pinned-fma'd onto the
    # score (`pointwise_scores.mojo`'s dynamic cosine kernel)
    score_std_dev: Float32 = Float32(0.0),
    seed: UInt64 = UInt64(0),
    plain_cosine: Bool = False,
) raises -> List[_OrdSplit]:
    """The doc-parallel oblivious searcher's level loop. `fold_count > 1`
    with `plain_l2` False is the ordered fold arm (the dynamic cosine
    scorer); `fold_count == 1` with `plain_l2` True is the single-task arm
    the pointwise lane runs (`find_optimal_split_single_fold_kernel` with
    `TL2ScoreCalcer` at meta exponent 1, `pointwise_scores.mojo:809-950`)."""
    # The single-fold cosine caller compiled in the CPU GBDT binding on
    # Apple M1 at O0 (2026-10-28); runtime/identity remain pending. It uses
    # TCosineScoreCalcer, not the paired-fold dynamic cosine formula.
    if plain_cosine and (fold_count != 1 or plain_l2):
        raise Error("plain cosine needs one fold and cannot select L2")
    var level_driver = len(tree_ctr) > 0
    if level_driver:
        if fold_count < 2 or plain_l2:
            raise Error("the tree-CTR level driver needs the ordered fold arm")
        if tree_p < 0 or tree_p >= len(tree_cindexes):
            raise Error("tree CTR tree: the learn permutation has no index")
        tree_ctr[0].begin_tree(tree_p, tree_cindexes[tree_p])
        tree_ctr[0].learn_changed = False
    var doc_count = len(doc_ids)
    var have_one_hot = len(one_hot_of) != 0
    var max_parts = 1 << (fold_bits + max_depth)
    var stripe = 1 << _int_log2_floor_ceil(fold_count)  # exact integer ceil(log2), as the device stripe (lane fb13)
    var s = _OrdSubsets(
        List[UInt32](length=doc_count, fill=UInt32(0)),
        List[Int](length=doc_count, fill=0),
        List[Int](length=max_parts, fill=0),
        List[Int](length=max_parts, fill=0),
        List[Float32](length=3 * max_parts, fill=Float32(0.0)),
        List[Float32](length=doc_count, fill=Float32(0.0)),
        List[Float32](length=doc_count, fill=Float32(0.0)),
    )
    _identity_rows(s.indices, doc_count)
    # `write_fold_based_initial_bins`: part p over its concatenated range
    _ord_initial_bins(part_bounds, fold_count, s.bins)
    _update_subsets_stats(s, 1 << fold_bits, sw, sg)

    for h in range(len(helpers)):  # driver-loop: over the policy helpers
        # every slot starts the tree at +0.0: a fresh plane the first time,
        # then only the slots the previous tree wrote are zeroed again
        ref hz = helpers[h]
        var slots = (1 << max_depth) * fold_count
        var line2 = hz.hist_line * 2
        if len(hz.hist) != slots * line2 or len(hz.dirty_flag) != slots:
            hz.hist = List[Float32](length=slots * line2, fill=Float32(0.0))
            hz.dirty = List[Int]()
            hz.dirty_flag = List[Bool](length=slots, fill=False)
        else:
            _pw_zero_dirty(hz)
    var structure = List[_OrdSplit]()
    var score_before = Float32(0.0)
    var docs = List[Int](length=doc_count, fill=0)
    var level_rand = TRandom(seed)
    # the level's computed slots (their rows laid end to end in `seg_at`)
    # and, on a partial pass, its (left slot, right slot, computed-is-left)
    # pairs: the same for every helper
    var seg_slot = List[Int]()
    var seg_off = List[Int]()
    var seg_at = List[Int]()
    var pair_l = List[Int]()
    var pair_r = List[Int]()
    var pair_left = List[Int]()
    for depth in range(max_depth):  # driver-loop: over the tree's levels (a level reads the split before it)
        var global_seed = level_rand.next_uniform_l()
        _ord_gather_docs(doc_ids, s.indices, docs)
        var part_count = 1 << depth
        var full_pass = depth == 0 or level_driver
        if level_driver and depth > 0:
            # the device's per-level `reset_for_tree`: every slot back to
            # +0.0 before a from-scratch level
            for h in range(len(helpers)):  # driver-loop: over the policy helpers
                ref hl = helpers[h]
                _pw_zero_dirty(hl)
        var ny = part_count if full_pass else part_count // 2
        seg_slot.clear()
        seg_off.clear()
        seg_at.clear()
        pair_l.clear()
        pair_r.clear()
        pair_left.clear()
        var seg_rows = 0
        for y in range(ny):  # driver-loop: over the level's (leaf, fold) slots
            for z in range(fold_count):
                var data_part: Int
                var hist_slot: Int
                if full_pass:
                    data_part = y * stripe + z
                    hist_slot = y * fold_count + z
                else:
                    var left = y * stripe + z
                    var right = (y | ny) * stripe + z
                    data_part = left if s.p_sz[left] < s.p_sz[right] else right
                    hist_slot = (y | ny) * fold_count + z
                    # an empty parent's slot and its computed child are
                    # all +0.0 (by induction from the fresh slots), and
                    # +0.0 - +0.0 is +0.0: nothing to subtract
                    if s.p_sz[left] != 0 or s.p_sz[right] != 0:
                        pair_l.append(y * fold_count + z)
                        pair_r.append(hist_slot)
                        pair_left.append(1 if s.p_sz[left] < s.p_sz[right] else 0)
                if s.p_sz[data_part] == 0:
                    continue
                seg_slot.append(hist_slot)
                seg_off.append(s.p_off[data_part])
                seg_at.append(seg_rows)
                seg_rows += s.p_sz[data_part]
        seg_at.append(seg_rows)
        var best_fid = GBDT_ORD_SENTINEL
        var best_bin = UInt32(0)
        var best_score = GBDT_ORD_FLOAT32_MAX
        var best_gain = GBDT_ORD_FLOAT32_MAX
        # `compute_optimal_split_dev`'s per-helper seed advance
        # (`pointwise_scores_calcer.mojo`): helper i scores with the i-th
        # draw of `TRandom(level seed)`
        var helper_rand = TRandom(global_seed)
        for h in range(len(helpers)):  # driver-loop: over the policy helpers (their seeds are one generator's draws in order)
            ref hp = helpers[h]
            var helper_seed = helper_rand.next_uniform_l()
            var hist_line = hp.hist_line
            var n_feat = len(hp.gids)
            # ---- the histograms (`compute_hist2`) ----
            for k in range(len(seg_slot)):  # driver-loop: over the computed slots, their marks
                hp.mark(seg_slot[k])
            _ord_hist_level(
                hp, cindex, docs, s.g_weight, s.g_target, seg_slot, seg_off,
                seg_at, fixed_scale,
            )
            # ---- the scan over the computed slots ----
            _ord_scan_level(hp, seg_slot, have_one_hot, one_hot_of)
            # ---- the subtraction on a partial pass ----
            if not full_pass:
                for k in range(len(pair_l)):  # driver-loop: over the level's slot pairs, their marks
                    hp.mark(pair_l[k])
                    hp.mark(pair_r[k])
                _ord_subtract_level(hp, pair_l, pair_r, pair_left)

            # ---- the dynamic cosine score and the block argmins ----
            var blocks = (hist_line + GBDT_ORD_SCORE_BLOCK - 1) // GBDT_ORD_SCORE_BLOCK
            if blocks > 32:
                blocks = 32
            if blocks < 1:
                blocks = 1
            var bf_feature = List[Int](length=hist_line, fill=0)
            var bf_bin = List[Int](length=hist_line, fill=0)
            for f in range(n_feat):  # driver-loop: over the helper's features and their borders (the layout)
                for b in range(hp.folds[f]):
                    bf_feature[hp.first[f] + b] = hp.gids[f]
                    bf_bin[hp.first[f] + b] = b
            var loc_fid = GBDT_ORD_SENTINEL
            var loc_bin = UInt32(0)
            var loc_score = GBDT_ORD_FLOAT32_MAX
            var loc_gain = GBDT_ORD_FLOAT32_MAX
            # the score of every candidate (each candidate's own adds run
            # in the kernel's (leaf, fold) order, the same expressions),
            # the candidates across the tasks
            var c_noisy = List[Float32]()
            var c_gain = List[Float32]()
            if plain_cosine:
                c_noisy = _plain_cosine_candidates(
                    hp, s.part_stats, part_count, l2, score_std_dev,
                    helper_seed, bf_feature, score_before, c_gain,
                )
            elif not plain_l2:
                c_noisy = _dynamic_cosine_candidates(
                    hp, s.part_stats, part_count, fold_count, stripe, l2,
                    score_std_dev, helper_seed, bf_feature, score_before,
                    c_gain,
                )
            var blk_score = List[Float32](length=blocks, fill=GBDT_ORD_FLOAT32_MAX)
            var blk_gain = List[Float32](length=blocks, fill=GBDT_ORD_FLOAT32_MAX)
            var blk_index = List[Int](length=blocks, fill=0)
            _ord_block_records(
                hp, s.part_stats, blocks, plain_l2, part_count, l2,
                score_before, c_noisy, c_gain, blk_score, blk_gain, blk_index,
            )
            for blk in range(blocks):  # driver-loop: over the helper's score blocks, the record fold in block order
                var c_fid = UInt32(0)
                var c_bin = UInt32(0)
                if blk_index[blk] < hist_line:
                    c_fid = UInt32(bf_feature[blk_index[blk]])
                    c_bin = UInt32(bf_bin[blk_index[blk]])
                if _record_less(blk_gain[blk], c_fid, c_bin, loc_gain, loc_fid, loc_bin):
                    loc_fid = c_fid
                    loc_bin = c_bin
                    loc_score = blk_score[blk]
                    loc_gain = blk_gain[blk]
            # `pw_fold_winner_kernel`: the helper's record into the incumbent
            if _record_less(loc_gain, loc_fid, loc_bin, best_gain, best_fid, best_bin):
                best_fid = loc_fid
                best_bin = loc_bin
                best_score = loc_score
                best_gain = loc_gain

        # ---- the gates (in level order) and the split ----
        if best_fid == GBDT_ORD_SENTINEL:
            raise Error(
                "best split is undefined at depth " + String(depth)
                + ": every candidate scored non-finite. Theirs raises the"
                " same way (`:122`)."
            )
        score_before = best_score
        var fid = Int(best_fid)
        var seen = False
        for i in range(len(structure)):  # driver-loop: over the tree's splits so far
            if structure[i].feature == fid and structure[i].bin == Int(best_bin):
                seen = True
        if seen:
            break
        var kept_fid = fid
        if level_driver:
            kept_fid = tree_ctr[0].take_winner(
                tree_p, tree_cindexes, tree_test_cindex, fid, Int(best_bin)
            )
            if kept_fid < 0:
                # the winner sits on a slot that holds no column: no split
                break
        structure.append(_OrdSplit(kept_fid, Int(best_bin)))
        if depth + 1 == max_depth:
            break
        var bit = UInt32(1) << UInt32(depth + fold_bits)
        var value = best_bin << shift_of[fid]
        var mask = mask_of[fid] << shift_of[fid]
        # take-bin (a one-hot feature): the rows of the split's category go
        # right; else the rows above the border
        var take_bin = False
        if have_one_hot:
            take_bin = one_hot_of[fid]
        _ord_split_subsets(
            s, cindex, word_offset_of[fid], docs, mask, value, take_bin, bit
        )
        _update_subsets_stats(s, 1 << (depth + 1 + fold_bits), sw, sg)
        if level_driver:
            # the next level's candidates on the search index (the
            # partition above read the won slot before this rewrites it)
            tree_ctr[0].mint_level(tree_p, tree_cindexes[tree_p])
            tree_ctr[0].learn_changed = False
    if level_driver:
        tree_ctr[0].end_tree()
    return structure^


def _plain_cosine_range(
    b_lo: Int,
    b_hi: Int,
    hist_line: Int,
    hist: HostF32Ptr,
    part_stats: HostF32Ptr,
    part_count: Int,
    l2: Float32,
    score_std_dev: Float32,
    helper_seed: UInt64,
    bf_feature: GbdtIntPtr,
    score_before: Float32,
    scores: HostF32Ptr,
    gains: HostF32Ptr,
):
    """`_plain_cosine_candidates` for the candidates `[b_lo, b_hi)`: each
    candidate's fma chain over the leaves in leaf order is its own."""
    for b in range(b_lo, b_hi):
        var score = Float32(0)
        var denum = Float32(1e-10)
        for leaf in range(part_count):
            var base = hist_line * leaf * 2 + 2 * b
            var wl = hist[base]
            var sl = hist[base + 1]
            var wr = max(part_stats[3 * leaf] - wl, Float32(0))
            var sr = part_stats[3 * leaf + 1] - sl
            var ml = sl / (wl + l2) if wl > Float32(0) else Float32(0)
            score = identical_mul_add(sl, ml, score)
            denum = identical_mul_add(wl * ml, ml, denum)
            var mr = sr / (wr + l2) if wr > Float32(0) else Float32(0)
            score = identical_mul_add(sr, mr, score)
            denum = identical_mul_add(wr * mr, mr, denum)
        var res = GBDT_ORD_FLOAT32_MAX
        if denum > Float32(1e-15):
            res = -score / identical_sqrt(denum)
        if score_std_dev != Float32(0):
            var seed = advance_seed_k(helper_seed + UInt64(bf_feature[b]), 4)
            var draw = next_normal_f(seed)
            res = identical_mul_add(draw[0], score_std_dev, res)
        scores[b] = res
        gains[b] = identical_mul_add(res, Float32(1), -score_before)


def _plain_cosine_candidates(
    hp: _PwHelper,
    part_stats: List[Float32],
    part_count: Int,
    l2: Float32,
    score_std_dev: Float32,
    helper_seed: UInt64,
    bf_feature: List[Int],
    score_before: Float32,
    mut gain_out: List[Float32],
) -> List[Float32]:
    """Score a plain single-fold histogram with Cosine or NewtonCosine. The
    unit is one CANDIDATE (its own chain over the leaves, its own draw),
    the candidates in `_gbdt_units` contiguous ranges."""
    # pointwise_scores.mojo::ScoreCalcer: 1e-10 starts DenumSqr for the
    # single-fold calcer; the ordered dynamic kernel instead starts 1e-20.
    # normalize is False on the public fit. Left then right leaf terms and
    # the feature-based normal draw follow the native GPU calcer exactly.
    var hist_line = hp.hist_line
    var scores = List[Float32](length=hist_line, fill=Float32(0))
    gain_out = List[Float32](length=hist_line, fill=Float32(0))
    var units = _gbdt_units(hist_line, 12 * hist_line * part_count)
    var chunk = host_predict_chunk(hist_line, units)
    var hist_p = rebind[HostF32Ptr](hp.hist.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](part_stats.unsafe_ptr())
    var bf_p = rebind[GbdtIntPtr](bf_feature.unsafe_ptr())
    var scores_p = rebind[HostF32Ptr](scores.unsafe_ptr())
    var gains_p = rebind[HostF32Ptr](gain_out.unsafe_ptr())

    def candidates(unit: Int) {imm chunk, imm hist_line, imm hist_p, imm stats_p, imm part_count, imm l2, imm score_std_dev, imm helper_seed, imm bf_p, imm score_before, imm scores_p, imm gains_p}:
        _plain_cosine_range(
            unit * chunk, min(hist_line, (unit + 1) * chunk), hist_line,
            hist_p, stats_p, part_count, l2, score_std_dev, helper_seed, bf_p,
            score_before, scores_p, gains_p,
        )

    host_parallelize(candidates, units)
    _ = len(hp.hist)
    _ = len(part_stats)
    _ = len(bf_feature)
    _ = len(gain_out)
    _ = len(scores)
    return scores^


@no_inline
def _dynamic_cosine_range(
    b_lo: Int,
    b_hi: Int,
    hist_line: Int,
    hptr: HostF32Ptr,
    part_stats: HostF32Ptr,
    part_count: Int,
    fold_count: Int,
    stripe: Int,
    l2: Float32,
    score_std_dev: Float32,
    helper_seed: UInt64,
    bf_feature: GbdtIntPtr,
    score_before: Float32,
    sptr: HostF32Ptr,
    dptr: HostF32Ptr,
    noisy_out: HostF32Ptr,
    gain_out: HostF32Ptr,
):
    """`_dynamic_cosine_candidates` for the candidates `[b_lo, b_hi)`: the
    (leaf, fold pair) nest outside, the range's candidates inside, eight
    to a step and the rest one by one (the two forms are the same
    operations lane by lane, so where a range ends moves no bit)."""
    comptime W = 8
    for leaf in range(part_count):
        var fold = 0
        while fold < fold_count:
            var learn_off = leaf * stripe + fold
            var test_off = leaf * stripe + fold + 1
            var plw = part_stats[3 * learn_off]
            var pls = part_stats[3 * learn_off + 1]
            var ptw = part_stats[3 * test_off]
            var pts = part_stats[3 * test_off + 1]
            var h_learn = hist_line * (leaf * fold_count + fold) * 2
            var h_test = hist_line * (leaf * fold_count + fold + 1) * 2
            var vb = b_lo
            while vb + W <= b_hi:
                # W candidates per step, lane k is candidate vb + k: the
                # scalar loop below, lane by lane (each `x += a * b` one fma,
                # written out rather than left to the compiler's contraction)
                var lpair = hptr.unsafe_load[width = 2 * W](h_learn + 2 * vb).deinterleave()
                var tpair = hptr.unsafe_load[width = 2 * W](h_test + 2 * vb).deinterleave()
                var sc = sptr.unsafe_load[width=W](vb)
                var dn = dptr.unsafe_load[width=W](vb)
                var wel = lpair[0]
                var sel = lpair[1]
                var wtl = tpair[0]
                var stl = tpair[1]
                var zero = SIMD[DType.float32, W](0.0)
                var wer = max(SIMD[DType.float32, W](plw) - wel, zero)
                var ser = SIMD[DType.float32, W](pls) - sel
                var wtr = max(SIMD[DType.float32, W](ptw) - wtl, zero)
                var sum_tr = SIMD[DType.float32, W](pts) - stl
                var mu_l = wel.gt(zero).select(sel / (wel + l2), zero)
                sc = identical_mul_add_simd[W](stl, mu_l, sc)
                dn = identical_mul_add_simd[W](wtl * mu_l, mu_l, dn)
                var mu_r = wer.gt(zero).select(ser / (wer + l2), zero)
                sc = identical_mul_add_simd[W](sum_tr, mu_r, sc)
                dn = identical_mul_add_simd[W](wtr * mu_r, mu_r, dn)
                sptr.unsafe_store(vb, sc)
                dptr.unsafe_store(vb, dn)
                vb += W
            for b in range(vb, b_hi):
                var current = 2 * b
                var sc = sptr[b]
                var dn = dptr[b]
                var wel = hptr[current + h_learn]
                var wer = max(plw - wel, Float32(0.0))
                var sel = hptr[current + h_learn + 1]
                var ser = pls - sel
                var wtl = hptr[current + h_test]
                var wtr = max(ptw - wtl, Float32(0.0))
                var stl = hptr[current + h_test + 1]
                var sum_tr = pts - stl
                var mu_l = Float32(0.0)
                if wel > Float32(0.0):
                    mu_l = sel / (wel + l2)
                # the four accumulates in ONE rounding each, the fusion the
                # default (contract=fast) build formed (lane/explicit-fma-contract-proof)
                sc = identical_mul_add(stl, mu_l, sc)
                dn = identical_mul_add(wtl * mu_l, mu_l, dn)
                var mu_r = Float32(0.0)
                if wer > Float32(0.0):
                    mu_r = ser / (wer + l2)
                sc = identical_mul_add(sum_tr, mu_r, sc)
                dn = identical_mul_add(wtr * mu_r, mu_r, dn)
                sptr[b] = sc
                dptr[b] = dn
            fold += 2
    var last_feature = -1
    var last_draw = Float32(0.0)
    for b in range(b_lo, b_hi):
        var sc = sptr[b]
        var denum_sqr = dptr[b]
        if denum_sqr > Float32(1e-15):
            sc = -sc / identical_sqrt(denum_sqr)
        else:
            sc = GBDT_ORD_FLOAT32_MAX
        sc *= Float32(1.0)
        var noisy = sc
        if score_std_dev != Float32(0.0):
            if bf_feature[b] != last_feature:
                var nseed = advance_seed_k(
                    helper_seed + UInt64(UInt32(bf_feature[b])), 4
                )
                last_draw = next_normal_f(nseed)[0]
                last_feature = bf_feature[b]
            noisy = identical_mul_add(last_draw, score_std_dev, noisy)
        noisy_out[b] = noisy
        gain_out[b] = (noisy - score_before) * Float32(1.0)


def _dynamic_cosine_candidates(
    hp: _PwHelper,
    part_stats: List[Float32],
    part_count: Int,
    fold_count: Int,
    stripe: Int,
    l2: Float32,
    score_std_dev: Float32,
    helper_seed: UInt64,
    bf_feature: List[Int],
    score_before: Float32,
    mut gain_out: List[Float32],
) -> List[Float32]:
    """`find_optimal_split_cosine_kernel`'s per-candidate score (the
    (estimate, test) fold pairs, `denum_sqr` from 1e-20), its noise and its
    gain, for every candidate of the helper at once: returns the noisy
    scores and fills `gain_out`. Each candidate accumulates over (leaf,
    fold pair) in the kernel's order with the kernel's expressions; only
    the loop nest is turned so a (leaf, fold pair)'s two histogram lines are
    read front to back. The noise draw depends on the candidate's feature
    only, so it is drawn once per feature run.

    The unit is one CANDIDATE (one device thread's strided share of the
    score block): its chain over (leaf, fold pair) and its draw are its
    own, so the candidates run in `_gbdt_units` contiguous ranges
    (`_dynamic_cosine_range`)."""
    var hist_line = hp.hist_line
    var score = List[Float32](length=hist_line, fill=Float32(0.0))
    var denum = List[Float32](length=hist_line, fill=Float32(1e-20))
    var noisy_out = List[Float32](length=hist_line, fill=Float32(0.0))
    gain_out = List[Float32](length=hist_line, fill=Float32(0.0))
    var units = _gbdt_units(hist_line, 8 * hist_line * part_count * max(fold_count, 1))
    var chunk = host_predict_chunk(hist_line, units)
    var hist_p = rebind[HostF32Ptr](hp.hist.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](part_stats.unsafe_ptr())
    var bf_p = rebind[GbdtIntPtr](bf_feature.unsafe_ptr())
    var score_p = rebind[HostF32Ptr](score.unsafe_ptr())
    var denum_p = rebind[HostF32Ptr](denum.unsafe_ptr())
    var noisy_p = rebind[HostF32Ptr](noisy_out.unsafe_ptr())
    var gains_p = rebind[HostF32Ptr](gain_out.unsafe_ptr())

    def candidates(unit: Int) {imm chunk, imm hist_line, imm hist_p, imm stats_p, imm part_count, imm fold_count, imm stripe, imm l2, imm score_std_dev, imm helper_seed, imm bf_p, imm score_before, imm score_p, imm denum_p, imm noisy_p, imm gains_p}:
        _dynamic_cosine_range(
            unit * chunk, min(hist_line, (unit + 1) * chunk), hist_line,
            hist_p, stats_p, part_count, fold_count, stripe, l2,
            score_std_dev, helper_seed, bf_p, score_before, score_p, denum_p,
            noisy_p, gains_p,
        )

    host_parallelize(candidates, units)
    _ = len(hp.hist)
    _ = len(part_stats)
    _ = len(bf_feature)
    _ = len(score)
    _ = len(denum)
    _ = len(gain_out)
    _ = len(noisy_out)
    return noisy_out^


def _record_less(
    gain_a: Float32, fid_a: UInt32, bin_a: UInt32,
    gain_b: Float32, fid_b: UInt32, bin_b: UInt32,
) -> Bool:
    """`_record_less` (`pointwise_split_resolve.mojo`): gain, then feature
    id, then bin, all strict."""
    if gain_a < gain_b:
        return True
    elif gain_a == gain_b:
        if fid_a < fid_b:
            return True
        elif fid_a == fid_b:
            return bin_a < bin_b
        return False
    return False


# ===========================================================================
# THE LEAVES (`ordered_estimate_and_apply`)
# ===========================================================================


def _ordered_estimate_and_apply(
    estimate_size: Int,
    apply_size: Int,
    n_leaves: Int,
    y: List[Float32],
    w: List[Float32],
    permutation: List[Int],
    bins: List[Int],
    mut cursor: List[Float32],
    rate: Float32,
    l2: Float32,
) raises -> List[Float32]:
    """`ordered_estimate_and_apply` (`dynamic_boosting.mojo:104-156`)."""
    if estimate_size < 1 or estimate_size > apply_size:
        raise Error("ordered estimation requires 0 < prefix <= cursor size")
    # THE SPLITS (lane gbdt-host-par; the helpers are below, with the
    # host pool's imports): every step is the device launch it names, its
    # unit a position range, a (sum, 512-thread chunk) or a leaf.
    var gy = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gw = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gc = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gb = List[Int](length=estimate_size, fill=0)
    _ord_prefix_gather(y, w, True, permutation, bins, cursor, estimate_size, gy, gw, gc, gb)
    # `partition_from_bins`: the stable counting sort
    var sizes = List[Int](length=n_leaves, fill=0)
    var offsets = List[Int](length=n_leaves, fill=0)
    var row_index = List[Int](length=estimate_size, fill=0)
    _ord_counting_sort(gb, estimate_size, n_leaves, sizes, offsets, row_index)
    # `_estimate_and_apply`: the gathers into leaf order
    var n = estimate_size
    var t_target = List[Float32](length=n, fill=Float32(0.0))
    var t_weight = List[Float32](length=n, fill=Float32(0.0))
    var t_cursor = List[Float32](length=n, fill=Float32(0.0))
    _gather_rows(gy, row_index, n, t_target)
    _gather_rows(gw, row_index, n, t_weight)
    _gather_rows(gc, row_index, n, t_cursor)
    # `make_bin_optimized_oracle`: WeightsCpu from the one-stat reduce
    # (`partition_stats_chunks(32, 1)` = 64 chunks)
    var weight_sums = List[Float32](length=n_leaves, fill=Float32(0.0))
    _pinned_partition_stats(
        t_weight, offsets, sizes, 2 * GBDT_ORD_PINNED_SM, weight_sums
    )
    var lambda_reg = Float64(l2)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0
    # `move_to(zero)` (`add_bin_model_value_kernel` adds 0.0 to every row),
    # then `pointwise_target_kernel[RMSE, estimation=True]`
    var stats = List[Float32](length=2 * n, fill=Float32(0.0))
    _ord_rmse_planes(t_target, t_weight, n, t_cursor, stats)
    comptime EPS_1E20F = Float64(Float32(1e-20))
    # the (leaf, plane) sums at two planes (`partition_stats_chunks(32, 2)`)
    var leaf_sums = _partition_stats_level(
        stats, n, 2, (2 * GBDT_ORD_PINNED_SM + 1) // 2, offsets, sizes, n_leaves
    )
    var leaves = List[Float32]()
    for leaf in range(n_leaves):  # driver-loop: over the leaves
        var g = Float64(leaf_sums[2 * leaf])
        var h = Float64(leaf_sums[2 * leaf + 1]) + lambda_reg
        var direction = Float32(0.0)
        if h > 0:
            direction = Float32(g / (h + EPS_1E20F))
        var moved = Float32(Float64(Float32(0.0)) + 1.0 * Float64(direction))
        if Float64(weight_sums[leaf]) < 1e-20:
            moved = Float32(0.0)
        leaves.append(moved)
    # `_ordered_apply_kernel` over every apply position
    _ord_apply_leaves(leaves, bins, permutation, rate, apply_size, cursor)
    return leaves^


# ===========================================================================
# THE FIT
# ===========================================================================


def _ord_all_finite(values: List[Float32]) -> Bool:
    """Whether every value is finite: a test per element (any non-finite
    element gives the caller's one refusal), the elements in `_gbdt_units`
    contiguous ranges, one flag a task."""
    var n = len(values)
    var units = _gbdt_units(n, n)
    var chunk = host_predict_chunk(n, units)
    var bad = List[Int](length=max(units, 1), fill=0)
    var v_p = rebind[HostF32Ptr](values.unsafe_ptr())
    var bad_p = rebind[GbdtIntPtr](bad.unsafe_ptr())

    def scan(unit: Int) {imm chunk, imm n, imm v_p, imm bad_p}:
        var found = 0
        for i in range(unit * chunk, min(n, (unit + 1) * chunk)):
            if not isfinite(v_p[i]):
                found = 1
        bad_p[unit] = found

    host_parallelize(scan, units)
    var any_bad = 0
    for u in range(len(bad)):  # driver-loop: over the tasks' flags
        any_bad += bad[u]
    _ = len(values)
    return any_bad == 0


def _ord_fold_doc_ids(
    permutation: List[Int], size: Int, offset: Int, mut doc_ids: List[Int]
):
    """`make_fold_doc_indices` for one fold: `doc_ids[offset + p] =
    permutation[p]` over the fold's prefix (a move per position), the
    positions in `_gbdt_units` contiguous ranges."""
    var units = _gbdt_units(size, 2 * size)
    var chunk = host_predict_chunk(size, units)
    var perm_p = rebind[GbdtIntPtr](permutation.unsafe_ptr())
    var doc_p = rebind[GbdtIntPtr](doc_ids.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm size, imm offset, imm perm_p, imm doc_p}:
        for i in range(unit * chunk, min(size, (unit + 1) * chunk)):
            doc_p[offset + i] = perm_p[i]

    host_parallelize(rows, units)
    _ = len(permutation)
    _ = len(doc_ids)


def _ord_rmse_fold_targets(
    y: List[Float32],
    w: List[Float32],
    permutation: List[Int],
    cursor: List[Float32],
    size: Int,
    offset: Int,
    mut sw: List[Float32],
    mut sg: List[Float32],
):
    """`_ordered_target_kernel` (`dynamic_boosting.mojo:44-60`) for one
    fold: position `i` of the fold's prefix takes the row's weight and
    `ftz(identical_mul(w, y - cursor[i]))`. One device thread a position,
    the positions in `_gbdt_units` contiguous ranges. (The folds are the
    caller's driver loop: a fold's prefix is the unit of the fit's plan,
    its positions are the units here.)"""
    var units = _gbdt_units(size, 6 * size)
    var chunk = host_predict_chunk(size, units)
    var y_p = rebind[HostF32Ptr](y.unsafe_ptr())
    var w_p = rebind[HostF32Ptr](w.unsafe_ptr())
    var perm_p = rebind[GbdtIntPtr](permutation.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm size, imm offset, imm y_p, imm w_p, imm perm_p, imm cursor_p, imm sw_p, imm sg_p}:
        for i in range(unit * chunk, min(size, (unit + 1) * chunk)):
            var row = perm_p[i]
            var wi = w_p[row]
            sw_p[offset + i] = wi
            sg_p[offset + i] = ftz(identical_mul(wi, y_p[row] - cursor_p[i]))

    host_parallelize(rows, units)
    _ = len(y)
    _ = len(w)
    _ = len(permutation)
    _ = len(cursor)
    _ = len(sw)
    _ = len(sg)


def gbdt_ordered_rmse_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    permutation_u32: List[UInt32],
    n_estimators: Int,
    max_depth: Int,
    border_count: Int,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    sample_weight: List[Float32] = List[Float32](),
) raises -> String:
    """`train_ordered_rmse` then `fit_ordered_rmse`; returns the model text.
    `sample_weight` (row order, empty: unit) is the device's `dw` buffer
    (`gbdt/methods/dynamic_boosting.mojo`), the caller's weights or a fill
    of 1.0, read by the same statements either way (lane gbdt-cpu-rest,
    2026-11-23)."""
    if len(sample_weight) != 0 and len(sample_weight) != n_rows:
        raise Error("ordered RMSE sample weight shape mismatch")
    if n_rows != len(y) or n_features < 1 or len(x_colmajor) != n_rows * n_features:
        raise Error("train_ordered_rmse input shape mismatch")
    if border_count < 1 or border_count > 255:
        raise Error("train_ordered_rmse supports 1..255 borders")
    # ---- 1. the grid and the compressed index ----
    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    var nan_treatment = List[Int]()
    comptime if IDN_ORDERED_RMSE_DEVICE_GRID:
        # lane/fam-gbdt: the device fit takes `device_float_borders` over
        # every row at NaN mode Forbidden; `gbdt_host_grid` is that build on
        # host memory (no subsample at `border_build_max_samples = 0`).
        if not _ord_all_finite(x_colmajor):
            raise Error("train_ordered_rmse requires finite numeric features")
        var dev_grid = gbdt_host_grid(
            x_colmajor, n_rows, n_features, border_count, 0, UInt64(0),
            NAN_MODE_FORBIDDEN,
        )
        for f in range(n_features):  # driver-loop: over the features, the grid's borders
            fold_counts.append(len(dev_grid.borders[f]))
            borders.append(dev_grid.borders[f].copy())
            nan_treatment.append(NAN_TREATMENT_AS_IS)
    else:
        # KEPT SERIAL, named (lane gbdt-host-par): the grid of the build
        # with IDN_ORDERED_RMSE_DEVICE_GRID off, which no shipped build
        # compiles (`best_split` per column on the calling thread, a
        # refusal at the first non-finite value in column order)
        for f in range(n_features):
            var column = List[Float32]()
            for r in range(n_rows):
                var value = x_colmajor[f * n_rows + r]
                if not isfinite(value):
                    raise Error("train_ordered_rmse requires finite numeric features")
                column.append(value)
            var grid = best_split(column^, border_count)
            fold_counts.append(len(grid))
            borders.append(grid^)
            nan_treatment.append(NAN_TREATMENT_AS_IS)
    var layout = build_layout(fold_counts)
    var host_grid = GbdtHostGrid(borders.copy(), fold_counts.copy(), nan_treatment.copy())
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, host_grid, layout)
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
        var gids = List[Int]()
        var offs = List[Int]()
        var firsts = List[Int]()
        var folds = List[Int]()
        var hist_line = 0
        for k in range(blk.count()):
            var f = blk.feature_ids[k]
            gids.append(f)
            offs.append(Int(layout.features[f].offset) * n_rows)
            firsts.append(Int(blk.fold_offset[k]))
            folds.append(Int(blk.folds[k]))
            hist_line += Int(blk.folds[k])
        helpers.append(_PwHelper(
            blk.policy, gids^, offs^, firsts^, folds^, hist_line, List[Float32](),
            List[Int](), List[Bool](),
        ))
    var feat_offset = List[Int](length=n_features, fill=0)
    var feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    for f in range(n_features):
        feat_offset[f] = Int(layout.features[f].offset) * n_rows
        feat_shift[f] = layout.features[f].shift
        feat_mask[f] = layout.features[f].mask

    # ---- 2. `fit_ordered_rmse` ----
    var n = n_rows
    if n < 4 or len(permutation_u32) != n or n_estimators < 1:
        raise Error("ordered RMSE needs >=4 rows, a permutation and >=1 tree")
    if max_depth < 1 or max_depth > 8:
        raise Error("ordered RMSE supports depth 1..8 and positive SM count")
    if not isfinite(learning_rate) or learning_rate <= Float32(0) or not isfinite(l2_leaf_reg) or l2_leaf_reg < Float32(0):
        raise Error("ordered RMSE requires finite positive rate and nonnegative L2")
    var permutation = List[Int](length=n, fill=0)
    var seen = List[Bool](length=n, fill=False)
    var w: List[Float32]
    if len(sample_weight) != 0:
        w = sample_weight.copy()
    else:
        w = List[Float32](length=n, fill=Float32(1.0))
    var weight_sum = Float64(0)
    var max_weight = Float64(0)
    var residual_bound = Float64(0)
    # SERIAL BY NATURE (admission, once per fit): the bijection test is a
    # running set (a row is refused when an EARLIER position named it) and
    # the weight total is one Float64 chain in position order
    for i in range(n):
        var row = Int(permutation_u32[i])
        if row >= n or seen[row]:
            raise Error("ordered RMSE permutation must be a bijection")
        seen[row] = True
        permutation[i] = row
        var wi = w[i]
        if not isfinite(y[i]) or not isfinite(wi) or wi < Float32(0) or not isfinite(identical_mul(wi, y[i])):
            raise Error("ordered RMSE targets/weights must be finite; weights nonnegative")
        weight_sum += Float64(wi)
        max_weight = max(max_weight, Float64(wi))
        residual_bound = max(residual_bound, abs(Float64(y[i])))
    if weight_sum <= Float64(0):
        raise Error("ordered RMSE weights sum to zero")
    var bounds = _ordered_folds(n, 2.0, 100)
    var n_folds = len(bounds) // 2
    var fold_count = 2 * n_folds
    var fold_bits = _int_log2_floor_ceil(fold_count)
    if fold_bits + max_depth >= 32:
        raise Error("1 << (FoldBits + maxDepth) does not fit a ui32 bin")
    var part_bounds = List[Int]()
    var total = 0
    part_bounds.append(0)
    for f in range(n_folds):  # driver-loop: over the folds, the running part bounds
        var est_right = bounds[2 * f]
        var qe_right = bounds[2 * f + 1]
        total += qe_right
        part_bounds.append(part_bounds[len(part_bounds) - 1] + est_right)
        part_bounds.append(part_bounds[len(part_bounds) - 1] + (qe_right - est_right))
    var doc_ids = List[Int](length=total, fill=0)
    var doc_at = 0
    for f in range(n_folds):  # driver-loop: over the folds, each prefix's positions across the tasks
        _ord_fold_doc_ids(permutation, bounds[2 * f + 1], doc_at, doc_ids)
        doc_at += bounds[2 * f + 1]
    var cursors = List[List[Float32]]()
    for f in range(n_folds):  # driver-loop: over the folds, one cursor each
        cursors.append(List[Float32](length=bounds[2 * f + 1], fill=Float32(0.0)))
    var estimation = List[Float32](length=n, fill=Float32(0.0))

    var tree_split_offsets = List[Int]()
    tree_split_offsets.append(0)
    var split_features = List[Int]()
    var split_bins = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()
    for iteration in range(n_estimators):
        var magnitude_bound = Float64(total) * max_weight * max(Float64(1), residual_bound)
        if not isfinite(magnitude_bound):
            raise Error("ordered RMSE fixed-point magnitude bound overflow")
        var scale = Float32(choose_scale(magnitude_bound, total))
        var sw = List[Float32](length=total, fill=Float32(0.0))
        var sg = List[Float32](length=total, fill=Float32(0.0))
        var offset = 0
        for f in range(n_folds):  # driver-loop: over the folds, each prefix's positions across the tasks
            var size = bounds[2 * f + 1]
            _ord_rmse_fold_targets(
                y, w, permutation, cursors[f], size, offset, sw, sg
            )
            offset += size
        # lane/sym-quality: `enqueue_snap_plane` (dynamic_boosting.
        # fit_ordered_rmse), the gradient plane onto the tree's grid
        _snap_plane(sg, total, scale)
        var splits = _ordered_tree_structure(
            cindex, helpers, max_depth, sw, sg, doc_ids, part_bounds,
            fold_count, fold_bits, scale, l2_leaf_reg, feat_offset,
            feat_shift, feat_mask, List[Bool](),
        )
        # `compute_bins_for_model` (level d is bit d), rows across the tasks
        var bins = List[Int](length=n, fill=0)
        _ordered_leaf_bins(
            cindex, splits, feat_offset, feat_shift, feat_mask, List[Bool](),
            n, bins,
        )
        var n_leaves = 1 << len(splits)
        for f in range(n_folds):  # driver-loop: over the folds (each fold's estimation splits its positions inside)
            _ = _ordered_estimate_and_apply(
                bounds[2 * f], bounds[2 * f + 1], n_leaves, y, w, permutation,
                bins, cursors[f], learning_rate, l2_leaf_reg,
            )
        var leaves = _ordered_estimate_and_apply(
            n, n, n_leaves, y, w, permutation, bins, estimation,
            learning_rate, l2_leaf_reg,
        )
        for level in range(len(splits)):
            split_features.append(splits[level].feature)
            split_bins.append(splits[level].bin)
        tree_split_offsets.append(len(split_features))
        for leaf in range(n_leaves):
            model_leaves.append(identical_mul(leaves[leaf], learning_rate))
        tree_leaf_offsets.append(len(model_leaves))
        residual_bound *= Float64(1) + Float64(learning_rate)

    var model = GbdtHostModel(
        fold_counts^, borders^, nan_treatment^, tree_split_offsets^,
        split_features^, split_bins^, tree_leaf_offsets^, model_leaves^,
        List[Float64](), -1, False,
    )
    return gbdt_host_model_text(model)


# ===========================================================================
# GradientBoosting(boosting_type='Ordered') ON THE HOST (lane/catboost-parity,
# 2026-09-19): the second spelling of `gbdt/train.mojo`'s Ordered branch and
# `gbdt/methods/ordered_boosting.mojo::fit_ordered`, for the CPU column.
#
# WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build)
#
#   1. the grid: `gbdt_host_grid` (the device's phase B border build, every
#      `feature_border_type`, the NaN modes, the border subsample), the
#      layout and the binarize, as the plain host fits read them;
#   2. `boost_from_average` on RMSE: `_rmse_starting_approx`, the bias;
#   3. the plan: `ordered_plan.ordered_permutations` and the block size, the
#      SAME host functions the device fit calls, and `_ordered_folds`;
#   4. per tree: the learn-permutation draw and the tree seed from the
#      `ORDERED_STREAM_SALT` stream; per fold the search planes at the fold
#      cursor (`_loss_row`, unit weights: plane 0 the weight or `w * der2`
#      under NewtonCosine, plane 1 `w * der`); the score noise's quality-slice
#      terms folded by `_deterministic_sum_lanes`; the bootstrap draws over the
#      concatenated positions (`bootstrap_kernel`'s grid, one plane of ones)
#      applied to the quality slices; the magnitudes and `choose_scale`; the
#      fold structure search with its noise (`_ordered_tree_structure`);
#   5. per (learn permutation, fold) and for the estimation permutation: the
#      prefix gather, `partition_from_bins`' stable counting sort, the loss's
#      leaf estimator (`gbdt_oracle_losses._estimate_leaves_for_loss`, the
#      walker the plain host fits restate), and `_ordered_apply_kernel`;
#   6. the learn loss at the estimation cursor (`_loss_value`).
#
# COVERED: RMSE, Logloss, CrossEntropy and the pointwise losses at UNIT
# weights, SymmetricTree, Cosine or NewtonCosine, any bootstrap, any
# random_strength, any permutation count, border type and NaN mode, with
# boost_from_average only on RMSE; an eval set with the detector and
# use_best_model (step 8 of `fit_ordered`). Everything else is refused by name in the
# binding (`_mojolearn_gbdt_host.mojo`).
#
# THE NEGATIVE CONTROL: `-D MOJOLEARN_ORDERED_SABOTAGE=1` (the device arm's
# own define) estimates every fold's leaves on the whole fold, quality slice
# included, as the device arm does; `-D MOJOLEARN_HOST_SABOTAGE=1` moves every leaf through the
# walker's regularizer.
# ===========================================================================

from std.sys.compile import is_defined
from std.memory import bitcast
from gbdt.data.ordered_plan import (
    ORDERED_BOOTSTRAP_SALT,
    ORDERED_STREAM_SALT,
    ordered_model_length_mult,
    ordered_permutations,
)
from gbdt.host.gbdt_oracle import (
    GbdtHostParams,
    gbdt_bootstrap_seeds,
    _deterministic_sum_lanes,
)
from gbdt.host.gbdt_oracle_losses import (
    GBDT_OBJ_MAE,
    GBDT_OBJ_MAPE,
    GBDT_OBJ_QUANTILE,
    GbdtHostLoss,
    _estimate_leaves_for_loss,
    _loss_row,
    _loss_value,
)
from gbdt.host.gbdt_oracle_rmse import (
    GbdtRmseHostFit,
    _rmse_starting_approx,
    gbdt_rmse_host_model_text,
)
from gbdt.metrics.sample_quantile import (
    calculate_optimal_const_approx_for_mape,
    calculate_weighted_target_quantile,
)
from checks.numerics import identical_log, identical_pow
from std.math import sqrt
from gbdt.host.gbdt_oracle_ctr import (
    gbdt_ctr_host_model_text_of,
    gbdt_ctr_host_prelude,
)
from gbdt.host.gbdt_oracle_rmse import gbdt_text_with_bias
from gbdt.models.combination_map import TCombinationCtrs
from gbdt.models.ctr_value_table import TCtrValueTable
from gbdt.overfitting_detector.overfitting_detector import (
    make_overfitting_detector,
)

comptime GBDT_ORDERED_SABOTAGE = is_defined["MOJOLEARN_ORDERED_SABOTAGE"]()

#: Lane S3 `MOJOLEARN_TREES_ORD_STD_GRIDFOLD`: the device's
#: `_ord_std_gridfold_kernel` fold order (ordered_boosting.mojo), taken
#: on the device's fused arm only (noise on, no bootstrap).
from gbdt.trees_identical_switches import ORD_STD_GRIDFOLD
from gbdt.host.gbdt_oracle import _halving_fold

#: `ORD_STD_GRID_TPB` / `ORD_STD_GRID_BLOCKS` (ordered_boosting.mojo).
comptime GBDT_ORD_STD_GRID_TPB = 64
comptime GBDT_ORD_STD_GRID_BLOCKS = 256


def _ord_gridfold_sum_lanes(
    partials: List[Float32], lanes: Int, count: Int
) -> List[Float32]:
    """`_ord_std_gridfold_kernel` then `_ord_std_combine_kernel`, per lane
    (NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED): grid lane
    `l = b * 64 + t` folds slots `l, l + 16384, ...` ascending from 0.0;
    block `b` folds its 64 lanes with the halving tree; the 256 block
    partials meet in the 256-wide halving tree. Host CPU-only column; the
    work is the device's statements, in the device's order."""
    comptime L = GBDT_ORD_STD_GRID_BLOCKS * GBDT_ORD_STD_GRID_TPB
    var out = List[Float32](length=lanes, fill=Float32(0.0))
    for lane in range(lanes):
        var top = List[Float32](
            length=GBDT_ORD_STD_GRID_BLOCKS, fill=Float32(0.0)
        )
        for b in range(GBDT_ORD_STD_GRID_BLOCKS):
            var slab = List[Float32](
                length=GBDT_ORD_STD_GRID_TPB, fill=Float32(0.0)
            )
            for t in range(GBDT_ORD_STD_GRID_TPB):
                var acc = Float32(0.0)
                var i = b * GBDT_ORD_STD_GRID_TPB + t
                while i < count:
                    acc += partials[i * lanes + lane]
                    i += L
                slab[t] = acc
            top[b] = _halving_fold(slab)
        out[lane] = _halving_fold(top)
    return out^

comptime GBDT_ORD_BOOT_BLOCK = 256
comptime GBDT_ORD_BOOT_SEEDS = 65536
#: `BOOTSTRAP_KERNEL_*` (`bootstrap.mojo`)
comptime GBDT_ORD_BOOT_BAYESIAN = 0
comptime GBDT_ORD_BOOT_BERNOULLI = 1
comptime GBDT_ORD_BOOT_POISSON = 2


@fieldwise_init
struct GbdtOrderedHostOptions(ImplicitlyCopyable, Movable):
    """What the Ordered branch of `train` resolves beyond `GbdtHostParams`
    and `GbdtHostLoss`."""

    var score_function_newton: Bool
    var random_strength: Float32
    var bootstrap_kind: Int
    var bootstrap_param: Float32
    var permutation_count: Int
    var fold_len_multiplier: Float64
    var permutation_block: Int
    var min_fold_size: Int
    var boost_from_average: Bool


@fieldwise_init
struct GbdtOrderedHostFit(Movable):
    var text: String
    var losses: List[Float64]
    var test_losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool


@fieldwise_init
struct GbdtOrderedHostEval(Movable):
    """The held-out set and what reads it, as `train` resolves them: the
    rows (column-major), the detector (`od_type_from_name` codes), and the
    RESOLVED `use_best_model` (1 on, 0 off) with `best_model_min_trees`."""

    var x_colmajor: List[Float32]
    var y: List[Float32]
    var n_rows: Int
    var od_type: Int
    var od_pvalue: Float64
    var od_wait: Int
    var want_best_model: Int
    var best_model_min_trees: Int


def _ordered_bootstrap_draws(
    kind: Int, mut seeds: List[UInt64], total: Int, param: Float32
) raises -> List[Float32]:
    """`launch_bootstrap` over one plane of ones (`bootstrap_kernel`,
    `bootstrap.mojo:120-246`): grid `min(65536 / 256, ceil(total / 256))`
    blocks of 256, one grid-stride walk per thread, the per-thread seed
    written back. Returns the draws (`1.0 * draw`)."""
    var by_rows = (total + GBDT_ORD_BOOT_BLOCK - 1) // GBDT_ORD_BOOT_BLOCK
    var blocks = GBDT_ORD_BOOT_SEEDS // GBDT_ORD_BOOT_BLOCK
    if by_rows < blocks:
        blocks = by_rows
    if blocks < 1:
        blocks = 1
    var stride = blocks * GBDT_ORD_BOOT_BLOCK
    var draws = List[Float32](length=total, fill=Float32(1.0))
    if (
        total > 0
        and kind != GBDT_ORD_BOOT_BAYESIAN
        and kind != GBDT_ORD_BOOT_BERNOULLI
        and kind != GBDT_ORD_BOOT_POISSON
    ):
        # the walk's refusal, raised before the launch (a task cannot raise)
        raise Error("ordered host: bootstrap kind " + String(kind))
    # THE SPLIT (lane gbdt-host-par): the unit is one device THREAD: it
    # owns seed `gid` and the rows `gid, gid + stride, ...`, its draws
    # ascending in its rows. No thread reads another's seed or rows (and
    # there is no fold here), so the threads run in `_gbdt_units`
    # contiguous ranges.
    var units = _gbdt_units(stride, 8 * total)
    var chunk = host_predict_chunk(stride, units)
    var seeds_p = rebind[GbdtU64Ptr](seeds.unsafe_ptr())
    var draws_p = rebind[HostF32Ptr](draws.unsafe_ptr())

    def thread_draws(unit: Int) {imm chunk, imm stride, imm total, imm kind, imm param, imm seeds_p, imm draws_p}:
        for gid in range(unit * chunk, min(stride, (unit + 1) * chunk)):
            var s = seeds_p[gid]
            var i = gid
            while i < total:
                var bw: Float32
                if kind == GBDT_ORD_BOOT_BAYESIAN:
                    var d = next_uniform_f(s)
                    s = d[1]
                    var tmp = -identical_log(d[0] + Float32(1e-20))
                    bw = tmp
                    if param != Float32(1.0):
                        bw = identical_pow(tmp, param)
                elif kind == GBDT_ORD_BOOT_BERNOULLI:
                    var d = next_uniform_f(s)
                    s = d[1]
                    bw = Float32(1.0) if d[0] < param else Float32(0.0)
                else:
                    var d = next_poisson_f(s, param)
                    s = d[1]
                    bw = d[0]
                draws_p[i] = draws_p[i] * bw
                i += stride
            seeds_p[gid] = s

    host_parallelize(thread_draws, units)
    _ = len(seeds)
    return draws^


@no_inline
def _ordered_task_host(
    loss: GbdtHostLoss,
    estimate_size: Int,
    apply_size: Int,
    n_leaves: Int,
    y: List[Float32],
    permutation: List[Int],
    bins: List[Int],
    mut cursor: List[Float32],
    rate: Float32,
    l2: Float32,
) raises -> List[Float32]:
    """`_ordered_estimate_task` (`ordered_boosting.mojo`): the prefix gather
    in permutation order, `partition_from_bins`' stable counting sort, the
    loss's estimator at the cursor, then `_ordered_apply_kernel` over
    `[0, apply_size)`."""
    if estimate_size < 1 or estimate_size > apply_size:
        raise Error("ordered estimation requires 0 < prefix <= cursor size")
    # THE SPLITS (lane gbdt-host-par): the prefix gather per position
    # range, the stable counting sort per row range, the apply per position
    var gy = List[Float32](length=estimate_size, fill=Float32(0.0))
    # no weights on this arm: the two empty lists are never read
    var no_w = List[Float32]()
    var gw = List[Float32]()
    var gc = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gb = List[Int](length=estimate_size, fill=0)
    _ord_prefix_gather(y, no_w, False, permutation, bins, cursor, estimate_size, gy, gw, gc, gb)
    var sizes = List[Int](length=n_leaves, fill=0)
    var offsets = List[Int](length=n_leaves, fill=0)
    var row_index = List[Int](length=estimate_size, fill=0)
    _ord_counting_sort(gb, estimate_size, n_leaves, sizes, offsets, row_index)
    var leaves = _estimate_leaves_for_loss(
        loss, gy, gc, row_index, offsets, sizes, estimate_size, l2
    )
    _ord_apply_leaves(leaves, bins, permutation, rate, apply_size, cursor)
    return leaves^


def _ord_prefix_gather(
    y: List[Float32],
    w: List[Float32],
    has_w: Bool,
    permutation: List[Int],
    bins: List[Int],
    cursor: List[Float32],
    n: Int,
    mut gy: List[Float32],
    mut gw: List[Float32],
    mut gc: List[Float32],
    mut gb: List[Int],
):
    """The ordered estimation's prefix gather in permutation order: target,
    weight (under `has_w`; otherwise `w` and `gw` are never read) and leaf
    by the row, the cursor by the position. A move per position (one device
    thread each), the positions in `_gbdt_units` contiguous ranges."""
    var units = _gbdt_units(n, 5 * n)
    var chunk = host_predict_chunk(n, units)
    var y_p = rebind[HostF32Ptr](y.unsafe_ptr())
    var w_p = rebind[HostF32Ptr](w.unsafe_ptr())
    var perm_p = rebind[GbdtIntPtr](permutation.unsafe_ptr())
    var bins_p = rebind[GbdtIntPtr](bins.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var gy_p = rebind[HostF32Ptr](gy.unsafe_ptr())
    var gw_p = rebind[HostF32Ptr](gw.unsafe_ptr())
    var gc_p = rebind[HostF32Ptr](gc.unsafe_ptr())
    var gb_p = rebind[GbdtIntPtr](gb.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm has_w, imm y_p, imm w_p, imm perm_p, imm bins_p, imm cursor_p, imm gy_p, imm gw_p, imm gc_p, imm gb_p}:
        for i in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var row = perm_p[i]
            gy_p[i] = y_p[row]
            if has_w:
                gw_p[i] = w_p[row]
            gc_p[i] = cursor_p[i]
            gb_p[i] = bins_p[row]

    host_parallelize(rows, units)
    _ = len(y)
    _ = len(w)
    _ = len(permutation)
    _ = len(bins)
    _ = len(cursor)
    _ = len(gy)
    _ = len(gw)
    _ = len(gc)
    _ = len(gb)


def _ord_counting_sort(
    gb: List[Int],
    n: Int,
    n_leaves: Int,
    mut sizes: List[Int],
    mut offsets: List[Int],
    mut row_index: List[Int],
):
    """`partition_from_bins`: the STABLE counting sort of `n` positions by
    their leaf `gb`, positions ascending within a leaf (integers only).

    The unit is a position range, as `_partition_for_model`: task `c`
    counts its positions per leaf; the leaf offsets and each (leaf, task)
    start are one running sum, leaves ascending and tasks ascending inside
    a leaf (serial by nature: a prefix sum over `leaves x tasks` counts);
    then task `c` places its positions from its own starts. The ranges
    ascend, so the order within a leaf is the serial sort's at any task
    count. A task holds one count per leaf, so the count is also bounded by
    `n / leaves`."""
    var units = _gbdt_units(n, 4 * n)
    units = max(1, min(units, n // max(n_leaves, 1)))
    var chunk = host_predict_chunk(n, units)
    var starts = List[Int](length=units * n_leaves, fill=0)
    var gb_p = rebind[GbdtIntPtr](gb.unsafe_ptr())
    var starts_p = rebind[GbdtIntPtr](starts.unsafe_ptr())
    var row_p = rebind[GbdtIntPtr](row_index.unsafe_ptr())

    def count(unit: Int) {imm chunk, imm n, imm n_leaves, imm gb_p, imm starts_p}:
        var base = unit * n_leaves
        for r in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var k = base + gb_p[r]
            starts_p[k] = starts_p[k] + 1

    host_parallelize(count, units)
    var running = 0
    for leaf in range(n_leaves):  # driver-loop: over (leaf, task), the running sum
        offsets[leaf] = running
        for u in range(units):
            var c = starts[u * n_leaves + leaf]
            starts[u * n_leaves + leaf] = running
            running += c
        sizes[leaf] = running - offsets[leaf]

    def place(unit: Int) {imm chunk, imm n, imm n_leaves, imm gb_p, imm starts_p, imm row_p}:
        var base = unit * n_leaves
        for r in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var at = starts_p[base + gb_p[r]]
            row_p[at] = r
            starts_p[base + gb_p[r]] = at + 1

    host_parallelize(place, units)
    _ = len(gb)
    _ = len(starts)
    _ = len(row_index)


def _ord_rmse_planes(
    t_target: List[Float32],
    t_weight: List[Float32],
    n: Int,
    mut t_cursor: List[Float32],
    mut stats: List[Float32],
):
    """`move_to(zero)` (`cursor + 0.0`, stored) then
    `pointwise_target_kernel[RMSE, estimation=True]`'s two planes,
    `ftz(w * (t - cursor))` and `ftz(w * 1)`: two statements per position
    in the device's order, the positions in `_gbdt_units` ranges."""
    var units = _gbdt_units(n, 6 * n)
    var chunk = host_predict_chunk(n, units)
    var target_p = rebind[HostF32Ptr](t_target.unsafe_ptr())
    var weight_p = rebind[HostF32Ptr](t_weight.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](t_cursor.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm target_p, imm weight_p, imm cursor_p, imm stats_p}:
        for pos in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var moved = cursor_p[pos] + Float32(0.0)
            cursor_p[pos] = moved
            var weight = weight_p[pos]
            stats_p[pos] = ftz(weight * (target_p[pos] - moved))
            stats_p[n + pos] = ftz(weight * Float32(1.0))

    host_parallelize(rows, units)
    _ = len(t_target)
    _ = len(t_weight)
    _ = len(t_cursor)
    _ = len(stats)


def _ord_apply_leaves(
    leaves: List[Float32],
    bins: List[Int],
    permutation: List[Int],
    rate: Float32,
    apply_size: Int,
    mut cursor: List[Float32],
):
    """`_ordered_apply_kernel` over `[0, apply_size)`: position `i` takes
    `identical_mul_add(leaves[bins[permutation[i]]], rate, cursor[i])`, ONE
    fma, as 0.8.19 computed it (the old pin fused `leaf * rate` into this
    add); dynamic_boosting.mojo's kernel says why
    (lane/pinned-mul-contract-free). One device thread a position, the
    positions in `_gbdt_units` contiguous ranges."""
    var units = _gbdt_units(apply_size, 4 * apply_size)
    var chunk = host_predict_chunk(apply_size, units)
    var leaves_p = rebind[HostF32Ptr](leaves.unsafe_ptr())
    var bins_p = rebind[GbdtIntPtr](bins.unsafe_ptr())
    var perm_p = rebind[GbdtIntPtr](permutation.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm apply_size, imm rate, imm leaves_p, imm bins_p, imm perm_p, imm cursor_p}:
        for i in range(unit * chunk, min(apply_size, (unit + 1) * chunk)):
            var leaf = bins_p[perm_p[i]]
            cursor_p[i] = identical_mul_add(leaves_p[leaf], rate, cursor_p[i])

    host_parallelize(rows, units)
    _ = len(leaves)
    _ = len(bins)
    _ = len(permutation)
    _ = len(cursor)


@no_inline
def _add_tree_values(
    mut cursor: List[Float32], bins: List[Int], values: List[Float32]
):
    """`compute_bins_and_add_kernel`'s add (`add_bin_values.mojo`):
    `cursor += values[bin]`, the STORED model values. `@no_inline` IS LOAD
    BEARING: inlined, the host compiler contracted `cursor + leaf * rate`
    into one fma, one rounding where the device kernel rounds the stored
    product first -- measured on the M4, 258 of 800 held-out rows one ulp
    apart after the first tree (lane/catboost-parity).

    One add per row (a device thread a row), the rows in `_gbdt_units`
    contiguous ranges; a task adds a STORED value read through a pointer,
    so there is no product here for a compiler to contract."""
    var n = len(bins)
    var units = _gbdt_units(n, 3 * n)
    var chunk = host_predict_chunk(n, units)
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var bins_p = rebind[GbdtIntPtr](bins.unsafe_ptr())
    var values_p = rebind[HostF32Ptr](values.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm cursor_p, imm bins_p, imm values_p}:
        for r in range(unit * chunk, min(n, (unit + 1) * chunk)):
            cursor_p[r] = cursor_p[r] + values_p[bins_p[r]]

    host_parallelize(rows, units)
    _ = len(cursor)
    _ = len(bins)
    _ = len(values)


comptime _OrdIntPtr = MutPointer[Int, MutUntrackedOrigin]
comptime _OrdU32Ptr = MutPointer[UInt32, MutUntrackedOrigin]


def _ordered_leaf_bins(
    cindex: List[UInt32],
    splits: List[_OrdSplit],
    word_offset_of: List[Int],
    shift_of: List[UInt32],
    mask_of: List[UInt32],
    one_hot_of: List[Bool],
    n_rows: Int,
    mut bins: List[Int],
):
    """`compute_bins_for_model` (`gbdt/models/add_bin_values.mojo`): every
    row's leaf of one oblivious tree read off one compressed index, bit
    `level` set when the row's bin exceeds the level's split bin, or EQUALS
    it when the level's feature is one-hot (`one_hot_of`, one flag per
    layout feature or empty for none: the take-bin split). A tree without
    splits sends every row to leaf 0.

    Rows are the independent units, split across the host's workers
    (`host_parallelize`, the worker count from
    `host_predict_task_count_work`); the levels of one row are a loop over
    model structure. `word_offset_of[f]` is feature `f`'s first word for
    this index's row count; `bins` holds at least `n_rows` entries."""
    var depth = len(splits)
    var slots = depth if depth > 0 else 1
    var offs = List[Int](length=slots, fill=0)
    var masks = List[UInt32](length=slots, fill=UInt32(0))
    var values = List[UInt32](length=slots, fill=UInt32(0))
    #: 1 for a take-bin level, 0 for a threshold level
    var equal = List[UInt32](length=slots, fill=UInt32(0))
    var any_equal = False
    for level in range(depth):
        var fid = splits[level].feature
        offs[level] = word_offset_of[fid]
        masks[level] = mask_of[fid] << shift_of[fid]
        values[level] = UInt32(splits[level].bin) << shift_of[fid]
        if len(one_hot_of) != 0 and one_hot_of[fid]:
            equal[level] = UInt32(1)
            any_equal = True
    var cp = _OrdU32Ptr(unsafe_from_address=Int(cindex.unsafe_ptr()))
    var op = _OrdIntPtr(unsafe_from_address=Int(offs.unsafe_ptr()))
    var mp = _OrdU32Ptr(unsafe_from_address=Int(masks.unsafe_ptr()))
    var vp = _OrdU32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))
    var ep = _OrdU32Ptr(unsafe_from_address=Int(equal.unsafe_ptr()))
    var bp = _OrdIntPtr(unsafe_from_address=Int(bins.unsafe_ptr()))
    var units = host_nested_units(host_predict_task_count_work(n_rows, n_rows * slots))
    var chunk = host_predict_chunk(n_rows, units)

    def rows(unit: Int) {imm chunk, imm n_rows, imm depth, imm cp, imm op, imm mp, imm vp, imm bp}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var leaf = 0
            for level in range(depth):
                if (cp[op[level] + r] & mp[level]) > vp[level]:
                    leaf += 1 << level
            bp[r] = leaf

    def rows_take_bin(unit: Int) {imm chunk, imm n_rows, imm depth, imm cp, imm op, imm mp, imm vp, imm ep, imm bp}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var leaf = 0
            for level in range(depth):
                var word = cp[op[level] + r] & mp[level]
                var right = word > vp[level]
                if ep[level] != UInt32(0):
                    right = word == vp[level]
                if right:
                    leaf += 1 << level
            bp[r] = leaf

    # a tree without a take-bin level keeps the threshold-only row loop
    if any_equal:
        host_parallelize(rows_take_bin, units)
    else:
        host_parallelize(rows, units)
    _ = len(equal)
    # the tasks read these through pointers: named again past the join
    _ = len(offs)
    _ = len(masks)
    _ = len(values)


def _ord_widen_perm(src: List[UInt32], mut dst: List[Int]):
    """`dst[i] = Int(src[i])`: one permutation's positions (a move), in
    `_gbdt_units` contiguous ranges."""
    var n = len(dst)
    var units = _gbdt_units(n, n)
    var chunk = host_predict_chunk(n, units)
    var src_p = _u32_ptr(src)
    var dst_p = rebind[GbdtIntPtr](dst.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm src_p, imm dst_p}:
        for i in range(unit * chunk, min(n, (unit + 1) * chunk)):
            dst_p[i] = Int(src_p[i])

    host_parallelize(rows, units)
    _ = len(src)
    _ = len(dst)


def _ord_fold_planes(
    objective: Int,
    alpha: Float32,
    newton: Bool,
    y: List[Float32],
    permutation: List[Int],
    cursor: List[Float32],
    size: Int,
    offset: Int,
    mut sw: List[Float32],
    mut sg: List[Float32],
    mut doc_ids: List[Int],
):
    """The fold planes of one fold of the learn permutation: position `i`
    of the fold's prefix files its row, the row's derivative at the fold
    cursor (`_loss_row`, unit weight) and the weight plane (the second
    derivative under a Newton score function, else 1). One device thread a
    position, the positions in `_gbdt_units` contiguous ranges; the folds
    are the caller's driver loop."""
    var units = _gbdt_units(size, 16 * size)
    var chunk = host_predict_chunk(size, units)
    var y_p = rebind[HostF32Ptr](y.unsafe_ptr())
    var perm_p = rebind[GbdtIntPtr](permutation.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())
    var doc_p = rebind[GbdtIntPtr](doc_ids.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm size, imm offset, imm objective, imm alpha, imm newton, imm y_p, imm perm_p, imm cursor_p, imm sw_p, imm sg_p, imm doc_p}:
        for i in range(unit * chunk, min(size, (unit + 1) * chunk)):
            var row = perm_p[i]
            doc_p[offset + i] = row
            # `kernel_alpha` carries Logloss's border (the binding builds
            # the loss that way, as `_loss_row` reads it)
            var lr_row = _loss_row(objective, y_p[row], cursor_p[i], alpha)
            var weight = Float32(1.0)
            if newton:
                weight = lr_row.der2
            sw_p[offset + i] = weight
            sg_p[offset + i] = lr_row.der

    host_parallelize(rows, units)
    _ = len(y)
    _ = len(permutation)
    _ = len(cursor)
    _ = len(sw)
    _ = len(sg)
    _ = len(doc_ids)


def _ord_noise_terms_range(
    lo: Int,
    hi: Int,
    n_folds: Int,
    bounds: GbdtIntPtr,
    offsets: GbdtIntPtr,
    sw: HostF32Ptr,
    sg: HostF32Ptr,
    terms: HostF32Ptr,
):
    for f in range(n_folds):
        var first = max(lo, offsets[f] + bounds[2 * f])
        var last = min(hi, offsets[f] + bounds[2 * f + 1])
        for i in range(first, last):
            var w = sw[i]
            if w > Float32(0.0):
                var q = ftz(sg[i] / w)
                terms[i] = ftz(ftz(q * q) * w)


def _ord_noise_terms(
    sw: List[Float32],
    sg: List[Float32],
    bounds: List[Int],
    offsets: List[Int],
    n_folds: Int,
    total: Int,
) -> List[Float32]:
    """The score noise's terms: `ftz(ftz(q * q) * w)` with `q = ftz(g / w)`
    on every QUALITY position with a positive weight (fold `f`'s positions
    `[bounds[2f], bounds[2f + 1])` at its offset), +0.0 elsewhere. One
    statement per position; a task owns a run of positions and visits each
    fold's quality slice clipped to it. The fold of the terms is the
    caller's (`_deterministic_sum_lanes` / the grid fold)."""
    var terms = List[Float32](length=total, fill=Float32(0.0))
    var units = _gbdt_units(total, 6 * total)
    var chunk = host_predict_chunk(total, units)
    var bounds_p = rebind[GbdtIntPtr](bounds.unsafe_ptr())
    var offsets_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())
    var terms_p = rebind[HostF32Ptr](terms.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm total, imm n_folds, imm bounds_p, imm offsets_p, imm sw_p, imm sg_p, imm terms_p}:
        _ord_noise_terms_range(
            unit * chunk, min(total, (unit + 1) * chunk), n_folds, bounds_p,
            offsets_p, sw_p, sg_p, terms_p,
        )

    host_parallelize(rows, units)
    _ = len(sw)
    _ = len(sg)
    _ = len(bounds)
    _ = len(offsets)
    _ = len(terms)
    return terms^


def _ord_boot_abs_range(
    lo: Int,
    hi: Int,
    n_folds: Int,
    bounds: GbdtIntPtr,
    offsets: GbdtIntPtr,
    boot: Bool,
    draws: HostF32Ptr,
    sw: HostF32Ptr,
    sg: HostF32Ptr,
    absv: HostF32Ptr,
):
    for f in range(n_folds):
        var quality_from = offsets[f] + bounds[2 * f]
        var first = max(lo, offsets[f])
        var last = min(hi, offsets[f] + bounds[2 * f + 1])
        for i in range(first, last):
            if boot and i >= quality_from:
                sw[i] = ftz(sw[i] * draws[i])
                sg[i] = ftz(sg[i] * draws[i])
            absv[2 * i] = abs(sw[i])
            absv[2 * i + 1] = abs(sg[i])


def _ord_boot_abs(
    mut sw: List[Float32],
    mut sg: List[Float32],
    draws: List[Float32],
    boot: Bool,
    bounds: List[Int],
    offsets: List[Int],
    n_folds: Int,
    total: Int,
) -> List[Float32]:
    """The bootstrap on the quality slices (`ftz(plane * draw)`, under
    `boot`; `draws` is never read otherwise), then the two magnitudes
    `|sw|`, `|sg|` interleaved for the scale's fold. Per position, the
    multiply before the magnitude as the device orders them; a task owns a
    run of positions and visits each fold clipped to it."""
    var absv = List[Float32](length=2 * total, fill=Float32(0.0))
    var units = _gbdt_units(total, 6 * total)
    var chunk = host_predict_chunk(total, units)
    var bounds_p = rebind[GbdtIntPtr](bounds.unsafe_ptr())
    var offsets_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var draws_p = rebind[HostF32Ptr](draws.unsafe_ptr())
    var sw_p = rebind[HostF32Ptr](sw.unsafe_ptr())
    var sg_p = rebind[HostF32Ptr](sg.unsafe_ptr())
    var absv_p = rebind[HostF32Ptr](absv.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm total, imm n_folds, imm boot, imm bounds_p, imm offsets_p, imm draws_p, imm sw_p, imm sg_p, imm absv_p}:
        _ord_boot_abs_range(
            unit * chunk, min(total, (unit + 1) * chunk), n_folds, bounds_p,
            offsets_p, boot, draws_p, sw_p, sg_p, absv_p,
        )

    host_parallelize(rows, units)
    _ = len(sw)
    _ = len(sg)
    _ = len(draws)
    _ = len(bounds)
    _ = len(offsets)
    _ = len(absv)
    return absv^


def gbdt_ordered_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    loss: GbdtHostLoss,
    opts: GbdtOrderedHostOptions,
    eval: GbdtOrderedHostEval,
    flags: List[UInt32],
    max_ctr_complexity: Int = 1,
) raises -> GbdtOrderedHostFit:
    """Train `params.n_estimators` trees by Ordered boosting on the host and
    return the model text (bias included), the learn and held-out losses,
    the best iteration and the detector's verdict, as `gbdt_fit` returns
    them. `eval.n_rows == 0` is no held-out set.

    `flags` is empty for a float-only fit, else one word per feature (bit 0
    categorical, bit 1 one-hot, as `gbdt_ctr_column_kinds` reads them). A
    categorical column above `one_hot_max_size` builds CTR columns as the
    device's Ordered fit does (`gbdt/train.mojo`, "ORDERED CTRS"): the CTR
    estimation orders ARE the fold permutations, the prelude
    (`gbdt_ctr_host_prelude`, shared with the Plain fit) builds one
    compressed index per permutation, each tree is searched on its learn
    permutation's index, and every permutation's leaves are read off that
    permutation's own index (`fit_ordered`'s `dep_bins`). A one-hot column
    (flagged, or categorical at or below `one_hot_max_size`) is searched and
    applied as equality splits, as on the device.

    `max_ctr_complexity` above 1 turns on feature combinations (tree CTRs)
    when a column feeds CTRs: every tree is then grown by the level driver
    of `_ordered_tree_structure_levels`, as the device's Ordered fit grows
    it, and the model gains one column per combination the trees chose."""
    if n_rows < 4:
        raise Error(
            "Error: pool has just " + String(n_rows) + " groups or docs,"
            " can't use #1 GPUs to learn on such small pool"
        )
    var max_depth = params.max_depth
    if params.n_estimators < 1 or max_depth < 1 or max_depth > 16:
        raise Error("ordered boosting needs n_estimators >= 1 and depth 1..16")
    # ---- 1. the grid, the layout, the binarize, the helpers ----
    # `pidx` holds the fit's compressed indexes: one, or one per
    # permutation when the fit has permutation-dependent CTR columns
    # (`fit_ordered`'s `pidx`). `n_cols` is the layout's column count: the
    # input features, or the categorical prelude's columns.
    var cat_on = len(flags) != 0
    var n_cols = n_features
    var grid = GbdtHostGrid(List[List[Float32]](), List[Int](), List[Int]())
    var pidx = List[List[UInt32]]()
    var test_cindex = List[UInt32]()
    var cat_one_hot = List[Bool]()
    var cat_tables = List[TCtrValueTable]()
    var cat_tree_state = List[GbdtTreeCtrHost]()
    var cat_ctr_columns = 0
    var cat_base_columns = n_features
    if cat_on:
        # the estimation permutation is the last (-1) and the orders take
        # the fold permutation block
        var shape = gbdt_ctr_host_prelude(
            x_colmajor, y, n_rows, n_features, params, flags,
            opts.permutation_count, -1, opts.permutation_block,
            eval.x_colmajor, eval.n_rows, max_ctr_complexity, grid, pidx,
            test_cindex, cat_one_hot, cat_tables, cat_tree_state,
        )
        n_cols = shape.n_columns
        cat_base_columns = shape.n_base_columns
        cat_ctr_columns = shape.ctr_column_count
    else:
        grid = gbdt_host_grid(
            x_colmajor, n_rows, n_features, params.border_count,
            params.border_build_max_samples, params.random_seed,
            params.nan_mode, params.border_type,
        )
    var tree_on = len(cat_tree_state) > 0
    # the prelude's own layout rule (`build_layout(fold_counts, one_hot)`);
    # `cat_one_hot` is empty for a float-only fit
    var layout = build_layout(grid.fold_counts, cat_one_hot)
    if not cat_on:
        pidx.append(
            _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
        )
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
        var gids = List[Int]()
        var offs = List[Int]()
        var firsts = List[Int]()
        var folds = List[Int]()
        var hist_line = 0
        for k in range(blk.count()):
            var f = blk.feature_ids[k]
            gids.append(f)
            offs.append(Int(layout.features[f].offset) * n_rows)
            firsts.append(Int(blk.fold_offset[k]))
            folds.append(Int(blk.folds[k]))
            hist_line += Int(blk.folds[k])
        helpers.append(_PwHelper(
            blk.policy, gids^, offs^, firsts^, folds^, hist_line, List[Float32](),
            List[Int](), List[Bool](),
        ))
    var feat_offset = List[Int](length=n_cols, fill=0)
    var feat_shift = List[UInt32](length=n_cols, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_cols, fill=UInt32(0))
    for f in range(n_cols):
        feat_offset[f] = Int(layout.features[f].offset) * n_rows
        feat_shift[f] = layout.features[f].shift
        feat_mask[f] = layout.features[f].mask

    # ---- 2. the starting point ----
    var start = Float64(0.0)
    if opts.boost_from_average:
        if loss.objective == GBDT_OBJ_MAPE:
            # `CalculateOptimalConstApproxForMAPE` (`sample_quantile.mojo`)
            start = Float64(
                calculate_optimal_const_approx_for_mape(y, List[Float32](), False)
            )
        elif loss.objective == GBDT_OBJ_MAE or loss.objective == GBDT_OBJ_QUANTILE:
            # `CalculateWeightedTargetQuantile` with their 1e-6 delta
            start = Float64(
                calculate_weighted_target_quantile(
                    y, List[Float32](), False,
                    0.5 if loss.objective == GBDT_OBJ_MAE else Float64(
                        loss.estimator_alpha
                    ),
                    1e-6,
                )
            )
        else:
            start = _rmse_starting_approx(y, n_rows)
    var start_value = Float32(start)

    # ---- 3. the plan ----
    var n = n_rows
    var perms_u32 = ordered_permutations(
        n, opts.permutation_count, opts.permutation_block, params.random_seed
    )
    var perm_count = len(perms_u32)
    var perms = List[List[Int]]()
    for p in range(perm_count):  # driver-loop: over the permutations, each one's positions across the tasks
        var one = List[Int](length=n, fill=0)
        _ord_widen_perm(perms_u32[p], one)
        perms.append(one^)
    var est_p = perm_count - 1
    var learn_count = est_p if est_p > 0 else 1
    # `fit_ordered`'s `dep_on`: one index per permutation. With one index
    # (no permutation-dependent column) every permutation reads index 0.
    if len(pidx) > 1 and len(pidx) != perm_count:
        raise Error(
            "ordered boosting: " + String(len(pidx)) + " compressed indexes"
            " for " + String(perm_count) + " permutations"
        )
    var dep_on = perm_count > 1 and len(pidx) == perm_count
    var est_ix = est_p if dep_on else 0
    if tree_on and not dep_on:
        # `fit_ordered`'s own refusal: over one shared index only
        # FeatureFreq combinations (a Borders one needs an index per
        # permutation)
        if len(cat_tree_state[0].dirty_persist) != 1 or (
            perm_count > 1 and cat_tree_state[0].has_borders
        ):
            raise Error(
                "ordered boosting: a tree-CTR fit over one shared index"
                " takes FeatureFreq combinations only"
            )
    #: a learn permutation's leaves under `dep_on` (`dep_bins[p]`), one
    #: buffer for the fit, rewritten per permutation per tree
    var lp_bins = List[Int](length=n if dep_on else 1, fill=0)
    var bounds = _ordered_folds(n, opts.fold_len_multiplier, opts.min_fold_size)
    var n_folds = len(bounds) // 2
    var fold_count = 2 * n_folds
    var fold_bits = _int_log2_floor_ceil(fold_count)
    if fold_bits + max_depth >= 32:
        raise Error("1 << (FoldBits + maxDepth) does not fit a ui32 bin")
    var part_bounds = List[Int]()
    var offsets = List[Int]()
    var total = 0
    part_bounds.append(0)
    # fold f's QUALITY slice is its positions `[bounds[2f], bounds[2f + 1])`
    # at `offsets[f]`: the per-tree passes read the slice off the bounds
    # (the per-position flag plane this replaced held the same test)
    var quality_count = 0
    for f in range(n_folds):  # driver-loop: over the folds, the running offsets and part bounds
        var est_right = bounds[2 * f]
        var qe_right = bounds[2 * f + 1]
        offsets.append(total)
        total += qe_right
        if qe_right > est_right:
            quality_count += qe_right - est_right
        part_bounds.append(part_bounds[len(part_bounds) - 1] + est_right)
        part_bounds.append(part_bounds[len(part_bounds) - 1] + (qe_right - est_right))
    var cursors = List[List[List[Float32]]]()
    for _ in range(learn_count):  # driver-loop: over the learn permutations and folds, one cursor each
        var per = List[List[Float32]]()
        for f in range(n_folds):
            per.append(List[Float32](length=bounds[2 * f + 1], fill=start_value))
        cursors.append(per^)
    var est_cursor = List[Float32](length=n, fill=start_value)
    var est_y = List[Float32](length=n, fill=Float32(0.0))
    _gather_rows(y, perms[est_p], n, est_y)
    var bootstrap_on = opts.bootstrap_kind >= 0
    var seeds = List[UInt64]()
    if bootstrap_on:
        seeds = gbdt_bootstrap_seeds(params.random_seed ^ ORDERED_BOOTSTRAP_SALT)
    # the held-out rows against the model's own borders, the test cursor at
    # the starting point, the detector (`fit_ordered`'s step 8)
    var has_test = eval.n_rows > 0
    var n_eval = eval.n_rows if has_test else 1
    # the categorical prelude already wrote the held-out index (the rows
    # expanded through the learn pool's final CTR tables)
    if has_test and not cat_on:
        test_cindex = _binarize_columns(eval.x_colmajor, n_eval, n_features, grid, layout)
    var test_feat_offset = List[Int](length=n_cols, fill=0)
    for f in range(n_cols):
        test_feat_offset[f] = Int(layout.features[f].offset) * n_eval
    var test_bins = List[Int](length=n_eval, fill=0)
    var test_cursor = List[Float32](length=n_eval, fill=start_value)
    var detector = make_overfitting_detector(
        eval.od_type, False, eval.od_pvalue, eval.od_wait, has_test
    )
    var test_losses = List[Float64]()
    var stopped_early = False
    var rng = TRandom(params.random_seed ^ ORDERED_STREAM_SALT)
    var lr = params.learning_rate
    var l2 = params.l2_leaf_reg

    var tree_split_offsets = List[Int]()
    tree_split_offsets.append(0)
    var split_features = List[Int]()
    var split_bins = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()
    var losses = List[Float64]()
    for iteration in range(params.n_estimators):
        var learn_p = 0
        if learn_count > 1:
            learn_p = Int(rng.next_uniform_l() % UInt64(learn_count - 1))
        var tree_seed = rng.next_uniform_l()
        # ---- the fold planes ----
        var sw = List[Float32](length=total, fill=Float32(0.0))
        var sg = List[Float32](length=total, fill=Float32(0.0))
        var doc_ids = List[Int](length=total, fill=0)
        for f in range(n_folds):  # driver-loop: over the folds, each prefix's positions across the tasks
            _ord_fold_planes(
                loss.objective, loss.kernel_alpha, opts.score_function_newton,
                y, perms[learn_p], cursors[learn_p][f], bounds[2 * f + 1],
                offsets[f], sw, sg, doc_ids,
            )
        # ---- the score noise ----
        var score_std = Float32(0.0)
        if opts.random_strength != Float32(0.0):
            var terms = _ord_noise_terms(sw, sg, bounds, offsets, n_folds, total)
            var s2: Float32
            comptime if ORD_STD_GRIDFOLD:
                if not bootstrap_on:
                    s2 = _ord_gridfold_sum_lanes(terms, 1, total)[0]
                else:
                    s2 = _deterministic_sum_lanes(terms, 1, total)[0]
            else:
                s2 = _deterministic_sum_lanes(terms, 1, total)[0]
            # the product pinned, as the device binding spells it (lane/pinned-mul-contract-free)
            var mult = ordered_model_length_mult(
                n, identical_mul64(Float64(iteration), Float64(lr))
            )
            score_std = Float32(
                mult * sqrt(Float64(s2) / (Float64(quality_count) + 1e-100))
                * Float64(opts.random_strength)
            )
        # ---- the bootstrap, quality slices only ----
        var draws = List[Float32]()
        if bootstrap_on:
            draws = _ordered_bootstrap_draws(
                opts.bootstrap_kind, seeds, total, opts.bootstrap_param
            )
        # ---- the scale (the bootstrap's multiply and the magnitudes are
        # one pass over the positions) ----
        var absv = _ord_boot_abs(
            sw, sg, draws, bootstrap_on, bounds, offsets, n_folds, total
        )
        var mags: List[Float32]
        comptime if ORD_STD_GRIDFOLD:
            # the device's fused arm (noise on, no bootstrap) takes the
            # magnitudes from the same grid fold
            if opts.random_strength != Float32(0.0) and not bootstrap_on:
                mags = _ord_gridfold_sum_lanes(absv, 2, total)
            else:
                mags = _deterministic_sum_lanes(absv, 2, total)
        else:
            mags = _deterministic_sum_lanes(absv, 2, total)
        var m0 = Float64(mags[0])
        var m1 = Float64(mags[1])
        var scale = Float32(choose_scale(m1 if m1 > m0 else m0, total))
        # lane/sym-quality: `enqueue_snap_plane` (ordered_boosting), the
        # gradient plane onto the tree's grid before the search
        _snap_plane(sg, total, scale)
        # ---- the structure ----
        # the structure is searched on the learn permutation's own index
        # when the fit has permutation-dependent columns (`search_idx`)
        # (tree CTRs on: the level driver, which rewrites the indexes'
        # slots in place; off: the same body with no tree-CTR state)
        var search_ix = learn_p if dep_on else 0
        var splits = _ordered_tree_structure_levels(
            _u32_ptr(pidx[search_ix]), helpers, max_depth, sw, sg,
            doc_ids, part_bounds,
            fold_count, fold_bits, scale, l2, feat_offset, feat_shift,
            feat_mask, cat_one_hot, cat_tree_state, search_ix, pidx,
            test_cindex, False, score_std, tree_seed,
        )
        # the estimation permutation's leaves (the model's), rows across
        # the workers
        var bins = List[Int](length=n, fill=0)
        _ordered_leaf_bins(
            pidx[est_ix], splits, feat_offset, feat_shift, feat_mask,
            cat_one_hot, n, bins,
        )
        var n_leaves = 1 << len(splits)
        # ---- the leaves ----
        for lp in range(learn_count):  # driver-loop: over the (learn permutation, fold) tasks; each task's estimation splits its positions inside
            if dep_on:
                # this learn permutation's leaves from its own index (a
                # learn permutation is never the estimation one here)
                _ordered_leaf_bins(
                    pidx[lp], splits, feat_offset, feat_shift, feat_mask,
                    cat_one_hot, n, lp_bins,
                )
            for f in range(n_folds):
                var est = bounds[2 * f]
                comptime if GBDT_ORDERED_SABOTAGE:
                    est = bounds[2 * f + 1]
                if dep_on:
                    _ = _ordered_task_host(
                        loss, est, bounds[2 * f + 1], n_leaves, y, perms[lp],
                        lp_bins, cursors[lp][f], lr, l2,
                    )
                else:
                    _ = _ordered_task_host(
                        loss, est, bounds[2 * f + 1], n_leaves, y, perms[lp],
                        bins, cursors[lp][f], lr, l2,
                    )
        var leaves = _ordered_task_host(
            loss, n, n, n_leaves, y, perms[est_p], bins, est_cursor, lr, l2,
        )
        for level in range(len(splits)):
            split_features.append(splits[level].feature)
            split_bins.append(splits[level].bin)
        tree_split_offsets.append(len(split_features))
        for leaf in range(n_leaves):
            model_leaves.append(identical_mul(leaves[leaf], lr))
        tree_leaf_offsets.append(len(model_leaves))
        # ---- the learn loss at the estimation cursor ----
        var fv = _loss_value(loss, est_y, est_cursor, n)
        losses.append(-Float64(fv) / Float64(n))
        # ---- the held-out cursor (`_apply_last_tree_to_test`: a depth-0
        # tree adds nothing), its loss, the detector ----
        if has_test:
            if len(splits) > 0:
                # one buffer for the fit, every row rewritten per tree
                _ordered_leaf_bins(
                    test_cindex, splits, test_feat_offset, feat_shift,
                    feat_mask, cat_one_hot, n_eval, test_bins,
                )
                var tree_values = List[Float32](capacity=n_leaves)
                for leaf in range(n_leaves):
                    tree_values.append(identical_mul(leaves[leaf], lr))
                _add_tree_values(test_cursor, test_bins, tree_values)
            var t_loss = -Float64(_loss_value(loss, eval.y, test_cursor, n_eval)) / Float64(n_eval)
            test_losses.append(t_loss)
            detector.add_error(t_loss)
            if detector.is_need_stop():
                stopped_early = True
                break

    var best = 0
    if has_test:
        best = detector.best_iteration
    else:
        for i in range(1, len(losses)):
            if losses[i] < losses[best]:
                best = i
    # ---- `use_best_model`: `ShrinkToBestIteration` ----
    if eval.want_best_model == 1 and len(test_losses) > 0:
        var min_trees_best = -1
        var min_trees_err = Float64(0.0)
        for i in range(len(test_losses)):
            if i + 1 < eval.best_model_min_trees:
                continue
            if min_trees_best < 0 or test_losses[i] < min_trees_err:
                min_trees_err = test_losses[i]
                min_trees_best = i
        var best_iter = min_trees_best + 1
        var n_trees = len(tree_split_offsets) - 1
        if 0 < best_iter and best_iter < n_trees:
            while len(tree_split_offsets) - 1 > best_iter:
                _ = tree_split_offsets.pop()
                _ = tree_leaf_offsets.pop()
            split_features.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
            split_bins.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
            model_leaves.resize(tree_leaf_offsets[len(tree_leaf_offsets) - 1], Float32(0.0))
    # TREE CTRS: the model's columns are the base columns followed by one
    # column per combination the trees chose, and the trees' persist-slot
    # splits are renumbered to them (`gbdt_ctr_host_fit`'s own tail)
    var combination_ctrs = TCombinationCtrs()
    if tree_on:
        var chosen = cat_tree_state.pop()
        var section = chosen.assemble()
        var split_offsets = tree_split_offsets.copy()
        chosen.rewrite_splits(split_offsets, split_features, section.column_of)
        var m_folds = List[Int]()
        var m_one_hot = List[Bool]()
        var m_borders = List[List[Float32]]()
        var m_nan = List[Int]()
        for c in range(cat_base_columns):
            m_folds.append(grid.fold_counts[c])
            m_one_hot.append(cat_one_hot[c])
            m_borders.append(grid.borders[c].copy())
            m_nan.append(grid.nan_treatment[c])
        for i in range(len(section.borders)):
            m_folds.append(len(section.borders[i]))
            m_one_hot.append(False)
            m_borders.append(section.borders[i].copy())
            m_nan.append(NAN_TREATMENT_AS_IS)
        grid.fold_counts = m_folds^
        grid.borders = m_borders^
        grid.nan_treatment = m_nan^
        cat_one_hot = m_one_hot^
        combination_ctrs = section.ctrs.copy()
    var model = GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses.copy(), best, stopped_early,
    )
    # the model text, then the `bias` record a seeded fit adds
    # (`gbdt_rmse_host_model_text`'s two steps); a categorical fit's text
    # is the CTR one (the `ctr_columns` header, the `type ctr` features,
    # the tables)
    var base_text = String("")
    if cat_on:
        base_text = gbdt_ctr_host_model_text_of(
            model, cat_one_hot, cat_tables, cat_ctr_columns,
            combination_ctrs,
        )
    else:
        base_text = gbdt_host_model_text(model)
    var text = gbdt_text_with_bias(base_text, len(model.losses), start)
    _ = len(cat_tree_state)
    return GbdtOrderedHostFit(text^, losses^, test_losses^, best, stopped_early)
