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
binding refuses by name `sample_weight` and any layout whose histogram policy
is not OneByteFeatures (a feature with 15 or fewer borders), the only policy
the covered data reaches.

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
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_sqrt
from gbdt.data.quantization import NAN_TREATMENT_AS_IS
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import POLICY_HALF_BYTE, POLICY_ONE_BYTE
from gbdt.grid_creator.binarization import best_split
from gbdt.data.permutation import TRandom
from gbdt.gpu_util.kernel.random_gen import (
    advance_seed_k,
    next_normal_f,
    next_poisson_f,
    next_uniform_f,
)
from gbdt.host.gbdt_oracle import (
    GBDT_ORACLE_HOST_SABOTAGE,
    GbdtHostGrid,
    GbdtHostModel,
    _binarize_columns,
    _halving_fold_live,
    _hist2_dither,
    _pinned_partition_stat,
    _hist2_quantize,
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


def _refuse_ordered(what: String) raises:
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_fit_ordered_rmse for "
        + what + "; the gbdt host binding trains the gbdt-ordered-rmse lane"
        " only (unit weights, OneByteFeatures), see"
        " gbdt/host/gbdt_oracle_ordered.mojo"
    )


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


def _partition_update_sum(values: List[Float32], offset: Int, size: Int) -> Float32:
    """`_compute_sum[1024]` per thread then `_block_reduce_sum[1024]`
    (`pointwise_scores.mojo:578-628`, `:1408-1443`)."""
    if size <= 0:
        return Float32(0.0)
    var live = min(size, GBDT_ORD_WIDE_BLOCK)
    var p = 1
    while p < live:
        p <<= 1
    var slab = List[Float32](length=p, fill=Float32(0.0))
    for tid in range(p):
        var s = Float32(0.0)
        var i = tid
        while i < size:
            s += values[offset + i]
            i += GBDT_ORD_WIDE_BLOCK
        slab[tid] = s
    # lanes past `size` hold +0.0 (`_halving_fold_live`)
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


@no_inline
def _update_subsets_stats(
    mut s: _OrdSubsets, part_count: Int, sw: List[Float32], sg: List[Float32]
):
    """`update_subsets_stats`: the offsets and sizes of the sorted bins
    (`update_partition_offsets_kernel`, `update_partition_sizes_kernel`, a
    lower bound per part), the two gathers, the partition update."""
    var doc_count = len(s.bins)
    var at = 0
    for b in range(part_count):
        while at < doc_count and Int(s.bins[at]) < b:
            at += 1
        s.p_off[b] = at
        var end = at
        while end < doc_count and Int(s.bins[end]) == b:
            end += 1
        s.p_sz[b] = end - at
    for i in range(doc_count):
        s.g_weight[i] = sw[s.indices[i]]
        s.g_target[i] = sg[s.indices[i]]
    for b in range(part_count):
        s.part_stats[3 * b] = _partition_update_sum(s.g_weight, s.p_off[b], s.p_sz[b])
        s.part_stats[3 * b + 1] = _partition_update_sum(s.g_target, s.p_off[b], s.p_sz[b])
        s.part_stats[3 * b + 2] = Float32(s.p_sz[b])


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


def _pw_thread_points(
    off: Int, sz: Int, full_pass: Bool
) -> List[List[Int]]:
    """Every thread's point sequence in a half-byte launch, `-1` for a zero
    point: `compute_histogram_2` on a full pass, `compute_histogram` (one
    point per iteration) on a partial one (`compute_point_hist2_loop.mojo`),
    at `PW_HB_BLOCK` 512, multiplier 1, the uniform iteration."""
    var out = List[List[Int]]()
    for t in range(GBDT_ORD_HB_BLOCK):
        var pts = List[Int]()
        if full_pass:
            var last_id = min(128 - (off & 127), sz)
            pts.append(off + t if (t < 128 and t < last_id) else -1)
            var ds = sz - last_id if sz > last_id else 0
            var base = off + last_id
            var tail = ds & 63
            if tail != 0:
                pts.append(base + ds - tail + t if (t < 64 and t < tail) else -1)
            ds -= tail
            if ds > 0:
                comptime stripe = GBDT_ORD_HB_BLOCK * 2
                var i = 2 * t
                var max_iters = (ds + stripe - 1) // stripe
                var own = 0
                if ds > i:
                    own = (ds - i + stripe - 1) // stripe
                var cur = base + i
                for j in range(max_iters):
                    if j < own:
                        pts.append(cur)
                        pts.append(cur + 1)
                    else:
                        pts.append(-1)
                        pts.append(-1)
                    cur += stripe
        else:
            var last_id = min(32 - (off & 31), sz)
            pts.append(off + t if t < last_id else -1)
            var ds = sz - last_id if sz > last_id else 0
            var base = off + last_id
            var tail = ds & 31
            if tail != 0:
                pts.append(base + ds - tail + t if t < tail else -1)
            ds -= tail
            if ds > 0:
                comptime stripe = GBDT_ORD_HB_BLOCK
                var max_iters = (ds + stripe - 1) // stripe
                var own = 0
                if ds > t:
                    own = (ds - t + stripe - 1) // stripe
                var cur = base + t
                for j in range(max_iters):
                    pts.append(cur if j < own else -1)
                    cur += stripe
        out.append(pts^)
    return out^


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
    plain_l2: Bool = False,
    # the per-tree noise (`fit_oblivious_tree_structure`'s `score_std_dev`
    # and `seed`): a `TRandom(seed)` draw per LEVEL is the level's
    # `global_seed`, and each candidate's noise is the normal of
    # `advance_seed_k(global_seed + feature, 4)`, pinned-fma'd onto the
    # score (`pointwise_scores.mojo`'s dynamic cosine kernel)
    score_std_dev: Float32 = Float32(0.0),
    seed: UInt64 = UInt64(0),
) raises -> List[_OrdSplit]:
    """The doc-parallel oblivious searcher's level loop. `fold_count > 1`
    with `plain_l2` False is the ordered fold arm (the dynamic cosine
    scorer); `fold_count == 1` with `plain_l2` True is the single-task arm
    the pointwise lane runs (`find_optimal_split_single_fold_kernel` with
    `TL2ScoreCalcer` at meta exponent 1, `pointwise_scores.mojo:809-950`)."""
    var doc_count = len(doc_ids)
    var max_parts = 1 << (fold_bits + max_depth)
    var stripe = 1 << Int(ceil(log2(Float32(fold_count))))
    var s = _OrdSubsets(
        List[UInt32](length=doc_count, fill=UInt32(0)),
        List[Int](length=doc_count, fill=0),
        List[Int](length=max_parts, fill=0),
        List[Int](length=max_parts, fill=0),
        List[Float32](length=3 * max_parts, fill=Float32(0.0)),
        List[Float32](length=doc_count, fill=Float32(0.0)),
        List[Float32](length=doc_count, fill=Float32(0.0)),
    )
    for i in range(doc_count):
        s.indices[i] = i
    # `write_fold_based_initial_bins`: part p over its concatenated range
    for p in range(fold_count):
        for i in range(part_bounds[p], part_bounds[p + 1]):
            s.bins[i] = UInt32(p)
    _update_subsets_stats(s, 1 << fold_bits, sw, sg)

    for h in range(len(helpers)):
        helpers[h].hist = List[Float32](
            length=(1 << max_depth) * fold_count * helpers[h].hist_line * 2,
            fill=Float32(0.0),
        )
    var structure = List[_OrdSplit]()
    var score_before = Float32(0.0)
    var docs = List[Int](length=doc_count, fill=0)
    var level_rand = TRandom(seed)
    for depth in range(max_depth):
        var global_seed = level_rand.next_uniform_l()
        for i in range(doc_count):
            docs[i] = doc_ids[s.indices[i]]
        var part_count = 1 << depth
        var full_pass = depth == 0
        var ny = part_count if full_pass else part_count // 2
        var best_fid = GBDT_ORD_SENTINEL
        var best_bin = UInt32(0)
        var best_score = GBDT_ORD_FLOAT32_MAX
        var best_gain = GBDT_ORD_FLOAT32_MAX
        # `compute_optimal_split_dev`'s per-helper seed advance
        # (`pointwise_scores_calcer.mojo`): helper i scores with the i-th
        # draw of `TRandom(level seed)`
        var helper_rand = TRandom(global_seed)
        for h in range(len(helpers)):
            ref hp = helpers[h]
            var helper_seed = helper_rand.next_uniform_l()
            var hist_line = hp.hist_line
            var n_feat = len(hp.gids)
            var group = 8 if hp.policy == POLICY_HALF_BYTE else 4
            var acc = List[Int32](length=hist_line * 2, fill=Int32(0))
            var reached = List[Bool](length=hist_line, fill=False)
            var reached_list = List[Int](capacity=hist_line)
            # ---- the histograms (`compute_hist2`) ----
            for y in range(ny):
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
                    var off = s.p_off[data_part]
                    var sz = s.p_sz[data_part]
                    if sz == 0:
                        continue
                    var base = hist_slot * hist_line * 2
                    if hp.policy == POLICY_ONE_BYTE:
                        # the 8-bit fixed-point kernel: Int32 sums, the
                        # dither keyed on the document id. A cell no row
                        # reaches sums to 0 and is not written (`|0| >
                        # 1e-20` is false), so only the reached cells are
                        # converted, then cleared for the next slot.
                        for pos in range(off, off + sz):
                            var row = docs[pos]
                            var u = _hist2_dither(row)
                            var qw = _hist2_quantize(s.g_weight[pos], fixed_scale, u)
                            var qt = _hist2_quantize(s.g_target[pos], fixed_scale, u)
                            var f_base = 0
                            while f_base < n_feat:
                                var ci = cindex[hp.offsets[f_base] + row]
                                var f_count = min(group, n_feat - f_base)
                                for j in range(f_count):
                                    var bin = Int((ci >> UInt32(24 - 8 * j)) & UInt32(255))
                                    if bin < hp.folds[f_base + j]:
                                        var at = (hp.first[f_base + j] + bin) * 2
                                        if not reached[at // 2]:
                                            reached[at // 2] = True
                                            reached_list.append(at)
                                        acc[at] = acc[at] + qw
                                        acc[at + 1] = acc[at + 1] + qt
                                f_base += group
                        for k in range(len(reached_list)):
                            var at = reached_list[k]
                            for c in range(at, at + 2):
                                var val = Float32(Int(acc[c])) / fixed_scale
                                if abs(val) > Float32(1e-20):
                                    hp.hist[base + c] = val
                                acc[c] = Int32(0)
                            reached[at // 2] = False
                        reached_list.clear()
                    else:
                        var points = _pw_thread_points(off, sz, full_pass)
                        var f_base = 0
                        while f_base < n_feat:
                            var f_count = min(group, n_feat - f_base)
                            var cells = _pw_half_byte_group(
                                points, docs, s.g_weight, s.g_target, cindex,
                                hp.offsets[f_base], f_count,
                            )
                            for j in range(f_count):
                                for fold in range(hp.folds[f_base + j]):
                                    for w in range(2):
                                        var result = cells[fold * 16 + 2 * j + w]
                                        if abs(result) > Float32(1e-20):
                                            hp.hist[base + (hp.first[f_base + j] + fold) * 2 + w] = result
                            f_base += group
            # ---- the scan over the computed slots ----
            for y in range(ny):
                for z in range(fold_count):
                    var slot = (y * fold_count + z) if full_pass else ((y | ny) * fold_count + z)
                    # a slot is first written at the level that computes
                    # it (leaves [ny, 2ny) here, leaf 0 at depth 0), so one
                    # whose partition is empty is still all +0.0 and its
                    # scan writes +0.0 over +0.0
                    var data_part: Int
                    if full_pass:
                        data_part = y * stripe + z
                    else:
                        var left = y * stripe + z
                        var right = (y | ny) * stripe + z
                        data_part = left if s.p_sz[left] < s.p_sz[right] else right
                    if s.p_sz[data_part] == 0:
                        continue
                    for f in range(n_feat):
                        if hp.folds[f] <= 1:
                            continue
                        var sbase = (slot * hist_line + hp.first[f]) * 2
                        for st in range(2):
                            var running = Float32(0.0)
                            for b in range(hp.folds[f]):
                                var at = sbase + b * 2 + st
                                running += hp.hist[at]
                                hp.hist[at] = running
            # ---- the subtraction on a partial pass ----
            if not full_pass:
                for y in range(ny):
                    for z in range(fold_count):
                        var left_part = y * stripe + z
                        var right_part = (y | ny) * stripe + z
                        # an empty parent's slot and its computed child are
                        # all +0.0 (by induction from the fresh slots), and
                        # +0.0 - +0.0 is +0.0: nothing to write
                        if s.p_sz[left_part] == 0 and s.p_sz[right_part] == 0:
                            continue
                        var is_left = s.p_sz[left_part] < s.p_sz[right_part]
                        var lslot = (y * fold_count + z) * hist_line * 2
                        var rslot = ((y | ny) * fold_count + z) * hist_line * 2
                        for c in range(hist_line * 2):
                            var calc = hp.hist[rslot + c]
                            var comp = hp.hist[lslot + c] - calc
                            hp.hist[lslot + c] = calc if is_left else comp
                            hp.hist[rslot + c] = comp if is_left else calc

            # ---- the dynamic cosine score and the block argmins ----
            var blocks = (hist_line + GBDT_ORD_SCORE_BLOCK - 1) // GBDT_ORD_SCORE_BLOCK
            if blocks > 32:
                blocks = 32
            if blocks < 1:
                blocks = 1
            var bf_feature = List[Int](length=hist_line, fill=0)
            var bf_bin = List[Int](length=hist_line, fill=0)
            for f in range(n_feat):
                for b in range(hp.folds[f]):
                    bf_feature[hp.first[f] + b] = hp.gids[f]
                    bf_bin[hp.first[f] + b] = b
            var loc_fid = GBDT_ORD_SENTINEL
            var loc_bin = UInt32(0)
            var loc_score = GBDT_ORD_FLOAT32_MAX
            var loc_gain = GBDT_ORD_FLOAT32_MAX
            var t_score = List[Float32](length=GBDT_ORD_SCORE_BLOCK, fill=Float32(0.0))
            var t_gain = List[Float32](length=GBDT_ORD_SCORE_BLOCK, fill=Float32(0.0))
            var t_index = List[Int](length=GBDT_ORD_SCORE_BLOCK, fill=0)
            # the dynamic cosine score of every candidate, (leaf, fold pair)
            # outer and the candidate inner: each candidate's own adds run
            # in the kernel's (leaf, fold) order, the same expressions
            var c_noisy = List[Float32]()
            var c_gain = List[Float32]()
            if not plain_l2:
                c_noisy = _dynamic_cosine_candidates(
                    hp, s.part_stats, part_count, fold_count, stripe, l2,
                    score_std_dev, helper_seed, bf_feature, score_before,
                    c_gain,
                )
            for blk in range(blocks):
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
                            # `TL2ScoreCalcer`: Score = 0, per leaf both
                            # sides' `(-sum * sum) / (weight + lambda)` when
                            # the weight exceeds 1e-20, no normalization
                            var l2score = Float32(0.0)
                            for leaf in range(part_count):
                                var pw = s.part_stats[3 * leaf]
                                var psum = s.part_stats[3 * leaf + 1]
                                var hb = 2 * b + hist_line * leaf * 2
                                var wl = hp.hist[hb]
                                var sl = hp.hist[hb + 1]
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
                var c_fid = UInt32(0)
                var c_bin = UInt32(0)
                if t_index[0] < hist_line:
                    c_fid = UInt32(bf_feature[t_index[0]])
                    c_bin = UInt32(bf_bin[t_index[0]])
                if _record_less(t_gain[0], c_fid, c_bin, loc_gain, loc_fid, loc_bin):
                    loc_fid = c_fid
                    loc_bin = c_bin
                    loc_score = t_score[0]
                    loc_gain = t_gain[0]
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
        for i in range(len(structure)):
            if structure[i].feature == fid and structure[i].bin == Int(best_bin):
                seen = True
        if seen:
            break
        structure.append(_OrdSplit(fid, Int(best_bin)))
        if depth + 1 == max_depth:
            break
        var bit = UInt32(1) << UInt32(depth + fold_bits)
        var value = best_bin << shift_of[fid]
        var mask = mask_of[fid] << shift_of[fid]
        for i in range(doc_count):
            var feature_val = cindex[word_offset_of[fid] + docs[i]] & mask
            if feature_val > value:
                s.bins[i] = s.bins[i] | bit
        var new_bins = List[UInt32](capacity=doc_count)
        var new_indices = List[Int](capacity=doc_count)
        for i in range(doc_count):
            if (s.bins[i] & bit) == UInt32(0):
                new_bins.append(s.bins[i])
                new_indices.append(s.indices[i])
        for i in range(doc_count):
            if (s.bins[i] & bit) != UInt32(0):
                new_bins.append(s.bins[i])
                new_indices.append(s.indices[i])
        s.bins = new_bins^
        s.indices = new_indices^
        _update_subsets_stats(s, 1 << (depth + 1 + fold_bits), sw, sg)
    return structure^


@no_inline
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
    read once, front to back. The noise draw depends on the candidate's
    feature only, so it is drawn once per feature."""
    var hist_line = hp.hist_line
    var score = List[Float32](length=hist_line, fill=Float32(0.0))
    var denum = List[Float32](length=hist_line, fill=Float32(1e-20))
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
            var hptr = hp.hist.unsafe_ptr()
            var sptr = score.unsafe_ptr()
            var dptr = denum.unsafe_ptr()
            comptime W = 8
            var vb = 0
            while vb + W <= hist_line:
                # W candidates per step, lane k is candidate vb + k: the
                # scalar loop below, lane by lane (the same contraction of
                # `x += a * b` into one fma, measured on the host compiler)
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
                sc += stl * mu_l
                dn += wtl * mu_l * mu_l
                var mu_r = wer.gt(zero).select(ser / (wer + l2), zero)
                sc += sum_tr * mu_r
                dn += wtr * mu_r * mu_r
                sptr.unsafe_store(vb, sc)
                dptr.unsafe_store(vb, dn)
                vb += W
            for b in range(vb, hist_line):
                var current = 2 * b
                var sc = score[b]
                var dn = denum[b]
                var wel = hp.hist[current + h_learn]
                var wer = max(plw - wel, Float32(0.0))
                var sel = hp.hist[current + h_learn + 1]
                var ser = pls - sel
                var wtl = hp.hist[current + h_test]
                var wtr = max(ptw - wtl, Float32(0.0))
                var stl = hp.hist[current + h_test + 1]
                var sum_tr = pts - stl
                var mu_l = Float32(0.0)
                if wel > Float32(0.0):
                    mu_l = sel / (wel + l2)
                sc += stl * mu_l
                dn += wtl * mu_l * mu_l
                var mu_r = Float32(0.0)
                if wer > Float32(0.0):
                    mu_r = ser / (wer + l2)
                sc += sum_tr * mu_r
                dn += wtr * mu_r * mu_r
                score[b] = sc
                denum[b] = dn
            fold += 2
    var noisy_out = List[Float32](length=hist_line, fill=Float32(0.0))
    gain_out = List[Float32](length=hist_line, fill=Float32(0.0))
    var last_feature = -1
    var last_draw = Float32(0.0)
    for b in range(hist_line):
        var sc = score[b]
        var denum_sqr = denum[b]
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


def _partition_stat_n(
    stats: List[Float32], line_size: Int, stat_id: Int, offset: Int, size: Int,
    n_stats: Int,
) -> Float32:
    """`compute_partition_stats` for one (leaf, stat) at
    `partition_stats_chunks(32, n_stats)` chunks (`partitions_reduce.mojo`)."""
    var max_chunks = (2 * GBDT_ORD_PINNED_SM + n_stats - 1) // n_stats
    return _pinned_partition_stat(
        stats, stat_id * line_size + offset, size, max_chunks
    )


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
    var gy = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gw = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gc = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gb = List[Int](length=estimate_size, fill=0)
    for i in range(estimate_size):
        var row = permutation[i]
        gy[i] = y[row]
        gw[i] = w[row]
        gc[i] = cursor[i]
        gb[i] = bins[row]
    # `partition_from_bins`: the stable counting sort
    var sizes = List[Int](length=n_leaves, fill=0)
    for r in range(estimate_size):
        sizes[gb[r]] += 1
    var offsets = List[Int](length=n_leaves, fill=0)
    var running = 0
    for i in range(n_leaves):
        offsets[i] = running
        running += sizes[i]
    var fill = offsets.copy()
    var row_index = List[Int](length=estimate_size, fill=0)
    for r in range(estimate_size):
        row_index[fill[gb[r]]] = r
        fill[gb[r]] += 1
    # `_estimate_and_apply`: the gathers into leaf order
    var n = estimate_size
    var t_target = List[Float32](length=n, fill=Float32(0.0))
    var t_weight = List[Float32](length=n, fill=Float32(0.0))
    var t_cursor = List[Float32](length=n, fill=Float32(0.0))
    for pos in range(n):
        t_target[pos] = gy[row_index[pos]]
        t_weight[pos] = gw[row_index[pos]]
        t_cursor[pos] = gc[row_index[pos]]
    # `make_bin_optimized_oracle`: WeightsCpu from the one-stat reduce
    var weights_cpu = List[Float64]()
    for leaf in range(n_leaves):
        weights_cpu.append(Float64(_partition_stat_n(t_weight, n, 0, offsets[leaf], sizes[leaf], 1)))
    var lambda_reg = Float64(l2)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0
    # `move_to(zero)`: `add_bin_model_value_kernel` adds 0.0 to every row
    for pos in range(n):
        t_cursor[pos] = t_cursor[pos] + Float32(0.0)
    # `pointwise_target_kernel[RMSE, estimation=True]`
    var stats = List[Float32](length=2 * n, fill=Float32(0.0))
    for pos in range(n):
        var weight = t_weight[pos]
        stats[pos] = ftz(weight * (t_target[pos] - t_cursor[pos]))
        stats[n + pos] = ftz(weight * Float32(1.0))
    comptime EPS_1E20F = Float64(Float32(1e-20))
    var leaves = List[Float32]()
    for leaf in range(n_leaves):
        var g = Float64(_partition_stat_n(stats, n, 0, offsets[leaf], sizes[leaf], 2))
        var h = Float64(_partition_stat_n(stats, n, 1, offsets[leaf], sizes[leaf], 2)) + lambda_reg
        var direction = Float32(0.0)
        if h > 0:
            direction = Float32(g / (h + EPS_1E20F))
        var moved = Float32(Float64(Float32(0.0)) + 1.0 * Float64(direction))
        if weights_cpu[leaf] < 1e-20:
            moved = Float32(0.0)
        leaves.append(moved)
    # `_ordered_apply_kernel` over every apply position
    for i in range(apply_size):
        var leaf = bins[permutation[i]]
        var scaled = identical_mul(leaves[leaf], rate)
        cursor[i] = identical_mul_add(scaled, Float32(1), cursor[i])
    return leaves^


# ===========================================================================
# THE FIT
# ===========================================================================


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
) raises -> String:
    """`train_ordered_rmse` then `fit_ordered_rmse` at unit weights; returns
    the model text."""
    if n_rows != len(y) or n_features < 1 or len(x_colmajor) != n_rows * n_features:
        raise Error("train_ordered_rmse input shape mismatch")
    if border_count < 1 or border_count > 255:
        raise Error("train_ordered_rmse supports 1..255 borders")
    # ---- 1. the grid and the compressed index ----
    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    var nan_treatment = List[Int]()
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
        if blk.policy != POLICY_ONE_BYTE and blk.policy != POLICY_HALF_BYTE:
            _refuse_ordered(
                "a feature with exactly one border (the BinaryFeatures"
                " histogram policy, feature "
                + String(blk.feature_ids[0]) + ")"
            )
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
    var w = List[Float32](length=n, fill=Float32(1.0))
    var weight_sum = Float64(0)
    var max_weight = Float64(0)
    var residual_bound = Float64(0)
    for i in range(n):
        var row = Int(permutation_u32[i])
        if row >= n or seen[row]:
            raise Error("ordered RMSE permutation must be a bijection")
        seen[row] = True
        permutation[i] = row
        var wi = Float32(1)
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
    var doc_ids = List[Int]()
    var total = 0
    part_bounds.append(0)
    for f in range(n_folds):
        var est_right = bounds[2 * f]
        var qe_right = bounds[2 * f + 1]
        total += qe_right
        part_bounds.append(part_bounds[len(part_bounds) - 1] + est_right)
        part_bounds.append(part_bounds[len(part_bounds) - 1] + (qe_right - est_right))
        for p in range(qe_right):
            doc_ids.append(permutation[p])
    var cursors = List[List[Float32]]()
    for f in range(n_folds):
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
        for f in range(n_folds):
            var size = bounds[2 * f + 1]
            for i in range(size):
                var row = permutation[i]
                var wi = w[row]
                sw[offset + i] = wi
                sg[offset + i] = ftz(identical_mul(wi, y[row] - cursors[f][i]))
            offset += size
        var splits = _ordered_tree_structure(
            cindex, helpers, max_depth, sw, sg, doc_ids, part_bounds,
            fold_count, fold_bits, scale, l2_leaf_reg, feat_offset,
            feat_shift, feat_mask,
        )
        # `compute_bins_for_model` (level d is bit d)
        var bins = List[Int](length=n, fill=0)
        for r in range(n):
            var leaf = 0
            for level in range(len(splits)):
                var fid = splits[level].feature
                var mask = feat_mask[fid] << feat_shift[fid]
                var value = UInt32(splits[level].bin) << feat_shift[fid]
                if (cindex[feat_offset[fid] + r] & mask) > value:
                    leaf += 1 << level
            bins[r] = leaf
        var n_leaves = 1 << len(splits)
        for f in range(n_folds):
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
    gbdt_host_grid,
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
from gbdt.overfitting_detector.overfitting_detector import (
    make_overfitting_detector,
)

comptime GBDT_ORDERED_SABOTAGE = is_defined["MOJOLEARN_ORDERED_SABOTAGE"]()
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
    for gid in range(stride):
        var s = seeds[gid]
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
            elif kind == GBDT_ORD_BOOT_POISSON:
                var d = next_poisson_f(s, param)
                s = d[1]
                bw = d[0]
            else:
                raise Error("ordered host: bootstrap kind " + String(kind))
            draws[i] = draws[i] * bw
            i += stride
        seeds[gid] = s
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
    var gy = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gc = List[Float32](length=estimate_size, fill=Float32(0.0))
    var gb = List[Int](length=estimate_size, fill=0)
    for i in range(estimate_size):
        var row = permutation[i]
        gy[i] = y[row]
        gc[i] = cursor[i]
        gb[i] = bins[row]
    var sizes = List[Int](length=n_leaves, fill=0)
    for r in range(estimate_size):
        sizes[gb[r]] += 1
    var offsets = List[Int](length=n_leaves, fill=0)
    var running = 0
    for i in range(n_leaves):
        offsets[i] = running
        running += sizes[i]
    var fill = offsets.copy()
    var row_index = List[Int](length=estimate_size, fill=0)
    for r in range(estimate_size):
        row_index[fill[gb[r]]] = r
        fill[gb[r]] += 1
    var leaves = _estimate_leaves_for_loss(
        loss, gy, gc, row_index, offsets, sizes, estimate_size, l2
    )
    for i in range(apply_size):
        var leaf = bins[permutation[i]]
        var scaled = identical_mul(leaves[leaf], rate)
        cursor[i] = identical_mul_add(scaled, Float32(1), cursor[i])
    return leaves^


@no_inline
def _add_tree_values(
    mut cursor: List[Float32], bins: List[Int], values: List[Float32]
):
    """`compute_bins_and_add_kernel`'s add (`add_bin_values.mojo`):
    `cursor += values[bin]`, the STORED model values. `@no_inline` IS LOAD
    BEARING: inlined, the host compiler contracted `cursor + leaf * rate`
    into one fma, one rounding where the device kernel rounds the stored
    product first -- measured on the M4, 258 of 800 held-out rows one ulp
    apart after the first tree (lane/catboost-parity)."""
    for r in range(len(bins)):
        cursor[r] = cursor[r] + values[bins[r]]


def gbdt_ordered_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    loss: GbdtHostLoss,
    opts: GbdtOrderedHostOptions,
    eval: GbdtOrderedHostEval,
) raises -> GbdtOrderedHostFit:
    """Train `params.n_estimators` trees by Ordered boosting on the host and
    return the model text (bias included), the learn and held-out losses,
    the best iteration and the detector's verdict, as `gbdt_fit` returns
    them. `eval.n_rows == 0` is no held-out set."""
    if n_rows < 4:
        raise Error(
            "Error: pool has just " + String(n_rows) + " groups or docs,"
            " can't use #1 GPUs to learn on such small pool"
        )
    var max_depth = params.max_depth
    if params.n_estimators < 1 or max_depth < 1 or max_depth > 16:
        raise Error("ordered boosting needs n_estimators >= 1 and depth 1..16")
    # ---- 1. the grid, the layout, the binarize, the helpers ----
    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, params.border_count,
        params.border_build_max_samples, params.random_seed, params.nan_mode,
        params.border_type,
    )
    var layout = build_layout(grid.fold_counts)
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var blocks = blocks_for(layout, n_rows)
    var helpers = List[_PwHelper]()
    for b in range(len(blocks)):
        ref blk = blocks[b]
        if blk.policy != POLICY_ONE_BYTE and blk.policy != POLICY_HALF_BYTE:
            raise Error(
                "no CPU implementation of _mojolearn_gbdt.gbdt_fit for Ordered"
                " boosting on a feature with exactly one border (the"
                " BinaryFeatures histogram policy, feature "
                + String(blk.feature_ids[0]) + ")"
            )
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
        ))
    var feat_offset = List[Int](length=n_features, fill=0)
    var feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    for f in range(n_features):
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
    for p in range(perm_count):
        var one = List[Int](capacity=n)
        for i in range(n):
            one.append(Int(perms_u32[p][i]))
        perms.append(one^)
    var est_p = perm_count - 1
    var learn_count = est_p if est_p > 0 else 1
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
    for f in range(n_folds):
        var est_right = bounds[2 * f]
        var qe_right = bounds[2 * f + 1]
        offsets.append(total)
        total += qe_right
        part_bounds.append(part_bounds[len(part_bounds) - 1] + est_right)
        part_bounds.append(part_bounds[len(part_bounds) - 1] + (qe_right - est_right))
    var quality = List[Bool](length=total, fill=False)
    var quality_count = 0
    for f in range(n_folds):
        for i in range(bounds[2 * f], bounds[2 * f + 1]):
            quality[offsets[f] + i] = True
            quality_count += 1
    var cursors = List[List[List[Float32]]]()
    for _ in range(learn_count):
        var per = List[List[Float32]]()
        for f in range(n_folds):
            per.append(List[Float32](length=bounds[2 * f + 1], fill=start_value))
        cursors.append(per^)
    var est_cursor = List[Float32](length=n, fill=start_value)
    var est_y = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        est_y[i] = y[perms[est_p][i]]
    var bootstrap_on = opts.bootstrap_kind >= 0
    var seeds = List[UInt64]()
    if bootstrap_on:
        seeds = gbdt_bootstrap_seeds(params.random_seed ^ ORDERED_BOOTSTRAP_SALT)
    # the held-out rows against the model's own borders, the test cursor at
    # the starting point, the detector (`fit_ordered`'s step 8)
    var has_test = eval.n_rows > 0
    var n_eval = eval.n_rows if has_test else 1
    var test_cindex = List[UInt32]()
    if has_test:
        test_cindex = _binarize_columns(eval.x_colmajor, n_eval, n_features, grid, layout)
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
        var doc_ids = List[Int](capacity=total)
        for f in range(n_folds):
            var r = bounds[2 * f + 1]
            for i in range(r):
                var row = perms[learn_p][i]
                doc_ids.append(row)
                # `kernel_alpha` carries Logloss's border (the binding
                # builds the loss that way, as `_loss_row` reads it)
                var lr_row = _loss_row(
                    loss.objective, y[row], cursors[learn_p][f][i],
                    loss.kernel_alpha,
                )
                sw[offsets[f] + i] = lr_row.der2 if opts.score_function_newton else Float32(1.0)
                sg[offsets[f] + i] = lr_row.der
        # ---- the score noise ----
        var score_std = Float32(0.0)
        if opts.random_strength != Float32(0.0):
            var terms = List[Float32](length=total, fill=Float32(0.0))
            for i in range(total):
                if quality[i]:
                    var w = sw[i]
                    if w > Float32(0.0):
                        var q = ftz(sg[i] / w)
                        terms[i] = ftz(ftz(q * q) * w)
            var s2 = _deterministic_sum_lanes(terms, 1, total)[0]
            var mult = ordered_model_length_mult(
                n, Float64(iteration) * Float64(lr)
            )
            score_std = Float32(
                mult * sqrt(Float64(s2) / (Float64(quality_count) + 1e-100))
                * Float64(opts.random_strength)
            )
        # ---- the bootstrap, quality slices only ----
        if bootstrap_on:
            var draws = _ordered_bootstrap_draws(
                opts.bootstrap_kind, seeds, total, opts.bootstrap_param
            )
            for i in range(total):
                if quality[i]:
                    sw[i] = ftz(sw[i] * draws[i])
                    sg[i] = ftz(sg[i] * draws[i])
        # ---- the scale ----
        var absv = List[Float32](length=2 * total, fill=Float32(0.0))
        for i in range(total):
            absv[2 * i] = abs(sw[i])
            absv[2 * i + 1] = abs(sg[i])
        var mags = _deterministic_sum_lanes(absv, 2, total)
        var m0 = Float64(mags[0])
        var m1 = Float64(mags[1])
        var scale = Float32(choose_scale(m1 if m1 > m0 else m0, total))
        # ---- the structure ----
        var splits = _ordered_tree_structure(
            cindex, helpers, max_depth, sw, sg, doc_ids, part_bounds,
            fold_count, fold_bits, scale, l2, feat_offset, feat_shift,
            feat_mask, False, score_std, tree_seed,
        )
        var bins = List[Int](length=n, fill=0)
        for r in range(n):
            var leaf = 0
            for level in range(len(splits)):
                var fid = splits[level].feature
                var mask = feat_mask[fid] << feat_shift[fid]
                var value = UInt32(splits[level].bin) << feat_shift[fid]
                if (cindex[feat_offset[fid] + r] & mask) > value:
                    leaf += 1 << level
            bins[r] = leaf
        var n_leaves = 1 << len(splits)
        # ---- the leaves ----
        for lp in range(learn_count):
            for f in range(n_folds):
                var est = bounds[2 * f]
                comptime if GBDT_ORDERED_SABOTAGE:
                    est = bounds[2 * f + 1]
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
                var test_bins = List[Int](length=n_eval, fill=0)
                for r in range(n_eval):
                    var leaf = 0
                    for level in range(len(splits)):
                        var fid = splits[level].feature
                        var mask = feat_mask[fid] << feat_shift[fid]
                        var value = UInt32(splits[level].bin) << feat_shift[fid]
                        if (test_cindex[Int(layout.features[fid].offset) * n_eval + r] & mask) > value:
                            leaf += 1 << level
                    test_bins[r] = leaf
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
    var model = GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses.copy(), best, stopped_early,
    )
    var text = gbdt_rmse_host_model_text(GbdtRmseHostFit(model^, start))
    return GbdtOrderedHostFit(text^, losses^, test_losses^, best, stopped_early)
