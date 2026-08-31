# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Quantized gradient pairs + one shared histogram per block (DEV 1911/1912).

================= DEVIATION BLOCK (whole file) =================
NO CATBOOST COUNTERPART. This family is the recons' borrowed design, not a
port of `hist_one_byte.cu`:

* **DEV 1911 -- fixed-point gradient pairs, packed one word per row.**
  XGBoost converts every float `(grad, hess)` to fixed-point integers ONCE
  per round with a scale derived from the round's magnitudes
  (`GradientQuantiser::ToFixedPoint`, `quantiser.cuh:21-31`;
  `CreateRoundingFactor`, `histogram.cu:90-115` -- recon_xgboost_gpu.md a),
  and LightGBM packs the discretized pair into ONE word per row so the hist
  kernel loads it in one instruction (`cuda_histogram_constructor.cu:291-294`
  -- recon_lightgbm_cuda.md b2). `quantize_pair_kernel` mirrors the DESIGN
  on this port's planes: both stat planes of a row become two Int32 through
  the SAME `hist2_quantize(stat, fixed_scale, hist2_dither(position))` the
  shared-Int32 arms already apply inline -- same scale (`choose_scale`'s
  bound, the round's magnitude contract), same dithered-floor rounding
  (this port's measured stand-in for XGBoost's rounding constant: plain
  truncation and round-to-nearest both failed, `hist2_quantize`'s docstring
  carries the numbers), same positions, same draws -- packed into one
  UInt64 via SIMD bitcast (lane 0 = stat plane 0 / weight-hess, lane 1 =
  stat plane 1 / gradient: their grad-high/hess-low halves). The pack is a
  LOAD format only: accumulation is per-stat Int32, so there is no
  cross-half carry hazard and none of LightGBM's width-laddering overflow
  machinery is needed (XGBoost's separate-words accumulation, exactly).
  NO shifts touch the pack/unpack -- SIMD bitcast both ways -- so the
  int-widening sign-extension trap cannot apply.

  Because the addends are bit-for-bit the values the fused 8-bit /
  shared-Int32 arms compute inline, and integer addition is associative,
  the per-cell totals this family flushes are BIT-IDENTICAL to that arm's
  for the same rows -- which is the gate the orchestrator can hold it to.

  XGBoost quantizes once per ROUND; this port re-permutes the stat planes
  at every split, so the quantize pass runs per LEVEL over exactly the
  rows being built (grid y = the non-zero compute leaves). One extra
  8 B/row write + read per level, in exchange for every feature-group
  pass over the rows loading 8 aligned bytes instead of two float planes
  plus a quantize per replica. DEVIATION 1902 (stop permuting stats) is
  what would make it per-round.

* **DEV 1912 -- ONE shared histogram per block, integer atomics, one
  global flush.** CatBoost's warp-private float slices cost 32 floats of
  shared memory PER THREAD (128 B/thread, `hist_2_one_byte_base.mojo`
  header); XGBoost keeps ONE histogram copy per thread block in shared
  memory, every thread accumulating with integer atomics, flushed to
  global memory once per block (`histogram.cu:118-233`,
  `AtomicAdd64As32` at `histogram.cuh:24-37` -- recon a / borrow 2). The
  cost becomes bytes PER BIN, amortized over the whole block. The
  threadgroup adds here are Int32 `Atomic.fetch_add` -- the one
  shared-memory atomic EVERY column has (Metal has no threadgroup float
  atomics but full integer ones, `column_has_threadgroup_int_atomics`),
  which is why the quantized integer histogram is the portable form of
  the shared-copy design. The global flush lands in a per-leaf Int32
  accumulator (`q_acc`); XGBoost flushes with native 64-bit global
  atomics, which Metal does not have, so the flush is per-stat 32-bit --
  the same two-words-per-pair shape their own shared-memory half uses.
  `qh_write_hist_kernel` dequantizes accumulator -> flat float histogram
  with the IDENTICAL expression `Float32(Int(q)) / fixed_scale` the
  standing bridge uses (`write_reduces_from_fixed_kernel`), and zeroes
  the cell it read, so the accumulator self-cleans and downstream --
  scan, subtract, score -- is untouched: dequantize-on-flush.

* **DEV 1913 -- the feature group fills the COLUMN's shared budget.**
  `QH_GROUP_FEATURES` comes off the kernel matrix
  (`quantized_hist_group_features_for`): 16 on Apple/32 KB, 24 on
  NVIDIA/48 KB, 32 on AMD/64 KB -- never XGBoost's 227 KB opt-in number.

* **DEV 1914 -- grid floors.** `QH_MIN_ITEMS_PER_BLOCK = 8192` is
  XGBoost's `kMinItemsPerBlock` (`histogram.cu:405-419`): a partition's
  active row-replicas are `ceil(rows / 8192)`, so a small leaf never
  launches full-device work (idle replicas return before touching shared
  memory). `QH_MIN_TOTAL_BLOCKS = 160` is LightGBM's `min_grid_dim_y_`
  (`cuda_histogram_constructor.hpp:152`): the HOST-side launch geometry
  never drops below it, so late small-leaf levels stay device-filling
  instead of launch-bound. Both live in the launcher's `qh_replicas`;
  the in-kernel trim is this file's `active_block_count`.

MODE: this family is reachable ONLY through `greedy_quantized_hist_for`,
which is comptime False under IDENTICAL -- the IDENTICAL column never
launches anything in this file and its schedule is byte-for-byte the
pre-round one.

LANE-AGNOSTIC BY CONSTRUCTION (contrast DEVIATIONS 1906/1910): Int32
atomics, block barriers, uniform trip counts, no lane-indexed slices --
32-lane and 64-lane columns compile this file unchanged.

TWO-STAT BY CONSTRUCTION: the packed word holds exactly two planes. The
launcher refuses other shapes to the standing arms (multi-stat keeps the
PASS route, the same split `launch_histograms_for_blocks` already makes
for the fused 8-bit arm).
=================================================================

Shared layout: `cell = (feature_in_group << 9) + (bin << 1) + stat` --
`[feature][bin][stat]`, 512 Int32 cells per feature, so one feature's two
stat columns for one bin are adjacent (LightGBM's interleaved-pair layout,
`cuda_histogram_constructor.cu:31-67`).

OVERFLOW GUARD, stated once: `fixed_scale` satisfies
`original/fixed_point.choose_scale`'s bound -- the FULL-plane sum of
magnitudes maps under 2^30 - 1 with the dither's +1/row allowance exact --
and every cell here (shared: one block's rows; global: one leaf's rows) is
a partial sum over a SUBSET of all rows, so no Int32 cell can wrap at any
depth. The same contract every shared-Int32 arm in this package rides on.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from original.kernel_matrix import (
    TARGET_COLUMN,
    quantized_hist_group_features_for,
)
from original.numerics import ftz

comptime QH_BLOCK = 512
comptime QH_BINS = 256
comptime QH_STATS = 2
#: DEV 1913: features per block from the COLUMN's shared budget
#: (`quantized_hist_group_features_for` -- 16/24/32, never 227 KB).
comptime QH_GROUP_FEATURES = quantized_hist_group_features_for[
    TARGET_COLUMN
]()
#: whole compressed-index words per group; the row guarantees 4 | G.
comptime QH_GROUP_WORDS = QH_GROUP_FEATURES // 4
#: Int32 cells of the ONE per-block shared histogram.
comptime QH_SMEM = QH_GROUP_FEATURES * QH_BINS * QH_STATS
#: DEV 1914: XGBoost `kMinItemsPerBlock` (`histogram.cu:405-419`).
comptime QH_MIN_ITEMS_PER_BLOCK = 8192
#: DEV 1914: LightGBM `min_grid_dim_y_ = 160`
#: (`cuda_histogram_constructor.hpp:152`); read by the launcher.
comptime QH_MIN_TOTAL_BLOCKS = 160


def quantize_pair_kernel[ridx_stats: Bool = False](
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    q_stats: MutPointer[UInt64, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """DEV 1911: both stat planes of a row -> one packed UInt64, once per
    level, over exactly the partitions being built (grid y = the compute
    id list; positions across partitions are disjoint, so no write races).

    The addends are the SAME `hist2_quantize(stat, scale, dither(pos))`
    values the in-kernel quantizers produce -- position-keyed dither, one
    draw per row shared by both planes, exactly as every hist kernel in
    this package draws it -- so downstream integer totals cannot differ
    from the inline arms'. Pack/unpack is SIMD bitcast, no shifts: lane 0
    = plane 0 (weight/hess), lane 1 = plane 1 (gradient)."""
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var stat_line_size = Int(stat_line_size_in)
    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < p_size:
        var pos = p_offset + i
        var u = hist2_dither(pos)
        var src = pos

        @parameter
        if ridx_stats:
            # DEVIATION 1902 x 1911: the stat plane is stationary, so the
            # VALUE is gathered through the row id at this position while
            # the dither stays keyed on the storage position -- the same
            # (value(ridx[pos]), dither(pos)) pairing the inline ridx_stats
            # hist arms produce, so `q_stats[pos]` holds bit-for-bit the
            # pair the permuted plane would have held here and the
            # positional load in `qh_hist_gather_kernel` stays correct
            # unchanged.
            src = Int(ldg(indices + pos))
        var q1 = hist2_quantize(ldg(stats + src), fixed_scale, u)
        var q2 = hist2_quantize(
            ldg(stats + (stat_line_size + src)), fixed_scale, u
        )
        var pair = SIMD[DType.int32, 2](q1, q2)
        q_stats.unsafe_store(pos, bitcast[DType.uint64, 1](pair)[0])
        i += stride


@always_inline
def qh_add_row(
    pair: SIMD[DType.int32, 2],
    words: Int,
    row: Int,
    bins_line_size: Int,
    bins_p: MutPointer[UInt32, MutAnyOrigin],
    tid: Int,
    smem: UnsafePointer[
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """One row into the block's ONE shared histogram: `words` cindex words
    (4 one-byte features each), two Int32 threadgroup atomics per feature.
    The `(tid + i) & 3` byte rotation decorrelates which feature's cells a
    warp hits together (the fused 8-bit arm's own trick). A bin past a
    narrow feature's fold count (the skip mark) lands in a cell the flush
    never reads -- `bin < folds` bounds every flush, the ladder's own drop.
    """
    for w in range(words):
        var ci = ldg(bins_p + (w * bins_line_size + row))
        comptime for i in range(4):
            var j = (tid + i) & 3
            var bin = Int((ci >> UInt32(24 - 8 * j)) & UInt32(255))
            var cell = (((w << 2) + j) << 9) + (bin << 1)
            # DEVIATION 1898: upstream's atomicAdd is relaxed; the
            # non-Apple Mojo default is seq_cst.
            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                smem.unsafe_offset(cell), pair[0]
            )
            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                smem.unsafe_offset(cell + 1), pair[1]
            )


@always_inline
def qh_flush(
    tid: Int,
    active_block_count: Int,
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: Int,
    f_count: Int,
    hist_block_offset: Int,
    hist_cell_count: Int,
    q_acc: MutPointer[Int32, MutAnyOrigin],
    smem: UnsafePointer[
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """DEV 1912's ONE global flush per block: every nonzero shared cell
    goes to the leaf's Int32 accumulator -- atomically when the partition
    is covered by several row-replicas, a plain store when this block is
    the only writer of its (leaf, group) cells (the fused 8-bit arm's own
    dispatch). `q_acc` is dense-leaf-major: `(blockIdx.y * 2 + stat) *
    hist_cell_count + cell`, the bridge's exact read layout."""
    barrier()
    var span = f_count * QH_BINS * QH_STATS
    var idx = tid
    while idx < span:
        var q = smem[idx]
        if q != Int32(0):
            var f_local = idx >> 9
            var bin = (idx >> 1) & (QH_BINS - 1)
            var stat = idx & 1
            var fid = feature_offset + f_local
            if bin < Int(feature_folds.unsafe_load(fid)):
                var cell = (
                    hist_block_offset
                    + Int(feature_fold_offset.unsafe_load(fid))
                    + bin
                )
                var dst = (
                    Int(block_idx.y) * QH_STATS + stat
                ) * hist_cell_count + cell
                if active_block_count > 1:
                    # DEVIATION 1898: relaxed, as at every atomic here.
                    _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                        q_acc.unsafe_offset(dst), q
                    )
                else:
                    # sole writer of these cells this level, and the
                    # bridge zeroed them on its last read
                    q_acc.unsafe_store(dst, q)
        idx += QH_BLOCK


def qh_hist_kernel(
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    q_stats: MutPointer[UInt64, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    q_acc: MutPointer[Int32, MutAnyOrigin],
    hist_block_offset_in: Int32,
    hist_cell_count_in: Int32,
):
    """Direct loads (depth 0: the row index is the identity, as every
    depth-0 kernel in this package assumes). Grid x = feature groups times
    row-replicas, y = the compute leaves; one shared histogram per block,
    one flush."""
    var tid = Int(thread_idx.x)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    var feature_blocks = (
        f_count_in + QH_GROUP_FEATURES - 1
    ) // QH_GROUP_FEATURES
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var group_id = Int(block_idx.x) // max_blocks_per_part
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var feature_offset = group_id * QH_GROUP_FEATURES
    var f_count = min(f_count_in - feature_offset, QH_GROUP_FEATURES)
    var words = (f_count + 3) // 4
    var bins_p = bins + Int(cindex_base_in) + bins_line_size * (
        group_id * QH_GROUP_WORDS
    )

    # DEV 1914: XGBoost's min-work trim -- a small partition activates
    # ceil(rows / kMinItemsPerBlock) replicas, the rest return here.
    var active_block_count = (
        p_size + QH_MIN_ITEMS_PER_BLOCK - 1
    ) // QH_MIN_ITEMS_PER_BLOCK
    if active_block_count > max_blocks_per_part:
        active_block_count = max_blocks_per_part
    if active_block_count < 1:
        active_block_count = 1
    if local_block_idx >= active_block_count:
        return

    var smem = stack_allocation[
        QH_SMEM,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < QH_SMEM:
        smem[z] = Int32(0)
        z += QH_BLOCK
    barrier()

    var pos = p_offset + local_block_idx * QH_BLOCK + tid
    var stride = active_block_count * QH_BLOCK
    var pend = p_offset + p_size
    while pos < pend:
        var pair = bitcast[DType.int32, 2](ldg(q_stats + pos))
        qh_add_row(pair, words, pos, bins_line_size, bins_p, tid, smem)
        pos += stride

    qh_flush(
        tid, active_block_count, feature_folds, feature_fold_offset,
        feature_offset, f_count, Int(hist_block_offset_in),
        Int(hist_cell_count_in), q_acc, smem,
    )


def qh_hist_gather_kernel(
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    indices: MutPointer[UInt32, MutAnyOrigin],
    q_stats: MutPointer[UInt64, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    q_acc: MutPointer[Int32, MutAnyOrigin],
    hist_block_offset_in: Int32,
    hist_cell_count_in: Int32,
):
    """Indexed loads (below the root): bins through `indices`, the packed
    pairs contiguous at the storage position -- the same conventions as
    every gather kernel in this package. Gradients never move for this
    kernel's sake: the 4-byte row index is the only indirection (XGBoost's
    own gather, `histogram.cu:186, 212`)."""
    var tid = Int(thread_idx.x)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    var feature_blocks = (
        f_count_in + QH_GROUP_FEATURES - 1
    ) // QH_GROUP_FEATURES
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var group_id = Int(block_idx.x) // max_blocks_per_part
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var feature_offset = group_id * QH_GROUP_FEATURES
    var f_count = min(f_count_in - feature_offset, QH_GROUP_FEATURES)
    var words = (f_count + 3) // 4
    var bins_p = cindex + Int(cindex_base_in) + bins_line_size * (
        group_id * QH_GROUP_WORDS
    )

    var active_block_count = (
        p_size + QH_MIN_ITEMS_PER_BLOCK - 1
    ) // QH_MIN_ITEMS_PER_BLOCK
    if active_block_count > max_blocks_per_part:
        active_block_count = max_blocks_per_part
    if active_block_count < 1:
        active_block_count = 1
    if local_block_idx >= active_block_count:
        return

    var smem = stack_allocation[
        QH_SMEM,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < QH_SMEM:
        smem[z] = Int32(0)
        z += QH_BLOCK
    barrier()

    var pos = p_offset + local_block_idx * QH_BLOCK + tid
    var stride = active_block_count * QH_BLOCK
    var pend = p_offset + p_size
    while pos < pend:
        var row = Int(ldg(indices + pos))
        var pair = bitcast[DType.int32, 2](ldg(q_stats + pos))
        qh_add_row(pair, words, row, bins_line_size, bins_p, tid, smem)
        pos += stride

    qh_flush(
        tid, active_block_count, feature_folds, feature_fold_offset,
        feature_offset, f_count, Int(hist_block_offset_in),
        Int(hist_cell_count_in), q_acc, smem,
    )


def qh_write_hist_kernel(
    hist_ids: MutPointer[UInt32, MutAnyOrigin],
    q_acc: MutPointer[Int32, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
    hist_cell_count_in: Int32,
    dst_histogram: MutPointer[Float32, MutAnyOrigin],
):
    """Dequantize-on-flush: accumulator -> flat float histogram, the SAME
    conversion expression as `write_reduces_from_fixed_kernel`
    (`Float32(Int(q)) / fixed_scale`, ftz'd at the store -- IDENTITY_PATHS
    ROW 10's seam discipline), and the accumulator cell zeroed where it was
    nonzero so the next level inherits nothing. Downstream -- scan,
    subtract, score, split -- reads the flat histogram exactly as before;
    nothing after this kernel knows the build was quantized.

    Grid: x over cells, y over the COMPUTE id list (dense, the build's
    `blockIdx.y`), z over the two stats. Every cell of the built leaves is
    stored (zero included): the flat slot must equal the build's output,
    exactly as the standing bridge leaves it."""
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var hist_cell_count = Int(hist_cell_count_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < hist_cell_count:
        var dense = Int(block_idx.y)
        var stat = Int(block_idx.z)
        var stat_count = Int(grid_dim.z)
        var dst_id = Int(hist_ids.unsafe_load(dense))
        var src = (dense * stat_count + stat) * hist_cell_count + cell
        var q = q_acc.unsafe_load(src)
        var val = Float32(0.0)
        if q != Int32(0):
            val = ftz(Float32(Int(q)) / fixed_scale)
            q_acc.unsafe_store(src, Int32(0))
        var dst = (
            dst_id * hist_cell_count * stat_count
            + stat * hist_cell_count
            + cell
        )
        dst_histogram.unsafe_store(dst, val)
