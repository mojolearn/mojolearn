"""lane/apple-fast-sym-iter (2026-10-03): the per-iteration FIXED cost of the
pointwise SymmetricTree arm (`doc_parallel_boosting.mojo`, Plain boosting,
one permutation) on the Apple FAST tier. Everything here compiles ONLY under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` and one
of the defines below; IDENTICAL and every other vendor compile main's loop
unchanged. The profile this answers is docs/apple-fast/notes/sym-iter.md:
main makes eight host waits and ~25 Metal allocations per tree before any
histogram work.

The defines (each default OFF, each its own A/B; `MOJOLEARN_SYM_ITER_ALL`
turns on SYM_BUF_ARENA only; the other three are recorded DROPs):

- `MOJOLEARN_SYM_BUF_ARENA` (`SYM_BUF_ARENA`): a per-fit pool of one
  (`SymIterPool`) for every buffer the loop made per tree -- the two split
  planes, the bins staging, the leaf partitioner (`DeviceLeafPartitioner`,
  pool of one), the oracle's host staging, the three dummies the searcher
  allocated for its fold arm -- and the magnitudes readback into the fit's
  existing `h_mags`. The two drains whose only job was to keep a per-call
  buffer alive (`compute_bins_for_model`'s, `partition_from_bins`' outer one)
  go with the buffers.
- `MOJOLEARN_SYM_REUSE_PARTITION`: deleted 2026-10-09 (DROPPED-noise;
  docs/TOMBSTONES.md).
- `MOJOLEARN_SYM_DERIV_FUSED`: deleted 2026-10-09 (DROPPED-noise;
  docs/TOMBSTONES.md).
- `MOJOLEARN_SYM_LEAF_FROM_STATS`: deleted 2026-10-09 (DROPPED-noise;
  docs/TOMBSTONES.md).

FAST only: the leaf value from the stats is the Newton step over the SNAPPED
gradient plane (`enqueue_snap_plane`, the fixed-point grid the histogram
quantizes at) summed in the partition reduce's order, where the oracle sums
the exact plane in `compute_partition_stats`' order; the partition reuse
changes the row order inside a leaf (the searcher's stable one-bit sorts
instead of a full-key radix sort), so the estimator's leaf sums fold in a
different order. Same leaves, same rows, same quality.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.math import isfinite, sqrt
from std.memory import stack_allocation
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    identical_mul_add,
)
from checks.fixed_point import choose_scale
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.gpu_util.kernel.transform import launch_split_planes_f32
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN, TBinarySplit
from gbdt.models.kernel.add_bin_values import compute_bins_kernel
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import (
    COMPUTE_BINS_BLOCK_SIZE,
    DeviceLeafPartitioner,
)
from gbdt.methods.kernel.pointwise_scores import _block_reduce_sum
from gbdt.methods.pointwise_optimization_subsets import (
    PARTITION_RECORD,
    PARTITION_STAT_STRIDE,
    PART_OFFSET,
    PART_SIZE,
    PART_STAT_SUM,
)
from gbdt.methods.random_score_helper import (
    STD_DEV_BLOCK,
    std_dev_blocks,
    std_dev_partials_kernel,
)
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_MULTICLASS_OVA,
    OBJECTIVE_MULTIRMSE,
    OBJECTIVE_PAIR_LOGIT,
    OBJECTIVE_QUERY_RMSE,
    OBJECTIVE_RMSE,
    OBJECTIVE_YETI_RANK,
    deterministic_sum_lanes_kernel,
    routed_exp,
)
from gbdt.options.catboost_options import LEAF_ESTIMATION_NEWTON

comptime SYM_ITER_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-iter@4956a2234; SYM_ITER_ALL compiled rc=0 on the
#: laptop 2026-10-03, singles never built, never timed.
#: apple-fast LEDGER 2026-10-03: DROP sym-iter-all istella (0.0%, auc
#: .980163 -> .980122). The umbrella now holds only SYM_BUF_ARENA, the one
#: switch with no verdict.
comptime SYM_ITER_ALL = (
    SYM_ITER_FAST_APPLE and is_defined["MOJOLEARN_SYM_ITER_ALL"]()
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-iter@4956a2234; SYM_ITER_ALL compiled rc=0 on the
#: laptop 2026-10-03, singles never built, never timed. Port: the searcher's
#: pooled fold dummies now sit in front of main's ORD_ALL observation
#: scratch.
comptime SYM_BUF_ARENA = SYM_ITER_FAST_APPLE and (
    # MEASURED M3 FAST; broader workload evidence remains separate.
# F12/arena M3 2026-10-06: 6 scored caller times; B/A
# 0.9336..1.0082 (mixed/regressing); FAST candidate remains OFF.
# Scored FAST quality 3/3 within existing bands; PASS.
# One warmup/one score; caller67d0efb29; exact cases/builds/hashes:
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena.
# Compilation/identity reused. No combined-toggle/full-board claim.
    is_defined["MOJOLEARN_SYM_BUF_ARENA"]() or SYM_ITER_ALL
)
#: TOMBSTONE: MOJOLEARN_SYM_LEAF_FROM_STATS (DROPPED-noise: sym-iter-leaf istella/taxi -0.2%, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (one Newton step from the searcher's partition stats, on the device); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_LEAF_FROM_STATS.patch; record in docs/TOMBSTONES.md.
#: TOMBSTONE: MOJOLEARN_SYM_REUSE_PARTITION (DROPPED-noise: sym-iter-reuse taxi +0.3%, old base; F12 2026-10-06 B/A 0.94..1.03 mixed)
#: deleted 2026-10-09 on lane/owed-deletions-D1 (the searcher's final partition as the estimator's); code recoverable at
#: b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SYM_REUSE_PARTITION.patch; record in docs/TOMBSTONES.md.
#: TOMBSTONE: MOJOLEARN_SYM_DERIV_FUSED (DROPPED-noise: sym-iter-fused istella/taxi -0.1%, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (the next tree's gradient pass behind this tree's tail drain); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_DERIV_FUSED.patch; record in docs/TOMBSTONES.md.
comptime SYM_ITER_ANY = (
    SYM_BUF_ARENA
)



struct SymIterPool(Movable):
    """The fit's pool of one for the pointwise symmetric arm's per-tree
    buffers, keyed on (n_rows, max_depth). Buffers a define does not use
    are one element wide, so a single define's A/B carries no other
    define's memory."""

    var n_rows_key: Int
    var max_depth_key: Int
    var n_leaves_cap: Int
    #: SYM_BUF_ARENA: the split planes (`split_stat_planes` made two
    #: n_rows buffers per tree)
    var w: DeviceBuffer[DType.float32]
    var t: DeviceBuffer[DType.float32]
    #: SYM_BUF_ARENA: `compute_bins_for_model`'s staging (five host and
    #: five device buffers per tree)
    var h_off: HostBuffer[DType.uint32]
    var h_shift: HostBuffer[DType.uint32]
    var h_mask: HostBuffer[DType.uint32]
    var h_bin: HostBuffer[DType.uint32]
    var h_eq: HostBuffer[DType.uint8]
    var d_off: DeviceBuffer[DType.uint32]
    var d_shift: DeviceBuffer[DType.uint32]
    var d_mask: DeviceBuffer[DType.uint32]
    var d_bin: DeviceBuffer[DType.uint32]
    var d_eq: DeviceBuffer[DType.uint8]
    #: SYM_BUF_ARENA: the leaf partitioner (`partition_from_bins` built one
    #: per call: eight n_rows-sized buffers); its `bins` is the bins target
    var parts: DeviceLeafPartitioner

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        max_depth: Int,
        sm_count: Int,
    ) raises:
        self.n_rows_key = n_rows
        self.max_depth_key = max_depth
        var n_leaves = 1 << max_depth
        self.n_leaves_cap = n_leaves
        var depth_cap = max_depth if max_depth > 0 else 1
        var rows_arena = n_rows if SYM_BUF_ARENA else 1
        var depth_arena = depth_cap if SYM_BUF_ARENA else 1
        var leaves_arena = n_leaves if SYM_BUF_ARENA else 1
        self.w = ctx.enqueue_create_buffer[DType.float32](rows_arena)
        self.t = ctx.enqueue_create_buffer[DType.float32](rows_arena)
        self.h_off = ctx.enqueue_create_host_buffer[DType.uint32](depth_arena)
        self.h_shift = ctx.enqueue_create_host_buffer[DType.uint32](
            depth_arena
        )
        self.h_mask = ctx.enqueue_create_host_buffer[DType.uint32](
            depth_arena
        )
        self.h_bin = ctx.enqueue_create_host_buffer[DType.uint32](depth_arena)
        self.h_eq = ctx.enqueue_create_host_buffer[DType.uint8](depth_arena)
        self.d_off = ctx.enqueue_create_buffer[DType.uint32](depth_arena)
        self.d_shift = ctx.enqueue_create_buffer[DType.uint32](depth_arena)
        self.d_mask = ctx.enqueue_create_buffer[DType.uint32](depth_arena)
        self.d_bin = ctx.enqueue_create_buffer[DType.uint32](depth_arena)
        self.d_eq = ctx.enqueue_create_buffer[DType.uint8](depth_arena)
        self.parts = DeviceLeafPartitioner(ctx, rows_arena, leaves_arena)
        ctx.synchronize()


def sym_pool_get(
    ctx: DeviceContext,
    mut pool: List[SymIterPool],
    n_rows: Int,
    max_depth: Int,
    sm_count: Int,
) raises:
    """The pool of one: build on the first tree, rebuild on a shape change."""
    if (
        len(pool) == 0
        or pool[0].n_rows_key != n_rows
        or pool[0].max_depth_key != max_depth
    ):
        pool.clear()
        pool.append(SymIterPool(ctx, n_rows, max_depth, sm_count))


def sym_split_planes(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    n_rows: Int,
    mut pool: SymIterPool,
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]]:
    """`split_stat_planes` into the pool's two planes: the same launch,
    no allocation. The handles returned keep the pool's buffers alive in
    the searcher's `TL2Target`."""
    launch_split_planes_f32(ctx, pool.w, pool.t, stats, n_rows, n_rows)
    return (pool.w.copy(), pool.t.copy())


def sym_scale_from_mags(
    h_mags: HostBuffer[DType.float32], n_rows: Int
) raises -> Float32:
    """`choose_scale` over the larger plane magnitude, as the loop derives
    it after its magnitudes drain."""
    var m0 = Float64(h_mags[0])
    if m0 < 0.0:
        m0 = -m0
    var m1 = Float64(h_mags[1])
    if m1 < 0.0:
        m1 = -m1
    return Float32(choose_scale(m1 if m1 > m0 else m0, n_rows))


def sym_compute_bins_pooled(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    splits: List[TBinarySplit],
    depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut pool: SymIterPool,
) raises:
    """`compute_bins_for_model` with the pool's staging and no drain: the
    same records, the same `compute_bins_kernel`, into the pooled
    partitioner's `bins`. The staging is rewritten by the NEXT tree only,
    after this tree's drains."""
    if depth <= 0 or depth > pool.max_depth_key:
        raise Error(
            "sym_compute_bins_pooled: depth " + String(depth)
            + " outside the pool's " + String(pool.max_depth_key)
        )
    if len(splits) < depth:
        raise Error(
            "sym_compute_bins_pooled: " + String(len(splits))
            + " splits for depth " + String(depth)
        )
    for level in range(depth):
        ref cf = layout.features[Int(splits[level].feature_id)]
        pool.h_off.unsafe_ptr().unsafe_store(
            level, cf.offset * UInt32(n_rows)
        )
        pool.h_shift.unsafe_ptr().unsafe_store(level, cf.shift)
        pool.h_mask.unsafe_ptr().unsafe_store(level, cf.mask)
        pool.h_bin.unsafe_ptr().unsafe_store(
            level, UInt32(Int(splits[level].bin_idx))
        )
        var take_bin = Int(splits[level].split_type) == BIN_SPLIT_TAKE_BIN
        pool.h_eq.unsafe_ptr().unsafe_store(
            level, UInt8(1) if take_bin else UInt8(0)
        )
    ctx.enqueue_copy(dst_buf=pool.d_off, src_ptr=pool.h_off.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=pool.d_shift, src_ptr=pool.h_shift.unsafe_ptr()
    )
    ctx.enqueue_copy(dst_buf=pool.d_mask, src_ptr=pool.h_mask.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=pool.d_bin, src_ptr=pool.h_bin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=pool.d_eq, src_ptr=pool.h_eq.unsafe_ptr())
    var blocks = (n_rows + COMPUTE_BINS_BLOCK_SIZE - 1) // COMPUTE_BINS_BLOCK_SIZE
    if blocks < 1:
        blocks = 1
    ctx.enqueue_function[compute_bins_kernel](
        cindex.unsafe_ptr(),
        pool.d_off.unsafe_ptr(),
        pool.d_shift.unsafe_ptr(),
        pool.d_mask.unsafe_ptr(),
        pool.d_bin.unsafe_ptr(),
        pool.d_eq.unsafe_ptr(),
        Int32(depth),
        Int32(n_rows),
        pool.parts.bins.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(COMPUTE_BINS_BLOCK_SIZE, 1, 1),
    )


def sym_objective_is_pointwise(objective: Int) -> Bool:
    """The objectives whose gradient pass is `launch_approximate` (the
    single-dimensional pointwise oracles SYM_BUF_ARENA stages)."""
    return not (
        objective == OBJECTIVE_MULTICLASS
        or objective == OBJECTIVE_MULTICLASS_OVA
        or objective == OBJECTIVE_MULTIRMSE
        or objective == OBJECTIVE_QUERY_RMSE
        or objective == OBJECTIVE_PAIR_LOGIT
        or objective == OBJECTIVE_YETI_RANK
    )


# TOMBSTONE: MOJOLEARN_SYM_LEAF_FROM_STATS (DROPPED-noise: sym-iter-leaf istella/taxi -0.2%, old base) deleted 2026-10-09 on
# lane/owed-deletions-D1 (one Newton step from the searcher's partition stats, on the device); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_LEAF_FROM_STATS.patch; record in docs/TOMBSTONES.md.


# TOMBSTONE: MOJOLEARN_SYM_DERIV_FUSED (DROPPED-noise: sym-iter-fused istella/taxi -0.1%, old base) deleted 2026-10-09 on
# lane/owed-deletions-D1 (the next tree's gradient pass behind this tree's tail drain); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_DERIV_FUSED.patch; record in docs/TOMBSTONES.md.
