"""lane/apple-fast-sym-iter (2026-10-03): the per-iteration FIXED cost of the
pointwise SymmetricTree arm (`doc_parallel_boosting.mojo`, Plain boosting,
one permutation) on the Apple FAST tier. Everything here compiles ONLY under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` and one
of the defines below; IDENTICAL and every other vendor compile main's loop
unchanged. The profile this answers is docs/apple-fast/notes/sym-iter.md:
main makes eight host waits and ~25 Metal allocations per tree before any
histogram work.

The defines (each default OFF, each its own A/B; `MOJOLEARN_SYM_ITER_ALL`
turns every one on):

- `MOJOLEARN_SYM_BUF_ARENA` (`SYM_BUF_ARENA`): a per-fit pool of one
  (`SymIterPool`) for every buffer the loop made per tree -- the two split
  planes, the bins staging, the leaf partitioner (`DeviceLeafPartitioner`,
  pool of one), the oracle's host staging, the three dummies the searcher
  allocated for its fold arm -- and the magnitudes readback into the fit's
  existing `h_mags`. The two drains whose only job was to keep a per-call
  buffer alive (`compute_bins_for_model`'s, `partition_from_bins`' outer one)
  go with the buffers.
- `MOJOLEARN_SYM_REUSE_PARTITION` (`SYM_REUSE_PARTITION`): the searcher's own
  final partition is the estimator's partition. After the last level
  `subsets.indices` is the row order grouped by leaf and `subsets.partitions`
  the (offset, size) per leaf; the searcher's single tail drain carries the
  partitions back to the host (`sym_parts_out`), and `compute_bins_for_model`
  + `partition_from_bins` (one bins launch, a 2^depth-way radix sort, two or
  three drains) are skipped. Falls back to main's path when the tree stopped
  early (a repeated split: the structure is shorter than `max_depth` and the
  subsets' bins carry a redundant bit).
- `MOJOLEARN_SYM_DERIV_FUSED` (`SYM_DERIV_FUSED`): the NEXT tree's gradient
  pass (`launch_approximate`, the fv and magnitude folds, the score-noise std
  dev) is enqueued behind this tree's cursor update and the estimator's tail
  drain settles both, so the magnitude drain and the std-dev drain at the top
  of the next iteration disappear. The launches stay separate kernels; what
  is fused is the command buffer and the wait.
- `MOJOLEARN_SYM_LEAF_FROM_STATS` (`SYM_LEAF_FROM_STATS`, implies
  `SYM_REUSE_PARTITION`): DEVIATION 64 for the pointwise arm. Under Newton
  with one iteration the leaf is `sum(der) / (sum(der2) + l2)` over the
  leaf's rows at the current cursor, and `subsets.partition_stats` already
  holds `sum(weight * der)` per leaf from the search planes; one block per
  leaf reduces the Hessian plane (`weight` for RMSE, `weight * p * (1 - p)`
  for Logloss), writes the leaf, and a second launch applies it through the
  searcher's partition. No gathers, no oracle, no walker drain; the 2^depth
  leaf values ride the tail drain back to the host for the model.

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
#: laptop 2026-10-03, singles never built, never timed. Umbrella: all four.
comptime SYM_ITER_ALL = (
    SYM_ITER_FAST_APPLE and is_defined["MOJOLEARN_SYM_ITER_ALL"]()
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-iter@4956a2234; SYM_ITER_ALL compiled rc=0 on the
#: laptop 2026-10-03, singles never built, never timed. Port: the searcher's
#: pooled fold dummies now sit in front of main's ORD_ALL observation
#: scratch.
comptime SYM_BUF_ARENA = SYM_ITER_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_BUF_ARENA"]() or SYM_ITER_ALL
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-iter@4956a2234; SYM_ITER_ALL compiled rc=0 on the
#: laptop 2026-10-03, singles never built, never timed.
comptime SYM_LEAF_FROM_STATS = SYM_ITER_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_LEAF_FROM_STATS"]() or SYM_ITER_ALL
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-iter@4956a2234; SYM_ITER_ALL compiled rc=0 on the
#: laptop 2026-10-03, singles never built, never timed.
comptime SYM_REUSE_PARTITION = SYM_ITER_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_REUSE_PARTITION"]() or SYM_LEAF_FROM_STATS
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-iter@4956a2234; SYM_ITER_ALL compiled rc=0 on the
#: laptop 2026-10-03, singles never built, never timed. Port: compiled with
#: EST_SHRINK_FUSED, this prefetch wins on its trees (the estimation hook
#: stands down).
comptime SYM_DERIV_FUSED = SYM_ITER_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_DERIV_FUSED"]() or SYM_ITER_ALL
)
comptime SYM_ITER_ANY = (
    SYM_BUF_ARENA or SYM_REUSE_PARTITION or SYM_DERIV_FUSED
    or SYM_LEAF_FROM_STATS
)

#: the one-block-per-leaf Hessian reduce
comptime SYM_LEAF_BLOCK = 256
#: the leaf-apply kernel's block
comptime SYM_APPLY_BLOCK = 256


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
    #: SYM_REUSE_PARTITION: the searcher's `subsets.partitions` on the host,
    #: filled by the searcher's own tail drain
    var h_parts: HostBuffer[DType.uint32]
    #: SYM_LEAF_FROM_STATS: the leaf values, device and host
    var d_est: DeviceBuffer[DType.float32]
    var h_est: HostBuffer[DType.float32]
    #: SYM_DERIV_FUSED: the score-noise std dev's partials, fold and readback
    var d_sd_partials: DeviceBuffer[DType.float32]
    var d_sd_out: DeviceBuffer[DType.float32]
    var h_sd: HostBuffer[DType.float32]

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
        var parts_reuse = 2 * n_leaves if SYM_REUSE_PARTITION else 1
        self.h_parts = ctx.enqueue_create_host_buffer[DType.uint32](
            parts_reuse
        )
        var leaves_stats = n_leaves if SYM_LEAF_FROM_STATS else 1
        self.d_est = ctx.enqueue_create_buffer[DType.float32](leaves_stats)
        self.h_est = ctx.enqueue_create_host_buffer[DType.float32](
            leaves_stats
        )
        var sd_blocks = 1
        if SYM_DERIV_FUSED:
            sd_blocks = std_dev_blocks(n_rows, sm_count)
            if sd_blocks < 1:
                sd_blocks = 1
        self.d_sd_partials = ctx.enqueue_create_buffer[DType.float32](
            sd_blocks
        )
        self.d_sd_out = ctx.enqueue_create_buffer[DType.float32](1)
        self.h_sd = ctx.enqueue_create_host_buffer[DType.float32](1)
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
    """The objectives whose gradient pass is `launch_approximate`: the
    prefetch (SYM_DERIV_FUSED) replays exactly that launch."""
    return not (
        objective == OBJECTIVE_MULTICLASS
        or objective == OBJECTIVE_MULTICLASS_OVA
        or objective == OBJECTIVE_MULTIRMSE
        or objective == OBJECTIVE_QUERY_RMSE
        or objective == OBJECTIVE_PAIR_LOGIT
        or objective == OBJECTIVE_YETI_RANK
    )


def sym_leaf_from_stats_ok(
    objective: Int,
    leaf_estimation_method: Int,
    leaf_estimation_iterations: Int,
    approx_dim: Int,
) -> Bool:
    """The closed form holds for one Newton step of a single-dimensional
    pointwise loss whose Hessian plane the kernel knows: RMSE (`der2 = 1`)
    and Logloss (`der2 = p (1 - p)`)."""
    return (
        (objective == OBJECTIVE_RMSE or objective == OBJECTIVE_LOGLOSS)
        and leaf_estimation_method == LEAF_ESTIMATION_NEWTON
        and leaf_estimation_iterations == 1
        and approx_dim == 1
    )


def sym_leaves_from_stats_kernel[
    objective: Int, block_size: Int
](
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    l2: Float32,
    est: MutPointer[Float32, MutAnyOrigin],
):
    """ONE BLOCK PER LEAF. The Newton step from zero,
    `gradient / (hessian + lambda + 1e-20)` (`_diagonal_direction`), with
    the gradient `sum(weight * der)` read from the searcher's partition
    statistics (`PART_STAT_SUM`, the snapped gradient plane) and the
    Hessian `sum(weight * der2)` reduced here over the leaf's rows at the
    current cursor: `weight` for RMSE, `weight * p * (1 - p)` for Logloss
    with `cross_entropy_kernel`'s `p`. An empty leaf is zero
    (`regularize`'s MinLeafWeight arm). Float32 throughout (FAST)."""
    var leaf = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var offset = Int(partitions.unsafe_load(leaf * PARTITION_RECORD + PART_OFFSET))
    var size = Int(partitions.unsafe_load(leaf * PARTITION_RECORD + PART_SIZE))
    var buf = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var hsum = Float32(0.0)
    var i = tid
    while i < size:
        var row = Int(indices.unsafe_load(offset + i))
        var weight = Float32(1.0)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(row)

        @parameter
        if objective == OBJECTIVE_LOGLOSS:
            var exp_val = routed_exp(cursor.unsafe_load(row))
            var p = Float32(1.0)
            if isfinite(exp_val):
                p = exp_val / (Float32(1.0) + exp_val)
            p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
            hsum += weight * (p * (Float32(1.0) - p))
        else:
            hsum += weight
        i += block_size
    buf[unsafe_offset=tid] = hsum
    barrier()
    var hess = _block_reduce_sum[block_size](buf, tid)
    if tid == 0:
        var value = Float32(0.0)
        if size > 0:
            var grad = part_stats.unsafe_load(
                leaf * PARTITION_STAT_STRIDE + PART_STAT_SUM
            )
            var denom = hess + l2
            if denom > Float32(0.0):
                value = grad / (denom + Float32(1e-20))
        est.unsafe_store(leaf, value)


def sym_add_leaves_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    est: MutPointer[Float32, MutAnyOrigin],
    learning_rate: Float32,
    cursor: MutPointer[Float32, MutAnyOrigin],
):
    """`add_model_value_kernel`'s single-dimensional arithmetic over the
    searcher's interleaved (offset, size) partition records: grid y the
    leaf, x strides its rows."""
    var leaf = Int(block_idx.y)
    var offset = Int(partitions.unsafe_load(leaf * PARTITION_RECORD + PART_OFFSET))
    var size = Int(partitions.unsafe_load(leaf * PARTITION_RECORD + PART_SIZE))
    var raw = est.unsafe_load(leaf)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var row = Int(indices.unsafe_load(offset + i))
        cursor.unsafe_store(
            row, identical_mul_add(raw, learning_rate, cursor.unsafe_load(row))
        )
        i += stride


def sym_estimate_leaves_device(
    ctx: DeviceContext,
    objective: Int,
    n_leaves: Int,
    mut partitions: DeviceBuffer[DType.uint32],
    mut part_stats: DeviceBuffer[DType.float32],
    mut indices: DeviceBuffer[DType.uint32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut cursor: DeviceBuffer[DType.float32],
    l2_leaf_reg: Float32,
    learning_rate: Float32,
    sm_count: Int,
    mut pool: SymIterPool,
) raises:
    """The leaves from the searcher's statistics, applied to the cursor,
    and their readback ENQUEUED into `pool.h_est`: the caller's tail drain
    settles it and the model reads the values after."""
    if n_leaves != pool.n_leaves_cap:
        raise Error(
            "sym_estimate_leaves_device: " + String(n_leaves)
            + " leaves for a pool of " + String(pool.n_leaves_cap)
        )
    var hw = Int32(1) if has_weights else Int32(0)
    if objective == OBJECTIVE_LOGLOSS:
        ctx.enqueue_function[
            sym_leaves_from_stats_kernel[OBJECTIVE_LOGLOSS, SYM_LEAF_BLOCK]
        ](
            partitions.unsafe_ptr(), part_stats.unsafe_ptr(),
            indices.unsafe_ptr(), weights.unsafe_ptr(), hw,
            cursor.unsafe_ptr(), l2_leaf_reg, pool.d_est.unsafe_ptr(),
            grid_dim=(n_leaves, 1, 1),
            block_dim=(SYM_LEAF_BLOCK, 1, 1),
        )
    elif objective == OBJECTIVE_RMSE:
        ctx.enqueue_function[
            sym_leaves_from_stats_kernel[OBJECTIVE_RMSE, SYM_LEAF_BLOCK]
        ](
            partitions.unsafe_ptr(), part_stats.unsafe_ptr(),
            indices.unsafe_ptr(), weights.unsafe_ptr(), hw,
            cursor.unsafe_ptr(), l2_leaf_reg, pool.d_est.unsafe_ptr(),
            grid_dim=(n_leaves, 1, 1),
            block_dim=(SYM_LEAF_BLOCK, 1, 1),
        )
    else:
        raise Error(
            "sym_estimate_leaves_device: objective " + String(objective)
            + " has no stats-based leaf (gate with sym_leaf_from_stats_ok)"
        )
    var gx = 2 * sm_count
    if gx < 1:
        gx = 1
    ctx.enqueue_function[sym_add_leaves_kernel](
        partitions.unsafe_ptr(), indices.unsafe_ptr(),
        pool.d_est.unsafe_ptr(), learning_rate, cursor.unsafe_ptr(),
        grid_dim=(gx, n_leaves, 1),
        block_dim=(SYM_APPLY_BLOCK, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=pool.h_est, src_buf=pool.d_est)


def sym_std_dev_enqueue(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    count: Int,
    stat_line_size: Int,
    sm_count: Int,
    mut pool: SymIterPool,
) raises:
    """`compute_std_dev` up to its drain, on the pool's buffers: the same
    two launches, the readback enqueued into `pool.h_sd`."""
    if count <= 0:
        return
    var n_blocks = std_dev_blocks(count, sm_count)
    ctx.enqueue_function[std_dev_partials_kernel](
        stats.unsafe_ptr(),
        Int32(count),
        Int32(stat_line_size),
        pool.d_sd_partials.unsafe_ptr(),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(STD_DEV_BLOCK, 1, 1),
    )
    ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
        pool.d_sd_partials.unsafe_ptr(), Int32(n_blocks),
        pool.d_sd_out.unsafe_ptr(),
        grid_dim=1, block_dim=256,
    )
    ctx.enqueue_copy(dst_buf=pool.h_sd, src_buf=pool.d_sd_out)


def sym_score_std_dev_collect(
    model_length_mult: Float64,
    random_strength: Float64,
    count: Int,
    pool: SymIterPool,
) -> Float64:
    """`compute_score_std_dev`'s host half over the settled `pool.h_sd`:
    the same product guard, the same multiplication order."""
    if count <= 0:
        return 0.0
    if model_length_mult * random_strength != 0.0:
        var sum2 = Float64(pool.h_sd[0])
        var std_dev = sqrt(sum2 / (Float64(count) + 1e-100))
        return model_length_mult * std_dev * random_strength
    return 0.0
