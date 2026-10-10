# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Leaf estimation for the FAST tier on Apple (lane/apple-fast-sym-est).

Four candidate experiments for `_estimate_and_apply`
(`gbdt/methods/doc_parallel_boosting.mojo`), each behind its own `-D`
define, every one compiled ONLY under `FAST + Apple`; IDENTICAL and every
other vendor compile main's code unchanged. All default OFF. They restate the
one-task estimation of a single-dimensional pointwise loss (the board's
Logloss cells; RMSE at one permutation never reaches the estimator,
DEVIATION 64) with the Newton or Gradient walker; Exact, the multiclass
family and the ranking targets keep main's path.

  MOJOLEARN_EST_STATS_FUSED   the per-leaf fold, the Newton step and
                              `RegularizeImpl` run on the device
                              (`est_walk_kernel`), so the task has ONE drain
                              (its tail) instead of two, no mid-task
                              readback, no host leaf arithmetic and no
                              `d_est` upload.
  MOJOLEARN_EST_REUSE_PART    (deleted 2026-10-09, DROPPED-BUG; see
                              docs/TOMBSTONES.md)
  MOJOLEARN_EST_ITERS_DEVICE  `leaf_estimation_iterations > 1`: the whole
                              Newton walk (AnyImprovement line search) runs
                              on the device, one `est_walk_kernel` launch per
                              evaluation deciding accept / halve / stop; one
                              drain per tree instead of one per evaluation.
  MOJOLEARN_EST_SHRINK_FUSED  the cursor add (`AppendModels`) is fused with
                              the NEXT iteration's derivative pass: one
                              full-row kernel writes the new cursor, the two
                              search planes and the value partials the loop
                              head would otherwise recompute.
  MOJOLEARN_SYM_EST_ALL       EST_STATS_FUSED + EST_ITERS_DEVICE (REUSE_PART
                              and SHRINK_FUSED are recorded DROPs). The FAST
                              + Apple default since 2026-10-04 (rab4-symest);
                              rollback MOJOLEARN_SYM_EST_ALL_OFF.

Apple has no f64 on the device, so the device walker computes in f32 what
the host walker computed in f64 (the Hessian plus lambda, the direction
quotient, the step); the differences are last-bit and FAST is not
identity-bound. The acceptance test compares f32 values exactly as the host
does (their `functionValue` is a float).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import fma
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, identical_mul_add
from gbdt.apple_fast_classical import AFCL_T06
from gbdt.gpu_data.apple_fast_trees_experiments import AFT_G07
from core.identity_trace import IdentityTrace
from gbdt.gpu_util.kernel.transform import (
    launch_gather_planes_with_mask_f32,
    launch_gather_with_mask_f32,
)
from gbdt.gpu_util.partitions_reduce import (
    compute_partition_stats,
    partition_stats_chunks,
)
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)
from gbdt.methods.kernel_add_model_value import (
    add_model_value_kernel,
    fill_bins_from_partition_kernel,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_EXPECTILE,
    OBJECTIVE_HUBER,
    OBJECTIVE_LOGLINQUANTILE,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_LQ,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_MULTICLASS_OVA,
    OBJECTIVE_MULTIRMSE,
    OBJECTIVE_PAIR_LOGIT,
    OBJECTIVE_POISSON,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_QUERY_RMSE,
    OBJECTIVE_RMSE,
    OBJECTIVE_TWEEDIE,
    OBJECTIVE_YETI_RANK,
    cross_entropy_kernel,
    launch_approximate,
    launch_approximate_move_eval,
    pinned_block_sum,
    pointwise_target_kernel,
)


comptime _APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-est@c8518eb52; never built (its slot was killed
#: unstarted), never timed.
#: apple-fast LEDGER 2026-10-03: DROP sym-est-all (-3.5%, auc .980177 ->
#: .980164; it carried EST_REUSE_PART's bug), 2026-10-04 DROP
#: sym-est-all-ord taxi (+0.7%). The umbrella now holds EST_STATS_FUSED +
#: EST_ITERS_DEVICE, the two switches with no verdict.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tag
#: rab4-symest): gbdt-symmetric istella 14721.49 -> 14291.94 ms (-2.9%); auc
#: 0.980132 -> 0.980155, logloss 0.186584 -> 0.186496 (both better). KEEP:
#: the FAST + Apple default since then; rollback -D MOJOLEARN_SYM_EST_ALL_OFF
#: (single defines then select flags again). rab7 per-define row:
#: EST_STATS_FUSED -2.5% (covered by this umbrella).
comptime EST_ALL = _APPLE_FAST and not is_defined["MOJOLEARN_SYM_EST_ALL_OFF"]()
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-est@c8518eb52; never built (its slot was killed
#: unstarted), never timed.
comptime EST_STATS_FUSED = _APPLE_FAST and (
    is_defined["MOJOLEARN_EST_STATS_FUSED"]() or EST_ALL
)
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-est@c8518eb52; never built (its slot was killed
#: unstarted), never timed. (no EXPERIMENTS row at the source; added here)
comptime EST_ITERS_DEVICE = _APPLE_FAST and (
    is_defined["MOJOLEARN_EST_ITERS_DEVICE"]() or EST_ALL
)
# TOMBSTONE: MOJOLEARN_EST_REUSE_PART (DROPPED-BUG: istella auc .980 -> .930, logloss .186 -> 2.15) deleted 2026-10-09 on
# lane/owed-deletions-D1 (row-order evaluation over the searcher's partition); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_EST_REUSE_PART.patch; record in docs/TOMBSTONES.md.
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-est@c8518eb52; never built (its slot was killed
#: unstarted), never timed. Port: stands down on a tree where
#: SYM_DERIV_FUSED already enqueues the next gradient pass
#: (doc_parallel_boosting `not sym_fuse`).
#: apple-fast LEDGER 2026-10-04: sym-est-sh-1k istella -2.8% (old base, auc
#: up), judged inconclusive and not worth a main verify;
#: recorded loser, OUT of SYM_EST_ALL, not in the A/B table.
comptime EST_SHRINK_FUSED = _APPLE_FAST and (
    is_defined["MOJOLEARN_EST_SHRINK_FUSED"]()
)
comptime EST_APPLE_ANY = (
    EST_STATS_FUSED or EST_ITERS_DEVICE or EST_SHRINK_FUSED
)

#: one block, one thread per leaf (looping past 256 leaves)
# G07: four Apple SIMD groups rather than eight reduce objective partials
# and traverse every leaf. Shared partition-statistics/fused block contracts
# remain untouched. Uncompiled/unverified/unmeasured; changed FAST rounding
# can affect a line-search acceptance boundary, so weighted/multi-iteration
# quality remains required before any decision.
comptime EST_WALK_BLOCK = 128 if AFT_G07 else 256
#: the walker's never-accepted cap, `iteration < 100` (`descent_helpers.cpp:179`)
comptime EST_WALK_TRY_CAP = 100
#: their `1e-20f` (`descent_helpers.cpp:87`), the f32 literal
comptime EST_EPS = Float32(1e-20)
#: `MinLeafWeight`, `leaves_estimation_config.h:60`
comptime EST_MIN_LEAF_WEIGHT = Float32(1e-20)


@always_inline
def _afcl_leaf_stats_sm(sm: Int) -> Int:
    # Scheduling proxy only: halve the partial-grid budget, not the device's
    # reported hardware. Allocation and both statistic launchers use this.
    comptime if AFCL_T06:
        return max(1, (sm + 1) // 2)
    return sm


def apple_est_handles(
    objective: Int,
    estimation_method: Int,
    approx_dim: Int,
    iterations: Int,
    has_group: Bool,
) -> Bool:
    """Whether `apple_fast_estimate_and_apply` takes this task. False in
    every build without one of the defines, so main's path runs."""
    comptime if not EST_APPLE_ANY:
        return False
    if approx_dim != 1 or has_group or iterations < 1:
        return False
    if (
        objective == OBJECTIVE_MULTICLASS
        or objective == OBJECTIVE_MULTICLASS_OVA
        or objective == OBJECTIVE_MULTIRMSE
        or objective == OBJECTIVE_PAIR_LOGIT
        or objective == OBJECTIVE_YETI_RANK
        or objective == OBJECTIVE_QUERY_RMSE
    ):
        return False
    if (
        estimation_method != LEAF_ESTIMATION_NEWTON
        and estimation_method != LEAF_ESTIMATION_GRADIENT
    ):
        return False
    if iterations == 1:
        return EST_STATS_FUSED or EST_SHRINK_FUSED
    return EST_ITERS_DEVICE


def apple_est_fuses_derivs(
    objective: Int,
    estimation_method: Int,
    approx_dim: Int,
    iterations: Int,
    has_group: Bool,
) -> Bool:
    """Whether the task leaves the NEXT iteration's search planes and value
    partials in the hook's buffers (`EST_SHRINK_FUSED`), so the loop head
    skips its derivative pass. The call site and the task consult the same
    predicate."""
    return EST_SHRINK_FUSED and apple_est_handles(
        objective, estimation_method, approx_dim, iterations, has_group
    )


struct EstDerivsHook(Movable):
    """The loop head's derivative pass, as the task restates it
    (`EST_SHRINK_FUSED`): handle views of the fit's `stats`, `fv_part` and
    `mag_part`, and the two flags its launch takes."""

    var stats: DeviceBuffer[DType.float32]
    var fv_part: DeviceBuffer[DType.float32]
    var mag_part: DeviceBuffer[DType.float32]
    var compute_magnitudes: Bool
    var second_order: Bool

    def __init__(
        out self,
        var stats: DeviceBuffer[DType.float32],
        var fv_part: DeviceBuffer[DType.float32],
        var mag_part: DeviceBuffer[DType.float32],
        compute_magnitudes: Bool,
        second_order: Bool,
    ):
        self.stats = stats^
        self.fv_part = fv_part^
        self.mag_part = mag_part^
        self.compute_magnitudes = compute_magnitudes
        self.second_order = second_order


struct AppleEstScratch(Movable):
    """The task's device and host buffers, owned by the fit (pool of one,
    keyed by `n_rows` and the device's core count; `leaves_cap` grows).
    Every cell is written before it is read in each task."""

    var n_rows: Int
    var leaves_cap: Int
    var sm: Int
    var fv_blocks: Int
    #: row -> leaf (row order) or the bins of the gathered order
    var d_bins: DeviceBuffer[DType.uint32]
    var d_eval_stats: DeviceBuffer[DType.float32]
    var d_fv: DeviceBuffer[DType.float32]
    var d_mag_dummy: DeviceBuffer[DType.float32]
    var d_leaves: DeviceBuffer[DType.uint32]
    var h_leaves: HostBuffer[DType.uint32]
    var d_partials: DeviceBuffer[DType.float32]
    var d_part_stats: DeviceBuffer[DType.float32]
    var d_wsum_stats: DeviceBuffer[DType.float32]
    #: the walk: 6 leaf-sized rows (cur_point, cursor_point, gradient,
    #: hessian, direction, weight) then two scalar slots of (value, step)
    var d_walk: DeviceBuffer[DType.float32]
    #: two scalar slots of (iteration, updated, finished, pad)
    var d_walk_i: DeviceBuffer[DType.int32]
    var h_walk_i: HostBuffer[DType.int32]
    var d_shift: DeviceBuffer[DType.float32]
    var d_est: DeviceBuffer[DType.float32]
    var h_est: HostBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, n_rows: Int, leaves_cap: Int, sm: Int
    ) raises:
        self.n_rows = n_rows
        self.leaves_cap = leaves_cap
        self.sm = sm
        self.fv_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
        self.d_bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.d_eval_stats = ctx.enqueue_create_buffer[DType.float32](
            2 * n_rows
        )
        self.d_fv = ctx.enqueue_create_buffer[DType.float32](self.fv_blocks)
        self.d_mag_dummy = ctx.enqueue_create_buffer[DType.float32](2)
        self.d_leaves = ctx.enqueue_create_buffer[DType.uint32](leaves_cap)
        self.h_leaves = ctx.enqueue_create_host_buffer[DType.uint32](
            leaves_cap
        )
        # the two-stat fold and the one-stat weight fold size their partials
        # from `partition_stats_chunks`, which is per stat count
        var two = 2 * partition_stats_chunks(_afcl_leaf_stats_sm(sm), 2)
        var one = partition_stats_chunks(_afcl_leaf_stats_sm(sm), 1)
        var per_leaf = two if two > one else one
        self.d_partials = ctx.enqueue_create_buffer[DType.float32](
            leaves_cap * per_leaf
        )
        self.d_part_stats = ctx.enqueue_create_buffer[DType.float32](
            2 * leaves_cap
        )
        self.d_wsum_stats = ctx.enqueue_create_buffer[DType.float32](
            2 * leaves_cap
        )
        self.d_walk = ctx.enqueue_create_buffer[DType.float32](
            6 * leaves_cap + 4
        )
        self.d_walk_i = ctx.enqueue_create_buffer[DType.int32](8)
        self.h_walk_i = ctx.enqueue_create_host_buffer[DType.int32](8)
        self.d_shift = ctx.enqueue_create_buffer[DType.float32](leaves_cap)
        self.d_est = ctx.enqueue_create_buffer[DType.float32](leaves_cap)
        self.h_est = ctx.enqueue_create_host_buffer[DType.float32](leaves_cap)
        # the identity leaf ids the folds read; the staging is a field, so
        # it outlives its copy
        for i in range(leaves_cap):
            self.h_leaves.unsafe_ptr().unsafe_store(i, UInt32(i))
        ctx.enqueue_copy(
            dst_buf=self.d_leaves, src_ptr=self.h_leaves.unsafe_ptr()
        )

    def handles(self) -> AppleEstScratch:
        """Handle copies onto the same memory."""
        return AppleEstScratch(
            self.n_rows, self.leaves_cap, self.sm, self.fv_blocks,
            self.d_bins.copy(), self.d_eval_stats.copy(), self.d_fv.copy(),
            self.d_mag_dummy.copy(), self.d_leaves.copy(),
            self.h_leaves.copy(), self.d_partials.copy(),
            self.d_part_stats.copy(), self.d_wsum_stats.copy(),
            self.d_walk.copy(), self.d_walk_i.copy(), self.h_walk_i.copy(),
            self.d_shift.copy(), self.d_est.copy(), self.h_est.copy(),
        )

    def __init__(
        out self,
        n_rows: Int,
        leaves_cap: Int,
        sm: Int,
        fv_blocks: Int,
        var d_bins: DeviceBuffer[DType.uint32],
        var d_eval_stats: DeviceBuffer[DType.float32],
        var d_fv: DeviceBuffer[DType.float32],
        var d_mag_dummy: DeviceBuffer[DType.float32],
        var d_leaves: DeviceBuffer[DType.uint32],
        var h_leaves: HostBuffer[DType.uint32],
        var d_partials: DeviceBuffer[DType.float32],
        var d_part_stats: DeviceBuffer[DType.float32],
        var d_wsum_stats: DeviceBuffer[DType.float32],
        var d_walk: DeviceBuffer[DType.float32],
        var d_walk_i: DeviceBuffer[DType.int32],
        var h_walk_i: HostBuffer[DType.int32],
        var d_shift: DeviceBuffer[DType.float32],
        var d_est: DeviceBuffer[DType.float32],
        var h_est: HostBuffer[DType.float32],
    ):
        self.n_rows = n_rows
        self.leaves_cap = leaves_cap
        self.sm = sm
        self.fv_blocks = fv_blocks
        self.d_bins = d_bins^
        self.d_eval_stats = d_eval_stats^
        self.d_fv = d_fv^
        self.d_mag_dummy = d_mag_dummy^
        self.d_leaves = d_leaves^
        self.h_leaves = h_leaves^
        self.d_partials = d_partials^
        self.d_part_stats = d_part_stats^
        self.d_wsum_stats = d_wsum_stats^
        self.d_walk = d_walk^
        self.d_walk_i = d_walk_i^
        self.h_walk_i = h_walk_i^
        self.d_shift = d_shift^
        self.d_est = d_est^
        self.h_est = h_est^


def apple_est_ensure(
    ctx: DeviceContext,
    mut pool: List[AppleEstScratch],
    n_rows: Int,
    n_leaves: Int,
    sm: Int,
) raises:
    """`pool[0]` fits this task on return: same `n_rows` and core count,
    `leaves_cap >= n_leaves`. A miss replaces the entry."""
    if (
        len(pool) == 0
        or pool[0].n_rows != n_rows
        or pool[0].sm != sm
        or pool[0].leaves_cap < n_leaves
    ):
        pool.clear()
        pool.append(AppleEstScratch(ctx, n_rows, n_leaves, sm))


# ------------------------------------------------------------------ kernels


def est_leaf_of_row_kernel(
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    leaf_of_row: MutPointer[UInt32, MutAnyOrigin],
):
    """`leaf_of_row[row_index[offset + i]] = leaf`: the partition read as a
    row -> leaf map. Grid y is the leaf, x strides its rows (the
    `fill_bins_from_partition_kernel` shape). Integer only."""
    var leaf = Int(block_idx.y)
    var offset = Int(part_offset.unsafe_load(leaf))
    var size = Int(part_size.unsafe_load(leaf))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        leaf_of_row.unsafe_store(
            Int(row_index.unsafe_load(offset + i)), UInt32(leaf)
        )
        i += stride


def est_walk_kernel(
    part_stats: MutPointer[Float32, MutAnyOrigin],
    wsum_stats: MutPointer[Float32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    fv: MutPointer[Float32, MutAnyOrigin],
    fv_blocks_in: Int32,
    n_leaves_in: Int32,
    cap_in: Int32,
    use_wsum_stats: Int32,
    newton: Int32,
    lambda_reg: Float32,
    iterations_in: Int32,
    phase_in: Int32,
    slot_in: Int32,
    slot_out: Int32,
    walk: MutPointer[Float32, MutAnyOrigin],
    walk_i: MutPointer[Int32, MutAnyOrigin],
    shift: MutPointer[Float32, MutAnyOrigin],
    est: MutPointer[Float32, MutAnyOrigin],
):
    """`TNewtonLikeWalker::Estimate` (`descent_helpers.cpp:128-204`), one
    evaluation's worth of walking, on the device. ONE block of
    `EST_WALK_BLOCK` threads, a thread per leaf (looping past the block).

    `phase_in < 0`: the start point's evaluation just ran (zero point).
    Read gradient / Hessian from `part_stats` (`[leaf][2]`: der, der2), the
    leaf weight from `wsum_stats` or the leaf size, form the direction
    `g / (h + 1e-20f)` where `h > 0` else 0 (Newton: `h = der2 + lambda`;
    Gradient: `h = weight + lambda`), and either finish at one iteration
    (one full step from zero, `RegularizeImpl`, `est`) or write the first
    try's shift (`next - cursor_point`) and start the line search.

    `phase_in >= 0`: a try's evaluation just ran. AnyImprovement
    (`step_estimator.cpp`: `FunctionValue <= next`, f32) accepts the
    point in the cursor (gradient / Hessian / direction from its stats,
    `iteration += 1`, `step = 1`) or halves the step (`iteration += 1`);
    then either the next try's shift is written, or the walk is finished
    at the current point (`est`), as the walker's two loop tests say.

    Scalar state is double-buffered by `slot_in` / `slot_out` so every
    thread reads one slot and thread 0 writes the other. `finished` makes
    every later launch a no-op (shift stays 0). `est` holds 0 until the
    walk is finished, so the enqueued cursor add adds nothing early."""
    var tid = Int(thread_idx.x)
    var cap = Int(cap_in)
    var n_leaves = Int(n_leaves_in)
    var fv_blocks = Int(fv_blocks_in)
    var phase = Int(phase_in)
    var s_in = Int(slot_in)
    var s_out = Int(slot_out)

    # the value: their f32 `functionValue`, the per-block partials folded
    var acc = Float32(0.0)
    var b = tid
    while b < fv_blocks:
        acc += fv.unsafe_load(b)
        b += EST_WALK_BLOCK
    var total = pinned_block_sum[EST_WALK_BLOCK](acc)
    # thread 0's return is the meaningful one: broadcast through a 4-byte
    # shared slab (fits on every Apple part), the one barrier of the kernel
    var slab = stack_allocation[
        1, Float32, address_space=AddressSpace.SHARED
    ]()
    if tid == 0:
        slab[unsafe_offset=0] = total
    barrier()
    var fv_total = slab[unsafe_offset=0]

    var iteration = 0
    var updated = 0
    var finished = 0
    var cur_value = Float32(0.0)
    var step = Float32(1.0)
    if phase >= 0:
        iteration = Int(walk_i.unsafe_load(4 * s_in))
        updated = Int(walk_i.unsafe_load(4 * s_in + 1))
        finished = Int(walk_i.unsafe_load(4 * s_in + 2))
        cur_value = walk.unsafe_load(6 * cap + 2 * s_in)
        step = walk.unsafe_load(6 * cap + 2 * s_in + 1)
    if finished != 0:
        if tid == 0:
            walk_i.unsafe_store(4 * s_out, Int32(iteration))
            walk_i.unsafe_store(4 * s_out + 1, Int32(updated))
            walk_i.unsafe_store(4 * s_out + 2, Int32(1))
            walk.unsafe_store(6 * cap + 2 * s_out, cur_value)
            walk.unsafe_store(6 * cap + 2 * s_out + 1, step)
        return

    # the decision, uniform across the block
    var iterations = Int(iterations_in)
    var accept = False
    var cont = False
    var one_step = False
    var new_value = cur_value
    var new_step = step
    var new_iter = iteration
    var new_upd = updated
    if phase < 0:
        new_value = fv_total
        if iterations == 1:
            # `:151-156`: one full step, regularize, no re-evaluation
            one_step = True
        else:
            new_step = Float32(1.0)
            cont = True
    else:
        accept = cur_value <= fv_total
        if accept:
            new_value = fv_total
            new_iter = iteration + 1
            new_upd = 1
            new_step = Float32(1.0)
            cont = new_iter < iterations
        else:
            new_iter = iteration + 1
            new_step = step / Float32(2.0)
            cont = new_iter < iterations or (
                new_upd == 0 and new_iter < EST_WALK_TRY_CAP
            )
    var new_fin = 0 if cont else 1

    var leaf = tid
    while leaf < n_leaves:
        if phase < 0 or accept:
            var g = part_stats.unsafe_load(2 * leaf)
            var h2 = part_stats.unsafe_load(2 * leaf + 1)
            var w: Float32
            if phase < 0:
                if use_wsum_stats != Int32(0):
                    w = wsum_stats.unsafe_load(leaf)
                else:
                    w = Float32(Int(part_size.unsafe_load(leaf)))
                walk.unsafe_store(5 * cap + leaf, w)
                walk.unsafe_store(leaf, Float32(0.0))
                walk.unsafe_store(cap + leaf, Float32(0.0))
            else:
                w = walk.unsafe_load(5 * cap + leaf)
                # the accepted point is the one in the cursor
                walk.unsafe_store(leaf, walk.unsafe_load(cap + leaf))
            # `(*Der2AtPoint)[i] += lambda` (`pointwise_oracle.cpp:86-89`);
            # Gradient: `WeightsCpu[bin] + lambda` (`:185-193`)
            var hess = (h2 if newton != Int32(0) else w) + lambda_reg
            var direction = Float32(0.0)
            if hess > Float32(0.0):
                direction = g / (hess + EST_EPS)
            walk.unsafe_store(2 * cap + leaf, g)
            walk.unsafe_store(3 * cap + leaf, hess)
            walk.unsafe_store(4 * cap + leaf, direction)
        var cur = walk.unsafe_load(leaf)
        var dir = walk.unsafe_load(4 * cap + leaf)
        var wl = walk.unsafe_load(5 * cap + leaf)
        var cp = walk.unsafe_load(cap + leaf)
        if one_step:
            # `MoveInOptimalDirection` at step 1, then `RegularizeImpl`
            var r = fma(Float32(1.0), dir, cur)
            if wl < EST_MIN_LEAF_WEIGHT:
                r = Float32(0.0)
            est.unsafe_store(leaf, r)
            shift.unsafe_store(leaf, Float32(0.0))
        elif new_fin != 0:
            # `:204`: the current point, regularized before its acceptance
            est.unsafe_store(leaf, cur)
            shift.unsafe_store(leaf, Float32(0.0))
        else:
            var nxt = fma(new_step, dir, cur)
            if wl < EST_MIN_LEAF_WEIGHT:
                nxt = Float32(0.0)
            shift.unsafe_store(leaf, nxt - cp)
            walk.unsafe_store(cap + leaf, nxt)
            est.unsafe_store(leaf, Float32(0.0))
        leaf += EST_WALK_BLOCK

    if tid == 0:
        if one_step:
            new_fin = 1
        walk_i.unsafe_store(4 * s_out, Int32(new_iter))
        walk_i.unsafe_store(4 * s_out + 1, Int32(new_upd))
        walk_i.unsafe_store(4 * s_out + 2, Int32(new_fin))
        walk.unsafe_store(6 * cap + 2 * s_out, new_value)
        walk.unsafe_store(6 * cap + 2 * s_out + 1, new_step)


# TOMBSTONE: MOJOLEARN_EST_REUSE_PART (DROPPED-BUG: istella auc .980 -> .930, logloss .186 -> 2.15) deleted 2026-10-09 on
# lane/owed-deletions-D1 (row-order evaluation over the searcher's partition); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_EST_REUSE_PART.patch; record in docs/TOMBSTONES.md.


def est_apply_derivs_xent_kernel[
    has_border: Bool,
    second_der_as_weights: Bool,
](
    leaf_of_row: MutPointer[UInt32, MutAnyOrigin],
    est: MutPointer[Float32, MutAnyOrigin],
    walk: MutPointer[Float32, MutAnyOrigin],
    cap_in: Int32,
    learning_rate: Float32,
    use_cursor_point: Int32,
    walk_i: MutPointer[Int32, MutAnyOrigin],
    slot_in: Int32,
    target_classes: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    border: Float32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`EST_SHRINK_FUSED`: the cursor add, then `cross_entropy_kernel`'s
    search body (the loop head's launch) on the just-stored value, the
    DEVIATION 2030 shape. The add is `add_model_value_kernel`'s
    `identical_mul_add(raw, rate, cursor)` on a cursor the walk left
    alone, or the fix-up form on one it shifted (`use_cursor_point`).
    Skipped (planes still written, overwritten later) while unfinished."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in) and walk_i.unsafe_load(4 * Int(slot_in) + 2) != Int32(0):
        var leaf = Int(leaf_of_row.unsafe_load(i))
        var raw = est.unsafe_load(leaf)
        if use_cursor_point != Int32(0):
            var delta = raw * learning_rate - walk.unsafe_load(
                Int(cap_in) + leaf
            )
            predictions.unsafe_store(i, predictions.unsafe_load(i) + delta)
        else:
            predictions.unsafe_store(
                i, identical_mul_add(raw, learning_rate, predictions.unsafe_load(i))
            )
    cross_entropy_kernel[has_border, False, second_der_as_weights](
        target_classes, weights, size_in, predictions, has_weights,
        border, stats, function_value, compute_fv,
        plane_magnitudes, compute_magnitudes,
    )


def est_apply_derivs_pointwise_kernel[
    objective: Int,
    second_der_as_weights: Bool,
](
    leaf_of_row: MutPointer[UInt32, MutAnyOrigin],
    est: MutPointer[Float32, MutAnyOrigin],
    walk: MutPointer[Float32, MutAnyOrigin],
    cap_in: Int32,
    learning_rate: Float32,
    use_cursor_point: Int32,
    walk_i: MutPointer[Int32, MutAnyOrigin],
    slot_in: Int32,
    relevs: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    alpha: Float32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """The twin of `est_apply_derivs_xent_kernel` for
    `pointwise_target_kernel`'s losses."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in) and walk_i.unsafe_load(4 * Int(slot_in) + 2) != Int32(0):
        var leaf = Int(leaf_of_row.unsafe_load(i))
        var raw = est.unsafe_load(leaf)
        if use_cursor_point != Int32(0):
            var delta = raw * learning_rate - walk.unsafe_load(
                Int(cap_in) + leaf
            )
            predictions.unsafe_store(i, predictions.unsafe_load(i) + delta)
        else:
            predictions.unsafe_store(
                i, identical_mul_add(raw, learning_rate, predictions.unsafe_load(i))
            )
    pointwise_target_kernel[objective, False, second_der_as_weights](
        relevs, weights, size_in, predictions, has_weights, alpha,
        stats, function_value, compute_fv,
        plane_magnitudes, compute_magnitudes,
    )


def _launch_apply_derivs(
    ctx: DeviceContext,
    objective: Int,
    second_order: Bool,
    mut leaf_of_row: DeviceBuffer[DType.uint32],
    mut est: DeviceBuffer[DType.float32],
    mut walk: DeviceBuffer[DType.float32],
    cap: Int,
    learning_rate: Float32,
    use_cursor_point: Int32,
    mut walk_i: DeviceBuffer[DType.int32],
    slot_in: Int32,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
    has_weights: Int32,
    alpha: Float32,
    border: Float32,
    mut stats: DeviceBuffer[DType.float32],
    mut fv_part: DeviceBuffer[DType.float32],
    mut mag_part: DeviceBuffer[DType.float32],
    compute_magnitudes: Int32,
    blocks: Int,
) raises:
    """`launch_approximate[False, second_order]`'s dispatch (the loop
    head's), fused kernels; `UseBorder()` is Logloss, the generic arm in
    `launch_pointwise_target_kernel`'s case order."""

    @parameter
    def _xent[has_border: Bool, so: Bool]() raises:
        ctx.enqueue_function[est_apply_derivs_xent_kernel[has_border, so]](
            leaf_of_row, est, walk, Int32(cap), learning_rate,
            use_cursor_point, walk_i, slot_in,
            targets, weights, Int32(n_rows), cursor, has_weights, border,
            stats, fv_part, Int32(1), mag_part, compute_magnitudes,
            grid_dim=(blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )

    @parameter
    def _go[obj: Int, so: Bool]() raises:
        ctx.enqueue_function[est_apply_derivs_pointwise_kernel[obj, so]](
            leaf_of_row, est, walk, Int32(cap), learning_rate,
            use_cursor_point, walk_i, slot_in,
            targets, weights, Int32(n_rows), cursor, has_weights, alpha,
            stats, fv_part, Int32(1), mag_part, compute_magnitudes,
            grid_dim=(blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )

    @parameter
    def _dispatch[so: Bool]() raises:
        if objective == OBJECTIVE_LOGLOSS:
            _xent[True, so]()
        elif objective == OBJECTIVE_CROSSENTROPY:
            _xent[False, so]()
        elif objective == OBJECTIVE_EXPECTILE:
            _go[OBJECTIVE_EXPECTILE, so]()
        elif objective == OBJECTIVE_QUANTILE:
            _go[OBJECTIVE_QUANTILE, so]()
        elif objective == OBJECTIVE_MAE:
            _go[OBJECTIVE_MAE, so]()
        elif objective == OBJECTIVE_LOGLINQUANTILE:
            _go[OBJECTIVE_LOGLINQUANTILE, so]()
        elif objective == OBJECTIVE_MAPE:
            _go[OBJECTIVE_MAPE, so]()
        elif objective == OBJECTIVE_POISSON:
            _go[OBJECTIVE_POISSON, so]()
        elif objective == OBJECTIVE_LQ:
            _go[OBJECTIVE_LQ, so]()
        elif objective == OBJECTIVE_RMSE:
            _go[OBJECTIVE_RMSE, so]()
        elif objective == OBJECTIVE_TWEEDIE:
            _go[OBJECTIVE_TWEEDIE, so]()
        elif objective == OBJECTIVE_HUBER:
            _go[OBJECTIVE_HUBER, so]()
        else:
            raise Error(
                "apple_fast_est: objective " + String(objective)
                + " does not reach the fused derivative pass"
            )

    if second_order:
        _dispatch[True]()
    else:
        _dispatch[False]()


# --------------------------------------------------------------- the task


def apple_fast_estimate_and_apply(
    ctx: DeviceContext,
    n_rows: Int,
    n_leaves: Int,
    sizes: List[Int],
    leaf_offsets: List[Int],
    mut row_index: DeviceBuffer[DType.uint32],
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut cursor: DeviceBuffer[DType.float32],
    objective: Int,
    alpha: Float32,
    logloss_border: Float32,
    l2_leaf_reg: Float32,
    est_sm: Int,
    estimation_method: Int,
    iterations: Int,
    learning_rate: Float32,
    mut leaf_values: List[Float32],
    mut trace: IdentityTrace,
    mut stage_times: StageTimes,
    leaf_tag: String,
    mut g_target: DeviceBuffer[DType.float32],
    mut g_weights: DeviceBuffer[DType.float32],
    mut g_cursor: DeviceBuffer[DType.float32],
    mut d_p_off: DeviceBuffer[DType.uint32],
    mut d_p_sz: DeviceBuffer[DType.uint32],
    mut h_po: HostBuffer[DType.uint32],
    mut h_ps: HostBuffer[DType.uint32],
    mut s: AppleEstScratch,
    var hook: Optional[EstDerivsHook],
) raises:
    """`_estimate_and_apply` for a task `apple_est_handles` accepts: the
    same estimate (`TNewtonLikeWalker::Estimate`, then `AppendModels`) with
    the device doing the folding and the walking.

    The partition arrives as the searcher's `row_index` / `sizes` /
    `leaf_offsets`; the live `targets`, `weights` and `cursor` are gathered
    into bin order (`g_*`) as main does. One drain per tree
    in the common case; a walk that has accepted nothing after
    `iterations` tries continues in device chunks, one drain each, up to
    the walker's cap of 100 tries."""
    var fuse_derivs = EST_SHRINK_FUSED and hook.__bool__()
    var newton = estimation_method == LEAF_ESTIMATION_NEWTON
    var sm = est_sm
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var mse_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    if mse_blocks < 1:
        mse_blocks = 1
    var hw = Int32(1) if has_weights else Int32(0)
    var max_leaf = 1
    for i in range(n_leaves):
        if sizes[i] > max_leaf:
            max_leaf = sizes[i]
    var gx = 2 * sm
    if gx < 1:
        gx = 1

    stage_times.begin(ctx)
    # the device's own offsets (bit-reversed leaf order; see
    # `_estimate_and_apply`), staged in the workspace's buffers
    for i in range(n_leaves):
        h_po.unsafe_ptr().unsafe_store(i, UInt32(leaf_offsets[i]))
        h_ps.unsafe_ptr().unsafe_store(i, UInt32(sizes[i]))
    ctx.enqueue_copy(dst_buf=d_p_off, src_ptr=h_po.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p_sz, src_ptr=h_ps.unsafe_ptr())

    # the shift needs the bins of the gathered order
    # TOMBSTONE: MOJOLEARN_EST_REUSE_PART (DROPPED-BUG) deleted 2026-10-09 on lane/owed-deletions-D1 (its row-order
    # branches in this task); code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_EST_REUSE_PART.patch.
    var need_shift = iterations > 1
    launch_gather_with_mask_f32(
        ctx, g_target, targets, row_index, n_rows, UInt32(0xFFFFFFFF),
    )
    if has_weights:
        launch_gather_with_mask_f32(
            ctx, g_weights, weights, row_index, n_rows,
            UInt32(0xFFFFFFFF),
        )
    launch_gather_planes_with_mask_f32(
        ctx, g_cursor, cursor, row_index, n_rows,
        UInt32(0xFFFFFFFF), 1, n_rows,
    )
    if need_shift:
        ctx.enqueue_function[fill_bins_from_partition_kernel](
            d_p_off.unsafe_ptr(), d_p_sz.unsafe_ptr(),
            s.d_bins.unsafe_ptr(),
            grid_dim=(gx, n_leaves, 1),
            block_dim=(256, 1, 1),
        )

    # `WeightsCpu` (`pointwise_oracle.cpp:236-243`): the weighted arm's
    # per-leaf weight fold, for `RegularizeImpl` and the Gradient Hessian
    if has_weights:
        compute_partition_stats(
            ctx, n_leaves, 0, 1, n_rows,
            s.d_leaves, d_p_off, d_p_sz, g_weights,
            s.d_partials, s.d_wsum_stats,
            sm_count=_afcl_leaf_stats_sm(sm), row_bound=max_leaf,
        )
    stage_times.end(ctx, "est.stage_in")

    @parameter
    def _reduce() raises:
        compute_partition_stats(
            ctx, n_leaves, 0, 2, n_rows,
            s.d_leaves, d_p_off, d_p_sz, s.d_eval_stats,
            s.d_partials, s.d_part_stats,
            sm_count=_afcl_leaf_stats_sm(sm), row_bound=max_leaf,
        )

    @parameter
    def _walk(phase: Int, slot_in: Int, slot_out: Int) raises:
        ctx.enqueue_function[est_walk_kernel](
            s.d_part_stats.unsafe_ptr(), s.d_wsum_stats.unsafe_ptr(),
            d_p_sz.unsafe_ptr(), s.d_fv.unsafe_ptr(),
            Int32(mse_blocks), Int32(n_leaves), Int32(s.leaves_cap),
            hw, Int32(1) if newton else Int32(0),
            l2_leaf_reg, Int32(iterations),
            Int32(phase), Int32(slot_in), Int32(slot_out),
            s.d_walk.unsafe_ptr(), s.d_walk_i.unsafe_ptr(),
            s.d_shift.unsafe_ptr(), s.d_est.unsafe_ptr(),
            grid_dim=(1, 1, 1),
            block_dim=(EST_WALK_BLOCK, 1, 1),
        )

    @parameter
    def _try(t: Int) raises:
        # `MoveTo` + `ApproximateAt` fused (DEVIATION 2030's kernels): the
        # shift the previous walk launch wrote, applied by the thread that
        # evaluates the row
        launch_approximate_move_eval[True](
            ctx, objective, s.d_shift, s.d_bins,
            g_target, g_weights, Int32(n_rows), g_cursor, hw,
            alpha, logloss_border,
            s.d_eval_stats, s.d_fv, Int32(1),
            s.d_mag_dummy, Int32(0),
            mse_blocks,
        )
        _reduce()
        _walk(t, t % 2, (t + 1) % 2)

    @parameter
    def _apply(slot: Int) raises:
        # `AppendModels`: the rescaled estimate onto the real cursor
        if fuse_derivs:
            ref h = hook.value()
            # the bins buffer is free once the walk is done: the row ->
            # leaf map goes there now (stream order covers the walk)
            ctx.enqueue_function[est_leaf_of_row_kernel](
                d_p_off.unsafe_ptr(), d_p_sz.unsafe_ptr(),
                row_index.unsafe_ptr(), s.d_bins.unsafe_ptr(),
                grid_dim=(gx, n_leaves, 1),
                block_dim=(256, 1, 1),
            )
            _launch_apply_derivs(
                ctx, objective, h.second_order,
                s.d_bins, s.d_est, s.d_walk, s.leaves_cap, learning_rate,
                Int32(0),
                s.d_walk_i, Int32(slot),
                targets, weights, n_rows, cursor, hw, alpha, logloss_border,
                h.stats, h.fv_part, h.mag_part,
                Int32(1) if h.compute_magnitudes else Int32(0),
                mse_blocks,
            )
            return
        # `est` is 0 while the walk is unfinished, so this adds nothing
        # early; the gathered copy took the shifts, the live cursor
        # takes the estimate
        ctx.enqueue_function[add_model_value_kernel](
            d_p_off.unsafe_ptr(), d_p_sz.unsafe_ptr(),
            row_index.unsafe_ptr(), s.d_est.unsafe_ptr(),
            learning_rate, cursor.unsafe_ptr(),
            Int32(1), Int32(n_rows),
            grid_dim=(gx, n_leaves, 1),
            block_dim=(256, 1, 1),
        )

    # the start point's evaluation (zero point, no shift); the value is
    # read only by a line search
    stage_times.begin(ctx)
    var fv_flag = Int32(1) if iterations > 1 else Int32(0)
    launch_approximate[True](
        ctx, objective, g_target, g_weights, Int32(n_rows), g_cursor, hw,
        alpha, logloss_border,
        s.d_eval_stats, s.d_fv, fv_flag,
        s.d_mag_dummy, Int32(0),
        mse_blocks,
    )
    _reduce()
    _walk(-1, 0, 0)
    var t = 0
    if iterations > 1:
        for _ in range(iterations):
            _try(t)
            t += 1
    var slot = t % 2
    _apply(slot)
    # the one drain of the task: the estimate and the walk state ride it
    ctx.enqueue_copy(dst_ptr=s.h_est.unsafe_ptr(), src_buf=s.d_est)
    ctx.enqueue_copy(dst_ptr=s.h_walk_i.unsafe_ptr(), src_buf=s.d_walk_i)
    ctx.synchronize()
    stage_times.end(ctx, "est.walk")

    # the walker's never-accepted corner: `iteration < 100` keeps halving
    # until a step is accepted; continue in device chunks, one drain each
    if iterations > 1:
        while s.h_walk_i.unsafe_ptr().unsafe_load(4 * slot + 2) == Int32(0):
            var iteration = Int(s.h_walk_i.unsafe_ptr().unsafe_load(4 * slot))
            var k = EST_WALK_TRY_CAP - iteration
            if k > iterations:
                k = iterations
            if k < 1:
                k = 1
            for _ in range(k):
                _try(t)
                t += 1
            slot = t % 2
            _apply(slot)
            ctx.enqueue_copy(dst_ptr=s.h_est.unsafe_ptr(), src_buf=s.d_est)
            ctx.enqueue_copy(
                dst_ptr=s.h_walk_i.unsafe_ptr(), src_buf=s.d_walk_i
            )
            ctx.synchronize()

    leaf_values.clear()
    for i in range(n_leaves):
        leaf_values.append(s.h_est.unsafe_ptr().unsafe_load(i))
    trace.record_list_f32(leaf_tag, leaf_values)
    _ = hook.__bool__()
