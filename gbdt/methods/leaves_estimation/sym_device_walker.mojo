# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-trees-symmetric: the Newton walker's leaf walk on the
device (`-D MOJOLEARN_SYM_DEVICE_LEAVES`, FAST + Apple only).

`newton_like_walker_estimate` (descent_helpers.mojo) drains the queue
once per evaluation: `write_value_and_first_derivatives` reads the
per-leaf (der, der2) sums and the function value back, the host forms
the direction, the step and the backtracking test, and `move_to` uploads
the shift. A Logloss tree at `leaf_estimation_iterations = 10` is up to
eleven drains; at ~0.2 ms per launch-plus-wait on Apple that is the
1000-iteration lane's largest per-tree host term after the partition.

Here the walk is enqueued whole: per evaluation the cursor shift
(`add_bin_model_value_kernel`), the row evaluation (`launch_approximate`,
value partials and per-row der/der2), the per-leaf fold
(`compute_partition_stats`) and ONE decide kernel -- a single block,
`SYM_WALK_BLOCK` threads over the `2^depth` leaves (a fixed, small size)
-- that folds the value partials, applies the backtracking test, takes
the Newton direction, the step and the next proposal, and writes the
cursor shift the next evaluation's launch reads. The host reads the
estimate back once, with `_estimate_and_apply`'s tail, after the
`add_model_value` launch that consumes it.

THE SAME NEWTON STATEMENTS, in float32 (Apple GPUs have no float64; the
host walker's Float64 gradient, Hessian and value become float32 here,
which is a FAST-only bit change):
  direction[leaf] = der2 > 0 ? grad / (der2 + 1e-20) : 0,
                    der2 = sum(w * d2) + lambda                (`_diagonal_direction`)
  proposal        = fma(step, direction, cur_point), then Regularize
                    (leaf weight < MinLeafWeight -> 0)          (`_move`, `regularize`)
  accept          = cur_value <= next_value                    (ANY_IMPROVEMENT)
                    accepted: cur = proposal, new direction, step = 1
                    rejected: step /= 2
  iterations == 1 : result = Regularize(cur + 1.0 * direction), no second
                    evaluation (the walker's early return).

FIXED TRIP COUNT: `1 + iterations` evaluations for `iterations > 1`
(the walker's initial evaluation plus the `iterations` it counts), one
for `iterations == 1`. The host walker keeps halving past `iterations`
evaluations, up to 100, when nothing was accepted yet; this walk stops
at `iterations` and returns the start point in that case, as the host
walker does when the 100 are exhausted too. Diagonal Hessian only
(every pointwise single-dimensional loss), no sample weights (the
Regularize weights are the leaf sizes, `d_p_sz`), no querywise /
pairwise / YetiRank oracle, at most `SYM_WALK_BLOCK` leaves; the caller
gates on all of these and falls back to the host walker otherwise.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import fma
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_zero import enqueue_fill
from gbdt.gpu_util.partitions_reduce import compute_partition_stats
from gbdt.methods.kernel_add_model_value import (
    ABMV_BLOCK,
    ABMV_ELEMENTS,
    add_bin_model_value_kernel,
)
from gbdt.methods.leaves_estimation.pointwise_oracle import BinOptimizedOracle
from gbdt.methods.leaves_estimation.step_estimator import (
    BACKTRACKING_ANY_IMPROVEMENT,
    BACKTRACKING_NONE,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    launch_approximate,
)


#: `-D MOJOLEARN_SYM_DEVICE_LEAVES`, FAST + Apple only.
comptime SYM_DEVICE_LEAVES = (
    is_defined["MOJOLEARN_SYM_DEVICE_LEAVES"]()
    and GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
)

#: the decide kernel's block: one thread per leaf, so the walk takes
#: trees of at most this many leaves (depth 8)
comptime SYM_WALK_BLOCK = 256

#: `scalars`: [cur_value, step]
comptime SYM_WALK_SCALARS = 4


struct SymWalkerScratch(Movable):
    """The walk's per-leaf state, a pool of one for the fit
    (`TEstimationWorkspace.sym_walker`): the accepted point, the point
    the cursor currently holds, the direction, and the two scalars."""

    var leaves_cap: Int
    var d_cur_point: DeviceBuffer[DType.float32]
    var d_cursor_point: DeviceBuffer[DType.float32]
    var d_dir: DeviceBuffer[DType.float32]
    var d_scalars: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, leaves_cap: Int) raises:
        if leaves_cap <= 0:
            raise Error("SymWalkerScratch: leaves_cap must be positive")
        self.leaves_cap = leaves_cap
        self.d_cur_point = ctx.enqueue_create_buffer[DType.float32](leaves_cap)
        self.d_cursor_point = ctx.enqueue_create_buffer[DType.float32](
            leaves_cap
        )
        self.d_dir = ctx.enqueue_create_buffer[DType.float32](leaves_cap)
        self.d_scalars = ctx.enqueue_create_buffer[DType.float32](
            SYM_WALK_SCALARS
        )
        ctx.synchronize()


def sym_newton_decide_kernel[
    block_size: Int
](
    fv: MutPointer[Float32, MutAnyOrigin],
    fv_blocks_in: Int32,
    part_stats: MutPointer[Float32, MutAnyOrigin],
    bin_count_in: Int32,
    p_sz: MutPointer[UInt32, MutAnyOrigin],
    lambda_reg: Float32,
    min_leaf_weight: Float32,
    backtracking: Int32,
    eval_index_in: Int32,
    n_evals_in: Int32,
    single_iteration: Int32,
    cur_point: MutPointer[Float32, MutAnyOrigin],
    cursor_point: MutPointer[Float32, MutAnyOrigin],
    direction: MutPointer[Float32, MutAnyOrigin],
    scalars: MutPointer[Float32, MutAnyOrigin],
    shift: MutPointer[Float32, MutAnyOrigin],
    est_out: MutPointer[Float32, MutAnyOrigin],
):
    """One evaluation's host statements, in one block. Phase A folds the
    value partials (`finish_single_dim_evaluation`'s `fv32 +=` loop, as a
    block tree); phase B is one thread per leaf. The scalars are read
    into registers before the first barrier and rewritten by thread 0
    after the last, so no thread reads a slot another is writing."""
    var tid = Int(thread_idx.x)
    var fv_blocks = Int(fv_blocks_in)
    var bin_count = Int(bin_count_in)
    var eval_index = Int(eval_index_in)
    var n_evals = Int(n_evals_in)
    var cur_value0 = scalars.unsafe_load(0)
    var step0 = scalars.unsafe_load(1)

    # ---- phase A: the function value ----
    var acc = Float32(0.0)
    var b = tid
    while b < fv_blocks:
        acc += fv.unsafe_load(b)
        b += block_size
    var s_val = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s_val[unsafe_offset=tid] = acc
    barrier()
    var s = block_size >> 1
    while s > 0:
        if tid < s:
            s_val[unsafe_offset=tid] = (
                s_val[unsafe_offset=tid] + s_val[unsafe_offset=tid + s]
            )
        barrier()
        s >>= 1
    var value = s_val[unsafe_offset=0]

    # ---- the backtracking test (`StepEstimator.is_satisfied`) ----
    var accept = True
    if eval_index != 0:
        if backtracking == Int32(BACKTRACKING_NONE):
            accept = True
        else:
            accept = cur_value0 <= value
    var step = step0
    if accept:
        step = Float32(1.0)
    else:
        step = step0 * Float32(0.5)

    # ---- phase B: one thread per leaf ----
    var leaf = tid
    if leaf < bin_count:
        var grad = part_stats.unsafe_load(2 * leaf)
        var der2 = part_stats.unsafe_load(2 * leaf + 1) + lambda_reg
        var cur = cur_point.unsafe_load(leaf)
        var dir = direction.unsafe_load(leaf)
        if accept:
            if eval_index != 0:
                # the accepted proposal is what the cursor holds
                cur = cursor_point.unsafe_load(leaf)
                cur_point.unsafe_store(leaf, cur)
            # `_diagonal_direction`
            if der2 > Float32(0.0):
                dir = grad / (der2 + Float32(1e-20))
            else:
                dir = Float32(0.0)
            direction.unsafe_store(leaf, dir)
        var small = Float32(p_sz.unsafe_load(leaf)) < min_leaf_weight
        if single_iteration != Int32(0):
            # the walker's `iterations == 1` return: one move, regularized
            var res = fma(Float32(1.0), dir, cur)
            if small:
                res = Float32(0.0)
            est_out.unsafe_store(leaf, res)
        elif eval_index + 1 < n_evals:
            # `_move` + `regularize` + `move_to`'s shift
            var nxt = fma(step, dir, cur)
            if small:
                nxt = Float32(0.0)
            var held = cursor_point.unsafe_load(leaf)
            shift.unsafe_store(leaf, nxt - held)
            cursor_point.unsafe_store(leaf, nxt)
        else:
            # the last evaluation decided: the result is the accepted point
            est_out.unsafe_store(leaf, cur)

    barrier()
    if tid == 0:
        if accept:
            scalars.unsafe_store(0, value)
        scalars.unsafe_store(1, step)


def sym_device_newton_walk(
    ctx: DeviceContext,
    mut oracle: BinOptimizedOracle,
    iterations: Int,
    backtracking_type: Int,
    mut pool: List[SymWalkerScratch],
    mut est_out: DeviceBuffer[DType.float32],
) raises:
    """The whole walk, enqueued; nothing read back. `est_out` (the
    workspace's `d_est`, `bin_count` floats) holds the estimate when the
    queue drains. The caller gates (see the module docstring)."""
    var bin_count = oracle.bin_count
    if bin_count > SYM_WALK_BLOCK:
        raise Error(
            "sym_device_newton_walk: " + String(bin_count)
            + " leaves exceed the decide block of " + String(SYM_WALK_BLOCK)
        )
    if iterations < 1:
        raise Error("sym_device_newton_walk: iterations must be positive")
    if len(pool) == 0 or pool[0].leaves_cap < bin_count:
        pool.clear()
        pool.append(SymWalkerScratch(ctx, bin_count))
    ref ws = pool[0]
    enqueue_fill(ctx, ws.d_cur_point, Float32(0.0))
    enqueue_fill(ctx, ws.d_cursor_point, Float32(0.0))
    enqueue_fill(ctx, ws.d_dir, Float32(0.0))
    enqueue_fill(ctx, ws.d_scalars, Float32(0.0))

    var single = iterations == 1
    var n_evals = 1 if single else iterations + 1
    var n_rows = oracle.n_rows
    var blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var abmv_blocks = (
        n_rows + ABMV_BLOCK * ABMV_ELEMENTS - 1
    ) // (ABMV_BLOCK * ABMV_ELEMENTS)

    for e in range(n_evals):
        if e > 0:
            # `move_to`'s `AddBinModelValue(shift, bins, cursor)`
            ctx.enqueue_function[add_bin_model_value_kernel](
                oracle.d_shift.unsafe_ptr(),
                oracle.d_bins.unsafe_ptr(),
                Int32(n_rows),
                Int32(1),
                Int32(n_rows),
                oracle.d_cursor.unsafe_ptr(),
                grid_dim=(abmv_blocks, 1, 1),
                block_dim=(ABMV_BLOCK, 1, 1),
            )
        # `enqueue_single_dim_evaluation`'s two launches
        launch_approximate[True](
            ctx, oracle.objective,
            oracle.d_target, oracle.d_weights, Int32(n_rows),
            oracle.d_cursor,
            Int32(1) if oracle.has_weights else Int32(0),
            oracle.alpha, oracle.border,
            oracle.d_eval_stats, oracle.d_fv, Int32(1),
            oracle.d_mag_dummy, Int32(0),
            blocks,
        )
        compute_partition_stats(
            ctx, bin_count, 0, 2, n_rows,
            oracle.d_leaves, oracle.d_p_off, oracle.d_p_sz,
            oracle.d_eval_stats, oracle.d_partials, oracle.d_part_stats,
            sm_count=oracle.sm_count,
            row_bound=oracle.max_leaf_size,
        )
        ctx.enqueue_function[sym_newton_decide_kernel[SYM_WALK_BLOCK]](
            oracle.d_fv.unsafe_ptr(),
            Int32(oracle.fv_blocks),
            oracle.d_part_stats.unsafe_ptr(),
            Int32(bin_count),
            oracle.d_p_sz.unsafe_ptr(),
            Float32(oracle.lambda_reg),
            Float32(oracle.min_leaf_weight),
            Int32(backtracking_type),
            Int32(e),
            Int32(n_evals),
            Int32(1) if single else Int32(0),
            ws.d_cur_point.unsafe_ptr(),
            ws.d_cursor_point.unsafe_ptr(),
            ws.d_dir.unsafe_ptr(),
            ws.d_scalars.unsafe_ptr(),
            oracle.d_shift.unsafe_ptr(),
            est_out.unsafe_ptr(),
            grid_dim=(1, 1, 1),
            block_dim=(SYM_WALK_BLOCK, 1, 1),
        )
