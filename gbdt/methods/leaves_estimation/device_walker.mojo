# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Newton / Gradient leaf walker ON THE DEVICE (lane cpu4-gbdt).

`descent_helpers.newton_like_walker_estimate` walked the leaves on the host:
every evaluation drained, read the per-leaf derivative sums and the value
partials back, folded them in host loops, took the step and the
regularizer on the host and uploaded the next shift. On a GPU route that is
CPU numeric work over the leaves, which the owner's rule forbids. Here the
walk's state (`cur_point`, `next_point`, `direction`, the oracle's
`CurrentPoint`, the current value) lives in device buffers and every
statement of the walk runs in a kernel; the host keeps only the loop
counters (`iteration`, `updated`, `step`) and reads ONE word per
line-search evaluation, the accept flag, which is the walk's control flow.

SCOPE: the single-dimensional arm (`SingleBinDim() == 1`, `cursorDim == 1`:
every pointwise loss, QueryRMSE, PairLogit, YetiRank) under Newton or
Gradient with `AnyImprovement` backtracking, which is the only backtracking
the GPU fit passes (`_estimate_and_apply`, `estimate_advance`). The
multi-dimensional walk (MultiClass, MultiClassOneVsAll, MultiRMSE at more
than one iteration) still runs `descent_helpers`' host walker.

THE STATEMENTS, each the host walker's (`descent_helpers.mojo`,
`pointwise_oracle.mojo`, `step_estimator.mojo`), with every double a
`checks/soft_f64.mojo` bit pattern (correctly rounded, so the bits of a
hardware double unit, on every vendor):

  move       next = float(fma(step, double(dir), double(cur)))   (`_move`)
  regularize next = 0 when double(weight) < MinLeafWeight        (`regularize`)
  MoveTo     shift = next - CurrentPoint (float, computed as the double
             difference narrowed once = the correctly rounded float
             difference); CurrentPoint = next; the cursor add is the
             oracle's own `add_bin_model_value_kernel`
  zero avg   PairLogit / YetiRank: the double sum of the leaves in two
             levels (256-leaf halving trees, the lanes fold of their
             partials; CHANGED from an ascending chain with the host column
             `gbdt_oracle_losses._zero_average_host`), `bias = -sum / n`
  value      the evaluation's per-block value partials folded in
             `deterministic_sum_lanes_kernel`'s fixed order (CHANGED from
             the host's ascending Float32 chain, with every host column:
             `gbdt/host/gbdt_oracle*.mojo` fold with `_deterministic_sum_lanes`)
  accept     `function_value <= next_value` (AnyImprovement; the start
             evaluation is always taken)
  direction  hess = double(der2) + lambda (Newton) or double(weight) +
             lambda (Gradient); dir = hess > 0 ? float(double(der) /
             (hess + double(1e-20f))) : 0                  (`_diagonal_direction`)

The weights are the oracle's `WeightsCpu`: the leaf's row count
(`d_p_sz`) on an unweighted fit, the weight fold's float32 sum on a
weighted one (stashed off `d_part_stats` before the first evaluation
overwrites it).
"""

from std.memory import bitcast, stack_allocation
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.soft_f64 import (
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_fma,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_lt,
    sf64_mul,
    sf64_neg,
    sf64_sub,
    sf64_to_f32,
)
from gbdt.gpu_util.arena import BufferArena
from gbdt.trees_identical_switches import T23
from gbdt.methods.leaves_estimation.pointwise_oracle import BinOptimizedOracle
from gbdt.methods.leaves_estimation.step_estimator import (
    BACKTRACKING_ANY_IMPROVEMENT,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
)

#: the per-leaf kernels' block
comptime WALK_BLOCK = 256
#: the one-block folds; MUST equal `pointwise_targets.REDUCE_LANES_BLOCK`
#: (the value fold is `deterministic_sum_lanes_kernel[1]`'s order) and the
#: host columns' `GBDT_REDUCE_LANES_BLOCK`
comptime WALK_FOLD_BLOCK = 256


# ===========================================================================
# KERNELS
# ===========================================================================


def walker_move_kernel(
    cur: MutPointer[Float32, MutAnyOrigin],
    direction: MutPointer[Float32, MutAnyOrigin],
    step_bits: UInt64,
    weight_stats: MutPointer[Float32, MutAnyOrigin],
    leaf_sizes: MutPointer[UInt32, MutAnyOrigin],
    has_weights: Int32,
    min_leaf_weight_bits: UInt64,
    n_leaves_in: Int32,
    oracle_point: MutPointer[Float32, MutAnyOrigin],
    shift: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    apply_move: Int32,
):
    """`_move` + `regularize` (+ `MoveTo`'s shift when `apply_move`), one
    thread per leaf. `dst` is never `cur` (no two arguments alias)."""
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return
    var p = sf64_to_f32(
        sf64_fma(
            step_bits,
            sf64_from_f32(direction.unsafe_load(leaf)),
            sf64_from_f32(cur.unsafe_load(leaf)),
        )
    )
    # `regularize`: the host test `weights_cpu[leaf] < MinLeafWeight`
    # (a NaN weight compares false there: keep)
    var w: UInt64
    if has_weights != Int32(0):
        w = sf64_from_f32(weight_stats.unsafe_load(leaf))
    else:
        w = sf64_from_int(Int(leaf_sizes.unsafe_load(leaf)))
    if (not sf64_is_nan(w)) and sf64_lt(w, min_leaf_weight_bits):
        p = Float32(0.0)
    dst.unsafe_store(leaf, p)
    if apply_move != Int32(0):
        var o = oracle_point.unsafe_load(leaf)
        shift.unsafe_store(
            leaf, sf64_to_f32(sf64_sub(sf64_from_f32(p), sf64_from_f32(o)))
        )
        oracle_point.unsafe_store(leaf, p)


def walker_fold_decide_kernel(
    fv: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    values: MutPointer[Float32, MutAnyOrigin],
    flag: MutPointer[UInt32, MutAnyOrigin],
    always_accept: Int32,
):
    """ONE block of `WALK_FOLD_BLOCK`: the evaluation's value (the per-block
    partials in `deterministic_sum_lanes_kernel[1]`'s order: thread `t`
    walks `t, t + 256, ...` ascending, then the halving tree), then
    AnyImprovement's test against the current value. `values[0]` is the
    current value, `values[1]` the evaluated one; `flag[0]` = 1 when the
    point is taken (always at the walk's start)."""
    var tid = Int(thread_idx.x)
    var count = Int(count_in)
    var acc = Float32(0.0)
    var i = tid
    while i < count:
        acc += fv.unsafe_load(i)
        i += WALK_FOLD_BLOCK
    var red = stack_allocation[
        WALK_FOLD_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = acc
    barrier()
    var step = WALK_FOLD_BLOCK // 2
    while step > 0:
        if tid < step:
            red[tid] = red[tid] + red[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        var next_value = red[0]
        var take = True
        if always_accept == Int32(0):
            # `function_value <= next_value`: both doubles are widened
            # floats, so the float comparison is the same test
            take = Bool(values.unsafe_load(0) <= next_value)
        values.unsafe_store(1, next_value)
        if take:
            values.unsafe_store(0, next_value)
        flag.unsafe_store(0, UInt32(1) if take else UInt32(0))


def walker_accept_kernel(
    flag: MutPointer[UInt32, MutAnyOrigin],
    next_point: MutPointer[Float32, MutAnyOrigin],
    cur: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    weight_stats: MutPointer[Float32, MutAnyOrigin],
    leaf_sizes: MutPointer[UInt32, MutAnyOrigin],
    has_weights: Int32,
    lambda_bits: UInt64,
    newton: Int32,
    n_leaves_in: Int32,
    direction: MutPointer[Float32, MutAnyOrigin],
):
    """When the evaluation was taken (`flag[0]`): `cur = next` and the new
    direction from the evaluation's per-leaf sums `part_stats[2 * leaf]` =
    sum(der), `part_stats[2 * leaf + 1]` = sum(der2) (`finish_single_dim_
    evaluation`, `write_second_derivatives`, `_diagonal_direction`), one
    thread per leaf."""
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return
    if flag.unsafe_load(0) == UInt32(0):
        return
    cur.unsafe_store(leaf, next_point.unsafe_load(leaf))
    var g = sf64_from_f32(part_stats.unsafe_load(2 * leaf))
    var h: UInt64
    if newton != Int32(0):
        h = sf64_add(
            sf64_from_f32(part_stats.unsafe_load(2 * leaf + 1)), lambda_bits
        )
    else:
        # GRADIENT: `WeightsCpu[bin] + lambda` (`write_second_derivatives`)
        var w: UInt64
        if has_weights != Int32(0):
            w = sf64_from_f32(weight_stats.unsafe_load(leaf))
        else:
            w = sf64_from_int(Int(leaf_sizes.unsafe_load(leaf)))
        h = sf64_add(w, lambda_bits)
    var v = Float32(0.0)
    if (not sf64_is_nan(h)) and sf64_gt(h, SF64_ZERO):
        var eps = sf64_from_f32(Float32(1e-20))
        v = sf64_to_f32(sf64_div(g, sf64_add(h, eps)))
    direction.unsafe_store(leaf, v)


def zero_average_partials_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    n_leaves_in: Int32,
    partials: MutPointer[UInt64, MutAnyOrigin],
):
    """`MakeZeroAverage`'s sum, level 1: block `b` folds leaves
    `[256 b, 256 b + 256)` (absent leaves +0.0) as doubles in the halving
    tree into `partials[b]`."""
    var tid = Int(thread_idx.x)
    var n = Int(n_leaves_in)
    var i = Int(block_idx.x) * WALK_FOLD_BLOCK + tid
    var red = stack_allocation[
        WALK_FOLD_BLOCK,
        Scalar[DType.uint64],
        address_space = AddressSpace.SHARED,
    ]()
    var v = SF64_ZERO
    if i < n:
        v = sf64_from_f32(values.unsafe_load(i))
    red[tid] = v
    barrier()
    var step = WALK_FOLD_BLOCK // 2
    while step > 0:
        if tid < step:
            red[tid] = sf64_add(red[tid], red[tid + step])
        barrier()
        step //= 2
    if tid == 0:
        partials.unsafe_store(Int(block_idx.x), red[0])


def zero_average_bias_kernel(
    partials: MutPointer[UInt64, MutAnyOrigin],
    part_count_in: Int32,
    n_leaves_in: Int32,
    bias_out: MutPointer[UInt64, MutAnyOrigin],
):
    """Level 2, ONE block over the `ceil(n / 256)` level-1 partials (thread
    `t` adds partials `t, t + 256, ...` ascending from +0.0, then the
    halving tree), and `bias = -sum / n` into `bias_out[0]`."""
    var tid = Int(thread_idx.x)
    var parts = Int(part_count_in)
    var acc = SF64_ZERO
    var i = tid
    while i < parts:
        acc = sf64_add(acc, partials.unsafe_load(i))
        i += WALK_FOLD_BLOCK
    var red = stack_allocation[
        WALK_FOLD_BLOCK,
        Scalar[DType.uint64],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = acc
    barrier()
    var step = WALK_FOLD_BLOCK // 2
    while step > 0:
        if tid < step:
            red[tid] = sf64_add(red[tid], red[tid + step])
        barrier()
        step //= 2
    if tid == 0:
        var leaves = Int(n_leaves_in)
        var bias = SF64_ZERO
        if leaves > 0:
            bias = sf64_div(sf64_neg(red[0]), sf64_from_int(leaves))
        bias_out.unsafe_store(0, bias)


def zero_average_apply_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    n_leaves_in: Int32,
    bias_in: MutPointer[UInt64, MutAnyOrigin],
):
    """`leaf = float(double(leaf) + bias)`, one thread per leaf."""
    var n = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n:
        return
    values.unsafe_store(
        leaf,
        sf64_to_f32(
            sf64_add(sf64_from_f32(values.unsafe_load(leaf)), bias_in.unsafe_load(0))
        ),
    )


def weight_keep_mask_strided_kernel(
    weight_stats: MutPointer[Float32, MutAnyOrigin],
    min_leaf_weight_bits: UInt64,
    n_leaves_in: Int32,
    stride_in: Int32,
    out_mask: MutPointer[Float32, MutAnyOrigin],
):
    """`weight_keep_mask_kernel` with the decision at `out_mask[leaf *
    stride]` (the MultiClass one-step reads it at each leaf's first slot of
    `d_est`). Replaces the host mask fill of `_estimate_and_apply`'s
    MultiClass one-step arm, statement for statement."""
    var n_leaves = Int(n_leaves_in)
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if leaf >= n_leaves:
        return
    var w = sf64_from_f32(weight_stats.unsafe_load(leaf))
    var keep = Float32(1.0)
    if (not sf64_is_nan(w)) and sf64_lt(w, min_leaf_weight_bits):
        keep = Float32(0.0)
    out_mask.unsafe_store(leaf * Int(stride_in), keep)


def count_nonzero_u32_kernel(
    flags: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    out_count: MutPointer[UInt32, MutAnyOrigin],
):
    """ONE block of `WALK_FOLD_BLOCK`: how many of `flags[0, n)` are nonzero
    (DEVIATION 74's per-leaf Cholesky fallback flags), written to
    `out_count[0]`. Integer, so any order gives the same count."""
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var acc = UInt32(0)
    var i = tid
    while i < n:
        if flags.unsafe_load(i) != UInt32(0):
            acc += 1
        i += WALK_FOLD_BLOCK
    var red = stack_allocation[
        WALK_FOLD_BLOCK,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = acc
    barrier()
    var step = WALK_FOLD_BLOCK // 2
    while step > 0:
        if tid < step:
            red[tid] = red[tid] + red[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        out_count.unsafe_store(0, red[0])


def scaled_copy_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale: Float32,
    use_scale: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
    dst_offset_in: Int32,
):
    """`dst[dst_offset + i] = src[i]` (or `src[i] * scale`, the model's leaf
    rescale, as the correctly rounded float product `identical_mul` pins)."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var v = src.unsafe_load(i)
    if use_scale != Int32(0):
        # the product of two floats is EXACT in a double (48 significant
        # bits), so narrowing it once is the correctly rounded float product
        # with its IEEE sign, subnormals honored, on every vendor
        v = sf64_to_f32(sf64_mul(sf64_from_f32(v), sf64_from_f32(scale)))
    dst.unsafe_store(Int(dst_offset_in) + i, v)


def scale_in_place_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale: Float32,
):
    """`values[i] = values[i] * scale`, the model's `leaf * learning_rate`
    rescale (`identical_mul`: the correctly rounded float product), one
    thread per value."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    values.unsafe_store(
        i,
        sf64_to_f32(
            sf64_mul(sf64_from_f32(values.unsafe_load(i)), sf64_from_f32(scale))
        ),
    )


# ===========================================================================
# THE WALK'S BUFFERS (owned by the fit's estimation workspace)
# ===========================================================================


struct WalkBuffers(Movable):
    """The device walk's state for up to `cap` leaves. Kept by the fit's
    `TEstimationWorkspace` (pool of one), handed to a task as handle copies
    (`handles`), so no task allocates."""

    var cap: Int
    var d_cur: DeviceBuffer[DType.float32]
    var d_next: DeviceBuffer[DType.float32]
    var d_dir: DeviceBuffer[DType.float32]
    #: the oracle's `CurrentPoint` (cursor gauge = walker gauge, single-dim)
    var d_opt: DeviceBuffer[DType.float32]
    #: the weighted fit's per-leaf weight fold (`WeightsCpu`), stashed
    var d_w: DeviceBuffer[DType.float32]
    #: [current value, last evaluated value]
    var d_vals: DeviceBuffer[DType.float32]
    var d_flag: DeviceBuffer[DType.uint32]
    var h_flag: HostBuffer[DType.uint32]
    #: `MakeZeroAverage`'s level-1 partials (`ceil(cap / 256)`) and bias
    var d_zpart: DeviceBuffer[DType.uint64]
    var d_zbias: DeviceBuffer[DType.uint64]
    #: True when the walk's result is in `d_next` (the one-step walk), else
    #: it is in `d_cur`
    var result_next: Bool

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        var n = cap if cap > 0 else 1
        self.cap = n
        self.d_cur = ctx.enqueue_create_buffer[DType.float32](n)
        self.d_next = ctx.enqueue_create_buffer[DType.float32](n)
        self.d_dir = ctx.enqueue_create_buffer[DType.float32](n)
        self.d_opt = ctx.enqueue_create_buffer[DType.float32](n)
        self.d_w = ctx.enqueue_create_buffer[DType.float32](n)
        self.d_vals = ctx.enqueue_create_buffer[DType.float32](2)
        self.d_flag = ctx.enqueue_create_buffer[DType.uint32](1)
        self.h_flag = ctx.enqueue_create_host_buffer[DType.uint32](1)
        self.d_zpart = ctx.enqueue_create_buffer[DType.uint64](
            (n + WALK_FOLD_BLOCK - 1) // WALK_FOLD_BLOCK
        )
        self.d_zbias = ctx.enqueue_create_buffer[DType.uint64](1)
        self.result_next = False

    def __init__(
        out self, ctx: DeviceContext, mut arena: BufferArena, cap: Int
    ) raises:
        var n = cap if cap > 0 else 1
        self.cap = n
        self.d_cur = arena.device[DType.float32](ctx, n)
        self.d_next = arena.device[DType.float32](ctx, n)
        self.d_dir = arena.device[DType.float32](ctx, n)
        self.d_opt = arena.device[DType.float32](ctx, n)
        self.d_w = arena.device[DType.float32](ctx, n)
        self.d_vals = arena.device[DType.float32](ctx, 2)
        self.d_flag = arena.device[DType.uint32](ctx, 1)
        self.h_flag = arena.host_buffer[DType.uint32](ctx, 1)
        self.d_zpart = arena.device[DType.uint64](
            ctx, (n + WALK_FOLD_BLOCK - 1) // WALK_FOLD_BLOCK
        )
        self.d_zbias = arena.device[DType.uint64](ctx, 1)
        self.result_next = False

    def __init__(out self, *, handles_of: WalkBuffers):
        """Handle copies (the same device memory)."""
        self.cap = handles_of.cap
        self.d_cur = handles_of.d_cur.copy()
        self.d_next = handles_of.d_next.copy()
        self.d_dir = handles_of.d_dir.copy()
        self.d_opt = handles_of.d_opt.copy()
        self.d_w = handles_of.d_w.copy()
        self.d_vals = handles_of.d_vals.copy()
        self.d_flag = handles_of.d_flag.copy()
        self.h_flag = handles_of.h_flag.copy()
        self.d_zpart = handles_of.d_zpart.copy()
        self.d_zbias = handles_of.d_zbias.copy()
        self.result_next = handles_of.result_next

    def handles(self) -> WalkBuffers:
        return WalkBuffers(handles_of=self)

    def result(self) -> DeviceBuffer[DType.float32]:
        """The walk's estimate (a handle), once the walk has finished."""
        if self.result_next:
            return self.d_next.copy()
        return self.d_cur.copy()


# ===========================================================================
# THE WALK
# ===========================================================================


def device_walk_supported(
    oracle: BinOptimizedOracle, backtracking_type: Int, n_leaves: Int,
    cap: Int,
) -> Bool:
    """The configurations this walk restates (see the module docstring)."""
    return (
        oracle.single_bin_dim == 1
        and oracle.cursor_dim == 1
        and oracle.hessian_block_size() == 1
        and (
            oracle.estimation_method == LEAF_ESTIMATION_NEWTON
            or oracle.estimation_method == LEAF_ESTIMATION_GRADIENT
        )
        and backtracking_type == BACKTRACKING_ANY_IMPROVEMENT
        and oracle.bin_count == n_leaves
        and n_leaves <= cap
    )


def _leaf_grid(n: Int) -> Int:
    var g = (n + WALK_BLOCK - 1) // WALK_BLOCK
    return g if g > 0 else 1


def _enqueue_move(
    ctx: DeviceContext,
    mut oracle: BinOptimizedOracle,
    mut w: WalkBuffers,
    step: Float64,
    apply_move: Bool,
) raises:
    """`_move(cur, dir, step)`, `regularize`, and (when `apply_move`) the
    oracle's `MoveTo`: the shift into `oracle.d_shift`, then the oracle's
    own cursor add. The point lands in `d_next`."""
    var n = oracle.bin_count
    ctx.enqueue_function[walker_move_kernel](
        w.d_cur.unsafe_ptr(),
        w.d_dir.unsafe_ptr(),
        bitcast[DType.uint64](step),
        w.d_w.unsafe_ptr(),
        oracle.d_p_sz.unsafe_ptr(),
        Int32(1) if oracle.has_weights else Int32(0),
        bitcast[DType.uint64](oracle.min_leaf_weight),
        Int32(n),
        w.d_opt.unsafe_ptr(),
        oracle.d_shift.unsafe_ptr(),
        w.d_next.unsafe_ptr(),
        Int32(1) if apply_move else Int32(0),
        grid_dim=(_leaf_grid(n), 1, 1),
        block_dim=(WALK_BLOCK, 1, 1),
    )
    if apply_move:
        oracle._launch_shift_abmv()


def _enqueue_eval_and_decide(
    ctx: DeviceContext,
    mut oracle: BinOptimizedOracle,
    mut w: WalkBuffers,
    always_accept: Bool,
) raises:
    """The evaluation at the moved point (no host readback), the value fold
    and AnyImprovement's test, and on acceptance `cur = next` and the new
    direction."""
    oracle.enqueue_single_dim_evaluation(readback=False)
    ctx.enqueue_function[walker_fold_decide_kernel](
        oracle.d_fv.unsafe_ptr(),
        Int32(oracle.fv_blocks),
        w.d_vals.unsafe_ptr(),
        w.d_flag.unsafe_ptr(),
        Int32(1) if always_accept else Int32(0),
        grid_dim=(1, 1, 1),
        block_dim=(WALK_FOLD_BLOCK, 1, 1),
    )
    var n = oracle.bin_count
    ctx.enqueue_function[walker_accept_kernel](
        w.d_flag.unsafe_ptr(),
        w.d_next.unsafe_ptr(),
        w.d_cur.unsafe_ptr(),
        oracle.d_part_stats.unsafe_ptr(),
        w.d_w.unsafe_ptr(),
        oracle.d_p_sz.unsafe_ptr(),
        Int32(1) if oracle.has_weights else Int32(0),
        bitcast[DType.uint64](oracle.lambda_reg),
        Int32(1) if oracle.estimation_method == LEAF_ESTIMATION_NEWTON else Int32(0),
        Int32(n),
        w.d_dir.unsafe_ptr(),
        grid_dim=(_leaf_grid(n), 1, 1),
        block_dim=(WALK_BLOCK, 1, 1),
    )


def _one_step_prepare_kernel(
    stats: MutPointer[Float32, MutAnyOrigin],
    cur: MutPointer[Float32, MutAnyOrigin],
    direction: MutPointer[Float32, MutAnyOrigin],
    optimum: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32, has_weights: Int32,
):
    """T23: exact device one-step preparation in one leaf pass.

    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    The three +0 fills and the constructor's exact weight copy keep their
    incumbent values. No gradient, Hessian, regularizer or step is changed.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        cur.unsafe_store(i, Float32(0.0))
        direction.unsafe_store(i, Float32(0.0))
        optimum.unsafe_store(i, Float32(0.0))
        if has_weights != Int32(0):
            weights.unsafe_store(i, stats.unsafe_load(i))


def device_walk_begin(
    ctx: DeviceContext,
    mut oracle: BinOptimizedOracle,
    mut w: WalkBuffers,
    iterations: Int,
) raises -> Bool:
    """The walk's start, ENQUEUED: `MoveTo(0)`, the evaluation, the first
    direction; and for a one-iteration walk the final full step (no
    re-evaluation, `descent_helpers.cpp:151-156`). Returns True when the
    walk is already finished (one iteration: the result is in `d_next`,
    nothing to read back). The weighted fit's weight fold must still be in
    `oracle.d_part_stats` (the oracle was just made)."""
    var n = oracle.bin_count
    w.result_next = False
    if T23 and iterations == 1:
        ctx.enqueue_function[_one_step_prepare_kernel](
            oracle.d_part_stats.unsafe_ptr(), w.d_cur.unsafe_ptr(),
            w.d_dir.unsafe_ptr(), w.d_opt.unsafe_ptr(), w.d_w.unsafe_ptr(),
            Int32(n), Int32(1) if oracle.has_weights else Int32(0),
            grid_dim=(_leaf_grid(n), 1, 1), block_dim=(WALK_BLOCK, 1, 1),
        )
    else:
        w.d_cur.enqueue_fill(Float32(0.0))
        w.d_dir.enqueue_fill(Float32(0.0))
        w.d_opt.enqueue_fill(Float32(0.0))
        if oracle.has_weights:
            # `WeightsCpu`: the constructor's per-leaf weight fold, before the
            # evaluation below overwrites `d_part_stats`
            ctx.enqueue_function[scaled_copy_kernel](
                oracle.d_part_stats.unsafe_ptr(),
                Int32(n),
                Float32(1.0),
                Int32(0),
                w.d_w.unsafe_ptr(),
                Int32(0),
                grid_dim=(_leaf_grid(n), 1, 1),
                block_dim=(WALK_BLOCK, 1, 1),
            )
    # `MoveTo(startPoint)` at the zero point: `fma(0, 0, +0) = +0`, kept
    # through the regularizer, shift +0.0 - +0.0 = +0.0, and the cursor add
    # runs as the host walker's did
    _enqueue_move(ctx, oracle, w, Float64(0.0), True)
    _enqueue_eval_and_decide(ctx, oracle, w, True)
    if iterations == 1:
        _enqueue_move(ctx, oracle, w, Float64(1.0), False)
        w.result_next = True
        return True
    return False


def device_walk_enqueue_line_search(
    ctx: DeviceContext,
    mut oracle: BinOptimizedOracle,
    mut w: WalkBuffers,
    step: Float64,
) raises:
    """One line-search evaluation at `cur + step * dir`, ENQUEUED with its
    accept flag's one-word readback; `device_walk_flag` reads it after the
    caller's drain."""
    _enqueue_move(ctx, oracle, w, step, True)
    _enqueue_eval_and_decide(ctx, oracle, w, False)
    ctx.enqueue_copy(dst_ptr=w.h_flag.unsafe_ptr(), src_buf=w.d_flag)


def device_walk_flag(mut w: WalkBuffers) -> Bool:
    """The last line-search evaluation's accept flag (after a drain)."""
    return w.h_flag.unsafe_ptr().unsafe_load(0) != UInt32(0)


def device_walk_estimate(
    ctx: DeviceContext,
    mut oracle: BinOptimizedOracle,
    mut w: WalkBuffers,
    iterations: Int,
) raises:
    """`TNewtonLikeWalker::Estimate` (`descent_helpers.cpp:128-204`) with
    AnyImprovement, the walk on the device. The host runs only the loop
    counters and reads one word (the accept flag) per line-search
    evaluation. The estimate is `w.result()` (the single-dimensional
    `MakeEstimationResult` is the identity), enqueued."""
    if device_walk_begin(ctx, oracle, w, iterations):
        return
    var updated = False
    var iteration = 0
    while iteration < iterations:
        var step = Float64(1.0)
        var accepted = False
        while iteration < iterations or ((not updated) and iteration < 100):
            device_walk_enqueue_line_search(ctx, oracle, w, step)
            # the walk's control flow: one word
            ctx.synchronize()
            if device_walk_flag(w):
                iteration += 1
                updated = True
                accepted = True
                break
            iteration += 1
            step /= 2
        if not accepted:
            break


def enqueue_zero_average(
    ctx: DeviceContext,
    mut w: WalkBuffers,
    mut values: DeviceBuffer[DType.float32],
    leaf_count: Int,
) raises:
    """`MakeZeroAverage` over `values[0, leaf_count)` in place: the double
    sum in two levels (256-leaf halving trees, then the lanes fold of their
    partials), the bias, the shift; `leaf_count <= w.cap`. The host column
    is `gbdt_oracle_losses._zero_average_host`, the same order."""
    var parts = (leaf_count + WALK_FOLD_BLOCK - 1) // WALK_FOLD_BLOCK
    if parts < 1:
        return
    ctx.enqueue_function[zero_average_partials_kernel](
        values.unsafe_ptr(),
        Int32(leaf_count),
        w.d_zpart.unsafe_ptr(),
        grid_dim=(parts, 1, 1),
        block_dim=(WALK_FOLD_BLOCK, 1, 1),
    )
    ctx.enqueue_function[zero_average_bias_kernel](
        w.d_zpart.unsafe_ptr(),
        Int32(parts),
        Int32(leaf_count),
        w.d_zbias.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(WALK_FOLD_BLOCK, 1, 1),
    )
    ctx.enqueue_function[zero_average_apply_kernel](
        values.unsafe_ptr(),
        Int32(leaf_count),
        w.d_zbias.unsafe_ptr(),
        grid_dim=(_leaf_grid(leaf_count), 1, 1),
        block_dim=(WALK_BLOCK, 1, 1),
    )


def enqueue_copy_values(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    n: Int,
    mut dst: DeviceBuffer[DType.float32],
    dst_offset: Int = 0,
) raises:
    """`dst[dst_offset, dst_offset + n) = src[0, n)` on the device."""
    ctx.enqueue_function[scaled_copy_kernel](
        src.unsafe_ptr(),
        Int32(n),
        Float32(1.0),
        Int32(0),
        dst.unsafe_ptr(),
        Int32(dst_offset),
        grid_dim=(_leaf_grid(n), 1, 1),
        block_dim=(WALK_BLOCK, 1, 1),
    )


def enqueue_scale_in_place(
    ctx: DeviceContext,
    mut values: DeviceBuffer[DType.float32],
    count: Int,
    scale: Float32,
) raises:
    """The model's leaf rescale on the device, in place, over `[0, count)`."""
    ctx.enqueue_function[scale_in_place_kernel](
        values.unsafe_ptr(),
        Int32(count),
        scale,
        grid_dim=(_leaf_grid(count), 1, 1),
        block_dim=(WALK_BLOCK, 1, 1),
    )
