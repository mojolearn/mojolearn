# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reference: `catboost/cuda/targets/kernel/query_softmax.cu`
(`ComputeGroupMaximalsImpl` `:8-64`, `ComputeQueryExponentsImpl` `:66-96`,
`ComputeGroupSumsImpl` `:98-147`, `QuerySoftMaxImpl` `:149-217`), the host
kernel that launches them, `TQuerySoftMaxKernel::Run`
(`catboost/cuda/targets/kernel.h:229-356`), and the QuerySoftMax arms of
`TQuerywiseTargetsImpl` (`targets/querywise_targets_impl.h:100-103`,
`:242-257`, `:271`, `:346-354`).

THEIR SEQUENCE, IN ORDER, per call:

    ApproxExp = Indices ? Gather(Predictions, Indices) : copy(Predictions)
    Qids = ComputeGroupIds(sizes, offsets, bias)
    QueryApprox, QuerySumWeightedTargets = ComputeGroupMaximals(Relevs, Weights, ApproxExp)
    ApproxExp[i] = exp(beta * (ApproxExp[i] - QueryApprox[qid])) * weight[i]
    QueryApprox = ComputeGroupSums(ApproxExp)
    QuerySoftMaxImpl(Relevs, Weights, ApproxExp, Qids, lambda, beta, QueryApprox,
                     QuerySumWeightedTargets, writeMap = Indices)

`launch_query_softmax_with` launches exactly that. The two group kernels share
`query_helper.mojo`'s layout (a block of 128 threads carries four queries of
32 lanes each) and its pinned `WarpReduce` tree: `line[x] = op(line[x],
line[x + s])` for `s = 16 .. 1`, with `op` the float32 add for the sums and
their `TCudaMax` (`kernel_helpers.cuh:75-80`, CUDA's `max(float, float)`,
which is `fmaxf`) for the maxima. Per lane, in row order: the maximum of the
approx over rows with `weight > 0`, starting at `-FLT_MAX`, and the sum of
`target * weight` over ALL rows of the query (their GPU kernel has no
`target > 0` guard here; their CPU der calcer does).

THE KERNEL (`query_softmax.cu:149-197`), per row `i` in ROW order:

    softmax = approxExp[i] / approxSum[qid];  wt = weight * target
    guard = weight > 0 && sumTargets[qid] > 0
    der[writeMap[i]]  = beta * ((guard ? -sumTargets * softmax : 0) + wt)
    der2[writeMap[i]] = guard ? beta * sumTargets * (beta * softmax * (1 - softmax) + lambda) : 0
    functionValue    += (weight > 0 && target > 0) ? wt * log(softmax) : 0

`writeMap` is null on the structure search and the INVERSE of the oracle's bin
order during leaf estimation, as for QueryRMSE (`query_rmse.mojo`).

THE STATS LAYOUT:

  - `estimation=False`, the SEARCH: plane 1 is `der`; plane 0 follows the
    search weight plane DEVIATION of `pair_logit.mojo` (the reference's
    querywise `StochasticDer` has the two arms of `secondDerAsWeights`
    reversed against the pointwise target): the row weight under Cosine and
    L2 (`second_order=False`), `der2` under NewtonL2 and NewtonCosine.
  - `estimation=True`, the ORACLE's `ApproximateAt`: plane 0 `der`, plane 1
    `der2`, both at `writeMap[i]`.

THE LOSS WEIGHT. `ComputeStats` divides by `TotalWeightedTarget`
(`querywise_targets_impl.h:100-103`), `DotProduct(targets, weights)` taken
once at `InitQuerySoftmax` (`:346-354`), which refuses a total that is not
positive in their words.

DEVIATIONS:

  * `functionValue` and the magnitudes are per-block partials through
    `pinned_block_sum` at the 256-thread block the pointwise kernels use,
    folded by `deterministic_sum_lanes_kernel`, where theirs is a 1024-thread
    `FastInBlockReduce` plus an `atomicAdd` (the pointwise family's policy).
  * `exp` and `log` route through `routed_exp` / `routed_log` (DEVIATION 254,
    IDENTITY_PATHS row 12), and every value one kernel stores for the next
    (the weighted exponent, the group sums, der, der2) passes through `ftz`
    (row 10), a comptime no-op outside IDENTICAL.
  * `TotalWeightedTarget` is a Float64 sum in row order on the host
    (`query_softmax_total_weighted_target`), where the reference reduces it
    on the device. It only scales the reported learn loss and gates the
    refusal; no tree reads it.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from gbdt.gpu_data.kernel.query_helper import (
    QUERIES_PER_BLOCK,
    QUERY_HELPER_BLOCK_SIZE,
    QUERY_LANES,
    launch_compute_group_ids,
    query_helper_blocks,
)
from gbdt.gpu_util.kernel.transform import TRANSFORM_BLOCK_SIZE
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    pinned_block_sum,
    routed_exp,
    routed_log,
)
from gbdt.targets.kernel.query_rmse import (
    QuerywiseTargetBuffers,
    make_querywise_target_buffers,
)

#: `GetQuerySoftMaxLambdaReg`'s default (`loss_description.cpp:209-212`)
comptime QUERY_SOFTMAX_DEFAULT_LAMBDA = 0.01
#: `GetQuerySoftMaxBeta`'s default (`loss_description.cpp:214-222`)
comptime QUERY_SOFTMAX_DEFAULT_BETA = 1.0


@always_inline
def query_softmax_fmax(a: Float32, b: Float32) -> Float32:
    """CUDA's `max(float, float)`, `fmaxf`: a NaN operand yields the other
    operand; otherwise the larger. (Which zero `fmaxf` returns for `-0` and
    `+0` is unspecified; the one reader, `exp(beta * (approx - max))`, gives
    the same bits for either.)"""
    if a != a:
        return b
    if b != b:
        return a
    return b if a < b else a


def query_softmax_total_weighted_target(
    targets: List[Float32], weights: List[Float32], has_weights: Bool
) raises -> Float64:
    """`InitQuerySoftmax` (`querywise_targets_impl.h:346-354`): the dot product
    of targets and weights, a Float64 sum in row order (the DEVIATION in the
    module docstring), refused in their words when it is not positive."""
    var total = Float64(0.0)
    for i in range(len(targets)):
        var w = Float64(1.0)
        if has_weights:
            w = Float64(weights[i])
        total += Float64(targets[i]) * w
    if not (total > Float64(0.0)):
        raise Error(
            "Observation targets and weights should be greater or equal zero."
            " Total weighted target should be greater, than zero"
        )
    return total


struct QuerySoftMaxTargetBuffers(Movable):
    """The querywise buffers (targets, weights, query offsets and sizes, row
    query ids, inverse order, `no_indices`), the loss parameters, the loss
    weight and `TQuerySoftMaxKernel::PrepareContext`'s scratch
    (`kernel.h:249-257`). Built once per fit by
    `make_query_softmax_target_buffers`; `handles()` gives views onto the same
    memory."""

    var query: QuerywiseTargetBuffers
    var lambda_reg: Float32
    var beta: Float32
    var total_weighted_target: Float64
    var approx_exp: DeviceBuffer[DType.float32]
    var query_approx: DeviceBuffer[DType.float32]
    var query_sum_wt: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        var query: QuerywiseTargetBuffers,
        lambda_reg: Float32,
        beta: Float32,
        total_weighted_target: Float64,
        var approx_exp: DeviceBuffer[DType.float32],
        var query_approx: DeviceBuffer[DType.float32],
        var query_sum_wt: DeviceBuffer[DType.float32],
    ):
        self.query = query^
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.total_weighted_target = total_weighted_target
        self.approx_exp = approx_exp^
        self.query_approx = query_approx^
        self.query_sum_wt = query_sum_wt^

    def handles(self) -> QuerySoftMaxTargetBuffers:
        """Handle copies onto the same device memory."""
        return QuerySoftMaxTargetBuffers(
            self.query.handles(), self.lambda_reg, self.beta,
            self.total_weighted_target, self.approx_exp.copy(),
            self.query_approx.copy(), self.query_sum_wt.copy(),
        )


def make_query_softmax_target_buffers(
    ctx: DeviceContext,
    group_sizes: List[UInt32],
    n_rows: Int,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    lambda_reg: Float32,
    beta: Float32,
) raises -> QuerySoftMaxTargetBuffers:
    """Upload the grouping, read the targets and weights back once for
    `TotalWeightedTarget`, and allocate the scratch."""
    var query = make_querywise_target_buffers(
        ctx, group_sizes, n_rows, targets, weights, has_weights
    )
    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=h_t.unsafe_ptr(), src_buf=targets)
    if has_weights:
        ctx.enqueue_copy(dst_ptr=h_w.unsafe_ptr(), src_buf=weights)
    ctx.synchronize()
    var t_list = List[Float32](capacity=n_rows)
    var w_list = List[Float32](capacity=n_rows)
    for i in range(n_rows):
        t_list.append(h_t.unsafe_ptr().unsafe_load(i))
        w_list.append(h_w.unsafe_ptr().unsafe_load(i) if has_weights else Float32(1.0))
    _ = h_t^  # past the drain (step-33 race class)
    _ = h_w^
    var total = query_softmax_total_weighted_target(t_list, w_list, has_weights)
    var q_count = query.q_count
    var approx_exp = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var query_approx = ctx.enqueue_create_buffer[DType.float32](q_count)
    var query_sum_wt = ctx.enqueue_create_buffer[DType.float32](q_count)
    return QuerySoftMaxTargetBuffers(
        query^, lambda_reg, beta, total, approx_exp^, query_approx^,
        query_sum_wt^,
    )


def query_softmax_gather_kernel(
    predictions: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    has_indices: Int32,
    size_in: Int32,
    approx_exp: MutPointer[Float32, MutAnyOrigin],
):
    """`Gather(ApproxExp, Predictions, Indices)` or the copy (`kernel.h:
    306-310`), per row."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        if has_indices != Int32(0):
            approx_exp.unsafe_store(i, predictions.unsafe_load(Int(indices.unsafe_load(i))))
        else:
            approx_exp.unsafe_store(i, predictions.unsafe_load(i))
        i += step


def compute_group_maximals_kernel(
    target: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    approx_exp: MutPointer[Float32, MutAnyOrigin],
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    offsets_bias: UInt32,
    q_sizes: MutPointer[UInt32, MutAnyOrigin],
    q_count_in: Int32,
    maximals: MutPointer[Float32, MutAnyOrigin],
    sum_weighted_targets: MutPointer[Float32, MutAnyOrigin],
):
    """`ComputeGroupMaximalsImpl` (`query_softmax.cu:8-51`)."""
    var q_count = Int(q_count_in)
    var tid = Int(thread_idx.x)
    var local_qid = tid // QUERY_LANES
    var qid = Int(block_idx.x) * QUERIES_PER_BLOCK + local_qid
    var lane = tid & (QUERY_LANES - 1)

    var read_offset = 0
    var query_size = 0
    if qid < q_count:
        read_offset = Int(q_offsets.unsafe_load(qid) - offsets_bias)
        query_size = Int(q_sizes.unsafe_load(qid))

    var max_approx = -Float32.MAX_FINITE
    var sum_wt = Float32(0.0)
    var i = lane
    while i < query_size:
        var t = target.unsafe_load(read_offset + i)
        var w = Float32(1.0)
        if has_weights != Int32(0):
            w = weights.unsafe_load(read_offset + i)
        var a = approx_exp.unsafe_load(read_offset + i)
        if w > Float32(0.0):
            max_approx = query_softmax_fmax(max_approx, a)
        sum_wt = sum_wt + t * w
        i += QUERY_LANES

    var line_max = stack_allocation[
        QUERY_HELPER_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var line_sum = stack_allocation[
        QUERY_HELPER_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    line_max[tid] = max_approx
    line_sum[tid] = sum_wt
    barrier()
    var base = local_qid * QUERY_LANES
    var step = QUERY_LANES // 2
    while step > 0:
        if lane < step:
            line_max[base + lane] = query_softmax_fmax(
                line_max[base + lane], line_max[base + lane + step]
            )
            line_sum[base + lane] = line_sum[base + lane] + line_sum[base + lane + step]
        barrier()
        step //= 2

    if lane == 0 and qid < q_count:
        maximals.unsafe_store(qid, line_max[base])
        sum_weighted_targets.unsafe_store(qid, ftz(line_sum[base]))


def query_softmax_exponents_kernel(
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    qids: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    maximals: MutPointer[Float32, MutAnyOrigin],
    approx_exp: MutPointer[Float32, MutAnyOrigin],
    beta: Float32,
):
    """`ComputeQueryExponentsImpl` (`query_softmax.cu:66-84`), per row."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var weight = Float32(1.0)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)
        var approx = approx_exp.unsafe_load(i)
        var appr_max = maximals.unsafe_load(Int(qids.unsafe_load(i)))
        approx_exp.unsafe_store(i, ftz(routed_exp(beta * (approx - appr_max)) * weight))
        i += step


def compute_group_sums_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    offsets_bias: UInt32,
    q_sizes: MutPointer[UInt32, MutAnyOrigin],
    q_count_in: Int32,
    group_sums: MutPointer[Float32, MutAnyOrigin],
):
    """`ComputeGroupSumsImpl` (`query_softmax.cu:98-136`); its `result`
    staging behind a `__syncthreads` stores the same value."""
    var q_count = Int(q_count_in)
    var tid = Int(thread_idx.x)
    var local_qid = tid // QUERY_LANES
    var qid = Int(block_idx.x) * QUERIES_PER_BLOCK + local_qid
    var lane = tid & (QUERY_LANES - 1)

    var read_offset = 0
    var query_size = 0
    if qid < q_count:
        read_offset = Int(q_offsets.unsafe_load(qid) - offsets_bias)
        query_size = Int(q_sizes.unsafe_load(qid))

    var sum_data = Float32(0.0)
    var i = lane
    while i < query_size:
        sum_data = sum_data + data.unsafe_load(read_offset + i)
        i += QUERY_LANES

    var line = stack_allocation[
        QUERY_HELPER_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    line[tid] = sum_data
    barrier()
    var base = local_qid * QUERY_LANES
    var step = QUERY_LANES // 2
    while step > 0:
        if lane < step:
            line[base + lane] = line[base + lane] + line[base + lane + step]
        barrier()
        step //= 2

    if lane == 0 and qid < q_count:
        group_sums.unsafe_store(qid, ftz(line[base]))


def query_softmax_kernel[estimation: Bool, second_order: Bool](
    target: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    approx_exp: MutPointer[Float32, MutAnyOrigin],
    qids: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    approx_sum: MutPointer[Float32, MutAnyOrigin],
    sum_weighted_targets: MutPointer[Float32, MutAnyOrigin],
    lambda_reg: Float32,
    beta: Float32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`QuerySoftMaxImpl` (`query_softmax.cu:149-197`); see the module
    docstring for the modes and the partials."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var in_range = i < size

    var weight = Float32(1.0)
    var der = Float32(0.0)
    var der2 = Float32(0.0)
    var score = Float32(0.0)
    if in_range:
        var target_val = target.unsafe_load(i)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)
        var approx = approx_exp.unsafe_load(i)
        var qid = Int(qids.unsafe_load(i))
        var a_sum = approx_sum.unsafe_load(qid)
        var sum_targets = sum_weighted_targets.unsafe_load(qid)
        var softmax = approx / a_sum
        var wt = weight * target_val
        var guard = weight > Float32(0.0) and sum_targets > Float32(0.0)
        var first = Float32(0.0)
        if guard:
            first = (-sum_targets) * softmax
        der = ftz(beta * (first + wt))
        if guard:
            der2 = ftz(
                beta * sum_targets * (beta * softmax * (Float32(1.0) - softmax) + lambda_reg)
            )
        if weight > Float32(0.0) and target_val > Float32(0.0):
            score = wt * routed_log(softmax)
        comptime if estimation:
            var dst = i
            if has_write_map != Int32(0):
                dst = Int(write_map.unsafe_load(i))
            stats.unsafe_store(dst, der)
            stats.unsafe_store(size + dst, der2)
        else:
            comptime if second_order:
                stats.unsafe_store(i, der2)
            else:
                stats.unsafe_store(i, weight)
            stats.unsafe_store(size + i, der)

    if compute_fv != Int32(0):
        var total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            comptime if second_order:
                w_abs = abs(der2)
            else:
                w_abs = abs(weight)
            g_abs = abs(der)
        var w_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x) + 1, g_total)


def launch_query_softmax_with[estimation: Bool, second_order: Bool](
    ctx: DeviceContext,
    mut s: QuerySoftMaxTargetBuffers,
    mut predictions: DeviceBuffer[DType.float32],
    use_inverse: Bool,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`TQuerySoftMaxKernel::Run` (`kernel.h:295-352`), one device, offsets
    bias 0: the search and final passes read the point in row order
    (`use_inverse` False, their null `Indices`); the leaf estimator reads it
    through `s.query.inverse`, which is also the write map."""
    var size = s.query.n_rows
    if size <= 0:
        return
    var q_count = s.query.q_count
    var t_blocks = (size + TRANSFORM_BLOCK_SIZE - 1) // TRANSFORM_BLOCK_SIZE
    var idx_ptr = s.query.inverse.unsafe_ptr() if use_inverse else s.query.no_indices.unsafe_ptr()
    ctx.enqueue_function[query_softmax_gather_kernel](
        predictions.unsafe_ptr(), idx_ptr, Int32(1) if use_inverse else Int32(0),
        Int32(size), s.approx_exp.unsafe_ptr(),
        grid_dim=(t_blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )
    launch_compute_group_ids(
        ctx, s.query.q_sizes, s.query.q_offsets, UInt32(0), q_count, s.query.qids
    )
    var q_blocks = query_helper_blocks(q_count)
    var has_w = Int32(1) if s.query.has_weights else Int32(0)
    if q_blocks > 0:
        ctx.enqueue_function[compute_group_maximals_kernel](
            s.query.targets.unsafe_ptr(), s.query.weights.unsafe_ptr(), has_w,
            s.approx_exp.unsafe_ptr(), s.query.q_offsets.unsafe_ptr(), UInt32(0),
            s.query.q_sizes.unsafe_ptr(), Int32(q_count),
            s.query_approx.unsafe_ptr(), s.query_sum_wt.unsafe_ptr(),
            grid_dim=(q_blocks, 1, 1),
            block_dim=(QUERY_HELPER_BLOCK_SIZE, 1, 1),
        )
    ctx.enqueue_function[query_softmax_exponents_kernel](
        s.query.weights.unsafe_ptr(), has_w, s.query.qids.unsafe_ptr(),
        Int32(size), s.query_approx.unsafe_ptr(), s.approx_exp.unsafe_ptr(),
        s.beta,
        grid_dim=(t_blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )
    if q_blocks > 0:
        ctx.enqueue_function[compute_group_sums_kernel](
            s.approx_exp.unsafe_ptr(), s.query.q_offsets.unsafe_ptr(), UInt32(0),
            s.query.q_sizes.unsafe_ptr(), Int32(q_count),
            s.query_approx.unsafe_ptr(),
            grid_dim=(q_blocks, 1, 1),
            block_dim=(QUERY_HELPER_BLOCK_SIZE, 1, 1),
        )
    var blocks = (size + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    ctx.enqueue_function[query_softmax_kernel[estimation, second_order]](
        s.query.targets.unsafe_ptr(), s.query.weights.unsafe_ptr(), has_w,
        s.approx_exp.unsafe_ptr(), s.query.qids.unsafe_ptr(), Int32(size),
        s.query_approx.unsafe_ptr(), s.query_sum_wt.unsafe_ptr(),
        s.lambda_reg, s.beta,
        idx_ptr, Int32(1) if use_inverse else Int32(0),
        stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(MSE_BLOCK_SIZE, 1, 1),
    )
