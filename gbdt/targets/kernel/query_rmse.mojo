# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reference: `catboost/cuda/targets/kernel/query_rmse.cu` (`QueryRmseImpl`,
`:12-51`) and the host kernel that launches it, `TQueryRmseKernel::Run`
(`catboost/cuda/targets/kernel.h:196-226`), the QueryRMSE arm of
`TQuerywiseTargetsImpl::ApproximateForPermutation`
(`targets/querywise_targets_impl.h:209-222`).

THEIR SEQUENCE, IN ORDER, per call:

    MseDer = Indices ? Gather(Predictions, Indices) : copy(Predictions)
    MseDer *= -1
    MseDer += Relevs                                  (row order, not gathered)
    QueryMeans = ComputeGroupMeans(MseDer, Weights, offsets, bias, sizes)
    Qids       = ComputeGroupIds(sizes, offsets, bias)
    QueryRmseImpl(MseDer, Weights, Qids, QueryMeans, writeMap = Indices, ...)

`launch_approximate_query_rmse` launches exactly that. The first three
lines are one per-element kernel here (`query_rmse_residual_kernel`): each
is an element-wise float operation with no reduction, so doing the three in
one pass stores the same bytes as three passes. The group means and ids are
`gbdt/gpu_data/kernel/query_helper.mojo`.

THE KERNEL (`query_rmse.cu:12-51`), per row `i` in ROW order:

    val = diffs[i];  queryMean = queryMeans[qids[i]];
    direction = val - queryMean;  weight = weights ? weights[i] : 1
    der[writeMap[i]]  = weight * direction
    der2[writeMap[i]] = weight
    functionValue    += -weight * (val - queryMean) * (val - queryMean)

`writeMap` is null on the structure search (`GradientAt` / `NewtonAt` pass
no indices, `querywise_targets_impl.h:129-157`) and is the INVERSE of the
oracle's bin order during leaf estimation
(`permutation_der_calcer.h:176-200`): the point arrives bin-ordered, the
gather brings it back to row order so the query means are taken over each
query's own rows, and the derivatives are written back bin-ordered.

THE STATS LAYOUT, the two modes `pointwise_target_kernel` already records:

  - `estimation=False`, the SEARCH: plane 0 is what their `StochasticDer`
    hands the searcher as weights, and for QueryRMSE both of its arms give
    the row weight (`GradientAt` copies the weights, `NewtonAt` writes
    `der2 = weight`), so plane 0 is the weight whatever the score function;
    plane 1 is `weight * direction`. The two magnitudes are `|plane 0|` and
    `|plane 1|`, as the fixed-point scale reads them.
  - `estimation=True`, the ORACLE's `ApproximateAt`: plane 0 `der`, plane 1
    `der2`, both at `writeMap[i]`.

DEVIATIONS, both the pointwise family's and for the same reasons:

  * `functionValue` and the magnitudes are PER-BLOCK PARTIALS through
    `pinned_block_sum` at the 256-thread block the pointwise kernels use,
    folded by `deterministic_sum_lanes_kernel`, where theirs is a 1024-thread
    `FastInBlockReduce` plus an `atomicAdd` (the 2026-08-21 determinism fix;
    one fold shape on every vendor).
  * `weight * direction` passes through `ftz` at derivation (IDENTITY_PATHS
    row 10), a comptime no-op outside IDENTICAL.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from gbdt.gpu_data.kernel.query_helper import (
    launch_compute_group_ids,
    launch_compute_group_means,
)
from gbdt.gpu_util.kernel.transform import TRANSFORM_BLOCK_SIZE
from gbdt.targets.kernel.pointwise_targets import MSE_BLOCK_SIZE, pinned_block_sum


struct QuerywiseTargetBuffers(Movable):
    """The pool's grouping on the device plus the scratch one
    `launch_approximate_query_rmse` writes: their `TGpuSamplesGrouping`
    offsets and sizes at bias 0 (`gpu_data/samples_grouping_gpu.h:23-33`)
    and `TQueryRmseKernel::PrepareContext`'s three allocations
    (`targets/kernel.h:158-164`). `inverse` holds the inverse of a bin
    order when the leaf estimator needs one (`permutation_der_calcer.h:
    176-183`); `no_indices` is the one-word stand-in the search pass hands
    the kernel where their pointer is null.

    Built once per fit by `make_querywise_target_buffers`. `DeviceBuffer.copy()`
    copies the HANDLE, so `handles()` gives the estimator views onto the same
    memory the boosting loop owns."""

    var q_count: Int
    var n_rows: Int
    #: the ROW-ORDER targets and weights (the estimator's own copies are
    #: gathered into bin order; this target reads the pool's order)
    var targets: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var has_weights: Bool
    var q_offsets: DeviceBuffer[DType.uint32]
    var q_sizes: DeviceBuffer[DType.uint32]
    var mse_der: DeviceBuffer[DType.float32]
    var query_means: DeviceBuffer[DType.float32]
    var qids: DeviceBuffer[DType.uint32]
    var inverse: DeviceBuffer[DType.uint32]
    var no_indices: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        q_count: Int,
        n_rows: Int,
        var targets: DeviceBuffer[DType.float32],
        var weights: DeviceBuffer[DType.float32],
        has_weights: Bool,
        var q_offsets: DeviceBuffer[DType.uint32],
        var q_sizes: DeviceBuffer[DType.uint32],
        var mse_der: DeviceBuffer[DType.float32],
        var query_means: DeviceBuffer[DType.float32],
        var qids: DeviceBuffer[DType.uint32],
        var inverse: DeviceBuffer[DType.uint32],
        var no_indices: DeviceBuffer[DType.uint32],
    ):
        self.q_count = q_count
        self.n_rows = n_rows
        self.targets = targets^
        self.weights = weights^
        self.has_weights = has_weights
        self.q_offsets = q_offsets^
        self.q_sizes = q_sizes^
        self.mse_der = mse_der^
        self.query_means = query_means^
        self.qids = qids^
        self.inverse = inverse^
        self.no_indices = no_indices^

    def handles(self) -> QuerywiseTargetBuffers:
        """Handle copies onto the same device memory."""
        return QuerywiseTargetBuffers(
            self.q_count, self.n_rows,
            self.targets.copy(), self.weights.copy(), self.has_weights,
            self.q_offsets.copy(), self.q_sizes.copy(),
            self.mse_der.copy(), self.query_means.copy(), self.qids.copy(),
            self.inverse.copy(), self.no_indices.copy(),
        )


def make_querywise_target_buffers(
    ctx: DeviceContext,
    group_sizes: List[UInt32],
    n_rows: Int,
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
) raises -> QuerywiseTargetBuffers:
    """Upload the grouping as offsets and sizes and allocate the scratch.
    The sizes must be positive and cover `n_rows`; the caller checked that
    (`gbdt/train.mojo::train`) and it is checked again here because this
    function has callers other than `train`."""
    var q_count = len(group_sizes)
    if q_count < 1:
        raise Error("QueryRMSE needs at least one query")
    var h_offsets = ctx.enqueue_create_host_buffer[DType.uint32](q_count)
    var h_sizes = ctx.enqueue_create_host_buffer[DType.uint32](q_count)
    var at = 0
    for q in range(q_count):
        var size = Int(group_sizes[q])
        if size < 1:
            raise Error("QueryRMSE: query " + String(q) + " has no rows")
        h_offsets.unsafe_ptr().unsafe_store(q, UInt32(at))
        h_sizes.unsafe_ptr().unsafe_store(q, UInt32(size))
        at += size
    if at != n_rows:
        raise Error(
            "QueryRMSE: the query sizes cover " + String(at) + " rows of "
            + String(n_rows)
        )
    var q_offsets = ctx.enqueue_create_buffer[DType.uint32](q_count)
    var q_sizes = ctx.enqueue_create_buffer[DType.uint32](q_count)
    ctx.enqueue_copy(dst_buf=q_offsets, src_ptr=h_offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=q_sizes, src_ptr=h_sizes.unsafe_ptr())
    var mse_der = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var query_means = ctx.enqueue_create_buffer[DType.float32](q_count)
    var qids = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var inverse = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var no_indices = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.synchronize()
    _ = h_offsets^  # past the drain (step-33 race class)
    _ = h_sizes^
    return QuerywiseTargetBuffers(
        q_count, n_rows, targets.copy(), weights.copy(), has_weights,
        q_offsets^, q_sizes^, mse_der^, query_means^, qids^, inverse^,
        no_indices^,
    )


def query_rmse_residual_kernel(
    predictions: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    has_indices: Int32,
    relevs: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    mse_der: MutPointer[Float32, MutAnyOrigin],
):
    """`TQueryRmseKernel::Run`'s first three statements (`kernel.h:208-215`),
    per row: the gather (or the copy), `MultiplyVector(-1)`, `AddVector`."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var p: Float32
        if has_indices != Int32(0):
            p = predictions.unsafe_load(Int(indices.unsafe_load(i)))
        else:
            p = predictions.unsafe_load(i)
        var v = p * Float32(-1.0)
        v = v + relevs.unsafe_load(i)
        mse_der.unsafe_store(i, v)
        i += step


def query_rmse_kernel[estimation: Bool](
    diffs: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    qids: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    query_means: MutPointer[Float32, MutAnyOrigin],
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`QueryRmseImpl` (`query_rmse.cu:12-51`); see the module docstring for
    the two modes and the partials."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var in_range = i < size

    # `val = i < size ? diffs[i] : 0`, `queryMean = i < size ? means[qids[i]] : 0`
    var val = Float32(0.0)
    var query_mean = Float32(0.0)
    var weight = Float32(1.0)
    if in_range:
        val = diffs.unsafe_load(i)
        query_mean = query_means.unsafe_load(Int(qids.unsafe_load(i)))
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)
    var direction = val - query_mean
    var der = ftz(weight * direction)

    if in_range:
        comptime if estimation:
            var dst = i
            if has_write_map != Int32(0):
                dst = Int(write_map.unsafe_load(i))
            stats.unsafe_store(dst, der)
            stats.unsafe_store(size + dst, weight)
        else:
            stats.unsafe_store(i, weight)
            stats.unsafe_store(size + i, der)

    # `tmpScores[x] = i < size ? -weight * (val - queryMean) * (val - queryMean) : 0`
    if compute_fv != Int32(0):
        var score = Float32(0.0)
        if in_range:
            score = -weight * (val - query_mean) * (val - query_mean)
        var total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            w_abs = abs(weight)
            g_abs = abs(der)
        var w_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x) + 1, g_total)


def launch_query_rmse_with[estimation: Bool](
    ctx: DeviceContext,
    mut q: QuerywiseTargetBuffers,
    mut predictions: DeviceBuffer[DType.float32],
    use_inverse: Bool,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`launch_approximate_query_rmse` over the fit's grouping: the search
    and final passes read the point in row order (`use_inverse` False, their
    null `Indices`); the leaf estimator reads it through `q.inverse`, which
    the caller filled for this tree's bin order."""
    if use_inverse:
        launch_approximate_query_rmse[estimation](
            ctx, q.targets, q.weights, q.has_weights, predictions, q.inverse,
            True, q.n_rows, q.q_offsets, q.q_sizes, q.q_count, q.mse_der,
            q.query_means, q.qids, stats, function_value, compute_fv,
            plane_magnitudes, compute_magnitudes,
        )
    else:
        launch_approximate_query_rmse[estimation](
            ctx, q.targets, q.weights, q.has_weights, predictions,
            q.no_indices, False, q.n_rows, q.q_offsets, q.q_sizes,
            q.q_count, q.mse_der, q.query_means, q.qids, stats,
            function_value, compute_fv, plane_magnitudes, compute_magnitudes,
        )


def launch_approximate_query_rmse[estimation: Bool](
    ctx: DeviceContext,
    mut relevs: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut predictions: DeviceBuffer[DType.float32],
    mut indices: DeviceBuffer[DType.uint32],
    has_indices: Bool,
    size: Int,
    mut q_offsets: DeviceBuffer[DType.uint32],
    mut q_sizes: DeviceBuffer[DType.uint32],
    q_count: Int,
    mut mse_der: DeviceBuffer[DType.float32],
    mut query_means: DeviceBuffer[DType.float32],
    mut qids: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`TQueryRmseKernel::Run` (`kernel.h:196-226`), one device, offsets
    bias 0. `indices` is their `Indices`, the oracle's inverse bin order,
    unread when `has_indices` is False; it is also the kernel's write map.
    `relevs` and `weights` are in ROW order. Scratch `mse_der` and `qids`
    hold `size` values and `query_means` holds `q_count`, their
    `PrepareContext` allocations (`kernel.h:158-164`)."""
    if size <= 0:
        return
    var t_blocks = (size + TRANSFORM_BLOCK_SIZE - 1) // TRANSFORM_BLOCK_SIZE
    ctx.enqueue_function[query_rmse_residual_kernel](
        predictions.unsafe_ptr(), indices.unsafe_ptr(),
        Int32(1) if has_indices else Int32(0),
        relevs.unsafe_ptr(), Int32(size), mse_der.unsafe_ptr(),
        grid_dim=(t_blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )
    launch_compute_group_means(
        ctx, mse_der, weights, has_weights, q_offsets, UInt32(0), q_sizes,
        q_count, query_means,
    )
    launch_compute_group_ids(ctx, q_sizes, q_offsets, UInt32(0), q_count, qids)
    var blocks = (size + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    ctx.enqueue_function[query_rmse_kernel[estimation]](
        mse_der.unsafe_ptr(), weights.unsafe_ptr(),
        Int32(1) if has_weights else Int32(0),
        qids.unsafe_ptr(), Int32(size), query_means.unsafe_ptr(),
        indices.unsafe_ptr(), Int32(1) if has_indices else Int32(0),
        stats.unsafe_ptr(), function_value.unsafe_ptr(),
        Int32(1) if compute_fv else Int32(0),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(MSE_BLOCK_SIZE, 1, 1),
    )
