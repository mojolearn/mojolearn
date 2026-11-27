"""Fit-owned buffers bridging query derivatives to the existing tree search and leaf walker."""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gbdt.data.group_layout import GroupLayout
from gbdt.gpu_data.kernel.query_helper import launch_compute_group_ids
from gbdt.targets.kernel.lightgbm_ranking import (
    LightGBMRankWorkspace, launch_lightgbm_ranking, launch_ranking_metric,
)
from gbdt.targets.lightgbm_ranking import LR_CUTOFF
from gbdt.targets.kernel.pointwise_targets import MSE_BLOCK_SIZE, pinned_block_sum


def _search_planes(
    raw: MutPointer[Float32, MutAnyOrigin], weights: MutPointer[Float32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin], mags: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32, second_order: Int32, compute_mags: Int32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var w = Float32(0)
    var d = Float32(0)
    if row < n:
        d = raw.unsafe_load(row)
        w = raw.unsafe_load(n + row) if second_order != 0 else weights.unsafe_load(row)
        stats.unsafe_store(row, w)
        stats.unsafe_store(n + row, d)
    if compute_mags != 0:
        var wm = pinned_block_sum[block_size=MSE_BLOCK_SIZE](abs(w))
        var dm = pinned_block_sum[block_size=MSE_BLOCK_SIZE](abs(d))
        if thread_idx.x == 0:
            mags.unsafe_store(2 * Int(block_idx.x), wm)
            mags.unsafe_store(2 * Int(block_idx.x) + 1, dm)


struct DeviceRankingMetric(Movable):
    var workspace: LightGBMRankWorkspace
    var layout: GroupLayout
    var qids: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, var layout: GroupLayout, n: Int) raises:
        self.layout = layout^
        self.qids = ctx.enqueue_create_buffer[DType.uint32](n)
        self.workspace = LightGBMRankWorkspace(ctx, n, self.layout.n_groups, self.layout.max_size)
        launch_compute_group_ids(ctx, self.layout.sizes, self.layout.offsets,
                                 UInt32(0), self.layout.n_groups, self.qids)

    def metric[reuse_terms: Bool = False](mut self, ctx: DeviceContext, mut targets: DeviceBuffer[DType.float32],
               mut scores: DeviceBuffer[DType.float32], n: Int,
               mut result: DeviceBuffer[DType.float32], objective: Int,
               mut weights: DeviceBuffer[DType.float32], has_weights: Bool, cutoff: Int = LR_CUTOFF) raises:
        launch_ranking_metric[reuse_terms](ctx, self.workspace, objective, targets, scores, weights, has_weights, self.layout.offsets,
                             self.layout.sizes, self.qids, n, self.layout.n_groups, result, cutoff)


struct DeviceRankingFit(Movable):
    var metric_state: DeviceRankingMetric
    # Unsampled [direction, Hessian] in row order; the search may snap or
    # bootstrap its own planes after this, without changing the Newton update.
    var raw: DeviceBuffer[DType.float32]
    var directions: DeviceBuffer[DType.float32]
    var hessians: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, var layout: GroupLayout, n: Int) raises:
        self.metric_state = DeviceRankingMetric(ctx, layout^, n)
        self.raw = ctx.enqueue_create_buffer[DType.float32](2 * n)
        self.directions = self.raw.create_sub_buffer[DType.float32](0, n)
        self.hessians = self.raw.create_sub_buffer[DType.float32](n, n)

    def search(mut self, ctx: DeviceContext, objective: Int, mut targets: DeviceBuffer[DType.float32],
               mut scores: DeviceBuffer[DType.float32], mut weights: DeviceBuffer[DType.float32],
               has_weights: Bool, n: Int, seed: Int, iteration: Int, second_order: Bool,
               mut stats: DeviceBuffer[DType.float32], mut fv: DeviceBuffer[DType.float32],
               mut mags: DeviceBuffer[DType.float32], compute_mags: Bool) raises:
        launch_lightgbm_ranking(ctx, self.metric_state.workspace, objective, targets, scores, weights,
                                has_weights, self.metric_state.layout.offsets, self.metric_state.layout.sizes, self.metric_state.qids,
                                n, self.metric_state.layout.n_groups, seed, iteration, self.directions, self.hessians)
        self.metric_state.metric[True](ctx, targets, scores, n, fv, objective, weights, has_weights)
        ctx.enqueue_function[_search_planes](
            self.raw.unsafe_ptr(), weights.unsafe_ptr(), stats.unsafe_ptr(), mags.unsafe_ptr(),
            Int32(n), Int32(1) if second_order else Int32(0), Int32(1) if compute_mags else Int32(0),
            grid_dim=((n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
