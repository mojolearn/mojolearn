# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reference: `catboost/cuda/targets/kernel/pair_logit.cu`
(`PairLogitPointwiseTargetImpl`, `:10-58`; `MakePairWeightsImpl`, `:61-72`), the
host kernel that launches it, `TPairLogitKernel::Run` (`targets/kernel.h:
524-556`), and `TQuerywiseTargetsImpl::InitPairLogit`
(`targets/querywise_targets_impl.h:326-346`), the PairLogit arm of the
querywise target.

THE PAIRS. `pairs` holds two row indices per pair, winner then loser, in the
order the reference's `TQueriesGrouping` flattens them (query by query, winner
row by winner row, each winner's competitors in insertion order,
`gpu_data/samples_grouping.h:79-101`); `gbdt/train.mojo::train` builds or
checks that order. `pair_weights` is one weight per pair.

THE ARITHMETIC PER PAIR `i` (`pair_logit.cu:25-40`):

    diff = point[winner] - point[loser];  expDiff = exp(diff)
    p = max(min(isfinite(1 + expDiff) ? expDiff / (1 + expDiff) : 1, 1 - 1e-40), 1e-40)
    direction = 1 - p;  scale = p * (1 - p)
    der[winner] += w * direction;  der[loser] += -w * direction
    der2[winner] += w * scale;     der2[loser] += w * scale
    functionValue += w * (diff - (isfinite(1 + expDiff) ? log(1 + expDiff) : diff))

The point is read in ROW order: directly on the structure search, and through
the inverse of the oracle's bin order during leaf estimation, whose derivatives
land at each row's bin position (`writeMap = Indices`, the same convention as
`gbdt/targets/kernel/query_rmse.mojo`).

================= DEVIATION BLOCK: the per-row sums get one order =================
The reference sums each row's pair contributions with `atomicAdd` over the pair
threads, so the float result depends on which thread lands first, and
`MakePairWeightsImpl` does the same for the per-row pair weights. Here:

  1. `make_pairwise_target_buffers` builds, once per fit, each row's ENDPOINT
     LIST: the codes `pair << 1` (the row is the winner) and `pair << 1 | 1`
     (the loser), in increasing pair index.
  2. `pair_logit_pair_kernel` stores `w * direction` and `w * scale` per pair
     (each flushed at derivation, IDENTITY_PATHS row 10) and the value partial
     of every 256-PAIR block through `pinned_block_sum`.
  3. `pair_logit_row_kernel` folds each row's list SEQUENTIALLY from 0.0,
     `acc = acc + contribution`, the loser's contribution being `-(w * direction)`.
     One thread per row, no shared memory, no per-segment block grid.

So a row's sum has one order on every vendor and on the host
(`gbdt/host/gbdt_oracle_pair.mojo` restates the same loop). The per-row pair
weights take the same fold on the host, in `make_pairwise_target_buffers`.
==================================================================================

================= DEVIATION BLOCK: the search weight plane =================
The reference's querywise `StochasticDer` (`querywise_targets_impl.h:161-181`)
reads `if (secondDerAsWeights) GradientAt(...) else NewtonAt(...)`. The flag
arrives unchanged from `IsSecondOrderScoreFunction(scoreFunction)`
(`greedy_search_helper.cpp:286-296`, `weak_objective_impl.h:21-45`,
`target_func.h:346-356`), so a Newton score gets the row weights and Cosine/L2
get the second derivatives: the reverse of the flag's name and of the pointwise
target (`pointwise_target_impl.h:173-216`), where true writes der2 into the
weight column. QueryRMSE cannot show it (its der2 is its weight). PairLogit
can, so the search plane 0 here follows the flag's meaning: the per-row pair
weights under Cosine and L2, the der2 sums under NewtonL2 and NewtonCosine.
The CatBoost CPU learner, the comparison this repository is measured against,
weights its Cosine score by the row weights too.
============================================================================

DEVIATION (partials, as the pointwise family's): `functionValue` and the plane
magnitudes are per-block partials through `pinned_block_sum`, folded by
`deterministic_sum_lanes_kernel`, where theirs is a 1024-thread
`FastInBlockReduce` plus an `atomicAdd`. The value partials are per 256 PAIRS,
the magnitudes per 256 ROWS.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from gbdt.data.pairs import prepare_pairs
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_f32
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    pinned_block_sum,
    routed_exp,
    routed_log,
)

def pair_blocks(n_pairs: Int) -> Int:
    """Value partials: one per 256 pairs (at least one, so an empty buffer is
    never allocated)."""
    var b = (n_pairs + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    return b if b > 0 else 1


struct PairwiseTargetBuffers(Movable):
    """The pairs, their weights and the fit-long endpoint lists on the device,
    the per-row pair weights (the reference's replaced target weights,
    `InitPairLogit`), the per-pair scratch, and the gathered point and inverse
    order the estimator uses. Built once per fit by
    `make_pairwise_target_buffers`; `handles()` gives views onto the same
    memory."""

    var n_rows: Int
    var n_pairs: Int
    #: `PairsTotalWeight` (`querywise_targets_impl.h:336-344`), a Float64 sum in
    #: pair order on the host; it normalizes the learn loss
    var pairs_total_weight: Float64
    var d_pairs: DeviceBuffer[DType.uint32]
    var d_pair_w: DeviceBuffer[DType.float32]
    var d_ep_offsets: DeviceBuffer[DType.uint32]
    var d_ep_codes: DeviceBuffer[DType.uint32]
    var d_row_weights: DeviceBuffer[DType.float32]
    var d_pair_dir: DeviceBuffer[DType.float32]
    var d_pair_scale: DeviceBuffer[DType.float32]
    var d_point: DeviceBuffer[DType.float32]
    var inverse: DeviceBuffer[DType.uint32]
    var no_indices: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        n_rows: Int,
        n_pairs: Int,
        pairs_total_weight: Float64,
        var d_pairs: DeviceBuffer[DType.uint32],
        var d_pair_w: DeviceBuffer[DType.float32],
        var d_ep_offsets: DeviceBuffer[DType.uint32],
        var d_ep_codes: DeviceBuffer[DType.uint32],
        var d_row_weights: DeviceBuffer[DType.float32],
        var d_pair_dir: DeviceBuffer[DType.float32],
        var d_pair_scale: DeviceBuffer[DType.float32],
        var d_point: DeviceBuffer[DType.float32],
        var inverse: DeviceBuffer[DType.uint32],
        var no_indices: DeviceBuffer[DType.uint32],
    ):
        self.n_rows = n_rows
        self.n_pairs = n_pairs
        self.pairs_total_weight = pairs_total_weight
        self.d_pairs = d_pairs^
        self.d_pair_w = d_pair_w^
        self.d_ep_offsets = d_ep_offsets^
        self.d_ep_codes = d_ep_codes^
        self.d_row_weights = d_row_weights^
        self.d_pair_dir = d_pair_dir^
        self.d_pair_scale = d_pair_scale^
        self.d_point = d_point^
        self.inverse = inverse^
        self.no_indices = no_indices^

    def handles(self) -> PairwiseTargetBuffers:
        """Handle copies onto the same device memory."""
        return PairwiseTargetBuffers(
            self.n_rows, self.n_pairs, self.pairs_total_weight,
            self.d_pairs.copy(), self.d_pair_w.copy(),
            self.d_ep_offsets.copy(), self.d_ep_codes.copy(),
            self.d_row_weights.copy(), self.d_pair_dir.copy(),
            self.d_pair_scale.copy(), self.d_point.copy(),
            self.inverse.copy(), self.no_indices.copy(),
        )

    def blocks(self) -> Int:
        return pair_blocks(self.n_pairs)


def make_pairwise_target_buffers(
    ctx: DeviceContext,
    winners: List[UInt32],
    losers: List[UInt32],
    pair_weights: List[Float32],
    n_rows: Int,
) raises -> PairwiseTargetBuffers:
    """Upload the pairs, their weights, the endpoint lists and the per-row
    pair weights, and allocate the scratch."""
    var prep = prepare_pairs(winners, losers, pair_weights, n_rows)
    var n_pairs = len(winners)
    var h_pairs = ctx.enqueue_create_host_buffer[DType.uint32](2 * n_pairs)
    var h_pw = ctx.enqueue_create_host_buffer[DType.float32](n_pairs)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_rows + 1)
    var h_codes = ctx.enqueue_create_host_buffer[DType.uint32](2 * n_pairs)
    var h_rw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for p in range(n_pairs):
        h_pairs.unsafe_ptr().unsafe_store(2 * p, winners[p])
        h_pairs.unsafe_ptr().unsafe_store(2 * p + 1, losers[p])
        h_pw.unsafe_ptr().unsafe_store(p, pair_weights[p])
    for r in range(n_rows + 1):
        h_off.unsafe_ptr().unsafe_store(r, prep.offsets[r])
    for k in range(2 * n_pairs):
        h_codes.unsafe_ptr().unsafe_store(k, prep.codes[k])
    for r in range(n_rows):
        h_rw.unsafe_ptr().unsafe_store(r, prep.row_weights[r])
    var d_pairs = ctx.enqueue_create_buffer[DType.uint32](2 * n_pairs)
    var d_pw = ctx.enqueue_create_buffer[DType.float32](n_pairs)
    var d_off = ctx.enqueue_create_buffer[DType.uint32](n_rows + 1)
    var d_codes = ctx.enqueue_create_buffer[DType.uint32](2 * n_pairs)
    var d_rw = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_buf=d_pairs, src_ptr=h_pairs.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_pw, src_ptr=h_pw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_codes, src_ptr=h_codes.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_rw, src_ptr=h_rw.unsafe_ptr())
    var d_dir = ctx.enqueue_create_buffer[DType.float32](n_pairs)
    var d_scale = ctx.enqueue_create_buffer[DType.float32](n_pairs)
    var d_point = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var inverse = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var no_indices = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.synchronize()
    _ = h_pairs^  # past the drain (step-33 race class)
    _ = h_pw^
    _ = h_off^
    _ = h_codes^
    _ = h_rw^
    return PairwiseTargetBuffers(
        n_rows, n_pairs, prep.total, d_pairs^, d_pw^, d_off^, d_codes^, d_rw^,
        d_dir^, d_scale^, d_point^, inverse^, no_indices^,
    )


def pair_logit_pair_kernel(
    point: MutPointer[Float32, MutAnyOrigin],
    pairs: MutPointer[UInt32, MutAnyOrigin],
    pair_w: MutPointer[Float32, MutAnyOrigin],
    n_pairs_in: Int32,
    pair_dir: MutPointer[Float32, MutAnyOrigin],
    pair_scale: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
):
    """`PairLogitPointwiseTargetImpl` (`pair_logit.cu:10-58`), per pair: the
    stored `w * direction` and `w * scale` and the value partial. See the
    module docstring for what differs."""
    var n_pairs = Int(n_pairs_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var in_range = i < n_pairs
    var w = Float32(1.0)
    var diff = Float32(0.0)
    if in_range:
        var a = Int(pairs.unsafe_load(2 * i))
        var b = Int(pairs.unsafe_load(2 * i + 1))
        w = pair_w.unsafe_load(i)
        diff = point.unsafe_load(a) - point.unsafe_load(b)
    var exp_diff = routed_exp(diff)
    var p = Float32(1.0)
    if isfinite(Float32(1.0) + exp_diff):
        p = exp_diff / (Float32(1.0) + exp_diff)
    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
    var direction = Float32(1.0) - p
    var scale = ftz(p * (Float32(1.0) - p))
    if in_range:
        pair_dir.unsafe_store(i, ftz(w * direction))
        pair_scale.unsafe_store(i, ftz(w * scale))
    if compute_fv != Int32(0):
        var score = Float32(0.0)
        if in_range:
            var log_exp_val_plus_one = diff
            if isfinite(Float32(1.0) + exp_diff):
                log_exp_val_plus_one = routed_log(Float32(1.0) + exp_diff)
            score = w * (diff - log_exp_val_plus_one)
        var total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)


def pair_logit_row_kernel[estimation: Bool, second_order: Bool](
    ep_offsets: MutPointer[UInt32, MutAnyOrigin],
    ep_codes: MutPointer[UInt32, MutAnyOrigin],
    pair_dir: MutPointer[Float32, MutAnyOrigin],
    pair_scale: MutPointer[Float32, MutAnyOrigin],
    row_weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """The per-row sums in the pinned order (the first deviation block), then
    the stats planes: SEARCH `[weight-or-der2, der]` at the row, ESTIMATION
    `[der, der2]` at `write_map[row]`."""
    comptime assert not (estimation and second_order), (
        "second_order is a SEARCH-mode flag"
    )
    var n_rows = Int(n_rows_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var in_range = i < n_rows
    var acc_der = Float32(0.0)
    var acc_der2 = Float32(0.0)
    var weight = Float32(0.0)
    if in_range:
        var k = Int(ep_offsets.unsafe_load(i))
        var end = Int(ep_offsets.unsafe_load(i + 1))
        while k < end:
            var code = ep_codes.unsafe_load(k)
            var p = Int(code >> UInt32(1))
            if (code & UInt32(1)) != UInt32(0):
                acc_der = acc_der + (-pair_dir.unsafe_load(p))
            else:
                acc_der = acc_der + pair_dir.unsafe_load(p)
            acc_der2 = acc_der2 + pair_scale.unsafe_load(p)
            k += 1
        weight = row_weights.unsafe_load(i)
    var der = ftz(acc_der)
    var der2 = ftz(acc_der2)
    var plane0 = weight
    comptime if second_order:
        plane0 = der2
    if in_range:
        comptime if estimation:
            var dst = i
            if has_write_map != Int32(0):
                dst = Int(write_map.unsafe_load(i))
            stats.unsafe_store(dst, der)
            stats.unsafe_store(n_rows + dst, der2)
        else:
            stats.unsafe_store(i, plane0)
            stats.unsafe_store(n_rows + i, der)
    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            w_abs = abs(plane0)
            g_abs = abs(der)
        var w_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x) + 1, g_total)


def launch_pair_logit_with[estimation: Bool, second_order: Bool](
    ctx: DeviceContext,
    mut q: PairwiseTargetBuffers,
    mut predictions: DeviceBuffer[DType.float32],
    use_inverse: Bool,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`TPairLogitKernel::Run` (`kernel.h:524-556`) with the per-row fold:
    the point in row order (gathered through `q.inverse` when `use_inverse`,
    the estimator's bin order), the pair kernel over `q.blocks()` 256-pair
    blocks, then the row kernel over 256-row blocks. `function_value` holds
    `q.blocks()` partials; `plane_magnitudes` holds two per 256-row block."""
    var n_rows = q.n_rows
    var n_pairs = q.n_pairs
    var row_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    if use_inverse:
        launch_gather_with_mask_f32(
            ctx, q.d_point, predictions, q.inverse, n_rows, UInt32(0xFFFFFFFF)
        )
        ctx.enqueue_function[pair_logit_pair_kernel](
            q.d_point.unsafe_ptr(), q.d_pairs.unsafe_ptr(), q.d_pair_w.unsafe_ptr(),
            Int32(n_pairs), q.d_pair_dir.unsafe_ptr(), q.d_pair_scale.unsafe_ptr(),
            function_value.unsafe_ptr(), Int32(1) if compute_fv else Int32(0),
            grid_dim=(q.blocks(), 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_function[pair_logit_row_kernel[estimation, second_order]](
            q.d_ep_offsets.unsafe_ptr(), q.d_ep_codes.unsafe_ptr(),
            q.d_pair_dir.unsafe_ptr(), q.d_pair_scale.unsafe_ptr(),
            q.d_row_weights.unsafe_ptr(), Int32(n_rows),
            q.inverse.unsafe_ptr(), Int32(1),
            stats.unsafe_ptr(), plane_magnitudes.unsafe_ptr(),
            Int32(1) if compute_magnitudes else Int32(0),
            grid_dim=(row_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
    else:
        ctx.enqueue_function[pair_logit_pair_kernel](
            predictions.unsafe_ptr(), q.d_pairs.unsafe_ptr(), q.d_pair_w.unsafe_ptr(),
            Int32(n_pairs), q.d_pair_dir.unsafe_ptr(), q.d_pair_scale.unsafe_ptr(),
            function_value.unsafe_ptr(), Int32(1) if compute_fv else Int32(0),
            grid_dim=(q.blocks(), 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_function[pair_logit_row_kernel[estimation, second_order]](
            q.d_ep_offsets.unsafe_ptr(), q.d_ep_codes.unsafe_ptr(),
            q.d_pair_dir.unsafe_ptr(), q.d_pair_scale.unsafe_ptr(),
            q.d_row_weights.unsafe_ptr(), Int32(n_rows),
            q.no_indices.unsafe_ptr(), Int32(0),
            stats.unsafe_ptr(), plane_magnitudes.unsafe_ptr(),
            Int32(1) if compute_magnitudes else Int32(0),
            grid_dim=(row_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
