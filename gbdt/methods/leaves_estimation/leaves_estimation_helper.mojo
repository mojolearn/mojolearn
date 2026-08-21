"""`ComputeWeightedQuantile` and `ComputeExactApprox`: the Exact driver.

PORT OF `catboost/cuda/methods/leaves_estimation/leaves_estimation_helper.h`
at CatBoost `54a8143a` -- `ComputeWeightedQuantile` (`:64-146`) and
`ComputeExactApprox` (`:148-185`). Transliterated. Do not improve.

Only those two functions are ported. The rest of their header is pairwise
and groupwise machinery (`MakeSupportPairsMatrix`, `ReorderPairs`,
`FilterZeroLeafBins`) belonging to the ranking oracles, which are NOT
PORTED; porting a function nothing reaches is the defect PORTING_RULES 3
names.

## The step of theirs this file DOES NOT do, and why that is right

Theirs opens with `ComputeByLeafOrder` (`:88-94`): radix-sort the bin ids,
carry an index permutation, and `Gather` targets and weights into leaf
order. This port skips it, because the rows ARRIVE in leaf order --
`split_points`' gather leaves `row_index` exactly bin-sorted, and
`doc_parallel_boosting` gathers target / weights / cursor by it before the
oracle is built. Their factory sorts; we inherit. That is the same
inheritance `pointwise_oracle.mojo`'s header already records for the
Newton path, and it is not a shortcut: the segment boundaries used here are
the searcher's OWN exported offsets, which is the only correct
segmentation, since the final partitions sit in memory in bit-reversed
leaf order and a prefix sum of the sizes would be the wrong one.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK
from gbdt.gpu_util.kernel.segmented_sort import (
    float_key_kernel,
    gather_f32_by_index_kernel,
    launch_segmented_radix_sort,
)
from gbdt.gpu_util.kernel.segmented_scan import launch_segmented_scan_vector
from gbdt.methods.kernel.exact_estimation import (
    BINARY_SEARCH_ITERATIONS,
    NEED_WEIGHTS_BLOCK,
    QUANTILE_SEARCH_BLOCK,
    WEIGHTS_WITH_TARGETS_BLOCK,
    compute_need_weights_kernel,
    compute_weighted_quantile_kernel,
    compute_weights_with_targets_kernel,
    make_end_of_bins_flags_kernel,
)

#: their `SegmentedRadixSort(..., 10, 32)` (`leaves_estimation_helper.h
#: :110-112`). The bottom ten mantissa bits are NOT sorted on; see
#: `segmented_sort.mojo` for what that costs.
comptime EXACT_SORT_FIRST_BIT = 10
comptime EXACT_SORT_LAST_BIT = 32

#: `SegmentedScanVector(orderedWeights, endOfBinsFlags, weightsPrefixSum,
#: true, 1)` (`:121`): inclusive, flag in BIT ZERO.
comptime EXACT_FLAG_MASK = 1


@fieldwise_init
struct ExactQuantileScratch(Movable):
    """Every buffer `ComputeWeightedQuantile` creates, hoisted to the
    oracle so a ten-iteration walker does not re-allocate ten times.

    Theirs allocates inside the function (`:78-134`, eleven
    `TSingleBuffer::Create` calls). Ours cannot: the Exact estimator runs
    once per TREE here, and the per-tree fixed cost is the number this
    port is judged on (`PERF_2026-08-20_fixed-cost.md`), so the buffers
    belong to the fit. No arithmetic depends on where they live.
    """

    var keys: DeviceBuffer[DType.uint32]
    var index: DeviceBuffer[DType.uint32]
    var temp_keys: DeviceBuffer[DType.uint32]
    var temp_index: DeviceBuffer[DType.uint32]
    var ordered_targets: DeviceBuffer[DType.float32]
    var ordered_weights: DeviceBuffer[DType.float32]
    var flags: DeviceBuffer[DType.uint32]
    var weights_prefix_sum: DeviceBuffer[DType.float32]
    var scan_scanned: DeviceBuffer[DType.float32]
    var scan_has_flag: DeviceBuffer[DType.uint8]
    var scan_block_sums: DeviceBuffer[DType.float32]
    var scan_block_flags: DeviceBuffer[DType.uint8]
    var sort_offsets: DeviceBuffer[DType.int32]
    var sort_block_sums: DeviceBuffer[DType.int32]
    var need_weights: DeviceBuffer[DType.float32]
    var point: DeviceBuffer[DType.float32]
    var h_point: HostBuffer[DType.float32]
    #: `ComputeExactValue`'s two outputs: the residual column and the
    #: weight copy. Named for what they hold, not for their C++ locals,
    #: because `values`/`weights` there is what made the MAPE argument
    #: confusing enough to need the note in `compute_exact_approx`.
    var residuals: DeviceBuffer[DType.float32]
    var residual_weights: DeviceBuffer[DType.float32]
    #: the MAPE arm's `weightsWithTargets` (`:169`); allocated always,
    #: because one `n_rows` float buffer is not worth a branch
    var mape_weights: DeviceBuffer[DType.float32]


def make_exact_quantile_scratch(
    ctx: DeviceContext,
    n_rows: Int,
    bin_count: Int,
    max_segment_size: Int,
) raises -> ExactQuantileScratch:
    """Size every buffer once. `blocks_wide` follows the widest leaf,
    which is what the sort's segment axis is sized from."""
    var blocks_wide = (
        max_segment_size + REORDER_BLOCK - 1
    ) // REORDER_BLOCK
    if blocks_wide < 1:
        blocks_wide = 1
    var scan_blocks = (n_rows + 768 - 1) // 768 + 1

    return ExactQuantileScratch(
        ctx.enqueue_create_buffer[DType.uint32](n_rows),
        ctx.enqueue_create_buffer[DType.uint32](n_rows),
        ctx.enqueue_create_buffer[DType.uint32](n_rows),
        ctx.enqueue_create_buffer[DType.uint32](n_rows),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
        ctx.enqueue_create_buffer[DType.uint32](n_rows),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
        ctx.enqueue_create_buffer[DType.uint8](n_rows),
        ctx.enqueue_create_buffer[DType.float32](scan_blocks),
        ctx.enqueue_create_buffer[DType.uint8](scan_blocks),
        ctx.enqueue_create_buffer[DType.int32](n_rows),
        ctx.enqueue_create_buffer[DType.int32](bin_count * blocks_wide),
        ctx.enqueue_create_buffer[DType.float32](bin_count),
        ctx.enqueue_create_buffer[DType.float32](bin_count),
        ctx.enqueue_create_host_buffer[DType.float32](bin_count),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
        ctx.enqueue_create_buffer[DType.float32](n_rows),
    )


def compute_weighted_quantile(
    ctx: DeviceContext,
    n_rows: Int,
    bin_count: Int,
    max_segment_size: Int,
    alpha: Float32,
    use_mape_weights: Bool,
    mut seg_offsets: DeviceBuffer[DType.uint32],
    mut seg_sizes: DeviceBuffer[DType.uint32],
    mut s: ExactQuantileScratch,
    mut out_point: List[Float32],
) raises:
    """`ComputeWeightedQuantile` (`leaves_estimation_helper.h:64-146`),
    their step order, one for one.

    Their two column arguments live in `s` here rather than in the
    signature: Mojo refuses a call that passes a struct and one of its own
    fields mutably in the same argument list, and the struct is what the
    other eleven buffers arrive in anyway. `use_mape_weights` picks
    `s.mape_weights` over `s.residual_weights`, which is the only choice
    their `targets`/`weights` pair ever expresses at this call site.

    The keys always come from `s.residuals`, which is the residual column
    `ComputeExactValue` wrote.
    """
    if bin_count <= 0:
        return
    var row_blocks = (n_rows + 255) // 256
    if row_blocks < 1:
        row_blocks = 1

    # `SegmentedRadixSort(orderedTargets, orderedWeights, ...)` (`:110`).
    # The key/index materialisation is DEVIATION 65's substitution for
    # handing CUB a float key directly.
    ctx.enqueue_function[float_key_kernel](
        s.residuals.unsafe_ptr(), Int32(n_rows),
        s.keys.unsafe_ptr(), s.index.unsafe_ptr(),
        grid_dim=row_blocks, block_dim=256,
    )
    launch_segmented_radix_sort(
        ctx, n_rows, bin_count, max_segment_size,
        EXACT_SORT_FIRST_BIT, EXACT_SORT_LAST_BIT,
        s.keys, s.index, s.temp_keys, s.temp_index,
        seg_offsets, seg_sizes, s.sort_offsets, s.sort_block_sums,
    )
    # their `Gather(orderedTargets, ...)` / `Gather(orderedWeights, ...)`
    # (`:99`, `:102`), applied AFTER the sort here because the sort
    # carried indices rather than the columns themselves
    ctx.enqueue_function[gather_f32_by_index_kernel](
        s.residuals.unsafe_ptr(), s.index.unsafe_ptr(), Int32(n_rows),
        s.ordered_targets.unsafe_ptr(),
        grid_dim=row_blocks, block_dim=256,
    )
    var src_weights = (
        s.mape_weights if use_mape_weights else s.residual_weights
    )
    ctx.enqueue_function[gather_f32_by_index_kernel](
        src_weights.unsafe_ptr(), s.index.unsafe_ptr(), Int32(n_rows),
        s.ordered_weights.unsafe_ptr(),
        grid_dim=row_blocks, block_dim=256,
    )

    # `FillBuffer(endOfBinsFlags, 0); MakeEndOfBinsFlags(...)` (`:114-117`)
    ctx.enqueue_memset(s.flags, UInt32(0))
    ctx.enqueue_function[make_end_of_bins_flags_kernel](
        seg_offsets.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        s.flags.unsafe_ptr(), UInt32(EXACT_FLAG_MASK),
        grid_dim=bin_count, block_dim=128,
    )

    # `SegmentedScanVector(orderedWeights, endOfBinsFlags,
    #                      weightsPrefixSum, true, 1)` (`:121`)
    launch_segmented_scan_vector(
        ctx, n_rows, True, EXACT_FLAG_MASK,
        s.ordered_weights, s.flags, s.weights_prefix_sum,
        s.scan_scanned, s.scan_has_flag,
        s.scan_block_sums, s.scan_block_flags,
    )

    # `FillBuffer(needWeights, 0.0f); ComputeNeedWeights(...)` (`:125-132`)
    ctx.enqueue_memset(s.need_weights, Float32(0.0))
    ctx.enqueue_function[compute_need_weights_kernel](
        s.ordered_weights.unsafe_ptr(),
        seg_offsets.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        s.need_weights.unsafe_ptr(), alpha,
        grid_dim=bin_count, block_dim=NEED_WEIGHTS_BLOCK,
    )

    # `CalculateQuantileWithBinarySearch(...)` (`:139-144`)
    var point_blocks = (
        bin_count + QUANTILE_SEARCH_BLOCK - 1
    ) // QUANTILE_SEARCH_BLOCK
    ctx.enqueue_function[compute_weighted_quantile_kernel](
        s.ordered_targets.unsafe_ptr(),
        s.weights_prefix_sum.unsafe_ptr(),
        s.need_weights.unsafe_ptr(),
        seg_offsets.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        Int32(bin_count),
        s.point.unsafe_ptr(),
        Int32(BINARY_SEARCH_ITERATIONS),
        grid_dim=point_blocks, block_dim=QUANTILE_SEARCH_BLOCK,
    )

    # `result.Read(point)` (`:145`)
    ctx.enqueue_copy(dst_ptr=s.h_point.unsafe_ptr(), src_buf=s.point)
    ctx.synchronize()
    out_point.clear()
    for i in range(bin_count):
        out_point.append(s.h_point.unsafe_ptr().unsafe_load(i))


def compute_exact_approx(
    ctx: DeviceContext,
    objective: Int,
    is_mape: Bool,
    n_rows: Int,
    bin_count: Int,
    max_segment_size: Int,
    alpha: Float32,
    mut seg_offsets: DeviceBuffer[DType.uint32],
    mut seg_sizes: DeviceBuffer[DType.uint32],
    mut s: ExactQuantileScratch,
    mut out_point: List[Float32],
) raises:
    """`ComputeExactApprox` (`leaves_estimation_helper.h:148-185`).

    Their switch has three reachable arms and a raising default:

        Quantile, MAE  ->  ComputeWeightedQuantile(targets, weights, ...)
        MAPE           ->  ComputeWeightsWithTargets first, then the same
        anything else  ->  CB_ENSURE(false, "Only MAPE, MAE and Quantile
                           are supported for Exact leaves estimation on
                           GPU")

    `is_mape` selects the middle one. `objective` is carried so the raise
    can name the loss, which is what their message does.
    """
    if is_mape:
        # `ComputeWeightsWithTargets(targets, weights, &weightsWithTargets)`
        # (`:170`).
        #
        # THE ARGUMENT IS THE RESIDUAL, NOT THE RAW LABEL, and that is
        # deliberate on their side rather than an accident of naming. At
        # the point `EstimateExact` calls in (`pointwise_oracle.cpp:
        # 204-210`) the buffer named `values` holds `target - cursor`, and
        # `ComputeWeightsWithTargets` divides by `max(1, |that|)`. This
        # port read that as a defect against `TMAPETarget::Der`, which
        # divides by `max(1, |raw target|)` (`pointwise_targets.cu:
        # 151-154`), and went looking for which of the two their CPU does.
        # THEIR CPU DOES THE SAME AS THEIR GPU:
        # `CalcExactLeafDeltas` fills `leafSamples[i]` with
        # `targets[i] - approxes[i]` (`algo/approx_calcer.cpp:693`) and
        # hands it to `CalcOneDimensionalOptimumConstApprox` as `target`,
        # whose MAPE arm is
        # `weightsWithTarget[idx] /= Max(1.0f, Abs(target[idx]))`
        # (`metrics/optimal_const_for_loss.h:112-114`) -- the residual
        # again. Both of their arms agree, so ours does too.
        var mape_blocks = (
            n_rows + WEIGHTS_WITH_TARGETS_BLOCK - 1
        ) // WEIGHTS_WITH_TARGETS_BLOCK
        ctx.enqueue_function[compute_weights_with_targets_kernel](
            s.residuals.unsafe_ptr(), s.residual_weights.unsafe_ptr(),
            s.mape_weights.unsafe_ptr(), Int32(n_rows),
            grid_dim=mape_blocks, block_dim=WEIGHTS_WITH_TARGETS_BLOCK,
        )
        compute_weighted_quantile(
            ctx, n_rows, bin_count, max_segment_size, alpha, True,
            seg_offsets, seg_sizes, s, out_point,
        )
        return

    compute_weighted_quantile(
        ctx, n_rows, bin_count, max_segment_size, alpha, False,
        seg_offsets, seg_sizes, s, out_point,
    )
