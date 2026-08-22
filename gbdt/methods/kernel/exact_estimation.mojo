"""The four kernels behind `ELeavesEstimation::Exact` on their GPU.

PORT OF `catboost/cuda/methods/kernel/exact_estimation.cu` at CatBoost
`54a8143a`. Transliterated. Do not improve.

WHAT EXACT IS. For MAE, MAPE and Quantile the second derivative is
identically zero, so a Newton step is meaningless and even the Gradient
step is only a weight-normalized average. Their answer is to solve the leaf
EXACTLY: the value that minimises a weighted absolute deviation over a leaf
is the weighted alpha-quantile of that leaf's residuals, and these four
kernels compute it.

The pipeline, from `ComputeWeightedQuantile`
(`leaves_estimation_helper.h:64-146`), which
`gbdt/methods/leaves_estimation/leaves_estimation_helper.mojo` drives:

    residual[i] = target[i] - cursor[i]     ComputeExactValue
                                            (`permutation_der_calcer.h:98`)
    sort residuals within each leaf         SegmentedRadixSort (:110)
    flags[segment start] = 1                MakeEndOfBinsFlags (:117)
    prefix sum of weights within leaf       SegmentedScanVector (:121)
    needWeights[leaf] = alpha * sum w       ComputeNeedWeights (:129)
    binary search for the crossing point    CalculateQuantileWithBinarySearch
                                            (:139)

MAPE differs in ONE step: its weights are divided by `max(1, |target|)`
first (`ComputeWeightsWithTargets`, `leaves_estimation_helper.h:169-170`),
which is exactly the `1 / max(1, |target|)` factor in `TMAPETarget::Der`.

## Two places their own file is approximate, both kept

Their binary search runs a FIXED SIXTEEN iterations
(`leaves_estimation_helper.h:69`, used at `:143`) rather than to
convergence, and their sort keeps only bits [10, 32) of the key. Neither is
an error to fix: the estimator is called "Exact" against Newton and
Gradient, not against infinite precision.

================= DEVIATION BLOCK =================
DEVIATION 66: `ComputeNeedWeightsImpl`'s EARLY RETURN IS MOVED BELOW ITS
BARRIER. Theirs is (`exact_estimation.cu:51-73`):

    const ui32 begin = beginOffsets[blockIdx.x] + threadIdx.x;
    const ui32 end   = endOffsets[blockIdx.x];
    __shared__ float localBuffer[BLOCK_SIZE];
    localBuffer[threadIdx.x] = 0;
    if (begin >= end) { return; }            // <-- BEFORE the barrier
    ...
    __syncthreads();
    float blocksSum = FastInBlockReduce<float>(...);

A leaf with fewer rows than the block width makes `begin >= end` true for
some threads and false for others, so the threads that stay reach
`__syncthreads()` with part of the block already retired. CUDA calls that
undefined and NVIDIA's hardware happens to survive it; Metal does not
promise to, and a barrier some threads never reach is the one Metal
failure mode that hangs rather than returning a wrong number.

Ours keeps every thread alive to the barrier and lets the out-of-range
ones contribute a zero, which is the shape `pointwise_target_kernel`
already uses for its own tail (`pointwise_targets.cu:255-257`, their own
idiom in their own file). THE SUM IS UNCHANGED: the threads that returned
early in theirs had nothing to add.

DEVIATION 67: `ComputeWeightedQuantileWithBinarySearchImpl` IS BOUNDED BY
THE BIN COUNT, not by the object count. Theirs guards with

    const ui32 i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= objectsCount) { return; }

and then indexes `beginOffsets[i]`, `endOffsets[i]`, `point[i]` -- all of
which are per-BIN. It is launched at `CeilDivide(binCount, 256)` blocks
(`:148`) and passed `Targets.Size()` as `objectsCount` (`exact_estimation.h
:113`), so with `binCount` far below `objectsCount` the guard never fires
and threads in the last block read `beginOffsets` past `binCount + 1` and
write `point` past `binCount`. On a 64-leaf tree that is 192 slots of
overrun into whatever follows. It cannot be copied: `point` here is the
leaf-value buffer and the overrun would be into live data.
===================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import sum as block_sum

from gbdt.targets.kernel.pointwise_targets import pinned_block_sum
from std.gpu import block_dim, block_idx, thread_idx

#: `const ui32 blockSize = 1024` (`exact_estimation.cu:107`). Metal's
#: threadgroup limit is 1024 as well, and the reduce below is sized from
#: this at comptime.
comptime NEED_WEIGHTS_BLOCK = 1024

#: `const ui32 blockSize = 512` (`:127`)
comptime WEIGHTS_WITH_TARGETS_BLOCK = 512

#: `const ui32 blockSize = 256` (`:146`)
comptime QUANTILE_SEARCH_BLOCK = 256

#: `binarySearchIterations = 16` (`leaves_estimation_helper.h:69`)
comptime BINARY_SEARCH_ITERATIONS = 16


def compute_exact_value_kernel(
    targets: MutPointer[Float32, MutAnyOrigin],
    approx: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    has_weights: Int32,
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_weights: MutPointer[Float32, MutAnyOrigin],
):
    """`TPermutationDerCalcer::ComputeExactValue`
    (`targets/permutation_der_calcer.h:98-113`).

    Theirs is five vector ops -- fill, add, fill, subtract, add, fill,
    add -- on whole buffers, which is `value = target - approx` and
    `weights = weights` written the long way because their `TCudaBuffer`
    has no fused form. One kernel here, same two results.

    THE RESIDUAL IS WHAT GETS QUANTILED, not the target: the walker's
    `MoveTo` then SHIFTS the cursor by the value this produces, so the
    leaf value is the quantile of what is left to explain.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(size_in):
        return
    out_values.unsafe_store(
        i, targets.unsafe_load(i) - approx.unsafe_load(i)
    )
    var w = Float32(1.0)
    if has_weights != Int32(0):
        w = weights.unsafe_load(i)
    out_weights.unsafe_store(i, w)


def compute_weights_with_targets_kernel(
    targets: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    weights_with_targets: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
):
    """`ComputeWeightsWithTargetsImpl` (`exact_estimation.cu:76-87`).

        const float delta = max(1.0f, abs(targets[i]));
        weightsWithTargets[i] = weights[i] / delta;

    The MAPE arm only (`leaves_estimation_helper.h:167-179`). `delta` is
    the same `max(1, |target|)` denominator `TMAPETarget::Der` divides by
    (`pointwise_targets.cu:151-154`), so the quantile it solves is the one
    that minimises MAPE rather than MAE.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(size_in):
        return
    var delta = max(Float32(1.0), abs(targets.unsafe_load(i)))
    weights_with_targets.unsafe_store(
        i, weights.unsafe_load(i) / delta
    )


def make_end_of_bins_flags_kernel(
    seg_offsets: MutPointer[UInt32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    flag_bit_in: UInt32,
):
    """`MakeEndOfBinsFlagsImpl` (`exact_estimation.cu:89-99`).

        const ui32 begin = beginOffsets[blockIdx.x];
        if (begin == end) return;
        flags[begin] = 1;

    One flag per segment START, which is what the segmented scan below
    resets its carry on. Theirs launches 128 threads per block and lets
    all of them write the same word; the race is benign because the value
    is a constant, and it is left alone here for the same reason -- except
    that this port's segmented scan reads a BIT rather than a whole word
    (`segmented_scan.mojo`'s `flag_mask`), so the constant is
    `flag_bit_in` and not a literal 1.

    Their `beginOffsets`/`endOffsets` pair is a single offsets array read
    at `i` and `i + 1`; ours is the offset/size pair the searcher already
    exports, because the final partitions sit in memory in BIT-REVERSED
    leaf order and a prefix sum of the sizes is the wrong segmentation
    (`doc_parallel_boosting.mojo` records that trap at its export).
    """
    var seg = Int(block_idx.x)
    if Int(seg_sizes.unsafe_load(seg)) == 0:
        return
    if thread_idx.x != 0:
        return
    var begin = Int(seg_offsets.unsafe_load(seg))
    flags.unsafe_store(
        begin, flags.unsafe_load(begin) | flag_bit_in
    )


def compute_need_weights_kernel(
    weights: MutPointer[Float32, MutAnyOrigin],
    seg_offsets: MutPointer[UInt32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    need_weights: MutPointer[Float32, MutAnyOrigin],
    alpha: Float32,
):
    """`ComputeNeedWeightsImpl<BLOCK_SIZE>` (`exact_estimation.cu:44-73`).

    One block per leaf; the block strides its leaf's rows, sums the
    weights, reduces, and thread 0 writes `alpha * total`. That number is
    the weight mass the quantile has to reach.

    An EMPTY leaf writes nothing, exactly as theirs does not -- the caller
    zero-fills `need_weights` first (`leaves_estimation_helper.h:125-127`),
    so an empty leaf keeps a zero and the binary search's own
    `left > right` arm gives it a leaf value of zero.

    See DEVIATION 66 in the module docstring for the one restructuring:
    every thread reaches the reduce.
    """
    var seg = Int(block_idx.x)
    var base = Int(seg_offsets.unsafe_load(seg))
    var size = Int(seg_sizes.unsafe_load(seg))
    var tid = Int(thread_idx.x)

    # `for (idx = begin; idx < end; idx += BLOCK_SIZE) totalSum += w[idx]`
    var total_sum = Float32(0.0)
    var idx = tid
    while idx < size:
        total_sum += weights.unsafe_load(base + idx)
        idx += NEED_WEIGHTS_BLOCK

    # IDENTITY_PATHS row 8's last site (E1 2026-08-22: gbdt_logloss's
    # first divergent stage is leaves.estimated, this path).
    var blocks_sum = pinned_block_sum[NEED_WEIGHTS_BLOCK](total_sum)
    if tid == 0 and size > 0:
        need_weights.unsafe_store(seg, blocks_sum * alpha)


def compute_weighted_quantile_kernel(
    targets: MutPointer[Float32, MutAnyOrigin],
    weights_prefix_sum: MutPointer[Float32, MutAnyOrigin],
    need_weights: MutPointer[Float32, MutAnyOrigin],
    seg_offsets: MutPointer[UInt32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    bin_count_in: Int32,
    point: MutPointer[Float32, MutAnyOrigin],
    binary_search_iterations_in: Int32,
):
    """`ComputeWeightedQuantileWithBinarySearchImpl<BLOCK_SIZE>`
    (`exact_estimation.cu:10-42`), one thread per LEAF.

    Their loop, term for term (`:31-39`):

        left  = beginOffsets[i]
        right = endOffsets[i] == 0 ? 0 : endOffsets[i] - 1
        if (left > right) { point[i] = 0; return; }
        for (index = 0; index < binarySearchIterations; ++index) {
            middle = left + (right - left) / 2;
            if (weightsPrefixSum[middle] < needWeights[i] - eps) left = middle;
            else right = middle;
        }
        point[i] = targets[right];

    Note what the loop does NOT do: it never terminates early, and `left`
    is set to `middle` rather than `middle + 1`, so the interval can stop
    shrinking before the sixteen iterations are up. Both are theirs and
    both are copied -- the second is why sixteen iterations resolve a leaf
    of up to 65,536 rows exactly and a wider one only approximately.

    `eps` is `std::numeric_limits<float>::epsilon()` (`:33`).

    See DEVIATION 67 in the module docstring for the bound: `bin_count`,
    not their object count.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(bin_count_in):
        return

    var base = Int(seg_offsets.unsafe_load(i))
    var size = Int(seg_sizes.unsafe_load(i))
    if size <= 0:
        point.unsafe_store(i, Float32(0.0))
        return

    var left = base
    var right = base + size - 1

    var eps = Float32(1.1920929e-07)  # FLT_EPSILON
    var need = need_weights.unsafe_load(i)
    for _ in range(Int(binary_search_iterations_in)):
        var middle = left + (right - left) // 2
        if weights_prefix_sum.unsafe_load(middle) < need - eps:
            left = middle
        else:
            right = middle

    point.unsafe_store(i, targets.unsafe_load(right))
