# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MultiClass: the softmax value, its gradient, and one Hessian row.

PORT OF `catboost/cuda/targets/kernel/multilogit.cu` at CatBoost `54a8143a`
-- `MultiLogitValAndFirstDerImpl` (`:10-102`), `MultiLogitSecondDerRowImpl`
(`:104-169`) and their two launchers (`:171-212`). Transliterated. Do not
improve.

**The MultiLogit pair and the MultiClassOneVsAll pair are ported.** Their
file also holds
`RMSEWithUncertainty`, `MultiCrossEntropy`, `MultiRMSE` and
`BuildConfusionMatrixBins`. Each of the others is a different `ELossFunction`
with its own dispatch, and porting a kernel no caller reaches is the defect
PORTING_RULES 3 names. They are listed in `NOT_IMPLEMENTED.tsv` rather than left
looking absent.

## THE LAST CLASS IS IMPLICIT, AND EVERY LOOP BOUND SAYS SO

`effectiveClassCount = numClasses - 1` (`:20`, `:113`). A `numClasses`-way
softmax over free approxes is over-parameterized -- adding a constant to
every approx changes nothing -- so CatBoost pins the last class's approx at
ZERO and carries only the other `numClasses - 1`. That is why:

    the prediction plane count is  numClasses - 1
    the leaf value dimension is    numClasses - 1
    the Hessian block is           (numClasses - 1) x (numClasses - 1)
    the softmax denominator gets   `+= __expf(0.0f - maxApprox)`  (`:53`)
                                   -- the pinned class's own term

and why `classApprox` is `0.0f` when `targetClass` is the last one
(`:46`, the `targetClass[j] < effectiveClassCount` test).

A reader who counts `numClasses` planes anywhere in this file has found a
bug.

## THE MAX SUBTRACTION IS PART OF THE ARITHMETIC, NOT A GUARD

`maxApprox` starts at **0**, not at `-inf` (`:41`), and is then maxed
against the free approxes. Starting at zero is not a missing initializer:
zero IS the pinned class's approx, so `maxApprox` is the max over ALL
`numClasses` of them, including the implicit one. Every `__expf` in the file
then takes `approx - maxApprox`, which keeps the largest exponent at
`exp(0) == 1` and the denominator at or above 1.

## THE HESSIAN ROW IS LOWER-TRIANGULAR AND ONE ROW PER LAUNCH

`MultiLogitSecondDerRowImpl` writes row `der2Row` only, and only its
columns `0..der2Row` (`:159-165`) -- their
`ComputeSecondDerRowLowerTriangleForAllBlocks` loops the row index and
launches this once per row (`pointwise_oracle.cpp:140-150`). The diagonal is
`w * (1 - p_row) * p_row` and the off-diagonals are `-w * p_k * p_row`,
which is the multinomial Hessian; the caller mirrors the lower triangle into
the upper one on the host (`:165-176`).

================= DEVIATION BLOCK =================
DEVIATION 71: `functionValue` ARRIVES AS PER-BLOCK PARTIALS, not through
their block-reduce-plus-`atomicAdd` scalar (`:96-99`). This is the same
substitution `pointwise_targets.mojo` and `bootstrap.mojo` already record
and for the same reason: a float atomic makes the sum depend on block
arrival order, and the same-seed-twice gate caught two fits differing on it.
Their `FillBuffer(functionValue, 0.0f, 1, stream)` prologue (`:186-188`),
which exists only because the atomic accumulates, is not needed and is not
ported -- each block writes its own slot.

DEVIATION 72: `ElementsPerThread` is a comptime parameter as theirs is a
template parameter, and **both of their launchers pass 1** (`:181`, `:205`).
The per-element arrays are `InlineArray`, which is registers at that size.
The unrolled shape is kept rather than collapsed to a scalar because their
`#pragma unroll` loops are the file's structure and a reader diffing against
`:59-92` needs to find them.

DEVIATION 73: `__ldg` is a plain load, and `predictionsAlignSize` /
`derAlignSize` / `der2AlignSize` are passed as arguments exactly as theirs
are, so a caller that pads its planes still works. Every caller in this port
passes `size`. Mojo 1.0 ships no non-temporal or read-only-cache load hint;
the same deviation `transform.mojo` and `fill.mojo` already record.

DEVIATION 254: `__expf` and `__logf` are `routed_exp` and `routed_log`,
the `@always_inline` shims in `pointwise_targets.mojo` over IDENTITY_PATHS
row 12's `identical_exp`/`identical_log` -- under FAST each inlines to the
`std.math` call it replaced at the site (measured bit for bit; the shims'
own block records why the plain wrapper was not enough); under IDENTICAL
both route through the one portable polynomial pair in
`checks/numerics.mojo`, one arithmetic on every backend. Their file uses
the CUDA fast-math forms at every site, so the substitution is in-family --
and `pointwise_target_check` measured Mojo's device `exp` at about twenty
ulp against libm, which is looser than `__expf`'s two. No check may expect
bitwise der parity against a CatBoost fit. The isfinite guard in the
one-vs-all score is kept exactly; only the transcendental calls are routed.
===================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext

# IDENTITY_PATHS row 8 (DEVIATION 251): under `NUMERIC_IDENTICAL` the
# within-block fold must not follow the hardware warp width (64 on AMD),
# so every fv/magnitude reduce below goes through the pinned-shape fold.
#
# DEVIATION 254 (see the module deviation block): `routed_exp`/`routed_log`
# are the `@always_inline` shims beside `pinned_block_sum` -- FAST arm the
# stdlib call inlined at the site (measured bit-for-bit against the direct
# call; a plain `def` wrapper MOVED FAST BITS, see the shims' block), the
# IDENTICAL arm row 12's `identical_exp`/`identical_log`.
from gbdt.targets.kernel.pointwise_targets import (
    pinned_block_sum,
    routed_exp,
    routed_log,
)

#: `const ui32 blockSize = 256` (`multilogit.cu:180`, `:204`)
comptime MULTILOGIT_BLOCK_SIZE = 256

#: `const ui32 elementsPerThreads = 1` (`:181`, `:205`). See DEVIATION 72.
comptime MULTILOGIT_ELEMENTS_PER_THREAD = 1


def multilogit_val_and_first_der_kernel[
    elements_per_thread: Int = MULTILOGIT_ELEMENTS_PER_THREAD,
    search: Bool = False,
](
    target_classes: MutPointer[Float32, MutAnyOrigin],
    num_classes_in: Int32,
    size_in: Int32,
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    load_indices: MutPointer[UInt32, MutAnyOrigin],
    has_load_indices: Int32,
    predictions_align_size_in: Int32,
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    der: MutPointer[Float32, MutAnyOrigin],
    der_align_size_in: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`MultiLogitValAndFirstDerImpl<BlockSize, ElementsPerThread>`
    (`multilogit.cu:10-102`), copied.

    Per document, their arithmetic term for term:

        eff       = numClasses - 1
        maxApprox = max(0, approx[0..eff))
        sumExp    = sum_k exp(approx[k] - maxApprox) + exp(-maxApprox)
        p_k       = exp(approx[k] - maxApprox) / sumExp
        der[k]    = w * ((targetClass == k) - p_k)
        score    += w * ((targetClass < eff ? approx[targetClass] : 0)
                         - maxApprox - log(sumExp))

    `load_indices` is their `loadPredictionsIndices` (`:29`): the estimator
    reads the cursor through a permutation while the boosting loop reads it
    straight, so the gather is a kernel argument rather than two kernels.

    ## TWO MODES, because their ONE kernel serves two callers

    `search=False` is the LEAVES ORACLE's `ComputeValueAndDerivative`
    (`permutation_der_calcer.h:114-126`): `der` is their standalone der
    buffer, one plane per free class starting at plane 0.

    `search=True` is `TMultiClassificationTargets::StochasticDer`
    (`multiclass_targets.cpp:22-45`), which fills their `StatsToAggregate`:

        statCount = 1 + NumClasses, minus one for MultiClass  (`:31-35`)
        column 0                = the weights                 (`:38-40`)
        columns [1, statCount)  = the ders                    (`:42-45`)

    so `der` is the STATS buffer and the class planes start at plane ONE.
    Their `CB_ENSURE(!secondDerAsWeights, "MultiClass loss doesn't support
    second derivatives in tree structure search currently")` (`:27`) is why
    plane 0 is always the weight here and never `der2`.

    ================= DEVIATION BLOCK =================
    DEVIATION 79: `plane_magnitudes` IN SEARCH MODE CARRIES A SINGLE
    GRADIENT BOUND FOR ALL CLASS PLANES, and it is
    `sum over rows of max over classes |w * der_k|` rather than one sum per
    plane.

    NO CATBOOST COUNTERPART, like the fixed-point accumulator it feeds:
    their histograms flush with a float `atomicAdd` and need no bound at
    all. Ours needs one, and `choose_scale` takes ONE scale for the whole
    histogram (`greedy_search_helper.mojo` maxes the weight and gradient
    magnitudes before calling it), so per-plane sums would be reduced to
    their max anyway.

    WHY THIS BOUND IS VALID FOR EVERY PLANE: for any class `k`,
    `sum_rows |der_k| <= sum_rows max_j |der_j|`, so the bound holds
    simultaneously for all of them and overflow stays impossible -- which
    is the only property `fixed_point.mojo` requires of it.

    WHAT IT COSTS: the bound is loose by at most a factor of `numClasses`
    against the tightest per-plane sum, so the fixed-point scale can be up
    to `numClasses` times smaller than it needed to be. At seven classes
    that is three bits of resolution out of the margin `choose_scale`
    documents as millionfold. UNMEASURED against a per-plane version;
    the alternative costs `numClasses` reduction lanes where the
    deterministic fold is comptime-fixed at two. OPEN ITEM.
    ===================================================
    """
    var size = Int(size_in)
    var num_classes = Int(num_classes_in)
    var eff = num_classes - 1
    var pred_align = Int(predictions_align_size_in)
    var der_align = Int(der_align_size_in)

    var tid = (
        Int(block_idx.x) * MULTILOGIT_BLOCK_SIZE * elements_per_thread
        + Int(thread_idx.x)
    )

    var tmp_score = Float32(0.0)

    var class_approx = InlineArray[Float32, elements_per_thread](
        fill=Float32(0.0)
    )
    var target_class = InlineArray[Int32, elements_per_thread](
        fill=Int32(0)
    )
    var sum_exp = InlineArray[Float32, elements_per_thread](
        fill=Float32(0.0)
    )
    var weight = InlineArray[Float32, elements_per_thread](
        fill=Float32(1.0)
    )
    var max_approx = InlineArray[Float32, elements_per_thread](
        fill=Float32(0.0)
    )
    var load_index = InlineArray[Int, elements_per_thread](fill=0)

    # their first `#pragma unroll` block (`:33-56`)
    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        var in_range = idx < size

        var li = idx
        if has_load_indices != Int32(0) and in_range:
            li = Int(load_indices.unsafe_load(idx))
        load_index[j] = li

        if in_range:
            target_class[j] = Int32(target_classes.unsafe_load(idx))
        else:
            target_class[j] = Int32(0)

        # `maxApprox[j] = 0` then max over the free approxes -- zero is the
        # PINNED class's approx, so this is the max over all of them
        var mx = Float32(0.0)
        if in_range:
            for k in range(eff):
                var v = predictions.unsafe_load(li + k * pred_align)
                if v > mx:
                    mx = v
        max_approx[j] = mx

        # `const float tmp = targetClass < eff && idx < size ? ... : 0.0f`
        var tmp = Float32(0.0)
        if in_range and Int(target_class[j]) < eff:
            tmp = predictions.unsafe_load(
                li + Int(target_class[j]) * pred_align
            )
        class_approx[j] = tmp - mx

        var se = Float32(0.0)
        if in_range:
            for k in range(eff):
                se += routed_exp(
                    predictions.unsafe_load(li + k * pred_align) - mx
                )
        # `sumExpApproxForAllClasses[j] += __expf(0.0f - maxApprox[j])`
        # -- the PINNED class's term, added whether or not idx < size,
        # exactly as theirs is (`:53` sits outside the ternary)
        se += routed_exp(Float32(0.0) - mx)
        sum_exp[j] = se

    # their second block (`:59-64`)
    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        if has_weights != Int32(0) and idx < size:
            weight[j] = weights.unsafe_load(idx)
        else:
            weight[j] = Float32(1.0)

    # their third block (`:67-92`)
    #
    # `mag_der` accumulates DEVIATION 79's bound: the per-row maximum over
    # class planes, summed over the rows this thread owns.
    var mag_der = Float32(0.0)
    var mag_weight = Float32(0.0)

    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        # `search` puts the class planes at 1.. and the weight at 0
        comptime plane_base = 1 if search else 0
        var max_abs_der = Float32(0.0)
        if idx < size:
            var li = load_index[j]

            comptime if search:
                # `weights = StatsToAggregate.ColumnsView(0)` (`:38`)
                der.unsafe_store(idx, weight[j])

            for k in range(eff):
                var pk = (
                    routed_exp(
                        predictions.unsafe_load(li + k * pred_align)
                        - max_approx[j]
                    )
                    / sum_exp[j]
                )
                var indicator = (
                    Float32(1.0) if Int(target_class[j]) == k
                    else Float32(0.0)
                )
                var d = weight[j] * (indicator - pk)
                der.unsafe_store(
                    idx + (plane_base + k) * der_align, d
                )
                var ad = abs(d)
                if ad > max_abs_der:
                    max_abs_der = ad
            mag_der += max_abs_der
            mag_weight += abs(weight[j])

        if compute_fv != Int32(0):
            var log_denum = routed_log(sum_exp[j])
            if idx < size:
                tmp_score += weight[j] * (class_approx[j] - log_denum)

    # DEVIATION 71: per-block partials, not their block reduce + atomicAdd
    if compute_fv != Int32(0):
        var total = pinned_block_sum[block_size=MULTILOGIT_BLOCK_SIZE](
            tmp_score
        )
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    # DEVIATION 79: the fixed-point scale's two inputs, same reduce shape
    # as the pointwise kernel's.
    comptime if search:
        if compute_magnitudes != Int32(0):
            var w_total = pinned_block_sum[block_size=MULTILOGIT_BLOCK_SIZE](
                mag_weight
            )
            var g_total = pinned_block_sum[block_size=MULTILOGIT_BLOCK_SIZE](
                mag_der
            )
            if thread_idx.x == 0:
                plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
                plane_magnitudes.unsafe_store(
                    2 * Int(block_idx.x) + 1, g_total
                )


def multilogit_second_der_row_kernel[
    elements_per_thread: Int = MULTILOGIT_ELEMENTS_PER_THREAD
](
    num_classes_in: Int32,
    size_in: Int32,
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    predictions_align_size_in: Int32,
    der2_row_in: Int32,
    der2_align_size_in: Int32,
    der2: MutPointer[Float32, MutAnyOrigin],
):
    """`MultiLogitSecondDerRowImpl<BlockSize, ElementsPerThread>`
    (`multilogit.cu:104-169`), copied.

    ONE ROW OF THE LOWER TRIANGLE per launch:

        p_row          = exp(approx[der2Row] - maxApprox) / sumExp
                         (or the pinned class's exp(-maxApprox)/sumExp when
                         der2Row == eff, their `else` at `:157`)
        der2[k]        = -w * p_k * p_row      for k < der2Row
        der2[der2Row]  =  w * (1 - p_row) * p_row

    NOTE THE ABSENT `targetClasses`: the multinomial Hessian does not depend
    on the label, only on the predicted probabilities. Their signature takes
    the pointer and never reads it (`:104`); this one does not take it,
    which is the only difference and is not arithmetic.
    """
    var size = Int(size_in)
    var num_classes = Int(num_classes_in)
    var eff = num_classes - 1
    var pred_align = Int(predictions_align_size_in)
    var der2_align = Int(der2_align_size_in)
    var der2_row = Int(der2_row_in)

    var tid = (
        Int(block_idx.x) * MULTILOGIT_BLOCK_SIZE * elements_per_thread
        + Int(thread_idx.x)
    )

    var sum_exp = InlineArray[Float32, elements_per_thread](
        fill=Float32(0.0)
    )
    var weight = InlineArray[Float32, elements_per_thread](
        fill=Float32(1.0)
    )
    var max_approx = InlineArray[Float32, elements_per_thread](
        fill=Float32(0.0)
    )

    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        var in_range = idx < size

        var mx = Float32(0.0)
        if in_range:
            for k in range(eff):
                var v = predictions.unsafe_load(idx + k * pred_align)
                if v > mx:
                    mx = v
        max_approx[j] = mx

        var se = Float32(0.0)
        if in_range:
            for k in range(eff):
                se += routed_exp(
                    predictions.unsafe_load(idx + k * pred_align) - mx
                )
        se += routed_exp(Float32(0.0) - mx)
        sum_exp[j] = se

    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        if has_weights != Int32(0) and idx < size:
            weight[j] = weights.unsafe_load(idx)
        else:
            weight[j] = Float32(1.0)

    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        if idx < size:
            var p_row: Float32
            if der2_row < eff:
                p_row = (
                    routed_exp(
                        predictions.unsafe_load(
                            idx + der2_row * pred_align
                        )
                        - max_approx[j]
                    )
                    / sum_exp[j]
                )
            else:
                # the PINNED class's probability (`:157-158`)
                p_row = routed_exp(-max_approx[j]) / sum_exp[j]

            for k in range(der2_row):
                var pk = (
                    routed_exp(
                        predictions.unsafe_load(idx + k * pred_align)
                        - max_approx[j]
                    )
                    / sum_exp[j]
                )
                der2.unsafe_store(
                    idx + k * der2_align, -weight[j] * pk * p_row
                )
            der2.unsafe_store(
                idx + der2_row * der2_align,
                weight[j] * (Float32(1.0) - p_row) * p_row,
            )


def multilogit_blocks(size: Int) -> Int:
    """`CeilDivide<ui32>(size, elementsPerThreads * blockSize)`
    (`:182`, `:206`), with their `if (numBlocks)` guard left to the
    caller."""
    var per = MULTILOGIT_BLOCK_SIZE * MULTILOGIT_ELEMENTS_PER_THREAD
    return (size + per - 1) // per


def launch_multilogit_value_and_der(
    ctx: DeviceContext,
    num_classes: Int,
    size: Int,
    mut target_classes: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut predictions: DeviceBuffer[DType.float32],
    predictions_align_size: Int,
    mut load_indices: DeviceBuffer[DType.uint32],
    has_load_indices: Bool,
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut der: DeviceBuffer[DType.float32],
    der_align_size: Int,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool = False,
) raises:
    """`MultiLogitValueAndDer` (`multilogit.cu:171-190`).

    Their `FillBuffer(functionValue, 0.0f, 1, stream)` prologue is not here;
    see DEVIATION 71.
    """
    var blocks = multilogit_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[
        multilogit_val_and_first_der_kernel[
            MULTILOGIT_ELEMENTS_PER_THREAD
        ]
    ](
        target_classes.unsafe_ptr(), Int32(num_classes), Int32(size),
        weights.unsafe_ptr(), Int32(1) if has_weights else Int32(0),
        predictions.unsafe_ptr(),
        load_indices.unsafe_ptr(),
        Int32(1) if has_load_indices else Int32(0),
        Int32(predictions_align_size),
        function_value.unsafe_ptr(), Int32(1) if compute_fv else Int32(0),
        der.unsafe_ptr(), Int32(der_align_size),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(MULTILOGIT_BLOCK_SIZE, 1, 1),
    )


def launch_multilogit_value_and_der_search(
    ctx: DeviceContext,
    num_classes: Int,
    size: Int,
    mut target_classes: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut predictions: DeviceBuffer[DType.float32],
    predictions_align_size: Int,
    mut load_indices: DeviceBuffer[DType.uint32],
    has_load_indices: Bool,
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut stats: DeviceBuffer[DType.float32],
    stats_align_size: Int,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool,
) raises:
    """`TMultiClassificationTargets::StochasticDer`'s MultiClass arm
    (`multiclass_targets.cpp:22-45`), which fills `StatsToAggregate`:
    column 0 the weights, columns 1.. the ders."""
    var blocks = multilogit_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[
        multilogit_val_and_first_der_kernel[
            MULTILOGIT_ELEMENTS_PER_THREAD, True
        ]
    ](
        target_classes.unsafe_ptr(), Int32(num_classes), Int32(size),
        weights.unsafe_ptr(), Int32(1) if has_weights else Int32(0),
        predictions.unsafe_ptr(),
        load_indices.unsafe_ptr(),
        Int32(1) if has_load_indices else Int32(0),
        Int32(predictions_align_size),
        function_value.unsafe_ptr(), Int32(1) if compute_fv else Int32(0),
        stats.unsafe_ptr(), Int32(stats_align_size),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(MULTILOGIT_BLOCK_SIZE, 1, 1),
    )


def launch_multilogit_second_der(
    ctx: DeviceContext,
    num_classes: Int,
    size: Int,
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut predictions: DeviceBuffer[DType.float32],
    predictions_align_size: Int,
    mut der2: DeviceBuffer[DType.float32],
    der2_row: Int,
    der2_align_size: Int,
) raises:
    """`MultiLogitSecondDer` (`multilogit.cu:193-212`)."""
    var blocks = multilogit_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[
        multilogit_second_der_row_kernel[
            MULTILOGIT_ELEMENTS_PER_THREAD
        ]
    ](
        Int32(num_classes), Int32(size),
        weights.unsafe_ptr(), Int32(1) if has_weights else Int32(0),
        predictions.unsafe_ptr(), Int32(predictions_align_size),
        Int32(der2_row), Int32(der2_align_size),
        der2.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(MULTILOGIT_BLOCK_SIZE, 1, 1),
    )


# =========================================================================
# MultiClassOneVsAll: `numClasses` INDEPENDENT logistic regressions.
#
# PORT OF `MultiClassOneVsAllValAndFirstDerImpl` (`multilogit.cu:613-673`)
# and `MultiClassOneVsAllSecondDerImpl` (`:675-704`).
#
# WHERE IT DIFFERS FROM MultiClass, and every line of the difference
# matters:
#
#   * NO PINNED CLASS. `GetDim()` returns `NumClasses`, not
#     `NumClasses - 1` (`multiclass_targets.h:129-134`), so there are
#     `numClasses` cursor planes and `numClasses` leaf dimensions. There
#     is no gauge to fix and no gradient component to reconstruct.
#   * NO MAX SUBTRACTION. Each plane is its own sigmoid; there is no
#     softmax denominator to stabilise, and overflow is handled by their
#     `isfinite(expVal)` fallback exactly as `CrossEntropyImpl` does.
#   * DIAGONAL HESSIAN. `GetHessianType()` is `Diagonal`
#     (`multiclass_targets.h:118-123`), so the walker takes its diagonal
#     arm and no Cholesky runs.
#   * THE SCORE IS DIVIDED BY `numClasses` (`:654`), which MultiLogit's is
#     not.
#
# AND ONE CONSTANT THAT IS NOT THE ONE NEXT DOOR: their `ClipProb`
# (`cuda_util/kernel/kernel_helpers.cuh:228-230`) clamps at 1e-7, where
# `CrossEntropyImpl`'s inline clamp is 1e-40
# (`pointwise_targets.cu:354`). The per-plane arithmetic is otherwise the
# cross-entropy kernel's term for term, so it is tempting to reuse it --
# and the answer would differ in the tail. Copied as theirs.
# =========================================================================


@always_inline
def clip_prob(p: Float32) -> Float32:
    """`ClipProb` (`cuda_util/kernel/kernel_helpers.cuh:228-230`).

        return max(min(p, 1.0f - 1e-7f), 1e-7f);

    NOTE 1e-7, NOT the 1e-40 that `CrossEntropyImpl` clamps at.
    """
    return max(
        min(p, Float32(1.0) - Float32(1e-7)), Float32(1e-7)
    )


def one_vs_all_val_and_first_der_kernel[
    elements_per_thread: Int = MULTILOGIT_ELEMENTS_PER_THREAD,
    search: Bool = False,
](
    target_classes: MutPointer[Float32, MutAnyOrigin],
    num_classes_in: Int32,
    size_in: Int32,
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    load_indices: MutPointer[UInt32, MutAnyOrigin],
    has_load_indices: Int32,
    predictions_align_size_in: Int32,
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    der: MutPointer[Float32, MutAnyOrigin],
    der_align_size_in: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`MultiClassOneVsAllValAndFirstDerImpl` (`multilogit.cu:613-673`).

    Per class, per document, their arithmetic term for term (`:643-656`):

        expVal = exp(val)
        p      = ClipProb(expVal / (1 + expVal))
        c      = (clazz == targetClass) ? 1 : 0
        der    = weight * (c - p)
        score += weight * (c*val - log(1 + expVal)) / numClasses

    The two modes are `multilogit_val_and_first_der_kernel`'s: `search`
    puts the weight in plane 0 and the class planes at 1.., matching their
    `StatsToAggregate` (`multiclass_targets.cpp:37-48`, where OneVsAll
    keeps the full `1 + NumClasses` because it has no pinned class).
    """
    var size = Int(size_in)
    var num_classes = Int(num_classes_in)
    var pred_align = Int(predictions_align_size_in)
    var der_align = Int(der_align_size_in)

    var tid = (
        Int(block_idx.x) * MULTILOGIT_BLOCK_SIZE * elements_per_thread
        + Int(thread_idx.x)
    )

    var tmp_score = Float32(0.0)
    var target_class = InlineArray[Int32, elements_per_thread](
        fill=Int32(0)
    )
    var weight = InlineArray[Float32, elements_per_thread](
        fill=Float32(1.0)
    )
    var load_index = InlineArray[Int, elements_per_thread](fill=0)
    var mag_der = Float32(0.0)
    var mag_weight = Float32(0.0)

    # their first `#pragma unroll` block (`:628-635`)
    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        var in_range = idx < size
        var li = idx
        if has_load_indices != Int32(0) and in_range:
            li = Int(load_indices.unsafe_load(idx))
        load_index[j] = li
        if in_range:
            target_class[j] = Int32(target_classes.unsafe_load(idx))
            if has_weights != Int32(0):
                weight[j] = weights.unsafe_load(idx)
            else:
                weight[j] = Float32(1.0)
        else:
            target_class[j] = Int32(0)
            weight[j] = Float32(1.0)

        comptime if search:
            if in_range:
                der.unsafe_store(idx, weight[j])
                mag_weight += abs(weight[j])

    comptime plane_base = 1 if search else 0

    # their class loop (`:638-660`). NOTE the loop order: class OUTSIDE,
    # document inside, which is theirs and keeps each plane's stores
    # contiguous.
    var max_abs = InlineArray[Float32, elements_per_thread](
        fill=Float32(0.0)
    )
    for clazz in range(num_classes):

        @parameter
        for j in range(elements_per_thread):
            var idx = tid + j * MULTILOGIT_BLOCK_SIZE
            var in_range = idx < size
            var val = Float32(0.0)
            if in_range:
                val = predictions.unsafe_load(
                    load_index[j] + clazz * pred_align
                )
            var exp_val = routed_exp(val)
            var p = clip_prob(exp_val / (Float32(1.0) + exp_val))
            var c = (
                Float32(1.0) if Int(target_class[j]) == clazz
                else Float32(0.0)
            )
            var direction = c - p

            if in_range:
                var d = weight[j] * direction
                der.unsafe_store(
                    idx + (plane_base + clazz) * der_align, d
                )
                var ad = abs(d)
                if ad > max_abs[j]:
                    max_abs[j] = ad

            if compute_fv != Int32(0):
                # `isfinite(expVal) ? __logf(1 + expVal) : val` (`:653`)
                # DEVIATION 254: their fallback kept exactly; only the
                # `log` call is routed.
                var log_term = val
                if isfinite(exp_val):
                    log_term = routed_log(Float32(1.0) + exp_val)
                if in_range:
                    tmp_score += (
                        weight[j]
                        * (c * val - log_term)
                        / Float32(num_classes)
                    )

    @parameter
    for j in range(elements_per_thread):
        mag_der += max_abs[j]

    if compute_fv != Int32(0):
        var total = pinned_block_sum[block_size=MULTILOGIT_BLOCK_SIZE](
            tmp_score
        )
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    comptime if search:
        if compute_magnitudes != Int32(0):
            var w_total = pinned_block_sum[block_size=MULTILOGIT_BLOCK_SIZE](
                mag_weight
            )
            var g_total = pinned_block_sum[block_size=MULTILOGIT_BLOCK_SIZE](
                mag_der
            )
            if thread_idx.x == 0:
                plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
                plane_magnitudes.unsafe_store(
                    2 * Int(block_idx.x) + 1, g_total
                )


def one_vs_all_second_der_kernel[
    elements_per_thread: Int = MULTILOGIT_ELEMENTS_PER_THREAD
](
    num_classes_in: Int32,
    size_in: Int32,
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    predictions_align_size_in: Int32,
    der2_align_size_in: Int32,
    der2: MutPointer[Float32, MutAnyOrigin],
):
    """`MultiClassOneVsAllSecondDerImpl` (`multilogit.cu:675-704`).

        der2[idx + clazz * der2Align] = weight * p * (1 - p)

    ALL `numClasses` PLANES IN ONE LAUNCH, because the Hessian is
    DIAGONAL: there is no lower triangle to walk a row at a time, which is
    the whole difference from `multilogit_second_der_row_kernel`.
    """
    var size = Int(size_in)
    var num_classes = Int(num_classes_in)
    var pred_align = Int(predictions_align_size_in)
    var der2_align = Int(der2_align_size_in)

    var tid = (
        Int(block_idx.x) * MULTILOGIT_BLOCK_SIZE * elements_per_thread
        + Int(thread_idx.x)
    )
    var weight = InlineArray[Float32, elements_per_thread](
        fill=Float32(1.0)
    )

    @parameter
    for j in range(elements_per_thread):
        var idx = tid + j * MULTILOGIT_BLOCK_SIZE
        if has_weights != Int32(0) and idx < size:
            weight[j] = weights.unsafe_load(idx)
        else:
            weight[j] = Float32(1.0)

    for clazz in range(num_classes):

        @parameter
        for j in range(elements_per_thread):
            var idx = tid + j * MULTILOGIT_BLOCK_SIZE
            if idx < size:
                var val = predictions.unsafe_load(
                    idx + clazz * pred_align
                )
                var exp_val = routed_exp(val)
                var p = clip_prob(exp_val / (Float32(1.0) + exp_val))
                der2.unsafe_store(
                    idx + clazz * der2_align,
                    weight[j] * p * (Float32(1.0) - p),
                )


def launch_one_vs_all_value_and_der[
    search: Bool = False
](
    ctx: DeviceContext,
    num_classes: Int,
    size: Int,
    mut target_classes: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut predictions: DeviceBuffer[DType.float32],
    predictions_align_size: Int,
    mut load_indices: DeviceBuffer[DType.uint32],
    has_load_indices: Bool,
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Bool,
    mut der: DeviceBuffer[DType.float32],
    der_align_size: Int,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Bool = False,
) raises:
    """`MultiClassOneVsAllValueAndDer` (`multilogit.cu:709-732`)."""
    var blocks = multilogit_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[
        one_vs_all_val_and_first_der_kernel[
            MULTILOGIT_ELEMENTS_PER_THREAD, search
        ]
    ](
        target_classes.unsafe_ptr(), Int32(num_classes), Int32(size),
        weights.unsafe_ptr(), Int32(1) if has_weights else Int32(0),
        predictions.unsafe_ptr(),
        load_indices.unsafe_ptr(),
        Int32(1) if has_load_indices else Int32(0),
        Int32(predictions_align_size),
        function_value.unsafe_ptr(), Int32(1) if compute_fv else Int32(0),
        der.unsafe_ptr(), Int32(der_align_size),
        plane_magnitudes.unsafe_ptr(),
        Int32(1) if compute_magnitudes else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(MULTILOGIT_BLOCK_SIZE, 1, 1),
    )


def launch_one_vs_all_second_der(
    ctx: DeviceContext,
    num_classes: Int,
    size: Int,
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut predictions: DeviceBuffer[DType.float32],
    predictions_align_size: Int,
    mut der2: DeviceBuffer[DType.float32],
    der2_align_size: Int,
) raises:
    """`MultiClassOneVsAllSecondDer` (`multilogit.cu:734-752`)."""
    var blocks = multilogit_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[
        one_vs_all_second_der_kernel[MULTILOGIT_ELEMENTS_PER_THREAD]
    ](
        Int32(num_classes), Int32(size),
        weights.unsafe_ptr(), Int32(1) if has_weights else Int32(0),
        predictions.unsafe_ptr(), Int32(predictions_align_size),
        Int32(der2_align_size), der2.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(MULTILOGIT_BLOCK_SIZE, 1, 1),
    )
