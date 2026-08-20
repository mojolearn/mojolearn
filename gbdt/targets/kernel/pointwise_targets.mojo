"""Pointwise objectives: value, first derivative, second derivative.

PORT OF `catboost/cuda/targets/kernel/pointwise_targets.cu` at CatBoost
`54a8143a`. Transliterated. Do not improve.

RMSE and the cross-entropy pair (Logloss / CrossEntropy) are ported. RMSE
came first because its derivatives are exact and its Newton step needs no
line search; the cross-entropy kernel is below it, and its LEAF story is
different -- their Logloss default is Newton with TEN estimation iterations
(`catboost_options.cpp:157-164`), so its leaves are the estimator's job,
not this file's.

## Which of their two MSE kernels this is, because they have two

`MseImpl` (`pointwise_targets.cu:285-321`) is the obvious one to port and it
is UNREACHED IN THEIR OWN TREE. Its only entry point is `MseTargetKernel`
(`:415-428`) behind `ApproximateMse` (`targets/kernel.h:828`), and
`ApproximateMse` has no caller anywhere in `catboost/`. This file used to cite
it, and the citation was to dead code.

What RMSE actually reaches is the generic
`PointwiseTargetImpl<TTarget, BLOCK_SIZE>` (`:246-281`), dispatched at
`case ELossFunction::RMSE: TRmseTarget target;` (`:496-500`) out of
`PointwiseTargetKernel`, which `ApproximatePointwise`
(`targets/kernel.h:840`) launches. The objective is a three-method struct
(`:178-191`):

    Score(t, p) = (t - p) * (t - p)
    Der  (t, p) = t - p
    Der2 (_, p) = 1.0f

and the generic kernel multiplies each by the weight (`:259-271`):

    const float weight = (weights && (i < size)) ? weights[i] : 1.0f;
    if (der)  der[i]  = weight * target.Der(relev, val);
    if (der2) der2[i] = weight * target.Der2(relev, val);
    functionValue += -weight * target.Score(relev, val);

**The two kernels compute the same three numbers**, term for term, which is
why the port was numerically right while its citation was wrong. It is
transcribed from the reachable one now, because a reader diffing against
`MseImpl` is diffing against a file CatBoost never runs.

So the FIRST derivative is the weighted residual and the SECOND is just the
weight. That is why an MSE tree's leaf value is a plain weighted mean and why
this objective is the one to build the loop against before anything harder.

Note the SIGN on `functionValue`: theirs accumulates the NEGATIVE squared
error, because everything downstream maximizes. Kept, because flipping it
here would silently invert every early-stopping comparison later.
"""

from std.atomic import Atomic
from std.math import exp, isfinite, log
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.primitives.block import sum as block_sum

#: the boosting loop launches this kernel at 256 threads; the reduce below
#: needs the block size at comptime.
comptime MSE_BLOCK_SIZE = 256


def mse_kernel(
    relevs: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`PointwiseTargetImpl<TRmseTarget>`, copied, writing straight into the
    stats planes. The name is kept from when this file cited `MseImpl`; see
    the module docstring for why that citation was to dead code and why the
    two kernels agree term for term anyway.

    ================= DEVIATION BLOCK =================
    Theirs writes `der` and `der2` into two separate buffers and sums
    `functionValue` with a block reduce and an `atomicAdd`.

    Ours writes the two derivatives into the ONE `stats` buffer the histogram
    kernels already read, in the layout they already expect:

        stats[0 * size + i]  = the WEIGHT plane
        stats[1 * size + i]  = der = weight * (y - pred)

    That is not a redesign, it is their own layout: `StochasticDer` builds
    `StatsToAggregate` with column 0 the weights and columns 1.. the ders
    (`targets/pointwise_target_impl.h:188-213`), and that buffer IS what their
    histogram kernels consume.

    WHAT GOES IN PLANE 0 DEPENDS ON THE SCORE FUNCTION, and this port writes
    the branch CatBoost's default takes. `ComputeTarget` passes
    `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`
    (`greedy_search_helper.cpp:286-296`), and Cosine, the shipped default, is
    NOT second order (`enum_helpers.cpp:829-841`), so plane 0 is
    `weightsView.Copy(weightsForIndices)` -- the sample weight -- and not
    `der2`. Under NewtonCosine or NewtonL2 it would be `der2` instead. For
    RMSE the distinction is invisible because `TRmseTarget::Der2` returns
    `1.0f`, so `weight * der2 == weight`; for any objective whose second
    derivative is not constant it is two different numbers and this kernel
    would need the flag.

    `functionValue` IS computed here now, as theirs is
    (`pointwise_targets.cu:309-317`): every in-range thread contributes
    `-weight * (val - relev)^2`, the block reduces, and thread 0 does one
    float `atomicAdd` into the scalar. `compute_fv` stands in for their
    null-pointer test on `functionValue`. The paragraph that stood here said
    the host reduction was free because "the boosting loop already reads the
    cursor back" -- that host read WAS the cost, ~5 ms/tree of host loop
    over 800k rows per iteration, and it is deleted with this.

    The block reduce goes through `max.gpu.primitives.block.sum` where
    theirs is `FastInBlockReduce`, the same substitution
    `partitions_reduce.mojo` records.

    `plane_magnitudes` / `compute_magnitudes` have NO CATBOOST COUNTERPART,
    like the scale they feed: they are the two sums of absolute values --
    `plane_magnitudes[0] = sum |weight plane|`, `[1] = sum |der plane|` --
    that `mojo_only/fixed_point.choose_scale` is specified against, reduced
    here by the SAME block-reduce-plus-one-atomicAdd shape as
    `functionValue` so the fixed-point builds get real magnitudes with no
    host loop and no full-stats readback (the host loop this replaces cost a
    6.4 MB copy plus a 2 * n_rows walk per tree at 800k rows). Accumulated
    in Float32 through a float atomic, so the total can round DOWN by a few
    parts in 1e6 relative; `choose_scale`'s remaining safety bit is a factor
    of TWO against exactly this kind of slack (a millionfold margin -- the
    row-count-aware limit spends the other two former headroom bits on
    resolution, with the dither's +1/row accounted separately), as its
    contract audit in `doc_parallel_boosting.mojo` prices out. `compute_magnitudes` stands in
    for a null-pointer test, exactly as `compute_fv` does.
    ===================================================
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)

    # their `(i < size) ? ... : 0` ternaries (`:296-299`): out-of-range
    # threads stay alive to take part in the reduce, contributing zero.
    var in_range = i < size
    var val = Float32(0.0)
    var relev = Float32(0.0)
    var weight = Float32(1.0)
    if in_range:
        val = predictions.unsafe_load(i)
        relev = relevs.unsafe_load(i)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)
        var direction = relev - val
        stats.unsafe_store(i, weight)
        stats.unsafe_store(size + i, weight * direction)

    # `if (functionValue) { tmpScores[tid] = -w * (val - relev)^2; ...
    # FastInBlockReduce; if (tid == 0) atomicAdd(functionValue, val); }`
    # (`pointwise_targets.cu:309-317`).
    # ============ DETERMINISM FIX, 2026-08-21 ============
    # These two reduces ENDED IN A FLOAT ATOMIC (their `atomicAdd`,
    # `pointwise_targets.cu:317`), so `functionValue` and, worse, the
    # fixed-point scale's magnitudes depended on block arrival order --
    # the same-seed-twice bootstrap gate caught two fits differing, and
    # the long-observed rep-to-rep last-bit jitter of the loss column was
    # exactly this. Every block now STORES its partial at its own slot
    # and `deterministic_sum_lanes_kernel` folds them in one fixed order,
    # so a seeded fit is bit-reproducible end to end. `function_value`
    # and `plane_magnitudes` are therefore PER-BLOCK PARTIALS buffers
    # (`n_blocks` and `2 * n_blocks`), not scalars.
    # =====================================================
    if compute_fv != Int32(0):
        var score = Float32(0.0)
        if in_range:
            score = -weight * (val - relev) * (val - relev)
        var total = block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    # the fixed-point scale's inputs: sum of |plane| for both stat planes,
    # same reduce shape as `functionValue` above.
    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            w_abs = abs(weight)
            g_abs = abs(weight * (relev - val))
        var w_total = block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(
                2 * Int(block_idx.x), w_total
            )
            plane_magnitudes.unsafe_store(
                2 * Int(block_idx.x) + 1, g_total
            )


comptime REDUCE_LANES_BLOCK = 256


def deterministic_sum_lanes_kernel[
    lanes: Int
](
    partials: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """Fold interleaved per-block partials in ONE FIXED ORDER.

    NO CATBOOST COUNTERPART: they accept the float atomic's arrival-order
    nondeterminism in `functionValue`; this port cannot, because the
    magnitudes feed `fixed_scale` and the dithered histograms behind the
    bit-reproducibility claim. One block; thread `t` walks slots
    `t, t + 256, ...` in ascending order and the shared tree fold has one
    shape -- same inputs, same bits, every run.

    `partials` is `[i * lanes + lane]` for `count_in` blocks."""
    var tid = Int(thread_idx.x)
    var count = Int(count_in)

    var acc = InlineArray[Float32, lanes](fill=Float32(0.0))
    var i = tid
    while i < count:

        @parameter
        for lane in range(lanes):
            acc[lane] += partials.unsafe_load(i * lanes + lane)
        i += REDUCE_LANES_BLOCK

    var red = stack_allocation[
        lanes * REDUCE_LANES_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    @parameter
    for lane in range(lanes):
        red[lane * REDUCE_LANES_BLOCK + tid] = acc[lane]
    barrier()
    var step = REDUCE_LANES_BLOCK // 2
    while step > 0:
        if tid < step:

            @parameter
            for lane in range(lanes):
                red[lane * REDUCE_LANES_BLOCK + tid] = (
                    red[lane * REDUCE_LANES_BLOCK + tid]
                    + red[lane * REDUCE_LANES_BLOCK + tid + step]
                )
        barrier()
        step //= 2
    if tid == 0:

        @parameter
        for lane in range(lanes):
            dst.unsafe_store(lane, red[lane * REDUCE_LANES_BLOCK])


def cross_entropy_kernel[
    has_border: Bool
](
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
    """`CrossEntropyImpl` (`pointwise_targets.cu:327-390`), the kernel
    Logloss and CrossEntropy actually reach: `pointwise_target_impl.h:333-345`
    dispatches both to `ApproximateCrossEntropy`, which launches this and
    nothing else (`targets/kernel.h:49`, `:876`). `has_border` is their
    `HAS_BORDER` template arm: Logloss thresholds the target at `border`
    (`GetLogLossBorder`, default 0.5, `pointwise_target_impl.h:285-287`);
    CrossEntropy takes the target as an already-soft class probability.

    Per element, THEIR arithmetic, term for term (`:353-360`):

        expVal = exp(val)                      // their __expf
        p  = clamp(isfinite ? expVal/(1+expVal) : 1, [1e-40, 1-1e-40])
        c  = has_border ? (targetClass > border) : targetClass
        der  = weight * (c - p)
        der2 = weight * (p * (1 - p))
        score += weight * (c*val - log(1+expVal))   // isfinite ? : w*c*val-val

    Note the SIGN convention matches `mse_kernel`: `score` is the
    log-LIKELIHOOD (their `tmpScore`, `:363`), so downstream maximizes, and
    the printed loss is its negation by the same reader that negates mse.

    ================= DEVIATION BLOCK =================
    Same three substitutions `mse_kernel` records, none new:

    * STATS PLANES instead of their separate `der` buffer: plane 0 the
      weight, plane 1 `weight * der` -- their own `StatsToAggregate` column
      order under a NON-second-order score function
      (`pointwise_target_impl.h:188-213`; Cosine and L2 are not second
      order, `enum_helpers.cpp:829-841`). `der2` is NOT written here: the
      split search never reads it under Cosine/L2, and the leaves estimator
      recomputes it at its own point each Newton iteration, which is the
      only place it is consumed. A NewtonCosine/NewtonL2 caller would need
      the `secondDerAsWeights` flag, exactly as the mse block already says.
    * PER-BLOCK PARTIALS instead of their block-reduce-plus-`atomicAdd`
      tail (`:381-388`) for `function_value` AND the two fixed-point plane
      magnitudes -- the 2026-08-21 determinism fix, same buffers, same
      `deterministic_sum_lanes_kernel` fold.
    * GRID: theirs is 512 threads x 2 elements each (`:396-397`); ours is
      the file's one-element-per-thread shape at `MSE_BLOCK_SIZE`.
      Scheduling only -- every per-document store is identical -- and the
      fv/magnitude summation order is ours either way, because the partials
      buffers already are.

    And one of substance: `exp`/`log` here are `std.math`, where theirs are
    CUDA's `__expf`/`__logf` fast-math approximations (~2 ulp). CatBoost
    itself accepts approximate transcendentals at this exact site, so the
    substitution is in-family, but the two arms' derivatives differ in last
    bits BY CONSTRUCTION and no check may expect bitwise der parity against
    a CatBoost fit. (The host-side `std.math.log` tie-decoding defect is a
    different site and does not apply: nothing here decodes ties.)
    ===================================================
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)

    var in_range = i < size
    var val = Float32(0.0)
    var target_class = Float32(1.0)  # their `scale[j]` OOB default (`:346`)
    var weight = Float32(1.0)
    if in_range:
        val = predictions.unsafe_load(i)
        target_class = target_classes.unsafe_load(i)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)

    # `const float expVal = idx < size ? __expf(val) : 0;` (`:353`)
    var exp_val = Float32(0.0)
    if in_range:
        exp_val = exp(val)
    # `p = max(min(isfinite(expVal) ? expVal / (1.0f + expVal) : 1.0f,
    #              1.0f - 1e-40f), 1e-40f)` (`:354`)
    var p = Float32(1.0)
    if isfinite(exp_val):
        p = exp_val / (Float32(1.0) + exp_val)
    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))

    # `const float c = HAS_BORDER ? targetClass > border : targetClass;`
    var c: Float32

    @parameter
    if has_border:
        c = Float32(1.0) if target_class > border else Float32(0.0)
    else:
        c = target_class

    var direction = c - p  # their `direction[j] = c - p` (`:358`)
    var scale = p * (Float32(1.0) - p)  # their `scale[j]` (`:359`)

    if in_range:
        stats.unsafe_store(i, weight)
        stats.unsafe_store(size + i, weight * direction)
    # der2 = weight * scale is NOT stored; see the deviation block.
    _ = scale

    if compute_fv != Int32(0):
        # `tmpScore += w * (c * val - __logf(1 + expVal))`, with their
        # infinity fallback `logExpValPlusOne = isfinite ? ... : val`
        # (`:362-364`).
        var score = Float32(0.0)
        if in_range:
            var log_exp_val_plus_one = val
            if isfinite(exp_val):
                log_exp_val_plus_one = log(Float32(1.0) + exp_val)
            score = weight * (c * val - log_exp_val_plus_one)
        var total = block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            w_abs = abs(weight)
            g_abs = abs(weight * direction)
        var w_total = block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(
                2 * Int(block_idx.x) + 1, g_total
            )
