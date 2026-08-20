"""Pointwise objectives: value, first derivative, second derivative.

PORT OF `catboost/cuda/targets/kernel/pointwise_targets.cu` at CatBoost
`54a8143a`. Transliterated. Do not improve.

Only the RMSE objective is ported. It is the smallest correct starting point
for a boosting loop, because its derivatives are exact rather than
approximated and its Newton step needs no line search.

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
    parts in 1e6 relative; `choose_scale`'s three headroom bits are a factor
    of eight against exactly this kind of slack, as its contract audit in
    `doc_parallel_boosting.mojo` prices out. `compute_magnitudes` stands in
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
    if compute_fv != Int32(0):
        var score = Float32(0.0)
        if in_range:
            score = -weight * (val - relev) * (val - relev)
        var total = block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            _ = Atomic.fetch_add(function_value, total)

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
            _ = Atomic.fetch_add(plane_magnitudes, w_total)
            _ = Atomic.fetch_add(plane_magnitudes.unsafe_offset(1), g_total)
