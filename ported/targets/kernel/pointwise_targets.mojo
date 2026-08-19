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

from std.gpu import block_dim, block_idx, thread_idx


def mse_kernel(
    relevs: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
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

    `functionValue` is NOT computed here. It is one scalar per iteration
    behind a block reduce, it feeds nothing this port computes on the device,
    and the boosting loop already reads the cursor back for its own reduction,
    so adding it here would be work for a number that is produced anyway. The
    host reduction lives in `doc_parallel_boosting.mojo`.
    ===================================================
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= size:
        return

    var val = predictions.unsafe_load(i)
    var relev = relevs.unsafe_load(i)
    var direction = relev - val
    var weight = Float32(1.0)
    if has_weights != Int32(0):
        weight = weights.unsafe_load(i)

    stats.unsafe_store(i, weight)
    stats.unsafe_store(size + i, weight * direction)
