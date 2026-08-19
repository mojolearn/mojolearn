"""Pointwise objectives: value, first derivative, second derivative.

PORT OF `catboost/cuda/targets/kernel/pointwise_targets.cu` at CatBoost
`54a8143a`. Transliterated. Do not improve.

Only `MseImpl` is ported. It is the objective behind `RMSE`
(`pointwise_targets.cu:496`) and it is the smallest correct starting point
for a boosting loop, because its derivatives are exact rather than
approximated and its Newton step needs no line search.

Their kernel, verbatim in structure:

    const float direction = relev - val;
    const float weight = (weights && (i < size)) ? weights[i] : 1.0f;
    if (der)  der[i]  = weight * direction;
    if (der2) der2[i] = weight;
    functionValue += -weight * (val - relev) * (val - relev);

So the FIRST derivative is the residual and the SECOND is just the weight.
That is why an MSE tree's leaf value is a plain weighted mean and why this
objective is the one to build the loop against before anything harder.

Note the SIGN on `functionValue`: theirs accumulates the NEGATIVE squared
error, because everything downstream maximises. Kept, because flipping it
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
    """`MseImpl`, copied, writing straight into the stats planes.

    ================= DEVIATION BLOCK =================
    Theirs writes `der` and `der2` into two separate buffers and sums
    `functionValue` with a block reduce and an `atomicAdd`.

    Ours writes the two derivatives into the ONE `stats` buffer the histogram
    kernels already read, in the layout they already expect:

        stats[0 * size + i]  = der2 = weight        (the weight plane)
        stats[1 * size + i]  = der  = weight * (y - pred)

    That is not a redesign, it is the same two numbers in the place the rest
    of this port already reads them from. Adding a copy to move them would be
    inventing work CatBoost does not do either, since their der buffers ARE
    what their histogram kernels consume.

    `functionValue` is NOT computed here. Their block reduce needs an
    `atomicAdd` on a float, which Metal does not have. The loss is reduced on
    the host in `doc_parallel_boosting.mojo` instead, where it costs one copy
    per iteration and is not on any hot path.
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
