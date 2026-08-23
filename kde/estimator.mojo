"""Host-pointer surface for KernelDensity: what `bindings/` will call.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's (COMMON_BRIEF).
The README's HAND-OFF names the exact Python surface; this file is the entry
it should reach, shaped like `glm/estimator.mojo::ols_fit_host`.

`kde_score_samples_host` takes the training rows, the query rows and the
optional weights as host lists, validates exactly as cuML's `fit` and
`score_samples` do (`kde/ported/neighbors/kernel_density.mojo::
kde_fit_validate`, `kernel_from_name`, `metric_from_name` -- every
unported choice REFUSED BY NAME) plus DEVIATION 604's finiteness rules
(`kde_validate_data`: no NaN/inf in the data, the sqeuclidean magnitude
bound, a normal bandwidth and normal finite weights), uploads, runs `ML::KDE::score_samples`
(`kde/ported/kde/kde.mojo`) with the environment's identity trace
(`MOJOLEARN_IDENTITY_TRACE`), and returns `n_query` float32 log-densities.

cuML's `fit` keeps `X` on the device and `score_samples` reuses it; a
bindings layer that wants that should keep the `DeviceBuffer` in the Python
object and call `kde/ported/kde/kde.mojo::score_samples` directly. This
entry is the one-shot form, which is what the gates and the card use.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from kde.ported.kde.kde import score_samples
from kde.ported.neighbors.kernel_density import (
    KDE_ELEM_TPB,
    KDE_LSE_TPB,
    host_sum_weights,
    kde_fit_validate,
    kde_validate_data,
    kernel_from_name,
    metric_from_name,
)


def _upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def kde_score_samples_host(
    train: List[Float32],
    n_train: Int,
    query: List[Float32],
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    weights: List[Float32],
    has_weights: Bool,
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
) raises -> List[Float32]:
    """`KernelDensity(bandwidth, kernel, metric).fit(X_train,
    sample_weight).score_samples(X_query)`, one shot, host in and host out.
    Row-major `train` (`n_train x n_features`) and `query` (`n_query x
    n_features`). Raises, by name, on everything cuML's validation raises
    on and on every unported metric/kernel."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    kde_fit_validate(n_train, n_features, bandwidth, k, m, weights, has_weights)
    if n_query <= 0:
        raise Error("kde: X must have at least one row (n_query)")
    # DEVIATION 604: finite data (and the sqeuclidean magnitude bound),
    # refused by name BEFORE any upload; the length check is inside.
    kde_validate_data(train, n_train, n_features, m, "train")
    kde_validate_data(query, n_query, n_features, m, "query")
    var ctx = DeviceContext()
    var dtrain = _upload(ctx, train)
    var dquery = _upload(ctx, query)
    var dweights: DeviceBuffer[DType.float32]
    var sum_w = Float32(n_train)
    if has_weights:
        dweights = _upload(ctx, weights)
        sum_w = host_sum_weights(weights)
    else:
        var one = List[Float32]()
        one.append(Float32(1.0))
        dweights = _upload(ctx, one)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_query)
    ctx.synchronize()
    var trace = IdentityTrace()
    trace.header(
        "kde: n_train=" + String(n_train) + " n_query=" + String(n_query)
        + " n_features=" + String(n_features) + " kernel=" + kernel
        + " metric=" + metric + " weighted=" + String(has_weights)
    )
    score_samples(
        ctx, dquery, dtrain, dweights, has_weights, dout,
        n_query, n_train, n_features, bandwidth, sum_w, k, m, Float32(2.0),
        trace, elem_tpb, lse_tpb,
    )
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_query)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n_query):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    _ = dtrain^
    _ = dquery^
    _ = dweights^
    _ = dout^
    return out^
