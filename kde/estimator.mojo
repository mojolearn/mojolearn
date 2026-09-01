# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for KernelDensity: what `bindings/` calls.

**WIRED**, corrected 2026-09-01. This docstring said "NOT YET WIRED into
`bindings/_mojolearn_estimators.mojo` or `python/mojolearn/`" and that has
been false since the identity lane took the hand-off:
`bindings/_mojolearn_estimators.mojo:367-426` is
`kde_score_samples_binding` over this function and
`python/mojolearn/density.py::KernelDensity.score_samples` calls it. The
sentence is deleted rather than annotated (fix-docs-on-discovery). Shape
is still `glm/estimator.mojo::ols_fit_host`'s.

ONE PARAMETER OF THIS FUNCTION IS NOT REACHABLE FROM PYTHON YET, and it is
named rather than left as a surprise: `metric_arg` (Minkowski's `p`).
`kde_score_samples_binding` length-checks its `params` list at exactly 5
entries (`:392-396`) and has no slot for a sixth, so the Python surface can
only reach the default `2.0`. `python/mojolearn/density.py` refuses
`metric_params` with any other value BY NAME for that reason and says so.
Closing it is a two-line change in `bindings/` (accept 5 or 6, read
`params[5]` as `metric_arg` when present) and `bindings/` is not this
lane's directory; the exact diff is in the hand-off note in
`kde/README.md`.

`kde_score_samples_host` takes the training rows, the query rows and the
optional weights as host lists, validates exactly as cuML's `fit` and
`score_samples` do (`kde/impl/neighbors/kernel_density.mojo::
kde_fit_validate`, `kernel_from_name`, `metric_from_name` -- every
unported choice REFUSED BY NAME) plus DEVIATION 604's finiteness rules
(`kde_validate_data`: no NaN/inf in the data, the sqeuclidean magnitude
bound, a normal bandwidth and normal finite weights), uploads, runs `ML::KDE::score_samples`
(`kde/impl/kde/kde.mojo`) with the environment's identity trace
(`MOJOLEARN_IDENTITY_TRACE`), and returns `n_query` float32 log-densities.

cuML's `fit` keeps `X` on the device and `score_samples` reuses it; a
bindings layer that wants that should keep the `DeviceBuffer` in the Python
object and call `kde/impl/kde/kde.mojo::score_samples` directly. This
entry is the one-shot form, which is what the gates and the card use.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from kde.impl.kde.kde import score_samples
from kde.impl.neighbors.kernel_density import (
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
    metric_arg: Float32 = Float32(2.0),
) raises -> List[Float32]:
    """`KernelDensity(bandwidth, kernel, metric).fit(X_train,
    sample_weight).score_samples(X_query)`, one shot, host in and host out.
    Row-major `train` (`n_train x n_features`) and `query` (`n_query x
    n_features`). Raises, by name, on everything cuML's validation raises
    on and on every unported metric/kernel.

    `metric_arg` is Minkowski's `p`, their name for it. cuML's
    `KernelDensity` has no `p` parameter at all; it takes `metric_params`
    and forwards `list(metric_params.values())[0]` as `metric_arg`
    (`kernel_density.py:302-313`), so a caller mirroring their surface
    passes the single value of that dict here. Defaults to their
    `pairwise_distances` default of 2 (`pairwise_distances.pyx:266`), which
    is also what `score_samples` uses when `metric_params` is empty. It is
    refused by value for non-Lp-valid p (DEVIATION 552) inside
    `pairwise_distance`, before any launch."""
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
        + " metric=" + metric + " metric_arg=" + String(metric_arg)
        + " weighted=" + String(has_weights)
    )
    score_samples(
        ctx, dquery, dtrain, dweights, has_weights, dout,
        n_query, n_train, n_features, bandwidth, sum_w, k, m, metric_arg,
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
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^
