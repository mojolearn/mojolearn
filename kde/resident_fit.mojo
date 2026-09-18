# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident KDE fit set (DEVIATION 3003, lane/knn-tiled-distance,
2026-09-17): `KernelDensity.score_samples` without the per-call upload and
validation of the training rows.

`kde_score_samples_host_ptr` (`kde/estimator.mojo`) validated the whole
training set (DEVIATION 604's finiteness scan, the cosine zero-row rule),
copied it into pinned memory and uploaded it on EVERY call: on an RTX 4090
that was 27.5 ms of a 75 ms Istella-S call at 2,000 queries and all of a
one-query call (the floor probe of lane/infer-speed-classical). cuML's
`fit` keeps `X` on the device and `score_samples` reuses it
(`kernel_density.py`, `self.X_`); this file gives the estimator the same
residency in the shape of the k-NN index registry (`neighbors/
resident_index.mojo`, DEVIATION 2921): the FIRST `score_samples` call
validates and uploads the training rows and the weights through
`kde_fit_prepare`, keeps the handle on the Python instance, and every later
call scores through `kde_score_samples_resident`, which validates and
uploads the QUERIES only.

WHAT DOES NOT CHANGE. The refusals are `kde_score_samples_host_ptr`'s, by
name and in its order (kernel and metric names, `kde_fit_validate`, the
training rows), then the query count and the query rows at score time;
the one difference is that a bad training set and a bad query count in the
same call now name the training set first. The score is `kde/impl/kde.mojo::
score_samples` over the same device bytes with the same `sum_weights`, and
the scores come back through the same pinned download, so the bits are the
one-shot entry's (`tools/identity_break.py`, the kde lanes, cuda against the
cpu host route). The context and the buffers are destroyed in DEVIATION
1946's order, buffers before the context.
"""
from std.ffi import _Global

from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
from core.identity_trace import IdentityTrace
from kde.impl.kde import score_samples
from kde.impl.neighbors.kernel_density import (
    KDE_ELEM_TPB,
    KDE_LSE_TPB,
    host_sum_weights,
    kde_fit_validate,
    kde_validate_data_ptr,
    kernel_from_name,
    metric_from_name,
)


struct ResidentKdeFit(Movable):
    """One fitted KDE on the device: the training rows, the weights (or a
    one-element placeholder), their sum, and the shape and names the
    handle was prepared with."""

    var ctx: DeviceContext
    var train: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var has_weights: Bool
    var sum_w: Float32
    var n_train: Int
    var n_features: Int
    var kernel: Int
    var metric: Int
    var bandwidth: Float32

    def __init__(
        out self,
        train_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_train: Int,
        n_features: Int,
        bandwidth: Float32,
        kernel: Int,
        metric: Int,
        weights: List[Float32],
        has_weights: Bool,
    ) raises:
        # `kde_score_samples_host_ptr`'s refusals for the fit set, in its
        # order: `kde_fit_validate`, then the training rows.
        kde_fit_validate(n_train, n_features, bandwidth, kernel, metric, weights, has_weights)
        kde_validate_data_ptr(train_ptr, n_train, n_features, metric, "train")
        var ctx = DeviceContext()
        var n = n_train * n_features
        var train = ctx.enqueue_create_buffer[DType.float32](n)
        var host = ctx.enqueue_create_host_buffer[DType.float32](n)
        copy_f32(train_ptr, host.unsafe_ptr(), n)
        ctx.enqueue_copy(dst_buf=train, src_ptr=host.unsafe_ptr())
        var sum_w = Float32(n_train)
        var wbuf: DeviceBuffer[DType.float32]
        if has_weights:
            wbuf = ctx.enqueue_create_buffer[DType.float32](n_train)
            var whost = ctx.enqueue_create_host_buffer[DType.float32](n_train)
            copy_f32(weights.unsafe_ptr(), whost.unsafe_ptr(), n_train)
            ctx.enqueue_copy(dst_buf=wbuf, src_ptr=whost.unsafe_ptr())
            ctx.synchronize()
            _ = whost^
            sum_w = host_sum_weights(weights)
        else:
            wbuf = ctx.enqueue_create_buffer[DType.float32](1)
            ctx.synchronize()
        _ = host^
        self.n_train = n_train
        self.n_features = n_features
        self.kernel = kernel
        self.metric = metric
        self.bandwidth = bandwidth
        self.has_weights = has_weights
        self.sum_w = sum_w
        self.train = train^
        self.weights = wbuf^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.weights^
        _ = self.train^
        # DEVIATION 3010 (DEVIATION 2520's drain): the frees enqueued by the
        # releases above must complete before the context is destroyed, or
        # the runtime allocator's lock is left held and the next context's
        # first allocation never returns. Host-side drain; no output bit.
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


struct KdeFitRegistry(Movable):
    var entries: Dict[Int, ResidentKdeFit]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentKdeFit]()
        self.next_id = 1


comptime KDE_FIT_REGISTRY = _Global[
    StorageType=KdeFitRegistry,
    name=(
        "MojoKdeResidentFitIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoKdeResidentFitDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
        "MojoKdeResidentFitFast"
    ),
    init_fn=KdeFitRegistry.__init__,
]


def kde_fit_prepare(
    train_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_train: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    weights: List[Float32],
    has_weights: Bool,
) raises -> Int:
    """Validate and upload the fit set once; the handle every later score
    names. Refuses, by name, everything `kde_score_samples_host_ptr`
    refuses about the fit set."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    var entry = ResidentKdeFit(train_ptr, n_train, n_features, bandwidth, k, m, weights, has_weights)
    if state[].next_id == 9223372036854775807:
        raise Error("resident KDE fit handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def kde_fit_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later call."""
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident KDE fit handle")
    var released = state[].entries.pop(handle)
    _ = released^


def kde_score_samples_resident(
    handle: Int,
    query: MutPointer[Float32, MutUntrackedOrigin],
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    scores: MutPointer[Float32, MutUntrackedOrigin],
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
    metric_arg: Float32 = Float32(2.0),
) raises:
    """`kde_score_samples_host_ptr` after its fit-set work: the query count
    and rows are validated as there, the queries staged once into pinned
    memory and uploaded, `score_samples` runs over the handle's training
    rows and weights, and the scores come back through the pinned download
    into `scores`. The names, the bandwidth and the feature count must be
    the handle's; a mismatch is refused rather than served."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident KDE fit handle")
    ref entry = state[].entries[handle]
    if entry.kernel != k or entry.metric != m or entry.n_features != n_features or entry.bandwidth != bandwidth:
        raise Error(
            "kde_score_samples_resident: the handle was prepared with kernel "
            + String(entry.kernel) + ", metric " + String(entry.metric)
            + ", " + String(entry.n_features) + " features and bandwidth "
            + String(entry.bandwidth) + "; the call names kernel " + String(k)
            + ", metric " + String(m) + ", " + String(n_features)
            + " features and bandwidth " + String(bandwidth)
        )
    if n_query <= 0:
        raise Error("kde: X must have at least one row (n_query)")
    kde_validate_data_ptr(query, n_query, n_features, m, "query")
    var n = n_query * n_features
    var dquery = entry.ctx.enqueue_create_buffer[DType.float32](n)
    var host = entry.ctx.enqueue_create_host_buffer[DType.float32](n)
    copy_f32(query, host.unsafe_ptr(), n)
    entry.ctx.enqueue_copy(dst_buf=dquery, src_ptr=host.unsafe_ptr())
    var dout = entry.ctx.enqueue_create_buffer[DType.float32](n_query)
    var trace = IdentityTrace()
    trace.header(
        "kde: n_train=" + String(entry.n_train) + " n_query=" + String(n_query)
        + " n_features=" + String(n_features) + " kernel=" + kernel
        + " metric=" + metric + " metric_arg=" + String(metric_arg)
        + " weighted=" + String(entry.has_weights)
    )
    score_samples(
        entry.ctx, dquery, entry.train, entry.weights, entry.has_weights, dout,
        n_query, entry.n_train, n_features, bandwidth, entry.sum_w, k, m,
        metric_arg, trace, elem_tpb, lse_tpb,
    )
    var hout = entry.ctx.enqueue_create_host_buffer[DType.float32](n_query)
    entry.ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
    entry.ctx.synchronize()
    copy_f32(hout.unsafe_ptr(), scores, n_query)
    _ = hout^
    _ = host^
    _ = dquery^
    _ = dout^
