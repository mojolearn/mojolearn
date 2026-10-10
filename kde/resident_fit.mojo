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
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import NUMERIC_FAST

from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoEstimatorsContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoEstimatorsContextFast"


from checks.numerics import GLOBAL_NUMERIC_MODE
from core.identity_trace import IdentityTrace
from core.device_fold import device_sum_f32_fixed
from kde.impl.kde import score_samples
from kde.impl.chunk_workspace import KDE_CHUNK_POOL_ON,KdeChunkWorkspace
from kde.impl.neighbors.kernel_density import (
    KDE_ELEM_TPB,
    KDE_LSE_TPB,
    kde_chunk_lse_metric_applies,
    kde_score_samples_chunk_lse_reused,
    host_sum_weights,
    kde_fit_validate,
    kde_validate_data_ptr,
    kernel_from_name,
    metric_from_name,
)


# Immutable retained fit snapshots and direct upload are separate controls.
# Buffer copies are transport; the GPU score/statistics remain unchanged.
# M3 2026-10-06 F18: one excluded warmup and one score per arm, 509x3,
# 521x7,997x13; bandwidth invalidation, two models, refit, refusal recovery.
# Direct prep cold B/A=0.6349/0.9221/0.8119; repeated totals=0.4917/0.7277/0.9234,
# all captured score quality equal. Promote only FAST Apple direct preparation;
# escape MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF. Evidence ab-20261006/repairs-54c1f35a5/F18.
# Immutable snapshots independently measured faster (cold0.7578/0.9473/0.6224,
# repeated0.4999/0.9233/0.7824) but remain OFF: ownership semantics change and
# combination with direct preparation is unmeasured; one sample limits inference.
comptime KDE_FAST_IMMUTABLE_FIT = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_KDE_FAST_IMMUTABLE_FIT"]()
comptime KDE_FAST_DIRECT_PREP = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF"]()


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
    var partial_pool: Optional[KdeChunkWorkspace]

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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var n = n_train * n_features
        var train = ctx.enqueue_create_buffer[DType.float32](n)
        var host = ctx.enqueue_create_host_buffer[DType.float32](1 if KDE_FAST_DIRECT_PREP else n)
        comptime if KDE_FAST_DIRECT_PREP:
            # Caller retains the immutable host snapshot through this function's
            # existing completion; no pinned staging copy is needed.
            ctx.enqueue_copy(dst_buf=train, src_ptr=train_ptr)
        else:
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
        self.partial_pool=Optional[KdeChunkWorkspace]()
        comptime if KDE_CHUNK_POOL_ON:
            if kde_chunk_lse_metric_applies(metric):
                self.partial_pool=KdeChunkWorkspace()
        self.has_weights = has_weights
        self.sum_w = sum_w
        self.train = train^
        self.weights = wbuf^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # I20 explicit owning-context drain/release before the fit buffers
        # and context, including an incomplete/failed score consumer.
        comptime if KDE_CHUNK_POOL_ON:
            if self.partial_pool:
                try:
                    self.partial_pool.value().close(self.ctx)
                except:
                    pass
        _ = self.partial_pool^
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
    want_total: Bool = False,
) raises -> Float32:
    """`kde_score_samples_host_ptr` after its fit-set work: the query count
    and rows are validated as there, the queries staged once into pinned
    memory and uploaded, `score_samples` runs over the handle's training
    rows and weights, and the scores come back through the pinned download
    into `scores`. The names, the bandwidth and the feature count must be
    the handle's; a mismatch is refused rather than served. Returns the
    scores' fixed-order float32 sum on the device (`device_sum_f32_fixed`)
    when `want_total` (KernelDensity.score; lane pyglue-numeric: a host sum
    of the downloaded scores), else 0."""
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
    # TOMBSTONE: MOJOLEARN_KDE_SAMPLE_FUSED (DROP) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
    # Tried: the one-drain score + download branch here.
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_SAMPLE_FUSED.patch; record in docs/TOMBSTONES.md.
    var pooled=False
    comptime if KDE_CHUNK_POOL_ON:
        if entry.partial_pool and not trace.enabled and kde_chunk_lse_metric_applies(m):
            ref pool=entry.partial_pool.value()
            # The registry's never-reused immutable handle owns the uploaded
            # fit and weights. Queries change freely; all outputs recompute.
            pooled=kde_score_samples_chunk_lse_reused(entry.ctx,entry.train,dquery,entry.weights,
                entry.has_weights,entry.sum_w,entry.n_train,n_query,n_features,bandwidth,k,m,dout,
                pool,UInt64(handle),elem_tpb,lse_tpb)
    if not pooled:
        score_samples(
            entry.ctx, dquery, entry.train, entry.weights, entry.has_weights, dout,
            n_query, entry.n_train, n_features, bandwidth, entry.sum_w, k, m,
            metric_arg, trace, elem_tpb, lse_tpb,
        )
    var total = Float32(0)
    if want_total:
        total = device_sum_f32_fixed(entry.ctx, dout, n_query)
    var hout = entry.ctx.enqueue_create_host_buffer[DType.float32](n_query)
    entry.ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
    entry.ctx.synchronize()
    copy_f32(hout.unsafe_ptr(), scores, n_query)
    _ = hout^
    _ = host^
    _ = dquery^
    _ = dout^
    return total
