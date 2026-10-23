# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident reference rows of `labeled_reference_predict`
(DBSCAN.predict and AgglomerativeClustering.predict, DEVIATION 2740), in the
shape of the resident KDE fit set (`kde/resident_fit.mojo`).

`labeled_reference_predict` (core/labeled_reference_predict.mojo) uploaded
the fitted reference rows, their keys and their labels on EVERY predict
call. `labeled_reference_prepare` uploads them once (at fit, or at the first
predict of a loaded or copied model) and returns a handle; every later
`labeled_reference_predict_resident` uploads the QUERIES only and runs the
same kernel (`labeled_reference_kernel`, imported, unchanged) with the same
launch chunks over the same device bytes, so the labels are the one-shot
entry's bits. Cosine keeps DEVIATION 5113: the reference rows are scaled to
unit rows by the same `cosine_unit_rows` before their one upload, the
queries before theirs.
"""
from std.ffi import _Global
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.cosine_rows import cosine_unit_rows
from core.labeled_reference_host_predict import (
    LABELED_METRIC_COSINE,
    labeled_reference_validate,
)
from core.labeled_reference_predict import (
    LABELED_PREDICT_TPB,
    LABELED_PREDICT_WORK,
    labeled_reference_kernel,
)
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L1,
    dbscan_metric_threshold,
)

#: The estimators binding's ONE process-lifetime DeviceContext slot (the
#: same name `bindings/_mojolearn_estimators.mojo` uses).
comptime _DEVCTX_SLOT = "MojoEstimatorsContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoEstimatorsContextFast"


struct ResidentLabeledReference(Movable):
    """One fitted reference set on the device: the rows (unit rows under
    cosine), their tie keys and their labels."""

    var ctx: DeviceContext
    var refs: DeviceBuffer[DType.float32]
    var keys: DeviceBuffer[DType.int32]
    var labels: DeviceBuffer[DType.int32]
    var n_refs: Int
    var n_features: Int
    var metric: Int

    def __init__(
        out self,
        refs_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_refs: Int,
        keys_ptr: MutPointer[Int32, MutUntrackedOrigin],
        labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
        n_features: Int,
        metric: Int,
    ) raises:
        # `labeled_reference_predict`'s refusals that concern the reference
        # set, in its order (a query count of 1 stands in for the queries,
        # which the predict call validates).
        labeled_reference_validate(n_refs, 1, n_features, metric)
        var d = n_features
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var refs = ctx.enqueue_create_buffer[DType.float32](n_refs * d)
        var keys = ctx.enqueue_create_buffer[DType.int32](n_refs)
        var ref_labels = ctx.enqueue_create_buffer[DType.int32](n_refs)
        ctx.synchronize()
        var ur = List[Float32]()
        if metric == LABELED_METRIC_COSINE:
            var rr = List[Float32](length=n_refs * d, fill=Float32(0))
            memcpy(dest=rr.unsafe_ptr(), src=refs_ptr, count=n_refs * d)
            ur = cosine_unit_rows(rr, n_refs, d, "labeled_reference_predict")
            ctx.enqueue_copy(dst_buf=refs, src_ptr=ur.unsafe_ptr())
        else:
            ctx.enqueue_copy(dst_buf=refs, src_ptr=refs_ptr)
        ctx.enqueue_copy(dst_buf=keys, src_ptr=keys_ptr)
        ctx.enqueue_copy(dst_buf=ref_labels, src_ptr=labels_ptr)
        ctx.synchronize()
        _ = ur^
        self.n_refs = n_refs
        self.n_features = n_features
        self.metric = metric
        self.refs = refs^
        self.keys = keys^
        self.labels = ref_labels^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946),
        # then the drain of their frees (DEVIATION 3010).
        _ = self.labels^
        _ = self.keys^
        _ = self.refs^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


struct LabeledReferenceRegistry(Movable):
    var entries: Dict[Int, ResidentLabeledReference]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentLabeledReference]()
        self.next_id = 1


comptime LABELED_REFERENCE_REGISTRY = _Global[
    StorageType=LabeledReferenceRegistry,
    name=(
        "MojoLabeledReferenceIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else
        "MojoLabeledReferenceFast"
    ),
    init_fn=LabeledReferenceRegistry.__init__,
]


def labeled_reference_prepare(
    refs_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_refs: Int,
    keys_ptr: MutPointer[Int32, MutUntrackedOrigin],
    labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_features: Int,
    metric: Int,
) raises -> Int:
    """Upload a reference set once; the handle every later predict names."""
    var state = LABELED_REFERENCE_REGISTRY.get_or_create_ptr()
    var entry = ResidentLabeledReference(refs_ptr, n_refs, keys_ptr, labels_ptr, n_features, metric)
    if state[].next_id == 9223372036854775807:
        raise Error("resident labeled reference handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def labeled_reference_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later call."""
    var state = LABELED_REFERENCE_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident labeled reference handle")
    var released = state[].entries.pop(handle)
    _ = released^


def labeled_reference_predict_resident(
    handle: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    metric: Int,
    eps: Float64,
    has_thresh: Bool,
    out_labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`labeled_reference_predict` over a resident reference set: the same
    refusals, the same kernel and launch chunks; only the queries cross."""
    var state = LABELED_REFERENCE_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident labeled reference handle")
    ref entry = state[].entries[handle]
    if entry.n_features != n_features or entry.metric != metric:
        raise Error(
            "labeled_reference_predict_resident: the handle was prepared with "
            + String(entry.n_features) + " features and metric " + String(entry.metric)
            + "; the call names " + String(n_features) + " features and metric " + String(metric)
        )
    var n_refs = entry.n_refs
    labeled_reference_validate(n_refs, n_queries, n_features, metric)
    var thresh = Float32(0.0)
    if has_thresh:
        thresh = dbscan_metric_threshold(metric, eps)
    var d = n_features
    var queries = entry.ctx.enqueue_create_buffer[DType.float32](n_queries * d)
    var out_labels = entry.ctx.enqueue_create_buffer[DType.int32](n_queries)
    var out_refs = entry.ctx.enqueue_create_buffer[DType.int32](n_queries)
    entry.ctx.synchronize()
    var uq = List[Float32]()
    if metric == LABELED_METRIC_COSINE:
        var qq = List[Float32](length=n_queries * d, fill=Float32(0))
        memcpy(dest=qq.unsafe_ptr(), src=queries_ptr, count=n_queries * d)
        uq = cosine_unit_rows(qq, n_queries, d, "labeled_reference_predict")
        entry.ctx.enqueue_copy(dst_buf=queries, src_ptr=uq.unsafe_ptr())
    else:
        entry.ctx.enqueue_copy(dst_buf=queries, src_ptr=queries_ptr)
    entry.ctx.synchronize()
    _ = uq^

    var per_query = n_refs * d
    var chunk = LABELED_PREDICT_WORK // per_query
    if chunk < 1:
        chunk = 1
    var start = 0
    while start < n_queries:  # driver-loop: over launch chunks of the query rows
        var n_launch = n_queries - start
        if n_launch > chunk:
            n_launch = chunk
        var blocks = (n_launch + LABELED_PREDICT_TPB - 1) // LABELED_PREDICT_TPB
        if metric == DBSCAN_METRIC_L1:
            comptime kernel_l1 = labeled_reference_kernel[DBSCAN_METRIC_L1]
            entry.ctx.enqueue_function[kernel_l1](
                queries.unsafe_ptr(),
                entry.refs.unsafe_ptr(),
                entry.keys.unsafe_ptr(),
                entry.labels.unsafe_ptr(),
                Int32(start),
                Int32(n_launch),
                Int32(n_refs),
                Int32(d),
                thresh,
                Int32(1) if has_thresh else Int32(0),
                out_labels.unsafe_ptr(),
                out_refs.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(LABELED_PREDICT_TPB, 1, 1),
            )
        else:
            comptime kernel_l2 = labeled_reference_kernel[0]
            entry.ctx.enqueue_function[kernel_l2](
                queries.unsafe_ptr(),
                entry.refs.unsafe_ptr(),
                entry.keys.unsafe_ptr(),
                entry.labels.unsafe_ptr(),
                Int32(start),
                Int32(n_launch),
                Int32(n_refs),
                Int32(d),
                thresh,
                Int32(1) if has_thresh else Int32(0),
                out_labels.unsafe_ptr(),
                out_refs.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(LABELED_PREDICT_TPB, 1, 1),
            )
        entry.ctx.synchronize()
        start += n_launch

    # Only the labels come back: the chosen reference row was never read.
    var hl = entry.ctx.enqueue_create_host_buffer[DType.int32](n_queries)
    entry.ctx.synchronize()
    entry.ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=out_labels)
    entry.ctx.synchronize()
    memcpy(dest=out_labels_ptr, src=hl.unsafe_ptr(), count=n_queries)
    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = hl^
    _ = queries^
    _ = out_labels^
    _ = out_refs^
