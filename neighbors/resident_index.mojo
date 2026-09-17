# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident k-NN index (lane/infer-speed-classical, 2026-09-17,
DEVIATION 2921).

`NearestNeighbors.kneighbors` used to upload the whole fitted index on
every call: 352 MB for the 400,000 x 220 Istella-S block, which on an RTX
4090 is 48 ms of a 140 ms call at 4,000 queries and all but the query
work of a call at one query. cuML's `NearestNeighbors.fit` keeps the
index on the device (`nearest_neighbors.pyx`, the `X_m` device array) and
its `kneighbors` reads it there; this file gives the estimator the same
residency without touching `fit`: the FIRST `kneighbors` call uploads
the index through `knn_index_prepare` and keeps the handle on the Python
instance, every later call searches through `knn_index_search`, and the
handle is released when the instance is collected or its index changes.

The registry is `core/forest_inference_model.mojo`'s (FOREST-RESIDENT-1):
one process-wide `_Global` keyed by a monotonically increasing integer
handle, the calling extension holding the GIL across prepare, search and
release. An entry owns its `DeviceContext` and the index buffer; the
buffer is destroyed before the context (DEVIATION 1946).

WHAT DOES NOT CHANGE. The search is `neighbors/estimator.mojo::
knn_search_resident`, the body `knn_search` runs after its own upload:
the norms, the transposed layout, the distance chain, the selection, the
sort and the outputs are the same statements over the same bytes, so a
search through a handle returns the bits `knn_search` returns
(`tools/identity_break.py` lanes knn and knn-<metric>, the cuda column
before and after this file, IDENTICAL). The host binding has no entry
here (a CPU search reads the caller's memory in place), so
`python/mojolearn/neighbors.py` takes this door only where the loaded
binding exports it.
"""
from std.ffi import _Global

from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
from neighbors.impl.selection.distance_weights import WEIGHTS_UNIFORM
from neighbors.estimator import (
    DEFAULT_QUERY_TILE,
    knn_classifier_predict_resident,
    knn_regressor_predict_resident,
    knn_search_resident,
)
from neighbors.impl.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    METRIC_FROM_IS_SQRT,
)


struct ResidentKnnIndex(Movable):
    """One fitted index on the device: the bytes of `n_index x n_features`
    row-major float32, uploaded once."""

    var ctx: DeviceContext
    var index: DeviceBuffer[DType.float32]
    var n_index: Int
    var n_features: Int

    def __init__(
        out self,
        index_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_index: Int,
        n_features: Int,
    ) raises:
        if n_index <= 0 or n_features <= 0:
            raise Error("knn_index_prepare: n_index and n_features must be positive")
        var ctx = DeviceContext()
        var index = ctx.enqueue_create_buffer[DType.float32](n_index * n_features)
        ctx.enqueue_copy(dst_buf=index, src_ptr=index_ptr)
        ctx.synchronize()
        self.n_index = n_index
        self.n_features = n_features
        self.index = index^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.index^
        _ = self.ctx^


struct KnnIndexRegistry(Movable):
    var entries: Dict[Int, ResidentKnnIndex]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentKnnIndex]()
        self.next_id = 1


comptime KNN_INDEX_REGISTRY = _Global[
    StorageType=KnnIndexRegistry,
    name=(
        "MojoKnnResidentIndexIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoKnnResidentIndexDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
        "MojoKnnResidentIndexFast"
    ),
    init_fn=KnnIndexRegistry.__init__,
]


def knn_index_prepare(
    index_ptr: MutPointer[Float32, MutUntrackedOrigin], n_index: Int, n_features: Int,
) raises -> Int:
    """Upload the index once; the handle every later search names."""
    var state = KNN_INDEX_REGISTRY.get_or_create_ptr()
    var entry = ResidentKnnIndex(index_ptr, n_index, n_features)
    if state[].next_id == 9223372036854775807:
        raise Error("resident k-NN index handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def knn_index_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later call."""
    var state = KNN_INDEX_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident k-NN index handle")
    var released = state[].entries.pop(handle)
    _ = released^


def knn_index_search(
    handle: Int,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    out_dist_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_idx_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    return_sqrt: Bool = True,
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    knn_method: Int = KNN_METHOD_AUTO,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """`knn_search_resident` over the handle's index and context. The shape
    the caller names must be the shape that was uploaded; `index_ptr` is
    the same bytes on the host, read for the refusals only."""
    var state = KNN_INDEX_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident k-NN index handle")
    ref entry = state[].entries[handle]
    if entry.n_index != n_index or entry.n_features != n_features:
        raise Error(
            "knn_search_resident: the handle holds a "
            + String(entry.n_index) + " x " + String(entry.n_features)
            + " index, the call names " + String(n_index) + " x "
            + String(n_features)
        )
    return knn_search_resident(
        entry.ctx, entry.index, index_ptr, n_index, queries_ptr, n_queries,
        n_features, k, out_dist_ptr, out_idx_ptr, return_sqrt,
        requested_query_tile, knn_method, metric, metric_arg,
    )


def _resident_entry_check(handle: Int, n_index: Int, n_features: Int) raises:
    """The handle exists and holds the shape the call names."""
    var state = KNN_INDEX_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident k-NN index handle")
    ref entry = state[].entries[handle]
    if entry.n_index != n_index or entry.n_features != n_features:
        raise Error(
            "knn resident predict: the handle holds a "
            + String(entry.n_index) + " x " + String(entry.n_features)
            + " index, the call names " + String(n_index) + " x "
            + String(n_features)
        )


def knn_index_classify(
    handle: Int,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    y_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_outputs: Int,
    n_classes: List[Int],
    out_labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_proba_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_uniq_ptr: MutPointer[Int32, MutUntrackedOrigin],
    want_proba: Bool,
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
    weights: Int = WEIGHTS_UNIFORM,
) raises -> Int:
    """`knn_classifier_predict_resident` over the handle's index and
    context (DEVIATION 3002)."""
    _resident_entry_check(handle, n_index, n_features)
    var state = KNN_INDEX_REGISTRY.get_or_create_ptr()
    ref entry = state[].entries[handle]
    return knn_classifier_predict_resident(
        entry.ctx, entry.index, index_ptr, n_index, queries_ptr, n_queries,
        n_features, k, y_ptr, n_outputs, n_classes, out_labels_ptr,
        out_proba_ptr, out_uniq_ptr, want_proba, requested_query_tile,
        metric, metric_arg, weights,
    )


def knn_index_regress(
    handle: Int,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_outputs: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
    weights: Int = WEIGHTS_UNIFORM,
) raises -> Int:
    """`knn_regressor_predict_resident` over the handle's index and
    context (DEVIATION 3002)."""
    _resident_entry_check(handle, n_index, n_features)
    var state = KNN_INDEX_REGISTRY.get_or_create_ptr()
    ref entry = state[].entries[handle]
    return knn_regressor_predict_resident(
        entry.ctx, entry.index, index_ptr, n_index, queries_ptr, n_queries,
        n_features, k, y_ptr, n_outputs, out_ptr, requested_query_tile,
        metric, metric_arg, weights,
    )
