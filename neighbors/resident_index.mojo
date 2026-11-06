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

THREADS AND DEVICES (lane pass2-forms-cleanup; the same two hazards lane
pass2-cluster-nbr fixed for KMeans in `cluster/resident_centers.mojo`, whose
pattern this copies). The search entries run with the GIL released
(bindings/_mojolearn.mojo), so a refit on another thread could release a
handle, or a prepare could grow the dictionary, while a search reads the
entry. Every registry read and write now takes the registry's lock word. A
search copies the entry's context and index HANDLES (reference counted)
under it and takes the entry's derived-data caches OUT of the registry for
the length of the search (a second search of the same handle meanwhile
builds its own, as a first search does), then puts them back when the
handle still holds the same upload. A release while a search runs drops
only the registry's references; the device memory lives until the search's
copies go. Each entry records the device it was uploaded to; a search whose
thread is bound to another device (the device plan's root) moves the index
there first (device to host to device, the same bytes) under the same
handle, so a handle never runs on a device the caller did not select.
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx, neural_device_plan
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoCoreContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoCoreContextFast"


from checks.numerics import GLOBAL_NUMERIC_MODE
from neighbors.impl.selection.distance_weights import WEIGHTS_UNIFORM
from neighbors.estimator import (
    DEFAULT_QUERY_TILE,
    knn_classifier_predict_resident,
    knn_regressor_predict_resident,
    knn_search_resident,
)
from neighbors.impl.knn.knn import KnnVoteCache
from neighbors.impl.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    KnnIndexCache,
    METRIC_FROM_IS_SQRT,
)


struct ResidentKnnIndex(Movable):
    """One fitted index on the device: the bytes of `n_index x n_features`
    row-major float32, uploaded once."""

    var ctx: DeviceContext
    var index: DeviceBuffer[DType.float32]
    var n_index: Int
    var n_features: Int
    # The device the copy lives on: the device plan's root at the upload,
    # 0 when no plan names another device (process_ctx's own rule).
    var device: Int
    # Which upload of its handle this is (set by the registry on insert and
    # on a device move): caches built on an older upload are never put back.
    var serial: Int

    def __init__(
        out self,
        index_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_index: Int,
        n_features: Int,
    ) raises:
        if n_index <= 0 or n_features <= 0:
            raise Error("knn_index_prepare: n_index and n_features must be positive")
        var dplan = neural_device_plan()
        var ctx = process_ctx[_DEVCTX_SLOT](dplan)
        var index = ctx.enqueue_create_buffer[DType.float32](n_index * n_features)
        ctx.enqueue_copy(dst_buf=index, src_ptr=index_ptr)
        ctx.synchronize()
        self.n_index = n_index
        self.n_features = n_features
        self.device = _device_of(dplan.root)
        self.serial = 0
        self.index = index^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.index^
        _ = self.ctx^


def _device_of(root: Int) -> Int:
    """The device a context made for plan root `root` runs on: `process_ctx`
    opens `root` when it is positive and device 0 otherwise."""
    return root if root > 0 else 0


struct KnnIndexRegistry(Movable):
    var entries: Dict[Int, ResidentKnnIndex]
    #: DEVIATION 3061: what every search derives from an entry's `index`
    #: alone, built by the first search that needs it and kept for the
    #: handle's life (per handle; absent while a search holds it).
    var caches: Dict[Int, KnnIndexCache]
    #: lane/fam2-neighbors (KNN_IDN_VOTE_CACHE): the classifier vote's labels
    #: and unique sets, built by the first vote and kept for the handle's
    #: life (per handle; absent while a vote holds it).
    var votes: Dict[Int, KnnVoteCache]
    var next_id: Int
    var next_serial: Int
    # One lock word: searches run with the GIL released (see the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ResidentKnnIndex]()
        self.caches = Dict[Int, KnnIndexCache]()
        self.votes = Dict[Int, KnnVoteCache]()
        self.next_id = 1
        self.next_serial = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime KNN_INDEX_REGISTRY = _Global[
    StorageType=KnnIndexRegistry,
    name=(
        "MojoKnnResidentIndexIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoKnnResidentIndexFast"
    ),
    init_fn=KnnIndexRegistry.__init__,
]


def _registry_lock(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin]) raises:
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _registry_unlock(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _registry() raises -> UnsafePointer[KnnIndexRegistry, MutAnyOrigin]:
    return KNN_INDEX_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct _ResidentView(Movable):
    """A search's own copies of one entry's context and index HANDLES
    (reference counted, the memory is shared) and its key words."""

    var ctx: DeviceContext
    var index: DeviceBuffer[DType.float32]
    var n_index: Int
    var n_features: Int
    var device: Int
    var serial: Int

    def __init__(out self, var ctx: DeviceContext, var index: DeviceBuffer[DType.float32],
                 n_index: Int, n_features: Int, device: Int, serial: Int):
        self.ctx = ctx^
        self.index = index^
        self.n_index = n_index
        self.n_features = n_features
        self.device = device
        self.serial = serial

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.index^
        _ = self.ctx^


def _locked_insert(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin],
                   var entry: ResidentKnnIndex) raises -> Int:
    _registry_lock(state)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident k-NN index handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        entry.serial = state[].next_serial
        state[].next_serial += 1
        state[].entries[handle] = entry^
        _registry_unlock(state)
        return handle
    except error:
        _registry_unlock(state)
        raise error^


def _locked_pop(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin],
                handle: Int) raises -> ResidentKnnIndex:
    """Remove the entry. Its caches stay behind; the caller drops them with
    `_locked_drop_caches` (each is destroyed outside the lock)."""
    _registry_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident k-NN index handle")
        var popped = state[].entries.pop(handle)
        _registry_unlock(state)
        return popped^
    except error:
        _registry_unlock(state)
        raise error^


def _locked_take_cache(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin],
                       handle: Int) raises -> KnnIndexCache:
    """The handle's derived-data cache, taken out for one search; an empty
    one when none is stored (first search, or another search holds it)."""
    _registry_lock(state)
    try:
        if handle in state[].caches:
            var cache = state[].caches.pop(handle)
            _registry_unlock(state)
            return cache^
        _registry_unlock(state)
        return KnnIndexCache()
    except error:
        _registry_unlock(state)
        raise error^


def _locked_take_vote(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin],
                      handle: Int) raises -> KnnVoteCache:
    """The handle's vote cache, taken out for one vote; an empty one when
    none is stored."""
    _registry_lock(state)
    try:
        if handle in state[].votes:
            var vote = state[].votes.pop(handle)
            _registry_unlock(state)
            return vote^
        _registry_unlock(state)
        return KnnVoteCache()
    except error:
        _registry_unlock(state)
        raise error^


def _locked_put_cache(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin], handle: Int,
                      serial: Int, var cache: KnnIndexCache) raises:
    """Store the search's cache back when the handle still holds the upload
    it was built on and no other search stored one meanwhile; otherwise the
    cache is destroyed here, outside the lock."""
    var keep = False
    _registry_lock(state)
    try:
        if handle in state[].entries and handle not in state[].caches:
            keep = state[].entries[handle].serial == serial
    except error:
        _registry_unlock(state)
        raise error^
    if keep:
        state[].caches[handle] = cache^
        _registry_unlock(state)
    else:
        _registry_unlock(state)
        _ = cache^


def _locked_put_vote(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin], handle: Int,
                     serial: Int, var vote: KnnVoteCache) raises:
    """`_locked_put_cache` for the vote cache."""
    var keep = False
    _registry_lock(state)
    try:
        if handle in state[].entries and handle not in state[].votes:
            keep = state[].entries[handle].serial == serial
    except error:
        _registry_unlock(state)
        raise error^
    if keep:
        state[].votes[handle] = vote^
        _registry_unlock(state)
    else:
        _registry_unlock(state)
        _ = vote^


def _locked_drop_caches(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin], handle: Int) raises:
    """Remove the handle's stored caches; destroyed outside the lock."""
    var cache = KnnIndexCache()
    var vote = KnnVoteCache()
    _registry_lock(state)
    try:
        if handle in state[].caches:
            cache = state[].caches.pop(handle)
        if handle in state[].votes:
            vote = state[].votes.pop(handle)
        _registry_unlock(state)
    except error:
        _registry_unlock(state)
        raise error^
    _ = vote^
    _ = cache^


def _locked_view(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin],
                 handle: Int) raises -> _ResidentView:
    _registry_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident k-NN index handle")
        ref entry = state[].entries[handle]
        var view = _ResidentView(entry.ctx.copy(), entry.index.copy(), entry.n_index, entry.n_features,
                                 entry.device, entry.serial)
        _registry_unlock(state)
        return view^
    except error:
        _registry_unlock(state)
        raise error^


def _locked_replace(state: UnsafePointer[KnnIndexRegistry, MutAnyOrigin], handle: Int,
                    var moved: ResidentKnnIndex) raises -> ResidentKnnIndex:
    """Put `moved` under `handle` with a new serial; the entry it replaces
    is returned so it is destroyed outside the lock. A handle released
    meanwhile is refused."""
    _registry_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident k-NN index handle")
        moved.serial = state[].next_serial
        state[].next_serial += 1
        var replaced = state[].entries.pop(handle)
        state[].entries[handle] = moved^
        _registry_unlock(state)
        return replaced^
    except error:
        _registry_unlock(state)
        raise error^


def knn_index_prepare(
    index_ptr: MutPointer[Float32, MutUntrackedOrigin], n_index: Int, n_features: Int,
) raises -> Int:
    """Upload the index once; the handle every later search names."""
    var state = _registry()
    # The upload runs outside the lock; only the dictionary step is locked.
    var entry = ResidentKnnIndex(index_ptr, n_index, n_features)
    return _locked_insert(state, entry^)


def knn_index_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later call."""
    var state = _registry()
    var released = _locked_pop(state, handle)
    # Caches a search holds now are dropped by its put-back (the entry is gone).
    _locked_drop_caches(state, handle)
    # Destroyed outside the lock: a search still running holds its own
    # handle copies of the context and the index.
    _ = released^


def _knn_resident_view(handle: Int, n_index: Int, n_features: Int, what: String) raises -> _ResidentView:
    """The entry's context and index buffer for one call, checked against
    the call's shape, on the device the calling thread selected (moved there
    first when the entry lives on another device)."""
    var device = _device_of(neural_device_plan().root)
    var state = _registry()
    var view = _locked_view(state, handle)
    if view.n_index != n_index or view.n_features != n_features:
        raise Error(
            what + ": the handle holds a "
            + String(view.n_index) + " x " + String(view.n_features)
            + " index, the call names " + String(n_index) + " x "
            + String(n_features)
        )
    if view.device == device:
        return view^
    # The device index is part of the resident key: move the same bytes to
    # the selected device (the new entry reads the plan again and opens its
    # root), then publish the moved entry under the same handle. The caches
    # were built on the old device: they are dropped, and a search still
    # holding one does not put it back (new serial).
    var host = List[Float32](length=n_index * n_features, fill=Float32(0))
    var host_ptr = rebind[MutPointer[Float32, MutUntrackedOrigin]](host.unsafe_ptr())
    view.ctx.enqueue_copy(dst_ptr=host_ptr, src_buf=view.index)
    view.ctx.synchronize()
    var moved = ResidentKnnIndex(host_ptr, n_index, n_features)
    _ = host^
    var replaced = _locked_replace(state, handle, moved^)
    _locked_drop_caches(state, handle)
    _ = replaced^
    _ = view^
    return _locked_view(state, handle)


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
    negate_products: Bool = False,
) raises -> Int:
    """`knn_search_resident` over the handle's index and context. The shape
    the caller names must be the shape that was uploaded; `index_ptr` is
    the same bytes on the host, read for the refusals only."""
    var state = _registry()
    var view = _knn_resident_view(handle, n_index, n_features, "knn_search_resident")
    var cache = _locked_take_cache(state, handle)
    var used = knn_search_resident(
        view.ctx, view.index, index_ptr, n_index, queries_ptr, n_queries,
        n_features, k, out_dist_ptr, out_idx_ptr, return_sqrt,
        requested_query_tile, knn_method, metric, metric_arg,
        MutPointer(to=cache).unsafe_origin_cast[MutAnyOrigin](),
        negate_products,
    )
    _locked_put_cache(state, handle, view.serial, cache^)
    _ = view^
    return used


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
    var state = _registry()
    var view = _knn_resident_view(handle, n_index, n_features, "knn resident predict")
    var cache = _locked_take_cache(state, handle)
    var vote = _locked_take_vote(state, handle)
    var used = knn_classifier_predict_resident(
        view.ctx, view.index, index_ptr, n_index, queries_ptr, n_queries,
        n_features, k, y_ptr, n_outputs, n_classes, out_labels_ptr,
        out_proba_ptr, out_uniq_ptr, want_proba, requested_query_tile,
        metric, metric_arg, weights,
        MutPointer(to=cache).unsafe_origin_cast[MutAnyOrigin](),
        MutPointer(to=vote).unsafe_origin_cast[MutAnyOrigin](),
    )
    _locked_put_vote(state, handle, view.serial, vote^)
    _locked_put_cache(state, handle, view.serial, cache^)
    _ = view^
    return used


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
    var state = _registry()
    var view = _knn_resident_view(handle, n_index, n_features, "knn resident predict")
    var cache = _locked_take_cache(state, handle)
    var used = knn_regressor_predict_resident(
        view.ctx, view.index, index_ptr, n_index, queries_ptr, n_queries,
        n_features, k, y_ptr, n_outputs, out_ptr, requested_query_tile,
        metric, metric_arg, weights,
        MutPointer(to=cache).unsafe_origin_cast[MutAnyOrigin](),
    )
    _locked_put_cache(state, handle, view.serial, cache^)
    _ = view^
    return used
