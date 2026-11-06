# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident KMeans centroids (lane migrate-classical-b2).

`KMeans.predict` and `KMeans.transform` used to upload the fitted
centroids on every call. The first call after a fit (or a load) now
uploads them ONCE through `kmeans_centers_prepare`, the handle lives in
the model's `_resident_holder` with the binding that minted it
(bindings/kmeans_native.mojo), and every later call reads the device copy
(`kmeans_predict_resident`, `kmeans_transform_resident`). The handle is
released by a refit, a change of the centroid array, or the native
finalizer when the instance is dropped (bindings/host_archive_lifetime.mojo
"entry" plans).

The registry is `neighbors/resident_index.mojo`'s (DEVIATION 2921): one
process-wide `_Global` keyed by a monotonically increasing handle, the
caller holding the GIL across prepare, query and release; an entry owns its
`DeviceContext` and the buffer, destroyed before the context (DEVIATION
1946).

THREADS AND DEVICES (lane pass2-cluster-nbr, hazard H8). The query
entries run with the GIL released (bindings/_mojolearn.mojo), so a refit on
another thread can release the handle while a query reads it. Every
registry read and write now takes the registry's lock word, and a query
copies the entry's context and buffer HANDLES (reference counted) under it:
a release while the query runs drops only the registry's references, and
the device memory lives until the query's copies go. Each entry records the
device it was minted on; a query whose thread is bound to another device
(the device plan's root) moves the centroids there first (device to host to
device, the same bytes), so a handle never runs on a device the caller did
not select.

WHAT DOES NOT CHANGE. A resident predict runs `kmeans_predict_device` and a
resident transform `kmeans_transform_device` (cluster/estimator.mojo),
the bodies `kmeans_predict` and `kmeans_transform` run after their own
upload, over the same centroid bytes: the bits are the same.
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx, neural_device_plan
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from cluster.estimator import kmeans_predict_device, kmeans_transform_device

comptime _DEVCTX_SLOT = "MojoCoreContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoCoreContextFast"


struct ResidentKMeansCenters(Movable):
    """One fitted model's `n_clusters x n_features` float32 centroids on
    the device, uploaded once."""

    var ctx: DeviceContext
    var centers: DeviceBuffer[DType.float32]
    var n_clusters: Int
    var n_features: Int
    # The device the copy lives on: the device plan's root at the upload,
    # 0 when no plan names another device (process_ctx's own rule).
    var device: Int

    def __init__(
        out self,
        centers_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_clusters: Int,
        n_features: Int,
    ) raises:
        if n_clusters <= 0 or n_features <= 0:
            raise Error("kmeans_centers_prepare: n_clusters and n_features must be positive")
        var dplan = neural_device_plan()
        var ctx = process_ctx[_DEVCTX_SLOT](dplan)
        var centers = ctx.enqueue_create_buffer[DType.float32](n_clusters * n_features)
        ctx.enqueue_copy(dst_buf=centers, src_ptr=centers_ptr)
        ctx.synchronize()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.device = _device_of(dplan.root)
        self.centers = centers^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.centers^
        _ = self.ctx^


def _device_of(root: Int) -> Int:
    """The device a context made for plan root `root` runs on: `process_ctx`
    opens `root` when it is positive and device 0 otherwise."""
    return root if root > 0 else 0


struct KMeansCentersRegistry(Movable):
    var entries: Dict[Int, ResidentKMeansCenters]
    var next_id: Int
    # One lock word: queries run with the GIL released (see the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ResidentKMeansCenters]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime KMEANS_CENTERS_REGISTRY = _Global[
    StorageType=KMeansCentersRegistry,
    name=(
        "MojoKMeansResidentCentersIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoKMeansResidentCentersFast"
    ),
    init_fn=KMeansCentersRegistry.__init__,
]


def _registry_lock(state: UnsafePointer[KMeansCentersRegistry, MutAnyOrigin]) raises:
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _registry_unlock(state: UnsafePointer[KMeansCentersRegistry, MutAnyOrigin]) raises:
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _registry() raises -> UnsafePointer[KMeansCentersRegistry, MutAnyOrigin]:
    return KMEANS_CENTERS_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct _ResidentView(Movable):
    """A query's own copies of one entry's context and buffer HANDLES
    (reference counted, the memory is shared) and its key words."""

    var ctx: DeviceContext
    var centers: DeviceBuffer[DType.float32]
    var n_clusters: Int
    var n_features: Int
    var device: Int

    def __init__(out self, var ctx: DeviceContext, var centers: DeviceBuffer[DType.float32],
                 n_clusters: Int, n_features: Int, device: Int):
        self.ctx = ctx^
        self.centers = centers^
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.device = device

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.centers^
        _ = self.ctx^


def _locked_insert(state: UnsafePointer[KMeansCentersRegistry, MutAnyOrigin],
                   var entry: ResidentKMeansCenters) raises -> Int:
    _registry_lock(state)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident KMeans centroid handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        state[].entries[handle] = entry^
        _registry_unlock(state)
        return handle
    except error:
        _registry_unlock(state)
        raise error^


def _locked_pop(state: UnsafePointer[KMeansCentersRegistry, MutAnyOrigin],
                handle: Int) raises -> ResidentKMeansCenters:
    _registry_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KMeans centroid handle")
        var popped = state[].entries.pop(handle)
        _registry_unlock(state)
        return popped^
    except error:
        _registry_unlock(state)
        raise error^


def _locked_view(state: UnsafePointer[KMeansCentersRegistry, MutAnyOrigin],
                 handle: Int) raises -> _ResidentView:
    _registry_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KMeans centroid handle")
        ref entry = state[].entries[handle]
        var view = _ResidentView(entry.ctx.copy(), entry.centers.copy(), entry.n_clusters, entry.n_features,
                                 entry.device)
        _registry_unlock(state)
        return view^
    except error:
        _registry_unlock(state)
        raise error^


def _locked_replace(state: UnsafePointer[KMeansCentersRegistry, MutAnyOrigin], handle: Int,
                    var moved: ResidentKMeansCenters) raises -> ResidentKMeansCenters:
    """Put `moved` under `handle`; the entry it replaces is returned so it
    is destroyed outside the lock. A handle released meanwhile is refused."""
    _registry_lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KMeans centroid handle")
        var replaced = state[].entries.pop(handle)
        state[].entries[handle] = moved^
        _registry_unlock(state)
        return replaced^
    except error:
        _registry_unlock(state)
        raise error^


def kmeans_centers_prepare(
    centers_ptr: MutPointer[Float32, MutUntrackedOrigin], n_clusters: Int, n_features: Int,
) raises -> Int:
    """Upload the centroids once; the handle every later query names."""
    var state = _registry()
    # The upload runs outside the lock; only the dictionary step is locked.
    var entry = ResidentKMeansCenters(centers_ptr, n_clusters, n_features)
    return _locked_insert(state, entry^)


def kmeans_centers_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later call."""
    var state = _registry()
    var released = _locked_pop(state, handle)
    # Destroyed outside the lock: a query still running holds its own
    # handle copies of the context and the buffer.
    _ = released^


def _kmeans_resident_view(handle: Int, n_clusters: Int, n_features: Int) raises -> _ResidentView:
    """The entry's context and centroid buffer for one query, checked
    against the call's shape, on the device the calling thread selected
    (moved there first when the entry lives on another device)."""
    var device = _device_of(neural_device_plan().root)
    var state = _registry()
    var view = _locked_view(state, handle)
    if view.n_clusters != n_clusters or view.n_features != n_features:
        raise Error(
            "kmeans resident: the handle holds " + String(view.n_clusters) + " x "
            + String(view.n_features) + " centroids, the call names "
            + String(n_clusters) + " x " + String(n_features)
        )
    if view.device == device:
        return view^
    # The device index is part of the resident key: move the same bytes to
    # the selected device (the new entry reads the plan again and opens its
    # root), then publish the moved entry under the same handle.
    var host = List[Float32](length=n_clusters * n_features, fill=Float32(0))
    var host_ptr = rebind[MutPointer[Float32, MutUntrackedOrigin]](host.unsafe_ptr())
    view.ctx.enqueue_copy(dst_ptr=host_ptr, src_buf=view.centers)
    view.ctx.synchronize()
    var moved = ResidentKMeansCenters(host_ptr, n_clusters, n_features)
    _ = host^
    var fresh = _ResidentView(moved.ctx.copy(), moved.centers.copy(), moved.n_clusters, moved.n_features,
                              moved.device)
    var replaced = _locked_replace(state, handle, moved^)
    _ = replaced^
    _ = view^
    return fresh^


def kmeans_predict_resident(
    handle: Int,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    out_labels_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    metric: Int,
) raises:
    """`kmeans_predict` with the handle's centroids: X uploaded, the same
    device pass (`kmeans_predict_device`), the labels read back."""
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error(
            "kmeans_predict needs n_samples, n_features and n_clusters >= 1: got "
            + String(n_samples) + ", " + String(n_features) + ", " + String(n_clusters)
        )
    var view = _kmeans_resident_view(handle, n_clusters, n_features)
    var x = view.ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var labels = view.ctx.enqueue_create_buffer[DType.uint32](n_samples)
    view.ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    kmeans_predict_device(
        view.ctx, x, n_samples, n_features, n_clusters, view.centers, labels, metric
    )
    view.ctx.enqueue_copy(dst_ptr=out_labels_ptr, src_buf=labels)
    view.ctx.synchronize()
    _ = x^
    _ = labels^
    _ = view^


def kmeans_transform_resident(
    handle: Int,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    metric: Int,
) raises:
    """`kmeans_transform` with the handle's centroids: X uploaded, the same
    device body (`kmeans_transform_device`) writing `out_ptr`."""
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error(
            "kmeans_transform needs n_samples, n_features and n_clusters >= 1: got "
            + String(n_samples) + ", " + String(n_features) + ", " + String(n_clusters)
        )
    var view = _kmeans_resident_view(handle, n_clusters, n_features)
    var x = view.ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    view.ctx.synchronize()
    view.ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    kmeans_transform_device(
        view.ctx, x, n_samples, n_features, n_clusters, view.centers, out_ptr, metric
    )
    _ = x^
    _ = view^
