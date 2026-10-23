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

WHAT DOES NOT CHANGE. A resident predict runs `kmeans_predict_device` and a
resident transform `kmeans_transform_device` (cluster/estimator.mojo),
the bodies `kmeans_predict` and `kmeans_transform` run after their own
upload, over the same centroid bytes: the bits are the same.
"""
from std.ffi import _Global

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
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

    def __init__(
        out self,
        centers_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_clusters: Int,
        n_features: Int,
    ) raises:
        if n_clusters <= 0 or n_features <= 0:
            raise Error("kmeans_centers_prepare: n_clusters and n_features must be positive")
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var centers = ctx.enqueue_create_buffer[DType.float32](n_clusters * n_features)
        ctx.enqueue_copy(dst_buf=centers, src_ptr=centers_ptr)
        ctx.synchronize()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.centers = centers^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.centers^
        _ = self.ctx^


struct KMeansCentersRegistry(Movable):
    var entries: Dict[Int, ResidentKMeansCenters]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentKMeansCenters]()
        self.next_id = 1


comptime KMEANS_CENTERS_REGISTRY = _Global[
    StorageType=KMeansCentersRegistry,
    name=(
        "MojoKMeansResidentCentersIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoKMeansResidentCentersFast"
    ),
    init_fn=KMeansCentersRegistry.__init__,
]


def kmeans_centers_prepare(
    centers_ptr: MutPointer[Float32, MutUntrackedOrigin], n_clusters: Int, n_features: Int,
) raises -> Int:
    """Upload the centroids once; the handle every later query names."""
    var state = KMEANS_CENTERS_REGISTRY.get_or_create_ptr()
    var entry = ResidentKMeansCenters(centers_ptr, n_clusters, n_features)
    if state[].next_id == 9223372036854775807:
        raise Error("resident KMeans centroid handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def kmeans_centers_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later call."""
    var state = KMEANS_CENTERS_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident KMeans centroid handle")
    var released = state[].entries.pop(handle)
    _ = released^


def _kmeans_resident_check(handle: Int, n_clusters: Int, n_features: Int) raises:
    var state = KMEANS_CENTERS_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident KMeans centroid handle")
    ref entry = state[].entries[handle]
    if entry.n_clusters != n_clusters or entry.n_features != n_features:
        raise Error(
            "kmeans resident: the handle holds " + String(entry.n_clusters) + " x "
            + String(entry.n_features) + " centroids, the call names "
            + String(n_clusters) + " x " + String(n_features)
        )


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
    _kmeans_resident_check(handle, n_clusters, n_features)
    var state = KMEANS_CENTERS_REGISTRY.get_or_create_ptr()
    ref entry = state[].entries[handle]
    var x = entry.ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var labels = entry.ctx.enqueue_create_buffer[DType.uint32](n_samples)
    entry.ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    kmeans_predict_device(
        entry.ctx, x, n_samples, n_features, n_clusters, entry.centers, labels, metric
    )
    entry.ctx.enqueue_copy(dst_ptr=out_labels_ptr, src_buf=labels)
    entry.ctx.synchronize()


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
    _kmeans_resident_check(handle, n_clusters, n_features)
    var state = KMEANS_CENTERS_REGISTRY.get_or_create_ptr()
    ref entry = state[].entries[handle]
    var x = entry.ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    entry.ctx.synchronize()
    entry.ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    kmeans_transform_device(
        entry.ctx, x, n_samples, n_features, n_clusters, entry.centers, out_ptr, metric
    )
