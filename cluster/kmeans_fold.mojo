# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KMeans fit, predict and transform over a model-selection fold resident on
the device (lane a7d-device-x-decomp-cluster, 2026-11-09; the model is
decomposition/pca_fold.mojo).

`kmeans_fit` / `kmeans_predict` / `kmeans_transform` (cluster/estimator.mojo)
upload X from host words. Under cross-validation the fold's rows are already
resident in model selection's store (core/msel_device_x.mojo): these entries
gather them on the device straight into the kernel's X buffer (no host
gather, no upload), with the finite and range tests in the same pass, upload
only the centroids, run the same device bodies and return the two status
words with the results. Only GPU bindings import this file (the store lives
on the device).

Numerics: the same rows in the same order, the same kernels on the same
words as the host-address path: no bit changes.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_fill
from core.msel_device_x import msel_device_x_f32
from cluster.estimator import (
    KMeansFitResult,
    kmeans_fit_device,
    kmeans_predict_device,
    kmeans_transform_device,
)
from cluster.impl.kmeans_params import KMeansParams


struct KMeansFoldFit(Copyable, Movable):
    """The fold fit's result and its two status words (finite, range)."""
    var fit: KMeansFitResult
    var finite: Int32
    var in_range: Int32

    def __init__(out self, fit: KMeansFitResult, finite: Int32, in_range: Int32):
        self.fit = fit
        self.finite = finite
        self.in_range = in_range


def _kmeans_fold_words(ctx: DeviceContext, mut status: DeviceBuffer[DType.int32]) raises -> Tuple[Int32, Int32]:
    """The two status words, read after the gather (one synchronisation)."""
    var flags = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var finite = flags.unsafe_ptr()[0]
    var in_range = flags.unsafe_ptr()[1]
    _ = flags^
    return (finite, in_range)


def kmeans_fit_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    out_centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_labels_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    max_iter: Int,
    tol: Float64,
    seed: UInt64,
    n_init: Int,
    init: Int,
    metric: Int,
    oversampling_factor: Float64,
) raises -> KMeansFoldFit:
    """The unit-weight fit of fold rows `index_id` of the resident base
    `base_id` (`base_rows` x `n_features`). `ctx` is
    `msel_fold_entry_context()`. The status words are read before the fit
    (the scale plan synchronises there anyway): a flagged fold is not fit
    and the door refuses."""
    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_samples, x, status.unsafe_ptr(), 0)
    var words = _kmeans_fold_words(ctx, status)
    if words[0] != 0 or words[1] != 0:
        _ = x^
        _ = status^
        return KMeansFoldFit(KMeansFitResult(0.0, 0, 0.0, 0.0), words[0], words[1])
    var fit = kmeans_fit_device(
        ctx, x, n_samples, n_features, n_clusters, out_centroids_ptr, out_labels_ptr,
        max_iter, tol, seed, n_init, init, metric, oversampling_factor,
    )
    _ = x^
    _ = status^
    return KMeansFoldFit(fit, words[0], words[1])


def kmeans_query_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: Int,
    metric: Int,
    transform: Bool,
) raises -> Tuple[Int32, Int32]:
    """`kmeans_predict` (labels, `out_ptr` n_samples uint32) or with
    `transform` `kmeans_transform` (distances, `out_ptr` n_samples x
    n_clusters float32) of fold rows `index_id` of the resident base. The
    same device bodies as the host-address entries. Returns (finite word,
    range word); on a flagged fold the door discards `out_ptr`."""
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error("kmeans fold query needs n_samples, n_features and n_clusters >= 1")
    var params = KMeansParams.default()
    params.n_clusters = n_clusters
    params.metric = metric
    params.validate()
    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var centroids = ctx.enqueue_create_buffer[DType.float32](n_clusters * n_features)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_samples, x, status.unsafe_ptr(), 0)
    ctx.enqueue_copy(dst_buf=centroids, src_ptr=centroids_ptr)
    var flags = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=status)
    if transform:
        # kmeans_transform_device downloads into out_ptr and synchronises.
        kmeans_transform_device(
            ctx, x, n_samples, n_features, n_clusters, centroids,
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_ptr), metric,
        )
    else:
        var labels = ctx.enqueue_create_buffer[DType.uint32](n_samples)
        kmeans_predict_device(ctx, x, n_samples, n_features, n_clusters, centroids, labels, metric)
        ctx.enqueue_copy(dst_ptr=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=out_ptr), src_buf=labels)
        ctx.synchronize()
        _ = labels^
    var finite = flags.unsafe_ptr()[0]
    var in_range = flags.unsafe_ptr()[1]
    _ = flags^
    _ = x^
    _ = centroids^
    _ = status^
    return (finite, in_range)
