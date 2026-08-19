"""The public surface: fit, predict, fit_predict, cluster_cost.

PORT OF `cuvs/src/cluster/kmeans.cuh` at cuVS `94c2819`. Partial. Do not
improve.

Their file is a dispatch layer: it takes mdspans, decides host or device
residency, picks an index type, and forwards to `detail::`. What is ported is
the SHAPE of that surface and the division of labor, not the mdspan
machinery, which has no counterpart.

The entry points, and which of them exist here:

    fit           PORTED
    predict       PORTED
    fit_predict   PORTED (theirs is fit then predict, and so is this)
    cluster_cost  PORTED
    transform     NOT PORTED, it materializes the full n x k distance matrix
                  and nothing in the fit path calls it

**`fit_predict` deserves one sentence because it looks redundant and is not.**
The fit already computed an assignment on its last iteration, so returning it
would be free. Neither cuVS nor scikit-learn does that: both run one more
full assignment against the FINAL centroids, because the last iteration's
labels belong to the centroids from BEFORE the final update. Returning the
stale ones is a real and well known off-by-one-iteration bug, and it would be
invisible in every aggregate metric.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.mojo_only.reduce_by_key import REDUCE_BY_KEY_TPB
from cluster.ported.cluster.detail.kmeans import (
    FitResult,
    kmeans_fit_main,
)
from cluster.ported.cluster.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
)
from cluster.ported.cluster.kmeans_params import (
    KMeansParams,
    get_centroids_batch_size,
    get_data_batch_size,
)


def fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    params: KMeansParams,
    n_samples: Int,
    n_features: Int,
    sum_scale: Float32,
    weight_scale: Float32,
) raises -> FitResult:
    """`cuvs::cluster::kmeans::fit`, device-resident inputs only.

    Theirs has no host-resident arm either: `kmeans_fit`
    (`detail/kmeans.cuh:811-951`) takes `device_matrix_view` and its only
    input-shape concessions are two RAFT_LOG_DEBUG lines saying `batch_samples`
    and `batch_centroids` will be ignored on the fused metrics
    (`:837-856`). This tree consumes device-resident data by construction,
    exactly as they do.
    """
    return kmeans_fit_main(
        ctx,
        x,
        weights,
        centroids,
        labels,
        params,
        n_samples,
        n_features,
        sum_scale,
        weight_scale,
    )


def predict(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut min_dist: DeviceBuffer[DType.float32],
    params: KMeansParams,
    n_samples: Int,
    n_features: Int,
) raises:
    """`kmeans_predict`, which is one assignment pass and nothing else.

    The caller supplies `x_norm` because predict against the same data that
    was fitted should not recompute it, which is the same reuse the fit makes
    across its iterations.
    """
    var data_batch = get_data_batch_size(params.batch_samples, n_samples)
    var centroid_batch = get_centroids_batch_size(
        params.batch_centroids, params.n_clusters
    )
    var centroid_norm = ctx.enqueue_create_buffer[DType.float32](
        params.n_clusters
    )
    var dist_buf = ctx.enqueue_create_buffer[DType.float32](
        data_batch * centroid_batch
    )
    ctx.synchronize()

    compute_centroid_norms(
        ctx,
        centroids,
        centroid_norm,
        params.n_clusters,
        n_features,
        params.metric,
    )
    min_cluster_and_distance_compute(
        ctx,
        x,
        x_norm,
        centroids,
        centroid_norm,
        dist_buf,
        labels,
        min_dist,
        n_samples,
        n_features,
        params.n_clusters,
        params.metric,
        params.batch_samples,
        params.batch_centroids,
    )
    ctx.synchronize()


def fit_predict(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut min_dist: DeviceBuffer[DType.float32],
    params: KMeansParams,
    n_samples: Int,
    n_features: Int,
    sum_scale: Float32,
    weight_scale: Float32,
) raises -> FitResult:
    """`fit_predict`: fit, then a FRESH assignment against the final centroids.

    See the module docstring for why the second pass is not redundant.
    """
    var result = fit(
        ctx,
        x,
        weights,
        centroids,
        labels,
        params,
        n_samples,
        n_features,
        sum_scale,
        weight_scale,
    )
    predict(
        ctx,
        x,
        x_norm,
        centroids,
        labels,
        min_dist,
        params,
        n_samples,
        n_features,
    )
    return result
