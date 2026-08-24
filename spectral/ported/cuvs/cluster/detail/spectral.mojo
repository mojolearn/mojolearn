"""cuVS `cpp/src/cluster/detail/spectral.cuh` (v26.08.00):
`cuvs::cluster::spectral::detail::fit_predict`, the graph overload
(`:17-62`) and the dataset overload (`:64-80`). This is what cuML 26.08's
`ML::SpectralClustering::fit_predict` reaches.

RUNG 2 = RUNG 1 + k-MEANS, and nothing else. The graph overload (`:26-61`):
  spectral_embedding params:  n_components = config.n_components,
                              n_neighbors, norm_laplacian = TRUE,
                              drop_first = FALSE, seed = rng_state.seed,
                              tolerance = config.tolerance
  kmeans params:              n_clusters, rng_state, n_init,
                              oversampling_factor = 0.0  (classic k-means++,
                              `detail/kmeans.cuh:910-915`)
  transform -> transpose to row-major -> kmeans::fit_predict(embedding) ->
  labels.
`drop_first = false` means the embedding INCLUDES the trivial eigenvector
(a constant column after the `/ diagonal` scaling when the graph is
connected): theirs clusters on all `n_components` columns and so do we.

THE k-MEANS IS `cluster/`'s PORTED `fit_predict` (`cluster/ported/cluster/
kmeans.mojo`), called with a `KMeansParams` whose `oversampling_factor` is
`0.0` -- the public host surface `cluster/estimator.mojo::kmeans_fit` does
not expose that field (it keeps cuVS's default `2.0`, the scalable arm), so
the device-resident entry one level down is the one whose interface fits.
The device setup around it (the row norms, the fixed-point scales) is
copied from `kmeans_fit` line for line; `cluster/` is frozen for this lane
and is READ, IMPORTED, not edited.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.estimator import plan_sum_scale
from cluster.ported.cluster.detail.kmeans_common import metric_is_sqrt
from cluster.ported.cluster.kmeans import fit_predict as kmeans_fit_predict
from cluster.ported.cluster.kmeans_params import (
    INIT_KMEANS_PLUS_PLUS,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from core.identity_trace import IdentityTrace
from core.row_norms import NORM_TPB, row_norm_kernel
from mojo_only.fixed_point import choose_scale
from spectral.mojo_only.device_io import download_u32, upload_f32
from spectral.ported.cuvs.preprocessing.spectral.detail.spectral_embedding import (
    SpectralEmbeddingParams,
    create_connectivity_graph,
    transform_graph,
)
from spectral.ported.sparse.coo import CooGraph


@fieldwise_init
struct SpectralClusteringParams(Copyable, Movable):
    """`cuvs::cluster::spectral::params` (`spectral.hpp:27-46`):
    `n_clusters`, `n_components`, `n_init`, `n_neighbors`, `tolerance`,
    `rng_state{0}` (its seed)."""

    var n_clusters: Int
    var n_components: Int
    var n_init: Int
    var n_neighbors: Int
    var tolerance: Float32
    var seed: UInt64


def fit_predict_graph(
    ctx: DeviceContext,
    config: SpectralClusteringParams,
    connectivity_graph: CooGraph,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut trace: IdentityTrace,
) raises:
    """`fit_predict` on a COO (`:17-62`). `labels` gets `n` cluster ids in
    `[0, n_clusters)`; `embedding_out` gets the `n x n_components` row-major
    embedding k-means ran on (ours exposes it for the gates)."""
    var n_samples = connectivity_graph.n
    if config.n_clusters < 1 or config.n_clusters > n_samples:
        raise Error(
            "spectral clustering: n_clusters=" + String(config.n_clusters)
            + " must satisfy 1 <= n_clusters <= n_samples"
        )
    var emb_params = SpectralEmbeddingParams(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        norm_laplacian=True,
        drop_first=False,
        tolerance=config.tolerance,
        has_seed=True,
        seed=config.seed,
    )
    var n_out = transform_graph(ctx, emb_params, connectivity_graph, embedding_out, trace)
    # embedding_row_major (:47-52): ours is row-major already.
    var n_features = n_out

    # --- kmeans::fit_predict (:54-61), through cluster/'s ported entry, with
    # the device setup `cluster/estimator.mojo::kmeans_fit` performs.
    var h = ctx.enqueue_create_host_buffer[DType.float32](n_samples * n_features)
    ctx.synchronize()
    for i in range(n_samples * n_features):
        h.unsafe_ptr().unsafe_store(i, embedding_out[i])
    var sum_scale = plan_sum_scale(h.unsafe_ptr(), n_samples, n_features)
    var weight_scale = choose_scale(Float64(n_samples), n_samples)
    var cd = config.n_clusters * n_features
    var x = upload_f32(ctx, embedding_out)
    var ones = List[Float32]()
    for _ in range(n_samples):
        ones.append(Float32(1.0))
    var weights = upload_f32(ctx, ones)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var d_labels = ctx.enqueue_create_buffer[DType.uint32](n_samples)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_samples)
    ctx.enqueue_memset(centroids, Float32(0.0))
    ctx.enqueue_memset(d_labels, UInt32(0))
    ctx.enqueue_memset(min_dist, Float32(0.0))
    ctx.synchronize()
    var take_sqrt = Int32(0)
    if metric_is_sqrt(METRIC_L2_EXPANDED):
        take_sqrt = Int32(1)
    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_features),
        take_sqrt,
        grid_dim=(n_samples, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()
    var params = KMeansParams.default()
    params.n_clusters = config.n_clusters
    params.init = INIT_KMEANS_PLUS_PLUS
    params.metric = METRIC_L2_EXPANDED
    params.seed = config.seed
    params.n_init = config.n_init
    params.oversampling_factor = 0.0  # `kmeans_config.oversampling_factor = 0.0` (:42)
    _ = kmeans_fit_predict(
        ctx,
        x,
        x_norm,
        weights,
        centroids,
        d_labels,
        min_dist,
        params,
        n_samples,
        n_features,
        Float32(sum_scale),
        Float32(weight_scale),
    )
    var got = download_u32(ctx, d_labels, n_samples)
    labels.clear()
    for i in range(n_samples):
        labels.append(Int32(got[i]))
    trace.record_list_i32("spectral.labels", labels)
    _ = h^
    _ = x^
    _ = weights^
    _ = centroids^
    _ = d_labels^
    _ = x_norm^
    _ = min_dist^


def fit_predict_dataset(
    ctx: DeviceContext,
    config: SpectralClusteringParams,
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut trace: IdentityTrace,
) raises:
    """`fit_predict` on a dataset (`:64-80`): `create_connectivity_graph`
    with `embed_params.n_neighbors = config.n_neighbors`, then the graph
    overload."""
    var embed_params = SpectralEmbeddingParams.default_with(
        config.n_components, config.n_neighbors
    )
    var graph = create_connectivity_graph(ctx, embed_params, dataset, n_samples, n_features, trace)
    trace.record_list_i32("spectral.W.rows", graph.rows)
    trace.record_list_i32("spectral.W.cols", graph.cols)
    trace.record_list_f32("spectral.W.vals", graph.vals)
    fit_predict_graph(ctx, config, graph, labels, embedding_out, trace)
