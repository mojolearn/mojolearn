# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/spectral/spectral_clustering.cu` and
`include/cuml/cluster/spectral_clustering.hpp` (v26.08.00):
`ML::SpectralClustering::params`, `to_cuvs` (`:22-33`) and the three
`fit_predict` overloads (`:35-64`), every one a forward to
`cuvs::cluster::spectral::fit_predict`.

No arithmetic here. The algorithm is `spectral/impl/cluster/detail/
spectral.mojo` (embedding, then k-means). cuML's Python
(`spectral_clustering.pyx`) defaults `n_components` to `n_clusters` when
`None`, `n_neighbors=10`, `n_init=10`, `eigen_tol='auto'` (cuVS's `1e-5`),
and refuses `affinity` outside `{'nearest_neighbors', 'precomputed'}` and
`assign_labels != 'kmeans'` -- those are the Python caller's and are in the
README's HAND-OFF.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from spectral.host.spectral_predict_host import SpectralPredictionState
from spectral.impl.cluster.detail.spectral import (
    SpectralClusteringParams,
    fit_predict_dataset_keep,
    fit_predict_graph_keep,
)
from spectral.impl.sparse.coo import CooGraph


@fieldwise_init
struct MLSpectralClusteringParams(Copyable, Movable):
    """`ML::SpectralClustering::params` (`spectral_clustering.hpp:19-32`):
    `n_clusters`, `n_components`, `n_init`, `n_neighbors`, `eigen_tol`,
    `seed`."""

    var n_clusters: Int
    var n_components: Int
    var n_init: Int
    var n_neighbors: Int
    var eigen_tol: Float32
    var seed: UInt64


def to_cuvs(config: MLSpectralClusteringParams) -> SpectralClusteringParams:
    """`to_cuvs` (`spectral_clustering.cu:22-33`): `tolerance = eigen_tol`,
    `rng_state = RngState(seed)`."""
    return SpectralClusteringParams(
        n_clusters=config.n_clusters,
        n_components=config.n_components,
        n_init=config.n_init,
        n_neighbors=config.n_neighbors,
        tolerance=config.eigen_tol,
        seed=config.seed,
    )


def fit_predict(
    ctx: DeviceContext,
    config: MLSpectralClusteringParams,
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut trace: IdentityTrace,
) raises:
    """`fit_predict(handle, config, dataset, labels)` (`:35-41`)."""
    var state = SpectralPredictionState()
    fit_predict_keep(ctx, config, dataset, n_samples, n_features, labels, embedding_out, state, False, trace)


def fit_predict_keep(
    ctx: DeviceContext,
    config: MLSpectralClusteringParams,
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut state: SpectralPredictionState,
    keep: Bool,
    mut trace: IdentityTrace,
) raises:
    """`fit_predict`, with `state` receiving the prediction data when `keep`
    (lane/spectral-predict, 2026-09-15)."""
    fit_predict_dataset_keep(
        ctx, to_cuvs(config), dataset, n_samples, n_features, labels, embedding_out, state, keep, trace
    )


def fit_predict_connectivity(
    ctx: DeviceContext,
    config: MLSpectralClusteringParams,
    connectivity_graph: CooGraph,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut trace: IdentityTrace,
) raises:
    """`fit_predict(handle, config, connectivity_graph, labels)` (`:43-49`)
    and the `(rows, cols, vals)` form (`:51-64`)."""
    var state = SpectralPredictionState()
    fit_predict_connectivity_keep(ctx, config, connectivity_graph, labels, embedding_out, state, False, trace)


def fit_predict_connectivity_keep(
    ctx: DeviceContext,
    config: MLSpectralClusteringParams,
    connectivity_graph: CooGraph,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut state: SpectralPredictionState,
    keep: Bool,
    mut trace: IdentityTrace,
) raises:
    """`fit_predict_connectivity`, keeping the prediction data when `keep`."""
    fit_predict_graph_keep(ctx, to_cuvs(config), connectivity_graph, labels, embedding_out, state, keep, trace)
