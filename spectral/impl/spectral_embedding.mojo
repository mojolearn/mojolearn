# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/spectral/spectral_embedding.hpp` + `spectral_embedding.cu`
(v26.08.00) and `include/cuml/manifold/spectral_embedding.hpp`:
`ML::SpectralEmbedding::params`, `to_cuvs`, and the three `transform`
overloads -- every one a forward to cuVS (`spectral_embedding.cu:14-47`).

cuML's file is 27 + 49 lines and does no arithmetic of its own; the
algorithm is `spectral/impl/preprocessing/detail/
spectral_embedding.mojo`. The `(rows, cols, vals)` overload (`:32-48`) builds
a COO view and calls the COO overload; ours takes the `CooGraph` directly
(the view is the struct).

The Python surface (`spectral_embedding.pyx:294`) passes `n_components + 1`
when `drop_first`; that arithmetic belongs to the Python caller and is
documented in README's HAND-OFF, not performed here.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from spectral.impl.preprocessing.detail.spectral_embedding import (
    SpectralEmbeddingParams,
    transform_dataset,
    transform_dense_keep,
    transform_graph,
)
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.spectral_predict_common import SpectralPredictionState


@fieldwise_init
struct MLSpectralEmbeddingParams(Copyable, Movable):
    """`ML::SpectralEmbedding::params` (`cuml/manifold/spectral_embedding.hpp:
    21-36`): `n_components`, `n_neighbors`, `norm_laplacian`, `drop_first`,
    `std::optional<uint64_t> seed`. No `tolerance` field: cuML's struct
    predates cuVS's and `to_cuvs` leaves cuVS's default `1e-5f`."""

    var n_components: Int
    var n_neighbors: Int
    var norm_laplacian: Bool
    var drop_first: Bool
    var has_seed: Bool
    var seed: UInt64


def to_cuvs(config: MLSpectralEmbeddingParams) -> SpectralEmbeddingParams:
    """`to_cuvs` (`spectral_embedding.hpp:14-25`), field for field; the
    cuVS `tolerance` keeps its default `1e-5f`."""
    return SpectralEmbeddingParams(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        norm_laplacian=config.norm_laplacian,
        drop_first=config.drop_first,
        tolerance=Float32(1e-5),
        has_seed=config.has_seed,
        seed=config.seed,
    )


def transform(
    ctx: DeviceContext,
    config: MLSpectralEmbeddingParams,
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    mut embedding: List[Float32],
    mut trace: IdentityTrace,
    tolerance: Float32 = Float32(1e-5),
) raises -> Int:
    """`transform(handle, config, dataset, embedding)` (`:14-21`). Returns
    the number of embedding columns (`n_components`, or `n_components - 1`
    when `drop_first`). `tolerance` is the Lanczos tolerance cuVS's params
    carry (default its `1e-5f`); scikit-learn's float `eigen_tol` sets it
    (lane/algos-decomp, 2026-09-27)."""
    var params = to_cuvs(config)
    params.tolerance = tolerance
    return transform_dataset(ctx, params, dataset, n_samples, n_features, embedding, trace)


def transform_connectivity(
    ctx: DeviceContext,
    config: MLSpectralEmbeddingParams,
    connectivity_graph: CooGraph,
    mut embedding: List[Float32],
    mut trace: IdentityTrace,
    tolerance: Float32 = Float32(1e-5),
) raises -> Int:
    """`transform(handle, config, connectivity_graph, embedding)` (`:22-29`)
    and the `(rows, cols, vals)` form (`:31-47`). `tolerance` as in
    `transform`."""
    var params = to_cuvs(config)
    params.tolerance = tolerance
    return transform_graph(ctx, params, connectivity_graph, embedding, trace)


def transform_dense_connectivity(
    ctx: DeviceContext,
    config: MLSpectralEmbeddingParams,
    dense: DeviceBuffer[DType.float32],
    n: Int,
    m: Int,
    var indptr: DeviceBuffer[DType.int32],
    mut embedding: List[Float32],
    mut trace: IdentityTrace,
    tolerance: Float32 = Float32(1e-5),
) raises -> Int:
    """`transform_connectivity` on a DENSE device affinity (lane
    cpu2-l9-neighbors): the diagonal is dropped as the COO route's
    `coo_remove_diagonal` drops it, then re-inserted as zero by the
    Laplacian's construction."""
    var params = to_cuvs(config)
    params.tolerance = tolerance
    var state = SpectralPredictionState()
    return transform_dense_keep(ctx, params, dense, n, m, True, indptr^, embedding, state, False, trace)
