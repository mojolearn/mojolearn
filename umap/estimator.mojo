# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public UMAP boundary; graph construction is available, layout is not."""

from max.gpu.host import DeviceContext
from neighbors.estimator import knn_search
from umap.graph import FuzzySimplicialGraph, fuzzy_simplicial_graph
from umap.params import UMAPParams


def fuzzy_graph_from_data(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
) raises -> FuzzySimplicialGraph:
    """Reuse mojolearn's exact k-NN surface, then build UMAP memberships."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    var hx = ctx.enqueue_create_host_buffer[DType.float32](len(x_rowmajor))
    for i in range(len(x_rowmajor)):
        hx.unsafe_ptr().unsafe_store(i, x_rowmajor[i])
    var hd = ctx.enqueue_create_host_buffer[DType.float32](
        n_samples * params.n_neighbors
    )
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](
        n_samples * params.n_neighbors
    )
    _ = knn_search(
        ctx, hx.unsafe_ptr(), n_samples, hx.unsafe_ptr(), n_samples,
        n_features, params.n_neighbors, hd.unsafe_ptr(), hi.unsafe_ptr(),
    )
    var distances = List[Float32]()
    var indices = List[UInt32]()
    for i in range(n_samples * params.n_neighbors):
        distances.append(hd.unsafe_ptr().unsafe_load(i))
        indices.append(hi.unsafe_ptr().unsafe_load(i))
    _ = hx^
    _ = hd^
    _ = hi^
    return fuzzy_simplicial_graph(
        indices^, distances^, n_samples, params.n_neighbors,
        params.set_op_mix_ratio,
    )


def fit_transform(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
) raises:
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    raise Error(
        "UMAP fit_transform REFUSED: fuzzy graph construction exists, but "
        "spectral initialization, negative-sampling optimization, and the "
        "output layout are not implemented"
    )
