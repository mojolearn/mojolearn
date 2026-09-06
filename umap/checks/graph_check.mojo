# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from umap.graph import fuzzy_simplicial_graph
from umap.params import UMAPParams


def main() raises:
    UMAPParams(n_neighbors=3).validate(3)
    var idx: List[UInt32] = [0, 1, 2, 1, 0, 2, 2, 1, 0]
    var dist: List[Float32] = [0, 1, 3, 0, 1, 2, 0, 2, 3]
    var graph = fuzzy_simplicial_graph(idx, dist, 3, 3)
    for i in range(3):
        if graph.rhos[i] != dist[i * 3 + 1] or not (
            graph.sigmas[i] > Float32(0.0)
        ):
            raise Error("UMAP rho/sigma search mismatch")
        if graph.weights[i * 3 + i] != Float32(0.0):
            raise Error("UMAP graph contains a self edge")
        for j in range(3):
            if graph.weights[i * 3 + j] != graph.weights[j * 3 + i]:
                raise Error("UMAP fuzzy union is not symmetric")
    if graph.weights[1] != Float32(1.0):
        raise Error("UMAP mutual nearest-neighbor edge lost unit weight")
    print("UMAP fuzzy simplicial graph PASS")
