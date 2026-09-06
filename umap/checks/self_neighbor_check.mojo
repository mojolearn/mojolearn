# SPDX-License-Identifier: Apache-2.0
"""Self-neighbor normalization: moved, missing, already first, and tied."""
from umap.graph import canonicalize_self_neighbors, fuzzy_simplicial_graph


def main() raises:
    var indices: List[UInt32] = [1, 0, 2, 0, 2, 3, 2, 0, 1, 0, 1, 3]
    var distances: List[Float32] = [0, 0.01, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0]
    var rejected = False
    try:
        _ = fuzzy_simplicial_graph(indices.copy(), distances.copy(), 4, 3, Float32(1))
    except:
        rejected = True
    if not rejected:
        raise Error("uncorrected self-neighbor control did not fail")
    var want_indices: List[UInt32] = [0, 1, 2, 1, 0, 2, 2, 0, 1, 3, 0, 1]
    var want_distances: List[Float32] = [0, 0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0]
    for repeat in range(2):
        canonicalize_self_neighbors(indices, distances, 4, 3)
        for i in range(12):
            if indices[i] != want_indices[i] or distances[i] != want_distances[i]:
                raise Error("self-neighbor normalization changed candidate order")
    _ = fuzzy_simplicial_graph(indices^, distances^, 4, 3, Float32(1))
    print("UMAP SELF NEIGHBORS PASS: four cases, idempotence, uncorrected refusal")
