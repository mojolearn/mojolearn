# SPDX-License-Identifier: Apache-2.0
"""Main-thread-only bounded control gate; direct FAST launch uses eight rows."""
from max.gpu.host import DeviceContext
from umap.checks.sparse_estimator_check import bits, fit_case
from umap.graph import fuzzy_simplicial_graph
from umap.sparse_graph import sparse_fuzzy_simplicial_graph
from umap.optimizer_fast import optimize_layout_fast
from umap.sparse_optimizer import optimize_sparse_layout_fast
from umap.params import UMAPParams


def main() raises:
    with DeviceContext() as ctx:
        var x: List[Float32] = [0, 1, 2.2, 4, 6.5, 10, 14.5, 20]
        for negatives in range(0, 4, 3):
            var params = UMAPParams(
                n_neighbors=3, n_components=2, n_epochs=4, random_seed=UInt64(19),
                learning_rate=Float32(0.5), repulsion_strength=Float32(2),
                negative_sample_rate=negatives,
            )
            fit_case(ctx, x, 8, 1, params, "controls-neg" + String(negatives))
            var indices = List[UInt32]()
            var distances = List[Float32]()
            var initial = List[Float32]()
            for row in range(8):
                indices.append(UInt32(row))
                indices.append(UInt32((row + 1) % 8))
                indices.append(UInt32((row + 7) % 8))
                distances.append(Float32(0))
                distances.append(Float32(1))
                distances.append(Float32(2))
                initial.append(Float32(row) / Float32(8))
                initial.append(Float32((row * 3) % 7) / Float32(8))
            var dense = fuzzy_simplicial_graph(indices, distances, 8, 3, Float32(0.5))
            var sparse = sparse_fuzzy_simplicial_graph(indices, distances, 8, 3, Float32(0.5))
            # Call FAST directly, avoiding the public small-sample fallback.
            var expected = optimize_layout_fast(
                ctx, initial, dense.weights, 8, 2, 4, params.learning_rate,
                negatives, params.repulsion_strength, Float32(1.5), Float32(0.9), UInt64(19),
            )
            var actual = optimize_sparse_layout_fast(
                ctx, initial, sparse, 8, 2, 4, params.learning_rate,
                negatives, params.repulsion_strength, Float32(1.5), Float32(0.9), UInt64(19),
            )
            bits(expected, actual, "controls-fast-neg" + String(negatives))
    print("UMAP optimizer controls PASS")
