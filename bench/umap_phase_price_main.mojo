# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP fit_transform timed phase by phase: k-NN, host graph, spectral, optimize.

Shape from the environment: MOJOLEARN_UMAP_ROWS (20000), MOJOLEARN_UMAP_FEATURES
(32), MOJOLEARN_UMAP_NEIGHBORS (15), MOJOLEARN_UMAP_EPOCHS (200),
MOJOLEARN_UMAP_ROUNDS (1). The data is the dyadic mixer of
`bench/knn_smallk_dispatch_fixture._coordinate` (salt 0), which
`tools/umap_cuml_reference.py` reproduces so cuML fits the same bytes. The
phases are the four calls `umap/sparse_estimator.mojo::sparse_fit_transform`
makes, timed one after another with a host clock; the embedding's FNV-1a64
fingerprint is printed so two builds can be compared.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from bench.knn_smallk_dispatch_fixture import _coordinate
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from neighbors.estimator import knn_search
from umap.curve import fit_umap_curve
from umap.graph import canonicalize_self_neighbors
from umap.params import UMAPParams
from umap.sparse_estimator import sparse_spectral_initialize
from umap.sparse_graph import sparse_fuzzy_simplicial_graph
from umap.sparse_optimizer import optimize_sparse_layout


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def _ms(begin: Int) -> Float64:
    return Float64(perf_counter_ns() - begin) / 1000000.0


def main() raises:
    var n = _env_int("MOJOLEARN_UMAP_ROWS", 20000)
    var d = _env_int("MOJOLEARN_UMAP_FEATURES", 32)
    var k = _env_int("MOJOLEARN_UMAP_NEIGHBORS", 15)
    var epochs = _env_int("MOJOLEARN_UMAP_EPOCHS", 200)
    var rounds = _env_int("MOJOLEARN_UMAP_ROUNDS", 1)
    var params = UMAPParams(n_neighbors=k, n_components=2, n_epochs=epochs)
    params.validate(n)
    print(
        "UMAP_PHASE_HEADER", "mode", numeric_mode_name(), "column",
        column_name(TARGET_COLUMN), "rows", n, "features", d, "neighbors", k,
        "epochs", epochs, "rounds", rounds, "fixture", "dyadic-v1",
    )
    var x = List[Float32]()
    for row in range(n):
        for f in range(d):
            x.append(_coordinate(row, f, 0))
    with DeviceContext() as ctx:
        for r in range(rounds):
            var t_all = perf_counter_ns()
            # phase 1: exact k-NN, index == queries
            var t0 = perf_counter_ns()
            var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
            for i in range(n * d):
                hx.unsafe_ptr().unsafe_store(i, x[i])
            var hd = ctx.enqueue_create_host_buffer[DType.float32](n * k)
            var hi = ctx.enqueue_create_host_buffer[DType.uint32](n * k)
            _ = knn_search(
                ctx, hx.unsafe_ptr(), n, hx.unsafe_ptr(), n, d, k,
                hd.unsafe_ptr(), hi.unsafe_ptr(),
            )
            var distances = List[Float32]()
            var indices = List[UInt32]()
            for i in range(n * k):
                distances.append(hd.unsafe_ptr().unsafe_load(i))
                indices.append(hi.unsafe_ptr().unsafe_load(i))
            _ = hx^
            _ = hd^
            _ = hi^
            var knn_ms = _ms(t0)
            # phase 2: the host fuzzy simplicial set (CSR)
            t0 = perf_counter_ns()
            canonicalize_self_neighbors(indices, distances, n, k)
            var graph = sparse_fuzzy_simplicial_graph(
                indices^, distances^, n, k, params.set_op_mix_ratio,
            )
            var graph_ms = _ms(t0)
            # phase 3: spectral initialization
            t0 = perf_counter_ns()
            var initial = sparse_spectral_initialize(
                ctx, graph.copy(), params.n_components, params.random_seed
            )
            var spectral_ms = _ms(t0)
            # phase 4: the optimizer
            t0 = perf_counter_ns()
            var curve = fit_umap_curve(params.min_dist, params.spread)
            var embedding = optimize_sparse_layout(
                ctx, initial^, graph, n, params.n_components, epochs,
                initial_learning_rate=params.learning_rate,
                negative_sample_rate=params.negative_sample_rate,
                repulsion_strength=params.repulsion_strength,
                a=curve.a, b=curve.b, seed=params.random_seed,
            )
            var optimize_ms = _ms(t0)
            var total_ms = _ms(t_all)
            var h = UInt64(1469598103934665603)
            for i in range(len(embedding)):
                if not isfinite(embedding[i]):
                    raise Error("UMAP phase price: embedding is not finite")
                h = (h ^ UInt64(bitcast[DType.uint32](embedding[i]))) * UInt64(1099511628211)
            print(
                "UMAP_PHASE_ROUND", r, "knn_ms", knn_ms, "graph_ms", graph_ms,
                "spectral_ms", spectral_ms, "optimize_ms", optimize_ms,
                "total_ms", total_ms, "edges", len(graph.values),
                "embedding_fnv1a64", h,
            )
    print("UMAP PHASE PRICE PASS")
