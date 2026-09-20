# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Repeatable whole-path timing and bit hash for sparse UMAP graph build."""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from umap.sparse_graph import sparse_fuzzy_simplicial_graph


def mix(h: UInt64, x: UInt64) -> UInt64:
    return (h ^ x) * UInt64(1099511628211)


def main() raises:
    var n = 16384
    var text = String(getenv("MOJOLEARN_UMAP_SPARSE_BENCH_ROWS"))
    if text != "":
        n = Int(atol(text))
    var k = 16
    var idx = List[UInt32](capacity=n * k)
    var dist = List[Float32](capacity=n * k)
    for i in range(n):
        for j in range(k):
            # Sorted distances, but varying gaps exercise the complete sigma search.
            idx.append(UInt32((i + j * 7919) % n) if j > 0 else UInt32(i))
            dist.append(Float32(j * j + (i % 7)) * Float32(0.03125) if j > 0 else Float32(0.0))
    var t0 = perf_counter_ns()
    var graph = sparse_fuzzy_simplicial_graph(idx, dist, n, k)
    var elapsed = perf_counter_ns() - t0
    var h = UInt64(1469598103934665603)
    for value in graph.rhos:
        h = mix(h, UInt64(bitcast[DType.uint32](value)))
    for value in graph.sigmas:
        h = mix(h, UInt64(bitcast[DType.uint32](value)))
    for value in graph.directed_indices:
        h = mix(h, UInt64(value))
    for value in graph.directed_values:
        h = mix(h, UInt64(bitcast[DType.uint32](value)))
    for value in graph.indices:
        h = mix(h, UInt64(value))
    for value in graph.values:
        h = mix(h, UInt64(bitcast[DType.uint32](value)))
    print("UMAP_SPARSE_BUILD_BENCH", n, k, elapsed, h, len(graph.values))
