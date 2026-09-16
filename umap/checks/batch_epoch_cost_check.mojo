# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the batch-invariance repair COSTS, at a size where it is felt.

lane/umap-batch-fix, 2026-09-16. Collapsing the epoch cliff means a request of
more than ten thousand queries refines for 100 epochs where it used to refine
for 30, and making the edge schedule divide by each row's own largest
membership rather than the request's makes more edges fire. Neither is free,
and "3.3x" is arithmetic on the epoch counts rather than a measurement. This
file measures.

Two sizes, both on the CPU host route, which is the right place for it: the
device route runs only the k-NN on a `DeviceContext` and does `refine_transform`
on the host exactly as this does, so refinement time is host time in both
routes.

  BELOW  5,000 queries. 100 epochs before the repair and 100 after, so the
         ratio here is the per-row maximum and the per-row key ALONE.
  ABOVE  10,001 queries. 30 epochs before and 100 after, so the ratio here is
         that cost plus the epoch collapse.

Run it on this branch and on the commit before the repair and divide. The
fixture is a saved model of 1,000 training rows in 8 features with k=10,
n_components=2, at the shipped `negative_sample_rate=5` and with `n_epochs`
unset, which is the only configuration in which any of this is reachable.
"""
from std.math import isfinite
from std.time import perf_counter_ns

from umap.host.umap_oracle import host_umap_transform
from umap.params import UMAPParams

comptime N_TRAIN = 1000
comptime D = 8
comptime C = 2
comptime K = 10
comptime ROWS_BELOW = 5000
comptime ROWS_ABOVE = 10001


def _mix(value: UInt64) -> UInt64:
    var z = value + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _spread(seed: UInt64, scale: Float32) -> Float32:
    var bits = Int(_mix(seed) >> 40)
    return (Float32(bits) / Float32(8388608.0) - Float32(1.0)) * scale


def _fixture() raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """Twelve clusters with jitter, never uniform, plus a 2D saved embedding."""
    var data = List[Float32]()
    var embedding = List[Float32]()
    for i in range(N_TRAIN):
        var cluster = i % 12
        for c in range(D):
            data.append(
                Float32(cluster) * Float32(3.0) + Float32(c) * Float32(0.7)
                + _spread(UInt64(i * 131 + c), Float32(0.9))
            )
        for c in range(C):
            embedding.append(
                Float32(cluster % 4) * Float32(4.0) + Float32(c) * Float32(2.0)
                + _spread(UInt64(i * 977 + c + 4096), Float32(0.6))
            )
    var queries = List[Float32]()
    for i in range(ROWS_ABOVE):
        var cluster = i % 12
        for c in range(D):
            queries.append(
                Float32(cluster) * Float32(3.0) + Float32(c) * Float32(0.7)
                + _spread(UInt64(i * 17 + c + 900000), Float32(1.1))
            )
    return (data^, embedding^, queries^)


def _head(queries: List[Float32], rows: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(rows * D):
        out.append(queries[i])
    return out^


def _time(
    label: String, training: List[Float32], embedding: List[Float32],
    queries: List[Float32], rows: Int,
) raises:
    var params = UMAPParams(
        n_neighbors=K, n_components=C, n_epochs=0, random_seed=UInt64(7),
        negative_sample_rate=5,
    )
    var batch = _head(queries, rows)
    var t0 = perf_counter_ns()
    var result = host_umap_transform(training, embedding, batch, N_TRAIN, rows, D, params)
    var t1 = perf_counter_ns()
    var checksum = Float64(0)
    for value in result:
        if not isfinite(value):
            raise Error("cost fixture produced a non-finite coordinate")
        checksum += Float64(value)
    print(
        "COST", label, "rows", rows, "ms", Float64(t1 - t0) / Float64(1000000.0),
        "checksum", checksum,
    )


def main() raises:
    print("UMAP transform batch-invariance cost, CPU host route, one core")
    var fx = _fixture()
    var training = fx[0].copy()
    var embedding = fx[1].copy()
    var queries = fx[2].copy()
    _time(String("BELOW"), training, embedding, queries, ROWS_BELOW)
    _time(String("ABOVE"), training, embedding, queries, ROWS_ABOVE)
    print("UMAP batch-invariance cost measurement COMPLETE")
