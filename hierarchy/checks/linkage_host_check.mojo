# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU agglomerative fit's fast walk equals the oracle's, edge for edge.

    tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/checks/linkage_host_check.mojo

`hierarchy/host/linkage_host.mojo::host_prim_mst` (Prim under the total
order, threaded) against `hierarchy/checks/linkage_oracle.mojo`'s
`host_kruskal(host_pinned_distance_matrix(..))`: every MST edge (lo, hi,
weight bits) in the same position, then `host_dendrogram_fast` against
`host_dendrogram` row for row, then the labels of the cut. Fixtures: the
oracle's five (blobs, a duplicate lattice whose zero and equal distances
tie, a chain, hashed, blobs with duplicates) and three larger ones that
split every Prim step over several vertex chunks (a small-integer grid with
many equal distances, duplicated hashed rows, hashed rows), at both
metrics. Run it at MOJOLEARN_CPU_THREADS=1 and at the default: the answer
may not move. The order is not vacuous: each fixture must hold two MST
edges of EQUAL weight (so a wrong tie-break could reorder or swap them), or
it is reported VACUOUS for the tie part."""
from std.memory import bitcast

from hierarchy.checks.linkage_oracle import (
    fixture_as_list,
    fixture_d,
    fixture_n,
    fixture_n_clusters,
    host_dendrogram,
    host_extract_flattened_clusters,
    host_kruskal,
    host_pinned_distance_matrix,
)
from hierarchy.host.linkage_host import host_dendrogram_fast, host_prim_mst


def _hash01(i: Int, f: Int, salt: Int) -> Float32:
    var h = UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(f) * 0xBF58476D1CE4E5B9 + UInt64(salt) * 0x94D049BB133111EB
    h = (h ^ (h >> 31)) * 0xD6E8FEB86659FD93
    h = h ^ (h >> 32)
    return Float32(Int(h & UInt64(0xFFFF))) / Float32(65536.0)


def _big(kind: Int, m: Int, d: Int) -> List[Float32]:
    var x = List[Float32](capacity=m * d)
    for i in range(m):
        for f in range(d):
            if kind == 0:
                # a small-integer grid: many exactly equal distances
                x.append(Float32(Int(_hash01(i, f, 7) * Float32(5.0))))
            elif kind == 1:
                # every row duplicated once: zero-weight ties
                x.append(_hash01(i // 2, f, 11) * Float32(4.0) - Float32(2.0))
            else:
                x.append(_hash01(i, f, 13) * Float32(8.0) - Float32(4.0))
    return x^


def _check(name: String, x: List[Float32], m: Int, d: Int, n_clusters: Int, is_sqrt: Bool) raises -> Int:
    var dists = host_pinned_distance_matrix(x, m, d, is_sqrt)
    var want = host_kruskal(dists, m)
    var got = host_prim_mst(x, m, d, is_sqrt)
    if len(got.lo) != len(want[0]):
        raise Error(name + ": " + String(len(got.lo)) + " edges, oracle " + String(len(want[0])))
    var ties = 0
    for i in range(len(got.lo)):
        var wb = bitcast[DType.uint32](want[2][i])
        if got.lo[i] != want[0][i] or got.hi[i] != want[1][i] or bitcast[DType.uint32](got.w[i]) != wb:
            raise Error(
                name + ": edge " + String(i) + " (" + String(got.lo[i]) + "," + String(got.hi[i]) + ","
                + String(got.w[i]) + ") oracle (" + String(want[0][i]) + "," + String(want[1][i]) + ","
                + String(want[2][i]) + ")"
            )
        if i > 0 and bitcast[DType.uint32](want[2][i - 1]) == wb:
            ties += 1
    var ch_w = host_dendrogram(want[0], want[1], m)
    var ch_g = host_dendrogram_fast(got.lo, got.hi, m)
    for i in range(len(ch_w)):
        if ch_w[i] != ch_g[i]:
            raise Error(name + ": children row " + String(i // 2) + " differs")
    var lw = host_extract_flattened_clusters(ch_w, n_clusters, m)
    var lg = host_extract_flattened_clusters(ch_g, n_clusters, m)
    for i in range(m):
        if lw[i] != lg[i]:
            raise Error(name + ": label " + String(i) + " differs")
    var note = String("") if ties > 0 else String(" (VACUOUS for ties: no two equal MST weights)")
    print("  " + name + (" sqrt" if is_sqrt else " sq") + ": m=" + String(m) + " edges equal, "
          + String(ties) + " equal-weight neighbours" + note)
    return ties


def main() raises:
    var tied = 0
    for sq in range(2):
        var is_sqrt = sq == 1
        for fix in range(5):
            var m = fixture_n(fix)
            tied += _check("fixture" + String(fix), fixture_as_list(fix), m, fixture_d(fix), fixture_n_clusters(fix), is_sqrt)
        tied += _check("grid", _big(0, 3000, 3), 3000, 3, 6, is_sqrt)
        tied += _check("dups", _big(1, 2500, 4), 2500, 4, 5, is_sqrt)
        _ = _check("hashed", _big(2, 4200, 7), 4200, 7, 9, is_sqrt)
    if tied == 0:
        raise Error("VACUOUS: no fixture held two equal MST weights")
    print("PASS hierarchy linkage_host_check")
