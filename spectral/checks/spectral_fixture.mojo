# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fixtures for the spectral lane. Hashed, non-uniform, per-cell distinct.

Four shapes, each with a known answer the gates can hold it to:

- `blobs_fixture`: `n_blobs` well-separated Gaussian-ish clusters in `d`
  dimensions, offsets hashed uniform in `[-1, 1)`, centers `20` apart. The
  kNN graph of this is nearly block-diagonal, so the embedding separates the
  blobs and k-means on it recovers the planted labels.
- `ring_graph_fixture`: the cycle `C_n` as a symmetric COO with unit
  weights. Its UNNORMALIZED Laplacian has the closed-form spectrum
  `2 - 2 cos(2 pi j / n)`, `j = 0..n-1`, every nonzero eigenvalue DOUBLE.
  An exact check, and the degeneracy hazard the README records.
- `path_graph_fixture`: the path `P_n` as a symmetric COO with unit weights.
  Its NORMALIZED Laplacian has the closed-form spectrum `1 - cos(pi j /
  (n - 1))`, `j = 0..n-1`, every eigenvalue SIMPLE (the reflecting
  birth-death chain). The exact check on the default `norm_laplacian=True`
  path, with no ties.
- `hashed_graph_fixture`: a symmetric COO with hashed weights in `(0, 1]`,
  `m` upper-triangle edges per row plus a ring backbone so it is connected.
  Non-uniform weights make every degree sum and every matvec row separate
  a serial fold from a split one.

Every float here is built from integer hashes (`splitmix64`), so a fixture
is a pure function of its arguments and carries no host arithmetic of its
own beyond the conversion of an integer to a float.
"""

from spectral.impl.sparse.coo import CooGraph


def splitmix64(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed_unit(tag: UInt64, i: Int, j: Int) -> Float32:
    """A Float32 in `[0, 1)` with 24 significant bits, a pure function of
    the three integers. `Float32(UInt32)` of a 24-bit integer is exact and
    `* 2^-24` is exact, so no rounding happens here."""
    var h = splitmix64(tag * UInt64(0x100000001B3) + UInt64(i) * UInt64(7919) + UInt64(j))
    var top = UInt32((h >> 40) & UInt64(0xFFFFFF))
    return Float32(top) * Float32(5.9604644775390625e-08)


def hashed_signed(tag: UInt64, i: Int, j: Int) -> Float32:
    """`[-1, 1)`, same construction (one exact subtraction after an exact
    scale by 2)."""
    return hashed_unit(tag, i, j) * Float32(2.0) - Float32(1.0)


def blobs_fixture(
    n_per_blob: Int, n_blobs: Int, d: Int, tag: UInt64
) -> Tuple[List[Float32], List[Int32]]:
    """Row-major `n_per_blob * n_blobs` x `d` data and the planted label
    per row. Blob `b` sits at center `20 * hashed_signed` per coordinate
    (so blobs are tens apart) with a radius-1 hashed cloud around it."""
    var data = List[Float32]()
    var labels = List[Int32]()
    for b in range(n_blobs):
        for i in range(n_per_blob):
            for f in range(d):
                var center = Float32(20.0) * hashed_signed(tag ^ UInt64(0xB10B), b, f)
                var off = hashed_signed(tag, b * n_per_blob + i, f)
                data.append(center + off)
            labels.append(Int32(b))
    return (data^, labels^)


def ring_graph_fixture(n: Int) -> CooGraph:
    """`C_n`: edges `(i, i+1 mod n)` in both directions, weight 1."""
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for i in range(n):
        var j = (i + 1) % n
        rows.append(Int32(i))
        cols.append(Int32(j))
        vals.append(Float32(1.0))
        rows.append(Int32(j))
        cols.append(Int32(i))
        vals.append(Float32(1.0))
    return CooGraph(n, rows^, cols^, vals^)


def path_graph_fixture(n: Int) -> CooGraph:
    """`P_n`: edges `(i, i+1)` for `i < n-1`, both directions, weight 1."""
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for i in range(n - 1):
        rows.append(Int32(i))
        cols.append(Int32(i + 1))
        vals.append(Float32(1.0))
        rows.append(Int32(i + 1))
        cols.append(Int32(i))
        vals.append(Float32(1.0))
    return CooGraph(n, rows^, cols^, vals^)


def hashed_graph_fixture(n: Int, m: Int, tag: UInt64) -> CooGraph:
    """A connected symmetric graph with hashed weights: the ring backbone
    (weight hashed in `(0, 1]`) plus `m` hashed extra neighbors per vertex
    (`j = hash mod n`, skipped when `j == i` or `j` is the ring neighbor),
    each edge emitted once per direction. A hashed `j` that repeats an edge
    already present is SKIPPED (a dense `n x n` seen-table on the host), so
    the fixture has no repeated `(row, col)` key: DEVIATION 775 refuses
    those, and `check_spectral_refusals` plants one on purpose."""
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    var seen = List[Bool]()
    for _ in range(n * n):
        seen.append(False)
    for i in range(n):
        var j = (i + 1) % n
        var w = hashed_unit(tag ^ UInt64(0x51), i, 0) * Float32(0.75) + Float32(0.25)
        if not seen[i * n + j]:
            seen[i * n + j] = True
            seen[j * n + i] = True
            rows.append(Int32(i))
            cols.append(Int32(j))
            vals.append(w)
            rows.append(Int32(j))
            cols.append(Int32(i))
            vals.append(w)
        for e in range(m):
            var h = splitmix64(tag * UInt64(31) + UInt64(i) * UInt64(1009) + UInt64(e))
            var jj = Int(h % UInt64(n))
            if jj == i or seen[i * n + jj]:
                continue
            seen[i * n + jj] = True
            seen[jj * n + i] = True
            var ww = hashed_unit(tag ^ UInt64(0x77), i, e) * Float32(0.9) + Float32(0.1)
            rows.append(Int32(i))
            cols.append(Int32(jj))
            vals.append(ww)
            rows.append(Int32(jj))
            cols.append(Int32(i))
            vals.append(ww)
    return CooGraph(n, rows^, cols^, vals^)


def closed_form_ring(n: Int, j: Int) -> Float64:
    """Unnormalized `C_n` Laplacian eigenvalue `2 - 2 cos(2 pi j / n)`."""
    from std.math import cos, pi

    return 2.0 - 2.0 * cos(2.0 * pi * Float64(j) / Float64(n))


def closed_form_path_normalized(n: Int, j: Int) -> Float64:
    """Normalized `P_n` Laplacian eigenvalue `1 - cos(pi j / (n - 1))`."""
    from std.math import cos, pi

    return 1.0 - cos(pi * Float64(j) / Float64(n - 1))
