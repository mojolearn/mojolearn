# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5119 (SpectralClustering assign_labels: the k x k SVD
LAPACK computes for scikit-learn is a one-sided Jacobi SVD, the column sums
folded rows ascending, every product pinned).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/spectral_assign_check.mojo

Pass 2, lane/algos-cluster. The fixture (the k x k class sums discretize
builds, and the pivot-row transposes cluster_qr builds, from a seam fixture
embedding) is first shown to SEPARATE the pinned fold from the reversed one,
VACUOUS otherwise; then `x_cluster/spectral_assign.mojo::jacobi_svd` must
equal the host oracle (`x_cluster/checks/oracles.mojo::oracle_jacobi_svd`)
bit for bit in u, the singular values and v. The routine is host code, one
source compiled into both bindings, so the device and CPU columns run the
same compiled body; the lane check (x-cluster-spectral-affinities) compares
the two columns end to end. With MOJOLEARN_IDENTITY_TRACE set, the stage
lands on the card."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from x_cluster.checks.oracles import oracle_jacobi_svd
from x_cluster.checks.seam_util import require_equal, require_separates, seam_fixture
from x_cluster.spectral_assign import cluster_qr_labels, discretize_labels, jacobi_svd


def _diff64(a: List[Float64], b: List[Float64]) -> Int:
    var c = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            c += 1
    return c


def _same(seam: String, differing: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        require_equal(seam, differing)
    else:
        print("  " + seam + ": FAST, " + String(differing) + " cells differ from the oracle (no claim)")


def main() raises:
    var n = 96
    var tr = IdentityTrace()
    var fixtures = List[List[Float64]]()
    var sizes = List[Int]()
    for k in range(3, 7):
        # the fixture's all-zero last column dropped (an embedding has none)
        var xf = seam_fixture(n, k + 1, UInt64(40 + k))
        var x = List[Float32](capacity=n * k)
        for i in range(n):
            for r in range(k):
                x.append(xf[i * (k + 1) + r])
        # the discretize class sums: rows grouped by (row index mod k)
        var m = List[Float64](length=k * k, fill=0)
        for i in range(n):
            for r in range(k):
                m[(i % k) * k + r] = m[(i % k) * k + r] + Float64(x[i * k + r])
        fixtures.append(m^)
        sizes.append(k)
        # a cluster_qr-shaped one: k rows of the embedding, transposed
        var b = List[Float64](length=k * k, fill=0)
        for r in range(k):
            for c in range(k):
                b[r * k + c] = Float64(x[(7 * c + 3) * k + r])
        fixtures.append(b^)
        sizes.append(k)
        # the full routes run (the trace records their labels)
        tr.record_list_i32("x_cluster.spectral_assign.cluster_qr", cluster_qr_labels(x, n, k))
        var it = 0
        tr.record_list_i32("x_cluster.spectral_assign.discretize", discretize_labels(x, n, k, UInt64(3), 30, 20, it))
    var sep = 0
    for f in range(len(fixtures)):
        var k = sizes[f]
        var u0 = List[Float64]()
        var s0 = List[Float64]()
        var v0 = List[Float64]()
        oracle_jacobi_svd(fixtures[f], k, u0, s0, v0)
        var u1 = List[Float64]()
        var s1 = List[Float64]()
        var v1 = List[Float64]()
        oracle_jacobi_svd(fixtures[f], k, u1, s1, v1, reversed_fold=True)
        sep += _diff64(u0, u1) + _diff64(s0, s1) + _diff64(v0, v1)
    require_separates("5119 Jacobi SVD column fold", sep)
    for f in range(len(fixtures)):
        var k = sizes[f]
        var uo = List[Float64]()
        var so = List[Float64]()
        var vo = List[Float64]()
        oracle_jacobi_svd(fixtures[f], k, uo, so, vo)
        var u = List[Float64]()
        var s = List[Float64]()
        var v = List[Float64]()
        jacobi_svd(fixtures[f], k, u, s, v)
        var tag = "fixture " + String(f) + " (k " + String(k) + ")"
        _same("5119 jacobi_svd u " + tag, _diff64(u, uo))
        _same("5119 jacobi_svd singular values " + tag, _diff64(s, so))
        _same("5119 jacobi_svd v " + tag, _diff64(v, vo))
        var sf = List[Float32]()
        for val in s:
            sf.append(Float32(val))
        tr.record_list_f32("x_cluster.spectral_assign.svd", sf)
    print("PASS x_cluster spectral_assign_check")
