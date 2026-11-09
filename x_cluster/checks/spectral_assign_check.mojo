# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5119 (SpectralClustering assign_labels: the k x k SVD
LAPACK computes for scikit-learn is a float-float one-sided Jacobi SVD, the
column sums folded rows ascending).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/spectral_assign_check.mojo

The fixtures (the k x k class sums discretize builds, and the pivot-row
matrices cluster_qr builds, from a seam fixture embedding) are first shown
to SEPARATE the pinned fold from the reversed one, VACUOUS otherwise; then
x_cluster/host/spectral_df64_host.mojo `_sd_svd_host` (the host column of
the device's one-block SVD, the same bodies) must equal the host oracle
(`x_cluster/checks/oracles.mojo::oracle_df64_svd`) bit for bit in u, the
singular values and v, both words. Lane m3-linalg-and-housekeeping
(2026-11-09): cluster_qr moved from binary64 host code to float-float on
every column, so this check moved from the binary64 `jacobi_svd` to the
float-float SVD both assignments share; the lane check
(x-cluster-spectral-affinities) compares the columns end to end. With
MOJOLEARN_IDENTITY_TRACE set, the stage lands on the card."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from x_cluster.checks.oracles import oracle_df64_svd
from x_cluster.checks.seam_util import require_equal, require_separates, seam_fixture
from x_cluster.host.spectral_df64_host import _sd_svd_host, cluster_qr_df64_host, discretize_df64_host
from x_cluster.spectral_df64 import SD_STATE_WORDS, SD_STATUS
from x_linear.ff import FF, ff_add
from x_linear.ops import FP, IP


def _diff(a: List[FF], b: List[FF]) -> Int:
    var c = 0
    for i in range(len(a)):
        if a[i].hi != b[i].hi or a[i].lo != b[i].lo:
            c += 1
    return c


def _same(seam: String, differing: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        require_equal(seam, differing)
    else:
        print("  " + seam + ": FAST, " + String(differing) + " cells differ from the oracle (no claim)")


def _shipped_svd(a: List[FF], k: Int, mut u: List[FF], mut sv: List[FF], mut v: List[FF]) raises:
    """`_sd_svd_host` on a copy of a (it rotates m in place)."""
    var kk = k * k
    var mh = List[Float32](length=kk, fill=0)
    var ml = List[Float32](length=kk, fill=0)
    for t in range(kk):
        mh[t] = a[t].hi
        ml[t] = a[t].lo
    var vh = List[Float32](length=kk, fill=0)
    var vl = List[Float32](length=kk, fill=0)
    var uh = List[Float32](length=kk, fill=0)
    var ul = List[Float32](length=kk, fill=0)
    var svh = List[Float32](length=k, fill=0)
    var svl = List[Float32](length=k, fill=0)
    var ch = List[Float32](length=k, fill=0)
    var cl = List[Float32](length=k, fill=0)
    var fil = List[Int32](length=k, fill=0)
    var st = List[Int32](length=SD_STATE_WORDS, fill=0)
    _sd_svd_host(
        FP(unsafe_from_address=Int(mh.unsafe_ptr())), FP(unsafe_from_address=Int(ml.unsafe_ptr())),
        FP(unsafe_from_address=Int(vh.unsafe_ptr())), FP(unsafe_from_address=Int(vl.unsafe_ptr())),
        FP(unsafe_from_address=Int(uh.unsafe_ptr())), FP(unsafe_from_address=Int(ul.unsafe_ptr())),
        FP(unsafe_from_address=Int(svh.unsafe_ptr())), FP(unsafe_from_address=Int(svl.unsafe_ptr())),
        IP(unsafe_from_address=Int(fil.unsafe_ptr())), FP(unsafe_from_address=Int(ch.unsafe_ptr())),
        FP(unsafe_from_address=Int(cl.unsafe_ptr())), k, IP(unsafe_from_address=Int(st.unsafe_ptr())),
    )
    if st[SD_STATUS] != 0:
        raise Error("spectral_assign_check: the shipped SVD refused a full-rank fixture")
    u = List[FF](capacity=kk)
    v = List[FF](capacity=kk)
    sv = List[FF](capacity=k)
    for t in range(kk):
        u.append(FF(uh[t], ul[t]))
        v.append(FF(vh[t], vl[t]))
    for j in range(k):
        sv.append(FF(svh[j], svl[j]))


def main() raises:
    var n = 96
    var tr = IdentityTrace()
    var fixtures = List[List[FF]]()
    var sizes = List[Int]()
    for k in range(3, 7):
        # the fixture's all-zero last column dropped (an embedding has none)
        var xf = seam_fixture(n, k + 1, UInt64(40 + k))
        var x = List[Float32](capacity=n * k)
        for i in range(n):
            for r in range(k):
                x.append(xf[i * (k + 1) + r])
        # the discretize class sums: rows grouped by (row index mod k)
        var m = List[FF](length=k * k, fill=FF(Float32(0), Float32(0)))
        for i in range(n):
            for r in range(k):
                m[(i % k) * k + r] = ff_add(m[(i % k) * k + r], FF(x[i * k + r], Float32(0)))
        fixtures.append(m^)
        sizes.append(k)
        # a cluster_qr-shaped one: M[c, r] = embedding row (7 c + 3)
        var b = List[FF](length=k * k, fill=FF(Float32(0), Float32(0)))
        for c in range(k):
            for r in range(k):
                b[c * k + r] = FF(x[(7 * c + 3) * k + r], Float32(0))
        fixtures.append(b^)
        sizes.append(k)
        # the full routes run (the trace records their labels)
        tr.record_list_i32("x_cluster.spectral_assign.cluster_qr", cluster_qr_df64_host(x, n, k))
        var it = 0
        tr.record_list_i32("x_cluster.spectral_assign.discretize", discretize_df64_host(x, n, k, UInt64(3), 30, 20, it))
    var sep = 0
    for f in range(len(fixtures)):
        var k = sizes[f]
        var u0 = List[FF]()
        var s0 = List[FF]()
        var v0 = List[FF]()
        oracle_df64_svd(fixtures[f], k, u0, s0, v0)
        var u1 = List[FF]()
        var s1 = List[FF]()
        var v1 = List[FF]()
        oracle_df64_svd(fixtures[f], k, u1, s1, v1, reversed_fold=True)
        sep += _diff(u0, u1) + _diff(s0, s1) + _diff(v0, v1)
    require_separates("5119 Jacobi SVD column fold", sep)
    for f in range(len(fixtures)):
        var k = sizes[f]
        var uo = List[FF]()
        var so = List[FF]()
        var vo = List[FF]()
        oracle_df64_svd(fixtures[f], k, uo, so, vo)
        var u = List[FF]()
        var s = List[FF]()
        var v = List[FF]()
        _shipped_svd(fixtures[f], k, u, s, v)
        var tag = "fixture " + String(f) + " (k " + String(k) + ")"
        _same("5119 df64 svd u " + tag, _diff(u, uo))
        _same("5119 df64 svd singular values " + tag, _diff(s, so))
        _same("5119 df64 svd v " + tag, _diff(v, vo))
        var sf = List[Float32]()
        for j in range(k):
            sf.append(s[j].hi)
        tr.record_list_f32("x_cluster.spectral_assign.svd", sf)
    print("PASS x_cluster spectral_assign_check")
