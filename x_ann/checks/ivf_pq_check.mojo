# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ's per-seam proof (DEVIATIONS 5800, 5801, 5803, 5804; IDENTITY_PATHS row 180):
the device build and search equal the independent host oracle
(x_ann/checks/ivf_pq_oracle.mojo) BIT FOR BIT, on fixtures each shown first
to SEPARATE the pinned spelling of its seam from the unpinned one (a fixture
that does not separate is refused as VACUOUS). Run under IDENTICAL:

    tools/with_identical_mode.sh pixi run mojo run -I . x_ann/checks/ivf_pq_check.mojo

With MOJOLEARN_IDENTITY_TRACE=<file> the device stages are written as a card
(ivfpq.centers, ivfpq.codebooks, ivfpq.codes, ivfpq.search.*) for
`tools/identity_trace_diff.py` across boxes. Exit 0 only when every
comparison matched."""

from std.memory import bitcast
from std.sys import exit
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_ann.ivf_pq_device import ivf_pq_build_device, ivf_pq_search_device, pq_encode_device
from x_ann.checks.ann_check_fixtures import fixture_ties, fixture_wide, hash_u, report, same_f32, same_i32
from x_ann.checks.ivf_pq_oracle import (
    OrIndex, or_argmin, or_before, or_build, or_dist, or_search,
)


def check_build(name: String, x: List[Float32], n: Int, dim: Int, mut trace: IdentityTrace, mut failed: Int) raises:
    var n_lists = 6
    var pq_dim = 3
    var pq_bits = 3
    var dev = ivf_pq_build_device(x, n, dim, n_lists, 4, 1, pq_dim, pq_bits, 4)
    var ora = or_build(x, n, dim, n_lists, 4, 1, pq_dim, pq_bits, 4)
    trace.record_list_f32(String("ivfpq.") + name + ".centers", dev.centers)
    trace.record_list_f32(String("ivfpq.") + name + ".codebooks", dev.codebooks)
    trace.record_list_i32(String("ivfpq.") + name + ".codes", dev.codes)
    report(name + ": coarse centres == host_ivf_build", same_f32(dev.centers, ora.centers), failed)
    report(name + ": lists == host_ivf_build", same_i32(dev.offsets, ora.offsets) and same_i32(dev.list_indices, ora.list_indices), failed)
    report(name + ": codebooks == host_kmeans_fit", same_f32(dev.codebooks, ora.codebooks), failed)
    report(name + ": codes == oracle (5800, 5801)", same_i32(dev.codes, ora.codes), failed)


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var trace = IdentityTrace()
    trace.header("x_ann ivf_pq_check")
    var n = 512
    var dim = 10
    var wide = fixture_wide(n, dim)
    var ties = fixture_ties(n, dim)

    # ---- separation, BEFORE any comparison is trusted
    var sep_fold = 0
    for i in range(n):
        for l in range(8):
            var j = (i * 37 + l * 101) % n
            if bitcast[DType.uint32](or_dist(wide, i * dim, wide, j * dim, dim)) != bitcast[DType.uint32](or_dist(wide, i * dim, wide, j * dim, dim, descending=True)):
                sep_fold += 1
    var sep_tie = 0
    var cb6 = List[Float32]()
    for c in range(6):
        var row = (c * 40503 + 7919) % n
        for t in range(dim):
            cb6.append(ties[row * dim + t])
    for i in range(n):
        if or_argmin(ties, i * dim, cb6, 0, 6, dim) != or_argmin(ties, i * dim, cb6, 0, 6, dim, high_tie=True):
            sep_tie += 1
    print("separation: fold", sep_fold, "encode-tie", sep_tie)
    if sep_fold == 0 or sep_tie == 0:
        print("VACUOUS: a fixture does not separate its seam")
        exit(2)

    # 5801 directly: codebooks with DUPLICATED codewords (code 2c+1 = code 2c),
    # so every row's encoding meets an exact tie
    var dup_cb = List[Float32]()
    for j in range(2):
        for c in range(6):
            var row = ((c // 2) * 97 + j * 13) % n
            for t in range(5):
                dup_cb.append(ties[row * dim + j * 5 + t])
    var enc = pq_encode_device(ties, dup_cb, n, 2, 5, 6)
    var enc_ok = True
    var enc_ties = 0
    for i in range(n):
        for j in range(2):
            var want = or_argmin(ties, i * dim + j * 5, dup_cb, j * 6 * 5, 6, 5)
            if want != or_argmin(ties, i * dim + j * 5, dup_cb, j * 6 * 5, 6, 5, high_tie=True):
                enc_ties += 1
            if Int(enc[i * 2 + j]) != want:
                enc_ok = False
    print("separation: duplicated-codeword ties", enc_ties)
    if enc_ties == 0:
        print("VACUOUS: the duplicated codebook does not separate 5801")
        exit(2)
    report("encode with duplicated codewords == oracle (5801)", enc_ok, failed)

    check_build("wide", wide, n, dim, trace, failed)
    check_build("ties", ties, n, dim, trace, failed)

    # ---- search on a planted index: lists 1 and 2 share a centre (probe
    # tie, 5804), rows 2t and 2t+1 share codes (distance tie, 5803)
    var sdim = 4
    var centers: List[Float32] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5]
    var offsets: List[Int32] = [0, 10, 20, 30, 40]
    var list_indices = List[Int32]()
    for i in range(40):
        list_indices.append(Int32(i))
    var codebooks = List[Float32]()
    for e in range(2 * 4 * 2):
        codebooks.append(Float32(Int(hash_u(e, 3) % UInt64(9))) * Float32(0.25) - Float32(1.0))
    var codes = List[Int32]()
    for i in range(40):
        var p = i // 2
        codes.append(Int32(p % 4))
        codes.append(Int32((p // 4) % 4))
    var mask = List[Int32](length=40, fill=Int32(1))
    var m = 16
    var queries = List[Float32]()
    for q in range(m):
        for c in range(sdim):
            queries.append(Float32(1.0) + Float32(Int(hash_u(q * sdim + c, 5) % UInt64(3))) * Float32(0.5) - Float32(0.5))
    var k = 3
    # separation: the probe tie and the top-k tie occur and are decisive
    var sep_probe = 0
    for q in range(m):
        var d1 = or_dist(queries, q * sdim, centers, 1 * sdim, sdim)
        var d0 = or_dist(queries, q * sdim, centers, 0, sdim)
        var d3 = or_dist(queries, q * sdim, centers, 3 * sdim, sdim)
        if d1 < d0 and d1 < d3:
            sep_probe += 1
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    or_search(centers, offsets, list_indices, codebooks, codes, mask, 4, sdim, 2, 2, queries, m, k, 1, od, oi, on)
    var sep_topk = 0
    for q in range(m):
        for s in range(k - 1):
            if od[q * k + s] == od[q * k + s + 1] and oi[q * k + s] != oi[q * k + s + 1]:
                sep_topk += 1
    print("separation: probe-tie", sep_probe, "topk-tie", sep_topk)
    if sep_probe == 0 or sep_topk == 0:
        print("VACUOUS: the planted index does not separate its seams")
        exit(2)
    var dd = List[Float32]()
    var di = List[Int32]()
    var dn = List[Int32]()
    ivf_pq_search_device(centers, offsets, list_indices, codebooks, codes, mask, 4, sdim, 2, 2, queries, m, k, 1, dd, di, dn)
    trace.record_list_f32("ivfpq.search.dist", dd)
    trace.record_list_i32("ivfpq.search.idx", di)
    report("search: distances == oracle (5800)", same_f32(dd, od), failed)
    report("search: ids == oracle (5803, 5804)", same_i32(di, oi), failed)
    report("search: candidate counts == oracle (5804)", same_i32(dn, on), failed)
    # the wide build searched end to end
    var idx = ivf_pq_build_device(wide, n, dim, 6, 4, 1, 3, 3, 4)
    var wm = List[Int32](length=n, fill=Int32(1))
    var wq = List[Float32]()
    for e in range(32 * dim):
        wq.append(wide[(e * 7) % (n * dim)])
    var wd = List[Float32]()
    var wi = List[Int32]()
    var wn = List[Int32]()
    ivf_pq_search_device(idx.centers, idx.offsets, idx.list_indices, idx.codebooks, idx.codes, wm, 6, dim, 3, 3, wq, 32, 5, 2, wd, wi, wn)
    var xd = List[Float32]()
    var xi = List[Int32]()
    var xn = List[Int32]()
    or_search(idx.centers, idx.offsets, idx.list_indices, idx.codebooks, idx.codes, wm, 6, dim, 3, 3, wq, 32, 5, 2, xd, xi, xn)
    report("wide search == oracle", same_f32(wd, xd) and same_i32(wi, xi) and same_i32(wn, xn), failed)
    if failed > 0:
        print("ivf_pq_check: FAILED", failed)
        exit(1)
    print("ivf_pq_check: ALL OK")
