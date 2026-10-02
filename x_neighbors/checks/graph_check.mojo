# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5216 (the PageRank step), 5217 (the min-label product),
5204 (Louvain's tie to the smallest community id, lane hr-graph) and 5205 (the SVGP system: Cholesky, the
substitutions, the collapsed bound) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/graph_check.mojo

Pattern as dist_check.mojo."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import o_pagerank_step, o_cc_step, o_svgp, o_kernel, o_matmul
from x_neighbors.checks.seam_util import (
    seam_fixture, fa, ia, zf, zi, count_diff_f32, count_diff_i32, require_separates, same,
)
from x_neighbors.device_ops import op_pagerank_step, op_cc_step, op_svgp
from x_neighbors.host_ops import op_pagerank_step as h_pagerank_step, op_cc_step as h_cc_step, op_svgp as h_svgp
from x_neighbors.graph_dev import op_louvain, pr_iterate_gpu
from x_neighbors.graph_host import op_louvain as h_louvain, GraphCpu, pr_iterate_cpu
from x_neighbors.graph_par import (
    Lay, LV_MOVE, L_NSLOT, L_G0, L_COMM, L_IC, L_DEG, L_STOT, L_FV, L_EK, L_ORD, _is, _fs, _ls,
)


def _graph(n: Int, seed: UInt64) -> List[Float32]:
    """A symmetric weighted graph with two clusters and a bridge, and ties."""
    var r = seam_fixture(n, n, seed)
    var a = zf(n * n)
    for i in range(n):
        for j in range(i + 1, n):
            var same_side = (i < n // 2) == (j < n // 2)
            var v = abs(r[i * n + j])
            if (same_side and v > Float32(0.3)) or (i == 0 and j == n - 1):
                var wt = Float32(1 + (i * j) % 3)
                a[i * n + j] = wt
                a[j * n + i] = wt
    return a^


def main() raises:
    var tr = IdentityTrace()
    var n = 26
    var q = seam_fixture(n, n, 41)
    for i in range(len(q)):
        q[i] = abs(q[i]) * Float32(0.01)
    var x = seam_fixture(1, n, 42)
    for i in range(n):
        x[i] = abs(x[i]) + Float32(0.01)
    var p = zf(n)
    for i in range(n):
        p[i] = Float32(1) / Float32(n)
    var dang = zi(n)
    dang[3] = 1
    dang[9] = 1
    var w = o_pagerank_step(q, x, p, p, dang, n, Float32(0.85))
    require_separates("5216 pagerank x@Q fold", count_diff_f32(w, o_pagerank_step(q, x, p, p, dang, n, Float32(0.85), 1)))
    var dv = zf(n)
    op_pagerank_step(fa(q), fa(x), fa(p), fa(p), ia(dang), fa(dv), n, Float32(0.85))
    same("5216 pagerank_step device", count_diff_f32(dv, w))
    var hv = zf(n)
    h_pagerank_step(fa(q), fa(x), fa(p), fa(p), ia(dang), fa(hv), n, Float32(0.85))
    same("5216 pagerank_step host", count_diff_f32(hv, w))
    tr.record_list_f32("x_neighbors.pagerank", dv)
    # networkx's `dangling` weights, apart from the personalization
    var dwt = zf(n)
    for i in range(n):
        dwt[i] = Float32(1 + (i * 7) % 5) / Float32(3 * n)
    var wpd = o_pagerank_step(q, x, p, dwt, dang, n, Float32(0.85))
    require_separates("5216 pagerank dangling weights", count_diff_f32(wpd, o_pagerank_step(q, x, p, dwt, dang, n, Float32(0.85), 2)))
    var ddw = zf(n)
    op_pagerank_step(fa(q), fa(x), fa(p), fa(dwt), ia(dang), fa(ddw), n, Float32(0.85))
    same("5216 pagerank dangling device", count_diff_f32(ddw, wpd))
    var hdw = zf(n)
    h_pagerank_step(fa(q), fa(x), fa(p), fa(dwt), ia(dang), fa(hdw), n, Float32(0.85))
    same("5216 pagerank dangling host", count_diff_f32(hdw, wpd))
    tr.record_list_f32("x_neighbors.pagerank_dangling", ddw)
    _ = dwt^

    # ---- PageRank's device route (graph_par.pr_drive): the column lists and
    # the power iteration on the device == the host column, and a real
    # distribution (sums to ~1, every score positive). Row 5 zeroed: a
    # dangling node. An all-zero or never-written device result fails here.
    var pa = _graph(n, 49)
    for j in range(n):
        pa[5 * n + j] = Float32(0)
    var pcols = List[List[Float32]]()
    var pinfs = List[List[Int32]]()
    for col in range(2):
        var px = zf(n)
        var pp = zf(n)
        var pdw = zf(n)
        for i in range(n):
            px[i] = Float32(1) / Float32(n)
            pp[i] = Float32(1) / Float32(n)
            pdw[i] = Float32(1) / Float32(n)
        var pinf = zi(2)
        if col == 0:
            pr_iterate_gpu(fa(pa), fa(px), fa(pp), fa(pdw), ia(pinf), n, 100, Float64(n) * 1e-6, 0, Float32(0.85))
        else:
            pr_iterate_cpu(fa(pa), fa(px), fa(pp), fa(pdw), ia(pinf), n, 100, Float64(n) * 1e-6, 0, Float32(0.85))
        _ = pp^
        _ = pdw^
        pcols.append(px^)
        pinfs.append(pinf^)
    var psum = Float64(0)
    var pmin = Float32(1)
    for i in range(n):
        psum += Float64(pcols[0][i])
        pmin = min(pmin, pcols[0][i])
    if not (psum > 0.99 and psum < 1.01 and pmin > Float32(0)):
        raise Error("5216 pagerank device iterate: not a distribution (sum " + String(psum) + ", min " + String(pmin) + ")")
    same("5216 pagerank iterate device == host", count_diff_f32(pcols[0], pcols[1]))
    same("5216 pagerank iterate info device == host", count_diff_i32(pinfs[0], pinfs[1]))
    tr.record_list_f32("x_neighbors.pagerank_iterate", pcols[0])
    _ = pa^

    var g = _graph(n, 43)
    var lab = zi(n)
    for i in range(n):
        lab[i] = Int32((i * 11) % n)
    var wc = o_cc_step(g, lab, n)
    require_separates("5217 min-label representative", count_diff_i32(wc, o_cc_step(g, lab, n, 1)))
    var dc = zi(n)
    op_cc_step(fa(g), ia(lab), ia(dc), n)
    same("5217 cc_step device", count_diff_i32(dc, wc))
    var hc = zi(n)
    h_cc_step(fa(g), ia(lab), ia(hc), n)
    same("5217 cc_step host", count_diff_i32(hc, wc))
    tr.record_list_i32("x_neighbors.cc", dc)

    # ---- 5204: one node with two candidate communities of EQUAL gain (node 0
    # of the path 1 - 0 - 2, singletons) moves to the smaller id, community 1
    var lay = Lay(L_NSLOT)
    for b in range(3):
        lay.i(L_G0 + 4 * b, 4)
        lay.i(L_G0 + 4 * b + 1, 4)
        lay.i(L_G0 + 4 * b + 2, 4)
        lay.f(L_G0 + 4 * b + 3, 4)
    lay.i(L_COMM, 3)
    lay.i(L_IC, 8)
    lay.f(L_DEG, 3)
    lay.f(L_STOT, 3)
    lay.f(L_FV, 4)
    lay.l(L_EK, 4)
    lay.l(L_ORD, 3)
    var ex = GraphCpu(0)
    var gx = ex.alloc(lay)
    var gip = _is(gx, L_G0)
    var gsrc = _is(gx, L_G0 + 1)
    var gcol = _is(gx, L_G0 + 2)
    var gval = _fs(gx, L_G0 + 3)
    var ips: List[Int32] = [0, 2, 3, 4]
    var srcs: List[Int32] = [0, 0, 1, 2]
    var cols: List[Int32] = [1, 2, 0, 0]
    for k in range(4):
        gip.unsafe_store(k, ips[k])
        gsrc.unsafe_store(k, srcs[k])
        gcol.unsafe_store(k, cols[k])
        gval.unsafe_store(k, Float32(1))
    var degs: List[Float32] = [2, 1, 1]
    for k in range(3):
        _is(gx, L_COMM).unsafe_store(k, Int32(k))
        _fs(gx, L_DEG).unsafe_store(k, degs[k])
        _fs(gx, L_STOT).unsafe_store(k, degs[k])
    _fs(gx, L_FV).unsafe_store(0, Float32(2))
    _ls(gx, L_ORD).unsafe_store(0, Int64(0))
    var mq = gx
    mq.n0 = 3
    mq.n1 = 0
    mq.n3 = L_G0
    mq.x0 = Float32(1)
    ex.run(LV_MOVE, 1, mq)
    var tie = zi(1)
    tie[0] = _is(gx, L_COMM).unsafe_load(0)
    var want = zi(1)
    want[0] = Int32(1)
    var other = zi(1)
    other[0] = Int32(2)
    require_separates("5204 louvain tie (smallest id vs largest)", count_diff_i32(want, other))
    same("5204 louvain tie to the smallest community", count_diff_i32(tie, want))
    _ = ex^

    # two disconnected 4-cliques: the partition is the two cliques, numbered
    # by their lowest node; the device and the host agree bit for bit
    var nq = 8
    var cl = zf(nq * nq)
    for i in range(nq):
        for j in range(nq):
            if i != j and (i < 4) == (j < 4):
                cl[i * nq + j] = Float32(1)
    var cw = zi(nq)
    for i in range(nq):
        cw[i] = Int32(0 if i < 4 else 1)
    var cdl = zi(nq)
    var cdi = zf(2)
    op_louvain(fa(cl), ia(cdl), fa(cdi), nq, 0, Float32(1), Float32(1e-7))
    same("5204 louvain device two cliques", count_diff_i32(cdl, cw))
    var chl = zi(nq)
    var chi = zf(2)
    h_louvain(fa(cl), ia(chl), fa(chi), nq, 0, Float32(1), Float32(1e-7))
    same("5204 louvain host two cliques", count_diff_i32(chl, cw))
    same("5204 louvain two cliques modularity device == host", count_diff_f32(cdi, chi))
    _ = cl^  # the host call read it: keep it alive past the call (seam_util.fa)

    # a ring of equal weights (every move a tie) and the weighted two-cluster
    # graph: device == host, labels and [modularity, levels]
    var ring = zf(n * n)
    for i in range(n):
        ring[i * n + (i + 1) % n] = Float32(1)
        ring[((i + 1) % n) * n + i] = Float32(1)
    var dl = zi(n)
    var di = zf(2)
    op_louvain(fa(ring), ia(dl), fa(di), n, 0, Float32(1), Float32(1e-7))
    var hl = zi(n)
    var hi = zf(2)
    h_louvain(fa(ring), ia(hl), fa(hi), n, 0, Float32(1), Float32(1e-7))
    same("5204 louvain ring labels device == host", count_diff_i32(dl, hl))
    same("5204 louvain ring info device == host", count_diff_f32(di, hi))
    var wg = _graph(n, 47)
    var wdl = zi(n)
    var wdi = zf(2)
    op_louvain(fa(wg), ia(wdl), fa(wdi), n, 0, Float32(1), Float32(1e-7))
    var whl = zi(n)
    var whi = zf(2)
    h_louvain(fa(wg), ia(whl), fa(whi), n, 0, Float32(1), Float32(1e-7))
    same("5204 louvain weighted labels device == host", count_diff_i32(wdl, whl))
    same("5204 louvain weighted info device == host", count_diff_f32(wdi, whi))
    _ = wg^
    tr.record_list_i32("x_neighbors.louvain", dl)

    # ---- 5205: the SVGP system on 60 rows, 8 inducing points
    var nn = 60
    var mm = 8
    var d = 3
    var xs = seam_fixture(nn, d, 44)
    for i in range(len(xs)):
        xs[i] = xs[i] * Float32(0.002)
    var z = zf(mm * d)
    for i in range(mm):
        for f in range(d):
            z[i * d + f] = xs[(i * 7) * d + f]
    var y = zf(nn)
    for i in range(nn):
        y[i] = xs[i * d] + Float32(0.1) * Float32(i % 5)
    var kuu = o_kernel(z, z, mm, mm, d, 2, Float32(0.5), Float32(0), 0)
    var kuf = o_kernel(z, xs, mm, nn, d, 2, Float32(0.5), Float32(0), 0)
    var kfu = o_kernel(xs, z, nn, mm, d, 2, Float32(0.5), Float32(0), 0)
    var bmat = o_matmul(kuf, kfu, mm, nn, mm)
    var b = o_matmul(kuf, y, mm, nn, 1)
    var ws = o_svgp(kuu, bmat, b, y, mm, nn, Float32(0.3), Float32(1e-4), Float32(1))
    require_separates("5205 cholesky fold", count_diff_f32(ws, o_svgp(kuu, bmat, b, y, mm, nn, Float32(0.3), Float32(1e-4), Float32(1), 1)))
    var outs = List[List[Float32]]()
    for col in range(2):
        var al = zf(mm)
        var cm = zf(mm * mm)
        var qm = zf(mm)
        var qs = zf(mm * mm)
        var inf = zf(2)
        if col == 0:
            op_svgp(fa(kuu), fa(bmat), fa(b), fa(y), fa(al), fa(cm), fa(qm), fa(qs), fa(inf), mm, nn, Float32(0.3), Float32(1e-4), Float32(1))
        else:
            h_svgp(fa(kuu), fa(bmat), fa(b), fa(y), fa(al), fa(cm), fa(qm), fa(qs), fa(inf), mm, nn, Float32(0.3), Float32(1e-4), Float32(1))
        var got = al.copy()
        got.extend(cm.copy())
        got.extend(qm.copy())
        got.extend(qs.copy())
        got.append(inf[0])
        if inf[1] != Float32(1):
            raise Error("5205: the SVGP system was not positive definite on the fixture")
        outs.append(got^)
    same("5205 svgp device", count_diff_f32(outs[0], ws))
    same("5205 svgp host", count_diff_f32(outs[1], ws))
    tr.record_list_f32("x_neighbors.svgp", outs[0])
    _ = q^
    _ = x^
    _ = p^
    _ = dang^
    _ = g^
    _ = lab^
    _ = ring^
    _ = kuu^
    _ = bmat^
    _ = b^
    _ = y^
    print("PASS x_neighbors graph_check")
