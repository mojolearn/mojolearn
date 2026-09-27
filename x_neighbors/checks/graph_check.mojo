# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5216 (the PageRank step), 5217 (the min-label product),
5204 (Louvain's pinned order) and 5205 (the SVGP system: Cholesky, the
substitutions, the collapsed bound) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/graph_check.mojo

Pattern as dist_check.mojo."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import o_pagerank_step, o_cc_step, o_louvain, o_svgp, o_kernel, o_matmul
from x_neighbors.checks.seam_util import (
    seam_fixture, fa, ia, zf, zi, count_diff_f32, count_diff_i32, require_separates, same,
)
from x_neighbors.device_ops import op_pagerank_step, op_cc_step, op_louvain, op_svgp
from x_neighbors.host_ops import (
    op_pagerank_step as h_pagerank_step, op_cc_step as h_cc_step, op_louvain as h_louvain, op_svgp as h_svgp,
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

    # a ring of equal weights: every move is a tie, so the visit order decides the partition
    var ring = zf(n * n)
    for i in range(n):
        ring[i * n + (i + 1) % n] = Float32(1)
        ring[((i + 1) % n) * n + i] = Float32(1)
    var lw = o_louvain(ring, n, 0, Float32(1), Float32(1e-7))
    var la = o_louvain(ring, n, 0, Float32(1), Float32(1e-7), 1)
    require_separates("5204 louvain visit order", count_diff_i32(lw[0], la[0]))
    var dl = zi(n)
    var di = zf(2)
    op_louvain(fa(ring), ia(dl), fa(di), n, 0, Float32(1), Float32(1e-7))
    same("5204 louvain device labels", count_diff_i32(dl, lw[0]))
    var wi = zf(2)
    wi[0] = lw[1]
    wi[1] = Float32(lw[2])
    same("5204 louvain device modularity", count_diff_f32(di, wi))
    var hl = zi(n)
    var hi = zf(2)
    h_louvain(fa(ring), ia(hl), fa(hi), n, 0, Float32(1), Float32(1e-7))
    same("5204 louvain host labels", count_diff_i32(hl, lw[0]))
    same("5204 louvain host modularity", count_diff_f32(hi, wi))
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
