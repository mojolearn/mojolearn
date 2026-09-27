# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5206 (distance folds), 5207 (selection tie-break) and 5208
(kernel epilogues) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/dist_check.mojo

Pass 2, lane/algos-neighbors. Each fixture is first shown to SEPARATE the
pinned spelling from its alternative (VACUOUS otherwise); then the device
column (x_neighbors/device_ops.mojo) and the CPU column
(x_neighbors/host_ops.mojo) must each equal the host oracle
(x_neighbors/checks/oracles.mojo) bit for bit under IDENTICAL (FAST: reported,
no claim). With MOJOLEARN_IDENTITY_TRACE set, every stage lands on the card."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import o_sqdist, o_l1dist, o_nan_sqdist, o_knn_select, o_kernel
from x_neighbors.checks.seam_util import (
    seam_fixture, positive_fixture, fa, ia, zf, zi, count_diff_f32, count_diff_i32, require_separates, same,
)
from x_neighbors.device_ops import op_sqdist, op_l1dist, op_nan_sqdist, op_knn_select, op_kernel
from x_neighbors.host_ops import (
    op_sqdist as h_sqdist, op_l1dist as h_l1dist, op_nan_sqdist as h_nan_sqdist, op_knn_select as h_knn_select,
    op_kernel as h_kernel,
)


def main() raises:
    var tr = IdentityTrace()
    var n = 40
    var m = 9
    var d = 7
    var x = seam_fixture(n, d, 1)
    var y = seam_fixture(m, d, 2)

    # ---- 5206: squared distance, L1, nan_euclidean
    var want = o_sqdist(x, y, n, m, d)
    require_separates("5206 distance fold order", count_diff_f32(want, o_sqdist(x, y, n, m, d, 1)))
    require_separates("5206 distance contraction", count_diff_f32(want, o_sqdist(x, y, n, m, d, 2)))
    var dv = zf(n * m)
    op_sqdist(fa(x), fa(y), fa(dv), n, m, d)
    same("5206 sqdist device", count_diff_f32(dv, want))
    var hv = zf(n * m)
    h_sqdist(fa(x), fa(y), fa(hv), n, m, d)
    same("5206 sqdist host", count_diff_f32(hv, want))
    tr.record_list_f32("x_neighbors.sqdist", dv)

    var wl1 = o_l1dist(x, y, n, m, d)
    require_separates("5206 L1 fold order", count_diff_f32(wl1, o_l1dist(x, y, n, m, d, 1)))
    var dl = zf(n * m)
    op_l1dist(fa(x), fa(y), fa(dl), n, m, d)
    same("5206 l1dist device", count_diff_f32(dl, wl1))
    var hl = zf(n * m)
    h_l1dist(fa(x), fa(y), fa(hl), n, m, d)
    same("5206 l1dist host", count_diff_f32(hl, wl1))
    tr.record_list_f32("x_neighbors.l1dist", dl)

    var xn = x.copy()
    for i in range(0, n, 3):
        xn[i * d + (i % d)] = Float32(0) / Float32(0)
    for f in range(d):
        xn[5 * d + f] = Float32(0) / Float32(0)            # a row with no coordinate
    var wn = o_nan_sqdist(xn, y, n, m, d)
    require_separates("5206 nan_euclidean scale order", count_diff_f32(wn, o_nan_sqdist(xn, y, n, m, d, 1)))
    var dn = zf(n * m)
    op_nan_sqdist(fa(xn), fa(y), fa(dn), n, m, d)
    same("5206 nan_sqdist device", count_diff_f32(dn, wn))
    var hn = zf(n * m)
    h_nan_sqdist(fa(xn), fa(y), fa(hn), n, m, d)
    same("5206 nan_sqdist host", count_diff_f32(hn, wn))
    tr.record_list_f32("x_neighbors.nan_sqdist", dn)
    _ = xn^                        # KEEPALIVE: an address was taken; no ASAP free before the op read it

    # ---- 5207: selection by (value, column); the fixture's rows 1 and 2 tie exactly
    var sq = o_sqdist(x, x, n, n, d)
    var k = 5
    var ws = o_knn_select(sq, n, n, k, False)
    var alt = o_knn_select(sq, n, n, k, False, 1)
    require_separates("5207 selection tie-break", count_diff_i32(ws[1], alt[1]))
    var sd = zf(n * k)
    var si = zi(n * k)
    op_knn_select(fa(sq), fa(sd), ia(si), n, n, k, 0)
    same("5207 knn_select device idx", count_diff_i32(si, ws[1]))
    same("5207 knn_select device dist", count_diff_f32(sd, ws[0]))
    var hd = zf(n * k)
    var hi = zi(n * k)
    h_knn_select(fa(sq), fa(hd), ia(hi), n, n, k, 0)
    same("5207 knn_select host idx", count_diff_i32(hi, ws[1]))
    same("5207 knn_select host dist", count_diff_f32(hd, ws[0]))
    var we = o_knn_select(sq, n, n, k, True)
    var ei = zi(n * k)
    var ed = zf(n * k)
    op_knn_select(fa(sq), fa(ed), ia(ei), n, n, k, 1)
    same("5207 knn_select exclude-self device", count_diff_i32(ei, we[1]))
    tr.record_list_i32("x_neighbors.knn_select", si)
    _ = sq^

    # ---- 5208: every kernel's epilogue on the pinned fold (small-scale rows so exp is not 0)
    var xs = seam_fixture(n, d, 3)
    var ys = seam_fixture(m, d, 4)
    for i in range(len(xs)):
        xs[i] = xs[i] * Float32(0.001)
    for i in range(len(ys)):
        ys[i] = ys[i] * Float32(0.001)
    var xp = positive_fixture(n, d, 5)
    var yp = positive_fixture(m, d, 6)
    for kind in range(8):
        var a = xp.copy() if kind >= 6 else xs.copy()
        var b = yp.copy() if kind >= 6 else ys.copy()
        var g = Float32(0.37)
        var c0 = Float32(0.25)
        var wk = o_kernel(a, b, n, m, d, kind, g, c0, 3)
        require_separates("5208 kernel " + String(kind) + " fold", count_diff_f32(wk, o_kernel(a, b, n, m, d, kind, g, c0, 3, 1)))
        var dk = zf(n * m)
        op_kernel(fa(a), fa(b), fa(dk), n, m, d, kind, g, c0, 3)
        same("5208 kernel " + String(kind) + " device", count_diff_f32(dk, wk))
        var hk = zf(n * m)
        h_kernel(fa(a), fa(b), fa(hk), n, m, d, kind, g, c0, 3)
        same("5208 kernel " + String(kind) + " host", count_diff_f32(hk, wk))
        tr.record_list_f32("x_neighbors.kernel" + String(kind), dk)
        _ = a^
        _ = b^
    _ = x^
    _ = y^
    print("PASS x_neighbors dist_check")
