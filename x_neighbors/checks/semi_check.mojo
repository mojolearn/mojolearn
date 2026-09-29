# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5214 (label propagation's hard clamp, label spreading's soft
clamp and normalized Laplacian) and 5215 (KNNImputer's donors and weighted
mean) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/semi_check.mojo

Pattern as dist_check.mojo."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import o_ls_laplacian, o_ls_clamp, o_lp_clamp, o_knn_impute
from x_neighbors.checks.seam_util import seam_fixture, fa, ia, zf, zi, count_diff_f32, require_separates, same
from x_neighbors.device_ops import op_ls_laplacian, op_ls_clamp, op_lp_clamp, op_knn_impute, op_col_degree, op_ls_laplacian_deg
from x_neighbors.host_ops import (
    op_ls_laplacian as h_ls_laplacian, op_ls_clamp as h_ls_clamp, op_lp_clamp as h_lp_clamp,
    op_knn_impute as h_knn_impute, op_col_degree as h_col_degree, op_ls_laplacian_deg as h_ls_laplacian_deg,
)


def main() raises:
    var tr = IdentityTrace()
    var n = 24
    var a = seam_fixture(n, n, 31)
    for i in range(len(a)):
        a[i] = abs(a[i])
    for j in range(n):
        a[j * n + 7] = Float32(0)                     # a node with in-degree 0: w = 1
    var wl = o_ls_laplacian(a, n)
    require_separates("5214 laplacian division order", count_diff_f32(wl, o_ls_laplacian(a, n, 1)))
    var dl = zf(n * n)
    op_ls_laplacian(fa(a), fa(dl), n)
    same("5214 ls_laplacian device", count_diff_f32(dl, wl))
    var hl = zf(n * n)
    h_ls_laplacian(fa(a), fa(hl), n)
    same("5214 ls_laplacian host", count_diff_f32(hl, wl))
    # LabelSpreading's DEFAULT graph (_expansion_neighbors.py): degrees once
    # (col_degree), then ls_laplacian_deg; the same words as the per-cell item.
    var dd = zf(n)
    op_col_degree(fa(a), fa(dd), n)
    var dg = zf(n * n)
    op_ls_laplacian_deg(fa(a), fa(dd), fa(dg), n)
    same("5214 ls_laplacian_deg device", count_diff_f32(dg, wl))
    var hd = zf(n)
    h_col_degree(fa(a), fa(hd), n)
    var hg = zf(n * n)
    h_ls_laplacian_deg(fa(a), fa(hd), fa(hg), n)
    same("5214 ls_laplacian_deg host", count_diff_f32(hg, wl))
    tr.record_list_f32("x_neighbors.laplacian", dl)

    var c = 3
    var ld = seam_fixture(n, c, 32)
    for i in range(len(ld)):
        ld[i] = abs(ld[i])
    var ys = zf(n * c)
    for i in range(n):
        ys[i * c + i % c] = Float32(0.7)
    var wc = o_ls_clamp(ld, ys, Float32(0.3))
    require_separates("5214 soft clamp contraction", count_diff_f32(wc, o_ls_clamp(ld, ys, Float32(0.3), 1)))
    var dc = zf(n * c)
    op_ls_clamp(fa(ld), fa(ys), fa(dc), n * c, Float32(0.3))
    same("5214 ls_clamp device", count_diff_f32(dc, wc))
    var hc = zf(n * c)
    h_ls_clamp(fa(ld), fa(ys), fa(hc), n * c, Float32(0.3))
    same("5214 ls_clamp host", count_diff_f32(hc, wc))
    var unl = zi(n)
    for i in range(n):
        unl[i] = Int32(i % 2)
    var wp = o_lp_clamp(ld, ys, unl, n, c)
    var dp = zf(n * c)
    op_lp_clamp(fa(ld), fa(ys), ia(unl), fa(dp), n, c)
    same("5214 lp_clamp device", count_diff_f32(dp, wp))
    var hp = zf(n * c)
    h_lp_clamp(fa(ld), fa(ys), ia(unl), fa(hp), n, c)
    same("5214 lp_clamp host", count_diff_f32(hp, wp))
    tr.record_list_f32("x_neighbors.clamp", dp)

    # ---- 5215: the imputer on the planted fixture (rows 1 and 2 tie) with holes
    var m = 30
    var d = 5
    var fx = seam_fixture(m, d, 33)
    var x = seam_fixture(12, d, 33)                  # the first rows of the same draw: exact ties
    var nan = Float32(0) / Float32(0)
    for i in range(12):
        x[i * d + (i % (d - 1))] = nan
    fx[0 * d + 1] = nan
    for f in range(d):
        x[5 * d + f] = nan                            # no coordinate: the column mean
    fx[2 * d + 0] = fx[2 * d + 0] + Float32(0.5)      # donors 1 and 2 now differ ONLY in column 0
    for r in range(6, 8):                             # receivers sitting on donor 1, column 0 missing:
        for f in range(d):                            # donors 1 and 2 tie at distance 0
            x[r * d + f] = fx[1 * d + f]
        x[r * d + 0] = nan
    for kk in range(1, 4, 2):
        for weights in range(2):
            var tag = " k" + String(kk) + " w" + String(weights)
            var w = o_knn_impute(x, fx, 12, m, d, kk, weights)
            if kk == 1:                               # at k = 3 both tied donors are in; the tie decides at k = 1
                require_separates("5215 donor tie-break" + tag, count_diff_f32(w, o_knn_impute(x, fx, 12, m, d, kk, weights, 1)))
            var dv = zf(12 * d)
            op_knn_impute(fa(x), fa(fx), fa(dv), 12, m, d, kk, weights)
            same("5215 knn_impute device" + tag, count_diff_f32(dv, w))
            var hv = zf(12 * d)
            h_knn_impute(fa(x), fa(fx), fa(hv), 12, m, d, kk, weights)
            same("5215 knn_impute host" + tag, count_diff_f32(hv, w))
            tr.record_list_f32("x_neighbors.impute" + tag, dv)
    _ = a^
    _ = ld^
    _ = ys^
    _ = unl^
    _ = fx^
    _ = x^
    print("PASS x_neighbors semi_check")
