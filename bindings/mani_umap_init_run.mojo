# SPDX-License-Identifier: Apache-2.0
"""RUN piece of UMAP's `_initial` (lane entangled-mkit, 2026-11-12): the
'random' and 'pca' starts on the typed manifold kit (bindings/mani_kit_run.mojo),
the steps bindings/mani_umap_native.mojo ran on the Python-holding MKit: the
identical-mode image `k`, `_svd_flip_v` and the transposes on the
default-mode image `hk`, the same operations, operands and order. The door
admits the bindings and X (UNPACK), releases the GIL around this step, and
packs the start into the `init` Array it admits (PACK)."""
from bindings.mani_kit_run import MRun, OP_SUB, OP_SCALE, OP_ADDS, OP_DIV, OP_MUL, OP_ABS, OP_ADD
from bindings.mani_decomp_run import mani_col_range
from x_ann.mani_umap_run import umap_check_pca_components, umap_check_singular_values, umap_check_spread


def umap_random_start_run(mut run: MRun, k: Int, n: Int, nc: Int, seed: Int) raises -> Int:
    """'random': uniform(-10, 10) on the Philox stream 90."""
    var u = run.rand(k, n, nc, seed, 90, 0)
    var scaled = run.ew(k, OP_SCALE, u, s=20.0)
    return run.ew(k, OP_ADDS, scaled, s=-10.0)


def umap_pca_start_run(mut run: MRun, k: Int, hk: Int, address: Int, rows: Int, cols: Int, ndim: Int, n: Int,
                       nc: Int, seed: Int) raises -> Int:
    """'pca': umap-learn's PCA of the data scaled to max |coordinate| 10 plus
    N(0, 1e-4) noise (stream 91)."""
    var m = run.from_snapshot(k, address, rows, cols, ndim, "X")
    umap_check_pca_components(nc, run.r(m), run.c(m))
    # k.center(M, k.colmean(M))
    var sums = run.colsum(k, m)
    var mean = run.ew(k, OP_SCALE, sums, s=1.0 / Float64(run.r(m)))
    var xc = run.ew(k, OP_SUB, m, mean)
    var vt: Int
    if run.r(m) >= run.c(m):
        var got = run.svd(k, xc, hk)
        vt = got[1]
    else:
        # wide data: the SVD of Xc^T gives Xc's left vectors U (its Vt rows);
        # Xc's right vectors are then U^T Xc / S
        var xct = run.t(xc, hk)
        var got = run.svd(k, xct, hk)
        var s = got[0]
        var ut = got[1]
        umap_check_singular_values(run.addr(s), nc)
        var head = run.rows(ut, 0, nc)
        vt = run.mm(k, head, xc, False, False)
        var kept = run.take_cols(s, mani_col_range(0, nc))
        vt = run.ew(k, OP_DIV, vt, run.t(kept, hk))
    # _svd_flip_v(Vt.rows(0, nc)) on the default-mode image
    var top = run.rows(vt, 0, nc)
    var signs = run.absmax_signs(hk, top, False)
    vt = run.ew(hk, OP_MUL, top, signs)
    var coords = run.mm(k, xc, vt, False, True)
    # max |coords| in Mojo: the positions of the minimum of -|coords|
    var mag = run.ew(k, OP_ABS, coords)
    var at = run.argmin_all(k, run.ew(k, OP_SCALE, mag, s=-1.0))
    var peak = 0.0
    if run.r(at) > 0:
        var where = Int(run.at(at, 0))
        peak = run.at(mag, where)
    umap_check_spread(peak)
    coords = run.ew(k, OP_SCALE, coords, s=10.0 / peak)
    var noise = run.ew(k, OP_SCALE, run.rand(k, n, nc, seed, 91, 1), s=1e-4)
    return run.ew(k, OP_ADD, coords, noise)
