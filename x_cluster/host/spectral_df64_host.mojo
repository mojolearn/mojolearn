# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of SpectralClustering's float-float discretize (lane
n9b-device-svd-qr, 2026-11-09): x_cluster/spectral_df64.mojo's bodies in the
device driver's order (x_cluster/spectral_df64_device.mojo), the row and
partial steps in parallel over the host pool. Host binding only."""
from cluster.host.host_cells import host_cells
from x_cluster.bodies import SplitMix64
from x_cluster.spectral_df64 import (
    SD_DONE, SD_ITER, SD_ROWS, SD_STATE_WORDS, SD_STATUS, SD_BAD_BASIS, SD_SVD_SWEEPS,
    sd_apply_row, sd_better_min, sd_cacc_row, sd_chunks, sd_class_fold, sd_class_part, sd_colnorm, sd_colscale_cell,
    sd_colsign_cell, sd_colsq_part, sd_complete_u, sd_dot_cols, sd_ld, sd_label_row, sd_load_cell, sd_ncut_step,
    sd_norm_col, sd_of_int, sd_polar_cell, sd_raise_status, sd_rotation, sd_rownorm_row, sd_st, sd_take_row_cell,
    sd_zero,
)
from x_linear.ff import FF, ff_sqrt
from x_linear.ops import FP, IP


@always_inline
def _lp(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


@always_inline
def _li(mut l: List[Int32]) -> IP:
    return IP(unsafe_from_address=Int(l.unsafe_ptr()))


def discretize_df64_host(
    x: List[Float32], n: Int, k: Int, seed: UInt64, max_svd_restarts: Int, n_iter_max: Int, mut n_iter_out: Int
) raises -> List[Int32]:
    """The host column of the device driver (x_cluster/spectral_df64_device.mojo):
    the same bodies in the same order, the row and partial steps in parallel
    over the host pool (each index writes only its own words), the k x k SVD
    serial (each rotation reads the previous one's words)."""
    if n < 1 or k < 1 or len(x) != n * k:
        raise Error("SpectralClustering discretize: the embedding is not n x k")
    if max_svd_restarts < 1:
        raise Error("SpectralClustering discretize: SVD did not converge")
    var nk = n * k
    var kk = k * k
    var nch = sd_chunks(n)
    var xl = x.copy()
    var vhl = List[Float32](length=nk, fill=Float32(0))
    var vll = List[Float32](length=nk, fill=Float32(0))
    var phl = List[Float32](length=max(nch * kk, nch * k), fill=Float32(0))
    var pll = List[Float32](length=max(nch * kk, nch * k), fill=Float32(0))
    var cnhl = List[Float32](length=k, fill=Float32(0))
    var cnll = List[Float32](length=k, fill=Float32(0))
    var sgnl = List[Float32](length=k, fill=Float32(0))
    var stl = List[Int32](length=SD_STATE_WORDS, fill=Int32(0))
    var px = _lp(xl)
    var vh = _lp(vhl)
    var vl = _lp(vll)
    var ph = _lp(phl)
    var pl = _lp(pll)
    var cnh = _lp(cnhl)
    var cnl = _lp(cnll)
    var sgn = _lp(sgnl)
    var state = _li(stl)

    def load(t: Int) {imm px, imm vh, imm vl}:
        sd_load_cell(px, vh, vl, t)

    host_cells(load, nk, 2)

    def colsq(t: Int) {imm vh, imm vl, imm n, imm k, imm ph, imm pl}:
        sd_colsq_part(vh, vl, n, k, t // k, t % k, ph, pl)

    host_cells(colsq, nch * k, 20 * SD_ROWS)
    for c in range(k):  # small-loop(k: embedding columns): each folds its nch partials
        sd_colnorm(ph, pl, nch, k, c, cnh, cnl, state)
    sd_raise_status(Int(stl[SD_STATUS]))
    var rootn = ff_sqrt(sd_of_int(n))

    def colscale(t: Int) {imm vh, imm vl, imm k, imm cnh, imm cnl, imm rootn, imm sgn}:
        sd_colscale_cell(vh, vl, k, cnh, cnl, rootn, sgn, t)

    host_cells(colscale, nk, 40)

    def colsign(t: Int) {imm vh, imm vl, imm k, imm sgn}:
        sd_colsign_cell(vh, vl, k, sgn, t)

    host_cells(colsign, nk, 2)

    def rownorm(i: Int) {imm vh, imm vl, imm k, imm state}:
        sd_rownorm_row(vh, vl, k, i, state)

    host_cells(rownorm, n, 40 * k)
    sd_raise_status(Int(stl[SD_STATUS]))
    # the greedy orthogonal start
    var rothl = List[Float32](length=kk, fill=Float32(0))
    var rotll = List[Float32](length=kk, fill=Float32(0))
    var cahl = List[Float32](length=n, fill=Float32(0))
    var call = List[Float32](length=n, fill=Float32(0))
    var roth = _lp(rothl)
    var rotl = _lp(rotll)
    var cah = _lp(cahl)
    var cal = _lp(call)
    var bmhl = List[Float32](length=nch, fill=Float32(0))
    var bmll = List[Float32](length=nch, fill=Float32(0))
    var bmil = List[Int32](length=nch, fill=Int32(0))
    var bmh = _lp(bmhl)
    var bml = _lp(bmll)
    var bmi = _li(bmil)
    var rng = SplitMix64(state=seed)
    var r0 = rng.below(n)
    for r in range(k):  # small-loop(k: embedding columns): the start row's words
        sd_take_row_cell(vh, vl, k, r0, roth, rotl, 0, r)
    for j in range(1, k):  # driver-loop: over the start's k - 1 greedy columns (model structure)
        var col = j - 1

        def cacc(i: Int) {imm vh, imm vl, imm k, imm roth, imm rotl, imm col, imm cah, imm cal}:
            sd_cacc_row(vh, vl, k, roth, rotl, col, cah, cal, i)

        host_cells(cacc, n, 20 * k)
        # the argmin: each SD_ROWS chunk's (value, index) minimum in
        # parallel, then the chunks' (exact compares: any order gives the
        # device's answer)

        def cmin(ch: Int) {imm cah, imm cal, imm n, imm bmh, imm bml, imm bmi}:
            var bi = -1
            var bv = sd_zero()
            for i in range(ch * SD_ROWS, min(ch * SD_ROWS + SD_ROWS, n)):
                var v = sd_ld(cah, cal, i)
                if sd_better_min(v, i, bv, bi):
                    bv = v
                    bi = i
            sd_st(bmh, bml, ch, bv)
            bmi.unsafe_store(ch, Int32(bi))

        host_cells(cmin, nch, 4 * SD_ROWS)
        var bi = -1
        var bv = sd_zero()
        for ch in range(nch):  # small-loop(nch: row chunks): the chunk minima, exact compares
            var v = sd_ld(bmh, bml, ch)
            var vi = Int(bmi.unsafe_load(ch))
            if sd_better_min(v, vi, bv, bi):
                bv = v
                bi = vi
        for r in range(k):  # small-loop(k: embedding columns): the argmin row's words
            sd_take_row_cell(vh, vl, k, bi, roth, rotl, j, r)
    # the iterations
    var labl = List[Int32](length=n, fill=Int32(0))
    var lab = _li(labl)
    var mhl = List[Float32](length=kk, fill=Float32(0))
    var mll = List[Float32](length=kk, fill=Float32(0))
    var mh = _lp(mhl)
    var ml = _lp(mll)
    var vvhl = List[Float32](length=kk, fill=Float32(0))
    var vvll = List[Float32](length=kk, fill=Float32(0))
    var uhl = List[Float32](length=kk, fill=Float32(0))
    var ull = List[Float32](length=kk, fill=Float32(0))
    var svhl = List[Float32](length=k, fill=Float32(0))
    var svll = List[Float32](length=k, fill=Float32(0))
    var chl = List[Float32](length=k, fill=Float32(0))
    var cll = List[Float32](length=k, fill=Float32(0))
    var fil = List[Int32](length=k, fill=Int32(0))
    var lastl = List[Float32](length=2, fill=Float32(0))
    var vvh = _lp(vvhl)
    var vvl = _lp(vvll)
    var uh = _lp(uhl)
    var ul = _lp(ull)
    var svh = _lp(svhl)
    var svl = _lp(svll)
    var cch = _lp(chl)
    var ccl = _lp(cll)
    var filled = _li(fil)
    var lh = _lp(lastl)
    var ll = lh + 1
    while stl[SD_DONE] == Int32(0):  # driver-loop: over the discretize iterations (at most n_iter_max + 1)
        stl[SD_ITER] = stl[SD_ITER] + 1

        def label(i: Int) {imm vh, imm vl, imm k, imm roth, imm rotl, imm lab}:
            lab.unsafe_store(i, sd_label_row(vh, vl, k, roth, rotl, i))

        host_cells(label, n, 20 * k * k)

        def cpart(t: Int) {imm vh, imm vl, imm lab, imm n, imm k, imm kk, imm ph, imm pl}:
            sd_class_part(vh, vl, lab, n, k, t // kk, t % kk, ph, pl)

        host_cells(cpart, nch * kk, 4 * SD_ROWS)

        def cfold(t: Int) {imm ph, imm pl, imm nch, imm kk, imm mh, imm ml}:
            sd_class_fold(ph, pl, nch, kk, t, mh, ml)

        host_cells(cfold, kk, 20 * nch)
        _sd_svd_host(mh, ml, vvh, vvl, uh, ul, svh, svl, filled, cch, ccl, k, state)
        sd_raise_status(Int(stl[SD_STATUS]))
        if not sd_ncut_step(svh, svl, k, n, lh, ll, state, n_iter_max):
            # rotation = V U^T

            def polar(t: Int) {imm vvh, imm vvl, imm uh, imm ul, imm k, imm roth, imm rotl}:
                sd_polar_cell(vvh, vvl, uh, ul, k, t, roth, rotl)

            host_cells(polar, kk, 20 * k)
    n_iter_out = Int(stl[SD_ITER])
    _ = xl^
    _ = vhl^
    _ = vll^
    _ = phl^
    _ = pll^
    _ = cnhl^
    _ = cnll^
    _ = sgnl^
    _ = rothl^
    _ = rotll^
    _ = cahl^
    _ = call^
    _ = bmhl^
    _ = bmll^
    _ = bmil^
    _ = mhl^
    _ = mll^
    _ = vvhl^
    _ = vvll^
    _ = uhl^
    _ = ull^
    _ = svhl^
    _ = svll^
    _ = chl^
    _ = cll^
    _ = fil^
    _ = lastl^
    return labl^


def _sd_svd_host(
    mh: FP, ml: FP, vh: FP, vl: FP, uh: FP, ul: FP, svh: FP, svl: FP, filled: IP, ch: FP, cl: FP, k: Int, state: IP
):
    """The device step kernel's SVD on the host: m (k x k) rotated in place
    (it is the class sums, rebuilt every iteration), v = I rotated, then sv,
    u and the completion. Serial: each rotation reads the previous one's
    words."""
    for t in range(k * k):
        var i = t // k
        sd_st(vh, vl, t, FF(Float32(1) if i * k + i == t else Float32(0), Float32(0)))
    for _sweep in range(SD_SVD_SWEEPS):
        var rotated = False
        for p in range(k - 1):
            for q in range(p + 1, k):
                var rot = sd_rotation(sd_dot_cols(mh, ml, k, p, p), sd_dot_cols(mh, ml, k, q, q), sd_dot_cols(mh, ml, k, p, q))
                if not rot[0]:
                    continue
                rotated = True
                for r in range(k):
                    sd_apply_row(mh, ml, vh, vl, k, p, q, rot[1], rot[2], r)
        if not rotated:
            break
    for j in range(k):
        sd_norm_col(mh, ml, uh, ul, svh, svl, filled, k, j)
    if not sd_complete_u(uh, ul, filled, ch, cl, k):
        state.unsafe_store(SD_STATUS, Int32(SD_BAD_BASIS))
