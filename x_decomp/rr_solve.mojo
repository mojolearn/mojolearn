# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE host eigh of the round-robin order (cgr-decomp, 2026-10-03): the
rounds of x_decomp/rr.mojo (`host_eigh_rr`), then `host_sign_flip` and
`eigh_ascending`. It computes the words of `DevExec.eigh` and of each
problem of `rr_batch_kernel`; the x_decomp host column and the spectral
oracle's projected solve call it. No GPU import."""
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.host.linalg_public import eigh_ascending
from decomposition.host.pca_oracle import host_sign_flip
from checks.numerics import ftz
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from x_decomp.cells import F32Ptr
from x_decomp.rr import RR_EIGH_SWEEPS, host_eigh_rr, rr_converged, rr_fro_kept, rr_off_fold
from x_decomp.rr_block import (
    RB_B,
    RB_W,
    rb_blocks,
    rb_gather_cell,
    rb_hi,
    rb_idx,
    rb_is_pad,
    rb_left_cell,
    rb_lo,
    rb_rank_real,
    rb_row_dot,
    rb_tol,
    rb_use,
)


def host_eigh_rr_sorted(mut m: List[Float32], n: Int, mut w: List[Float32], mut v: List[Float32]) raises -> Int:
    """`m` (n x n row major, consumed): w = n values ascending, v = n x n row
    major, vector c in COLUMN c. Not converged in RR_EIGH_SWEEPS raises (no
    cyclic fallback). Returns the sweeps run."""
    # lane fam2-decomp: the block Jacobi at large n, the device's choice
    # (`rb_use`, x_decomp/rr_block.mojo; `DevExec._eigh_par_on`)
    if rb_use(n):
        return host_eigh_rb_sorted(m, n, w, v)
    var vr = List[Float32](length=n * n, fill=Float32(0.0))
    var rr = host_eigh_rr(m, vr, n, RR_EIGH_SWEEPS, Float32(JACOBI_TOL))
    if not rr[0]:
        raise Error(
            "eigh: the round-robin Jacobi did not converge in " + String(RR_EIGH_SWEEPS)
            + " sweeps at n = " + String(n) + ". An unconverged decomposition is not returned"
            " as if it were one (DEVIATION 590)."
        )
    host_sign_flip(vr, n)
    var diag = List[Float32]()
    for i in range(n):
        diag.append(m[i * n + i])
    var got = eigh_ascending(diag, vr, n, True, rr[1])
    w = got.w.copy()
    v = got.v.copy()
    return rr[1]


def _rb_par[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """Tasks 0 .. n - 1 in contiguous groups on the host pool (x_decomp/host.mojo
    `xd_parallel`'s split). Every task writes only its own cells, so the
    words are the serial loop's."""
    var groups = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, groups)

    def group(g: Int) {imm func, imm n, imm chunk}:
        for i in range(g * chunk, min(n, (g + 1) * chunk)):
            func(i)

    if groups <= 1:
        group(0)
        return
    host_parallelize(group, groups)


def host_eigh_rb_sorted(mut a_in: List[Float32], n: Int, mut w: List[Float32], mut v: List[Float32]) raises -> Int:
    """THE host eigh at n >= RB_MIN_N in an IDENTICAL build: the block Jacobi
    of x_decomp/rr_block.mojo, `DevExec._eigh_block_on`'s walk (the same
    cells in the same order: the test, then per block round the gather, the
    pivot solves, T = A W, A = W^T T, V = V W; then the sign flip and the
    real columns ascending). `a_in` n x n row major (read); w = n values
    ascending, v = n x n row major, vector c in column c. Returns the sweeps
    run; an unconverged solve raises."""
    var m = rb_blocks(n)
    var h = m // 2
    var nn = m * RB_B
    var ww = RB_W * RB_W
    var al = List[Float32](length=nn * nn, fill=Float32(0.0))
    for i in range(n):
        for j in range(n):
            al[i * nn + j] = a_in[i * n + j]
    var tl = List[Float32](length=nn * nn, fill=Float32(0.0))
    var v0 = List[Float32](length=nn * nn, fill=Float32(0.0))
    var v1 = List[Float32](length=nn * nn, fill=Float32(0.0))
    for i in range(nn):
        v0[i * nn + i] = Float32(1.0)
    var wvl = List[Float32](length=h * ww, fill=Float32(0.0))
    var wll = List[Float32](length=h * RB_W, fill=Float32(0.0))
    var pa = F32Ptr(unsafe_from_address=Int(al.unsafe_ptr()))
    var pt = F32Ptr(unsafe_from_address=Int(tl.unsafe_ptr()))
    var pv0 = F32Ptr(unsafe_from_address=Int(v0.unsafe_ptr()))
    var pv1 = F32Ptr(unsafe_from_address=Int(v1.unsafe_ptr()))
    var pwv = F32Ptr(unsafe_from_address=Int(wvl.unsafe_ptr()))
    var pwl = F32Ptr(unsafe_from_address=Int(wll.unsafe_ptr()))
    var tol = Float32(JACOBI_TOL)
    var tloc = rb_tol(tol, m)
    var cur = 0
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    for sweep in range(RR_EIGH_SWEEPS + 1):
        var sums = rr_off_fold(pa, nn)
        fro_now = ftz(sums[0] + sums[1])
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(sums[0], sums[1], tol):
            converged = True
            break
        if sweep == RR_EIGH_SWEEPS:
            break
        executed += 1
        for rd in range(m - 1):
            # the pivot problems, each solved completely (rr_batch_kernel's
            # words: host_eigh_rr to the local tolerance, the sign flip, the
            # ascending order)
            for g in range(h):
                var lo = rb_lo(m, rd, g)
                var hi = rb_hi(m, rd, g)
                var pl = List[Float32](length=ww, fill=Float32(0.0))
                for x in range(RB_W):
                    for y in range(RB_W):
                        pl[x * RB_W + y] = rb_gather_cell(pa, nn, lo, hi, x, y)
                var vr = List[Float32](length=ww, fill=Float32(0.0))
                var got = host_eigh_rr(pl, vr, RB_W, RR_EIGH_SWEEPS, tloc)
                if not got[0]:
                    raise Error(
                        "eigh: a " + String(RB_W) + " x " + String(RB_W) + " pivot problem of the block Jacobi did"
                        " not converge in " + String(RR_EIGH_SWEEPS) + " sweeps at n = " + String(n)
                        + ". An unconverged decomposition is not returned as if it were one (DEVIATION 590)."
                    )
                host_sign_flip(vr, RB_W)
                var dg = List[Float32]()
                for i in range(RB_W):
                    dg.append(pl[i * RB_W + i])
                var srt = eigh_ascending(dg, vr, RB_W, True, got[1])
                for i in range(RB_W):
                    wll[g * RB_W + i] = srt.w[i]
                for i in range(ww):
                    wvl[g * ww + i] = srt.v[i]
            var vsrc = pv0 if cur == 0 else pv1
            var vdst = pv1 if cur == 0 else pv0

            def right_row(u: Int) {imm pa, imm pt, imm pwv, imm nn, imm m, imm rd}:
                var gi = u // RB_W
                var row = rb_idx(rb_lo(m, rd, gi), rb_hi(m, rd, gi), u - gi * RB_W)
                for c in range((gi + 1) * RB_W, nn):
                    var gj = c // RB_W
                    pt.unsafe_store(
                        u * nn + c,
                        rb_row_dot(pa + row * nn, pwv + gj * RB_W * RB_W, rb_lo(m, rd, gj), rb_hi(m, rd, gj), c - gj * RB_W),
                    )

            _rb_par(right_row, nn)

            def left_row(u: Int) {imm pa, imm pt, imm pwv, imm pwl, imm nn, imm m, imm rd}:
                var gi = u // RB_W
                var lr = u - gi * RB_W
                var ra = rb_idx(rb_lo(m, rd, gi), rb_hi(m, rd, gi), lr)
                for c in range(gi * RB_W, nn):
                    var gj = c // RB_W
                    var lc = c - gj * RB_W
                    var ca = rb_idx(rb_lo(m, rd, gj), rb_hi(m, rd, gj), lc)
                    if gi < gj:
                        var x = rb_left_cell(pt, pwv + gi * RB_W * RB_W, nn, gi, lr, c)
                        pa.unsafe_store(ra * nn + ca, x)
                        pa.unsafe_store(ca * nn + ra, x)
                    else:
                        var d = Float32(0.0)
                        if lr == lc:
                            d = pwl.unsafe_load(gi * RB_W + lr)
                        pa.unsafe_store(ra * nn + ca, d)

            _rb_par(left_row, nn)

            def v_row(k: Int) {imm vsrc, imm vdst, imm pwv, imm nn, imm m, imm rd}:
                for c in range(nn):
                    var g = c // RB_W
                    var lc = c - g * RB_W
                    var lo = rb_lo(m, rd, g)
                    var hi = rb_hi(m, rd, g)
                    vdst.unsafe_store(k * nn + rb_idx(lo, hi, lc), rb_row_dot(vsrc + k * nn, pwv + g * RB_W * RB_W, lo, hi, lc))

            _rb_par(v_row, nn)
            cur = 1 - cur
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    if not converged:
        raise Error(
            "eigh: the block Jacobi did not converge in " + String(RR_EIGH_SWEEPS)
            + " sweeps at n = " + String(n) + ". An unconverged decomposition is not returned"
            " as if it were one (DEVIATION 590)."
        )
    # the tail: sign flip on the N x N basis, the pad columns marked, the
    # real ones ascending (rb_pad_kernel, rb_rank_kernel, rb_scatter_kernel)
    if cur == 0:
        host_sign_flip(v0, nn)
    else:
        host_sign_flip(v1, nn)
    var vfin = pv0 if cur == 0 else pv1
    var key = List[Float32](length=nn, fill=Float32(0.0))
    var pad = List[Float32](length=nn, fill=Float32(0.0))
    for c in range(nn):
        key[c] = al[c * nn + c]
        if rb_is_pad(vfin, nn, n, c):
            pad[c] = Float32(1.0)
    var pkey = F32Ptr(unsafe_from_address=Int(key.unsafe_ptr()))
    var ppad = F32Ptr(unsafe_from_address=Int(pad.unsafe_ptr()))
    var wo = List[Float32](length=n, fill=Float32(0.0))
    var vo = List[Float32](length=n * n, fill=Float32(0.0))
    for i in range(nn):
        if pad[i] == Float32(0.0):
            var dcol = n - 1 - rb_rank_real(pkey, ppad, nn, i)
            if dcol >= 0 and dcol < n:
                wo[dcol] = key[i]
                for r in range(n):
                    vo[r * n + dcol] = vfin.unsafe_load(r * nn + i)
    w = wo^
    v = vo^
    # the pointers above read these lists to here
    _ = key^
    _ = pad^
    _ = al^
    _ = tl^
    _ = v0^
    _ = v1^
    _ = wvl^
    _ = wll^
    return executed
