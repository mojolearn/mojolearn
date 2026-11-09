# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/fa_em.mojo's FactorAnalysis EM loop on resident device matrices
(lane py-runtime-b): the same statements on `DKit`. The d x d spectral
step stays resident on both branches: the SVD (n >= d) on the executor op
(`DKit.svd`, the descending order on the device; lane n9b-device-svd-qr),
the eigh (n < d) on `DKit.eigh` (its reversal and transpose as device
copies), and the
log-likelihood's float32 words come home for `_dsum`, as Python read them.
GPU binding only."""
from std.math import inf, sqrt
from experiments.classical_identical_ideas.linear_controls import C27_FA_COMPONENTS
from x_decomp.cells import F32Ptr, OP_CLASSICAL_FA_LOAD
from x_decomp.fa_em import FA_LOG_FLOOR, FA_SMALL, FaArgs, dsum, fa_mask
from x_decomp.kit import (
    OP_ADD, OP_ADDS, OP_DIV, OP_LOGS, OP_MAXS, OP_MINS, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT, mat_from,
)
from x_decomp.kit_device import DKit, DMat


def fa_em_dev(mut k: DKit, A: DMat, mut psi: DMat, mut W: DMat, mut ll_out: List[Float64], a: FaArgs) raises -> Int:
    var n = a.n
    var d = a.d
    var nc = a.nc
    var nsqrt = sqrt(Float64(n))
    var old_ll = -inf[DType.float64]()
    var it = 0
    # perf-decomp-slow (2026-11-06): the psi update's keep / drop masks are
    # constants of (nc, d) (Vfull is d x d on both branches): uploaded once
    # per fit instead of twice per EM iteration. Same words, same order.
    var keep = k.upload(fa_mask(nc, d, True))
    var drop = k.upload(fa_mask(nc, d, False))
    for i in range(1, a.max_iter + 1):  # driver-loop: over the EM iterations
        it = i
        var sqrt_psi = k.ew1(OP_ADDS, k.ew1(OP_SQRT, psi, 0.0), FA_SMALL)
        var s2: DMat
        var Vfull: DMat
        if n >= d:
            var B = k.ew1(OP_SCALE, k.ew2(OP_DIV, A, sqrt_psi), 1.0 / nsqrt)
            # lane n9b-device-svd-qr: the SVD op, S descending and V^T left
            # resident (was s and V home, `svd_desc` on the host, both up
            # again); the same words
            var sv = k.zeros(0, 0)
            var Vh = k.zeros(0, 0)
            k.svd(B, sv, Vh)
            s2 = k.ew1(OP_SQ, sv, 0.0)
            Vfull = Vh^
        else:
            var Z = k.ew1(OP_SCALE, k.ew2(OP_DIV, A, sqrt_psi), 1.0 / nsqrt)
            var e = k.eigh(k.mm(Z, Z, True, False))
            # lane n9-decomp-eigh-families: the descending order and V^T on
            # the device (the same words `rev_cols` / `mat_t` copied on the
            # host after w and V came home and went up again)
            s2 = k.ew1(OP_MAXS, k.rev_vec(e.wd), 0.0)
            Vfull = k.rev_cols_t(e.vd)
        var Vt = k.rows(Vfull, 0, nc)
        # perf-decomp-slow (2026-11-06): s2's words come home with the two
        # log vectors below, at ONE sync per iteration (was a second sync
        # here). s2 is not written in between, so the same words reach the
        # same `dsum` in the same order.
        var s2h = k.get(s2)
        var sk = k.rows(k.vec_t(k.copy(s2)), 0, nc)
        sk = k.vec_t(sk^)
        comptime if C27_FA_COMPONENTS:
            W = k.ew3(OP_CLASSICAL_FA_LOAD, Vt, k.vec_t(k.copy(sk)), sqrt_psi, 0.0)
        else:
            W = k.ew2(OP_MUL, Vt, k.vec_t(k.ew1(OP_SQRT, k.ew1(OP_MAXS, k.ew1(OP_ADDS, sk, -1.0), 0.0), 0.0)))
            W = k.ew2(OP_MUL, W, sqrt_psi)
        var lskd = k.ew1(OP_LOGS, sk, FA_LOG_FLOOR)
        var lpsd = k.ew1(OP_LOGS, psi, FA_LOG_FLOOR)
        var lsk = k.get(lskd)
        var lps = k.get(lpsd)
        k.sync()
        _ = lskd^
        _ = lpsd^
        var unexp = dsum(s2h, nc, d) if nc < d else 0.0
        var slog = dsum(lsk, 0, lsk.n())
        var plog = dsum(lps, 0, lps.n())
        var ll = (a.llconst + slog + unexp + plog) * (-Float64(n) / 2.0)
        ll_out.append(ll)
        if (ll - old_ll) < a.tol:
            break
        old_ll = ll
        var wts = k.ew2(OP_ADD, k.ew2(OP_MUL, k.ew1(OP_MINS, s2, 1.0), keep), k.ew2(OP_MUL, s2, drop))
        var share = k.mm(wts, k.ew1(OP_SQ, Vfull, 0.0), False, False)
        psi = k.ew1(OP_MAXS, k.ew2(OP_MUL, k.ew1(OP_SQ, sqrt_psi, 0.0), share), FA_SMALL)
    return it


def fa_em_main_dev_run(
    pa: F32Ptr, pp: F32Ptr, pw: F32Ptr, pl: MutPointer[Float64, MutUntrackedOrigin], ar: Int, args: FaArgs
) raises -> Int:
    """`fa_em_main_run` on the resident kit."""
    var k = DKit()
    var A = k.upload(mat_from(pa, ar, args.d))
    var P = k.upload(mat_from(pp, 1, args.d))
    var W = k.zeros(args.nc, args.d)
    var lls = List[Float64]()
    var it = fa_em_dev(k, A, P, W, lls, args)
    var hp = k.get(P)
    var hw = k.get(W)
    k.sync()
    for i in range(args.d):
        pp.unsafe_store(i, hp.d[i])
    for i in range(hw.n()):
        pw.unsafe_store(i, hw.d[i])
    for i in range(len(lls)):
        pl.unsafe_store(i, lls[i])
    return it
