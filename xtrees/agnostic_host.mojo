# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-agnostic explainers on the host, for the CPU-only binding and
the CPU verification column (lane cgr2-metrics-shap, 2026-10-03):
xtrees/agnostic.mojo's units in xtrees/agnostic_device.mojo's stage order,
each stage's units in ascending t, so the words are the GPU columns'
words. GPU installs never run this file."""
from xtrees.agnostic import (
    F32P, I32P, U64P, I64P, kshap_mask_unit, kshap_synth_unit, bg_mean_unit, logit_unit, kshap_gram_unit,
    kshap_rhs_unit, kshap_pivot_unit, kshap_elim_unit, kshap_back_unit, fx_unit, pshap_perm_unit,
    pshap_synth_unit, pshap_marginal_unit,
)


@always_inline
def _i32(mut v: List[Int32]) -> I32P:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _u64(mut v: List[UInt64]) -> U64P:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct _Masks(Movable):
    var masks: List[Int32]
    var w: List[UInt64]
    var perm: List[Int32]

    def __init__(out self, size_off: Int, size_w: Int, cdf: Int, R: Int, d: Int, m: Int, nfixed: Int, nfull: Int,
                 npaired: Int, L: Int, seed: Int, row0: Int, wrand: UInt64):
        self.masks = List[Int32](length=max(R * m * d, 1), fill=0)
        self.perm = List[Int32](length=max(R * m * d, 1), fill=0)
        self.w = List[UInt64](length=max(R * m, 1), fill=0)
        var so = I64P(unsafe_from_address=size_off)
        var sw = U64P(unsafe_from_address=size_w)
        var cd = U64P(unsafe_from_address=cdf)
        var pm = _i32(self.perm)
        var mk = _i32(self.masks)
        var wp = _u64(self.w)
        for t in range(R * m):
            kshap_mask_unit(t, d, m, nfixed, nfull, npaired, L, seed, row0, so, sw, cd, wrand, pm, mk, wp)


def kshap_synth(x: Int, bg: Int, size_off: Int, size_w: Int, cdf: Int, syn: Int, R: Int, nb: Int, d: Int, m: Int,
                nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int, wrand: UInt64) raises:
    var total = R * m * nb * d
    if total <= 0:
        return
    var mk = _Masks(size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var xp = F32P(unsafe_from_address=x)
    var bp = F32P(unsafe_from_address=bg)
    var sp = F32P(unsafe_from_address=syn)
    var mp = _i32(mk.masks)
    for t in range(total):
        kshap_synth_unit(t, nb, d, m, xp, bp, mp, sp)
    _ = len(mk.masks)


def kshap_solve(yout: Int, fx: Int, fnull: Int, size_off: Int, size_w: Int, cdf: Int, phi: Int, R: Int, nb: Int,
                d: Int, k: Int, m: Int, nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int,
                wrand: UInt64, link: Bool) raises:
    if R <= 0:
        return
    var mk = _Masks(size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var ey = List[UInt64](length=max(R * m * k, 1), fill=0)
    var eyp = _u64(ey)
    var op = F32P(unsafe_from_address=yout)
    for t in range(R * m * k):
        bg_mean_unit(t, nb, k, op, eyp)
    if link:
        for t in range(R * m * k):
            logit_unit(t, eyp)
    var f64 = List[UInt64](length=max(R * k, 1), fill=0)
    var fxp = _u64(f64)
    var fx32 = F32P(unsafe_from_address=fx)
    for t in range(R * k):
        fx_unit(t, fx32, fxp)
    if link:
        for t in range(R * k):
            logit_unit(t, fxp)
    var nul = U64P(unsafe_from_address=fnull)
    var q = d - 1
    var A = List[UInt64](length=max(R * q * q, 1), fill=0)
    var B = List[UInt64](length=max(R * q * k, 1), fill=0)
    var p0 = List[Int32](length=max(R * q, 1), fill=0)
    var p1 = List[Int32](length=max(R * q, 1), fill=0)
    var sol = List[UInt64](length=max(R * k * q, 1), fill=0)
    var Ap = _u64(A)
    var Bp = _u64(B)
    var mp = _i32(mk.masks)
    var wp = _u64(mk.w)
    if q > 0:
        for t in range(R * q * q):
            kshap_gram_unit(t, d, m, mp, wp, Ap, _i32(p0))
        for t in range(R * q * k):
            kshap_rhs_unit(t, d, m, k, mp, wp, eyp, fxp, nul, Bp)
    var in0 = True
    for col in range(q):
        var pin = _i32(p0) if in0 else _i32(p1)
        var pout = _i32(p1) if in0 else _i32(p0)
        for t in range(R):
            kshap_pivot_unit(t, q, col, Ap, pin, pout)
        var nr = q - 1 - col
        for t in range(R * nr * (nr + k)):
            kshap_elim_unit(t, q, k, col, Ap, Bp, pout)
        in0 = not in0
    var pfin = _i32(p0) if in0 else _i32(p1)
    var php = U64P(unsafe_from_address=phi)
    for t in range(R * k):
        kshap_back_unit(t, d, k, Ap, Bp, pfin, fxp, nul, _u64(sol), php)
    _ = len(mk.masks)
    _ = len(ey)
    _ = len(f64)
    _ = len(A)
    _ = len(B)
    _ = len(p0)
    _ = len(p1)
    _ = len(sol)


def pshap_synth(x: Int, bg: Int, syn: Int, R: Int, nb: Int, d: Int, np: Int, seed: Int, row0: Int) raises:
    var total = R * np * (2 * d + 1) * nb * d
    if total <= 0:
        return
    var perm = List[Int32](length=R * np * d, fill=0)
    var inv = List[Int32](length=R * np * d, fill=0)
    for t in range(R * np):
        pshap_perm_unit(t, d, np, seed, row0, _i32(perm), _i32(inv))
    var xp = F32P(unsafe_from_address=x)
    var bp = F32P(unsafe_from_address=bg)
    var sp = F32P(unsafe_from_address=syn)
    var ip = _i32(inv)
    for t in range(total):
        pshap_synth_unit(t, nb, d, np, xp, bp, ip, sp)
    _ = len(perm)
    _ = len(inv)


def pshap_values(yout: Int, phi: Int, R: Int, nb: Int, d: Int, k: Int, np: Int, seed: Int, row0: Int) raises:
    var mm = np * (2 * d + 1)
    if R * d * k <= 0 or mm <= 0:
        return
    var perm = List[Int32](length=R * np * d, fill=0)
    var inv = List[Int32](length=R * np * d, fill=0)
    for t in range(R * np):
        pshap_perm_unit(t, d, np, seed, row0, _i32(perm), _i32(inv))
    var ey = List[UInt64](length=R * mm * k, fill=0)
    var op = F32P(unsafe_from_address=yout)
    for t in range(R * mm * k):
        bg_mean_unit(t, nb, k, op, _u64(ey))
    var php = U64P(unsafe_from_address=phi)
    for t in range(R * d * k):
        pshap_marginal_unit(t, d, k, np, _i32(inv), _u64(ey), php)
    _ = len(perm)
    _ = len(inv)
    _ = len(ey)


def bg_mean(y: Int, res: Int, m: Int, nb: Int, k: Int) raises:
    var yp = F32P(unsafe_from_address=y)
    var rp = U64P(unsafe_from_address=res)
    for t in range(m * k):
        bg_mean_unit(t, nb, k, yp, rp)
