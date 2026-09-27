# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Adafactor, as PyTorch 2.5 states it (`torch/optim/_adafactor.py`,
`_single_tensor_adafactor`): relative step alpha = max(eps2, RMS(p)) rho_t,
rho_t = min(lr, 1/sqrt(t)); decoupled weight decay; for a matrix the
factored second moment row_var (mean over columns of g^2) and col_var (mean
over rows), each lerped toward the new mean with weight t^beta2_decay, the
estimate row_var col_var / max(mean(row_var), eps1); for a vector the full
second moment; the update g / sqrt(max(estimate, eps1^2)) scaled by
1 / max(1, RMS(update) / d). One tensor per call; each reduction is one
thread's ascending loop. torch's norms are sqrt(sum of squares), squared
back where the reference squares them, and its lerp is torch's two-branch
formula."""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_rsqrt, identical_sqrt


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def lerp(s: Float32, e: Float32, w: Float32) -> Float32:
    """torch.lerp: s + w (e - s) for w < 0.5, else e - (e - s)(1 - w)."""
    if w < Float32(0.5):
        return fma3(w, sub(e, s), s)
    return sub(e, mul(sub(e, s), sub(Float32(1.0), w)))


def _sumsq(p: FP, start: Int, n: Int, stride: Int) -> Float32:
    var acc = Float32(0.0)
    for k in range(n):
        var v = ld(p, start + k * stride)
        acc = fma3(v, v, acc)
    return acc


def op_af_alpha(t: Int, a: Args):
    """One thread: p1[1] = max(eps2, ||p0|| / sqrt(numel)) rho.
    i0 numel; f0 eps2, f1 rho."""
    var n = a.i0
    var rms = div(ftz(identical_sqrt(_sumsq(a.p0, 0, n, 1))), ftz(identical_sqrt(Float32(n))))
    var m = rms if rms > a.f0 else a.f0
    st(a.p1, 1, mul(m, a.f1))


def op_af_row(t: Int, a: Args):
    """Row t of grad p0 [R, C]: p1[t] = lerp(p1[t], ||g_t||^2 / C, f0); i0 C."""
    var C = a.i0
    var nrm = ftz(identical_sqrt(_sumsq(a.p0, t * C, C, 1)))
    st(a.p1, t, lerp(ld(a.p1, t), div(mul(nrm, nrm), Float32(C)), a.f0))


def op_af_col(t: Int, a: Args):
    """Column t: p1[t] = lerp(p1[t], ||g_:,t||^2 / R, f0); i0 R, i1 C."""
    var R = a.i0
    var nrm = ftz(identical_sqrt(_sumsq(a.p0, t, R, a.i1)))
    st(a.p1, t, lerp(ld(a.p1, t), div(mul(nrm, nrm), Float32(R)), a.f0))


def op_af_rmean(t: Int, a: Args):
    """One thread: p1[2] = max(mean(p0[0:R]), eps1); i0 R, f0 eps1."""
    var s = Float32(0.0)
    for k in range(a.i0):
        s = add(s, ld(a.p0, k))
    var m = div(s, Float32(a.i0))
    st(a.p1, 2, m if m > a.f0 else a.f0)


@always_inline
def _upd(v: Float32, g: Float32, eps1sq: Float32) -> Float32:
    var c = v if v > eps1sq else eps1sq
    return mul(ftz(identical_rsqrt(c)), g)


def op_af_update_mat(t: Int, a: Args):
    """p4[r, c] = rsqrt(max(row[r] col[c] / p3[2], eps1^2)) g[r, c]; i0 C,
    f0 eps1^2. p0 grad, p1 row_var, p2 col_var, p3 scalars."""
    var C = a.i0
    var r = t // C
    var c = t - r * C
    var ve = div(mul(ld(a.p1, r), ld(a.p2, c)), ld(a.p3, 2))
    st(a.p4, t, _upd(ve, ld(a.p0, t), a.f0))


def op_af_vec(t: Int, a: Args):
    """p1[t] = lerp(p1[t], g^2, f0); p2[t] = rsqrt(max(p1[t], f1)) g."""
    var g = ld(a.p0, t)
    var v = lerp(ld(a.p1, t), mul(g, g), a.f0)
    st(a.p1, t, v)
    st(a.p2, t, _upd(v, g, a.f1))


def op_af_denom(t: Int, a: Args):
    """One thread: p1[3] = -p1[1] / max(1, ||p0|| / (sqrt(numel) d));
    i0 numel, f0 d."""
    var n = a.i0
    var r = div(ftz(identical_sqrt(_sumsq(a.p0, 0, n, 1))), mul(ftz(identical_sqrt(Float32(n))), a.f0))
    var den = r if r > Float32(1.0) else Float32(1.0)
    st(a.p1, 3, div(-ld(a.p1, 1), den))


def op_af_apply(t: Int, a: Args):
    """p0[t] += p1[t] p2[3]."""
    st(a.p0, t, fma3(ld(a.p1, t), ld(a.p2, 3), ld(a.p0, t)))


# ------------------------------------------------------------------ LAMB
def op_seg_sumsq(t: Int, a: Args):
    """Segment t of p0 (offsets p1[t] .. p1[t + 1], as floats):
    p2[t] = its sum of squares, ascending."""
    var s = Int(a.p1.unsafe_load(t))
    var e = Int(a.p1.unsafe_load(t + 1))
    st(a.p2, t, _sumsq(a.p0, s, e - s, 1))


def op_lamb_upd(t: Int, a: Args):
    """m = b1 m + beta3 g; v = b2 v + (1 - b2) g g; u = (m / bc1) /
    (sqrt(v) / sqrt(bc2) + eps) (+ wd p). p0 param, p1 grad, p2 m, p3 v,
    p4 u; f1 b1, f2 b2, f3 eps, f4 wd, f5 bc1, f6 sqrt(bc2), f7 beta3."""
    var g = ld(a.p1, t)
    var m = fma3(a.f1, ld(a.p2, t), mul(a.f7, g))
    var v = fma3(a.f2, ld(a.p3, t), mul(mul(sub(Float32(1.0), a.f2), g), g))
    st(a.p2, t, m)
    st(a.p3, t, v)
    var den = add(div(ftz(identical_sqrt(v)), a.f6), a.f3)
    var u = div(div(m, a.f5), den)
    if a.f4 != Float32(0.0):
        u = fma3(a.f4, ld(a.p0, t), u)
    st(a.p4, t, u)


def op_lamb_ratio(t: Int, a: Args):
    """Segment t: p3[t] = ||p|| / ||u|| (1 where either is 0; at most 1
    when i0 & 1, timm's trust_clip). p0 param, p1 u, p2 offsets."""
    var s = Int(a.p2.unsafe_load(t))
    var e = Int(a.p2.unsafe_load(t + 1))
    var wn = ftz(identical_sqrt(_sumsq(a.p0, s, e - s, 1)))
    var gn = ftz(identical_sqrt(_sumsq(a.p1, s, e - s, 1)))
    var r = Float32(1.0)
    if wn > Float32(0.0) and gn > Float32(0.0):
        r = div(wn, gn)
    if (a.i0 & 1) != 0 and r > Float32(1.0):
        r = Float32(1.0)
    st(a.p3, t, r)


def op_lamb_apply(t: Int, a: Args):
    """p0[t] = p0[t] - lr (u[t] ratio): p1 u, p2 ratios, i0 the segment of
    this launch, f0 lr."""
    var r = ld(a.p2, a.i0)
    st(a.p0, t, fma3(-a.f0, mul(ld(a.p1, t), r), ld(a.p0, t)))
