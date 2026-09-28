# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S HOST ORACLES, one per numeric seam (pass 2).

Each restates its seam as plain host loops over the numeric primitives of
checks/numerics.mojo (ftz, identical_mul / mul_add / div / exp / log / sqrt
/ rsqrt / sigmoid / tanh), written from the reference semantics and NOT from
sequence/: the lane's element bodies run on the device AND on the host
(sequence/exec.mojo), so a bug or a sabotage in a body moves both columns
together, and only an independent restatement catches it. Every oracle
takes `alt`, the seam spelled the OTHER legal way; sequence/checks/
seams_check.mojo first requires pinned != alt on its fixture (VACUOUS
otherwise), then device == oracle and host == oracle bit for bit.

Seams (DEVIATION numbers, lane sequence 5500-5599):
  5500 GEMM: k ascending, one fused multiply-add per term           alt: k descending
  5501 column sums (bias gradients): rows ascending                 alt: descending
  5502 LSTM cell state: c = fma(f, c_prev, i*g)                     alt: fma(i, g, f*c_prev)
  5503 GRU update: h = n + z (h_prev - n), one fma                  alt: (1 - z) n + z h_prev
  5504 BPTT weight gradient: ONE fold over (step, batch) rows
       ascending after the reverse sweep                            alt: per-step, steps descending
  5505 softmax cross entropy: the exp-sum in column order           alt: reversed
  5506 Adam denominator: sqrt(v) / sqrt(1 - b2^t) + eps (torch)     alt: sqrt(v / (1 - b2^t)) + eps
  5507 torch.lerp: two branches at w = 0.5 (Adamax, NAdam, Adafactor) alt: s + w (e - s) always
  5508 MLP log loss: p clipped to [eps32, 1 - eps32] (sklearn)      alt: no clip
  5509 MLP shuffle: splitmix64, Fisher-Yates i descending,
       j by rejection                                               alt: i ascending
  5510 STL robustness weights: cmad = 3 (r[m0] + r[m1])             alt: 3 r[m0] + 3 r[m1]
  5511 VAR column scaling: an exact power of two                    alt: 1 / max|x|
  5512 VAR Cholesky: inner sums k ascending, first bad pivot's
       status word                                                  alt: k descending
  5513 Nelder-Mead simplex order: ties keep the lower index        alt: the higher index
  5514 MoE top-k routing: ties go to the lower expert index         alt: the higher
  5515 LayerNorm statistics: the mean folded columns ascending,
       then the centred squares                                     alt: mean descending
  5516 SES recursion (Croston): alpha x + (1 - alpha) f, one fma    alt: f + alpha (x - f)
"""
from std.memory import bitcast

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
    identical_sqrt,
    identical_tanh,
)
from checks.fixture_rng import splitmix64_next


@always_inline
def _m(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


@always_inline
def _f(a: Float32, b: Float32, c: Float32) -> Float32:
    return ftz(identical_mul_add(ftz(a), ftz(b), ftz(c)))


@always_inline
def _a(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def _s(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def _d(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(ftz(a), ftz(b)))


@always_inline
def _z(x: Float32) -> Float32:
    return ftz(x)


# ---------------------------------------------------------------- 5500
def o_gemm(A: List[Float32], B: List[Float32], C0: List[Float32], M: Int, N: Int, K: Int, alt: Bool) -> List[Float32]:
    """C = C0 + A B, A [M, K], B [K, N] row-major."""
    var out = List[Float32](capacity=M * N)
    for m in range(M):
        for n in range(N):
            var acc = _z(C0[m * N + n])
            for kk in range(K):
                var k = K - 1 - kk if alt else kk
                acc = _f(A[m * K + k], B[k * N + n], acc)
            out.append(acc)
    return out^


# ---------------------------------------------------------------- 5540
def o_gemm_split(A: List[Float32], B: List[Float32], C0: List[Float32], M: Int, N: Int, K: Int) -> List[Float32]:
    """5540's alternative: `o_gemm`'s cells with the product rounded before
    the add (two roundings per term instead of the one fused rounding)."""
    var out = List[Float32](capacity=M * N)
    for m in range(M):
        for n in range(N):
            var acc = _z(C0[m * N + n])
            for k in range(K):
                acc = _a(_m(A[m * K + k], B[k * N + n]), acc)
            out.append(acc)
    return out^


# ---------------------------------------------------------------- 5501
def o_colsum(X: List[Float32], R: Int, C: Int, alt: Bool) -> List[Float32]:
    var out = List[Float32](capacity=C)
    for c in range(C):
        var acc = Float32(0.0)
        for rr in range(R):
            var r = R - 1 - rr if alt else rr
            acc = _a(acc, X[r * C + c])
        out.append(acc)
    return out^


# ---------------------------------------------------------------- 5502
def o_lstm(gx: List[Float32], gh: List[Float32], c_prev: List[Float32], B: Int, H: Int, alt: Bool) -> List[Float32]:
    """[h (B H), c (B H)] of one LSTM step (gate order i, f, g, o)."""
    var hs = List[Float32]()
    var cs = List[Float32]()
    for b in range(B):
        var row = b * 4 * H
        for u in range(H):
            var i_ = _z(identical_sigmoid(_a(gx[row + u], gh[row + u])))
            var f_ = _z(identical_sigmoid(_a(gx[row + H + u], gh[row + H + u])))
            var g_ = _z(identical_tanh(_a(gx[row + 2 * H + u], gh[row + 2 * H + u])))
            var o_ = _z(identical_sigmoid(_a(gx[row + 3 * H + u], gh[row + 3 * H + u])))
            var cp = c_prev[b * H + u]
            var c: Float32
            if alt:
                c = _f(i_, g_, _m(f_, cp))
            else:
                c = _f(f_, cp, _m(i_, g_))
            cs.append(c)
            hs.append(_m(o_, _z(identical_tanh(c))))
    for i in range(len(cs)):
        hs.append(cs[i])
    return hs^


# ---------------------------------------------------------------- 5503
def o_gru(gx: List[Float32], gh: List[Float32], h_prev: List[Float32], B: Int, H: Int, alt: Bool) -> List[Float32]:
    """h of one GRU step, PyTorch's gate order (r, z, n)."""
    var out = List[Float32]()
    for b in range(B):
        var row = b * 3 * H
        for u in range(H):
            var r = _z(identical_sigmoid(_a(gx[row + u], gh[row + u])))
            var z = _z(identical_sigmoid(_a(gx[row + H + u], gh[row + H + u])))
            var n = _z(identical_tanh(_f(r, gh[row + 2 * H + u], gx[row + 2 * H + u])))
            var hp = h_prev[b * H + u]
            if alt:
                out.append(_a(_m(_s(Float32(1.0), z), n), _m(z, hp)))
            else:
                out.append(_f(z, _s(hp, n), n))
    return out^


# ---------------------------------------------------------------- 5504
def o_bptt_dw(dgx: List[Float32], x: List[Float32], T: Int, B: Int, GH: Int, D: Int, alt: Bool) -> List[Float32]:
    """dW_ih [GH, D] = sum over rows r = s B + b of dgx[r, g] x[r, d]:
    pinned, r ascending in one fold; alt, the steps descending (the order a
    per-step accumulation during the reverse sweep folds them)."""
    var out = List[Float32](capacity=GH * D)
    for g in range(GH):
        for d in range(D):
            var acc = Float32(0.0)
            for ss in range(T):
                var s = T - 1 - ss if alt else ss
                for b in range(B):
                    var r = s * B + b
                    acc = _f(dgx[r * GH + g], x[r * D + d], acc)
            out.append(acc)
    return out^


# ---------------------------------------------------------------- 5505
def o_ce(logits: List[Float32], labels: List[Int], B: Int, C: Int, scale: Float32, alt: Bool) -> List[Float32]:
    """[grad (B C), loss (B)] of softmax cross entropy, the max first."""
    var grad = List[Float32]()
    var loss = List[Float32]()
    for b in range(B):
        var base = b * C
        var mx = _z(logits[base])
        for c in range(1, C):
            if _z(logits[base + c]) > mx:
                mx = _z(logits[base + c])
        var s = Float32(0.0)
        for cc in range(C):
            var c = C - 1 - cc if alt else cc
            s = _a(s, _z(identical_exp(_s(logits[base + c], mx))))
        var ls = _z(identical_log(s))
        var y = labels[b]
        loss.append(_s(ls, _s(logits[base + y], mx)))
        for c in range(C):
            var p = _d(_z(identical_exp(_s(logits[base + c], mx))), s)
            if c == y:
                p = _s(p, Float32(1.0))
            grad.append(_m(p, scale))
    for i in range(len(loss)):
        grad.append(loss[i])
    return grad^


# ---------------------------------------------------------------- 5506
def o_adam(
    p: List[Float32], g: List[Float32], m0: List[Float32], v0: List[Float32],
    b1: Float32, b2: Float32, eps: Float32, wd: Float32, step_scale: Float32, bc2_sqrt: Float32, alt: Bool,
) -> List[Float32]:
    """[p', m', v'] of torch.optim.Adam (L2 weight decay); step_scale =
    lr / (1 - b1^t) and bc2_sqrt = sqrt(1 - b2^t) are the caller's scalars."""
    var n = len(p)
    var ps = List[Float32]()
    var ms = List[Float32]()
    var vs = List[Float32]()
    for i in range(n):
        var gi = _z(g[i])
        if wd != Float32(0.0):
            gi = _f(wd, p[i], gi)
        var m = _f(b1, m0[i], _m(_s(Float32(1.0), b1), gi))
        var v = _f(b2, v0[i], _m(_m(_s(Float32(1.0), b2), gi), gi))
        var den: Float32
        if alt:
            den = _a(_z(identical_sqrt(_d(v, _m(bc2_sqrt, bc2_sqrt)))), eps)
        else:
            den = _a(_d(_z(identical_sqrt(v)), bc2_sqrt), eps)
        ps.append(_f(-step_scale, _d(m, den), p[i]))
        ms.append(m)
        vs.append(v)
    for i in range(n):
        ps.append(ms[i])
    for i in range(n):
        ps.append(vs[i])
    return ps^


# ---------------------------------------------------------------- 5507
def o_lerp(s: Float32, e: Float32, w: Float32, alt: Bool) -> Float32:
    if alt or w < Float32(0.5):
        return _f(w, _s(e, s), s)
    return _s(e, _m(_s(e, s), _s(Float32(1.0), w)))


def o_adamax(
    p: List[Float32], g: List[Float32], m0: List[Float32], u0: List[Float32],
    b1: Float32, b2: Float32, eps: Float32, wd: Float32, clr: Float32, alt: Bool,
) -> List[Float32]:
    """[p', m', u'] of torch.optim.Adamax; clr = lr / (1 - b1^t)."""
    var n = len(p)
    var ps = List[Float32]()
    var ms = List[Float32]()
    var us = List[Float32]()
    for i in range(n):
        var gi = _z(g[i])
        if wd != Float32(0.0):
            gi = _f(wd, p[i], gi)
        var m = o_lerp(_z(m0[i]), gi, _s(Float32(1.0), b1), alt)
        var ub = _m(b2, u0[i])
        var ga = _a(abs(gi), eps)
        var u = ub if ub > ga else ga
        ps.append(_f(-clr, _d(m, u), p[i]))
        ms.append(m)
        us.append(u)
    for i in range(n):
        ps.append(ms[i])
    for i in range(n):
        ps.append(us[i])
    return ps^


# ---------------------------------------------------------------- 5508
def o_binary_logloss(p: List[Float32], y: List[Float32], alt: Bool) -> List[Float32]:
    """[delta (n), row loss (n)] of sklearn's binary log loss, one output
    per row; the clip is float32's machine epsilon."""
    comptime EPS32 = Float32(1.1920928955078125e-07)
    var d = List[Float32]()
    var l = List[Float32]()
    for i in range(len(p)):
        var pi = _z(p[i])
        var yi = _z(y[i])
        d.append(_s(pi, yi))
        var pc = pi
        if not alt:
            if pc < EPS32:
                pc = EPS32
            if pc > _s(Float32(1.0), EPS32):
                pc = _s(Float32(1.0), EPS32)
        var acc = Float32(0.0)
        if yi != Float32(0.0):
            acc = _f(yi, _z(identical_log(pc)), acc)
        var ny = _s(Float32(1.0), yi)
        if ny != Float32(0.0):
            acc = _f(ny, _z(identical_log(_s(Float32(1.0), pc))), acc)
        l.append(-acc)
    for i in range(len(l)):
        d.append(l[i])
    return d^


# ---------------------------------------------------------------- 5509
def _below(mut state: UInt64, n: Int) -> Int:
    var un = UInt64(n)
    var threshold = (UInt64(0) - un) % un
    while True:
        var r = splitmix64_next(state)
        if r >= threshold:
            return Int(r % un)


def o_shuffle(n: Int, seed: UInt64, alt: Bool) -> List[Float32]:
    """The permutation of 0..n-1 (as floats) the MLP trainer draws."""
    var perm = List[Float32]()
    for i in range(n):
        perm.append(Float32(i))
    var state = seed
    if alt:
        for i in range(1, n):
            var j = _below(state, i + 1)
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
    else:
        var i = n - 1
        while i > 0:
            var j = _below(state, i + 1)
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
            i -= 1
    return perm^


# ---------------------------------------------------------------- 5510
def o_stl_rwts(y: List[Float32], fit: List[Float32], alt: Bool) -> List[Float32]:
    """statsmodels _rwts: bisquare weights of |y - fit| at 6 MAD."""
    var n = len(y)
    var r = List[Float32]()
    for i in range(n):
        r.append(abs(_s(y[i], fit[i])))
    var srt = r.copy()
    for i in range(1, n):   # insertion sort: the order statistics are values
        var v = srt[i]
        var j = i - 1
        while j >= 0 and srt[j] > v:
            srt[j + 1] = srt[j]
            j -= 1
        srt[j + 1] = v
    var m0 = n // 2
    var m1 = n - m0 - 1
    var cmad: Float32
    if alt:
        cmad = _a(_m(Float32(3.0), srt[m0]), _m(Float32(3.0), srt[m1]))
    else:
        cmad = _m(Float32(3.0), _a(srt[m0], srt[m1]))
    var out = List[Float32]()
    if cmad == Float32(0.0):
        for _ in range(n):
            out.append(Float32(1.0))
        return out^
    var c9 = _m(Float32(0.999), cmad)
    var c1 = _m(Float32(0.001), cmad)
    for i in range(n):
        if r[i] <= c1:
            out.append(Float32(1.0))
        elif r[i] <= c9:
            var q = _d(r[i], cmad)
            var u = _s(Float32(1.0), _m(q, q))
            out.append(_m(u, u))
        else:
            out.append(Float32(0.0))
    return out^


# ---------------------------------------------------------------- 5511
def o_colscale(X: List[Float32], R: Int, M: Int, alt: Bool) -> List[Float32]:
    """[X scaled (R M), scales (M)]: each column times 2^-e, e the exponent
    of its largest magnitude (found by exact doubling / halving, not by the
    bit trick the lane uses); a zero column keeps scale 1."""
    var out = List[Float32](length=R * M, fill=Float32(0.0))
    var scales = List[Float32]()
    for c in range(M):
        var mx = Float32(0.0)
        for r in range(R):
            var v = abs(_z(X[r * M + c]))
            if v > mx:
                mx = v
        var s = Float32(1.0)
        if mx > Float32(0.0):
            if alt:
                s = _d(Float32(1.0), mx)
            else:
                var t = mx
                while t >= Float32(2.0):
                    t = t * Float32(0.5)
                    s = s * Float32(0.5)
                while t < Float32(1.0):
                    t = t * Float32(2.0)
                    s = s * Float32(2.0)
        for r in range(R):
            out[r * M + c] = _m(X[r * M + c], s)
        scales.append(s)
    for i in range(M):
        out.append(scales[i])
    return out^


# ---------------------------------------------------------------- 5512
def o_cholsolve(G0: List[Float32], B0: List[Float32], m: Int, K: Int, alt: Bool) -> List[Float32]:
    """[G (m m, lower factor in place), X (m K), status (1)]."""
    var G = List[Float32]()
    for i in range(len(G0)):
        G.append(_z(G0[i]))
    var X = List[Float32]()
    for i in range(len(B0)):
        X.append(_z(B0[i]))
    var status = Float32(0.0)
    var ok = True
    for j in range(m):
        var d = G[j * m + j]
        for kk in range(j):
            var k = j - 1 - kk if alt else kk
            d = _s(d, _m(G[j * m + k], G[j * m + k]))
        if not (d > Float32(0.0)):
            status = Float32(1 + j)
            ok = False
            break
        var ljj = _z(identical_sqrt(d))
        G[j * m + j] = ljj
        for i in range(j + 1, m):
            var v = G[i * m + j]
            for kk in range(j):
                var k = j - 1 - kk if alt else kk
                v = _s(v, _m(G[i * m + k], G[j * m + k]))
            G[i * m + j] = _d(v, ljj)
    if ok:
        for c in range(K):
            for i in range(m):
                var v = X[i * K + c]
                for kk in range(i):
                    var k = i - 1 - kk if alt else kk
                    v = _s(v, _m(G[i * m + k], X[k * K + c]))
                X[i * K + c] = _d(v, G[i * m + i])
            var i = m - 1
            while i >= 0:
                var v = X[i * K + c]
                for kk in range(i + 1, m):
                    var k = m + i - kk if alt else kk
                    v = _s(v, _m(G[k * m + i], X[k * K + c]))
                X[i * K + c] = _d(v, G[i * m + i])
                i -= 1
    var out = G^
    for i in range(len(X)):
        out.append(X[i])
    out.append(status)
    return out^


# ---------------------------------------------------------------- 5513
def o_nm_quantized(x0: List[Float32], lower: List[Float32], upper: List[Float32], centre: List[Float32],
                   init_step: Float32, zero_pert: Float32, max_iter: Int, tol_std: Float32, alt: Bool) -> List[Float32]:
    """statsforecast's Nelder-Mead on the quantized objective of
    seams_check (`_quant_obj`), restated: [best point (n), iterations].
    Ties in the simplex order: pinned, the lower vertex index first."""
    var n = len(x0)
    var nf = Float32(n)
    var gamma = _a(Float32(1.0), _d(Float32(2.0), nf))
    var rho = _s(Float32(0.75), _d(Float32(1.0), _m(Float32(2.0), nf)))
    var sigma = _s(Float32(1.0), _d(Float32(1.0), nf))
    var S = List[List[Float32]]()
    for _ in range(n + 1):
        var v = List[Float32]()
        for j in range(n):
            v.append(o_clamp(_z(x0[j]), lower[j], upper[j]))
        S.append(v^)
    for i in range(n):
        var v = S[i][i]
        if v == Float32(0.0):
            v = zero_pert
        else:
            v = _m(v, _a(Float32(1.0), init_step))
        S[i][i] = o_clamp(v, lower[i], upper[i])
    var fs = List[Float32]()
    for i in range(n + 1):
        fs.append(quant_obj(S[i], centre))
    var it = 0
    var best = 0
    while it < max_iter:
        var order = List[Int]()
        for i in range(n + 1):
            var pos = len(order)
            for q in range(len(order)):
                var before = fs[i] < fs[order[q]] if not alt else fs[i] <= fs[order[q]]
                if before:
                    pos = q
                    break
            order.insert(pos, i)
        best = order[0]
        var worst = order[n]
        var second = order[n - 1]
        var mean = Float32(0.0)
        for i in range(n + 1):
            mean = _a(mean, fs[i])
        mean = _d(mean, Float32(n + 1))
        var ss = Float32(0.0)
        for i in range(n + 1):
            var d = _s(fs[i], mean)
            ss = _f(d, d, ss)
        if _z(identical_sqrt(_d(ss, Float32(n + 1)))) < tol_std:
            break
        var xo = List[Float32]()
        for j in range(n):
            var s = Float32(0.0)
            for i in range(n + 1):
                s = _a(s, S[i][j])
            xo.append(_d(_s(s, S[worst][j]), nf))
        var xr = List[Float32]()
        for j in range(n):
            xr.append(o_clamp(_a(xo[j], _s(xo[j], S[worst][j])), lower[j], upper[j]))
        var fr = quant_obj(xr, centre)
        if fs[best] <= fr and fr < fs[second]:
            S[worst] = xr.copy()
            fs[worst] = fr
            it += 1
            continue
        if fr < fs[best]:
            var xe = List[Float32]()
            for j in range(n):
                xe.append(o_clamp(_f(gamma, _s(xr[j], xo[j]), xo[j]), lower[j], upper[j]))
            var fe = quant_obj(xe, centre)
            if fe < fr:
                S[worst] = xe.copy()
                fs[worst] = fe
            else:
                S[worst] = xr.copy()
                fs[worst] = fr
            it += 1
            continue
        var accepted = False
        var xt = List[Float32]()
        if fs[second] <= fr and fr < fs[worst]:
            for j in range(n):
                xt.append(o_clamp(_f(rho, _s(xr[j], xo[j]), xo[j]), lower[j], upper[j]))
            var fc = quant_obj(xt, centre)
            if fc <= fr:
                S[worst] = xt.copy()
                fs[worst] = fc
                accepted = True
        else:
            for j in range(n):
                xt.append(o_clamp(_s(xo[j], _m(rho, _s(xr[j], xo[j]))), lower[j], upper[j]))
            var fc = quant_obj(xt, centre)
            if fc < fs[worst]:
                S[worst] = xt.copy()
                fs[worst] = fc
                accepted = True
        if not accepted:
            for i in range(n + 1):
                if i == best:
                    continue
                for j in range(n):
                    var b = S[best][j]
                    S[i][j] = o_clamp(_f(sigma, _s(S[i][j], b), b), lower[j], upper[j])
                fs[i] = quant_obj(S[i], centre)
        it += 1
    var out = List[Float32]()
    for j in range(n):
        out.append(S[best][j])
    out.append(Float32(it + 1))
    return out^


def o_clamp(v: Float32, lo: Float32, hi: Float32) -> Float32:
    var r = v if v > lo else lo
    return r if r < hi else hi


def quant_obj(x: List[Float32], centre: List[Float32]) -> Float32:
    """sum (x_j - c_j)^2 rounded down to a multiple of 1/4: a staircase, so
    simplex values tie often. Shared by the oracle and the lane's call."""
    var s = Float32(0.0)
    for j in range(len(x)):
        var d = _s(x[j], centre[j])
        s = _f(d, d, s)
    return _m(Float32(Int(_m(s, Float32(4.0)))), Float32(0.25))


# ---------------------------------------------------------------- 5514
def o_moe_route(x: List[Float32], Wg: List[Float32], T: Int, D: Int, E: Int, k: Int, renorm: Bool, alt: Bool) -> List[Float32]:
    """[logits (T E), selected (T k), weights (T k)] of Mixtral's router:
    softmax over experts, top k by probability, ties to the lower index."""
    var logits = List[Float32]()
    var sel = List[Float32]()
    var wts = List[Float32]()
    for t in range(T):
        var row = List[Float32]()
        for e in range(E):
            var s = Float32(0.0)
            for d in range(D):
                s = _f(x[t * D + d], Wg[e * D + d], s)
            row.append(s)
            logits.append(s)
        var mx = row[0]
        for e in range(1, E):
            if row[e] > mx:
                mx = row[e]
        var pr = List[Float32]()
        var zsum = Float32(0.0)
        for e in range(E):
            var v = _z(identical_exp(_s(row[e], mx)))
            pr.append(v)
            zsum = _a(zsum, v)
        for e in range(E):
            pr[e] = _d(pr[e], zsum)
        var taken = List[Bool](length=E, fill=False)
        var tw = List[Float32]()
        var tot = Float32(0.0)
        for _ in range(k):
            var best = -1
            for e in range(E):
                if taken[e]:
                    continue
                if best < 0 or pr[e] > pr[best] or (alt and pr[e] == pr[best]):
                    best = e
            taken[best] = True
            sel.append(Float32(best))
            tw.append(pr[best])
            tot = _a(tot, pr[best])
        for j in range(k):
            wts.append(_d(tw[j], tot) if renorm else tw[j])
    for i in range(len(sel)):
        logits.append(sel[i])
    for i in range(len(wts)):
        logits.append(wts[i])
    return logits^


# ---------------------------------------------------------------- 5515
def o_layer_norm(x: List[Float32], w: List[Float32], b: List[Float32], M: Int, D: Int, eps: Float32, alt: Bool) -> List[Float32]:
    """[y (M D), mean (M), rstd (M)] of torch.nn.functional.layer_norm."""
    var y = List[Float32]()
    var means = List[Float32]()
    var rstds = List[Float32]()
    for r in range(M):
        var base = r * D
        var s = Float32(0.0)
        for cc in range(D):
            var c = D - 1 - cc if alt else cc
            s = _a(s, x[base + c])
        var mean = _d(s, Float32(D))
        var q = Float32(0.0)
        for c in range(D):
            var d = _s(x[base + c], mean)
            q = _f(d, d, q)
        var rstd = _z(identical_rsqrt(_a(_d(q, Float32(D)), eps)))
        for c in range(D):
            y.append(_a(_m(_m(_s(x[base + c], mean), rstd), w[c]), b[c]))
        means.append(mean)
        rstds.append(rstd)
    for i in range(M):
        y.append(means[i])
    for i in range(M):
        y.append(rstds[i])
    return y^


# ---------------------------------------------------------------- 5516
def _ses_step(alpha: Float32, x: Float32, f: Float32, alt: Bool) -> Float32:
    if alt:
        return _a(f, _m(alpha, _s(x, f)))
    return _f(alpha, x, _m(_s(Float32(1.0), alpha), f))


def _ses_forecast(x: List[Float32], alpha: Float32, alt: Bool) -> Float32:
    var n = len(x)
    if n == 1:
        return x[0]
    var f = x[0]
    for i in range(1, n):
        f = _ses_step(alpha, x[i - 1], f, alt)
    return _ses_step(alpha, x[n - 1], f, alt)


def _ses_sse(alpha: Float32, x: List[Float32], alt: Bool) -> Float32:
    var n = len(x)
    if n < 2:
        return Float32(0.0)
    var f = x[0]
    var sse = Float32(0.0)
    for i in range(1, n):
        f = _ses_step(alpha, x[i - 1], f, alt)
        var e = _s(x[i], f)
        sse = _f(e, e, sse)
    return sse


def _golden(x: List[Float32], lo: Float32, hi: Float32, alt: Bool) -> Float32:
    """statsforecast golden_section_ses on [lo, hi], float32, at most 200
    steps, stopping at an exact tie."""
    var gr = _d(_a(_z(identical_sqrt(Float32(5.0))), Float32(1.0)), Float32(2.0))
    var a = lo
    var b = hi
    var c = _s(b, _d(_s(b, a), gr))
    var d = _a(a, _d(_s(b, a), gr))
    var fc = _ses_sse(c, x, alt)
    var fd = _ses_sse(d, x, alt)
    var it = 0
    while abs(_s(b, a)) >= Float32(1e-12) and it < 200:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = _s(b, _d(_s(b, a), gr))
            fc = _ses_sse(c, x, alt)
        elif fd < fc:
            a = c
            c = d
            fc = fd
            d = _a(a, _d(_s(b, a), gr))
            fd = _ses_sse(d, x, alt)
        else:
            break
        it += 1
    return _d(_a(b, a), Float32(2.0))


def o_croston(y: List[Float32], B: Int, n: Int, variant: Int, alt: Bool) -> List[Float32]:
    """The mean forecast per series: 0 classic, 1 optimized, 2 SBA."""
    var out = List[Float32]()
    for s in range(B):
        var dem = List[Float32]()
        var itv = List[Float32]()
        var prev = 0
        for i in range(n):
            var v = _z(y[s * n + i])
            if v > Float32(0.0):
                dem.append(v)
            if v != Float32(0.0):
                itv.append(Float32(i + 1 - prev))
                prev = i + 1
        var mean: Float32
        if len(dem) == 0:
            mean = _z(y[s * n + n - 1])
        else:
            var ad = Float32(0.1)
            var ai = Float32(0.1)
            if variant == 1:
                ad = _golden(dem, Float32(0.1), Float32(0.3), alt)
                ai = _golden(itv, Float32(0.1), Float32(0.3), alt)
            var ydp = _ses_forecast(dem, ad, alt)
            var yip = _ses_forecast(itv, ai, alt)
            mean = _d(ydp, yip) if yip != Float32(0.0) else ydp
            if variant == 2:
                mean = _m(mean, Float32(0.95))
        out.append(mean)
    return out^
