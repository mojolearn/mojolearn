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
       ascending after the reverse sweep (IDENTICAL, K > 512: the
       blocked order of sequence/recurrent.mojo)                     alt: per-step, steps descending
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
  5517 ETS seasonal update: s + gamma (t - s), one fma             alt: (1 - gamma) s + gamma t
  5518 ETS decomposition moving average: taps ascending, 0.5 x
       ends, ONE division by m                                      alt: every tap times w / m
  5536 RMSprop centered variance: v - ga*ga, two roundings          alt: fma(-ga, ga, v)
  5537 Adagrad accumulator: sum + g*g, one fma                      alt: g*g rounded, then the add
  5538 Lion momentum: b2 m + (1 - b2) g, one fma                    alt: both products rounded
  5539 LAMB trust ratio: sqrt(sum p^2) / sqrt(sum u^2)              alt: sqrt(sum p^2 / sum u^2)
  5540 LR schedulers: exact rational, ONE rounding to float32       alt: float64 closed form, rounded
       (python/mojolearn/_x_sequence_sched.py; its own driver sequence/checks/sched_check.py)
  5541 Theta level: alpha y + (1 - alpha) level, one fma            alt: level + alpha (y - level)
  5542 GARCH variance recursion: each term one fma, in order        alt: product rounded, then the add
  5543 Prophet Fourier argument: (2 pi i) frac                      alt: 2 pi (i frac)
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
    identical_cos,
    identical_sigmoid,
    identical_sin,
    identical_sqrt,
    identical_tanh,
)
from checks.fixture_rng import splitmix64_next
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined


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


# ---------------------------------------------------------------- 5544
def o_gemm_split(A: List[Float32], B: List[Float32], C0: List[Float32], M: Int, N: Int, K: Int) -> List[Float32]:
    """5544's alternative: `o_gemm`'s cells with the product rounded before
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
def _o_wgrad_block(K: Int) -> Int:
    """Restated (sequence/recurrent.mojo `wgrad_block`): K itself when the
    blocked order is off, else the smallest power of two R >= 512 with
    R R >= K."""
    # S03 independent numerical-profile restatement. NOT TESTED — NOT
    # COMPILED — NOT MEASURED; no production default or acceptance claim.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NEURAL_S03_WGRAD_LEAF128"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]():
        return 128
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or is_defined["MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]():
        return K
    var r = 512
    while r * r < K:
        r *= 2
    return r


def o_bptt_dw(dgx: List[Float32], x: List[Float32], T: Int, B: Int, GH: Int, D: Int, alt: Bool) -> List[Float32]:
    """dW_ih [GH, D] = sum over rows r = s B + b of dgx[r, g] x[r, d]:
    pinned, r ascending in one fold (under IDENTICAL since nr-small D3:
    the blocked order, blocks of `_o_wgrad_block(T B)` rows each folded
    from zero, the partials added ascending from zero; one block is the one
    fold); alt, the steps descending (the order a per-step accumulation
    during the reverse sweep folds them)."""
    var out = List[Float32](capacity=GH * D)
    var K = T * B
    var KB = _o_wgrad_block(K)
    if not alt and KB < K:
        for g in range(GH):
            for d in range(D):
                var tot = Float32(0.0)
                var lo = 0
                while lo < K:
                    var part = Float32(0.0)
                    for r in range(lo, min(lo + KB, K)):
                        part = _f(dgx[r * GH + g], x[r * D + d], part)
                    tot = _a(tot, part)
                    lo += KB
                out.append(tot)
        return out^
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
# V02 oracle control — NOT TESTED — NOT COMPILED — NOT MEASURED.
# Independent adjacent-pair implementation, never imports sequence/layernorm.
comptime _O_LN_TREE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NEURAL_V02_LN_ADJACENT_TREE"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()


def _o_ln_tree(var values: List[Float32]) -> Float32:
    """Pairs (0,1),(2,3),... per level; the unpaired last value carries."""
    var live = len(values)
    if live == 0:
        return Float32(0.0)
    while live > 1:
        var out = 0
        var i = 0
        while i < live:
            values[out] = _a(values[i], values[i + 1]) if i + 1 < live else values[i]
            out += 1
            i += 2
        live = out
    return values[0]


def o_layer_norm(x: List[Float32], w: List[Float32], b: List[Float32], M: Int, D: Int, eps: Float32, alt: Bool) -> List[Float32]:
    """[y (M D), mean (M), rstd (M)] of torch.nn.functional.layer_norm."""
    var y = List[Float32]()
    var means = List[Float32]()
    var rstds = List[Float32]()
    for r in range(M):
        var base = r * D
        var s = Float32(0.0)
        comptime if _O_LN_TREE:
            var terms = List[Float32]()
            for cc in range(D):
                var c = D - 1 - cc if alt else cc
                terms.append(_a(Float32(0.0), x[base + c]))
            s = _o_ln_tree(terms^)
        else:
            for cc in range(D):
                var c = D - 1 - cc if alt else cc
                s = _a(s, x[base + c])
        var mean = _d(s, Float32(D))
        var q = Float32(0.0)
        comptime if _O_LN_TREE:
            var terms = List[Float32]()
            for c in range(D):
                var d = _s(x[base + c], mean)
                terms.append(_f(d, d, Float32(0.0)))
            q = _o_ln_tree(terms^)
        else:
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


def o_layer_norm_tree_backward(
    x: List[Float32], w: List[Float32], dy: List[Float32],
    M: Int, D: Int, eps: Float32,
) raises -> List[Float32]:
    """V02 independent dx,dweight,dbias reference; source only, never run.
    This explicitly requires the new profile, avoiding a mislabeled oracle
    that silently returns candidate words for a baseline binary.
    """
    comptime if not _O_LN_TREE:
        raise Error("LayerNorm tree backward oracle requires the V02 profile")
    var bias = List[Float32](length=D, fill=Float32(0.0))
    var fwd = o_layer_norm(x, w, bias, M, D, eps, False)
    var result = List[Float32]()
    for r in range(M):
        var mean = fwd[M * D + r]
        var rs = fwd[M * D + M + r]
        var gs = List[Float32]()
        var gxs = List[Float32]()
        for c in range(D):
            var g = _m(dy[r * D + c], w[c])
            var xh = _m(_s(x[r * D + c], mean), rs)
            gs.append(_a(Float32(0.0), g))
            gxs.append(_f(g, xh, Float32(0.0)))
        var mg = _d(_o_ln_tree(gs^), Float32(D))
        var mgx = _d(_o_ln_tree(gxs^), Float32(D))
        for c in range(D):
            var g = _m(dy[r * D + c], w[c])
            var xh = _m(_s(x[r * D + c], mean), rs)
            result.append(_m(rs, _s(_s(g, mg), _m(xh, mgx))))
    var db = List[Float32]()
    for c in range(D):
        var ws = List[Float32]()
        var bs = List[Float32]()
        for r in range(M):
            var xh = _m(_s(x[r * D + c], fwd[M * D + r]), fwd[M * D + M + r])
            ws.append(_f(dy[r * D + c], xh, Float32(0.0)))
            bs.append(_a(Float32(0.0), dy[r * D + c]))
        result.append(_o_ln_tree(ws^))
        db.append(_o_ln_tree(bs^))
    result.extend(db^)
    return result^


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


# ---------------------------------------------------------------- 5517 ETS seasonal update
def _ets_seas(s_old: Float32, gamma: Float32, t: Float32, alt: Bool) -> Float32:
    """statsforecast Update: s[0] = old_s[m-1] + gamma (t - old_s[m-1])."""
    if alt:
        return _a(_m(_s(Float32(1.0), gamma), s_old), _m(gamma, t))
    return _f(gamma, _s(t, s_old), s_old)


def o_ets_calc(y: List[Float32], B: Int, n: Int, err: Int, trend: Bool, season: Int, m: Int,
               par: List[Float32], alt: Bool) -> List[Float32]:
    """statsforecast Calc per series from its initial state (the layout of
    sequence OP_ETS_LIK): par [B, 6 + m] = alpha, beta, gamma, phi, l0, b0,
    s[0..m-1] (s[m-1] the oldest, used first); out [B, 3 + m] = lik, the
    final level, the final trend, the final s[0..m-1]. The seasonal vector
    is shifted explicitly (s[1:] = old_s[:-1]) as in the reference.
    err 0 A, 1 M; season 0 N, 1 A, 2 M."""
    var out = List[Float32]()
    var mm = m if season != 0 else 1
    for sr in range(B):
        var pr = sr * (6 + mm)
        var alpha = par[pr]
        var beta = par[pr + 1]
        var gamma = par[pr + 2]
        var phi = par[pr + 3]
        var l = _z(par[pr + 4])
        var b = _z(par[pr + 5]) if trend else Float32(0.0)
        var s = List[Float32]()
        for j in range(mm):
            s.append(_z(par[pr + 6 + j]))
        var sse = Float32(0.0)
        var slog = Float32(0.0)
        var ba = _d(beta, alpha)
        for i in range(n):
            var yi = _z(y[sr * n + i])
            var old_l = l
            var old_b = b
            var phib = _m(phi, old_b) if trend else Float32(0.0)
            var q = _a(old_l, phib)
            var so = s[mm - 1] if season != 0 else Float32(0.0)
            var f0 = q
            if season == 1:
                f0 = _a(q, so)
            elif season == 2:
                f0 = _m(q, so)
            var e: Float32
            if err == 0:
                e = _s(yi, f0)
            else:
                var fd = _a(f0, Float32(1e-10)) if abs(f0) < Float32(1e-10) else f0
                e = _d(_s(yi, f0), fd)
            var p = yi
            if season == 1:
                p = _s(yi, so)
            elif season == 2:
                p = Float32(1e10) if abs(so) < Float32(1e-10) else _d(yi, so)
            l = _f(alpha, _s(p, q), q)
            if trend:
                b = _f(ba, _s(_s(l, old_l), phib), phib)
            if season != 0:
                var t: Float32
                if season == 1:
                    t = _s(yi, q)
                else:
                    t = Float32(1e10) if abs(q) < Float32(1e-10) else _d(yi, q)
                var s0 = _ets_seas(so, gamma, t, alt)
                var j = mm - 1
                while j > 0:
                    s[j] = s[j - 1]
                    j -= 1
                s[0] = s0
            sse = _f(e, e, sse)
            var v = abs(f0)
            slog = _a(slog, _z(identical_log(v)) if v > Float32(0.0) else _z(identical_log(_a(v, Float32(1e-8)))))
        var lik: Float32
        if sse > Float32(0.0):
            lik = _m(Float32(n), _z(identical_log(sse)))
        else:
            lik = _m(Float32(n), _z(identical_log(_a(sse, Float32(1e-8)))))
        if err == 1:
            lik = _f(Float32(2.0), slog, lik)
        out.append(lik)
        out.append(l)
        out.append(b)
        for j in range(mm):
            out.append(s[j] if season != 0 else Float32(0.0))
    return out^


# ---------------------------------------------------------------- 5518 ETS initstate (decomposition)
def _ets_ma(y: List[Float32], base: Int, i: Int, m: Int, alt: Bool) -> Float32:
    """statsmodels' centred moving average at i: filter [0.5, 1, ..., 1, 0.5] / m
    (even m) or ones / m (odd m). Pinned: taps ascending, the half-weight ends
    as 0.5 x, ONE division by m. alt: every tap times its weight w / m."""
    var half = m // 2
    var even = m % 2 == 0
    if alt:
        var w = _d(Float32(1.0), Float32(m))
        var hw = _d(Float32(0.5), Float32(m))
        var acc = _m(hw if even else w, y[base + i - half])
        for k in range(1, m):
            acc = _f(w, y[base + i - half + k], acc)
        if even:
            acc = _f(hw, y[base + i + half], acc)
        return acc
    var acc: Float32
    if even:
        acc = _m(Float32(0.5), y[base + i - half])
        for k in range(1, m):
            acc = _a(acc, y[base + i - half + k])
        acc = _f(Float32(0.5), y[base + i + half], acc)
    else:
        acc = _z(y[base + i - half])
        for k in range(1, m):
            acc = _a(acc, y[base + i - half + k])
    return _d(acc, Float32(m))


def o_ets_init(y: List[Float32], B: Int, n: Int, trend: Bool, season: Int, m: Int, alt: Bool) -> List[Float32]:
    """statsforecast initstate for n >= 3m (seasonal_decompose): out [B, 1 + m]
    = l0, b0 (0 without a trend), the m - 1 free seasonal states
    seasonal[m-1], ..., seasonal[1]. season 1 A, 2 M."""
    var out = List[Float32]()
    var half = m // 2
    for sr in range(B):
        var base = sr * n
        # period averages of the detrended series where the trend exists
        var pa = List[Float32]()
        for p in range(m):
            var acc = Float32(0.0)
            var cnt = 0
            for i in range(half, n - half):
                if i % m != p:
                    continue
                var tr = _ets_ma(y, base, i, m, alt)
                var yi = _z(y[base + i])
                acc = _a(acc, _s(yi, tr) if season == 1 else _d(yi, tr))
                cnt += 1
            pa.append(_d(acc, Float32(cnt)))
        var mean = Float32(0.0)
        for p in range(m):
            mean = _a(mean, pa[p])
        mean = _d(mean, Float32(m))
        for p in range(m):
            pa[p] = _s(pa[p], mean) if season == 1 else _d(pa[p], mean)
        var init = List[Float32]()
        for k in range(1, m):
            init.append(pa[m - k])
        if season == 2:
            var sm = Float32(0.0)
            for k in range(m - 1):
                init[k] = init[k] if init[k] > Float32(1e-2) else Float32(1e-2)
                sm = _a(sm, init[k])
            if sm > Float32(m):
                var den = Float32(0.0)
                for k in range(m - 1):
                    den = _a(den, _a(init[k], Float32(1e-2)))
                for k in range(m - 1):
                    init[k] = _d(init[k], den)
        var mx = 2 * m if 2 * m > 10 else 10
        var maxn = mx if mx < n else n
        var ysa = List[Float32]()
        for i in range(maxn):
            var sv = pa[i % m]
            var yi = _z(y[base + i])
            ysa.append(_s(yi, sv) if season == 1 else _d(yi, sv if sv > Float32(1e-2) else Float32(1e-2)))
        var sy = Float32(0.0)
        for i in range(maxn):
            sy = _a(sy, ysa[i])
        var ybar = _d(sy, Float32(maxn))
        var l0 = ybar
        var b0 = Float32(0.0)
        if trend:
            # the least-squares line through (t, ysa), t = 1..maxn, centred sums
            var tbar = _d(Float32(maxn + 1), Float32(2.0))
            var sxy = Float32(0.0)
            var sxx = Float32(0.0)
            for i in range(maxn):
                var dt = _s(Float32(i + 1), tbar)
                sxy = _f(dt, _s(ysa[i], ybar), sxy)
                sxx = _f(dt, dt, sxx)
            b0 = _d(sxy, sxx) if sxx > Float32(0.0) else Float32(0.0)
            l0 = _s(ybar, _m(b0, tbar))
            if abs(_a(l0, b0)) < Float32(1e-8):
                l0 = _m(l0, Float32(1.001))
                b0 = _m(b0, Float32(0.999))
        out.append(l0)
        out.append(b0)
        for k in range(m - 1):
            out.append(init[k])
    return out^


# ---------------------------------------------------------------- 5507 (NAdam, Adafactor)
def o_nadam(
    p: List[Float32], g: List[Float32], m0: List[Float32], v0: List[Float32],
    b1: Float32, b2: Float32, eps: Float32, bc2: Float32, c1: Float32, c2: Float32, alt: Bool,
) -> List[Float32]:
    """[p', m', v'] of torch.optim.NAdam (no weight decay): m.lerp_(g, 1 - b1),
    v = b2 v + (1 - b2) g^2, den = sqrt(v / bc2) + eps, p += c1 g / den, then
    p += c2 m / den; bc2, c1, c2 are the caller's host scalars."""
    var n = len(p)
    var ps = List[Float32]()
    var ms = List[Float32]()
    var vs = List[Float32]()
    for i in range(n):
        var gi = _z(g[i])
        var m = o_lerp(_z(m0[i]), gi, _s(Float32(1.0), b1), alt)
        var v = _f(b2, v0[i], _m(_m(_s(Float32(1.0), b2), gi), gi))
        var den = _a(_z(identical_sqrt(_d(v, bc2))), eps)
        var p1 = _f(c1, _d(gi, den), p[i])
        ps.append(_f(c2, _d(m, den), p1))
        ms.append(m)
        vs.append(v)
    for i in range(n):
        ps.append(ms[i])
    for i in range(n):
        ps.append(vs[i])
    return ps^


def o_af_vec(g: List[Float32], v0: List[Float32], w: Float32, eps1sq: Float32, alt: Bool) -> List[Float32]:
    """[v', u] of torch's Adafactor on a vector: v.lerp_(g^2, w), u = g /
    sqrt(max(v, eps1^2)) (torch: grad * rsqrt of the clamped estimate)."""
    var n = len(g)
    var vs = List[Float32]()
    var us = List[Float32]()
    for i in range(n):
        var gi = _z(g[i])
        var v = o_lerp(_z(v0[i]), _m(gi, gi), w, alt)
        vs.append(v)
        var c = max(v, eps1sq)
        us.append(_m(_z(identical_rsqrt(c)), gi))
    for i in range(n):
        vs.append(us[i])
    return vs^


# ---------------------------------------------------------------- 5536
def o_rmsprop(
    p: List[Float32], g: List[Float32], buf0: List[Float32], v0: List[Float32], ga0: List[Float32],
    lr: Float32, alpha: Float32, eps: Float32, wd: Float32, mu: Float32, alt: Bool,
) -> List[Float32]:
    """[p', buf', v', gavg'] of torch.optim.RMSprop, centered, with momentum
    mu > 0 and L2 weight decay. torch's `square_avg.addcmul(grad_avg,
    grad_avg, value=-1)` is v - ga*ga, two roundings (alt: one fma). A
    negative difference (rounding only) is taken as 0, never a NaN sqrt."""
    var n = len(p)
    var ps = List[Float32]()
    var bs = List[Float32]()
    var vs = List[Float32]()
    var gs = List[Float32]()
    var oma = _s(Float32(1.0), alpha)
    for i in range(n):
        var gi = _z(g[i])
        if wd != Float32(0.0):
            gi = _f(wd, p[i], gi)
        var v = _f(alpha, v0[i], _m(_m(oma, gi), gi))
        var ga = _f(alpha, ga0[i], _m(oma, gi))
        var var_: Float32
        if alt:
            var_ = _f(-ga, ga, v)
        else:
            var_ = _s(v, _m(ga, ga))
        if var_ < Float32(0.0):
            var_ = Float32(0.0)
        var avg = _a(_z(identical_sqrt(var_)), eps)
        var b = _f(mu, buf0[i], _d(gi, avg))
        ps.append(_f(-lr, b, p[i]))
        bs.append(b)
        vs.append(v)
        gs.append(ga)
    for i in range(n):
        ps.append(bs[i])
    for i in range(n):
        ps.append(vs[i])
    for i in range(n):
        ps.append(gs[i])
    return ps^


# ---------------------------------------------------------------- 5537
def o_adagrad(p: List[Float32], g: List[Float32], s0: List[Float32], clr: Float32, eps: Float32, wd: Float32, alt: Bool) -> List[Float32]:
    """[p', sum'] of torch.optim.Adagrad: sum += g^2 (pinned one fma, alt
    g*g then the add), p -= clr g / (sqrt(sum) + eps); clr = lr / (1 + (t -
    1) lr_decay) is the caller's scalar."""
    var n = len(p)
    var ps = List[Float32]()
    var ss = List[Float32]()
    for i in range(n):
        var gi = _z(g[i])
        if wd != Float32(0.0):
            gi = _f(wd, p[i], gi)
        var s: Float32
        if alt:
            s = _a(s0[i], _m(gi, gi))
        else:
            s = _f(gi, gi, s0[i])
        var std = _a(_z(identical_sqrt(s)), eps)
        ps.append(_f(-clr, _d(gi, std), p[i]))
        ss.append(s)
    for i in range(n):
        ps.append(ss[i])
    return ps^


# ---------------------------------------------------------------- 5538
def o_lion(p: List[Float32], g: List[Float32], m0: List[Float32], lr: Float32, b1: Float32, b2: Float32, wd: Float32, alt: Bool) -> List[Float32]:
    """[p', m'] of Lion (lion-pytorch): p *= 1 - lr wd; p -= lr sign(b1 m +
    (1 - b1) g); m = b2 m + (1 - b2) g, pinned one fma (alt: two products
    and an add)."""
    var n = len(p)
    var ps = List[Float32]()
    var ms = List[Float32]()
    var shrink = _s(Float32(1.0), _m(lr, wd))
    for i in range(n):
        var gi = _z(g[i])
        var mi = _z(m0[i])
        var pd = _m(p[i], shrink)
        var c = _f(b1, mi, _m(_s(Float32(1.0), b1), gi))
        var u = Float32(0.0)
        if c > Float32(0.0):
            u = Float32(1.0)
        elif c < Float32(0.0):
            u = Float32(-1.0)
        ps.append(_f(-lr, u, pd))
        if alt:
            ms.append(_a(_m(b2, mi), _m(_s(Float32(1.0), b2), gi)))
        else:
            ms.append(_f(b2, mi, _m(_s(Float32(1.0), b2), gi)))
    for i in range(n):
        ps.append(ms[i])
    return ps^


# ---------------------------------------------------------------- 5539
def _o_sumsq(x: List[Float32], s: Int, e: Int) -> Float32:
    # S10 independent restatement — NOT TESTED — NOT COMPILED — NOT MEASURED.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NEURAL_S10_NORM_LEAF256"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]():
        if e - s > 256:
            var total = Float32(0.0)
            var lo = s
            while lo < e:
                var part = Float32(0.0)
                for k in range(lo, min(lo + 256, e)):
                    part = _f(x[k], x[k], part)
                total = _a(total, part)
                lo += 256
            return total
    var acc = Float32(0.0)
    for k in range(s, e):
        acc = _f(x[k], x[k], acc)
    return acc


def o_lamb_ratio(p: List[Float32], u: List[Float32], offs: List[Int], clip: Bool, alt: Bool) -> List[Float32]:
    """timm Lamb's trust ratio per parameter tensor (segment): ||p|| / ||u||,
    each norm sqrt of its ascending sum of squares (alt: sqrt of the
    quotient of the two sums); 1 when either norm is 0; at most 1 under
    trust_clip."""
    var out = List[Float32]()
    for s in range(len(offs) - 1):
        var sw = _o_sumsq(p, offs[s], offs[s + 1])
        var sg = _o_sumsq(u, offs[s], offs[s + 1])
        var wn = _z(identical_sqrt(sw))
        var gn = _z(identical_sqrt(sg))
        var r = Float32(1.0)
        if wn > Float32(0.0) and gn > Float32(0.0):
            if alt:
                r = _z(identical_sqrt(_d(sw, sg)))
            else:
                r = _d(wn, gn)
        if clip and r > Float32(1.0):
            r = Float32(1.0)
        out.append(r)
    return out^


# ---------------------------------------------------------------- 5541
def o_theta_run(y: List[Float32], model: Int, level0: Float32, alpha: Float32, theta: Float32, alt: Bool) -> List[Float32]:
    """statsforecast theta.cpp `init_state` + `update` over the sample and
    `calc`'s objective: [states (n x 5: level, meany, A, B, mu), e (n),
    sum(e[3:]^2) / max(mean|y|, 1e-10)]. Models 0 STM, 1 OTM (fixed A, B
    from the OLS line), 2 DSTM, 3 DOTM (A, B updated each step). The level
    is `alpha y + (1 - alpha) level`, one fma (alt: level + alpha (y -
    level)); (1 - alpha)^i a running product."""
    var n = len(y)
    var dyn = model >= 2
    var k = _s(Float32(1.0), _d(Float32(1.0), theta))
    var y0 = _z(y[0])
    var A: Float32
    var B: Float32
    var mu: Float32
    if dyn:
        A = y0
        B = Float32(0.0)
        mu = y0
    else:
        var s = Float32(0.0)
        var w = Float32(0.0)
        for i in range(n):
            s = _a(s, y[i])
            w = _f(y[i], Float32(i + 1), w)
        var ym = _d(s, Float32(n))
        var wa = _d(w, Float32(n))
        B = _d(_m(Float32(6.0), _s(_m(Float32(2.0), wa), _m(Float32(n + 1), ym))), Float32(n * n - 1))
        A = _s(ym, _d(_m(Float32(n + 1), B), Float32(2.0)))
        mu = _f(k, _a(A, B), level0)
    var oma = _s(Float32(1.0), alpha)
    var lev: Float32
    if alt:
        lev = _a(level0, _m(alpha, _s(y0, level0)))
    else:
        lev = _f(alpha, y0, _m(oma, level0))
    var st = List[Float32]()
    var e = List[Float32]()
    st.append(lev); st.append(y0); st.append(A); st.append(B); st.append(mu)
    e.append(_s(y0, mu))
    var pw = oma
    for i in range(1, n):
        var r0 = (i - 1) * 5
        var levp = st[r0]
        var my = st[r0 + 1]
        var An = st[r0 + 2]
        var Bn = st[r0 + 3]
        var pw1 = _m(pw, oma)
        var m = _f(k, _a(_m(An, pw), _d(_m(Bn, _s(Float32(1.0), pw1)), alpha)), levp)
        var yi = _z(y[i])
        e.append(_s(yi, m))
        var nl: Float32
        if alt:
            nl = _a(levp, _m(alpha, _s(yi, levp)))
        else:
            nl = _f(alpha, yi, _m(oma, levp))
        var my2 = _d(_f(Float32(i), my, yi), Float32(i + 1))
        var A2 = An
        var B2 = Bn
        if dyn:
            B2 = _d(_a(_m(Float32(i - 1), Bn), _d(_m(Float32(6.0), _s(yi, my)), Float32(i + 1))), Float32(i + 2))
            A2 = _s(my2, _d(_m(B2, Float32(i + 2)), Float32(2.0)))
        st.append(nl); st.append(my2); st.append(A2); st.append(B2); st.append(m)
        pw = pw1
    var sa = Float32(0.0)
    for i in range(n):
        sa = _a(sa, abs(_z(y[i])))
    var mean_y = _d(sa, Float32(n))
    if mean_y < Float32(1e-10):
        mean_y = Float32(1e-10)
    var sse = Float32(0.0)
    for i in range(3, n):
        sse = _f(e[i], e[i], sse)
    for i in range(n):
        st.append(e[i])
    st.append(_d(sse, mean_y))
    return st^


# ---------------------------------------------------------------- 5542
def o_garch_sigma2(
    par: List[Float32], r: List[Float32], p: Int, o: Int, q: Int, backcast: Float32, vb: List[Float32], alt: Bool,
) -> List[Float32]:
    """arch's `garch_recursion` (power 2) with `bounds_check`: sigma2[t] =
    omega + sum alpha_j r[t-1-j]^2 + sum gamma_j r[t-1-j]^2 [r < 0] + sum
    beta_j sigma2[t-1-j], the backcast (half of it for gamma) before the
    sample; every term folded in that order, pinned one fma each (alt: the
    product rounded, then the add). Below the lower bound: the bound;
    above the upper: hi + log(v / hi); a NaN: hi."""
    var n = len(r)
    var s2 = List[Float32]()
    for t in range(n):
        var v = _z(par[0])
        var loc = 1
        for j in range(p + o + q):
            var term: Float32
            var take = True
            if j < p:
                if t - 1 - j < 0:
                    term = backcast
                else:
                    term = _m(r[t - 1 - j], r[t - 1 - j])
            elif j < p + o:
                var jj = j - p
                if t - 1 - jj < 0:
                    term = _m(Float32(0.5), backcast)
                else:
                    var x = _z(r[t - 1 - jj])
                    term = _m(x, x)
                    take = x < Float32(0.0)
            else:
                var jj = j - p - o
                if t - 1 - jj < 0:
                    term = backcast
                else:
                    term = s2[t - 1 - jj]
            if take:
                if alt:
                    v = _a(v, _m(par[loc], term))
                else:
                    v = _f(par[loc], term, v)
            loc += 1
        var lo = _z(vb[2 * t])
        var hi = _z(vb[2 * t + 1])
        if not (v == v):
            v = hi
        if v < lo:
            v = lo
        elif v > hi:
            v = _a(hi, _z(identical_log(_d(v, hi))))
        s2.append(v)
    return s2^


# ---------------------------------------------------------------- 5543
comptime _TWO_PI: Float32 = 6.283185307179586


def o_prophet_features(frac: List[Float32], orders: List[Int], hol: List[Float32], N: Int, nh: Int, alt: Bool) -> List[Float32]:
    """Prophet's Fourier design rows: for each seasonality s and i in 1 ..
    order_s, sin and cos of 2 pi i frac[t, s] (frac = t / period, the phase
    in [0, 1)); then the holiday columns. The argument pinned (2 pi i) frac
    (alt: 2 pi (i frac))."""
    var ns = len(orders)
    var out = List[Float32]()
    for t in range(N):
        for s in range(ns):
            var f = _z(frac[t * ns + s])
            for i in range(orders[s]):
                var c: Float32
                if alt:
                    c = _m(_TWO_PI, _m(Float32(i + 1), f))
                else:
                    c = _m(_m(_TWO_PI, Float32(i + 1)), f)
                out.append(_z(identical_sin(c)))
                out.append(_z(identical_cos(c)))
        for h in range(nh):
            out.append(_z(hol[t * nh + h]))
    return out^


def _o_fmix32(x: UInt32) -> UInt32:
    """murmur3's 32-bit finalizer."""
    var z = x ^ (x >> 16)
    z = z * UInt32(0x85EBCA6B)
    z = z ^ (z >> 13)
    z = z * UInt32(0xC2B2AE35)
    return z ^ (z >> 16)


def o_mlp_perm(n: Int, seed: UInt64, epoch: Int, alt: Bool) -> List[Float32]:
    """Seam 5545 (lane cpu3-python): the MLP device epoch order (roadmap D13,
    `sequence/mlp.mojo::op_mlp_perm`) restated: the epoch key is splitmix64's
    output at state seed + (epoch + 1) * golden; position t maps through six
    Feistel rounds on two h-bit halves (h = ceil(bits / 2), 2^bits >= n),
    round key fmix32(k0 + round * 0x9E3779B9) ^ k1, then cycle-walks until
    the value is below n. alt: five rounds (a dropped round)."""
    var z = seed + UInt64(epoch + 1) * UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    var key = z ^ (z >> 31)
    var k0 = UInt32(key & UInt64(0xFFFFFFFF))
    var k1 = UInt32(key >> UInt64(32))
    var bits = 1
    while (1 << bits) < n:
        bits += 1
    var h = UInt32((bits + 1) // 2)
    var mask = (UInt32(1) << h) - UInt32(1)
    var rounds = 5 if alt else 6
    var out = List[Float32]()
    for t in range(n):
        var x = UInt32(t)
        while True:
            var l = (x >> h) & mask
            var r = x & mask
            for rd in range(rounds):
                var f = _o_fmix32(r + (_o_fmix32(k0 + UInt32(rd) * UInt32(0x9E3779B9)) ^ k1)) & mask
                var nl = r
                r = l ^ f
                l = nl
            x = (l << h) | r
            if Int(x) < n:
                break
        out.append(Float32(Int(x)))
    return out^
