# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MIXTURES' k-SIZED WORK ON THE DEVICE (lane cgr4-device-optim-bgmm,
2026-10-03; Andrew: no CPU work on the GPU path, even k- or d-sized). One
source for the device kernels (`x_cluster/bgmm_kernels.mojo`) and the host
column's loops (`bgmm_host_step` below, `x_cluster/host/host_ops.mojo`).

The Wishart and Dirichlet(-process) updates, the d x d Cholesky and its
triangular inverse, digamma and log-gamma, the per-component constants of
the E-step and the variational lower bound all run in FLOAT-FLOAT
(`x_linear/ff.mojo`, about 48 bits, every operation the IDENTICAL float32
arithmetic; Metal has no float64), so a word is the same on NVIDIA, AMD,
Apple and the host. The state lives in ONE float workspace on the device
for the whole fit (`BgL`: each value a (hi, lo) pair); between EM
iterations only the scalar block (`SC_*`: the bound, the convergence and
the error flags) comes home.

A STEP is a set of independent cells (`bgmm_cells`, `bgmm_cell`): the
device runs one thread per cell, the host walks the cells ascending. Cells
that need every component's scalars (the tied pool, the Dirichlet-process
tail and prefix, the sums of the bound) recompute them in their own fixed
order; a step only reads what an EARLIER step wrote. The Cholesky step
(`ST_CHOL`) is one block per component: column j's diagonal, a barrier,
the rows below it in parallel, a barrier; then the inverse column by
column, one thread per column."""
from std.memory import bitcast

from x_cluster.bodies import FPtr
from x_linear.ff import FF, ff_add, ff_add_f, ff_div, ff_f32, ff_mul, ff_mul_f, ff_neg, ff_sqrt, ff_sub
from x_linear.ops import fm

# ------------------------------------------------------------ the scalar block
comptime SC_WCP = 0
comptime SC_MPP = 1
comptime SC_DOFP = 2
comptime SC_REG = 3
comptime SC_TOL = 4
comptime SC_LB = 5
comptime SC_N = 6
comptime SC_CONV = 7
comptime SC_ERR = 8
comptime SC_HAVE = 9
comptime SC_SIZE = 10
"""Pairs in the scalar block; the host reads 2 * SC_SIZE words per iteration."""

# ------------------------------------------------------------------ the steps
comptime ST_PRIOR = 0
"""The priors the caller did not give: X's mean and np.cov(X.T) (p1 = the
mean, p2 = the biased covariance of one all-ones component; aux bits 1 mean
given, 2 covariance given, 4 spherical)."""
comptime ST_A1 = 1
"""Per component: nk, the weight concentrations, the mean precision and the
degrees of freedom from the moments' nk (p1)."""
comptime ST_B = 2
"""Per cell: the covariances (kc x d x d) and the means (kc x d) from the
moments' means (p1) and covariances (p2)."""
comptime ST_CHOL = 3
"""Per component (a block): the precision Cholesky (aux 0: cov -> pchol;
aux 1: the precisions_init -> u)."""
comptime ST_UUT = 4
"""Per cell: cov = u u^T (the precisions_init's inverse)."""
comptime ST_OVR = 5
"""GaussianMixture's weights_init (aux bit 1) and means_init (aux bit 2)."""
comptime ST_A2 = 6
"""Per component: the digamma and log-gamma parts of the weights."""
comptime ST_E = 7
"""Per component: the E-step constant and its term of the lower bound."""
comptime ST_F = 8
"""Per cell: the float32 means (p1), precision factors (p2), constants (p3)."""
comptime ST_D = 9
"""One thread: the lower bound (p1 = the fold's (hi, lo)), the change test."""
comptime ST_KEEP = 10
"""Per word: the workspace into p1 (the best init so far)."""
comptime ST_W = 11
"""One thread: the mixture weights."""

comptime BGMM_CHOL_ERROR = "Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, increase reg_covar, or scale the input data."


struct BgL(ImplicitlyCopyable, Movable):
    """The workspace layout, in pairs: the scalar block, the priors, then
    per component the scalars, the means and the d x d matrices."""
    var pm: Int
    var pc: Int
    var nk: Int
    var wc0: Int
    var wc1: Int
    var mp: Int
    var dof: Int
    var ga: Int
    var gb: Int
    var gw: Int
    var cst: Int
    var lbk: Int
    var wts: Int
    var wi: Int
    var means: Int
    var mi: Int
    var cov: Int
    var pchol: Int
    var l: Int
    var pi: Int
    var u: Int
    var total: Int

    @always_inline
    def __init__(out self, kc: Int, d: Int):
        var dd = d * d
        self.pm = SC_SIZE
        self.pc = self.pm + d
        self.nk = self.pc + dd
        self.wc0 = self.nk + kc
        self.wc1 = self.wc0 + kc
        self.mp = self.wc1 + kc
        self.dof = self.mp + kc
        self.ga = self.dof + kc
        self.gb = self.ga + kc
        self.gw = self.gb + kc
        self.cst = self.gw + kc
        self.lbk = self.cst + kc
        self.wts = self.lbk + kc
        self.wi = self.wts + kc
        self.means = self.wi + kc
        self.mi = self.means + kc * d
        self.cov = self.mi + kc * d
        self.pchol = self.cov + kc * dd
        self.l = self.pchol + kc * dd
        self.pi = self.l + kc * dd
        self.u = self.pi + kc * dd
        self.total = self.u + kc * dd


# ------------------------------------------------------------ float-float math
comptime _ZERO = FF(Float32(0), Float32(0))
comptime _ONE = FF(Float32(1), Float32(0))
comptime _LN2 = FF(Float32(0.6931471824645996), Float32(-1.9046542121259336e-09))
comptime _LOG_2PI = FF(Float32(1.8378770351409912), Float32(3.126835323996602e-08))
comptime _SQRT2_HI = Float32(1.4142135381698608)
comptime _C12 = FF(Float32(0.0833333358168602), Float32(-2.4835269396561444e-09))
comptime _C120 = FF(Float32(0.008333333767950535), Float32(-4.34617203337595e-10))
comptime _C252 = FF(Float32(0.003968254197388887), Float32(-2.291349193717096e-10))
comptime _C240 = FF(Float32(0.004166666883975267), Float32(-2.173086016687975e-10))
comptime _C132 = FF(Float32(0.007575757801532745), Float32(-2.2577517633237676e-10))
comptime _C360 = FF(Float32(0.0027777778450399637), Float32(-6.726218887420643e-11))
comptime _C1260 = FF(Float32(0.0007936508045531809), Float32(-1.0902387499733823e-11))
comptime _C1680 = FF(Float32(0.0005952381179668009), Float32(-2.2728706070007654e-11))


@always_inline
def ff_half(x: FF) -> FF:
    """x / 2, exact."""
    return FF(fm(x.hi, Float32(0.5)), fm(x.lo, Float32(0.5)))


def ff_log(x: FF) -> FF:
    """log(x), x > 0 normal: x = 2^e m with m in [sqrt(1/2), sqrt(2)), then
    log m = 2 atanh(s), s = (m - 1) / (m + 1), |s| <= 0.1716, the odd series
    to s^23 (its first omitted term is below 2^-60)."""
    var bits = bitcast[DType.uint32](x.hi)
    var e = Int((bits >> 23) & UInt32(0xFF)) - 127
    var sc = bitcast[DType.float32](UInt32(127 - e) << 23)
    var m = FF(fm(x.hi, sc), fm(x.lo, sc))
    if m.hi >= _SQRT2_HI:
        m = ff_half(m)
        e += 1
    var s = ff_div(ff_add_f(m, Float32(-1)), ff_add_f(m, Float32(1)))
    var s2 = ff_mul(s, s)
    var p = FF(Float32(0.043478261679410934), Float32(-8.098456905081264e-10))
    p = ff_add(FF(Float32(0.0476190485060215), Float32(-8.869738832295582e-10)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.05263157933950424), Float32(-3.9213582381236733e-10)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.05882352963089943), Float32(-2.1913472425527658e-10)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.06666667014360428), Float32(-3.47693762670076e-09)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.07692307978868484), Float32(-2.8656079731348427e-09)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.09090909361839294), Float32(-2.709302115988521e-09)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.1111111119389534), Float32(-8.278422947149977e-10)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.1428571492433548), Float32(-6.38621200366174e-09)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.20000000298023224), Float32(-2.9802322831784522e-09)), ff_mul(s2, p))
    p = ff_add(FF(Float32(0.3333333432674408), Float32(-9.934107758624577e-09)), ff_mul(s2, p))
    p = ff_add(_ONE, ff_mul(s2, p))
    var r = ff_mul(ff_add(s, s), p)
    return ff_add(ff_mul_f(_LN2, Float32(e)), r)


def digamma_ff(x_in: FF) -> FF:
    """psi(x), x > 0: the recurrence up to x >= 6, then the asymptotic series
    (the float64 `digamma64` it replaces, term for term)."""
    var x = x_in
    var acc = _ZERO
    var steps = 0
    while x.hi < Float32(6) and steps < 64:
        acc = ff_sub(acc, ff_div(_ONE, x))
        x = ff_add_f(x, Float32(1))
        steps += 1
    var inv = ff_div(_ONE, x)
    var inv2 = ff_mul(inv, inv)
    var t = ff_sub(_C240, ff_mul(inv2, _C132))
    t = ff_sub(_C252, ff_mul(inv2, t))
    t = ff_sub(_C120, ff_mul(inv2, t))
    t = ff_sub(_C12, ff_mul(inv2, t))
    var series = ff_mul(inv2, t)
    return ff_sub(ff_sub(ff_add(acc, ff_log(x)), ff_half(inv)), series)


def lgamma_ff(x_in: FF) -> FF:
    """log Gamma(x), x > 0: shift to x >= 7 (one product, one log), then
    Stirling (the float64 `lgamma64` it replaces, term for term)."""
    var x = x_in
    var prod = _ONE
    var steps = 0
    while x.hi < Float32(7) and steps < 64:
        prod = ff_mul(prod, x)
        x = ff_add_f(x, Float32(1))
        steps += 1
    var inv = ff_div(_ONE, x)
    var inv2 = ff_mul(inv, inv)
    var t = ff_sub(_C1260, ff_mul(inv2, _C1680))
    t = ff_sub(_C360, ff_mul(inv2, t))
    t = ff_sub(_C12, ff_mul(inv2, t))
    var series = ff_mul(inv, t)
    var r = ff_sub(ff_mul(ff_add_f(x, Float32(-0.5)), ff_log(x)), x)
    r = ff_add(r, ff_half(_LOG_2PI))
    r = ff_add(r, series)
    return ff_sub(r, ff_log(prod))


@always_inline
def ld2(w: FPtr, t: Int) -> FF:
    return FF(w[2 * t], w[2 * t + 1])


@always_inline
def st2(w: FPtr, t: Int, v: FF):
    w[2 * t] = v.hi
    w[2 * t + 1] = v.lo


@always_inline
def _of(v: Float32) -> FF:
    return FF(v, Float32(0))


@always_inline
def _abs_ff(x: FF) -> FF:
    return ff_neg(x) if x.hi < Float32(0) else x


@always_inline
def _lt_ff(x: FF, y: FF) -> Bool:
    """x < y, (hi, lo) lexicographic."""
    if x.hi != y.hi:
        return x.hi < y.hi
    return x.lo < y.lo


# ------------------------------------------------------------------- the cells
@always_inline
def bgmm_cells(step: Int, kc: Int, d: Int) -> Int:
    """Cells of a step (threads on the device; ST_CHOL: blocks)."""
    var dd = d * d
    if step == ST_PRIOR:
        return d + dd
    if step == ST_A1 or step == ST_A2 or step == ST_E or step == ST_CHOL:
        return kc
    if step == ST_B:
        return kc * dd + kc * d
    if step == ST_UUT:
        return kc * dd
    if step == ST_OVR:
        return kc * d + kc
    if step == ST_F:
        return kc * dd + kc * d + kc
    if step == ST_KEEP:
        return 2 * BgL(kc, d).total
    return 1


@always_inline
def _prior_cell(w: FPtr, xm: FPtr, sk: FPtr, kc: Int, d: Int, aux: Int, t: Int):
    var L = BgL(kc, d)
    if t < d:
        if (aux & 1) == 0:
            st2(w, L.pm + t, _of(xm[t]))
        return
    if (aux & 2) != 0:
        return
    var c = t - d
    var a = c // d
    var b = c - a * d
    # np.cov(X.T), ddof 1: the biased moment times n / (n - 1)
    var n = ld2(w, SC_N)
    var ratio = ff_div(n, ff_add_f(n, Float32(-1))) if n.hi > Float32(1) else _ONE
    if (aux & 4) != 0:
        # spherical: var(X, ddof=1).mean() as s * I
        if a != b:
            st2(w, L.pc + c, _ZERO)
            return
        var acc = _ZERO
        for q in range(d):
            acc = ff_add(acc, ff_mul(_of(sk[q * d + q]), ratio))
        st2(w, L.pc + c, ff_div(acc, _of(Float32(d))))
        return
    st2(w, L.pc + c, ff_mul(_of(sk[c]), ratio))


@always_inline
def _a1_cell(w: FPtr, nkp: FPtr, kc: Int, d: Int, cfg: Int, k: Int):
    var L = BgL(kc, d)
    var dp = (cfg & 1) != 0
    var variational = (cfg & 2) != 0
    var ct = cfg >> 2
    var nk = _of(nkp[k])
    st2(w, L.nk + k, nk)
    if not variational:
        var tot = _ZERO
        for q in range(kc):
            tot = ff_add_f(tot, nkp[q])
        st2(w, L.wc0 + k, ff_div(nk, tot))
        st2(w, L.wc1 + k, _ZERO)
        st2(w, L.mp + k, _ONE)
        st2(w, L.dof + k, _ZERO)
        return
    if dp:
        # tail[k] = nk[kc - 1] + ... + nk[k + 1], added in that order
        var tail = _ZERO
        var q = kc - 1
        while q > k:
            tail = ff_add_f(tail, nkp[q])
            q -= 1
        st2(w, L.wc0 + k, ff_add_f(_ONE, nkp[k]))
        st2(w, L.wc1 + k, ff_add(ld2(w, SC_WCP), tail))
    else:
        st2(w, L.wc0 + k, ff_add_f(ld2(w, SC_WCP), nkp[k]))
        st2(w, L.wc1 + k, _ZERO)
    st2(w, L.mp + k, ff_add_f(ld2(w, SC_MPP), nkp[k]))
    if ct == 1:
        var ntot = _ZERO
        for q in range(kc):
            ntot = ff_add_f(ntot, nkp[q])
        st2(w, L.dof + k, ff_add(ld2(w, SC_DOFP), ff_div(ntot, _of(Float32(kc)))))
    else:
        st2(w, L.dof + k, ff_add_f(ld2(w, SC_DOFP), nkp[k]))


@always_inline
def _dev(w: FPtr, xk: FPtr, L: BgL, k: Int, d: Int, a: Int) -> FF:
    """xk[k, a] - mean_prior[a]."""
    return ff_add_f(ff_neg(ld2(w, L.pm + a)), xk[k * d + a])


@always_inline
def _cov_cell(w: FPtr, xk: FPtr, sk: FPtr, kc: Int, d: Int, cfg: Int, t: Int) -> FF:
    """sklearn `_estimate_wishart_{full,tied,diag,spherical}` (variational) or
    `_estimate_gaussian_covariances_*` (plain EM), cell (k, a, b) of the full
    d x d matrix (zeros off the diagonal for diag and spherical)."""
    var L = BgL(kc, d)
    var variational = (cfg & 2) != 0
    var ct = cfg >> 2
    var dd = d * d
    var k = t // dd
    var r = t - k * dd
    var a = r // d
    var b = r - a * d
    var diag = a == b
    var reg = ld2(w, SC_REG)
    if not variational:
        if ct == 0:
            return _of(sk[t])
        if ct == 1:
            var tot = _ZERO
            var acc = _ZERO
            for q in range(kc):
                var nq = ld2(w, L.nk + q)
                tot = ff_add(tot, nq)
                var v = _of(sk[q * dd + r])
                if diag:
                    v = ff_sub(v, reg)
                acc = ff_add(acc, ff_mul(nq, v))
            var v = ff_div(acc, tot)
            return ff_add(v, reg) if diag else v
        if not diag:
            return _ZERO
        if ct == 2:
            return _of(sk[t])
        var acc = _ZERO
        for q in range(d):
            acc = ff_add_f(acc, sk[k * dd + q * d + q])
        return ff_div(acc, _of(Float32(d)))
    var mpp = ld2(w, SC_MPP)
    var nk = ld2(w, L.nk + k)
    var mp = ld2(w, L.mp + k)
    var dof = ld2(w, L.dof + k)
    if ct == 0:
        var coef = ff_div(ff_mul(nk, mpp), mp)
        var da = _dev(w, xk, L, k, d, a)
        var db = _dev(w, xk, L, k, d, b)
        var v = ff_add(ld2(w, L.pc + r), ff_mul(nk, _of(sk[t])))
        v = ff_add(v, ff_mul(coef, ff_mul(da, db)))
        return ff_div(v, dof)
    if ct == 1:
        # the tied pool of every component's moments, the same in each cell
        var ntot = _ZERO
        var skp = _ZERO
        var acc = _ZERO
        for q in range(kc):
            var nq = ld2(w, L.nk + q)
            ntot = ff_add(ntot, nq)
            var v = _of(sk[q * dd + r])
            if diag:
                v = ff_sub(v, reg)
            skp = ff_add(skp, ff_mul(nq, v))
            var da = _dev(w, xk, L, q, d, a)
            var db = _dev(w, xk, L, q, d, b)
            acc = ff_add(acc, ff_mul(ff_div(nq, ld2(w, L.mp + q)), ff_mul(da, db)))
        skp = ff_div(skp, ntot)
        if diag:
            skp = ff_add(skp, reg)
        var fk = _of(Float32(kc))
        var v = ff_add(ld2(w, L.pc + r), ff_mul(skp, ff_div(ntot, fk)))
        v = ff_add(v, ff_mul(ff_div(mpp, fk), acc))
        return ff_div(v, dof)
    if not diag:
        return _ZERO
    var ratio = ff_div(mpp, mp)
    if ct == 2:
        var da = _dev(w, xk, L, k, d, a)
        var v = ff_add(_of(sk[t]), ff_mul(ratio, ff_mul(da, da)))
        v = ff_add(ld2(w, L.pc + r), ff_mul(nk, v))
        return ff_div(v, dof)
    var skm = _ZERO
    var dm = _ZERO
    for q in range(d):
        var dq = _dev(w, xk, L, k, d, q)
        skm = ff_add_f(skm, sk[k * dd + q * d + q])
        dm = ff_add(dm, ff_mul(dq, dq))
    var fd = _of(Float32(d))
    var v = ff_add(ff_div(skm, fd), ff_mul(ratio, ff_div(dm, fd)))
    return ff_div(ff_add(ld2(w, L.pc), ff_mul(nk, v)), dof)


@always_inline
def _b_cell(w: FPtr, xk: FPtr, sk: FPtr, kc: Int, d: Int, cfg: Int, t: Int):
    var L = BgL(kc, d)
    var ncov = kc * d * d
    if t < ncov:
        st2(w, L.cov + t, _cov_cell(w, xk, sk, kc, d, cfg, t))
        return
    var u = t - ncov
    var k = u // d
    var a = u - k * d
    if (cfg & 2) == 0:
        st2(w, L.means + u, _of(xk[u]))
        return
    var nk = ld2(w, L.nk + k)
    var v = ff_add(ff_mul(ld2(w, SC_MPP), ld2(w, L.pm + a)), ff_mul(nk, _of(xk[k * d + a])))
    st2(w, L.means + u, ff_div(v, ld2(w, L.mp + k)))


# The Cholesky cells: src (d x d per component) -> L (lower, the `l` region)
# -> dst = inv(L)^T (upper).
@always_inline
def chol_diag_cell(w: FPtr, kc: Int, d: Int, src: Int, k: Int, j: Int):
    var L = BgL(kc, d)
    var o = k * d * d
    var s = ld2(w, src + o + j * d + j)
    for q in range(j):
        var lq = ld2(w, L.l + o + j * d + q)
        s = ff_sub(s, ff_mul(lq, lq))
    if not (s.hi > Float32(0)):
        w[2 * SC_ERR] = Float32(1)
        s = _ONE
    st2(w, L.l + o + j * d + j, ff_sqrt(s))


@always_inline
def chol_off_cell(w: FPtr, kc: Int, d: Int, src: Int, k: Int, j: Int, i: Int):
    var L = BgL(kc, d)
    var o = k * d * d
    var t = ld2(w, src + o + i * d + j)
    for q in range(j):
        t = ff_sub(t, ff_mul(ld2(w, L.l + o + i * d + q), ld2(w, L.l + o + j * d + q)))
    st2(w, L.l + o + i * d + j, ff_div(t, ld2(w, L.l + o + j * d + j)))


@always_inline
def chol_inv_cell(w: FPtr, kc: Int, d: Int, dst: Int, k: Int, c: Int):
    """Row c of inv(L)^T: forward substitution down column c of inv(L)."""
    var L = BgL(kc, d)
    var o = k * d * d
    for i in range(d):
        if i < c:
            st2(w, dst + o + c * d + i, _ZERO)
            continue
        var t = _ONE if i == c else _ZERO
        for q in range(c, i):
            t = ff_sub(t, ff_mul(ld2(w, L.l + o + i * d + q), ld2(w, dst + o + c * d + q)))
        st2(w, dst + o + c * d + i, ff_div(t, ld2(w, L.l + o + i * d + i)))


@always_inline
def chol_regions(kc: Int, d: Int, aux: Int) -> Tuple[Int, Int]:
    var L = BgL(kc, d)
    if aux == 1:
        return (L.pi, L.u)
    return (L.cov, L.pchol)


@always_inline
def _uut_cell(w: FPtr, kc: Int, d: Int, t: Int):
    var L = BgL(kc, d)
    var dd = d * d
    var k = t // dd
    var r = t - k * dd
    var a = r // d
    var b = r - a * d
    var o = L.u + k * dd
    var acc = _ZERO
    for q in range(d):
        acc = ff_add(acc, ff_mul(ld2(w, o + a * d + q), ld2(w, o + b * d + q)))
    st2(w, L.cov + t, acc)


@always_inline
def _ovr_cell(w: FPtr, kc: Int, d: Int, aux: Int, t: Int):
    var L = BgL(kc, d)
    if t < kc * d:
        if (aux & 2) != 0:
            st2(w, L.means + t, ld2(w, L.mi + t))
        return
    var k = t - kc * d
    if (aux & 1) != 0:
        st2(w, L.wc0 + k, ld2(w, L.wi + k))


@always_inline
def _a2_cell(w: FPtr, kc: Int, d: Int, cfg: Int, k: Int):
    var L = BgL(kc, d)
    if (cfg & 2) == 0:
        return
    var wc0 = ld2(w, L.wc0 + k)
    if (cfg & 1) != 0:
        var wc1 = ld2(w, L.wc1 + k)
        var s = ff_add(wc0, wc1)
        var ds = digamma_ff(s)
        st2(w, L.ga + k, ff_sub(digamma_ff(wc0), ds))
        st2(w, L.gb + k, ff_sub(digamma_ff(wc1), ds))
        st2(w, L.gw + k, ff_sub(ff_add(lgamma_ff(wc0), lgamma_ff(wc1)), lgamma_ff(s)))
    else:
        st2(w, L.ga + k, digamma_ff(wc0))
        st2(w, L.gb + k, _ZERO)
        st2(w, L.gw + k, lgamma_ff(wc0))


@always_inline
def _e_cell(w: FPtr, kc: Int, d: Int, cfg: Int, k: Int):
    """c_k with wlp[i, k] = c_k - q_ik / 2 (`_estimate_weighted_log_prob`)
    and the component's term of the lower bound (`_compute_lower_bound`):
    lbk = dof (log det - d/2 log dof) + dof d/2 log 2 + sum_j lgamma((dof - j) / 2)
    + (betaln of the stick | lgamma(wc0)) - d/2 log(mean_precision)."""
    var L = BgL(kc, d)
    var o = L.pchol + k * d * d
    var ld = _ZERO
    for j in range(d):
        ld = ff_add(ld, ff_log(ld2(w, o + j * d + j)))
    var hd = Float32(0.5) * Float32(d)
    var hlog2pi = ff_mul_f(_LOG_2PI, hd)
    var wc0 = ld2(w, L.wc0 + k)
    if (cfg & 2) == 0:
        st2(w, L.cst + k, ff_add(ff_sub(ff_log(wc0), hlog2pi), ld))
        st2(w, L.lbk + k, _ZERO)
        return
    var lw: FF
    if (cfg & 1) != 0:
        var run = _ZERO
        for q in range(k):
            run = ff_add(run, ld2(w, L.gb + q))
        lw = ff_add(ld2(w, L.ga + k), run)
    else:
        var tot = _ZERO
        for q in range(kc):
            tot = ff_add(tot, ld2(w, L.wc0 + q))
        lw = ff_sub(ld2(w, L.ga + k), digamma_ff(tot))
    var dof = ld2(w, L.dof + k)
    var mp = ld2(w, L.mp + k)
    var loglam = ff_mul_f(_LN2, Float32(d))
    var g = _ZERO
    for j in range(d):
        var x = ff_half(ff_add_f(dof, -Float32(j)))
        loglam = ff_add(loglam, digamma_ff(x))
        g = ff_add(g, lgamma_ff(x))
    var ldof = ff_log(dof)
    var hdl = ff_mul_f(ldof, hd)
    var c = ff_sub(ff_sub(ld, hlog2pi), hdl)
    c = ff_add(c, ff_half(ff_sub(loglam, ff_div(_of(Float32(d)), mp))))
    c = ff_add(c, lw)
    st2(w, L.cst + k, c)
    var term = ff_add(ff_mul(dof, ff_sub(ld, hdl)), ff_mul(dof, ff_mul_f(_LN2, hd)))
    var lbk = ff_add(ff_add(term, g), ld2(w, L.gw + k))
    st2(w, L.lbk + k, ff_sub(lbk, ff_mul_f(ff_log(mp), hd)))


@always_inline
def _f_cell(w: FPtr, ms: FPtr, ps: FPtr, cs: FPtr, kc: Int, d: Int, t: Int):
    var L = BgL(kc, d)
    var ncov = kc * d * d
    if t < ncov:
        ps[t] = ff_f32(ld2(w, L.pchol + t))
        return
    var u = t - ncov
    if u < kc * d:
        ms[u] = ff_f32(ld2(w, L.means + u))
        return
    var k = u - kc * d
    cs[k] = ff_f32(ld2(w, L.cst + k))


@always_inline
def _d_cell(w: FPtr, es: FPtr, kc: Int, d: Int, cfg: Int):
    """The bound: -entropy + sum_k lbk (- lgamma(sum wc0) for the Dirichlet
    distribution), or the plain EM's mean log-likelihood; then sklearn's
    `abs(lb - prev) < tol`."""
    var L = BgL(kc, d)
    var ent = FF(es[0], es[1])
    var lb: FF
    if (cfg & 2) != 0:
        var s = _ZERO
        for k in range(kc):
            s = ff_add(s, ld2(w, L.lbk + k))
        if (cfg & 1) == 0:
            var tot = _ZERO
            for k in range(kc):
                tot = ff_add(tot, ld2(w, L.wc0 + k))
            s = ff_sub(s, lgamma_ff(tot))
        lb = ff_sub(s, ent)
    else:
        lb = ff_div(ent, ld2(w, SC_N))
    if w[2 * SC_HAVE] != Float32(0):
        var ch = _abs_ff(ff_sub(lb, ld2(w, SC_LB)))
        if _lt_ff(ch, ld2(w, SC_TOL)):
            w[2 * SC_CONV] = Float32(1)
    w[2 * SC_HAVE] = Float32(1)
    st2(w, SC_LB, lb)


@always_inline
def _w_cell(w: FPtr, kc: Int, d: Int, cfg: Int):
    """`weights_`: the stick-breaking products (Dirichlet process) or wc0,
    normalized; the plain EM's nk / n as they are."""
    var L = BgL(kc, d)
    if (cfg & 2) == 0:
        for k in range(kc):
            st2(w, L.wts + k, ld2(w, L.wc0 + k))
        return
    if (cfg & 1) != 0:
        var run = _ONE
        for k in range(kc):
            var wc0 = ld2(w, L.wc0 + k)
            var wc1 = ld2(w, L.wc1 + k)
            var s = ff_add(wc0, wc1)
            st2(w, L.wts + k, ff_mul(ff_div(wc0, s), run))
            run = ff_mul(run, ff_div(wc1, s))
    else:
        for k in range(kc):
            st2(w, L.wts + k, ld2(w, L.wc0 + k))
    var tot = _ZERO
    for k in range(kc):
        tot = ff_add(tot, ld2(w, L.wts + k))
    for k in range(kc):
        st2(w, L.wts + k, ff_div(ld2(w, L.wts + k), tot))


@always_inline
def bgmm_cell(step: Int, w: FPtr, p1: FPtr, p2: FPtr, p3: FPtr, kc: Int, d: Int, cfg: Int, aux: Int, t: Int):
    """Cell t of a step (every step but ST_CHOL)."""
    if step == ST_PRIOR:
        _prior_cell(w, p1, p2, kc, d, aux, t)
    elif step == ST_A1:
        _a1_cell(w, p1, kc, d, cfg, t)
    elif step == ST_B:
        _b_cell(w, p1, p2, kc, d, cfg, t)
    elif step == ST_UUT:
        _uut_cell(w, kc, d, t)
    elif step == ST_OVR:
        _ovr_cell(w, kc, d, aux, t)
    elif step == ST_A2:
        _a2_cell(w, kc, d, cfg, t)
    elif step == ST_E:
        _e_cell(w, kc, d, cfg, t)
    elif step == ST_F:
        _f_cell(w, p1, p2, p3, kc, d, t)
    elif step == ST_D:
        _d_cell(w, p1, kc, d, cfg)
    elif step == ST_KEEP:
        p1[t] = w[t]
    elif step == ST_W:
        _w_cell(w, kc, d, cfg)


def bgmm_host_step(step: Int, w: FPtr, p1: FPtr, p2: FPtr, p3: FPtr, kc: Int, d: Int, cfg: Int, aux: Int):
    """The host column: the device's cells ascending; the Cholesky's
    columns, then each column's rows, then the inverse's columns."""
    if step == ST_CHOL:
        var rg = chol_regions(kc, d, aux)
        for k in range(kc):
            for j in range(d):
                chol_diag_cell(w, kc, d, rg[0], k, j)
                for i in range(j + 1, d):
                    chol_off_cell(w, kc, d, rg[0], k, j, i)
            for c in range(d):
                chol_inv_cell(w, kc, d, rg[1], k, c)
        return
    for t in range(bgmm_cells(step, kc, d)):
        bgmm_cell(step, w, p1, p2, p3, kc, d, cfg, aux, t)
