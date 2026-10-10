# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# TOMBSTONE: MOJOLEARN_BGMM_ENT (DROPPED-noise) deleted 2026-10-03 by 31b0cf94c; code recoverable at 31b0cf94c^.
# Restore: git apply experiments/removed/MOJOLEARN_BGMM_ENT.patch; record in docs/TOMBSTONES.md.
"""BayesianGaussianMixture and the EM GaussianMixture, every covariance
type. Reference: scikit-learn `sklearn/mixture/_bayesian_mixture.py` (the
variational updates, `_estimate_log_weights`, `_estimate_log_prob`,
`_compute_lower_bound`, `_set_parameters`), `_base.py::fit_predict` (the EM
loop, `abs(change) < tol`, the best of `n_init`) and `_gaussian_mixture.py`
(`_estimate_gaussian_parameters`, `_compute_precision_cholesky`).

EVERYTHING IS THE DEVICE'S. The n-sized work in float32: the Mahalanobis
squares against the upper-triangular precision Cholesky factors
(`bodies.gauss_q_cell`), the row log-sum-exp of the E-step
(`bodies.resp_row`), the M-step moments (`nk_cell` rows ascending; the
means and covariances through the identical GEMM, DEVIATION 5110 revised
2026-09-29, `x_cluster/host/moments_gemm.mojo`), the bound's entropy or mean
log-likelihood (the float-float fold, lane cgr2-cluster), the default priors
(X's mean and covariance as the moments of one all-ones component). The
k-sized work in float-float on the device (lane cgr4-device-optim-bgmm,
`x_cluster/bgmm_device.mojo`): the Wishart and Dirichlet(-process) updates,
the d x d Cholesky and its triangular inverse, digamma and log-gamma, the
E-step constants and the lower bound. The state stays in one device
workspace for the whole fit; per iteration only the scalar block (the bound,
the convergence and error flags) comes home, and the state once at the end.
The host column runs the same cells in the same order (`bgmm_host_step`).
The k-means start is this library's KMeans through `ClusterOps.kmeans`."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.compile import is_defined
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS
from x_cluster.bgmm_device import (
    BGMM_CHOL_ERROR,
    SC_CONV,
    SC_DOFP,
    SC_ERR,
    SC_HAVE,
    SC_LB,
    SC_MPP,
    SC_N,
    SC_REG,
    SC_SIZE,
    SC_TOL,
    SC_WCP,
    ST_A1,
    ST_A2,
    ST_B,
    ST_CHOL,
    ST_D,
    ST_E,
    ST_F,
    ST_KEEP,
    ST_OVR,
    ST_PRIOR,
    ST_UUT,
    ST_W,
    BgL,
)
from x_cluster.bodies import SPLITMIX_GAMMA, SplitMix64
from x_cluster.common import greedy_kmeans_pp_indices
from x_cluster.ops import ClusterOps
from x_cluster.post_bodies import FM_PROD, FM_VAL, ff_of_f64
from x_linear.ff import FF, ff_f32

# `-D MOJOLEARN_BGMM_ESTEP1=1` (FAST, the GPU binding): the E-step's three
# kernels as one launch, a row per thread (`ops.estep`; the same values).
comptime BGMM_ESTEP1 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_BGMM_ESTEP1"]()


struct BgmmState(Copyable, Movable):
    var nk: List[Float64]
    var wc0: List[Float64]
    var wc1: List[Float64]
    var mean_prec: List[Float64]
    var means: List[Float64]
    var dof: List[Float64]
    var cov: List[Float64]
    var pchol: List[Float64]
    var weights: List[Float64]
    """`weights_` (the device's ST_W)."""
    var consts: List[Float32]
    """The E-step constants c_k (the device's ST_E, as published)."""
    var mean_prior: List[Float64]
    var cov_prior: List[Float64]

    def __init__(out self):
        self.nk = List[Float64]()
        self.wc0 = List[Float64]()
        self.wc1 = List[Float64]()
        self.mean_prec = List[Float64]()
        self.means = List[Float64]()
        self.dof = List[Float64]()
        self.cov = List[Float64]()
        self.pchol = List[Float64]()
        self.weights = List[Float64]()
        self.consts = List[Float32]()
        self.mean_prior = List[Float64]()
        self.cov_prior = List[Float64]()


@fieldwise_init
struct BgmmPriors(Copyable, Movable):
    var kc: Int
    var d: Int
    var dp: Bool
    var wcp: Float64
    var mpp: Float64
    var mean_prior: List[Float64]
    """The caller's mean prior (d), or empty: X's mean, on the device."""
    var dofp: Float64
    var cov_prior: List[Float64]
    """The caller's covariance prior (d x d), or empty: np.cov(X.T) (its
    mean variance times I when spherical), on the device."""
    var cov_type: Int
    var variational: Bool
    """False: the PLAIN EM GaussianMixture (sklearn `_gaussian_mixture.py`)
    on the same machinery: weights nk / n, means xk, the covariances by type,
    the lower bound the mean log-likelihood of the E-step. The priors above
    are then unused.

    cov_type: 0 full, 1 tied, 2 diag, 3 spherical. The state always holds FULL d x d
    matrices per component (diag embedded, spherical as s * I, tied repeated),
    so one Cholesky and one device Mahalanobis kernel serve every type; the
    zeros off the diagonal add exactly nothing."""


def _distinct_rows(mut rng: SplitMix64, n: Int, kc: Int) -> List[Int]:
    """'random_from_data': the first kc of a partial Fisher-Yates shuffle of
    0 .. n - 1 (q swaps with q + below(n - q)), the moved positions kept in
    a k-sized list instead of an n-sized pool."""
    var pos = List[Int]()
    var val = List[Int]()

    def at(p: Int, pos: List[Int], val: List[Int]) -> Int:
        for t in range(len(pos)):
            if pos[t] == p:
                return val[t]
        return p

    def put(p: Int, v: Int, mut pos: List[Int], mut val: List[Int]):
        for t in range(len(pos)):
            if pos[t] == p:
                val[t] = v
                return
        pos.append(p)
        val.append(v)

    var picks = List[Int](capacity=kc)
    for q in range(kc):
        var r = q + rng.below(n - q)
        var vq = at(q, pos, val)
        var vr = at(r, pos, val)
        put(q, vr, pos, val)
        put(r, vq, pos, val)
        picks.append(vr)
    return picks^


@fieldwise_init
struct BgmmFit(Copyable, Movable):
    var lower_bound: Float64
    var n_iter: Int
    var converged: Bool


def _put(mut h: List[Float32], t: Int, v: Float64):
    """Pair t of a host-built workspace: an input double as float-float."""
    var f = ff_of_f64(v)
    h[2 * t] = f.hi
    h[2 * t + 1] = f.lo


def _scalars(pr: BgmmPriors, reg: Float32, tol: Float64, n: Int, lb: Float64, have: Bool) -> List[Float32]:
    """The scalar block: the priors' scalars, reg_covar, tol, the bound so
    far, n, and the flags (converged, error, have a bound)."""
    var h = List[Float32](length=2 * SC_SIZE, fill=Float32(0))
    _put(h, SC_WCP, pr.wcp)
    _put(h, SC_MPP, pr.mpp)
    _put(h, SC_DOFP, pr.dofp)
    _put(h, SC_REG, Float64(reg))
    _put(h, SC_TOL, tol)
    _put(h, SC_LB, lb)
    _put(h, SC_N, Float64(n))
    if have:
        h[2 * SC_HAVE] = Float32(1)
    return h^


def _ff64(h: List[Float32], t: Int) -> Float64:
    return Float64(h[2 * t]) + Float64(h[2 * t + 1])


def _region(h: List[Float32], off: Int, m: Int) -> List[Float64]:
    var out = List[Float64](capacity=m)
    for t in range(m):
        out.append(_ff64(h, off + t))
    return out^


def _state_of(h: List[Float32], kc: Int, d: Int, variational: Bool) -> BgmmState:
    """The best init's workspace, read once at the end, as the fit's state."""
    var L = BgL(kc, d)
    var st = BgmmState()
    st.nk = _region(h, L.nk, kc)
    st.wc0 = _region(h, L.wc0, kc)
    if variational:
        st.wc1 = _region(h, L.wc1, kc)
    st.mean_prec = _region(h, L.mp, kc)
    st.means = _region(h, L.means, kc * d)
    st.dof = _region(h, L.dof, kc)
    st.cov = _region(h, L.cov, kc * d * d)
    st.pchol = _region(h, L.pchol, kc * d * d)
    st.weights = _region(h, L.wts, kc)
    st.consts = List[Float32](capacity=kc)
    for k in range(kc):
        var t = L.cst + k
        st.consts.append(ff_f32(FF(h[2 * t], h[2 * t + 1])))
    st.mean_prior = _region(h, L.pm, d)
    st.cov_prior = _region(h, L.pc, d * d)
    return st^


def _m_step[O: ClusterOps](mut ops: O, kc: Int, d: Int, cfg: Int, ws: Int, nks: Int, xks: Int, sks: Int) raises:
    """The M-step's k-sized work from the moments: the scalars, the means and
    covariances, the precision Cholesky."""
    ops.bgmm_step(ST_A1, kc, d, cfg, 0, ws, nks, -1, -1)
    ops.bgmm_step(ST_B, kc, d, cfg, 0, ws, xks, sks, -1)
    ops.bgmm_step(ST_CHOL, kc, d, cfg, 0, ws, -1, -1, -1)


def _derive[O: ClusterOps](mut ops: O, kc: Int, d: Int, cfg: Int, ws: Int, ms: Int, ps: Int, cs: Int) raises:
    """From the state: the weights' digamma parts, the E-step constants and
    the bound's terms, then the float32 means, factors and constants."""
    ops.bgmm_step(ST_A2, kc, d, cfg, 0, ws, -1, -1, -1)
    ops.bgmm_step(ST_E, kc, d, cfg, 0, ws, -1, -1, -1)
    ops.bgmm_step(ST_F, kc, d, cfg, 0, ws, ms, ps, cs)


def bgmm_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, pr: BgmmPriors, reg: Float32, tol: Float64,
    max_iter: Int, n_init: Int, init_mode: Int, seed: UInt64,
    mut best: BgmmState, mut labels: List[Int32],
    warm: Bool = False, warm_lb: Float64 = 0,
    w_init: List[Float64] = List[Float64](), m_init: List[Float64] = List[Float64](),
    p_init: List[Float64] = List[Float64](),
) raises -> BgmmFit:
    """init_mode: 0 'kmeans', 1 'random', 2 'k-means++' (one-hot at the greedy
    k-means++ picks), 3 'random_from_data' (one-hot at k distinct rows).
    `best` holds the warm state on entry (warm) and the best init's state
    on return."""
    var kc = pr.kc
    if kc < 1 or kc > n:
        raise Error("Expected n_samples >= n_components but got n_components = " + String(kc) + ", n_samples = " + String(n))
    var rng = SplitMix64(seed)
    var L = BgL(kc, d)
    var cfg = (1 if pr.dp else 0) + (2 if pr.variational else 0) + 4 * pr.cov_type
    var has_m = len(pr.mean_prior) > 0
    var has_c = len(pr.cov_prior) > 0
    # the workspace: the scalar block, the caller's priors, a warm state and
    # GaussianMixture's inits, the input words as float-float
    var h = List[Float32](length=2 * L.total, fill=Float32(0))
    var sc0 = _scalars(pr, reg, tol, n, warm_lb if warm else Float64(0), warm)
    for t in range(2 * SC_SIZE):
        h[t] = sc0[t]
    if has_m:
        for a in range(d):
            _put(h, L.pm + a, pr.mean_prior[a])
    if has_c:
        for t in range(d * d):
            _put(h, L.pc + t, pr.cov_prior[t])
    if warm:
        for k in range(kc):
            _put(h, L.nk + k, best.nk[k])
            _put(h, L.wc0 + k, best.wc0[k])
            if k < len(best.wc1):
                _put(h, L.wc1 + k, best.wc1[k])
            _put(h, L.mp + k, best.mean_prec[k])
            _put(h, L.dof + k, best.dof[k])
        for t in range(kc * d):
            _put(h, L.means + t, best.means[t])
        for t in range(kc * d * d):
            _put(h, L.cov + t, best.cov[t])
            _put(h, L.pchol + t, best.pchol[t])
    for k in range(len(w_init)):
        _put(h, L.wi + k, w_init[k])
    for t in range(len(m_init)):
        _put(h, L.mi + t, m_init[t])
    for t in range(len(p_init)):
        _put(h, L.pi + t, p_init[t])
    var ws = ops.put(h)
    var wb = ops.zeros(2 * L.total)
    var xs = ops.put(x)
    var rs = ops.zeros(n * kc)
    var qs = ops.zeros(n * kc)
    var lpn = ops.zeros(n)
    var nks = ops.zeros(kc)
    var xks = ops.zeros(kc * d)
    var sks = ops.zeros(kc * d * d)
    var ms = ops.zeros(kc * d)
    var ps = ops.zeros(kc * d * d)
    var cs = ops.zeros(kc)
    var es = ops.zeros(2)
    if not (has_m and has_c):
        # the default priors: X's mean and np.cov(X.T), the moments of one
        # all-ones component, on the device
        var ones = ops.zeros(n)
        ops.onehot(ops.zeros_i(n), n, 1, True, ones)
        var n1 = ops.zeros(1)
        var x1 = ops.zeros(d)
        var s1 = ops.zeros(d * d)
        ops.moments(ones, xs, n, d, 1, Float32(0), n1, x1, s1)
        var aux = (1 if has_m else 0) + (2 if has_c else 0) + (4 if pr.cov_type == 3 else 0)
        ops.bgmm_step(ST_PRIOR, kc, d, cfg, aux, ws, x1, s1, -1)
    var estep1 = False
    comptime if BGMM_ESTEP1:
        estep1 = ops.fast_device()
    var ovr = (1 if len(w_init) > 0 else 0) + (2 if len(m_init) > 0 else 0)
    var max_lb = Float64(0)
    var have_best = False
    var best_iter = 0
    var converged_best = False
    var runs = 1 if warm else n_init
    for _init in range(runs):
        # the start, written on the device (lane cgr2-cluster): one-hot rows
        # at the k-means labels or the picked rows, or normalized uniforms
        var r0 = ops.zeros(n * kc)
        if warm:
            pass
        elif init_mode == 2 or init_mode == 3:
            var picks: List[Int]
            if init_mode == 2:
                picks = greedy_kmeans_pp_indices(ops, x, n, d, kc, rng)
            else:
                picks = _distinct_rows(rng, n, kc)
            var pi = List[Int32](capacity=kc)
            for k in range(kc):
                pi.append(Int32(picks[k]))
            ops.onehot(ops.put_i(pi), kc, kc, False, r0)
        elif init_mode == 1:
            # draw i * kc + k + 1 of the stream for cell (i, k), then the
            # stream moves past all n * kc draws
            ops.rand_resp(r0, n, kc, rng.state)
            rng.state = rng.state + UInt64(n * kc) * SPLITMIX_GAMMA
        else:
            var c = List[Float32]()
            var l = List[Int32]()
            _ = ops.kmeans(x, n, d, kc, 300, 1e-4, rng.next() >> 1, 1, INIT_KMEANS_PLUS_PLUS, c, l)
            ops.onehot(ops.put_i(l), n, kc, True, r0)
        if not warm:
            ops.set(ws, _scalars(pr, reg, tol, n, Float64(0), False))
            ops.moments(r0, xs, n, d, kc, reg, nks, xks, sks)
            _m_step(ops, kc, d, cfg, ws, nks, xks, sks)
            # sklearn GaussianMixture._initialize: weights_init, means_init and
            # precisions_init replace what the start responsibilities gave
            if ovr != 0:
                ops.bgmm_step(ST_OVR, kc, d, cfg, ovr, ws, -1, -1, -1)
            if len(p_init) > 0:
                # the precisions' inverse as covariances, then the usual factor
                ops.bgmm_step(ST_CHOL, kc, d, cfg, 1, ws, -1, -1, -1)
                ops.bgmm_step(ST_UUT, kc, d, cfg, 0, ws, -1, -1, -1)
                ops.bgmm_step(ST_CHOL, kc, d, cfg, 0, ws, -1, -1, -1)
        _derive(ops, kc, d, cfg, ws, ms, ps, cs)
        var converged = False
        var n_iter = 0
        for it in range(1, max_iter + 1):
            n_iter = it
            # E-step
            if estep1:
                ops.estep(xs, n, d, ms, ps, cs, kc, qs, rs, lpn)
            else:
                ops.gauss_q(xs, n, d, ms, ps, kc, qs)
                ops.resp(qs, cs, n, kc, lpn)
                ops.exp(qs, rs, n * kc)
            # M-step
            ops.moments(rs, xs, n, d, kc, reg, nks, xks, sks)
            _m_step(ops, kc, d, cfg, ws, nks, xks, sks)
            _derive(ops, kc, d, cfg, ws, ms, ps, cs)
            if pr.variational:
                # the entropy sum(resp * log_resp): the float-float fold
                ops.fold_into(rs, qs, -1, n * kc, FM_PROD, es)
            else:
                # sklearn GaussianMixture: the mean log-likelihood of THIS E-step
                ops.fold_into(lpn, -1, -1, n, FM_VAL, es)
            ops.bgmm_step(ST_D, kc, d, cfg, 0, ws, es, -1, -1)
            # the one wait per iteration: the scalar block
            var sc = ops.get(ws, 2 * SC_SIZE)
            if sc[2 * SC_ERR] != Float32(0):
                raise Error(BGMM_CHOL_ERROR)
            if sc[2 * SC_CONV] != Float32(0):
                converged = True
                break
        var sc = ops.get(ws, 2 * SC_SIZE)
        if sc[2 * SC_ERR] != Float32(0):
            raise Error(BGMM_CHOL_ERROR)
        var lb = _ff64(sc, SC_LB)
        if not have_best or lb > max_lb:
            max_lb = lb
            have_best = True
            best_iter = n_iter
            converged_best = converged
            ops.bgmm_step(ST_KEEP, kc, d, cfg, 0, ws, wb, -1, -1)
    # the best init: its weights, then the final E-step on its parameters
    ops.bgmm_step(ST_W, kc, d, cfg, 0, wb, -1, -1, -1)
    ops.bgmm_step(ST_F, kc, d, cfg, 0, wb, ms, ps, cs)
    ops.gauss_q(xs, n, d, ms, ps, kc, qs)
    ops.resp(qs, cs, n, kc, lpn)
    var ls = ops.zeros_i(n)
    ops.argmax_rows(qs, n, kc, ls)
    var hb = List[Float32]()
    ops.get_if(ls, n, wb, 2 * L.total, labels, hb)
    best = _state_of(hb, kc, d, pr.variational)
    return BgmmFit(max_lb, best_iter, converged_best)


def bgmm_score[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, kc: Int, means: List[Float32], pchol: List[Float32],
    c: List[Float32], mut log_resp: List[Float32], mut lpn_out: List[Float32],
    mut labels_out: List[Int32], flags: Int, mut proba_out: List[Float32], mut total: Float64,
) raises:
    """flags (lane apple-fast-py2mojo-cluster, what Python computed): bit 0
    `proba_out` = the pinned exp of every log responsibility (predict_proba),
    bit 1 `total` = the float-float fold of log_prob_norm (score)."""
    var xs = ops.put(x)
    var ms = ops.put(means)
    var ps = ops.put(pchol)
    var cs = ops.put(c)
    var qs = ops.zeros(n * kc)
    var lpn = ops.zeros(n)
    ops.gauss_q(xs, n, d, ms, ps, kc, qs)
    ops.resp(qs, cs, n, kc, lpn)
    var ls = ops.zeros_i(n)
    ops.argmax_rows(qs, n, kc, ls)
    if flags & 1:
        var pr = ops.zeros(n * kc)
        ops.exp(qs, pr, n * kc)
        proba_out = ops.get(pr, n * kc)
    if flags & 2:
        total = ops.sum_ff(lpn, -1, -1, n, FM_VAL)
    log_resp = ops.get(qs, n * kc)
    lpn_out = ops.get(lpn, n)
    labels_out = ops.get_i(ls, n)
