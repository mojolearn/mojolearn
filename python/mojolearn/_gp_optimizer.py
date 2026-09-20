# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The bounded quasi-Newton optimizer behind
`GaussianProcessRegressor(optimizer="fmin_l_bfgs_b")` (2026-09-15).

PRIVATE MODULE. The reference is scikit-learn 1.9.0 `_gpr.py`'s
`_constrained_optimization`, which calls SciPy's L-BFGS-B
(`scipy.optimize.minimize(method="L-BFGS-B", jac=True, bounds=bounds)`) on
`-log_marginal_likelihood(theta)`.

DEVIATION 2881: A PROJECTED L-BFGS WITH EXPLICIT RULES, NOT SCIPY'S L-BFGS-B.
The answer is the same bits on every column (Apple, NVIDIA, AMD and the CPU
verifier), not SciPy's bits. Every rule below is a function of the objective's
bits alone:

  state        Python float64. CPython float `+ - * /`, `abs`, comparisons and
               `math.sqrt` are correctly rounded IEEE double with no
               contraction, so this module computes the same doubles on every
               host (DEVIATION 2833's argument). `sum()` is never used: every
               dot product is a written loop, index ascending from +0.0.
  objective    f = -lml and g = -grad, the float32 values the binding returns
               widened exactly. A failed factorization or a non-finite value
               is f = +inf (scikit-learn returns -inf likelihood, `_gpr.py:593`).
  start        theta0 clipped into the bounds.
  history      M = 10 pairs (SciPy's `maxcor`), newest replaces oldest. A pair
               is stored only when `s.y > EPS * y.y`, EPS = 2**-52.
  active set   entry i is ACTIVE when `x_i <= lo_i and g_i > 0` or
               `x_i >= hi_i and g_i < 0`; its direction entry is zero.
  direction    the two-loop recursion over the history applied to g with the
               active entries zeroed, scaled by `s.y / y.y` of the newest pair,
               negated, active entries zeroed. If `g.d >= 0` the history is
               cleared and `d = -g` (active entries zeroed).
  line search  backtracking on the PROJECTED path `x(t) = clip(x + t d)`:
               t = 1, except the very first step, t = min(1, 1 / max|d_i|);
               accept when f(x(t)) is finite and `f(x(t)) <= f + C1 * g.(x(t) - x)`,
               C1 = 1e-4; otherwise t = t / 2, at most MAX_LS = 20 trials.
               No curvature (Wolfe) condition.
  stops, in    `pg <= PGTOL` (1e-5, SciPy's `pgtol`) where
  this order   `pg = max_i |clip(x_i - g_i) - x_i|`, tested before a step;
               no descent direction; the line search exhausting MAX_LS or
               reaching a trial equal to x; after an accepted step,
               `f_old - f <= FTOL * max(|f_old|, |f|, 1)` with FTOL = 1e7 * 2**-52
               (SciPy's `factr` times machine epsilon); MAX_ITER = 200 accepted
               steps.

How it differs from SciPy's L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995): SciPy
finds a generalized Cauchy point along the projected steepest descent path and
then minimizes the quadratic model over the free variables, uses the compact
limited-memory matrix, and runs the More-Thuente line search with the strong
Wolfe conditions (`dcsrch`); it also caps function evaluations at 15000 and
iterations at 15000. This optimizer keeps only the two-loop direction with an
active-set mask, an Armijo backtracking search on the projected path and the
three stops above. On a smooth likelihood both approach the same local
maximum; the optimized theta and likelihood agree with scikit-learn's to the
precision the float32 likelihood allows, and can differ where the likelihood
has several maxima. Restarts (DEVIATION 2881, `gp_theta.mojo`) draw log-uniform
starting points from position-mapped Philox rather than NumPy's RandomState.
"""

from . import _portable_math as math

M = 10
EPS = 2.0 ** -52
C1 = 1e-4
MAX_LS = 20
PGTOL = 1e-5
FTOL = 1e7 * 2.0 ** -52
MAX_ITER = 200


def _dot(a, b):
    acc = 0.0
    for i in range(len(a)):
        acc = acc + a[i] * b[i]
    return acc


def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def minimize(fun, x0, lo, hi):
    """Minimize `fun(x) -> (f, g)` over the box `[lo, hi]` from `x0`.
    Returns `(x, f, n_iter, n_eval, stop)`, `stop` one of "pgtol",
    "no-descent", "line-search", "ftol", "max-iter", "nonfinite-start"."""
    n = len(x0)
    x = [_clip(float(x0[i]), lo[i], hi[i]) for i in range(n)]
    f, g = fun(x)
    n_eval = 1
    if not math.isfinite(f):
        return x, f, 0, n_eval, "nonfinite-start"
    hist_s, hist_y, hist_rho, hist_gamma = [], [], [], []
    n_iter = 0
    first = True
    while True:
        if n_iter >= MAX_ITER:
            return x, f, n_iter, n_eval, "max-iter"
        pg = 0.0
        for i in range(n):
            v = abs(_clip(x[i] - g[i], lo[i], hi[i]) - x[i])
            if v > pg:
                pg = v
        if pg <= PGTOL:
            return x, f, n_iter, n_eval, "pgtol"
        active = [(x[i] <= lo[i] and g[i] > 0.0) or (x[i] >= hi[i] and g[i] < 0.0) for i in range(n)]
        q = [0.0 if active[i] else g[i] for i in range(n)]
        k = len(hist_s)
        a = [0.0] * k
        for p in range(k - 1, -1, -1):
            a[p] = hist_rho[p] * _dot(hist_s[p], q)
            yp = hist_y[p]
            for i in range(n):
                q[i] = q[i] - a[p] * yp[i]
        if k > 0:
            gamma = hist_gamma[k - 1]
            q = [gamma * v for v in q]
        for p in range(k):
            b = hist_rho[p] * _dot(hist_y[p], q)
            sp = hist_s[p]
            for i in range(n):
                q[i] = q[i] + (a[p] - b) * sp[i]
        d = [0.0 if active[i] else -q[i] for i in range(n)]
        gd = _dot(g, d)
        if not gd < 0.0:
            hist_s, hist_y, hist_rho, hist_gamma = [], [], [], []
            d = [0.0 if active[i] else -g[i] for i in range(n)]
            gd = _dot(g, d)
            if not gd < 0.0:
                return x, f, n_iter, n_eval, "no-descent"
        t = 1.0
        if first:
            dmax = 0.0
            for i in range(n):
                if abs(d[i]) > dmax:
                    dmax = abs(d[i])
            if dmax > 1.0:
                t = 1.0 / dmax
        accepted = None
        for _ in range(MAX_LS):
            xt = [_clip(x[i] + t * d[i], lo[i], hi[i]) for i in range(n)]
            if xt == x:
                break
            ft, gt = fun(xt)
            n_eval += 1
            step = [xt[i] - x[i] for i in range(n)]
            if math.isfinite(ft) and ft <= f + C1 * _dot(g, step):
                accepted = (xt, ft, gt, step)
                break
            t = t / 2.0
        if accepted is None:
            return x, f, n_iter, n_eval, "line-search"
        xt, ft, gt, s = accepted
        y = [gt[i] - g[i] for i in range(n)]
        sy = _dot(s, y)
        yy = _dot(y, y)
        if sy > EPS * yy:
            hist_s.append(s)
            hist_y.append(y)
            hist_rho.append(1.0 / sy)
            # The next two-loop recursion used to recompute these same two
            # ascending dots on every iteration.  Cache their already-
            # computed quotient with the pair: identical operands, division,
            # and bits, without two extra Python loops over theta.
            hist_gamma.append(sy / yy)
            if len(hist_s) > M:
                hist_s.pop(0)
                hist_y.pop(0)
                hist_rho.pop(0)
                hist_gamma.pop(0)
        f_old = f
        x, f, g = xt, ft, gt
        n_iter += 1
        first = False
        if f_old - f <= FTOL * max(abs(f_old), abs(f), 1.0):
            return x, f, n_iter, n_eval, "ftol"
