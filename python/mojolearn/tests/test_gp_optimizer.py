# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `GaussianProcessRegressor(optimizer="fmin_l_bfgs_b",
n_restarts_optimizer=k)` (2026-09-15). The reference is scikit-learn 1.9.0's
`_gpr.py` (the log marginal likelihood and its gradient, the restarts) and
`kernels.py` (theta, bounds, the gradient arms). The gradient is DEVIATION
2880 (`gaussian_process/host/gp_theta.mojo`), the optimizer DEVIATION 2881
(`python/mojolearn/_gp_optimizer.py`). On the GPU set it runs
`gaussian_process/estimator.mojo::gpr_lml_grad_host`; on a CPU-only install
the gp host binding's `gpr_grad_oracle.mojo` (inside the reference context).

    cd python && python3 -m mojolearn.tests.test_gp_optimizer

The REFERENCE arm prints the scikit-learn agreement (max absolute difference
of the optimized theta and of the likelihood) and checks it against bounds
that were set once from the float32 precision argument, never tuned.
Exit 2 naming the build script when the binding is unbuilt.
"""
import math
import sys

import numpy as np

from mojolearn._cpu_reference import reference_training
from mojolearn._gp_impl import RBF, ConstantKernel, GaussianProcessRegressor, Matern, WhiteKernel
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, run


def _data(seed=3, n=48, d=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (np.sin(1.5 * x[:, 0]) + 0.5 * x[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return x, y


def _fit(kernel, x, y, **kw):
    with reference_training():
        return GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", **kw).fit(x, y)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def arm_kernel(rep):
    k = ConstantKernel(2.0) * RBF([1.0, 3.0]) + WhiteKernel(0.5, "fixed")
    rep.check("KERNEL", k.n_dims == 3, "a constant, an ARD pair and a fixed white kernel give 3 entries", k.n_dims)
    rep.check("KERNEL", np.allclose(k.theta, np.log([2.0, 1.0, 3.0])), "theta is the log in leaf order", k.theta)
    rep.check("KERNEL", np.allclose(k.bounds, np.log([[1e-5, 1e5]] * 3)), "the default bounds are scikit-learn's")
    rep.check("KERNEL", k._free_flags() == [1, 1, 0, 0, 0], "one free flag per postfix node", k._free_flags())
    rep.check("KERNEL", GaussianProcessRegressor().kernel.n_dims == 0, "the default kernel is fixed, as scikit-learn's")
    m = Matern(1.0, (1e-2, 1e2), nu=2.5)
    rep.check("KERNEL", m.nu == 2.5 and np.allclose(m.bounds, [np.log([1e-2, 1e2])]),
              "Matern takes scikit-learn's (length_scale, length_scale_bounds, nu) order")


def arm_gradient(rep):
    """The analytic gradient against central differences of the same float32
    likelihood, and against scikit-learn's analytic gradient at the same theta."""
    x, y = _data(4, n=40)
    for name, k in (("rbf", ConstantKernel(1.5) * RBF([0.8, 1.2, 2.0]) + WhiteKernel(0.2)),
                    ("matern05", ConstantKernel(1.0) * Matern(1.1, nu=0.5) + WhiteKernel(0.1)),
                    ("matern15", ConstantKernel(0.7) * Matern([1.0, 0.6, 1.4], nu=1.5) + WhiteKernel(0.1)),
                    ("matern25", Matern(0.9, nu=2.5) + ConstantKernel(0.3) + WhiteKernel(0.2))):
        with reference_training():
            m = GaussianProcessRegressor(kernel=k).fit(x, y)
        th = np.array(k.theta)
        lml, g = m.log_marginal_likelihood(th, eval_gradient=True)
        h = 1e-2
        fd = []
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = h
            fd.append((m.log_marginal_likelihood(th + e) - m.log_marginal_likelihood(th - e)) / (2 * h))
        err = float(np.max(np.abs(np.array(g) - np.array(fd)) / (np.abs(fd) + 1.0)))
        rep.check("GRADIENT", err < 5e-3, f"{name}: the gradient matches central differences", err)
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor as SkGPR
            from sklearn.gaussian_process import kernels as skk
        except ImportError:
            continue
        sk = SkGPR(kernel=_sk_kernel(k, skk), alpha=2.0 ** -20, optimizer=None).fit(
            x.astype(np.float64), y.astype(np.float64))
        slml, sg = sk.log_marginal_likelihood(th, eval_gradient=True)
        gerr = float(np.max(np.abs(np.array(g) - sg) / (np.abs(sg) + 1.0)))
        rep.check("GRADIENT", gerr < 1e-3 and abs(lml - slml) < 1e-3 * (abs(slml) + 1.0),
                  f"{name}: the likelihood and gradient match scikit-learn's at the same theta", (gerr, lml, slml))


def _sk_kernel(k, skk):
    from mojolearn._gp_impl import _Combined
    if isinstance(k, _Combined):
        a, b = _sk_kernel(k.a, skk), _sk_kernel(k.b, skk)
        return a + b if k.sym == "+" else a * b

    def bnd(b):
        return "fixed" if b == "fixed" else (b[0][0], b[0][1])
    if isinstance(k, ConstantKernel):
        return skk.ConstantKernel(k.constant_value, bnd(k.constant_value_bounds))
    if isinstance(k, WhiteKernel):
        return skk.WhiteKernel(k.noise_level, bnd(k.noise_level_bounds))
    ls = k.length_scale if len(k.length_scale) > 1 else k.length_scale[0]
    if isinstance(k, RBF):
        return skk.RBF(ls, bnd(k.length_scale_bounds))
    return skk.Matern(ls, bnd(k.length_scale_bounds), nu=k.nu)


def arm_optimize(rep):
    x, y = _data(5)
    k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
    m = _fit(k, x, y)
    rep.check("OPTIMIZE", m.kernel_ is not m.kernel and m.kernel.theta == list(np.log([1.0, 1.0, 0.1])),
              "kernel_ is a new kernel and kernel is untouched")
    runs = m._optimizer_runs
    rep.check("OPTIMIZE", len(runs) == 1 and runs[0][0] > 0, "one run that took steps", runs)
    start = m.log_marginal_likelihood(np.log([1.0, 1.0, 0.1]))
    rep.check("OPTIMIZE", m.log_marginal_likelihood_value_ > start, "the likelihood rose from the start",
              (start, m.log_marginal_likelihood_value_))
    again = m.log_marginal_likelihood(m.kernel_.theta)
    rep.check("OPTIMIZE", again == m.log_marginal_likelihood_value_ == runs[0][3],
              "the fit's likelihood is bit for bit the optimizer's and a re-evaluation's",
              (again, m.log_marginal_likelihood_value_, runs[0][3]))
    params = m.kernel_._free_values()
    rep.check("OPTIMIZE", all(float(np.float32(p)) == p for p in params), "every hyperparameter is a float32")
    m2 = _fit(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1), x, y)
    rep.check("OPTIMIZE", _bits(m2.alpha_) == _bits(m.alpha_) and m2.kernel_.theta == m.kernel_.theta,
              "a second fit gives the same bits")
    with reference_training():
        none = GaussianProcessRegressor(kernel=k).fit(x, y)
    rep.check("OPTIMIZE", none.kernel_ is none.kernel and none._optimizer_runs == [],
              "optimizer=None fits the kernel passed")
    fixed = _fit(ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed"), x, y)
    rep.check("OPTIMIZE", fixed._optimizer_runs == [], "a kernel with no free hyperparameter is not optimized")


def arm_restarts(rep):
    x, y = _data(6)
    k = ConstantKernel(1.0) * Matern([1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(0.1)
    m = _fit(k, x, y, n_restarts_optimizer=2, random_state=7)
    runs = m._optimizer_runs
    rep.check("RESTARTS", len(runs) == 3, "the kernel's theta plus two restarts", runs)
    best = max(r[3] for r in runs)
    first = [r[3] for r in runs].index(best)
    rep.check("RESTARTS", m.log_marginal_likelihood_value_ == best, "the best likelihood wins")
    m2 = _fit(k, x, y, n_restarts_optimizer=2, random_state=7)
    rep.check("RESTARTS", m2.kernel_.theta == m.kernel_.theta and m2._optimizer_runs == runs,
              "the same random_state gives the same runs")
    m3 = _fit(k, x, y, n_restarts_optimizer=2, random_state=2 ** 40 + 7)
    rep.check("RESTARTS", m3._optimizer_runs[1:] != runs[1:], "the key's high word moves the restart starts")
    rep.check("RESTARTS", first == [r[3] for r in runs].index(best), "a tie goes to the earliest run")


def arm_reference(rep):
    """scikit-learn's optimized theta and likelihood on three fixtures, the
    agreement printed. Float64 L-BFGS-B there, the float32 likelihood and
    DEVIATION 2881's optimizer here; the bound is the float32 precision
    argument (a likelihood resolution near 1e-6 relative), not a tuned one."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor as SkGPR
        from sklearn.gaussian_process import kernels as skk
    except ImportError:
        rep.check("REFERENCE", True, "SKIPPED: scikit-learn is not installed")
        return
    fixtures = (
        ("rbf-white", 11, ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)),
        ("ard-matern25", 12, ConstantKernel(1.0) * Matern([1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(0.1)),
        ("matern15-white", 13, Matern(1.0, nu=1.5) + WhiteKernel(0.1)),
    )
    for name, seed, k in fixtures:
        x, y = _data(seed, n=64)
        m = _fit(k, x, y)
        sk = SkGPR(kernel=_sk_kernel(k, skk), alpha=2.0 ** -20).fit(x.astype(np.float64), y.astype(np.float64))
        dth = float(np.max(np.abs(np.array(m.kernel_.theta) - sk.kernel_.theta)))
        dl = abs(m.log_marginal_likelihood_value_ - sk.log_marginal_likelihood_value_)
        rep.check("REFERENCE", True, f"{name}: max |theta - sklearn theta| = {dth:.3e}, "
                  f"|lml - sklearn lml| = {dl:.3e} (ours {m.log_marginal_likelihood_value_:.6f})")
        rep.check("REFERENCE", dl < 1e-3 * (abs(sk.log_marginal_likelihood_value_) + 1.0),
                  f"{name}: the optimized likelihood agrees with scikit-learn's", dl)


def arm_refuse(rep):
    x, y = _data(8, n=24)
    rep.raises("REFUSE", NotImplementedError, "callable optimizer", "a callable optimizer",
               GaussianProcessRegressor, optimizer=lambda f, t, bounds: (t, 0.0))
    rep.raises("REFUSE", ValueError, "not one of", "an unknown optimizer name",
               GaussianProcessRegressor, optimizer="adam")
    rep.raises("REFUSE", ValueError, "random_state", "restarts with random_state=None",
               GaussianProcessRegressor, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=1)
    rep.raises("REFUSE", ValueError, "non-negative", "a negative n_restarts_optimizer",
               GaussianProcessRegressor, n_restarts_optimizer=-1)
    rep.raises("REFUSE", ValueError, "bound", "a non-positive bound", RBF, 1.0, (0.0, 1.0))
    rep.raises("REFUSE", ValueError, "bounds must be", "a scalar bound (Matern's old positional nu)",
               Matern, 1.0, 2.5)
    with reference_training():
        m = GaussianProcessRegressor(kernel=RBF(1.0) + WhiteKernel(0.1)).fit(x, y)
    rep.raises("REFUSE", ValueError, "free hyperparameters", "a theta of the wrong length",
               m.log_marginal_likelihood, [0.0])
    rep.raises("REFUSE", ValueError, "needs a theta", "eval_gradient without theta",
               m.log_marginal_likelihood, None, True)
    lml = m.log_marginal_likelihood([math.log(1e5), math.log(1e-5)])
    rep.check("REFUSE", lml == -math.inf or math.isfinite(lml),
              "an extreme theta answers a number or -inf, never raises", lml)


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_gp", "build_gp.sh")
    rep = Report("test_gp_optimizer")
    return run("test_gp_optimizer", [("KERNEL", arm_kernel), ("GRADIENT", arm_gradient), ("OPTIMIZE", arm_optimize),
                                     ("RESTARTS", arm_restarts), ("REFERENCE", arm_reference),
                                     ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
