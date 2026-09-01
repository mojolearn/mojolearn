# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.GaussianProcessRegressor`.

Written 2026-09-01, the day the surface landed -- the day the estimator
left `_NOT_YET`, where it had been withheld over a cross-vendor divergence
reading WITHDRAWN at `9835094e` (the divergent lines were a sabotage arm's
own block; `python/mojolearn/_gp_impl.py`'s header carries the history).
The model for this file is `test_svr_surface.py` and the house standard is
`gemm/PYTHON_SURFACE_GATE.md`.

WHAT THIS CLOSES. `gaussian_process/estimator.mojo`'s `gpr_fit_host` /
`gpr_predict_host`, `bindings/_mojolearn_gp.mojo`'s `gpr_fit` /
`gpr_predict` and `python/mojolearn/_gp_impl.py` put the GP lane in reach
of a Python caller. NOTHING IN THAT PATH IS COVERED BY THE LANE'S ELEVEN
CHECKS: those run inside Mojo (`pixi run check-gaussian-process`), against
their own oracle, and stop at the host entries. Everything here is
downstream of that point: a params list whose order is written out twice
and could be written out wrong once, a postfix kernel spec flattened on
this side and rebuilt on that side, output buffers the caller sizes, and
an `info` that must travel from fit to predict for the failed-fit refusal
to stay reachable.

WHAT IS ASSERTED AND WHAT IS REPORTED. Under `identical` the bitwise
comparison (one row predicted alone against that row in a batch) is
ASSERTED; under `fast` it is REPORTED, per `[[fast-is-not-identical]]`:
FAST promises speed only and its bits are allowed to move. The posterior
recovery, the shapes, the variance clamp bookkeeping and every refusal are
tolerance or name comparisons and are asserted in every tier.

WHAT THIS DOES NOT PROVE. It does not gate the arithmetic -- the lane's
own checks do, eleven per tier with nine runtime sabotage arms. It is one
machine and one vendor per run. And it makes no speed claim: the gp speed
ladder is UNRUN (HANDOFF_2026-09-01.md section 5).

HOW TO RUN IT
-------------
    # 1. build the extension (fast is the default tier)
    bash bindings/build_gp.sh

    # 2. the gate
    cd python && python3 -m mojolearn.tests.test_gp_surface

    # 3. the identical tier, which is where the bitwise arm is asserted
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gp.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_gp_surface
"""

import os
import sys

import numpy as np

from mojolearn import (
    ConstantKernel,
    GaussianProcessRegressor,
    Matern,
    RBF,
    WhiteKernel,
)
from mojolearn import _backend


class Report(object):
    def __init__(self):
        self.rows = []

    def check(self, arm, cond, what, detail=""):
        self.rows.append((bool(cond), arm,
                          what + (("" if cond else " -- " + detail)
                                  if detail else "")))
        return bool(cond)

    def report_only(self, arm, cond, what):
        self.rows.append((True, arm, "[REPORT, not asserted] %s: %s"
                          % (what, "same" if cond else "MOVED")))
        return bool(cond)

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal; each one here is
        made to fire by name and by message (`test_svr_surface.py`'s rule).
        A Mojo `Error` crossing the binding is caught as `Exception`."""
        try:
            fn(*a, **kw)
        except exc_type as exc:
            if needle in str(exc):
                return self.check(arm, True, what)
            return self.check(arm, False, what,
                              "raised %s but the message does not contain "
                              "%r: %s" % (exc_type.__name__, needle, exc))
        except Exception as exc:  # noqa: BLE001 - the wrong exception is data
            return self.check(arm, False, what, "raised %s, want %s: %s"
                              % (type(exc).__name__, exc_type.__name__, exc))
        return self.check(arm, False, what, "IS INERT: the call was ACCEPTED")

    def bits_equal(self, arm, got, want, what, assert_it):
        same = (np.ascontiguousarray(got).tobytes()
                == np.ascontiguousarray(want).tobytes())
        if assert_it:
            return self.check(arm, same, what, "bits moved")
        return self.report_only(arm, same, what)

    @property
    def failures(self):
        return [r for r in self.rows if not r[0]]

    def render(self, out):
        arm = None
        for ok, a, what in self.rows:
            if a != arm:
                out.write("\n  %s\n" % a)
                arm = a
            out.write("    %s %s\n" % ("pass" if ok else "FAIL", what))
        out.write("\n  %d checks, %d failed\n"
                  % (len(self.rows), len(self.failures)))


# HASHED, NOT `np.random` -- the same splitmix64-shaped generator
# `test_svr_surface.py` uses, so the fixture is a pure function of
# `(count, salt)` on every host and `uniform-test-data-hides-permutation`
# does not apply.
_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


def hashed_unit(count, salt):
    idx = np.arange(count, dtype=np.uint64) + np.uint64(1)
    z = (idx * np.uint64(_GOLDEN) + np.uint64(salt & _MASK64))
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(_MIX_A)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(_MIX_B)
    z = z ^ (z >> np.uint64(31))
    q = (z >> np.uint64(43)).astype(np.int64)
    return ((q - 1048576).astype(np.float32) / np.float32(1048576.0))


#: 16 points, 2 features -- the shape of the lane's own
#: `check_posterior_recovers_training` fixture class, small enough that a
#: unit-length-scale RBF correlates the points without making K singular.
N, D = 16, 2
#: The lane's recovery bound: worst |mean - y| against 2^-14 on 16 planted
#: observations (gaussian_process/README.md's Status transcript).
BOUND = 2.0 ** -14


def main(out=sys.stdout):
    rep = Report()
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    assert_bits = mode == "identical"

    x = hashed_unit(N * D, 0x67700001).reshape(N, D)
    y = hashed_unit(N, 0x67700002)

    # -- PROVENANCE: which binary answers, and the tier it reports --------
    arm = "PROVENANCE"
    gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=2.0 ** -20)
    rep.check(arm, gp._extension() is not None, "the _mojolearn_gp binding loads")
    rep.check(arm, gp.numeric_mode_used() == mode,
              "the binding's tier is the requested tier",
              "%r vs %r" % (gp.numeric_mode_used(), mode))

    # -- REFUSALS: every guard on this surface made to fire by name -------
    arm = "REFUSALS"
    rep.raises(arm, NotImplementedError, "DEVIATION 1761",
               "optimizer='fmin_l_bfgs_b' (sklearn's default) is refused",
               GaussianProcessRegressor, optimizer="fmin_l_bfgs_b")
    rep.raises(arm, NotImplementedError, "n_restarts_optimizer",
               "n_restarts_optimizer=3 is refused",
               GaussianProcessRegressor, n_restarts_optimizer=3)
    rep.raises(arm, NotImplementedError, "DEVIATION 1764",
               "normalize_y=True is refused",
               GaussianProcessRegressor, normalize_y=True)
    rep.raises(arm, NotImplementedError, "copy_X_train",
               "copy_X_train=False is refused",
               GaussianProcessRegressor, copy_X_train=False)
    rep.raises(arm, NotImplementedError, "random_state",
               "random_state=0 is refused",
               GaussianProcessRegressor, random_state=0)
    rep.raises(arm, TypeError, "composition",
               "a non-Kernel kernel object is refused",
               GaussianProcessRegressor, kernel="rbf")
    rep.raises(arm, ValueError, "must be 1-D",
               "a 2-D y is refused by name",
               GaussianProcessRegressor().fit, x, y.reshape(N, 1))
    rep.raises(arm, NotImplementedError, "DEVIATION 1759",
               "predict(return_cov=True) is refused",
               gp.predict, x, **{"return_cov": True})
    rep.raises(arm, NotImplementedError, "NOT PORTED",
               "sample_y is refused with the closure condition",
               gp.sample_y, x)
    # The two below are MOJO refusals and are the reason the surface judges
    # neither value: they would be unreachable if Python judged first.
    rep.raises(arm, Exception, "CLOSED FORMS",
               "Matern nu=0.7 is refused at fit, in Mojo, by name",
               GaussianProcessRegressor(kernel=Matern(1.0, nu=0.7)).fit, x, y)
    rep.raises(arm, Exception, "alpha",
               "alpha=NaN is refused at fit, in Mojo, by name",
               GaussianProcessRegressor(alpha=float("nan")).fit, x, y)

    # -- POSTERIOR: the planted recovery, the lane's own bound ------------
    # FIXTURE CORRECTED 2026-09-01: at lengthscale 1.0 this fixture's
    # kernel matrix has cond ~1.2e6 and the 2^-14 recovery bound was
    # never achievable -- sklearn in FLOAT64 on the identical data reads
    # worst |mean - y| = 0.0280 (ours read 0.0270, closer than the
    # reference). The fixture was the defect, not the assertion (the
    # fix-the-fixture rule). At lengthscale 0.25 the matrix conditions
    # to ~253 and sklearn float64 recovers to 2.39e-05 against the
    # 6.1e-05 bound -- honest with margin, and the off-diagonal
    # coupling stays real (cond >> 1), so a broken kernel still fails.
    arm = "POSTERIOR"
    k = ConstantKernel(1.0) * RBF([0.25, 0.25]) + WhiteKernel(0.0)
    model = GaussianProcessRegressor(kernel=k).fit(x, y)
    rep.check(arm, model.info_ == 0, "the factorization succeeded",
              "info_=%d" % model.info_)
    rep.check(arm, model.L_.shape == (N, N) and model.alpha_.shape == (N,),
              "L_ is (n, n) and alpha_ (the dual vector) is (n,)")
    mean, std = model.predict(x, return_std=True)
    worst = float(np.max(np.abs(mean - y)))
    rep.check(arm, worst <= BOUND,
              "the posterior mean recovers the training targets",
              "worst |mean - y| = %g against %g" % (worst, BOUND))
    rep.check(arm, bool((std >= 0.0).all()), "every std is non-negative")
    rep.check(arm, model.clamped_.shape == (N,)
              and int(model.clamped_.sum()) == model.n_clamped_,
              "the per-point clamp flags and their count agree "
              "(DEVIATION 1760)")
    lml = model.log_marginal_likelihood()
    rep.check(arm, np.isfinite(lml)
              and lml == model.log_marginal_likelihood_value_,
              "log_marginal_likelihood() is the fit-time value and finite")

    # -- INVARIANCE: one row alone against that row in a batch ------------
    # Bitwise ASSERTED under identical, REPORTED under fast
    # ([[fast-is-not-identical]]: a bitwise question has no right answer
    # on the fast tier).
    arm = "INVARIANCE"
    alone = model.predict(x[:1])
    rep.bits_equal(arm, alone, model.predict(x)[:1],
                   "row 0 predicted alone == row 0 in the batch",
                   assert_bits)

    # -- FAILED FIT: info is a RESULT, and nothing solves past it ---------
    # Two identical rows with alpha=0 are exactly singular; the lane's
    # check_duplicate_inputs_need_the_ridge is the Mojo-side twin.
    arm = "FAILED-FIT"
    xdup = x.copy()
    xdup[1] = xdup[0]
    failed = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.0).fit(xdup, y)
    rep.check(arm, failed.info_ != 0,
              "duplicate rows with alpha=0 fail the factorization "
              "(info_ != 0), reported rather than raised (DEVIATION 1634)")
    rep.raises(arm, Exception, "FAILED fit",
               "predict on the failed fit is refused, in Mojo, by name",
               failed.predict, x)
    rep.raises(arm, RuntimeError, "DEVIATION 1634",
               "log_marginal_likelihood on the failed fit is refused",
               failed.log_marginal_likelihood)

    rep.render(out)
    out.write("\n")
    if rep.failures:
        out.write("test_gp_surface: RED. %d checks failed.\n"
                  % len(rep.failures))
        return 1
    if not assert_bits:
        out.write(
            "test_gp_surface: the %s arms passed. The posterior recovery,\n"
            "the shapes, the clamp bookkeeping, the failed-fit path and the\n"
            "refusals hold. THE BITWISE ARM WAS REPORTED, NOT ASSERTED:\n"
            "this tier promises speed only. Re-run under\n"
            "MOJOLEARN_NUMERIC_MODE=identical for the asserted form.\n"
            % mode)
        return 0
    out.write(
        "test_gp_surface: GREEN. The Python surface recovers the planted\n"
        "posterior to the lane's own bound, one row alone is bit-identical\n"
        "to that row in a batch, a failed factorization is a named result\n"
        "and both downstream refusals fire, and every guard on the path\n"
        "fires by name. One box, one vendor: the cross-vendor statement\n"
        "belongs to the lane's cards, not to this file.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
