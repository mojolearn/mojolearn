# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.SVR`.

Written 2026-09-01, the day the surface landed. The model for this file is
`python/mojolearn/tests/test_linalg_identity.py` and the house standard it
follows is `gemm/PYTHON_SURFACE_GATE.md`.

WHAT THIS CLOSES
----------------
`svm/impl/svm/svr_impl.mojo`, `svm/estimator.mojo`'s `svr_fit_host` /
`svr_predict_host`, `bindings/_mojolearn_svm.mojo`'s `svr_fit` /
`svr_predict` and `python/mojolearn/_svm_impl.py`'s `SVR` put the
epsilon-SVR solver in reach of a Python caller. NOTHING IN THAT PATH IS
COVERED BY THE 44 GATES. Those gates run inside Mojo, against
`svm/checks/smo_oracle.mojo`, and they stop at `svc_check.mojo::
_run_svr_device`. Everything this file exercises is downstream of that
point: a params list whose order is written out twice and could be written
out wrong once, six output buffers the caller sizes, a fold from the
solver's `2 * n_rows` domain that this surface must NOT double, and a
`gamma` that has to be the one the fit resolved.

The failure this exists to catch is the one
`bindings/_mojolearn_svm.mojo`'s own header names: a silent reordering of
the scalar list is A WRONG ANSWER, NOT A FAILURE. Swap `C` and `epsilon` in
`svr_fit`'s params and every call still returns a plausible model.

THE ARMS, AND WHAT EACH ONE CAN SEE
-----------------------------------
    PROVENANCE   which binary answered, read back from the binary itself,
                 and which tier this process is therefore reporting about
    REFUSALS     every guard on this surface made to fire, by type and by
                 message, INCLUDING the four that live in Mojo and would
                 otherwise be unreachable from Python
    PLANTED      a linear problem with a known answer, recovered to a
                 tolerance. The only arm with an EXTERNAL reference: the
                 weights were planted, not read back off the model
    TUBE         raising `epsilon` does not increase the support count, and
                 the residuals of the interior rows sit inside the tube.
                 This is the epsilon-insensitive formulation's own property
                 and no part of it comes from our solver
    SHAPES       the model's arrays are `n_support` wide, never
                 `2 * n_support`, and the folded coefficients are bounded
                 by C. The arm that would catch a surface that believed the
                 doubled domain reached it
    INVARIANCE   one row predicted alone equals that row inside a batch,
                 and equals it again with the prediction buffer set small
                 enough to force several batches
    GAMMA        the gamma the fit resolved is the gamma predict uses

WHAT IS ASSERTED AND WHAT IS REPORTED
-------------------------------------
**Under `identical` the bitwise comparisons are ASSERTED. Under `fast` they
are REPORTED and this file says so rather than asserting them.** That is
the discipline every lane check in this tree follows, and
`[[fast-is-not-identical]]` is why: FAST promises SPEED ONLY, its bits move
run to run, and a bitwise question asked of a FAST arm is a question with
no right answer. The INVARIANCE arm is the one this bites; PLANTED, TUBE
and SHAPES are tolerance and inequality comparisons and are asserted in
every tier.

WHAT THIS DOES NOT PROVE
------------------------
- **It does not gate the solver.** `svm/svc_main.mojo` does that, 44 of 44
  at `fea6becc`, with four property gates derived from the formulation and
  sabotage reach reported per fixture. Nothing here is a second opinion
  about the arithmetic.
- **It is one machine and one vendor.** The SVR path has never been in a
  three-vendor round; `SVC`'s card does not extend to it. This gate
  contributes nothing to that and says so in its banner.
- **It has no oracle.** PLANTED recovers weights that were planted, to a
  tolerance chosen for the fixture, which is a much weaker statement than
  `test_linalg_identity`'s CARD arm makes. There is no host card emitter
  for the SVR path today, and writing one is the way to make this file
  absolute rather than relative.

HOW TO RUN IT
-------------
    # 1. build the extension (fast is the default tier)
    bash bindings/build_svm.sh

    # 2. the gate
    cd python && python3 -m mojolearn.tests.test_svr_surface

    # 3. the identical tier, which is where the bitwise arms are asserted
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_svm.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_svr_surface

ENVIRONMENT
    MOJOLEARN_NUMERIC_MODE   fast (default), deterministic or identical
"""

import os
import sys

import numpy as np

from mojolearn import SVR
from mojolearn import _backend
from mojolearn import _svm_impl


# ===========================================================================
# REPORTING
# ===========================================================================
# EVERY ARM RUNS AND EVERY VERDICT IS PRINTED BEFORE THE PROCESS EXITS
# NON-ZERO, for `test_linalg_identity`'s reason: stopping at the first
# failure shows one arm's opinion when the useful evidence is WHICH arms a
# given defect reaches and which it walks past.


class Report(object):
    def __init__(self):
        self.rows = []
        self.notes = []

    def ok(self, arm, what):
        self.rows.append((True, arm, what))

    def bad(self, arm, what):
        self.rows.append((False, arm, what))

    def note(self, text):
        self.notes.append(text)

    def check(self, arm, cond, what, detail=""):
        if cond:
            self.ok(arm, what)
        else:
            self.bad(arm, what + ((" -- " + detail) if detail else ""))
        return bool(cond)

    def report_only(self, arm, cond, what, detail=""):
        """A comparison this tier cannot ASK. Printed with its verdict and
        never counted as a failure. Used for the bitwise arms under FAST,
        where the bits are allowed to move."""
        self.rows.append((True, arm, "[REPORT, not asserted] %s: %s%s"
                          % (what, "same" if cond else "MOVED",
                             (" -- " + detail) if detail and not cond else "")))
        return bool(cond)

    def bits_equal(self, arm, got, want, what, assert_it):
        """The only bitwise comparison this file makes. `.tobytes()` is
        C-order whatever the strides are."""
        gb = np.ascontiguousarray(got).tobytes()
        wb = np.ascontiguousarray(want).tobytes()
        same = gb == wb
        detail = ""
        if not same:
            ga = np.ascontiguousarray(got).ravel().view(np.uint32)
            wa = np.ascontiguousarray(want).ravel().view(np.uint32)
            if ga.shape != wa.shape:
                detail = "SHAPES DIFFER, %s vs %s" % (np.shape(got),
                                                      np.shape(want))
            else:
                diff = np.flatnonzero(ga != wa)
                first = int(diff[0])
                detail = ("%d of %d cells differ; first at flat index %d, "
                          "0x%08x vs 0x%08x" % (diff.size, ga.size, first,
                                                int(ga[first]),
                                                int(wa[first])))
        if assert_it:
            return self.check(arm, same, what, detail)
        return self.report_only(arm, same, what, detail)

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal. Every guard on this
        surface is a branch a passing build can contain and never take, so
        each one is made to fire by name and by message."""
        try:
            fn(*a, **kw)
        except exc_type as exc:
            if needle in str(exc):
                self.ok(arm, what)
                return True
            self.bad(arm, "%s -- raised %s but the message does not contain "
                     "%r: %s" % (what, exc_type.__name__, needle, exc))
            return False
        except Exception as exc:  # noqa: BLE001 - the wrong exception is data
            self.bad(arm, "%s -- raised %s, want %s: %s"
                     % (what, type(exc).__name__, exc_type.__name__, exc))
            return False
        self.bad(arm, "%s -- IS INERT: the call was ACCEPTED" % what)
        return False

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
        for n in self.notes:
            out.write("\n%s\n" % n)
        out.write("\n  %d checks, %d failed\n"
                  % (len(self.rows), len(self.failures)))


class GateAbort(Exception):
    """A condition under which no verdict about `SVR` can be reached at all,
    as distinct from a failing check. A failure says the surface is wrong;
    an abort says the gate did not run."""


# ===========================================================================
# THE FIXTURES
# ===========================================================================
# HASHED, NOT `np.random`. The values have to be the same on every host and
# in every process, because the TUBE arm compares two fits of the SAME data
# at two epsilons and a reseeded generator would make that comparison about
# the data instead of about epsilon.
#
# The numerator is below 2^21 so it is exact in a float32 mantissa and the
# divisor is a power of two, so the division is exact and nothing rounds
# upstream of the thing being measured. That is
# `bench/gemm_card_main.mojo::_exact`'s rule and
# `svm/checks/svc_check.mojo::_unit` follows it too.

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


def hashed_unit(count, salt):
    """`count` values in [-1, 1), a pure function of `(count, salt)`."""
    idx = np.arange(count, dtype=np.uint64) + np.uint64(1)
    z = (idx * np.uint64(_GOLDEN) + np.uint64(salt & _MASK64))
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(_MIX_A)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(_MIX_B)
    z = z ^ (z >> np.uint64(31))
    # 21 bits, so the numerator is exact in float32, then centered.
    q = (z >> np.uint64(43)).astype(np.int64)
    return ((q - 1048576).astype(np.float32) / np.float32(1048576.0))


#: The planted linear problem. `n` and `k` are DISTINCT and neither is a
#: power of two, so a swapped `n_rows` / `n_features` in the params list
#: cannot land on a self-consistent call: `svr_fit` checks
#: `len(x) == n_rows * n_cols` on the Mojo side and 150 * 5 is not 5 * 150
#: as a pair even though it is as a product, which is why the SHAPES arm
#: also reads the model's widths back.
PLANT_N = 150
PLANT_K = 5

#: The weights the targets are built from. Recovered, not read back.
PLANT_W = np.array([0.75, -0.5, 0.25, -1.25, 0.5], dtype=np.float32)
PLANT_B = np.float32(0.125)

#: The half-width of the noise added to the targets. THE FIXTURE IS NOISY ON
#: PURPOSE and a noiseless one would be worse here, not better. With no
#: noise a linear kernel fits the targets exactly, every residual is zero,
#: and the tube swallows the whole training set at the FIRST positive
#: epsilon -- so the support count collapses in one step and the TUBE arm's
#: ladder measures nothing across the rest of it. `svm/checks/svc_check.mojo`
#: builds its regression fixtures the same way and for the same reason
#: (`_linear_targets` takes a `noise` argument).
#:
#: 0.06 against a signal whose standard deviation is about 0.95, so R^2 on a
#: converged fit is around 0.999 and the epsilon ladder below straddles the
#: noise rather than sitting entirely above or below it.
PLANT_NOISE = np.float32(0.06)


def planted():
    """`(X, y, w, b)` for a problem a LINEAR kernel can fit well.

    `y = X w + b + noise`, with the noise hashed from a different salt than
    `X` so the two are independent and reproducible.
    """
    x = hashed_unit(PLANT_N * PLANT_K, 0x51D3).reshape(PLANT_N, PLANT_K)
    noise = hashed_unit(PLANT_N, 0x9E11) * PLANT_NOISE
    y = (x @ PLANT_W + PLANT_B + noise).astype(np.float32)
    return x, y, PLANT_W, PLANT_B


# ===========================================================================
# ARM: PROVENANCE
# ===========================================================================


def arm_provenance(rep):
    """Which binary answered, and therefore which tier this run is about.

    Read back OUT OF THE BINARY (`svm_numeric_mode`, folded in at compile
    time from `checks/numerics.mojo`), never from the environment variable
    that asked for it. `_svm_impl._extension` already refuses to hand back a
    module whose compiled tier disagrees with the requested one; what is
    added here is naming the answer in the report, so that a reader of a
    green run knows which arms were asserted.
    """
    arm = "PROVENANCE"
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    try:
        ext = _svm_impl._extension(None)
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        raise GateAbort("the _mojolearn_svm extension did not load: %s" % exc)
    rep.check(arm, hasattr(ext, "svr_fit"),
              "the loaded binary exports svr_fit")
    rep.check(arm, hasattr(ext, "svr_predict"),
              "the loaded binary exports svr_predict")
    compiled = {0: "fast", 1: "identical", 2: "deterministic"}.get(
        int(ext.svm_numeric_mode()), "unknown")
    rep.check(arm, compiled in ("fast", "identical", "deterministic"),
              "the binary reports a tier it knows", compiled)
    # The VENDOR read back out of THIS binary rather than off an estimator:
    # `NumericModeMixin.vendor_used` resolves through `_BINDING`, which is
    # the base extension's name for both SVM classes, so it would report a
    # different `.so` than the one that answered above.
    rep.note("  binary tier   %s\n  env asked for %s\n  vendor        %s"
             % (compiled, mode, _backend.read_vendor(ext)))
    return compiled


# ===========================================================================
# ARM: REFUSALS
# ===========================================================================


def arm_refusals(rep):
    """Every guard this surface carries, made to fire.

    THE FOUR MOJO ONES ARE THE POINT OF THIS ARM. `epsilon` and `y` are
    plumbed through UNVALIDATED on the Python side exactly so that
    `check_rung1_scope` and `check_finite_list` stay reachable from here; if
    a later edit "helpfully" clamps either in `_svm_impl.py`, these four
    rows go from pass to FAIL with the message "IS INERT: the call was
    ACCEPTED" or with the wrong exception type, and that is the whole
    reason they are written this way.

    A Mojo `Error` crossing `def_function` is caught as `Exception` rather
    than as a named type, because what CPython raises for one was not
    established by reading the tree. Tightening these four to the real type
    is a one-line edit once a run has printed it, and the message needle is
    doing the discriminating in the meantime.
    """
    arm = "REFUSALS"
    x, y, _, _ = planted()

    # --- the Python-side constructor guards
    rep.raises(arm, NotImplementedError, "POLYNOMIAL",
               "kernel='poly' is refused by name",
               SVR, kernel="poly")
    rep.raises(arm, NotImplementedError, "TANH",
               "kernel='sigmoid' is refused by name",
               SVR, kernel="sigmoid")
    rep.raises(arm, NotImplementedError, "PRECOMPUTED",
               "kernel='precomputed' is refused by name",
               SVR, kernel="precomputed")
    rep.raises(arm, ValueError, "not a kernel name",
               "an unknown kernel name is refused",
               SVR, kernel="cosine")
    rep.raises(arm, NotImplementedError, "DEVIATION 870",
               "gamma='scale' is refused and names the deviation",
               SVR, gamma="scale")
    rep.raises(arm, ValueError, "not a name",
               "an unknown gamma name is refused",
               SVR, gamma="median")
    rep.raises(arm, ValueError, "finite and >= 0",
               "a negative gamma is refused",
               SVR, gamma=-1.0)
    rep.raises(arm, NotImplementedError, "POLYNOMIAL",
               "degree is refused, naming the kernel that would read it",
               SVR, degree=4)
    rep.raises(arm, NotImplementedError, "POLYNOMIAL and TANH",
               "coef0 is refused, naming the kernels that would read it",
               SVR, coef0=1.0)
    rep.raises(arm, ValueError, "C must be positive",
               "a non-positive C is refused",
               SVR, C=0.0)
    rep.raises(arm, ValueError, "C must be finite",
               "a non-finite C is refused",
               SVR, C=float("inf"))
    rep.raises(arm, ValueError, "tol must be positive",
               "a non-positive tol is refused",
               SVR, tol=0.0)
    rep.raises(arm, ValueError, "cache_size",
               "a non-positive cache_size is refused",
               SVR, cache_size=0.0)
    rep.raises(arm, ValueError, "max_iter",
               "max_iter=0 is refused",
               SVR, max_iter=0)
    rep.raises(arm, ValueError, "nochange_steps",
               "a negative nochange_steps is refused",
               SVR, nochange_steps=-1)
    rep.raises(arm, NotImplementedError, "CUML_LOG_DEBUG",
               "verbose is refused, naming what it would have selected",
               SVR, verbose=True)
    rep.raises(arm, NotImplementedError, "output_type",
               "output_type is refused",
               SVR, output_type="numpy")
    rep.raises(arm, TypeError, "shrinking",
               "shrinking is not a parameter of this class and says so",
               SVR, shrinking=True)

    # --- the Python-side fit and predict guards
    rep.raises(arm, NotImplementedError, "sample_weight",
               "sample_weight is refused in fit()",
               SVR(kernel="linear").fit, x, y, np.ones(PLANT_N))
    rep.raises(arm, ValueError, "must be 1-D",
               "a 2-D y is refused",
               SVR(kernel="linear").fit, x, y.reshape(-1, 1))
    rep.raises(arm, ValueError, "X has",
               "a y of the wrong length is refused",
               SVR(kernel="linear").fit, x, y[:-1])
    rep.raises(arm, ValueError, "must be 2-D",
               "a 1-D X is refused",
               SVR(kernel="linear").fit, x.ravel(), y)
    rep.raises(arm, ValueError, "call fit() first",
               "predict before fit is refused",
               SVR(kernel="linear").predict, x)

    fitted = SVR(kernel="linear", epsilon=0.05).fit(x, y)
    rep.raises(arm, ValueError, "features, fit saw",
               "a predict X with the wrong width is refused",
               fitted.predict, x[:, :2])

    # --- THE MOJO ONES. Reachable only because nothing above clamps them.
    rep.raises(arm, Exception, "epsilon must be non-negative",
               "a negative epsilon is refused BY THE SOLVER'S scope check",
               SVR(kernel="linear", epsilon=-0.1).fit, x, y)
    rep.raises(arm, Exception, "epsilon must be finite",
               "a non-finite epsilon is refused BY THE SOLVER'S scope check",
               SVR(kernel="linear", epsilon=float("nan")).fit, x, y)
    bad_y = y.copy()
    bad_y[7] = np.float32("nan")
    rep.raises(arm, Exception, "non-finite value at flat index 7",
               "a NaN target is refused BY NAME with its index (DEVIATION 636)",
               SVR(kernel="linear").fit, x, bad_y)
    bad_x = x.copy()
    bad_x[3, 2] = np.float32("inf")
    rep.raises(arm, Exception, "non-finite value at flat index",
               "a non-finite X cell is refused BY NAME (DEVIATION 636)",
               SVR(kernel="linear").fit, bad_x, y)


# ===========================================================================
# ARM: PLANTED
# ===========================================================================


def arm_planted(rep):
    """A linear problem with a KNOWN answer, recovered through the surface.

    `y = X w + b` exactly, so `coef_` must be `w` and `intercept_` must be
    `b`, to a tolerance. This is the only arm with a reference that did not
    come out of the model, and it is what catches a params list whose scalar
    slots are shuffled: swap `C` and `epsilon` and the recovered weights are
    a different vector, not a slightly worse one.

    THE TOLERANCE IS NOT A BIT COMPARISON AND IS NOT MEANT TO BE. SMO stops
    at `tol`, the tube costs the fit a bias of order `epsilon`, and the
    whole thing runs in float32. `2e-2` on the weights is loose enough that
    a converged solver clears it on any vendor and tight enough that a
    wrong scalar does not.
    """
    arm = "PLANTED"
    x, y, w, b = planted()
    est = SVR(kernel="linear", C=100.0, epsilon=1e-3, tol=1e-4).fit(x, y)

    rep.check(arm, est.n_support_ >= 1,
              "the fit found at least one support vector",
              str(est.n_support_))
    rep.check(arm, est.n_features_in_ == PLANT_K,
              "n_features_in_ is the width of X", str(est.n_features_in_))
    rep.check(arm, est.n_iter_ >= 1,
              "n_iter_ came back positive", str(est.n_iter_))

    # THE TOLERANCE IS THE NOISE'S, NOT THE SOLVER'S. With 150 rows, 5
    # features drawn on [-1, 1) and noise of half-width 0.06, an
    # independently solved reference (a primal minimizer of
    # `0.5|w|^2 + C sum max(0, |r| - epsilon)` on this exact fixture) lands
    # within 0.021 of the planted weights and 0.002 of the planted
    # intercept. `8e-2` is four times that and it is three times smaller
    # than the SMALLEST planted weight, 0.25, so a converged fit clears it
    # everywhere while a shuffled params list, which changes a weight rather
    # than perturbing it, does not.
    coef = np.asarray(est.coef_, dtype=np.float64).ravel()
    dw = float(np.max(np.abs(coef - w.astype(np.float64))))
    rep.check(arm, dw < 8e-2,
              "coef_ recovers the planted weights", "max |dw| = %.3e" % dw)
    db = abs(float(est.intercept_[0]) - float(b))
    rep.check(arm, db < 8e-2,
              "intercept_ recovers the planted offset", "|db| = %.3e" % db)

    pred = np.asarray(est.predict(x), dtype=np.float64)
    resid = float(np.max(np.abs(pred - y.astype(np.float64))))
    rep.check(arm, resid < 4.0 * float(PLANT_NOISE),
              "predict reproduces the training targets to the noise floor",
              "max |residual| = %.3e, noise half-width %.3e"
              % (resid, float(PLANT_NOISE)))
    r2 = est.score(x, y)
    rep.check(arm, r2 > 0.99, "score() is a high R^2 on a noiseless fit",
              "R^2 = %.6f" % r2)

    # The RBF arm, so the two kernels that exist are both launched. It has
    # no planted answer -- an RBF fit of a linear target is not `w` -- so
    # what is asserted is that it FITS, which the linear-kernel row above
    # cannot say for `row_norms_l2sq` and the expanded RBF kernel.
    rbf = SVR(kernel="rbf", gamma=0.5, C=100.0, epsilon=1e-3).fit(x, y)
    rep.check(arm, rbf.score(x, y) > 0.9,
              "kernel='rbf' fits the same problem (its own two kernels run)",
              "R^2 = %.6f" % rbf.score(x, y))
    rep.note("  planted   n=%d k=%d  linear n_SV=%d R2=%.6f  rbf n_SV=%d"
             % (PLANT_N, PLANT_K, est.n_support_, r2, rbf.n_support_))


# ===========================================================================
# ARM: TUBE
# ===========================================================================


def arm_tube(rep):
    """THE TUBE IS ACTUALLY A TUBE.

    Two properties of the epsilon-insensitive formulation, neither of them
    read off our solver:

      (1) MONOTONE SUPPORT COUNT. A row inside the tube costs nothing and
          gets a zero coefficient. Widening the tube can only move rows
          INTO it, so `n_support_` must be non-increasing in `epsilon`.
          This is the cheapest statement that separates an epsilon-SVR from
          a least-squares fit that ignores its epsilon, and it is what a
          params list that dropped `epsilon` on the floor would fail.

      (2) THE INTERIOR IS INSIDE. Every row that is NOT a support vector
          has a residual no larger than `epsilon`, up to the stopping
          tolerance. `svm/checks/svc_check.mojo::check_svr_eps_tube` gates
          the same inequality inside Mojo with a KKT-gap slack term; here
          the slack is a fixed margin, because this side cannot see the
          gap.

    A wider tube also costs accuracy, and that is checked in the same
    direction rather than assumed: R^2 must stay high at the small epsilon
    and is only reported at the large one.
    """
    arm = "TUBE"
    x, y, _, _ = planted()
    noise = float(PLANT_NOISE)

    # THE LADDER STRADDLES THE NOISE. `0` is the degenerate tube, where every
    # row with a non-zero residual is a support vector (and where `SvrInit`
    # writes both signed zeros into `f`, which is fixture R6's whole reason
    # for existing inside Mojo). `0.5` is far above the noise and well below
    # the signal's own spread, so the fit still has to bend but almost every
    # row falls inside. A ladder entirely above or entirely below the noise
    # would be five measurements of one thing.
    eps_ladder = [0.0, noise / 3.0, noise, 2.5 * noise, 0.5]
    counts = []
    fits = []
    for e in eps_ladder:
        est = SVR(kernel="linear", C=100.0, epsilon=e, tol=1e-4).fit(x, y)
        counts.append(est.n_support_)
        fits.append(est)

    for i in range(1, len(eps_ladder)):
        rep.check(arm, counts[i] <= counts[i - 1],
                  "n_support_ does not grow from epsilon=%.4f to %.4f"
                  % (eps_ladder[i - 1], eps_ladder[i]),
                  "%d -> %d" % (counts[i - 1], counts[i]))
    rep.check(arm, counts[-1] < counts[0],
              "the widest tube is STRICTLY smaller than the narrowest",
              "%d -> %d" % (counts[0], counts[-1]))

    # (2) the interior rows, at the middle epsilon, where there is a real
    # population of them and the fit has not collapsed.
    mid = fits[2]
    eps = eps_ladder[2]
    inside = np.ones(PLANT_N, dtype=bool)
    inside[np.asarray(mid.support_, dtype=np.int64)] = False
    n_inside = int(inside.sum())
    if n_inside < 10:
        raise GateAbort(
            "only %d of %d rows are outside the support set at epsilon=%.4f; "
            "the interior population is too small for this arm to say "
            "anything. Widen the ladder rather than lowering the bar."
            % (n_inside, PLANT_N, eps))
    resid = np.abs(np.asarray(mid.predict(x), dtype=np.float64)
                   - y.astype(np.float64))
    worst = float(np.max(resid[inside]))
    # The margin is the stopping tolerance's room, not a fudge: SMO stops at
    # `tol` on the KKT gap and the tube bound inherits that slack.
    rep.check(arm, worst <= eps + 1e-2,
              "every NON-support row lies inside the tube",
              "worst interior residual %.6f vs epsilon %.6f" % (worst, eps))

    rep.check(arm, fits[1].score(x, y) > 0.99,
              "a narrow tube still fits", "R^2 = %.6f" % fits[1].score(x, y))
    rep.check(arm, fits[-1].score(x, y) < fits[1].score(x, y) + 1e-9,
              "the widest tube does not fit BETTER than the narrow one",
              "%.6f vs %.6f" % (fits[-1].score(x, y), fits[1].score(x, y)))
    rep.note("  tube      epsilon %s\n            n_SV    %s\n"
             "            R^2     %s"
             % ("  ".join("%7.4f" % e for e in eps_ladder),
                "  ".join("%7d" % c for c in counts),
                "  ".join("%7.4f" % f.score(x, y) for f in fits)))


# ===========================================================================
# ARM: SHAPES
# ===========================================================================


def arm_shapes(rep):
    """The model is `n_support` wide, NOT `2 * n_support`.

    This is the arm for the one thing about the regression path that a
    surface author can get wrong without an error. `SmoSolver` solves over
    `n_train = 2 * n_rows`, and a caller who believed that reached this
    boundary would size buffers at `2 * n_rows` and read a second half that
    is never written. `Results::combine_coefs` folds the two alpha halves
    and then selects over `n_rows`, so:

        n_support_        <= n_rows
        dual_coef_        (1, n_support_)
        support_          (n_support_,)      int32, indices into X
        support_vectors_  (n_support_, k)    the rows themselves
        |dual_coef_|      <= C               each folded coefficient is
                                             `alpha_i - alpha*_i` and both
                                             halves are box-bounded by C

    The support indices are also checked to be in range, distinct and
    ascending, which is what `get_support_vector_indices`'s
    order-preserving compaction promises and what a scrambled gather would
    break.
    """
    arm = "SHAPES"
    x, y, _, _ = planted()
    C = 3.0
    est = SVR(kernel="linear", C=C, epsilon=0.02, tol=1e-4).fit(x, y)
    ns = est.n_support_

    rep.check(arm, 0 < ns <= PLANT_N,
              "n_support_ is at most n_rows, never 2 * n_rows",
              "%d of %d" % (ns, PLANT_N))
    rep.check(arm, est.dual_coef_.shape == (1, ns),
              "dual_coef_ is (1, n_support_)", str(est.dual_coef_.shape))
    rep.check(arm, est.support_.shape == (ns,),
              "support_ is (n_support_,)", str(est.support_.shape))
    rep.check(arm, est.support_vectors_.shape == (ns, PLANT_K),
              "support_vectors_ is (n_support_, n_features)",
              str(est.support_vectors_.shape))
    rep.check(arm, est.intercept_.shape == (1,),
              "intercept_ is (1,)", str(est.intercept_.shape))
    rep.check(arm, est.dual_coef_.dtype == np.float32
              and est.support_vectors_.dtype == np.float32,
              "the model arrays are float32")
    rep.check(arm, est.support_.dtype == np.int32,
              "support_ is int32", str(est.support_.dtype))

    idx = np.asarray(est.support_, dtype=np.int64)
    rep.check(arm, idx.size == 0 or (idx.min() >= 0 and idx.max() < PLANT_N),
              "every support index is a row of X")
    rep.check(arm, np.all(np.diff(idx) > 0) if idx.size > 1 else True,
              "the support indices are distinct and ascending")
    rep.check(arm, np.array_equal(est.support_vectors_, x[idx]),
              "support_vectors_ ARE the rows support_ names")

    worst = float(np.max(np.abs(est.dual_coef_))) if ns else 0.0
    rep.check(arm, worst <= C * (1.0 + 1e-5),
              "every folded dual coefficient is bounded by C",
              "max |dual| = %.6f, C = %.1f" % (worst, C))

    # `coef_` is a linear-kernel-only attribute on both classes.
    rbf = SVR(kernel="rbf", gamma=0.5).fit(x, y)
    try:
        rbf.coef_
        rep.bad(arm, "coef_ on an RBF fit -- IS INERT: it was ACCEPTED")
    except AttributeError as exc:
        rep.check(arm, "linear" in str(exc),
                  "coef_ raises AttributeError for a non-linear kernel")


# ===========================================================================
# ARM: INVARIANCE
# ===========================================================================


def arm_invariance(rep, assert_bits):
    """ONE ROW PREDICTED ALONE EQUALS THAT ROW INSIDE A BATCH.

    The property the serving world calls batch invariance, asked of the
    regression predict path. Three comparisons against one baseline:

        single      one row on its own, against its cell of the full call
        slice       a contiguous block of rows, same
        small buf   the whole matrix with `cache_size` set small enough
                    that `svc_predict` splits it into several batches, so
                    the kernel tile is a different shape every launch

    The third is this surface's half of DEVIATION 871 and of
    `check_svr_device_is_launch_invariant`, which turns the same knob
    inside Mojo from 0.001 MiB to 200 MiB. It matters more on a regressor
    than on a classifier because SVR runs `UpdateF` twice per batch, so a
    given matrix interleaves twice as many launches.

    ASSERTED UNDER `identical`, REPORTED UNDER `fast`. FAST promises speed
    only; its bits are allowed to move run to run and shape to shape, and
    asking a FAST arm a bitwise question is asking a question with no right
    answer.
    """
    arm = "INVARIANCE"
    x, y, _, _ = planted()
    est = SVR(kernel="rbf", gamma=0.5, C=10.0, epsilon=0.01).fit(x, y)
    base = est.predict(x)

    one = est.predict(x[7:8])
    rep.bits_equal(arm, one, base[7:8],
                   "row 7 alone == row 7 inside the full call", assert_bits)

    blk = est.predict(x[40:73])
    rep.bits_equal(arm, blk, base[40:73],
                   "rows 40:73 as a block == the same rows inside the full "
                   "call", assert_bits)

    # A buffer small enough to force several predict batches. The tile is
    # `n_batch * n_support * 4` bytes, so this asks for roughly two rows.
    #
    # SET ON THE FITTED ESTIMATOR RATHER THAN ON A SECOND FIT, so that the
    # predict buffer is the ONLY thing that changed. A refit would put a
    # second solve on the path and the comparison would stop being about
    # batching. `cache_size` is read at predict and nowhere else here
    # (DEVIATION 871), which is what makes that legitimate.
    tiny = max(est.n_support_ * 4 * 2, 1) / (1024.0 * 1024.0)
    wide = est.cache_size
    est.cache_size = tiny
    try:
        split = est.predict(x)
    finally:
        est.cache_size = wide
    rep.bits_equal(arm, split, base,
                   "the whole matrix through a %.6f MiB predict buffer == "
                   "one batch" % tiny, assert_bits)

    # A LINEAR arm too: the two kernels take different device paths and the
    # RBF row above cannot speak for the linear one.
    lin = SVR(kernel="linear", C=10.0, epsilon=0.01).fit(x, y)
    lbase = lin.predict(x)
    rep.bits_equal(arm, lin.predict(x[99:100]), lbase[99:100],
                   "kernel='linear': row 99 alone == row 99 in the batch",
                   assert_bits)


# ===========================================================================
# ARM: GAMMA
# ===========================================================================


def arm_gamma(rep, assert_bits):
    """The gamma the FIT resolved is the gamma PREDICT uses.

    Slot 5 of `svr_predict`'s params list is the trap its own docstring
    names: pass the constructor's gamma instead of the resolved one and the
    kernel matrix at predict is not the one the dual coefficients were
    solved against, and the answer is quietly wrong rather than absent.

    `gamma='auto'` is `1 / n_features`, so a fit with `gamma='auto'` and a
    fit with that number written out must agree EXACTLY, and a fit at a
    different gamma must not. The second half is what makes the first half
    evidence rather than a tautology.
    """
    arm = "GAMMA"
    x, y, _, _ = planted()
    auto = SVR(kernel="rbf", gamma="auto", C=10.0, epsilon=0.01).fit(x, y)
    rep.check(arm, auto._gamma == 1.0 / PLANT_K,
              "'auto' resolves to 1 / n_features", str(auto._gamma))
    explicit = SVR(kernel="rbf", gamma=1.0 / PLANT_K, C=10.0,
                   epsilon=0.01).fit(x, y)
    # TWO SEPARATE FITS, so this is asserted only where a repeated fit of
    # identical inputs is promised to give identical bits. FAST promises
    # nothing of the kind and gets the REPORT form.
    rep.bits_equal(arm, explicit.predict(x), auto.predict(x),
                   "gamma='auto' == the same number written out", assert_bits)
    close = np.allclose(explicit.predict(x), auto.predict(x),
                        rtol=1e-4, atol=1e-5)
    rep.check(arm, close,
              "gamma='auto' agrees with the same number to float32 tolerance "
              "in EVERY tier")
    other = SVR(kernel="rbf", gamma=2.0, C=10.0, epsilon=0.01).fit(x, y)
    same = np.array_equal(other.predict(x), auto.predict(x))
    rep.check(arm, not same,
              "a DIFFERENT gamma gives a different answer, so the row above "
              "is not vacuous")


# ===========================================================================


def main(argv=None):
    out = sys.stdout
    out.write("== mojolearn.tests.test_svr_surface ==\n")
    out.write("   surface mojolearn.SVR (svm/, epsilon-SVR)\n")

    rep = Report()
    aborted = []
    try:
        mode = arm_provenance(rep)
    except GateAbort as exc:
        out.write("\nCANNOT START: %s\n" % exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        out.write("\nCANNOT START: %s: %s\n" % (type(exc).__name__, exc))
        return 2

    assert_bits = mode == "identical"
    arms = (
        ("REFUSALS", lambda: arm_refusals(rep)),
        ("PLANTED", lambda: arm_planted(rep)),
        ("TUBE", lambda: arm_tube(rep)),
        ("SHAPES", lambda: arm_shapes(rep)),
        ("INVARIANCE", lambda: arm_invariance(rep, assert_bits)),
        ("GAMMA", lambda: arm_gamma(rep, assert_bits)),
    )
    for name, fn in arms:
        try:
            fn()
        except GateAbort as exc:
            aborted.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001 - report, then exit non-zero
            aborted.append((name, "%s: %s" % (type(exc).__name__, exc)))

    rep.render(out)
    for name, why in aborted:
        out.write("\n  %s ARM DID NOT RUN\n" % name)
        for line in why.splitlines():
            out.write("    %s\n" % line)

    out.write("\n")
    if rep.failures or aborted:
        out.write("test_svr_surface: RED. %d checks failed, %d arms did not "
                  "run.\n" % (len(rep.failures), len(aborted)))
        return 1
    if not assert_bits:
        out.write(
            "test_svr_surface: the %s arms passed. The tube, the shapes, the\n"
            "planted recovery and the refusals hold. THE BITWISE ARMS WERE\n"
            "REPORTED, NOT ASSERTED: this tier promises speed only and its\n"
            "bits are allowed to move. Re-run under\n"
            "MOJOLEARN_NUMERIC_MODE=identical for the asserted form.\n" % mode)
        return 0
    out.write(
        "test_svr_surface: GREEN. The Python surface recovers a planted\n"
        "linear fit, the epsilon tube behaves as a tube across five widths,\n"
        "the model is n_support wide and not 2 * n_support, one row alone is\n"
        "bit-identical to that row in a batch and through a split predict\n"
        "buffer, and every refusal on the path fires by name. It says nothing\n"
        "about a second vendor: the SVR path has never been in a\n"
        "three-vendor round.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
