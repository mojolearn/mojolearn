# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `arima/`, `mojolearn.ARIMA`.

DEVIATIONS 990 through 993. Written 2026-09-01, the day the surface landed.

WHAT THIS CLOSES
----------------
`arima/estimator.mojo`, `bindings/_mojolearn_arima.mojo` and
`python/mojolearn/_arima_impl.py` put a fitted, predicting, forecasting
ARIMA in reach of a Python caller for the first time. Nothing in that path
had been compared against anything.

`arima/checks/fit_check.mojo` and `arima/checks/arima_check.mojo` gate the
KERNELS, sixteen gates, both numeric tiers, including planted-parameter
recovery, batch-composition invariance, launch invariance and eleven
sabotage arms. THE ARITHMETIC IS GATED THERE AND IS NOT RE-GATED HERE. This
file is about the PYTHON PATH specifically, which those files cannot see at
all:

    the marshalling      numpy address in, numpy bytes out
    the element counts   N * batch, 2 * batch, (end - start) * batch, and
                         which of them is which
    the packing          mu, ar, ma, sar, sma, sigma2 per series, and the
                         named views that slice it
    the params list      thirteen scalars in one order on both sides, and
                         nine of them are small integers that would look
                         plausible in each other's slots
    the refusals         every one of them made to fire, by type and by
                         message, since a refusal that never fires is not a
                         refusal
    the mode selection   which .so actually loaded, read back from it

WHAT MAKES A NUMBER HERE ABSOLUTE. Almost every arm below is a RELATIVE
comparison and can see an inconsistency in the Python path without seeing an
answer that is wrong the same way everywhere. Two arms are not:

    RECOVERY   the series are GENERATED from known coefficients, in this
               file, so the right answer existed before any of this code
               ran. It is the same generator `tsa/checks/fixtures.mojo`
               uses, transcribed, at the same length and the same salt the
               lane's own recovery gate uses, so a failure here and a
               failure there are the same failure.
    FORECAST   an AR(1) forecast has NO INNOVATION IN IT. The h-step
               forecast of a fitted AR(1) with no intercept and no
               differencing is exactly `phi^h` times the last observation,
               in closed form, and that closed form is written here rather
               than taken from anything the device computed.

WHAT THIS FILE DOES *NOT* PROVE. It does not prove the lane is right; that
is `check-arima` and `check-fit`'s job. It does not touch a second vendor,
and the fit has never run on one: `arima/`'s three-vendor card at `221aa141`
is the KALMAN FILTER's, not the fit's, and `_arima_impl.py` says so on the
class. And it cannot prove an identity claim under FAST, which brings us to

THE TWO TIERS, AND WHAT EACH RUN IS WORTH
-----------------------------------------
**UNDER `identical` THE BITWISE ARMS ARE ASSERTED. UNDER `fast` THEY ARE
REPORTS AND THIS FILE SAYS SO IN ITS BANNER, IN EVERY AFFECTED LINE, AND IN
ITS VERDICT.** That is `arima/checks/fit_check.mojo::_gate`'s contract,
copied deliberately: a vendor-shaped claim is asserted only in the tier that
makes it. A FAST run of this file checks the marshalling, the shapes, the
refusals, the recovery and the closed-form forecast, all of which are real,
AND IT CHECKS NO BIT.

Everything that is NOT vendor-shaped is asserted in both tiers, because a
wrong element count is wrong in every tier.

HOW TO RUN IT
-------------
    # 1. build both extensions (the identical one is the gated one)
    bash bindings/build_arima.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_arima.sh

    # 2. the gate
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_arima_surface

    # 3. the FAST half, in its own process, because the mode is chosen at
    #    import and cannot be changed inside one
    cd python && python3 -m mojolearn.tests.test_arima_surface

ENVIRONMENT
    MOJOLEARN_ARIMA_GATE_N_OBS    the recovery length, default 512, which is
                                  `FIT_N_OBS` in `arima/checks/fixtures.mojo`.
                                  LOWERING IT INVALIDATES THE TOLERANCES,
                                  which are multiples of a standard error
                                  that goes as 1/sqrt(n), so the multiple is
                                  recomputed from whatever this says and the
                                  banner prints the length that ran.
    MOJOLEARN_ARIMA_GATE_QUICK    1 runs the AR(1) recovery case only, for a
                                  smoke pass. The verdict says so.
"""

import math
import os
import sys

import numpy as np

# `mojolearn.ARIMA` is the public name and is what this file prefers, so
# that no edit is needed on the day the private module is renamed. The
# fallback exists for a checkout where the package `__init__` has not been
# re-pointed yet.
try:  # pragma: no cover - one of the two branches is dead per checkout
    from mojolearn import ARIMA
    SURFACE = "mojolearn.ARIMA"
except ImportError:
    from mojolearn._arima_impl import ARIMA
    SURFACE = "mojolearn._arima_impl.ARIMA"

import mojolearn


# ===========================================================================
# THE FIXTURE, TRANSCRIBED FROM `tsa/checks/fixtures.mojo`
# ===========================================================================
# TRANSCRIBED AND NOT IMPORTED, because it lives in Mojo and Python cannot
# call it. Every innovation is a splitmix64 hash of `(series, t, salt)`, so
# no two cells repeat, a permutation of cells moves every comparison, and
# the fixture is reproducible on every host from its integers alone
# (`uniform-test-data-hides-permutation`).
#
# THE ARITHMETIC IS float64 UNTIL THE LAST LINE, exactly as the Mojo is:
# `ar1_series` and its siblings work in Float64 and `to_f32` casts once at
# the end. Doing the recursion in float32 would give a DIFFERENT series and
# the tolerances below, which are the lane's own, would then be tolerances
# for a different problem.
#
# `MASK` is not decoration. Mojo's UInt64 wraps; Python's int does not, so
# every product and every sum is masked back to 64 bits by hand. Getting
# that wrong does not raise, it silently produces another series.
_MASK = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB

#: `SALT` in `arima/checks/fit_check.mojo`, `arima_check.mojo` and
#: `arima_main.mojo`, all three.
SALT = 7
#: `FIT_N_OBS` in `arima/checks/fixtures.mojo`.
FIT_N_OBS = 512


def _splitmix(a, b, salt):
    z = (
        ((a + 1) * _GOLDEN)
        + ((b + 1) * _MIX_A)
        + ((salt + 1) * _MIX_B)
    ) & _MASK
    z = ((z ^ (z >> 30)) * _MIX_A) & _MASK
    z = ((z ^ (z >> 27)) * _MIX_B) & _MASK
    return z ^ (z >> 31)


def _u01(a, b, salt):
    return float(_splitmix(a, b, salt) >> 11) * (1.0 / 9007199254740992.0)


def _innovation(series, t, salt):
    """Approximately N(0, 1), `(u1 + u2 + u3 + u4 - 2) * sqrt(3)`. The sum
    is accumulated in the same order the Mojo loop accumulates it."""
    s = 0.0
    for j in range(4):
        s += _u01(series, t * 4 + j, salt)
    return (s - 2.0) * 1.7320508075688772


def ar1_series(n, phi, sigma, series, salt):
    out = np.empty(n, dtype=np.float64)
    x = 0.0
    for t in range(n):
        x = phi * x + sigma * _innovation(series, t, salt)
        out[t] = x
    return out


def ma1_series(n, theta, sigma, series, salt):
    out = np.empty(n, dtype=np.float64)
    prev = 0.0
    for t in range(n):
        e = sigma * _innovation(series, t, salt)
        out[t] = e + theta * prev
        prev = e
    return out


def arma11_series(n, phi, theta, sigma, series, salt):
    out = np.empty(n, dtype=np.float64)
    x = 0.0
    prev = 0.0
    for t in range(n):
        e = sigma * _innovation(series, t, salt)
        x = phi * x + e + theta * prev
        prev = e
        out[t] = x
    return out


# ===========================================================================
# THE PLANTED CASES, AND THE MULTIPLE OF THE STANDARD ERROR EACH IS ALLOWED
# ===========================================================================
# These are `arima/checks/fixtures.mojo::planted_cases`, the same three
# fixtures, the same six series each (ids 100+b, 200+b, 300+b), the same
# planted coefficients.
#
# THE TOLERANCE IS A MULTIPLE OF AN ASYMPTOTIC STANDARD ERROR AND THE
# MULTIPLE IS WRITTEN OUT, which is the whole reason this arm means
# anything. For a stationary AR(1) the asymptotic standard error of the
# maximum likelihood estimate of phi is
#
#     se(phi) = sqrt((1 - phi^2) / n)
#
# and for an invertible MA(1) the same expression in theta. At n = 512 and
# the coefficients below that is 0.0316, 0.0383, 0.0354 and 0.0422, so the
# bounds are 0.1499, 0.2001, 0.2500 and 0.2500, which are
# `fixtures.mojo`'s 0.15, 0.20, 0.25 and 0.25 to four figures.
#
# THE MULTIPLES ARE NOT CHOSEN HERE. They are the numbers that reproduce
# `fixtures.mojo::planted_cases`'s own `tol` field at n = 512, so this gate
# is EXACTLY as strict as the lane's own recovery gate and no stricter. A
# tighter bound invented in this file would go red on a fit the lane
# considers correct, which would be this file being wrong about the lane
# rather than the lane being wrong.
#
# THE ARMA(1,1) MULTIPLES ARE THE LOOSEST AND THAT IS NOT SLACK. phi_hat and
# theta_hat in an ARMA(1,1) are strongly correlated, so the MARGINAL formula
# above understates the spread of either one taken alone; the lane's own
# fixture allows 0.25 for both, which is 7.07 marginal standard errors on
# phi and 5.93 on theta.
PLANTED = (
    dict(name="ar1", order=(1, 0, 0), seasonal=(0, 0, 0, 0), first=100,
         phi=0.7, theta=None, mult_phi=4.75, mult_theta=None),
    dict(name="ma1", order=(0, 0, 1), seasonal=(0, 0, 0, 0), first=200,
         phi=None, theta=0.5, mult_phi=None, mult_theta=5.23),
    dict(name="arma11", order=(1, 0, 1), seasonal=(0, 0, 0, 0), first=300,
         phi=0.6, theta=0.3, mult_phi=7.07, mult_theta=5.93),
)
BATCH = 6


def planted_batch(case, n_obs):
    """The six series of one planted case, `(6, n_obs)` float32."""
    rows = []
    for b in range(BATCH):
        sid = case["first"] + b
        if case["phi"] is not None and case["theta"] is not None:
            rows.append(arma11_series(n_obs, case["phi"], case["theta"], 1.0,
                                      sid, SALT))
        elif case["phi"] is not None:
            rows.append(ar1_series(n_obs, case["phi"], 1.0, sid, SALT))
        else:
            rows.append(ma1_series(n_obs, case["theta"], 1.0, sid, SALT))
    return np.ascontiguousarray(np.array(rows), dtype=np.float32)


def standard_error(coef, n_obs):
    return math.sqrt((1.0 - coef * coef) / float(n_obs))


# ===========================================================================
# REPORTING
# ===========================================================================
# EVERY ARM RUNS AND EVERY VERDICT IS PRINTED BEFORE THE PROCESS EXITS
# NON-ZERO. `arima/checks/arima_check.mojo` states the reason and it is the
# same one here: stopping at the first failure shows one arm's opinion when
# the useful evidence is WHICH arms a given defect reaches and which it
# walks past.


class Report(object):
    def __init__(self, mode):
        self.mode = mode
        self.identical = mode == "identical"
        self.rows = []
        self.notes = []
        self.reported = 0

    def ok(self, arm, what):
        self.rows.append((True, arm, what))

    def bad(self, arm, what):
        self.rows.append((False, arm, what))

    def note(self, text):
        self.notes.append(text)

    def check(self, arm, cond, what, detail=""):
        """An arm that is true in every tier. Asserted always."""
        if cond:
            self.ok(arm, what)
        else:
            self.bad(arm, what + ((" -- " + detail) if detail else ""))
        return bool(cond)

    def gate(self, arm, cond, what, detail=""):
        """A VENDOR-SHAPED claim: asserted under IDENTICAL, RECORDED under
        FAST. The exact contract of `fit_check.mojo::_gate`, and the reason
        is the same. Under FAST the bits are allowed to move run to run and
        box to box, so asking a FAST arm a bitwise question and failing on
        the answer would be a gate that is wrong rather than a gate that
        found something."""
        if cond:
            self.ok(arm, what + ("" if self.identical
                                 else "  [FAST: not asserted]"))
            return True
        if self.identical:
            self.bad(arm, what + ((" -- " + detail) if detail else ""))
            return False
        self.reported += 1
        self.ok(arm, "RECORDED [FAST] " + what
                + ((" -- " + detail) if detail else "")
                + "  (vendor-shaped under FAST; not asserted)")
        return True

    def bits_equal(self, arm, got, want, what, gated=True):
        """The only comparison this file makes about a fitted number.
        `.tobytes()` is C-order whatever the strides are."""
        ga = np.ascontiguousarray(got).ravel().view(np.uint32)
        wa = np.ascontiguousarray(want).ravel().view(np.uint32)
        if ga.shape != wa.shape:
            self.bad(arm, "%s -- SHAPES DIFFER, %s vs %s"
                     % (what, np.shape(got), np.shape(want)))
            return False
        diff = np.flatnonzero(ga != wa)
        if diff.size == 0:
            return (self.gate if gated else self.check)(arm, True, what)
        first = int(diff[0])
        detail = ("%d of %d cells differ; first at flat index %d, "
                  "0x%08x vs 0x%08x (%r vs %r)"
                  % (diff.size, ga.size, first, int(ga[first]), int(wa[first]),
                     float(np.ascontiguousarray(got).ravel()[first]),
                     float(np.ascontiguousarray(want).ravel()[first])))
        return (self.gate if gated else self.check)(arm, False, what, detail)

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal. Every guard on this
        surface is a branch a passing build can contain and never take, so
        each one is made to fire BY NAME and BY MESSAGE.

        A Mojo `Error` crossing `def_function` is caught as `Exception`
        rather than as a named type, because what CPython raises for one is
        not this surface's choice. Those rows pass `Exception` and lean
        entirely on the message, which is why the needles below are quoted
        from the Mojo source rather than paraphrased."""
        try:
            fn(*a, **kw)
        except exc_type as exc:
            if needle in str(exc):
                self.ok(arm, what)
                return True
            self.bad(arm, "%s -- raised %s but the message does not contain "
                     "%r: %s" % (what, type(exc).__name__, needle, exc))
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
    """A condition under which no verdict about ARIMA can be reached at all,
    a binary that will not load, a fixture that will not build. Distinct
    from a FAILING check: a failure says the surface is wrong, an abort says
    the gate did not run."""


# ===========================================================================
# ARM: PROVENANCE. DEVIATION 990.
# ===========================================================================


def arm_provenance(rep, model):
    """What this process is actually holding, checked against itself.

    `_backend.load_set` already refuses a binary whose VENDOR disagrees with
    the directory it loaded from, and cross-checks the TIER through the gbdt
    binary. `_arima_impl.ARIMA._extension` adds the tier read-back for THIS
    binary. What is added here is the direction neither of them checks, that
    the module the estimator holds is the one under the tier directory this
    process asked for."""
    arm = "PROVENANCE (990)"
    mod = model._bind()
    binary = getattr(mod, "__file__", "") or ""
    parent = os.path.basename(os.path.dirname(os.path.abspath(binary)))
    mode = model.numeric_mode_used()
    rep.check(arm, mode == rep.mode,
              "the estimator's tier is the process's tier",
              "%s vs %s" % (mode, rep.mode))
    if rep.identical:
        rep.check(arm, parent == "identical",
                  "the loaded binary is the one under identical/", binary)
    else:
        rep.check(arm, parent != "identical",
                  "the loaded binary is not the identical one", binary)
    rep.check(arm, hasattr(mod, "arima_vendor"),
              "the binary exports arima_vendor for _backend.py's cross-check")
    rep.check(arm, hasattr(mod, "arima_numeric_mode"),
              "the binary exports arima_numeric_mode for the tier read-back")
    rep.note("  binary  %s\n  surface %s\n  mode    %s\n  vendor  %s"
             % (binary, SURFACE, mode, model.vendor_used()))
    return mode


# ===========================================================================
# ARM: REFUSALS. DEVIATION 992.
# ===========================================================================


def arm_refusals(rep, y):
    """Every guard reachable from this surface, made to fire.

    THE TWO HALVES ARE LABELLED, because they behave differently. The
    PYTHON half raises named types and its policy lives in
    `_arima_impl.py`; the MOJO half arrives as a bare `Exception` and its
    policy lives in `arima/impl/tsa/arima_common.mojo::validate_order`,
    `arima/estimator.mojo::_refuse_method` and
    `arima/impl/arima/batched_arima.mojo::_refuse_non_finite`. A refusal
    that migrated from the second half to the first would be a policy moved
    out of reach of every Mojo gate, and the type in these rows is what
    would catch that.
    """
    arm = "REFUSALS, python side (992)"

    rep.raises(arm, NotImplementedError, "trend",
               "trend='t' is refused by name",
               ARIMA, (1, 0, 0), trend="t")
    rep.raises(arm, NotImplementedError, "trend",
               "trend='ct' is refused by name",
               ARIMA, (1, 0, 0), trend="ct")
    rep.raises(arm, NotImplementedError, "trend",
               "a polynomial trend specification is refused by name",
               ARIMA, (1, 0, 0), trend=[1, 1, 0, 1])
    rep.raises(arm, ValueError, "unknown method",
               "an unspelled method is refused",
               ARIMA, (1, 0, 0), method="mle")
    rep.raises(arm, NotImplementedError, "verbose",
               "verbose is refused, it selects log lines this port never prints",
               ARIMA, (1, 0, 0), verbose=True)
    rep.raises(arm, NotImplementedError, "output_type",
               "output_type is refused, this package returns NumPy",
               ARIMA, (1, 0, 0), output_type="cudf")
    rep.raises(arm, ValueError, "3 entries",
               "a 2-tuple order is refused",
               ARIMA, (1, 0))
    rep.raises(arm, ValueError, "non-negative",
               "a negative order entry is refused",
               ARIMA, (-1, 0, 0))
    rep.raises(arm, ValueError, "4 entries",
               "a 3-tuple seasonal_order is refused",
               ARIMA, (1, 0, 0), seasonal_order=(0, 0, 0))
    rep.raises(arm, TypeError, "level",
               "level is NOT A PARAMETER of this class, so it is a TypeError;"
               " cuML's confidence intervals are unported",
               ARIMA, (1, 0, 0), level=0.95)
    rep.raises(arm, TypeError, "simple_differencing",
               "simple_differencing is NOT A PARAMETER of this class; only"
               " the True arm exists in this lane",
               ARIMA, (1, 0, 0), simple_differencing=False)
    rep.raises(arm, TypeError, "start_params",
               "start_params is NOT A PARAMETER; every fit starts from"
               " estimate_x0",
               ARIMA, (1, 0, 0), start_params=[0.1, 1.0])
    rep.raises(arm, ValueError, "1-D",
               "a 3-D y is refused",
               ARIMA(order=(1, 0, 0)).fit, np.zeros((2, 3, 4), np.float32))
    rep.raises(arm, ValueError, "call fit()",
               "predict before fit is refused",
               ARIMA(order=(1, 0, 0)).predict, 0, 4)
    rep.raises(arm, ValueError, "call fit()",
               "forecast before fit is refused",
               ARIMA(order=(1, 0, 0)).forecast, 4)

    arm = "REFUSALS, mojo side (992)"
    # THE NEEDLES ARE QUOTED FROM THE MOJO SOURCE. A paraphrase here would
    # pass against any message at all containing the paraphrase's words.
    rep.raises(arm, Exception, "exog",
               "exog is refused by validate_order, as n_exog != 0",
               ARIMA(order=(1, 0, 0)).fit, y, np.ones((BATCH, y.shape[1], 1)))
    rep.raises(arm, Exception, "method='css'",
               "method='css' is refused by arima/estimator.mojo, only MLE is"
               " offered",
               ARIMA(order=(1, 0, 0), method="css").fit, y)
    rep.raises(arm, Exception, "method='css-ml'",
               "method='css-ml' is refused by the same file",
               ARIMA(order=(1, 0, 0), method="css-ml").fit, y)
    rep.raises(arm, Exception, "p, q, P, Q <= 8",
               "p > 8 is refused by validate_order, in cuML's words",
               ARIMA(order=(9, 0, 0)).fit, y)
    rep.raises(arm, Exception, "d + D <= 2",
               "d + D > 2 is refused by validate_order",
               ARIMA(order=(1, 3, 0)).fit, y)
    rep.raises(arm, Exception, "at least one of p, q, P, Q",
               "an order with no parameters at all is refused",
               ARIMA(order=(0, 0, 0), trend="n").fit, y)
    rep.raises(arm, Exception, "invalid period for seasonal",
               "a seasonal term with s < 2 is refused",
               ARIMA(order=(0, 0, 0), seasonal_order=(1, 0, 0, 1),
                     trend="n").fit, y)
    # rd = d + s*D + max(p + s*P, q + s*Q + 1) = 0 + 8 + 1 = 9, and r = 1,
    # so this reaches the rd bound WITHOUT tripping the r bound first.
    rep.raises(arm, Exception, "block-per-series",
               "rd > 8 is refused; it selects cuML's block-per-series Kalman"
               " kernel, which is unported",
               ARIMA(order=(0, 0, 0), seasonal_order=(0, 1, 0, 8),
                     trend="c").fit, y)
    # r = max(p, q + 1) = 6 with no differencing, so rd = 6 and the rd bound
    # is NOT what fires here.
    rep.raises(arm, Exception, "Lyapunov",
               "r > 5 is refused; it selects cuML's Schur/Sylvester Lyapunov"
               " solver, which is unported",
               ARIMA(order=(6, 0, 0), trend="n").fit, y)
    bad = y.copy()
    bad[1, 7] = np.float32("nan")
    rep.raises(arm, Exception, "non-finite value at index",
               "a NaN is refused by name and by INDEX; missing observations"
               " are a cuML path with no port, so a NaN here is not missing"
               " data, it is refused input",
               ARIMA(order=(1, 0, 0)).fit, bad)
    rep.raises(arm, Exception, "n_obs must be at least 2",
               "a one-observation series is refused",
               ARIMA(order=(1, 0, 0)).fit, y[:, :1])

    arm = "REFUSALS, after a fit (992)"
    m = ARIMA(order=(1, 0, 0), trend="n").fit(y)
    rep.raises(arm, ValueError, "start < end",
               "predict with end <= start is refused, and the message names"
               " BOTH conventions",
               m.predict, 5, 5)
    rep.raises(arm, ValueError, "steps must be >= 1",
               "forecast(0) is refused",
               m.forecast, 0)
    rep.raises(arm, AttributeError, "no intercept",
               "mu_ raises when trend resolved to 'n', rather than returning"
               " a zero-width mean",
               lambda: m.mu_)
    return m


# ===========================================================================
# ARM: RECOVERY. The answer came from outside this file's device code.
# ===========================================================================


def arm_recovery(rep, n_obs, quick):
    """PLANTED-PARAMETER RECOVERY through the Python surface.

    Every series is GENERATED above from known coefficients, so the right
    answer existed before any of this ran. No oracle that shares a spelling
    with the thing it checks, no bound derived from our own output. If the
    fit recovers phi = 0.7 from a series built with phi = 0.7, then the
    least squares, the Jones transform, the Kalman filter, the finite
    differences, the optimizer AND the whole marshalling path are doing what
    they claim, because a fault in any one of them moves the answer.

    IT ALSO CATCHES THE ONE MARSHALLING BUG A SHAPE CHECK CANNOT SEE. If the
    thirteen-entry params list were reordered, or if `params_` were unpacked
    with the wrong stride, the numbers would still be finite and the shapes
    would still be right; they would simply be the wrong series' parameters.
    Six series with six different innovation seeds and one shared truth is
    what separates those cases, and the per-series spread is printed.

    ASSERTED IN BOTH TIERS. Recovering a planted coefficient is not a
    vendor-shaped claim: it is a statement about the estimator, and a FAST
    build that could not recover phi would be broken in every tier.
    """
    arm = "RECOVERY"
    cases = PLANTED[:1] if quick else PLANTED
    fitted = {}
    for case in cases:
        y = planted_batch(case, n_obs)
        m = ARIMA(order=case["order"], seasonal_order=case["seasonal"],
                  trend="n").fit(y)
        fitted[case["name"]] = (y, m)
        p, d, q = case["order"]
        rep.check(arm, m.params_.shape == (BATCH, p + q + 1),
                  "%s: params_ is (batch, N) with N = p + q + k + 1"
                  % case["name"], str(m.params_.shape))
        rep.check(arm, m.retcode_.shape == (BATCH,) and m.llf_.shape == (BATCH,),
                  "%s: the per-series arrays are batch-long" % case["name"])

        for label, key, mult_key, block in (
            ("phi", "phi", "mult_phi", "ar_"),
            ("theta", "theta", "mult_theta", "ma_"),
        ):
            truth = case[key]
            if truth is None:
                continue
            got = np.asarray(getattr(m, block), dtype=np.float64)[:, 0]
            se = standard_error(truth, n_obs)
            mult = case[mult_key]
            bound = mult * se
            worst = float(np.max(np.abs(got - truth)))
            rep.check(
                arm, worst <= bound,
                "%s: %s recovered on all %d series, worst |error| %.5f "
                "within %.2f standard errors (se %.5f, bound %.5f)"
                % (case["name"], label, BATCH, worst, mult, se, bound),
                "achieved %.2f standard errors" % (worst / se if se else 0.0))
            rep.note("    %-8s %-5s planted %.3f  fitted %s"
                     % (case["name"], label, truth,
                        " ".join("%.4f" % v for v in got)))
            # SIX SERIES, SIX ANSWERS. A broadcast bug, a stride bug or a
            # params list whose batch_size landed in the wrong slot would
            # give one number six times, and every tolerance above would
            # still pass.
            rep.check(arm, len(set(np.round(got, 7))) == BATCH,
                      "%s: the %d series produced %d DISTINCT %s, so nothing "
                      "was broadcast" % (case["name"], BATCH, BATCH, label))

        rep.check(arm, int(np.count_nonzero(m.retcode_)) == 0,
                  "%s: every series reported OPT_SUCCESS" % case["name"],
                  "retcode_ = %s, n_iter_ = %s" % (m.retcode_, m.n_iter_))
        # The two log-likelihoods the Mojo side computes by different routes
        # must agree: `fx_` is the optimizer's objective, -loglike/(n_obs-1),
        # and `llf_` is a SEPARATE Kalman pass at the fitted point
        # (`arima/estimator.mojo::_loglike_at`). This is the only check that
        # the extra pass evaluated the model the optimizer returned.
        implied = -np.asarray(m.fx_, dtype=np.float64) * (n_obs - 1)
        rel = np.max(np.abs(implied - m.llf_) / np.maximum(np.abs(m.llf_), 1.0))
        rep.check(arm, rel < 1e-5,
                  "%s: llf_ agrees with -fx_ * (n_obs - 1) to %.2e, so the "
                  "second Kalman pass evaluated the fit's own point"
                  % (case["name"], rel))
    return fitted


# ===========================================================================
# ARM: INFORMATION CRITERIA. DEVIATION 991.
# ===========================================================================


def arm_criteria(rep, fitted, n_obs):
    """`aic_` and `bic_` against raft's formula, written out again here.

    A TRANSCRIPTION CHECK AND NOTHING MORE, and it says so. cuML's
    `information_criterion` is unported and its arithmetic beyond the
    log-likelihood is one unary op, `ic_base - 2 * loglike`, with
    `ic_base = 2 * N` for AIC and `log(T) * N` for BIC
    (`raft/stats/detail/batched/information_criterion.cuh`). What this can
    catch is a wrong `N` or a wrong `T`, which are the two ways to get an
    information criterion wrong without getting the likelihood wrong, and
    `T` is the one people miss: it is the observation count AFTER
    differencing, `n_obs - (d + s * D)`, not `n_obs`.
    """
    arm = "CRITERIA (991)"
    for name, (y, m) in fitted.items():
        p, d, q = m.order
        P, D, Q, s = m.seasonal_order
        N = p + q + P + Q + m.k_ + 1
        T = n_obs - (d + s * D)
        want_aic = 2.0 * N - 2.0 * m.llf_
        want_bic = math.log(T) * N - 2.0 * m.llf_
        rep.check(arm, np.allclose(m.aic_, want_aic, rtol=0, atol=1e-9),
                  "%s: aic_ == 2N - 2 llf with N = %d" % (name, N))
        rep.check(arm, np.allclose(m.bic_, want_bic, rtol=0, atol=1e-9),
                  "%s: bic_ == log(T) N - 2 llf with T = %d, the count AFTER "
                  "differencing" % (name, T))
        rep.check(arm, np.all(m.bic_ - m.aic_ > 0.0),
                  "%s: the BIC penalty exceeds the AIC penalty at n = %d, "
                  "since log(T) > 2" % (name, T))


# ===========================================================================
# ARM: PREDICT. The window, the exclusive end, and the NaN prefix.
# ===========================================================================


def arm_predict(rep, y, m, n_obs):
    arm = "PREDICT"
    full = m.predict(0, n_obs)
    rep.check(arm, full.shape == (BATCH, n_obs),
              "predict(0, n_obs) is (batch, n_obs); `end` IS EXCLUDED, which "
              "is cuML's convention and not statsmodels'", str(full.shape))
    rep.check(arm, m.predict().shape == full.shape,
              "end=None means n_obs, as upstream")
    win = m.predict(3, 9)
    rep.check(arm, win.shape == (BATCH, 6),
              "predict(3, 9) is six wide, not seven", str(win.shape))
    rep.check(arm, bool(np.isfinite(full).all()),
              "with d = 0 no in-sample prediction is NaN")
    # THE WINDOW MUST NOT CHANGE THE ANSWER. The Kalman pass is over the
    # whole series whatever `start` and `end` are; only the copy-out is
    # windowed. If a window changed a value, the windowing would be inside
    # the filter rather than after it.
    rep.bits_equal(arm, win, full[:, 3:9],
                   "a windowed predict is the same bits as the slice of the "
                   "full one")

    # THE DIFFERENCED PATH, which reaches a different in-sample kernel arm
    # (dD == 1) and the tsa differencing kernels.
    w = np.ascontiguousarray(np.cumsum(y, axis=1), dtype=np.float32)
    md = ARIMA(order=(1, 1, 1), trend="c").fit(w)
    pd_ = md.predict(0, n_obs)
    rep.check(arm, pd_.shape == (BATCH, n_obs),
              "the d = 1 model predicts (batch, n_obs) too")
    rep.check(arm, bool(np.isnan(pd_[:, 0]).all()),
              "step 0 is NaN when d = 1: differencing consumes it, and the "
              "sentinel is the canonical 0x7fc00000 by constant "
              "(DEVIATION 676)")
    rep.check(arm, bool(np.isfinite(pd_[:, 1:]).all()),
              "everything from step d onward is finite")
    nan_bits = np.ascontiguousarray(pd_[:, 0]).view(np.uint32)
    rep.gate(arm, bool(np.all(nan_bits == np.uint32(0x7FC00000))),
             "the undefined prediction is the CANONICAL quiet NaN "
             "0x7fc00000 on every vendor, not the hardware's own payload",
             "got 0x%08x" % int(nan_bits[0]))
    rep.check(arm, md.params_.shape == (BATCH, 4),
              "the (1,1,1) model with trend='c' packs mu, ar, ma, sigma2",
              str(md.params_.shape))
    rep.check(arm, md.mu_.shape == (BATCH,),
              "mu_ exists and is batch-long when trend='c'")
    return md


# ===========================================================================
# ARM: FORECAST. The closed form, and the identity with predict.
# ===========================================================================


def arm_forecast(rep, y, m, md, n_obs):
    """THE ONE ARM WITH AN ANSWER THAT OWES NOTHING TO THE DEVICE.

    A stationary AR(1) with no intercept and no differencing has, as its
    h-step forecast from the end of the series,

        y_hat(n + h) = phi^h * y(n)

    with NO INNOVATION TERM IN IT, because the expectation of every future
    innovation is zero. That is a closed form in the FITTED phi and the LAST
    OBSERVATION, both of which this file already holds, so the whole
    forecast path (the Kalman forecast steps, `finalize_forecast`,
    `copy_forecast_kernel`, the copy back and the reshape) is checked
    against arithmetic no kernel touched. It is "a noiseless planted series"
    in the only sense that matters here: the CONTINUATION is noiseless by
    construction even though the data was not.

    THE TOLERANCE IS NOT AN IDENTITY TOLERANCE and this arm is not an
    identity arm. The reference is float64 and the device is float32
    (DEVIATION 670), so they cannot agree bitwise and it would be wrong to
    ask them to.
    """
    arm = "FORECAST"
    h = 6
    fc = m.forecast(h)
    rep.check(arm, fc.shape == (BATCH, h),
              "forecast(6) is (batch, 6)", str(fc.shape))
    rep.check(arm, bool(np.isfinite(fc).all()),
              "no forecast step is NaN, there is no in-sample block in one")

    phi = np.asarray(m.ar_, dtype=np.float64)[:, 0]
    last = np.asarray(y[:, -1], dtype=np.float64)
    want = np.empty((BATCH, h), dtype=np.float64)
    acc = last.copy()
    for i in range(h):
        acc = phi * acc
        want[:, i] = acc
    got = np.asarray(fc, dtype=np.float64)
    scale = np.maximum(np.abs(want), 1e-3)
    worst = float(np.max(np.abs(got - want) / scale))
    rep.check(arm, worst < 2e-3,
              "the AR(1) forecast is phi^h times the last observation to "
              "%.2e relative, in closed form, against a reference no kernel "
              "produced" % worst)
    rep.check(arm, bool(np.all(np.abs(got[:, -1]) <= np.abs(got[:, 0]) + 1e-6)),
              "a stationary AR(1) forecast decays; |step 6| does not exceed "
              "|step 1|")

    # UPSTREAM `forecast` IS `predict(n_obs, n_obs + steps)` AND THIS PORT
    # SAYS SO. If the two entry points ever stop agreeing, one of them has
    # grown a behaviour the other has not, and it will be visible here
    # before it is visible in an answer.
    rep.bits_equal(arm, fc, m.predict(n_obs, n_obs + h),
                   "forecast(h) is predict(n_obs, n_obs + h), bit for bit")

    # THE DIFFERENCED FORECAST, which is the only path through
    # `finalize_forecast`. Its first step continues the LEVEL of the series
    # rather than restarting near zero, which a missing undiff would break
    # loudly and nothing else here would catch.
    fd = md.forecast(4)
    rep.check(arm, fd.shape == (BATCH, 4) and bool(np.isfinite(fd).all()),
              "the d = 1 model forecasts four finite steps")
    w_last = np.abs(np.asarray(np.cumsum(y, axis=1)[:, -1], dtype=np.float64))
    step0 = np.abs(np.asarray(fd[:, 0], dtype=np.float64))
    rep.check(arm, bool(np.all(step0 > 0.25 * w_last)),
              "the differenced forecast continues the LEVEL of the series, "
              "so step 1 is the same order as the last observation; a "
              "missing finalize_forecast would return the differenced scale")


# ===========================================================================
# ARM: BATCH COMPOSITION. The vendor-shaped one.
# ===========================================================================


def arm_batch(rep, n_obs):
    """A SERIES FITTED ALONE MUST GIVE THE SAME BITS AS THAT SERIES INSIDE A
    BATCH.

    `arima/checks/fit_check.mojo::check_fit_is_batch_composition_invariant`
    asserts this at the kernel boundary and explains what it catches: the
    optimizer is a HOST state machine with per-series branches, the Armijo
    test, the skipping test, the convergence test, any of which could read
    another series' state; and it is the gate that would catch the shortcut
    of dropping converged series out of the batch, which would make the
    answer depend on convergence order.

    WHAT IS ADDED BY REPEATING IT HERE is everything between that boundary
    and a caller: the wrapper builds a DIFFERENT params list for a batch of
    one, a different set of buffers, and a different reshape. A
    `batch_size` that leaked into an index would show up here and nowhere in
    that Mojo gate.

    THREE COMPOSITIONS, not two. One alone, three of six, and all six, so a
    difference between "alone" and "in company" cannot be confused with a
    difference between two batch sizes.

    GATED, not asserted, under FAST. Bit equality is a vendor-shaped claim
    and FAST promises speed only.
    """
    arm = "BATCH COMPOSITION"
    case = PLANTED[0]
    y6 = planted_batch(case, n_obs)
    m6 = ARIMA(order=case["order"], trend="n").fit(y6)

    pick = [0, 2, 4]
    y3 = np.ascontiguousarray(y6[pick])
    m3 = ARIMA(order=case["order"], trend="n").fit(y3)
    rep.bits_equal(arm, m3.params_, m6.params_[pick],
                   "three series drawn from six fit to the same bits")
    rep.check(arm, bool(np.array_equal(m3.n_iter_, m6.n_iter_[pick])),
              "and to the same ITERATION COUNTS, which is the stage that "
              "catches a per-series branch reading another series' state")
    rep.check(arm, bool(np.array_equal(m3.retcode_, m6.retcode_[pick])),
              "and the same retcodes")

    y1 = np.ascontiguousarray(y6[2:3])
    m1 = ARIMA(order=case["order"], trend="n").fit(y1)
    rep.bits_equal(arm, m1.params_, m6.params_[2:3],
                   "ONE series fitted alone is the same bits as that series "
                   "inside a batch of six")
    rep.check(arm, m1.params_.shape == (1, 2),
              "a batch of one is still 2-D, (1, N)", str(m1.params_.shape))

    # AND SO IS EVERY DOWNSTREAM ANSWER. Parameters agreeing while a
    # prediction does not would mean the prediction path reads batch_size
    # somewhere it should read a series index.
    rep.bits_equal(arm, m1.forecast(5), m6.forecast(5)[2:3],
                   "the forecast is batch-composition invariant too")
    rep.bits_equal(arm, m1.predict(0, 8), m6.predict(0, 8)[2:3],
                   "and so is the in-sample prediction")

    # A 1-D INPUT IS ONE SERIES AND MUST BE THE SAME ANSWER AS ITS (1, n)
    # FORM. Same bytes in, same bits out; if this ever differs, the reshape
    # in `_series_major` has stopped being a view of the same values.
    #
    # GATED LIKE THE REST, even though it is the same input through the same
    # code. FAST promises SPEED ONLY, so it does not promise a repeat of one
    # call, and asking a FAST arm a bitwise question is the mistake
    # `fast-is-not-identical` names.
    m1d = ARIMA(order=case["order"], trend="n").fit(y6[2])
    rep.bits_equal(arm, m1d.params_, m1.params_,
                   "a 1-D y is exactly the (1, n_obs) fit, not a different "
                   "problem")


# ===========================================================================
# THE RUNNER
# ===========================================================================


def main(argv=None):
    out = sys.stdout
    n_obs = int(os.environ.get("MOJOLEARN_ARIMA_GATE_N_OBS", FIT_N_OBS))
    quick = os.environ.get("MOJOLEARN_ARIMA_GATE_QUICK") == "1"

    try:
        mode = mojolearn.numeric_mode()
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        out.write("\nCANNOT START: %s\n" % exc)
        return 2
    rep = Report(mode)

    out.write("test_arima_surface: %s, tier %s, n_obs %d%s\n"
              % (SURFACE, mode, n_obs, ", QUICK" if quick else ""))
    if not rep.identical:
        out.write(
            "\nTHIS PROCESS LOADED THE %s BUILD, WHICH MAKES NO CROSS-VENDOR\n"
            "CLAIM. The bitwise arms below are RECORDED and NOT ASSERTED, and\n"
            "each one says so on its own line. What this run does check is\n"
            "real: the marshalling, the element counts, the packing, every\n"
            "refusal, the planted-parameter recovery and the closed-form\n"
            "forecast. It checks NO BIT. For the identity half:\n"
            "\n"
            "  MOJOLEARN_NUMERIC_MODE=identical \\\n"
            "      python3 -m mojolearn.tests.test_arima_surface\n"
            % mode.upper())
    if n_obs != FIT_N_OBS:
        out.write(
            "\nWARNING: n_obs is %d, not the lane's %d. The recovery bounds\n"
            "are recomputed from it (they go as 1/sqrt(n)), but the MULTIPLES\n"
            "were chosen to reproduce fit_check.mojo's tolerances AT %d, so a\n"
            "pass here is no longer the same statement as a pass there.\n"
            % (n_obs, FIT_N_OBS, FIT_N_OBS))

    aborted = []
    fitted = {}
    y = planted_batch(PLANTED[0], n_obs)

    def _provenance():
        arm_provenance(rep, ARIMA(order=(1, 0, 0)))

    def _refusals():
        arm_refusals(rep, y)

    def _recovery():
        fitted.update(arm_recovery(rep, n_obs, quick))

    def _criteria():
        if not fitted:
            raise GateAbort("RECOVERY did not run, so there is nothing fitted")
        arm_criteria(rep, fitted, n_obs)

    def _shapes():
        if "ar1" not in fitted:
            raise GateAbort("RECOVERY did not run")
        yy, m = fitted["ar1"]
        md = arm_predict(rep, yy, m, n_obs)
        arm_forecast(rep, yy, m, md, n_obs)

    def _batch():
        arm_batch(rep, n_obs)

    arms = (
        ("PROVENANCE", _provenance),
        ("REFUSALS", _refusals),
        ("RECOVERY", _recovery),
        ("CRITERIA", _criteria),
        ("PREDICT/FORECAST", _shapes),
        ("BATCH", _batch),
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
        out.write("test_arima_surface: RED. %d checks failed, %d arms did not "
                  "run.\n" % (len(rep.failures), len(aborted)))
        return 1
    if not rep.identical:
        out.write(
            "test_arima_surface: the %s arms passed, AND NO BIT WAS CHECKED.\n"
            "%d bitwise comparisons were RECORDED rather than asserted. This\n"
            "run says the surface marshals correctly, refuses what it claims\n"
            "to refuse, recovers planted coefficients and forecasts the\n"
            "closed form. It says NOTHING about cross-vendor identity.\n"
            % (rep.mode, rep.reported))
        return 0
    out.write(
        "test_arima_surface: GREEN. The Python surface fits a batch, recovers\n"
        "planted coefficients within the lane's own multiples of a standard\n"
        "error, predicts on an exclusive end with the canonical NaN before\n"
        "d, forecasts the AR(1) closed form, and gives one series the same\n"
        "BITS alone as inside a batch of six. What it does NOT prove is a\n"
        "second vendor: arima/'s three-vendor card is the Kalman filter's,\n"
        "and the fit has never left one Apple M4.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
