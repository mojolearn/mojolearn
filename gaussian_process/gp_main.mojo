# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gaussian-process driver: one fit and predict, the identity card, the mode.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.gp.card \\
        tools/with_build_lock.sh pixi run mojo run -I . gaussian_process/gp_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.gp.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . gaussian_process/gp_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.gp.identical.card /tmp/<other>.gp.identical.card

Environment knobs (all optional): `MOJOLEARN_GP_FIXTURE` (`planted`,
`duplicate`, `handworked`, `ard`, `signed_zero`; default `ard`, because it
is the only one that exercises ARD length scales, a multi-panel-free but
non-trivial factorization and a cross-covariance at a different row count
all at once).

THE CARD: sixteen stages, FNV-1a64 over raw bytes, in the sequence a
divergence would first show up in.

    gp.x_train    the training inputs as this run built them
    gp.y_train    the targets
    gp.kernel     K = k(X, X)                      -- the covariance function
    gp.ridged     K + alpha I                      -- the ridge (host replay)
    gp.factor     L, lower Cholesky of the above   -- the cholesky lane
    gp.dual_coef  K^-1 y                           -- the two substitutions
    gp.logdet     log |K + alpha I|                -- one device fold, one log
    gp.ydotalpha  y^T K^-1 y                       -- one host fold
    gp.lml        the log marginal likelihood      -- three terms
    gp.kss        k(x*, x*)                        -- a host scalar
    gp.kcross     k(X_train, X_star)               -- the covariance function
    gp.mean       K_trans alpha_                   -- the gemm profile
    gp.v          L^-1 k_star                      -- one triangular solve
    gp.var        k** - v^T v, clamped             -- one per-point fold
    gp.clamped    the per-test-point clamp FLAGS   -- Int32, DEVIATION 1760
    gp.std        sqrt(var)                        -- one root

**A CARD THAT DIVERGES HAS AN ADDRESS AND THE ADDRESS IS THE DIAGNOSIS.**

`gp.x_train` or `gp.y_train` moving means the FIXTURE moved and nothing
below it is comparable; that is a bug in the fixture or in the hash, never
in the algorithm.

`gp.kernel` moving with the inputs identical is the covariance function
itself: `identical_exp` (IDENTITY_PATHS row 12), `identical_sqrt` (row 10,
and DEVIATION 258's approximate NVIDIA sqrt), `identical_div` (row 49), the
`fma` pin (row 9) or the flush (row 10). Which of those it is has a further
address, because the sabotage arms map one to one onto them.

`gp.ridged` moving with `gp.kernel` identical is one float add on the
diagonal, so it is the flush and nothing else. If `gp.ridged` agrees and the
Cholesky lane's own `chol.jittered` (present in the same card when its trace
is enabled) does not, that is a HOST-versus-DEVICE disagreement about that
same add.

`gp.factor` moving with `gp.ridged` identical is the CHOLESKY LANE's
certificate and not this one's -- go read `chol.panelNNN.*` in the same card,
where the panel that moved has its own tag.

`gp.dual_coef` moving with `gp.factor` identical is the two triangular
substitutions. `gp.logdet` moving with `gp.factor` identical is
`identical_log` over the diagonal and nothing else. `gp.ydotalpha` moving
with `gp.dual_coef` identical is a HOST fold, which should be impossible
across vendors and would mean the host arithmetic is not what this lane
believes (`gaussian_process/estimator.mojo::_y_dot_alpha` states the belief).
`gp.lml` moving with those two identical is the three-term assembly, which
is `log(2 pi)` and two exact products.

`gp.kcross` moving with `gp.kernel` identical is a cross-covariance shape
issue rather than an arithmetic one, because the two go through the same
kernels. `gp.mean` moving with `gp.kcross` and `gp.dual_coef` identical is
the GEMM, which is `mojolearn.identical.gemm.fp32.v1`'s certificate and not
this one's. `gp.v` moving with `gp.kcross` and `gp.factor` identical is the
triangular solve, which is the Cholesky lane's. `gp.var` moving with `gp.v`
identical is this lane's own per-point fold or its clamp; `gp.clamped`
moving with `gp.var` identical is impossible by construction (the flag is
derived from the variance's bits) and would mean the flag is not derived
from what it says. `gp.std` moving with `gp.var` identical is
`identical_sqrt`.

**TWO CARDS HAVE BEEN EMITTED AND COMPARED**, both from
`gaussian_process/checks/gp_check.mojo` rather than from this driver, on
2026-08-28:

    Apple  bench/results/e1/2026-08-28_162228-MacBook-Air-1-terrabyte/lanes/
    AMD    bench/results/e1/2026-08-28_203552-mojolearn-e2-amd/lanes/

3,494 lines each, and the only eight lines that differ are inside the ONE
fit block the `GP_SAB_STD_EXP` sabotage arm produced. See
`bench/results/e1/GP_CROSS_VENDOR_DIVERGENCE.md`. This driver's own single
block has not been compared across two boxes.

**THE FIT AND PREDICT HEADERS NAME THE SABOTAGE ARM AND THE KERNEL'S
HYPERPARAMETERS**, since 2026-09-01, because one card file collects every
fit a run makes and until then thirty blocks of one fixture shared a
byte-identical header. `gaussian_process/estimator.mojo`'s comment above
`trace.header` carries the argument and the list of what is deliberately
left out.

WHAT THIS DRIVER IS NOT. It is not a benchmark and it prints no time. A
traced run drains the queue at every stage by construction
(`core/identity_trace.mojo` rule 4), so a number taken from it would be a
number about the instrument.
"""

from std.os import getenv

from gaussian_process.estimator import (
    gp_profile_alpha,
    gpr_fit_host,
    gpr_log_marginal_likelihood,
    gpr_predict_host,
)
from gaussian_process.checks.gp_fixture import (
    GP_FIX_ARD,
    GP_FIX_DUPLICATE,
    GP_FIX_HANDWORKED,
    GP_FIX_PLANTED,
    GP_FIX_SIGNED_ZERO,
    gp_fixture_alpha,
    gp_fixture_d,
    gp_fixture_kernel,
    gp_fixture_n,
    gp_fixture_n_star,
    gp_fixture_name,
    gp_fixture_x,
    gp_fixture_x_star,
    gp_fixture_y,
)
from gaussian_process.checks.kernels import (
    GP_PROFILE,
    gp_hex32_bits,
    gp_kernel_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    """Decimal AND hex, because `String(Float32)` does not round trip
    (`[[mojo-string-float-roundtrip]]`) and a printed number a reader
    cannot reproduce is a printed number nobody can check."""
    return String("0x") + gp_hex32_bits(v)


def _fixture_from_env() -> Int:
    var s = String(getenv("MOJOLEARN_GP_FIXTURE"))
    if s == "planted":
        return GP_FIX_PLANTED
    if s == "duplicate":
        return GP_FIX_DUPLICATE
    if s == "handworked":
        return GP_FIX_HANDWORKED
    if s == "signed_zero":
        return GP_FIX_SIGNED_ZERO
    return GP_FIX_ARD


def main() raises:
    var which = _fixture_from_env()
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var ns = gp_fixture_n_star(which)
    var x = gp_fixture_x(which, 0)
    var y = gp_fixture_y(which, 0)
    var xs = gp_fixture_x_star(which, 0)
    var spec = gp_fixture_kernel(which)
    var alpha = gp_fixture_alpha(which)

    print(
        "== gaussian_process/gp_main.mojo ["
        + _mode_name()
        + "] profile="
        + GP_PROFILE
        + " fixture="
        + gp_fixture_name(which)
        + " n_train="
        + String(n)
        + " n_star="
        + String(ns)
        + " d="
        + String(d)
        + " =="
    )
    print("  kernel  = " + gp_kernel_name(spec))
    print(
        "  alpha   = "
        + _hex32(alpha)
        + "  (the RIDGE, which IS the Cholesky profile's jitter;"
        " the pinned one is "
        + _hex32(gp_profile_alpha())
        + ")"
    )

    var model = gpr_fit_host(x, n, d, y, spec, alpha)
    if model.info != 0:
        # A pivot failure is a RESULT for a Gaussian process, not a crash:
        # it says this kernel and this ridge do not describe these points.
        # DEVIATION 1634.
        print(
            "  FIT DID NOT FACTOR: info="
            + String(model.info)
            + " -- the leading minor of that order of K + alpha I was not"
            " positive definite. Nothing below it exists"
        )
        return

    print("  info    = 0  (LAPACK's contract; nb=" + String(model.nb) + ")")
    print(
        "  log|K|  = "
        + String(model.logdet)
        + "  "
        + _hex32(model.logdet)
    )
    print(
        "  y^T a   = "
        + String(model.ydotalpha)
        + "  "
        + _hex32(model.ydotalpha)
    )
    var lml = gpr_log_marginal_likelihood(model)
    print("  lml     = " + String(lml) + "  " + _hex32(lml))

    var pred = gpr_predict_host(model, xs, ns, True)
    print(
        "  k(x*,x*)= "
        + String(pred.kss)
        + "  "
        + _hex32(pred.kss)
    )
    var upto = 8
    if ns < upto:
        upto = ns
    for i in range(upto):
        print(
            "    mean["
            + String(i)
            + "] = "
            + String(pred.mean[i])
            + "  "
            + _hex32(pred.mean[i])
            + "   std = "
            + String(pred.std[i])
            + "  "
            + _hex32(pred.std[i])
            + "   clamped = "
            + String(Int(pred.clamped[i]))
        )
    # **THE CLAMP COUNT IS PRINTED WHETHER IT IS ZERO OR NOT.** DEVIATION
    # 1760: a Gaussian process that clamps a negative predictive variance
    # and says nothing is a Gaussian process that lies quietly.
    print(
        "  CLAMPED: "
        + String(pred.n_clamped)
        + " of "
        + String(ns)
        + " predictive variances came out non-positive and were replaced"
        " by +0.0"
    )

    var trace_path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if trace_path == "":
        print(
            "  no MOJOLEARN_IDENTITY_TRACE set: the fit and the prediction"
            " ran, no card was written"
        )
    else:
        print(
            "  card written to "
            + trace_path
            + " (16 gp.* stages; the cholesky lane's own chol.* stages are"
            " in the same file, because cholesky_factor_host and"
            " cholesky_solve_host read the same environment variable)"
        )
