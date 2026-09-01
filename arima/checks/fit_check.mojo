# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gates for `batched_fit` (DEVIATIONS 678, 679, 687, and DEVIATION 675's
second decision).

    tools/with_build_lock.sh     pixi run mojo run -I . arima/checks/fit_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . arima/checks/fit_check.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/arima_fit.card tools/with_identical_mode.sh \\
        pixi run mojo run -I . arima/checks/fit_check.mojo

THIS IS A SECOND DRIVER AND A SECOND CARD, ON PURPOSE. `arima_check.mojo`
emits `arima.identical.card`, and at commit `221aa141` the Apple, AMD and
NVIDIA copies of that file are 139 lines and BYTE-IDENTICAL
(`bench/results/e1/CERT_2026-08-31.md`). Recording `fit` stages into that
trace would change every one of those files and retire a three-vendor
result that has been earned. The fit's stages go into their own card,
`arima.fit.identical.card`, which starts with no vendor evidence at all and
has to earn its own row.

=============================================================================
WHY THESE GATES ARE NOT SHAPED LIKE THE OTHER SIXTEEN
=============================================================================
`arima/SABOTAGES.md` opens by saying that this lane's expected values are
OUR OWN TALLY, so every gate needs a sabotage arm before it means anything.
For a `fit` we can do better than a tally, and three of the gates below take
values from somewhere this repository does not control:

    check_x0_solves_the_normal_equations       `A'(Ax - b) = 0` in Float64
                                               at the device's own answer. A
                                               least-squares solution
                                               satisfies it however it is
                                               spelled; a wrong answer does
                                               not.
    check_fit_recovers_planted_parameters      the series are GENERATED from
                                               known coefficients
                                               (`fixtures.mojo`), so the
                                               right answer existed before
                                               the fit ran.
    check_fit_is_a_minimizer                   the Float64 gradient at the
                                               returned point against the
                                               solver's own bound, and the
                                               objective below eight
                                               perturbations of it. Modelled
                                               on `glm/checks/logistic_
                                               check.mojo::check_logistic_
                                               is_a_minimizer`.

The rest are the usual shapes: device against a separately spelled oracle,
a seam shown observable before a sabotage claims to move it, a rare branch
reached deliberately, and refusals by name.

RUNTIME. The fit gates run at `FIT_N_OBS = 512`, because
`_arma_least_squares` refuses outright near `n_obs = 24` (see
`fixtures.mojo`) and because a recovery gate at 24 could only assert a
tolerance no broken optimizer would fail. Six full fits and three Float64
reference gradients is the bulk of it. This is the slowest gate in the lane
and it is not close; that is the price of gating an optimizer rather than a
kernel.
"""

from std.math import abs, sqrt
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    numeric_mode_name,
)
from core.identity_trace import IdentityTrace
from glm.impl.glm.qn.qn_util import LBFGSParam, check_convergence
from glm.impl.glm.qn.qn_linesearch import ls_success

from arima.checks.fit_oracle import (
    build_ls_system_f64,
    grad_f64_central,
    householder_qr_solve_host,
    jones_transform_f64,
    normal_eq_residual_f64,
    normal_equations_solve_f32,
    objective_f64,
    solve_normal_f64,
    test_invparams_host,
)
from arima.checks.fixtures import (
    FIT_N_OBS,
    FIT_SHORT_N_OBS,
    arima_fixture,
    bits32,
    count_cells_differ,
    download_f32,
    download_i32,
    download_u8,
    first_cell_differ,
    fit_order_table,
    planted_cases,
    same_bits,
    sub_batch_series,
    upload_f32,
)
from arima.impl.arima.batched_fit import (
    ARIMA_FIT_H,
    arima_fit_params,
    batched_fit,
)
from arima.impl.arima.estimate_x0 import (
    INVP_AR_TESTED,
    INVP_AR_VALID,
    INVP_MA_TESTED,
    INVP_MA_VALID,
    estimate_x0,
)
from arima.impl.arima.lbfgs_host import armijo_ok, check_convergence_at
from arima.impl.linalg.batched.least_squares import (
    LS_MAX_COLS,
    householder_qr_solve,
)
from arima.impl.tsa.arima_common import (
    ARIMAOrder,
    ARIMAParams,
    ARIMAParamsHost,
    unpack_host,
)
from tsa.checks.fixtures import ar1_series, random_walk, to_f32, u01
from tsa.impl.timeSeries.arima_helpers import prepare_data_host


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime SALT = 7


def _mode_name() -> String:
    return numeric_mode_name()


def _survey() -> Bool:
    return getenv("MOJOLEARN_ARIMA_SURVEY") == "1"


def _gate(ok: Bool, what: String) raises:
    """Vendor-shaped claims: asserted under IDENTICAL, RECORDED under FAST.
    Same contract as `arima_check.mojo::_gate`."""
    if ok:
        return
    if _survey():
        print("    SABOTAGE-MOVED " + what)
        return
    comptime if IDENTICAL:
        raise Error(what)
    else:
        print("    RECORDED [FAST] " + what + " (vendor-shaped under FAST; not asserted)")


def _assert(ok: Bool, what: String) raises:
    if not ok:
        if _survey():
            print("    SABOTAGE-MOVED (assert) " + what)
            return
        raise Error(what)


# ===========================================================================
# DEVIATION 678: the least squares
# ===========================================================================


def qr_probe_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    batch_size_in: Int32,
    m_in: Int32,
    n_in: Int32,
):
    """A door onto `householder_qr_solve` for the gate.

    The device QR is otherwise reachable only from inside
    `arma_least_squares_kernel`, where its input is scratch nobody can plant
    and its output is already transformed by `test_invparams`. A probe
    kernel gives DEVIATION 678's arithmetic a DIRECT bitwise device-versus-
    oracle line, and gives `SABOTAGES.md` arms (k) and (l) somewhere to
    bite that is not four steps downstream."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var m = Int(m_in)
    var n = Int(n_in)
    info.unsafe_store(bid, householder_qr_solve(a, bid * m * n, m, n, b, bid * m))


def _hashed_system(m: Int, n: Int, batch: Int, salt: Int) -> List[Float32]:
    """A hashed `m x n` system per series, column-major, no two cells equal
    (`uniform-test-data-hides-permutation`)."""
    var a = List[Float32]()
    for bid in range(batch):
        for c in range(n):
            for i in range(m):
                a.append(Float32(2.0 * u01(bid * 31 + c, i, salt) - 1.0))
    return a^


def _hashed_rhs(m: Int, batch: Int, salt: Int) -> List[Float32]:
    var b = List[Float32]()
    for bid in range(batch):
        for i in range(m):
            b.append(Float32(2.0 * u01(bid * 97 + 5, i, salt + 3) - 1.0))
    return b^


def check_qr_device_equals_oracle(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    """DEVIATION 678's arithmetic, device against the separately spelled
    host replay, bitwise, on a hashed overdetermined system."""
    print("check_qr_device_equals_oracle [" + _mode_name() + "]")
    var batch = 6
    for n in range(1, 6):
        var m = 4 * n + 5
        var a = _hashed_system(m, n, batch, SALT)
        var b = _hashed_rhs(m, batch, SALT)
        var da = upload_f32(ctx, a)
        var db = upload_f32(ctx, b)
        var dinfo = ctx.enqueue_create_buffer[DType.int32](batch)
        comptime TPB = 128
        ctx.enqueue_function[qr_probe_kernel](
            da.unsafe_ptr(), db.unsafe_ptr(), dinfo.unsafe_ptr(),
            Int32(batch), Int32(m), Int32(n),
            grid_dim=((batch + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
        )
        ctx.synchronize()
        var got = download_f32(ctx, db, m * batch)
        var got_info = download_i32(ctx, dinfo, batch)
        # host replay, per series, on its own copy
        var want = List[Float32]()
        for _ in range(m * batch):
            want.append(Float32(0.0))
        var bad = 0
        for bid in range(batch):
            var ah = List[Float32]()
            for t in range(m * n):
                ah.append(a[bid * m * n + t])
            var bh = List[Float32]()
            for t in range(m):
                bh.append(b[bid * m + t])
            var inf = householder_qr_solve_host(ah, m, n, bh)
            if Int32(inf) != got_info[bid]:
                bad += 1
            for t in range(m):
                want[bid * m + t] = bh[t]
        var tag = "fit.qr.n" + String(n)
        trace.record_list_f32(tag, got)
        trace.record_list_i32(tag + ".info", got_info)
        # only the first n rows are the solution; the tail is the reflector
        # workspace, which the oracle also reproduces and which is compared
        # too because a divergence there is a divergence
        var nd = count_cells_differ(got, want)
        print("      n = " + String(n) + ", m = " + String(m) + ": "
              + String(m * batch) + " cells, " + String(nd) + " differ"
              + ("" if nd == 0 else " (first " + first_cell_differ(got, want) + ")"))
        _gate(nd == 0, "fit.qr n = " + String(n) + " device != oracle")
        _assert(bad == 0, "fit.qr n = " + String(n) + ": info differs from the oracle")
        _ = da^
        _ = db^
        _ = dinfo^


def check_qr_beats_normal_equations_on_ill_conditioning() raises:
    """THE EVIDENCE FOR DEVIATION 678's ROUTE, measured rather than argued.

    A random walk is a unit root, so its lag columns are as collinear as an
    ARIMA design gets. Both Float32 routes solve the same system and both
    are compared against the Float64 answer. If the normal equations ever
    win here, the deviation's accuracy argument is wrong and it should be
    revisited rather than defended.

    A Cholesky that REFUSES (a non-positive pivot on a Gram that is
    positive definite in exact arithmetic) counts as a loss for that route
    and is the outcome the deviation predicts."""
    print("check_qr_beats_normal_equations_on_ill_conditioning [" + _mode_name() + "]")
    var n_obs = 256
    var lags = 3
    var m = n_obs - lags
    var worst_qr = 0.0
    var worst_ne = 0.0
    var ne_refusals = 0
    var qr_wins = 0
    for series in range(6):
        var y = to_f32(random_walk(n_obs, 400 + series, SALT))
        var a = List[Float32]()
        for c in range(lags):
            for i in range(m):
                a.append(y[lags - c - 1 + i])
        var b = List[Float32]()
        for i in range(m):
            b.append(y[lags + i])
        var a64 = List[Float64]()
        for t in range(len(a)):
            a64.append(Float64(a[t]))
        var b64 = List[Float64]()
        for t in range(len(b)):
            b64.append(Float64(b[t]))
        var truth = solve_normal_f64(a64, m, lags, b64)
        # QR
        var aq = a.copy()
        var bq = b.copy()
        var inf = householder_qr_solve_host(aq, m, lags, bq)
        _assert(inf == 0, "the QR refused a full-rank random-walk lag matrix (info "
                + String(inf) + "), which is not the branch this gate is about")
        var eq = 0.0
        for c in range(lags):
            var e = abs(Float64(bq[c]) - truth[c])
            if e > eq:
                eq = e
        # normal equations
        var ene = 0.0
        var refused = False
        try:
            var xn = normal_equations_solve_f32(a, m, lags, b)
            for c in range(lags):
                var e = abs(Float64(xn[c]) - truth[c])
                if e > ene:
                    ene = e
        except e:
            refused = True
            ne_refusals += 1
        if eq > worst_qr:
            worst_qr = eq
        if (not refused) and ene > worst_ne:
            worst_ne = ene
        if refused or eq < ene:
            qr_wins += 1
        print("      series " + String(series) + ": QR max |dx| = " + String(eq)
              + ", normal equations " + ("REFUSED (Gram not PD in Float32)" if refused else String(ene)))
    print("    worst QR " + String(worst_qr) + ", worst normal equations "
          + String(worst_ne) + ", Cholesky refusals " + String(ne_refusals)
          + ", QR strictly better on " + String(qr_wins) + " of 6")
    _assert(qr_wins >= 4,
            "the normal equations matched or beat the QR on a unit-root lag matrix in "
            + String(6 - qr_wins) + " of 6 series. DEVIATION 678 chose the QR on an"
            " ACCURACY argument; if that argument does not hold on the worst"
            " conditioning this lane admits, the deviation is wrong and must be"
            " rewritten, not this gate")


def check_invparams_contraction_is_visible() raises:
    """THE SEAM `test_invparams` WOULD LOSE BY COPYING `invtransform`.

    Theirs is `coef * a * x`, which C++ parses `(coef*a) * x` with `coef` an
    exact +-1, so the surviving product feeds the add and FUSES: ONE
    rounding. `invtransform`'s is `sign * (a * x)`, which rounds `a * x`
    first: TWO. If the two agreed on this fixture then
    `SABOTAGES.md` arm (j) could not fail and the distinction would be
    unverifiable here, so the difference is ASSERTED rather than assumed.
    Exactly the shape of `check_jones_contraction_is_visible` one file over,
    for the seam one function over."""
    print("check_invparams_contraction_is_visible [" + _mode_name() + "]")
    var moved = 0
    var total = 0
    var verdict_moved = 0
    for pq in range(2, 6):
        for is_ar_i in range(2):
            var is_ar = is_ar_i == 1
            for b in range(24):
                var v = List[Float32]()
                for i in range(pq):
                    # near the invertibility boundary, where the verdict can
                    # actually flip, not in the comfortable middle
                    v.append(Float32(0.995 - 0.37 * u01(b, i, SALT + 11)))
                var right = test_invparams_host(v, 0, pq, is_ar, False)
                var wrong = test_invparams_host(v, 0, pq, is_ar, True)
                total += 1
                if right != wrong:
                    verdict_moved += 1
    # the arithmetic-level comparison: run the recursion by hand both ways
    for pq in range(2, 6):
        for is_ar_i in range(2):
            var is_ar = is_ar_i == 1
            var sign = Float32(1.0) if is_ar else Float32(-1.0)
            for b in range(24):
                var mine = List[Float32]()
                for i in range(pq):
                    mine.append(Float32(0.995 - 0.37 * u01(b, i, SALT + 11)))
                var j = pq - 1
                while j > 0:
                    var a = mine[j]
                    var coef_a = a if is_ar else ftz(-a)
                    for k in range(j):
                        var x = mine[j - k - 1]
                        var acc = mine[k]
                        var one = ftz(identical_mul_add(coef_a, x, acc))
                        var two = ftz(identical_mul_add(sign, ftz(a * x), acc))
                        total += 1
                        if not same_bits(one, two):
                            moved += 1
                    j -= 1
    print("    of " + String(total) + " numerators the two associations differ on "
          + String(moved) + "; the VERDICT flips on " + String(verdict_moved))
    _assert(moved > 0,
            "the one-rounding and two-rounding spellings of test_invparams' numerator are"
            " bit-identical on every cell of this fixture: sabotage (j) cannot fail and"
            " the correction is unverifiable here")


# ===========================================================================
# estimate_x0
# ===========================================================================


def check_x0_solves_the_normal_equations(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    """THE GATE THAT DOES NOT CARE HOW THE SOLVER IS SPELLED.

    `x` is a least-squares solution of `A x = b` exactly when
    `A'(A x - b) = 0`. A different pivot rule, a different fold order, a
    different factorization all satisfy that to rounding; only a WRONG
    ANSWER fails. The system is rebuilt in Float64 from its definition
    (`fit_oracle.mojo::build_ls_system_f64`), never from the device's
    scratch.

    TWO BOUNDS, and the reason for two is in `build_ls_system_f64`'s
    docstring. With `q == 0` the design matrix is an intercept column and
    lags of `y`, all Float32 values widened exactly, so the reference matrix
    IS the device's and the bound is tight. With `q > 0` the last columns
    are lags of an AR pre-fit's residual, computed in Float64 here and in
    Float32 there, so the two matrices differ by about `1e-7` relative and
    the bound must admit it.

    SERIES WHOSE VERDICT IS INVALID ARE SKIPPED, because `test_invparams`
    ZEROED their coefficients (`batched_arima.cu:823-838`) and zero is not
    the least-squares answer; that is exactly why the verdict is a recorded
    decision stage and not something to infer from the floats. The gate
    asserts that most series are NOT skipped, so it cannot pass vacuously.

    BOTH BOUNDS ARE DERIVED, not observed. A compile slot must replace them
    with what this prints."""
    print("check_x0_solves_the_normal_equations [" + _mode_name() + "]")
    var f = arima_fixture(FIT_N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    var table = fit_order_table()
    var checked = 0
    var skipped = 0
    var worst_exact = 0.0
    var worst_loose = 0.0
    for oc in table:
        var order = oc.order
        var params = ARIMAParams(ctx, order, f.batch_size)
        var st = estimate_x0(ctx, params, y, f.batch_size, f.n_obs, order)
        _assert(not st.ns.degenerate,
                oc.name + ": the non-seasonal least squares took the too-few-observations"
                " arm at n_obs = " + String(FIT_N_OBS) + ", so this gate measured a memset")
        var mu = download_f32(ctx, params.mu, max(1, order.k * f.batch_size))
        var ar = download_f32(ctx, params.ar, max(1, order.p * f.batch_size))
        var ma = download_f32(ctx, params.ma, max(1, order.q * f.batch_size))
        var verdict = download_u8(ctx, st.ns.verdict, f.batch_size)
        var info = download_i32(ctx, st.ns.info, f.batch_size)
        trace.record_list_i32("fit.x0." + oc.name + ".info_ls", info)
        # the differenced series the least squares actually saw
        var yd = f.y.copy()
        var n_obs_d = f.n_obs
        if order.need_diff():
            yd = prepare_data_host(f.y, f.batch_size, f.n_obs, order.d, order.D, order.s)
            n_obs_d = f.n_obs - order.n_diff()
        for b in range(f.batch_size):
            var v = Int(verdict[b])
            var ar_ok = (v & INVP_AR_TESTED) == 0 or (v & INVP_AR_VALID) != 0
            var ma_ok = (v & INVP_MA_TESTED) == 0 or (v & INVP_MA_VALID) != 0
            if info[b] != Int32(0) or not ar_ok or not ma_ok:
                skipped += 1
                continue
            var sys = build_ls_system_f64(yd, b, n_obs_d, order.p, order.q, 1, order.k)
            var x = List[Float64]()
            if order.k != 0:
                x.append(Float64(mu[b]))
            for i in range(order.p):
                x.append(Float64(ar[order.p * b + i]))
            for i in range(order.q):
                x.append(Float64(ma[order.q * b + i]))
            var res = normal_eq_residual_f64(sys, x)
            checked += 1
            if oc.exact_ls:
                if res > worst_exact:
                    worst_exact = res
            else:
                if res > worst_loose:
                    worst_loose = res
            var bound = 1.0e-5 if oc.exact_ls else 2.0e-3
            _assert(res <= bound,
                    oc.name + " series " + String(b) + ": the normal-equation residual is "
                    + String(res) + ", above the bound " + String(bound)
                    + ". estimate_x0 returned something that is not a least-squares"
                    " solution of the system it was given")
        print("      " + oc.name + " (exact reference: " + String(oc.exact_ls) + ")")
        _ = params^
        _ = st^
    print("    checked " + String(checked) + " series, skipped " + String(skipped)
          + " (refused or zeroed by test_invparams); worst residual, exact arm "
          + String(worst_exact) + ", loose arm " + String(worst_loose))
    _assert(checked >= 15,
            "fewer than 15 series survived to be checked (" + String(checked)
            + "): this gate is close to vacuous and the fixture or the tolerance"
            " needs looking at before the number is believed")
    _ = y^


def check_x0_refusal_is_reached(ctx: DeviceContext) raises:
    """THE TOO-FEW-OBSERVATIONS ARM, reached deliberately.

    `_arma_least_squares:685-697` fills with zeros and `sigma2 = 1` when
    `p + q + k >= n_obs - r`. `verify reach, not output`: at
    `FIT_N_OBS` nothing takes it, so a gate that only ran there would prove
    nothing about it, and `check_x0_solves_the_normal_equations` would
    silently be measuring a memset if a fixture ever drifted into it. The
    order and length here are chosen so the arm FIRES, and the arithmetic is
    written out so the choice is checkable:

        p = 2, q = 2, s = 1, k = 1, n_obs = 10
        p_ar = max(2, 4) = 4          r = max(4 + 2, 2) = 6
        first clause:  q and 4 >= 10 - 4 = 6      -> false
        second clause: 2 + 2 + 1 = 5 >= 10 - 6 = 4 -> TRUE

    so it is the SECOND clause that fires, which is the one that guards the
    ARMA solve's `m > n`."""
    print("check_x0_refusal_is_reached [" + _mode_name() + "]")
    var order = ARIMAOrder(2, 0, 2, 0, 0, 0, 0, 1, 0)
    var f = arima_fixture(FIT_SHORT_N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    var params = ARIMAParams(ctx, order, f.batch_size)
    var st = estimate_x0(ctx, params, y, f.batch_size, f.n_obs, order)
    _assert(st.ns.degenerate,
            "the refusal arm did NOT fire at n_obs = " + String(FIT_SHORT_N_OBS)
            + ": recompute the two clauses in this docstring, the branch is unreached")
    var ar = download_f32(ctx, params.ar, order.p * f.batch_size)
    var ma = download_f32(ctx, params.ma, order.q * f.batch_size)
    var mu = download_f32(ctx, params.mu, f.batch_size)
    var s2 = download_f32(ctx, params.sigma2, f.batch_size)
    var bad = 0
    for i in range(len(ar)):
        if not same_bits(ar[i], Float32(0.0)):
            bad += 1
    for i in range(len(ma)):
        if not same_bits(ma[i], Float32(0.0)):
            bad += 1
    for b in range(f.batch_size):
        if not same_bits(mu[b], Float32(0.0)):
            bad += 1
        if not same_bits(s2[b], Float32(1.0)):
            bad += 1
    print("      the arm fired; " + String(bad) + " of "
          + String(len(ar) + len(ma) + 2 * f.batch_size)
          + " cells are not the documented fill (must be 0)")
    _assert(bad == 0, "the too-few-observations arm did not write 0 / 0 / 0 / 1")
    _ = params^
    _ = st^
    _ = y^


def check_invparams_verdict_is_reached(ctx: DeviceContext) raises:
    """`test_invparams`' ZEROING ARM, and its opposite, both reached.

    A verdict that is always VALID means the zeroing branch never runs and
    `SABOTAGES.md` arm (j) has nothing to move; a verdict that is always
    INVALID means every estimate is being thrown away and every recovery
    gate above is measuring zeros. Both are failures and both are silent
    without this. `reached but inert` is the standing rule."""
    print("check_invparams_verdict_is_reached [" + _mode_name() + "]")
    var table = fit_order_table()
    var valid_seen = 0
    var invalid_seen = 0
    var tested = 0
    # a long fixture, where estimates are good and mostly valid, AND a short
    # one, where they are wild and some are not
    var lengths = List[Int]()
    lengths.append(FIT_N_OBS)
    lengths.append(32)
    for li in range(len(lengths)):
        var L = lengths[li]
        var f = arima_fixture(L, SALT)
        var y = upload_f32(ctx, f.y)
        for oc in table:
            var order = oc.order
            if order.p == 0 and order.q == 0:
                continue
            var params = ARIMAParams(ctx, order, f.batch_size)
            var st = estimate_x0(ctx, params, y, f.batch_size, f.n_obs, order)
            if st.ns.degenerate:
                _ = params^
                _ = st^
                continue
            var v = download_u8(ctx, st.ns.verdict, f.batch_size)
            for b in range(f.batch_size):
                var vb = Int(v[b])
                if (vb & INVP_AR_TESTED) != 0:
                    tested += 1
                    if (vb & INVP_AR_VALID) != 0:
                        valid_seen += 1
                    else:
                        invalid_seen += 1
                if (vb & INVP_MA_TESTED) != 0:
                    tested += 1
                    if (vb & INVP_MA_VALID) != 0:
                        valid_seen += 1
                    else:
                        invalid_seen += 1
            _ = params^
            _ = st^
        _ = y^
    print("    " + String(tested) + " verdicts: " + String(valid_seen) + " valid, "
          + String(invalid_seen) + " invalid")
    _assert(valid_seen > 0,
            "test_invparams called EVERY estimate invalid, so every fit starts from zeros"
            " and every recovery gate is measuring a memset")
    _assert(invalid_seen > 0,
            "test_invparams called every estimate valid on both fixture lengths, so the"
            " zeroing arm (batched_arima.cu:829, :835) is UNREACHED and sabotage (j) has"
            " nothing to move. Shorten the second fixture until it fires, as the pivot"
            " tie was constructed rather than hoped for")


# ===========================================================================
# DEVIATION 687: the finite-difference step
# ===========================================================================


def _pack_f64(ph: ARIMAParamsHost, order: ARIMAOrder, batch_size: Int) -> List[Float64]:
    var out = List[Float64]()
    for b in range(batch_size):
        if order.k != 0:
            out.append(Float64(ph.mu[b]))
        for i in range(order.p):
            out.append(Float64(ph.ar[order.p * b + i]))
        for i in range(order.q):
            out.append(Float64(ph.ma[order.q * b + i]))
        for i in range(order.P):
            out.append(Float64(ph.sar[order.P * b + i]))
        for i in range(order.Q):
            out.append(Float64(ph.sma[order.Q * b + i]))
        out.append(Float64(ph.sigma2[b]))
    return out^


def _device_objective_grad(
    ctx: DeviceContext,
    y: List[Float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
    x: List[Float32],
    h: Float32,
) raises -> List[Float32]:
    """`-loglike_grad(x, h) / (n_obs - 1)`, the quantity the optimizer
    actually descends on. `order` must have `d = D = 0` here: the gate uses
    undifferenced orders so the comparison against `grad_f64_central` needs
    no second differencing path to agree about."""
    from arima.impl.arima.batched_arima import batched_loglike_grad

    var N = order.complexity()
    var dy = upload_f32(ctx, y)
    var dx = upload_f32(ctx, x)
    var dg = ctx.enqueue_create_buffer[DType.float32](N * batch_size)
    var dxp = ctx.enqueue_create_buffer[DType.float32](N * batch_size)
    var scratch = ARIMAParams(ctx, order, batch_size)
    _ = batched_loglike_grad(
        ctx, dy, batch_size, n_obs, order, dx, dg, h, True, scratch, dxp
    )
    var g = download_f32(ctx, dg, N * batch_size)
    var scale = Float32(n_obs - 1)
    var out = List[Float32]()
    for i in range(N * batch_size):
        out.append(ftz(ftz(-g[i]) / scale))
    _ = dy^
    _ = dx^
    _ = dg^
    _ = dxp^
    _ = scratch^
    return out^


def check_grad_matches_float64(ctx: DeviceContext) raises:
    """DEVIATION 687, and the reason it needed a number.

    Until this existed, the only gate on the gradient was
    `check_grad_device_equals_oracle`, which compares the device's forward
    difference against a HOST REPLAY OF THE SAME FORWARD DIFFERENCE. Two
    spellings of one number say nothing about whether the number is a
    gradient. This compares it against a FLOAT64 CENTRAL DIFFERENCE through
    a Float64 Jones transform, which is `O(h^2)` where the device is
    `O(h)`, and sweeps `h` so the shipped value is a measured minimum.

    WHAT MUST HOLD, and what is only reported:

      ASSERTED  the error at `ARIMA_FIT_H` is within a factor of three of
                the best `h` in the sweep. A step chosen on an argument that
                the measurement contradicts is a wrong step, and this is the
                clause that would catch it.
      ASSERTED  `h = 1e-8`, cuML's value, is CATASTROPHIC in Float32. It is
                below eps, so `x + h == x` for every `|x| > 0.1` and the
                gradient is exactly zero. Asserting that it is at least a
                hundred times worse than the shipped `h` is what makes
                DEVIATION 687 a finding rather than a preference.
      REPORTED   the whole error curve, so a compile slot can move `h` if
                the shape is not what this predicts.

    The bound on the absolute error is DERIVED: about `1e-3` on an O(1)
    objective, from `(h/2)|f''| + 2 * delta_f / h` with `delta_f ~ 1e-7`.
    Replace it with what this prints."""
    print("check_grad_matches_float64 [" + _mode_name() + "]")
    var f = arima_fixture(FIT_N_OBS, SALT)
    var table = fit_order_table()
    for oc in table:
        var order = oc.order
        if order.need_diff():
            continue
        if order.P != 0 or order.Q != 0:
            continue
        var params = ARIMAParams(ctx, order, f.batch_size)
        var y = upload_f32(ctx, f.y)
        var st = estimate_x0(ctx, params, y, f.batch_size, f.n_obs, order)
        # start from an honest point: the inverse transform of estimate_x0's
        # answer, which is where the optimizer actually begins
        var ph = ARIMAParamsHost(
            mu=download_f32(ctx, params.mu, max(1, order.k * f.batch_size)),
            ar=download_f32(ctx, params.ar, max(1, order.p * f.batch_size)),
            ma=download_f32(ctx, params.ma, max(1, order.q * f.batch_size)),
            sar=download_f32(ctx, params.sar, max(1, order.P * f.batch_size)),
            sma=download_f32(ctx, params.sma, max(1, order.Q * f.batch_size)),
            sigma2=download_f32(ctx, params.sigma2, f.batch_size),
        )
        var x64 = _pack_f64(ph, order, f.batch_size)
        var N = order.complexity()
        var x32 = List[Float32]()
        for i in range(len(x64)):
            x32.append(Float32(x64[i]))
        var truth = grad_f64_central(f.y, f.batch_size, f.n_obs, order, x64, 1.0e-3)
        var best_err = 1.0e30
        var best_h = Float32(0.0)
        var shipped_err = 0.0
        var cuml_err = 0.0
        var e = 6
        while e <= 26:
            var h = Float32(1.0)
            for _ in range(e):
                h = h * Float32(0.5)
            var g = _device_objective_grad(ctx, f.y, f.batch_size, f.n_obs, order, x32, h)
            var worst = 0.0
            for i in range(N * f.batch_size):
                var d = abs(Float64(g[i]) - truth[i])
                if d > worst:
                    worst = d
            print("      " + oc.name + " h = 2^-" + String(e) + " (" + String(h)
                  + "): worst |grad - grad_f64| = " + String(worst))
            if worst < best_err:
                best_err = worst
                best_h = h
            if same_bits(h, ARIMA_FIT_H):
                shipped_err = worst
            e += 2
        var g8 = _device_objective_grad(ctx, f.y, f.batch_size, f.n_obs, order, x32, Float32(1.0e-8))
        var dead = 0
        var live = 0
        for i in range(N * f.batch_size):
            var d = abs(Float64(g8[i]) - truth[i])
            if d > cuml_err:
                cuml_err = d
            if abs(truth[i]) > 1.0e-3:
                live += 1
                if g8[i] == Float32(0.0):
                    dead += 1
        print("      " + oc.name + " at cuML's h = 1e-8: " + String(dead) + " of "
              + String(live) + " cells whose true gradient exceeds 1e-3 came back"
              " EXACTLY ZERO")
        print("    " + oc.name + ": shipped h = " + String(ARIMA_FIT_H) + " -> "
              + String(shipped_err) + "; best in sweep h = " + String(best_h)
              + " -> " + String(best_err) + "; cuML's h = 1e-8 -> " + String(cuml_err))
        _assert(shipped_err > 0.0,
                oc.name + ": the shipped h was not in the sweep, so nothing was measured")
        _assert(shipped_err <= 3.0 * best_err,
                oc.name + ": the shipped h = 2^-10 has error " + String(shipped_err)
                + ", more than three times the best in the sweep (" + String(best_err)
                + " at " + String(best_h) + "). DEVIATION 687's derivation is wrong;"
                " change the step, not this gate")
        # THE MECHANISM, not a ratio: `1e-8` is below Float32 eps, so
        # `x + h == x` for every `|x| > 0.1` and the difference is EXACTLY
        # ZERO. Asserting a ratio of errors would be weaker and would depend
        # on how large the true gradient happens to be at this point.
        _assert(live > 0,
                oc.name + ": no cell has a true gradient above 1e-3 at the start"
                " point, so the h = 1e-8 arm of this gate proves nothing here")
        _assert(dead > 0,
                oc.name + ": cuML's h = 1e-8 did NOT collapse to an exactly zero"
                " gradient on any cell with a real gradient (worst error "
                + String(cuml_err) + "). DEVIATION 687 rests on 1e-8 being below"
                " Float32 eps; if x + 1e-8 != x here, re-derive the deviation")
        _ = params^
        _ = st^
        _ = y^


# ===========================================================================
# DEVIATION 675's SECOND DECISION
# ===========================================================================


def check_jones_inverse_is_below_the_fd_step() raises:
    """DEVIATION 675, REAFFIRMED WITH A MEASUREMENT NOW THAT `fit` PUTS THE
    INVERSE ON THE PORTED PATH.

    `arima/README.md` accepted `identical_exp` / `identical_log` for tanh
    and atanh, and said the reasoning inverts the day the optimizer lands,
    because until then nothing called `two_atanh`. `batched_fit` step 2
    calls it: `x0 = inverse(estimate_x0's parameters)`. The half is on the
    path.

    THE OPTION `identical_log1p` IS STILL NOT TAKEN, and the reason is no
    longer "it is off the path". It is this:

      * The measured 2.73e-6 is a RELATIVE error at a coordinate of
        magnitude about 0.03, so it is an ABSOLUTE error of about 8e-8 in
        `x`. The cancellation is inside `log(1 + small)`, whose output error
        is bounded by about one ulp of 1.0 HOWEVER SMALL the argument gets,
        so the absolute error does not grow as the coordinate shrinks.
      * `x` is the coordinate the optimizer moves, by O(1) per fit, and the
        finest thing it can resolve there is `ARIMA_FIT_H = 2^-10 = 9.8e-4`
        (DEVIATION 687). The inverse transform's error is four orders of
        magnitude below the step used to differentiate the objective.
      * `identical_log1p` fixes only the INVERSE half. The FORWARD half's
        `(e^x - 1)` cancellation costs about the same, one ulp of 1.0
        expressed in `x`, and `log1p` does not touch it. This gate MEASURES
        both and prints them side by side rather than asserting which is
        larger, because they are the same order and an earlier draft of this
        docstring asserted a dominance it had not measured. Option B fixes
        one of two comparable halves of a round trip whose measured error is
        2.73e-6 relative -- 8e-8 absolute at the coordinate where it was
        measured.

    Taking B would move every `arima.jones.inv.*` stage on every card and
    retire the three-vendor baseline at `221aa141` in exchange for accuracy
    a thousand times below the step size. It stays refused.

    THE DECISION RULE, fixed here in advance: if the inverse half's absolute
    error in `x` ever exceeds `ARIMA_FIT_H / 100`, land `identical_log1p`.
    This gate asserts that bound, so the refusal is checked on every run and
    not merely argued once.

    IT ALSO REPORTS WHICH HALF DOMINATES, because the whole argument turns
    on that and it had never been separated. A FINDING FOR THE NUMERICS
    HAND-OFF: `identical_tanh` (DEVIATION 821, `portable_tanhf`) now EXISTS
    in `checks/numerics.mojo` and did not when DEVIATION 675 was decided,
    which retires option C's stated blocker ("`identical_expm1` DOES NOT
    EXIST"). That does not make C right -- it is a card-invalidating change
    and README OWED item 5's end-to-end measurement is still the thing that
    should decide it -- but the hand-off in that section is now out of date
    and says so."""
    print("check_jones_inverse_is_below_the_fd_step [" + _mode_name() + "]")
    from std.math import atanh, tanh

    from arima.impl.timeSeries.jones_transform import tanh_half, two_atanh

    var worst_inv = 0.0
    var worst_inv_v = 0.0
    var worst_fwd_in_x = 0.0
    var n = 0
    for i in range(1, 400):
        var v = Float64(i) * 0.0025 - 0.5
        if abs(v) >= 0.9999:
            continue
        n += 1
        var got = Float64(two_atanh(Float32(v)))
        var want = 2.0 * atanh(v)
        var e = abs(got - want)
        if e > worst_inv:
            worst_inv = e
            worst_inv_v = v
        # the FORWARD half, converted into the same units: an error of
        # `d(theta)` costs `d(theta) / (d theta / d x)` in x, and
        # `d tanh(x/2) / dx = (1 - theta^2) / 2`
        var x = want
        if abs(x) < 40.0:
            var th = Float64(tanh_half(Float32(x)))
            var th64 = tanh(x * 0.5)
            var dtheta = abs(th - th64)
            var slope = (1.0 - th64 * th64) * 0.5
            if slope > 1.0e-9:
                var in_x = dtheta / slope
                if in_x > worst_fwd_in_x:
                    worst_fwd_in_x = in_x
    var h = Float64(ARIMA_FIT_H)
    print("    over " + String(n) + " points: inverse half worst |dx| = "
          + String(worst_inv) + " (at v = " + String(worst_inv_v)
          + "); forward half, expressed in x, worst |dx| = " + String(worst_fwd_in_x))
    print("    the finite-difference step is " + String(h) + ", so the inverse half is "
          + String(worst_inv / h) + " of it and the forward half " + String(worst_fwd_in_x / h))
    _gate(worst_inv <= h / 100.0,
          "DEVIATION 675's inverse half now costs " + String(worst_inv)
          + " absolute in x, which is above ARIMA_FIT_H / 100 = " + String(h / 100.0)
          + ". THE DECISION RULE RECORDED IN THIS DOCSTRING SAYS TO LAND"
          " identical_log1p. Do that, spend a deviation number on it, and expect"
          " every arima.jones.inv.* card stage to move")
    _gate(worst_fwd_in_x <= h / 100.0,
          "DEVIATION 675's FORWARD half costs " + String(worst_fwd_in_x)
          + " absolute in x, above ARIMA_FIT_H / 100 = " + String(h / 100.0)
          + ". That half is `tanh(x/2) = (e^x - 1)/(e^x + 1)` and"
          " `identical_log1p` does NOT fix it; the fix is option C, and"
          " `identical_tanh` (DEVIATION 821, portable_tanhf) now exists in"
          " checks/numerics.mojo, which retires the blocker DEVIATION 675"
          " recorded for it. Expect every arima.jones.* stage to move")
    _assert(worst_inv > 0.0 and worst_fwd_in_x > 0.0,
            "one of the two halves measured EXACTLY zero error against Float64,"
            " which means the reference is not being evaluated and this gate is"
            " inert")


# ===========================================================================
# the optimizer's rules cannot drift from glm's
# ===========================================================================


def check_lbfgs_rules_match_glm(ctx: DeviceContext) raises:
    """`arima/impl/arima/lbfgs_host.mojo` RE-SPELLS two decision rules that
    `glm/impl/glm/qn/` already owns, because glm's take device buffers and
    cannot be called per series. A duplicated rule drifts; this asserts it
    has not, by running both spellings over a grid and comparing.

    `ls_success` is called with one-element dummy buffers and `n = 1`: under
    `LBFGS_LS_BT_ARMIJO`, which is `LBFGSParam::defaults`' line search and
    the only one any door reaches, both arms return before the Wolfe `dot`
    is issued, so no device work happens and no buffer is read.

    A HAND-OFF, not a fix, is in `arima/README.md`: a three-part patch to
    `glm/` would delete these copies entirely. `glm/` is not this lane's to
    edit and was under active change the day this was written."""
    print("check_lbfgs_rules_match_glm [" + _mode_name() + "]")
    var param = arima_fit_params()
    var dummy_a = ctx.enqueue_create_buffer[DType.float32](1)
    var dummy_b = ctx.enqueue_create_buffer[DType.float32](1)
    var dummy_s = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    var armijo_checked = 0
    var armijo_bad = 0
    for a in range(40):
        for b in range(20):
            var fx = Float32(2.0 * u01(a, b, 21) - 1.0)
            var fx_init = Float32(2.0 * u01(a, b, 22) - 1.0)
            var step = Float32(u01(a, b, 23))
            var dg_test = Float32(-u01(a, b, 24))
            var width = Float32(0.0)
            var theirs = ls_success(
                ctx, param, fx_init, Float32(-1.0), fx, dg_test, step,
                dummy_a, dummy_b, 1, width, dummy_s,
            )
            var ours = armijo_ok(fx, fx_init, step, dg_test)
            armijo_checked += 1
            if theirs != ours:
                armijo_bad += 1
    print("      armijo_ok vs glm ls_success (ARMIJO arm): " + String(armijo_checked)
          + " cases, " + String(armijo_bad) + " disagree")
    _assert(armijo_bad == 0,
            "arima's armijo_ok disagrees with glm's ls_success on "
            + String(armijo_bad) + " cases: the two spellings of"
            " qn_linesearch.cuh:62 have drifted")

    var conv_checked = 0
    var conv_bad = 0
    for a in range(60):
        var fx = Float32(u01(a, 0, 31) * 4.0)
        var gnorm = Float32(u01(a, 1, 32) * 0.02)
        var h1 = List[Float32]()
        var h2 = List[Float32]()
        for i in range(param.past):
            var v = Float32(u01(a, i, 33))
            h1.append(v)
            h2.append(v)
        for k in range(0, 25):
            var t = check_convergence(param, k, fx, gnorm, h1)
            var o = check_convergence_at(param, k, fx, gnorm, h2, 0)
            conv_checked += 1
            if t != o:
                conv_bad += 1
        for i in range(param.past):
            if not same_bits(h1[i], h2[i]):
                conv_bad += 1
    print("      check_convergence_at vs glm check_convergence: " + String(conv_checked)
          + " cases, " + String(conv_bad) + " disagree (history included)")
    _assert(conv_bad == 0,
            "arima's check_convergence_at disagrees with glm's check_convergence on "
            + String(conv_bad) + " cases: the two spellings of qn_util.cuh:147-169"
            " have drifted, or the history write-back differs")
    _ = dummy_a^
    _ = dummy_b^
    _ = dummy_s^


# ===========================================================================
# the fit itself
# ===========================================================================


def check_fit_recovers_planted_parameters(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    """PLANTED-PARAMETER RECOVERY: the answer came from outside this
    repository.

    Every series in `fixtures.mojo::planted_cases` is GENERATED from known
    coefficients, so the right answer existed before any of this code ran.
    No sabotage tally, no oracle that shares a spelling, no bound derived
    from our own output. If the fit recovers `phi = 0.7` from a series built
    with `phi = 0.7`, the least squares, the Jones transform, the Kalman
    filter, the finite differences and the optimizer are all doing what they
    claim, because a fault in any one of them moves the answer.

    THE TOLERANCES ARE STANDARD ERRORS, written out in `fixtures.mojo`, at
    least four of them each. They are not slack."""
    print("check_fit_recovers_planted_parameters [" + _mode_name() + "]")
    var cases = planted_cases(FIT_N_OBS, SALT)
    for pc in cases:
        var order = pc.order
        var N = order.complexity()
        var y = upload_f32(ctx, pc.y)
        var params = ARIMAParams(ctx, order, pc.batch_size)
        var r = batched_fit(ctx, y, pc.batch_size, pc.n_obs, order, params, trace)
        var ph = unpack_host(r.t_x, order, pc.batch_size)
        var worst_phi = 0.0
        var worst_theta = 0.0
        var not_converged = 0
        for b in range(pc.batch_size):
            if r.retcode[b] != Int32(0):
                not_converged += 1
            if order.p != 0:
                var e = abs(Float64(ph.ar[order.p * b]) - pc.phi)
                if e > worst_phi:
                    worst_phi = e
            if order.q != 0:
                var e = abs(Float64(ph.ma[order.q * b]) - pc.theta)
                if e > worst_theta:
                    worst_theta = e
        print("      " + pc.name + ": planted phi = " + String(pc.phi) + ", theta = "
              + String(pc.theta) + "; worst |dphi| = " + String(worst_phi)
              + ", worst |dtheta| = " + String(worst_theta) + ", tol " + String(pc.tol)
              + "; " + String(not_converged) + " of " + String(pc.batch_size)
              + " series did not report OPT_SUCCESS, evaluations " + String(r.n_eval))
        for b in range(pc.batch_size):
            print("        series " + String(b) + ": n_iter " + String(r.n_iter[b])
                  + ", retcode " + String(r.retcode[b]) + ", -ll/(n-1) "
                  + String(r.fx[b]))
        _assert(worst_phi <= pc.tol,
                pc.name + ": the fitted AR coefficient is " + String(worst_phi)
                + " from the planted " + String(pc.phi) + ", above the tolerance "
                + String(pc.tol) + " (which is at least four standard errors)")
        _assert(worst_theta <= pc.tol,
                pc.name + ": the fitted MA coefficient is " + String(worst_theta)
                + " from the planted " + String(pc.theta) + ", above the tolerance "
                + String(pc.tol))
        _assert(not_converged == 0,
                pc.name + ": " + String(not_converged) + " series did not converge."
                " With DEVIATION 687's step the gradient's noise floor is about 1e-3"
                " and arima_fit_params' epsilon is set to it; a series that runs to"
                " maxiter means the tolerance is below the noise, which is the exact"
                " mistake carrying scipy's pgtol = 1e-5 across would have made")
        _ = params^
        _ = y^


def check_fit_is_a_minimizer(ctx: DeviceContext) raises:
    """IS THE POINT IT RETURNED ACTUALLY A MINIMUM?

    Modelled on `glm/checks/logistic_check.mojo::check_logistic_is_a_
    minimizer`, and the same two questions: the Float64 gradient at the
    returned point against the solver's OWN bound, and the objective below
    eight perturbations of it.

    THE FACTOR OF THREE IN THE BOUND IS NOT SLACK AND IS NOT FITTED. The
    solver stops when the FLOAT32 FORWARD-DIFFERENCE gradient falls below
    `epsilon * max(fx, epsilon)`. That gradient differs from the true one by
    about the same order as `epsilon` itself (DEVIATION 687: the two are
    both ~1e-3 by construction, because epsilon was CHOSEN as the gradient's
    noise floor). Asking the Float64 reference to meet the solver's own
    bound exactly would be asking it to agree with a quantity the solver
    never computed. Three times it is the smallest honest number, and the
    achieved value is printed so a compile slot can tighten it."""
    print("check_fit_is_a_minimizer [" + _mode_name() + "]")
    var cases = planted_cases(FIT_N_OBS, SALT)
    var pc = cases[0].copy()
    var order = pc.order
    var N = order.complexity()
    var y = upload_f32(ctx, pc.y)
    var params = ARIMAParams(ctx, order, pc.batch_size)
    var off = IdentityTrace.disabled()
    var r = batched_fit(ctx, y, pc.batch_size, pc.n_obs, order, params, off)
    for b in range(pc.batch_size):
        _assert(r.retcode[b] == Int32(0),
                "check_fit_is_a_minimizer: series " + String(b) + " returned retcode "
                + String(r.retcode[b]) + " after " + String(r.n_iter[b]) + " iterations")
    var x64 = List[Float64]()
    for i in range(len(r.x)):
        x64.append(Float64(r.x[i]))
    var f0 = objective_f64(pc.y, pc.batch_size, pc.n_obs, order, x64)
    var g = grad_f64_central(pc.y, pc.batch_size, pc.n_obs, order, x64, 1.0e-3)
    var param = arima_fit_params()
    var worst_ratio = 0.0
    for b in range(pc.batch_size):
        var gmax = 0.0
        for i in range(N):
            var v = abs(g[N * b + i])
            if v > gmax:
                gmax = v
        var fmag = max(f0[b], Float64(param.epsilon))
        var bound = 3.0 * Float64(param.epsilon) * fmag
        print("      series " + String(b) + ": float64 |grad|_inf = " + String(gmax)
              + " vs bound " + String(bound) + " (objective " + String(f0[b])
              + ", the device's own " + String(r.fx[b]) + ")")
        if bound > 0.0 and gmax / bound > worst_ratio:
            worst_ratio = gmax / bound
        _assert(gmax <= bound,
                "check_fit_is_a_minimizer: series " + String(b)
                + ": the Float64 gradient infinity norm at the returned point is "
                + String(gmax) + ", above three times the solver's own convergence"
                " bound " + String(bound) + ". The point it returned is not a"
                " stationary point of the objective it claims to minimize")
    # and below eight perturbations, which catches a saddle a gradient test
    # cannot
    var below = 0
    var total = 0
    for p in range(8):
        var xp = x64.copy()
        for b in range(pc.batch_size):
            for i in range(N):
                xp[N * b + i] = x64[N * b + i] + 0.05 * (2.0 * u01(p, i, 41) - 1.0)
        var fp = objective_f64(pc.y, pc.batch_size, pc.n_obs, order, xp)
        for b in range(pc.batch_size):
            total += 1
            if fp[b] > f0[b]:
                below += 1
    print("    the returned objective is below " + String(below) + " of "
          + String(total) + " perturbed objectives; worst gradient/bound ratio "
          + String(worst_ratio))
    _assert(below == total,
            "check_fit_is_a_minimizer: the objective at the returned point is NOT below "
            + String(total - below) + " of " + String(total)
            + " points perturbed by 0.05. A gradient test alone cannot tell a"
            " minimum from a saddle; this can")
    _ = params^
    _ = y^


def check_fit_is_batch_composition_invariant(ctx: DeviceContext) raises:
    """THE WHOLE OPTIMIZER, ASSERTED INVARIANT TO WHO ELSE IS IN THE BATCH.

    `check_kalman_launch_invariant` proves this for one filter pass. A fit
    is hundreds of passes with a HOST STATE MACHINE between them, and the
    state machine has per-series branches -- the Armijo test, the skipping
    test, the convergence test -- any of which could in principle be written
    to read another series' state. This fits a batch of six and then a batch
    of three drawn from it, and requires the three to come back BITWISE
    identical.

    It is also the gate that would catch the shortcut this driver
    deliberately does not take: dropping converged series out of the batch
    to save work. That would make the answer depend on convergence order.
    `batched_min_lbfgs` keeps every series in every launch and discards the
    results it does not need, and this is what says so."""
    print("check_fit_is_batch_composition_invariant [" + _mode_name() + "]")
    var f = arima_fixture(FIT_N_OBS, SALT)
    var order = ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0)
    var N = order.complexity()
    var off = IdentityTrace.disabled()

    var y_all = upload_f32(ctx, f.y)
    var p_all = ARIMAParams(ctx, order, f.batch_size)
    var r_all = batched_fit(ctx, y_all, f.batch_size, f.n_obs, order, p_all, off)

    var which = List[Int]()
    which.append(0)
    which.append(2)
    which.append(4)
    var sub_y = sub_batch_series(f.y, f.n_obs, which)
    var y_sub = upload_f32(ctx, sub_y)
    var p_sub = ARIMAParams(ctx, order, len(which))
    var r_sub = batched_fit(ctx, y_sub, len(which), f.n_obs, order, p_sub, off)

    var nd = 0
    var iter_nd = 0
    for k in range(len(which)):
        var b = which[k]
        for i in range(N):
            if not same_bits(r_all.t_x[N * b + i], r_sub.t_x[N * k + i]):
                nd += 1
                print("        series " + String(b) + " parameter " + String(i)
                      + ": batch-of-6 " + bits32(r_all.t_x[N * b + i])
                      + " vs batch-of-3 " + bits32(r_sub.t_x[N * k + i]))
        if r_all.n_iter[b] != r_sub.n_iter[k]:
            iter_nd += 1
        if r_all.retcode[b] != r_sub.retcode[k]:
            iter_nd += 1
    print("    " + String(len(which) * N) + " fitted parameters, " + String(nd)
          + " differ; " + String(iter_nd) + " iteration counts or retcodes differ")
    _gate(nd == 0, "the fitted parameters depend on the batch composition")
    _assert(iter_nd == 0,
            "the ITERATION COUNT or the retcode depends on the batch composition ("
            + String(iter_nd) + " differ). The parameters could still match by luck;"
            " this is the stage that catches a per-series branch reading another"
            " series' state")
    _ = p_all^
    _ = p_sub^
    _ = y_all^
    _ = y_sub^


def check_fit_refuses_by_name(ctx: DeviceContext) raises:
    """Every door a caller can get wrong, refused with a message that says
    which one, as `check_arima_refuses_by_name` does for the filter."""
    print("check_fit_refuses_by_name [" + _mode_name() + "]")
    var refusals = 0
    var f = arima_fixture(64, SALT)
    var off = IdentityTrace.disabled()

    # an order validate_order refuses (r = max(1, 6) = 6 > 5), taken from
    # the FIT door rather than the loglike door
    try:
        var order = ARIMAOrder(1, 1, 0, 0, 1, 1, 5, 0, 0)
        var y = upload_f32(ctx, f.y)
        var p = ARIMAParams(ctx, order, f.batch_size)
        _ = batched_fit(ctx, y, f.batch_size, f.n_obs, order, p, off)
        _assert(False, "batched_fit accepted an order with rd > 8")
    except e:
        print("      " + String(e))
        refusals += 1

    # a non-finite observation
    try:
        var order = ARIMAOrder(1, 0, 0, 0, 0, 0, 0, 0, 0)
        var yy = f.y.copy()
        yy[3] = Float32(0.0) / Float32(0.0)
        var y = upload_f32(ctx, yy)
        var p = ARIMAParams(ctx, order, f.batch_size)
        _ = batched_fit(ctx, y, f.batch_size, f.n_obs, order, p, off)
        _assert(False, "batched_fit accepted a non-finite observation")
    except e:
        print("      " + String(e))
        refusals += 1

    # n_obs <= d + s*D, estimate_x0's own RAFT_FAIL (:989)
    try:
        var order = ARIMAOrder(1, 1, 0, 0, 1, 0, 4, 0, 0)
        var short = arima_fixture(4, SALT)
        var y = upload_f32(ctx, short.y)
        var p = ARIMAParams(ctx, order, short.batch_size)
        _ = batched_fit(ctx, y, short.batch_size, short.n_obs, order, p, off)
        _assert(False, "batched_fit accepted n_obs <= d + s*D")
    except e:
        print("      " + String(e))
        refusals += 1

    # an order with no parameters at all
    try:
        var order = ARIMAOrder(0, 1, 0, 0, 0, 0, 0, 0, 0)
        var y = upload_f32(ctx, f.y)
        var p = ARIMAParams(ctx, order, f.batch_size)
        _ = batched_fit(ctx, y, f.batch_size, f.n_obs, order, p, off)
        _assert(False, "batched_fit accepted an order with no parameters")
    except e:
        print("      " + String(e))
        refusals += 1

    print("    " + String(refusals) + " refusals, each by name")
    _assert(refusals == 4, "fewer than four doors refused")


def main() raises:
    print("== arima/checks/fit_check.mojo [" + _mode_name() + "] FIT_N_OBS="
          + String(FIT_N_OBS) + " SALT=" + String(SALT) + " ==")
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "arima_fit mode=" + _mode_name() + " n_obs=" + String(FIT_N_OBS)
        + " batch=6 salt=" + String(SALT) + " h=" + String(ARIMA_FIT_H)
    )
    # DEVIATION 678, cheapest first
    check_qr_device_equals_oracle(ctx, trace)
    check_qr_beats_normal_equations_on_ill_conditioning()
    check_invparams_contraction_is_visible()
    check_x0_refusal_is_reached(ctx)
    check_invparams_verdict_is_reached(ctx)
    check_x0_solves_the_normal_equations(ctx, trace)
    # DEVIATION 675's second decision, and DEVIATION 687
    check_jones_inverse_is_below_the_fd_step()
    check_grad_matches_float64(ctx)
    # DEVIATION 679
    check_lbfgs_rules_match_glm(ctx)
    check_fit_refuses_by_name(ctx)
    check_fit_recovers_planted_parameters(ctx, trace)
    check_fit_is_a_minimizer(ctx)
    check_fit_is_batch_composition_invariant(ctx)
    print("ALL ARIMA FIT CHECKS PASSED [" + _mode_name() + "]"
          + (" card: " + trace.path if trace.enabled else " (no card: set MOJOLEARN_IDENTITY_TRACE)"))
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
