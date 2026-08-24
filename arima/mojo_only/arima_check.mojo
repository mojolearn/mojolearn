"""Gates for the batched ARIMA Kalman filter (DEVIATIONS 670, 673-679).

    tools/with_build_lock.sh     pixi run mojo run -I . arima/mojo_only/arima_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . arima/mojo_only/arima_check.mojo

Every printed line carries the mode the binary COMPILED in. The device-vs-
oracle bitwise lines ASSERT under IDENTICAL and print `RECORDED [FAST]`
under FAST (the transcendentals are the stdlib's there, DEVIATION 675);
refusals, the Float64 tolerance and the structural claims assert in both.

    check_jones_device_equals_oracle    the Jones transform and its inverse,
                                        AR and MA, p = 1..4, device vs the
                                        host replay, bitwise; and
                                        inverse(forward(x)) back to x, the
                                        round-trip error REPORTED in ulp
    check_kalman_device_equals_oracle   EVERY stage per cell -- Z, R, T, RQ,
                                        RQR, P0, alpha0, pred, vs, loglike,
                                        fc -- for all seven orders of
                                        `fixtures.order_table()`
    check_lyapunov_solves_the_equation  the residual `Ts P Ts' - P + RQRs`
                                        per cell against a Float64 solve of
                                        the same system (DEVIATION 674)
    check_predict_device_equals_oracle  `predict`'s in-sample block, the
                                        undifferenced forecast and the
                                        NaN sentinel (DEVIATION 676),
                                        bitwise against the host replay
    check_grad_device_equals_oracle     `batched_loglike_grad`'s forward
                                        differences, bitwise, and the reset
                                        of a `-0.0` parameter (the audit
                                        defect: `x + 0` is not a copy)
    check_kalman_matches_float64        the log-likelihood and P0 against
                                        the Float64 reference (DEVIATION 670)
    check_arima_refuses_by_name         rd > 8, r > 5, exog, s < 2, d+D > 2,
                                        p >= s, all-zero order, p > 8,
                                        NaN / inf in y, a non-positive
                                        innovation variance (DEVIATION 673)
    check_kalman_launch_invariant       block width 32 / 64 / 128, 41 floats
                                        of poisoned padding, a batch of 3 in
                                        another order, run twice
    check_unit_root_guard_is_reached    `rd == 2 && p == 2` REWRITES T[1] to
                                        -0.99 on the `ar2_unit` order, and
                                        the guard's absence moves the bits
    check_guard_decisions_are_recorded  the two RARE rewrites -- the
                                        `rd == 2 && p == 2` unit-root guard
                                        and the `r == 1` intercept nudge --
                                        read back from the `guards` decision
                                        stage: one arm that must set each
                                        bit and one that must set neither
    check_lu_pivot_tie_is_reached       DEVIATION 674's pivot tie rule: an
                                        EXACT magnitude tie is planted in
                                        column 0 of `I - T (x) T` on the
                                        `ar2_tie` order and asserted both
                                        PRESENT and MAXIMAL, so sabotage (e)
                                        is a live arm and not a recorded
                                        reach failure
    check_fold_order_is_visible         a descending `_mm` fold and a
                                        Float64 fold both move the T*P
                                        contraction, swept over EVERY order
                                        so the argument does not rest on one
                                        fixture: the bitwise gates above
                                        have teeth

SABOTAGES PERFORMED (2026-08-23), each reverted; outputs in arima/README.md.
The two the brief names are (a) and (c):
    (a) `batched_kalman_loop_kernel`'s `_mm` k loop run DESCENDING
    (b) `_numerical_stability`'s symmetrization dropped
    (c) `jones_transform_kernel`'s `transform` inner `k` loop run DESCENDING
    (d) `jones_transform_kernel`'s contraction spelled `fma(sign*a, x, t)`
        instead of `fma(sign, round(a*x), t)` -- the ONE-ROUNDING spelling
        this file's audit removed; it must FAIL, which is the evidence the
        correction was real and not cosmetic
    (e) `lu_inverse`'s pivot search changed from `>` to `>=` (the tie rule)
"""

from std.math import abs
from std.os import getenv
from std.memory import bitcast
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)

from arima.mojo_only.fixtures import (
    ArimaFixture,
    OrderCase,
    PLANT_INTERCEPT_NUDGE,
    PLANT_NONE,
    PLANT_PIVOT_TIE,
    PLANT_UNIT_ROOT,
    download_u8,
    arima_fixture,
    arima_params_fixture,
    bits32,
    count_cells_differ,
    count_cells_differ_i32,
    download_f32,
    download_params,
    first_cell_differ,
    order_table,
    same_bits,
    sub_batch_params,
    sub_batch_series,
    upload_f32,
    upload_f32_padded,
    upload_params,
)
from arima.ported.linalg.batched.matrix import kron_minus_identity_host
from arima.mojo_only.kalman_oracle import (
    KalmanHostStages,
    copy_forecast_host,
    in_sample_prediction_host,
    kalman_host_f32,
    kalman_host_f64,
)
from arima.ported.arima.batched_arima import (
    CANONICAL_NAN_BITS,
    batched_loglike,
    batched_loglike_grad,
    predict,
)
from arima.ported.arima.batched_kalman import batched_kalman_filter
from arima.ported.timeSeries.arima_helpers import (
    batched_jones_transform_host,
    finalize_forecast_host,
)
from arima.ported.timeSeries.jones_transform import (
    JONES_MAX_PARAMS,
    jones_transform,
    jones_transform_host,
)
from arima.ported.tsa.arima_common import (
    ARIMAOrder,
    ARIMAParams,
    ARIMAParamsHost,
    pack_host,
    unpack_host,
)
from tsa.ported.timeSeries.arima_helpers import prepare_data_host


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N_OBS = 24
comptime SALT = 7


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _survey() -> Bool:
    """SURVEY MODE, for sabotage runs only (`MOJOLEARN_ARIMA_SURVEY=1`).

    A gate that raises on its FIRST differing stage answers "did this
    sabotage bite" and nothing else. The question a sabotage table has to
    answer is WHICH stages moved and HOW MANY cells in each, because an arm
    that moves an intermediate stage but leaves the final output alone is
    the arm a naive output-only gate would call inert. Under survey mode
    `_gate` prints `SABOTAGE-MOVED` and CONTINUES, so one run enumerates
    every stage the sabotage reached.

    Never set this for a real gate run: it turns every assertion into a
    print. The green runs recorded in `arima/README.md` are made with it
    unset."""
    return getenv("MOJOLEARN_ARIMA_SURVEY") == "1"


def _gate(ok: Bool, what: String) raises:
    """Vendor-shaped claims (a bitwise device-vs-oracle line rides on the
    stdlib transcendental under FAST): asserted under IDENTICAL, RECORDED
    under FAST. Structural, refusal and integer claims use `_assert`."""
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


def _stage_list(name: String, got: List[Float32], want: List[Float32]) raises:
    var nd = count_cells_differ(got, want)
    print("      " + name + ": " + String(len(got)) + " cells, " + String(nd) + " differ"
          + ("" if nd == 0 else " (first " + first_cell_differ(got, want) + ")"))
    _gate(nd == 0, name + " device != oracle")


# ---------------------------------------------------------------------------
# the Jones transform
# ---------------------------------------------------------------------------


def check_jones_device_equals_oracle(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    print("check_jones_device_equals_oracle [" + _mode_name() + "]")
    var batch = 6
    for p in range(1, 5):
        for is_ar_i in range(2):
            var is_ar = is_ar_i == 1
            var x = List[Float32]()
            for b in range(batch):
                for i in range(p):
                    # hashed, in the untransformed range the optimizer works
                    # in; series 0 slot 0 is a planted -0.0
                    var v = Float32(0.31 * Float64(i + 1) - 0.17 * Float64(b + 1))
                    x.append(Float32(-0.0) if (b == 0 and i == 0) else v)
            var buf = upload_f32(ctx, x)
            var out = ctx.enqueue_create_buffer[DType.float32](batch * p)
            jones_transform(ctx, buf, batch, p, out, is_ar, False)
            var got = download_f32(ctx, out, batch * p)
            var want = jones_transform_host(x, batch, p, is_ar, False)
            var tag = "arima.jones.fwd.p" + String(p) + (".ar" if is_ar else ".ma")
            trace.record_device[DType.float32](ctx, tag, out, batch * p)
            _stage_list(tag, got, want)
            # the inverse, on the transformed values
            var buf2 = upload_f32(ctx, got)
            var out2 = ctx.enqueue_create_buffer[DType.float32](batch * p)
            jones_transform(ctx, buf2, batch, p, out2, is_ar, True)
            var back = download_f32(ctx, out2, batch * p)
            var want2 = jones_transform_host(got, batch, p, is_ar, True)
            var tag2 = "arima.jones.inv.p" + String(p) + (".ar" if is_ar else ".ma")
            trace.record_device[DType.float32](ctx, tag2, out2, batch * p)
            _stage_list(tag2, back, want2)
            # round trip, REPORTED (the identity is not the libm algorithm,
            # DEVIATION 675: this is a sanity number, not a bitwise claim)
            var worst = Float64(0.0)
            for i in range(batch * p):
                if x[i] == Float32(0.0):
                    continue
                var rel = abs(Float64(back[i]) - Float64(x[i])) / abs(Float64(x[i]))
                if rel > worst:
                    worst = rel
            print("      round trip p=" + String(p) + (" ar" if is_ar else " ma")
                  + ": worst relative " + String(worst) + " (REPORTED, DEVIATION 675)")
            _assert(worst < 1e-2, "the Jones round trip lost more than 1% relative")
            _ = buf^
            _ = buf2^
            _ = out^
            _ = out2^


def check_jones_refuses_by_name(ctx: DeviceContext) raises:
    print("check_jones_refuses_by_name [" + _mode_name() + "]")
    var x = List[Float32]()
    for _ in range(4):
        x.append(Float32(0.2))
    var buf = upload_f32(ctx, x)
    var out = ctx.enqueue_create_buffer[DType.float32](4)
    var bad = [(0, 1, String("batchSize < 1")), (1, 0, String("parameter < 1")), (1, 9, String("parameter > 8"))]
    for c in bad:
        var raised = False
        var msg = String("")
        try:
            jones_transform(ctx, buf, c[0], c[1], out, True, False)
        except e:
            raised = True
            msg = String(e)
        print("    " + c[2] + ": " + ("raised: " + msg if raised else "DID NOT RAISE"))
        _assert(raised, c[2] + " was not refused")
    _assert(JONES_MAX_PARAMS == 8, "their local arrays are DataT[8]")
    _ = buf^
    _ = out^


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------


def _run_pair(
    ctx: DeviceContext,
    f: ArimaFixture,
    order: ARIMAOrder,
    ph: ARIMAParamsHost,
    fc_steps: Int,
    kalman_tpb: Int = 32,
) raises -> Tuple[List[List[Float32]], KalmanHostStages]:
    """The device filter on TRANSFORMED parameters beside its oracle.

    `trans = True` on the device (so the Jones transform and the sigma2
    floor are on the recorded path); the oracle is fed the same transform's
    HOST replay, which is how their `Tparams` reaches `batched_kalman_
    filter` too."""
    var y = upload_f32(ctx, f.y)
    var params = upload_params(ctx, ph, order, f.batch_size)
    var res = batched_loglike(ctx, y, f.batch_size, f.n_obs, order, params, True, fc_steps, kalman_tpb)
    var tp = batched_jones_transform_host(order, f.batch_size, False, ph)
    var host = kalman_host_f32(f.y, f.batch_size, f.n_obs, order, tp, fc_steps)
    var rd = order.rd()
    var rd2 = rd * rd
    var b = f.batch_size
    var got = List[List[Float32]]()
    got.append(download_f32(ctx, res.ws.Z, rd * b))
    got.append(download_f32(ctx, res.ws.R, rd * b))
    got.append(download_f32(ctx, res.ws.T, rd2 * b))
    got.append(download_f32(ctx, res.ws.RQ, rd * b))
    got.append(download_f32(ctx, res.ws.RQR, rd2 * b))
    got.append(download_f32(ctx, res.ws.P0, rd2 * b))
    got.append(download_f32(ctx, res.ws.alpha0, rd * b))
    got.append(download_f32(ctx, res.ws.pred, f.n_obs * b))
    got.append(download_f32(ctx, res.ws.vs, f.n_obs * b))
    got.append(download_f32(ctx, res.ws.loglike, b))
    if fc_steps > 0:
        got.append(download_f32(ctx, res.ws.fc, fc_steps * b))
    else:
        got.append(List[Float32]())
    _ = res^
    _ = params^
    _ = y^
    return (got^, host^)


def _compare_stages(
    name: String, got: List[List[Float32]], host: KalmanHostStages
) raises:
    var tags = ["Z", "R", "T", "RQ", "RQR", "P0", "alpha0", "pred", "vs", "loglike", "fc"]
    var wants = List[List[Float32]]()
    wants.append(host.Z.copy())
    wants.append(host.R.copy())
    wants.append(host.T.copy())
    wants.append(host.RQ.copy())
    wants.append(host.RQR.copy())
    wants.append(host.P0.copy())
    wants.append(host.alpha0.copy())
    wants.append(host.pred.copy())
    wants.append(host.vs.copy())
    wants.append(host.loglike.copy())
    wants.append(host.fc.copy())
    for i in range(len(tags)):
        if len(got[i]) == 0:
            continue
        _stage_list(name + "." + tags[i], got[i], wants[i])


def check_kalman_device_equals_oracle(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    print("check_kalman_device_equals_oracle [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var table = order_table()
    for oc in table:
        var order = oc.order
        var ph = arima_params_fixture(order, f.batch_size, SALT, oc.plant)
        print("    " + oc.name + " (p" + String(order.p) + " d" + String(order.d)
              + " q" + String(order.q) + " P" + String(order.P) + " D" + String(order.D)
              + " Q" + String(order.Q) + " s" + String(order.s) + " k" + String(order.k)
              + ") rd=" + String(order.rd()) + " r=" + String(order.r()))
        var pair = _run_pair(ctx, f, order, ph, 3)
        _record_card(trace, "arima." + oc.name, pair[0])
        _compare_stages(oc.name, pair[0], pair[1])
        var ll = pair[0][9].copy()
        for b in range(f.batch_size):
            # v != v is the NaN test, spelled without an import
            _assert(ll[b] == ll[b], "log-likelihood is NaN for series " + String(b)
                    + " on order " + oc.name + " (ADDENDUM 11: no computed NaN in a stage)")
        print("      loglike[0] = " + String(ll[0]) + " " + bits32(ll[0]))


def _record_card(mut trace: IdentityTrace, prefix: String, got: List[List[Float32]]) raises:
    """The card's stages by name, hashed from the DOWNLOADED device bytes.

    Recording the host copy rather than the buffer is deliberate: the same
    list is what the oracle comparison sees, so a card that matches across
    vendors and a gate that passes on one cannot disagree about which bytes
    they meant."""
    var tags = ["Z", "R", "T", "RQ", "RQR", "P0", "alpha0", "pred", "vs", "loglike", "fc"]
    for i in range(len(tags)):
        if len(got[i]) == 0:
            continue
        trace.record_list_f32(prefix + "." + tags[i], got[i])


def check_lyapunov_solves_the_equation(ctx: DeviceContext) raises:
    """DEVIATION 674: our own LU replaces cuBLAS getrf/getri, so `P0` is
    checked against the EQUATION it is supposed to solve, not only against
    the oracle that shares its spelling. `Ts P Ts' - P + RQRs = 0` on the
    stationary block, residual in Float64, relative to `|RQRs|`."""
    print("check_lyapunov_solves_the_equation [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var table = order_table()
    for oc in table:
        var order = oc.order
        var ph = arima_params_fixture(order, f.batch_size, SALT, oc.plant)
        var pair = _run_pair(ctx, f, order, ph, 0)
        var T = pair[0][2].copy()
        var RQR = pair[0][4].copy()
        var P0 = pair[0][5].copy()
        var rd = order.rd()
        var r = order.r()
        var nd = order.n_diff()
        var worst = Float64(0.0)
        for b in range(f.batch_size):
            var mb = b * rd * rd
            var scale = Float64(0.0)
            for i in range(r):
                for j in range(r):
                    var q = abs(Float64(RQR[mb + (i + nd) + (j + nd) * rd]))
                    if q > scale:
                        scale = q
            if scale == 0.0:
                scale = 1.0
            for i in range(r):
                for j in range(r):
                    # (T P T')_ij on the r x r block at offset nd
                    var acc = Float64(0.0)
                    for k in range(r):
                        var tik = Float64(T[mb + (i + nd) + (k + nd) * rd])
                        for l in range(r):
                            var pkl = Float64(P0[mb + (k + nd) + (l + nd) * rd])
                            var tjl = Float64(T[mb + (j + nd) + (l + nd) * rd])
                            acc += tik * pkl * tjl
                    var resid = acc - Float64(P0[mb + (i + nd) + (j + nd) * rd]) \
                        + Float64(RQR[mb + (i + nd) + (j + nd) * rd])
                    var rel = abs(resid) / scale
                    if rel > worst:
                        worst = rel
        print("    " + oc.name + ": worst relative Lyapunov residual " + String(worst))
        _assert(worst < 5e-3,
                "DEVIATION 674: the direct Lyapunov solve does not satisfy its own equation on "
                + oc.name + " (worst relative residual " + String(worst) + ")")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def check_predict_device_equals_oracle(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    """`predict` had NO oracle before 2026-08-23 (the audit's finding): the
    in-sample undifferencing, the sentinel and the forecast copy are all
    device-only code. This replays them on the host and compares bitwise."""
    print("check_predict_device_equals_oracle [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var table = order_table()
    for oc in table:
        var order = oc.order
        if order.rd() > 8:
            continue
        var ph_raw = arima_params_fixture(order, f.batch_size, SALT, oc.plant)
        # `predict` is handed FITTED parameters and calls `batched_loglike`
        # with `trans = false` (`batched_arima.cu:175`), so the vector it
        # receives must ALREADY be transformed. Transform on the host once
        # and use that same vector for the device and the oracle.
        #
        # The first run of this gate uploaded the RAW vector, and on
        # `ar2_unit` -- whose fixture carries the untransformed `ar[1] =
        # -20.0` that the Jones clamp is supposed to fold to -0.9999 --
        # the filter ran with `phi_2 = -20`, which is a violently explosive
        # AR(2). DEVIATION 673 caught it and raised: "innovation variance
        # F <= 0 at step 0". That refusal firing on a fixture mistake, in
        # the one place an unported NaN would otherwise have propagated
        # silently, is DEVIATION 673 earning itself.
        var ph = batched_jones_transform_host(order.without_diff(), f.batch_size, False, ph_raw)
        var start = 0
        var end = f.n_obs + 3
        var y = upload_f32(ctx, f.y)
        var params = upload_params(ctx, ph, order, f.batch_size)
        var pr = predict(ctx, y, f.batch_size, f.n_obs, start, end, order, params, True)
        var ld = pr.predict_ld
        var got = download_f32(ctx, pr.y_p, ld * f.batch_size)
        trace.record_list_f32("arima." + oc.name + ".predict", got)

        # -------- the host replay of predict's own body --------
        var diff = order.need_diff()
        var n_obs_kf = f.n_obs - order.n_diff() if diff else f.n_obs
        var order_kf = order.without_diff() if diff else order
        var y_kf = prepare_data_host(f.y, f.batch_size, f.n_obs, order.d, order.D, order.s) if diff else f.y.copy()
        # NO Jones transform here. `predict` calls `batched_loglike` with
        # `trans = false` (`batched_arima.cu:175`) because the parameters it
        # is handed are the FITTED ones, already transformed by the caller.
        # The oracle has to be fed the same RAW vector. The first run of this
        # gate applied `batched_jones_transform_host` at this line and every
        # one of the 162 predict cells differed, which is what an oracle
        # running a different model from the device looks like: not a near
        # miss, a total one. That is the gate working, on the gate.
        var num_steps = end - f.n_obs
        var host = kalman_host_f32(y_kf, f.batch_size, n_obs_kf, order_kf, ph, num_steps)
        var want = List[Float32]()
        for _ in range(ld * f.batch_size):
            want.append(Float32(0.0))
        var res_offset = order.d + order.s * order.D if diff else 0
        var p_start = start if start > res_offset else res_offset
        var p_end = f.n_obs if f.n_obs < end else end
        var dD = order.d + order.D if diff else 0
        var period1 = 1 if order.d != 0 else order.s
        var period2 = 1 if order.d == 2 else order.s
        if start < f.n_obs:
            in_sample_prediction_host(
                f.y, host.pred, f.batch_size, f.n_obs, n_obs_kf, start, ld,
                res_offset, p_start, p_end, dD, period1, period2, want,
            )
        var fc = host.fc.copy()
        if diff:
            finalize_forecast_host(fc, f.y, num_steps, f.batch_size, f.n_obs, f.n_obs,
                                   order.d, order.D, order.s)
        copy_forecast_host(fc, f.batch_size, num_steps, ld, f.n_obs - start, want)
        _stage_list(oc.name + ".predict", got, want)

        # -------- DEVIATION 676: the sentinel is the CONSTANT --------
        var n_nan = 0
        for b in range(f.batch_size):
            for i in range(res_offset - start):
                var u = bitcast[DType.uint32](got[b * ld + i])
                _assert(u == CANONICAL_NAN_BITS,
                        "DEVIATION 676: the undefined prediction at series " + String(b)
                        + " step " + String(i) + " is " + bits32(got[b * ld + i])
                        + ", not the canonical 0x7fc00000")
                n_nan += 1
        print("      sentinel cells checked: " + String(n_nan) + " (res_offset - start = "
              + String(res_offset - start) + ")")
        _ = pr^
        _ = params^
        _ = y^
    _assert(True, "")


def check_predict_sentinel_is_reached() raises:
    """`res_offset - start` is 0 on most orders, so the sentinel branch is
    RARE and a gate that never enters it proves nothing (`sabotage-when-
    required`: a rare branch must be reached deliberately). This asserts the
    table CONTAINS an order with `res_offset > 0`."""
    print("check_predict_sentinel_is_reached [" + _mode_name() + "]")
    var table = order_table()
    var with_sentinel = 0
    for oc in table:
        if oc.order.need_diff():
            with_sentinel += 1
            print("    " + oc.name + ": res_offset = " + String(oc.order.n_diff())
                  + " sentinel cells per series at start = 0")
    _assert(with_sentinel >= 2,
            "fewer than two orders reach the NaN sentinel: DEVIATION 676 would be unchecked")


# ---------------------------------------------------------------------------
# the gradient
# ---------------------------------------------------------------------------


def check_grad_device_equals_oracle(ctx: DeviceContext, mut trace: IdentityTrace) raises:
    print("check_grad_device_equals_oracle [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var order = ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 1, 0)
    var ph = arima_params_fixture(order, f.batch_size, SALT)
    var N = order.complexity()
    var xh = pack_host(ph, order, f.batch_size)
    var h = Float32(1e-3)
    var y = upload_f32(ctx, f.y)
    var xb = upload_f32(ctx, xh)
    var gb = ctx.enqueue_create_buffer[DType.float32](N * f.batch_size)
    var scratch = ARIMAParams(ctx, order, f.batch_size)
    var xpert = ctx.enqueue_create_buffer[DType.float32](N * f.batch_size)
    var base_ll = batched_loglike_grad(ctx, y, f.batch_size, f.n_obs, order, xb, gb, h, True, scratch, xpert)
    var grad = download_f32(ctx, gb, N * f.batch_size)
    trace.record_list_f32("arima.grad", grad)

    # host replay: the same forward differences over the host filter
    var base_host = _loglike_host(f, order, unpack_host(xh, order, f.batch_size))
    var want = List[Float32]()
    for _ in range(N * f.batch_size):
        want.append(Float32(0.0))
    for i in range(N):
        var xp = xh.copy()
        for b in range(f.batch_size):
            xp[N * b + i] = ftz(ftz(xh[N * b + i]) + h)
        var pert = _loglike_host(f, order, unpack_host(xp, order, f.batch_size))
        for b in range(f.batch_size):
            var d = ftz(ftz(pert[b]) - ftz(base_host[b]))
            want[N * b + i] = ftz(d / h)
    _stage_list("grad", grad, want)
    for b in range(f.batch_size):
        _gate(same_bits(base_ll[b], base_host[b]),
              "the base log-likelihood differs from the host replay at series " + String(b))
    print("      N = " + String(N) + " parameters x " + String(f.batch_size) + " series")
    _ = gb^
    _ = xb^
    _ = y^
    _ = scratch^


def _loglike_host(f: ArimaFixture, order: ARIMAOrder, ph: ARIMAParamsHost) raises -> List[Float32]:
    var tp = batched_jones_transform_host(order, f.batch_size, False, ph)
    var host = kalman_host_f32(f.y, f.batch_size, f.n_obs, order, tp, 0)
    return host.loglike.copy()


def check_grad_reset_preserves_negative_zero(ctx: DeviceContext) raises:
    """THE AUDIT DEFECT, gated. Their reset (`batched_arima.cu:583-586`) is
    `x_pert[i] = x[i]`; ours reused the perturbation kernel with `h = 0`,
    and `-0.0 + 0.0` is `+0.0`. This plants `-0.0` in parameter 0 of every
    series, runs the gradient, and asserts the gradient of the LAST
    parameter -- the one evaluated after every reset -- equals the gradient
    computed from a vector whose parameter 0 is still `-0.0`.

    With the old `h = 0` reset, parameter 0 came back `+0.0` after its own
    iteration and every later column was evaluated one bit away."""
    print("check_grad_reset_preserves_negative_zero [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var order = ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 1, 0)
    var ph = arima_params_fixture(order, f.batch_size, SALT)
    var N = order.complexity()
    var xh = pack_host(ph, order, f.batch_size)
    for b in range(f.batch_size):
        xh[N * b] = Float32(-0.0)
    var neg = 0
    for b in range(f.batch_size):
        if bitcast[DType.uint32](xh[N * b]) == UInt32(0x80000000):
            neg += 1
    _assert(neg == f.batch_size, "the -0.0 plant did not take")
    var h = Float32(1e-3)
    var y = upload_f32(ctx, f.y)
    var xb = upload_f32(ctx, xh)
    var gb = ctx.enqueue_create_buffer[DType.float32](N * f.batch_size)
    var scratch = ARIMAParams(ctx, order, f.batch_size)
    var xpert = ctx.enqueue_create_buffer[DType.float32](N * f.batch_size)
    _ = batched_loglike_grad(ctx, y, f.batch_size, f.n_obs, order, xb, gb, h, True, scratch, xpert)
    var grad = download_f32(ctx, gb, N * f.batch_size)
    # the input vector must come back UNTOUCHED: `d_x` is never written
    var xback = download_f32(ctx, xb, N * f.batch_size)
    var nd = count_cells_differ(xh, xback)
    print("      d_x after the gradient: " + String(nd) + " of " + String(N * f.batch_size)
          + " cells differ (must be 0)")
    _assert(nd == 0, "batched_loglike_grad wrote to its input parameter vector")
    # THE DIRECT ASSERTION, and the only one with teeth. After the last
    # reset, x_pert must be d_x BITWISE, negative zeros included. The
    # indirect test below (the gradient of the last parameter) was written
    # first and is INERT: sabotage (g) restored the broken `x + 0.0` reset
    # and moved nothing, because a `-0.0` in `mu` is washed out by the very
    # first `0.0 + x` in the filter and never reaches the log-likelihood.
    # An inert gate is worse than no gate, so this reads the buffer itself.
    var xp_back = download_f32(ctx, xpert, N * f.batch_size)
    var np_ = count_cells_differ(xh, xp_back)
    print("      x_pert after the last reset: " + String(np_) + " of "
          + String(N * f.batch_size) + " cells differ from d_x (must be 0)")
    for b in range(f.batch_size):
        if not same_bits(xp_back[N * b], xh[N * b]):
            print("        series " + String(b) + " parameter 0: x_pert " + bits32(xp_back[N * b])
                  + " vs d_x " + bits32(xh[N * b]))
    _assert(np_ == 0,
            "the reset did not restore x_pert to d_x bitwise: a later column was evaluated "
            + "on a different vector (this is the `x + 0.0` is not a copy defect)")
    # host replay with the -0.0 KEPT throughout
    var base_host = _loglike_host(f, order, unpack_host(xh, order, f.batch_size))
    var last = N - 1
    var xp = xh.copy()
    for b in range(f.batch_size):
        xp[N * b + last] = ftz(ftz(xh[N * b + last]) + h)
    var pert = _loglike_host(f, order, unpack_host(xp, order, f.batch_size))
    var moved = 0
    for b in range(f.batch_size):
        var d = ftz(ftz(pert[b]) - ftz(base_host[b]))
        var wanted = ftz(d / h)
        if not same_bits(grad[N * b + last], wanted):
            moved += 1
    print("      gradient of the LAST parameter (after N-1 resets): "
          + String(moved) + " of " + String(f.batch_size) + " series differ")
    _gate(moved == 0,
          "the reset did not preserve -0.0: a later column was evaluated on a different vector")
    _ = gb^
    _ = xb^
    _ = y^
    _ = scratch^


# ---------------------------------------------------------------------------
# tolerance, refusals, invariance
# ---------------------------------------------------------------------------


def check_kalman_matches_float64(ctx: DeviceContext) raises:
    print("check_kalman_matches_float64 [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var table = order_table()
    for oc in table:
        var order = oc.order
        var ph = arima_params_fixture(order, f.batch_size, SALT, oc.plant)
        var pair = _run_pair(ctx, f, order, ph, 0)
        var ll32 = pair[0][9].copy()
        var tp = batched_jones_transform_host(order, f.batch_size, False, ph)
        var r64 = kalman_host_f64(f.y, f.batch_size, f.n_obs, order, tp, 0)
        var worst = Float64(0.0)
        for b in range(f.batch_size):
            var a = Float64(ll32[b])
            var e = r64.loglike[b]
            var rel = abs(a - e) / (abs(e) + 1e-30)
            if rel > worst:
                worst = rel
        # the diffuse kappa = 1e6 costs Float32 about ulp(1e6) = 0.0625 of
        # absolute error in the stationary block after the first step, so
        # the differenced orders are LOOSER by construction (DEVIATION 670)
        var bound = 2e-2 if order.n_diff() > 0 else 5e-3
        print("    " + oc.name + ": worst relative log-likelihood gap " + String(worst)
              + " (bound " + String(bound) + ", n_diff = " + String(order.n_diff()) + ")")
        _assert(worst <= bound,
                "DEVIATION 670: Float32 is further from Float64 than the recorded bound on " + oc.name)


def _expect_raise(what: String, raised: Bool, msg: String) raises:
    print("    " + what + ": " + ("raised: " + msg if raised else "DID NOT RAISE"))
    _assert(raised, what + " was not refused")


def check_arima_refuses_by_name(ctx: DeviceContext) raises:
    print("check_arima_refuses_by_name [" + _mode_name() + "]")
    var f = arima_fixture(32, SALT)
    var cases = [
        (ARIMAOrder(1, 0, 1, 1, 0, 1, 1, 0, 0), String("s < 2 with seasonal terms")),
        (ARIMAOrder(1, 2, 1, 0, 1, 0, 4, 0, 0), String("d + D > 2")),
        (ARIMAOrder(4, 0, 1, 1, 0, 1, 4, 0, 0), String("p >= s")),
        (ARIMAOrder(0, 1, 0, 0, 0, 0, 0, 0, 0), String("p+q+P+Q+k == 0")),
        (ARIMAOrder(9, 0, 0, 0, 0, 0, 0, 0, 0), String("p > 8")),
        (ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 0, 2), String("n_exog != 0")),
        (ARIMAOrder(1, 0, 8, 0, 0, 0, 0, 0, 0), String("rd = 9 > 8 (block-per-series kernel)")),
        (ARIMAOrder(6, 0, 0, 0, 0, 0, 0, 0, 0), String("r = 6 > 5 (Schur Lyapunov)")),
    ]
    for c in cases:
        var bad_order = c[0]
        var raised = False
        var msg = String("")
        try:
            var ph = arima_params_fixture(bad_order, f.batch_size, SALT)
            var y = upload_f32(ctx, f.y)
            var params = upload_params(ctx, ph, bad_order, f.batch_size)
            var res = batched_loglike(ctx, y, f.batch_size, f.n_obs, bad_order, params, True, 0)
            _ = res^
            _ = params^
            _ = y^
        except e:
            raised = True
            msg = String(e)
        _expect_raise(c[1], raised, msg)

    # non-finite input
    var order = ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 0, 0)
    for k in range(2):
        var yy = f.y.copy()
        yy[5 + k] = Float32(0.0) / Float32(0.0) if k == 0 else Float32(1.0) / Float32(0.0)
        var raised = False
        var msg = String("")
        try:
            var ph = arima_params_fixture(order, f.batch_size, SALT)
            var y = upload_f32(ctx, yy)
            var params = upload_params(ctx, ph, order, f.batch_size)
            var res = batched_loglike(ctx, y, f.batch_size, f.n_obs, order, params, True, 0)
            _ = res^
            _ = params^
            _ = y^
        except e:
            raised = True
            msg = String(e)
        _expect_raise("NaN in y" if k == 0 else "inf in y", raised, msg)

    # DEVIATION 673: a non-positive innovation variance, reached with
    # sigma2 = 0 and trans = False (the Jones floor would otherwise lift it)
    var raised2 = False
    var msg2 = String("")
    try:
        var ph = arima_params_fixture(order, f.batch_size, SALT)
        for b in range(f.batch_size):
            ph.sigma2[b] = Float32(0.0)
            ph.ar[b] = Float32(0.0)
            ph.ma[b] = Float32(0.0)
        var y = upload_f32(ctx, f.y)
        var params = upload_params(ctx, ph, order, f.batch_size)
        var res = batched_loglike(ctx, y, f.batch_size, f.n_obs, order, params, False, 0)
        _ = res^
        _ = params^
        _ = y^
    except e:
        raised2 = True
        msg2 = String(e)
    _expect_raise("DEVIATION 673: F <= 0 with sigma2 = 0, trans = False", raised2, msg2)
    # the message is PRINTED above rather than pattern-matched: a substring
    # test on an error string is a gate that passes when the wording drifts.
    # OWED: the compile slot reads the printed line and confirms it names
    # both the series and the step, as DEVIATION 673 says it does.


def check_kalman_launch_invariant(ctx: DeviceContext) raises:
    print("check_kalman_launch_invariant [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var order = ARIMAOrder(2, 1, 2, 0, 0, 0, 0, 1, 0)
    var ph = arima_params_fixture(order, f.batch_size, SALT)
    var rd = order.rd()
    var b = f.batch_size

    var y0 = upload_f32(ctx, f.y)
    var p0 = upload_params(ctx, ph, order, b)
    var r0 = batched_loglike(ctx, y0, b, f.n_obs, order, p0, True, 3, 32)
    var ll0 = download_f32(ctx, r0.ws.loglike, b)
    var P0_0 = download_f32(ctx, r0.ws.P0, rd * rd * b)
    var vs0 = download_f32(ctx, r0.ws.vs, f.n_obs * b)
    var fc0 = download_f32(ctx, r0.ws.fc, 3 * b)

    for tpb in [64, 128]:
        var y1 = upload_f32_padded(ctx, f.y, 41, Float32(12345.678))
        var p1 = upload_params(ctx, ph, order, b)
        var r1 = batched_loglike(ctx, y1, b, f.n_obs, order, p1, True, 3, tpb)
        var nd = count_cells_differ(ll0, download_f32(ctx, r1.ws.loglike, b))
        nd += count_cells_differ(P0_0, download_f32(ctx, r1.ws.P0, rd * rd * b))
        nd += count_cells_differ(vs0, download_f32(ctx, r1.ws.vs, f.n_obs * b))
        nd += count_cells_differ(fc0, download_f32(ctx, r1.ws.fc, 3 * b))
        print("    block " + String(tpb) + " / 41 floats of poison: " + String(nd) + " cells differ")
        _assert(nd == 0, "the loop kernel's block width or the padding moved the bytes")
        _ = r1^
        _ = p1^
        _ = y1^

    # a different poison
    var y2 = upload_f32_padded(ctx, f.y, 41, Float32(-0.0))
    var p2 = upload_params(ctx, ph, order, b)
    var r2 = batched_loglike(ctx, y2, b, f.n_obs, order, p2, True, 3, 32)
    var nd2 = count_cells_differ(ll0, download_f32(ctx, r2.ws.loglike, b))
    print("    poison -0.0: " + String(nd2) + " cells differ")
    _assert(nd2 == 0, "the poison moved the bytes")

    # a batch of three in another order
    var which: List[Int] = [4, 0, 2]
    var ys = sub_batch_series(f.y, f.n_obs, which)
    var phs = sub_batch_params(ph, order, which)
    var y3 = upload_f32(ctx, ys)
    var p3 = upload_params(ctx, phs, order, 3)
    var r3 = batched_loglike(ctx, y3, 3, f.n_obs, order, p3, True, 3, 32)
    var ll3 = download_f32(ctx, r3.ws.loglike, 3)
    var vs3 = download_f32(ctx, r3.ws.vs, f.n_obs * 3)
    var nb = 0
    for k in range(3):
        var src = which[k]
        if not same_bits(ll3[k], ll0[src]):
            nb += 1
        for t in range(f.n_obs):
            if not same_bits(vs3[k * f.n_obs + t], vs0[src * f.n_obs + t]):
                nb += 1
    print("    batch of 6 vs batch of 3 [4, 0, 2]: " + String(nb) + " cells differ")
    _assert(nb == 0, "batch composition moved the bytes")

    # run twice in one process
    var y4 = upload_f32(ctx, f.y)
    var p4 = upload_params(ctx, ph, order, b)
    var r4 = batched_loglike(ctx, y4, b, f.n_obs, order, p4, True, 3, 32)
    var nd4 = count_cells_differ(ll0, download_f32(ctx, r4.ws.loglike, b))
    print("    run twice: " + String(nd4) + " cells differ")
    _assert(nd4 == 0, "two runs differ")
    _ = r0^
    _ = r2^
    _ = r3^
    _ = r4^
    _ = p0^
    _ = p2^
    _ = p3^
    _ = p4^
    _ = y0^
    _ = y2^
    _ = y3^
    _ = y4^


def check_unit_root_guard_is_reached(ctx: DeviceContext) raises:
    """`if (rd == 2 && order.p == 2 && abs(batch_T[1] + 1) < 0.01)
    batch_T[1] = -0.99` (`batched_kalman.cu:1243-1244`) is a RARE branch: it
    only fires on an ARMA(2,0) whose transformed `phi_2` sits within 0.01 of
    -1. `sabotage-when-required` says a rare branch must be entered
    deliberately. This asserts `T[1]` IS -0.99 exactly on the `ar2_unit`
    order, and that the untransformed value is NOT -0.99 (so the guard, not
    the fixture, put it there)."""
    print("check_unit_root_guard_is_reached [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var order = ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0)
    _assert(order.rd() == 2 and order.p == 2, "the ar2_unit order does not select the guard")
    var ph = arima_params_fixture(order, f.batch_size, SALT, PLANT_UNIT_ROOT)
    var pair = _run_pair(ctx, f, order, ph, 0)
    var T = pair[0][2].copy()
    var fired = 0
    for b in range(f.batch_size):
        var t1 = T[b * 4 + 1]
        if same_bits(t1, Float32(-0.99)):
            fired += 1
        else:
            print("      series " + String(b) + ": T[1] = " + String(t1) + " " + bits32(t1))
    print("    the guard fired on " + String(fired) + " of " + String(f.batch_size) + " series")
    _assert(fired == f.batch_size,
            "the rd == 2 && p == 2 guard did NOT fire: the branch is unreached and unchecked")
    # and the transformed phi_2 before the guard is not -0.99 by accident
    # For p = 2, s = 0, `T[1]` IS the transformed `ar[1]`: reduced_polynomial
    # returns `-(-ar[1] * 1)` at idx = 2. So the value the guard overwrote is
    # readable directly, and it must be the CLAMP (-0.9999) and not -0.99,
    # otherwise the fixture and not the guard is what put -0.99 in T.
    var tp = batched_jones_transform_host(order, f.batch_size, False, ph)
    var phi2_raw = tp.ar[1]
    print("      transformed phi_2 (series 0) before the guard: " + String(phi2_raw)
          + " " + bits32(phi2_raw))
    _assert(not same_bits(phi2_raw, Float32(-0.99)),
            "the fixture already equals -0.99: the guard is not what wrote T[1]")
    _assert(same_bits(phi2_raw, Float32(-0.9999)),
            "the transformed phi_2 is not at the Jones clamp, so the guard was reached by accident")


def check_guard_decisions_are_recorded(ctx: DeviceContext) raises:
    """THE TWO RARE REWRITES, and the stage that finally records them.

    `guards` is one byte per series. Bit 0 says the `rd == 2 && p == 2`
    unit-root guard rewrote `T[1]` to -0.99; bit 1 says the `r == 1`
    intercept guard nudged `I - T*` off zero. Both CHANGE THE MODEL, and
    before 2026-08-24 neither appeared in any recorded stage on any card:
    bit 0 was only inferable from a suspicious -0.99 sitting in `T` if you
    already knew to look, and bit 1 was not inferable at all, because `ImT`
    is overwritten in place by its own LU factorization before anything
    could have read it.

    Three arms, because a decision stage that is always 0 or always 1 is not
    a stage, it is a constant:

      * `ar2_unit` must set bit 0 and NOT bit 1
      * a locally built `(1,0,0) k=1` with the intercept plant must set
        bit 1 and NOT bit 0
      * `arma11_k` must set NEITHER

    The second fixture is built HERE rather than added to `order_table`
    because it is deliberately near-singular (`1 - phi^2 = 2e-4`), which is
    what makes it reach the guard and also what would make it a dishonest
    row in the Float64 precision gate."""
    print("check_guard_decisions_are_recorded [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var arms = [
        (ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0), PLANT_UNIT_ROOT, UInt8(1), String("ar2_unit: unit-root rewrite")),
        (ARIMAOrder(1, 0, 0, 0, 0, 0, 0, 1, 0), PLANT_INTERCEPT_NUDGE, UInt8(2), String("ar1_k_nudge: r == 1 intercept nudge")),
        (ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 1, 0), PLANT_NONE, UInt8(0), String("arma11_k: neither guard")),
    ]
    for arm in arms:
        var order = arm[0]
        var ph = arima_params_fixture(order, f.batch_size, SALT, arm[1])
        var y = upload_f32(ctx, f.y)
        var params = upload_params(ctx, ph, order, f.batch_size)
        var res = batched_loglike(ctx, y, f.batch_size, f.n_obs, order, params, True, 0)
        var g = download_u8(ctx, res.ws.guards, f.batch_size)
        var want = arm[2]
        var bad = 0
        for b in range(f.batch_size):
            if g[b] != want:
                bad += 1
        print("      " + arm[3] + ": guards byte " + String(Int(g[0]))
              + " on series 0, expected " + String(Int(want)) + "; "
              + String(bad) + " of " + String(f.batch_size) + " series disagree")
        _assert(bad == 0,
                arm[3] + ": the guards decision stage does not read back as expected, so either "
                + "the guard did not fire where it must or the stage is not recording it")
        _ = res^
        _ = params^
        _ = y^


def check_lu_pivot_tie_is_reached(ctx: DeviceContext) raises:
    """DEVIATION 674's PIVOT TIE RULE, made gateable.

    Sabotage (e) flips `lu_inverse`'s `>` to `>=` and, before the `ar2_tie`
    fixture existed, moved nothing on any order: no column of
    `I - T (x) T` contained two entries of equal maximum magnitude, so the
    tie branch was UNREACHED and the rule -- which is OURS, cuBLAS's being
    unreadable -- was gated by nothing.

    This asserts the tie is really there, in the matrix the DEVICE built,
    before the arm is claimed to be live. It checks three things, because
    only all three together mean the pivot loop meets it:

      1. rows 1 and 2 of column 0 are equal BIT FOR BIT (they are the same
         product with the operands commuted, so IEEE-754 makes this exact);
      2. that shared magnitude is the strict maximum of the column, so the
         search actually selects it;
      3. the tie is at rows > 0, so the search reaches it at all (the loop
         starts at `best = j` and scans `j+1 ..`).

    A tie that exists but is not the maximum is `reached but inert` all over
    again, which is why (2) is asserted and not assumed.

    THIS GATE FAILS LOUDLY BY DESIGN. A fixture that silently misses its
    branch is WORSE THAN NO FIXTURE: it converts an uncovered branch into an
    apparently covered one, and the next reader sees a green arm and moves
    on. So none of the three properties is checked with a print and a shrug;
    each raises, naming the series and the numbers, and the arm below asserts
    the unit-root guard did NOT fire on this order so a passing
    `check_unit_root_guard_is_reached` can never be mistaken for this one.

    (Under `MOJOLEARN_ARIMA_SURVEY=1` these become prints, as every assertion
    does, because a sabotage run must enumerate rather than stop. Survey mode
    is never used for a green run and the baseline prints nothing.)"""
    print("check_lu_pivot_tie_is_reached [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var order = ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0)
    var ph = arima_params_fixture(order, f.batch_size, SALT, PLANT_PIVOT_TIE)
    var pair = _run_pair(ctx, f, order, ph, 0)
    var T = pair[0][2].copy()
    var rd = order.rd()
    var r = order.r()
    var r2 = r * r
    _assert(r == 2 and rd == 2, "ar2_tie must have r = rd = 2 for this derivation")
    var tied = 0
    for b in range(f.batch_size):
        var imaa = kron_minus_identity_host(T, b * rd * rd, rd, 0, r)
        # column 0 of the r2 x r2 matrix
        var mags = List[Float32]()
        for i in range(r2):
            mags.append(abs(imaa[i]))
        print("      series " + String(b) + ": phi1 " + bits32(T[b * 4])
              + " phi2 " + bits32(T[b * 4 + 1]) + "  column 0 magnitudes "
              + String(mags[0]) + " " + String(mags[1]) + " "
              + String(mags[2]) + " " + String(mags[3]))
        # (1) exact bitwise tie between rows 1 and 2
        _assert(same_bits(imaa[1], imaa[2]),
                "series " + String(b) + ": rows 1 and 2 of column 0 are not bit-identical, "
                + "so the commuted product did not tie")
        # (2) it is the strict maximum
        var mx = mags[1]
        var second = mags[0] if mags[0] > mags[3] else mags[3]
        _assert(mx > second,
                "series " + String(b) + ": the tied magnitude " + String(mx)
                + " is not the strict maximum of column 0 (nearest rival " + String(second)
                + "), so the pivot search never selects it and sabotage (e) stays inert")
        # (3) reached by the search, which starts at row 0 and scans upward
        tied += 1
    print("    an exact pivot-magnitude tie is present and maximal in column 0 on "
          + String(tied) + " of " + String(f.batch_size) + " series")
    _assert(tied == f.batch_size, "the planted pivot tie is not present on every series")
    # and the unit-root guard must NOT be what produced this T
    for b in range(f.batch_size):
        _assert(not same_bits(T[b * 4 + 1], Float32(-0.99)),
                "series " + String(b) + ": the rd == 2 && p == 2 guard fired on ar2_tie; "
                + "this arm would then be testing the guard, not the pivot rule")


def check_fold_order_is_visible(ctx: DeviceContext) raises:
    """Host only, and WIDENED 2026-08-24. The oracle's `T*P` contraction
    against a DESCENDING k fold and against a Float64 fold of the same
    products, over EVERY order in the table rather than one.

    Why it changed. This gate is the one whose entire job is to show the
    bitwise gates have teeth, and it used to run on `arma11_k` alone, where
    `rd = 2` gives 24 cells and a fold chain two terms long. It moved 2 of
    24 cells under IDENTICAL and 1 of 24 under FAST. That is a real signal
    but a thin one, and a gate that argues for every other gate should not
    rest on one or two cells of one fixture.

    REACH IS PER FIXTURE, which is the lesson other lanes have now paid for
    twice: the spectral lane found three of its arms each carried by a
    single fixture, and the mamba lane's fold arm went from 1 of 4 rows to 3
    of 4 purely by widening `d_model` from 8 to 16. The width that matters
    here is `rd`, because `rd` IS the length of the fold chain: at `rd = 2`
    a reversal permutes two terms, and two-term sums are very often
    associative by accident. The table now runs to `rd = 8` (`sarima_rd8`)
    and `r = 5` (`arma44`), so this gate sweeps chain lengths 1 through 8
    and cell counts from 6 to 384 instead of one length and 24 cells.

    THE PREDICTION WAS RECORDED IN ADVANCE AND WAS MOSTLY WRONG. It said:
    `ar1` (`rd = 1`) moves ZERO, `rd = 2` orders move very few, and
    `rd >= 4` orders move a substantial fraction. Measured 2026-08-24,
    IDENTICAL / FAST:

        order         rd  n_phi   cells   moved   moved%
        arma11_k       2      1      24     2/1     8.3
        ar1            1      1       6     0/0     0.0
        ma2            3      0      54     0/0     0.0
        arima111       3      1      54     2/1     3.7
        arima212       4      2      96     5/7     5.2
        ar2_unit       2      2      24     1/3     4.2
        sarima_full    7      3     294    13/18    4.4
        sarima_rd8     8      1     384     1/2     0.3
        arma44         5      4     150    25/24   16.7
        ar2_tie        2      2      24     6/7    25.0

    One of three clauses held. `ar1` moved zero, as predicted and now
    asserted. The other two were wrong, and wrong in the same way: CHAIN
    LENGTH IS NOT THE DRIVER. `sarima_rd8` has the LONGEST chain in the
    table (`rd = 8`) and moves 0.3%, the least of anything that moved at
    all; `ar2_tie` has one of the shortest (`rd = 2`) and moves 25%, the
    most.

    THE DRIVER IS `n_phi`, the count of NON-TRIVIAL entries in `T`. A
    differenced or seasonal `T` is mostly structural: exact `0.0` and exact
    `1.0` from the differencing rows, the seasonal shift and the companion
    superdiagonal. A fold over exact values is order-independent, so those
    cells cannot move however long the chain is. Only the `n_phi` hashed AR
    coefficients carry rounding, and the movement tracks them: `ma2` has
    `n_phi = 0` and moves NOTHING despite `rd = 3`; `arma44` has `n_phi = 4`
    and moves most.

    The lesson generalizes past this gate and is the reason the wrong
    prediction is left here rather than quietly replaced: widening a fixture
    means widening the part that CARRIES ARITHMETIC, not the part that
    carries structure. `rd` was the wrong knob and `n_phi` is the right one.

    The floors below are the observed numbers with headroom, and they hold
    in both modes (IDENTICAL totals 55, FAST 63)."""
    print("check_fold_order_is_visible [" + _mode_name() + "]")
    var f = arima_fixture(N_OBS, SALT)
    var table = order_table()
    var total_cells = 0
    var total_desc = 0
    var total_f64 = 0
    var orders_that_moved = 0
    var best_order_moved = 0
    for oc in table:
        var order = oc.order
        var ph = arima_params_fixture(order, f.batch_size, SALT, oc.plant)
        var tp = batched_jones_transform_host(order, f.batch_size, False, ph)
        var host = kalman_host_f32(f.y, f.batch_size, f.n_obs, order, tp, 0)
        var rd = order.rd()
        var moved_desc = 0
        var moved_f64 = 0
        var cells = 0
        for b in range(f.batch_size):
            var mb = b * rd * rd
            for i in range(rd):
                for j in range(rd):
                    var asc = Float32(0.0)
                    var desc = Float32(0.0)
                    var f64 = Float64(0.0)
                    for k in range(rd):
                        var t = ftz(host.T[mb + i + k * rd])
                        var pv = ftz(host.P0[mb + k + j * rd])
                        asc = ftz(identical_mul_add(t, pv, asc))
                        f64 += Float64(t) * Float64(pv)
                    var k2 = rd - 1
                    while k2 >= 0:
                        var t = ftz(host.T[mb + i + k2 * rd])
                        var pv = ftz(host.P0[mb + k2 + j * rd])
                        desc = ftz(identical_mul_add(t, pv, desc))
                        k2 -= 1
                    cells += 1
                    if not same_bits(asc, desc):
                        moved_desc += 1
                    if not same_bits(asc, Float32(f64)):
                        moved_f64 += 1
        # n_phi is the count of NON-TRIVIAL entries in T and is what actually
        # drives this; rd is only the chain length and most of a differenced
        # T is exact 0.0 / 1.0, which no reordering can move.
        print("      " + oc.name + " (rd=" + String(rd) + ", n_phi=" + String(order.n_phi())
              + "): " + String(cells) + " T*P cells, descending fold moves "
              + String(moved_desc) + ", Float64 fold moves " + String(moved_f64))
        total_cells += cells
        total_desc += moved_desc
        total_f64 += moved_f64
        if moved_desc > 0:
            orders_that_moved += 1
        if moved_desc > best_order_moved:
            best_order_moved = moved_desc
        if rd == 1:
            _assert(moved_desc == 0,
                    oc.name + " has rd = 1, a one-term fold with no order, yet reversing it "
                    + "moved " + String(moved_desc) + " cells: the fold helper is not what we think")
    print("    TOTAL over " + String(len(table)) + " orders: " + String(total_cells)
          + " cells, descending fold moves " + String(total_desc)
          + ", Float64 fold moves " + String(total_f64)
          + "; orders in which the descending fold moved something: "
          + String(orders_that_moved))
    _assert(total_desc > 0,
            "a descending fold moves no cell anywhere: the fixture cannot see a fold and every "
            + "bitwise gate in this file is empty")
    _assert(total_f64 > 0, "a Float64 fold moves no cell anywhere: the fixture cannot see a fold")
    print("      strongest single order moved " + String(best_order_moved) + " cells")
    # FLOORS, raised 2026-08-24 from "at least 2 orders" to the observed
    # numbers with headroom. Observed: 8 orders moved, 55 cells total
    # (IDENTICAL) / 63 (FAST), strongest order 25 / 24.
    _assert(orders_that_moved >= 6,
            "the descending fold moves cells in fewer than six orders (observed 8), so this gate "
            + "has lost reach it used to have (spectral lane, per-fixture reach)")
    _assert(total_desc >= 35,
            "the descending fold moves fewer than 35 cells in total (observed 55 IDENTICAL, "
            + "63 FAST): the fixture has stopped seeing the fold")
    _assert(best_order_moved >= 15,
            "no single order moves 15 cells (observed 25 IDENTICAL, 24 FAST on arma44): the "
            + "argument that the bitwise gates have teeth has lost its strongest witness. "
            + "n_phi is the knob -- an order with more non-trivial AR coefficients, not a "
            + "longer state vector")


def check_jones_contraction_is_visible() raises:
    """THE AUDIT CORRECTION, gated on the host. `tmp[k] += sign * (a * x)`
    contracts as `fma(sign, round(a*x), tmp)`, TWO roundings. The removed
    spelling was `fma(sign*a, x, tmp)`, ONE rounding. If those two agreed on
    this fixture the correction would be unobservable and the sabotage (d)
    could not fail, so the difference is asserted here rather than assumed."""
    print("check_jones_contraction_is_visible [" + _mode_name() + "]")
    var moved = 0
    var total = 0
    for pi in range(2, 5):
        for is_ar_i in range(2):
            var is_ar = is_ar_i == 1
            var sign = Float32(-1.0) if is_ar else Float32(1.0)
            for b in range(8):
                var mine = List[Float32]()
                for i in range(pi):
                    mine.append(Float32(0.31 * Float64(i + 1) - 0.17 * Float64(b + 1)))
                for j in range(1, pi):
                    var a = mine[j]
                    for k in range(j):
                        var x = mine[j - k - 1]
                        var acc = mine[k]
                        var two = ftz(identical_mul_add(sign, ftz(a * x), acc))
                        var one = ftz(identical_mul_add(ftz(sign * a), x, acc))
                        total += 1
                        if not same_bits(two, one):
                            moved += 1
    print("    of " + String(total) + " accumulations, the two spellings differ on " + String(moved))
    _assert(moved > 0,
            "the corrected contraction is bit-identical to the one it replaced on every cell: "
            + "sabotage (d) cannot fail and the correction is unverifiable here")


def main() raises:
    print("== arima/mojo_only/arima_check.mojo [" + _mode_name() + "] N_OBS=" + String(N_OBS)
          + " SALT=" + String(SALT) + " ==")
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header("arima_check mode=" + _mode_name() + " n_obs=" + String(N_OBS)
                 + " batch=6 salt=" + String(SALT))
    check_jones_device_equals_oracle(ctx, trace)
    check_jones_refuses_by_name(ctx)
    check_jones_contraction_is_visible()
    check_kalman_device_equals_oracle(ctx, trace)
    check_lyapunov_solves_the_equation(ctx)
    check_predict_sentinel_is_reached()
    check_predict_device_equals_oracle(ctx, trace)
    check_grad_device_equals_oracle(ctx, trace)
    check_grad_reset_preserves_negative_zero(ctx)
    check_kalman_matches_float64(ctx)
    check_arima_refuses_by_name(ctx)
    check_kalman_launch_invariant(ctx)
    check_unit_root_guard_is_reached(ctx)
    check_lu_pivot_tie_is_reached(ctx)
    check_guard_decisions_are_recorded(ctx)
    check_fold_order_is_visible(ctx)
    print("ALL ARIMA CHECKS PASSED [" + _mode_name() + "]"
          + (" card: " + trace.path if trace.enabled else " (no card: set MOJOLEARN_IDENTITY_TRACE)"))
