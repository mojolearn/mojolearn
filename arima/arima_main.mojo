"""The batched ARIMA Kalman filter on one hashed batch, with an identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . arima/arima_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . arima/arima_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/arima.card tools/with_identical_mode.sh \
        pixi run mojo run -I . arima/arima_main.mojo

    python3 tools/identity_trace_diff.py /tmp/arima.mac.card /tmp/arima.other.card

The card (`core/identity_trace.mojo`) carries the INPUT bytes first, then the
untransformed parameters, then the Jones-TRANSFORMED parameters (read those
second when two cards disagree: a different `t_params` is a different model
and everything downstream of it is noise), then every stage of the filter in
the order the device writes them --  `Z`, `R`, `T`, `RQ`, `RQR`, `P0`,
`alpha0`, `pred`, `vs`, `loglike`, `fc` -- for each of the seven orders in
`arima/mojo_only/fixtures.mojo`'s table. Tags are unique and carry no launch
parameter, so a card is a function of the fixture and the mode alone.

READ THE STAGES IN ORDER. `T` before `P0` before `alpha0` before `pred`: the
first stage that differs is the one to diagnose, and every later one inherits
it. `P0` is the stage most likely to move between vendors, because it is
DEVIATION 674's hand-written LU where theirs is a closed cuBLAS batched
`getrf`/`getri`.

Not a port: cuML ships one backend and needs no card.

STATUS: WRITTEN, NEVER RUN. No card has been produced from this file on any
vendor. See `arima/README.md`'s OWED list.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from arima.mojo_only.fixtures import (
    arima_fixture,
    arima_params_fixture,
    bits32,
    download_f32,
    order_table,
    upload_f32,
    upload_params,
)
from arima.ported.arima.batched_arima import batched_loglike


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N_OBS = 24
comptime SALT = 7
comptime FC_STEPS = 3


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def main() raises:
    print("== arima/arima_main.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "arima_main mode=" + _mode_name() + " n_obs=" + String(N_OBS)
        + " batch=6 salt=" + String(SALT) + " fc_steps=" + String(FC_STEPS)
    )

    var f = arima_fixture(N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    trace.record_device[DType.float32](ctx, "arima.input", y, f.batch_size * f.n_obs)

    var table = order_table()
    for oc in table:
        var order = oc.order
        var tag = "arima." + oc.name
        var rd = order.rd()
        var rd2 = rd * rd
        var b = f.batch_size

        var ph = arima_params_fixture(order, b, SALT, oc.plant)
        var params = upload_params(ctx, ph, order, b)
        if order.k != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.mu", params.mu, b)
        if order.p != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.ar", params.ar, order.p * b)
        if order.q != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.ma", params.ma, order.q * b)
        if order.P != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.sar", params.sar, order.P * b)
        if order.Q != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.sma", params.sma, order.Q * b)
        trace.record_device[DType.float32](ctx, tag + ".param.sigma2", params.sigma2, b)

        var res = batched_loglike(ctx, y, b, f.n_obs, order, params, True, FC_STEPS)

        # the TRANSFORMED parameters: read these second when cards disagree
        if order.p != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.ar", res.t_params.ar, order.p * b)
        if order.q != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.ma", res.t_params.ma, order.q * b)
        if order.P != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.sar", res.t_params.sar, order.P * b)
        if order.Q != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.sma", res.t_params.sma, order.Q * b)
        trace.record_device[DType.float32](ctx, tag + ".jones.sigma2", res.t_params.sigma2, b)

        # the filter, in the order the device writes them
        trace.record_device[DType.float32](ctx, tag + ".Z", res.ws.Z, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".R", res.ws.R, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".T", res.ws.T, rd2 * b)
        trace.record_device[DType.float32](ctx, tag + ".RQ", res.ws.RQ, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".RQR", res.ws.RQR, rd2 * b)
        trace.record_device[DType.float32](ctx, tag + ".P0", res.ws.P0, rd2 * b)
        trace.record_device[DType.float32](ctx, tag + ".alpha0", res.ws.alpha0, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".pred", res.ws.pred, f.n_obs * b)
        trace.record_device[DType.float32](ctx, tag + ".vs", res.ws.vs, f.n_obs * b)
        trace.record_device[DType.float32](ctx, tag + ".loglike", res.ws.loglike, b)
        trace.record_device[DType.float32](ctx, tag + ".fc", res.ws.fc, FC_STEPS * b)

        var ll = download_f32(ctx, res.ws.loglike, b)
        print("  " + oc.name + " rd=" + String(rd) + " r=" + String(order.r())
              + ": loglike[0] = " + String(ll[0]) + " " + bits32(ll[0]))
        _ = res^
        _ = params^

    print("arima_main done [" + _mode_name() + "]"
          + (" card: " + trace.path if trace.enabled else " (no card: set MOJOLEARN_IDENTITY_TRACE)"))
    _ = y^
