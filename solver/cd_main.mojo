# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The coordinate-descent driver: one Lasso fit on the planted fixture, its
IDENTITY CARD, and the oracle's verdict.

    pixi run mojo run -I . solver/cd_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/cd.apple.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo
    python3 tools/identity_trace_diff.py /tmp/cd.apple.card /tmp/cd.other.card

The card's stages, in emission order: `cd.input.x`, `cd.input.y`,
[`cd.mu_input`, `cd.mu_labels` under fit_intercept], `cd.l1_alpha`,
`cd.l2_alpha`, `cd.colnorm`, `cd.squared`, then per epoch
`cd.sweepNNN.coef`, `cd.sweepNNN.resid`, `cd.sweepNNN.conv` (the
`ConvState` triple `{-last r, coefMax, diffMax}`), then `cd.final.coef`,
`cd.intercept`, `cd.n_iter` (INTEGER). A cross-vendor diff that first moves
at `cd.n_iter` says the two machines disagreed about how long to run, which
is a bigger thing than a last bit; one that first moves at a `.resid` says
which epoch and which row.

Environment: `MOJOLEARN_CD_ROWS` (default 2048), `MOJOLEARN_CD_COLS` (16),
`MOJOLEARN_CD_ALPHA` (0.01), `MOJOLEARN_CD_L1_RATIO` (1.0 = Lasso),
`MOJOLEARN_CD_FIT_INTERCEPT` (1), `MOJOLEARN_CD_EPOCHS` (1000),
`MOJOLEARN_CD_TOL` (1e-3, cuML's default; scikit-learn's is 1e-4).

Every line carries the mode the binary COMPILED in. No timing is printed
and none is to be read from this file.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from solver.original.cd_oracle import cd_oracle_fit, fixture_planted_sparse
from solver.derived.solver.cd import CdLaunch, cd_fit_traced, cd_predict
from solver.derived.solvers.params import LOSS_SQRD_LOSS


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
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def _env_f32(name: String, default: Float32) raises -> Float32:
    var s = String(getenv(name))
    if s == "":
        return default
    return Float32(atof(s))


def main() raises:
    var mode = _mode_name()
    print("== solver/cd_main.mojo [" + mode + "] ==")
    var n = _env_int("MOJOLEARN_CD_ROWS", 2048)
    var d = _env_int("MOJOLEARN_CD_COLS", 16)
    var alpha = _env_f32("MOJOLEARN_CD_ALPHA", Float32(0.01))
    var l1_ratio = _env_f32("MOJOLEARN_CD_L1_RATIO", Float32(1.0))
    var fit_intercept = _env_int("MOJOLEARN_CD_FIT_INTERCEPT", 1) != 0
    var epochs = _env_int("MOJOLEARN_CD_EPOCHS", 1000)
    var tol = _env_f32("MOJOLEARN_CD_TOL", Float32(1.0e-3))
    print(
        "[" + mode + "] n_rows=" + String(n) + " n_cols=" + String(d)
        + " alpha=" + String(alpha) + " l1_ratio=" + String(l1_ratio)
        + " fit_intercept=" + String(fit_intercept) + " epochs="
        + String(epochs) + " tol=" + String(tol) + " shuffle=false"
    )

    var fx = fixture_planted_sparse(n, d, 610)
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var coef = ctx.enqueue_create_buffer[DType.float32](d)
    var resid = ctx.enqueue_create_buffer[DType.float32](n)
    var hx = fx[0].copy()
    var hy = fx[1].copy()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_memset(coef, Float32(0.0))
    ctx.synchronize()

    var trace = IdentityTrace()
    trace.header(
        "solver/cd_main.mojo mode=" + mode + " fixture=planted_sparse n="
        + String(n) + " d=" + String(d)
    )
    var res = cd_fit_traced(
        ctx, x, n, d, y, coef, fit_intercept, epochs, LOSS_SQRD_LOSS, alpha,
        l1_ratio, False, tol, False, trace, "cd", CdLaunch.default(), resid, True,
    )
    var n_iter = res[0]
    var intercept = res[1]
    var h = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=coef)
    ctx.synchronize()

    print("[" + mode + "] n_iter=" + String(n_iter) + " intercept=" + String(intercept) + " " + _hex32(intercept))
    var orc = cd_oracle_fit(fx[0], fx[1], n, d, fit_intercept, epochs, alpha, l1_ratio, tol, True)
    var moved = 0
    var support_ok = True
    for k in range(d):
        var dv = h.unsafe_ptr().unsafe_load(k)
        var ov = orc.coef[k]
        var same = bitcast[DType.uint32](dv) == bitcast[DType.uint32](ov)
        if not same:
            moved += 1
        var planted = fx[2][k] != Float32(0.0)
        if planted != (dv != Float32(0.0)):
            support_ok = False
        print(
            "[" + mode + "]   coef[" + String(k) + "] device " + String(dv) + " "
            + _hex32(dv) + "  oracle " + _hex32(ov) + ("" if same else "  DIFFERS")
            + "  planted " + String(fx[2][k])
        )
    print(
        "[" + mode + "] oracle: n_iter=" + String(orc.n_iter) + " intercept "
        + _hex32(orc.intercept) + "; coefficients differing from the device: "
        + String(moved) + " of " + String(d)
        + ("  (IDENTICAL asserts zero)" if mode == "IDENTICAL" else "  (FAST: a report)")
    )
    print("[" + mode + "] planted support recovered: " + String(support_ok))
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if moved != 0 or orc.n_iter != n_iter:
            raise Error("cd_main: device and oracle disagree under IDENTICAL")

    # cdPredict on the (un-centered) training rows, as a reach witness.
    var preds = ctx.enqueue_create_buffer[DType.float32](n)
    cd_predict(ctx, x, n, d, coef, intercept, preds, LOSS_SQRD_LOSS)
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hp.unsafe_ptr(), src_buf=preds)
    ctx.synchronize()
    var sse = 0.0
    for i in range(n):
        var e = Float64(hp.unsafe_ptr().unsafe_load(i)) - Float64(fx[1][i])
        sse += e * e
    print("[" + mode + "] predict: training MSE " + String(sse / Float64(n)) + " (noise variance 0.1^2/12 = 0.000833)")
    if trace.enabled:
        print("[" + mode + "] card written to " + trace.path)
    _ = x^
    _ = y^
    _ = coef^
    _ = resid^
    _ = preds^
    _ = h^
    _ = hp^
