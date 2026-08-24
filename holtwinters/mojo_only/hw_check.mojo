"""Holt-Winters: refusals, oracle sanity, device identity, launch
invariance, signed zero, NaN, reach.

DEVIATIONS 660-665's gates. The checks, in order:

    check_hw_refusals                  every parameter cuML's wrapper refuses
                                       RAISES BY NAME; DEVIATION 664's NaN /
                                       inf / non-positive-under-multiplicative
    check_hw_decompose_vs_reference    DEVIATION 660: the float32 R1Qt is
                                       the float64 closed form cast; a
                                       planted noiseless level+trend series
                                       gives back its slope and intercept; a
                                       planted noiseless season gives back its
                                       pattern; float32 oracle start values
                                       vs float64 within tolerance
    check_hw_optimizer_reduces_sse     float32 oracle: the fitted SSE is at
                                       most the SSE at the start parameters
                                       (0.4, 0.3, 0.3), every series, both
                                       seasonal types; parameters in [0, 1];
                                       float32 vs float64 SSE within tolerance
    check_hw_forecast_continues_pattern
                                       the forecast of a planted noiseless
                                       additive / multiplicative series stays
                                       within a stated bound of the planted
                                       continuation, h = 2 seasons
    check_hw_device_equals_oracle      six fixtures (additive, multiplicative,
                                       constant, zero, short n = 2 seasons,
                                       a mixed batch): every stage's bits --
                                       start values, every recorded
                                       iteration's parameters, niter,
                                       criterion, final parameters, SSE,
                                       level, trend, season, forecast -- device
                                       == oracle under IDENTICAL (canonicalized
                                       NaN, DEVIATION 661); REPORT under FAST;
                                       plus the two cards through
                                       `first_divergence`. REACH: additive and
                                       multiplicative parameters differ on one
                                       positive fixture; niter > 0; the
                                       multiplicative stmp_eps branch planted
                                       through a direct eval
    check_hw_launch_invariance         THE HEADLINE: every output byte
                                       identical across tpb_decomp 32/256,
                                       tpb_optim 128/64, pad 0/37, two
                                       poisons, run twice, and series s fitted
                                       in a batch of 1, of 7 and of 512
    check_hw_signed_zero_clamp         DEVIATION 663: bound(-0.0) = +0.0,
                                       bound(NaN) = +0.0, bound(+inf) = 1 on
                                       the host helper; a direct eval launch
                                       with alpha = beta = gamma = -0.0 equals
                                       the +0.0 launch bit for bit (device,
                                       both modes) and equals the oracle
                                       (IDENTICAL)
    check_hw_zero_series_keeps_start   DEVIATION 662: an all-zero additive
                                       series keeps (0.4, 0.3, 0.3), criterion
                                       MIN_GRAD_NORM, niter 0, SSE 0, device
                                       and oracle; prints iter000's bits
    check_hw_card_is_emitted           the stage list and the run-to-run
                                       control

SABOTAGES (each a `-D MOJOLEARN_HW_SABOTAGE_<NAME>=1` build, run under
IDENTICAL, nothing edited; the README carries the failing lines and the
measured verdict of every arm):
    FAIL on Apple, as they must
      ROTATE_CONV       conv1d's sum starts at block_idx.x % filter_size
      NO_FTZ            ftz dropped at every stored intermediate
      SWAP_FMA          the other product fused in the eval's level update
      NO_ZERO_DIR_GUARD DEVIATION 662 off
      CLAMP_GE          DEVIATION 663's lower test loosened to `>=`, so a
                        -0.0 survives the clamp
      LS_TIE            the line-search acceptance test loosened to `>=`
      CRIT_ORDER        the two stop criteria tested in the other order
    NULL on Apple, RECORDED with the reason (both are other vendors' arms)
      STD_SQRT          std.math.sqrt for the BFGS step size -- on Metal
                        both spellings are the same correctly-rounded sqrt
      HW_MAX_CLAMP      bound_device as min(max(0.0, v), 1.0) -- the
                        constant zero lets LLVM fold the maxnum away

Run:

    tools/with_build_lock.sh     pixi run mojo run -I . holtwinters/mojo_only/hw_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . holtwinters/mojo_only/hw_check.mojo
"""

from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, first_divergence, read_trace_lines
from holtwinters.estimator import (
    HWFit,
    download_f32,
    holtwinters_fit_host_traced,
    holtwinters_forecast_host_traced,
    upload_f32,
)
from holtwinters.mojo_only.hw_fixture import (
    HWFixtureSpec,
    hex32,
    hw_fixture,
    hw_fixture_mixed,
    hw_series_value,
    spec_additive,
    spec_additive_no_season,
    spec_additive_noiseless,
    spec_constant,
    spec_multiplicative,
    spec_multiplicative_noiseless,
    spec_zero,
)
from holtwinters.mojo_only.hw_oracle import (
    HWOracleFit,
    oracle_eval,
    oracle_fit,
    oracle_forecast,
    oracle_sse_at,
)
from holtwinters.ported.holtwinters.internal.hw_decompose import host_filter, host_r1qt
from holtwinters.ported.holtwinters.internal.hw_utils import (
    bound_device,
    hw_float32_inf,
    hw_sabotage_name,
)
from holtwinters.ported.holtwinters.runner import (
    HW_ALPHA0,
    HW_BETA0,
    HW_GAMMA0,
    HW_DEFAULT_EPS,
    holtwinters_eval,
    holtwinters_optim,
)
from holtwinters.ported.tsa.holtwinters_params import (
    OPTIM_MIN_GRAD_NORM,
    SEASONAL_ADDITIVE,
    SEASONAL_MULTIPLICATIVE,
    criterion_name,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from solver.mojo_only.record_canon import canon_nan_f32, canon_nan_list

comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime SCRATCH = "/tmp"

comptime N = 72
comptime FREQ = 12
comptime START_PERIODS = 2
comptime BATCH = 7
comptime TRACE_ITERS = 64
comptime H = 24


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _f32_list[dt: DType](v: List[Scalar[dt]]) -> List[Float32]:
    var out = List[Float32]()
    out.reserve(len(v))
    for e in v:
        out.append(Float32(e))
    return out^


def _first_diff(tag: String, a: List[Float32], b: List[Float32]) -> String:
    """"" if every cell agrees bitwise after NaN canonicalization, else
    the first mismatch."""
    if len(a) != len(b):
        return tag + ": length " + String(len(a)) + " vs " + String(len(b))
    for i in range(len(a)):
        var x = canon_nan_f32(a[i])
        var y = canon_nan_f32(b[i])
        if bitcast[DType.uint32](x) != bitcast[DType.uint32](y):
            return tag + "[" + String(i) + "]: " + hex32(x) + " vs " + hex32(y) + " (" + String(x) + " vs " + String(y) + ")"
    return String("")


def _first_diff_i(tag: String, a: List[Int32], b: List[Int]) -> String:
    if len(a) != len(b):
        return tag + ": length " + String(len(a)) + " vs " + String(len(b))
    for i in range(len(a)):
        if Int(a[i]) != b[i]:
            return tag + "[" + String(i) + "]: " + String(a[i]) + " vs " + String(b[i])
    return String("")


def _device_fit(
    ctx: DeviceContext,
    data: List[Float32],
    n: Int,
    batch_size: Int,
    seasonal: String,
    mut trace: IdentityTrace,
    tpb_decomp: Int = -1,
    tpb_optim: Int = 128,
    pad: Int = 0,
    poison: Float32 = Float32(0.0),
) raises -> HWFit:
    return holtwinters_fit_host_traced(
        ctx, data, n, batch_size, FREQ, START_PERIODS, seasonal, HW_DEFAULT_EPS, trace,
        TRACE_ITERS, tpb_decomp, tpb_optim, pad, poison,
    )


def _oracle32(data: List[Float32], n: Int, batch_size: Int, seasonal: Int) raises -> HWOracleFit[DType.float32]:
    return oracle_fit[DType.float32](data, n, batch_size, FREQ, START_PERIODS, seasonal, HW_DEFAULT_EPS, TRACE_ITERS)


def _oracle64(data: List[Float32], n: Int, batch_size: Int, seasonal: Int) raises -> HWOracleFit[DType.float64]:
    return oracle_fit[DType.float64](data, n, batch_size, FREQ, START_PERIODS, seasonal, HW_DEFAULT_EPS, TRACE_ITERS)


def _seasonal_str(seasonal: Int) -> String:
    return String("additive") if seasonal == SEASONAL_ADDITIVE else String("multiplicative")


def _rel_err(a: Float64, b: Float64) -> Float64:
    var d = abs(a - b)
    var m = abs(b)
    if m < 1.0:
        m = 1.0
    return d / m


# ---------------------------------------------------------------------------


def _expect_raise(ctx: DeviceContext, mut tr: IdentityTrace, what: String, data: List[Float32], n: Int, bs: Int, freq: Int, sp: Int, seasonal: String, eps: Float32, needle: String) raises -> Int:
    var raised = False
    try:
        _ = holtwinters_fit_host_traced(ctx, data, n, bs, freq, sp, seasonal, eps, tr, 0)
    except e:
        raised = True
        if String(e).find(needle) < 0:
            raise Error("check_hw_refusals: " + what + " raised '" + String(e) + "' without naming '" + needle + "'")
    if not raised:
        raise Error("check_hw_refusals: " + what + " did not raise")
    return 1


def check_hw_refusals() raises:
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var good = hw_fixture(spec_additive(), N, 2, FREQ, 1)
    var n_refused = 0

    n_refused += _expect_raise(ctx, tr, "frequency 1", good, N, 2, 1, 2, "additive", HW_DEFAULT_EPS, "seasonal_periods")
    n_refused += _expect_raise(ctx, tr, "start_periods 1", good, N, 2, FREQ, 1, "additive", HW_DEFAULT_EPS, "start_periods")
    n_refused += _expect_raise(ctx, tr, "frequency < start_periods", good, N, 2, FREQ, FREQ + 1, "additive", HW_DEFAULT_EPS, "cannot be less than start_periods")
    n_refused += _expect_raise(ctx, tr, "n short", good, 2 * FREQ - 1, 2, FREQ, 2, "additive", HW_DEFAULT_EPS, "must be at least freq*start_periods")
    n_refused += _expect_raise(ctx, tr, "eps 0", good, N, 2, FREQ, 2, "additive", Float32(0.0), "Epsilon must be positive")
    n_refused += _expect_raise(ctx, tr, "seasonal name", good, N, 2, FREQ, 2, "damped", HW_DEFAULT_EPS, "seasonal='damped'")
    n_refused += _expect_raise(ctx, tr, "ts_num 0", good, N, 0, FREQ, 2, "additive", HW_DEFAULT_EPS, "ts_num")
    var short = List[Float32]()
    for i in range(N):
        short.append(good[i])
    n_refused += _expect_raise(ctx, tr, "wrong length", short, N, 2, FREQ, 2, "additive", HW_DEFAULT_EPS, "endog: expected")
    # DEVIATION 664
    var nan_data = good.copy()
    nan_data[N + 5] = bitcast[DType.float32](UInt32(0x7FC00000))
    n_refused += _expect_raise(ctx, tr, "NaN", nan_data, N, 2, FREQ, 2, "additive", HW_DEFAULT_EPS, "endog[series 1, t=5]")
    var inf_data = good.copy()
    inf_data[3] = hw_float32_inf()
    n_refused += _expect_raise(ctx, tr, "inf", inf_data, N, 2, FREQ, 2, "additive", HW_DEFAULT_EPS, "endog[series 0, t=3]")
    var zero_data = hw_fixture(spec_multiplicative(), N, 2, FREQ, 1)
    zero_data[N + 7] = Float32(0.0)
    n_refused += _expect_raise(ctx, tr, "zero under multiplicative", zero_data, N, 2, FREQ, 2, "multiplicative", HW_DEFAULT_EPS, "strictly positive under seasonal='multiplicative'")
    var neg_data = hw_fixture(spec_multiplicative(), N, 2, FREQ, 1)
    neg_data[11] = Float32(-1.0)
    n_refused += _expect_raise(ctx, tr, "negative under multiplicative", neg_data, N, 2, FREQ, 2, "mul", HW_DEFAULT_EPS, "endog[series 0, t=11]")
    # the same zero is LEGAL under additive
    _ = holtwinters_fit_host_traced(ctx, zero_data, N, 2, FREQ, 2, "add", HW_DEFAULT_EPS, tr, 0)
    # forecast h <= 0
    var f = holtwinters_fit_host_traced(ctx, good, N, 2, FREQ, 2, "additive", HW_DEFAULT_EPS, tr, 0)
    var raised = False
    try:
        _ = holtwinters_forecast_host_traced(ctx, f, 0, tr)
    except e:
        raised = True
        if String(e).find("h must be > 0") < 0:
            raise Error("check_hw_refusals: h=0 raised '" + String(e) + "'")
    if not raised:
        raise Error("check_hw_refusals: h=0 did not raise")
    n_refused += 1
    print("check_hw_refusals OK [" + _mode_name() + "]: " + String(n_refused) + " refusals by name (cuML's wrapper rules + DEVIATION 664); a zero under additive accepted")


def check_hw_decompose_vs_reference() raises:
    # (a) DEVIATION 660: R1Qt float32 == float64 closed form cast, for several trend_len
    var n_rq = 0
    for m in [3, 7, 13, 24, 25, 100]:
        var rq = host_r1qt(m)
        var mm = Float64(m)
        var tbar = (mm + 1.0) / 2.0
        var sxx = (mm * (mm * mm - 1.0)) / 12.0
        for i in range(m):
            var t = Float64(i + 1)
            var w1 = (t - tbar) / sxx
            var w0 = 1.0 / mm - tbar * w1
            if _rel_err(Float64(rq[2 * i + 1]), w1) > 1e-6 or _rel_err(Float64(rq[2 * i]), w0) > 1e-6:
                raise Error("check_hw_decompose_vs_reference: R1Qt[m=" + String(m) + ", i=" + String(i) + "] off: " + String(rq[2 * i]) + "," + String(rq[2 * i + 1]) + " vs " + String(w0) + "," + String(w1))
            n_rq += 1
        # pinv property: sum w0 = 1, sum w1 = 0, sum t w1 = 1, sum t w0 = 0
        var s0 = 0.0
        var s1 = 0.0
        var st1 = 0.0
        var st0 = 0.0
        for i in range(m):
            s0 += Float64(rq[2 * i])
            s1 += Float64(rq[2 * i + 1])
            st1 += Float64(i + 1) * Float64(rq[2 * i + 1])
            st0 += Float64(i + 1) * Float64(rq[2 * i])
        if abs(s0 - 1.0) > 1e-5 or abs(s1) > 1e-5 or abs(st1 - 1.0) > 1e-4 or abs(st0) > 1e-4:
            raise Error("check_hw_decompose_vs_reference: pinv identities fail at m=" + String(m))
    # (b) a planted noiseless level+trend series: slope T/16 and intercept
    var bs = 3
    var spec = spec_additive_no_season()
    var data = hw_fixture(spec, N, bs, FREQ, 3)
    var o = _oracle32(data, N, bs, SEASONAL_ADDITIVE)
    var planted_slope = Float64(spec.trend) / 16.0
    for s in range(bs):
        var slope = Float64(o.start_trend[s])
        if _rel_err(slope, planted_slope) > 1e-3:
            raise Error("check_hw_decompose_vs_reference: series " + String(s) + " start_trend " + String(slope) + " vs planted " + String(planted_slope))
        # the smoothed trend is the line itself: trend[o] = value at t = o + half
        var half = FREQ // 2
        var expect0 = Float64(hw_series_value(spec, s, half, FREQ, 3))
        if _rel_err(Float64(o.decomp_trend[s]), expect0) > 1e-5:
            raise Error("check_hw_decompose_vs_reference: series " + String(s) + " decomp_trend[0] " + String(o.decomp_trend[s]) + " vs " + String(expect0))
        # intercept: the regression line at t = 0 is trend[0] - slope
        var expect_level = Float64(o.decomp_trend[s]) - planted_slope
        if _rel_err(Float64(o.start_level[s]), expect_level) > 1e-4:
            raise Error("check_hw_decompose_vs_reference: series " + String(s) + " start_level " + String(o.start_level[s]) + " vs " + String(expect_level))
        # no season planted -> start_season ~ 0
        for j in range(FREQ):
            if abs(Float64(o.start_season[j * bs + s])) > 1e-2:
                raise Error("check_hw_decompose_vs_reference: series " + String(s) + " start_season[" + String(j) + "] = " + String(o.start_season[j * bs + s]) + " with no season planted")
    # (c) a planted noiseless additive season is recovered (the residual of
    # a linear trend plus a zero-mean season is the season itself)
    var spec_s = spec_additive_noiseless()
    var data_s = hw_fixture(spec_s, N, bs, FREQ, 5)
    var os_ = _oracle32(data_s, N, bs, SEASONAL_ADDITIVE)
    var worst_season = 0.0
    for s in range(bs):
        # recover S_j = value(t) - (mean over one period of value) - T*(t - mean t), for t in first period
        var mean_v = 0.0
        for t in range(FREQ):
            mean_v += Float64(hw_series_value(spec_s, s, t, FREQ, 5))
        mean_v /= Float64(FREQ)
        var tbar = Float64(FREQ - 1) / 2.0
        for t in range(FREQ):
            var planted = Float64(hw_series_value(spec_s, s, t, FREQ, 5)) - mean_v - (Float64(spec_s.trend) / 16.0) * (Float64(t) - tbar)
            var got = Float64(os_.start_season[t * bs + s])
            var e = abs(got - planted)
            if e > worst_season:
                worst_season = e
            if e > 1e-3:
                raise Error("check_hw_decompose_vs_reference: series " + String(s) + " start_season[" + String(t) + "] " + String(got) + " vs planted " + String(planted))
    # (d) float32 oracle vs float64 oracle start values on the hashed fixtures
    var worst = 0.0
    for seasonal in [SEASONAL_ADDITIVE, SEASONAL_MULTIPLICATIVE]:
        var sp = spec_additive() if seasonal == SEASONAL_ADDITIVE else spec_multiplicative()
        var d = hw_fixture(sp, N, BATCH, FREQ, 2)
        var o32 = _oracle32(d, N, BATCH, seasonal)
        var o64 = _oracle64(d, N, BATCH, seasonal)
        for s in range(BATCH):
            var e = _rel_err(Float64(o32.start_level[s]), o64.start_level[s])
            if e > worst:
                worst = e
            e = _rel_err(Float64(o32.start_trend[s]), o64.start_trend[s])
            if e > worst:
                worst = e
            for j in range(FREQ):
                e = _rel_err(Float64(o32.start_season[j * BATCH + s]), o64.start_season[j * BATCH + s])
                if e > worst:
                    worst = e
    if worst > 1e-4:
        raise Error("check_hw_decompose_vs_reference: float32 vs float64 start values worst rel err " + String(worst))
    print(
        "check_hw_decompose_vs_reference OK [" + _mode_name() + "]: R1Qt == float64 pinv at " + String(n_rq)
        + " weights (6 trend_len) with the four pinv identities; planted slope/intercept/season recovered (worst season err "
        + String(worst_season) + "); float32 vs float64 start values worst rel err " + String(worst)
    )


def check_hw_optimizer_reduces_sse() raises:
    var n_series = 0
    var worst = 0.0
    var n_at_limit = 0
    for seasonal in [SEASONAL_ADDITIVE, SEASONAL_MULTIPLICATIVE]:
        var sp = spec_additive() if seasonal == SEASONAL_ADDITIVE else spec_multiplicative()
        var d = hw_fixture(sp, N, BATCH, FREQ, 2)
        var o32 = _oracle32(d, N, BATCH, seasonal)
        var o64 = _oracle64(d, N, BATCH, seasonal)
        for s in range(BATCH):
            var start_sse = oracle_sse_at[DType.float32](o32, s, HW_ALPHA0, HW_BETA0, HW_GAMMA0)
            if not (o32.sse[s] <= start_sse):
                raise Error("check_hw_optimizer_reduces_sse: " + _seasonal_str(seasonal) + " series " + String(s) + " fitted SSE " + String(o32.sse[s]) + " > start SSE " + String(start_sse))
            for v in [o32.alpha[s], o32.beta[s], o32.gamma[s]]:
                if not (v >= Float32(0.0) and v <= Float32(1.0)):
                    raise Error("check_hw_optimizer_reduces_sse: parameter out of [0,1]: " + String(v))
            if o32.criterion[s] == 0:
                n_at_limit += 1
            var e = _rel_err(Float64(o32.sse[s]), o64.sse[s])
            if e > worst:
                worst = e
            n_series += 1
        print(
            "  " + _seasonal_str(seasonal) + " series 0: start SSE "
            + String(oracle_sse_at[DType.float32](o32, 0, HW_ALPHA0, HW_BETA0, HW_GAMMA0))
            + " -> fitted " + String(o32.sse[0]) + " (float64 " + String(o64.sse[0]) + "), params "
            + String(o32.alpha[0]) + "/" + String(o32.beta[0]) + "/" + String(o32.gamma[0])
            + ", niter " + String(o32.niter[0]) + ", " + criterion_name(o32.criterion[0])
        )
    # float32 vs float64 SSE: a TOLERANCE report (the two optimizers walk
    # different float paths and can stop at different iterations), not an
    # identity claim
    print(
        "check_hw_optimizer_reduces_sse OK [" + _mode_name() + "]: " + String(n_series)
        + " series, fitted SSE <= start SSE everywhere, parameters in [0,1], " + String(n_at_limit)
        + " hit the 1000-iteration limit; float32 vs float64 SSE worst rel diff " + String(worst) + " (report)"
    )


def check_hw_forecast_continues_pattern() raises:
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var worst_all = 0.0
    for seasonal in [SEASONAL_ADDITIVE, SEASONAL_MULTIPLICATIVE]:
        var sp = spec_additive_noiseless() if seasonal == SEASONAL_ADDITIVE else spec_multiplicative_noiseless()
        var d = hw_fixture(sp, N, 3, FREQ, 4)
        var f = _device_fit(ctx, d, N, 3, _seasonal_str(seasonal), tr)
        var fc = holtwinters_forecast_host_traced(ctx, f, H, tr)
        var worst = 0.0
        var worst_rel = 0.0
        for s in range(3):
            for i in range(H):
                var planted = Float64(hw_series_value(sp, s, N + i, FREQ, 4))
                var got = Float64(fc[s + i * 3])
                var e = abs(got - planted)
                if e > worst:
                    worst = e
                var r = e / abs(planted)
                if r > worst_rel:
                    worst_rel = r
        # THE BOUND (chosen, stated): 0.5% of the value. The season
        # amplitude is ~5% of the value (additive 800/16 on ~1000;
        # multiplicative 300/1024 on ~4000) and the trend over 2 seasons is
        # ~0.7% (additive) / ~1% (multiplicative), so a forecast that
        # dropped the season or the trend fails it.
        if worst_rel > 5e-3:
            raise Error("check_hw_forecast_continues_pattern: " + _seasonal_str(seasonal) + " worst rel err " + String(worst_rel) + " (abs " + String(worst) + ") over h=" + String(H))
        if worst_rel > worst_all:
            worst_all = worst_rel
        print("  " + _seasonal_str(seasonal) + ": worst |forecast - planted| " + String(worst) + " (rel " + String(worst_rel) + ") over h=" + String(H) + ", 3 series; params s0 " + String(f.alpha[0]) + "/" + String(f.beta[0]) + "/" + String(f.gamma[0]))
    print("check_hw_forecast_continues_pattern OK [" + _mode_name() + "]: noiseless planted additive and multiplicative series continued within 0.5% over 2 seasons (worst rel " + String(worst_all) + ")")


def _tag_of(line: String) -> String:
    """The tag column of a trace record (`idx\ttag\tdtype\tcount\thash`)."""
    var parts = line.split("\t")
    if len(parts) >= 2:
        return String(parts[1])
    return String(line)


def _stage_delta(path_a: String, path_b: String) raises -> String:
    """HOW MANY STAGES MOVED, not just the first.

    The mamba lane measured (2026-08-23, one M4) an arm that moves 13 of 16
    stages and 23 of 64 cells of the second-to-last stage while leaving the
    FINAL OUTPUT bit-identical, because a later add puts a small term beside
    a large one and rounds the difference away. First-divergence alone
    cannot tell that arm from one that moves a single trailing stage, so
    every sabotage arm here reports a COUNT.
    """
    var a = read_trace_lines(path_a)
    var b = read_trace_lines(path_b)
    if len(a) != len(b):
        return (
            String("stages STRUCTURAL (") + String(len(a)) + " vs " + String(len(b))
            + " records: the two runs did not take the same stages)"
        )
    var n_diff = 0
    var first = String("")
    for i in range(len(a)):
        if a[i] != b[i]:
            n_diff += 1
            if first == "":
                first = _tag_of(a[i])
    if n_diff == 0:
        return String("stages 0/") + String(len(a))
    return (
        String("stages ") + String(n_diff) + "/" + String(len(a))
        + " (first " + first + ")"
    )


def _count_diff(a: List[Float32], b: List[Float32]) -> Int:
    var n = len(a) if len(a) < len(b) else len(b)
    var c = 0
    for i in range(n):
        if bitcast[DType.uint32](canon_nan_f32(a[i])) != bitcast[DType.uint32](canon_nan_f32(b[i])):
            c += 1
    return c


def _cell_delta(f: HWFit, o: HWOracleFit[DType.float32], fc: List[Float32], ofc: List[Float32]) -> String:
    """HOW MANY CELLS MOVED, per field and in total. An arm that fails only
    on the last stage may be hiding that it fails on nothing that matters;
    an arm that moves the fitted parameters but no component cell is saying
    something different from one that moves every component."""
    var names = ["sse", "alpha", "beta", "gamma", "iter", "level", "trend", "season", "fcast"]
    var da = List[Int]()
    da.append(_count_diff(f.sse, o.sse))
    da.append(_count_diff(f.alpha, o.alpha))
    da.append(_count_diff(f.beta, o.beta))
    da.append(_count_diff(f.gamma, o.gamma))
    da.append(_count_diff(f.iter_trace, o.iter_trace))
    da.append(_count_diff(f.level, o.level))
    da.append(_count_diff(f.trend, o.trend))
    da.append(_count_diff(f.season, o.season))
    da.append(_count_diff(fc, ofc))
    var tot = 0
    var totn = len(f.sse) + len(f.alpha) + len(f.beta) + len(f.gamma) + len(f.iter_trace)
    totn += len(f.level) + len(f.trend) + len(f.season) + len(fc)
    var parts = String("")
    for i in range(len(da)):
        tot += da[i]
        if da[i] != 0:
            if parts != "":
                parts += " "
            parts += names[i] + "=" + String(da[i])
    var out = String("cells ") + String(tot) + "/" + String(totn)
    if parts != "":
        out += " [" + parts + "]"
    # THE OUTPUT-ONLY BLIND SPOT, named: forecast is the last stage.
    if tot != 0 and da[len(da) - 1] == 0:
        out += " (FINAL FORECAST UNCHANGED)"
    return out


def _compare_fit(tag: String, f: HWFit, o: HWOracleFit[DType.float32], fc: List[Float32], ofc: List[Float32]) -> String:
    var checks = List[String]()
    checks.append(_first_diff(tag + " hw.sse", f.sse, o.sse))
    checks.append(_first_diff(tag + " hw.params.alpha", f.alpha, o.alpha))
    checks.append(_first_diff(tag + " hw.params.beta", f.beta, o.beta))
    checks.append(_first_diff(tag + " hw.params.gamma", f.gamma, o.gamma))
    checks.append(_first_diff_i(tag + " hw.opt.niter", f.niter, o.niter))
    checks.append(_first_diff_i(tag + " hw.opt.criterion", f.criterion, o.criterion))
    checks.append(_first_diff(tag + " hw.opt.iter*.params", f.iter_trace, o.iter_trace))
    checks.append(_first_diff(tag + " hw.level", f.level, o.level))
    checks.append(_first_diff(tag + " hw.trend", f.trend, o.trend))
    checks.append(_first_diff(tag + " hw.season", f.season, o.season))
    checks.append(_first_diff(tag + " hw.forecast", fc, ofc))
    for c in checks:
        if c != "":
            return c
    return String("")


def _oracle_card(path: String, o: HWOracleFit[DType.float32], ofc: List[Float32], header: String) raises:
    """The oracle's card in the runner's stage order, through canon_nan_list."""
    var t = IdentityTrace.to_path(path)
    t.header(header)
    t.record_list_f32("hw.input", canon_nan_list(o.ts))
    t.record_list_f32("hw.decomp.trend", canon_nan_list(o.decomp_trend))
    t.record_list_f32("hw.decomp.season", canon_nan_list(o.decomp_season))
    t.record_list_f32("hw.start.level", canon_nan_list(o.start_level))
    t.record_list_f32("hw.start.trend", canon_nan_list(o.start_trend))
    t.record_list_f32("hw.start.season", canon_nan_list(o.start_season))
    var max_iter = 0
    for v in o.niter:
        if v > max_iter:
            max_iter = v
    var shown = max_iter if max_iter < o.trace_iters else o.trace_iters
    for it in range(shown):
        var s = String(it)
        while s.byte_length() < 3:
            s = "0" + s
        var chunk = List[Float32]()
        for k in range(3 * o.batch_size):
            chunk.append(o.iter_trace[it * 3 * o.batch_size + k])
        t.record_list_f32("hw.opt.iter" + s + ".params", canon_nan_list(chunk))
    var ni = List[Int32]()
    var cr = List[Int32]()
    for v in o.niter:
        ni.append(Int32(v))
    for v in o.criterion:
        cr.append(Int32(v))
    t.record_list_i32("hw.opt.niter", ni)
    t.record_list_i32("hw.opt.criterion", cr)
    var params = List[Float32]()
    for v in o.alpha:
        params.append(v)
    for v in o.beta:
        params.append(v)
    for v in o.gamma:
        params.append(v)
    t.record_list_f32("hw.params", canon_nan_list(params))
    t.record_list_f32("hw.sse", canon_nan_list(o.sse))
    t.record_list_f32("hw.level", canon_nan_list(o.level))
    t.record_list_f32("hw.trend", canon_nan_list(o.trend))
    t.record_list_f32("hw.season", canon_nan_list(o.season))
    t.record_list_f32("hw.forecast", canon_nan_list(ofc))


def check_hw_device_equals_oracle() raises:
    var ctx = DeviceContext()
    var n_stages_ok = 0
    var n_reported = 0
    var names = ["additive", "multiplicative", "constant", "zero", "short", "mixed"]
    for fi in range(len(names)):
        var name = names[fi]
        var n = N
        var bs = BATCH
        var seasonal = SEASONAL_ADDITIVE
        var data: List[Float32]
        if name == "additive":
            data = hw_fixture(spec_additive(), n, bs, FREQ, 2)
        elif name == "multiplicative":
            seasonal = SEASONAL_MULTIPLICATIVE
            data = hw_fixture(spec_multiplicative(), n, bs, FREQ, 2)
        elif name == "constant":
            data = hw_fixture(spec_constant(), n, bs, FREQ, 2)
        elif name == "zero":
            data = hw_fixture(spec_zero(), n, bs, FREQ, 2)
        elif name == "short":
            n = 2 * FREQ
            data = hw_fixture(spec_additive(), n, bs, FREQ, 9)
        else:
            var specs: List[HWFixtureSpec] = [spec_additive(), spec_constant(), spec_additive_noiseless(), spec_zero()]
            data = hw_fixture_mixed(specs, n, bs, FREQ, 6)
        var dpath = SCRATCH + "/mojolearn_hw_dev_" + name + ".card"
        var opath = SCRATCH + "/mojolearn_hw_orc_" + name + ".card"
        var dt = IdentityTrace.to_path(dpath)
        dt.header("hw device " + name)
        # the mixed fixture runs at tpb 4 (two blocks of a 7-series batch), so
        # a launch-geometry-dependent order (the ROTATE_CONV sabotage) is
        # visible to THIS gate and not only to the batch-512 arm of
        # check_hw_launch_invariance
        var tpb_d = 4 if name == "mixed" else -1
        var f = _device_fit(ctx, data, n, bs, _seasonal_str(seasonal), dt, tpb_d, 4 if name == "mixed" else 128)
        var fc = holtwinters_forecast_host_traced(ctx, f, H, dt)
        var o = oracle_fit[DType.float32](data, n, bs, FREQ, START_PERIODS, seasonal, HW_DEFAULT_EPS, TRACE_ITERS)
        var ofc = oracle_forecast[DType.float32](o, H)
        _oracle_card(opath, o, ofc, "hw oracle " + name)
        # the card first (stage order, so the FIRST divergent stage is
        # named), then every cell
        var bad = first_divergence(dpath, opath)
        if bad != "":
            bad = name + " card: " + bad
        else:
            bad = _compare_fit(name, f, o, fc, ofc)
        # ALWAYS, pass or fail: which stages moved and how many cells.
        var delta = _stage_delta(dpath, opath) + "; " + _cell_delta(f, o, fc, ofc)
        if bad != "" or hw_sabotage_name() != "none":
            print("  DELTA " + name + ": " + delta)
        if bad != "":
            comptime if IDENTICAL:
                raise Error("check_hw_device_equals_oracle FAILED under IDENTICAL (sabotage " + hw_sabotage_name() + "): " + bad)
            else:
                print("  RECORDED [FAST] " + bad)
                n_reported += 1
        else:
            n_stages_ok += 1
        print("  " + name + ": series 0 params " + String(f.alpha[0]) + "/" + String(f.beta[0]) + "/" + String(f.gamma[0]) + " niter " + String(f.niter[0]) + " " + criterion_name(Int(f.criterion[0])) + " sse " + String(f.sse[0]) + " " + hex32(f.sse[0]) + (" == oracle" if bad == "" else ""))
    # REACH (per branch): additive vs multiplicative on one positive fixture
    var tr = IdentityTrace.disabled()
    var pos = hw_fixture(spec_multiplicative(), N, BATCH, FREQ, 2)
    var fa = _device_fit(ctx, pos, N, BATCH, "additive", tr)
    var fm = _device_fit(ctx, pos, N, BATCH, "multiplicative", tr)
    var n_param_diff = 0
    var n_season_diff = 0
    var n_iter_pos = 0
    for s in range(BATCH):
        if bitcast[DType.uint32](fa.alpha[s]) != bitcast[DType.uint32](fm.alpha[s]) or bitcast[DType.uint32](fa.gamma[s]) != bitcast[DType.uint32](fm.gamma[s]):
            n_param_diff += 1
        if bitcast[DType.uint32](fa.season[s]) != bitcast[DType.uint32](fm.season[s]):
            n_season_diff += 1
        if fa.niter[s] > 0 and fm.niter[s] > 0:
            n_iter_pos += 1
    if n_param_diff == 0 or n_season_diff != BATCH or n_iter_pos != BATCH:
        raise Error("check_hw_device_equals_oracle REACH: additive vs multiplicative params differ on " + String(n_param_diff) + " series, season cells differ on " + String(n_season_diff) + ", both iterate on " + String(n_iter_pos))
    # REACH: the multiplicative `stmp_eps` branch, planted through a direct
    # eval with start_season = 0 (|stmp| <= 1e-6) -- device vs oracle
    var bs2 = 3
    var ts_h = List[Float32]()
    for t in range(N):
        for s in range(bs2):
            ts_h.append(pos[s * N + t])
    var o_m = oracle_fit[DType.float32](pos, N, bs2, FREQ, START_PERIODS, SEASONAL_MULTIPLICATIVE, HW_DEFAULT_EPS, 0)
    var ts_d = upload_f32(ctx, ts_h)
    var sl = upload_f32(ctx, o_m.start_level)
    var st = upload_f32(ctx, o_m.start_trend)
    var zero_season = List[Float32]()
    for _ in range(FREQ * bs2):
        zero_season.append(Float32(0.0))
    var ss = upload_f32(ctx, zero_season)
    var pa = List[Float32]()
    for _ in range(bs2):
        pa.append(Float32(0.35))
    var al = upload_f32(ctx, pa)
    var be = upload_f32(ctx, pa)
    var ga = upload_f32(ctx, pa)
    var comps = (N - FREQ) * bs2
    var lv = ctx.enqueue_create_buffer[DType.float32](comps)
    var tv = ctx.enqueue_create_buffer[DType.float32](comps)
    var sv = ctx.enqueue_create_buffer[DType.float32](comps)
    var ev = ctx.enqueue_create_buffer[DType.float32](bs2)
    ctx.synchronize()
    holtwinters_eval(ctx, ts_d, N, bs2, FREQ, sl, st, ss, al, be, ga, lv, tv, sv, ev, SEASONAL_MULTIPLICATIVE)
    var dev_level = download_f32(ctx, lv, comps)
    var dev_sse = download_f32(ctx, ev, bs2)
    # the oracle's eval at the same planted inputs
    var ol = List[Float32]()
    var ot = List[Float32]()
    var osn = List[Float32]()
    for _ in range(comps):
        ol.append(Float32(0.0))
        ot.append(Float32(0.0))
        osn.append(Float32(0.0))
    var bad2 = String("")
    for s in range(bs2):
        var ps = List[Float32]()
        for _ in range(FREQ):
            ps.append(Float32(0.0))
        var e = oracle_eval[DType.float32](s, o_m.ts, N, bs2, FREQ, FREQ, o_m.start_level[s], o_m.start_trend[s], ps, zero_season, Float32(0.35), Float32(0.35), Float32(0.35), False, True, ol, ot, osn)
        if bitcast[DType.uint32](canon_nan_f32(e)) != bitcast[DType.uint32](canon_nan_f32(dev_sse[s])):
            bad2 = "stmp_eps eval sse series " + String(s) + ": device " + hex32(dev_sse[s]) + " oracle " + hex32(e)
    if bad2 == "":
        bad2 = _first_diff("stmp_eps eval level", dev_level, ol)
    if bad2 != "":
        comptime if IDENTICAL:
            raise Error("check_hw_device_equals_oracle stmp_eps branch FAILED: " + bad2)
        else:
            print("  RECORDED [FAST] " + bad2)
            n_reported += 1
    _ = ts_d^
    _ = sl^
    _ = st^
    _ = ss^
    _ = al^
    _ = be^
    _ = ga^
    _ = lv^
    _ = tv^
    _ = sv^
    _ = ev^
    print(
        "check_hw_device_equals_oracle " + ("OK" if IDENTICAL else "REPORT") + " [" + _mode_name() + "]: " + String(n_stages_ok)
        + " of 6 fixtures bit-identical at every stage and cell (cards through first_divergence); REACH: additive vs multiplicative params differ on "
        + String(n_param_diff) + "/" + String(BATCH) + " series, season cells on all; stmp_eps branch planted (sse s0 " + hex32(dev_sse[0]) + ")"
        + ("" if n_reported == 0 else "; " + String(n_reported) + " RECORDED [FAST]")
    )


def _all_bits(f: HWFit, fc: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for v in f.sse:
        out.append(v)
    for v in f.alpha:
        out.append(v)
    for v in f.beta:
        out.append(v)
    for v in f.gamma:
        out.append(v)
    for v in f.level:
        out.append(v)
    for v in f.trend:
        out.append(v)
    for v in f.season:
        out.append(v)
    for v in f.iter_trace:
        out.append(v)
    for v in fc:
        out.append(v)
    return out^


def check_hw_launch_invariance() raises:
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var n_ok = 0
    for seasonal in [SEASONAL_ADDITIVE, SEASONAL_MULTIPLICATIVE]:
        var sp = spec_additive() if seasonal == SEASONAL_ADDITIVE else spec_multiplicative()
        var name = _seasonal_str(seasonal)
        var d = hw_fixture(sp, N, BATCH, FREQ, 2)
        # A: tpb_decomp 32, tpb_optim 128, pad 0, poison -987654
        var fa = _device_fit(ctx, d, N, BATCH, name, tr, 32, 128, 0, Float32(-987654.0))
        var fca = holtwinters_forecast_host_traced(ctx, fa, H, tr, 32)
        # B: tpb_decomp 256, tpb_optim 64, pad 37, poison 13.5
        var fb = _device_fit(ctx, d, N, BATCH, name, tr, 256, 64, 37, Float32(13.5))
        var fcb = holtwinters_forecast_host_traced(ctx, fb, H, tr, 256)
        # C: A again
        var fcc = _device_fit(ctx, d, N, BATCH, name, tr, 32, 128, 0, Float32(-987654.0))
        var fccf = holtwinters_forecast_host_traced(ctx, fcc, H, tr, 32)
        # NEGATIVE CONTROL (the mamba lane's clause-(c) lesson, 2026-08-23).
        # Everything below is a chain of `_first_diff(...) == ""`. If
        # `_all_bits` came back empty, or `_device_fit` quietly ignored its
        # tpb / pad / poison arguments, every comparison here would compare a
        # thing to ITSELF and this gate would pass for ever, on every vendor,
        # while proving nothing. So first prove the machinery can SEE a
        # difference at all: a fit of a DIFFERENT series must differ.
        var d_other = hw_fixture(sp, N, BATCH, FREQ, 77)
        var fo = _device_fit(ctx, d_other, N, BATCH, name, tr, 32, 128, 0, Float32(-987654.0))
        var fco = holtwinters_forecast_host_traced(ctx, fo, H, tr, 32)
        var ctrl_a = _all_bits(fa, fca)
        var ctrl_b = _all_bits(fo, fco)
        if len(ctrl_a) == 0:
            raise Error(
                "check_hw_launch_invariance VACUOUS [" + _mode_name() + "]:"
                " _all_bits returned 0 cells, so every comparison below is empty"
            )
        if _first_diff("ctrl", ctrl_a, ctrl_b) == "":
            raise Error(
                "check_hw_launch_invariance VACUOUS [" + _mode_name() + "]: a fit of a"
                " DIFFERENT series produced identical bits over " + String(len(ctrl_a))
                + " cells, so the comparisons below cannot see a difference and"
                " their passes mean nothing"
            )
        var bad = _first_diff(name + " A vs B", _all_bits(fa, fca), _all_bits(fb, fcb))
        if bad == "":
            bad = _first_diff(name + " A vs C (run twice)", _all_bits(fa, fca), _all_bits(fcc, fccf))
        # D: batch composition -- series 0, 3, 6 of the batch of 7 fitted
        # alone (batch of 1) and inside a batch of 512 (positions 0, 255, 511)
        if bad == "":
            var picks = [0, 3, 6]
            var at = [0, 255, 511]
            var big = List[Float32]()
            var filler = hw_fixture(sp, N, 512, FREQ, 11)
            for q in range(512):
                var src = -1
                for k in range(3):
                    if q == at[k]:
                        src = picks[k]
                for t in range(N):
                    if src >= 0:
                        big.append(d[src * N + t])
                    else:
                        big.append(filler[q * N + t])
            var fbig = _device_fit(ctx, big, N, 512, name, tr)
            var fcbig = holtwinters_forecast_host_traced(ctx, fbig, H, tr)
            # NEGATIVE CONTROL for the batch arm: the batch of 512 must
            # actually HOLD different series. Had the construction above
            # filled every row from one source, each slice comparison would
            # compare a row to an identical row and pass vacuously.
            var distinct = False
            for q in range(1, 8):
                if bitcast[DType.uint32](canon_nan_f32(fbig.sse[0])) != bitcast[DType.uint32](canon_nan_f32(fbig.sse[q])):
                    distinct = True
            if not distinct:
                raise Error(
                    "check_hw_launch_invariance VACUOUS [" + _mode_name() + "]: rows 1-7"
                    " of the batch of 512 all have the SSE of row 0, so the batch"
                    " does not hold distinct series and the slice comparisons are"
                    " comparing rows to themselves"
                )
            for k in range(3):
                var one = List[Float32]()
                for t in range(N):
                    one.append(d[picks[k] * N + t])
                var f1 = _device_fit(ctx, one, N, 1, name, tr)
                var fc1 = holtwinters_forecast_host_traced(ctx, f1, H, tr)
                # compare series picks[k] of A, at[k] of big, 0 of one
                var s7 = picks[k]
                var sb = at[k]
                var cells_a = List[Float32]()
                var cells_b = List[Float32]()
                var cells_1 = List[Float32]()
                cells_a.append(fa.sse[s7])
                cells_b.append(fbig.sse[sb])
                cells_1.append(f1.sse[0])
                cells_a.append(fa.alpha[s7])
                cells_b.append(fbig.alpha[sb])
                cells_1.append(f1.alpha[0])
                cells_a.append(fa.beta[s7])
                cells_b.append(fbig.beta[sb])
                cells_1.append(f1.beta[0])
                cells_a.append(fa.gamma[s7])
                cells_b.append(fbig.gamma[sb])
                cells_1.append(f1.gamma[0])
                for i in range(N - FREQ):
                    cells_a.append(fa.level[s7 + i * BATCH])
                    cells_b.append(fbig.level[sb + i * 512])
                    cells_1.append(f1.level[i])
                    cells_a.append(fa.season[s7 + i * BATCH])
                    cells_b.append(fbig.season[sb + i * 512])
                    cells_1.append(f1.season[i])
                    cells_a.append(fa.trend[s7 + i * BATCH])
                    cells_b.append(fbig.trend[sb + i * 512])
                    cells_1.append(f1.trend[i])
                for i in range(H):
                    cells_a.append(fca[s7 + i * BATCH])
                    cells_b.append(fcbig[sb + i * 512])
                    cells_1.append(fc1[i])
                for it in range(TRACE_ITERS):
                    for c in range(3):
                        cells_a.append(fa.iter_trace[(it * 3 + c) * BATCH + s7])
                        cells_b.append(fbig.iter_trace[(it * 3 + c) * 512 + sb])
                        cells_1.append(f1.iter_trace[(it * 3 + c) * 1 + 0])
                if fa.niter[s7] != fbig.niter[sb] or fa.niter[s7] != f1.niter[0]:
                    bad = name + " series " + String(s7) + " niter differs across batch sizes: " + String(fa.niter[s7]) + "/" + String(fbig.niter[sb]) + "/" + String(f1.niter[0])
                if bad == "":
                    bad = _first_diff(name + " series " + String(s7) + " batch 7 vs 512", cells_a, cells_b)
                if bad == "":
                    bad = _first_diff(name + " series " + String(s7) + " batch 7 vs 1", cells_a, cells_1)
                if bad != "":
                    break
        if bad != "":
            raise Error("check_hw_launch_invariance FAILED [" + _mode_name() + "] (sabotage " + hw_sabotage_name() + "): " + bad)
        n_ok += 1
    print(
        "check_hw_launch_invariance OK [" + _mode_name() + "]: " + String(n_ok)
        + " seasonal types byte-identical across tpb_decomp 32/256, tpb_optim 128/64, pad 0/37, two poisons, run twice, and series in batch 1 == batch 7 == batch 512 (every stage incl. per-iteration params)"
    )


def check_hw_signed_zero_clamp() raises:
    var neg0 = bitcast[DType.float32](UInt32(0x80000000))
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var nan2 = bitcast[DType.float32](UInt32(0xFFC0BEEF))
    var b0 = bound_device(neg0)
    if bitcast[DType.uint32](b0) != UInt32(0):
        raise Error("check_hw_signed_zero_clamp FAILED (sabotage " + hw_sabotage_name() + "): bound_device(-0.0) = " + hex32(b0) + ", expected 0x00000000")
    if bitcast[DType.uint32](bound_device(nan)) != UInt32(0) or bitcast[DType.uint32](bound_device(nan2)) != UInt32(0):
        raise Error("check_hw_signed_zero_clamp FAILED: bound_device(NaN) = " + hex32(bound_device(nan)) + " / " + hex32(bound_device(nan2)))
    if bound_device(hw_float32_inf()) != Float32(1.0) or bound_device(Float32(-5.0)) != Float32(0.0) or bound_device(Float32(0.5)) != Float32(0.5) or bound_device(Float32(1.5)) != Float32(1.0):
        raise Error("check_hw_signed_zero_clamp FAILED: bound_device values")
    # the device clamp: a direct eval with -0.0 and with +0.0 parameters
    var ctx = DeviceContext()
    var bs = 4
    var d = hw_fixture(spec_additive(), N, bs, FREQ, 21)
    var o = oracle_fit[DType.float32](d, N, bs, FREQ, START_PERIODS, SEASONAL_ADDITIVE, HW_DEFAULT_EPS, 0)
    var ts_h = List[Float32]()
    for t in range(N):
        for s in range(bs):
            ts_h.append(d[s * N + t])
    var ts_d = upload_f32(ctx, ts_h)
    var sl = upload_f32(ctx, o.start_level)
    var st = upload_f32(ctx, o.start_trend)
    var ss = upload_f32(ctx, o.start_season)
    var comps = (N - FREQ) * bs
    var results = List[List[Float32]]()
    for which in range(2):
        var pv = List[Float32]()
        for _ in range(bs):
            pv.append(neg0 if which == 0 else Float32(0.0))
        var al = upload_f32(ctx, pv)
        var be = upload_f32(ctx, pv)
        var ga = upload_f32(ctx, pv)
        var lv = ctx.enqueue_create_buffer[DType.float32](comps)
        var tv = ctx.enqueue_create_buffer[DType.float32](comps)
        var sv = ctx.enqueue_create_buffer[DType.float32](comps)
        var ev = ctx.enqueue_create_buffer[DType.float32](bs)
        ctx.synchronize()
        holtwinters_eval(ctx, ts_d, N, bs, FREQ, sl, st, ss, al, be, ga, lv, tv, sv, ev, SEASONAL_ADDITIVE)
        var vals = download_f32(ctx, ev, bs)
        for v in download_f32(ctx, lv, comps):
            vals.append(v)
        for v in download_f32(ctx, tv, comps):
            vals.append(v)
        for v in download_f32(ctx, sv, comps):
            vals.append(v)
        results.append(vals^)
        _ = al^
        _ = be^
        _ = ga^
        _ = lv^
        _ = tv^
        _ = sv^
        _ = ev^
    var bad = _first_diff("eval(-0.0) vs eval(+0.0)", results[0], results[1])
    if bad != "":
        raise Error("check_hw_signed_zero_clamp FAILED on the device (sabotage " + hw_sabotage_name() + "): " + bad)
    # and against the oracle at +0.0 (the clamp makes both +0.0)
    var ol = List[Float32]()
    var ot = List[Float32]()
    var osn = List[Float32]()
    for _ in range(comps):
        ol.append(Float32(0.0))
        ot.append(Float32(0.0))
        osn.append(Float32(0.0))
    var oall = List[Float32]()
    for s in range(bs):
        var ps = List[Float32]()
        for _ in range(FREQ):
            ps.append(Float32(0.0))
        oall.append(oracle_eval[DType.float32](s, o.ts, N, bs, FREQ, FREQ, o.start_level[s], o.start_trend[s], ps, o.start_season, neg0, neg0, neg0, True, True, ol, ot, osn))
    for v in ol:
        oall.append(v)
    for v in ot:
        oall.append(v)
    for v in osn:
        oall.append(v)
    var bad2 = _first_diff("eval(-0.0) device vs oracle", results[0], oall)
    if bad2 != "":
        comptime if IDENTICAL:
            raise Error("check_hw_signed_zero_clamp FAILED device vs oracle: " + bad2)
        else:
            print("  RECORDED [FAST] " + bad2)
    _ = ts_d^
    _ = sl^
    _ = st^
    _ = ss^
    # (c) THE RECORDED CLAMP: a direct optimizer launch on the ZERO series
    # (gradient exactly 0, so DEVIATION 662 returns the START parameters
    # unmoved) with start alpha = -0.0, beta = a payload NaN, gamma = -0.0:
    # the bounded parameters the kernel records must be +0.0 (0x00000000)
    # on every vendor. The HW_MAX_CLAMP sabotage spells the clamp
    # min(max(0.0, v), 1.0) and on Apple max(0.0, -0.0) is -0.0.
    var bsz = 3
    var zero_series = List[Float32]()
    for _ in range(N * bsz):
        zero_series.append(Float32(0.0))
    var zts = upload_f32(ctx, zero_series)
    var zstart = List[Float32]()
    for _ in range(bsz):
        zstart.append(Float32(0.0))
    var zseason = List[Float32]()
    for _ in range(FREQ * bsz):
        zseason.append(Float32(0.0))
    var zsl = upload_f32(ctx, zstart)
    var zst = upload_f32(ctx, zstart)
    var zss = upload_f32(ctx, zseason)
    var pa = List[Float32]()
    var pb = List[Float32]()
    var pg = List[Float32]()
    for _ in range(bsz):
        pa.append(neg0)
        pb.append(nan2)
        pg.append(neg0)
    var zal = upload_f32(ctx, pa)
    var zbe = upload_f32(ctx, pb)
    var zga = upload_f32(ctx, pg)
    var zcomps = (N - FREQ) * bsz
    var zlv = ctx.enqueue_create_buffer[DType.float32](zcomps)
    var ztv = ctx.enqueue_create_buffer[DType.float32](zcomps)
    var zsv = ctx.enqueue_create_buffer[DType.float32](zcomps)
    var zev = ctx.enqueue_create_buffer[DType.float32](bsz)
    var zcr = ctx.enqueue_create_buffer[DType.int32](bsz)
    var zni = ctx.enqueue_create_buffer[DType.int32](bsz)
    var ztr = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    holtwinters_optim(ctx, zts, N, bsz, FREQ, zsl, zst, zss, zal, zbe, zga, zlv, ztv, zsv, zev, zcr, zni, ztr, 0, HW_DEFAULT_EPS, SEASONAL_ADDITIVE)
    var ra = download_f32(ctx, zal, bsz)
    var rb = download_f32(ctx, zbe, bsz)
    var rg = download_f32(ctx, zga, bsz)
    var rsse = download_f32(ctx, zev, bsz)
    var got = hex32(ra[0]) + "/" + hex32(rb[0]) + "/" + hex32(rg[0])
    for s in range(bsz):
        if bitcast[DType.uint32](ra[s]) != UInt32(0) or bitcast[DType.uint32](rb[s]) != UInt32(0) or bitcast[DType.uint32](rg[s]) != UInt32(0):
            raise Error("check_hw_signed_zero_clamp FAILED at the RECORDED clamp (sabotage " + hw_sabotage_name() + "): start (-0.0, NaN 0xffc0beef, -0.0) on the zero series bounded to " + hex32(ra[s]) + "/" + hex32(rb[s]) + "/" + hex32(rg[s]) + ", expected 0x00000000 x3")
        if bitcast[DType.uint32](rsse[s]) != UInt32(0):
            raise Error("check_hw_signed_zero_clamp: zero series SSE " + hex32(rsse[s]))
    _ = zts^
    _ = zsl^
    _ = zst^
    _ = zss^
    _ = zal^
    _ = zbe^
    _ = zga^
    _ = zlv^
    _ = ztv^
    _ = zsv^
    _ = zev^
    _ = zcr^
    _ = zni^
    _ = ztr^
    print("check_hw_signed_zero_clamp OK [" + _mode_name() + "]: host bound(-0.0)=" + hex32(b0) + ", bound(NaN)=+0.0, bound(inf)=1; device eval at alpha=beta=gamma=-0.0 == eval at +0.0 bit for bit (" + String(len(results[0])) + " cells), == oracle" + ("" if bad2 == "" else " (RECORDED)") + "; RECORDED clamp: start (-0.0, NaN, -0.0) on the zero series -> " + got)


def check_hw_zero_series_keeps_start() raises:
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var bs = 3
    var d = hw_fixture(spec_zero(), N, bs, FREQ, 0)
    var f = _device_fit(ctx, d, N, bs, "additive", tr)
    var o = _oracle32(d, N, bs, SEASONAL_ADDITIVE)
    var it0 = hex32(f.iter_trace[0]) + "/" + hex32(f.iter_trace[bs]) + "/" + hex32(f.iter_trace[2 * bs])
    for s in range(bs):
        var ok = (
            bitcast[DType.uint32](f.alpha[s]) == bitcast[DType.uint32](HW_ALPHA0)
            and bitcast[DType.uint32](f.beta[s]) == bitcast[DType.uint32](HW_BETA0)
            and bitcast[DType.uint32](f.gamma[s]) == bitcast[DType.uint32](HW_GAMMA0)
            and Int(f.criterion[s]) == OPTIM_MIN_GRAD_NORM
            and Int(f.niter[s]) == 0
            and bitcast[DType.uint32](f.sse[s]) == UInt32(0)
        )
        if not ok:
            raise Error(
                "check_hw_zero_series_keeps_start FAILED (sabotage " + hw_sabotage_name() + "): series " + String(s)
                + " params " + String(f.alpha[s]) + "/" + String(f.beta[s]) + "/" + String(f.gamma[s])
                + " (" + hex32(f.alpha[s]) + ") criterion " + criterion_name(Int(f.criterion[s])) + " niter " + String(f.niter[s])
                + " sse " + hex32(f.sse[s]) + " iter000 " + it0
            )
        if o.alpha[s] != HW_ALPHA0 or o.criterion[s] != OPTIM_MIN_GRAD_NORM or o.niter[s] != 0:
            raise Error("check_hw_zero_series_keeps_start: ORACLE series " + String(s) + " params " + String(o.alpha[s]) + " criterion " + String(o.criterion[s]))
    # the level array is the start level (0) everywhere, trend 0, season 0
    for v in f.level:
        if bitcast[DType.uint32](v) != UInt32(0):
            raise Error("check_hw_zero_series_keeps_start: nonzero level " + hex32(v))
    print("check_hw_zero_series_keeps_start OK [" + _mode_name() + "]: all-zero additive series keeps (0.4, 0.3, 0.3), OPTIM_MIN_GRAD_NORM, niter 0, SSE 0 on device and oracle; iter000 slots " + it0 + " (unwritten fill)")


def check_hw_card_is_emitted() raises:
    var ctx = DeviceContext()
    var d = hw_fixture(spec_additive(), N, BATCH, FREQ, 2)
    var p1 = SCRATCH + "/mojolearn_hw_card_1.card"
    var p2 = SCRATCH + "/mojolearn_hw_card_2.card"
    var t1 = IdentityTrace.to_path(p1)
    t1.header("hw card 1")
    var f1 = _device_fit(ctx, d, N, BATCH, "additive", t1)
    _ = holtwinters_forecast_host_traced(ctx, f1, H, t1)
    var t2 = IdentityTrace.to_path(p2)
    t2.header("hw card 2")
    var f2 = _device_fit(ctx, d, N, BATCH, "additive", t2)
    _ = holtwinters_forecast_host_traced(ctx, f2, H, t2)
    var div = first_divergence(p1, p2)
    if div != "":
        raise Error("check_hw_card_is_emitted: two runs of one fixture diverge: " + div)
    var max_iter = 0
    for v in f1.niter:
        if Int(v) > max_iter:
            max_iter = Int(v)
    var expect = List[String]()
    expect.append("hw.input")
    expect.append("hw.decomp.trend")
    expect.append("hw.decomp.season")
    expect.append("hw.start.level")
    expect.append("hw.start.trend")
    expect.append("hw.start.season")
    var shown = max_iter if max_iter < TRACE_ITERS else TRACE_ITERS
    for it in range(shown):
        var s = String(it)
        while s.byte_length() < 3:
            s = "0" + s
        expect.append("hw.opt.iter" + s + ".params")
    expect.append("hw.opt.niter")
    expect.append("hw.opt.criterion")
    expect.append("hw.params")
    expect.append("hw.sse")
    expect.append("hw.level")
    expect.append("hw.trend")
    expect.append("hw.season")
    expect.append("hw.forecast")
    var lines = List[String]()
    with open(p1, "r") as fh:
        var text = fh.read()
        for ln in text.split("\n"):
            var s = String(ln)
            if s != "" and not s.startswith("#"):
                lines.append(s)
    if len(lines) != len(expect):
        raise Error("check_hw_card_is_emitted: " + String(len(lines)) + " records, expected " + String(len(expect)))
    for i in range(len(expect)):
        if lines[i].find(expect[i]) < 0:
            raise Error("check_hw_card_is_emitted: record " + String(i) + " is '" + lines[i] + "', expected tag " + expect[i])
    print("check_hw_card_is_emitted OK [" + _mode_name() + "]: " + String(len(expect)) + " stages (" + expect[0] + " ... " + String(shown) + " iteration stages ... hw.forecast), run-to-run control identical")


def main() raises:
    print("== holtwinters/mojo_only/hw_check.mojo [" + _mode_name() + "] sabotage=" + hw_sabotage_name() + " ==")
    check_hw_refusals()
    check_hw_decompose_vs_reference()
    check_hw_optimizer_reduces_sse()
    check_hw_forecast_continues_pattern()
    check_hw_signed_zero_clamp()
    check_hw_zero_series_keeps_start()
    check_hw_device_equals_oracle()
    check_hw_launch_invariance()
    check_hw_card_is_emitted()
    print("== hw_check: ALL OK [" + _mode_name() + "] ==")
