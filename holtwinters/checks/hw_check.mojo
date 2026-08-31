# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Holt-Winters: refusals, oracle sanity, device identity, launch
invariance, signed zero, NaN, reach, and the packed pointer surface.

DEVIATIONS 660-665, 697-699 and 930's gates. (The sentence this replaced
said "DEVIATIONS 660-665's gates" and listed nine checks; DEVIATION 699's
`check_hw_decision_branches` had been added to `main` without being added
here, and DEVIATION 930's `check_hw_pack_round_trip` is new.) The checks,
in order:

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
    check_hw_decision_branches         DEVIATION 699: the decision mask is
                                       the oracle's on five fixtures; the
                                       REACH CENSUS of which branch bits the
                                       standard fixtures set, and on how many
                                       of the 35 series each, so a bit held up
                                       by ONE series prints as THIN
                                       (DEVIATION 962); a bounded 128-fit
                                       search for LS_LIMIT and RHO_ZERO whose
                                       result is printed BEFORE anything can
                                       raise, feeding the RHO_ZERO fixture two
                                       named numbers rather than one packed
                                       integer a human decodes (DEVIATION
                                       960, which is how this gate came to
                                       abort on its own reach assertion from
                                       79b238d until it was repaired);
                                       bfgs_iter_limit = 2 for ITER_LIMIT and
                                       linesearch_iter_limit = 1 for LS_LIMIT,
                                       both now ASSERTED and not merely
                                       claimed (DEVIATION 961)
    check_hw_pack_round_trip           DEVIATION 930, and the only gate that
                                       BREAKS THE ROUND TRIP: what
                                       `holtwinters_fit_ptr` packs into one
                                       flat buffer for `bindings/` is
                                       asserted cell by cell, BITWISE,
                                       against `holtwinters_fit_host`'s
                                       structured level / trend / season /
                                       stats / flags, which never saw the
                                       pack; each block is read the way
                                       `_tsa_impl.py` reads it and compared
                                       against series s fitted ALONE, which
                                       is what pins TIME-MAJOR; and
                                       `holtwinters_forecast_ptr` on the flat
                                       buffer is asserted against
                                       `holtwinters_forecast_host` on the
                                       structured `HWFit`. A self-consistent
                                       pack mistake round-trips through every
                                       end-to-end test; only a comparison
                                       against a witness that never saw the
                                       pack can see it. ASSERTS IN BOTH
                                       MODES: nothing here is a numerics
                                       question

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

THE PACK ARMS (DEVIATION 930) are the same `-D MOJOLEARN_HW_SABOTAGE_<NAME>=1`
switch, but they are declared in THIS file rather than on the switchboard in
`holtwinters/impl/holtwinters/internal/hw_utils.mojo`, because they do not
sabotage a kernel: they corrupt the flat buffer after the pointer entry wrote
it, which is byte for byte what a wrongly spelled writer would have left, and
the buffer is all the assertions can see. Each must break the clauses named
beside it and NO others; a run prints a `CLAUSES ...=BROKE/held` line whenever
any arm is named, so the table is checkable rather than asserted.
    PACK_SWAP_TREND_SEASON  the trend and season blocks exchanged. Breaks
                            (a) comps and (e) time-major. Clause (f) HOLDS,
                            deliberately: the arm hands the forecast the
                            pristine buffer, which is the dangerous defect
                            drawn exactly -- writer and reader wrong the same
                            way, forecast still round-tripping
    PACK_SWAP_LEVEL_TREND   the level and trend blocks exchanged. Same two
    PACK_SERIES_MAJOR       each block written series-major, `s * rows + i`,
                            instead of time-major, `s + i * batch_size`.
                            Same two
    PACK_STRIDE_BATCH       `batch_size` used as the block stride where
                            `components_len` belongs. Same two
    PACK_SWAP_SSE_ALPHA     the stats pack's first two blocks exchanged.
                            Breaks (c) stats only
    PACK_SWAP_BETA_GAMMA    the stats pack's last two blocks exchanged.
                            Breaks (c) stats only
    PACK_SWAP_FLAGS         niter and criterion exchanged. Breaks (d) flags
                            only
    PACK_FORECAST_SWAP      the FORECAST side only reads the buffer with
                            trend and season exchanged, the fit clauses
                            seeing the pristine one. Breaks (f) forecast
                            only: a writer and a reader that disagree

Run:

    tools/with_build_lock.sh     pixi run mojo run -I . holtwinters/checks/hw_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . holtwinters/checks/hw_check.mojo

and one sabotage arm, which must FAIL:

    tools/with_identical_mode.sh pixi run mojo run -I . \
        -D MOJOLEARN_HW_SABOTAGE_PACK_SWAP_TREND_SEASON=1 \
        holtwinters/checks/hw_check.mojo
"""

from std.memory import bitcast
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, first_divergence, read_trace_lines
from holtwinters.estimator import (
    HWFit,
    download_f32,
    download_i32,
    holtwinters_fit_host,
    holtwinters_fit_host_traced,
    holtwinters_fit_ptr,
    holtwinters_forecast_host,
    holtwinters_forecast_host_traced,
    holtwinters_forecast_ptr,
    upload_f32,
)
from holtwinters.checks.hw_fixture import (
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
from holtwinters.checks.hw_oracle import (
    HWOracleFit,
    oracle_eval,
    oracle_fit,
    oracle_forecast,
    oracle_sse_at,
)
from holtwinters.impl.holtwinters.internal.hw_decompose import host_filter, host_r1qt
from holtwinters.impl.holtwinters.internal.hw_utils import (
    HW_OPTIM_TPB,
    bound_device,
    hw_float32_inf,
    hw_sabotage_name,
)
from holtwinters.impl.holtwinters.runner import (
    HW_ALPHA0,
    HW_BETA0,
    HW_GAMMA0,
    HW_DEFAULT_EPS,
    holtwinters_eval,
    holtwinters_optim,
)
from holtwinters.impl.tsa.holtwinters_params import (
    OPTIM_MIN_GRAD_NORM,
    SEASONAL_ADDITIVE,
    SEASONAL_MULTIPLICATIVE,
    criterion_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from solver.checks.record_canon import canon_nan_f32, canon_nan_list

comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime SCRATCH = "/tmp"

comptime N = 72
comptime FREQ = 12
comptime START_PERIODS = 2
comptime BATCH = 7
comptime TRACE_ITERS = 64
comptime H = 24

# THE PACK FIXTURE (DEVIATION 930), separate from the six above and chosen
# so that no index arithmetic error can alias onto a correct answer:
# `batch_size`, `frequency`, the per-series row count `n - frequency` and
# `components_len = (n - frequency) * batch_size` are PAIRWISE DISTINCT (3,
# 5, 15, 45), and so is every other extent the pack uses (h = 7,
# 3*components_len = 135, 4*batch_size = 12, 2*batch_size = 6, h*batch_size
# = 21). `check_hw_pack_round_trip` ASSERTS that rather than leaving a
# reader to re-derive it. The standard fixture's extents would separate too
# (N = 72, FREQ = 12, BATCH = 7 gives 420 and 60, distinct from 7 and 12),
# so this is not a fixture the standard one could not have been; it is a
# smaller one, because the gate runs four extra fits and three forecasts,
# and a separate one, because the pack's separation argument should not
# silently depend on a fixture chosen for a different clause.
comptime PACK_N = 20
comptime PACK_FREQ = 5
comptime PACK_BATCH = 3
comptime PACK_H = 7
comptime PACK_SALT = 930
#: cells past the end of every packed buffer, poisoned and asserted
#: untouched, so an over-striding writer is a failure and not a silent
#: overwrite of whatever numpy put next in memory.
comptime PACK_GUARD = 16
comptime PACK_POISON = Float32(-424242.0)
comptime PACK_POISON_I = Int32(-424242)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


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
    checks.append(_first_diff_i(tag + " hw.opt.decisions", f.decisions, o.decisions))
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
    var dec = List[Int32]()
    for v in o.decisions:
        dec.append(Int32(v))
    t.record_list_i32("hw.opt.decisions", dec)
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
        # `mixed` runs the decomposition AND optimizer at tpb 4 (two blocks
        # for a batch of 7); the FORECAST kernel had no such fixture and ran
        # at one block on all six, so a launch-geometry defect in it was
        # compared against the oracle by nothing. tpb 4 here puts it on two
        # blocks. (ROTATE_CONV showed what a one-block fixture hides: the
        # rotation is the identity when block_idx.x is always 0.)
        var fc = holtwinters_forecast_host_traced(ctx, f, H, dt, 4 if name == "mixed" else -1)
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
    var zde = ctx.enqueue_create_buffer[DType.int32](bsz)
    zde.enqueue_fill(Int32(-7))
    ctx.synchronize()
    holtwinters_optim(ctx, zts, N, bsz, FREQ, zsl, zst, zss, zal, zbe, zga, zlv, ztv, zsv, zev, zcr, zni, zde, ztr, 0, HW_DEFAULT_EPS, SEASONAL_ADDITIVE)
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
    expect.append("hw.opt.decisions")
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


def _dec_bit_names() -> List[String]:
    """DEVIATION 699's mask, bit 0 first.

    ONE list. `_dec_names` spells a mask with it and the reach tally counts
    with it, so a bit added to the optimizer cannot be spelled in one place
    and forgotten in the other -- which is the shape of the defect DEVIATION
    960 repaired one function below.
    """
    var out = List[String]()
    out.append(String("ZERO_DIR"))
    out.append(String("HESSIAN_RESET"))
    out.append(String("LS_LIMIT"))
    out.append(String("RHO_ZERO"))
    out.append(String("H_NAN"))
    out.append(String("BOTH_CRIT"))
    out.append(String("ITER_LIMIT"))
    out.append(String("ALPHA_LO"))
    out.append(String("ALPHA_HI"))
    out.append(String("BETA_LO"))
    out.append(String("BETA_HI"))
    out.append(String("GAMMA_LO"))
    out.append(String("GAMMA_HI"))
    return out^


def _dec_names(d: Int) -> String:
    """DEVIATION 699's mask, spelled out."""
    var nm = _dec_bit_names()
    var out = String("")
    for b in range(len(nm)):
        if (d & (1 << b)) != 0:
            out += nm[b] + " "
    var halv = d >> 16
    if out == "":
        out = "(none) "
    return out + "halvings=" + String(halv)


#: DEVIATION 960: the RHO_ZERO fixture the last RECORDED run of the census
#: selected, as two numbers in the units the fixture actually takes -- the
#: `hw_fixture` salt, and 0 for additive / 1 for multiplicative. `-1` means
#: no run has recorded one yet, and the census prints the pair it selects so
#: it can be pasted here. It is a WITNESS, not a filter: the census never
#: fits only what is pinned here, it fits what the search finds and says out
#: loud whether that still matches the pin. Never edit these to make a run
#: pass; the point of the pair is that a change in WHICH series reaches the
#: branch is a change in the port, and has to be read before it is recorded.
# MEASURED 2026-08-24, and the drift detector found it on its FIRST use.
# THE PIN IS THE IDENTICAL ARM'S WITNESS AND IT CANNOT SERVE BOTH MODES.
#
#   IDENTICAL   RHO_ZERO first at salt 102 multiplicative, 12/128 fits
#   FAST        RHO_ZERO first at salt 119 multiplicative
#
# Both arms are ALL OK and the branch is covered in both. They disagree about
# WHICH series gets there, and that is not a defect: `rho_ == 0.0` fires when
# two consecutive gradients are BITWISE EQUAL, which is a plateau event, and
# FAST and IDENTICAL are different arithmetic, so they plateau on different
# series. A single pin therefore reports DRIFTED under FAST by construction,
# and that message is information rather than a failure. Read it; do not
# silence it by widening the pin to accept both, because then a genuine drift
# inside one arm would have nowhere to show up.
comptime HW_RHO_PIN_SALT = 102
comptime HW_RHO_PIN_KIND = 1


def check_hw_decision_branches() raises:
    """DEVIATION 699's gate, and the REACH CENSUS the lane could not take
    before it.

    Four jobs. (1) Print which decision bits the five standard fixtures
    actually set, and on HOW MANY of the 35 series each, so `LS_LIMIT`,
    `RHO_ZERO`, `ITER_LIMIT` and the clamp arms stop being branches nobody
    can say are reached -- and so a bit reached by exactly ONE series is
    named THIN while it is still covered, rather than after it goes vacuous.
    (1b) Search a bounded family of natural series for the two bits the
    standard fixtures do not set. (1c) Re-fit the series the search names
    for `RHO_ZERO` and assert the branch is genuinely taken, by the device
    AND by the host oracle, with the whole mask compared. (2) Reach the two
    LIMIT branches deliberately, at the smallest size that reaches them,
    using cuML's own `bfgs_iter_limit` / `linesearch_iter_limit` (their
    override block, `runner.cuh:236-253`) rather than hunting for a series
    that fails to converge in 1000 iterations.

    DEVIATION 960 (2026-08-24). Jobs 1b and 1c used to be joined by a HUMAN:
    the search printed its hit as one packed integer, `salt * 10 + kind`, and
    the fixture below was pinned by hand from that print as `multiplicative,
    salt 102`. That pin sets `RHO_ZERO` on NO series, so from 79b238d
    onwards this gate aborted on its own reach assertion, on the committed
    tree, in both modes -- and once `check_hw_pack_round_trip` was added
    after it in `main`, that gate could never run at all. The census caught
    its own coverage going vacuous, which is what it is for; what it could
    not do was tell anyone WHICH series to pin instead, because the search
    line was built before the raise and printed after it. Both are fixed
    here: the search carries the salt and the kind as two named numbers that
    the fixture consumes directly, and it prints before anything can raise.
    """
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var names = ["additive", "multiplicative", "constant", "zero", "mixed"]
    var seen = 0
    var max_halv = 0
    # DEVIATION 962: the UNION of bits is blind to thinness. A bit set by one
    # series in 35 is covered by an accident that the next arithmetic change
    # takes away, and it is indistinguishable in `seen` from a bit set by all
    # 35. Tally per bit and name the ones at 1.
    var bit_names = _dec_bit_names()
    var bit_hits = List[Int]()
    for _ in range(len(bit_names)):
        bit_hits.append(0)
    var n_std_series = 0
    for fi in range(len(names)):
        var name = names[fi]
        var seasonal = SEASONAL_ADDITIVE
        var data: List[Float32]
        if name == "additive":
            data = hw_fixture(spec_additive(), N, BATCH, FREQ, 2)
        elif name == "multiplicative":
            seasonal = SEASONAL_MULTIPLICATIVE
            data = hw_fixture(spec_multiplicative(), N, BATCH, FREQ, 2)
        elif name == "constant":
            data = hw_fixture(spec_constant(), N, BATCH, FREQ, 2)
        elif name == "zero":
            data = hw_fixture(spec_zero(), N, BATCH, FREQ, 2)
        else:
            var specs: List[HWFixtureSpec] = [spec_additive(), spec_constant(), spec_additive_noiseless(), spec_zero()]
            data = hw_fixture_mixed(specs, N, BATCH, FREQ, 6)
        var f = _device_fit(ctx, data, N, BATCH, _seasonal_str(seasonal), tr)
        var o = oracle_fit[DType.float32](data, N, BATCH, FREQ, START_PERIODS, seasonal, HW_DEFAULT_EPS, TRACE_ITERS)
        var bad = _first_diff_i(name + " hw.opt.decisions", f.decisions, o.decisions)
        if bad != "":
            comptime if IDENTICAL:
                raise Error("check_hw_decision_branches FAILED (sabotage " + hw_sabotage_name() + "): " + bad)
            else:
                print("  RECORDED [FAST] " + bad)
        for s in range(BATCH):
            # OR the FLAG bits only. ORing the packed halving COUNT would
            # produce a number that is not any fixture's count (the first
            # run of this census printed halvings=255, which is the OR of
            # 145, 112 and 68 and means nothing); the max is the useful one.
            seen |= Int(f.decisions[s]) & 0xFFFF
            var hv = Int(f.decisions[s]) >> 16
            if hv > max_halv:
                max_halv = hv
            for b in range(len(bit_names)):
                if (Int(f.decisions[s]) & (1 << b)) != 0:
                    bit_hits[b] += 1
            n_std_series += 1
        print("  census " + name + " s0: " + _dec_names(Int(f.decisions[0])))

    # (1a) HOW THIN each bit's coverage is, in series out of `n_std_series`.
    # DEVIATION 962. Printed, not asserted: the standard fixtures are pinned
    # for the identity comparison, not chosen to reach bits, so a bit falling
    # to zero here is a fact to read rather than a failure. A bit at exactly 1
    # is named THIN, because that is what RHO_ZERO's own pinned fixture was
    # claimed to be (1 series of 7) right before it turned out to be 0.
    var tally = String("")
    var thin = String("")
    for b in range(len(bit_names)):
        if b > 0:
            tally += " "
        tally += bit_names[b] + "=" + String(bit_hits[b])
        if bit_hits[b] == 1:
            thin += bit_names[b] + " "
    print("  census reach over " + String(n_std_series) + " standard series: " + tally)
    if thin != "":
        print("  census THIN (one series only, a bit away from vacuous): " + thin)

    # (1b) A BOUNDED REACH SEARCH for the two bits the standard fixtures do
    # NOT set: LS_LIMIT (bit 2, cuml#888's path) and RHO_ZERO (bit 3, the
    # second NaN route). Both need a pathological series rather than a
    # pathological parameter, so the honest way to look is to LOOK: 64
    # hashed salts x two seasonal types, all at the standard small size.
    # This is a reach search, not a sweep and not a benchmark -- it either
    # names a series for (1c) to re-fit, or it records, with a number, that
    # 128 tries did not reach the branch.
    #
    # DEVIATION 960: the salt and the seasonal kind are carried as TWO NAMED
    # numbers in the units `hw_fixture` and `_seasonal_str` take, and (1c)
    # consumes them directly. They used to be squashed into one `salt * 10 +
    # kind` integer that a human decoded into a hand-written fixture, and the
    # decode was wrong. Nothing between the search and the fixture is read by
    # eye now. The COUNTS matter as much as the first hit: `1/128` is a bit
    # that will go vacuous the next time the arithmetic moves, `40/128` will
    # not, and the difference is invisible in a first-hit-only search.
    var rho_salt = -1
    var rho_kind = -1
    var rho_fits = 0
    var rho_series = 0
    var ls_salt = -1
    var ls_kind = -1
    var ls_fits = 0
    for salt in range(64):
        for k in range(2):
            var sp2 = spec_additive() if k == 0 else spec_multiplicative()
            var seas2 = SEASONAL_ADDITIVE if k == 0 else SEASONAL_MULTIPLICATIVE
            var dsc = hw_fixture(sp2, N, BATCH, FREQ, salt + 100)
            var fsc = _device_fit(ctx, dsc, N, BATCH, _seasonal_str(seas2), tr)
            var hit_ls = 0
            var hit_rho = 0
            for s in range(BATCH):
                var dv = Int(fsc.decisions[s])
                if (dv & 4) != 0:
                    hit_ls += 1
                if (dv & 8) != 0:
                    hit_rho += 1
            if hit_ls > 0:
                ls_fits += 1
                if ls_salt < 0:
                    ls_salt = salt + 100
                    ls_kind = k
            if hit_rho > 0:
                rho_fits += 1
                rho_series += hit_rho
                if rho_salt < 0:
                    rho_salt = salt + 100
                    rho_kind = k
    # PRINTED HERE, before anything below can raise. The previous spelling
    # built this line first and printed it last, so the one run that needed
    # it -- the run where the pinned fixture stopped reaching -- was the one
    # run that never saw it.
    var ls_kind_str = _seasonal_str(SEASONAL_ADDITIVE) if ls_kind == 0 else _seasonal_str(SEASONAL_MULTIPLICATIVE)
    var rho_kind_str = _seasonal_str(SEASONAL_ADDITIVE) if rho_kind == 0 else _seasonal_str(SEASONAL_MULTIPLICATIVE)
    var srch = String("  reach search 128 fits (64 hashed salts, seed = salt, x additive and multiplicative): LS_LIMIT ")
    srch += ("REACHED, first at salt " + String(ls_salt) + " " + ls_kind_str + ", " + String(ls_fits) + "/128 fits") if ls_salt >= 0 else "NOT REACHED in 128 fits"
    srch += "; RHO_ZERO "
    srch += ("REACHED, first at salt " + String(rho_salt) + " " + rho_kind_str + ", " + String(rho_fits) + "/128 fits, " + String(rho_series) + " series") if rho_salt >= 0 else "NOT REACHED in 128 fits"
    print(srch)
    if ls_salt >= 0:
        print(
            "  NOTE: LS_LIMIT is now REACHED BY A NATURAL SERIES on DEFAULT"
            " limits, which is what holtwinters/README.md's OWED item 1 says"
            " 128 fits could not do. Read that row before trusting it."
        )

    # (1c) THE RHO_ZERO FIXTURE, re-fitted from what the search just named.
    # NOT_IMPLEMENTED.tsv ARGUES that the second NaN route "terminates
    # deterministically on every vendor"; this is the fixture that exercises
    # it instead. Asserted four ways, because a reach claim is worth nothing
    # unless the branch is shown to be taken rather than assumed:
    #   the search's hit must REPRODUCE on a fresh fit of the same fixture
    #     (two fits of one fixture that disagree is not a fixture problem,
    #      it is a determinism problem, and it gets its own message);
    #   the DEVICE must set the bit on at least one series;
    #   the HOST ORACLE, which shares no line of code with the kernel, must
    #     set it on the SAME COUNT of series, so the branch is a property of
    #     the algorithm and not of one kernel's rounding;
    #   the whole decision mask must equal the oracle's, cell for cell.
    # The last two are IDENTICAL-only raises for the usual reason: under FAST
    # the two are allowed to diverge, and that divergence is the recorded
    # result, not a failure.
    if rho_salt < 0:
        raise Error(
            "check_hw_decision_branches REACH FAILURE: RHO_ZERO (the second NaN"
            " route) is set by NO series in 128 natural fits (64 hashed salts x"
            " additive and multiplicative at n=" + String(N) + ", batch "
            + String(BATCH) + "), so the branch is UNCOVERED. Do not hand-pin a"
            " fixture to make this line go away. Either a reaching series exists"
            " and the search has to be widened, or the route is unreachable on"
            " this arithmetic and NOT_IMPLEMENTED.tsv's claim that it 'terminates"
            " deterministically on every vendor' has to be rewritten as an"
            " unreachable-branch row -- an acknowledged gap beats a census that"
            " pretends to cover it."
        )
    var pin_line = String("  RHO_ZERO pin: ")
    if HW_RHO_PIN_SALT < 0:
        pin_line += (
            "UNRECORDED. The census selects salt " + String(rho_salt) + " "
            + rho_kind_str + "; put that pair in HW_RHO_PIN_SALT / HW_RHO_PIN_KIND"
            " so the NEXT run can tell drift from agreement"
        )
    elif HW_RHO_PIN_SALT == rho_salt and HW_RHO_PIN_KIND == rho_kind:
        pin_line += "CONFIRMED at salt " + String(rho_salt) + " " + rho_kind_str
    else:
        pin_line += (
            "DRIFTED. The recorded witness is salt " + String(HW_RHO_PIN_SALT)
            + " kind " + String(HW_RHO_PIN_KIND) + ", the census now selects salt "
            + String(rho_salt) + " " + rho_kind_str + ". The branch is still"
            " covered, by a DIFFERENT series, and WHICH series reaches a NaN"
            " route is a fact about the port -- read it before recording it"
        )
    print(pin_line)
    var rho_spec = spec_additive() if rho_kind == 0 else spec_multiplicative()
    var rho_seasonal = SEASONAL_ADDITIVE if rho_kind == 0 else SEASONAL_MULTIPLICATIVE
    var d_rho = hw_fixture(rho_spec, N, BATCH, FREQ, rho_salt)
    var f_rho = _device_fit(ctx, d_rho, N, BATCH, rho_kind_str, tr)
    var o_rho = oracle_fit[DType.float32](d_rho, N, BATCH, FREQ, START_PERIODS, rho_seasonal, HW_DEFAULT_EPS, TRACE_ITERS)
    var n_rho = 0
    var n_rho_oracle = 0
    var n_rho_hnan = 0
    for s in range(BATCH):
        if (Int(f_rho.decisions[s]) & 8) != 0:
            n_rho += 1
            if (Int(f_rho.decisions[s]) & 16) != 0:
                n_rho_hnan += 1
        if (o_rho.decisions[s] & 8) != 0:
            n_rho_oracle += 1
    if n_rho == 0:
        raise Error(
            "check_hw_decision_branches NONDETERMINISM: the reach search set"
            " RHO_ZERO on the fixture at salt " + String(rho_salt) + " "
            + rho_kind_str + ", and re-fitting that same fixture in the same"
            " process sets it on no series. Two fits of one fixture must agree"
            " bit for bit; the suspect is the optimizer's run-to-run"
            " determinism, not the fixture."
        )
    if n_rho_oracle != n_rho:
        var oracle_msg = String(
            "check_hw_decision_branches: RHO_ZERO is set by the device on "
        ) + String(n_rho) + " series and by the HOST ORACLE on " + String(n_rho_oracle) + " at salt " + String(rho_salt) + " " + rho_kind_str + " (sabotage " + hw_sabotage_name() + "). The oracle shares no code with the kernel, so the branch is only really reached when both take it."
        comptime if IDENTICAL:
            raise Error(oracle_msg)
        else:
            print("  RECORDED [FAST] " + oracle_msg)
    var bad_rho = _first_diff_i("rho_zero hw.opt.decisions", f_rho.decisions, o_rho.decisions)
    if bad_rho != "":
        comptime if IDENTICAL:
            raise Error("check_hw_decision_branches FAILED on the RHO_ZERO fixture (sabotage " + hw_sabotage_name() + "): " + bad_rho)
        else:
            print("  RECORDED [FAST] " + bad_rho)
    # H_NAN rides along, and whether it does is arithmetic, not luck: `rho_ ==
    # 0` makes `rho = 1/0 = +-inf`, and every H entry then takes an `inf * 0`
    # or an `inf - inf`. It is NOT asserted, because `inf - (-inf)` stays inf
    # and `H11 == H11` is true of an infinity, so the implication is a
    # near-certainty rather than a theorem. Printed instead, on every run,
    # because if it holds then README.md's OWED item 4 ("H_NAN remains
    # unreached by any natural fixture") is false and has to go.
    print(
        "  RHO_ZERO fixture: device " + String(n_rho) + "/" + String(BATCH)
        + " series, oracle " + String(n_rho_oracle) + "/" + String(BATCH)
        + ", H_NAN co-reached on " + String(n_rho_hnan) + " of those "
        + String(n_rho) + ("; mask == oracle" if IDENTICAL else "; mask vs oracle RECORDED, not asserted [FAST]")
    )

    # (2) the two LIMIT branches, reached exactly.
    var d2 = hw_fixture(spec_additive(), N, BATCH, FREQ, 2)
    var ts_h2 = List[Float32]()
    for t in range(N):
        for s in range(BATCH):
            ts_h2.append(d2[s * N + t])
    var ts_d = upload_f32(ctx, ts_h2)
    var sl = ctx.enqueue_create_buffer[DType.float32](BATCH)
    var st = ctx.enqueue_create_buffer[DType.float32](BATCH)
    var ss = ctx.enqueue_create_buffer[DType.float32](FREQ * BATCH)
    var comps = (N - FREQ) * BATCH
    var lv = ctx.enqueue_create_buffer[DType.float32](comps)
    var tv = ctx.enqueue_create_buffer[DType.float32](comps)
    var sv = ctx.enqueue_create_buffer[DType.float32](comps)
    var ev = ctx.enqueue_create_buffer[DType.float32](BATCH)
    var cr = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var ni = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var de = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var itr = ctx.enqueue_create_buffer[DType.float32](1)
    # the decomposition's start values, from the oracle (a host quantity)
    var ob = oracle_fit[DType.float32](d2, N, BATCH, FREQ, START_PERIODS, SEASONAL_ADDITIVE, HW_DEFAULT_EPS, 0, True, 2, 1)
    var h_sl = List[Float32]()
    var h_st = List[Float32]()
    for s in range(BATCH):
        h_sl.append(ob.start_level[s])
        h_st.append(ob.start_trend[s])
    var h_ss = List[Float32]()
    for i in range(FREQ * BATCH):
        h_ss.append(ob.start_season[i])
    var sl2 = upload_f32(ctx, h_sl)
    var st2 = upload_f32(ctx, h_st)
    var ss2 = upload_f32(ctx, h_ss)
    var a0 = List[Float32]()
    var b0 = List[Float32]()
    var g0 = List[Float32]()
    for _ in range(BATCH):
        a0.append(HW_ALPHA0)
        b0.append(HW_BETA0)
        g0.append(HW_GAMMA0)
    var al = upload_f32(ctx, a0)
    var be = upload_f32(ctx, b0)
    var ga = upload_f32(ctx, g0)
    ctx.synchronize()
    holtwinters_optim(
        ctx, ts_d, N, BATCH, FREQ, sl2, st2, ss2, al, be, ga, lv, tv, sv, ev,
        cr, ni, de, itr, 0, HW_DEFAULT_EPS, SEASONAL_ADDITIVE, HW_OPTIM_TPB, 2, 1,
    )
    var dd = download_i32(ctx, de, BATCH)
    var cc = download_i32(ctx, cr, BATCH)
    var n_limit = 0
    var n_ls_limit = 0
    for s in range(BATCH):
        if (Int(dd[s]) & 64) != 0:
            n_limit += 1
        # DEVIATION 961: count LS_LIMIT here too. This block is titled "the
        # two LIMIT branches, reached exactly" and passes
        # `linesearch_iter_limit = 1` for no other reason, but only
        # ITER_LIMIT had a reach assertion; LS_LIMIT was a branch the README
        # CLAIMED this fixture covers with nothing in the gate that would
        # notice if it stopped. That is the same shape as the RHO_ZERO
        # failure, one line away from it, and it is closed here.
        if (Int(dd[s]) & 4) != 0:
            n_ls_limit += 1
        if Int(dd[s]) != Int(ob.decisions[s]):
            comptime if IDENTICAL:
                raise Error(
                    "check_hw_decision_branches FAILED at the limit fixture (sabotage "
                    + hw_sabotage_name() + "): series " + String(s) + " device decisions "
                    + String(dd[s]) + " [" + _dec_names(Int(dd[s])) + "] oracle "
                    + String(ob.decisions[s]) + " [" + _dec_names(Int(ob.decisions[s])) + "]"
                )
    if n_limit == 0:
        raise Error(
            "check_hw_decision_branches REACH FAILURE: bfgs_iter_limit = 2 did not"
            " set ITER_LIMIT on any series, so the fall-through return is still"
            " uncovered and the fixture does not do what it claims"
        )
    if n_ls_limit == 0 and ls_salt < 0:
        raise Error(
            "check_hw_decision_branches REACH FAILURE: LS_LIMIT (cuml#888's"
            " path, where `x = nx` stores the LAST trial step rather than the"
            " best one) is set by NO series anywhere in this census -- not by"
            " the linesearch_iter_limit = 1 override fixture, and not by any of"
            " the 128 natural fits. holtwinters/README.md states the override"
            " fixture reaches the BRANCH; if this line is printing, that"
            " sentence is false and the branch is uncovered. Delete the"
            " sentence before touching the fixture."
        )
    print(
        "check_hw_decision_branches " + ("OK" if IDENTICAL else "REPORT") + " [" + _mode_name()
        + "]: decisions device == oracle on 5 fixtures; union of bits seen on the"
        " standard fixtures = [" + _dec_names(seen) + "] max halvings " + String(max_halv)
        + "; RHO_ZERO reached on a natural series at salt " + String(rho_salt) + " "
        + rho_kind_str + " (" + String(rho_fits) + "/128 fits in the search, "
        + String(n_rho) + "/" + String(BATCH) + " series on the re-fit, oracle agrees),"
        " decisions == oracle; bfgs_iter_limit=2 reached ITER_LIMIT on "
        + String(n_limit) + "/" + String(BATCH) + " series and linesearch_iter_limit=1"
        " reached LS_LIMIT on " + String(n_ls_limit) + "/" + String(BATCH)
        + " series, criterion " + criterion_name(Int(cc[0]))
    )
    _ = ts_d^
    _ = sl^
    _ = st^
    _ = ss^
    _ = sl2^
    _ = st2^
    _ = ss2^
    _ = al^
    _ = be^
    _ = ga^
    _ = lv^
    _ = tv^
    _ = sv^
    _ = ev^
    _ = cr^
    _ = ni^
    _ = de^
    _ = itr^


# ---------------------------------------------------------------------------
# THE PACKED POINTER SURFACE (DEVIATION 930)
# ---------------------------------------------------------------------------
# `holtwinters/estimator.mojo`'s `holtwinters_fit_ptr` and
# `holtwinters_forecast_ptr` are what `bindings/_mojolearn_tsa.mojo` and
# `python/mojolearn/_tsa_impl.py` call. They exist because
# `PythonModuleBuilder.def_function` infers its signature from arity and
# gives out around nine arguments, so `fit` cannot hand back eight separate
# addresses. The three component series share ONE float32 buffer, the four
# per-series scalars share another, and the two integer flags share a third.
#
# THE LAYOUT, in the same words as the other three files:
#
#     comps   3 * components_len float32
#         [0 * components_len, 1 * components_len)   level
#         [1 * components_len, 2 * components_len)   trend
#         [2 * components_len, 3 * components_len)   season
#       each block TIME-MAJOR, series `s` at step `i` at `[s + i *
#       batch_size]` (`holtwinters.pyx:341-344`), which is a TRANSPOSE of
#       the series-major input `fit` reads
#     stats   4 * batch_size float32:  sse | alpha | beta | gamma
#     flags   2 * batch_size int32:    niter | criterion
#
# WHY THIS GATE EXISTS. `holtwinters_fit_ptr` writes that pack and
# `holtwinters_forecast_ptr` reads it back, and the two live eighty lines
# apart in ONE file. A SELF-CONSISTENT mistake -- trend and season swapped
# in both, or written series-major and read series-major -- round-trips
# perfectly. `fit(...).forecast(h)` then returns plausible numbers that are
# simply the wrong component, every end-to-end smoke test passes, and the
# three copies of the layout comment catch nothing, because the three copies
# agree with each other and not one of them is executed. Nothing else in the
# build can see this class of defect.
#
# So this gate BREAKS THE ROUND TRIP. `holtwinters_fit_host` returns the
# same fit as three structured `List`s and never sees the pack: it is the
# INDEPENDENT WITNESS, and the flat buffer is asserted against it cell by
# cell, BITWISE. `holtwinters_forecast_host` on a structured `HWFit` is the
# witness from the other side. There is no tolerance anywhere below and
# there must not be: both sides run the same arithmetic on the same inputs
# through the same `holtwinters_fit_host`, so the only thing a tolerance
# could absorb is the defect being hunted. Run-to-run determinism is not
# assumed here either; `check_hw_card_is_emitted` asserts it, in both
# numeric modes, before this check runs.
#
# AND THE CLAUSES ASSERT IN BOTH MODES. Everywhere else in this file a
# device-vs-oracle comparison RECORDS under FAST and raises under
# IDENTICAL, because FAST is allowed to take a different float path from
# the host oracle. Nothing here is a numerics question: both sides of every
# comparison are the SAME `holtwinters_fit_host` call's output, and a
# reordered buffer is wrong in every mode.
#
# THE VACUITY PROBLEM. A fixture where `batch_size` equals `components_len`,
# or where `batch_size` is 1, or where the three components have similar
# magnitudes, lets a swapped or transposed layout agree with the witness by
# accident. Three answers, all of them checked at run time rather than
# argued in a comment:
#   (i)   the fixture's extents are PAIRWISE DISTINCT (see PACK_N and
#         friends beside the other fixture constants);
#   (ii)  the fitted level, trend and season sit at plainly different
#         magnitudes and the season changes sign (on this fixture level is
#         ~1e3 to ~3e3, season ~5e1 to ~9e2 and signed, trend ~3e-1 to ~2),
#         so a block swap moves essentially every cell rather than a few;
#   (iii) THE SEPARATION CENSUS below re-emits the very buffer the pointer
#         entry wrote under each named mis-spelling and counts the cells
#         that move. A count of zero means the fixture cannot tell the
#         pinned layout from that mis-spelling, and the gate raises VACUOUS
#         instead of passing. That census runs on EVERY build, clean ones
#         included, so nobody has to rebuild to learn the gate is empty.
#
# THE SABOTAGE ARMS. Nine arms already in this lane are `-D
# MOJOLEARN_HW_SABOTAGE_<NAME>=1` switches on the ONE switchboard in
# `holtwinters/impl/holtwinters/internal/hw_utils.mojo`, because they
# sabotage device kernels. These eight do not: they corrupt the flat BUFFER
# after the pointer entry has written it, which is byte for byte what a
# wrong writer would have left there, and the buffer is the only thing the
# assertions can see anyway. They are declared here, next to the assertions
# they must break, because the code they model lives in a file this gate
# does not own.
#
# TWO DEFECT CLASSES, TWO KINDS OF ARM, and the difference is the whole
# point. `PACK_SWAP_*`, `PACK_SERIES_MAJOR` and `PACK_STRIDE_BATCH` corrupt
# the buffer the FIT clauses read and hand `holtwinters_forecast_ptr` the
# PRISTINE one. That is the dangerous defect drawn exactly: writer and
# reader wrong in the same way, forecast still round-tripping, and only the
# witness clause firing. `PACK_FORECAST_SWAP` does the opposite -- pristine
# fit clauses, corrupted forecast input -- which is a writer and a reader
# that disagree.

comptime SAB_PACK_SWAP_TREND_SEASON = is_defined[
    "MOJOLEARN_HW_SABOTAGE_PACK_SWAP_TREND_SEASON"
]()
comptime SAB_PACK_SWAP_LEVEL_TREND = is_defined[
    "MOJOLEARN_HW_SABOTAGE_PACK_SWAP_LEVEL_TREND"
]()
comptime SAB_PACK_SERIES_MAJOR = is_defined["MOJOLEARN_HW_SABOTAGE_PACK_SERIES_MAJOR"]()
comptime SAB_PACK_STRIDE_BATCH = is_defined["MOJOLEARN_HW_SABOTAGE_PACK_STRIDE_BATCH"]()
comptime SAB_PACK_SWAP_SSE_ALPHA = is_defined["MOJOLEARN_HW_SABOTAGE_PACK_SWAP_SSE_ALPHA"]()
comptime SAB_PACK_SWAP_BETA_GAMMA = is_defined[
    "MOJOLEARN_HW_SABOTAGE_PACK_SWAP_BETA_GAMMA"
]()
comptime SAB_PACK_SWAP_FLAGS = is_defined["MOJOLEARN_HW_SABOTAGE_PACK_SWAP_FLAGS"]()
comptime SAB_PACK_FORECAST_SWAP = is_defined["MOJOLEARN_HW_SABOTAGE_PACK_FORECAST_SWAP"]()


def _pack_comps_sabotage() -> String:
    """The comps-pack arm this build names, "" on a clean build."""
    comptime if SAB_PACK_SWAP_TREND_SEASON:
        return String("SWAP_TREND_SEASON")
    comptime if SAB_PACK_SWAP_LEVEL_TREND:
        return String("SWAP_LEVEL_TREND")
    comptime if SAB_PACK_SERIES_MAJOR:
        return String("SERIES_MAJOR")
    comptime if SAB_PACK_STRIDE_BATCH:
        return String("STRIDE_BATCH")
    return String("")


def _pack_stats_sabotage() -> String:
    comptime if SAB_PACK_SWAP_SSE_ALPHA:
        return String("SWAP_SSE_ALPHA")
    comptime if SAB_PACK_SWAP_BETA_GAMMA:
        return String("SWAP_BETA_GAMMA")
    return String("")


def _pack_flags_sabotage() -> String:
    comptime if SAB_PACK_SWAP_FLAGS:
        return String("SWAP_FLAGS")
    return String("")


def _pack_forecast_sabotage() -> String:
    comptime if SAB_PACK_FORECAST_SWAP:
        return String("SWAP_TREND_SEASON")
    return String("")


def _pack_sabotage_name() -> String:
    """The pack arms this build names, in the shape `hw_sabotage_name` uses."""
    var s = String("")
    if _pack_comps_sabotage() != "":
        s += "PACK_" + _pack_comps_sabotage() + " "
    if _pack_stats_sabotage() != "":
        s += "PACK_" + _pack_stats_sabotage() + " "
    if _pack_flags_sabotage() != "":
        s += "PACK_" + _pack_flags_sabotage() + " "
    if _pack_forecast_sabotage() != "":
        s += "PACK_FORECAST_SWAP "
    if s == "":
        return String("none")
    return s


def _first_diff_bits(tag: String, a: List[Float32], b: List[Float32]) -> String:
    """`_first_diff` WITHOUT the NaN canonicalization.

    A pack is a copy, not arithmetic. `_first_diff` canonicalizes because it
    compares two different computations of one quantity, which may reach
    different NaN payloads for the same reason. Here both sides come out of
    the SAME `holtwinters_fit_host` call, so a payload that changed
    on the way through the flat buffer is a defect, and canonicalizing it
    away would hide a member of the exact class this gate hunts.
    """
    if len(a) != len(b):
        return tag + ": length " + String(len(a)) + " vs " + String(len(b))
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return (
                tag + "[" + String(i) + "]: " + hex32(a[i]) + " vs " + hex32(b[i])
                + " (" + String(a[i]) + " vs " + String(b[i]) + ")"
            )
    return String("")


def _first_diff_i32(tag: String, a: List[Int32], b: List[Int32]) -> String:
    if len(a) != len(b):
        return tag + ": length " + String(len(a)) + " vs " + String(len(b))
    for i in range(len(a)):
        if a[i] != b[i]:
            return tag + "[" + String(i) + "]: " + String(a[i]) + " vs " + String(b[i])
    return String("")


def _count_diff_bits(a: List[Float32], b: List[Float32]) -> Int:
    var n = len(a) if len(a) < len(b) else len(b)
    var c = 0
    for i in range(n):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            c += 1
    return c


def _count_diff_i32(a: List[Int32], b: List[Int32]) -> Int:
    var n = len(a) if len(a) < len(b) else len(b)
    var c = 0
    for i in range(n):
        if a[i] != b[i]:
            c += 1
    return c


def _pack_reorder(
    variant: String, comps: List[Float32], cl: Int, batch: Int, rows: Int
) raises -> List[Float32]:
    """The `3 * cl` comps buffer as a DIFFERENTLY SPELLED writer would have
    left it.

    The three blocks are read out of `comps` per the DOCUMENTED layout and
    re-emitted per `variant` in `holtwinters_fit_ptr`'s own loop order --
    level, then trend, then season, `i` ascending -- so an arm whose stores
    overlap its own earlier stores lands exactly where the real mis-spelling
    would land. A cell the variant never writes keeps `PACK_POISON`: the
    Python surface hands the binding an `np.empty`, so an unwritten cell is
    garbage, and a definite poison is the honest stand-in for garbage.
    """
    if 2 * batch + cl > 3 * cl:
        raise Error(
            "_pack_reorder: the STRIDE_BATCH spelling would write past the"
            " buffer on this fixture (2*batch + cl = " + String(2 * batch + cl)
            + " > " + String(3 * cl) + "), so the arm is not expressible here"
        )
    var out = List[Float32]()
    out.reserve(3 * cl)
    for _ in range(3 * cl):
        out.append(PACK_POISON)
    for i in range(cl):
        var lv = comps[i]
        var tr = comps[cl + i]
        var sn = comps[2 * cl + i]
        if variant == "SWAP_TREND_SEASON":
            out[i] = lv
            out[cl + i] = sn
            out[2 * cl + i] = tr
        elif variant == "SWAP_LEVEL_TREND":
            out[i] = tr
            out[cl + i] = lv
            out[2 * cl + i] = sn
        elif variant == "SERIES_MAJOR":
            # the TRANSPOSE: cell (series `s`, step `k`) of each block put at
            # `s * rows + k` instead of `k * batch + s`
            var s = i % batch
            var k = i // batch
            var j = s * rows + k
            out[j] = lv
            out[cl + j] = tr
            out[2 * cl + j] = sn
        elif variant == "STRIDE_BATCH":
            # `batch_size` used where `components_len` belongs, which is the
            # stats pack's stride copied into the comps loop by hand
            out[i] = lv
            out[batch + i] = tr
            out[2 * batch + i] = sn
        else:
            raise Error("_pack_reorder: unknown variant '" + variant + "'")
    return out^


def _stats_reorder(variant: String, stats: List[Float32], batch: Int) raises -> List[Float32]:
    """The `4 * batch` stats buffer under a named mis-spelling."""
    var out = List[Float32]()
    out.reserve(4 * batch)
    for _ in range(4 * batch):
        out.append(PACK_POISON)
    for b in range(batch):
        var sse = stats[b]
        var alpha = stats[batch + b]
        var beta = stats[2 * batch + b]
        var gamma = stats[3 * batch + b]
        if variant == "SWAP_SSE_ALPHA":
            out[b] = alpha
            out[batch + b] = sse
            out[2 * batch + b] = beta
            out[3 * batch + b] = gamma
        elif variant == "SWAP_BETA_GAMMA":
            out[b] = sse
            out[batch + b] = alpha
            out[2 * batch + b] = gamma
            out[3 * batch + b] = beta
        else:
            raise Error("_stats_reorder: unknown variant '" + variant + "'")
    return out^


def _flags_reorder(variant: String, flags: List[Int32], batch: Int) raises -> List[Int32]:
    """The `2 * batch` flags buffer under a named mis-spelling."""
    var out = List[Int32]()
    out.reserve(2 * batch)
    for _ in range(2 * batch):
        out.append(PACK_POISON_I)
    for b in range(batch):
        if variant == "SWAP_FLAGS":
            out[b] = flags[batch + b]
            out[batch + b] = flags[b]
        else:
            raise Error("_flags_reorder: unknown variant '" + variant + "'")
    return out^


def check_hw_pack_round_trip() raises:
    """DEVIATION 930: the flat pack `bindings/` reads is the structured fit.

    Seven clauses, every one bitwise, and every one of them a comparison
    against something that never saw the pack:

      (a) comps    `holtwinters_fit_ptr`'s `3 * components_len` buffer,
                   unpacked per the documented layout, cell by cell against
                   `holtwinters_fit_host`'s `level` / `trend` / `season`
      (b) extent   the return value, and guard cells past the end of all
                   four buffers still holding their poison -- a writer that
                   over-strides is a wrong answer, not a crash
      (c) stats    `sse` / `alpha` / `beta` / `gamma`, same principle
      (d) flags    `niter` / `criterion`, same principle
      (e) meaning  each block read the way `_tsa_impl.py` reads it, `[s + i
                   * batch_size]`, against series `s` FITTED ALONE. This is
                   the clause that pins TIME-MAJOR: (a) would pass a pack
                   that faithfully copied a wrongly ordered `HWFit`, and
                   this one would not. When the structured witness itself
                   disagrees with the batch-of-1 fit the message says so,
                   because that is `check_hw_launch_invariance`'s claim and
                   not this gate's.
      (f) forecast `holtwinters_forecast_ptr` reading the flat buffer
                   against `holtwinters_forecast_host` on the structured
                   `HWFit`, plus its own guard cells and return value
      (g) refusals `holtwinters_forecast_ptr`'s three by-name refusals (its
                   own `n <= frequency` and `batch_size < 1`, plus the `h
                   must be > 0` it inherits), which nothing else reaches

    Plus the controls, which raise VACUOUS rather than pass: the fixture's
    extents are pairwise distinct, the separation census finds every named
    mis-spelling visible on this fixture, and a mis-read comps buffer moves
    the FORECAST and not only the buffer.
    """
    var ctx = DeviceContext()
    var rows = PACK_N - PACK_FREQ
    var cl = rows * PACK_BATCH
    var n_comps = 3 * cl
    var n_stats = 4 * PACK_BATCH
    var n_flags = 2 * PACK_BATCH
    var n_fc = PACK_H * PACK_BATCH

    # CONTROL 1: the fixture's extents, asserted and not merely asserted in
    # prose. If any two of these coincide, an index arithmetic error can
    # land on a correct answer and every clause below is worth less than it
    # looks.
    var dims: List[Int] = [
        PACK_BATCH, PACK_FREQ, rows, cl, PACK_H, n_comps, n_stats, n_flags, n_fc
    ]
    var dnames: List[String] = [
        "batch_size", "frequency", "rows_per_series", "components_len", "h",
        "3*components_len", "4*batch_size", "2*batch_size", "h*batch_size",
    ]
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            if dims[i] == dims[j]:
                raise Error(
                    "check_hw_pack_round_trip VACUOUS FIXTURE [" + _mode_name() + "]: "
                    + dnames[i] + " == " + dnames[j] + " == " + String(dims[i])
                    + ", so an index arithmetic error can alias onto a correct"
                    " answer here; change PACK_N / PACK_FREQ / PACK_BATCH / PACK_H"
                )

    # The three series are three DIFFERENT families, fitted with one
    # seasonal type, so the optimizer has no reason to land on the same
    # parameters twice (the stats pack's swap arms need at least one series
    # where the swapped pair differs, and CONTROL 2 checks that it got one).
    var specs: List[HWFixtureSpec] = [
        spec_additive(), spec_multiplicative(), spec_additive_noiseless()
    ]
    var data = hw_fixture_mixed(specs, PACK_N, PACK_BATCH, PACK_FREQ, PACK_SALT)

    # THE INDEPENDENT WITNESS. `holtwinters_fit_ptr` calls exactly this
    # function inside itself and then packs; this call is the same fit with
    # the packing left out.
    var fitted = holtwinters_fit_host(
        data, PACK_N, PACK_BATCH, PACK_FREQ, START_PERIODS, "additive", HW_DEFAULT_EPS
    )
    if len(fitted.level) != cl or len(fitted.sse) != PACK_BATCH:
        raise Error(
            "check_hw_pack_round_trip: the witness returned " + String(len(fitted.level))
            + " component cells and " + String(len(fitted.sse)) + " stats cells,"
            " expected " + String(cl) + " and " + String(PACK_BATCH)
        )

    # The four host buffers, every cell poisoned first so an UNWRITTEN cell
    # inside the documented extent fails a clause instead of reading as a
    # coincidence, and PACK_GUARD poisoned cells past the end of each.
    var hdata = ctx.enqueue_create_host_buffer[DType.float32](PACK_N * PACK_BATCH)
    for i in range(PACK_N * PACK_BATCH):
        hdata.unsafe_ptr().unsafe_store(i, data[i])
    var hcomps = ctx.enqueue_create_host_buffer[DType.float32](n_comps + PACK_GUARD)
    for i in range(n_comps + PACK_GUARD):
        hcomps.unsafe_ptr().unsafe_store(i, PACK_POISON)
    var hstats = ctx.enqueue_create_host_buffer[DType.float32](n_stats + PACK_GUARD)
    for i in range(n_stats + PACK_GUARD):
        hstats.unsafe_ptr().unsafe_store(i, PACK_POISON)
    var hflags = ctx.enqueue_create_host_buffer[DType.int32](n_flags + PACK_GUARD)
    for i in range(n_flags + PACK_GUARD):
        hflags.unsafe_ptr().unsafe_store(i, PACK_POISON_I)

    var got_cl = holtwinters_fit_ptr(
        hdata.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        hcomps.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        hstats.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        hflags.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        PACK_N, PACK_BATCH, PACK_FREQ, START_PERIODS, "additive", HW_DEFAULT_EPS,
    )

    var comps = List[Float32]()
    comps.reserve(n_comps)
    for i in range(n_comps):
        comps.append(hcomps.unsafe_ptr().unsafe_load(i))
    var stats = List[Float32]()
    stats.reserve(n_stats)
    for i in range(n_stats):
        stats.append(hstats.unsafe_ptr().unsafe_load(i))
    var flags = List[Int32]()
    flags.reserve(n_flags)
    for i in range(n_flags):
        flags.append(hflags.unsafe_ptr().unsafe_load(i))

    # CONTROL 2: THE SEPARATION CENSUS, on the buffer the entry actually
    # wrote and on every build. Each named mis-spelling is re-emitted and
    # its moved cells counted; a zero means this fixture cannot tell that
    # mis-spelling from the pinned layout, which makes the matching arm
    # unfalsifiable and the clause it guards worth nothing.
    var sep = String("")
    var cvars: List[String] = [
        "SWAP_TREND_SEASON", "SWAP_LEVEL_TREND", "SERIES_MAJOR", "STRIDE_BATCH"
    ]
    for vi in range(len(cvars)):
        var v = cvars[vi]
        var moved = _count_diff_bits(_pack_reorder(v, comps, cl, PACK_BATCH, rows), comps)
        if moved == 0:
            raise Error(
                "check_hw_pack_round_trip VACUOUS [" + _mode_name() + "]: the comps"
                " mis-spelling " + v + " leaves all " + String(n_comps) + " cells"
                " unchanged on this fixture, so clause (a) cannot see it and the"
                " PACK_" + v + " arm cannot fail; change PACK_SALT or the fixture"
            )
        sep += v + " " + String(moved) + "/" + String(n_comps) + " "
    var svars: List[String] = ["SWAP_SSE_ALPHA", "SWAP_BETA_GAMMA"]
    for si in range(len(svars)):
        var sv = svars[si]
        var smoved = _count_diff_bits(_stats_reorder(sv, stats, PACK_BATCH), stats)
        if smoved == 0:
            raise Error(
                "check_hw_pack_round_trip VACUOUS [" + _mode_name() + "]: the stats"
                " mis-spelling " + sv + " leaves all " + String(n_stats) + " cells"
                " unchanged, so the two swapped blocks are bit-equal on every one"
                " of the " + String(PACK_BATCH) + " series and clause (c) cannot"
                " see the swap; change PACK_SALT or the fixture families"
            )
        sep += sv + " " + String(smoved) + "/" + String(n_stats) + " "
    var fmoved = _count_diff_i32(_flags_reorder("SWAP_FLAGS", flags, PACK_BATCH), flags)
    if fmoved == 0:
        raise Error(
            "check_hw_pack_round_trip VACUOUS [" + _mode_name() + "]: niter equals"
            " criterion on every series, so SWAP_FLAGS is invisible and clause (d)"
            " cannot see it; change PACK_SALT or the fixture families"
        )
    sep += "SWAP_FLAGS " + String(fmoved) + "/" + String(n_flags)

    # The arms. `comps_seen` is what the fit clauses read; the forecast
    # clause gets `comps` itself unless PACK_FORECAST_SWAP is named, which
    # is what makes the fit arms a model of the SELF-CONSISTENT defect (a
    # writer and a reader wrong the same way) rather than of a disagreement.
    var comps_seen = comps.copy()
    if _pack_comps_sabotage() != "":
        comps_seen = _pack_reorder(
            _pack_comps_sabotage(), comps, cl, PACK_BATCH, rows
        )
    var stats_seen = stats.copy()
    if _pack_stats_sabotage() != "":
        stats_seen = _stats_reorder(_pack_stats_sabotage(), stats, PACK_BATCH)
    var flags_seen = flags.copy()
    if _pack_flags_sabotage() != "":
        flags_seen = _flags_reorder(_pack_flags_sabotage(), flags, PACK_BATCH)

    var cnames = List[String]()
    var cmsgs = List[String]()

    # (a) comps vs the structured witness, cell by cell.
    var want = List[Float32]()
    want.reserve(n_comps)
    for i in range(cl):
        want.append(fitted.level[i])
    for i in range(cl):
        want.append(fitted.trend[i])
    for i in range(cl):
        want.append(fitted.season[i])
    cnames.append("a comps")
    cmsgs.append(_first_diff_bits("comps pack vs fit_host level|trend|season", comps_seen, want))

    # (b) the extent: the returned length, and the guard cells.
    var bad_ext = String("")
    if got_cl != cl:
        bad_ext = (
            "holtwinters_fit_ptr returned " + String(got_cl) + ", expected"
            " (n - frequency) * batch_size = " + String(cl)
        )
    for i in range(PACK_GUARD):
        if bad_ext == "" and bitcast[DType.uint32](hcomps.unsafe_ptr().unsafe_load(n_comps + i)) != bitcast[DType.uint32](PACK_POISON):
            bad_ext = "comps guard cell " + String(i) + " past " + String(n_comps) + " was written"
        if bad_ext == "" and bitcast[DType.uint32](hstats.unsafe_ptr().unsafe_load(n_stats + i)) != bitcast[DType.uint32](PACK_POISON):
            bad_ext = "stats guard cell " + String(i) + " past " + String(n_stats) + " was written"
        if bad_ext == "" and hflags.unsafe_ptr().unsafe_load(n_flags + i) != PACK_POISON_I:
            bad_ext = "flags guard cell " + String(i) + " past " + String(n_flags) + " was written"
    cnames.append("b extent")
    cmsgs.append(bad_ext)

    # (c) stats.
    var want_stats = List[Float32]()
    want_stats.reserve(n_stats)
    for b in range(PACK_BATCH):
        want_stats.append(fitted.sse[b])
    for b in range(PACK_BATCH):
        want_stats.append(fitted.alpha[b])
    for b in range(PACK_BATCH):
        want_stats.append(fitted.beta[b])
    for b in range(PACK_BATCH):
        want_stats.append(fitted.gamma[b])
    cnames.append("c stats")
    cmsgs.append(_first_diff_bits("stats pack vs fit_host sse|alpha|beta|gamma", stats_seen, want_stats))

    # (d) flags.
    var want_flags = List[Int32]()
    want_flags.reserve(n_flags)
    for b in range(PACK_BATCH):
        want_flags.append(fitted.niter[b])
    for b in range(PACK_BATCH):
        want_flags.append(fitted.criterion[b])
    cnames.append("d flags")
    cmsgs.append(_first_diff_i32("flags pack vs fit_host niter|criterion", flags_seen, want_flags))

    # (e) THE MEANING OF THE INDEX. Series `s` fitted ALONE, against the
    # cells `_tsa_impl.py`'s `reshape((ts_num, num_rows), order="F")` reads
    # for that series. Clause (a) compares the pack to the `HWFit` and would
    # pass a faithful copy of a wrongly ordered one; this compares the pack
    # to a fit that shares no buffer, no launch geometry and no batch with
    # it.
    var bad_tm = String("")
    var bad_bi = String("")
    for s in range(PACK_BATCH):
        var one = List[Float32]()
        one.reserve(PACK_N)
        for t in range(PACK_N):
            one.append(data[s * PACK_N + t])
        var f1 = holtwinters_fit_host(
            one, PACK_N, 1, PACK_FREQ, START_PERIODS, "additive", HW_DEFAULT_EPS
        )
        var w_lv = List[Float32]()
        var w_tr = List[Float32]()
        var w_sn = List[Float32]()
        var p_lv = List[Float32]()
        var p_tr = List[Float32]()
        var p_sn = List[Float32]()
        for i in range(rows):
            var k = s + i * PACK_BATCH
            w_lv.append(fitted.level[k])
            w_tr.append(fitted.trend[k])
            w_sn.append(fitted.season[k])
            p_lv.append(comps_seen[k])
            p_tr.append(comps_seen[cl + k])
            p_sn.append(comps_seen[2 * cl + k])
        var tag1 = "series " + String(s) + " batch-1 vs batch-" + String(PACK_BATCH) + " "
        if bad_bi == "":
            bad_bi = _first_diff_bits(tag1 + "level", f1.level, w_lv)
        if bad_bi == "":
            bad_bi = _first_diff_bits(tag1 + "trend", f1.trend, w_tr)
        if bad_bi == "":
            bad_bi = _first_diff_bits(tag1 + "season", f1.season, w_sn)
        var tag2 = "series " + String(s) + " batch-1 vs comps[block + s + i*batch_size] "
        if bad_tm == "":
            bad_tm = _first_diff_bits(tag2 + "level", f1.level, p_lv)
        if bad_tm == "":
            bad_tm = _first_diff_bits(tag2 + "trend", f1.trend, p_tr)
        if bad_tm == "":
            bad_tm = _first_diff_bits(tag2 + "season", f1.season, p_sn)
    if bad_bi != "":
        # NOT the pack. The STRUCTURED witness already disagrees with the
        # batch-of-1 fit, which is check_hw_launch_invariance's claim, so
        # say whose claim broke instead of blaming the buffer.
        bad_tm = (
            "BATCH INVARIANCE, not the pack (" + bad_bi + "); the structured HWFit"
            " itself disagrees with a batch-of-1 fit, which is"
            " check_hw_launch_invariance's clause"
        )
    cnames.append("e time-major")
    cmsgs.append(bad_tm)

    # (f) the forecast, from the other side.
    var comps_fc = comps.copy()
    if _pack_forecast_sabotage() != "":
        comps_fc = _pack_reorder(
            _pack_forecast_sabotage(), comps, cl, PACK_BATCH, rows
        )
    var hfc_in = ctx.enqueue_create_host_buffer[DType.float32](n_comps)
    for i in range(n_comps):
        hfc_in.unsafe_ptr().unsafe_store(i, comps_fc[i])
    var hfc_out = ctx.enqueue_create_host_buffer[DType.float32](n_fc + PACK_GUARD)
    for i in range(n_fc + PACK_GUARD):
        hfc_out.unsafe_ptr().unsafe_store(i, PACK_POISON)
    var got_fc = holtwinters_forecast_ptr(
        hfc_in.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        hfc_out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        PACK_N, PACK_BATCH, PACK_FREQ, "additive", PACK_H,
    )
    var fc_ptr = List[Float32]()
    fc_ptr.reserve(n_fc)
    for i in range(n_fc):
        fc_ptr.append(hfc_out.unsafe_ptr().unsafe_load(i))
    var fc_host = holtwinters_forecast_host(fitted, PACK_H)
    var bad_fc = String("")
    if got_fc != n_fc:
        bad_fc = (
            "holtwinters_forecast_ptr returned " + String(got_fc) + ", expected h *"
            " batch_size = " + String(n_fc)
        )
    for i in range(PACK_GUARD):
        if bad_fc == "" and bitcast[DType.uint32](hfc_out.unsafe_ptr().unsafe_load(n_fc + i)) != bitcast[DType.uint32](PACK_POISON):
            bad_fc = "forecast guard cell " + String(i) + " past " + String(n_fc) + " was written"
    if bad_fc == "":
        bad_fc = _first_diff_bits("forecast_ptr(comps) vs forecast_host(HWFit)", fc_ptr, fc_host)
    cnames.append("f forecast")
    cmsgs.append(bad_fc)

    # CONTROL 3: clause (f) is a chain of comparisons that would pass for
    # ever if the forecast did not depend on the packed order at all. So
    # prove the dependence: the same entry, on a comps buffer read with
    # trend and season exchanged, must move the forecast.
    var hfc_in2 = ctx.enqueue_create_host_buffer[DType.float32](n_comps)
    var comps_sw = _pack_reorder("SWAP_TREND_SEASON", comps, cl, PACK_BATCH, rows)
    for i in range(n_comps):
        hfc_in2.unsafe_ptr().unsafe_store(i, comps_sw[i])
    var hfc_out2 = ctx.enqueue_create_host_buffer[DType.float32](n_fc)
    for i in range(n_fc):
        hfc_out2.unsafe_ptr().unsafe_store(i, PACK_POISON)
    _ = holtwinters_forecast_ptr(
        hfc_in2.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        hfc_out2.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        PACK_N, PACK_BATCH, PACK_FREQ, "additive", PACK_H,
    )
    var fc_sw = List[Float32]()
    fc_sw.reserve(n_fc)
    for i in range(n_fc):
        fc_sw.append(hfc_out2.unsafe_ptr().unsafe_load(i))
    var fc_moved = _count_diff_bits(fc_sw, fc_host)
    if fc_moved == 0:
        raise Error(
            "check_hw_pack_round_trip VACUOUS [" + _mode_name() + "]: a comps buffer"
            " with trend and season EXCHANGED forecasts the same " + String(n_fc)
            + " bits, so clause (f) cannot see a mis-read pack and its pass means"
            " nothing"
        )

    # (g) the three by-name refusals `holtwinters_forecast_ptr` makes: its
    # own `n <= frequency` and `batch_size < 1`, and the `h must be > 0` it
    # inherits from `holtwinters_forecast_host_traced`. `check_hw_refusals`
    # reaches only the last of the three, and only through the list-shaped
    # entry, so the two the pointer entry owns are reached nowhere else.
    var bad_ref = String("")
    var n_ref = 0
    var raised = False
    try:
        _ = holtwinters_forecast_ptr(
            hfc_in.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            hfc_out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            PACK_FREQ, PACK_BATCH, PACK_FREQ, "additive", PACK_H,
        )
    except e:
        raised = True
        if String(e).find("must exceed frequency") < 0:
            bad_ref = "n == frequency raised '" + String(e) + "' without naming 'must exceed frequency'"
    if not raised:
        bad_ref = "n == frequency did not raise"
    else:
        n_ref += 1
    raised = False
    try:
        _ = holtwinters_forecast_ptr(
            hfc_in.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            hfc_out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            PACK_N, 0, PACK_FREQ, "additive", PACK_H,
        )
    except e:
        raised = True
        if bad_ref == "" and String(e).find("batch_size must be >= 1") < 0:
            bad_ref = "batch_size 0 raised '" + String(e) + "' without naming 'batch_size must be >= 1'"
    if not raised:
        if bad_ref == "":
            bad_ref = "batch_size 0 did not raise"
    else:
        n_ref += 1
    raised = False
    try:
        _ = holtwinters_forecast_ptr(
            hfc_in.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            hfc_out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            PACK_N, PACK_BATCH, PACK_FREQ, "additive", 0,
        )
    except e:
        raised = True
        if bad_ref == "" and String(e).find("h must be > 0") < 0:
            bad_ref = "h 0 raised '" + String(e) + "' without naming 'h must be > 0'"
    if not raised:
        if bad_ref == "":
            bad_ref = "h 0 did not raise"
    else:
        n_ref += 1
    cnames.append("g refusals")
    cmsgs.append(bad_ref)

    # The verdict, one line per broken clause, because an operator running
    # eight arms wants to read which clauses each arm moves and not just the
    # first assertion that happened to fire.
    var n_failed = 0
    var failed = String("")
    var first_msg = String("")
    for i in range(len(cmsgs)):
        if cmsgs[i] != "":
            n_failed += 1
            failed += " " + cnames[i]
            if first_msg == "":
                first_msg = cnames[i] + ": " + cmsgs[i]
    if n_failed != 0 or _pack_sabotage_name() != "none":
        var line = String("")
        for i in range(len(cmsgs)):
            line += cnames[i] + "=" + ("BROKE" if cmsgs[i] != "" else "held") + " "
        print("  CLAUSES " + line)
    _ = hdata^
    _ = hcomps^
    _ = hstats^
    _ = hflags^
    _ = hfc_in^
    _ = hfc_out^
    _ = hfc_in2^
    _ = hfc_out2^
    _ = fitted^
    if n_failed != 0:
        raise Error(
            "check_hw_pack_round_trip FAILED [" + _mode_name() + "] (pack sabotage "
            + _pack_sabotage_name() + "): " + String(n_failed) + " of "
            + String(len(cmsgs)) + " clauses broke [" + failed + " ]; first "
            + first_msg
        )
    print(
        "check_hw_pack_round_trip OK [" + _mode_name() + "]: fit_ptr's "
        + String(n_comps) + "-cell comps, " + String(n_stats) + "-cell stats and "
        + String(n_flags) + "-cell flags packs are bitwise equal to fit_host's"
        " structured lists (n " + String(PACK_N) + ", batch_size " + String(PACK_BATCH)
        + ", frequency " + String(PACK_FREQ) + ", components_len " + String(cl)
        + ", rows " + String(rows) + ", all extents distinct); guards past all four"
        " buffers untouched; " + String(PACK_BATCH) + " batch-of-1 fits pin every"
        " block TIME-MAJOR at [s + i*batch_size]; forecast_ptr == forecast_host over "
        + String(n_fc) + " cells (" + String(n_ref) + " refusals by name); SEPARATION "
        + sep + "; swapped-read forecast moves " + String(fc_moved) + "/" + String(n_fc)
    )


def main() raises:
    print(
        "== holtwinters/checks/hw_check.mojo [" + _mode_name() + "] sabotage="
        + hw_sabotage_name() + " pack_sabotage=" + _pack_sabotage_name() + " =="
    )
    check_hw_refusals()
    check_hw_decompose_vs_reference()
    check_hw_optimizer_reduces_sse()
    check_hw_forecast_continues_pattern()
    check_hw_signed_zero_clamp()
    check_hw_zero_series_keeps_start()
    check_hw_device_equals_oracle()
    check_hw_launch_invariance()
    check_hw_card_is_emitted()
    check_hw_decision_branches()
    check_hw_pack_round_trip()
    print("== hw_check: ALL OK [" + _mode_name() + "] ==")
