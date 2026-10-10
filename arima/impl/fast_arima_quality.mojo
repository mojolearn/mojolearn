# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-arima-quality (2026-10-05), FAST + Apple only: three
AutoARIMA quality rules of statsforecast's `auto_arima_f` (`arima.py`), each
behind its own default-off define.

On taxi-hourly our forecast_rmse was 75.71 against statsforecast's 68.21; the
comparison found these algorithm gaps:

  ARIMA_FAST_CONST_BOTH: statsforecast's `search_arima` fits every (p, q)
    with and without the constant (`for K in range(max_K + 1)`, max_K = 1
    when allowmean (d + D == 0) or allowdrift (d + D == 1)); ours fixed the
    constant by `fit_intercept="auto"`. The define doubles the grid (k
    innermost, the reference's order) in the SAME batched search call; the
    choice is the board's ic (aicc). The grid change is Python glue
    (`_x_sequence_autoarima.py`, `_k_options`); this module only reports it.
  ARIMA_FAST_ROOT_CHECK: statsforecast's `myarima` (like R's auto.arima)
    sets a candidate's ic to +inf when its AR or MA polynomial has a root of
    modulus < 1.01. Here, after each candidate's fit, one thread per series
    tests the fitted AR and MA polynomials of every candidate on the device
    (`root_check_kernel`) and writes the candidate's log-likelihood as -inf
    (its ic +inf in every criterion fold). The test is the Schur-Cohn
    step-down recursion on the polynomial rescaled by 1.01: P(z) has a root
    with |z| < 1.01 iff P(1.01 w) has a root with |w| < 1 iff a reflection
    coefficient of P(1.01 w) has modulus >= 1 (no eigenvalues, no iteration,
    degree <= 4). Coefficients are trimmed at the last |c| > 1e-8 first, as
    `myarima` does.
  ARIMA_FAST_KPSS_D: our d came from cuML's KPSS rule (`select_d`: lags
    ceil(12 (n/100)^0.25), the first stationary order of the p-value against
    0.05). statsforecast's `ndiffs` differs: lags floor(3 sqrt(n) / 13)
    (re-derived on each differenced series), statsmodels' `kpss(x, "c")`
    statistic, p-value by `np.interp` over the table (0.347, 0.463, 0.574,
    0.739) -> (0.10, 0.05, 0.025, 0.01), difference while p < alpha = 0.05
    and d < max_d, and the `is_constant` stops (a constant series takes d =
    0; a series constant after a difference stops there). `ndiffs_kpss_kernel`
    runs that whole loop on the device, one block per series, every series
    in one launch.

Each is UNMEASURED as of 2026-10-05, merged on compile by Andrew's decision.
The A/B that judges them: lane autoarima on taxi-hourly and a synthetic set,
arm A FAST main, arm B the define; forecast_rmse must not go up (quality
must not go down).
"""
from std.math import inf, sqrt, floor
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from core.neural_context import process_ctx
from arima.estimator import _DEVCTX_SLOT, _upload_f32, _write_list_i32
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, unpack
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()

#: MOJOLEARN_ARIMA_FAST_CONST_BOTH (default off, READY-AB). Every order of
#: the AutoARIMA grid fitted with and without the constant (mean for d + D ==
#: 0, drift for d + D == 1; none for d + D == 2) in the same batched search
#: call, statsforecast `search_arima`'s K loop; picked by the board's ic.
#: Works with SEARCH_REUSE. (ARIMA_FAST_STEPWISE and ARIMA_FAST_CSS_SEARCH,
#: which it also fed, were deleted 2026-10-09: docs/TOMBSTONES.md.)
#: UNMEASURED as of 2026-10-05, merged on compile by Andrew's decision. A/B:
#: lane autoarima, taxi-hourly + synthetic, forecast_rmse must not go up.
comptime ARIMA_FAST_CONST_BOTH = (
    _FAST_APPLE and ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FAST_CONST_BOTH"]()
)
#: MOJOLEARN_ARIMA_FAST_ROOT_CHECK (default off, READY-AB). After each
#: candidate's fit in the device search (grouped exact), a candidate whose AR or MA polynomial
#: has a root of modulus < 1.01 gets log-likelihood -inf (ic +inf), as
#: statsforecast's `myarima` / R's auto.arima.
#: UNMEASURED as of 2026-10-05, merged on compile by Andrew's decision. A/B:
#: lane autoarima, taxi-hourly + synthetic, forecast_rmse must not go up.
comptime ARIMA_FAST_ROOT_CHECK = (
    _FAST_APPLE and ARIMA_ORDER_BATCH and is_defined["MOJOLEARN_ARIMA_FAST_ROOT_CHECK"]()
)
#: MOJOLEARN_ARIMA_FAST_KPSS_D (default off, READY-AB). AutoARIMA's d by
#: statsforecast's `ndiffs` (KPSS level test, lags floor(3 sqrt(n) / 13),
#: alpha 0.05, up to max_d) on the device, batched over series
#: (`ndiffs_kpss`), in place of cuML's KPSS rule (`select_d`).
#: UNMEASURED as of 2026-10-05, merged on compile by Andrew's decision. A/B:
#: lane autoarima, taxi-hourly + synthetic, forecast_rmse must not go up.
comptime ARIMA_FAST_KPSS_D = _FAST_APPLE and is_defined["MOJOLEARN_ARIMA_FAST_KPSS_D"]()


def quality_mode() -> Int:
    """This build's quality switches for the Python glue (bits 3..5 of
    `fast_search_mode`): bit 3 CONST_BOTH, bit 4 ROOT_CHECK, bit 5 KPSS_D."""
    var mode = 0
    comptime if ARIMA_FAST_CONST_BOTH:
        mode |= 8
    comptime if ARIMA_FAST_ROOT_CHECK:
        mode |= 16
    comptime if ARIMA_FAST_KPSS_D:
        mode |= 32
    return mode


# ---------------------------------------------------------------------------
# ARIMA_FAST_ROOT_CHECK
# ---------------------------------------------------------------------------

#: statsforecast `myarima`: `if minroot < 1 + 1e-2: fit["ic"] = math.inf`
comptime ROOT_MIN_MODULUS = Float32(1.01)
#: statsforecast `myarima`: `k = abs(testvec) > 1e-8` trims trailing zeros
comptime ROOT_TRIM = Float32(1e-8)
#: the largest polynomial degree the FAST searches fit (r <= 4)
comptime ROOT_MAX_DEG = 4
comptime ROOT_TPB = 128


def root_check_kernel(
    ll: MutPointer[Float32, MutAnyOrigin],
    ar: MutPointer[Float32, MutAnyOrigin],
    ma: MutPointer[Float32, MutAnyOrigin],
    bs_in: Int32, row_in: Int32, p_in: Int32, q_in: Int32,
):
    """ARIMA_FAST_ROOT_CHECK: one thread per series of one candidate order.
    `ar` / `ma` are the fitted (natural, Jones-transformed) coefficients,
    `[b * p + i]` / `[b * q + i]`. AR polynomial 1 - phi_1 z - ..., MA
    polynomial 1 + theta_1 z + ... (statsforecast's `np.append(1, -phi)` /
    `np.append(1, theta)`). A series with a root of modulus < 1.01 in either
    gets `ll[row * bs + b] = -inf`.

    The root test, per polynomial 1 + a[1] z + ... + a[deg] z^deg, in this
    thread (every loop is bounded by ROOT_MAX_DEG): trim at the last
    |a[i]| > 1e-8 (statsforecast's `testvec` trim), then the Schur-Cohn
    step-down on the coefficients scaled by 1.01^i: every reflection
    coefficient k_m = a[m] must have |k_m| < 1, and the order m - 1
    polynomial is (a[i] - k_m a[m - i]) / (1 - k_m^2). A NaN coefficient
    fails the test (rejects). The AR polynomial is tested first and the MA
    one only when the AR one passes (the same operations, in the same order,
    as the former `_root_inside` helper called twice)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var bs = Int(bs_in)
    if b >= bs:
        return
    var p = Int(p_in)
    var q = Int(q_in)
    var reject = False
    for poly in range(2):
        var deg = p if poly == 0 else q
        if reject or deg <= 0 or deg > ROOT_MAX_DEG:
            continue
        var a = InlineArray[Float32, ROOT_MAX_DEG + 1](fill=0.0)
        a[0] = 1.0
        for i in range(deg):
            if poly == 0:
                a[i + 1] = -ar.unsafe_load(b * p + i)
            else:
                a[i + 1] = ma.unsafe_load(b * q + i)
        while deg > 0 and abs(a[deg]) <= ROOT_TRIM:
            deg -= 1
        var scale = Float32(1.0)
        for i in range(1, deg + 1):
            scale *= ROOT_MIN_MODULUS
            a[i] = a[i] * scale
        var m = deg
        var tmp = InlineArray[Float32, ROOT_MAX_DEG + 1](fill=0.0)
        while m >= 1 and not reject:
            var k = a[m]
            if not (abs(k) < Float32(1.0)):
                reject = True
            else:
                var den = Float32(1.0) - k * k
                for i in range(1, m):
                    tmp[i] = (a[i] - k * a[m - i]) / den
                for i in range(1, m):
                    a[i] = tmp[i]
                m -= 1
    if reject:
        ll.unsafe_store(Int(row_in) * bs + b, -inf[DType.float32]())


def enqueue_root_check(
    ctx: DeviceContext, order: ARIMAOrder, bs: Int,
    mut x: DeviceBuffer[DType.float32], mut ll: DeviceBuffer[DType.float32], row: Int,
    mut keep: List[ARIMAParams],
) raises:
    """ARIMA_FAST_ROOT_CHECK for one candidate: its optimum `x` (packed,
    optimizer space) unpacked and Jones-transformed to the fitted
    coefficients on the device (`_enqueue_fit_block`'s steps), then
    `root_check_kernel` on row `row` of the (order, series) table `ll`.
    Nothing waits; the temporaries move into `keep`, which the caller drops
    after its next synchronize."""
    if order.p == 0 and order.q == 0:
        return
    var raw = ARIMAParams(ctx, order, bs)
    var nat = ARIMAParams(ctx, order, bs)
    unpack(ctx, raw, order, bs, x)
    batched_jones_transform(ctx, order, bs, False, raw, nat)
    ctx.enqueue_function[root_check_kernel](
        ll.unsafe_ptr(), nat.ar.unsafe_ptr(), nat.ma.unsafe_ptr(),
        Int32(bs), Int32(row), Int32(order.p), Int32(order.q),
        grid_dim=((bs + ROOT_TPB - 1) // ROOT_TPB, 1, 1), block_dim=(ROOT_TPB, 1, 1),
    )
    keep.append(raw^)
    keep.append(nat^)


# ---------------------------------------------------------------------------
# ARIMA_FAST_KPSS_D
# ---------------------------------------------------------------------------

comptime KD_TPB = 256
#: statsmodels `kpss` table for regression "c" (`crit = [0.347, 0.463,
#: 0.574, 0.739]`, `pvals = [0.10, 0.05, 0.025, 0.01]`)
comptime KD_C0 = Float32(0.347)
comptime KD_C1 = Float32(0.463)
comptime KD_C2 = Float32(0.574)
comptime KD_C3 = Float32(0.739)
comptime KD_P0 = Float32(0.10)
comptime KD_P1 = Float32(0.05)
comptime KD_P2 = Float32(0.025)
comptime KD_P3 = Float32(0.01)
#: statsforecast `ndiffs` default alpha
comptime KD_ALPHA = Float32(0.05)


@always_inline
def _kpss_pvalue(stat: Float32) -> Float32:
    """`np.interp(stat, crit, pvals)`: clamped at both ends, linear between
    table points. NaN stays NaN."""
    if stat != stat:
        return stat
    if stat <= KD_C0:
        return KD_P0
    if stat >= KD_C3:
        return KD_P3
    if stat <= KD_C1:
        return KD_P0 + (KD_P1 - KD_P0) * (stat - KD_C0) / (KD_C1 - KD_C0)
    if stat <= KD_C2:
        return KD_P1 + (KD_P2 - KD_P1) * (stat - KD_C1) / (KD_C2 - KD_C1)
    return KD_P2 + (KD_P3 - KD_P2) * (stat - KD_C2) / (KD_C3 - KD_C2)


@always_inline
def _xd(y: MutPointer[Float32, MutAnyOrigin], base: Int, dd: Int, t: Int) -> Float32:
    """Element t of the series differenced dd times (R's `diff` applied dd
    times: the second difference is the difference of first differences)."""
    if dd == 0:
        return y.unsafe_load(base + t)
    var d0 = y.unsafe_load(base + t + 1) - y.unsafe_load(base + t)
    if dd == 1:
        return d0
    var d1 = y.unsafe_load(base + t + 2) - y.unsafe_load(base + t + 1)
    return d1 - d0


@always_inline
def _block_sum[so: MutOrigin](
    sh: MutPointer[Float32, so, address_space = AddressSpace.SHARED], tid: Int, v: Float32,
) -> Float32:
    """The block's sum of `v` (a fixed pairwise tree), returned to every
    thread; `sh` is free again on return."""
    sh[tid] = v
    barrier()
    var w = KD_TPB // 2
    while w > 0:
        if tid < w:
            sh[tid] += sh[tid + w]
        barrier()
        w //= 2
    var r = sh[0]
    barrier()
    return r


def ndiffs_kpss_kernel(
    d_out: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32, d_max_in: Int32,
):
    """ARIMA_FAST_KPSS_D: one block per series, statsforecast's `ndiffs`
    loop:
        if is_constant(x): return 0
        dodiff = kpss_p(x) < 0.05
        while dodiff and d < max_d:
            d += 1; x = diff(x)
            if is_constant(x): return d
            dodiff = kpss_p(x) < 0.05
        return d
    with statsmodels' `kpss(x, "c", nlags=floor(3 sqrt(n) / 13))`:
    e = x - mean(x), stat = sum(cumsum(e)^2) / n^2 / s_hat, s_hat =
    (sum e^2 + 2 sum_{l=1..L} (1 - l / (L + 1)) sum_t e_t e_{t-l}) / n. A test
    statsmodels raises on (L >= n) is "no difference" (`run_tests`' except
    arm). Every decision is a block-wide sum, so the control flow is uniform.
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nobs = Int(nobs_in)
    var d_max = Int(d_max_in)
    var base = b * nobs
    var sh = stack_allocation[KD_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dd = 0
    while True:
        var n = nobs - dd
        if n < 2:
            break
        # is_constant: np.all(x[0] == x); and the sum for the mean
        var x0 = _xd(y, base, dd, 0)
        var miss = Float32(0.0)
        var part = Float32(0.0)
        var t = tid
        while t < n:
            var v = _xd(y, base, dd, t)
            if v != x0:
                miss += 1.0
            part += v
            t += KD_TPB
        var nmiss = _block_sum(sh, tid, miss)
        if nmiss == Float32(0.0):
            break
        var mean = _block_sum(sh, tid, part) / Float32(n)
        var lags = Int(floor(Float32(3.0) * sqrt(Float32(n)) / Float32(13.0)))
        if lags >= n:
            break
        # cumsum(e): each thread a contiguous chunk, the chunk totals scanned
        # in shared memory (Hillis-Steele), then each chunk walked again
        var chunk = (n + KD_TPB - 1) // KD_TPB
        var lo = min(tid * chunk, n)
        var hi = min(lo + chunk, n)
        var local = Float32(0.0)
        for t2 in range(lo, hi):
            local += _xd(y, base, dd, t2) - mean
        sh[tid] = local
        barrier()
        var off = 1
        while off < KD_TPB:
            var acc = sh[tid]
            if tid >= off:
                acc += sh[tid - off]
            barrier()
            sh[tid] = acc
            barrier()
            off *= 2
        var run = Float32(0.0)
        if tid > 0:
            run = sh[tid - 1]
        barrier()
        var s2 = Float32(0.0)
        var lrv = Float32(0.0)
        var lp1 = Float32(lags + 1)
        for t2 in range(lo, hi):
            var e = _xd(y, base, dd, t2) - mean
            run += e
            s2 += run * run
            var cross = Float32(0.0)
            for l in range(1, lags + 1):
                if t2 - l < 0:
                    break
                cross += (Float32(1.0) - Float32(l) / lp1) * (_xd(y, base, dd, t2 - l) - mean)
            lrv += e * e + Float32(2.0) * e * cross
        var s2_tot = _block_sum(sh, tid, s2)
        var lrv_tot = _block_sum(sh, tid, lrv)
        var stat = s2_tot / (Float32(n) * lrv_tot)
        var pv = _kpss_pvalue(stat)
        var dodiff = pv == pv and pv < KD_ALPHA
        if not dodiff or dd >= d_max:
            break
        dd += 1
    if tid == 0:
        d_out.unsafe_store(b, Int32(dd))


def ndiffs_kpss(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Int32, MutUntrackedOrigin],
    bs: Int, nobs: Int, d_max: Int,
) raises -> Int:
    """ARIMA_FAST_KPSS_D: every series' statsforecast `ndiffs` d (alpha
    0.05, up to `d_max`) into `out_ptr`, one launch for the batch, one
    download. `y` is (bs, nobs), one series per row. Returns `bs`."""
    comptime if not ARIMA_FAST_KPSS_D:
        raise Error("AutoARIMA ndiffs was not compiled (MOJOLEARN_ARIMA_FAST_KPSS_D)")
    else:
        if bs < 1 or nobs < 1 or d_max < 0 or d_max > 2:
            raise Error("AutoARIMA ndiffs: invalid shape or d_max")
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var y = _upload_f32(ctx, y_ptr, bs * nobs)
        var out = ctx.enqueue_create_buffer[DType.int32](bs)
        ctx.enqueue_function[ndiffs_kpss_kernel](
            out.unsafe_ptr(), y.unsafe_ptr(), Int32(nobs), Int32(d_max),
            grid_dim=(bs, 1, 1), block_dim=(KD_TPB, 1, 1),
        )
        var res = List[Int32](length=bs, fill=Int32(0))
        ctx.enqueue_copy(dst_ptr=res.unsafe_ptr(), src_buf=out)
        ctx.synchronize()
        _write_list_i32(out_ptr, res, 0)
        _ = out^
        _ = y^
        _ = ctx^
        return bs
