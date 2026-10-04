# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple `select_d` (lane/apple-fast-select, 2026-10-02): every KPSS
round on the device, one download.

FAST ON APPLE ONLY, the FAST + Apple default (`-D MOJOLEARN_SELECT_D_OFF` turns it off); tsa/impl/auto_arima.mojo
gates the one call on `SELECT_D_FAST` below, so the IDENTICAL binding, the
other vendors and the host compile the host-controlled loop unchanged.

`select_d` (tsa/impl/auto_arima.mojo) runs `kpss_test` once per candidate
order d and reads every round back: each round is a host scan of the whole
input for non-finite values (a download and a wait), eight launches, a
wait, then the flags downloaded and masked on the host (M3 0.8.34:
select-d taxi-hourly 7.0 ms, statsmodels 3.4 ms). Here the non-finite scan
is one kernel writing one flag word, every round's differencing and KPSS
launches go on the stream back to back with no wait between rounds (the
kernels are the primitive's own, tsa/impl/timeSeries/stationarity.mojo, on
one workspace), a kernel picks each series' first stationary order, and the
chosen orders and the flag come back together. The refusal of a non-finite
input is raised after the download, by the same name; the statistic of
every round is the primitive's, so the chosen orders are the same words.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from bindings.hostptr import copy_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from core.column_stats import STATS_TPB
from tsa.impl.timeSeries.arima_helpers import prepare_data
from tsa.impl.timeSeries.stationarity import (
    KPSS_ELEM_TPB, series_sum_kernel, center_kernel, s2B_accumulation_kernel, cumsum_by_series_kernel,
    kpss_stationarity_check_kernel, kpss_lags, kpss_s2B_coefficients,
    KPSS_SCAN_BLOCKED, KPSS_SCAN_TPB, cumsum_blocked_kernel,
)

#: FAST on Apple only. The FAST + Apple default since the M3 A/B (select-d taxi-hourly
#: 7.8 -> 3.9 ms, -50%, digest identical); -D MOJOLEARN_SELECT_D_OFF turns it off
#: (-D MOJOLEARN_SELECT_D is now harmless).
comptime SELECT_D_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_SELECT_D_OFF"]()
)

#: lane/fam-timeseries (2026-10-04), IDENTICAL on every vendor: `select_d`
#: takes `select_d_fast` (every KPSS round queued on the device, the first
#: stationary order chosen by a kernel, ONE download and ONE wait) instead of
#: one `kpss_test` + scan + download per candidate order. The launches are
#: the primitive's own kernels, so the chosen orders are the same words; a
#: refusal falls back to the host-controlled loop, which raises it by its
#: own name and index (`tsa/impl/auto_arima.mojo`).
#: `-D MOJOLEARN_IDN_SELECT_D_OFF=1` (or MOJOLEARN_IDN_ALL_OFF) restores the
#: loop. FAST is unchanged.
comptime SELECT_D_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_SELECT_D_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: lane/fam-timeseries (2026-10-04), IDENTICAL on every vendor: `kpss_test`
#: from the binding as ONE wait (`kpss_one_wait` below: the upload, the
#: non-finite flag kernel, the differencing, `_kpss_test`'s launches and the
#: download queued back to back) instead of four (upload, non-finite scan,
#: the test, the download). Same kernels, same bits.
#: `-D MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF=1` (or MOJOLEARN_IDN_ALL_OFF) restores
#: the four-wait sequence. FAST is unchanged.
comptime KPSS_ONE_WAIT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def nonfinite_flag_kernel(flag: MutPointer[UInt32, MutAnyOrigin], y: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    """flag[0] = 1 when any of the n values is NaN or infinite (every such
    thread writes the same word; the flag starts 0)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        if not isfinite(y.unsafe_load(i)):
            flag.unsafe_store(0, UInt32(1))


def choose_d_kernel(chosen: MutPointer[Int32, MutAnyOrigin], res: MutPointer[UInt8, MutAnyOrigin],
                    batch_in: Int32, d_max_in: Int32):
    """One thread per series: the first order d (rounds ascending, res[d *
    batch + b] its stationary flag) at which the series tests stationary,
    else d_max (auto_arima's first-stationary rule)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch = Int(batch_in)
    if b >= batch:
        return
    var d_max = Int(d_max_in)
    var c = d_max
    for d_ in range(d_max):
        if res.unsafe_load(d_ * batch + b) != UInt8(0):
            c = d_
            break
    chosen.unsafe_store(b, Int32(c))


def select_d_fast(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    D: Int,
    s: Int,
    d_max: Int,
    pval_threshold: Float32,
) raises -> List[Int32]:
    """`select_d`'s answer (d per series in 0 .. d_max) with every round on
    the device and one download; the caller has checked d_max against D."""
    if batch_size < 1:
        raise Error("kpss_test: batch_size must be >= 1 (batch_size=" + String(batch_size) + ")")
    var total = batch_size * n_obs
    # the rounds' shapes and one workspace for all of them: per round the
    # differenced series, the centred series, the accumulator (3 x batch x
    # n_diff) and y_means, s2A, s2B, eta, stat (5 x batch)
    var words = 0
    for d_ in range(d_max):
        var d_sD = d_ + s * D
        if n_obs <= d_sD:
            raise Error(
                "stationarity: n_obs (" + String(n_obs)
                + ") must be greater than d + s*D (" + String(d_sD) + ")"
            )
        words += 3 * batch_size * (n_obs - d_sD) + 5 * batch_size
    var w = ctx.enqueue_create_buffer[DType.float32](words if words > 0 else 1)
    var res = ctx.enqueue_create_buffer[DType.uint8](batch_size * d_max if d_max > 0 else 1)
    var flag = ctx.enqueue_create_buffer[DType.uint32](1)
    var chosen = ctx.enqueue_create_buffer[DType.int32](batch_size)
    ctx.enqueue_memset(flag, UInt32(0))
    ctx.enqueue_function[nonfinite_flag_kernel](
        flag.unsafe_ptr(), d_y.unsafe_ptr(), Int32(total),
        grid_dim=((total + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    comptime sum_kernel = series_sum_kernel[False]
    comptime sumsq_kernel = series_sum_kernel[True]
    var off = 0
    for d_ in range(d_max):
        var nd = n_obs - d_ - s * D
        var tot = batch_size * nd
        var y_at = off
        var cent_at = off + tot
        var acc_at = off + 2 * tot
        var means_at = off + 3 * tot
        var s2a_at = means_at + batch_size
        var s2b_at = s2a_at + batch_size
        var eta_at = s2b_at + batch_size
        var stat_at = eta_at + batch_size
        off += 3 * tot + 5 * batch_size
        var y_diff = w.create_sub_buffer[DType.float32](y_at, tot)
        if d_ == 0 and D == 0:
            # the round tests the input itself: a device-to-device copy into the workspace
            var src = d_y.create_sub_buffer[DType.float32](0, tot)
            ctx.enqueue_copy(dst_buf=y_diff, src_buf=src)
        else:
            prepare_data(ctx, y_diff, d_y, batch_size, n_obs, d_, D, s)
        # `_kpss_test`'s launches (tsa/impl/timeSeries/stationarity.mojo), on the workspace
        var nd_f = Float32(nd)
        var ratio = Float32(1.0) / nd_f
        var elem_grid = (tot + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
        var series_grid = (batch_size + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
        var wp = w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_function[sum_kernel](
            wp + means_at, wp + y_at, Int32(nd), ratio,
            grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        ctx.enqueue_function[center_kernel](
            wp + cent_at, wp + y_at, wp + means_at, Int32(nd), Int32(tot),
            grid_dim=(elem_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[sumsq_kernel](
            wp + s2a_at, wp + cent_at, Int32(nd), Float32(1.0),
            grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        var lags = kpss_lags(nd)
        var coeffs = kpss_s2B_coefficients(nd, lags)
        ctx.enqueue_function[s2B_accumulation_kernel](
            wp + acc_at, wp + cent_at, Int32(lags), Int32(nd), Int32(tot), coeffs[0], coeffs[1],
            grid_dim=(elem_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[sum_kernel](
            wp + s2b_at, wp + acc_at, Int32(nd), Float32(1.0),
            grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        comptime if KPSS_SCAN_BLOCKED:
            # IDENTICAL (lane/fam-timeseries): `_kpss_test`'s blocked scan
            ctx.enqueue_function[cumsum_blocked_kernel](
                wp + acc_at, wp + cent_at, Int32(nd),
                grid_dim=(batch_size, 1, 1), block_dim=(KPSS_SCAN_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[cumsum_by_series_kernel](
                wp + acc_at, wp + cent_at, Int32(nd), Int32(batch_size),
                grid_dim=(series_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
            )
        ctx.enqueue_function[sumsq_kernel](
            wp + eta_at, wp + acc_at, Int32(nd), Float32(1.0),
            grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        ctx.enqueue_function[kpss_stationarity_check_kernel](
            res.unsafe_ptr() + d_ * batch_size, wp + stat_at, wp + s2a_at, wp + s2b_at, wp + eta_at,
            Int32(batch_size), nd_f, pval_threshold,
            grid_dim=(series_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
        )
        _ = y_diff^
    ctx.enqueue_function[choose_d_kernel](
        chosen.unsafe_ptr(), res.unsafe_ptr(), Int32(batch_size), Int32(d_max),
        grid_dim=((batch_size + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    # the one download: the chosen orders and the non-finite flag
    var hc = ctx.enqueue_create_host_buffer[DType.int32](batch_size)
    var hf = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=chosen)
    ctx.enqueue_copy(dst_ptr=hf.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    if hf.unsafe_ptr().unsafe_load(0) != UInt32(0):
        raise Error(
            "kpss_test: y contains a non-finite value; missing or infinite observations are refused by name"
        )
    var out = List[Int32]()
    for b in range(batch_size):
        out.append(hc.unsafe_ptr().unsafe_load(b))
    _ = hc^
    _ = hf^
    _ = chosen^
    _ = flag^
    _ = res^
    _ = w^
    return out^


def kpss_one_wait(
    ctx: DeviceContext,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    stat_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32,
) raises -> Bool:
    """`kpss_test` of the caller's series at (d, D, s) with ONE wait
    (KPSS_ONE_WAIT; lane/fam-timeseries). Writes `batch_size` flags and
    statistics and returns True. Returns False, with nothing written, when
    the call is one the primitive refuses (a shape or differencing refusal,
    or a non-finite input): the caller then runs the primitive, which raises
    the refusal by its own name and index. The launches are `_kpss_test`'s
    (tsa/impl/timeSeries/stationarity.mojo) on one workspace."""
    var d_sD = d + s * D
    if batch_size < 1 or d < 0 or D < 0 or d + D > 2 or (D > 0 and s < 2) or n_obs <= d_sD:
        return False
    var total = batch_size * n_obs
    var nd = n_obs - d_sD
    var tot = batch_size * nd
    # one workspace: the differenced series, the centred series, the
    # accumulator (3 x batch x nd) and y_means, s2A, s2B, eta, stat (5 x batch)
    var cent_at = tot
    var acc_at = 2 * tot
    var means_at = 3 * tot
    var s2a_at = means_at + batch_size
    var s2b_at = s2a_at + batch_size
    var eta_at = s2b_at + batch_size
    var stat_at = eta_at + batch_size
    var d_y = ctx.enqueue_create_buffer[DType.float32](total)
    var h_y = ctx.enqueue_create_host_buffer[DType.float32](total)
    copy_f32(y_ptr, h_y.unsafe_ptr(), total)
    ctx.enqueue_copy(dst_buf=d_y, src_ptr=h_y.unsafe_ptr())
    var w = ctx.enqueue_create_buffer[DType.float32](3 * tot + 5 * batch_size)
    var res = ctx.enqueue_create_buffer[DType.uint8](batch_size)
    var flag = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.enqueue_memset(flag, UInt32(0))
    ctx.enqueue_function[nonfinite_flag_kernel](
        flag.unsafe_ptr(), d_y.unsafe_ptr(), Int32(total),
        grid_dim=((total + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    var y_diff = w.create_sub_buffer[DType.float32](0, tot)
    # d + D == 0 is `prepare_data`'s device-to-device copy
    prepare_data(ctx, y_diff, d_y, batch_size, n_obs, d, D, s)
    comptime sum_kernel = series_sum_kernel[False]
    comptime sumsq_kernel = series_sum_kernel[True]
    var nd_f = Float32(nd)
    var ratio = Float32(1.0) / nd_f
    var elem_grid = (tot + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
    var series_grid = (batch_size + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
    var wp = w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[sum_kernel](
        wp + means_at, wp, Int32(nd), ratio,
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    ctx.enqueue_function[center_kernel](
        wp + cent_at, wp, wp + means_at, Int32(nd), Int32(tot),
        grid_dim=(elem_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[sumsq_kernel](
        wp + s2a_at, wp + cent_at, Int32(nd), Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    var lags = kpss_lags(nd)
    var coeffs = kpss_s2B_coefficients(nd, lags)
    ctx.enqueue_function[s2B_accumulation_kernel](
        wp + acc_at, wp + cent_at, Int32(lags), Int32(nd), Int32(tot), coeffs[0], coeffs[1],
        grid_dim=(elem_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[sum_kernel](
        wp + s2b_at, wp + acc_at, Int32(nd), Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    comptime if KPSS_SCAN_BLOCKED:
        ctx.enqueue_function[cumsum_blocked_kernel](
            wp + acc_at, wp + cent_at, Int32(nd),
            grid_dim=(batch_size, 1, 1), block_dim=(KPSS_SCAN_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[cumsum_by_series_kernel](
            wp + acc_at, wp + cent_at, Int32(nd), Int32(batch_size),
            grid_dim=(series_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
        )
    ctx.enqueue_function[sumsq_kernel](
        wp + eta_at, wp + acc_at, Int32(nd), Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    ctx.enqueue_function[kpss_stationarity_check_kernel](
        res.unsafe_ptr(), wp + stat_at, wp + s2a_at, wp + s2b_at, wp + eta_at,
        Int32(batch_size), nd_f, pval_threshold,
        grid_dim=(series_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    # the one download: the flags, the statistics and the non-finite word
    var hr = ctx.enqueue_create_host_buffer[DType.uint8](batch_size)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](batch_size)
    var hf = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var stat_view = w.create_sub_buffer[DType.float32](stat_at, batch_size)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=res)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=stat_view)
    ctx.enqueue_copy(dst_ptr=hf.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    var ok = hf.unsafe_ptr().unsafe_load(0) == UInt32(0)
    if ok:
        for b in range(batch_size):
            flags_ptr.unsafe_store(b, Int32(1) if hr.unsafe_ptr().unsafe_load(b) != UInt8(0) else Int32(0))
            stat_ptr.unsafe_store(b, hs.unsafe_ptr().unsafe_load(b))
    _ = stat_view^
    _ = y_diff^
    _ = hr^
    _ = hs^
    _ = hf^
    _ = h_y^
    _ = flag^
    _ = res^
    _ = w^
    _ = d_y^
    return ok
