# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple `select_d` (lane/apple-fast-select, 2026-10-02): every KPSS
round on the device, one download.

FAST ON APPLE ONLY, the FAST + Apple default; tsa/impl/auto_arima.mojo
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
from max.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from max.gpu.host import DeviceBuffer, DeviceContext
from core.hostptr import copy_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz
from core.column_stats import STATS_TPB
from tsa.impl.timeSeries.arima_helpers import prepare_data
from tsa.tsa_store import TsaSeriesIn, tsa_series_fill
from tsa.impl.timeSeries.stationarity import (
    KPSS_ELEM_TPB, series_sum_kernel, center_kernel, s2B_accumulation_kernel, cumsum_by_series_kernel,
    kpss_stationarity_check_kernel, kpss_lags, kpss_s2B_coefficients,
    KPSS_SCAN_BLOCKED, KPSS_SCAN_TPB, cumsum_blocked_kernel,
)

#: FAST on Apple only. The FAST + Apple default since the M3 A/B (select-d taxi-hourly
#: 7.8 -> 3.9 ms, -50%, digest identical) (-D MOJOLEARN_SELECT_D is harmless).
# TOMBSTONE: MOJOLEARN_SELECT_D_OFF (old path, EXP:552 KEPT default) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_SELECT_D_OFF.patch; record in docs/TOMBSTONES.md.
comptime SELECT_D_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU

#: lane/fam-timeseries (2026-10-04), IDENTICAL on every vendor: `select_d`
#: takes `select_d_fast` (every KPSS round queued on the device, the first
#: stationary order chosen by a kernel, ONE download and ONE wait) instead of
#: one `kpss_test` + scan + download per candidate order. The launches are
#: the primitive's own kernels, so the chosen orders are the same words; a
#: refusal falls back to the host-controlled loop, which raises it by its
#: own name and index (`tsa/impl/auto_arima.mojo`). FAST is unchanged.
# TOMBSTONE: MOJOLEARN_IDN_SELECT_D_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SELECT_D_OFF.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella, EXP:2438; every IDENTICAL default kept) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-36-neural-series-linear.patch; record in docs/TOMBSTONES.md.
comptime SELECT_D_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
#: lane/fam-timeseries (2026-10-04), IDENTICAL on every vendor: `kpss_test`
#: from the binding as ONE wait (`kpss_one_wait` below: the upload, the
#: non-finite flag kernel, the differencing, `_kpss_test`'s launches and the
#: download queued back to back) instead of four (upload, non-finite scan,
#: the test, the download). Same kernels, same bits. FAST is unchanged.
# TOMBSTONE: MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF.patch; record in docs/TOMBSTONES.md.
comptime KPSS_ONE_WAIT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


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
    # ONE device buffer (lane fb15-sync-counts, 2026-11-18; was four: the
    # workspace, the flags, the non-finite word and the chosen orders), laid
    # out in 32-bit words as
    #   [chosen batch | non-finite word 1 | round flags (batch * d_max bytes) | workspace]
    # so the chosen orders and the non-finite word come home in ONE download.
    # Per round the workspace holds the differenced series (not for the round
    # that tests the input itself: it reads `d_y` directly, where it copied
    # it device-to-device before), the centred series, the accumulator, and
    # y_means, s2A, s2B, eta, stat (5 x batch). The kernels and their inputs
    # are the same words, so the chosen orders are the same words.
    var flag_at = batch_size
    var res_at = batch_size + 1
    var res_words = (batch_size * d_max + 3) // 4
    var ws_at = res_at + res_words
    var words = ws_at
    for d_ in range(d_max):
        var d_sD = d_ + s * D
        if n_obs <= d_sD:
            raise Error(
                "stationarity: n_obs (" + String(n_obs)
                + ") must be greater than d + s*D (" + String(d_sD) + ")"
            )
        var tot_ = batch_size * (n_obs - d_sD)
        words += (2 if (d_ == 0 and D == 0) else 3) * tot_ + 5 * batch_size
    var buf = ctx.enqueue_create_buffer[DType.float32](words)
    var flag = buf.create_sub_buffer[DType.float32](flag_at, 1)
    var head = buf.create_sub_buffer[DType.float32](0, batch_size + 1)
    ctx.enqueue_memset(flag, Float32(0.0))
    var bp = buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var chosen_p = bp.bitcast[Int32]()
    var flag_p = (bp + flag_at).bitcast[UInt32]()
    var res_p = (bp + res_at).bitcast[UInt8]()
    var yp = d_y.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[nonfinite_flag_kernel](
        flag_p, yp, Int32(total),
        grid_dim=((total + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    comptime sum_kernel = series_sum_kernel[False]
    comptime sumsq_kernel = series_sum_kernel[True]
    var off = ws_at
    for d_ in range(d_max):
        var nd = n_obs - d_ - s * D
        var tot = batch_size * nd
        var direct = d_ == 0 and D == 0
        var y_at = off
        if not direct:
            off += tot
        var cent_at = off
        var acc_at = off + tot
        var means_at = off + 2 * tot
        var s2a_at = means_at + batch_size
        var s2b_at = s2a_at + batch_size
        var eta_at = s2b_at + batch_size
        var stat_at = eta_at + batch_size
        off += 2 * tot + 5 * batch_size
        var wp = bp
        # the round's series: the input itself, or its differences in the workspace
        var src = yp
        if not direct:
            var y_diff = buf.create_sub_buffer[DType.float32](y_at, tot)
            prepare_data(ctx, y_diff, d_y, batch_size, n_obs, d_, D, s)
            src = bp + y_at
            _ = y_diff^
        # `_kpss_test`'s launches (tsa/impl/timeSeries/stationarity.mojo), on the workspace
        var nd_f = Float32(nd)
        var ratio = Float32(1.0) / nd_f
        var elem_grid = (tot + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
        var series_grid = (batch_size + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
        ctx.enqueue_function[sum_kernel](
            wp + means_at, src, Int32(nd), ratio,
            grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        ctx.enqueue_function[center_kernel](
            wp + cent_at, src, wp + means_at, Int32(nd), Int32(tot),
            grid_dim=(elem_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
        )
        ctx.enqueue_function[sumsq_kernel](
            wp + s2a_at, wp + cent_at, Int32(nd), Float32(1.0),
            grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        var lags = kpss_lags(nd)
        var coeffs = kpss_s2B_coefficients(nd, lags)
        # TOMBSTONE: MOJOLEARN_C60_LAG4 (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
        # Restore: git apply experiments/removed/MOJOLEARN_C60_LAG4.expdel2-36-neural-series-linear.patch; record in docs/TOMBSTONES.md.
        var lag_tasks = tot
        ctx.enqueue_function[s2B_accumulation_kernel](
            wp + acc_at, wp + cent_at, Int32(lags), Int32(nd), Int32(tot), coeffs[0], coeffs[1],
            grid_dim=((lag_tasks + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
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
            res_p + d_ * batch_size, wp + stat_at, wp + s2a_at, wp + s2b_at, wp + eta_at,
            Int32(batch_size), nd_f, pval_threshold,
            grid_dim=(series_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
        )
    ctx.enqueue_function[choose_d_kernel](
        chosen_p, res_p, Int32(batch_size), Int32(d_max),
        grid_dim=((batch_size + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    # the one download: the chosen orders and the non-finite word, adjacent
    var hh = ctx.enqueue_create_host_buffer[DType.float32](batch_size + 1)
    ctx.enqueue_copy(dst_ptr=hh.unsafe_ptr(), src_buf=head)
    ctx.synchronize()
    var hp = hh.unsafe_ptr()
    if hp.bitcast[UInt32]().unsafe_load(flag_at) != UInt32(0):
        raise Error(
            "kpss_test: y contains a non-finite value; missing or infinite observations are refused by name"
        )
    var out = List[Int32]()
    for b in range(batch_size):
        out.append(hp.bitcast[Int32]().unsafe_load(b))
    _ = hh^
    _ = head^
    _ = flag^
    _ = buf^
    return out^


def kpss_one_wait(
    ctx: DeviceContext,
    src: TsaSeriesIn,
    flags_ptr: MutPointer[UInt8, MutUntrackedOrigin],
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
    var direct = d == 0 and D == 0
    # ONE device buffer and ONE host stage (lane fb15-sync-counts, 2026-11-18;
    # were four device buffers and four host buffers, a device-to-device copy
    # of the input and three downloads), in 32-bit words:
    #   [y total | non-finite word 1 | stat batch | flags (batch bytes) | workspace]
    # The upload carries y and a zero non-finite word (no memset); the ONE
    # download brings the non-finite word, the statistics and the flags home
    # together. The test at d = D = 0 reads y directly (it copied y into the
    # workspace device-to-device before). Same kernels on the same words.
    var flag_at = total
    var stat_at = total + 1
    var res_at = stat_at + batch_size
    var res_words = (batch_size + 3) // 4
    var ws_at = res_at + res_words
    # the workspace: the differenced series (not when the test reads y), the
    # centred series, the accumulator, then y_means, s2A, s2B, eta (4 x batch)
    var y_at = ws_at
    var cent_at = ws_at + (0 if direct else tot)
    var acc_at = cent_at + tot
    var means_at = acc_at + tot
    var s2a_at = means_at + batch_size
    var s2b_at = s2a_at + batch_size
    var eta_at = s2b_at + batch_size
    var words = eta_at + batch_size
    var buf = ctx.enqueue_create_buffer[DType.float32](words)
    var host = ctx.enqueue_create_host_buffer[DType.float32](res_at + res_words)
    var hp = host.unsafe_ptr()
    var up = buf.create_sub_buffer[DType.float32](0, total + 1)
    var tail = buf.create_sub_buffer[DType.float32](flag_at, res_at + res_words - flag_at)
    if src.resident():
        # Lane gd-resident-2: the series from the scope's resident slot, laid
        # out series-major on the device (tsa/tsa_store.mojo), and the zero
        # non-finite word: no host copy, no upload.
        var y_dst = buf.create_sub_buffer[DType.float32](0, total)
        tsa_series_fill(ctx, y_dst, src, batch_size, n_obs)
        var flag_dst = buf.create_sub_buffer[DType.float32](flag_at, 1)
        flag_dst.enqueue_fill(Float32(0.0))
        _ = y_dst^
        _ = flag_dst^
    else:
        copy_f32(src.host, hp, total)
        hp.unsafe_store(flag_at, Float32(0.0))
        ctx.enqueue_copy(dst_buf=up, src_ptr=hp)
    var wp = buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[nonfinite_flag_kernel](
        (wp + flag_at).bitcast[UInt32](), wp, Int32(total),
        grid_dim=((total + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    var src = wp
    if not direct:
        var d_y = buf.create_sub_buffer[DType.float32](0, total)
        var y_diff = buf.create_sub_buffer[DType.float32](y_at, tot)
        prepare_data(ctx, y_diff, d_y, batch_size, n_obs, d, D, s)
        src = wp + y_at
        _ = y_diff^
        _ = d_y^
    comptime sum_kernel = series_sum_kernel[False]
    comptime sumsq_kernel = series_sum_kernel[True]
    var nd_f = Float32(nd)
    var ratio = Float32(1.0) / nd_f
    var elem_grid = (tot + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
    var series_grid = (batch_size + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB
    ctx.enqueue_function[sum_kernel](
        wp + means_at, src, Int32(nd), ratio,
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    ctx.enqueue_function[center_kernel](
        wp + cent_at, src, wp + means_at, Int32(nd), Int32(tot),
        grid_dim=(elem_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    ctx.enqueue_function[sumsq_kernel](
        wp + s2a_at, wp + cent_at, Int32(nd), Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    var lags = kpss_lags(nd)
    var coeffs = kpss_s2B_coefficients(nd, lags)
    # TOMBSTONE: MOJOLEARN_C60_LAG4 (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
    # Restore: git apply experiments/removed/MOJOLEARN_C60_LAG4.expdel2-36-neural-series-linear.patch; record in docs/TOMBSTONES.md.
    var lag_tasks = tot
    ctx.enqueue_function[s2B_accumulation_kernel](
        wp + acc_at, wp + cent_at, Int32(lags), Int32(nd), Int32(tot), coeffs[0], coeffs[1],
        grid_dim=((lag_tasks + KPSS_ELEM_TPB - 1) // KPSS_ELEM_TPB, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
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
        (wp + res_at).bitcast[UInt8](), wp + stat_at, wp + s2a_at, wp + s2b_at, wp + eta_at,
        Int32(batch_size), nd_f, pval_threshold,
        grid_dim=(series_grid, 1, 1), block_dim=(KPSS_ELEM_TPB, 1, 1),
    )
    # the one download: the non-finite word, the statistics and the flags
    ctx.enqueue_copy(dst_ptr=hp + flag_at, src_buf=tail)
    ctx.synchronize()
    var ok = hp.bitcast[UInt32]().unsafe_load(flag_at) == UInt32(0)
    if ok:
        var hr = (hp + res_at).bitcast[UInt8]()
        for b in range(batch_size):
            flags_ptr.unsafe_store(b, UInt8(1) if hr.unsafe_load(b) != UInt8(0) else UInt8(0))
            stat_ptr.unsafe_store(b, hp.unsafe_load(stat_at + b))
    _ = up^
    _ = tail^
    _ = host^
    _ = buf^
    return ok
