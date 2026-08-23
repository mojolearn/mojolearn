"""Gates for the KPSS test and the `d` choice (DEVIATIONS 671-672).

    tools/with_build_lock.sh     pixi run mojo run -I . tsa/mojo_only/stationarity_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . tsa/mojo_only/stationarity_check.mojo

Every printed line carries the mode the binary COMPILED in. The device-vs-
oracle bitwise lines ASSERT under IDENTICAL and print `RECORDED [FAST]`
under FAST (the fold is the library's there); refusals, the Float64
tolerance, the decisions and the integer `d` assert in both modes.

    check_kpss_device_equals_oracle     EVERY stage per cell -- y_diff,
                                        means, y_cent, s2A, the s2B
                                        accumulator, s2B, the cumulative
                                        sum, eta, the statistic, the flags
                                        -- at (d,D,s) = (0,0,0), (1,0,0),
                                        (2,0,0), (0,1,4), (1,1,4)
    check_kpss_matches_float64          statistic within 2e-3 relative of
                                        the paper's formulas in Float64
                                        (looser on the 2^-66 series, where
                                        FTZ is the point), decisions equal
                                        away from the critical values
    check_kpss_constant_series_is_defined  DEVIATION 672: stat bits
                                        0x00000000, stationary, both modes
    check_kpss_refuses_by_name          n_obs <= d + s*D; NaN; inf; D with
                                        s < 2; d + D > 2
    check_kpss_launch_invariant         elem block 256 / 64, 0 / 37 floats of
                                        poisoned padding, the same series in
                                        a batch of 8 and a batch of 3 in a
                                        different order: the bytes of every
                                        stage do not move
    check_kpss_fold_order_is_visible    a descending serial fold and a
                                        Float64 fold both move the statistic
                                        bits on the fixture: the bitwise
                                        gates above have teeth
    check_kpss_ftz_seam_is_reached      IDENTICAL only: the oracle WITHOUT
                                        `ftz` disagrees with the device on
                                        the 2^-66 series, so the flush is on
                                        the path and not decorative
    check_select_d                      auto_arima's d loop: AR(1) -> 0,
                                        random walk -> 1, linear trend -> 1
                                        (constant after one difference,
                                        DEVIATION 672), the constant -> 0;
                                        equal to the host replay of the loop
                                        over the oracle's flags

SABOTAGES PERFORMED (2026-08-23), each reverted; outputs in arima/README.md:
    (a) `series_sum_kernel`'s stride start ROTATED by the block id
        (`t = (tid + b) % STATS_TPB`): NO bit moves -- and that is a
        PROPERTY, not a missed check: a halving tree pairs slots
        `{j, j + step}` at every level, so a uniform rotation of the slots
        maps pairs to pairs and the sum is invariant. Recorded so nobody
        reads a rotation as evidence about this fold again.
    (a') `series_sum_kernel`'s per-thread stride folded DESCENDING
        (`t -= STATS_TPB` from the top): check_kpss_device_equals_oracle
        FAILS at `y_means` cell 0 (0x3c28d21d vs 0x3c28d220) under IDENTICAL
    (b) the oracle's fold tree swapped for a serial ascending sum of the
        partials: FAILS at `y_means` cell 0 (0x3d447329 vs 0x3d447325)
    (c) `s2B_accumulation_kernel`'s lag loop reversed (k descending):
        FAILS at `s2B` cell 0 (0x3ed8ff89 vs 0x3ed8ff8a) under IDENTICAL
    (d) `ftz` dropped from `s2B_accumulation_kernel`'s product: NO bit
        moves on Apple (the flush is inert on an FTZ backend; numerics.mojo
        says so of every pin on this column) -- recorded as the expected
        null, and check_kpss_ftz_seam_is_reached is the host-side evidence
        that the seam is on the path
"""

from std.math import abs
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from tsa.mojo_only.fixtures import (
    KpssFixture,
    bits32,
    count_cells_differ,
    download_f32,
    first_cell_differ,
    kpss_fixture,
    same_bits,
    sub_batch,
    upload_f32,
    upload_f32_padded,
)
from tsa.mojo_only.kpss_oracle import (
    KpssHostStages,
    kpss_host_f32,
    kpss_host_f64,
    pinned_fold_host,
)
from tsa.ported.timeSeries.stationarity import (
    KpssResult,
    download_results,
    kpss_lags,
    kpss_pvalue,
    kpss_stat_from_sums,
)
from tsa.ported.tsa.auto_arima import select_d
from tsa.ported.tsa.stationarity import kpss_test


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N_OBS = 520
comptime SALT = 1


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _gate(ok: Bool, what: String) raises:
    """Vendor-shaped claims (a bitwise device-vs-oracle line rides on the
    library fold under FAST): asserted under IDENTICAL, RECORDED under FAST.
    Integer, decision and host-only claims use `_assert` in both modes."""
    if ok:
        return
    comptime if IDENTICAL:
        raise Error(what)
    else:
        print("    RECORDED [FAST] " + what + " (vendor-shaped under FAST; not asserted)")


def _assert(ok: Bool, what: String) raises:
    if not ok:
        raise Error(what)


def _stage(
    ctx: DeviceContext,
    name: String,
    dev: DeviceBuffer[DType.float32],
    n: Int,
    host: List[Float32],
) raises:
    var got = download_f32(ctx, dev, n)
    var nd = count_cells_differ(got, host)
    print("    " + name + ": " + String(n) + " cells, " + String(nd) + " differ"
          + ("" if nd == 0 else " (first " + first_cell_differ(got, host) + ")"))
    _gate(nd == 0, name + " device != oracle")


def _run_both(
    ctx: DeviceContext, f: KpssFixture, d: Int, D: Int, s: Int
) raises -> Tuple[KpssResult, KpssHostStages]:
    var y = upload_f32(ctx, f.y)
    var res = kpss_test(ctx, y, f.batch_size, f.n_obs, d, D, s, 0.05)
    var host = kpss_host_f32(f.y, f.batch_size, f.n_obs, d, D, s, 0.05)
    _ = y^
    return (res^, host^)


def _compare_all_stages(
    ctx: DeviceContext, res: KpssResult, host: KpssHostStages, batch_size: Int
) raises:
    var n = host.n_obs_diff
    _assert(res.n_obs_diff == n, "n_obs_diff differs")
    _assert(res.scratch.lags == host.lags, "lags differ")
    _stage(ctx, "y_diff", res.y_diff, batch_size * n, host.y_diff)
    _stage(ctx, "y_means", res.scratch.y_means, batch_size, host.y_means)
    _stage(ctx, "y_cent", res.scratch.y_cent, batch_size * n, host.y_cent)
    _stage(ctx, "s2A", res.scratch.s2A, batch_size, host.s2A)
    _stage(ctx, "s2B", res.scratch.s2B, batch_size, host.s2B)
    # the accumulator holds the CUMULATIVE SUM by the end (theirs recycles
    # it, `stationarity.cuh:236-262`); the s2B partials are checked through
    # their fold and the scan through its cells
    _stage(ctx, "cumsum", res.scratch.accumulator, batch_size * n, host.cumsum)
    _stage(ctx, "eta", res.scratch.eta, batch_size, host.eta)
    _stage(ctx, "stat", res.scratch.stat, batch_size, host.stat)
    var rs = download_results(ctx, res, batch_size)
    var flags = rs[0].copy()
    var nflag = 0
    for b in range(batch_size):
        if flags[b] != host.stationary[b]:
            nflag += 1
    print("    flags: " + String(nflag) + " of " + String(batch_size) + " differ")
    _gate(nflag == 0, "stationary flags device != oracle")


def check_kpss_device_equals_oracle(ctx: DeviceContext) raises:
    print("check_kpss_device_equals_oracle [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    var orders = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 4), (1, 1, 4)]
    for o in orders:
        print("  (d, D, s) = (" + String(o[0]) + ", " + String(o[1]) + ", " + String(o[2]) + ")")
        var pair = _run_both(ctx, f, o[0], o[1], o[2])
        _compare_all_stages(ctx, pair[0], pair[1], f.batch_size)
        print("    lags = " + String(pair[1].lags))


def check_kpss_matches_float64(ctx: DeviceContext) raises:
    print("check_kpss_matches_float64 [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    var orders = [(0, 0, 0), (1, 0, 0), (0, 1, 4)]
    for o in orders:
        var y = upload_f32(ctx, f.y)
        var res = kpss_test(ctx, y, f.batch_size, f.n_obs, o[0], o[1], o[2], 0.05)
        var rs = download_results(ctx, res, f.batch_size)
        var flags = rs[0].copy()
        var stats = rs[1].copy()
        var r64 = kpss_host_f64(f.y, f.batch_size, f.n_obs, o[0], o[1], o[2])
        var st64 = r64[0].copy()
        var pv64 = r64[1].copy()
        for b in range(f.batch_size):
            var s32 = Float64(stats[b])
            var rel = abs(s32 - st64[b]) / (abs(st64[b]) + 1e-30)
            # the 2^-66 series is BUILT to flush: its centered products sit
            # at 2^-120 and the weighted terms at 2^-127, so on an FTZ
            # column a large share of s2B is zero by design and the Float64
            # reference (which keeps them) is not a tolerance target there
            # -- it is reported (measured 1.7e-3 at d = 0, 2.8e-1 at d = 1
            # on the M4), not asserted
            if st64[b] == 0.0:
                _assert(s32 == 0.0, "float64 stat is 0 but device is not, series " + String(b))
            else:
                print("    d=" + String(o[0]) + " D=" + String(o[1]) + " " + f.names[b]
                      + ": f32 " + String(s32) + " f64 " + String(st64[b])
                      + " rel " + String(rel) + (" (FTZ-dominated by construction: reported, not asserted)" if b == 3 else ""))
                if b != 3:
                    _assert(rel <= 2e-3, "statistic off the Float64 reference, series " + String(b))
            # decisions agree unless the statistic sits within 1e-2 of a
            # critical value (where Float32 may legitimately land on the
            # other side)
            var near = False
            var crit = [0.347, 0.463, 0.574, 0.739]
            for k in range(4):
                if abs(st64[b] - crit[k]) < 1e-2:
                    near = True
            if not near and b != 3:
                _assert((pv64[b] > 0.05) == flags[b],
                        "decision differs from the Float64 reference, series " + String(b))
        _ = y^


def check_kpss_constant_series_is_defined(ctx: DeviceContext) raises:
    print("check_kpss_constant_series_is_defined [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    # series 4 is constant at d = 0; series 5 (linear trend) is constant at d = 1
    for d in range(2):
        var y = upload_f32(ctx, f.y)
        var res = kpss_test(ctx, y, f.batch_size, f.n_obs, d, 0, 0, 0.05)
        var rs = download_results(ctx, res, f.batch_size)
        var flags = rs[0].copy()
        var stats = rs[1].copy()
        var b = 4 if d == 0 else 5
        print("    d=" + String(d) + " " + f.names[b] + ": stat " + bits32(stats[b])
              + " stationary " + String(flags[b]))
        _assert(bitcast[DType.uint32](stats[b]) == UInt32(0), "DEVIATION 672: stat is not +0.0")
        _assert(flags[b], "DEVIATION 672: the constant series must be stationary")
        _ = y^


def _expect_raise(ctx: DeviceContext, y: List[Float32], batch: Int, n_obs: Int,
                  d: Int, D: Int, s: Int, what: String) raises:
    var buf = upload_f32(ctx, y)
    var raised = False
    var msg = String("")
    try:
        var r = kpss_test(ctx, buf, batch, n_obs, d, D, s, 0.05)
        _ = r^
    except e:
        raised = True
        msg = String(e)
    print("    " + what + ": " + ("raised: " + msg if raised else "DID NOT RAISE"))
    _assert(raised, what + " was not refused")
    _ = buf^


def check_kpss_refuses_by_name(ctx: DeviceContext) raises:
    print("check_kpss_refuses_by_name [" + _mode_name() + "]")
    var f = kpss_fixture(16, SALT)
    _expect_raise(ctx, f.y, f.batch_size, f.n_obs, 1, 1, 16, "n_obs <= d + s*D")
    _expect_raise(ctx, f.y, f.batch_size, f.n_obs, 0, 1, 1, "D > 0 with s < 2")
    _expect_raise(ctx, f.y, f.batch_size, f.n_obs, 2, 1, 4, "d + D > 2")
    var ynan = f.y.copy()
    ynan[5] = Float32(0.0) / Float32(0.0)
    _expect_raise(ctx, ynan, f.batch_size, f.n_obs, 0, 0, 0, "NaN in y")
    var yinf = f.y.copy()
    yinf[7] = Float32(1.0) / Float32(0.0)
    _expect_raise(ctx, yinf, f.batch_size, f.n_obs, 0, 0, 0, "inf in y")


def check_kpss_launch_invariant(ctx: DeviceContext) raises:
    print("check_kpss_launch_invariant [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    var d = 1
    # reference: full batch, elem block 256, no padding
    var y0 = upload_f32(ctx, f.y)
    var r0 = kpss_test(ctx, y0, f.batch_size, f.n_obs, d, 0, 0, 0.05, 256)
    var stat0 = download_f32(ctx, r0.scratch.stat, f.batch_size)
    var cent0 = download_f32(ctx, r0.scratch.y_cent, f.batch_size * r0.n_obs_diff)
    var eta0 = download_f32(ctx, r0.scratch.eta, f.batch_size)
    # arm 1: elem block 64, buffer padded by 37 poisoned floats
    var y1 = upload_f32_padded(ctx, f.y, 37, Float32(12345.678))
    var r1 = kpss_test(ctx, y1, f.batch_size, f.n_obs, d, 0, 0, 0.05, 64)
    var stat1 = download_f32(ctx, r1.scratch.stat, f.batch_size)
    var cent1 = download_f32(ctx, r1.scratch.y_cent, f.batch_size * r1.n_obs_diff)
    var eta1 = download_f32(ctx, r1.scratch.eta, f.batch_size)
    var nd = count_cells_differ(stat0, stat1) + count_cells_differ(cent0, cent1) + count_cells_differ(eta0, eta1)
    print("    elem 256/no pad vs elem 64/pad 37 poisoned: " + String(nd) + " cells differ")
    _assert(nd == 0, "elementwise block width or padding moved the bytes")
    # arm 2: a different poison
    var y2 = upload_f32_padded(ctx, f.y, 37, Float32(-0.0))
    var r2 = kpss_test(ctx, y2, f.batch_size, f.n_obs, d, 0, 0, 0.05, 128)
    var stat2 = download_f32(ctx, r2.scratch.stat, f.batch_size)
    nd = count_cells_differ(stat0, stat2)
    print("    elem 128/pad 37 of -0.0: " + String(nd) + " cells differ")
    _assert(nd == 0, "poison moved the bytes")
    # arm 3: a batch of three series in another order
    var which: List[Int] = [3, 0, 7]
    var ys = sub_batch(f.y, f.n_obs, which)
    var y3 = upload_f32(ctx, ys)
    var r3 = kpss_test(ctx, y3, 3, f.n_obs, d, 0, 0, 0.05, 256)
    var stat3 = download_f32(ctx, r3.scratch.stat, 3)
    var eta3 = download_f32(ctx, r3.scratch.eta, 3)
    var cent3 = download_f32(ctx, r3.scratch.y_cent, 3 * r3.n_obs_diff)
    var nb = 0
    for k in range(3):
        var b = which[k]
        if not same_bits(stat3[k], stat0[b]):
            nb += 1
        if not same_bits(eta3[k], eta0[b]):
            nb += 1
        for t in range(r0.n_obs_diff):
            if not same_bits(cent3[k * r3.n_obs_diff + t], cent0[b * r0.n_obs_diff + t]):
                nb += 1
    print("    batch of 8 vs batch of 3 [3, 0, 7]: " + String(nb) + " cells differ")
    _assert(nb == 0, "batch composition moved the bytes")
    # arm 4: run twice in one process
    var y4 = upload_f32(ctx, f.y)
    var r4 = kpss_test(ctx, y4, f.batch_size, f.n_obs, d, 0, 0, 0.05, 256)
    var stat4 = download_f32(ctx, r4.scratch.stat, f.batch_size)
    nd = count_cells_differ(stat0, stat4)
    print("    run twice: " + String(nd) + " cells differ")
    _assert(nd == 0, "two runs differ")
    _ = y0^
    _ = y1^
    _ = y2^
    _ = y3^
    _ = y4^


def check_kpss_fold_order_is_visible(ctx: DeviceContext) raises:
    """Host only: the pinned fold against a descending serial fold and a
    Float64 fold of the same s2A, eta terms. At least one series' statistic
    must move, else the bitwise gates could not see a fold."""
    print("check_kpss_fold_order_is_visible [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    var host = kpss_host_f32(f.y, f.batch_size, f.n_obs, 0, 0, 0, 0.05)
    var n = host.n_obs_diff
    var moved_desc = 0
    var moved_f64 = 0
    for b in range(f.batch_size):
        # descending serial fold of the same squared terms
        var acc = Float32(0.0)
        var acc_e = Float32(0.0)
        var t = n - 1
        while t >= 0:
            var x = host.y_cent[b * n + t]
            acc = ftz(identical_mul_add(x, x, acc))
            var c = host.cumsum[b * n + t]
            acc_e = ftz(identical_mul_add(c, c, acc_e))
            t -= 1
        var st_desc = kpss_stat_from_sums(acc, host.s2B[b], acc_e, Float32(n))
        if not same_bits(st_desc, host.stat[b]):
            moved_desc += 1
        var a64 = 0.0
        var e64 = 0.0
        for u in range(n):
            var x = Float64(host.y_cent[b * n + u])
            a64 += x * x
            var c = Float64(host.cumsum[b * n + u])
            e64 += c * c
        var st_64 = kpss_stat_from_sums(Float32(a64), host.s2B[b], Float32(e64), Float32(n))
        if not same_bits(st_64, host.stat[b]):
            moved_f64 += 1
    print("    descending fold moves " + String(moved_desc) + " of " + String(f.batch_size)
          + " statistics; Float64 fold moves " + String(moved_f64))
    _assert(moved_desc > 0, "a descending fold does not move the statistic: the fixture cannot see a fold")
    _assert(moved_f64 > 0, "a Float64 fold does not move the statistic: the fixture cannot see a fold")


def check_kpss_ftz_seam_is_reached(ctx: DeviceContext) raises:
    """IDENTICAL only (under FAST the device flushes and the helper is a
    no-op, so the same comparison is the expected FAST disagreement): the
    s2A of the 2^-66 series replayed WITHOUT `ftz` differs from the device."""
    print("check_kpss_ftz_seam_is_reached [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    var res = kpss_test(ctx, y, f.batch_size, f.n_obs, 0, 0, 0, 0.05)
    var s2A_dev = download_f32(ctx, res.scratch.s2A, f.batch_size)
    var host = kpss_host_f32(f.y, f.batch_size, f.n_obs, 0, 0, 0, 0.05)
    var n = host.n_obs_diff
    var b = 3
    var partials = List[Float32]()
    for tid in range(STATS_TPB):
        var acc = Float32(0.0)
        var t = tid
        while t < n:
            var x = host.y_cent[b * n + t]
            acc = identical_mul_add(x, x, acc)  # NO ftz: gradual underflow kept
            t += STATS_TPB
        partials.append(acc)
    var s_noftz = pinned_fold_host(partials)
    print("    " + f.names[b] + ": device s2A " + bits32(s2A_dev[b]) + ", oracle with ftz "
          + bits32(host.s2A[b]) + ", oracle without ftz " + bits32(s_noftz))
    comptime if IDENTICAL:
        _assert(same_bits(s2A_dev[b], host.s2A[b]), "device s2A != oracle with ftz")
        _assert(not same_bits(s2A_dev[b], s_noftz),
                "the oracle without ftz matches the device: the flush seam is not reached")
    else:
        print("    RECORDED [FAST] (the host keeps subnormals; Apple flushes: "
              + ("differ" if not same_bits(s2A_dev[b], s_noftz) else "equal") + ")")
    _ = y^


def check_select_d(ctx: DeviceContext) raises:
    print("check_select_d [" + _mode_name() + "]")
    var f = kpss_fixture(N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    var d = select_d(ctx, y, f.batch_size, f.n_obs, 0, 0, 2, 0.05)
    # host replay of auto_arima.pyx's loop over the oracle's flags
    var want = List[Int32]()
    for _ in range(f.batch_size):
        want.append(Int32(2))
    var decided = List[Bool]()
    for _ in range(f.batch_size):
        decided.append(False)
    for d_ in range(2):
        var h = kpss_host_f32(f.y, f.batch_size, f.n_obs, d_, 0, 0, 0.05)
        for b in range(f.batch_size):
            if not decided[b] and h.stationary[b]:
                want[b] = Int32(d_)
                decided[b] = True
    var nd = 0
    for b in range(f.batch_size):
        print("    " + f.names[b] + ": d = " + String(d[b]) + " (host replay " + String(want[b]) + ")")
        if d[b] != want[b]:
            nd += 1
    _gate(nd == 0, "select_d differs from the host replay")
    _assert(d[0] == 0, "AR(1) phi=0.5 must take d = 0")
    _assert(d[1] == 1, "the random walk must take d = 1")
    _assert(d[4] == 0, "the constant must take d = 0 (DEVIATION 672)")
    _assert(d[5] == 1, "the linear trend must take d = 1 (constant after one difference)")
    _ = y^


def main() raises:
    print("== tsa/mojo_only/stationarity_check.mojo [" + _mode_name() + "] STATS_TPB=" + String(STATS_TPB) + " N_OBS=" + String(N_OBS) + " ==")
    var ctx = DeviceContext()
    check_kpss_device_equals_oracle(ctx)
    check_kpss_matches_float64(ctx)
    check_kpss_constant_series_is_defined(ctx)
    check_kpss_refuses_by_name(ctx)
    check_kpss_launch_invariant(ctx)
    check_kpss_fold_order_is_visible(ctx)
    check_kpss_ftz_seam_is_reached(ctx)
    check_select_d(ctx)
    print("ALL TSA CHECKS PASSED [" + _mode_name() + "]")
