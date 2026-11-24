# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GEMM filter of the brute epsilon pass against the exact pass, word for word (lane sp-dbscan-filter,
2026-11-20; `dbscan/impl/brute_eps_filter.mojo`, the default from DBF_MIN_FEATURES features since 2026-11-24).

For every unit and every threshold: `bef_eps_kernel` (the reference accumulation) writes the bit matrix and the
degrees of all n rows; the filtered pass (`dbf_prepare`, then `dbf_range_launch` over forced-small query ranges
so the range and tile seams are crossed) writes them again; every 64-bit word and every degree must be equal.
The kernels are called directly, so the check does not depend on the cost rule (its 1-feature unit runs the
filter below DBF_MIN_FEATURES, which the fit never takes but which must still be exact): it runs in the
ordinary device column build wherever the brute edge-free route exists (IDENTICAL, NVIDIA and AMD).

Units (the adversarial inputs): values on a 1/4 grid with thresholds that are exact pair distances (pairs AT
eps); duplicated rows with thresholds 0 and tiny; a large common offset (the centre matters; threshold 0 is
there because every row is at distance exactly 0 from itself, which the band never decides); magnitudes whose
centred norms pass the filter's cap (every pair rechecked); magnitudes whose squares underflow; one outlier
row in fifty dragging the mean; one feature; 1,000 features; continuous data. Each unit also states what it
must exercise: the recheck (some pair left open) and/or the filter (some pair decided without it).

Serial by nature: the units share one device context, whose stream is in order, and their host work is the
fixture fill and the word compare of at most a few hundred thousand values.
"""

from max.gpu.host import DeviceContext

from checks.native_tests.kit import download, upload_host_into
from checks.native_tests.testsuite_cases import require
from dbscan.impl.brute_edge_free import bef_eps_kernel, bef_words
from dbscan.impl.brute_eps_filter import (
    dbf_band_scale,
    dbf_prepare,
    dbf_range_launch,
)
from dbscan.impl.edge_free import IDN_DBSCAN_EDGE_FREE
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L2,
    EPS_MBLK,
    EPS_THREADS,
)
from gemm.checks.gemm_identical import GemmWorkspace


comptime KIND_GRID = 0
comptime KIND_DUPLICATES = 1
comptime KIND_OFFSET = 2
comptime KIND_HUGE = 3
comptime KIND_TINY = 4
comptime KIND_OUTLIERS = 5
comptime KIND_CONTINUOUS = 6


def _u01(row: Int, feature: Int) -> Float64:
    """A fixed value in [0, 1) per (row, feature)."""
    var z = UInt64(row) * 0x9E3779B97F4A7C15 + UInt64(feature + 1) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 31)) * 0x94D049BB133111EB
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _value(kind: Int, row: Int, feature: Int) -> Float32:
    if kind == KIND_GRID:
        # 0, 0.25, ..., 2.0: every pair distance is exact in float32
        return Float32(Float64(Int(_u01(row, feature) * 9.0)) * 0.25)
    if kind == KIND_DUPLICATES:
        return Float32(_u01(row % 37, feature) * 4.0)
    if kind == KIND_OFFSET:
        return Float32(10000.0 + _u01(row, feature))
    if kind == KIND_HUGE:
        return Float32((_u01(row, feature) - 0.5) * 1.0e18)
    if kind == KIND_TINY:
        return Float32((_u01(row, feature) - 0.5) * 1.0e-20)
    if kind == KIND_OUTLIERS:
        if row % 50 == 0:
            return Float32(_u01(row, feature) * 1.0e25)
        return Float32(_u01(row, feature))
    return Float32(_u01(row, feature) * 10.0)


def _unit(
    ctx: DeviceContext,
    name: String,
    kind: Int,
    n: Int,
    k: Int,
    g_rows: Int,
    thresholds: List[Float32],
    want_recheck: Bool,
    want_decided: Bool,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n * k)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    ctx.synchronize()
    for i in range(n):
        for f in range(k):
            hx.unsafe_ptr().unsafe_store(i * k + f, _value(kind, i, f))
    upload_host_into[DType.float32](ctx, x, hx)

    var n_words = bef_words(n)
    var bm_exact = ctx.enqueue_create_buffer[DType.uint64](n_words * n)
    var bm_filter = ctx.enqueue_create_buffer[DType.uint64](n_words * n)
    var deg_exact = ctx.enqueue_create_buffer[DType.int32](n)
    var deg_filter = ctx.enqueue_create_buffer[DType.int32](n)
    var bm_none = ctx.enqueue_create_buffer[DType.uint64](1)
    var unc = ctx.enqueue_create_buffer[DType.int32](1)
    var xc = ctx.enqueue_create_buffer[DType.float32](n * k)
    var nc = ctx.enqueue_create_buffer[DType.float32](n)
    var g = ctx.enqueue_create_buffer[DType.float32](g_rows * n)
    var gws = GemmWorkspace(ctx)
    dbf_prepare(ctx, x, xc, nc, n, k)
    var band = dbf_band_scale(k)
    var xa = x.create_sub_buffer[DType.float32](0, n * k)
    var grid_rows = (n + EPS_MBLK - 1) // EPS_MBLK

    var rechecked_total = 0
    var decided_total = 0
    for ti in range(len(thresholds)):
        var thresh = thresholds[ti]
        ctx.enqueue_memset(deg_exact, Int32(0))
        ctx.enqueue_memset(deg_filter, Int32(0))
        ctx.enqueue_memset(unc, Int32(0))
        # the exact pass: the reference kernel over all n rows, degrees then bits
        ctx.enqueue_function[bef_eps_kernel[DBSCAN_METRIC_L2, False]](
            bm_none.unsafe_ptr(), deg_exact.unsafe_ptr(), xa.unsafe_ptr(),
            x.unsafe_ptr(), Int32(n), Int32(n), Int32(k), thresh,
            grid_dim=(n_words, grid_rows, 1), block_dim=(EPS_THREADS, 1, 1),
        )
        ctx.enqueue_function[bef_eps_kernel[DBSCAN_METRIC_L2, True]](
            bm_exact.unsafe_ptr(), deg_exact.unsafe_ptr(), xa.unsafe_ptr(),
            x.unsafe_ptr(), Int32(n), Int32(n), Int32(k), thresh,
            grid_dim=(n_words, grid_rows, 1), block_dim=(EPS_THREADS, 1, 1),
        )
        # the filtered pass, in ranges of g_rows queries; the bits pass counts its rechecked pairs
        var st = 0
        while st < n:
            var m = min(n - st, g_rows)
            dbf_range_launch[DBSCAN_METRIC_L2, False](
                ctx, gws, g, xc, nc, x, bm_none, deg_filter, unc, st, m, n, k,
                m, 0, thresh, band,
            )
            dbf_range_launch[DBSCAN_METRIC_L2, True, True](
                ctx, gws, g, xc, nc, x, bm_filter, deg_filter, unc, st, m, n,
                k, n, st, thresh, band,
            )
            st += m
        var words_exact = download[DType.uint64](ctx, bm_exact, n_words * n)
        var words_filter = download[DType.uint64](ctx, bm_filter, n_words * n)
        var d_exact = download[DType.int32](ctx, deg_exact, n)
        var d_filter = download[DType.int32](ctx, deg_filter, n)
        var u = download[DType.int32](ctx, unc, 1)
        var bad_words = 0
        for i in range(n_words * n):
            if words_exact[i] != words_filter[i]:
                bad_words += 1
        var bad_deg = 0
        var inside = 0
        for i in range(n):
            if d_exact[i] != d_filter[i]:
                bad_deg += 1
            inside += Int(d_exact[i])
        var rechecked = Int(u[0])
        print(
            "dbscan_eps_filter " + name + " n " + String(n) + " k " + String(k)
            + " thresh " + String(Float64(thresh)) + " inside " + String(inside)
            + " of " + String(n * n) + " rechecked " + String(rechecked)
            + " differing words " + String(bad_words) + " degrees "
            + String(bad_deg)
        )
        require(
            bad_words == 0 and bad_deg == 0,
            "dbscan_eps_filter " + name + ": the filtered pass differs from the exact pass at thresh "
            + String(Float64(thresh)) + ": " + String(bad_words) + " words, " + String(bad_deg) + " degrees",
        )
        require(
            rechecked >= 0 and rechecked <= n * n,
            "dbscan_eps_filter " + name + ": rechecked count " + String(rechecked) + " outside [0, n^2]",
        )
        rechecked_total += rechecked
        decided_total += n * n - rechecked
    require(
        (not want_recheck) or rechecked_total > 0,
        "dbscan_eps_filter " + name + ": no pair took the exact recheck; the unit does not exercise it",
    )
    require(
        (not want_decided) or decided_total > 0,
        "dbscan_eps_filter " + name + ": the filter decided no pair; the unit does not exercise it",
    )
    _ = x^
    _ = hx^
    _ = xa^
    _ = bm_exact^
    _ = bm_filter^
    _ = deg_exact^
    _ = deg_filter^
    _ = bm_none^
    _ = unc^
    _ = xc^
    _ = nc^
    _ = g^
    _ = gws^


def run_checks() raises:
    comptime if IDN_DBSCAN_EDGE_FREE:
        var ctx = DeviceContext()
        # pairs AT eps: 0, 1/16, 1 and 2 + 1/16 are exact squared distances on the 1/4 grid
        var t_grid: List[Float32] = [0.0, 0.0625, 1.0, 2.0625, 100.0]
        _unit(ctx, "grid-at-eps", KIND_GRID, 700, 3, 192, t_grid, True, True)
        var t_grid1: List[Float32] = [0.0, 0.0625, 0.25, 4.0]
        _unit(ctx, "one-feature", KIND_GRID, 700, 1, 192, t_grid1, True, True)
        var t_dup: List[Float32] = [0.0, 1.0e-30, 0.5]
        _unit(ctx, "duplicates", KIND_DUPLICATES, 650, 5, 128, t_dup, True, True)
        var t_off: List[Float32] = [0.0, 1.0, 2.6666667, 4.0]
        _unit(ctx, "offset", KIND_OFFSET, 700, 16, 192, t_off, True, True)
        var t_huge: List[Float32] = [1.0e30, 1.0e35, 3.0e37]
        _unit(ctx, "huge", KIND_HUGE, 500, 8, 192, t_huge, True, False)
        var t_tiny: List[Float32] = [0.0, 1.0e-42, 1.0e-39]
        _unit(ctx, "tiny", KIND_TINY, 500, 8, 192, t_tiny, True, False)
        var t_out: List[Float32] = [0.5, 1.5, 1.0e30]
        _unit(ctx, "outliers", KIND_OUTLIERS, 600, 8, 192, t_out, True, False)
        var t_cont: List[Float32] = [150.0, 266.66666, 400.0]
        _unit(ctx, "continuous-16", KIND_CONTINUOUS, 1000, 16, 320, t_cont, False, True)
        var t_wide: List[Float32] = [15500.0, 16666.666, 18000.0]
        _unit(ctx, "thousand-features", KIND_CONTINUOUS, 330, 1000, 128, t_wide, False, True)
        print("dbscan_eps_filter: PASS")
    else:
        print("dbscan_eps_filter: SKIP (no brute edge-free route in this build)")
