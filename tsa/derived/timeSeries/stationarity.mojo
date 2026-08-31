# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The batched KPSS stationarity test (Kwiatkowski et al. 1992).

PORT OF `cuml/cpp/src_prims/timeSeries/stationarity.cuh` at cuML 265b9da6
(v26.08.00): `s2B_accumulation_kernel` (:81-102),
`kpss_stationarity_check_kernel` (:121-160), `_kpss_test` (:191-281),
`kpss_test` (:300-336). COPY, DO NOT IMPROVE. The RAFT primitives that file
calls -- `raft::stats::mean`, `raft::linalg::matrixVectorOp`,
`raft::linalg::reduce` (sum, and sum of `L2Op` squares),
`thrust::inclusive_scan_by_key` -- are written here at their call sites,
as `core/column_stats.mojo` does for `pca_fit`; RAFT is not mirrored file
for file in this tree.

THE ALGORITHM, in their order (`stationarity.cuh:205-280`):

    y_means[b]  = sum_t y[b,t] * (1/n)                    raft::stats::mean
    y_cent      = y - y_means[b]                           matrixVectorOp
    s2A[b]      = sum_t y_cent^2                           reduce<L2Op>
    lags        = ceil(12 * (n/100)^0.25)                  host, Schwert 1989
    acc[b,t]    = sum_{k=1..lags, t<n-k} w(k) y_cent[t] y_cent[t+k]
                  with w(k) = coeff_a * k + coeff_b        s2B_accumulation_kernel
    s2B[b]      = sum_t acc[b,t]                           reduce
    acc[b,t]    = sum_{u<=t} y_cent[b,u]                   inclusive_scan_by_key
    eta[b]      = sum_t acc[b,t]^2                         reduce<L2Op>
    stat        = (eta/n^2) / (s2A/n + s2B)                check kernel
    pvalue      = table 1 of the paper, linearly interpolated
    result[b]   = pvalue > pval_threshold

Layout is theirs: column-major, series in columns, series `b` contiguous at
`[b*n, (b+1)*n)`. Precision: their `DataT` is `float` or `double`; this is
the `float` instantiation (`stationarity.pyx:93`), and `double` is not
offered because Metal has no Float64 (`arima/README.md`, DEVIATION 670).

=============================================================================
DEVIATION 671: THE FOLDS AND THE SCAN HAVE ONE SHAPE ON EVERY VENDOR
=============================================================================
THEIRS. Every per-series sum above is `raft::linalg::reduce<false,false>`
-> `coalescedReduction` (`raft/linalg/detail/coalesced_reduction-inl.cuh`),
whose `add_op` arm is `coalescedSumThinKernel` for `n_obs <= 512` (and for
larger `n_obs` when `batch_size >= 16 * numSMs`), else the CUB
`BlockReduce` "Medium" kernel: a PER-THREAD KAHAN-BABUSKA-NEUMAIER chain
over a strided subset, then `logicalWarpReduceVector` shuffles at a
logical warp width chosen from `n_obs` (2/4/8/16/32), then a warp-width
fold. The thread->element assignment, the compensation and the fold tree
are therefore functions of `n_obs`, of the policy table, of the hardware
warp width AND of the SM count (`coalescedReduction`'s dispatch reads
`raft::getMultiProcessorCount()`). The cumulative sum is Thrust's
decoupled-look-back scan, whose association tree is a function of the
Thrust tile size and the block count.

OURS. One block of `STATS_TPB` threads per series; thread `t` folds
elements `t, t + STATS_TPB, ...` serially ascending (`x*x + acc` is one
`identical_mul_add` under IDENTICAL), then `core/pinned_reduce.
pinned_block_sum` -- `block.sum` under FAST, a lane-width-independent
halving tree under IDENTICAL. The scan is one thread per series, serial
ascending, in both modes (there is no vendor scan primitive in this tree
to stand in under FAST, so the serial spelling IS the FAST arm). No
compensation: the Kahan term is part of their fold shape, not of the
statistic, and carrying it would make the pinned shape a third arithmetic
wearing the name of neither. Under IDENTICAL every sum here is a pure
function of the series bits and of `STATS_TPB` (pinned to one value on
every column by `lib_block_bounds_a_float_fold`); under FAST the library
fold decides. MEASURED: `tsa/original/stationarity_check.mojo::
check_kpss_device_equals_oracle` -- bitwise against the host replay of
this exact shape under IDENTICAL on the M4, and the fixture's
`check_kpss_fold_order_is_visible` shows a descending fold moves the bits,
so the gate has teeth.

=============================================================================
DEVIATION 672: 0/0 IN THE STATISTIC IS DEFINED, NOT COMPUTED
=============================================================================
THEIRS. A series that is constant after differencing has `y_cent == 0`
everywhere, so `s2A = s2B = eta = 0` and `kpss_stat = 0/0 = NaN`
(`stationarity.cuh:144`); every `>=`/`<` on NaN is false, so `pvalue`
keeps its seed `0.10` and the series is declared STATIONARY.
OURS. The same DECISION, reached without a computed NaN: when
`s2A/n + s2B == 0` the statistic is set to `0.0`, which interpolates to
the same `pvalue = 0.10` (`0.0 < crit_vals[0]`, no branch taken). The
statistic is a RECORDED STAGE of the card (`tsa.stat`) and a NaN payload
is vendor-specific (Apple 0x7fc00000, NVIDIA 0x7fffffff: ADDENDUM 11 /
IDENTITY_PATHS row 39), so a computed NaN may not sit in a hashed stage.
Non-finite INPUTS are refused by name before any stage
(`kpss_test` below), so the guard is reached only by the constant case.
MEASURED: `check_kpss_constant_series_is_defined` plants a constant
column and a column constant after one difference; both report
`stat = 0x00000000`, `stationary = True`, in both modes.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceil, isfinite

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from original.numerics import ftz, identical_mul_add
from tsa.derived.timeSeries.arima_helpers import prepare_data


#: Table 1, Kwiatkowski 1992 (`stationarity.cuh:131-132`), as `float`.
comptime KPSS_CRIT_0 = Float32(0.347)
comptime KPSS_CRIT_1 = Float32(0.463)
comptime KPSS_CRIT_2 = Float32(0.574)
comptime KPSS_CRIT_3 = Float32(0.739)
comptime KPSS_PVAL_0 = Float32(0.10)
comptime KPSS_PVAL_1 = Float32(0.05)
comptime KPSS_PVAL_2 = Float32(0.025)
comptime KPSS_PVAL_3 = Float32(0.01)

#: Elementwise launch width (the centering, the s2B accumulation). Their
#: `TPB = 256` with a `choose_block_dims` 2-D split: SCHEDULING, no fold.
comptime KPSS_ELEM_TPB = 256


def _crit(k: Int) -> Float32:
    if k == 0:
        return KPSS_CRIT_0
    if k == 1:
        return KPSS_CRIT_1
    if k == 2:
        return KPSS_CRIT_2
    return KPSS_CRIT_3


def _pval(k: Int) -> Float32:
    if k == 0:
        return KPSS_PVAL_0
    if k == 1:
        return KPSS_PVAL_1
    if k == 2:
        return KPSS_PVAL_2
    return KPSS_PVAL_3


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def series_sum_kernel[
    square: Bool
](
    out_v: MutPointer[Float32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    n_obs_in: Int32,
    scale: Float32,
):
    """`raft::linalg::reduce<false,false>` along one series, with `L2Op` when
    `square` and `mul_const_op(scale)` as the final op (`raft::stats::mean`
    passes `1/n`; the plain sums pass `1.0`, an exact multiply). One block
    per series, `block_dim == STATS_TPB`, every thread reaches the fold
    (DEVIATION 671)."""
    var n = Int(n_obs_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * n
    var acc = Float32(0.0)
    var t = tid
    while t < n:
        var x = ftz(data.unsafe_load(base + t))
        comptime if square:
            acc = ftz(identical_mul_add(x, x, acc))
        else:
            acc = ftz(acc + x)
        t += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(b, ftz(s0 * scale))


def center_kernel(
    y_cent: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    y_means: MutPointer[Float32, MutAnyOrigin],
    n_obs_in: Int32,
    n_total_in: Int32,
):
    """`raft::linalg::matrixVectorOp<false,true>(..., a - b)`
    (`stationarity.cuh:211-218`): one thread per cell."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_total_in):
        return
    var b = idx // Int(n_obs_in)
    var yv = ftz(y.unsafe_load(idx))
    var mv = ftz(y_means.unsafe_load(b))
    y_cent.unsafe_store(idx, ftz(yv - mv))


def s2B_accumulation_kernel(
    accumulator: MutPointer[Float32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    lags_in: Int32,
    n_obs_in: Int32,
    n_total_in: Int32,
    coeff_a: Float32,
    coeff_b: Float32,
):
    """`stationarity.cuh:81-102`: `acc[t] = sum_{k=1..lags, t<n-k} (a*k+b) *
    (y[t]*y[t+k])`, serial in `k` ascending per cell. `coeff_a * k + coeff_b`
    and `acc += coeff * dp` are both row-9 contractions (nvcc fuses them):
    `identical_mul_add`."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_total_in):
        return
    var n = Int(n_obs_in)
    var lags = Int(lags_in)
    var sample = idx % n
    var acc = Float32(0.0)
    var x0 = ftz(data.unsafe_load(idx))
    var k = 1
    while k <= lags and sample < n - k:
        var xk = ftz(data.unsafe_load(idx + k))
        var dp = ftz(x0 * xk)
        var coeff = ftz(identical_mul_add(coeff_a, Float32(k), coeff_b))
        acc = ftz(identical_mul_add(coeff, dp, acc))
        k += 1
    accumulator.unsafe_store(idx, acc)


def cumsum_by_series_kernel(
    accumulator: MutPointer[Float32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    n_obs_in: Int32,
    batch_size_in: Int32,
):
    """`thrust::inclusive_scan_by_key` over the series (`stationarity.cuh:
    255-262`): one thread per series, serial ascending (DEVIATION 671)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b >= Int(batch_size_in):
        return
    var n = Int(n_obs_in)
    var base = b * n
    var acc = Float32(0.0)
    for t in range(n):
        acc = ftz(acc + ftz(data.unsafe_load(base + t)))
        accumulator.unsafe_store(base + t, acc)


def kpss_stationarity_check_kernel(
    results: MutPointer[UInt8, MutAnyOrigin],
    stat_out: MutPointer[Float32, MutAnyOrigin],
    s2A: MutPointer[Float32, MutAnyOrigin],
    s2B: MutPointer[Float32, MutAnyOrigin],
    eta: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    n_obs_f: Float32,
    pval_threshold: Float32,
):
    """`stationarity.cuh:121-160`, one thread per series. `stat_out` is ours
    (the card's `tsa.stat` stage); theirs keeps the statistic in a
    register. The `0/0` guard is DEVIATION 672."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(batch_size_in):
        return
    var stat = kpss_stat_from_sums(
        ftz(s2A.unsafe_load(i)), ftz(s2B.unsafe_load(i)),
        ftz(eta.unsafe_load(i)), n_obs_f,
    )
    var pvalue = kpss_pvalue(stat)
    stat_out.unsafe_store(i, stat)
    results.unsafe_store(i, UInt8(1) if pvalue > pval_threshold else UInt8(0))


@always_inline
def kpss_stat_from_sums(
    s2A_in: Float32, s2B_in: Float32, eta_in: Float32, n_obs_f: Float32
) -> Float32:
    """`stationarity.cuh:141-144` with DEVIATION 672's guard. Shared by the
    kernel and the host oracle so the two cannot drift."""
    var s2Ai = ftz(s2A_in / n_obs_f)
    var n2 = ftz(n_obs_f * n_obs_f)
    var etai = ftz(eta_in / n2)
    var den = ftz(s2Ai + s2B_in)
    if den == Float32(0.0):
        return Float32(0.0)
    return ftz(etai / den)


@always_inline
def kpss_pvalue(kpss_stat: Float32) -> Float32:
    """`stationarity.cuh:146-155`: `pvals[k] + (pvals[k+1]-pvals[k]) *
    (stat - crit[k]) / (crit[k+1]-crit[k])`, which C++ binds as
    `p + ((dp * ds) / dc)` -- no multiply-add, so no contraction seam; one
    local per op for row 10."""
    var pvalue = KPSS_PVAL_0
    for k in range(3):
        var ck = _crit(k)
        var ck1 = _crit(k + 1)
        if kpss_stat >= ck and kpss_stat < ck1:
            var dp = ftz(_pval(k + 1) - _pval(k))
            var ds = ftz(kpss_stat - ck)
            var num = ftz(dp * ds)
            var dc = ftz(ck1 - ck)
            var q = ftz(num / dc)
            pvalue = ftz(_pval(k) + q)
    if kpss_stat >= KPSS_CRIT_3:
        pvalue = KPSS_PVAL_3
    return pvalue


# ---------------------------------------------------------------------------
# host side: the launch sequence, exactly theirs
# ---------------------------------------------------------------------------


def kpss_lags(n_obs: Int) -> Int:
    """`stationarity.cuh:233-234`: `ceil(12.0 * pow(n_obs_f / 100.0, 0.25))`
    in double (the `float` instantiation promotes: `n_obs_f / 100.0` is a
    float-by-double division). Integer-valued, so a last-bit of `pow`
    cannot move it except at an exact boundary, where IEEE `pow` returns
    the exact value (`n = 100 -> 12`, `n = 1600 -> 24`)."""
    var n_obs_f = Float64(Float32(n_obs))
    var lags_f = ceil(12.0 * ((n_obs_f / 100.0) ** 0.25))
    return Int(lags_f)


def kpss_s2B_coefficients(n_obs: Int, lags: Int) -> Tuple[Float32, Float32]:
    """`stationarity.cuh:241-249`: `coeff_base = 2.0 / n_obs_f` (a double
    quotient stored in `DataT`), `coeff_a = -coeff_base / (lags_f + 1.0)`
    (float over double, stored as the kernel's `DataT` argument),
    `coeff_b = coeff_base`. Returns `(coeff_a, coeff_b)`."""
    var n_obs_f = Float32(n_obs)
    var coeff_base = Float32(2.0 / Float64(n_obs_f))
    var lags_f = Float32(lags)
    var coeff_a = Float32(Float64(-coeff_base) / (Float64(lags_f) + 1.0))
    return (coeff_a, coeff_base)


struct KpssScratch(Movable):
    """The device buffers `_kpss_test` allocates (`stationarity.cuh:206-265`),
    held in a struct so every one outlives the last launch that reads it
    (`[[mojo-buffer-freed-at-last-use]]`). The card records them as stages."""

    var y_means: DeviceBuffer[DType.float32]
    var y_cent: DeviceBuffer[DType.float32]
    var s2A: DeviceBuffer[DType.float32]
    var accumulator: DeviceBuffer[DType.float32]
    var s2B: DeviceBuffer[DType.float32]
    var eta: DeviceBuffer[DType.float32]
    var stat: DeviceBuffer[DType.float32]
    var results: DeviceBuffer[DType.uint8]
    var lags: Int

    def __init__(out self, ctx: DeviceContext, batch_size: Int, n_obs: Int) raises:
        var total = batch_size * n_obs
        self.y_means = ctx.enqueue_create_buffer[DType.float32](batch_size)
        self.y_cent = ctx.enqueue_create_buffer[DType.float32](total)
        self.s2A = ctx.enqueue_create_buffer[DType.float32](batch_size)
        self.accumulator = ctx.enqueue_create_buffer[DType.float32](total)
        self.s2B = ctx.enqueue_create_buffer[DType.float32](batch_size)
        self.eta = ctx.enqueue_create_buffer[DType.float32](batch_size)
        self.stat = ctx.enqueue_create_buffer[DType.float32](batch_size)
        self.results = ctx.enqueue_create_buffer[DType.uint8](batch_size)
        self.lags = 0


def _kpss_test(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    pval_threshold: Float32,
    elem_tpb: Int = KPSS_ELEM_TPB,
) raises -> KpssScratch:
    """`stationarity.cuh:191-281`, launch for launch. `d_y` is the
    (already differenced) batch, `batch_size * n_obs` values. `elem_tpb` is
    the elementwise block width (SCHEDULING; the launch-invariance gate
    varies it and the bytes must not move). The fold width is `STATS_TPB`
    and is not a parameter."""
    var sc = KpssScratch(ctx, batch_size, n_obs)
    var total = batch_size * n_obs
    var n_obs_f = Float32(n_obs)
    var elem_grid = (total + elem_tpb - 1) // elem_tpb
    var series_grid = (batch_size + elem_tpb - 1) // elem_tpb

    # Compute mean: `raft::stats::mean<false>(y_means, d_y, batch_size,
    # n_obs, false)` = reduce then `mul_const_op(1/n)`; the ratio is a host
    # `OutType(1) / OutType(N)` (`raft/stats/detail/mean.cuh`).
    var ratio = Float32(1.0) / n_obs_f
    comptime sum_kernel = series_sum_kernel[False]
    comptime sumsq_kernel = series_sum_kernel[True]
    ctx.enqueue_function[sum_kernel](
        sc.y_means.unsafe_ptr(), d_y.unsafe_ptr(), Int32(n_obs), ratio,
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    # Center the data around its mean
    ctx.enqueue_function[center_kernel](
        sc.y_cent.unsafe_ptr(), d_y.unsafe_ptr(), sc.y_means.unsafe_ptr(),
        Int32(n_obs), Int32(total),
        grid_dim=(elem_grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    # First sum in eq. 10 (first part of s^2)
    ctx.enqueue_function[sumsq_kernel](
        sc.s2A.unsafe_ptr(), sc.y_cent.unsafe_ptr(), Int32(n_obs), Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    # From Kwiatkowski et al. referencing Schwert (1989)
    var lags = kpss_lags(n_obs)
    sc.lags = lags
    var coeffs = kpss_s2B_coefficients(n_obs, lags)
    # Second sum in eq. 10 (second part of s^2)
    ctx.enqueue_function[s2B_accumulation_kernel](
        sc.accumulator.unsafe_ptr(), sc.y_cent.unsafe_ptr(), Int32(lags),
        Int32(n_obs), Int32(total), coeffs[0], coeffs[1],
        grid_dim=(elem_grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    ctx.enqueue_function[sum_kernel](
        sc.s2B.unsafe_ptr(), sc.accumulator.unsafe_ptr(), Int32(n_obs),
        Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    # Cumulative sum (inclusive scan with + operator), recycling `accumulator`
    # as theirs does: a second buffer is not needed because the scan reads
    # `y_cent` and the previous contents are dead after the s2B fold.
    ctx.enqueue_function[cumsum_by_series_kernel](
        sc.accumulator.unsafe_ptr(), sc.y_cent.unsafe_ptr(), Int32(n_obs),
        Int32(batch_size),
        grid_dim=(series_grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    # Eq. 11 (eta)
    ctx.enqueue_function[sumsq_kernel](
        sc.eta.unsafe_ptr(), sc.accumulator.unsafe_ptr(), Int32(n_obs),
        Float32(1.0),
        grid_dim=(batch_size, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    # Decide whether each series is stationary based on s^2 and eta
    ctx.enqueue_function[kpss_stationarity_check_kernel](
        sc.results.unsafe_ptr(), sc.stat.unsafe_ptr(), sc.s2A.unsafe_ptr(),
        sc.s2B.unsafe_ptr(), sc.eta.unsafe_ptr(), Int32(batch_size), n_obs_f,
        pval_threshold,
        grid_dim=(series_grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    return sc^


@fieldwise_init
struct KpssResult(Movable):
    """What `kpss_test` hands back: the differenced series it tested (their
    `diff_buffer`, or the input when `d + D == 0`), its length, and the
    scratch whose `results`/`stat` are the answer."""

    var y_diff: DeviceBuffer[DType.float32]
    var n_obs_diff: Int
    var scratch: KpssScratch


def kpss_test(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32 = 0.05,
    elem_tpb: Int = KPSS_ELEM_TPB,
) raises -> KpssResult:
    """`stationarity.cuh:300-336`: refuse `n_obs <= d + s*D` by name,
    difference with `prepare_data`, run `_kpss_test` on the result.

    NON-FINITE INPUTS ARE REFUSED BY NAME HERE, before any recorded stage
    (ADDENDUM 11): `y` must be finite. Theirs lets NaN flow (the Python
    surface passes `ensure_all_finite=False`) and the statistic becomes NaN.
    """
    if batch_size < 1:
        raise Error("kpss_test: batch_size must be >= 1 (batch_size=" + String(batch_size) + ")")
    var d_sD = d + s * D
    if n_obs <= d_sD:
        raise Error(
            "stationarity: n_obs (" + String(n_obs)
            + ") must be greater than d + s*D (" + String(d_sD) + ")"
        )
    var n_obs_diff = n_obs - d_sD
    _refuse_non_finite(ctx, d_y, batch_size * n_obs, "y")
    var y_diff: DeviceBuffer[DType.float32]
    if d == 0 and D == 0:
        # their `d_y_diff = d_y`; a COPY here because the result struct owns
        # its buffer and Mojo will not alias the caller's
        y_diff = ctx.enqueue_create_buffer[DType.float32](batch_size * n_obs)
        var src = d_y.create_sub_buffer[DType.float32](0, batch_size * n_obs)
        ctx.enqueue_copy(dst_buf=y_diff, src_buf=src)
    else:
        y_diff = ctx.enqueue_create_buffer[DType.float32](batch_size * n_obs_diff)
        prepare_data(ctx, y_diff, d_y, batch_size, n_obs, d, D, s)
    var sc = _kpss_test(ctx, y_diff, batch_size, n_obs_diff, pval_threshold, elem_tpb)
    return KpssResult(y_diff=y_diff^, n_obs_diff=n_obs_diff, scratch=sc^)


def _refuse_non_finite(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int, name: String
) raises:
    """Host scan of the input: a NaN or an infinity anywhere is a refusal
    carrying the parameter name and the offending index."""
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[DType.float32](0, n)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var p = h.unsafe_ptr()
    for i in range(n):
        var v = p.unsafe_load(i)
        if not isfinite(v):
            raise Error(
                "kpss_test: " + name + " contains a non-finite value at index "
                + String(i) + "; missing or infinite observations are refused by name"
            )
    _ = h^


def download_results(
    ctx: DeviceContext, res: KpssResult, batch_size: Int
) raises -> Tuple[List[Bool], List[Float32]]:
    """Read `(stationary flags, statistics)` back to the host."""
    var hr = ctx.enqueue_create_host_buffer[DType.uint8](batch_size)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](batch_size)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=res.scratch.results)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=res.scratch.stat)
    ctx.synchronize()
    var flags = List[Bool]()
    var stats = List[Float32]()
    for i in range(batch_size):
        flags.append(hr.unsafe_ptr().unsafe_load(i) != 0)
        stats.append(hs.unsafe_ptr().unsafe_load(i))
    _ = hr^
    _ = hs^
    return (flags^, stats^)
