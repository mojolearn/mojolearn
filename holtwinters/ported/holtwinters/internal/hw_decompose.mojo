"""cuML `cpp/src/holtwinters/internal/hw_decompose.cuh` (v26.08.00).

`conv1d`, the seasonal residual (their cuBLAS `geam` / RAFT `eltwiseDivide`,
both elementwise), `season_mean`, `batched_ls` and `stl_decomposition_gpu`:
the start-value decomposition over the first `start_periods` seasons.

THE IDENTITY CONTENT. Every reduction here is ONE THREAD PER SERIES walking
an index ascending -- `conv1d`'s filter sum over `i = 0 .. filter_size-1`,
`season_mean`'s per-phase sum over `k = i, i+f, ...`, its sum of phase
means over `i = 0 .. f-1`, and `batched_ls_solver`'s two dots over `i = 0 ..
trend_len-1` -- so the order is a pure function of (`frequency`,
`start_periods`), never of the launch. What moves between vendors is the
contraction (row 9) and the denormal policy (row 10), pinned below with
`identical_mul_add` + `ftz`; the divisions (`/ count`, `/ frequency`, `/
mean`, the multiplicative residual) are correctly rounded on every column
measured (row 10). The filter is host-built exactly as theirs:
`(Dtype)(1. / frequency)` (a float64 division rounded to float32, a basic
op, the same bits on every host) with the two ends halved for an even
frequency (exact).

`SAB_ROTATE_CONV` starts `conv1d`'s sum at `block_idx.x % filter_size` and
wraps: the same terms in a launch-dependent order. It MUST fail
device-vs-oracle and launch invariance (README).

============ DEVIATION 660 (2026-08-23): `batched_ls` WITHOUT cuSOLVER:
============ THE CLOSED-FORM PSEUDO-INVERSE OF [1, t] ======================
WHAT THEIRS DOES (`hw_decompose.cuh:153-257`): builds the CONSTANT design
matrix `A = [1, t]` (`t = 1 .. trend_len`), QR-factors it with cuSOLVER
`geqrf`, inverts the 2x2 `R` in a one-thread kernel, forms `Q` with
`orgqr`, multiplies `R^-1 Q^T` with cuBLAS `gemm` into `R1Qt` (2 x
trend_len), and only THEN touches the data: `batched_ls_solver_kernel`
dots each series' smoothed trend against the two rows of `R1Qt`.
`R^-1 Q^T` IS `pinv(A) = (A^T A)^-1 A^T` -- a function of `trend_len`
ALONE, not of any series -- computed by two CLOSED libraries whose float32
results are neither readable nor bit-reproducible across GPU models.
WHAT OURS DOES: `R1Qt` is written on the HOST in float64 from the closed
form of the same pseudo-inverse -- `tbar = (m+1)/2`, `Sxx = m (m^2 - 1) /
12`, slope row `w1[t] = (t - tbar) / Sxx`, intercept row `w0[t] = fma(-tbar,
w1[t], 1/m)` (one explicit fma so no host codegen can contract it
differently) -- then rounded to float32 and uploaded. The DATA-TOUCHING
step, `batched_ls_solver_kernel` (one thread per series, the two dots
ascending through `identical_mul_add` + `ftz`), is ported unchanged.
WHY: PORTING_RULES 0b-i: where their path calls a CLOSED library there is
nothing to port and the MAX equivalent is the substitute; but MAX's QR
would be another vendor-shaped float32 result for a quantity that is a
CONSTANT of `trend_len`, and the identity claim needs the constant to be
the same bits on every machine. Host float64 basic ops are.
MEASURED: `hw_check::check_hw_decompose_vs_reference` -- the float32
`R1Qt` against the float64 closed form (the cast, <= 0.5 ulp by
construction) and the start level/trend of a planted noiseless
level+trend series recovered within tolerance; their cuSOLVER bits were
not run (no NVIDIA here) and are expected to differ from ours in the last
bits of `R1Qt`, which is the one place this port's decomposition is NOT
their bits -- stated in the README as such.
============================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import fma
from max.gpu.host import DeviceBuffer, DeviceContext

from holtwinters.ported.holtwinters.internal.hw_utils import (
    SAB_NO_FTZ,
    SAB_ROTATE_CONV,
    get_num_blocks,
    get_threads_per_block,
)
from holtwinters.ported.tsa.holtwinters_params import SEASONAL_ADDITIVE
from mojo_only.numerics import ftz, identical_mul_add


@always_inline
def _f(x: Float32) -> Float32:
    comptime if SAB_NO_FTZ:
        return x
    return ftz(x)


def conv1d_kernel(
    inp: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    filter: MutPointer[Float32, MutAnyOrigin],
    filter_size_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
    output_size_in: Int32,
):
    """`conv1d_kernel` (`hw_decompose.cuh:24-39`): one thread per series,
    `out += filter[i] * input[tid + (i + o) * batch_size]` ascending in
    `i`, from `0.`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch_size = Int(batch_size_in)
    var filter_size = Int(filter_size_in)
    var output_size = Int(output_size_in)
    if tid < batch_size:
        for o in range(output_size):
            var out = Float32(0.0)
            comptime if SAB_ROTATE_CONV:
                var start = Int(block_idx.x) % filter_size
                for ii in range(filter_size):
                    var i = (start + ii) % filter_size
                    out = _f(identical_mul_add(
                        filter.unsafe_load(i), inp.unsafe_load(tid + (i + o) * batch_size), out
                    ))
            else:
                for i in range(filter_size):
                    out = _f(identical_mul_add(
                        filter.unsafe_load(i), inp.unsafe_load(tid + (i + o) * batch_size), out
                    ))
            output.unsafe_store(tid + o * batch_size, out)


def conv1d(
    ctx: DeviceContext,
    mut inp: DeviceBuffer[DType.float32],
    batch_size: Int,
    mut filter: DeviceBuffer[DType.float32],
    filter_size: Int,
    mut output: DeviceBuffer[DType.float32],
    output_size: Int,
    tpb: Int,
) raises:
    """`conv1d` (`:41-56`); `tpb` is their `GET_THREADS_PER_BLOCK(batch_size)`."""
    ctx.enqueue_function[conv1d_kernel](
        inp.unsafe_ptr(), Int32(batch_size), filter.unsafe_ptr(), Int32(filter_size),
        output.unsafe_ptr(), Int32(output_size),
        grid_dim=((batch_size + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )


def season_residual_kernel(
    season: MutPointer[Float32, MutAnyOrigin],
    ts: MutPointer[Float32, MutAnyOrigin],
    ts_offset_in: Int32,
    trend: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    additive_in: Int32,
):
    """Their `cublasgeam(1, ts + ts_offset, -1, trend)` (additive,
    `:303-316`) / `eltwiseDivide(season, aligned_ts, trend)` (multiplicative,
    `:317-322`): both are ELEMENTWISE over the contiguous `trend_len *
    batch_size` block (`geam`'s lda = ldb = ldc = trend_len on a contiguous
    buffer is cell `k` against cell `k`), so one thread per cell is the
    whole of it."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k < Int(cells_in):
        var x = ts.unsafe_load(Int(ts_offset_in) + k)
        var t = trend.unsafe_load(k)
        if additive_in != 0:
            season.unsafe_store(k, _f(x - t))
        else:
            season.unsafe_store(k, _f(x / t))


def season_mean_kernel(
    season: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    batch_size_in: Int32,
    start_season: MutPointer[Float32, MutAnyOrigin],
    frequency_in: Int32,
    half_filter_size_in: Int32,
    additive_in: Int32,
):
    """`season_mean_kernel` (`:59-88`) line for line: per phase `i` the
    ascending sum over `k = i, i + f, ...` divided by its count, written at
    `(i + half_filter_size) % frequency`; the mean of the phase means; then
    `-= mean` (additive) or `/= mean` (multiplicative)."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var length = Int(len_in)
    var batch_size = Int(batch_size_in)
    var frequency = Int(frequency_in)
    var half_filter_size = Int(half_filter_size_in)
    if tid < batch_size:
        var mean = Float32(0.0)
        for i in range(frequency):
            var period_mean = Float32(0.0)
            var k = i
            while k < length:
                period_mean = _f(period_mean + season.unsafe_load(k * batch_size + tid))
                k += frequency
            var count = 1 + ((length - i - 1) // frequency)
            period_mean = _f(period_mean / Float32(count))
            var ss_idx = (i + half_filter_size) % frequency
            start_season.unsafe_store(ss_idx * batch_size + tid, period_mean)
            mean = _f(mean + period_mean)
        mean = _f(mean / Float32(frequency))
        for i in range(frequency):
            var v = start_season.unsafe_load(i * batch_size + tid)
            if additive_in != 0:
                start_season.unsafe_store(i * batch_size + tid, _f(v - mean))
            else:
                start_season.unsafe_store(i * batch_size + tid, _f(v / mean))


def season_mean(
    ctx: DeviceContext,
    mut season: DeviceBuffer[DType.float32],
    length: Int,
    batch_size: Int,
    mut start_season: DeviceBuffer[DType.float32],
    frequency: Int,
    half_filter_size: Int,
    seasonal: Int,
    tpb: Int,
) raises:
    """`season_mean` (`:91-107`)."""
    ctx.enqueue_function[season_mean_kernel](
        season.unsafe_ptr(), Int32(length), Int32(batch_size), start_season.unsafe_ptr(),
        Int32(frequency), Int32(half_filter_size),
        Int32(1 if seasonal == SEASONAL_ADDITIVE else 0),
        grid_dim=((batch_size + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )


def host_r1qt(trend_len: Int) -> List[Float32]:
    """DEVIATION 660: `R1Qt` = pinv([1, t]), t = 1 .. trend_len, in the
    `rq[2 i] = intercept weight, rq[2 i + 1] = slope weight` layout their
    solver kernel reads. Host float64 basic ops + one explicit fma, cast."""
    var m = Float64(trend_len)
    var tbar = (m + Float64(1.0)) / Float64(2.0)
    var sxx = (m * (m * m - Float64(1.0))) / Float64(12.0)
    var inv_m = Float64(1.0) / m
    var out = List[Float32]()
    out.reserve(2 * trend_len)
    for i in range(trend_len):
        var t = Float64(i + 1)
        var w1 = (t - tbar) / sxx
        var w0 = fma(-tbar, w1, inv_m)
        out.append(Float32(w0))
        out.append(Float32(w1))
    return out^


def batched_ls_solver_kernel(
    B: MutPointer[Float32, MutAnyOrigin],
    rq: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    len_in: Int32,
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
):
    """`batched_ls_solver_kernel` (`:136-151`): one thread per series, the
    two dots ascending in `i` from `0.`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch_size = Int(batch_size_in)
    var length = Int(len_in)
    if tid < batch_size:
        var level_ = Float32(0.0)
        var trend_ = Float32(0.0)
        for i in range(length):
            var b = B.unsafe_load(tid + i * batch_size)
            level_ = _f(identical_mul_add(rq.unsafe_load(2 * i), b, level_))
            trend_ = _f(identical_mul_add(rq.unsafe_load(2 * i + 1), b, trend_))
        level.unsafe_store(tid, level_)
        trend.unsafe_store(tid, trend_)


def batched_ls(
    ctx: DeviceContext,
    mut data: DeviceBuffer[DType.float32],
    trend_len: Int,
    batch_size: Int,
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    tpb: Int,
) raises:
    """`batched_ls` (`:153-257`) with DEVIATION 660's host `R1Qt`."""
    var rq_h = host_r1qt(trend_len)
    var rq = ctx.enqueue_create_buffer[DType.float32](2 * trend_len)
    var host = ctx.enqueue_create_host_buffer[DType.float32](2 * trend_len)
    for i in range(2 * trend_len):
        host.unsafe_ptr().unsafe_store(i, rq_h[i])
    ctx.enqueue_copy(dst_buf=rq, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    ctx.enqueue_function[batched_ls_solver_kernel](
        data.unsafe_ptr(), rq.unsafe_ptr(), Int32(batch_size), Int32(trend_len),
        level.unsafe_ptr(), trend.unsafe_ptr(),
        grid_dim=((batch_size + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    _ = host^
    _ = rq^


def host_filter(frequency: Int) -> List[Float32]:
    """`filter_h(filter_size, 1. / frequency)` with the ends halved for an
    even frequency (`:282-287`): `(Dtype)(1. / frequency)` is a float64
    division rounded to float32."""
    var filter_size = (frequency // 2) * 2 + 1
    var v = Float32(Float64(1.0) / Float64(frequency))
    var out = List[Float32]()
    for _ in range(filter_size):
        out.append(v)
    if frequency % 2 == 0:
        out[0] = out[0] / Float32(2.0)
        out[filter_size - 1] = out[filter_size - 1] / Float32(2.0)
    return out^


def stl_decomposition_gpu(
    ctx: DeviceContext,
    mut ts: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    seasonal: Int,
    tpb: Int,
    scratch_pad: Int,
    scratch_poison: Float32,
    mut trend_scratch_out: DeviceBuffer[DType.float32],
    mut season_scratch_out: DeviceBuffer[DType.float32],
) raises:
    """`stl_decomposition_gpu` (`:259-324`). `tpb` is their
    `GET_THREADS_PER_BLOCK(batch_size)` for the per-series kernels (the
    elementwise residual takes the same width). `scratch_pad` /
    `scratch_poison` over-allocate and pre-fill the two scratch buffers
    (the gates' padding/poison knobs; no kernel reads a padded cell). The
    scratch buffers are handed back so the card can record the smoothed
    trend and the residual (`hw.decomp.trend`, `hw.decomp.season`)."""
    var end = start_periods * frequency
    var filter_size = (frequency // 2) * 2 + 1
    if end < filter_size:
        raise Error(
            "HoltWintersDecompose: start_periods*frequency (" + String(end)
            + ") must be >= filter_size (" + String(filter_size)
            + "); increase start_periods or decrease frequency"
        )
    var trend_len = end - (filter_size - 1)
    var batch_trend_n = batch_size * trend_len

    # Set filter
    var filter_h = host_filter(frequency)
    var filter_d = ctx.enqueue_create_buffer[DType.float32](filter_size)
    var fhost = ctx.enqueue_create_host_buffer[DType.float32](filter_size)
    for i in range(filter_size):
        fhost.unsafe_ptr().unsafe_store(i, filter_h[i])
    ctx.enqueue_copy(dst_buf=filter_d, src_ptr=fhost.unsafe_ptr())
    ctx.synchronize()

    # Set Trend
    var trend_d = ctx.enqueue_create_buffer[DType.float32](batch_trend_n + scratch_pad)
    trend_d.enqueue_fill(scratch_poison)
    conv1d(ctx, ts, batch_size, filter_d, filter_size, trend_d, trend_len, tpb)

    var season_d = ctx.enqueue_create_buffer[DType.float32](batch_trend_n + scratch_pad)
    season_d.enqueue_fill(scratch_poison)
    var ts_offset = (filter_size // 2) * batch_size
    ctx.enqueue_function[season_residual_kernel](
        season_d.unsafe_ptr(), ts.unsafe_ptr(), Int32(ts_offset), trend_d.unsafe_ptr(),
        Int32(batch_trend_n), Int32(1 if seasonal == SEASONAL_ADDITIVE else 0),
        grid_dim=((batch_trend_n + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    season_mean(
        ctx, season_d, trend_len, batch_size, start_season, frequency, filter_size // 2,
        seasonal, tpb,
    )
    batched_ls(ctx, trend_d, trend_len, batch_size, start_level, start_trend, tpb)
    ctx.synchronize()
    _ = fhost^
    _ = filter_d^
    trend_scratch_out = trend_d^
    season_scratch_out = season_d^
