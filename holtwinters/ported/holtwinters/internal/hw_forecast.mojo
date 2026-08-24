"""cuML `cpp/src/holtwinters/internal/hw_forecast.cuh` (v26.08.00).

`holtwinters_seasonal_forecast_kernel` and `holtwinters_forecast_gpu`'s
seasonal arm (the public `ML::HoltWinters::forecast` always passes level,
trend and season; the level-only and non-seasonal kernels are UNPORTED).
One thread per series, `h` steps serial: `level + trend * (i + 1) + season`
is `fma(trend, i + 1, level) + season` (additive) or `fma(...) * season`
(multiplicative); `i + 1` is exact in float32.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from holtwinters.ported.holtwinters.internal.hw_utils import SAB_NO_FTZ
from mojo_only.numerics import ftz, identical_mul_add


@always_inline
def _f(x: Float32) -> Float32:
    comptime if SAB_NO_FTZ:
        return x
    return ftz(x)


def holtwinters_seasonal_forecast_kernel(
    forecast: MutPointer[Float32, MutAnyOrigin],
    h_in: Int32,
    batch_size_in: Int32,
    frequency_in: Int32,
    level_coef: MutPointer[Float32, MutAnyOrigin],
    trend_coef: MutPointer[Float32, MutAnyOrigin],
    season_coef: MutPointer[Float32, MutAnyOrigin],
    additive_in: Int32,
):
    """`holtwinters_seasonal_forecast_kernel` (`hw_forecast.cuh:9-32`).
    `level_coef`/`trend_coef` are already offset to the last fitted row,
    `season_coef` to the last `frequency` rows (the runner's shifts)."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var h = Int(h_in)
    var batch_size = Int(batch_size_in)
    var frequency = Int(frequency_in)
    if tid < batch_size:
        var level = level_coef.unsafe_load(tid)
        var trend = trend_coef.unsafe_load(tid)
        for i in range(h):
            var season = season_coef.unsafe_load(tid + (i % frequency) * batch_size)
            var lt = _f(identical_mul_add(trend, Float32(i + 1), level))
            if additive_in != 0:
                forecast.unsafe_store(tid + i * batch_size, _f(lt + season))
            else:
                forecast.unsafe_store(tid + i * batch_size, _f(lt * season))


def holtwinters_forecast_gpu(
    ctx: DeviceContext,
    mut forecast: DeviceBuffer[DType.float32],
    h: Int,
    batch_size: Int,
    frequency: Int,
    mut level_coef: DeviceBuffer[DType.float32],
    level_offset: Int,
    mut trend_coef: DeviceBuffer[DType.float32],
    trend_offset: Int,
    mut season_coef: DeviceBuffer[DType.float32],
    season_offset: Int,
    additive: Bool,
    tpb: Int,
) raises:
    """`holtwinters_forecast_gpu` (`:60-87`), the seasonal arm. The
    `*_offset` are the runner's `leveltrend_coef_shift` /
    `season_coef_shift` (their pointer arithmetic, spelled out)."""
    if tpb <= 0:
        raise Error("holtwinters_forecast_gpu: tpb must be positive")
    ctx.enqueue_function[holtwinters_seasonal_forecast_kernel](
        forecast.unsafe_ptr(), Int32(h), Int32(batch_size), Int32(frequency),
        level_coef.unsafe_ptr().unsafe_offset(level_offset),
        trend_coef.unsafe_ptr().unsafe_offset(trend_offset),
        season_coef.unsafe_ptr().unsafe_offset(season_offset),
        Int32(1 if additive else 0),
        grid_dim=((batch_size + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
