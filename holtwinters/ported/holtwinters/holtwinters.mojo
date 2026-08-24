"""cuML `cpp/src/holtwinters/holtwinters.cu` + `include/cuml/tsa/holtwinters.h`
(v26.08.00): `ML::HoltWinters::buffer_size`, `fit`, `forecast` -- the
public surface, float instantiation only (double: UNPORTED.tsv). Each is
the thin call into `runner.mojo` that theirs is.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from holtwinters.ported.holtwinters.runner import (
    HWBufferSizes,
    HW_DEFAULT_TRACE_ITERS,
    holtwinters_buffer_size,
    holtwinters_fit_helper,
    holtwinters_forecast_helper,
)
from holtwinters.ported.holtwinters.internal.hw_utils import HW_OPTIM_TPB


def buffer_size(n: Int, batch_size: Int, frequency: Int) raises -> HWBufferSizes:
    """`ML::HoltWinters::buffer_size` (`holtwinters.cu:13-35`): use_beta =
    use_gamma = true."""
    return holtwinters_buffer_size(n, batch_size, frequency, True, True)


def fit(
    ctx: DeviceContext,
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    seasonal: Int,
    epsilon: Float32,
    mut data: DeviceBuffer[DType.float32],
    mut level_d: DeviceBuffer[DType.float32],
    mut trend_d: DeviceBuffer[DType.float32],
    mut season_d: DeviceBuffer[DType.float32],
    mut error_d: DeviceBuffer[DType.float32],
    mut alpha_d: DeviceBuffer[DType.float32],
    mut beta_d: DeviceBuffer[DType.float32],
    mut gamma_d: DeviceBuffer[DType.float32],
    mut criterion_d: DeviceBuffer[DType.int32],
    mut niter_d: DeviceBuffer[DType.int32],
    mut decisions_d: DeviceBuffer[DType.int32],
    mut iter_trace_d: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    trace_iters: Int = HW_DEFAULT_TRACE_ITERS,
    tpb_decomp: Int = -1,
    tpb_optim: Int = HW_OPTIM_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = Float32(0.0),
) raises:
    """`ML::HoltWinters::fit` (`holtwinters.cu:37-62`, float). The extra
    outputs (`alpha/beta/gamma`, `criterion`, `niter`, `iter_trace`) are
    DEVIATION 665's instrumentation, `decisions` is DEVIATION 699's; theirs leaves the fitted parameters on
    the device and discards them."""
    holtwinters_fit_helper(
        ctx, data, n, batch_size, frequency, start_periods, seasonal, epsilon,
        level_d, trend_d, season_d, error_d, alpha_d, beta_d, gamma_d,
        criterion_d, niter_d, decisions_d, iter_trace_d, trace_iters, trace,
        tpb_decomp, tpb_optim, scratch_pad, scratch_poison,
    )


def forecast(
    ctx: DeviceContext,
    n: Int,
    batch_size: Int,
    frequency: Int,
    h: Int,
    seasonal: Int,
    mut level_d: DeviceBuffer[DType.float32],
    mut trend_d: DeviceBuffer[DType.float32],
    mut season_d: DeviceBuffer[DType.float32],
    mut forecast_d: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    tpb: Int = -1,
) raises:
    """`ML::HoltWinters::forecast` (`holtwinters.cu:91-104`, float)."""
    holtwinters_forecast_helper(
        ctx, n, batch_size, frequency, h, seasonal, level_d, trend_d, season_d,
        forecast_d, trace, tpb,
    )
