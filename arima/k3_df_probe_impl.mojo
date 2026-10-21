# arima/k3_df_probe_impl.mojo -- the typed body of the opt-in K3 DF quality probe
# (moved from bindings/arima_k3_df_probe.mojo on 2026-10-21 so the shell holds
# only the Python door). Same launches, same order, same bits.
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from core.hostptr import f32_ptr, i32_ptr, copy_f32
from arima.estimator import _upload_f32
from arima.impl.fast_scalar_df import DF_TPB, DF_MAX_OBS, k3_df_parts_kernel, k3_df_finish_kernel, k3_df_gradient_kernel


def _download_df(ctx: DeviceContext, data: DeviceBuffer[DType.float32], address: Int, count: Int) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=data)
    ctx.synchronize()
    copy_f32(host.unsafe_ptr(), f32_ptr(address), count)
    _ = host^


def _probe_df(yp: Int, sp: Int, op: Int, lp: Int, ip: Int, gp: Int,
              batch: Int, nobs: Int, intercept: Int, gradient: Int) raises:
    var ctx = process_ctx["MojoArimaScalarDFQuality"]()
    var y = _upload_f32(ctx, f32_ptr(yp), batch * nobs)
    var state = _upload_f32(ctx, f32_ptr(sp), 5 * batch)
    var chunks = (nobs + DF_TPB - 1) // DF_TPB
    var parts = ctx.enqueue_create_buffer[DType.float32](5 * batch * chunks)
    var capture = ctx.enqueue_create_buffer[DType.float32](3 * batch * nobs)
    var stats = ctx.enqueue_create_buffer[DType.float32](4 * batch)
    var info = ctx.enqueue_create_buffer[DType.int32](batch)
    var grad = ctx.enqueue_create_buffer[DType.float32](max(1, (batch // 4) * 3))
    ctx.enqueue_function[k3_df_parts_kernel](
        y.unsafe_ptr(), state.unsafe_ptr(), parts.unsafe_ptr(), capture.unsafe_ptr(),
        Int32(nobs), Int32(batch), Int32(intercept),
        grid_dim=(batch * chunks, 1, 1), block_dim=(DF_TPB, 1, 1),
    )
    ctx.enqueue_function[k3_df_finish_kernel](
        parts.unsafe_ptr(), state.unsafe_ptr(), stats.unsafe_ptr(), info.unsafe_ptr(),
        Int32(nobs), Int32(batch), grid_dim=(batch, 1, 1), block_dim=(DF_TPB, 1, 1),
    )
    if gradient != 0:
        ctx.enqueue_function[k3_df_gradient_kernel](
            stats.unsafe_ptr(), info.unsafe_ptr(), grad.unsafe_ptr(), Int32(nobs), Int32(batch),
            grid_dim=(((batch // 4) * 3 + 127) // 128, 1, 1), block_dim=(128, 1, 1),
        )
        _download_df(ctx, grad, gp, (batch // 4) * 3)
    _download_df(ctx, capture, op, 3 * batch * nobs)
    _download_df(ctx, stats, lp, 4 * batch)
    var host = ctx.enqueue_create_host_buffer[DType.int32](batch)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=info)
    ctx.synchronize()
    memcpy(dest=i32_ptr(ip), src=host.unsafe_ptr(), count=batch)
    _ = host^
