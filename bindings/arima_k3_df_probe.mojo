# SPDX-License-Identifier: Apache-2.0
"""Private GPU probe; no product dispatch and no host numerical arithmetic."""
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from bindings.hostptr import f32_ptr, i32_ptr, copy_f32
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


def arima_k3_df_probe_binding(y: PythonObject, state: PythonObject,
        stages: PythonObject, stats: PythonObject, info: PythonObject,
        gradients: PythonObject, config: PythonObject) raises -> PythonObject:
    """ABI3 config=[batch,nobs,intercept,gradient]. Private borrowed pointers.

    state=(5,batch): T,Q,P0,alpha0,mu; y=(batch,nobs).
    stages=(3,batch,nobs): prediction,residual,variance.
    stats=(4,batch): rounded LL,final P,LL hi,LL lo; info=(batch,) i32.
    Enabled gradient=(batch//4,3), members model-major [base,p0+h,p1+h,p2+h].
    All other arrays contiguous f32; no model initialization on this route.
    """
    if len(config) != 4:
        raise Error("K3 DF probe: expected [batch,nobs,intercept,gradient]")
    var batch = Int(py=config[0])
    var nobs = Int(py=config[1])
    var intercept = Int(py=config[2])
    var gradient = Int(py=config[3])
    if batch < 1 or batch > 4096 or nobs < 1 or nobs > DF_MAX_OBS:
        raise Error("K3 DF probe: unsupported dimensions")
    if (intercept != 0 and intercept != 1) or (gradient != 0 and gradient != 1):
        raise Error("K3 DF probe: invalid flags")
    if gradient != 0 and (batch % 4 != 0 or nobs < 2):
        raise Error("K3 DF probe: gradient needs four members/model and nobs>=2")
    var yp = Int(py=y)
    var sp = Int(py=state)
    var op = Int(py=stages)
    var lp = Int(py=stats)
    var ip = Int(py=info)
    var gp = Int(py=gradients)
    if yp == 0 or sp == 0 or op == 0 or lp == 0 or ip == 0 or (gradient != 0 and gp == 0):
        raise Error("K3 DF probe: null input/output")
    with GILReleased(Python()):
        _probe_df(yp, sp, op, lp, ip, gp, batch, nobs, intercept, gradient)
    return PythonObject(3)
