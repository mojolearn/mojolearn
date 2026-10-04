# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Candidate-only numerical probe, not a product fit/predict route.

Launch the existing main RD1 kernel and K3 on identical supplied state
inputs. Host work is byte-preserving transfer only. Python quality tools
provide independent reference calculations and finite-difference inputs.
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from bindings.hostptr import f32_ptr, i32_ptr, copy_f32
from arima.estimator import _upload_f32
from arima.impl.batched_kalman import KalmanWorkspace, batched_kalman_loop_kernel
from arima.impl.fast_scalar_ll import ARIMA_FAST_SCALAR_LL, SCALAR_LL_MAX_OBS, launch_scalar_ll
from arima.impl.tsa.arima_common import ARIMAOrder


def _download(ctx: DeviceContext, data: DeviceBuffer[DType.float32], address: Int, count: Int) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=data)
    ctx.synchronize()
    copy_f32(host.unsafe_ptr(), f32_ptr(address), count)
    _ = host^


def _download_info(ctx: DeviceContext, data: DeviceBuffer[DType.int32], address: Int, count: Int) raises:
    var host = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=data)
    ctx.synchronize()
    memcpy(dest=i32_ptr(address), src=host.unsafe_ptr(), count=count)
    _ = host^


def _probe(y_address: Int, state_address: Int, stage_address: Int,
           stats_address: Int, info_address: Int, batch: Int, nobs: Int, intercept: Int) raises:
    var ctx = process_ctx["MojoArimaScalarQuality"]()
    var order = ARIMAOrder(1, 0, 0, 0, 0, 0, 0, intercept, 0)
    var total = batch * nobs
    var y = _upload_f32(ctx, f32_ptr(y_address), total)
    var mu = _upload_f32(ctx, f32_ptr(state_address + 4 * 4 * batch), batch)
    var a = KalmanWorkspace(ctx, order, batch, nobs, 0)
    var b = KalmanWorkspace(ctx, order, batch, nobs, 0)
    a.T = _upload_f32(ctx, f32_ptr(state_address), batch)
    a.RQR = _upload_f32(ctx, f32_ptr(state_address + 4 * batch), batch)
    a.P = _upload_f32(ctx, f32_ptr(state_address + 4 * 2 * batch), batch)
    a.alpha = _upload_f32(ctx, f32_ptr(state_address + 4 * 3 * batch), batch)
    ctx.enqueue_copy(dst_buf=b.T, src_buf=a.T)
    ctx.enqueue_copy(dst_buf=b.RQR, src_buf=a.RQR)
    ctx.enqueue_copy(dst_buf=b.P, src_buf=a.P)
    ctx.enqueue_copy(dst_buf=b.alpha, src_buf=a.alpha)
    # LL_ONLY=False only adds diagnostic stores; arithmetic is main's
    # existing recurrence and reduction, not a rewritten reference kernel.
    ctx.enqueue_function[batched_kalman_loop_kernel[1, False]](
        y.unsafe_ptr(), a.T.unsafe_ptr(), a.Z.unsafe_ptr(), a.RQR.unsafe_ptr(),
        a.P.unsafe_ptr(), a.alpha.unsafe_ptr(), mu.unsafe_ptr(),
        a.pred.unsafe_ptr(), a.vs.unsafe_ptr(), a.Fs.unsafe_ptr(),
        a.loglike.unsafe_ptr(), a.fc.unsafe_ptr(), a.info_loop.unsafe_ptr(),
        a.obs.unsafe_ptr(), a.obs_fut.unsafe_ptr(),
        Int32(1), Int32(nobs), Int32(batch), Int32(intercept), Int32(0), Int32(0), Int32(0),
        grid_dim=((batch + 31) // 32, 1, 1), block_dim=(32, 1, 1),
    )
    var capture = ctx.enqueue_create_buffer[DType.float32](3 * total)
    launch_scalar_ll[True](ctx, y, b.T, b.RQR, b.P, b.alpha, mu,
        b.pred, b.vs, b.Fs, b.loglike, b.info_loop, capture, nobs, batch, intercept)
    _download(ctx, a.pred, stage_address, total)
    _download(ctx, a.vs, stage_address + 4 * total, total)
    _download(ctx, a.Fs, stage_address + 4 * 2 * total, total)
    _download(ctx, capture, stage_address + 4 * 3 * total, 3 * total)
    _download(ctx, a.loglike, stats_address, batch)
    _download(ctx, b.loglike, stats_address + 4 * batch, batch)
    _download(ctx, a.P, stats_address + 4 * 2 * batch, batch)
    _download(ctx, b.P, stats_address + 4 * 3 * batch, batch)
    _download_info(ctx, a.info_loop, info_address, batch)
    _download_info(ctx, b.info_loop, info_address + 4 * batch, batch)


def arima_scalar_probe_binding(y: PythonObject, state: PythonObject,
        stages: PythonObject, stats: PythonObject, info: PythonObject, config: PythonObject) raises -> PythonObject:
    """Private probe: config=[batch,nobs,intercept]. All arrays C contiguous.

    y: batch*nobs f32. state: [T,Q,P0,alpha0,mu], each batch f32.
    stages: [A_pred,A_residual,A_F,B_pred,B_residual,B_F], each batch*nobs.
    stats: [A_LL,B_LL,A_final_P,B_final_P], each batch f32.
    info: [A_info,B_info], each batch int32. Buffers owned by caller.
    """
    if len(config) != 3:
        raise Error("K3 probe: expected [batch,nobs,intercept]")
    var batch = Int(py=config[0])
    var nobs = Int(py=config[1])
    var intercept = Int(py=config[2])
    if batch < 1 or batch > 4096 or nobs < 1 or nobs > SCALAR_LL_MAX_OBS or (intercept != 0 and intercept != 1):
        raise Error("K3 probe: unsupported dimensions or intercept")
    var yp = Int(py=y)
    var sp = Int(py=state)
    var op = Int(py=stages)
    var lp = Int(py=stats)
    var ip = Int(py=info)
    with GILReleased(Python()):
        _probe(yp, sp, op, lp, ip, batch, nobs, intercept)
    return PythonObject(1)
