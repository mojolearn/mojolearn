# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resumable GPU optimizer state for heterogeneous AutoARIMA order groups.

The existing device state machine is unchanged. Launching and polling are
separate methods so a group can evaluate multiple orders together before
ONE synchronization. The single-order entrypoint shares the validated refactor. Apple FAST
defaults to it; MOJOLEARN_ARIMA_ORDER_BATCH_OFF restores the original path.
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.sys.compile import is_defined
from arima.impl.batched_kalman import KALMAN_FAST_EVAL_WS, KALMAN_LL_ONLY
from arima.impl.fast_eval_ws import FastEvalWS
from arima.impl.fast_lbfgs_async import (
    ASYNC_READ_EVERY, AsyncLBFGSOut, I_N, F_N, F_FX, F_FXC, F_FXP,
    F_GNORM, F_STEP, I_ACTIVE, I_SEARCHING, I_ENDV, I_NVEC, I_LSRET,
    I_LSITERS, I_NITER, I_RETCODE, async_start_kernel, async_step_kernel,
    _down_f32, _down_i32,
)
from arima.impl.lbfgs_device import LBFGS_TPB, lbfgs_init_kernel
from arima.impl.tsa.arima_common import ARIMAOrder
from glm.impl.qn.qn_util import LBFGSParam

# KEEP candidate for default promotion, M3 2026-10-04, source 7ba385b30:
# gap26-orders-current-synthetic 13780.317 -> 9287.287 ms (-32.6%);
# gap26-orders-current-taxi-hourly 22706.248 -> 13747.215 ms (-39.5%).
# Digests/RMSE unchanged; FULL_PASS 44 paired 512/2048-observation fitted,
# order, likelihood and forecast arrays exact. Accepted fused tail in both.
# Taxi's existing opponent-quality hold remains (RMSE74.659 vs68.21).
# Default only within existing FAST+Apple guards; named OFF restores the
# pre-batching GPU optimizer/search while leaving fused eval tail enabled.
# See docs/apple-fast/ab/arima-orders-default.md and EXPERIMENTS.md.
comptime ARIMA_ORDER_BATCH = (
    KALMAN_FAST_EVAL_WS and KALMAN_LL_ONLY
    and not is_defined["MOJOLEARN_ARIMA_ORDER_BATCH_OFF"]()
)


struct OrderOptimizer(Movable):
    var n: Int
    var bs: Int
    var b_n: Int
    var m: Int
    var past: Int
    var grid: Int
    var rounds: Int
    var max_rounds: Int
    var param: LBFGSParam
    var order: ARIMAOrder
    var scale: Float32
    var h: Float32
    var ist: DeviceBuffer[DType.int32]
    var fst: DeviceBuffer[DType.float32]
    var x: DeviceBuffer[DType.float32]
    var cand: DeviceBuffer[DType.float32]
    var xp: DeviceBuffer[DType.float32]
    var grad: DeviceBuffer[DType.float32]
    var gradp: DeviceBuffer[DType.float32]
    var gradc: DeviceBuffer[DType.float32]
    var drt: DeviceBuffer[DType.float32]
    var d_grad: DeviceBuffer[DType.float32]
    var d_x_pert: DeviceBuffer[DType.float32]
    var S: DeviceBuffer[DType.float32]
    var Y: DeviceBuffer[DType.float32]
    var yhist: DeviceBuffer[DType.float32]
    var alpha: DeviceBuffer[DType.float32]
    var fx_hist: DeviceBuffer[DType.float32]
    var bad: DeviceBuffer[DType.int32]
    var any_active: DeviceBuffer[DType.int32]
    var flag_host: HostBuffer[DType.int32]
    var f_fx: DeviceBuffer[DType.float32]
    var f_fxc: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, mut ews: FastEvalWS,
                 batch_size: Int, scale: Float32, order_kf: ARIMAOrder,
                 x0: List[Float32], param: LBFGSParam, h: Float32) raises:
        var n = order_kf.complexity()
        var bs = batch_size
        var b_n = bs * n
        var m = param.m
        var past = param.past if param.past > 0 else 0
        var grid = (bs + LBFGS_TPB - 1) // LBFGS_TPB
        var ist = ctx.enqueue_create_buffer[DType.int32](max(1, I_N * bs))
        var fst = ctx.enqueue_create_buffer[DType.float32](max(1, F_N * bs))
        var x = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var cand = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var xp = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var grad = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var gradp = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var gradc = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var drt = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var d_grad = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var d_x_pert = ctx.enqueue_create_buffer[DType.float32](max(1, b_n))
        var S = ctx.enqueue_create_buffer[DType.float32](max(1, b_n * m))
        var Y = ctx.enqueue_create_buffer[DType.float32](max(1, b_n * m))
        var yhist = ctx.enqueue_create_buffer[DType.float32](max(1, bs * m))
        var alpha = ctx.enqueue_create_buffer[DType.float32](max(1, bs * m))
        var fx_hist = ctx.enqueue_create_buffer[DType.float32](max(1, bs * (past if past > 0 else 1)))
        var bad = ctx.enqueue_create_buffer[DType.int32](max(1, bs))
        var any_active = ctx.enqueue_create_buffer[DType.int32](1)
        var flag_host = ctx.enqueue_create_host_buffer[DType.int32](1)
        var f_fx = fst.create_sub_buffer[DType.float32](F_FX * bs, max(1, bs))
        var f_fxc = fst.create_sub_buffer[DType.float32](F_FXC * bs, max(1, bs))
        ctx.enqueue_memset(ist, Int32(0))
        ctx.enqueue_memset(fst, Float32(0.0))
        ctx.enqueue_memset(xp, Float32(0.0))
        ctx.enqueue_memset(gradp, Float32(0.0))
        ctx.enqueue_memset(gradc, Float32(0.0))
        ctx.enqueue_memset(drt, Float32(0.0))
        ctx.enqueue_memset(S, Float32(0.0))
        ctx.enqueue_memset(Y, Float32(0.0))
        ctx.enqueue_memset(yhist, Float32(0.0))
        ctx.enqueue_memset(alpha, Float32(0.0))
        ctx.enqueue_memset(fx_hist, Float32(0.0))
        if b_n > 0:
            var hx = x0.copy()
            ctx.enqueue_copy(dst_buf=x.create_sub_buffer[DType.float32](0, b_n), src_ptr=hx.unsafe_ptr())
            ctx.synchronize()
            _ = hx^
        ctx.enqueue_copy(dst_buf=cand, src_buf=x)
        # the evaluation at x0, then `lbfgs_init_kernel` on the packed fields
        ews.eval(ctx, order_kf, h, scale, cand, d_grad, d_x_pert, f_fx, grad, bad)
        var ip = ist.unsafe_ptr()
        var fp = fst.unsafe_ptr()
        ctx.enqueue_memset(any_active, Int32(0))
        # one sub-buffer per field (one argument may not alias another)
        var s_fxp = fst.create_sub_buffer[DType.float32](F_FXP * bs, max(1, bs))
        var s_gn = fst.create_sub_buffer[DType.float32](F_GNORM * bs, max(1, bs))
        var s_st = fst.create_sub_buffer[DType.float32](F_STEP * bs, max(1, bs))
        var s_ac = ist.create_sub_buffer[DType.int32](I_ACTIVE * bs, max(1, bs))
        var s_se = ist.create_sub_buffer[DType.int32](I_SEARCHING * bs, max(1, bs))
        var s_en = ist.create_sub_buffer[DType.int32](I_ENDV * bs, max(1, bs))
        var s_nv = ist.create_sub_buffer[DType.int32](I_NVEC * bs, max(1, bs))
        var s_lr = ist.create_sub_buffer[DType.int32](I_LSRET * bs, max(1, bs))
        var s_li = ist.create_sub_buffer[DType.int32](I_LSITERS * bs, max(1, bs))
        var s_ni = ist.create_sub_buffer[DType.int32](I_NITER * bs, max(1, bs))
        var s_rc = ist.create_sub_buffer[DType.int32](I_RETCODE * bs, max(1, bs))
        ctx.enqueue_function[lbfgs_init_kernel](
            grad.unsafe_ptr(), drt.unsafe_ptr(), f_fx.unsafe_ptr(),
            s_fxp.unsafe_ptr(), fx_hist.unsafe_ptr(), s_gn.unsafe_ptr(), s_st.unsafe_ptr(),
            s_ac.unsafe_ptr(), s_se.unsafe_ptr(), s_en.unsafe_ptr(), s_nv.unsafe_ptr(),
            s_lr.unsafe_ptr(), s_li.unsafe_ptr(), s_ni.unsafe_ptr(), s_rc.unsafe_ptr(),
            any_active.unsafe_ptr(),
            Int32(bs), Int32(n), Int32(past), param.epsilon, param.delta,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        ctx.enqueue_memset(any_active, Int32(0))
        ctx.enqueue_function[async_start_kernel](
            ip, fp, x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
            drt.unsafe_ptr(), cand.unsafe_ptr(), S.unsafe_ptr(), Y.unsafe_ptr(),
            yhist.unsafe_ptr(), alpha.unsafe_ptr(), fx_hist.unsafe_ptr(), any_active.unsafe_ptr(),
            Int32(bs), Int32(n), Int32(m), Int32(past), Int32(param.max_iterations),
            param.epsilon, param.delta, param.ftol,
            grid_dim=(grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        _ = s_fxp^
        _ = s_gn^
        _ = s_st^
        _ = s_ac^
        _ = s_se^
        _ = s_en^
        _ = s_nv^
        _ = s_lr^
        _ = s_li^
        _ = s_ni^
        _ = s_rc^
        self.n = n
        self.bs = bs
        self.b_n = b_n
        self.m = m
        self.past = past
        self.grid = grid
        self.rounds = 0
        self.max_rounds = param.max_iterations * max(1, param.max_linesearch) + 1
        self.param = param
        self.order = order_kf
        self.scale = scale
        self.h = h
        self.ist = ist^
        self.fst = fst^
        self.x = x^
        self.cand = cand^
        self.xp = xp^
        self.grad = grad^
        self.gradp = gradp^
        self.gradc = gradc^
        self.drt = drt^
        self.d_grad = d_grad^
        self.d_x_pert = d_x_pert^
        self.S = S^
        self.Y = Y^
        self.yhist = yhist^
        self.alpha = alpha^
        self.fx_hist = fx_hist^
        self.bad = bad^
        self.any_active = any_active^
        self.flag_host = flag_host^
        self.f_fx = f_fx^
        self.f_fxc = f_fxc^

    def enqueue_poll(mut self, ctx: DeviceContext) raises:
        ctx.enqueue_copy(dst_ptr=self.flag_host.unsafe_ptr(), src_buf=self.any_active)

    def running(self) -> Bool:
        return self.rounds < self.max_rounds and self.flag_host.unsafe_ptr()[0] != 0

    def evaluate(mut self, ctx: DeviceContext, mut ews: FastEvalWS) raises:
        ews.eval(ctx, self.order, self.h, self.scale, self.cand,
                 self.d_grad, self.d_x_pert, self.f_fxc, self.gradc, self.bad)

    def advance(mut self, ctx: DeviceContext) raises:
        ctx.enqueue_memset(self.any_active, Int32(0))
        ctx.enqueue_function[async_step_kernel](
            self.ist.unsafe_ptr(), self.fst.unsafe_ptr(), self.x.unsafe_ptr(), self.xp.unsafe_ptr(), self.grad.unsafe_ptr(), self.gradp.unsafe_ptr(),
            self.drt.unsafe_ptr(), self.cand.unsafe_ptr(), self.gradc.unsafe_ptr(), self.S.unsafe_ptr(),
            self.Y.unsafe_ptr(), self.yhist.unsafe_ptr(), self.alpha.unsafe_ptr(), self.fx_hist.unsafe_ptr(),
            self.any_active.unsafe_ptr(),
            Int32(self.bs), Int32(self.n), Int32(self.m), Int32(self.past), Int32(self.param.max_iterations),
            Int32(self.param.max_linesearch), self.param.epsilon, self.param.delta, self.param.ftol,
            self.param.min_step, self.param.max_step, self.param.ls_dec,
            grid_dim=(self.grid, 1, 1), block_dim=(LBFGS_TPB, 1, 1),
        )
        self.rounds += 1

    def result(mut self, ctx: DeviceContext) raises -> AsyncLBFGSOut:
        var x_out = _down_f32(ctx, self.x, 0, self.b_n)
        var fx_out = _down_f32(ctx, self.fst, F_FX * self.bs, self.bs)
        var n_iter_out = _down_i32(ctx, self.ist, I_NITER * self.bs, self.bs)
        var retcode_out = _down_i32(ctx, self.ist, I_RETCODE * self.bs, self.bs)
        return AsyncLBFGSOut(x=x_out^, fx=fx_out^, n_iter=n_iter_out^,
                             retcode=retcode_out^, n_eval=self.rounds + 1)


def order_min_lbfgs(ctx: DeviceContext, mut ews: FastEvalWS, batch_size: Int,
                    scale: Float32, order: ARIMAOrder, x0: List[Float32],
                    param: LBFGSParam, h: Float32) raises -> AsyncLBFGSOut:
    var state = OrderOptimizer(ctx, ews, batch_size, scale, order, x0, param, h)
    while state.rounds < state.max_rounds:
        if state.rounds % ASYNC_READ_EVERY == 0:
            state.enqueue_poll(ctx)
            ctx.synchronize()
            if not state.running():
                break
        state.evaluate(ctx, ews)
        state.advance(ctx)
    return state.result(ctx)
