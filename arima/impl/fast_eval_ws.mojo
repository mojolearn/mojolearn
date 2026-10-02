# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-tsa (2026-10-02): the FAST ARIMA fit's evaluation
workspace, allocated ONCE per solve (`-D MOJOLEARN_ARIMA_FAST_EVAL_WS=1`).

THE CAUSE. `batched_loglike_grad_host` (`arima/impl/batched_arima.mojo`,
the evaluation `batched_fit.mojo::eval_batch` makes at every candidate
point of the shared L-BFGS, hundreds of times per fit) allocates at every
call: `y_ext` ((N + 1) x batch_size series, the input replicated N + 1
times through N + 1 device copies), `x_ext`, `d_grad`, two `ARIMAParams`
(7 buffers each) and a `KalmanWorkspace` (26 buffers), plus four host
buffers, then issues N `perturb_kernel` launches, N `grad_kernel`
launches and the card's `P0` / `alpha0` copies, and synchronizes once.
About 45 buffer creations and 2 (N + 1) copies per evaluation on a batch
whose Kalman pass is 320 threads of 2,000 steps (the board's 64 ARMA(1,1)
series): the evaluation is host and allocator time, not filter time
(board, 0.8.25 L40S: arima 547 ms vs statsmodels 143).

THE CHANGE. `FastEvalWS` holds every buffer for the solve; `eval` issues
one host-to-device copy of the candidate vector, ONE stacking kernel
(`ew_stack_kernel`: member m of series b is x, member m >= 1 perturbed at
parameter m - 1 with `perturb_kernel`'s statement), `unpack`, the Jones
transform, `fast_kalman_into` (the filter's three launches into the held
workspace), ONE gradient kernel (`grad_kernel`'s statement over every
(parameter, series)) and the same four readbacks and one synchronize. The
per-element arithmetic is `perturb_kernel`'s, `grad_kernel`'s and the
filter's, so the log-likelihood and gradient bits are the sequence's; the
batch composition and launch geometry are unchanged. The compaction
(`FIT_COMPACT`) rebuilds the workspace at the packed size.

FAST on Apple only; IDENTICAL never imports this file's launches."""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx

from arima.impl.batched_kalman import (
    KALMAN_FAST_EVAL_WS,
    KalmanWorkspace,
    fast_kalman_into,
    kalman_raise_info_init,
    kalman_raise_info_loop,
)
from arima.impl.timeSeries.arima_helpers import batched_jones_transform
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, unpack
from checks.numerics import ftz

comptime EW_TPB = 128


def ew_stack_kernel(
    x_ext: MutPointer[Float32, MutAnyOrigin],
    d_x: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    h: Float32,
):
    """One thread per (member m, series b): block m of `x_ext` is `d_x`;
    for m >= 1 parameter m - 1 is `ftz(ftz(x) + h)` (`perturb_kernel`,
    `batched_arima.mojo`), the others the copy."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var B = Int(batch_size_in)
    var N = Int(N_in)
    if t >= (N + 1) * B:
        return
    var m = t // B
    var b = t - m * B
    var src = N * b
    var dst = m * (N * B) + N * b
    for i in range(N):
        var v = d_x.unsafe_load(src + i)
        if i == m - 1:
            v = ftz(ftz(v) + h)
        x_ext.unsafe_store(dst + i, v)


def ew_grad_kernel(
    d_grad: MutPointer[Float32, MutAnyOrigin],
    d_ll: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    N_in: Int32,
    h: Float32,
):
    """One thread per (parameter i, series b): `grad_kernel`'s
    `(ll_pert - ll_base) / h` with member i + 1's log-likelihood."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var B = Int(batch_size_in)
    var N = Int(N_in)
    if t >= N * B:
        return
    var i = t // B
    var b = t - i * B
    var diff = ftz(ftz(d_ll.unsafe_load((i + 1) * B + b)) - ftz(d_ll.unsafe_load(b)))
    d_grad.unsafe_store(N * b + i, ftz(diff / h))


struct FastEvalWS(Copyable, Movable):
    """Every buffer of one stacked evaluation, sized for `nb` series.
    Copyable because every field is a refcounted buffer or an Int, so the
    optimizer can keep one in a List that is empty when the switch is off."""

    var nb: Int
    var n_obs: Int
    var N: Int
    var eb: Int
    var y_ext: DeviceBuffer[DType.float32]
    var d_x: DeviceBuffer[DType.float32]
    var x_ext: DeviceBuffer[DType.float32]
    var d_grad: DeviceBuffer[DType.float32]
    var p_ext: ARIMAParams
    var t_params: ARIMAParams
    var ws: KalmanWorkspace
    var h_ll: HostBuffer[DType.float32]
    var h_g: HostBuffer[DType.float32]
    var h_i0: HostBuffer[DType.int32]
    var h_i1: HostBuffer[DType.int32]

    def __init__(
        out self,
        ctx: DeviceContext,
        mut d_y: DeviceBuffer[DType.float32],
        nb: Int,
        n_obs: Int,
        order: ARIMAOrder,
    ) raises:
        """`d_y` holds the `nb` series (contiguous, `nb * n_obs`); it is
        replicated N + 1 times here, once per solve."""
        var N = order.complexity()
        var M1 = N + 1
        var eb = M1 * nb
        var nb_y = nb * n_obs
        var nb_x = nb * N
        var y_ext = ctx.enqueue_create_buffer[DType.float32](max(1, eb * n_obs))
        for m in range(M1):
            ctx.enqueue_copy(
                dst_buf=y_ext.create_sub_buffer[DType.float32](m * nb_y, nb_y),
                src_buf=d_y.create_sub_buffer[DType.float32](0, nb_y),
            )
        var d_x = ctx.enqueue_create_buffer[DType.float32](max(1, nb_x))
        var x_ext = ctx.enqueue_create_buffer[DType.float32](max(1, eb * N))
        var d_grad = ctx.enqueue_create_buffer[DType.float32](max(1, nb_x))
        var p_ext = ARIMAParams(ctx, order, eb)
        var t_params = ARIMAParams(ctx, order, eb)
        var ws = KalmanWorkspace(ctx, order, eb, n_obs, 0)
        var h_ll = ctx.enqueue_create_host_buffer[DType.float32](max(1, nb))
        var h_g = ctx.enqueue_create_host_buffer[DType.float32](max(1, nb_x))
        var h_i0 = ctx.enqueue_create_host_buffer[DType.int32](max(1, eb))
        var h_i1 = ctx.enqueue_create_host_buffer[DType.int32](max(1, eb))
        ctx.synchronize()
        self.nb = nb
        self.n_obs = n_obs
        self.N = N
        self.eb = eb
        self.y_ext = y_ext^
        self.d_x = d_x^
        self.x_ext = x_ext^
        self.d_grad = d_grad^
        self.p_ext = p_ext^
        self.t_params = t_params^
        self.ws = ws^
        self.h_ll = h_ll^
        self.h_g = h_g^
        self.h_i0 = h_i0^
        self.h_i1 = h_i1^

    def eval(
        mut self,
        ctx: DeviceContext,
        order: ARIMAOrder,
        h: Float32,
        xin: List[Float32],
        mut ll_out: List[Float32],
        mut g_out: List[Float32],
    ) raises:
        """`batched_loglike_grad_host` with `trans = True` on the held
        buffers: the base log-likelihood of `xin` (`nb * N` values) into
        `ll_out[0:nb]`, the forward-difference gradient into `g_out[0:nb *
        N]`; the Kalman refusals are raised after the one synchronize."""
        comptime if not KALMAN_FAST_EVAL_WS:
            raise Error("FastEvalWS.eval: not compiled in this build (MOJOLEARN_ARIMA_FAST_EVAL_WS)")
        var nb = self.nb
        var N = self.N
        var eb = self.eb
        var nb_x = nb * N
        if len(xin) != nb_x:
            raise Error(
                "FastEvalWS.eval: len(xin)=" + String(len(xin)) + " is not nb * N = "
                + String(nb_x)
            )
        ctx.enqueue_copy(
            dst_buf=self.d_x.create_sub_buffer[DType.float32](0, nb_x), src_ptr=xin.unsafe_ptr()
        )
        var g1 = (eb + EW_TPB - 1) // EW_TPB
        ctx.enqueue_function[ew_stack_kernel](
            self.x_ext.unsafe_ptr(), self.d_x.unsafe_ptr(), Int32(nb), Int32(N), h,
            grid_dim=(g1, 1, 1), block_dim=(EW_TPB, 1, 1),
        )
        unpack(ctx, self.p_ext, order, eb, self.x_ext)
        batched_jones_transform(ctx, order, eb, False, self.p_ext, self.t_params)
        fast_kalman_into(ctx, self.y_ext, self.t_params, order, eb, self.n_obs, self.ws)
        var g2 = (nb_x + EW_TPB - 1) // EW_TPB
        ctx.enqueue_function[ew_grad_kernel](
            self.d_grad.unsafe_ptr(), self.ws.loglike.unsafe_ptr(), Int32(nb), Int32(N), h,
            grid_dim=(g2, 1, 1), block_dim=(EW_TPB, 1, 1),
        )
        ctx.enqueue_copy(
            dst_ptr=self.h_ll.unsafe_ptr(),
            src_buf=self.ws.loglike.create_sub_buffer[DType.float32](0, nb),
        )
        ctx.enqueue_copy(
            dst_ptr=self.h_g.unsafe_ptr(),
            src_buf=self.d_grad.create_sub_buffer[DType.float32](0, nb_x),
        )
        ctx.enqueue_copy(dst_ptr=self.h_i0.unsafe_ptr(), src_buf=self.ws.info_init)
        ctx.enqueue_copy(dst_ptr=self.h_i1.unsafe_ptr(), src_buf=self.ws.info_loop)
        ctx.synchronize()
        var i0 = List[Int32](capacity=eb)
        var i1 = List[Int32](capacity=eb)
        for b in range(eb):
            i0.append(self.h_i0.unsafe_ptr()[b])
            i1.append(self.h_i1.unsafe_ptr()[b])
        kalman_raise_info_init(i0)
        kalman_raise_info_loop(i1, order.n_diff())
        for b in range(nb):
            ll_out[b] = self.h_ll.unsafe_ptr()[b]
        for t in range(nb_x):
            g_out[t] = self.h_g.unsafe_ptr()[t]
