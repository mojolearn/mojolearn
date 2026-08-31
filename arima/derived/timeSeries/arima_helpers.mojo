# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ARIMA helpers: the reduced polynomials, undifferencing, the batched
Jones transform with the sigma2 floor.

PORT OF `cuml/cpp/src_prims/timeSeries/arima_helpers.cuh` at cuML 265b9da6
(v26.08.00): `_param_to_poly` (:35-44), `_select_read` (:58-62),
`_undiff_kernel` (:136-163), `reduced_polynomial` (:183-193),
`finalize_forecast` (:321-350), `batched_jones_transform` (:358-379).
`prepare_data` (:209-239) is imported from the tsa lane's port of the same
file; `prepare_future_data` / `_future_diff_kernel` (exog only) are not
reached (exog refused) and are in `arima/NOT_IMPLEMENTED.tsv`. COPY, DO NOT
IMPROVE.

`reduced_polynomial<isAr>(bid, param, lags, sparam, slags, s, idx)` is
`-coef0 * coef1` for AR and `coef0 * coef1` for MA, ONE product (exact sign);
it is evaluated inside `init_batched_kalman_matrices` (the arima lane's
`batched_kalman.mojo`) per series and its host replay, both through this
file's `reduced_polynomial`, which is pure scalar arithmetic on values
already loaded and so is shared by kernel and oracle without hiding a
seam. The undifferencing is `b_fc[i] += select(i - s0)` (one add) or
`b_fc[i] += ((-x0 + x1) + x2)` (`:154-157`, left to right); the sigma2
floor is `max(input, 1e-6)` -- value-first already (`:376`).

ONE ADDITION TO THEIR FUNCTION, recorded because it is not in their body:
`batched_jones_transform` here also COPIES `mu` into `t_params.mu`. Theirs
cannot: `ARIMAParams` is a struct of raw pointers there, so the caller
aliases `Tparams.mu = params.mu` in the aggregate initializer
(`batched_arima.cu:419-425`) and the transform never touches it. Ours owns its
buffers, so the same VALUE has to be moved rather than aliased. No
arithmetic, and `_copy_params` in `batched_arima.mojo` does the identical
thing on the `trans = false` arm where theirs assigns the other pointers.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from arima.derived.timeSeries.jones_transform import jones_transform, jones_transform_host
from arima.derived.tsa.arima_common import ARIMAOrder, ARIMAParams, ARIMAParamsHost
from original.numerics import ftz

from tsa.derived.timeSeries.arima_helpers import prepare_data, prepare_data_host


comptime MIN_SIGMA2 = Float32(1e-6)


@always_inline
def param_to_poly(is_ar: Bool, param0: Float32, idx: Int, lags: Int) -> Float32:
    """`_param_to_poly<isAr>(param, lags, idx)` with `param[idx - 1]` already
    loaded as `param0` (the caller loads it; a pointer read is not arithmetic):
    `0` past the lags, `1` at index 0, else `-param` (AR) or `param` (MA)."""
    if idx > lags:
        return Float32(0.0)
    elif idx != 0:
        return -param0 if is_ar else param0
    return Float32(1.0)


@always_inline
def reduced_polynomial(is_ar: Bool, coef0: Float32, coef1: Float32) -> Float32:
    """`reduced_polynomial<isAr>`'s last line: `isAr ? -coef0 * coef1 : coef0
    * coef1`, one rounding."""
    var prod = ftz(coef0 * coef1)
    return -prod if is_ar else prod


@always_inline
def reduced_poly_indices(idx: Int, s: Int) -> Tuple[Int, Int]:
    """`idx1 = s ? idx / s : 0; idx0 = idx - s * idx1` (`:187-188`)."""
    var idx1 = idx // s if s != 0 else 0
    var idx0 = idx - s * idx1
    return (idx0, idx1)


def undiff_kernel[
    double_diff: Bool
](
    d_fc: MutPointer[Float32, MutAnyOrigin],
    d_in: MutPointer[Float32, MutAnyOrigin],
    num_steps_in: Int32,
    batch_size_in: Int32,
    in_ld_in: Int32,
    n_in_in: Int32,
    s0_in: Int32,
    s1_in: Int32,
):
    """`_undiff_kernel<double_diff>` (`:137-163`): one thread per series,
    serial over the forecast steps; `_select_read` reads the past when the
    index is negative and the forecast otherwise."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var num_steps = Int(num_steps_in)
    var n_in = Int(n_in_in)
    var s0 = Int(s0_in)
    var s1 = Int(s1_in)
    var fc_base = bid * num_steps
    var in_base = bid * Int(in_ld_in)
    for i in range(num_steps):
        var cur = ftz(d_fc.unsafe_load(fc_base + i))
        comptime if not double_diff:
            var x = _select_read(d_in, in_base, n_in, d_fc, fc_base, i - s0)
            d_fc.unsafe_store(fc_base + i, ftz(cur + x))
        else:
            var x0 = _select_read(d_in, in_base, n_in, d_fc, fc_base, i - s0 - s1)
            var x1 = _select_read(d_in, in_base, n_in, d_fc, fc_base, i - s0)
            var x2 = _select_read(d_in, in_base, n_in, d_fc, fc_base, i - s1)
            var acc = ftz(-x0 + x1)
            acc = ftz(acc + x2)
            d_fc.unsafe_store(fc_base + i, ftz(cur + acc))


@always_inline
def _select_read(
    src0: MutPointer[Float32, MutAnyOrigin], base0: Int, size0: Int,
    src1: MutPointer[Float32, MutAnyOrigin], base1: Int, idx: Int,
) -> Float32:
    if idx < 0:
        return ftz(src0.unsafe_load(base0 + size0 + idx))
    return ftz(src1.unsafe_load(base1 + idx))


def finalize_forecast(
    ctx: DeviceContext,
    mut d_fc: DeviceBuffer[DType.float32],
    mut d_in: DeviceBuffer[DType.float32],
    num_steps: Int,
    batch_size: Int,
    in_ld: Int,
    n_in: Int,
    d: Int,
    D: Int,
    s: Int,
) raises:
    """`:321-350`, `TPB = 64` one thread per series (scheduling)."""
    comptime TPB = 64
    var grid = (batch_size + TPB - 1) // TPB
    if d + D == 1:
        comptime k1 = undiff_kernel[False]
        ctx.enqueue_function[k1](
            d_fc.unsafe_ptr(), d_in.unsafe_ptr(), Int32(num_steps), Int32(batch_size),
            Int32(in_ld), Int32(n_in), Int32(1 if d != 0 else s), Int32(0),
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )
    elif d + D == 2:
        comptime k2 = undiff_kernel[True]
        ctx.enqueue_function[k2](
            d_fc.unsafe_ptr(), d_in.unsafe_ptr(), Int32(num_steps), Int32(batch_size),
            Int32(in_ld), Int32(n_in), Int32(1 if d != 0 else s), Int32(1 if d == 2 else s),
            grid_dim=(grid, 1, 1), block_dim=(TPB, 1, 1),
        )


def finalize_forecast_host(
    mut fc: List[Float32], y: List[Float32], num_steps: Int, batch_size: Int,
    in_ld: Int, n_in: Int, d: Int, D: Int, s: Int,
):
    """The host replay of `_undiff_kernel` (the oracle)."""
    if d + D == 0:
        return
    var s0 = 1 if d != 0 else s
    var s1 = 1 if d == 2 else s
    for bid in range(batch_size):
        var fb = bid * num_steps
        var ib = bid * in_ld
        for i in range(num_steps):
            var cur = ftz(fc[fb + i])
            if d + D == 1:
                var x = _sel_host(y, ib, n_in, fc, fb, i - s0)
                fc[fb + i] = ftz(cur + x)
            else:
                var x0 = _sel_host(y, ib, n_in, fc, fb, i - s0 - s1)
                var x1 = _sel_host(y, ib, n_in, fc, fb, i - s0)
                var x2 = _sel_host(y, ib, n_in, fc, fb, i - s1)
                var acc = ftz(-x0 + x1)
                acc = ftz(acc + x2)
                fc[fb + i] = ftz(cur + acc)


def _sel_host(
    src0: List[Float32], base0: Int, size0: Int, src1: List[Float32], base1: Int, idx: Int
) -> Float32:
    if idx < 0:
        return ftz(src0[base0 + size0 + idx])
    return ftz(src1[base1 + idx])


def sigma2_floor_kernel(
    t_sigma2: MutPointer[Float32, MutAnyOrigin],
    sigma2: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
):
    """`raft::linalg::unaryOp(Tparams.sigma2, params.sigma2, ..., max(input,
    min_sigma2))` (`:372-377`), value-first as theirs."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(batch_size_in):
        return
    t_sigma2.unsafe_store(i, max(ftz(sigma2.unsafe_load(i)), MIN_SIGMA2))


def batched_jones_transform(
    ctx: DeviceContext,
    order: ARIMAOrder,
    batch_size: Int,
    is_inv: Bool,
    mut params: ARIMAParams,
    mut t_params: ARIMAParams,
) raises:
    """`:358-379`: AR, MA, SAR, SMA through `jones_transform` when present;
    sigma2 through the floor. `mu` is not transformed (theirs aliases
    `Tparams.mu = params.mu` at `batched_arima.cu:419-425`); ours copies the
    value, see the module docstring."""
    if order.p != 0:
        jones_transform(ctx, params.ar, batch_size, order.p, t_params.ar, True, is_inv)
    if order.q != 0:
        jones_transform(ctx, params.ma, batch_size, order.q, t_params.ma, False, is_inv)
    if order.P != 0:
        jones_transform(ctx, params.sar, batch_size, order.P, t_params.sar, True, is_inv)
    if order.Q != 0:
        jones_transform(ctx, params.sma, batch_size, order.Q, t_params.sma, False, is_inv)
    comptime TPB = 128
    ctx.enqueue_function[sigma2_floor_kernel](
        t_params.sigma2.unsafe_ptr(), params.sigma2.unsafe_ptr(), Int32(batch_size),
        grid_dim=((batch_size + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
    )
    if order.k != 0:
        var src = params.mu.create_sub_buffer[DType.float32](0, batch_size)
        var dst = t_params.mu.create_sub_buffer[DType.float32](0, batch_size)
        ctx.enqueue_copy(dst_buf=dst, src_buf=src)


def batched_jones_transform_host(
    order: ARIMAOrder, batch_size: Int, is_inv: Bool, params: ARIMAParamsHost
) raises -> ARIMAParamsHost:
    var ar = params.ar.copy()
    var ma = params.ma.copy()
    var sar = params.sar.copy()
    var sma = params.sma.copy()
    if order.p != 0:
        ar = jones_transform_host(params.ar, batch_size, order.p, True, is_inv)
    if order.q != 0:
        ma = jones_transform_host(params.ma, batch_size, order.q, False, is_inv)
    if order.P != 0:
        sar = jones_transform_host(params.sar, batch_size, order.P, True, is_inv)
    if order.Q != 0:
        sma = jones_transform_host(params.sma, batch_size, order.Q, False, is_inv)
    var sigma2 = List[Float32]()
    for i in range(batch_size):
        sigma2.append(max(ftz(params.sigma2[i]), MIN_SIGMA2))
    return ARIMAParamsHost(mu=params.mu.copy(), ar=ar^, ma=ma^, sar=sar^, sma=sma^, sigma2=sigma2^)
