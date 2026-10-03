# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Mamba-1 input-side fusions (lane afn-mamba, 2026-10-03;
`-D MOJOLEARN_AFN_MAMBA1_FUSE_IN`, FAST + Apple only; `mamba_mixer_forward`
dispatches here).

Main's conv is one thread per `(batch, channel)` walking `l` ascending
(768 threads x 2048 steps at the board shape) plus a second launch for the
window update; `A = -exp(A_log)` and the x_proj split are two more
launches, each followed by a wait.

  afn_m1_conv_token_kernel   ONE launch, one thread per `(token, channel)`
                             (1.5 M threads at the board shape): the four
                             bias-seeded taps, SiLU, and the window update
                             by the threads that own the last D_CONV
                             positions (or, at l < D_CONV, the token-0
                             thread copying the old window's slots).
  afn_m1_split_a_kernel      ONE launch covering the x_proj split (one
                             thread per element, not per row) and
                             A = -exp(A_log).

Per-element arithmetic is main's (bias seed, taps k ascending, one fma
per tap, `identical_silu`, `identical_exp`); only the launch shape
changes, so no float crosses a thread boundary differently.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_exp, identical_mul_add, identical_silu
from mamba.checks.mamba_fixture import D_CONV, D_STATE

comptime AFN_M1_TPB = 256


def _afn_grid(n: Int) -> Int:
    var g = (n + AFN_M1_TPB - 1) // AFN_M1_TPB
    if g < 1:
        return 1
    return g


def afn_m1_conv_token_kernel(
    conv_out: MutPointer[Float32, MutAnyOrigin],  # [M, di]
    silu_out: MutPointer[Float32, MutAnyOrigin],  # [M, di]
    new_win: MutPointer[Float32, MutAnyOrigin],  # [B, di, D_CONV]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, 2 di]
    conv_w: MutPointer[Float32, MutAnyOrigin],  # [di, D_CONV]
    conv_b: MutPointer[Float32, MutAnyOrigin],  # [di]
    old_win: MutPointer[Float32, MutAnyOrigin],  # [B, di, D_CONV]
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
):
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * di:
        return
    var bb = cell // (l * di)
    var rem = cell - bb * l * di
    var li = rem // di
    var d = rem - li * di
    var row = (bb * l + li) * 2 * di + d

    # S13, FUSED, BIAS-SEEDED, taps k = 0..3 ascending (oldest first).
    var acc = ftz(conv_b.unsafe_load(d))
    for k in range(D_CONV):
        var p = li - (D_CONV - 1) + k
        var xv: Float32
        if p >= 0:
            xv = in_proj.unsafe_load((bb * l + p) * 2 * di + d)
        else:
            xv = old_win.unsafe_load((bb * di + d) * D_CONV + (D_CONV + p))
        acc = ftz(
            identical_mul_add(ftz(conv_w.unsafe_load(d * D_CONV + k)), ftz(xv), acc)
        )
    conv_out.unsafe_store((bb * l + li) * di + d, acc)
    silu_out.unsafe_store((bb * l + li) * di + d, ftz(identical_silu(acc)))

    # The window AFTER the call: the last D_CONV conv INPUTS, oldest first
    # (copies, not a seam). Slot j holds position l - D_CONV + j.
    var j = li - (l - D_CONV)
    if j >= 0:
        new_win.unsafe_store((bb * di + d) * D_CONV + j, in_proj.unsafe_load(row))
    if li == 0:
        for jj in range(D_CONV):
            var p = l - D_CONV + jj
            if p < 0:
                new_win.unsafe_store(
                    (bb * di + d) * D_CONV + jj,
                    old_win.unsafe_load((bb * di + d) * D_CONV + (D_CONV + p)),
                )


def afn_m1_split_a_kernel(
    dt_low: MutPointer[Float32, MutAnyOrigin],  # [M, r]
    b_mat: MutPointer[Float32, MutAnyOrigin],  # [M, D_STATE]
    c_mat: MutPointer[Float32, MutAnyOrigin],  # [M, D_STATE]
    x_proj: MutPointer[Float32, MutAnyOrigin],  # [M, xr]
    a_out: MutPointer[Float32, MutAnyOrigin],  # [di, D_STATE]
    a_log: MutPointer[Float32, MutAnyOrigin],  # [di, D_STATE]
    m_in: Int32,
    r_in: Int32,
    xr_in: Int32,
    n_a_in: Int32,
):
    var m = Int(m_in)
    var r = Int(r_in)
    var xr = Int(xr_in)
    var n_a = Int(n_a_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < m * xr:
        var t = i // xr
        var j = i - t * xr
        var v = x_proj.unsafe_load(i)
        if j < r:
            dt_low.unsafe_store(t * r + j, v)
        elif j < r + D_STATE:
            b_mat.unsafe_store(t * D_STATE + (j - r), v)
        else:
            c_mat.unsafe_store(t * D_STATE + (j - r - D_STATE), v)
        return
    var ia = i - m * xr
    if ia >= n_a:
        return
    # S15: `-ftz(identical_exp(ftz(A_log)))`; the negation is exact.
    a_out.unsafe_store(ia, -ftz(identical_exp(ftz(a_log.unsafe_load(ia)))))


def afn_m1_conv_token(
    ctx: DeviceContext,
    mut conv_out: DeviceBuffer[DType.float32],
    mut silu_out: DeviceBuffer[DType.float32],
    mut new_win: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut conv_w: DeviceBuffer[DType.float32],
    mut conv_b: DeviceBuffer[DType.float32],
    mut old_win: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    d_inner: Int,
) raises:
    """`causal_conv1d_fn` + the window update as ONE token-parallel launch.
    ASYNCHRONOUS."""
    ctx.enqueue_function[afn_m1_conv_token_kernel](
        conv_out.unsafe_ptr(),
        silu_out.unsafe_ptr(),
        new_win.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        conv_w.unsafe_ptr(),
        conv_b.unsafe_ptr(),
        old_win.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(d_inner),
        grid_dim=(_afn_grid(b * l * d_inner), 1, 1),
        block_dim=(AFN_M1_TPB, 1, 1),
    )


def afn_m1_split_a(
    ctx: DeviceContext,
    mut dt_low: DeviceBuffer[DType.float32],
    mut b_mat: DeviceBuffer[DType.float32],
    mut c_mat: DeviceBuffer[DType.float32],
    mut x_proj: DeviceBuffer[DType.float32],
    mut a_out: DeviceBuffer[DType.float32],
    mut a_log: DeviceBuffer[DType.float32],
    m: Int,
    r: Int,
    xr: Int,
    d_inner: Int,
) raises:
    """The x_proj split (per element) and A = -exp(A_log) as ONE launch.
    ASYNCHRONOUS."""
    var n_a = d_inner * D_STATE
    ctx.enqueue_function[afn_m1_split_a_kernel](
        dt_low.unsafe_ptr(),
        b_mat.unsafe_ptr(),
        c_mat.unsafe_ptr(),
        x_proj.unsafe_ptr(),
        a_out.unsafe_ptr(),
        a_log.unsafe_ptr(),
        Int32(m),
        Int32(r),
        Int32(xr),
        Int32(n_a),
        grid_dim=(_afn_grid(m * xr + n_a), 1, 1),
        block_dim=(AFN_M1_TPB, 1, 1),
    )
