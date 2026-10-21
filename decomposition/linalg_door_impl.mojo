# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed entries of the linalg door of `bindings/_mojolearn_linalg.mojo`: the
float16 widening launch and the result copies of qr_r and svdvals, moved
here with no Python import (lane thin-core, 2026-10-21: a pure move, no bit
change)."""
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from gemm.f16_widen import f16_bits_to_f32
from decomposition.linalg_public_device import (
    device_qr_r,
    device_svdvals,
)


def f16_widen_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[UInt16, MutAnyOrigin],
    n_in: Int32,
):
    """IEEE float16 bits to float32 (`gemm/f16_widen.mojo`), one element
    per thread."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, f16_bits_to_f32(src.unsafe_load(i)))


comptime F16_WIDEN_TPB = 256


def from_f16_device(
    ctx: DeviceContext,
    dst: MutPointer[Float32, MutUntrackedOrigin],
    src: MutPointer[UInt16, MutUntrackedOrigin],
    count: Int,
) raises:
    """Widen `count` float16 words at `src` into float32 at `dst` on the device."""
    var n = count
    if n < 1:
        n = 1
    var dsrc = ctx.enqueue_create_buffer[DType.uint16](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=dsrc, src_ptr=src)
    var ddst = ctx.enqueue_create_buffer[DType.float32](count)
    ctx.enqueue_function[f16_widen_kernel](
        ddst.unsafe_ptr(),
        dsrc.unsafe_ptr(),
        Int32(count),
        grid_dim=((count + F16_WIDEN_TPB - 1) // F16_WIDEN_TPB, 1, 1),
        block_dim=(F16_WIDEN_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=dst, src_buf=ddst)
    ctx.synchronize()
    _ = dsrc^
    _ = ddst^
    # DEVIATION 3010: drain the frees before the context goes.
    ctx.synchronize()


def qr_r_device_into(
    a: List[Float32], n_rows: Int, n_cols: Int,
    rp: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`device_qr_r(a, n_rows, n_cols)`, the n_cols x n_cols R into rp."""
    var r = device_qr_r(a, n_rows, n_cols)
    for i in range(n_cols * n_cols):
        rp.unsafe_store(i, r[i])
    _ = r^


def svdvals_device_into(
    a: List[Float32], n_rows: Int, n_cols: Int,
    sp: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`device_svdvals(a, n_rows, n_cols)`, the n_cols values into sp."""
    var s = device_svdvals(a, n_rows, n_cols)
    for i in range(n_cols):
        sp.unsafe_store(i, s[i])
    _ = s^
