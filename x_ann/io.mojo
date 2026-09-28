# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host <-> device copies for the x_ann drivers (lane ann-apple2, 2026-09-28).

`metrics/checks/device_io.mojo`'s four functions with the same contract
(each synchronizes before it returns), minus two host passes: an upload
copies straight from the caller's list (no private copy first), and a
download moves the staged words with one memcpy instead of appending them
one by one. Plain copies: the same words. At the IVF shapes these lists are
1M x 28 floats (the residuals) and 1M x 14 codes."""
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext


def upload_f32(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
    ctx.synchronize()
    return buf^


def upload_i32(ctx: DeviceContext, values: List[Int32]) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
    ctx.synchronize()
    return buf^


def download_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[DType.float32](0, n)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^


def download_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[DType.int32](0, n)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^
