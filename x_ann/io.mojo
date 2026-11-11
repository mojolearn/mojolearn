# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host <-> device copies for the x_ann drivers (lane ann-apple2, 2026-09-28).

`metrics/checks/device_io.mojo`'s four functions with the same contract
(each synchronizes before it returns), minus two host passes: an upload
copies straight from the caller's list (no private copy first), and a
download moves the staged words with one memcpy instead of appending them
one by one. Plain copies: the same words. At the IVF shapes these lists are
1M x 28 floats (the residuals) and 1M x 14 codes."""
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from core.admitted_finite_status import enqueue_finite_status
from core.device_zero import enqueue_fill


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
        unsafe_memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
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
        unsafe_memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^


def enqueue_query_status(
    ctx: DeviceContext, mut dq: DeviceBuffer[DType.float32], n: Int, status_addr: Int,
) raises -> DeviceBuffer[DType.int32]:
    """The finite status of the `n` query words just uploaded into `dq`
    (lane a8b-device-x-neighbors-rest; core/admitted_finite_status.mojo
    contract: 0 when every word is finite, 1 otherwise), written to the host
    int32 word at `status_addr` with the search's results: the test and the
    copy are enqueued behind the upload, and the search's own downloads
    synchronise them (no new synchronisation). `status_addr` 0: nothing is
    enqueued. The caller keeps the returned buffer alive until its
    downloads have synchronised; the door refuses on a 1 word through
    `admitted_finite_refuse` (its outputs are then discarded)."""
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    if status_addr == 0:
        return status^
    if status_addr < 0 or status_addr % 4 != 0:
        raise Error("x_ann: the status address must be a positive int32-aligned address")
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    enqueue_finite_status(ctx, dq.unsafe_ptr(), n, status.unsafe_ptr(), 0)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutAnyOrigin](unsafe_from_address=status_addr), src_buf=status)
    return status^


def enqueue_query_status(
    ctx: DeviceContext, mut da: DeviceBuffer[DType.float32], na: Int,
    mut db: DeviceBuffer[DType.float32], nb: Int, status_addr: Int,
) raises -> DeviceBuffer[DType.int32]:
    """`enqueue_query_status` over two uploads in one word (refine's dataset
    and queries): 1 when either holds a NaN or an infinity."""
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    if status_addr == 0:
        return status^
    if status_addr < 0 or status_addr % 4 != 0:
        raise Error("x_ann: the status address must be a positive int32-aligned address")
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    enqueue_finite_status(ctx, da.unsafe_ptr(), na, status.unsafe_ptr(), 0)
    enqueue_finite_status(ctx, db.unsafe_ptr(), nb, status.unsafe_ptr(), 0)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutAnyOrigin](unsafe_from_address=status_addr), src_buf=status)
    return status^
