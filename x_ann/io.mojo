# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host <-> device copies for the x_ann drivers (lane ann-apple2, 2026-09-28).

`metrics/checks/device_io.mojo`'s four functions with the same contract
(each synchronizes before it returns), minus the host passes: an upload
copies straight from the caller's list (no private copy first), and a
download lands in the returned list's own words (core/transfer.mojo). Plain copies: the same words. At the IVF shapes these lists are
1M x 28 floats (the residuals) and 1M x 14 codes."""
from max.gpu.host import DeviceBuffer, DeviceContext
from core.transfer import (
    TRANSFER_TIER_IDENTICAL,
    transfer_download_f32_into,
    transfer_download_f32_list,
    transfer_download_i32_into,
    transfer_download_i32_list,
    transfer_upload_ptr,
)
from core.admitted_finite_status import enqueue_finite_status, enqueue_finite_word
from core.device_zero import enqueue_fill

#: The pinned download stages of this module and tier (core/transfer.mojo);
#: a slot holds stages of one element type, so the int32 ones have their own.
comptime _XFER_POOL = "MojolearnXAnnIoTransferStagesIdenticalV1" if TRANSFER_TIER_IDENTICAL else "MojolearnXAnnIoTransferStagesFastV1"
comptime _XFER_I32_POOL = "MojolearnXAnnIoTransferI32StagesIdenticalV1" if TRANSFER_TIER_IDENTICAL else "MojolearnXAnnIoTransferI32StagesFastV1"


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
    # core/transfer.mojo (lane fx-dup-next, 2026-11-19): the copy engine
    # writes the list's own words, through the pooled pinned stages from
    # DOWNLOAD_STAGE_MIN floats. Gone: a pinned host buffer the size of the
    # result per call and a memcpy out of it. The wait for an empty result is
    # kept (each helper here synchronizes before it returns).
    if n <= 0:
        ctx.synchronize()
        return List[Float32]()
    return transfer_download_f32_list[_XFER_POOL](ctx, buf, n)


def download_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    # As `download_f32`, through the pooled pinned int32 stages (the 1M x 14
    # codes).
    if n <= 0:
        ctx.synchronize()
        return List[Int32]()
    return transfer_download_i32_list[_XFER_I32_POOL](ctx, buf, n)


def upload_f32_ptr(
    ctx: DeviceContext, src: MutPointer[Float32, MutUntrackedOrigin], n: Int,
) raises -> DeviceBuffer[DType.float32]:
    """`upload_f32` for `n` words read in place in the CALLER's array (kept
    alive by the caller for the whole call): no list copy of the array
    first. Waits inside, as `upload_f32` does; a one-cell buffer for n <= 0."""
    return transfer_upload_ptr[DType.float32](ctx, src, n if n > 0 else 0, n if n > 0 else 1)


def upload_i32_ptr(
    ctx: DeviceContext, src: MutPointer[Int32, MutUntrackedOrigin], n: Int,
) raises -> DeviceBuffer[DType.int32]:
    """`upload_f32_ptr` for int32 words."""
    return transfer_upload_ptr[DType.int32](ctx, src, n if n > 0 else 0, n if n > 0 else 1)


def download_f32_into(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int, dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`download_f32` written straight into the CALLER's array at `dst` (no
    result list and no copy out of it); waits inside, also for n <= 0."""
    if n <= 0:
        ctx.synchronize()
        return
    transfer_download_f32_into[_XFER_POOL](ctx, buf, n, dst)


def download_i32_into(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int, dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`download_f32_into` for int32 words."""
    if n <= 0:
        ctx.synchronize()
        return
    transfer_download_i32_into[_XFER_I32_POOL](ctx, buf, n, dst)


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
    enqueue_finite_word(ctx, dq.unsafe_ptr(), n, status, 0)
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
