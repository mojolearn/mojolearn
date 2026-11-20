# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE FORM of the family host <-> device helpers (lane fx-transfer-helpers,
2026-11-18): a family's `_upload` / `_download` is one call into this file.

Before, 94 upload and 55 download helpers each carried their own body; 27 of
them walked the words one at a time on one host thread (an `unsafe_store` or
an `append` per element) through a pinned host buffer allocated per call, and
five uploads were the same eleven lines. The forms here are the fastest ones
already in the tree, unchanged:

- upload: one host-to-device copy that reads the caller's storage in place
  (`x_ann/io.mojo`, `resample/estimator.mojo::_upload_from`): no staging
  buffer, no host pass, one wait. The source is the CALLER's (a borrowed List
  or an admitted pointer) and the copy is drained before the return, so the
  queue never reads storage this file could free.
- download of float32: `core/staged_download.mojo::download_f32_into`, straight
  into the destination's pages; from DOWNLOAD_STAGE_MIN floats a pipeline over
  two pooled pinned stages. A List result is created at its final length
  (`unsafe_uninit_length`, every word then written by the copy) and is named
  by the `return` after the drain.
- download of the integer types: one device-to-host copy into the List's own
  storage (labels, counts). int32 results that can be large (drawn row
  indices) take `transfer_download_i32_list`: the same two-stage pipeline as
  the float32 one from DOWNLOAD_STAGE_MIN words, over its own pooled pinned
  int32 stages (lane fx-dup-next, 2026-11-19).
- upload into a buffer the caller already holds: `transfer_upload_into`, the
  same in-place copy as `transfer_upload_list` without the allocation.

Nothing here allocates host memory besides the returned List or takes a lock;
each entry reads the activity record's switch once (`core/launch_log.mojo`
`activity_uploaded` / `activity_downloaded`: one slot read and one Bool test
beside a transfer that waits on the device, and a count only when a
verification asked); no helper waits more often than the one it replaced. Not
split over host tasks: the copy engine is the unit (one transfer per call),
and the staged copy-out is `core/staged_download.mojo`'s. Copies only: no bit
moves.

`pool` names the pinned stages' process slot, as `download_f32_into`'s does:
one name per family and numeric tier (`TRANSFER_TIER_IDENTICAL` picks the
tier half of the name), so a stage is never handed to another binding's
context.
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.launch_log import activity_downloaded, activity_uploaded
from core.shared_slot import SharedSlot
from core.staged_download import (
    DOWNLOAD_STAGE_KEEP, download_chunk_floats, download_f32_into, download_staged,
)
from core.step_phase import step_count_d2h, step_count_h2d, step_count_host_alloc, step_count_sync

#: The tier half of a family's stage pool name:
#: `comptime _POOL = "Mojolearn<Family>TransferStagesIdenticalV1" if TRANSFER_TIER_IDENTICAL else "Mojolearn<Family>TransferStagesFastV1"`.
comptime TRANSFER_TIER_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def transfer_upload_list[dt: DType](
    ctx: DeviceContext, values: List[Scalar[dt]], n: Int, length: Int,
) raises -> DeviceBuffer[dt]:
    """A new device buffer of `length` cells whose first `n` are
    `values[0:n]`, copied from the list's storage in place; waits inside.
    Cells from `n` on are not written (`length` is `n`, or the callers'
    one-cell floor for an empty list)."""
    if n > len(values) or n > length:
        raise Error(
            "transfer_upload_list: " + String(n) + " elements from a list of " + String(len(values))
            + " into a buffer of " + String(length)
        )
    var buf = ctx.enqueue_create_buffer[dt](length)
    if n > 0:
        activity_uploaded(n * size_of[Scalar[dt]]())
        step_count_h2d()
        if n == length:
            ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
            step_count_sync()
            ctx.synchronize()
        else:
            var view = buf.create_sub_buffer[dt](0, n)
            ctx.enqueue_copy(dst_buf=view, src_ptr=values.unsafe_ptr())
            step_count_sync()
            ctx.synchronize()
            _ = view^
    else:
        step_count_sync()
        ctx.synchronize()
    return buf^


def transfer_upload_into[dt: DType](
    ctx: DeviceContext, mut buf: DeviceBuffer[dt], values: List[Scalar[dt]], n: Int,
) raises:
    """`values[0:n]` into the first `n` cells of `buf`, a buffer the CALLER
    already holds, copied from the list's storage in place (no staging copy
    of the list, no host buffer); waits inside, so the queue never reads the
    list after the return. Nothing is queued and nothing waits for `n <= 0`."""
    if n <= 0:
        return
    if n > len(values) or n > len(buf):
        raise Error(
            "transfer_upload_into: " + String(n) + " elements from a list of " + String(len(values))
            + " into a buffer of " + String(len(buf))
        )
    activity_uploaded(n * size_of[Scalar[dt]]())
    step_count_h2d()
    if n == len(buf):
        ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
        step_count_sync()
        ctx.synchronize()
    else:
        var view = buf.create_sub_buffer[dt](0, n)
        ctx.enqueue_copy(dst_buf=view, src_ptr=values.unsafe_ptr())
        step_count_sync()
        ctx.synchronize()
        _ = view^


def transfer_upload_ptr[dt: DType](
    ctx: DeviceContext, src: MutPointer[Scalar[dt], MutUntrackedOrigin], n: Int, length: Int,
) raises -> DeviceBuffer[dt]:
    """`transfer_upload_list` for `n` words read in place at `src`, host
    memory the CALLER owns and keeps alive for the whole call; waits inside."""
    if n > length:
        raise Error("transfer_upload_ptr: " + String(n) + " elements into a buffer of " + String(length))
    var buf = ctx.enqueue_create_buffer[dt](length)
    if n > 0:
        activity_uploaded(n * size_of[Scalar[dt]]())
        step_count_h2d()
        if n == length:
            ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
            step_count_sync()
            ctx.synchronize()
        else:
            var view = buf.create_sub_buffer[dt](0, n)
            ctx.enqueue_copy(dst_buf=view, src_ptr=src)
            step_count_sync()
            ctx.synchronize()
            _ = view^
    else:
        step_count_sync()
        ctx.synchronize()
    return buf^


def transfer_download_f32_into[pool: StaticString](
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.float32],
    n: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """The first `n` floats of `buf` written straight into `dst`, host memory
    the CALLER owns and keeps alive for the whole call; waits inside (once
    below the stage threshold). `buf` is a read borrow: the staged route goes
    through a view of it, because `download_f32_into` takes a mutable one."""
    if n <= 0:
        return
    if n > len(buf):
        raise Error("transfer_download_f32_into: " + String(n) + " elements from a buffer of " + String(len(buf)))
    activity_downloaded(n * 4)
    if download_staged(n):
        var staged = buf.create_sub_buffer[DType.float32](0, n)
        download_f32_into[pool](ctx, staged, n, dst)
        _ = staged^
        return
    step_count_d2h()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
        step_count_sync()
        ctx.synchronize()
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
        step_count_sync()
        ctx.synchronize()
        _ = view^


def transfer_download_f32_list[pool: StaticString](
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int,
) raises -> List[Float32]:
    """The first `n` floats of `buf` as a List of that length (empty for
    `n <= 0`): `transfer_download_f32_into` on the list's own storage."""
    if n <= 0:
        return List[Float32]()
    var dst = List[Float32](unsafe_uninit_length=n)
    transfer_download_f32_into[pool](
        ctx, buf, n, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr())),
    )
    # `dst` is named here, after the drain inside the call above.
    return dst^


def transfer_download_list[dt: DType](
    ctx: DeviceContext, buf: DeviceBuffer[dt], n: Int,
) raises -> List[Scalar[dt]]:
    """The first `n` cells of `buf` as a List of that length (empty for
    `n <= 0`): one device-to-host copy into the list's own storage, one wait.
    For the integer types; float32 goes through `transfer_download_f32_list`
    (the pooled stage above the threshold)."""
    if n <= 0:
        return List[Scalar[dt]]()
    if n > len(buf):
        raise Error("transfer_download_list: " + String(n) + " elements from a buffer of " + String(len(buf)))
    activity_downloaded(n * size_of[Scalar[dt]]())
    var dst = List[Scalar[dt]](unsafe_uninit_length=n)
    step_count_d2h()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=dst.unsafe_ptr(), src_buf=buf)
        step_count_sync()
        ctx.synchronize()
    else:
        var view = buf.create_sub_buffer[dt](0, n)
        ctx.enqueue_copy(dst_ptr=dst.unsafe_ptr(), src_buf=view)
        step_count_sync()
        ctx.synchronize()
        _ = view^
    # `dst` is named here, after the drain.
    return dst^


# ---------------------------------------------------------------------------
# The int32 stage pool (lane fx-dup-next, 2026-11-19): `core/staged_download`'s
# two-stage pipeline, word for word, over pinned int32 stages. Its float32
# stages cannot carry an int32 buffer (the copy engine's source and
# destination share one element type), so the int32 results that can be
# millions of words (resample's drawn row indices) had only the direct copy
# into pageable list pages. The threshold and the chunk are the float32
# pipeline's (`download_staged`, `download_chunk_floats`: both count 4-byte
# words). A transport choice: the same words land in the same cells.
# ---------------------------------------------------------------------------


struct _I32StagePool(Defaultable, Movable):
    """The pinned int32 stages kept between calls, all of `n` words."""
    var bufs: List[HostBuffer[DType.int32]]
    var n: Int

    def __init__(out self):
        self.bufs = List[HostBuffer[DType.int32]]()
        self.n = 0


def _i32_stage_take[name: StaticString](ctx: DeviceContext, chunk: Int) raises -> HostBuffer[DType.int32]:
    """A pooled pinned stage of at least `chunk` words, else a new one (a
    larger chunk retires the smaller stages)."""
    comptime SLOT = SharedSlot[_I32StagePool, name]
    var pool = SLOT.get_or_create_ptr()
    if pool[].n < chunk:
        pool[].bufs.clear()
        pool[].n = chunk
    if len(pool[].bufs) > 0:
        return pool[].bufs.pop()
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.int32](pool[].n)


def _i32_stage_give[name: StaticString](var stage: HostBuffer[DType.int32]) raises:
    comptime SLOT = SharedSlot[_I32StagePool, name]
    var pool = SLOT.get_or_create_ptr()
    if len(stage) == pool[].n and len(pool[].bufs) < DOWNLOAD_STAGE_KEEP:
        pool[].bufs.append(stage^)


def _i32_stage_enqueue(
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.int32],
    n: Int,
    k: Int,
    chunk: Int,
    mut stage: HostBuffer[DType.int32],
) raises:
    """Chunk `k` of the first `n` words of `buf` onto `stage`: buffer to
    buffer when the chunk is whole and the stage is exactly `chunk` long,
    otherwise the chunk's view onto the stage's pointer (the pool hands out
    stages of ITS length, which can be longer than `chunk`)."""
    var lo = k * chunk
    step_count_d2h()
    if lo + chunk <= len(buf) and len(stage) == chunk:
        var view = buf.create_sub_buffer[DType.int32](lo, chunk)
        ctx.enqueue_copy(dst_buf=stage, src_buf=view)
        _ = view^
    else:
        var view = buf.create_sub_buffer[DType.int32](lo, min(chunk, n - lo))
        ctx.enqueue_copy(dst_ptr=stage.unsafe_ptr(), src_buf=view)
        _ = view^


@always_inline
def _i32_copy_out(
    dst: MutPointer[Int32, MutUntrackedOrigin], src: MutPointer[Int32, MutUntrackedOrigin], n: Int,
):
    """`memcpy(dst, src, n)` from a pinned stage into the caller's memory, on
    the calling thread while the copy engine fills the other stage (serial by
    nature: one stage is drained at a time, as in the float32 pipeline)."""
    unsafe_memcpy(dest=dst, src=src, count=n)


def transfer_download_i32_into[pool: StaticString](
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.int32],
    n: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """The first `n` words of `buf` written straight into `dst`, host memory
    the CALLER owns and keeps alive for the whole call; waits inside. Below
    the stage threshold one direct copy and one wait; from it, a pipeline
    over chunks through two pooled pinned stages named by `pool` (one name
    per binding and tier, and NOT the name of a float32 stage pool: the slot
    holds stages of one element type)."""
    if n <= 0:
        return
    if n > len(buf):
        raise Error("transfer_download_i32_into: " + String(n) + " elements from a buffer of " + String(len(buf)))
    activity_downloaded(n * 4)
    if not download_staged(n):
        step_count_d2h()
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
            step_count_sync()
            ctx.synchronize()
        else:
            var view = buf.create_sub_buffer[DType.int32](0, n)
            ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
            step_count_sync()
            ctx.synchronize()
            _ = view^
        return
    var chunk = download_chunk_floats()
    if chunk > n:
        chunk = n
    var nchunks = (n + chunk - 1) // chunk
    var st0 = _i32_stage_take[pool](ctx, chunk)
    var st1 = _i32_stage_take[pool](ctx, chunk)
    _i32_stage_enqueue(ctx, buf, n, 0, chunk, st0)
    var k = 0
    # driver-loop: over the download's chunks, two a turn (one per stage)
    while k < nchunks:
        step_count_sync()
        ctx.synchronize()
        if k + 1 < nchunks:
            _i32_stage_enqueue(ctx, buf, n, k + 1, chunk, st1)
        _i32_copy_out(dst + k * chunk, st0.unsafe_ptr(), min(chunk, n - k * chunk))
        if k + 1 >= nchunks:
            break
        step_count_sync()
        ctx.synchronize()
        if k + 2 < nchunks:
            _i32_stage_enqueue(ctx, buf, n, k + 2, chunk, st0)
        _i32_copy_out(dst + (k + 1) * chunk, st1.unsafe_ptr(), min(chunk, n - (k + 1) * chunk))
        k += 2
    _i32_stage_give[pool](st0^)
    _i32_stage_give[pool](st1^)


def transfer_download_i32_list[pool: StaticString](
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int,
) raises -> List[Int32]:
    """The first `n` words of `buf` as a List of that length (empty for
    `n <= 0`): `transfer_download_i32_into` on the list's own storage."""
    if n <= 0:
        return List[Int32]()
    var dst = List[Int32](unsafe_uninit_length=n)
    transfer_download_i32_into[pool](
        ctx, buf, n, MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(dst.unsafe_ptr())),
    )
    # `dst` is named here, after the drain inside the call above.
    return dst^
