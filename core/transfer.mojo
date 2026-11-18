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
  storage. There is no pooled stage for them (the stage pool is float32); the
  callers' integer results are labels, counts and indices.

Nothing here allocates host memory besides the returned List, takes a lock or
looks anything up; no helper waits more often than the one it replaced. Not
split over host tasks: the copy engine is the unit (one transfer per call),
and the staged copy-out is `core/staged_download.mojo`'s. Copies only: no bit
moves.

`pool` names the pinned stages' process slot, as `download_f32_into`'s does:
one name per family and numeric tier (`TRANSFER_TIER_IDENTICAL` picks the
tier half of the name), so a stage is never handed to another binding's
context.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.staged_download import download_f32_into, download_staged
from core.step_phase import step_count_d2h, step_count_h2d, step_count_sync

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


def transfer_upload_ptr[dt: DType](
    ctx: DeviceContext, src: MutPointer[Scalar[dt], MutUntrackedOrigin], n: Int, length: Int,
) raises -> DeviceBuffer[dt]:
    """`transfer_upload_list` for `n` words read in place at `src`, host
    memory the CALLER owns and keeps alive for the whole call; waits inside."""
    if n > length:
        raise Error("transfer_upload_ptr: " + String(n) + " elements into a buffer of " + String(length))
    var buf = ctx.enqueue_create_buffer[dt](length)
    if n > 0:
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
