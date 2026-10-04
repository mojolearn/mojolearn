# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE DEVICE-TO-HOST DOWNLOAD of a float32 buffer into caller memory
(`download_f32_into`, `download_f32_into_scanned`): a pipeline over chunks
through two pooled pinned stages, the copy-out over host tasks.

MOVED here unchanged from `training/checks/train_loop.mojo` (lane
neural-pass138, 2026-10-02) so that a binding which does not link the
training lane (the embedding binding) and the cross-entropy host entry
(`training/estimator.mojo`) download through the same transport instead of
their own per-call pinned allocation or raw copy. `train_loop` re-imports
every name, so `from training.checks.train_loop import download_f32_into`
still resolves. The stage pool keeps its `_Global` slot name, so the
process still holds ONE pool per numeric tier. Copies only: no bit moves.
"""
from std.memory import bitcast, memcpy
from std.os import getenv
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.step_phase import (
    step_count_d2h,
    step_count_host_alloc,
    step_count_sync,
)


def download_f32_into[pool: StaticString = _STAGE_NAME](
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`n` elements of a device buffer written straight into `dst`, host
    memory the CALLER owns and keeps alive for the whole call; waits inside.
    DEVIATION 3120: the byte-LM exports (state, gradient, device fold) used
    `download_f32` and then copied the returned List into the caller's
    array: a pinned host buffer allocated per call, a device-to-host copy
    into it, an element loop appending into a List, then a third pass into
    the caller. At 162M parameters that was about 4 s a step for 2.59 GB.
    Here the copy engine writes the caller's pages directly, one transfer,
    no host buffer and no loop: the same bytes, as `cluster/estimator.mojo`
    (DEVIATION 2672) already does on every vendor. `n` IS PASSED, as in
    `download_f32`; a prefix of a larger buffer goes through a sub-buffer.
    On a raise the caller's memory holds an unspecified prefix of the copy;
    a caller that must not see a partial export stages it elsewhere.
    From DOWNLOAD_STAGE_MIN floats the copy goes through the pinned stage
    pipeline (`download_f32_into_scanned` with the scan off).
    """
    _ = download_f32_into_scanned[pool](ctx, buf, n, dst, False)


def download_f32_into_scanned[pool: StaticString = _STAGE_NAME](
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
    scan: Bool,
) raises -> Int:
    """`download_f32_into`, returning the first flat index of a non-finite
    element of `dst` (-1 when every element is finite, or when `scan` is
    off): the exponent-all-ones test on the bits, the same test
    `_refuse_nonfinite_logits` makes, folded into the copy-out so the pages
    are scanned while they are hot.
    lane/neural-pass49 (2026-10-01): the staged copy is a PIPELINE over
    chunks of `download_chunk_floats()` through TWO pinned stages kept in a
    process pool (`DOWNLOAD_STAGES`): the copy engine fills one stage while
    one host thread drains the other into `dst`. The per-call 64 MiB pinned
    allocation of the first staged form (lane neural-pass43) is gone; it was
    what made the stage lose to the raw host-pointer copy on the L40S. A
    transport choice: the same bytes land in the same places.
    `pool` names the `_Global` slot of the stages: the default is the byte
    LM's; a binding that links this file for its own downloads passes its
    own name (one per binding and tier, as `core/neural_context.mojo`'s
    contexts), so no pinned stage is handed to another binding's context.
    """
    if n < 1:
        return -1
    if n > len(buf):
        raise Error("download_f32_into: " + String(n) + " elements from a buffer of " + String(len(buf)))
    if not download_staged(n):
        if n == len(buf):
            step_count_d2h()
            ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
            step_count_sync()
            ctx.synchronize()
        else:
            var view = buf.create_sub_buffer[DType.float32](0, n)
            step_count_d2h()
            ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
            step_count_sync()
            ctx.synchronize()
            _ = view^
        if scan:
            return _first_nonfinite(dst, n)
        return -1
    var chunk = download_chunk_floats()
    if chunk > n:
        chunk = n
    var nchunks = (n + chunk - 1) // chunk
    var st0 = _stage_take[pool](ctx, chunk)
    var st1 = _stage_take[pool](ctx, chunk)
    var bad = -1
    _stage_enqueue(ctx, buf, n, 0, chunk, st0)
    var k = 0
    while k < nchunks:
        step_count_sync()
        ctx.synchronize()
        if k + 1 < nchunks:
            _stage_enqueue(ctx, buf, n, k + 1, chunk, st1)
        var b0 = _copy_out(dst + k * chunk, st0.unsafe_ptr(), min(chunk, n - k * chunk), scan)
        if b0 >= 0 and bad < 0:
            bad = k * chunk + b0
        if k + 1 >= nchunks:
            break
        step_count_sync()
        ctx.synchronize()
        if k + 2 < nchunks:
            _stage_enqueue(ctx, buf, n, k + 2, chunk, st0)
        var b1 = _copy_out(dst + (k + 1) * chunk, st1.unsafe_ptr(), min(chunk, n - (k + 1) * chunk), scan)
        if b1 >= 0 and bad < 0:
            bad = (k + 1) * chunk + b1
        k += 2
    _stage_give[pool](st0^)
    _stage_give[pool](st1^)
    return bad


def _stage_enqueue(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    k: Int,
    chunk: Int,
    mut stage: HostBuffer[DType.float32],
) raises:
    """Chunk `k` of the first `n` elements of `buf` onto `stage` (`chunk`
    floats long). A full chunk the stage's exact length goes buffer to
    buffer; otherwise the chunk's view (at most `chunk`, ending at `n`) lands
    on the stage's pointer. The pool hands out stages of ITS length, longer
    than `chunk` when an earlier call used a larger chunk (an output of 1M to
    2M floats after a larger one): the buffer-to-buffer copy then raised
    "not enough data in src" (lane/apple-fast-batchv, 2026-10-03)."""
    var lo = k * chunk
    step_count_d2h()
    if lo + chunk <= len(buf) and len(stage) == chunk:
        var view = buf.create_sub_buffer[DType.float32](lo, chunk)
        ctx.enqueue_copy(dst_buf=stage, src_buf=view)
        _ = view^
    else:
        var view = buf.create_sub_buffer[DType.float32](lo, min(chunk, n - lo))
        ctx.enqueue_copy(dst_ptr=stage.unsafe_ptr(), src_buf=view)
        _ = view^


comptime DOWNLOAD_STAGE_MIN = 1 << 20
comptime DOWNLOAD_CHUNK_FLOATS = 1 << 21
comptime DOWNLOAD_STAGE_KEEP = 2


def download_staged(n: Int) -> Bool:
    """Whether `download_f32_into` goes through the pinned stage pipeline:
    from DOWNLOAD_STAGE_MIN floats on every column (lane neural-pass49,
    2026-10-01; the Apple column since lane neural-pass43). The raw
    host-pointer copy is `MOJOLEARN_DOWNLOAD_STAGE=0`; `=1` forces the
    stage below the minimum. A transport choice: no bit moves."""
    var v = String(getenv("MOJOLEARN_DOWNLOAD_STAGE"))
    if v == "0":
        return False
    if v == "1":
        return True
    return n >= DOWNLOAD_STAGE_MIN


def download_chunk_floats() -> Int:
    """Floats per pipeline chunk: DOWNLOAD_CHUNK_FLOATS (2M, 8 MiB), or
    `MOJOLEARN_DOWNLOAD_CHUNK` (floats, at least 64K) for the sweep."""
    var v = String(getenv("MOJOLEARN_DOWNLOAD_CHUNK"))
    if v != "":
        try:
            var c = Int(v)
            if c >= (1 << 16):
                return c
        except:
            pass
    return DOWNLOAD_CHUNK_FLOATS


struct _StagePool(Defaultable, Movable):
    """The pinned stages kept between calls, all of `n` floats."""
    var bufs: List[HostBuffer[DType.float32]]
    var n: Int

    def __init__(out self):
        self.bufs = List[HostBuffer[DType.float32]]()
        self.n = 0


comptime _STAGE_NAME = "MojoDownloadStagesIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoDownloadStagesFast"
comptime DOWNLOAD_STAGES = _Global[StorageType=_StagePool, name=_STAGE_NAME, init_fn=_StagePool.__init__]


def _stage_take[name: StaticString](ctx: DeviceContext, chunk: Int) raises -> HostBuffer[DType.float32]:
    """A pooled pinned stage of at least `chunk` floats, else a new one (a
    larger chunk retires the smaller stages)."""
    comptime SLOT = _Global[StorageType=_StagePool, name=name, init_fn=_StagePool.__init__]
    var pool = SLOT.get_or_create_ptr()
    if pool[].n < chunk:
        pool[].bufs.clear()
        pool[].n = chunk
    if len(pool[].bufs) > 0:
        return pool[].bufs.pop()
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.float32](pool[].n)


def _stage_give[name: StaticString](var stage: HostBuffer[DType.float32]) raises:
    comptime SLOT = _Global[StorageType=_StagePool, name=name, init_fn=_StagePool.__init__]
    var pool = SLOT.get_or_create_ptr()
    if len(stage) == pool[].n and len(pool[].bufs) < DOWNLOAD_STAGE_KEEP:
        pool[].bufs.append(stage^)


@always_inline
def _first_nonfinite(p: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> Int:
    """The first index in `p[0:n]` whose exponent bits are all ones (an
    infinity or a NaN), -1 when none: eight lanes at a time, the scalar
    tail once a vector has one. No floating-point arithmetic."""
    comptime W = 8
    var exp = SIMD[DType.uint32, W](0x7F800000)
    var i = 0
    var body = n - n % W
    while i < body:
        var bits = bitcast[DType.uint32, W](p.unsafe_load[width=W](i)) & exp
        if bits.eq(exp).reduce_or():
            break
        i += W
    while i < n:
        if (bitcast[DType.uint32](p.unsafe_load(i)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return i
        i += 1
    return -1


def _copy_out(dst: MutPointer[Float32, MutUntrackedOrigin], src: MutPointer[Float32, MutUntrackedOrigin], n: Int, scan: Bool) -> Int:
    """`memcpy(dst, src, n)` from a pinned stage into the caller's memory.
    With `scan`, the result is the first non-finite index of `dst[0:n]`
    (-1 when none). lane gap-neural-overhead2 (2026-10-02): ONE thread; the
    copy-out over host tasks is gone (no CPU threads in GPU code, the
    no_host_routes hook). The pipeline still overlaps this copy with the
    copy engine filling the other stage. Copies only: no bit moves."""
    memcpy(dest=dst, src=src, count=n)
    if scan:
        return _first_nonfinite(dst, n)
    return -1
