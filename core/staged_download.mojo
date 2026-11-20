# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE DEVICE-TO-HOST DOWNLOAD of a float32 buffer into caller memory
(`download_f32_into`, `download_f32_into_scanned`, and
`download_f32_into_canon` for float32 results): a pipeline over chunks
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
from std.memory import bitcast, unsafe_memcpy
from std.sys import is_defined
from std.sys.defines import get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.step_phase import (
    step_count_d2h,
    step_count_host_alloc,
    step_count_sync,
)
from core.shared_slot import SharedSlot
from core.ctx_key import ctx_cache_key
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.nan_canon import NAN_CANON_ON, nan_canon_f32, nan_canon_span_f32


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
    the host drains the other into `dst` (`_copy_out`: spans across cores). The per-call 64 MiB pinned
    allocation of the first staged form (lane neural-pass43) is gone; it was
    what made the stage lose to the raw host-pointer copy on the L40S. A
    transport choice: the same bytes land in the same places.
    `pool` names the `_Global` slot of the stages: the default is the byte
    LM's; a binding that links this file for its own downloads passes its
    own name (one per binding and tier, as `core/neural_context.mojo`'s
    contexts), so no pinned stage is handed to another binding's context.
    """
    return _download_f32_into[pool](ctx, buf, n, dst, scan, False)


def download_f32_into_canon[pool: StaticString = _STAGE_NAME](
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`download_f32_into` for a buffer that holds float32 RESULTS (never
    int32 words moved through a float32 buffer, as x_prep's outputs are):
    with NAN_CANON_ON (default on) every NaN word
    of `dst[0:n]` is the one canonical quiet NaN (core/nan_canon.mojo), so a
    computed NaN is the same word on every vendor and on the host column.
    The pass is folded into the copy-out's spans (the words are read while
    the copy has them hot; a store only where a NaN is). Without the define
    this is `download_f32_into`, the same code."""
    _ = _download_f32_into[pool](ctx, buf, n, dst, False, NAN_CANON_ON)


def _download_f32_into[pool: StaticString](
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
    scan: Bool,
    canon: Bool,
) raises -> Int:
    """The body of `download_f32_into_scanned` and `download_f32_into_canon`.
    `canon` rewrites NaN words to the canonical one after the copy; the scan
    (an exponent test) reads the same verdict before or after it."""
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
        if canon:
            _ = nan_canon_f32(dst, n)
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
        var b0 = _copy_out(dst + k * chunk, st0.unsafe_ptr(), min(chunk, n - k * chunk), scan, canon)
        if b0 >= 0 and bad < 0:
            bad = k * chunk + b0
        if k + 1 >= nchunks:
            break
        step_count_sync()
        ctx.synchronize()
        if k + 2 < nchunks:
            _stage_enqueue(ctx, buf, n, k + 2, chunk, st0)
        var b1 = _copy_out(dst + (k + 1) * chunk, st1.unsafe_ptr(), min(chunk, n - (k + 1) * chunk), scan, canon)
        if b1 >= 0 and bad < 0:
            bad = (k + 1) * chunk + b1
        k += 2
    _stage_give[pool](ctx, st0^)
    _stage_give[pool](ctx, st1^)
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
#: floats per pipeline chunk (2M, 8 MiB); `-D MOJOLEARN_DOWNLOAD_CHUNK=<n>`
#: (floats, at least 64K) for the sweep. A size choice for any buffer, not a
#: board shape.
comptime DOWNLOAD_CHUNK_FLOATS = get_defined_int["MOJOLEARN_DOWNLOAD_CHUNK", 1 << 21]()
comptime DOWNLOAD_STAGE_KEEP = 2

# Download route (lane envgate-train-seq-mamba, 2026-11-15). Before, every
# download read MOJOLEARN_DOWNLOAD_STAGE ("0" raw, "1" force the stage) and
# MOJOLEARN_DOWNLOAD_CHUNK from the environment, per call. Now both are build
# constants holding today's default (staged from DOWNLOAD_STAGE_MIN floats on
# every column, lane neural-pass49 2026-10-01; the Apple column since lane
# neural-pass43). `-D MOJOLEARN_DOWNLOAD_STAGE` forces the stage below the
# minimum. A transport choice: no bit moves.
# TOMBSTONE: MOJOLEARN_DOWNLOAD_STAGE_OFF (LOSER: raw host-pointer copy at every size, arm/base geomean 1.016 over 7 neural cells, lm-forward fit 39.7 -> 44.9 ms (1.13), the other six 0.98-1.02, digests equal; NV L40S ab3 2026-11-18, job nvc-ab-20261118T182558Z-cb6e02) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
# Restore: git apply experiments/removed/MOJOLEARN_DOWNLOAD_STAGE_OFF.patch; record in docs/TOMBSTONES.md.
comptime DOWNLOAD_STAGE_FORCE = is_defined["MOJOLEARN_DOWNLOAD_STAGE"]()


@always_inline
def download_staged(n: Int) -> Bool:
    """Whether `download_f32_into` goes through the pinned stage pipeline:
    from DOWNLOAD_STAGE_MIN floats unless -D MOJOLEARN_DOWNLOAD_STAGE forces it."""
    comptime if DOWNLOAD_STAGE_FORCE:
        return True
    else:
        return n >= DOWNLOAD_STAGE_MIN


@always_inline
def download_chunk_floats() -> Int:
    """Floats per pipeline chunk: the build constant DOWNLOAD_CHUNK_FLOATS."""
    comptime assert DOWNLOAD_CHUNK_FLOATS >= (1 << 16), "MOJOLEARN_DOWNLOAD_CHUNK must be >= 65536"
    return DOWNLOAD_CHUNK_FLOATS


struct _StagePool(Defaultable, Movable):
    """The pinned stages kept between calls, all of `n` floats and all made
    on the context `key` names (core/ctx_key.mojo)."""
    var bufs: List[HostBuffer[DType.float32]]
    var n: Int
    #: fx-leftovers-product (2026-11-19): the context the kept stages were
    #: made on, as core/msel_device.mojo `_MselStages.key` holds it. A pool
    #: name is per module and tier, and a module can be linked into more than
    #: one binding (cholesky into the gaussian_process binding, ivf_flat_build
    #: into x_ann, the metrics device_io into x_cnn), so a name alone did not
    #: stop a pinned stage made by one context reaching a copy on another.
    var key: Int

    def __init__(out self):
        self.bufs = List[HostBuffer[DType.float32]]()
        self.n = 0
        self.key = 0


comptime _STAGE_NAME = "MojolearnDownloadStagesIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnDownloadStagesFastV1"
#: Layout version of `_StagePool` for every pool name (2: `key` added).
comptime _STAGE_LAYOUT = 2
comptime DOWNLOAD_STAGES = SharedSlot[_StagePool, _STAGE_NAME, _STAGE_LAYOUT]


def _stage_take[name: StaticString](ctx: DeviceContext, chunk: Int) raises -> HostBuffer[DType.float32]:
    """A pooled pinned stage of at least `chunk` floats, else a new one (a
    larger chunk retires the smaller stages, and so does a take on another
    context: a stage is never handed across contexts)."""
    comptime SLOT = SharedSlot[_StagePool, name, _STAGE_LAYOUT]
    var pool = SLOT.get_or_create_ptr()
    # One word read from the context's handle; no lookup (core/ctx_key.mojo).
    var key = ctx_cache_key(ctx)
    if pool[].key != key:
        pool[].bufs.clear()
        pool[].key = key
    if pool[].n < chunk:
        pool[].bufs.clear()
        pool[].n = chunk
    if len(pool[].bufs) > 0:
        return pool[].bufs.pop()
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.float32](pool[].n)


def _stage_give[name: StaticString](ctx: DeviceContext, var stage: HostBuffer[DType.float32]) raises:
    """Return a stage made on `ctx`: kept while the pool is still that
    context's and holds fewer than DOWNLOAD_STAGE_KEEP; dropped otherwise."""
    comptime SLOT = SharedSlot[_StagePool, name, _STAGE_LAYOUT]
    var pool = SLOT.get_or_create_ptr()
    if (pool[].key == ctx_cache_key(ctx) and len(stage) == pool[].n
            and len(pool[].bufs) < DOWNLOAD_STAGE_KEEP):
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


def staged_copy_f32(dst: MutPointer[Float32, MutUntrackedOrigin], src: MutPointer[Float32, MutUntrackedOrigin],
                    n: Int):
    """`memcpy(dst, src, n)` of float32 words between a pinned stage and the
    caller's host memory (either direction), split into contiguous spans
    across the host's cores above the grain floor. Spans are independent
    (each writes its own words), so the bytes do not depend on the split."""
    _ = _copy_out(dst, src, n, False, False)


def _copy_out(dst: MutPointer[Float32, MutUntrackedOrigin], src: MutPointer[Float32, MutUntrackedOrigin], n: Int, scan: Bool,
              canon: Bool) -> Int:
    """`memcpy(dst, src, n)` from a pinned stage into the caller's memory.
    With `scan`, the result is the first non-finite index of `dst[0:n]`
    (-1 when none). Copies only: no bit moves, unless `canon` (lane
    fx-nan-canon, 2026-11-19; only `download_f32_into_canon` passes it, and
    only under NAN_CANON_ON, the default): each span's NaN words
    then become the canonical one in `dst` right after its memcpy.
    fx-leftovers-product (2026-11-19): the copy (and its scan) is split
    into contiguous spans across the host's cores, the form of
    core/msel_device.mojo `_msel_copy_out`. It was ONE thread (lane
    gap-neural-overhead2, 2026-10-02, for a source hook that is retired),
    and that thread also took every first-touch page fault of a fresh
    output array: the per-megabyte cost of a large download that the
    cross-entropy row showed (256 MB of gradient down at 0.34 ms/MB with no
    scan in its door) and that the pinned link does not explain. Whole pages
    per span, so no two tasks first-touch one page of a page-aligned output.
    The pipeline still overlaps this copy with the copy engine filling the
    other stage. The first non-finite index is the lowest span's, so the
    verdict does not depend on the split."""
    # parallel: float32 copy (and non-finite scan) over contiguous spans across the host's cores
    if n <= 0:
        return -1
    var tasks = host_predict_task_count_work(n, n)
    if tasks <= 1:
        unsafe_memcpy(dest=dst, src=src, count=n)
        if canon:
            _ = nan_canon_span_f32(dst, n)
        if scan:
            return _first_nonfinite(dst, n)
        return -1
    # 1024 floats = one 4 KiB page
    var span = ((host_predict_chunk(n, tasks) + 1023) // 1024) * 1024
    var d = Int(dst)
    var s = Int(src)
    var found = List[Int](length=tasks, fill=-1)
    var found_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](found.unsafe_ptr())

    def _span(t: Int) {imm d, imm s, imm n, imm span, imm scan, imm canon, imm found_ptr}:
        var lo = t * span
        if lo >= n:
            return
        var hi = min(lo + span, n)
        var to = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=d + lo * 4)
        unsafe_memcpy(dest=to, src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=s + lo * 4),
                      count=hi - lo)
        if canon:
            _ = nan_canon_span_f32(to, hi - lo)
        if scan:
            var at = _first_nonfinite(to, hi - lo)
            if at >= 0:
                found_ptr.unsafe_store(t, lo + at)

    host_parallelize(_span, tasks)
    var bad = -1
    for t in range(tasks):  # driver-loop: over the copy tasks' results (at most one per core), lowest span first
        if bad < 0 and found[t] >= 0:
            bad = found[t]
    _ = found^
    return bad
