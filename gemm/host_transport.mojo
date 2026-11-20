# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Transport for the host-pointer GEMM surfaces (lane/gap-neural-models, 2026-10-02).

NO ARITHMETIC. Copies, pooled allocations and waits only, so nothing here can
move a bit of any GEMM profile. One exception, on by default: with
NAN_CANON_ON (the default; core/nan_canon.mojo) `gemm_down_f32` rewrites the NaN words of
the downloaded product to the one canonical word (core/nan_canon.mojo).

WHY. `linalg.matmul` at the board shape (4096^3, 64 MB per operand) paid, per
call: three fresh device allocations plus the dispatcher's fresh workspace,
and on Apple a device-to-host-pointer download that runs at ~3 GB/s (~21 ms
per 64 MB, memory note metal-transfer-costs-on-apple). This module gives the
host surfaces:

  * POOLED DEVICE BUFFERS: grow-only process slots, one per role, reused by
    every call. A busy flag guards them; a concurrent second caller (two
    Python threads, the GIL is released) takes fresh buffers instead.
  * A STAGED DOWNLOAD: DMA into a pinned host stage, then one memcpy out,
    double-buffered so the next chunk's DMA overlaps this chunk's copy. A
    pinned buffer DMAs at full speed on every vendor.
  * A STAGED UPLOAD (same stage, the other direction), off by default on
    Apple where a raw host-pointer upload is already 1.6-2.4 ms per 64 MB.

A/B (build arms; run-time environment reads per matmul until 2026-11-15):
  -D MOJOLEARN_GEMM_STAGE_UP_OFF      raw upload (default staged except Apple)
  -D MOJOLEARN_GEMM_STAGE_UP          staged upload on Apple too
"""

from std.atomic import Atomic, Ordering
from std.memory import unsafe_memcpy
from std.sys import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from core.shared_slot import SharedSlot
from core.nan_canon import NAN_CANON_ON, nan_canon_f32

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]

#: Floats per stage half (16 MB); two halves double-buffer.
comptime GEMM_STAGE_FLOATS = 1 << 22
#: Transfers smaller than this go raw (a wait per chunk costs more than it saves).
comptime GEMM_STAGE_MIN = 1 << 18

#: Pool roles.
comptime ROLE_A = 0
comptime ROLE_B = 1
comptime ROLE_C = 2
comptime ROLE_WS = 3
comptime ROLE_X = 4
comptime N_ROLES = 5


struct _GemmPool(Defaultable, Movable):
    var busy: Int64
    var f32: List[Optional[DeviceBuffer[DType.float32]]]
    var u16: List[Optional[DeviceBuffer[DType.uint16]]]
    var i8: List[Optional[DeviceBuffer[DType.int8]]]
    var i32: List[Optional[DeviceBuffer[DType.int32]]]
    var stage: Optional[HostBuffer[DType.float32]]

    def __init__(out self):
        self.busy = 0
        self.f32 = List[Optional[DeviceBuffer[DType.float32]]]()
        self.u16 = List[Optional[DeviceBuffer[DType.uint16]]]()
        self.i8 = List[Optional[DeviceBuffer[DType.int8]]]()
        self.i32 = List[Optional[DeviceBuffer[DType.int32]]]()
        for _ in range(N_ROLES):
            self.f32.append(Optional[DeviceBuffer[DType.float32]]())
            self.u16.append(Optional[DeviceBuffer[DType.uint16]]())
            self.i8.append(Optional[DeviceBuffer[DType.int8]]())
            self.i32.append(Optional[DeviceBuffer[DType.int32]]())
        self.stage = Optional[HostBuffer[DType.float32]]()


comptime GEMM_POOL = SharedSlot[_GemmPool, "MojolearnGemmHostPoolV1"]


# Transport switches (lane envgate-train-seq-mamba, 2026-11-15): build
# constants holding today's defaults; before, each was an environment read per
# matmul (`_env_flag`, now deleted).
# A/B ab3 (NVIDIA L40S, wheel d652a4b0a1, 2026-11-18, job
# nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests and quality equal;
# cells gemm / gemm-bf16 / gemm-int8 fit): the pool and the staged download
# stay, with no switch.
#   fresh buffers per call:  1.044 geomean (0.94, 1.09, 1.11)
#   raw download:            1.125 geomean (1.03, 1.18, 1.17)
#   raw upload (the B arm below): 0.986 geomean (0.87, 1.07, 1.03), three cells,
#   one of them faster: UNDECIDED, kept opt-in, re-time owed (owed-pre0867.json
#   "gemm-stage-up-off").
# TOMBSTONE: MOJOLEARN_GEMM_POOL_OFF (LOSER: fresh device buffers per matmul, arm/base geomean 1.044 over 3 cells, NV L40S ab3 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_POOL_OFF-STAGE_DOWN_OFF.patch (restores both gemm switches; re-delete the other); record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_GEMM_STAGE_DOWN_OFF (LOSER: raw device-to-host-pointer download, arm/base geomean 1.125 over 3 cells, NV L40S ab3 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_POOL_OFF-STAGE_DOWN_OFF.patch (restores both gemm switches; re-delete the other); record in docs/TOMBSTONES.md.
comptime GEMM_STAGE_UP_ON = (
    TARGET_COLUMN != COLUMN_APPLE or is_defined["MOJOLEARN_GEMM_STAGE_UP"]()
) and not is_defined["MOJOLEARN_GEMM_STAGE_UP_OFF"]()


@always_inline
def gemm_stage_up_on() -> Bool:
    return GEMM_STAGE_UP_ON


struct GemmHostLease(Movable):
    """The pool for one call, or nothing when the pool is busy.
    `release()` before the lease ends (after the call's final wait)."""

    var held: Bool

    def __init__(out self) raises:
        self.held = False
        var p = GEMM_POOL.get_or_create_ptr()
        var flag = UnsafePointer(to=p[].busy)
        var old = Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](flag, Int64(1))
        if old == 0:
            self.held = True
        else:
            _ = Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](flag, Int64(-1))

    def release(mut self) raises:
        if self.held:
            var p = GEMM_POOL.get_or_create_ptr()
            var flag = UnsafePointer(to=p[].busy)
            _ = Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](flag, Int64(-1))
            self.held = False

    def f32(self, ctx: DeviceContext, role: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
        """A device buffer of at least `n` floats (a view of exactly `n`)."""
        var want = n if n > 0 else 1
        if not self.held:
            return ctx.enqueue_create_buffer[DType.float32](want)
        var p = GEMM_POOL.get_or_create_ptr()
        if not p[].f32[role] or len(p[].f32[role].value()) < want:
            # Grow: the previous allocation may still be read by queued work.
            ctx.synchronize()
            p[].f32[role] = ctx.enqueue_create_buffer[DType.float32](want)
        return p[].f32[role].value().create_sub_buffer[DType.float32](0, want)

    def u16(self, ctx: DeviceContext, role: Int, n: Int) raises -> DeviceBuffer[DType.uint16]:
        var want = n if n > 0 else 1
        if not self.held:
            return ctx.enqueue_create_buffer[DType.uint16](want)
        var p = GEMM_POOL.get_or_create_ptr()
        if not p[].u16[role] or len(p[].u16[role].value()) < want:
            ctx.synchronize()
            p[].u16[role] = ctx.enqueue_create_buffer[DType.uint16](want)
        return p[].u16[role].value().create_sub_buffer[DType.uint16](0, want)

    def i8(self, ctx: DeviceContext, role: Int, n: Int) raises -> DeviceBuffer[DType.int8]:
        """int8 codes (gemm_int8's operands; lane a3, 2026-11-08: they were
        fresh allocations per call while the f32/bf16 surfaces pooled theirs)."""
        var want = n if n > 0 else 1
        if not self.held:
            return ctx.enqueue_create_buffer[DType.int8](want)
        var p = GEMM_POOL.get_or_create_ptr()
        if not p[].i8[role] or len(p[].i8[role].value()) < want:
            ctx.synchronize()
            p[].i8[role] = ctx.enqueue_create_buffer[DType.int8](want)
        return p[].i8[role].value().create_sub_buffer[DType.int8](0, want)

    def i32(self, ctx: DeviceContext, role: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
        """int32 words (gemm_int8's per-row exponents)."""
        var want = n if n > 0 else 1
        if not self.held:
            return ctx.enqueue_create_buffer[DType.int32](want)
        var p = GEMM_POOL.get_or_create_ptr()
        if not p[].i32[role] or len(p[].i32[role].value()) < want:
            ctx.synchronize()
            p[].i32[role] = ctx.enqueue_create_buffer[DType.int32](want)
        return p[].i32[role].value().create_sub_buffer[DType.int32](0, want)

    def stage(self, ctx: DeviceContext) raises -> Int:
        """The address of two halves of GEMM_STAGE_FLOATS pinned floats, or 0
        when the pool is not held."""
        if not self.held:
            return 0
        var p = GEMM_POOL.get_or_create_ptr()
        if not p[].stage:
            p[].stage = ctx.enqueue_create_host_buffer[DType.float32](2 * GEMM_STAGE_FLOATS)
            ctx.synchronize()
        return Int(p[].stage.value().unsafe_ptr())


def _par_copy(dst: F32P, src: F32P, n: Int):
    """One `memcpy(dst, src, n)`: the stage's single host copy (transport, no
    arithmetic, no host threads)."""
    unsafe_memcpy(dest=dst, src=src, count=n)


def gemm_up_f32(ctx: DeviceContext, lease: GemmHostLease, mut dst: DeviceBuffer[DType.float32], src: F32P, n: Int) raises:
    """Host floats `src[0, n)` into `dst[0, n)`."""
    if n <= 0:
        return
    var addr = lease.stage(ctx) if (n >= GEMM_STAGE_MIN and gemm_stage_up_on()) else 0
    if addr == 0:
        ctx.enqueue_copy(dst_buf=dst.create_sub_buffer[DType.float32](0, n), src_ptr=src)
        return
    var stage = F32P(unsafe_from_address=addr)
    # Double-buffered: chunk i is written into its half while chunk i-1's
    # DMA runs. The wait before DMA i retires DMA i-1, so when chunk i+1
    # refills that half nothing reads it any more.
    var off = 0
    var i = 0
    while off < n:
        var cnt = min(GEMM_STAGE_FLOATS, n - off)
        var half = stage + (i % 2) * GEMM_STAGE_FLOATS
        _par_copy(half, src + off, cnt)
        if i > 0:
            ctx.synchronize()
        ctx.enqueue_copy(dst_buf=dst.create_sub_buffer[DType.float32](off, cnt), src_ptr=half)
        off += cnt
        i += 1
    # The next host write into the stage (the next upload) needs these DMAs
    # retired; the kernels the caller enqueues next are ordered after them.
    ctx.synchronize()


def gemm_up_u16(ctx: DeviceContext, lease: GemmHostLease, mut dst: DeviceBuffer[DType.uint16], src: MutPointer[UInt16, MutUntrackedOrigin], n: Int) raises:
    """Raw upload of bf16 bits (half the bytes of a float operand)."""
    if n > 0:
        ctx.enqueue_copy(dst_buf=dst.create_sub_buffer[DType.uint16](0, n), src_ptr=src)


@always_inline
def _down_canon(dst: F32P, n: Int):
    """The NaN words of a downloaded GEMM result `dst[0, n)` made the one
    canonical word (core/nan_canon.mojo), across the host's cores. Compiles
    to nothing under -D MOJOLEARN_IDENTITY_FAST_NAN_CANON_OFF."""
    comptime if NAN_CANON_ON:
        _ = nan_canon_f32(dst, n)


def gemm_down_f32(ctx: DeviceContext, lease: GemmHostLease, src: DeviceBuffer[DType.float32], dst: F32P, n: Int) raises:
    """`src[0, n)` into host floats `dst[0, n)`, then waits.
    Every caller hands a float32 product C (fp32, fused, bf16, int8, int15),
    so this is where a GEMM result leaves the device: with
    NAN_CANON_ON (default on) each chunk's NaN
    words become the one canonical word right after its host copy, while the
    copy engine fills the other half (`_down_canon`). No other word moves."""
    if n <= 0:
        ctx.synchronize()
        return
    var addr = lease.stage(ctx) if n >= GEMM_STAGE_MIN else 0
    if addr == 0:
        ctx.enqueue_copy(dst_ptr=dst, src_buf=src.create_sub_buffer[DType.float32](0, n))
        ctx.synchronize()
        _down_canon(dst, n)
        return
    var stage = F32P(unsafe_from_address=addr)
    # Double-buffered: chunk i's DMA runs while chunk i-1 (retired by the
    # previous wait) is copied out of the other half.
    var off = 0
    var i = 0
    var prev_off = 0
    var prev_cnt = 0
    while off < n:
        var cnt = min(GEMM_STAGE_FLOATS, n - off)
        var half = stage + (i % 2) * GEMM_STAGE_FLOATS
        ctx.enqueue_copy(dst_ptr=half, src_buf=src.create_sub_buffer[DType.float32](off, cnt))
        if i > 0:
            _par_copy(dst + prev_off, stage + ((i - 1) % 2) * GEMM_STAGE_FLOATS, prev_cnt)
            _down_canon(dst + prev_off, prev_cnt)
        ctx.synchronize()
        prev_off = off
        prev_cnt = cnt
        off += cnt
        i += 1
    _par_copy(dst + prev_off, stage + ((i - 1) % 2) * GEMM_STAGE_FLOATS, prev_cnt)
    _down_canon(dst + prev_off, prev_cnt)
