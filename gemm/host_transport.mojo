# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Transport for the host-pointer GEMM surfaces (lane/gap-neural-models, 2026-10-02).

NO ARITHMETIC. Copies, pooled allocations and waits only, so nothing here can
move a bit of any GEMM profile.

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

A/B (runtime, one build serves both arms):
  MOJOLEARN_GEMM_POOL=0        fresh buffers every call (the old path)
  MOJOLEARN_GEMM_STAGE_DOWN=0|1   staged download off/on (default on)
  MOJOLEARN_GEMM_STAGE_UP=0|1     staged upload off/on (default: on except Apple)
"""

from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.memory import memcpy
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN

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
    var stage: Optional[HostBuffer[DType.float32]]

    def __init__(out self):
        self.busy = 0
        self.f32 = List[Optional[DeviceBuffer[DType.float32]]]()
        self.u16 = List[Optional[DeviceBuffer[DType.uint16]]]()
        for _ in range(N_ROLES):
            self.f32.append(Optional[DeviceBuffer[DType.float32]]())
            self.u16.append(Optional[DeviceBuffer[DType.uint16]]())
        self.stage = Optional[HostBuffer[DType.float32]]()


comptime GEMM_POOL = _Global[StorageType=_GemmPool, name="MojoGemmHostPool", init_fn=_GemmPool.__init__]


def _env_flag(name: StringLiteral, default: Bool) -> Bool:
    var v = String(getenv(name))
    if v == "0":
        return False
    if v == "1":
        return True
    return default


def gemm_pool_on() -> Bool:
    return _env_flag("MOJOLEARN_GEMM_POOL", True)


def gemm_stage_down_on() -> Bool:
    return _env_flag("MOJOLEARN_GEMM_STAGE_DOWN", True)


def gemm_stage_up_on() -> Bool:
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        return _env_flag("MOJOLEARN_GEMM_STAGE_UP", False)
    return _env_flag("MOJOLEARN_GEMM_STAGE_UP", True)


struct GemmHostLease(Movable):
    """The pool for one call, or nothing when the pool is off or busy.
    `release()` before the lease ends (after the call's final wait)."""

    var held: Bool

    def __init__(out self) raises:
        self.held = False
        if not gemm_pool_on():
            return
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
    memcpy(dest=dst, src=src, count=n)


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


def gemm_down_f32(ctx: DeviceContext, lease: GemmHostLease, src: DeviceBuffer[DType.float32], dst: F32P, n: Int) raises:
    """`src[0, n)` into host floats `dst[0, n)`, then waits."""
    if n <= 0:
        ctx.synchronize()
        return
    var addr = lease.stage(ctx) if (n >= GEMM_STAGE_MIN and gemm_stage_down_on()) else 0
    if addr == 0:
        ctx.enqueue_copy(dst_ptr=dst, src_buf=src.create_sub_buffer[DType.float32](0, n))
        ctx.synchronize()
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
        ctx.synchronize()
        prev_off = off
        prev_cnt = cnt
        off += cnt
        i += 1
    _par_copy(dst + prev_off, stage + ((i - 1) % 2) * GEMM_STAGE_FLOATS, prev_cnt)
