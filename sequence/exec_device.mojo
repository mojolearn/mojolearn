# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`DeviceExec`: the lane's operations on the GPU, one thread per element,
over `sequence/ops.mojo::apply`, the body `HostExec` loops over on the CPU."""
from std.ffi import _Global
from std.memory import bitcast
from std.gpu import block_dim, block_idx, thread_idx
from std.os import getenv
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from sequence.moe_tiled import MOE_TPB, moe_combine_kernel, moe_hidden_tiled_kernel, moe_out_tiled_kernel
from sequence.moe_reg import (
    MOE_DEVGROUP,
    MOE_REGTILE,
    RT as MOE_RT,
    moe_group_count_kernel,
    moe_group_offsets_kernel,
    moe_group_scatter_kernel,
    moe_group_zero_kernel,
    moe_hidden_reg_kernel,
    moe_logits_reg_kernel,
    moe_out_reg_kernel,
    moe_reg_blocks,
    moe_route_tail_kernel,
)

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from sequence.exec_trait import Exec
from sequence.dispatch import apply
from sequence.ops import OP_MOE_ROUTE, OP_MOE_OUT, OP_MOE_HIDDEN, FP, Args, OP_AF_ALPHA, OP_AF_BLK_SUMSQ, OP_AF_DENOM, OP_GEMM, OP_LAMB_RATIO, OP_SEG_SUMSQ
from sequence.coop import COOP_W, apply_coop
from sequence.ops import OP_CHOLSOLVE, OP_VAR_FORECAST, TSA2_VAR
from sequence.vecar_block import VAR_SMEM, VAR_TPB, var_chol_block_kernel, var_forecast_block_kernel
from sequence.fit_team import SeqTeam, garch_team, prophet_fit_team
from sequence.ets_team import ETS_TEAM, ets_team
from sequence.prophet_coop import PROPHET_COOP, prophet_fit_coop
from sequence.ops import OP_ETS, OP_GARCH
from x_linear.ops import IP
from x_linear.witness import witness_end
from std.sys.info import has_apple_gpu_accelerator

#: the simdgroup-cooperative long folds (sequence/coop.mojo): Apple only
comptime SEQ_COOP = has_apple_gpu_accelerator()

comptime TPB = 128


struct _SeqContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every `DeviceExec`. A context
    per binding call exhausts Metal's per-process command queues within one
    fit (memory: METAL QUEUE LIMIT IS PER-PROCESS; the cnn lane's M2 Pro
    finding). Storage is `std.ffi._Global`, one slot per numeric tier so a
    FAST and an IDENTICAL .so in one process never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXSequenceContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXSequenceContextFast"
comptime X_SEQUENCE_CONTEXT = _Global[StorageType=_SeqContext, name=_CTX_NAME, init_fn=_SeqContext.__init__]


def sequence_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_SEQUENCE_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


@always_inline
def _pack_ii(lo: Int, hi: Int) -> Int64:
    """Two Int32 values in one Int64 word (lo in the low half)."""
    return Int64(Int(UInt32(Int32(lo))) | (Int(UInt32(Int32(hi))) << 32))


@always_inline
def _lo(w: Int64) -> Int:
    return Int(Int32(w & 0xFFFFFFFF))


@always_inline
def _hi(w: Int64) -> Int:
    return Int(Int32((w >> 32) & 0xFFFFFFFF))


@always_inline
def _pack_ff(lo: Float32, hi: Float32) -> Int64:
    """Two floats' bit patterns in one Int64 word (exact)."""
    return Int64(Int(bitcast[DType.uint32](lo)) | (Int(bitcast[DType.uint32](hi)) << 32))


@always_inline
def _flo(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32(w & 0xFFFFFFFF))


@always_inline
def _fhi(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32((w >> 32) & 0xFFFFFFFF))



def _moe_tiled_on() -> Bool:
    return String(getenv("MOJOLEARN_SEQ_MOE_TILED")) != "0"


def _var_block_on() -> Bool:
    """VAR's Cholesky solve and forecast on one threadgroup
    (sequence/vecar_block.mojo, lane gap-prep2); MOJOLEARN_SEQ_VAR_BLOCK=0
    keeps the one-thread ops (the A/B arm)."""
    return String(getenv("MOJOLEARN_SEQ_VAR_BLOCK")) != "0"


def seq_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64,
):
    """One thread per element. The twelve integers and eight floats travel
    packed two to an Int64 word (bit-exact): Metal binds every kernel
    argument to its own buffer slot and has 31, so the unpacked 33-argument
    signature failed to compile on Apple."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                     _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                     _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                     _flo(f01), _fhi(f01), _flo(f23), _fhi(f23),
                     _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))
        apply[OP](t, a)


def team_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64, wf: IP, woff: Int32, nonce: Int32,
):
    """One block per series of the group (the arguments packed as
    seq_kernel's). The early exit is block-uniform, and every thread then
    reaches the completion witness (x_linear/witness.mojo: on Apple each
    block's word reports the slice ran to its end; elsewhere nothing)."""
    var blk = Int(block_idx.x)
    if blk < Int(n):
        var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                     _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                     _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                     _flo(f01), _fhi(f01), _flo(f23), _fhi(f23),
                     _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))
        var team = SeqTeam(Int(thread_idx.x), Int(block_dim.x))
        comptime if OP == OP_GARCH:
            garch_team(blk, team, a)
        elif ETS_TEAM and OP == OP_ETS:
            # Apple FAST default (off: -D MOJOLEARN_ETS_TEAM_OFF; sequence/ets_team.mojo)
            ets_team(blk, team, a)
        elif PROPHET_COOP:
            # Apple FAST default (off: -D MOJOLEARN_PROPHET_COOP_OFF; sequence/prophet_coop.mojo)
            prophet_fit_coop(blk, team, a)
        else:
            prophet_fit_team(blk, team, a)
    witness_end(wf, woff, nonce)



def _pcopy(dst: FP, src: FP, n: Int):
    """A host transport copy: one memcpy on the calling thread (no host task
    pool on a GPU install; cpu-gpu-cleanup n-seq, 2026-10-02). Data movement
    only, the bytes are the same."""
    if n > 0:
        memcpy(dest=dst, src=src, count=n)


def coop_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64,
):
    """One simdgroup (COOP_W threads) per cell of OP (sequence/coop.mojo);
    TPB is a multiple of COOP_W, so a cell's lanes share one simdgroup and
    leave together."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cell = g // COOP_W
    if cell < Int(n):
        var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                     _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                     _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                     _flo(f01), _fhi(f01), _flo(f23), _fhi(f23),
                     _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))
        apply_coop[OP](cell, g - cell * COOP_W, a)



# ---- the buffer pool (lane neural-pass40, 2026-10-01) ----------------------------------------
#: Every binding call built a fresh DeviceExec that created its device
#: buffers and one PINNED host buffer per upload and per download, all
#: freed at the end of the call: at the board's layer cells (a 64 MB x in,
#: a 64 MB y out) that is three device allocations and two pinned
#: allocations of 64 MB per call, and on CUDA a pinned allocation of that
#: size costs tens of milliseconds (the layernorm cell: 99.7 ms on an L40S
#: against a ~1 ms kernel). The pool keeps the process's device buffers and
#: pinned stages across calls, handed out by exact count (device) or
#: capacity (pinned) and returned when the executor syncs (stages) or ends
#: (device buffers); the words are copied exactly as before, so no bit
#: moves. MOJOLEARN_SEQ_POOL=0 restores the per-call allocations;
#: SEQ_POOL_BYTES bounds what is kept (the oldest free entries go first).
comptime SEQ_POOL_BYTES = 3 << 30


struct _SeqPool(Defaultable, Movable):
    var dev: List[DeviceBuffer[DType.float32]]
    var dev_n: List[Int]
    var dev_free: List[Bool]
    var host: List[HostBuffer[DType.float32]]
    var host_n: List[Int]
    var host_free: List[Bool]
    var bytes: Int

    def __init__(out self):
        self.dev = List[DeviceBuffer[DType.float32]]()
        self.dev_n = List[Int]()
        self.dev_free = List[Bool]()
        self.host = List[HostBuffer[DType.float32]]()
        self.host_n = List[Int]()
        self.host_free = List[Bool]()
        self.bytes = 0


comptime _POOL_NAME = "MojoXSequencePoolIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXSequencePoolFast"
comptime X_SEQUENCE_POOL = _Global[StorageType=_SeqPool, name=_POOL_NAME, init_fn=_SeqPool.__init__]


def seq_pool_on() -> Bool:
    return String(getenv("MOJOLEARN_SEQ_POOL")) != "0"


def _pool_trim(pool: UnsafePointer[_SeqPool, MutUntrackedOrigin], need: Int):
    """Drop free entries, oldest first, until `need` more bytes fit."""
    while pool[].bytes + need > SEQ_POOL_BYTES:
        var dropped = False
        for i in range(len(pool[].dev)):
            if pool[].dev_free[i]:
                pool[].bytes -= pool[].dev_n[i] * 4
                _ = pool[].dev.pop(i)
                _ = pool[].dev_n.pop(i)
                _ = pool[].dev_free.pop(i)
                dropped = True
                break
        if not dropped:
            for i in range(len(pool[].host)):
                if pool[].host_free[i]:
                    pool[].bytes -= pool[].host_n[i] * 4
                    _ = pool[].host.pop(i)
                    _ = pool[].host_n.pop(i)
                    _ = pool[].host_free.pop(i)
                    dropped = True
                    break
        if not dropped:
            return


def _pool_dev(ctx: DeviceContext, count: Int) raises -> Int:
    """A free device buffer of exactly `count` floats from the pool, or a
    new one added to it; returns its index (held until released)."""
    var pool = X_SEQUENCE_POOL.get_or_create_ptr()
    for i in range(len(pool[].dev)):
        if pool[].dev_free[i] and pool[].dev_n[i] == count:
            pool[].dev_free[i] = False
            return i
    _pool_trim(pool, count * 4)
    pool[].dev.append(ctx.enqueue_create_buffer[DType.float32](count))
    pool[].dev_n.append(count)
    pool[].dev_free.append(False)
    pool[].bytes += count * 4
    return len(pool[].dev) - 1


def _pool_host(ctx: DeviceContext, n: Int) raises -> Int:
    """A free pinned stage of capacity >= n (the smallest that fits), or a
    new one of n floats; returns its index."""
    var pool = X_SEQUENCE_POOL.get_or_create_ptr()
    var best = -1
    for i in range(len(pool[].host)):
        if pool[].host_free[i] and pool[].host_n[i] >= n:
            if best < 0 or pool[].host_n[i] < pool[].host_n[best]:
                best = i
    if best >= 0:
        pool[].host_free[best] = False
        return best
    _pool_trim(pool, n * 4)
    pool[].host.append(ctx.enqueue_create_host_buffer[DType.float32](n))
    pool[].host_n.append(n)
    pool[].host_free.append(False)
    pool[].bytes += n * 4
    return len(pool[].host) - 1


def _pool_release_dev(i: Int) raises:
    var pool = X_SEQUENCE_POOL.get_or_create_ptr()
    pool[].dev_free[i] = True


def _pool_release_host(i: Int) raises:
    var pool = X_SEQUENCE_POOL.get_or_create_ptr()
    pool[].host_free[i] = True

struct DeviceExec(Exec):
    var ctx: DeviceContext
    var bufs: List[DeviceBuffer[DType.float32]]
    var base: List[Int]
    var size: List[Int]
    #: upload staging buffers still read by queued copies; released at the
    #: next sync (an upload no longer waits for its own copy)
    var staged: List[HostBuffer[DType.float32]]
    #: download_async copies waiting for the next sync: (host buffer, dst, n)
    var pend_host: List[HostBuffer[DType.float32]]
    var pend_dst: List[Int]
    var pend_n: List[Int]
    #: the pool (lane neural-pass40): `pooled` says which route this
    #: executor took; `pdev` the pool indices of its device buffers;
    #: `pstaged` / `ppend_host` the pool indices of the stages in flight
    var pooled: Bool
    var pdev: List[Int]
    var pstaged: List[Int]
    var ppend_host: List[Int]

    def __init__(out self) raises:
        self.ctx = sequence_ctx()
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.base = List[Int]()
        self.size = List[Int]()
        self.staged = List[HostBuffer[DType.float32]]()
        self.pend_host = List[HostBuffer[DType.float32]]()
        self.pend_dst = List[Int]()
        self.pend_n = List[Int]()
        self.pooled = seq_pool_on()
        self.pdev = List[Int]()
        self.pstaged = List[Int]()
        self.ppend_host = List[Int]()

    def alloc(mut self, n: Int) raises -> FP:
        return self._alloc(n, True)

    def _alloc(mut self, n: Int, zero: Bool) raises -> FP:
        var count = n if n > 0 else 1
        if self.pooled:
            var i = _pool_dev(self.ctx, count)
            var pool = X_SEQUENCE_POOL.get_or_create_ptr()
            if zero:
                pool[].dev[i].enqueue_fill(Float32(0.0))
            var pp = FP(unsafe_from_address=Int(pool[].dev[i].unsafe_ptr()))
            self.base.append(Int(pp))
            self.size.append(count)
            self.pdev.append(i)
            return pp
        var buf = self.ctx.enqueue_create_buffer[DType.float32](count)
        if zero:
            buf.enqueue_fill(Float32(0.0))
        var p = FP(unsafe_from_address=Int(buf.unsafe_ptr()))
        self.base.append(Int(p))
        self.size.append(count)
        self.bufs.append(buf^)
        return p

    def _sub(self, slot: Int, off: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
        """The view [off, off + n) of this executor's slot (pooled or owned)."""
        if self.pooled:
            var pool = X_SEQUENCE_POOL.get_or_create_ptr()
            return pool[].dev[self.pdev[slot]].create_sub_buffer[DType.float32](off, n)
        return self.bufs[slot].create_sub_buffer[DType.float32](off, n)

    def __deinit__(deinit self):
        # The context outlives this Exec: drain its queue before the buffers
        # go, so no queued kernel reads a freed buffer.
        try:
            self.ctx.synchronize()
        except:
            pass
        try:
            for i in range(len(self.pstaged)):
                _pool_release_host(self.pstaged[i])
            for i in range(len(self.ppend_host)):
                _pool_release_host(self.ppend_host[i])
            for i in range(len(self.pdev)):
                _pool_release_dev(self.pdev[i])
        except:
            pass

    def _find(self, p: FP, n: Int) raises -> Tuple[Int, Int]:
        var addr = Int(p)
        for i in range(len(self.base)):
            var off = (addr - self.base[i]) // 4
            if addr >= self.base[i] and off + n <= self.size[i]:
                return (i, off)
        raise Error("sequence: a copy names no buffer this device Exec allocated")

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        """The caller's floats are copied (memcpy, bit for bit) into a staging
        buffer at once, so `src` may change as soon as this returns; the
        device copy is queued behind the work already queued, and the staging
        buffer lives until the next sync. No wait here (Apple speed,
        2026-09-28: every upload was a synchronize, about 4 ms on Metal, and
        an element-at-a-time host loop)."""
        if n <= 0:
            return
        var found = self._find(dst, n)
        var view = self._sub(found[0], found[1], n)
        if self.pooled:
            var hi = _pool_host(self.ctx, n)
            var pool = X_SEQUENCE_POOL.get_or_create_ptr()
            var hp = pool[].host[hi].unsafe_ptr()
            _pcopy(hp, src, n)
            self.ctx.enqueue_copy(dst_buf=view, src_ptr=hp)
            self.pstaged.append(hi)
            _ = view^
            return
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        _pcopy(host.unsafe_ptr(), src, n)
        self.ctx.enqueue_copy(dst_buf=view, src_ptr=host.unsafe_ptr())
        _ = view^
        self.staged.append(host^)

    def bind(mut self, src: FP, n: Int) raises -> FP:
        # every word is uploaded over at once: no zero fill first (apple2)
        var p = self._alloc(n, n <= 0)
        self.upload(p, src, n)
        return p

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        if n <= 0:
            return
        var found = self._find(src, n)
        var view = self._sub(found[0], found[1], n)
        if self.pooled:
            var hi = _pool_host(self.ctx, n)
            var pool = X_SEQUENCE_POOL.get_or_create_ptr()
            var hp = pool[].host[hi].unsafe_ptr()
            self.ctx.enqueue_copy(dst_ptr=hp, src_buf=view)
            self.sync()
            _pcopy(dst, hp, n)
            _pool_release_host(hi)
            _ = view^
            return
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        self.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        self.sync()
        _pcopy(dst, host.unsafe_ptr(), n)
        _ = view^
        _ = host^

    def download_async(mut self, dst: FP, src: FP, n: Int) raises:
        """The copy is queued now; `dst` is written at the next sync, so
        several downloads share one wait (apple2)."""
        if n <= 0:
            return
        var found = self._find(src, n)
        if self.pooled:
            var hi = _pool_host(self.ctx, n)
            var pool = X_SEQUENCE_POOL.get_or_create_ptr()
            var hp = pool[].host[hi].unsafe_ptr()
            var pview = self._sub(found[0], found[1], n)
            self.ctx.enqueue_copy(dst_ptr=hp, src_buf=pview)
            _ = pview^
            self.ppend_host.append(hi)
            self.pend_dst.append(Int(dst))
            self.pend_n.append(n)
            return
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        var view = self._sub(found[0], found[1], n)
        self.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        _ = view^
        self.pend_host.append(host^)
        self.pend_dst.append(Int(dst))
        self.pend_n.append(n)

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        if n <= 0:
            return
        comptime I32_MAX = 2147483647
        for v in [a.i0, a.i1, a.i2, a.i3, a.i4, a.i5, a.i6, a.i7, a.i8, a.i9, a.i10, a.i11]:
            if v > I32_MAX or v < -I32_MAX - 1:
                raise Error("sequence DeviceExec: an integer argument does not fit Int32 (" + String(v) + ")")
        # The MoE products as tiled kernels with the items' chains (lane
        # neural-pass29, sequence/moe_tiled.mojo) when the entry grouped the
        # pairs by expert (a.i4 = the block count); MOJOLEARN_SEQ_MOE_TILED=0
        # keeps the one-thread-per-cell items.
        # Apple FAST (lane apple-fast-moespeed, sequence/moe_reg.mojo):
        # MOJOLEARN_MOE_REGTILE, the router's logits tiled and the products
        # register-tiled, the same chains; MOJOLEARN_MOE_DEVGROUP, the pairs
        # grouped by expert on the device (a.i6 = 1 from the entry).
        comptime if MOE_REGTILE and OP == OP_MOE_ROUTE:
            if a.i1 <= MOE_RT:
                var bt = MOE_RT // a.i1
                self.ctx.enqueue_function[moe_logits_reg_kernel](
                    a.p0, a.p1, a.p2, Int32(n), Int32(a.i0), Int32(a.i1),
                    grid_dim=((n + bt - 1) // bt, 1, 1), block_dim=(MOE_RT, 1, 1),
                )
                self.ctx.enqueue_function[moe_route_tail_kernel](
                    a.p2, a.p3, a.p4, a.p5, Int32(n), Int32(a.i1), Int32(a.i2), Int32(a.i3),
                    grid_dim=((n + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
                )
                return
        comptime if MOE_REGTILE and OP == OP_MOE_HIDDEN:
            if a.i4 > 0:
                if n % a.i1 != 0:
                    raise Error("moe_reg hidden: the cell count is not pairs x F")
                var npairs = n // a.i1
                comptime if MOE_DEVGROUP:
                    if a.i6 == 1:
                        # a.p7 the int32 counts | cursors (2E words)
                        self.ctx.enqueue_function[moe_group_zero_kernel](
                            a.p7, Int32(a.i3), grid_dim=(1, 1, 1), block_dim=(MOE_RT, 1, 1),
                        )
                        self.ctx.enqueue_function[moe_group_count_kernel](
                            a.p2, a.p7, Int32(npairs),
                            grid_dim=((npairs + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
                        )
                        self.ctx.enqueue_function[moe_group_offsets_kernel](
                            a.p7, a.p5, Int32(a.i3), grid_dim=(1, 1, 1), block_dim=(MOE_RT, 1, 1),
                        )
                        self.ctx.enqueue_function[moe_group_scatter_kernel](
                            a.p2, a.p7, a.p5, a.p4, Int32(npairs), Int32(a.i3),
                            grid_dim=((npairs + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
                        )
                self.ctx.enqueue_function[moe_hidden_reg_kernel](
                    a.p0, a.p1, a.p4, a.p5, a.p3,
                    Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3),
                    grid_dim=(moe_reg_blocks(npairs, a.i3, a.i1), 1, 1), block_dim=(MOE_RT, 1, 1),
                )
                return
        comptime if MOE_REGTILE and OP == OP_MOE_OUT:
            if a.i4 > 0:
                var npairs = (n // a.i0) * a.i2
                self.ctx.enqueue_function[moe_out_reg_kernel](
                    a.p0, a.p1, a.p6, a.p7, a.p5,
                    Int32(a.i0), Int32(a.i1), Int32(a.i3),
                    grid_dim=(moe_reg_blocks(npairs, a.i3, a.i0), 1, 1), block_dim=(MOE_RT, 1, 1),
                )
                self.ctx.enqueue_function[moe_combine_kernel](
                    a.p3, a.p5, a.p4, Int32(a.i0), Int32(a.i2), Int32(n),
                    grid_dim=((n + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
                )
                return
        comptime if OP == OP_MOE_HIDDEN:
            if a.i4 > 0 and _moe_tiled_on():
                self.ctx.enqueue_function[moe_hidden_tiled_kernel](
                    a.p0, a.p1, a.p4, a.p5, a.p6, a.p3,
                    Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3), Int32(a.i5),
                    grid_dim=(a.i4, 1, 1), block_dim=(MOE_TPB, 1, 1),
                )
                return
        comptime if OP == OP_MOE_OUT:
            if a.i4 > 0 and _moe_tiled_on():
                self.ctx.enqueue_function[moe_out_tiled_kernel](
                    a.p0, a.p1, a.p6, a.p7, a.p8, a.p5,
                    Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3), Int32(a.i5),
                    grid_dim=(a.i4, 1, 1), block_dim=(MOE_TPB, 1, 1),
                )
                self.ctx.enqueue_function[moe_combine_kernel](
                    a.p3, a.p5, a.p4, Int32(a.i0), Int32(a.i2), Int32(n),
                    grid_dim=((n + TPB - 1) // TPB, 1, 1), block_dim=(TPB, 1, 1),
                )
                return
        # lane/apple-fast-tsa2 (-D MOJOLEARN_TSA2_VAR): the threadgroup VAR
        # kernels without the env read of `_var_block_on` on the fit path
        comptime if TSA2_VAR and OP == OP_CHOLSOLVE:
            if a.i0 * a.i0 + a.i0 * a.i1 <= VAR_SMEM:
                self.ctx.enqueue_function[var_chol_block_kernel](
                    a.p0, a.p1, a.p2, Int32(a.i0), Int32(a.i1),
                    grid_dim=(1, 1, 1), block_dim=(VAR_TPB, 1, 1),
                )
                return
        comptime if TSA2_VAR and OP == OP_VAR_FORECAST:
            if (a.i1 + a.i3) * a.i0 <= VAR_SMEM:
                self.ctx.enqueue_function[var_forecast_block_kernel](
                    a.p0, a.p1, a.p2, Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3),
                    grid_dim=(1, 1, 1), block_dim=(VAR_TPB, 1, 1),
                )
                return
        # VAR's one-thread ops on one threadgroup (sequence/vecar_block.mojo):
        # the same chain per cell, so the same words
        comptime if OP == OP_CHOLSOLVE:
            if a.i0 * a.i0 + a.i0 * a.i1 <= VAR_SMEM and _var_block_on():
                self.ctx.enqueue_function[var_chol_block_kernel](
                    a.p0, a.p1, a.p2, Int32(a.i0), Int32(a.i1),
                    grid_dim=(1, 1, 1), block_dim=(VAR_TPB, 1, 1),
                )
                return
        comptime if OP == OP_VAR_FORECAST:
            if (a.i1 + a.i3) * a.i0 <= VAR_SMEM and _var_block_on():
                self.ctx.enqueue_function[var_forecast_block_kernel](
                    a.p0, a.p1, a.p2, Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3),
                    grid_dim=(1, 1, 1), block_dim=(VAR_TPB, 1, 1),
                )
                return
        comptime if SEQ_COOP and (OP == OP_AF_ALPHA or OP == OP_AF_DENOM or OP == OP_SEG_SUMSQ
                                  or OP == OP_LAMB_RATIO or OP == OP_GEMM or OP == OP_AF_BLK_SUMSQ):
            var coop = True
            comptime if OP == OP_GEMM:
                coop = a.i0 * a.i1 <= 1024 and a.i2 >= 32768
            elif OP == OP_AF_ALPHA or OP == OP_AF_DENOM:
                coop = a.i0 >= 4096
            if coop:
                self.ctx.enqueue_function[coop_kernel[OP]](
                    a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
                    _pack_ii(a.i0, a.i1), _pack_ii(a.i2, a.i3), _pack_ii(a.i4, a.i5),
                    _pack_ii(a.i6, a.i7), _pack_ii(a.i8, a.i9), _pack_ii(a.i10, a.i11),
                    _pack_ff(a.f0, a.f1), _pack_ff(a.f2, a.f3), _pack_ff(a.f4, a.f5), _pack_ff(a.f6, a.f7),
                    Int64(n),
                    grid_dim=((n * COOP_W + TPB - 1) // TPB, 1, 1),
                    block_dim=(TPB, 1, 1),
                )
                return
        self.ctx.enqueue_function[seq_kernel[OP]](
            a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
            _pack_ii(a.i0, a.i1), _pack_ii(a.i2, a.i3), _pack_ii(a.i4, a.i5),
            _pack_ii(a.i6, a.i7), _pack_ii(a.i8, a.i9), _pack_ii(a.i10, a.i11),
            _pack_ff(a.f0, a.f1), _pack_ff(a.f2, a.f3), _pack_ff(a.f4, a.f5), _pack_ff(a.f6, a.f7),
            Int64(n),
            grid_dim=((n + TPB - 1) // TPB, 1, 1),
            block_dim=(TPB, 1, 1),
        )

    def launch_team[OP: Int](mut self, a: Args, nblocks: Int, tpb: Int, wf: IP, woff: Int, nonce: Int32) raises:
        """`team_kernel[OP]` (lane neural-pass143): one block of tpb threads
        per series of a group of nblocks (sequence/fit_team.mojo)."""
        if nblocks <= 0:
            return
        self.ctx.enqueue_function[team_kernel[OP]](
            a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
            _pack_ii(a.i0, a.i1), _pack_ii(a.i2, a.i3), _pack_ii(a.i4, a.i5),
            _pack_ii(a.i6, a.i7), _pack_ii(a.i8, a.i9), _pack_ii(a.i10, a.i11),
            _pack_ff(a.f0, a.f1), _pack_ff(a.f2, a.f3), _pack_ff(a.f4, a.f5), _pack_ff(a.f6, a.f7),
            Int64(nblocks), wf, Int32(woff), nonce,
            grid_dim=(nblocks, 1, 1),
            block_dim=(tpb, 1, 1),
        )

    def copy(mut self, dst: FP, src: FP, n: Int) raises:
        """A device-to-device copy of n words between this executor's
        buffers, queued (no wait)."""
        if n <= 0:
            return
        var fd = self._find(dst, n)
        var fs = self._find(src, n)
        var vd = self._sub(fd[0], fd[1], n)
        var vs = self._sub(fs[0], fs[1], n)
        self.ctx.enqueue_copy(dst_buf=vd, src_buf=vs)
        _ = vd^
        _ = vs^

    def sync(mut self) raises:
        self.ctx.synchronize()
        self.staged.clear()
        for i in range(len(self.pstaged)):
            _pool_release_host(self.pstaged[i])
        self.pstaged.clear()
        if self.pooled:
            var pool = X_SEQUENCE_POOL.get_or_create_ptr()
            for i in range(len(self.pend_dst)):
                _pcopy(FP(unsafe_from_address=self.pend_dst[i]), pool[].host[self.ppend_host[i]].unsafe_ptr(), self.pend_n[i])
                _pool_release_host(self.ppend_host[i])
            self.ppend_host.clear()
        else:
            for i in range(len(self.pend_dst)):
                _pcopy(FP(unsafe_from_address=self.pend_dst[i]), self.pend_host[i].unsafe_ptr(), self.pend_n[i])
        self.pend_host.clear()
        self.pend_dst.clear()
        self.pend_n.clear()
