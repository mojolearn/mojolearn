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
from sequence.moe_group import (
    MOE_GROUP_TPB, moe_group_count_all_kernel, moe_group_offsets_all_kernel, moe_group_scatter_all_kernel,
    moe_group_zero_all_kernel,
)
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
from sequence.moe_mma import MM_BNH, MM_BNO, MM_NT, MOE_MMA, moe_hidden_mma_kernel, moe_mma_blocks, moe_out_mma_kernel

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int

from sequence.exec_trait import Exec
from sequence.dispatch import apply
from sequence.ops import OP_MOE_ROUTE, OP_MOE_OUT, OP_MOE_HIDDEN, FP, Args, OP_AF_ALPHA, OP_AF_BLK_SUMSQ, OP_AF_DENOM, OP_GEMM, OP_LAMB_RATIO, OP_SEG_SUMSQ
from sequence.coop import COOP_W, apply_coop
from sequence.gemm_tiled import GT_TPB, SEQ_GEMM_TILED, seq_gemm_tiled_blocks, seq_gemm_tiled_kernel, seq_gemm_tiled_on
from sequence.ops import OP_THETA
from sequence.ops import OP_LN_BWD_X, OP_LN_FWD, OP_AF_RMEAN, OP_AF_ROW
from sequence.theta_spec import THETA_SPEC
from sequence.ops import OP_CHOLSOLVE, OP_VAR_FORECAST, TSA2_VAR
from sequence.vecar_block import VAR_SMEM, VAR_TPB, var_chol_block_kernel, var_forecast_block_kernel
from sequence.fit_team import SeqTeam, garch_team, prophet_fit_team
from sequence.ets_team import ETS_TEAM, ets_team
from sequence.prophet_coop import PROPHET_COOP, prophet_fit_coop
from sequence.ops import OP_ETS, OP_GARCH
from sequence.recurrent_scan import OP_CELL_BWD_SCAN, OP_CELL_FWD_SCAN, cell_bwd_scan_kernel, cell_fwd_scan_kernel, scan_tpb
from x_linear.ops import IP
from x_linear.witness import witness_end
from std.sys.info import has_apple_gpu_accelerator

#: the simdgroup-cooperative long folds (sequence/coop.mojo): Apple, and
#: since nr-small D1/D11 (2026-10-04) NVIDIA and AMD in IDENTICAL. The
#: cooperative fold is the one-thread op's chain (same fmas, same values,
#: same order; only the loads are spread over the warp), so no bit moves on
#: any column; `coop_bcast` keeps a cell inside its 32-lane half of a CDNA
#: wavefront. -D MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF (or MOJOLEARN_IDN_ALL_OFF)
#: restores Apple only.
comptime SEQ_COOP_NVAMD = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime SEQ_COOP = has_apple_gpu_accelerator() or SEQ_COOP_NVAMD
#: nr-small D10: LayerNorm forward / backward-x rows on a simdgroup (the
#: same chains over broadcast words, coalesced loads; sequence/coop.mojo),
#: IDENTICAL on every GPU. -D MOJOLEARN_IDN_SEQ_LN_COOP_OFF (or
#: MOJOLEARN_IDN_ALL_OFF) restores one thread per row.
#: nr-small D11: Adafactor's row factor (op_af_row) and row-var mean
#: (op_af_rmean, one thread) on a simdgroup, the same chains; IDENTICAL.
#: -D MOJOLEARN_IDN_SEQ_AF_COOP_OFF (or MOJOLEARN_IDN_ALL_OFF).
comptime SEQ_AF_COOP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_SEQ_AF_COOP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime SEQ_LN_COOP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_SEQ_LN_COOP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

comptime TPB = 128

#: Apple FAST pipelined transport (lane apple-fast-regress, 2026-10-03;
#: docs/apple-fast/notes/regress-oct3.md). cpu-gpu-cleanup n-seq
#: (1774263e0) made `_pcopy` one memcpy on the calling thread; at the
#: board's 64 MB tensors (layernorm, adafactor, the optimizers) each
#: transfer is then a serial DMA plus a serial single-thread copy, the
#: rows' regression against 0.8.34. Instead of restoring host threads, the
#: copy and the DMA overlap, chunk by chunk (SEQ_PIPE_CH floats):
#:  MOJOLEARN_SEQ_FAST_PIPE_UP: an upload copies chunk i into its stage and
#:    queues its DMA at once, so the DMA of chunk i runs while chunk i + 1
#:    is copied (the stage still holds every chunk until the next sync).
#:  MOJOLEARN_SEQ_FAST_PIPE_DOWN: the downloads of a sync go through two
#:    pinned halves: the DMA of chunk i overlaps the read of chunk i - 1
#:    (opt_resident's OPT_PIPE_DOWN, which took the optimizers 318 -> 191 ms
#:    on the M3, for every x_sequence download).
#: Copies only: the same bytes, the same launches, no bit moves. Default on
#: FAST + Apple since the M3 A/B (n=3, digests identical): layernorm
#: 76.0 -> 52.2 ms (UP + DOWN), adafactor 685.7 -> 390.0 ms (DOWN), adagrad
#: 195.5 -> 199.1 ms (neutral). -D MOJOLEARN_SEQ_FAST_PIPE_UP_OFF /
#: -D MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF restore the serial copies; the old
#: -D MOJOLEARN_SEQ_FAST_PIPE_UP=1 / _DOWN=1 are harmless. IDENTICAL and
#: the other vendors compile the main path unchanged.
comptime _SEQ_APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: lane idn-opt-resident (2026-10-04): IDENTICAL takes the same pipelined
#: copies on every vendor (NVIDIA, AMD, Apple). Copies only: the same bytes,
#: the same launches, no bit moves. -D MOJOLEARN_IDN_SEQ_PIPE_UP_OFF /
#: -D MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF restore IDENTICAL's serial copies.
#: FAST on NVIDIA and AMD is unchanged.
comptime _SEQ_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime SEQ_PIPE_UP = (_SEQ_APPLE_FAST and not is_defined["MOJOLEARN_SEQ_FAST_PIPE_UP_OFF"]()) or (
    _SEQ_IDN and not (is_defined["MOJOLEARN_IDN_SEQ_PIPE_UP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime SEQ_PIPE_DOWN = (_SEQ_APPLE_FAST and not is_defined["MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF"]()) or (
    _SEQ_IDN and not (is_defined["MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
#: the pipelined chunk, floats (8 MB; lane apple-fast-gap-optim:
#: -D MOJOLEARN_SEQ_FAST_PIPE_CH=<floats> for the A/B)
comptime SEQ_PIPE_CH = get_defined_int["MOJOLEARN_SEQ_FAST_PIPE_CH", 1 << 21]()
#: lane apple-fast-gap-optim (2026-10-03, docs/apple-fast/notes/gap-optim.md),
#: default OFF, FAST + Apple only: the pipelined downloads' read-back.
#:  MOJOLEARN_SEQ_FAST_MAP_DOWN: each deferred download is read through
#:    `DeviceBuffer.map_to_host` (the runtime's own mapping) and one memcpy,
#:    instead of the two pinned (write-combined) halves.
#:  MOJOLEARN_SEQ_FAST_RAW_DOWN: each deferred download is DMAd straight
#:    into the caller's array in SEQ_PIPE_CH chunks, all queued, one wait
#:    (no stage and no host read).
#: Copies only: the same bytes.
#: SEQ_FAST_MAP_DOWN OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-04, lane/apple-fast-rec-ab2 @ 40027eb8e): layernorm 50.2 -> 79.8 ms.
#: DROPPED-slower: stays off.
comptime SEQ_MAP_DOWN = _SEQ_APPLE_FAST and SEQ_PIPE_DOWN and is_defined["MOJOLEARN_SEQ_FAST_MAP_DOWN"]()
comptime SEQ_RAW_DOWN = _SEQ_APPLE_FAST and SEQ_PIPE_DOWN and is_defined["MOJOLEARN_SEQ_FAST_RAW_DOWN"]() and not SEQ_MAP_DOWN


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
    for i in range(len(pool[].dev)):  # small-loop(pool: pooled device buffers): free-buffer slot search, no data
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
    #: SEQ_PIPE_DOWN: downloads deferred to the next sync's pipeline
    #: (destination address, device source address, count)
    var pipe_dst: List[Int]
    var pipe_src: List[Int]
    var pipe_n: List[Int]

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
        self.pipe_dst = List[Int]()
        self.pipe_src = List[Int]()
        self.pipe_n = List[Int]()

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
        for i in range(len(self.base)):  # small-loop(base: buffers this Exec allocated): address-to-buffer lookup, no data
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
            comptime if SEQ_PIPE_UP:
                if n >= 2 * SEQ_PIPE_CH:
                    # chunk i's DMA is queued as soon as it is staged, and
                    # runs while chunk i + 1 is copied
                    _ = view^
                    var hpf = FP(unsafe_from_address=Int(hp))
                    var done = 0
                    while done < n:
                        var cnt = min(SEQ_PIPE_CH, n - done)
                        memcpy(dest=hpf + done, src=src + done, count=cnt)
                        var cv = self._sub(found[0], found[1] + done, cnt)
                        self.ctx.enqueue_copy(dst_buf=cv, src_ptr=hpf + done)
                        _ = cv^
                        done += cnt
                    self.pstaged.append(hi)
                    return
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
        comptime if SEQ_PIPE_DOWN:
            if self.pooled and n >= 2 * SEQ_PIPE_CH:
                self.download_async(dst, src, n)
                self.sync()
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
        comptime if SEQ_PIPE_DOWN:
            if self.pooled and n >= 2 * SEQ_PIPE_CH:
                # nothing queued now: the next sync runs the pipeline
                self.pipe_dst.append(Int(dst))
                self.pipe_src.append(Int(src))
                self.pipe_n.append(n)
                return
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
        # lane apple-fast-gap-lstm (sequence/recurrent_scan.mojo): the whole
        # recurrence of a layer, one block per batch row (n = B rows)
        comptime if OP == OP_CELL_FWD_SCAN:
            self.ctx.enqueue_function[cell_fwd_scan_kernel](
                a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
                Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3), Int32(a.i4),
                grid_dim=(n, 1, 1), block_dim=(scan_tpb(a.i2), 1, 1),
            )
            return
        comptime if OP == OP_CELL_BWD_SCAN:
            self.ctx.enqueue_function[cell_bwd_scan_kernel](
                a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
                Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3), Int32(a.i4),
                grid_dim=(n, 1, 1), block_dim=(scan_tpb(a.i2), 1, 1),
            )
            return
        # The MoE products as tiled kernels with the items' chains (lane
        # neural-pass29, sequence/moe_tiled.mojo) when the entry grouped the
        # pairs by expert (a.i4 = the block count); MOJOLEARN_SEQ_MOE_TILED=0
        # keeps the one-thread-per-cell items.
        # Apple FAST (lane apple-fast-moespeed, sequence/moe_reg.mojo):
        # MOJOLEARN_MOE_REGTILE, the router's logits tiled and the products
        # register-tiled, the same chains; MOJOLEARN_MOE_DEVGROUP, the pairs
        # grouped by expert on the device (a.i6 = 1 from the entry).
        # lane cgr5-owed: the pairs grouped by expert on the device for any E
        # (sequence/moe_group.mojo), a.i6 = 2 from the entry: p2 the picks,
        # p4 order, p5 poff, p6 boff_h, p7 counts | cursors (2E int32
        # words), p8 boff_o; i3 E, i5 / i7 the F / D tiles.
        comptime if OP == OP_MOE_HIDDEN:
            if a.i6 == 2:
                var gp = n // a.i1
                var gt = MOE_GROUP_TPB
                self.ctx.enqueue_function[moe_group_zero_all_kernel](
                    a.p7, Int32(2 * a.i3), grid_dim=((2 * a.i3 + gt - 1) // gt, 1, 1), block_dim=(gt, 1, 1),
                )
                self.ctx.enqueue_function[moe_group_count_all_kernel](
                    a.p2, a.p7, Int32(gp), grid_dim=((gp + gt - 1) // gt, 1, 1), block_dim=(gt, 1, 1),
                )
                self.ctx.enqueue_function[moe_group_offsets_all_kernel](
                    a.p7, a.p5, a.p6, a.p8, Int32(a.i3), Int32(a.i5), Int32(a.i7),
                    grid_dim=((a.i3 + gt) // gt, 1, 1), block_dim=(gt, 1, 1),
                )
                self.ctx.enqueue_function[moe_group_scatter_all_kernel](
                    a.p2, a.p7, a.p5, a.p4, Int32(gp), Int32(a.i3),
                    grid_dim=((gp + gt - 1) // gt, 1, 1), block_dim=(gt, 1, 1),
                )
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
                comptime if MOE_MMA:
                    # lane apple-fast-gap-misc: simdgroup matrix products
                    # (sequence/moe_mma.mojo), MOJOLEARN_MOE_FAST_MMA*
                    if a.i6 == 1:
                        self.ctx.enqueue_function[moe_hidden_mma_kernel](
                            a.p0, a.p1, a.p4, a.p5, a.p3,
                            Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3),
                            grid_dim=(moe_mma_blocks(npairs, a.i3, a.i1, MM_BNH), 1, 1), block_dim=(MM_NT, 1, 1),
                        )
                        return
                self.ctx.enqueue_function[moe_hidden_reg_kernel](
                    a.p0, a.p1, a.p4, a.p5, a.p3,
                    Int32(a.i0), Int32(a.i1), Int32(a.i2), Int32(a.i3),
                    grid_dim=(moe_reg_blocks(npairs, a.i3, a.i1), 1, 1), block_dim=(MOE_RT, 1, 1),
                )
                return
        comptime if MOE_REGTILE and OP == OP_MOE_OUT:
            if a.i4 > 0:
                var npairs = (n // a.i0) * a.i2
                var mma_done = False
                comptime if MOE_MMA:
                    if a.i6 == 1:
                        self.ctx.enqueue_function[moe_out_mma_kernel](
                            a.p0, a.p1, a.p6, a.p7, a.p5,
                            Int32(a.i0), Int32(a.i1), Int32(a.i3),
                            grid_dim=(moe_mma_blocks(npairs, a.i3, a.i0, MM_BNO), 1, 1), block_dim=(MM_NT, 1, 1),
                        )
                        mma_done = True
                if not mma_done:
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
        comptime if (SEQ_COOP and (OP == OP_AF_ALPHA or OP == OP_AF_DENOM or OP == OP_SEG_SUMSQ
                                   or OP == OP_LAMB_RATIO or OP == OP_GEMM or OP == OP_AF_BLK_SUMSQ
                                   or (THETA_SPEC and OP == OP_THETA))) or (
                SEQ_LN_COOP and (OP == OP_LN_FWD or OP == OP_LN_BWD_X)) or (
                SEQ_AF_COOP and (OP == OP_AF_ROW or OP == OP_AF_RMEAN)):
            var coop = True
            comptime if OP == OP_LN_FWD or OP == OP_LN_BWD_X or OP == OP_AF_ROW or OP == OP_AF_RMEAN:
                coop = a.i0 >= COOP_W
            elif OP == OP_GEMM:
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
        # nr-small D1: op_gemm's chain with the A/B slabs staged in
        # threadgroup memory (sequence/gemm_tiled.mojo), same words
        comptime if SEQ_GEMM_TILED and OP == OP_GEMM:
            if seq_gemm_tiled_on(a.i0, a.i1):
                self.ctx.enqueue_function[seq_gemm_tiled_kernel](
                    a.p0, a.p1, a.p2,
                    Int32(a.i0), Int32(a.i1), Int32(a.i2),
                    Int32(a.i3), Int32(a.i4), Int32(a.i5), Int32(a.i6),
                    Int32(a.i7), Int32(a.i8),
                    grid_dim=(seq_gemm_tiled_blocks(a.i0, a.i1), 1, 1),
                    block_dim=(GT_TPB, 1, 1),
                )
                return
        comptime if OP != OP_CELL_FWD_SCAN and OP != OP_CELL_BWD_SCAN:
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
        comptime if SEQ_PIPE_DOWN:
            if len(self.pipe_n) > 0:
                self._pipe_down()

    def _pipe_down(mut self) raises:
        """SEQ_PIPE_DOWN: the deferred downloads through two pinned halves of
        SEQ_PIPE_CH floats, after the queue has drained (every kernel that
        writes them is done): the DMA of chunk i runs into one half while
        chunk i - 1 is read out of the other. Each chunk's wait comes before
        its half is reused (that half was last read for chunk i - 2, before
        the previous wait). Returns with every byte in place."""
        comptime if SEQ_MAP_DOWN:
            for j in range(len(self.pipe_n)):  # small-loop(pipe_n: deferred download chunks): one mapped DMA copy per chunk into caller memory
                var nj = self.pipe_n[j]
                var f = self._find(FP(unsafe_from_address=self.pipe_src[j]), nj)
                var v = self._sub(f[0], f[1], nj)
                with v.map_to_host() as h:
                    memcpy(dest=FP(unsafe_from_address=self.pipe_dst[j]),
                           src=FP(unsafe_from_address=Int(h.unsafe_ptr())), count=nj)
                _ = v^
            self.pipe_dst.clear()
            self.pipe_src.clear()
            self.pipe_n.clear()
            return
        comptime if SEQ_RAW_DOWN:
            for j in range(len(self.pipe_n)):
                var nj = self.pipe_n[j]
                var done = 0
                while done < nj:
                    var cnt = min(SEQ_PIPE_CH, nj - done)
                    var f = self._find(FP(unsafe_from_address=self.pipe_src[j] + done * 4), cnt)
                    var v = self._sub(f[0], f[1], cnt)
                    self.ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=self.pipe_dst[j] + done * 4), src_buf=v)
                    _ = v^
                    done += cnt
            self.ctx.synchronize()
            self.pipe_dst.clear()
            self.pipe_src.clear()
            self.pipe_n.clear()
            return
        var h0 = _pool_host(self.ctx, SEQ_PIPE_CH)
        var h1 = _pool_host(self.ctx, SEQ_PIPE_CH)
        var pool = X_SEQUENCE_POOL.get_or_create_ptr()
        var st0 = FP(unsafe_from_address=Int(pool[].host[h0].unsafe_ptr()))
        var st1 = FP(unsafe_from_address=Int(pool[].host[h1].unsafe_ptr()))
        var have_prev = False
        var prev_dst = 0
        var prev_cnt = 0
        var prev_half = 0
        var half = 0
        for j in range(len(self.pipe_n)):
            var nj = self.pipe_n[j]
            var done = 0
            while done < nj:
                var cnt = min(SEQ_PIPE_CH, nj - done)
                var f = self._find(FP(unsafe_from_address=self.pipe_src[j] + done * 4), cnt)
                var v = self._sub(f[0], f[1], cnt)
                self.ctx.enqueue_copy(dst_ptr=st0 if half == 0 else st1, src_buf=v)
                _ = v^
                if have_prev:
                    # overlaps the DMA just queued, into the other half
                    memcpy(dest=FP(unsafe_from_address=prev_dst), src=st0 if prev_half == 0 else st1,
                           count=prev_cnt)
                self.ctx.synchronize()
                have_prev = True
                prev_dst = self.pipe_dst[j] + done * 4
                prev_cnt = cnt
                prev_half = half
                half = 1 - half
                done += cnt
        if have_prev:
            memcpy(dest=FP(unsafe_from_address=prev_dst), src=st0 if prev_half == 0 else st1, count=prev_cnt)
        _pool_release_host(h0)
        _pool_release_host(h1)
        self.pipe_dst.clear()
        self.pipe_src.clear()
        self.pipe_n.clear()
