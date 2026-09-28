# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from std.os import getenv
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.dsort import sort_cols_device, sort_scratch_words
from x_prep.fastred import (
    TGR, col_stats_fast_kernel, pt_fold_fast_kernel, class_stats_fast_kernel, ii_mean_fast_kernel,
    ii_gram_fast_kernel,
)
from x_prep.dmi import mi_cd_device, mi_big_n, mi_scratch_words

#: op 69 (`mi_cd`) runs as the sorted neighbour search of x_prep/dmi.mojo
#: (the host's argument, x_prep/host/mutual_info.mojo: the same words)
comptime OP_MI_CD = 69

#: FAST only: ops folded by a threadgroup per column (x_prep/fastred.mojo)
comptime OP_COL_STATS = 1
comptime OP_CLASS_STATS = 16
comptime OP_II_MEAN = 53
comptime OP_II_GRAM = 54
comptime OP_PT_FOLD = 106

#: op 0 (`sort_cols`) runs as the device sort of x_prep/dsort.mojo, not as
#: one heapsort thread per column: the same words (a sort under a total
#: order has one answer), at every thread of the GPU.
comptime OP_SORT_COLS = 0

comptime BLOCK = 128


struct _PrepContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every `x_prep_run` (the x_cnn
    `_Global` pattern; CURRENT DIRECTIVES, 2026-09-27: a context per call hung
    the SECOND call of x_cluster / x_neighbors on an RTX 4090, and on Metal a
    context per call exhausts the per-process command queues). The slot keeps
    a reference for the life of the process, so every call's buffers die
    inside it. One slot per numeric tier, so a FAST and an IDENTICAL .so in
    one process never share it. Context lifetime moves no bit."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXPrepContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXPrepContextFast"
comptime X_PREP_CONTEXT = _Global[StorageType=_PrepContext, name=_CTX_NAME, init_fn=_PrepContext.__init__]


def x_prep_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_PREP_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


def prep_kernel[OP: Int](f: FP, q: IP, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        run_unit[OP](t, f, q)


def run_program_device(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int, scratch_len: Int = 0,
                       out_addr: Int = 0, out_len: Int = 0) raises:
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages, scratch_len,
        out_addr, out_len,
    )


def run_program_device_ptr(host_f: FP, arena_len: Int, host_q: IP, stages: Int, scratch_len: Int = 0,
                           out_addr: Int = 0, out_len: Int = 0) raises:
    """scratch_len (lane prep-apple2): words of DEVICE-ONLY arena after the
    host's arena_len words (offsets arena_len ..); they never cross to or
    from the host and start undefined, so a program writes each scratch word
    before it reads it. out_len words after those (offsets arena_len +
    scratch_len ..) are the program's OUTPUT: zeroed on the device (as the
    host arena's words arrive zeroed), never uploaded, and copied back into
    the host buffer at out_addr, not into the arena. Where a word lives moves
    no bit."""
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        if op < 0 or op >= N_OPS:
            raise Error(String("x_prep: unknown op ", op))
    var scratch = 1
    for s in range(stages):
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_SORT_COLS:
            var sq = host_q + (s * STAGE_INTS + 2)
            scratch = max(scratch, sort_scratch_words(Int(sq[1]), Int(host_q.unsafe_load(s * STAGE_INTS + 1))))
    # FAST: MOJOLEARN_XPREP_FAST_FOLDS=0 keeps the row-order units (the A/B arm of
    # bench/x_prep_quality.py and bench/x_prep_speed.py); unset or 1 folds by threadgroup
    var fast_folds = getenv("MOJOLEARN_XPREP_FAST_FOLDS", "1") != "0"
    var mi_sorted = getenv("MOJOLEARN_XPREP_MI_SORTED", "1") != "0"
    var mi_w = 1
    var mi_u = 1
    for s in range(stages):
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_MI_CD:
            var mq = host_q + (s * STAGE_INTS + 2)
            mi_w = max(mi_w, Int(mq[2]) * mi_big_n(Int(mq[1])))
            mi_u = max(mi_u, mi_scratch_words(Int(mq[1]), Int(mq[2])))
    var ctx = x_prep_ctx()
    var dmw = ctx.enqueue_create_buffer[DType.uint64](mi_w if mi_sorted else 1)
    var dmu = ctx.enqueue_create_buffer[DType.uint32](mi_u if mi_sorted else 1)
    var out_n = out_len if out_addr != 0 and out_len > 0 else 0
    var out_at = arena_len + max(scratch_len, 0)
    var dev_len = out_at + out_n
    var df = ctx.enqueue_create_buffer[DType.float32](dev_len if dev_len > 0 else 1)
    var dw = ctx.enqueue_create_buffer[DType.uint32](scratch)
    var dq = ctx.enqueue_create_buffer[DType.int32](stages * STAGE_INTS if stages > 0 else 1)
    if arena_len > 0:
        if dev_len > arena_len:
            ctx.enqueue_copy(dst_buf=df.create_sub_buffer[DType.float32](0, arena_len), src_ptr=host_f)
        else:
            ctx.enqueue_copy(dst_buf=df, src_ptr=host_f)
    if out_n > 0:
        ctx.enqueue_memset(df.create_sub_buffer[DType.float32](out_at, out_n), Float32(0))
    if stages > 0:
        ctx.enqueue_copy(dst_buf=dq, src_ptr=host_q)
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
        if total <= 0:
            continue
        var qp = dq.unsafe_ptr() + (s * STAGE_INTS + 2)
        if mi_sorted and op == OP_MI_CD:
            var hq = host_q + (s * STAGE_INTS + 2)
            mi_cd_device(ctx, df, dmw, dmu, dq, s * STAGE_INTS + 2, total, Int(hq[1]), Int(hq[2]), Int(hq[0]))
            continue
        if op == OP_SORT_COLS:
            var hq = host_q + (s * STAGE_INTS + 2)
            sort_cols_device(ctx, df, dw, total, Int(hq[0]), Int(hq[1]), Int(hq[2]),
                             Int(hq[3]), Int(hq[4]))
            continue
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            if fast_folds and op == OP_COL_STATS:
                var hq = host_q + (s * STAGE_INTS + 2)
                ctx.enqueue_function[col_stats_fast_kernel](
                    df.unsafe_ptr(), hq[0], hq[1], hq[2], hq[3], grid_dim=total, block_dim=TGR,
                )
                continue
            if fast_folds and op == OP_PT_FOLD:
                ctx.enqueue_function[pt_fold_fast_kernel](df.unsafe_ptr(), qp, grid_dim=total, block_dim=TGR)
                continue
            if fast_folds and op == OP_CLASS_STATS and host_q.unsafe_load(s * STAGE_INTS + 2 + 9) == 0:
                ctx.enqueue_function[class_stats_fast_kernel](df.unsafe_ptr(), qp, grid_dim=total, block_dim=TGR)
                continue
            if fast_folds and op == OP_II_MEAN:
                ctx.enqueue_function[ii_mean_fast_kernel](df.unsafe_ptr(), qp, grid_dim=total, block_dim=TGR)
                continue
            if fast_folds and op == OP_II_GRAM:
                ctx.enqueue_function[ii_gram_fast_kernel](df.unsafe_ptr(), qp, grid_dim=total, block_dim=TGR)
                continue
        comptime for k in range(N_OPS):
            if op == k:
                ctx.enqueue_function[prep_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
    if arena_len > 0:
        if dev_len > arena_len:
            ctx.enqueue_copy(dst_ptr=host_f, src_buf=df.create_sub_buffer[DType.float32](0, arena_len))
        else:
            ctx.enqueue_copy(dst_ptr=host_f, src_buf=df)
    if out_n > 0:
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=out_addr), src_buf=df.create_sub_buffer[DType.float32](out_at, out_n))
    ctx.synchronize()
    _ = dw^
    _ = dmw^
    _ = dmu^
    _ = dq^
    _ = df^
    _ = ctx^
