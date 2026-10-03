# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.dsort import sort_cols_device, sort_scratch_words
from x_prep.dradix import RADIX_SORT, RADIX_MIN_ROWS, radix_sort_cols_device, radix_scratch_words
from x_prep.fastred import (
    TGR, col_stats_fast_kernel, pt_fold_fast_kernel, class_stats_fast_kernel, ii_mean_fast_kernel,
    ii_gram_fast_kernel,
)
from x_prep.dmi import mi_cd_device, mi_w_words, mi_scratch_words
from x_prep.select_fast import (
    SELECT_FREG, SELECT_FCLS, OP_F_CLASSIF, OP_F_REGRESSION, program_has_op, select_scratch_words,
    select_freg_device, select_fcls_device, select_cstats_device,
)
from x_prep.fastprep2 import PREP2_FAST, Prep2Switches, prep2_scratch_words, prep2_fast_stage
from core.arena_io import check_in_ranges, check_out_ranges, upload_ranges, download_ranges
from core.device_store import DeviceStore

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


#: The binding's resident input store (core/device_store.mojo; lane
#: py-shared): `x_prep_dev_put` / `_free` / `_live`, one per tier.
comptime _STORE_NAME = "MojoXPrepStoreIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXPrepStoreFast"
comptime X_PREP_STORE = _Global[StorageType=DeviceStore, name=_STORE_NAME, init_fn=DeviceStore.__init__]


def run_program_device_ranges(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int, scratch_len: Int,
                              out_addr: Int, out_len: Int, ins_addr: Int, nins: Int, outs_addr: Int,
                              nouts: Int) raises:
    """`run_program_device` whose host arena crosses by RANGES (lane
    py-shared, core/arena_io.mojo): only the `nins` input triples go up
    (every other arena word starts zero on the device, as the host's do;
    src >= 0 copies a resident `x_prep_dev_put` slot), and only the `nouts`
    output quads of the host arena come back. The scratch and the output
    region behave as in `run_program_device`."""
    check_in_ranges(ins_addr, nins, arena_len)
    check_out_ranges(outs_addr, nouts, arena_len)
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages, scratch_len,
        out_addr, out_len, ins_addr, nins, outs_addr, nouts,
    )


def _env_int(name: String, default: Int) -> Int:
    """The integer value of an environment switch (the default when unset or not a number)."""
    var v = String(getenv(name))
    if v == "":
        return default
    try:
        return Int(v)
    except:
        return default


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
                           out_addr: Int = 0, out_len: Int = 0, ins_addr: Int = 0, nins: Int = -1,
                           outs_addr: Int = 0, nouts: Int = -1) raises:
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
    # FAST on Apple (lane prep-apple3): sort_cols by radix (x_prep/dradix.mojo, the same words).
    # Default since request 1790627886703 (M3 Ultra, 16 columns x 1M rows: RobustScaler 0.141 ->
    # 0.070 s, every digest equal); MOJOLEARN_XPREP_SORT_RADIX=0 is the bitonic sort.
    # MOJOLEARN_XPREP_SORT_CHUNK = positions per chunk (512 to 4096 measured within 0.006 s).
    var radix = False
    var radix_rows = 2048
    comptime if RADIX_SORT:
        radix = getenv("MOJOLEARN_XPREP_SORT_RADIX", "1") != "0"
        radix_rows = max(1, _env_int("MOJOLEARN_XPREP_SORT_CHUNK", 2048))
    # MOJOLEARN_XPREP_PROFILE=1: XPPHASE lines (a wait after every phase; timing only)
    var prof = getenv("MOJOLEARN_XPREP_PROFILE", "0") == "1"
    var t_last = perf_counter_ns()
    var scratch = 1
    for s in range(stages):
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_SORT_COLS:
            var sq = host_q + (s * STAGE_INTS + 2)
            var units = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
            scratch = max(scratch, sort_scratch_words(Int(sq[1]), units))
            if radix and Int(sq[1]) >= RADIX_MIN_ROWS:
                scratch = max(scratch, radix_scratch_words(Int(sq[1]), units, radix_rows))
    comptime if SELECT_FREG or SELECT_FCLS:
        # FAST on Apple (lane/apple-fast-select, -D MOJOLEARN_SELECT_FREG / _FCLS): the
        # f_regression / f_classif tiles' partials live in the sort scratch (x_prep/select_fast.mojo)
        var sel_fcls = program_has_op(host_q, stages, OP_F_CLASSIF)
        for s in range(stages):
            var sq = host_q + (s * STAGE_INTS + 2)
            scratch = max(scratch, select_scratch_words(Int(host_q.unsafe_load(s * STAGE_INTS)), Int(sq[1]),
                                                        Int(sq[2]), Int(sq[4]), sel_fcls))
    # FAST: MOJOLEARN_XPREP_FAST_FOLDS=0 keeps the row-order units (the A/B arm of
    # bench/x_prep_quality.py and bench/x_prep_speed.py); unset or 1 folds by threadgroup
    var fast_folds = getenv("MOJOLEARN_XPREP_FAST_FOLDS", "1") != "0"
    # lane/apple-fast-prep2 (2026-10-02): FAST + Apple paths behind env switches, each default OFF
    # (x_prep/fastprep2.mojo: te_global / te_enc / ii_conv / ii_gram / eigh by threadgroups, the quantile
    # stage by radix select). Every field is False outside FAST + Apple.
    var p2 = Prep2Switches()
    comptime if PREP2_FAST:
        scratch = max(scratch, prep2_scratch_words(host_q, stages, p2))
    var mi_sorted = getenv("MOJOLEARN_XPREP_MI_SORTED", "1") != "0"
    var mi_ties = getenv("MOJOLEARN_XPREP_MI_TIES", "1") != "0"
    var mi_w = 1
    var mi_u = 1
    for s in range(stages):
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_MI_CD:
            var mq = host_q + (s * STAGE_INTS + 2)
            mi_w = max(mi_w, mi_w_words(Int(mq[1]), Int(mq[2])))
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
    if prof:
        ctx.synchronize()
        var now = perf_counter_ns()
        print("XPPHASE alloc us", (now - t_last) // 1000, "arena", arena_len, "scratch", max(scratch_len, 0), "out", out_n,
              "sort", scratch)
        t_last = now
    if nins >= 0:
        upload_ranges(ctx, df, host_f, arena_len, ins_addr, nins, X_PREP_STORE.get_or_create_ptr()[])
    elif arena_len > 0:
        if dev_len > arena_len:
            ctx.enqueue_copy(dst_buf=df.create_sub_buffer[DType.float32](0, arena_len), src_ptr=host_f)
        else:
            ctx.enqueue_copy(dst_buf=df, src_ptr=host_f)
    if out_n > 0:
        ctx.enqueue_memset(df.create_sub_buffer[DType.float32](out_at, out_n), Float32(0))
    if stages > 0:
        ctx.enqueue_copy(dst_buf=dq, src_ptr=host_q)
    if prof:
        ctx.synchronize()
        var now = perf_counter_ns()
        print("XPPHASE upload us", (now - t_last) // 1000, "ranges", nins)
        t_last = now
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
        if prof and s > 0:
            ctx.synchronize()
            var now = perf_counter_ns()
            print("XPPHASE stage", s - 1, "op", Int(host_q.unsafe_load((s - 1) * STAGE_INTS)), "us", (now - t_last) // 1000)
            t_last = now
        if total <= 0:
            continue
        var qp = dq.unsafe_ptr() + (s * STAGE_INTS + 2)
        if mi_sorted and op == OP_MI_CD:
            var hq = host_q + (s * STAGE_INTS + 2)
            mi_cd_device(ctx, df, dmw, dmu, dq, s * STAGE_INTS + 2, total, Int(hq[1]), Int(hq[2]), Int(hq[0]),
                         Int(hq[7]), mi_ties)
            continue
        if op == OP_SORT_COLS:
            var hq = host_q + (s * STAGE_INTS + 2)
            var by_radix = False
            comptime if RADIX_SORT:
                if radix and Int(hq[1]) >= RADIX_MIN_ROWS:
                    radix_sort_cols_device(ctx, df, dw, total, Int(hq[0]), Int(hq[1]), Int(hq[2]),
                                           Int(hq[3]), Int(hq[4]), radix_rows)
                    by_radix = True
            if not by_radix:
                sort_cols_device(ctx, df, dw, total, Int(hq[0]), Int(hq[1]), Int(hq[2]),
                                 Int(hq[3]), Int(hq[4]))
            continue
        comptime if SELECT_FREG:
            # FAST on Apple (lane/apple-fast-select): f_regression as row x feature tiles
            if op == OP_F_REGRESSION:
                var hq = host_q + (s * STAGE_INTS + 2)
                if select_freg_device(ctx, df, dw, Int(hq[0]), Int(hq[1]), Int(hq[2]), Int(hq[3]), Int(hq[4]),
                                      Int(hq[5]), Int(hq[6]), Int(hq[7]), Int(hq[8])):
                    continue
        comptime if SELECT_FCLS:
            # FAST on Apple (lane/apple-fast-select): f_classif, and the class_stats ahead of
            # it in the same program, as row x feature tiles
            if op == OP_F_CLASSIF:
                var hq = host_q + (s * STAGE_INTS + 2)
                if select_fcls_device(ctx, df, dw, Int(hq[0]), Int(hq[1]), Int(hq[2]), Int(hq[3]), Int(hq[4]),
                                      Int(hq[5]), Int(hq[6]), Int(hq[7]), Int(hq[8])):
                    continue
            if op == OP_CLASS_STATS and program_has_op(host_q, stages, OP_F_CLASSIF):
                var hq = host_q + (s * STAGE_INTS + 2)
                if select_cstats_device(ctx, df, dw, Int(hq[0]), Int(hq[1]), Int(hq[2]), Int(hq[3]), Int(hq[4]),
                                        Int(hq[5]), Int(hq[6]), Int(hq[7]), Int(hq[8]), Int(hq[9])):
                    continue
        comptime if PREP2_FAST:
            if prep2_fast_stage(ctx, df, dw, host_q, s, op, total, qp.unsafe_origin_cast[MutAnyOrigin](), p2):
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
    if prof and stages > 0:
        ctx.synchronize()
        var now = perf_counter_ns()
        print("XPPHASE stage", stages - 1, "op", Int(host_q.unsafe_load((stages - 1) * STAGE_INTS)), "us",
              (now - t_last) // 1000)
        t_last = now
    if nouts >= 0:
        download_ranges(ctx, df, host_f, outs_addr, nouts)
    elif arena_len > 0:
        if dev_len > arena_len:
            ctx.enqueue_copy(dst_ptr=host_f, src_buf=df.create_sub_buffer[DType.float32](0, arena_len))
        else:
            ctx.enqueue_copy(dst_ptr=host_f, src_buf=df)
    if out_n > 0:
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=out_addr), src_buf=df.create_sub_buffer[DType.float32](out_at, out_n))
    ctx.synchronize()
    if prof:
        var now = perf_counter_ns()
        print("XPPHASE download us", (now - t_last) // 1000, "ranges", nouts)
        t_last = now
    _ = dw^
    _ = dmw^
    _ = dmu^
    _ = dq^
    _ = df^
    _ = ctx^
    if prof:
        print("XPPHASE release us", (perf_counter_ns() - t_last) // 1000)
