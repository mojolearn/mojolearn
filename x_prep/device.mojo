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
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.dsort import sort_cols_device, sort_scratch_words
from x_prep.dradix import RADIX_SORT, RADIX_MIN_ROWS, radix_sort_cols_device, radix_scratch_words
from x_prep.fastred import (
    TGR, col_stats_fast_kernel, pt_fold_fast_kernel, class_stats_fast_kernel, ii_mean_fast_kernel,
    ii_gram_fast_kernel,
)
from x_prep.dmi import mi_cd_device, mi_w_words, mi_scratch_words
#: lane af-ptimpute (2026-10-03), FAST + Apple + define only (x_prep/fastpt.mojo): the import
#: instantiates nothing; every launch below sits inside `comptime if PT_* / SI_*`
from x_prep.fastpt import (
    PT_COLBATCH, PT_SPEC, PT_FUSED_TRANSFORM, SI_ONEPASS, OP_PT_MAP, OP_PT_SMAP, OP_PT_SFOLD, OP_PT_APPLY,
    pt_colbatch_fold, pt_spec_fold, cs_tile_stats, ptimpute_part_words, fused_tail_pair,
)
from x_prep.dmi_fast import mi_cc_device, mi_cd_device_rank, mi_colscale_fast_kernel, mi_reduce_fast_kernel, TGF
from core.arena_io import check_in_ranges, check_out_ranges, upload_ranges, download_ranges
from core.device_store import DeviceStore
from x_linear.fast_gram import fast_sym_gram_into, fg_part_words
from x_prep.rr_eigh import rr_eigh_into, rre_words
from x_prep.da_par import (
    DA_TPB, DT, DT_TPB, lda2_rank_kernel, lda2_scal1_kernel, lda2_ms_kernel, lda2_g2_kernel, lda3_rank_kernel,
    lda3_scal_kernel, lda3_tmp_kernel, lda3_inter_kernel, lda3_coef_kernel, lda3_dot_kernel, qda_prep_scal_kernel,
    qda_prep_rot_kernel, qda_dec_tile_kernel,
)


def _da_blocks(t: Int) -> Int:
    return max((t + DA_TPB - 1) // DA_TPB, 1)

#: FAST on Apple by default (lane/apple-fast-ldaqda; off: -D MOJOLEARN_LDAQDA_RR_EIGH_OFF): the
#: `eigh` stage (op 18) as x_prep/rr_eigh.mojo's round-robin Jacobi on the
#: whole GPU instead of `eigh_unit`'s one thread per matrix (cyclic, 24,090
#: serial rotations a sweep at LDA's / QDA's d = 220 on Istella).
comptime OP_EIGH = 18
#: FAST on Apple by default (off: -D MOJOLEARN_LDAQDA_PAR_STAGES_OFF / _DEC_TILE_OFF): x_prep/da_par.mojo,
#: lda_stage2 / lda_stage3 / qda_prep a thread a cell, qda_dec by shared tiles
#: (each cell the unit's own chain: the units' words)
comptime OP_LDA_STAGE2 = 38
comptime OP_LDA_STAGE3 = 39
comptime OP_QDA_PREP = 41
comptime OP_QDA_DEC = 42
#: RR_EIGH, PAR_STAGES and DEC_TILE are the FAST + Apple default since the M3
#: A/B on lane/apple-fast-ldaqda ef187d37b (n=1, Istella), all three on:
#: lda-clf 19,696 -> 536 ms, qda 15,794 -> 425 ms (RR_EIGH alone 607 / 435);
#: accuracy / logloss unchanged (lda .9131 / .2357, qda .8805 / 3.608 -> 3.609).
#: `-D MOJOLEARN_LDAQDA_<NAME>_OFF` restores main's path for that stage; the old
#: `-D MOJOLEARN_LDAQDA_<NAME>` stays harmless. The three are independent.
comptime PAR_STAGES = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                       and not is_defined["MOJOLEARN_LDAQDA_PAR_STAGES_OFF"]())
comptime DEC_TILE = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                     and not is_defined["MOJOLEARN_LDAQDA_DEC_TILE_OFF"]())
comptime RR_EIGH = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                    and not is_defined["MOJOLEARN_LDAQDA_RR_EIGH_OFF"]())

#: lane/apple-fast-mi (2026-10-03): FAST + Apple only, default OFF, one `-D MOJOLEARN_MI_<NAME>`
#: each (docs/apple-fast/ab/mi.md; MOJOLEARN_MI_ALL turns every one on). x_prep/dmi_fast.mojo:
#: REG_SORTCOUNT: op 68 (`mi_cc`, mutual_info_regression) as the sorted Kraskov search (the
#: host's argument, x_prep/host/mutual_info.mojo `_cc_column`) instead of the brute-force
#: unit; REG_TIES: its tie-aware form (implies SORTCOUNT); REG_RANKMAJOR: its point kernel
#: one thread per (column, sorted rank) (implies SORTCOUNT); FAST_FOLDS: ops 66 / 70
#: (`mi_colscale`, `mi_reduce`) as threadgroup folds; CLF_RANKMAJOR: op 69's point kernel
#: one thread per (column, sorted rank). Every launch is inside these guards; IDENTICAL and
#: the other vendors compile main's code unchanged.
comptime _MI_FA = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime MI_ALL = is_defined["MOJOLEARN_MI_ALL"]()
comptime MI_REG_TIES = _MI_FA and (MI_ALL or is_defined["MOJOLEARN_MI_REG_TIES"]())
comptime MI_REG_RANKMAJOR = _MI_FA and (MI_ALL or is_defined["MOJOLEARN_MI_REG_RANKMAJOR"]())
comptime MI_REG_SORTED = _MI_FA and (MI_REG_TIES or MI_REG_RANKMAJOR or is_defined["MOJOLEARN_MI_REG_SORTCOUNT"]())
comptime MI_FAST_FOLDS = _MI_FA and (MI_ALL or is_defined["MOJOLEARN_MI_FAST_FOLDS"]())
comptime MI_CLF_RANKMAJOR = _MI_FA and (MI_ALL or is_defined["MOJOLEARN_MI_CLF_RANKMAJOR"]())
comptime OP_MI_CC = 68
comptime OP_MI_COLSCALE = 66
comptime OP_MI_REDUCE = 70

#: op 69 (`mi_cd`) runs as the sorted neighbour search of x_prep/dmi.mojo
#: (the host's argument, x_prep/host/mutual_info.mojo: the same words)
comptime OP_MI_CD = 69

#: FAST only: ops folded by a threadgroup per column (x_prep/fastred.mojo)
comptime OP_COL_STATS = 1
#: FAST on Apple (lane/apple-fast-gram, 2026-10-02), the FAST + Apple default
#: since the M3 re-A/B on lane head c338b88dd (n=1, Istella): qda 16,310 ->
#: 15,811 ms with acc .866 -> .881; lda 19,026 -> 19,641 ms with acc .909 ->
#: .913, logloss .264 -> .236 (kept for quality). `-D
#: MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF` restores main's path; the old
#: `-D MOJOLEARN_X_PREP_CLASS_COV_GRID` stays harmless.
#: QDA's per-class covariances (op 40, naive_bayes/da.mojo `qda_cov_unit`: one
#: thread per (class, cell) walking every row, K d^2 chains of a million rows)
#: and LDA's Gram `matmul` (op 13, x_prep/prims.mojo `matmul_unit` as Z'Z: one
#: thread per cell walking every row) run as x_linear/fast_gram.mojo's grid
#: Gram: row chunks x 32 x 32 tiles through shared memory, then a sum over the
#: chunks. Board: lda-clf Istella 5.1x, qda Istella 2.6x behind scikit-learn.
comptime OP_MATMUL = 13
comptime OP_QDA_COV = 40
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


def _matmul_is_gram(hq: IP, total: Int) -> Bool:
    """`matmul` q = [A, sa0, sa1, B, sb0, sb1, C, ncols, K, BIAS, ALPHA] as
    the d x d Gram A'A of a K x d row-major A (LDA's Z'Z): both operands the
    same array, A read down a column and B along a row, no bias, no scale."""
    var d = Int(hq[7])
    return (Int(hq[0]) == Int(hq[3]) and Int(hq[1]) == 1 and Int(hq[2]) == d and Int(hq[4]) == d
            and Int(hq[5]) == 1 and Int(hq[8]) > 0 and Int(hq[9]) < 0 and Int(hq[10]) < 0
            and d > 0 and total == d * d)


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
    # FAST: MOJOLEARN_XPREP_FAST_FOLDS=0 keeps the row-order units (the A/B arm of
    # bench/x_prep_quality.py and bench/x_prep_speed.py); unset or 1 folds by threadgroup
    var fast_folds = getenv("MOJOLEARN_XPREP_FAST_FOLDS", "1") != "0"
    var mi_sorted = getenv("MOJOLEARN_XPREP_MI_SORTED", "1") != "0"
    var mi_ties = getenv("MOJOLEARN_XPREP_MI_TIES", "1") != "0"
    var mi_w = 1
    var mi_u = 1
    for s in range(stages):
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_MI_CD:
            var mq = host_q + (s * STAGE_INTS + 2)
            mi_w = max(mi_w, mi_w_words(Int(mq[1]), Int(mq[2])))
            mi_u = max(mi_u, mi_scratch_words(Int(mq[1]), Int(mq[2])))
    # the grid Gram's per-chunk partials (OP_QDA_COV / the Gram-shaped
    # OP_MATMUL above), sized over the program; 1 word when unused
    var cov_grid = False
    var cov_words = 1
    comptime if (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                 and not is_defined["MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF"]()):
        cov_grid = True
        if cov_grid:
            for s in range(stages):
                var op = Int(host_q.unsafe_load(s * STAGE_INTS))
                var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
                var cq = host_q + (s * STAGE_INTS + 2)
                if op == OP_QDA_COV and Int(cq[2]) > 0:
                    cov_words = max(cov_words, fg_part_words(Int(cq[1]), Int(cq[2])))
                elif op == OP_MATMUL and _matmul_is_gram(cq, total):
                    cov_words = max(cov_words, fg_part_words(Int(cq[8]), Int(cq[7])))
    # the round-robin eigh's scratch and done marks, sized over the program
    var rre_scr = 1
    comptime if RR_EIGH:
        for s in range(stages):
            if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_EIGH:
                var eb = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
                var en = Int(host_q.unsafe_load(s * STAGE_INTS + 3))
                if eb > 0 and en > 0:
                    rre_scr = max(rre_scr, rre_words(en, eb))
    # lane af-ptimpute: the row-tiled folds' per-(chunk, column) partials, sized over the program
    var ptw = 1
    comptime if PT_COLBATCH or PT_FUSED_TRANSFORM or SI_ONEPASS:
        ptw = ptimpute_part_words(host_q, stages)
    var ctx = x_prep_ctx()
    var dpt = ctx.enqueue_create_buffer[DType.float32](ptw)
    var dcg = ctx.enqueue_create_buffer[DType.float32](cov_words)
    var dre = ctx.enqueue_create_buffer[DType.float32](rre_scr)
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
        comptime if MI_CLF_RANKMAJOR:
            if mi_sorted and mi_ties and op == OP_MI_CD:
                var hq = host_q + (s * STAGE_INTS + 2)
                mi_cd_device_rank(ctx, df, dmw, dmu, dq, s * STAGE_INTS + 2, total, Int(hq[1]), Int(hq[2]),
                                  Int(hq[0]), Int(hq[7]))
                continue
        comptime if MI_REG_SORTED:
            if op == OP_MI_CC:
                # q = [Z, n, d, Y, k, TERM, ZS, YS]
                var hq = host_q + (s * STAGE_INTS + 2)
                mi_cc_device[MI_REG_TIES, MI_REG_RANKMAJOR](ctx, df, dq, s * STAGE_INTS + 2, total, Int(hq[1]),
                                                            Int(hq[2]), Int(hq[0]), Int(hq[6]), Int(hq[3]),
                                                            Int(hq[7]))
                continue
        comptime if MI_FAST_FOLDS:
            if op == OP_MI_COLSCALE:
                ctx.enqueue_function[mi_colscale_fast_kernel](df.unsafe_ptr(), qp, grid_dim=total, block_dim=TGF)
                continue
            if op == OP_MI_REDUCE:
                ctx.enqueue_function[mi_reduce_fast_kernel](df.unsafe_ptr(), qp, grid_dim=total, block_dim=TGF)
                continue
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
        comptime if RR_EIGH:
            if op == OP_EIGH:
                # q = [A, m, astride, EVAL, EVEC], one unit a matrix
                var hq = host_q + (s * STAGE_INTS + 2)
                var pf = FP(unsafe_from_address=Int(df.unsafe_ptr()))
                var pr = FP(unsafe_from_address=Int(dre.unsafe_ptr()))
                rr_eigh_into(ctx, pf, Int(hq[0]), Int(hq[1]), Int(hq[2]), total, Int(hq[3]), Int(hq[4]), pr)
                continue
        comptime if PAR_STAGES:
            if op == OP_LDA_STAGE2 or op == OP_LDA_STAGE3 or op == OP_QDA_PREP:
                var hq = host_q + (s * STAGE_INTS + 2)
                var pf = df.unsafe_ptr()
                if op == OP_LDA_STAGE2:
                    # q = [E1, V1, STD, MEAN, XBAR, PRIORS, K, d, n, META, SCAL1, G2, MS]
                    var kk = Int(hq[6])
                    var dd = Int(hq[7])
                    ctx.enqueue_function[lda2_rank_kernel](pf, qp, grid_dim=1, block_dim=1)
                    ctx.enqueue_function[lda2_scal1_kernel](pf, qp, grid_dim=_da_blocks(dd * dd), block_dim=DA_TPB)
                    ctx.enqueue_function[lda2_ms_kernel](pf, qp, grid_dim=_da_blocks(kk * dd), block_dim=DA_TPB)
                    ctx.enqueue_function[lda2_g2_kernel](pf, qp, grid_dim=_da_blocks(dd * dd), block_dim=DA_TPB)
                elif op == OP_LDA_STAGE3:
                    # q = [E2, V2, SCAL1, MEAN, XBAR, PRIORS, K, d, META, SCAL, COEF, INTER, EVR, TMP]
                    var kk = Int(hq[6])
                    var dd = Int(hq[7])
                    ctx.enqueue_function[lda3_rank_kernel](pf, qp, grid_dim=1, block_dim=1)
                    ctx.enqueue_function[lda3_scal_kernel](pf, qp, grid_dim=_da_blocks(dd * dd), block_dim=DA_TPB)
                    ctx.enqueue_function[lda3_tmp_kernel](pf, qp, grid_dim=_da_blocks(kk * dd), block_dim=DA_TPB)
                    ctx.enqueue_function[lda3_inter_kernel](pf, qp, grid_dim=_da_blocks(kk), block_dim=DA_TPB)
                    ctx.enqueue_function[lda3_coef_kernel](pf, qp, grid_dim=_da_blocks(kk * dd), block_dim=DA_TPB)
                    ctx.enqueue_function[lda3_dot_kernel](pf, qp, grid_dim=_da_blocks(kk), block_dim=DA_TPB)
                else:
                    # q = [EVAL, EVEC, K, d, REG, CNT, n, R, LOGC, S2OUT, GIVEN, PIN]
                    var kk = Int(hq[2])
                    var dd = Int(hq[3])
                    ctx.enqueue_function[qda_prep_scal_kernel](pf, qp, grid_dim=_da_blocks(kk), block_dim=DA_TPB)
                    ctx.enqueue_function[qda_prep_rot_kernel](pf, qp, grid_dim=_da_blocks(kk * dd * dd),
                                                              block_dim=DA_TPB)
                continue
        comptime if DEC_TILE:
            if op == OP_QDA_DEC:
                # q = [X, n, d, MEAN, R, LOGC, K, OUT]
                var hq = host_q + (s * STAGE_INTS + 2)
                var nn = Int(hq[1])
                var kk = Int(hq[6])
                if nn > 0 and kk > 0 and Int(hq[2]) > 0:
                    ctx.enqueue_function[qda_dec_tile_kernel](df.unsafe_ptr(), qp, grid_dim=((nn + DT - 1) // DT) * kk,
                                                              block_dim=DT_TPB)
                    continue
        comptime if PT_COLBATCH:
            # the (pt_map, pt_fold) pair as one tiled evaluation: pt_map is skipped (nothing reads T or
            # LG; the Python layer shrinks both to a word by `x_prep_ptimpute_flags`)
            if op == OP_PT_MAP:
                continue
            if op == OP_PT_FOLD:
                pt_colbatch_fold(ctx, FP(unsafe_from_address=Int(df.unsafe_ptr())),
                                 FP(unsafe_from_address=Int(dpt.unsafe_ptr())), host_q + (s * STAGE_INTS + 2),
                                 IP(unsafe_from_address=Int(dq.unsafe_ptr())) + (s * STAGE_INTS + 2))
                continue
        comptime if PT_SPEC:
            # the speculated round's (pt_smap, pt_sfold) likewise; pt_spts and pt_sres stay units
            if op == OP_PT_SMAP:
                continue
            if op == OP_PT_SFOLD:
                pt_spec_fold(ctx, FP(unsafe_from_address=Int(df.unsafe_ptr())),
                             FP(unsafe_from_address=Int(dpt.unsafe_ptr())), host_q + (s * STAGE_INTS + 2),
                             IP(unsafe_from_address=Int(dq.unsafe_ptr())) + (s * STAGE_INTS + 2))
                continue
        comptime if PT_FUSED_TRANSFORM:
            # the standardize tail: pt_apply whose output only feeds the next col_stats runs as the
            # tiled stats of the transform (no TX block), and that col_stats stage is skipped
            if op == OP_PT_APPLY and fused_tail_pair(host_q, s, stages):
                var hq = host_q + (s * STAGE_INTS + 2)
                var hq2 = host_q + ((s + 1) * STAGE_INTS + 2)
                cs_tile_stats(ctx, FP(unsafe_from_address=Int(df.unsafe_ptr())),
                              FP(unsafe_from_address=Int(dpt.unsafe_ptr())), Int(hq[0]), Int(hq[1]), Int(hq[2]),
                              Int(hq[3]), Int(hq[4]), Int(hq2[3]))
                continue
            if op == OP_COL_STATS and s > 0 and fused_tail_pair(host_q, s - 1, stages):
                continue
        comptime if SI_ONEPASS:
            # every col_stats as the one-pass tiled fold (the imputer's statistics, the power
            # transformer's opening col_stats)
            if op == OP_COL_STATS:
                var hq = host_q + (s * STAGE_INTS + 2)
                cs_tile_stats(ctx, FP(unsafe_from_address=Int(df.unsafe_ptr())),
                              FP(unsafe_from_address=Int(dpt.unsafe_ptr())), Int(hq[0]), Int(hq[1]), Int(hq[2]),
                              -1, 0, Int(hq[3]))
                continue
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            if cov_grid and op == OP_QDA_COV:
                # q = [X, n, d, Y, MEAN, CNT, COV]: class k's covariance
                # sum_{y_i = k} (x_i - MEAN_k)(x_i - MEAN_k)' / CNT[k], the
                # grid Gram with the label mask, one launch pair a class
                var hq = host_q + (s * STAGE_INTS + 2)
                var pf = FP(unsafe_from_address=Int(df.unsafe_ptr()))
                var pc = FP(unsafe_from_address=Int(dcg.unsafe_ptr()))
                var xo = Int(hq[0])
                var nn = Int(hq[1])
                var dd = Int(hq[2])
                var yo = Int(hq[3])
                var mo = Int(hq[4])
                var co = Int(hq[5])
                var vo = Int(hq[6])
                if dd > 0 and nn > 0:
                    for k in range(total // (dd * dd)):
                        fast_sym_gram_into(ctx, pf + xo, 0, nn, dd, pf + (mo + k * dd), True, pf + yo, k, pc,
                                           pf + (vo + k * dd * dd), pf + (co + k), True)
                    continue
            if cov_grid and op == OP_MATMUL:
                var hq = host_q + (s * STAGE_INTS + 2)
                if _matmul_is_gram(hq, total):
                    # C = A'A over the K rows of A (LDA's Z'Z), uncentered, no mask
                    var pf = FP(unsafe_from_address=Int(df.unsafe_ptr()))
                    var pc = FP(unsafe_from_address=Int(dcg.unsafe_ptr()))
                    fast_sym_gram_into(ctx, pf + Int(hq[0]), 0, Int(hq[8]), Int(hq[7]), pf, False, pf, -1, pc,
                                       pf + Int(hq[6]), pf, False)
                    continue
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
    _ = dcg^
    _ = dre^
    _ = dpt^
    _ = dq^
    _ = df^
    _ = ctx^
    if prof:
        print("XPPHASE release us", (perf_counter_ns() - t_last) // 1000)
