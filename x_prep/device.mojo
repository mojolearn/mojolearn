"""The prep lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from experiments.classical_identical_ideas.shared_controls import C07_RADIX_ROWS, C08_GROUPED_OUTPUT, C08_DICTIONARY, C08_UNIQUE_SCAN
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.sys import has_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.py2mojo import P2M_BASE, P2M_N, is_p2m_op, run_p2m_unit
#: lane fam2-prep-metrics: ops F2_BASE .. (x_prep/fam2.mojo), IDENTICAL only
from x_prep.fam2 import F2_BASE, F2_N, is_f2_op, run_f2_unit
from x_prep.dsort import sort_cols_device, sort_scratch_words
from x_prep.ddict import dict_inverse_device, dict_inverse_scratch_words, unique_runs_device, unique_runs_scratch_words
from x_prep.dradix import RADIX_SORT, IDN_XPREP_RADIX, RADIX_MIN_ROWS, radix_sort_cols_device, radix_scratch_words
from x_prep.fastred import (
    TGR, col_stats_fast_kernel, pt_fold_fast_kernel, class_stats_fast_kernel, ii_mean_fast_kernel,
    ii_gram_fast_kernel,
)
from x_prep.dmi import mi_cd_device, mi_w_words, mi_scratch_words
from x_prep.fastnb import NB_CAT_ATOMIC, AFCL_P03, cat_hist_atomic_kernel, cat_hist_convert_kernel
from x_prep.label_fast import LABEL_SCATTER, label_scatter_kernel
from x_prep.select_fast import (
    SELECT_FREG, SELECT_FCLS, OP_F_CLASSIF, OP_F_REGRESSION, program_has_op, select_scratch_words,
    select_freg_device, select_fcls_device, select_cstats_device,
)
from x_prep.fastprep2 import PREP2_FAST, Prep2Switches, prep2_scratch_words, prep2_fast_stage
#: lane ml-prep-nb: te_global / te_enc / ii_gram in the lane-tree order on a threadgroup (IDENTICAL)
from x_prep.idn_tree import IDN_TREE_ANY, idn_tree_scratch_words, idn_tree_stage
#: lane classical-te-gmm (2026-10-07): te_global / te_enc in the blocked order BLT (IDENTICAL A/B,
#: -D MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD, default off; x_prep/idn_blocked.mojo)
from x_prep.idn_blocked import IDN_TE_BLOCKED, te_blocked_scratch_words, te_blocked_stage

#: lane af-ptimpute (2026-10-03), FAST + Apple + define only (x_prep/fastpt.mojo): the import
#: instantiates nothing; every launch below sits inside `comptime if PT_* / SI_*`
from x_prep.fastpt import (
    PT_COLBATCH, PT_SPEC, PT_FUSED_TRANSFORM, SI_ONEPASS, OP_PT_MAP, OP_PT_SMAP, OP_PT_SFOLD, OP_PT_APPLY,
    pt_colbatch_fold, pt_spec_fold, cs_tile_stats, ptimpute_part_words, fused_tail_pair,
)
from x_prep.dmi_fast import mi_cc_device, mi_cd_device_rank, mi_colscale_fast_kernel, mi_reduce_fast_kernel, TGF
from core.arena_io import check_in_ranges, check_out_ranges, upload_ranges, download_ranges
from core.staged_download import download_f32_into
from core.device_pool import pool_take, pool_give

#: lane/apple-fast-gap-manprep: on FAST + Apple the program's output region
#: (and arena ranges of 1M+ words) download through core/staged_download.mojo.
#: Default since the M3 A/Bs gmp-staged-* (output digests the same):
#: label-binarizer taxi 563 -> 345 ms, multilabel-binarizer taxi 287 -> 193 ms,
#: target-encoder taxi 281 -> 272 ms. -D MOJOLEARN_X_PREP_FAST_STAGED_OUT_OFF:
#: the raw host-pointer copies.
comptime X_PREP_STAGED_OUT = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                              and not is_defined["MOJOLEARN_X_PREP_FAST_STAGED_OUT_OFF"]())
comptime _XP_STAGE_POOL = "MojoXPrepDownloadStagesFast"

#: lane/apple-fast-w2-prep (2026-10-04), DEFAULT in FAST + Apple; rollback
#: -D MOJOLEARN_X_PREP_POOL_ARENA_OFF (IDENTICAL and other vendors compile
#: none of it). M3, one run per arm: label-binarizer taxi 319.9 -> 270.4 ms,
#: multilabel-binarizer taxi 175.6 -> 167.5 ms, istella 129.3 -> 125.7 ms;
#: w2-pool-quality PASS (25 output arrays sha256-identical A vs B, dirty-buffer
#: repeats). Cost: up to POOL_KEEP_BYTES (2 GB) of idle device memory stays
#: held by the pool between programs. The
#: program's device buffer `df` (arena + scratch + output region) comes from
#: core/device_pool.mojo instead of a fresh allocation when it is at least
#: _XP_POOL_MIN_WORDS long, and goes back to the pool after the final wait.
#: Cause: the M3 profile of label-binarizer taxi (prep-apple3 request
#: 1790627886703, XPPHASE) spent 60 ms in the "upload" phase, which for that
#: program is the memset of a FRESH 972 MB output region (first touch of new
#: device pages, ~16 GB/s); the same memset on resident pages is bandwidth
#: bound (~2 ms). The board times one round after a warm-up of the same
#: shape, so the exact-size pool hits. A pooled buffer's words are whatever
#: the last program left, so every region is defined before any stage reads
#: it: upload_ranges zeroes every arena word outside the inputs (the
#: non-ranges path copies the whole arena), the output region keeps its
#: memset, and the scratch region is cleared here (a fresh buffer's scratch
#: was zero pages; the contract says a stage writes scratch before reading
#: it, and the clear keeps a pooled run word-identical even if one does not).
#: Copies and clears only: no bit moves. Idle pooled bytes are capped by
#: core/device_pool.mojo POOL_KEEP_BYTES (2 GB).
#: G2 (lane fg-knn-nb, 2026-10-09, arena half; the slot half is
#: core/device_store.mojo STORE_SLOT_POOL): the same pooled `df` outside
#: FAST, on every GPU vendor. Cost reasoning: an x_prep fit allocates and
#: frees its device arena (the whole X range plus outputs and scratch, ~880
#: MB for a 1M x 220 float32 X) once per call; repeated fits of one shape
#: (cross-validation, the board's rounds) paid a device allocation, a device
#: free and first-touch page mapping of it every call. The region contract
#: above makes a pooled buffer word-identical (inputs uploaded, every other
#: arena word zeroed by upload_ranges or copied whole, output memset, scratch
#: cleared), and the idle cap is POOL_KEEP_BYTES (2 GB). DEFAULT ON (storage
#: only, no bit moves); `-D MOJOLEARN_XPREP_ARENA_POOL_OFF=1` restores the
#: fresh allocation (and the store's free-at-once slots).
comptime X_PREP_POOL_ARENA_IDN = (GLOBAL_NUMERIC_MODE != NUMERIC_FAST and has_accelerator()
                                 and not is_defined["MOJOLEARN_XPREP_ARENA_POOL_OFF"]())
comptime X_PREP_POOL_ARENA = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                             and not is_defined["MOJOLEARN_X_PREP_POOL_ARENA_OFF"]()) or X_PREP_POOL_ARENA_IDN
comptime _XP_DF_POOL = "MojoXPrepArenaPoolFast" if GLOBAL_NUMERIC_MODE == NUMERIC_FAST else "MojoXPrepArenaPoolIdentical"
#: 2^24 words = 64 MB: smaller programs keep the fresh allocation
comptime _XP_POOL_MIN_WORDS = 1 << 24


def _download_ranges_staged(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], host_addr: Int,
                            outs_addr: Int, nouts: Int) raises:
    """core/arena_io.mojo `download_ranges` with every range of at least
    1M words through the pinned-stage pipeline (synchronizes)."""
    if nouts <= 0:
        return
    var hf = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=host_addr)
    var o = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=outs_addr)
    var bounded = False
    for k in range(nouts):
        var cn = Int(o.unsafe_load(4 * k + 2))
        if cn >= 0:
            bounded = True
            ctx.enqueue_copy(dst_ptr=hf + cn, src_buf=df.create_sub_buffer[DType.float32](cn, 1))
    if bounded:
        ctx.synchronize()
    for k in range(nouts):
        var lo = Int(o.unsafe_load(4 * k))
        var hi = Int(o.unsafe_load(4 * k + 1))
        var cn = Int(o.unsafe_load(4 * k + 2))
        if cn >= 0:
            var c = Int(bitcast[DType.int32](hf.unsafe_load(cn))) * Int(o.unsafe_load(4 * k + 3))
            hi = lo + max(0, min(hi - lo, c))
        if hi - lo >= (1 << 20):
            ctx.synchronize()
            var view = df.create_sub_buffer[DType.float32](lo, hi - lo)
            download_f32_into[_XP_STAGE_POOL](ctx, view, hi - lo, hf + lo)
            _ = view^
        elif hi > lo:
            ctx.enqueue_copy(dst_ptr=hf + lo, src_buf=df.create_sub_buffer[DType.float32](lo, hi - lo))
from core.device_store import DeviceStore
from x_linear.fast_gram import fast_sym_gram_into, fg_part_words
from x_prep.rr_eigh import rr_eigh_into, rre_words, rr_eigh_clear
#: lane fam-prep-metrics: IDENTICAL takes the round-robin eigh on every vendor (the host column runs
#: the same rounds, x_prep/host/rr_eigh_host.mojo; -D MOJOLEARN_IDN_RR_EIGH_OFF: `eigh_unit`)
from x_prep.host.rr_eigh_host import IDN_RR_EIGH, eigh_rr_takes
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
#: eigh q[5] != 0 keeps that stage on `eigh_unit`'s cyclic path under RR_EIGH.
#: IterativeImputer sets it (python/mojolearn/_expansion_prep.py, one small
#: eigh per feature per round): M3, iterative-imputer istella, main b2dd5dfe5,
#: RR_EIGH off 4,593 ms vs on 8,821 ms, masked_rmse unchanged. LDA / QDA leave
#: it 0 and keep the round-robin path.
comptime EIGH_CYCLIC_Q = 5

#: lane/apple-fast-mi (2026-10-03): FAST + Apple only, default OFF, one `-D MOJOLEARN_MI_<NAME>`
#: each (docs/apple-fast/ab/mi.md). x_prep/dmi_fast.mojo:
#: REG_SORTCOUNT: op 68 (`mi_cc`, mutual_info_regression) as the sorted Kraskov search (the
#: host's argument, x_prep/host/mutual_info.mojo `_cc_column`) instead of the brute-force
#: unit; REG_TIES: its tie-aware form (implies SORTCOUNT); REG_RANKMAJOR: its point kernel
#: one thread per (column, sorted rank) (implies SORTCOUNT); FAST_FOLDS: ops 66 / 70
#: (`mi_colscale`, `mi_reduce`) as threadgroup folds; CLF_RANKMAJOR: op 69's point kernel
#: one thread per (column, sorted rank). Every launch is inside these guards; IDENTICAL and
#: the other vendors compile main's code unchanged.
comptime _MI_FA = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
# TOMBSTONE: MOJOLEARN_MI_ALL (DROP-quality: M3 miv-reg-all-* reg istella 46,430 -> 1,151 ms, taxi 2,316 -> 81, but
# it includes FAST_FOLDS, which fails the selected-set gate) deleted 2026-10-09 on lane/owed-deletions-D2: the
# bundle alias only, each member keeps its own define; code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_MI_ALL.patch
#: MI_REG_TIES: FAST + Apple DEFAULT since lane/apple-fast-miv (2026-10-03). M3 A/B vs main:
#: select-mutual-info-reg istella 46,465 -> 1,340 ms, taxi 2,339 -> 160 ms (n_selected 110 / 5
#: both arms; main's istella arm swings 46-70 s run to run, far inside the gap). Quality
#: (tools/miv_quality.sh, M2): scores bit-identical to main's FAST route on a tie-heavy fixture.
#: -D MOJOLEARN_MI_REG_TIES_OFF: main's search.
comptime MI_REG_TIES = _MI_FA and (is_defined["MOJOLEARN_MI_REG_TIES"]()
                                   or not is_defined["MOJOLEARN_MI_REG_TIES_OFF"]())
comptime MI_REG_RANKMAJOR = _MI_FA and is_defined["MOJOLEARN_MI_REG_RANKMAJOR"]()
comptime MI_REG_SORTED = _MI_FA and (MI_REG_TIES or MI_REG_RANKMAJOR or is_defined["MOJOLEARN_MI_REG_SORTCOUNT"]())
#: MI_FAST_FOLDS stays opt-in (lane/apple-fast-miv): its scores move up to 3% of the largest
#: score and the selected set changes on the tools/miv_quality.py fixture.
# DROP-quality / INCONCLUSIVE-speed: mi-reg-folds-istella-x was -0.2%
# on the old base; M2 quality showed 3.2% score shift and selected-set
# symmetric difference 2. See docs/apple-fast/EXPERIMENTS.md.
comptime MI_FAST_FOLDS = _MI_FA and is_defined["MOJOLEARN_MI_FAST_FOLDS"]()
#: MI_CLF_RANKMAJOR: FAST + Apple DEFAULT since lane/apple-fast-miv (2026-10-03). M3 A/B vs main:
#: select-mutual-info istella 715.2 -> 645.2 ms, taxi 202.2 -> 196.4 ms (n_selected identical);
#: scores bit-identical (tools/miv_quality.sh, M2). -D MOJOLEARN_MI_CLF_RANKMAJOR_OFF: main's layout.
comptime MI_CLF_RANKMAJOR = _MI_FA and (is_defined["MOJOLEARN_MI_CLF_RANKMAJOR"]()
                                        or not is_defined["MOJOLEARN_MI_CLF_RANKMAJOR_OFF"]())
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
#: x_prep/blocked.mojo's CategoricalNB histogram stages (x_prep/fastnb.mojo intercepts them)
comptime OP_CAT_HPART = 133
comptime OP_CAT_HFOLD = 134

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


#: x_prep ops with a device form in x_prep/ddict.mojo
comptime OP_UNIQUE_COLS = 5
comptime OP_UNIQUE_INVERSE = 177


def _sort_feeds_dict(hq: IP, s: Int) -> Bool:
    """Stage s is sort_cols q = [X, n, d, S, cn] and stage s+1 is the
    unique_inverse q = [S, n, d, U, CNT, X, CODES] reading that S, over the
    same X, n, d and columns, canonical: the device form never reads S."""
    if Int(hq.unsafe_load((s + 1) * STAGE_INTS)) != OP_UNIQUE_INVERSE:
        return False
    var a = hq + (s * STAGE_INTS + 2)
    var b = hq + ((s + 1) * STAGE_INTS + 2)
    return (Int(hq.unsafe_load(s * STAGE_INTS + 1)) == Int(hq.unsafe_load((s + 1) * STAGE_INTS + 1))
            and Int(a[3]) == Int(b[0]) and Int(a[0]) == Int(b[5]) and Int(a[1]) == Int(b[1])
            and Int(a[2]) == Int(b[2]) and Int(a[4]) != 0)


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


def p2m_kernel[OP: Int](f: FP, q: IP, total: Int32):
    """Lane apple-fast-py2mojo-prep: ops P2M_BASE .. (x_prep/py2mojo.mojo)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        run_p2m_unit[OP](t, f, q)


def f2_kernel[OP: Int](f: FP, q: IP, total: Int32):
    """Lane fam2-prep-metrics: ops F2_BASE .. (x_prep/fam2.mojo)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        run_f2_unit[OP](t, f, q)


def run_program_device(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int, scratch_len: Int = 0,
                       out_addr: Int = 0, out_len: Int = 0) raises:
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages, scratch_len,
        out_addr, out_len,
    )


#: TOMBSTONE: G3 (lane fg-knn-nb, MOJOLEARN_XPREP_NO_SLOT_HOP: large direct
#: inputs as host spans straight into the arena, `run_program_device_ranges_host`,
#: core/arena_io.mojo `upload_ranges_host`) deleted 2026-10-09 (lane
#: postmerge-act-2): slower on the average, gaussian-nb istella NV 1.25x / AMD
#: 1.00x, taxi NV 1.02x / AMD 1.03x (nv n0668->n0630, amd a1066->a1090),
#: accuracy SAME, same digests; refused in core/six_lane_experiment_guards.mojo;
#: code recoverable at main 5c137b55e.

#: TOMBSTONE: G5 (lane fg-knn-nb, MOJOLEARN_XPREP_DEVICE_CODES: GaussianNB.fit
#: label codes from the device unique_inverse) deleted 2026-10-09 (lane
#: postmerge-act-2): SLOWER, gaussian-nb istella NV 1.29x / AMD 1.12x, taxi NV
#: 1.70x / AMD 1.35x (nv n0668->n0631, amd a1066->a1091), accuracy SAME, same
#: digests; refused in core/six_lane_experiment_guards.mojo; main 5c137b55e.


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
    for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        if (op < 0 or op >= N_OPS) and not is_p2m_op(op) and not is_f2_op(op):
            raise Error(String("x_prep: unknown op ", op))
    # FAST on Apple (lane prep-apple3): sort_cols by radix (x_prep/dradix.mojo, the same words).
    # Default since request 1790627886703 (M3 Ultra, 16 columns x 1M rows: RobustScaler 0.141 ->
    # 0.070 s, every digest equal); MOJOLEARN_XPREP_SORT_RADIX=0 is the bitonic sort.
    # MOJOLEARN_XPREP_SORT_CHUNK = positions per chunk (512 to 4096 measured within 0.006 s).
    var radix = False
    # C07 (IDENTICAL int sweep MOJOLEARN_CLASSICAL_C07_RADIX_ROWS = 1024|2048|4096, default
    # 2048): keys one (column, chunk) task walks, independent of the digit width; same words.
    var radix_rows = C07_RADIX_ROWS
    comptime if IDN_XPREP_RADIX:
        # K1: IDENTICAL takes the radix sort by define, not by env (the same words either way)
        radix = True
    elif RADIX_SORT:
        radix = getenv("MOJOLEARN_XPREP_SORT_RADIX", "1") != "0"
        radix_rows = max(1, _env_int("MOJOLEARN_XPREP_SORT_CHUNK", 2048))
    # MOJOLEARN_XPREP_PROFILE=1: XPPHASE lines (a wait after every phase; timing only)
    var prof = getenv("MOJOLEARN_XPREP_PROFILE", "0") == "1"
    var t_last = perf_counter_ns()
    var scratch = 1
    for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_SORT_COLS:
            var sq = host_q + (s * STAGE_INTS + 2)
            var units = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
            scratch = max(scratch, sort_scratch_words(Int(sq[1]), units))
            if radix and Int(sq[1]) >= RADIX_MIN_ROWS:
                scratch = max(scratch, radix_scratch_words(Int(sq[1]), units, radix_rows))
    comptime if C08_DICTIONARY or C08_UNIQUE_SCAN:
        # x_prep/ddict.mojo: the device unique_inverse / unique_cols scratch
        for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
            var dop = Int(host_q.unsafe_load(s * STAGE_INTS))
            var dq0 = host_q + (s * STAGE_INTS + 2)
            var dunits = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
            comptime if C08_DICTIONARY:
                if dop == OP_UNIQUE_INVERSE:
                    scratch = max(scratch, dict_inverse_scratch_words(Int(dq0[1]), dunits, radix_rows))
            comptime if C08_UNIQUE_SCAN:
                if dop == OP_UNIQUE_COLS:
                    scratch = max(scratch, unique_runs_scratch_words(Int(dq0[1]), dunits))
    comptime if SELECT_FREG or SELECT_FCLS:
        # FAST on Apple (lane/apple-fast-select, -D MOJOLEARN_SELECT_FREG / _FCLS): the
        # f_regression / f_classif tiles' partials live in the sort scratch (x_prep/select_fast.mojo)
        var sel_fcls = program_has_op(host_q, stages, OP_F_CLASSIF)
        for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
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
    comptime if IDN_TREE_ANY:
        scratch = max(scratch, idn_tree_scratch_words(host_q, stages))
    comptime if IDN_TE_BLOCKED:
        scratch = max(scratch, te_blocked_scratch_words(host_q, stages))
    var mi_sorted = getenv("MOJOLEARN_XPREP_MI_SORTED", "1") != "0"
    var mi_ties = getenv("MOJOLEARN_XPREP_MI_TIES", "1") != "0"
    var mi_w = 1
    var mi_u = 1
    for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
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
            for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
                var op = Int(host_q.unsafe_load(s * STAGE_INTS))
                var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
                var cq = host_q + (s * STAGE_INTS + 2)
                if op == OP_QDA_COV and Int(cq[2]) > 0:
                    cov_words = max(cov_words, fg_part_words(Int(cq[1]), Int(cq[2])))
                elif op == OP_MATMUL and _matmul_is_gram(cq, total):
                    cov_words = max(cov_words, fg_part_words(Int(cq[8]), Int(cq[7])))
    # the round-robin eigh's scratch and done marks, sized over the program
    var rre_scr = 1
    comptime if RR_EIGH or IDN_RR_EIGH:
        for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
            if (Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_EIGH
                    and Int(host_q.unsafe_load(s * STAGE_INTS + 2 + EIGH_CYCLIC_Q)) == 0):
                var eb = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
                var en = Int(host_q.unsafe_load(s * STAGE_INTS + 3))
                if eb > 0 and en > 0 and eigh_rr_takes(en):
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
    var pooled = False
    comptime if X_PREP_POOL_ARENA:
        pooled = dev_len >= _XP_POOL_MIN_WORDS
    var df: DeviceBuffer[DType.float32]
    if pooled:
        df = pool_take[_XP_DF_POOL](ctx, dev_len)
    else:
        df = ctx.enqueue_create_buffer[DType.float32](dev_len if dev_len > 0 else 1)
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
    if pooled and scratch_len > 0:
        # X_PREP_POOL_ARENA: a pooled buffer's scratch words start zero, as a fresh one's did
        ctx.enqueue_memset(df.create_sub_buffer[DType.float32](arena_len, scratch_len), Float32(0))
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
        comptime if C08_DICTIONARY:
            # C08 routes: unique_inverse as the index radix sort + run scan (x_prep/ddict.mojo),
            # q = [S, n, d, U, CNT, X, CODES]. It reads X, not S, so a sort_cols whose only
            # job is to fill this stage's S (the next stage, same X, canonical) is skipped.
            if op == OP_UNIQUE_INVERSE:
                var hq = host_q + (s * STAGE_INTS + 2)
                dict_inverse_device(ctx, df, dw, total, Int(hq[5]), Int(hq[1]), Int(hq[2]), Int(hq[3]), Int(hq[4]),
                                    Int(hq[6]), radix_rows)
                continue
            if op == OP_SORT_COLS and s + 1 < stages and _sort_feeds_dict(host_q, s):
                continue
        comptime if C08_UNIQUE_SCAN:
            # C08_UNIQUE_SCAN: unique_cols (q = [S, n, d, U, CNT]) as one chunked run scan of every column
            if op == OP_UNIQUE_COLS:
                var hq = host_q + (s * STAGE_INTS + 2)
                unique_runs_device(ctx, df, dw, total, Int(hq[0]), Int(hq[1]), Int(hq[3]), Int(hq[4]))
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
        comptime if LABEL_SCATTER:
            if op == 51:
                var hq = host_q + (s * STAGE_INTS + 2)
                if Int(hq[4]) == 0:
                    # Explicit clear also handles reused arena/output regions.
                    ctx.enqueue_memset(df.create_sub_buffer[DType.float32](Int(hq[7]), total), Float32(0))
                    var rows = Int(hq[1])
                    ctx.enqueue_function[label_scatter_kernel](
                        df.unsafe_ptr(), qp, Int32(rows),
                        grid_dim=(rows + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                    )
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
            if prep2_fast_stage(ctx, df, dw, host_q, s, op, total, IP(unsafe_from_address=Int(qp)), p2):
                continue
        comptime if IDN_TE_BLOCKED:
            if te_blocked_stage(ctx, df, dw, host_q, s, op, total, IP(unsafe_from_address=Int(qp))):
                continue
        comptime if IDN_TREE_ANY:
            if idn_tree_stage(ctx, df, dw, host_q, s, op, total, IP(unsafe_from_address=Int(qp))):
                continue
        comptime if RR_EIGH or IDN_RR_EIGH:
            if (op == OP_EIGH and Int(host_q.unsafe_load(s * STAGE_INTS + 2 + EIGH_CYCLIC_Q)) == 0
                    and eigh_rr_takes(Int(host_q.unsafe_load(s * STAGE_INTS + 3)))):
                # q = [A, m, astride, EVAL, EVEC, cyclic], one unit a matrix
                var hq = host_q + (s * STAGE_INTS + 2)
                var pf = FP(unsafe_from_address=Int(df.unsafe_ptr()))
                var pr = FP(unsafe_from_address=Int(dre.unsafe_ptr()))
                rr_eigh_into(ctx, pf, Int(hq[0]), Int(hq[1]), Int(hq[2]), total, Int(hq[3]), Int(hq[4]), pr)
                comptime if IDN_RR_EIGH:
                    # the destroyed A zeroed, as the host column leaves it
                    rr_eigh_clear(ctx, pf, Int(hq[0]), Int(hq[1]), Int(hq[2]), total)
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
        comptime if NB_CAT_ATOMIC:
            # lane apple-fast-nb: the (row, feature) atomic count table; W < 0 only
            # (the weighted fold keeps the units). The dispatcher zeroes block 0
            # of the histogram scratch, the kernel adds into it, and the fold
            # stage copies it out (x_prep/fastnb.mojo).
            if op == OP_CAT_HPART and host_q.unsafe_load(s * STAGE_INTS + 2 + 7) < 0:
                var hq = host_q + (s * STAGE_INTS + 2)
                var cells = Int(hq[1]) * Int(hq[2])
                var words = Int(hq[2]) * Int(hq[4]) * Int(hq[6])
                if cells > 0 and cells <= 2147483647 and words > 0:
                    ctx.enqueue_memset(df.create_sub_buffer[DType.float32](Int(hq[8]), words), Float32(0))
                    var cat_block = BLOCK // 2 if AFCL_P03 else BLOCK
                    ctx.enqueue_function[cat_hist_atomic_kernel](
                        df.unsafe_ptr(), qp, Int32(cells),
                        grid_dim=(cells + cat_block - 1) // cat_block, block_dim=cat_block,
                    )
                    continue
            if op == OP_CAT_HFOLD and host_q.unsafe_load(s * STAGE_INTS + 2 + 6) < 0:
                ctx.enqueue_function[cat_hist_convert_kernel](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
                continue
        comptime if C08_GROUPED_OUTPUT:
            if op == 9:
                total = (total+3)//4
        comptime for k in range(N_OPS):
            if op == k:
                ctx.enqueue_function[prep_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
        comptime for k in range(P2M_BASE, P2M_BASE + P2M_N):
            if op == k:
                ctx.enqueue_function[p2m_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
        comptime for k in range(F2_BASE, F2_BASE + F2_N):
            if op == k:
                ctx.enqueue_function[f2_kernel[k]](
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
        comptime if X_PREP_STAGED_OUT:
            _download_ranges_staged(ctx, df, Int(host_f), outs_addr, nouts)
        else:
            download_ranges(ctx, df, host_f, outs_addr, nouts)
    elif arena_len > 0:
        if dev_len > arena_len:
            ctx.enqueue_copy(dst_ptr=host_f, src_buf=df.create_sub_buffer[DType.float32](0, arena_len))
        else:
            ctx.enqueue_copy(dst_ptr=host_f, src_buf=df)
    comptime if X_PREP_STAGED_OUT:
        if out_n > 0:
            # lane/apple-fast-gap-manprep (2026-10-03): the output region
            # (LabelBinarizer's 1M x 259 int32 words at the board, 1 GB)
            # through the pooled pinned-stage pipeline instead of one raw
            # host-pointer copy (~21 ms per 64 MB on Apple). Copies only.
            ctx.synchronize()
            var oview = df.create_sub_buffer[DType.float32](out_at, out_n)
            download_f32_into[_XP_STAGE_POOL](ctx, oview, out_n,
                                              MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr))
            _ = oview^
    else:
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
    if pooled:
        # every use of df was waited on by the synchronize above
        pool_give[_XP_DF_POOL](df^)
    else:
        _ = df^
    _ = ctx^
    if prof:
        print("XPPHASE release us", (perf_counter_ns() - t_last) // 1000)
