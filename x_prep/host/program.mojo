"""The prep lane's host runner: the same program as x_prep/device.mojo, on
the caller's arena, in place. No accelerator import, so the CPU-only host
binding compiles it.

THE THREAD SPLIT (lane prep-cpu, 2026-09-28). The device launches one thread
per unit of a stage, all at once, and joins before the next stage; a unit
therefore reads only what earlier stages wrote and writes only its own
outputs (x_prep/common.mojo: every loop INSIDE a unit is its reduction
order). The host runs the same contract: each stage's units are cut into
contiguous ranges of `t`, one task per range, run by `host_parallelize` in
the calling thread's floating-point environment (DEVIATION 5900), joined
before the next stage. Which thread runs a unit never changes what the unit
computes, so the bits are the serial walk's at every task count
(MOJOLEARN_CPU_THREADS = 1, 3 or the default; `core/host_predict_threads.mojo`
holds the one count policy). THIS IS NOT A NUMERIC ROW."""
from experiments.classical_identical_ideas.shared_controls import C08_GROUPED_OUTPUT
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from core.host_parallel import host_parallelize
from core.host_predict_threads import host_grain_cap, host_predict_chunk, host_predict_task_count
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.py2mojo import P2M_BASE, P2M_N, is_p2m_op, run_p2m_unit
from x_prep.fam2 import F2_BASE, F2_N, is_f2_op, run_f2_unit
from x_prep.host.sort import sort_cols_host_unit
from x_prep.host.power import pt_fit_host_unit
from x_prep.host.rr_eigh_host import IDN_RR_EIGH, eigh_rr_host_unit
from x_prep.kbins import kbins_edges
from x_prep.host.target import te_enc_host_groups, te_enc_host_group
from x_prep.host.mutual_info import mi_cc_host_stage, mi_cd_host_stage, mi_dc_host_stage
from x_prep.host.dense import (
    matmul_host_groups, matmul_host_row, class_stats_host_groups, class_stats_host_col,
    qda_cov_host_groups, qda_cov_host_row, qda_dec_host_unit,
)

#: Ops of x_prep/units.mojo the host runs through its own spelling of the
#: SAME words (each file says why): 0 `sort_cols` (x_prep/host/sort.mojo),
#: 44 `pt_fit` (x_prep/host/power.mojo), 42 `qda_dec` (x_prep/host/dense.mojo),
#: 25 `kbins_edges` (its kmeans update, x_prep/kbins.mojo `kbins_edges[True]`);
#: ops whose units the host runs GROUPED, one task item per group of units:
#: 21 `te_enc` (x_prep/host/target.mojo, one group per fold, feature and
#: target column), 13 `matmul` (per output row), 16 `class_stats` (per
#: column), 40 `qda_cov` (per covariance row) (x_prep/host/dense.mojo); a
#: stage whose shape does not group (0 groups) runs its units;
#: and ops the host runs as a WHOLE STAGE (an index per column, then the
#: points across the pool): 68 `mi_cc`, 69 `mi_cd`, 94 `mi_dc`
#: (x_prep/host/mutual_info.mojo).
comptime OP_SORT_COLS = 0
comptime OP_MATMUL = 13
comptime OP_CLASS_STATS = 16
comptime OP_TE_ENC = 21
comptime OP_EIGH = 18
comptime OP_KBINS_EDGES = 25
comptime OP_QDA_COV = 40
comptime OP_QDA_DEC = 42
comptime OP_PT_FIT = 44
comptime OP_MI_CC = 68
comptime OP_MI_CD = 69
comptime OP_MI_DC = 94


@always_inline
def _host_unit[K: Int](t: Int, f: FP, q: IP):
    comptime if K == OP_SORT_COLS:
        sort_cols_host_unit(t, f, q)
    elif K == OP_PT_FIT:
        pt_fit_host_unit(t, f, q)
    elif K == OP_QDA_DEC:
        qda_dec_host_unit(t, f, q)
    elif K == OP_KBINS_EDGES:
        kbins_edges[True](t, f, q)
    elif K == OP_EIGH and IDN_RR_EIGH:
        # lane fam-prep-metrics: the round-robin eigh's words (x_prep/host/rr_eigh_host.mojo)
        eigh_rr_host_unit(t, f, q)
    elif K >= F2_BASE:
        # lane fam2-prep-metrics (x_prep/fam2.mojo)
        run_f2_unit[K](t, f, q)
    elif K >= P2M_BASE:
        run_p2m_unit[K](t, f, q)
    else:
        run_unit[K](t, f, q)


def run_program_host(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_host_ptr(FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages)


def _groups[K: Int](total: Int, q: IP) -> Int:
    """The groups of a grouped op's stage (0: run its units), else 0."""
    comptime if K == OP_TE_ENC:
        return te_enc_host_groups(total, q)
    elif K == OP_MATMUL:
        return matmul_host_groups(total, q)
    elif K == OP_CLASS_STATS:
        return class_stats_host_groups(total, q)
    elif K == OP_QDA_COV:
        return qda_cov_host_groups(total, q)
    else:
        return 0


@always_inline
def _host_group[K: Int](g: Int, total: Int, f: FP, q: IP):
    comptime if K == OP_TE_ENC:
        te_enc_host_group(g, f, q)
    elif K == OP_MATMUL:
        matmul_host_row(g, f, q)
    elif K == OP_CLASS_STATS:
        class_stats_host_col(g, f, q)
    elif K == OP_QDA_COV:
        qda_cov_host_row(g, total, f, q)


@always_inline
def _host_item[K: Int](t: Int, grouped: Bool, total: Int, f: FP, q: IP):
    if grouped:
        _host_group[K](t, total, f, q)
    else:
        _host_unit[K](t, f, q)


#: Lane perf-small-fixes (2026-11-06): the per-op work table of the host
#: split. `_unit_work[K]()` is the scalar element operations ONE unit of op K
#: costs (a load, an arithmetic step and a store each count one), or 0 when
#: the unit carries an inner loop whose length lives in its stage words (a
#: column reduction, a sort, a row softmax, a grouped stage): 0 means
#: unknown, and `host_grain_cap` then leaves the task count as it was.
#: Listed are only the ops whose unit is ONE output element with no loop
#: inside it (each body read in x_prep/prims.mojo, transform.mojo,
#: iterative.mojo, kbins.mojo, naive_bayes/nb.mojo): an element op is about
#: eight scalar steps (index arithmetic, two to four loads, a compare or
#: two, one store); a transcendental per element (log, the Yeo-Johnson /
#: Box-Cox power and its inverse) about 32; the quantile maps search their
#: column's quantiles twice by bisection, 2 * log2(nq) steps of about four
#: operations, which stays under 64 for every nq up to 256 and is taken as
#: 64. The weights are cost estimates, never a shape: with
#: HOST_MIN_TASK_WORK = 2^16 an element op runs inline below about 16k
#: elements and gets one task per 8k elements above it, on every shape.
#: Every op here is a stage of independent units (each unit writes only its
#: own outputs, the device contract above), so the task count moves no bit
#: and no fold partition depends on it.
comptime _ELEM_WORK = 8
comptime _TRANSCENDENTAL_WORK = 32
comptime _QUANTILE_WORK = 64


@always_inline
def _unit_work[K: Int]() -> Int:
    # element ops: 3 affine, 10 i2f, 11 f2i, 12 binarize, 19 where_neg,
    # 23 mark_missing, 24 fill, 51 label_binarize, 60 nan_mask,
    # 61 gather_cols, 77 ord_inverse, 78 cat_gather, 79 where_code,
    # 80 kbins_inverse, 87 indicator, 89 remap_codes, 90 add_arrays,
    # 102 std_scale, 103 nan_keep, 140 hcat, 141 colblock.
    comptime if (
        K == 3 or K == 10 or K == 11 or K == 12 or K == 19 or K == 23
        or K == 24 or K == 51 or K == 60 or K == 61 or K == 77 or K == 78
        or K == 79 or K == 80 or K == 87 or K == 89 or K == 90 or K == 102
        or K == 103 or K == 140 or K == 141
    ):
        return _ELEM_WORK
    # per-element transcendental: 45 pt_apply, 72 log, 75 pt_inverse.
    elif K == 45 or K == 72 or K == 75:
        return _TRANSCENDENTAL_WORK
    # per-element bisection: 43 qt_apply, 74 qt_inverse.
    elif K == 43 or K == 74:
        return _QUANTILE_WORK
    else:
        return 0


def _run_stage[K: Int](total: Int, f: FP, q: IP):
    """Items [0, items) of one stage: serially on the calling thread when one
    task covers them, else contiguous ranges across the pool."""
    comptime if K == OP_MI_CC:
        mi_cc_host_stage(total, f, q)
        return
    elif K == OP_MI_CD:
        mi_cd_host_stage(total, f, q)
        return
    elif K == OP_MI_DC:
        mi_dc_host_stage(total, f, q)
        return
    var groups = _groups[K](total, q)
    var grouped = groups > 0
    var items = groups if grouped else total
    var tasks = host_predict_task_count(items)
    comptime if _unit_work[K]() > 0:
        # Row-independent element stage: small calls run on the calling
        # thread (the per-op work table above).
        if not grouped:
            tasks = host_grain_cap(tasks, items * _unit_work[K]())
    if tasks <= 1:
        for t in range(items):
            _host_item[K](t, grouped, total, f, q)
        return
    var chunk = host_predict_chunk(items, tasks)

    def body(c: Int) {imm f, imm q, imm chunk, imm items, imm grouped, imm total}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > items:
            hi = items
        for t in range(lo, hi):
            _host_item[K](t, grouped, total, f, q)

    host_parallelize(body, tasks)


def run_program_host_ptr(f: FP, arena_len: Int, qbase: IP, stages: Int) raises:
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        if (op < 0 or op >= N_OPS) and not is_p2m_op(op) and not is_f2_op(op):
            raise Error(String("x_prep: unknown op ", op))
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        var total = Int(qbase.unsafe_load(s * STAGE_INTS + 1))
        comptime if C08_GROUPED_OUTPUT:
            if op == 9:
                total = (total+3)//4
        var q = qbase + (s * STAGE_INTS + 2)
        comptime for k in range(N_OPS):
            if op == k:
                _run_stage[k](total, f, q)
        comptime for k in range(P2M_BASE, P2M_BASE + P2M_N):
            if op == k:
                _run_stage[k](total, f, q)
        comptime for k in range(F2_BASE, F2_BASE + F2_N):
            if op == k:
                _run_stage[k](total, f, q)
