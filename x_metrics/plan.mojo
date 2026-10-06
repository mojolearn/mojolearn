# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PLANNER (lane/metrics phase C): a caller's program of units in, the
program both runners execute out. Every stage whose unit is one thread
walking all n rows (group_sort, group_sum, col_sort, wpercentile, bin_curve,
permute, col_max) is replaced by the wide stages of x_metrics/par.mojo, which return
the same bits (par.mojo's header says why for each); every other stage is
passed through. The replacement stages address SCRATCH slots past the
caller's arena (`arena_len` onward); the runners allocate arena + scratch,
and copy only the caller's arena back.

The plan is a pure function of the caller's program (never of a device, a
core count or the data), so the host and the device run the same stages.
A stage the planner cannot size (a group_sum whose OFF no earlier
group_sort wrote) runs its sequential unit unchanged.
"""
from experiments.classical_identical_ideas.shared_controls import C05_PHASE_SCRATCH, C09_REG_BUNDLE, C10_RANK_REUSE, C12_SPARSE_COUNTS
from x_metrics.common import IP, STAGE_INTS, PARAMS, LEAF
from x_metrics.par import RUN, KEY_COL, KEY_CURVE, KEY_PERM, KEY_GROUP
from x_metrics.cls_epi import OP_CLS_EPI
from x_metrics.curve_out import (
    OP_CURVE_OUT, OP_CO_KEEP, OP_CO_EMIT, OP_CO_DET, CO_DET, OP_AUC_XY, OP_AX_CHUNK, OP_AX_FINAL, AX_REC,
)
from x_metrics.contingency import (
    OP_CONT_STATS, OP_CT_CELL, OP_CT_ROW, OP_CT_COL, OP_CT_ISUM, OP_CT_PAIRS, OP_CT_ENT, OP_FF_CHUNK, OP_FF_FIN,
    OP_MI_CELL, OP_EMI_CELL, CT_CH, CT_IREC, FF_REC, FF_ENT, FF_MI, FF_SUM,
)

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# AFCL-P09: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified.
# The planner and its emitted chunk parameters use the same selected width.
comptime AFCL_P09 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_AFCL_P09"]()

comptime OP_GROUP_SORT = 0
comptime OP_GROUP_SUM = 1
comptime OP_COL_SORT = 4
comptime OP_WPERCENTILE = 5
comptime OP_BIN_CURVE = 7
comptime OP_PERMUTE = 10
#: ops a caller may name (python/mojolearn/_expansion_metrics.py `_OPS`)
comptime N_USER_OPS = 11
comptime OP_CS_HIST = 11
comptime OP_CS_SCAN_ROWS = 12
comptime OP_CS_SCAN_GROUPS = 13
comptime OP_CS_PLACE = 14
comptime OP_FOLD_LEAF = 15
comptime OP_FOLD_LEVEL = 16
comptime OP_FOLD_FINAL = 17
comptime OP_SORT_KEY = 18
comptime OP_SORT_RUNS = 19
comptime OP_SORT_MERGE = 20
comptime OP_SORT_EMIT = 21
comptime OP_CURVE_GATHER = 22
#: 23 (the curve's sequential prefix) is retired: the weighted curve runs
#: OP_CURVE_CNT/OFF/FILL as a blocked scan (lane cgr2-metrics-shap)
comptime OP_WPCT_GATHER = 24
#: 25 (wpercentile's sequential CDF) is retired: OP_WPCT_CSUM/COFF/CFILL
comptime OP_WPCT_SELECT = 26
comptime OP_CURVE_EMIT = 27
comptime OP_COPY = 28
comptime OP_CM_CHUNK = 29
comptime OP_CM_FINAL = 30
comptime OP_COL_MAX = 6
#: rows per chunk of the column maximum
comptime CM_CHUNK = 512
comptime OP_WPCT_IOTA = 31
comptime OP_CURVE_CNT = 32
comptime OP_CURVE_OFF = 33
comptime OP_CURVE_FILL = 34
comptime OP_CURVE_KEEP = 35
#: lane metrics-apple2: a caller's K-fold rows (x_metrics/split.mojo
#: fold_rows_unit) and its wide schedule (x_metrics/par.mojo fr_*)
comptime OP_FOLD_ROWS = 36
comptime OP_FR_SCATTER = 37
comptime OP_FR_CNT = 38
comptime OP_FR_OFF = 39
comptime OP_FR_FILL = 40
#: a caller's Int32 -> Int64 row words (x_metrics/split.mojo rows64_unit)
comptime OP_ROWS64 = 41
#: lane metrics-apple2: the kept points of an unweighted curve, compacted
#: (x_metrics/par.mojo ck_*; bin_curve params 12 = CF, 13 = CM)
comptime OP_CK_CNT = 42
comptime OP_CK_OFF = 43
comptime OP_CK_FILL = 44
#: slots per chunk of the curve compaction
comptime CK_CHUNK = 1024
#: a caller's StratifiedKFold row -> fold (x_metrics/split.mojo strat_codes_unit)
comptime OP_STRAT_CODES = 45
#: lane cgr2-metrics-shap: a caller's curve fold (the ROC AUC, the partial
#: AUC and the average precision sums; x_metrics/par.mojo curve_fold_unit)
#: and its wide schedule (cf_*); wpercentile's blocked weighted CDF
comptime OP_CURVE_FOLD = 46
comptime OP_CF_CHUNK = 47
comptime OP_CF_FINAL = 48
#: the label layout units (x_metrics/onehot.mojo; lane apple-fast-py2mojo-core)
comptime OP_ONEHOT = 52
comptime OP_REP_ROWS = 53
comptime OP_PAIR_COLS = 54
comptime OP_WPCT_CSUM = 49
comptime OP_WPCT_COFF = 50
comptime OP_WPCT_CFILL = 51
#: points per chunk of the curve folds
comptime CF_CHUNK = 1024
#: rows per chunk of the K-fold row partition
comptime FR_CHUNK = 1024
#: rows per chunk of the unweighted curve counts
# AFCL-P09: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified.
# Halving the prefix chunk doubles independent chunks while retaining every
# score threshold and tie group. Weighted scans change association, not rows.
comptime CURVE_CHUNK = 512 if AFCL_P09 else 1024
#: the unweighted CDF is Float32(i + 1) only while it stays exact
comptime IOTA_EXACT = 1 << 24
#: ops 55 (cls_epi) .. 62 (cl_epi): the metric tails and scans (x_metrics/cls_epi.mojo,
#: tail.mojo, cm_epi.mojo, reg_epi.mojo, rank_epi.mojo; lane cpu2-l7-metrics), caller-named
comptime OP_LAST_TAIL = 62
#: StratifiedShuffleSplit's per-class draw counts (x_metrics/split.mojo approx_mode_unit; lane py-runtime-b)
comptime OP_APPROX_MODE = 81


@always_inline
def is_user_op(op: Int) -> Bool:
    """An op a caller may name: 0..N_USER_OPS-1, fold_rows, rows64, strat_codes, curve_fold and the
    label layout units onehot, rep_rows and pair_cols."""
    return ((op >= 0 and op < N_USER_OPS) or op == OP_FOLD_ROWS or op == OP_ROWS64 or op == OP_STRAT_CODES
            or op == OP_CURVE_FOLD or op == OP_ONEHOT or op == OP_REP_ROWS or op == OP_PAIR_COLS
            or (op >= OP_CLS_EPI and op <= OP_LAST_TAIL) or op == OP_CURVE_OUT or op == OP_AUC_XY or op == OP_CONT_STATS or op == OP_APPROX_MODE)
#: the chunk length the counting sort aims for, and the bound on its
#: (groups x chunks) count table
comptime CS_CHUNK = 256
comptime CS_TABLE = 1 << 22
comptime ARENA_BOUND = 2147483647


struct Plan(Movable):
    var rows: List[Int32]
    var stages: Int
    var size: Int
    #: OFF offset -> rows n, for each group_sort planned so far
    var off_at: List[Int]
    var off_n: List[Int]

    def __init__(out self, arena_len: Int):
        self.rows = List[Int32]()
        self.stages = 0
        self.size = arena_len
        self.off_at = List[Int]()
        self.off_n = List[Int]()

    def alloc(mut self, k: Int) -> Int:
        var off = self.size
        self.size += k if k > 0 else 0
        return off

    def emit(mut self, op: Int, total: Int, params: List[Int]):
        self.rows.append(Int32(op))
        self.rows.append(Int32(total))
        for i in range(PARAMS):
            self.rows.append(Int32(params[i]) if i < len(params) else Int32(0))
        self.stages += 1

    def copy_stage(mut self, q: IP, s: Int):
        for k in range(STAGE_INTS):
            self.rows.append(q.unsafe_load(s * STAGE_INTS + k))
        self.stages += 1

    def fits(self, extra: Int) -> Bool:
        return self.size + extra <= ARENA_BOUND

    def rows_of(self, off: Int) -> Int:
        var i = len(self.off_at) - 1
        while i >= 0:
            if self.off_at[i] == off:
                return self.off_n[i]
            i -= 1
        return -1

    def sort(mut self, mode: Int, n: Int, P: Int, a0: Int, a1: Int, a2: Int) -> Int:
        """Key, run and merge stages sorting P problems of n rows; returns
        the buffer (KH, KL, IX at B, B+N, B+2N) holding the sorted result."""
        var N = P * n
        var b0 = self.alloc(3 * N)
        var b1 = self.alloc(3 * N)
        self.emit(OP_SORT_KEY, N, [mode, n, b0, N, a0, a1, a2])
        self.emit(OP_SORT_RUNS, P * ((n + RUN - 1) // RUN), [n, b0, N])
        var src = b0
        var dst = b1
        var w = RUN
        while w < n:
            self.emit(OP_SORT_MERGE, N, [n, w, src, dst, N])
            var tmp = src
            src = dst
            dst = tmp
            w *= 2
        return src


@always_inline
def _a(r: IP, k: Int) -> Int:
    return Int(r.unsafe_load(2 + k))


def _separate(a: Int, an: Int, b: Int, bn: Int) -> Bool:
    # Native program metadata only. Missing optional regions are disjoint.
    return a < 0 or b < 0 or an <= 0 or bn <= 0 or a + an <= b or b + bn <= a


def _reg_pair_safe(r: IP, following: IP, total: Int) -> Bool:
    var y = _a(r, 0)
    var pred = _a(r, 1)
    var pn = _a(r, 3) if _a(r, 6) == 1 else total
    var a = _a(r, 2)
    var b = _a(following, 2)
    return (a >= 0 and b >= 0 and _separate(a, total, b, total)
            and _separate(a, total, y, total) and _separate(a, total, pred, pn)
            and _separate(b, total, y, total) and _separate(b, total, pred, pn))


def _curve_pair_safe(r: IP, following: IP, n: Int, total: Int) -> Bool:
    # Reuse only independent arenas. Aliased producer programs retain B.
    # KEEP has extra write regions; it is left to the ordinary planner.
    if _a(r, 11) != 0 or _a(following, 11) != 0 or _a(r, 1) < total:
        return False
    var N = n * total
    var score_n = (n - 1) * _a(r, 1) + total
    for stage in range(2):
        var a = r if stage == 0 else following
        for k in range(5, 10):
            var at = _a(a, k)
            var count = total if k == 9 else N
            if at < 0 or not _separate(at, count, _a(r, 0), score_n):
                return False
            if not _separate(at, count, _a(r, 2), N) or not _separate(at, count, _a(r, 3), n):
                return False
            for other_stage in range(stage, 2):
                var b = r if other_stage == 0 else following
                for j in range(5, 10):
                    if other_stage == stage and j <= k:
                        continue
                    if not _separate(at, count, _a(b, j), total if j == 9 else N):
                        return False
    return True


def _plan_keep(mut pl: Plan, r: IP, n: Int, total: Int):
    """bin_curve params 10 and 11 (lane metrics-apple): with KEEP (10) and
    the flag (11) == 1, the collinear-drop flags of every slot follow the
    curve (x_metrics/par.mojo curve_keep_unit; weighted curves too, lane
    cgr2-metrics-shap)."""
    if _a(r, 11) == 1 and n > 0:
        pl.emit(OP_CURVE_KEEP, n * total, [n, _a(r, 6), _a(r, 7), _a(r, 9), _a(r, 10)])
        # params 12, 13 (lane metrics-apple2): CF > 0 = the kept slots'
        # fps, tps and threshold words, in order, at CF + p*n, CF + N + p*n,
        # CF + 2N + p*n, and their count at CM + p
        var CF = _a(r, 12)
        var C = (n + CK_CHUNK - 1) // CK_CHUNK
        if CF > 0 and pl.fits(C * total):
            var S = pl.alloc(C * total)
            pl.emit(OP_CK_CNT, C * total, [n, _a(r, 10), _a(r, 9), S, C, CK_CHUNK])
            pl.emit(OP_CK_OFF, total, [S, C, _a(r, 13)])
            pl.emit(OP_CK_FILL, C * total, [n, _a(r, 10), _a(r, 9), S, C, CK_CHUNK,
                                            _a(r, 6), _a(r, 7), _a(r, 8), CF, n * total])


def _plan_curve_out(mut pl: Plan, r: IP, total: Int) raises:
    """curve_out (lane cpu4-python, x_metrics/curve_out.mojo): q = [kind, n,
    FPS, TPS, THR, CNT, DROP, OUT, LEN], one problem. DROP (precision-recall
    and DET): the drop rule's flags (co_keep), then the curve compaction
    (ck_cnt, ck_off, ck_fill) into scratch; DET: its slice (co_det); then
    one co_emit unit per output index."""
    var kind = _a(r, 0)
    var n = _a(r, 1)
    if total != 1 or n <= 0:
        raise Error("x_metrics: curve_out takes one problem of at least one point")
    var fps = _a(r, 2)
    var tps = _a(r, 3)
    var thr = _a(r, 4)
    var cnt = _a(r, 5)
    var C = (n + CK_CHUNK - 1) // CK_CHUNK
    if not pl.fits(n + C + 3 * n + 2):
        raise Error("x_metrics: curve_out exceeds the arena bound")
    if _a(r, 6) != 0:
        var KEEP = pl.alloc(n)
        var S = pl.alloc(C)
        var CF = pl.alloc(3 * n)
        var CM = pl.alloc(1)
        pl.emit(OP_CO_KEEP, n, [n, tps, cnt, KEEP])
        pl.emit(OP_CK_CNT, C, [n, KEEP, cnt, S, C, CK_CHUNK])
        pl.emit(OP_CK_OFF, 1, [S, C, CM])
        pl.emit(OP_CK_FILL, C, [n, KEEP, cnt, S, C, CK_CHUNK, fps, tps, thr, CF, n])
        fps = CF
        tps = CF + n
        thr = CF + 2 * n
        cnt = CM
    var B = pl.alloc(1)
    if kind == CO_DET:
        pl.emit(OP_CO_DET, 1, [fps, tps, cnt, _a(r, 8), B])
    pl.emit(OP_CO_EMIT, n + 1, [kind, n, fps, tps, thr, cnt, _a(r, 7), _a(r, 8), B])


def _plan_ff(mut pl: Plan, src: Int, m: Int, dst: Int, mode: Int, x: Int):
    """The float-float fold of m records at src (x_metrics/contingency.mojo
    ff_chunk, ff_fin) into the binary64 at dst."""
    var NC = (m + CT_CH - 1) // CT_CH
    var S = pl.alloc(FF_REC * NC)
    if NC > 0:
        pl.emit(OP_FF_CHUNK, NC, [src, m, S, CT_CH, NC])
    pl.emit(OP_FF_FIN, 1, [S, NC, dst, mode, x])


def _plan_cont_stats(mut pl: Plan, r: IP, total: Int) raises:
    """cont_stats (lane cpu4-python, x_metrics/contingency.mojo): q = [OFF,
    ka, kb, kk, EPS, C, F, R, K, P, E, MI, EMI, n], one problem."""
    var ka = _a(r, 1)
    var kb = _a(r, 2)
    var n = _a(r, 13)
    if total != 1 or ka < 0 or kb < 0 or _a(r, 3) < ka or _a(r, 3) < kb:
        raise Error("x_metrics: cont_stats sizes")
    var m = ka * kb
    var NI = (max(max(ka, kb), 1) + CT_CH - 1) // CT_CH
    if not pl.fits(2 * ka + CT_IREC * NI + FF_REC * (ka + kb + 2 * m) + 4 * FF_REC * (NI + m // CT_CH + 4)):
        raise Error("x_metrics: cont_stats exceeds the arena bound")
    var C = _a(r, 5)
    var R = _a(r, 7)
    var K = _a(r, 8)
    var SQ = pl.alloc(2 * ka)
    pl.emit(OP_CT_CELL, m, [_a(r, 0), kb, _a(r, 3), _a(r, 4), C, _a(r, 6), ka])
    pl.emit(OP_CT_ROW, ka, [C, ka, kb, R, SQ])
    pl.emit(OP_CT_COL, kb, [C, ka, kb, K])
    var S = pl.alloc(CT_IREC * NI)
    pl.emit(OP_CT_ISUM, NI, [R, K, SQ, ka, kb, S, CT_CH, NI])
    pl.emit(OP_CT_PAIRS, 1, [S, NI, n, _a(r, 9)])
    var T = pl.alloc(FF_REC * (ka + kb))
    pl.emit(OP_CT_ENT, ka + kb, [R, K, ka, kb, n, T])
    var E = _a(r, 10)
    _plan_ff(pl, T, ka, E, FF_ENT, n)
    _plan_ff(pl, T + FF_REC * ka, kb, E + 2, FF_ENT, n)
    if _a(r, 11) >= 0:
        var TM = pl.alloc(FF_REC * m)
        pl.emit(OP_MI_CELL, m, [C, R, K, ka, kb, n, TM])
        _plan_ff(pl, TM, m, _a(r, 11), FF_MI, 1 if (ka == 1 or kb == 1) else 0)
    if _a(r, 12) >= 0:
        var TE = pl.alloc(FF_REC * m)
        pl.emit(OP_EMI_CELL, m, [R, K, ka, kb, n, TE])
        _plan_ff(pl, TE, m, _a(r, 12), FF_SUM, 0)


def plan_program(q: IP, stages: Int, arena_len: Int) raises -> Plan:
    var pl = Plan(arena_len)
    var peak = arena_len
    var skip = -1
    for s in range(stages):
        # C05: every expansion finishes into caller arena before the next
        # source stage. Retain one bounded, invocation-owned workspace and
        # reuse its offsets after that boundary; no cross-call stale state.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        comptime if C05_PHASE_SCRATCH:
            peak = max(peak, pl.size)
            pl.size = arena_len
        if s == skip:
            continue
        var r = q + s * STAGE_INTS
        var op = Int(r.unsafe_load(0))
        var total = Int(r.unsafe_load(1))
        if not is_user_op(op):
            raise Error(String("x_metrics: unknown op ", op))
        if total <= 0:
            pl.copy_stage(q, s)
            continue
        comptime if C09_REG_BUNDLE:
            if op == 3 and s+1 < stages:
                var following = q + (s+1)*STAGE_INTS
                # Only fuse adjacent independent SQ/ABS stages with the same
                # inputs, broadcast and output geometry. B keeps both maps.
                if (Int(following[0]) == 3 and Int(following[1]) == total
                    and _a(r, 4) == 0 and _a(following, 4) == 1
                    and _a(r, 0) == _a(following, 0) and _a(r, 1) == _a(following, 1)
                    and _a(r, 3) == _a(following, 3) and _a(r, 6) == _a(following, 6)
                    and _reg_pair_safe(r, following, total)):
                    pl.emit(82, total, [_a(r,0), _a(r,1), _a(r,2), _a(following,2), _a(r,3), _a(r,6)])
                    skip = s+1
                    continue
        if op == OP_GROUP_SORT:
            var K = _a(r, 0)
            var n = _a(r, 1)
            var m = _a(r, 2)
            var OFF = _a(r, 3)
            var ORD = _a(r, 4)
            var C = (n + CS_CHUNK - 1) // CS_CHUNK
            var cap = CS_TABLE // (m + 1)
            if cap < 1:
                cap = 1
            if C > cap:
                C = cap
            comptime if C12_SPARSE_COUNTS:
                # Occupancy can never exceed n. Prefer row storage when
                # full class/chunk tables cost more than sorting n keys.
                # No measured-shape threshold; dense OFF output still paid.
                if n > 1 and 6*n < m*C+m and pl.fits(6*n):
                    var B = pl.sort(KEY_GROUP, n, 1, K, m, 0)
                    pl.emit(OP_SORT_EMIT, n, [B, n, ORD])
                    pl.emit(83, m+1, [K, n, m, ORD, OFF])
                    pl.off_at.append(OFF)
                    pl.off_n.append(n)
                    continue
            if n <= 0 or m <= 0 or C < 1 or not pl.fits(m * C + m):
                pl.copy_stage(q, s)
                continue
            var CH = (n + C - 1) // C
            C = (n + CH - 1) // CH
            var H = pl.alloc(m * C)
            var GT = pl.alloc(m)
            pl.emit(OP_CS_HIST, C, [K, n, m, H, C, CH])
            pl.emit(OP_CS_SCAN_ROWS, m, [H, C, GT])
            pl.emit(OP_CS_SCAN_GROUPS, 1, [GT, m, OFF])
            pl.emit(OP_CS_PLACE, C, [K, n, m, H, C, CH, OFF, ORD])
            pl.off_at.append(OFF)
            pl.off_n.append(n)
        elif op == OP_GROUP_SUM:
            var OFF = _a(r, 0)
            var V = _a(r, 2)
            var W = _a(r, 4)
            var D = _a(r, 6)
            var n = pl.rows_of(OFF)
            if n < 0 or D <= 0 or (V < 0 and W < 0):
                pl.copy_stage(q, s)
                continue
            var m = total // D
            var NS = (n // LEAF + m + 1) * D
            if not pl.fits(NS):
                pl.copy_stage(q, s)
                continue
            var S = pl.alloc(NS)
            pl.emit(OP_FOLD_LEAF, NS, [OFF, _a(r, 1), V, _a(r, 3), W, S, D, m])
            var full = n // LEAF
            var j = 0
            while (2 << j) <= full:
                pl.emit(OP_FOLD_LEVEL, NS, [OFF, S, D, m, j])
                j += 1
            pl.emit(OP_FOLD_FINAL, total, [OFF, S, D, _a(r, 5), V, W])
        elif op == OP_COL_SORT:
            var V = _a(r, 0)
            var n = _a(r, 1)
            var D = _a(r, 2)
            if n <= 1 or not pl.fits(6 * n * total):
                pl.copy_stage(q, s)
                continue
            var B = pl.sort(KEY_COL, n, total, V, D, 0)
            pl.emit(OP_SORT_EMIT, n * total, [B, n * total, _a(r, 3)])
        elif op == OP_WPERCENTILE:
            var n = _a(r, 1)
            var WC = (n + CURVE_CHUNK - 1) // CURVE_CHUNK
            if n <= 0 or not pl.fits(2 * n * total + WC * total):
                pl.copy_stage(q, s)
                continue
            var N = n * total
            var G = pl.alloc(2 * N)
            pl.emit(OP_WPCT_GATHER, N, [n, _a(r, 3), _a(r, 4), G])
            if _a(r, 4) < 0 and n <= IOTA_EXACT:
                pl.emit(OP_WPCT_IOTA, N, [n, G + N])
            else:
                # the blocked weighted CDF (x_metrics/par.mojo wpct_c*)
                var S = pl.alloc(WC * total)
                pl.emit(OP_WPCT_CSUM, WC * total, [n, G, S, WC, CURVE_CHUNK])
                pl.emit(OP_WPCT_COFF, total, [S, WC])
                pl.emit(OP_WPCT_CFILL, WC * total, [n, G, G + N, S, WC, CURVE_CHUNK])
            pl.emit(OP_COPY, N, [G + N, _a(r, 8)])
            var sel = List[Int]()
            for k in range(9):
                sel.append(_a(r, k))
            pl.emit(OP_WPCT_SELECT, total, sel)
        elif op == OP_BIN_CURVE:
            var n = _a(r, 4)
            if n <= 1 or not pl.fits(12 * n * total + total + 3 * total * ((n + CURVE_CHUNK - 1) // CURVE_CHUNK)):
                pl.copy_stage(q, s)
                _plan_keep(pl, r, n, total)
                continue
            var N = n * total
            var B = pl.sort(KEY_CURVE, n, total, _a(r, 0), _a(r, 1), _a(r, 3))
            var G = pl.alloc(6 * N + total)
            pl.emit(OP_CURVE_GATHER, N, [n, B, N, _a(r, 0), _a(r, 1), _a(r, 2), _a(r, 3), G, _a(r, 5)])
            var C = (n + CURVE_CHUNK - 1) // CURVE_CHUNK
            var S = pl.alloc(3 * C * total)
            # counts unweighted, the blocked Float32 scan weighted (W = param 6)
            pl.emit(OP_CURVE_CNT, C * total, [n, G, N, S, C, CURVE_CHUNK, _a(r, 3)])
            pl.emit(OP_CURVE_OFF, total, [G, N, S, C, _a(r, 3)])
            pl.emit(OP_CURVE_FILL, C * total, [n, G, N, S, C, CURVE_CHUNK, _a(r, 3)])
            pl.emit(OP_CURVE_EMIT, N, [n, G, N, _a(r, 6), _a(r, 7), _a(r, 8), _a(r, 9)])
            _plan_keep(pl, r, n, total)
            comptime if C10_RANK_REUSE:
                if s+1 < stages:
                    var next_curve = q+(s+1)*STAGE_INTS
                    var same = Int(next_curve[0]) == OP_BIN_CURVE and Int(next_curve[1]) == total
                    for k in range(5):
                        same = same and _a(r,k) == _a(next_curve,k)
                    if same and _curve_pair_safe(r, next_curve, n, total):
                        # Sources were not touched between these adjacent
                        # producers. Reuse tie blocks and weighted prefixes;
                        # each consumer still receives its own output layout.
                        pl.emit(84, N, [n, _a(r,6), _a(r,7), _a(r,8), _a(r,9),
                                      _a(next_curve,6), _a(next_curve,7), _a(next_curve,8), _a(next_curve,9),
                                      _a(r,5), _a(next_curve,5), G])
                        _plan_keep(pl, next_curve, n, total)
                        skip = s+1
        elif op == OP_PERMUTE:
            var n = _a(r, 0)
            if n <= 1 or not pl.fits(6 * n):
                pl.copy_stage(q, s)
                continue
            var B = pl.sort(KEY_PERM, n, 1, _a(r, 2), _a(r, 3), 0)
            pl.emit(OP_SORT_EMIT, n, [B, n, _a(r, 1)])
        elif op == OP_COL_MAX:
            var V = _a(r, 0)
            var n = _a(r, 1)
            var D = _a(r, 2)
            var C = (n + CM_CHUNK - 1) // CM_CHUNK
            if n <= 1 or D != total or not pl.fits(2 * C * total):
                pl.copy_stage(q, s)
                continue
            var S = pl.alloc(2 * C * total)
            pl.emit(OP_CM_CHUNK, C * total, [V, n, D, S, C, CM_CHUNK])
            pl.emit(OP_CM_FINAL, total, [V, D, S, C, _a(r, 3)])
        elif op == OP_CURVE_FOLD:
            # q = [n, FPS, TPS, CNT, OUT, mode, MH, ML, CH]; total = problems
            var n = _a(r, 0)
            var CH = _a(r, 8)
            if n <= 0 or CH < 1:
                pl.copy_stage(q, s)
                continue
            var C = (n + CH - 1) // CH
            if not pl.fits(3 * C * total):
                pl.copy_stage(q, s)
                continue
            var S = pl.alloc(3 * C * total)
            pl.emit(OP_CF_CHUNK, C * total, [n, _a(r, 1), _a(r, 2), _a(r, 3), S, C, CH, _a(r, 5), _a(r, 6), _a(r, 7)])
            pl.emit(OP_CF_FINAL, total, [n, _a(r, 1), _a(r, 2), _a(r, 3), S, C, _a(r, 4), _a(r, 5)])
        elif op == OP_FOLD_ROWS:
            var n = _a(r, 0)
            var K = _a(r, 1)
            var C = (n + FR_CHUNK - 1) // FR_CHUNK
            if n <= 1 or K < 1 or total != 1 or not pl.fits(K * C):
                pl.copy_stage(q, s)
                continue
            var S = pl.alloc(K * C)
            if _a(r, 3) >= 0:
                pl.emit(OP_FR_SCATTER, n, [n, K, _a(r, 2), _a(r, 3)])
            pl.emit(OP_FR_CNT, K * C, [n, K, _a(r, 2), S, C, FR_CHUNK])
            pl.emit(OP_FR_OFF, K, [S, C, _a(r, 5)])
            pl.emit(OP_FR_FILL, K * C, [n, K, _a(r, 2), S, C, FR_CHUNK, _a(r, 4), _a(r, 5)])
        elif op == OP_CURVE_OUT:
            _plan_curve_out(pl, r, total)
        elif op == OP_CONT_STATS:
            _plan_cont_stats(pl, r, total)
        elif op == OP_AUC_XY:
            # q = [n, X, Y, OUT, CH] (lane cpu4-python, x_metrics/curve_out.mojo)
            var n = _a(r, 0)
            var CH = _a(r, 4)
            if total != 1 or n < 2 or CH < 1:
                raise Error("x_metrics: auc_xy takes one problem of at least 2 points")
            var C = (n + CH - 1) // CH
            if not pl.fits(AX_REC * C):
                raise Error("x_metrics: auc_xy exceeds the arena bound")
            var S = pl.alloc(AX_REC * C)
            pl.emit(OP_AX_CHUNK, C, [n, _a(r, 1), _a(r, 2), S, C, CH])
            pl.emit(OP_AX_FINAL, 1, [S, C, _a(r, 3)])
        else:
            pl.copy_stage(q, s)
    comptime if C05_PHASE_SCRATCH:
        pl.size = max(peak, pl.size)
    return pl^
