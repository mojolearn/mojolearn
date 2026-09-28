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
from x_metrics.common import IP, STAGE_INTS, PARAMS, LEAF
from x_metrics.par import RUN, KEY_COL, KEY_CURVE, KEY_PERM

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
comptime OP_CURVE_PREFIX = 23
comptime OP_WPCT_GATHER = 24
comptime OP_WPCT_PREFIX = 25
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
#: rows per chunk of the unweighted curve counts
comptime CURVE_CHUNK = 1024
#: the unweighted CDF is Float32(i + 1) only while it stays exact
comptime IOTA_EXACT = 1 << 24
#: HOST STAGES: a stage whose unit is one sequential walk of a Float32
#: prefix (DEVIATION 6107 keeps it sequential, so no wide schedule returns
#: its bits). The device runner runs it on the host, over a copy of the
#: slots it reads (params [HOST_RD, HOST_RD+1)) and writes back the slots it
#: writes ([HOST_WR, HOST_WR+1)); the host runner runs it like any stage.
comptime HOST_RD = 10
comptime HOST_WR = 12


@always_inline
def is_host_op(op: Int) -> Bool:
    return op == OP_WPCT_PREFIX or op == OP_CURVE_PREFIX
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


def plan_program(q: IP, stages: Int, arena_len: Int) raises -> Plan:
    var pl = Plan(arena_len)
    for s in range(stages):
        var r = q + s * STAGE_INTS
        var op = Int(r.unsafe_load(0))
        var total = Int(r.unsafe_load(1))
        if op < 0 or op >= N_USER_OPS:
            raise Error(String("x_metrics: unknown op ", op))
        if total <= 0:
            pl.copy_stage(q, s)
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
            if n <= 0 or not pl.fits(2 * n * total):
                pl.copy_stage(q, s)
                continue
            var N = n * total
            var G = pl.alloc(2 * N)
            pl.emit(OP_WPCT_GATHER, N, [n, _a(r, 3), _a(r, 4), G])
            if _a(r, 4) < 0 and n <= IOTA_EXACT:
                pl.emit(OP_WPCT_IOTA, N, [n, G + N])
            else:
                pl.emit(OP_WPCT_PREFIX, total, [n, G, G + N, 0, 0, 0, 0, 0, 0, 0, G, G + N, G + N, G + 2 * N])
            pl.emit(OP_COPY, N, [G + N, _a(r, 8)])
            var sel = List[Int]()
            for k in range(9):
                sel.append(_a(r, k))
            pl.emit(OP_WPCT_SELECT, total, sel)
        elif op == OP_BIN_CURVE:
            var n = _a(r, 4)
            if n <= 1 or not pl.fits(12 * n * total + total):
                pl.copy_stage(q, s)
                continue
            var N = n * total
            var B = pl.sort(KEY_CURVE, n, total, _a(r, 0), _a(r, 1), _a(r, 3))
            var G = pl.alloc(6 * N + total)
            pl.emit(OP_CURVE_GATHER, N, [n, B, N, _a(r, 0), _a(r, 1), _a(r, 2), _a(r, 3), G, _a(r, 5)])
            if _a(r, 3) < 0 and pl.fits(12 * n * total + total + 3 * total * ((n + CURVE_CHUNK - 1) // CURVE_CHUNK)):
                var C = (n + CURVE_CHUNK - 1) // CURVE_CHUNK
                var S = pl.alloc(3 * C * total)
                pl.emit(OP_CURVE_CNT, C * total, [n, G, N, S, C, CURVE_CHUNK])
                pl.emit(OP_CURVE_OFF, total, [G, N, S, C])
                pl.emit(OP_CURVE_FILL, C * total, [n, G, N, S, C, CURVE_CHUNK])
            else:
                pl.emit(OP_CURVE_PREFIX, total, [n, G, N, _a(r, 3), 0, 0, 0, 0, 0, 0, G, G + 3 * N, G + 3 * N, G + 6 * N + total])
            pl.emit(OP_CURVE_EMIT, N, [n, G, N, _a(r, 6), _a(r, 7), _a(r, 8), _a(r, 9)])
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
        else:
            pl.copy_stage(q, s)
    return pl^
