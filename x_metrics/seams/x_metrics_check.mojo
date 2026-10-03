# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's seam gate (DEVIATIONS 6100-6108, IDENTITY_PATHS rows
190-197). For each seam of the x_metrics units: first show the fixture
SEPARATES the pinned spelling from the unpinned one (else VACUOUS, a
failure), then run the SHIPPED unit through the host runner and, when this
box has an accelerator, through the device runner, and require both to equal
the pinned oracle (x_metrics.seams.x_metrics_oracle.mojo) bit for bit; the
shipped outputs go on the card.

    tools/with_identical_mode.sh pixi run mojo run -I . x_metrics/seams/x_metrics_check.mojo

The card goes to $MOJOLEARN_XMETRICS_CARD (default /tmp/x_metrics_seams.card);
`python3 tools/identity_trace_diff.py a.card b.card` compares two boxes.
"""
from std.os import getenv
from std.memory import bitcast
from std.sys import has_accelerator
from core.identity_trace import IdentityTrace
from x_metrics.common import STAGE_INTS, LEAF
from x_metrics.host.program import run_program_host_ptr
from x_metrics.device import run_program_device_ptr
from checks.fixture_rng import u01_triple
from x_metrics.seams.x_metrics_oracle import (
    seq_sum, pair_sum, argsort_key, argsort_value, pinned_dot_add, fused_dot_add,
    seq_prefix, refold_prefix, key_permutation, fisher_yates,
)

comptime OP_GROUP_SORT = 0
comptime OP_GROUP_SUM = 1
comptime OP_REG_TERM = 3
comptime OP_COL_SORT = 4
comptime OP_WPERCENTILE = 5
comptime OP_COL_MAX = 6
comptime OP_BIN_CURVE = 7
comptime OP_PERMUTE = 10


def _b(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _f(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _i(x: Int) -> Float32:
    """An int32 slot's word."""
    return bitcast[DType.float32](Int32(x))


def _ival(x: Float32) -> Int32:
    return bitcast[DType.int32](x)


def _require(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL: " + what)


struct Prog(Movable):
    """A small program over a small arena."""
    var arena: List[Float32]
    var prog: List[Int32]
    var stages: Int

    def __init__(out self, arena: List[Float32]):
        self.arena = arena.copy()
        self.prog = List[Int32]()
        self.stages = 0

    def stage(mut self, op: Int, total: Int, params: List[Int]):
        var row = List[Int32](length=STAGE_INTS, fill=0)
        row[0] = Int32(op)
        row[1] = Int32(total)
        for i in range(len(params)):
            row[2 + i] = Int32(params[i])
        for v in row:
            self.prog.append(v)
        self.stages += 1

    def run(self, device: Bool, legacy: Bool = False, threads: Int = 0) raises -> List[Float32]:
        var f = self.arena.copy()
        var q = self.prog.copy()
        var fp = f.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var qp = q.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if device:
            comptime if has_accelerator():
                run_program_device_ptr(fp, len(f), qp, self.stages, legacy)
        else:
            run_program_host_ptr(fp, len(f), qp, self.stages, legacy, threads)
        _ = len(q)
        return f^


def _both(p: Prog) raises -> List[List[Float32]]:
    var out = List[List[Float32]]()
    out.append(p.run(False))
    comptime if has_accelerator():
        out.append(p.run(True))
    return out^


def _grouped_sum(values: List[Float32], weights: List[Float32]) raises -> Prog:
    """One group (every key 0) summed by group_sort + group_sum. Arena:
    keys [0, n), values, weights, OFF (2), ORD (n), OUT (1)."""
    var n = len(values)
    var arena = List[Float32]()
    for _ in range(n):
        arena.append(_i(0))
    var V = len(arena)
    for v in values:
        arena.append(v)
    var W = -1
    if len(weights):
        W = len(arena)
        for w in weights:
            arena.append(w)
    var OFF = len(arena)
    arena.append(0)
    arena.append(0)
    var ORD = len(arena)
    for _ in range(n):
        arena.append(0)
    var OUT = len(arena)
    arena.append(0)
    var p = Prog(arena)
    p.stage(OP_GROUP_SORT, 1, [0, n, 1, OFF, ORD])
    p.stage(OP_GROUP_SUM, 1, [OFF, ORD, V, 1, W, OUT, 1])
    return p^


def check_fold_shape(mut card: IdentityTrace) raises:
    """DEVIATION 6100: every float sum is the fixed-shape pairwise fold."""
    var v = List[Float32]()
    v.append(Float32(16777216.0))
    for _ in range(LEAF + 7):
        v.append(Float32(1.0))
    _require(_b(seq_sum(v)) != _b(pair_sum(v, 32)), "VACUOUS fold fixture: sequential == pairwise")
    var p = _grouped_sum(v, List[Float32]())
    var want = pair_sum(v, 32)
    for r in _both(p):
        _require(_b(r[len(r) - 1]) == _b(want), "fold shape: group_sum " + String(r[len(r) - 1]) + " != " + String(want))
    card.record_scalar_f32("6100.fold", p.run(False)[len(v) * 3 + 2])
    print("PASS 6100 fold shape: group_sum folds leaves of 32 as a binary counter (the fixture separates it from a sequential fold)")


def check_sort_key(mut card: IdentityTrace) raises:
    """DEVIATION 6101: sorts order by the total key (-0.0 below +0.0), ties by row."""
    var v: List[Float32] = [0.0, -0.0, 1.0, 1.0, -1.0]
    var want = argsort_key(v)
    var other = argsort_value(v)
    var differ = False
    for i in range(len(v)):
        differ = differ or want[i] != other[i]
    _require(differ, "VACUOUS sort fixture: key order == value order")
    var arena = v.copy()
    for _ in range(len(v)):
        arena.append(0)
    var p = Prog(arena)
    p.stage(OP_COL_SORT, 1, [0, len(v), 1, len(v)])
    for r in _both(p):
        for i in range(len(v)):
            _require(_ival(r[len(v) + i]) == want[i], "sort key: col_sort position " + String(i))
    var got = p.run(False)
    var rec = List[Int32]()
    for i in range(len(v)):
        rec.append(_ival(got[len(v) + i]))
    card.record_list_i32("6101.sort", rec)
    print("PASS 6101 sort key: col_sort orders -0.0 below +0.0 and ties by row (the fixture separates it from a value sort)")


def check_group_order(mut card: IdentityTrace) raises:
    """DEVIATION 6102: a group's rows are folded in ascending row order."""
    var v: List[Float32] = [16777216.0, 1.0, 1.0]
    var rev: List[Float32] = [1.0, 1.0, 16777216.0]
    _require(_b(seq_sum(v)) != _b(seq_sum(rev)), "VACUOUS group-order fixture")
    var p = _grouped_sum(v, List[Float32]())
    for r in _both(p):
        _require(_b(r[len(r) - 1]) == _b(seq_sum(v)), "group order: group_sum " + String(r[len(r) - 1]))
    card.record_scalar_f32("6102.group", p.run(False)[len(v) * 3 + 2])
    print("PASS 6102 group order: a group's rows fold ascending (the fixture separates it from descending)")


def check_nan_guard(mut card: IdentityTrace) raises:
    """DEVIATION 6103: no computed NaN; APE floors |y| at eps64, so 0/0 never happens."""
    var zero = Float32(0)
    var raw = zero / zero
    _require(raw != raw, "VACUOUS NaN-guard fixture: 0/0 is not a NaN here")
    # reg_term APE: q = [Y, P, OUT, D, kind, S, PB]
    var arena: List[Float32] = [0.0, 0.0, 0.0, 7.0]
    var p = Prog(arena)
    p.stage(OP_REG_TERM, 1, [0, 1, 2, 1, 3, 0, 0])
    for r in _both(p):
        _require(_b(r[2]) == _b(Float32(0)), "NaN guard: APE of (0, 0) is " + String(r[2]))
    card.record_scalar_f32("6103.ape0", p.run(False)[2])
    print("PASS 6103 NaN guard: APE of y = p = 0 is +0.0, never 0/0 (the fixture's raw spelling is a NaN)")


def check_operand_ftz(mut card: IdentityTrace) raises:
    """DEVIATION 6104: every operand a unit loads is flushed."""
    var sub = _f(UInt32(0x00000100))  # a positive subnormal: loads as +0.0
    var v: List[Float32] = [sub, 0.0]
    var flushed: List[Float32] = [0.0, 0.0]
    var a = argsort_key(v)
    var b = argsort_key(flushed)
    _require(a[0] != b[0], "VACUOUS ftz fixture: the raw and flushed orders agree")
    var arena = v.copy()
    arena.append(0)
    arena.append(0)
    var p = Prog(arena)
    p.stage(OP_COL_SORT, 1, [0, 2, 1, 2])
    for r in _both(p):
        _require(_ival(r[2]) == b[0] and _ival(r[3]) == b[1], "ftz: col_sort read the subnormal unflushed")
    var got = p.run(False)
    var rec: List[Int32] = [_ival(got[2]), _ival(got[3])]
    card.record_list_i32("6104.ftz", rec)
    print("PASS 6104 operand ftz: a subnormal loads as +0.0 and ties +0.0 by row (the fixture separates it from the raw word)")


def check_contraction(mut card: IdentityTrace) raises:
    """DEVIATION 6105: a weighted value is rounded before it is added."""
    var ulp = Float32(1.1920929e-07)
    var a = Float32(1) + Float32(3) * ulp
    var b = Float32(1) + Float32(5) * ulp
    _require(_b(pinned_dot_add(Float32(-1), a, b)) != _b(fused_dot_add(Float32(-1), a, b)),
             "VACUOUS contraction fixture: fused == pinned")
    var v: List[Float32] = [-1.0, a]
    var w: List[Float32] = [1.0, b]
    var p = _grouped_sum(v, w)
    var want = pinned_dot_add(Float32(-1), a, b)
    for r in _both(p):
        _require(_b(r[len(r) - 1]) == _b(want), "contraction: weighted group_sum " + String(r[len(r) - 1]))
    card.record_scalar_f32("6105.dot", p.run(False)[len(p.arena) - 1])
    print("PASS 6105 contraction: v * w is rounded before the fold adds it (the fixture separates it from fma)")


def check_prefix(mut card: IdentityTrace) raises:
    """DEVIATION 6107: the weighted percentile's CDF is a blocked prefix,
    sequential inside a chunk of 1024 (this fixture is one chunk)."""
    var w: List[Float32] = [1.0, 1.0, 16777216.0, 1.0]
    var want = seq_prefix(w)
    var other = refold_prefix(w)
    var differ = False
    for i in range(len(w)):
        differ = differ or _b(want[i]) != _b(other[i])
    _require(differ, "VACUOUS prefix fixture")
    # values 0,1,2,3 already sorted; ORD identity; q = [V, n, D, ORD, W, R, avg, OUT, CDF]
    var arena: List[Float32] = [0.0, 1.0, 2.0, 3.0]            # V 0..3
    for i in range(4):
        arena.append(_i(i))                                      # ORD 4..7
    for x in w:
        arena.append(x)                                          # W 8..11
    arena.append(50.0)                                           # R 12
    arena.append(0)                                              # OUT 13
    for _ in range(4):
        arena.append(0)                                          # CDF 14..17
    var p = Prog(arena)
    p.stage(OP_WPERCENTILE, 1, [0, 4, 1, 4, 8, 12, 1, 13, 14])
    for r in _both(p):
        for i in range(4):
            _require(_b(r[14 + i]) == _b(want[i]), "prefix: CDF[" + String(i) + "] " + String(r[14 + i]))
    var got = p.run(False)
    var rec = List[Float32]()
    for i in range(5):
        rec.append(got[13 + i])
    card.record_list_f32("6107.cdf", rec)
    print("PASS 6107 prefix: the weight CDF is an ascending prefix within its chunk (the fixture separates it from a refold)")


def check_rng_mapping(mut card: IdentityTrace) raises:
    """DEVIATION 6108: a permutation is the sort of splitmix_pair keys."""
    var n = 16
    var salt = 0x2545F4914F6CDD1D
    var want = key_permutation(n, salt)
    var other = fisher_yates(n, salt)
    var differ = False
    for i in range(n):
        differ = differ or want[i] != other[i]
    _require(differ, "VACUOUS RNG fixture: key sort == Fisher-Yates")
    var arena = List[Float32]()
    for _ in range(n):
        arena.append(0)
    var p = Prog(arena)
    var lo = salt & 0xFFFFFFFF
    var hi = (salt >> 32) & 0xFFFFFFFF
    p.stage(OP_PERMUTE, 1, [n, 0, lo - (1 << 32) if lo >= (1 << 31) else lo, hi - (1 << 32) if hi >= (1 << 31) else hi])
    for r in _both(p):
        for i in range(n):
            _require(_ival(r[i]) == want[i], "RNG mapping: permute position " + String(i))
    var got = p.run(False)
    var rec = List[Int32]()
    for i in range(n):
        rec.append(_ival(got[i]))
    card.record_list_i32("6108.perm", rec)
    print("PASS 6108 RNG mapping: permute sorts by splitmix_pair keys (the fixture separates it from Fisher-Yates)")


def _u(i: Int, k: Int) -> Float64:
    return u01_triple(i, k, 0x6109)


def _arena_digest(a: List[Float32]) -> Int32:
    var h = UInt32(2166136261)
    for x in a:
        h = (h ^ _b(x)) * UInt32(16777619)
    return Int32(bitcast[DType.int32](h))


def check_parallel_schedules(mut card: IdentityTrace) raises:
    """THE PLANNER (x_metrics/plan.mojo, phase C): the wide schedules of
    x_metrics/par.mojo return the SAME ARENA, word for word, as the
    sequential units they replace (the caller's program run unplanned on
    the host), at shapes with many leaves, fold levels, merge passes, ties,
    signed zeros, dropped keys and zero weights."""
    var n = 70001
    var m = 7
    var arena = List[Float32]()
    var K = len(arena)
    for i in range(n):
        arena.append(_i(Int(_u(i, 0) * 8.0) - 1))                # -1 = dropped
    var Z = len(arena)
    for _ in range(n):
        arena.append(_i(0))                                       # one group
    var V = len(arena)
    for i in range(n):
        for c in range(2):
            arena.append(Float32(Int(_u(i, 1 + c) * 2001.0) - 1000) * Float32(0.37))
    var W = len(arena)
    for i in range(n):
        arena.append(Float32(Int(_u(i, 3) * 4.0)) * Float32(0.25))   # zeros included
    var OFF = len(arena)
    for _ in range(m + 1):
        arena.append(0)
    var ORD = len(arena)
    for _ in range(n):
        arena.append(0)
    var OUT = len(arena)
    for _ in range(4 * m):
        arena.append(0)
    var OFF1 = len(arena)
    arena.append(0)
    arena.append(0)
    var ORD1 = len(arena)
    for _ in range(n):
        arena.append(0)
    var OUT1 = len(arena)
    arena.append(0)
    arena.append(0)
    # col_sort + wpercentile over 3 columns of ns rows (ties, signed zeros)
    var ns = 5003
    var CV = len(arena)
    for i in range(ns):
        for c in range(3):
            var v = Float32(Int(_u(i, 10 + c) * 21.0) - 10) / Float32(4)
            if v == Float32(0) and _u(i, 13 + c) < 0.5:
                v = Float32(-0.0)
            arena.append(v)
    var CO = len(arena)
    for _ in range(3 * ns):
        arena.append(0)
    var R = len(arena)
    arena.append(Float32(37.5))
    var PO = len(arena)
    for _ in range(3):
        arena.append(0)
    var CD = len(arena)
    for _ in range(3 * ns):
        arena.append(0)
    # the same columns unweighted (the parallel CDF, lane metrics-apple)
    var PO2 = len(arena)
    for _ in range(3):
        arena.append(0)
    var CD2 = len(arena)
    for _ in range(3 * ns):
        arena.append(0)
    var MX = len(arena)
    for _ in range(3):
        arena.append(0)
    # bin_curve: 2 problems (scores n x 2 row-major, ties), weighted and not
    var nc = 20011
    var S = len(arena)
    for i in range(nc):
        for c in range(2):
            arena.append(Float32(Int(_u(i, 20 + c) * 300.0)) / Float32(299))
    var POS = len(arena)
    for i in range(2 * nc):
        arena.append(_i(1 if _u(i, 22) < 0.4 else 0))
    var BO = len(arena)
    var cur = List[Int]()
    for _ in range(2):
        var off = len(arena)
        cur.append(off)
        for _ in range(4 * 2 * nc + 2):
            arena.append(0)
    var PM = len(arena)
    var np = 10007
    for _ in range(np):
        arena.append(0)
    var p = Prog(arena)
    p.stage(OP_GROUP_SORT, 1, [K, n, m, OFF, ORD])
    p.stage(OP_GROUP_SUM, 2 * m, [OFF, ORD, V, 2, W, OUT, 2])
    p.stage(OP_GROUP_SUM, 2 * m, [OFF, ORD, V, 2, -1, OUT + 2 * m, 2])
    p.stage(OP_GROUP_SORT, 1, [Z, n, 1, OFF1, ORD1])
    p.stage(OP_GROUP_SUM, 2, [OFF1, ORD1, V, 2, W, OUT1, 2])
    p.stage(OP_COL_SORT, 3, [CV, ns, 3, CO])
    p.stage(OP_WPERCENTILE, 3, [CV, ns, 3, CO, W, R, 1, PO, CD])
    p.stage(OP_WPERCENTILE, 3, [CV, ns, 3, CO, -1, R, 0, PO2, CD2])
    p.stage(OP_COL_MAX, 3, [CV, ns, 3, MX])
    for k in range(2):
        var b = cur[k]
        var N = 2 * nc
        p.stage(OP_BIN_CURVE, 2, [S, 2, POS, W if k == 0 else -1, nc, b, b + N, b + 2 * N, b + 3 * N, b + 4 * N])
    p.stage(OP_PERMUTE, 1, [np, PM, 0x2545F491, 0x4F6CDD1D])
    _ = BO
    var want = p.run(False, True)
    var moved = 0
    for i in range(len(arena)):
        if _b(want[i]) != _b(arena[i]):
            moved += 1
    _require(moved > n, "VACUOUS schedule fixture: the program wrote almost nothing")
    var runs = _both(p)
    # THE HOST THREAD COUNT (phase 5, x_metrics/host/program.mojo): the
    # planned program on 1, 2, 3 and 8 host tasks writes the same arena.
    var counts: List[Int] = [1, 2, 3, 8]
    for c in counts:
        runs.append(p.run(False, False, c))
    for r in runs:
        _require(len(r) == len(want), "schedules: arena length")
        for i in range(len(want)):
            if _b(r[i]) != _b(want[i]):
                _require(False, "schedules: planned arena word " + String(i) + " is " + String(r[i]) + ", the sequential units wrote " + String(want[i]))
    var rec: List[Int32] = [_arena_digest(want)]
    card.record_list_i32("plan.arena", rec)
    print("PASS planner: the parallel schedules write the sequential units' arena word for word, host at 1, 2, 3 and 8 tasks (" + String(moved) + " words written)")


def main() raises:
    var path = getenv("MOJOLEARN_XMETRICS_CARD", "/tmp/x_metrics_seams.card")
    var card = IdentityTrace.to_path(path)
    card.header("x_metrics seams (DEVIATIONS 6100-6108)")
    check_fold_shape(card)
    check_sort_key(card)
    check_group_order(card)
    check_nan_guard(card)
    check_operand_ftz(card)
    check_contraction(card)
    check_prefix(card)
    check_rng_mapping(card)
    check_parallel_schedules(card)
    comptime if has_accelerator():
        print("PASS x_metrics seams: 8 seams, host and device, card", path)
    else:
        print("PASS x_metrics seams: 8 seams, host only (no accelerator), card", path)
