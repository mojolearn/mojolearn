# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's seam gate (DEVIATIONS 5400-5409, IDENTITY_PATHS rows
140-149). For each seam of the x_prep / naive_bayes units: first show the
fixture SEPARATES the pinned spelling from the unpinned one (else VACUOUS,
a failure), then run the SHIPPED unit through the host runner and, when this
box has an accelerator, through the device runner, and require both to equal
the pinned oracle (x_prep/seams/prep_oracle.mojo) bit for bit; the shipped
outputs go on the card.

    tools/with_identical_mode.sh pixi run mojo run -I . x_prep/seams/prep_check.mojo

The card goes to $MOJOLEARN_XPREP_CARD (default /tmp/x_prep_seams.card);
`python3 tools/identity_trace_diff.py a.card b.card` compares two boxes.
"""
from std.os import getenv
from std.memory import bitcast
from std.sys import has_accelerator
from core.identity_trace import IdentityTrace
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.host.program import run_program_host_ptr
from x_prep.device import run_program_device_ptr
from checks.numerics import identical_cos
from x_prep.prims import add, mul, logf, sqrtf, sort_cols_unit
from x_prep.transform import pt_fit_unit
from x_prep.target import te_enc_unit
from x_prep.mutual_info import digammaf
from x_prep.seams.prep_oracle import (
    seq_sum, rev_sum, pinned_dot, fused_dot, key_sorted, value_sorted, guarded_mean, raw_mean,
    first_max, last_max, largest_positive, first_positive, top24, low24, count_strict, count_closed,
    flushed_max, raw_max, numpy_lerp, naive_lerp, splitmix,
)

comptime OP_SORT = 0
comptime OP_COL_STATS = 1
comptime OP_QUANTILE = 2
comptime OP_PT_FIT = 44
comptime OP_TE_ENC = 21
comptime OP_MATMUL = 13
comptime OP_ARGMAX = 15
comptime OP_EIGH = 18
comptime OP_MI_NOISE = 67
comptime OP_MI_CC = 68


def _b(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _f(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _require(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL: " + what)


struct Prog(Movable):
    """A one-stage program over a small arena."""
    var arena: List[Float32]
    var prog: List[Int32]

    def __init__(out self, arena: List[Float32], op: Int, total: Int, params: List[Int]):
        self.arena = arena.copy()
        self.prog = List[Int32](length=STAGE_INTS, fill=0)
        self.prog[0] = Int32(op)
        self.prog[1] = Int32(total)
        for i in range(len(params)):
            self.prog[2 + i] = Int32(params[i])

    def run(self, device: Bool) raises -> List[Float32]:
        var f = self.arena.copy()
        var q = self.prog.copy()
        var fp = f.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var qp = q.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if device:
            comptime if has_accelerator():
                run_program_device_ptr(fp, len(f), qp, 1)
        else:
            run_program_host_ptr(fp, len(f), qp, 1)
        _ = len(q)
        return f^


def _both(p: Prog) raises -> List[List[Float32]]:
    """The host run, and the device run when there is an accelerator."""
    var out = List[List[Float32]]()
    out.append(p.run(False))
    comptime if has_accelerator():
        out.append(p.run(True))
    return out^


def check_fold_order(mut card: IdentityTrace) raises:
    """DEVIATION 5400: every sum in a unit folds rows ascending."""
    var big = Float32(1152921504606846976.0)  # 2^60
    var v: List[Float32] = [big, 1.0, -big, 1.0]
    _require(_b(seq_sum(v)) != _b(rev_sum(v)), "VACUOUS fold fixture: ascending == descending")
    var arena = v.copy()
    for _ in range(6):
        arena.append(0)
    var want = guarded_mean(seq_sum(v), 4)
    for r in _both(Prog(arena, OP_COL_STATS, 1, [0, 4, 1, 4])):
        _require(_b(r[5]) == _b(want), "fold order: col_stats mean " + String(r[5]) + " != " + String(want))
    card.record_list_f32("5400.fold", _both(Prog(arena, OP_COL_STATS, 1, [0, 4, 1, 4]))[0])
    print("PASS 5400 fold order: col_stats sums rows ascending (the fixture separates it from descending)")


def check_contraction(mut card: IdentityTrace) raises:
    """DEVIATION 5401: a product is rounded before its add (no FMA)."""
    var ulp = Float32(1.1920929e-07)
    var a: List[Float32] = [1.0, Float32(1) + Float32(3) * ulp]
    var b: List[Float32] = [-1.0, Float32(1) + Float32(5) * ulp]
    _require(_b(pinned_dot(a, b)) != _b(fused_dot(a, b)), "VACUOUS contraction fixture: fused == pinned")
    var arena: List[Float32] = [a[0], a[1], b[0], b[1], 0]
    # C = A (1 x 2) . B (2 x 1): q = [A, sa0, sa1, B, sb0, sb1, C, ncols, K, BIAS, ALPHA]
    var p = Prog(arena, OP_MATMUL, 1, [0, 2, 1, 2, 1, 0, 4, 1, 2, -1, -1])
    for r in _both(p):
        _require(_b(r[4]) == _b(pinned_dot(a, b)), "contraction: matmul " + String(r[4]))
    card.record_list_f32("5401.dot", p.run(False))
    print("PASS 5401 contraction: matmul rounds each product (the fixture separates it from fma)")


def check_sort_key(mut card: IdentityTrace) raises:
    """DEVIATION 5402: sorts order words by the total key (-0.0 < +0.0, NaN last)."""
    var nan = _f(UInt32(0x7FC00000))
    var v: List[Float32] = [0.0, -0.0, 2.0, nan, -1.0, -0.0, 0.0]
    var ks = key_sorted(v)
    var vs = value_sorted(v)
    var sep = False
    for i in range(len(v)):
        if _b(ks[i]) != _b(vs[i]):
            sep = True
    _require(sep, "VACUOUS sort fixture: key order == value order")
    var arena = v.copy()
    for _ in range(len(v)):
        arena.append(0)
    var n = len(v)
    var p = Prog(arena, OP_SORT, 1, [0, n, 1, n, 0])
    for r in _both(p):
        for i in range(n):
            _require(_b(r[n + i]) == _b(ks[i]), "sort key: position " + String(i))
    card.record_list_f32("5402.sort", p.run(False))
    print("PASS 5402 sort key: -0.0 before +0.0, NaN last (the fixture separates it from a value compare)")


def _heap_sorted(arena: List[Float32], params: List[Int], cols: Int) raises -> List[Float32]:
    """The device's unit (`sort_cols_unit`, the heap sort) called directly,
    column by column: the words x_prep/host/sort.mojo must write."""
    var f = arena.copy()
    var q = List[Int32](length=STAGE_INTS, fill=0)
    for i in range(len(params)):
        q[i] = Int32(params[i])
    var fp = f.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var qp = q.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for c in range(cols):
        sort_cols_unit(c, fp, qp)
    _ = len(q)
    return f^


def check_host_sort(mut card: IdentityTrace) raises:
    """DEVIATION 5402, the host's key sort (x_prep/host/sort.mojo): the host
    runner's `sort_cols` writes the heap sort's words on every column. The
    fixture holds -0.0/+0.0, a subnormal (flushed), infinities, ties, one NaN
    word in column 0 and two NaN payloads in column 1; the two NaN payloads
    SEPARATE a sort that writes one NaN word for the suffix from the heap
    sort's own order (else VACUOUS), with and without `canon`."""
    var nan = _f(UInt32(0x7FC00000))
    var nan2 = _f(UInt32(0x7FC00123))
    var nnan = _f(UInt32(0xFFC00000))
    var inf = _f(UInt32(0x7F800000))
    var sub_ = _f(UInt32(0x00000005))
    var nsub = _f(UInt32(0x80000005))
    var c0: List[Float32] = [2.0, -0.0, nan, 0.0, -inf, sub_, 2.0, nsub, inf, nan, -3.5, 0.0, 1.0e-30, nan]
    var c1: List[Float32] = [nan2, 1.0, nan, -0.0, nnan, 5.0, nan2, 0.0, -1.0, nan, 5.0, -2.0, nan2, 3.0]
    var n = len(c0)
    var arena = List[Float32]()
    for i in range(n):
        arena.append(c0[i])
        arena.append(c1[i])
    for _ in range(2 * n):
        arena.append(0)
    var one_word_sep = False
    for cn in range(2):
        var params: List[Int] = [0, n, 2, 2 * n, cn]
        var want = _heap_sorted(arena, params, 2)
        if cn == 0:
            # a suffix of one NaN word (the first) differs from the heap sort's column 1
            var first = UInt32(0)
            var seen = False
            for i in range(n):
                var w = _b(want[2 * n + n + i])
                if (w & UInt32(0x7F800000)) == UInt32(0x7F800000) and (w & UInt32(0x007FFFFF)) != UInt32(0):
                    if not seen:
                        first = w
                        seen = True
                    elif w != first:
                        one_word_sep = True
        var p = Prog(arena, OP_SORT, 2, params)
        var got = p.run(False)
        for i in range(2 * n):
            _require(_b(got[2 * n + i]) == _b(want[2 * n + i]),
                     "host sort canon=" + String(cn) + " position " + String(i))
        card.record_list_f32("5402.host_sort.canon" + String(cn), got)
    _require(one_word_sep, "VACUOUS host sort fixture: the two NaN payloads do not separate")
    print("PASS 5402 host key sort: the heap sort's words on every column (mixed NaN payloads, canon 0 and 1)")


def check_host_power(mut card: IdentityTrace) raises:
    """The host's `pt_fit` (x_prep/host/power.mojo) writes `pt_fit_unit`'s
    lambdas: yeo-johnson over a column with negatives, zeros and NaNs (the
    Jacobian's two signs and the NaN skip) and box-cox over a positive
    column; the device's unit is called directly as the oracle."""
    var nan = _f(UInt32(0x7FC00000))
    var c0: List[Float32] = [-3.0, 0.5, nan, 2.0, -0.25, 7.5, 0.0, -1.5, nan, 11.0, 0.125, -6.0]
    var c1: List[Float32] = [0.5, 3.0, 1.25, nan, 8.0, 0.0625, 2.5, 40.0, 1.0, 0.75, nan, 5.0]
    var n = len(c0)
    for method in range(2):
        var arena = List[Float32]()
        for i in range(n):
            arena.append(c0[i] if method == 0 else c1[i])
        var st_ = len(arena)
        for _ in range(6):
            arena.append(Float32(1))  # ST rows of d = 1: a nonzero variance (row 2), not a constant column
        var lam = len(arena)
        arena.append(0)
        var params: List[Int] = [0, n, 1, method, st_, lam]
        var f = arena.copy()
        var q = List[Int32](length=STAGE_INTS, fill=0)
        for i in range(len(params)):
            q[i] = Int32(params[i])
        pt_fit_unit(0, f.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), q.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
        _ = len(q)
        var got = Prog(arena, OP_PT_FIT, 1, params).run(False)
        _require(_b(got[lam]) == _b(f[lam]), "host pt_fit method " + String(method) + ": lambda word")
        card.record_list_f32("host_power.method" + String(method), got)
    print("PASS host pt_fit: pt_fit_unit's lambdas (yeo-johnson with NaN and both signs, box-cox)")


def check_host_te(mut card: IdentityTrace) raises:
    """The host's `te_enc` (x_prep/host/target.mojo) writes `te_enc_unit`'s
    encodings: 2 features, 2 target columns, 3 folds (+ the full fit), an
    unknown code, a code past NCAT, an empty category, both the "auto" and a
    fixed smoothing; the device's unit, called per unit, is the oracle."""
    var n = 11
    var d = 2
    var T = 2
    var F = 3
    var cmax = 4
    var codes: List[Float32] = [0, 1, 2, 0, 1, 1, 0, 3, 2, 0, -1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 0, 1]
    var y: List[Float32] = [0.5, 1.0, 2.0, -1.0, 3.0, 0.25, 7.0, 0.0, 1.5, 2.5, -3.0, 4.0, 0.125, 1.0,
                            -2.0, 6.0, 0.75, 1.0, 5.0, -0.5, 2.0, 3.5]
    var fold: List[Float32] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    for sm in range(2):
        var arena = List[Float32]()
        var co = 0
        for v in codes:
            arena.append(v)
        var yo = len(arena)
        for v in y:
            arena.append(v)
        var fo = len(arena)
        for v in fold:
            arena.append(v)
        var nco = len(arena)
        arena.append(3)  # feature 0: categories 0..2 (its code 3 is past NCAT)
        arena.append(3)  # feature 1: category 3 never occurs within NCAT; category 2 is present
        var meta = len(arena)
        for k in range(2 * (F + 1) * T):
            arena.append(Float32(0.5) + Float32(k) * Float32(0.25))
        var smo = len(arena)
        arena.append(Float32(-1) if sm == 0 else Float32(2.5))
        var enc = len(arena)
        var total = (F + 1) * d * cmax * T
        for _ in range(total):
            arena.append(Float32(-9))
        var params: List[Int] = [co, n, d, yo, T, fo, cmax, nco, meta, smo, enc]
        var want = arena.copy()
        var q = List[Int32](length=STAGE_INTS, fill=0)
        for i in range(len(params)):
            q[i] = Int32(params[i])
        var wp = want.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var qp = q.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for t in range(total):
            te_enc_unit(t, wp, qp)
        _ = len(q)
        var got = Prog(arena, OP_TE_ENC, total, params).run(False)
        for i in range(total):
            _require(_b(got[enc + i]) == _b(want[enc + i]), "host te_enc smooth " + String(sm) + " unit " + String(i))
        card.record_list_f32("host_te.smooth" + String(sm), got)
    print("PASS host te_enc: te_enc_unit's encodings (folds, unknown and past-NCAT codes, auto and fixed smoothing)")


def check_empty_guard(mut card: IdentityTrace) raises:
    """DEVIATION 5403: an empty (all-NaN) column's statistics are 0, never 0/0."""
    var nan = _f(UInt32(0x7FC00000))
    _require(_b(guarded_mean(0, 0)) != _b(raw_mean(0, 0)), "VACUOUS empty fixture: 0/0 is not a NaN here")
    var arena: List[Float32] = [nan, nan, 0, 0, 0, 0, 0, 0]
    var p = Prog(arena, OP_COL_STATS, 1, [0, 2, 1, 2])
    for r in _both(p):
        for i in range(2, 8):
            _require(_b(r[i]) == UInt32(0), "empty guard: col_stats row " + String(i - 2) + " is not +0")
    card.record_list_f32("5403.empty", p.run(False))
    print("PASS 5403 empty column: every statistic +0 (the fixture separates it from 0/0)")


def check_first_max(mut card: IdentityTrace) raises:
    """DEVIATION 5404: argmax takes the FIRST maximum."""
    var v: List[Float32] = [1.0, 3.0, 3.0, 2.0]
    _require(first_max(v) != last_max(v), "VACUOUS tie fixture")
    var arena = v.copy()
    arena.append(0)
    var p = Prog(arena, OP_ARGMAX, 1, [0, 1, 4, 4])
    for r in _both(p):
        _require(Int(bitcast[DType.int32](r[4])) == first_max(v), "first max: argmax " +
                 String(Int(bitcast[DType.int32](r[4]))))
    card.record_list_f32("5404.argmax", p.run(False))
    print("PASS 5404 tie-break: first maximum (the fixture separates it from the last)")


def check_eigen_sign(mut card: IdentityTrace) raises:
    """DEVIATION 5405: each eigenvector's largest-magnitude component (first
    on a tie) is positive; eigenvalues descending."""
    # A = [[1, -2], [-2, 4]]: eigenvalue 5 along (1, -2), 0 along (2, 1)
    var arena: List[Float32] = [1.0, -2.0, -2.0, 4.0, 0, 0, 0, 0, 0, 0]
    var p = Prog(arena, OP_EIGH, 1, [0, 2, 0, 4, 6])
    for r in _both(p):
        _require(r[4] >= r[5], "eigen: eigenvalues not descending")
        for c in range(2):
            var col: List[Float32] = [r[6 + c], r[8 + c]]
            _require(largest_positive(col), "eigen sign: column " + String(c))
        var top: List[Float32] = [r[6], r[8]]
        _require(not first_positive(top), "VACUOUS eigen fixture: the top vector's first component is >= 0")
    card.record_list_f32("5405.eigh", p.run(False))
    print("PASS 5405 eigenvector sign: largest component positive (the fixture separates it from first-positive)")


def oracle_gauss(seed: Int, t: Int) -> Float32:
    """5406 restated: two splitmix64 words of the counter, their top 24 bits
    as uniforms in (0, 1] and [0, 1), Box-Muller with the portable log/cos."""
    var base = UInt64(seed) * UInt64(0x100000000) + UInt64(2 * t)
    var u1 = mul(Float32(top24(seed, t) + 1), Float32(5.9604645e-08))
    var u2 = mul(Float32(Int(splitmix(base + UInt64(1)) >> 40)), Float32(5.9604645e-08))
    return mul(sqrtf(mul(Float32(-2), logf(u1))), identical_cos(mul(Float32(6.2831855), u2)))


def check_gauss_mapping(mut card: IdentityTrace) raises:
    """DEVIATION 5406: the noise uniform is the top 24 bits of splitmix64."""
    var sep = False
    for t in range(16):
        if top24(7, t) != low24(7, t):
            sep = True
    _require(sep, "VACUOUS rng fixture")
    # mi_noise with X = 0, SCALE = 1, MABS = 1e10 writes 1e-10 * 1e10 * N(0, 1)
    var n = 16
    var arena = List[Float32](length=n + 2 + n, fill=0)
    arena[n] = 1
    arena[n + 1] = Float32(1.0e10)
    var p = Prog(arena, OP_MI_NOISE, n, [0, n, 1, n, n + 1, 7, n + 2])
    var scale = mul(Float32(1.0e-10), Float32(1.0e10))
    for r in _both(p):
        for t in range(n):
            var want = add(Float32(0), mul(scale, oracle_gauss(7, t)))
            _require(_b(r[n + 2 + t]) == _b(want), "gauss: element " + String(t))
    card.record_list_f32("5406.gauss", p.run(False))
    print("PASS 5406 rng mapping: splitmix64 top-24-bit uniforms through Box-Muller")


def check_radius_boundary(mut card: IdentityTrace) raises:
    """DEVIATION 5407: counts within nextafter(r, 0) are dist < r."""
    var x: List[Float32] = [0.0, 1.0, 2.0, 3.0]
    _require(count_strict(x, 0, 1) != count_closed(x, 0, 1), "VACUOUS radius fixture")
    # mi_cc on (x, y = x), k = 1: point 0's joint radius is 1
    var arena: List[Float32] = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0, 0, 0, 0]
    var p = Prog(arena, OP_MI_CC, 4, [0, 4, 1, 4, 1, 8])
    var c = count_strict(x, 0, 1)
    var want = add(digammaf(Float32(c)), digammaf(Float32(c)))
    for r in _both(p):
        _require(_b(r[8]) == _b(want), "radius: point 0 term " + String(r[8]))
    card.record_list_f32("5407.radius", p.run(False))
    print("PASS 5407 radius boundary: strict inside the k-th distance (the fixture separates it from <=)")


def check_operand_ftz(mut card: IdentityTrace) raises:
    """DEVIATION 5408: every operand a unit loads is flushed (ftz)."""
    var sub = _f(UInt32(0x80000100))  # a negative subnormal: flushed, -0.0
    var v: List[Float32] = [sub, 0.0]
    _require(_b(flushed_max(v)) != _b(raw_max(v)), "VACUOUS ftz fixture")
    var arena: List[Float32] = [sub, 0.0, 0, 0, 0, 0, 0, 0]
    var p = Prog(arena, OP_COL_STATS, 1, [0, 2, 1, 2])
    for r in _both(p):
        _require(_b(r[2 + 4]) == _b(flushed_max(v)), "ftz: col_stats max " + String(r[6]))
    card.record_list_f32("5408.ftz", p.run(False))
    print("PASS 5408 operand ftz: a negative subnormal loads as -0.0, so +0.0 does not beat it (the fixture separates it from the raw word)")


def check_lerp(mut card: IdentityTrace) raises:
    """DEVIATION 5409: percentiles interpolate with numpy's `_lerp` spelling."""
    var a = _f(UInt32(1037489631))
    var b = _f(UInt32(3202125258))
    var g = _f(UInt32(1064621357))
    _require(_b(numpy_lerp(a, b, g)) != _b(naive_lerp(a, b, g)), "VACUOUS lerp fixture")
    # quantile over a 2-row column at fraction g: q = [S, n, d, QF, nq, OUT, CNT]
    var arena: List[Float32] = [a, b, g, 0]
    var p = Prog(arena, OP_QUANTILE, 1, [0, 2, 1, 2, 1, 3, -1])
    for r in _both(p):
        _require(_b(r[3]) == _b(numpy_lerp(a, b, g)), "lerp: quantile " + String(r[3]))
    card.record_list_f32("5409.lerp", p.run(False))
    print("PASS 5409 percentile lerp: numpy's two-sided spelling (the fixture separates it from a + d g)")


def main() raises:
    var path = getenv("MOJOLEARN_XPREP_CARD", "/tmp/x_prep_seams.card")
    var card = IdentityTrace.to_path(path)
    card.header("x_prep seams (DEVIATIONS 5400-5409)")
    check_fold_order(card)
    check_contraction(card)
    check_sort_key(card)
    check_host_sort(card)
    check_host_power(card)
    check_host_te(card)
    check_empty_guard(card)
    check_first_max(card)
    check_eigen_sign(card)
    check_gauss_mapping(card)
    check_radius_boundary(card)
    check_operand_ftz(card)
    check_lerp(card)
    comptime if has_accelerator():
        print("PASS x_prep seams: 10 seams, host and device, card", path)
    else:
        print("PASS x_prep seams: 10 seams, host only (no accelerator), card", path)
