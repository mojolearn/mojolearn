# SPDX-License-Identifier: Apache-2.0
# Lane ag-rankn-fold (docs/plans/AUTOGRAD_RANKN.md 2.1, 2.2, 2.7): checks of
# the rank-N broadcast binary ops, axis-set sums and means, max and softmax
# along an axis, on any executor (HostExec, Metal, NVIDIA, AMD). Each case
# prints one `RANKN fold <case> digest <d>` line; the digests must be equal
# on every executor. Bit equality (_same) where the result is a composition
# of 2-D tape ops or an exact host chain, the tolerance form (check_values)
# against a host oracle written elementwise in Mojo otherwise (never a
# NumPy product). Shapes are small, odd and off any board row: one-leaf
# folds (R <= 64), multi-leaf folds and folds with a tree level (R > 4096).
from std.memory import bitcast
from std.math import nan
from std.collections import InlineArray
from autograd.tape import Tape, Tensor
from autograd.ops import ADD, SUBTRACT, MULTIPLY, DIVIDE, AXIS_COLS
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_fold import (
    rankn_binary, rankn_sum, rankn_mean, rankn_max_argmax, rankn_softmax, RankNFoldRules,
)
from autograd.checks.support import check_values
from sequence.exec_trait import Exec
from sequence.ops import FP, add


def _data(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _positive(n: Int, salt: Int) -> List[Float32]:
    """Data in [0.5, 1.7): a safe divisor."""
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(0.5) + Float32((i * 7919 + salt * 104729) % 1000) * Float32(0.0012)
    return v^


def _digest(v: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _mix(d: UInt64, e: UInt64) -> UInt64:
    return (d ^ e) * UInt64(1099511628211)


def _same(name: String, got: List[Float32], want: List[Float32]) raises -> UInt64:
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")
    return _digest(got)


def _shape3(a: Int, b: Int, c: Int) -> ShapeN:
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    d[0] = a
    d[1] = b
    d[2] = c
    return ShapeN(3, d)


def _bcast_host(v: List[Float32], s: ShapeN, o: ShapeN) -> List[Float32]:
    """v of rank-3 shape s broadcast to rank-3 shape o (host copy of words)."""
    var r = List[Float32](length=o.count(), fill=Float32(0))
    for i in range(o.d[0]):  # small-loop(check data, host side)
        for j in range(o.d[1]):  # small-loop(check data, host side)
            for k in range(o.d[2]):  # small-loop(check data, host side)
                var si = 0 if s.d[0] == 1 else i
                var sj = 0 if s.d[1] == 1 else j
                var sk = 0 if s.d[2] == 1 else k
                r[(i * o.d[1] + j) * o.d[2] + k] = v[(si * s.d[1] + sj) * s.d[2] + sk]
    return r^


# ---------------------------------------------------------------- binary
def _bin_case[E: Exec](mut ex: E, mut tape: Tape, op: Int, so: ShapeN, sy: ShapeN, trailing: Bool) raises -> UInt64:
    """x [so] op y [sy] (a middle-axis or non-matrix broadcast): forward vs
    the same-shape op on y broadcast on the host; dx vs the same-shape VJP
    expression; dy vs Tape.sum(view, axis=1) of the elementwise VJP when the
    broadcast axes trail (`trailing`), else a host oracle (tolerance)."""
    var name = String("bin/") + String(op) + "/" + String(sy.d[0]) + "x" + String(sy.d[1]) + "x" + String(sy.d[2])
    var n = so.count()
    var xv = _data(n, 1 + op)
    var yv = _positive(sy.count(), 2 + op)
    var gv = _data(n, 3 + op)
    var yfull = _bcast_host(yv, sy, so)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, so, True)
    var y = tape.tensor_nd(ex, yv, sy, True)
    var z = rankn_binary(tape, ex, x, y, op)
    var seed = tape.tensor_nd(ex, gv, so, False)
    tape.backward_with_rules(ex, z, seed, RankNFoldRules())
    var zv = tape.values(ex, z)
    var dx = tape.gradients(ex, x)
    var dy = tape.gradients(ex, y)
    # references on a fresh graph: same-shape 2-D ops on the broadcast copy
    tape.reset()
    var x2 = tape.tensor_nd(ex, xv, so, False)
    var yf = tape.tensor_nd(ex, yfull, so, False)
    var g2 = tape.tensor_nd(ex, gv, so, False)
    var d = _same(name + "/value", zv, tape.values(ex, tape._binary(ex, x2, yf, op)))
    if op == ADD:
        d = _mix(d, _same(name + "/dx", dx, gv))
    elif op == MULTIPLY:
        d = _mix(d, _same(name + "/dx", dx, tape.values(ex, tape.multiply(ex, g2, yf))))
    elif op == DIVIDE:
        d = _mix(d, _same(name + "/dx", dx, tape.values(ex, tape.divide(ex, g2, yf))))
    else:
        d = _mix(d, _same(name + "/dx", dx, gv))
    if trailing and (op == ADD or op == MULTIPLY):
        var terms = g2 if op == ADD else tape.multiply(ex, g2, x2)
        var O = sy.count()
        var v = tape.view_nd(ex, terms, ShapeN.matrix(O, n // O))
        d = _mix(d, _same(name + "/dy-eq2d", dy, tape.values(ex, tape.sum(ex, v, AXIS_COLS))))
    else:
        # host oracle: dy[s] = sum over the outputs that read s of the VJP term
        var want = List[Float64](length=sy.count(), fill=Float64(0))
        for i in range(so.d[0]):  # small-loop(check oracle, host side)
            for j in range(so.d[1]):  # small-loop(check oracle, host side)
                for k in range(so.d[2]):  # small-loop(check oracle, host side)
                    var t = (i * so.d[1] + j) * so.d[2] + k
                    var si = 0 if sy.d[0] == 1 else i
                    var sj = 0 if sy.d[1] == 1 else j
                    var sk = 0 if sy.d[2] == 1 else k
                    var s = (si * sy.d[1] + sj) * sy.d[2] + sk
                    var g = Float64(gv[t])
                    var term: Float64
                    if op == ADD:
                        term = g
                    elif op == SUBTRACT:
                        term = -g
                    elif op == MULTIPLY:
                        term = g * Float64(xv[t])
                    else:
                        var yy = Float64(yv[s])
                        term = -g * Float64(xv[t]) / (yy * yy)
                    want[s] += term
        var w32 = List[Float32](length=len(want), fill=Float32(0))
        for i in range(len(want)):  # small-loop(check oracle, host side)
            w32[i] = Float32(want[i])
        check_values(name + "/dy-oracle", dy, w32)
        d = _mix(d, _digest(dy))
    print("RANKN fold", name, "digest", d)
    return d


# ---------------------------------------------------------------- sums
def _sum_perm_case[E: Exec](mut ex: E, mut tape: Tape, sx: ShapeN, mean: Bool) raises -> UInt64:
    """sum / mean of x [a, b, c] over axes (0, 2) (the rank-N fold) vs
    permute(x, (1, 0, 2)) materialised, viewed (b, a c) and summed along
    axis 1 by the tape: the same terms in the same order, so the same
    words; the value and x's gradient."""
    var name = String("sum02/") + ("mean/" if mean else "sum/") + String(sx.d[0]) + "x" + String(sx.d[1]) + "x" + String(sx.d[2])
    var n = sx.count()
    var b = sx.d[1]
    var xv = _data(n, 11)
    var gv = _data(b, 12)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var axes: List[Int] = [0, 2]
    var r = rankn_mean(tape, ex, x, axes) if mean else rankn_sum(tape, ex, x, axes)
    tape.backward_with_rules(ex, r, tape.tensor_nd(ex, gv, tape.shape_of(r), False), RankNFoldRules())
    var rv = tape.values(ex, r)
    var dx = tape.gradients(ex, x)
    tape.reset()
    var x2 = tape.tensor_nd(ex, xv, sx, True)
    var perm: List[Int] = [1, 0, 2]
    var p = tape.permute(ex, x2, perm)
    tape.materialize(ex, p)
    var v = tape.view_nd(ex, p, ShapeN.matrix(b, n // b))
    var r2 = tape.mean(ex, v, AXIS_COLS) if mean else tape.sum(ex, v, AXIS_COLS)
    tape.backward_with_rules(ex, r2, tape.tensor(ex, gv, b, 1), RankNFoldRules())
    var d = _same(name + "/value", rv, tape.values(ex, r2))
    d = _mix(d, _same(name + "/dx", dx, tape.gradients(ex, x2)))
    print("RANKN fold", name, "digest", d)
    return d


def _sum_middle_case[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """sum over the middle axis of [3, 4, 5] (R = 4, one leaf) vs the exact
    host chain (+0.0 then add ascending, ftz), and its gradient (the
    upstream broadcast back)."""
    var sx = _shape3(3, 4, 5)
    var xv = _data(60, 21)
    var gv = _data(15, 22)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var axes: List[Int] = [1]
    var r = rankn_sum(tape, ex, x, axes)
    tape.backward_with_rules(ex, r, tape.tensor_nd(ex, gv, tape.shape_of(r), False), RankNFoldRules())
    var want = List[Float32](length=15, fill=Float32(0))
    var wdx = List[Float32](length=60, fill=Float32(0))
    for i in range(3):  # small-loop(check oracle, host side)
        for k in range(5):  # small-loop(check oracle, host side)
            var s = Float32(0)
            for j in range(4):  # small-loop(check oracle, host side)
                s = add(s, xv[(i * 4 + j) * 5 + k])
                wdx[(i * 4 + j) * 5 + k] = gv[i * 5 + k]
            want[i * 5 + k] = add(Float32(0), s)
    var d = _same("sum1/value", tape.values(ex, r), want)
    d = _mix(d, _same("sum1/dx", tape.gradients(ex, x), wdx))
    print("RANKN fold sum1/3x4x5 digest", d)
    return d


def _sum_routes_case[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """Routed reductions keep the tape's words: a trailing block (1, 2) of
    [3, 4, 5] equals Tape.sum(view (3, 20), axis 1); the leading block (0, 1)
    equals Tape.sum(matrix view, axis 0); every axis equals Tape.sum."""
    var sx = _shape3(3, 4, 5)
    var xv = _data(60, 31)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, False)
    var a12: List[Int] = [1, 2]
    var a01: List[Int] = [0, 1]
    var none = List[Int]()
    var d = _same("routes/trailing", tape.values(ex, rankn_sum(tape, ex, x, a12)),
                  tape.values(ex, tape.sum(ex, tape.view_nd(ex, x, ShapeN.matrix(3, 20)), AXIS_COLS)))
    d = _mix(d, _same("routes/leading", tape.values(ex, rankn_sum(tape, ex, x, a01)),
                      tape.values(ex, tape.sum(ex, x, 0))))
    d = _mix(d, _same("routes/all", tape.values(ex, rankn_mean(tape, ex, x, none)),
                      tape.values(ex, tape.mean(ex, x, -1))))
    print("RANKN fold routes/3x4x5 digest", d)
    return d


# ---------------------------------------------------------------- max
def _max_data(n: Int, salt: Int, with_nan: Bool) -> List[Float32]:
    """Few distinct values (many ties); NaN at a few positions when asked."""
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 37 + salt * 11) % 7) - 3) * Float32(0.25)
        if with_nan and (i * 13 + salt) % 97 == 0:
            v[i] = nan[DType.float32]()
    return v^


def _beats(v: Float32, i: Int, bv: Float32, bi: Int) -> Bool:
    var vn = v != v
    var bn = bv != bv
    if vn and bn:
        return i < bi
    if vn:
        return True
    if bn:
        return False
    if v > bv:
        return True
    return v == bv and i < bi


def _max_case[E: Exec](mut ex: E, mut tape: Tape, sx: ShapeN, axes: List[Int], with_nan: Bool) raises -> UInt64:
    """max and argmax of x [a, b, c] over `axes` (a subset of {0, 1, 2}) vs
    a host scan in the reduced axes' row-major order with the same combine;
    the gradient is the upstream at the argmax and zero elsewhere."""
    var name = String("max/") + String(sx.d[0]) + "x" + String(sx.d[1]) + "x" + String(sx.d[2]) + "/" + String(len(axes)) + ("/nan" if with_nan else "")
    var red = InlineArray[Bool, 3](fill=False)
    for k in range(len(axes)):  # small-loop(check axes)
        red[axes[k]] = True
    var n = sx.count()
    var xv = _max_data(n, sx.d[1] + len(axes), with_nan)
    var ko = InlineArray[Int, 3](fill=1)
    var ro = InlineArray[Int, 3](fill=1)
    for k in range(3):  # small-loop(check axes)
        if red[k]:
            ro[k] = sx.d[k]
        else:
            ko[k] = sx.d[k]
    var O = ko[0] * ko[1] * ko[2]
    var R = n // O
    var gv = _data(O, 41)
    tape.reset()
    var x = tape.tensor_nd(ex, xv, sx, True)
    var res = rankn_max_argmax(tape, ex, x, axes)
    var m = res[0]
    tape.backward_with_rules(ex, m, tape.tensor_nd(ex, gv, tape.shape_of(m), False), RankNFoldRules())
    var got = tape.values(ex, m)
    var gotarg = List[Float32](length=O, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(gotarg.unsafe_ptr())), res[1], O)
    var dx = tape.gradients(ex, x)
    var wdx = List[Float32](length=n, fill=Float32(0))
    var d = UInt64(1469598103934665603)
    for o in range(O):  # small-loop(check oracle, host side)
        var ok = o
        var c = InlineArray[Int, 3](fill=0)
        var k = 2
        while k >= 0:  # small-loop(check axes)
            c[k] = ok % ko[k]
            ok //= ko[k]
            k -= 1
        var bv = Float32(0)
        var bi = -1
        var bx = 0
        for r in range(R):  # small-loop(check oracle, host side)
            var rr = r
            var e = InlineArray[Int, 3](fill=0)
            var q = 2
            while q >= 0:  # small-loop(check axes)
                e[q] = rr % ro[q]
                rr //= ro[q]
                q -= 1
            var xi = ((c[0] + e[0]) * sx.d[1] + (c[1] + e[1])) * sx.d[2] + (c[2] + e[2])
            var v = xv[xi]
            if bi < 0 or _beats(v, r, bv, bi):
                bv = v
                bi = r
                bx = xi
        var ga = Int(bitcast[DType.int32](gotarg[o]))
        if ga != bi:
            print("FAIL", name, "output", o, "argmax", ga, "want", bi)
            raise Error(name + ": argmax differs")
        var both_nan = (got[o] != got[o]) and (bv != bv)
        if not both_nan and bitcast[DType.uint32](got[o]) != bitcast[DType.uint32](bv):
            print("FAIL", name, "output", o, "got", got[o], "want", bv)
            raise Error(name + ": max differs")
        wdx[bx] = gv[o]
        d = _mix(d, UInt64(bi))
    d = _mix(d, _same(name + "/dx", dx, wdx))
    print("RANKN fold", name, "digest", d)
    return d


# ---------------------------------------------------------------- softmax
def _softmax_case[E: Exec](mut ex: E, mut tape: Tape, log: Bool) raises -> UInt64:
    """softmax (log) of [4, 6, 5] along axis 1 vs the tape's softmax of the
    host-transposed [4, 5, 6] tensor: the same words at transposed
    positions, value and gradient."""
    var name = String("softmax1/") + ("log" if log else "plain")
    var A = 4
    var B = 6
    var C = 5
    var n = A * B * C
    var xv = _data(n, 51)
    var gv = _data(n, 52)
    var xt = List[Float32](length=n, fill=Float32(0))
    var gt = List[Float32](length=n, fill=Float32(0))
    for i in range(A):  # small-loop(check data, host side)
        for j in range(B):  # small-loop(check data, host side)
            for k in range(C):  # small-loop(check data, host side)
                xt[(i * C + k) * B + j] = xv[(i * B + j) * C + k]
                gt[(i * C + k) * B + j] = gv[(i * B + j) * C + k]
    tape.reset()
    var x = tape.tensor_nd(ex, xv, _shape3(A, B, C), True)
    var s = rankn_softmax(tape, ex, x, 1, log)
    tape.materialize(ex, s)
    tape.backward_with_rules(ex, s, tape.tensor_nd(ex, gv, _shape3(A, B, C), False), RankNFoldRules())
    var sv = tape.values(ex, s)
    var dx = tape.gradients(ex, x)
    tape.reset()
    var x2 = tape.tensor(ex, xt, A * C, B, True)
    var s2 = tape.log_softmax(ex, x2) if log else tape.softmax(ex, x2)
    tape.backward_from(ex, s2, tape.tensor(ex, gt, A * C, B))
    var rv = tape.values(ex, s2)
    var rdx = tape.gradients(ex, x2)
    var want = List[Float32](length=n, fill=Float32(0))
    var wdx = List[Float32](length=n, fill=Float32(0))
    for i in range(A):  # small-loop(check data, host side)
        for j in range(B):  # small-loop(check data, host side)
            for k in range(C):  # small-loop(check data, host side)
                want[(i * B + j) * C + k] = rv[(i * C + k) * B + j]
                wdx[(i * B + j) * C + k] = rdx[(i * C + k) * B + j]
    var d = _same(name + "/value", sv, want)
    d = _mix(d, _same(name + "/dx", dx, wdx))
    print("RANKN fold", name, "digest", d)
    return d


def run_rankn_fold_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    var cases = 0
    var so = _shape3(3, 4, 5)
    var ops: List[Int] = [ADD, SUBTRACT, MULTIPLY, DIVIDE]
    for i in range(len(ops)):  # small-loop(4: check ops)
        # [3,4,5] op [3,1,5]: a middle-axis broadcast (host oracle for dy)
        summary = _mix(summary, _bin_case(ex, tape, ops[i], so, _shape3(3, 1, 5), False))
        # [3,4,5] op [3,1,1]: broadcast axes trail (EQ2D for ADD, MULTIPLY)
        summary = _mix(summary, _bin_case(ex, tape, ops[i], so, _shape3(3, 1, 1), True))
        cases += 2
    # multi-leaf fold with a tree level (R = 5600): trailing (EQ2D) and middle
    var big = _shape3(3, 70, 80)
    summary = _mix(summary, _bin_case(ex, tape, MULTIPLY, big, _shape3(3, 1, 1), True))
    summary = _mix(summary, _bin_case(ex, tape, ADD, _shape3(70, 3, 80), _shape3(1, 3, 1), False))
    summary = _mix(summary, _bin_case(ex, tape, DIVIDE, _shape3(5, 9, 130), _shape3(1, 9, 1), False))
    cases += 3
    summary = _mix(summary, _sum_middle_case(ex, tape))
    summary = _mix(summary, _sum_routes_case(ex, tape))
    cases += 2
    # (0, 2) reductions: R = 15 (one leaf), 5600 (a tree level), 1170 (leaves)
    var sums = [_shape3(3, 4, 5), _shape3(70, 3, 80), _shape3(9, 4, 130)]
    for i in range(len(sums)):  # small-loop(3: check shapes)
        summary = _mix(summary, _sum_perm_case(ex, tape, sums[i], False))
        summary = _mix(summary, _sum_perm_case(ex, tape, sums[i], True))
        cases += 2
    var a1: List[Int] = [1]
    var a02: List[Int] = [0, 2]
    var a12: List[Int] = [1, 2]
    var a012: List[Int] = [0, 1, 2]
    summary = _mix(summary, _max_case(ex, tape, _shape3(5, 7, 6), a1, False))
    summary = _mix(summary, _max_case(ex, tape, _shape3(5, 7, 6), a02, False))
    summary = _mix(summary, _max_case(ex, tape, _shape3(5, 7, 6), a1, True))
    summary = _mix(summary, _max_case(ex, tape, _shape3(3, 70, 80), a12, False))
    summary = _mix(summary, _max_case(ex, tape, _shape3(3, 70, 80), a12, True))
    summary = _mix(summary, _max_case(ex, tape, _shape3(9, 4, 130), a012, False))
    cases += 6
    summary = _mix(summary, _softmax_case(ex, tape, False))
    summary = _mix(summary, _softmax_case(ex, tape, True))
    cases += 2
    print("PASS rankn fold cases", cases, "summary", summary)
