# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S SEAM GATE (DEVIATIONS 5000-5009, IDENTITY_PATHS rows
100-109). For each seam of x_linear/: first show the fixture SEPARATES the
pinned spelling from the unpinned one (else VACUOUS, a failure), then assert
the SHIPPED function (the one both bindings run) equals the oracle
(x_linear/checks/seams_oracle.mojo) bit for bit, and record it on the card.

    tools/with_identical_mode.sh pixi run mojo run -I . x_linear/checks/seams_check.mojo

The card goes to $MOJOLEARN_XLINEAR_CARD (default /tmp/x_linear_seams.card);
`python3 tools/identity_trace_diff.py a.card b.card` compares two boxes. Each
seam's sabotage arm is x_linear/checks/sabotage/seam_50NN_*.patch, listed in
tools/identity_lanes/linear.checks.
"""
from std.os import getenv
from std.memory import bitcast
from std.sys.info import CompilationTarget
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.numerics import ftz
from x_linear.ops import FP, IP, dot, fa, fm, fz_branchless, cholesky, jacobi_eig, shuffle, fexp, flog
from x_linear.dispatch import decision_one
from x_linear.team import team_work, solo
from x_linear.lars import lars_fit
from x_linear.isotonic import isotonic_predict
from x_linear.cd import alpha_grid_value
from x_linear.checks.seams_oracle import (
    flush, dot_pinned, dot_reversed, dot_unfused, add, add_unflushed,
    cholesky_pinned, cholesky_reversed_inner, jacobi, splitmix, shuffle_mod, shuffle_mulshift,
    argmax_abs_first, argmax_abs_last, CANONICAL_NAN_BITS,
    score_intercept_last, score_intercept_first, exp_pinned, log_pinned, exp_libm, log_libm,
    grid_pinned, grid_logspace,
)


def _p(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


def _pi(mut l: List[Int32]) -> IP:
    return IP(unsafe_from_address=Int(l.unsafe_ptr()))


def _u(mut l: List[Float32]) -> MutPointer[Float32, MutUntrackedOrigin]:
    return l.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _require(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL: " + what)


def _values(n: Int, seed: UInt64, spread: Bool) -> List[Float32]:
    """Deterministic non-uniform float32s: splitmix bits to [-1, 1), and
    (spread) a magnitude from 2^-12 to 2^12 so the folds can disagree."""
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        var r = splitmix(s)
        var u = Float32(Int(r >> 40)) / Float32(1 << 24) * 2 - 1
        if spread:
            var e = Int((r >> 8) & 31) - 16
            var sc = Float32(1)
            if e > 0:
                for _ in range(e // 2):
                    sc = sc * 2
            else:
                for _ in range((-e) // 2):
                    sc = sc / 2
            u = u * sc
        out.append(u)
    return out^


def check_dot(mut card: IdentityTrace) raises:
    """DEVIATION 5000: every x_linear reduction folds j ascending, one fused
    multiply-add per term (ops.dot / row_dot)."""
    var a = List[Float32]()
    var b = List[Float32]()
    var found = False
    for seed in range(1, 200):
        a = _values(37, UInt64(10 + 2 * seed), False)
        b = _values(37, UInt64(11 + 2 * seed), False)
        var w = dot_pinned(a, b)
        if _bits(w) != _bits(dot_reversed(a, b)) and _bits(w) != _bits(dot_unfused(a, b)):
            found = True
            break
    _require(found, "VACUOUS dot fixture: no seed separates the ascending fused fold from both others")
    var want = dot_pinned(a, b)
    var got = List[Float32](length=1, fill=0)
    got[0] = dot(_p(a), 0, _p(b), 0, 37)
    _require(_bits(got[0]) == _bits(want), "dot: shipped fold != ascending fused fold")
    card.record_host("x_linear.seam5000.dot", _u(got), 1)
    print("PASS 5000 dot: ascending fused fold (fixture separates descending and unfused)")


def check_flush(mut card: IdentityTrace) raises:
    """DEVIATION 5001: every operand and result passes through ftz."""
    var a = bitcast[DType.float32](UInt32(0x00400000))  # subnormal
    var b = bitcast[DType.float32](UInt32(0x00200001))  # subnormal
    _require(_bits(add(a, b)) != _bits(add_unflushed(a, b)), "VACUOUS flush fixture: flushed and raw sums agree")
    var got = List[Float32](length=2, fill=0)
    got[0] = fa(a, b)
    got[1] = fm(bitcast[DType.float32](UInt32(0x3F800000)), a)
    _require(_bits(got[0]) == _bits(add(a, b)), "flush: shipped add kept a subnormal operand")
    _require(_bits(got[1]) == _bits(flush(a)), "flush: shipped product kept a subnormal operand")
    # the device's branchless flush (speed phase) is ftz's word on every edge
    var edges: List[UInt32] = [
        UInt32(0x00000000), UInt32(0x80000000), UInt32(0x00000001), UInt32(0x80000001),
        UInt32(0x007FFFFF), UInt32(0x807FFFFF), UInt32(0x00400000), UInt32(0x00800000),
        UInt32(0x80800000), UInt32(0x3F800000), UInt32(0xBF800000), UInt32(0x7F7FFFFF),
        UInt32(0x7F800000), UInt32(0xFF800000), UInt32(0x7FC00000), UInt32(0xFFC00001),
    ]
    for e in edges:
        var v = bitcast[DType.float32](e)
        _require(_bits(fz_branchless(v)) == _bits(ftz(v)), "flush: the branchless flush and ftz give different words")
    card.record_host("x_linear.seam5001.flush", _u(got), 2)
    print("PASS 5001 flush: subnormal operands are signed zeros")


def _spd(m: Int, seed: UInt64) -> List[Float32]:
    var r = _values(m * m, seed, False)
    var a = List[Float32](length=m * m, fill=0)
    for i in range(m):
        for j in range(m):
            var acc = Float32(0)
            for k in range(m):
                acc = add(acc, flush(r[i * m + k] * r[j * m + k]))
            a[i * m + j] = acc
        a[i * m + i] = add(a[i * m + i], Float32(0.5))
    for i in range(m):
        for j in range(i):
            a[i * m + j] = a[j * m + i]
    return a^


def check_cholesky(mut card: IdentityTrace) raises:
    """DEVIATION 5002: Cholesky column j ascending, inner sums k ascending."""
    var m = 9
    var a = _spd(m, 21)
    var want = a.copy()
    cholesky_pinned(want, m)
    var alt = a.copy()
    cholesky_reversed_inner(alt, m)
    var sep = False
    for i in range(m * m):
        if _bits(want[i]) != _bits(alt[i]):
            sep = True
    _require(sep, "VACUOUS cholesky fixture: ascending and descending inner sums agree")
    var got = a.copy()
    _require(cholesky(_p(got), 0, m), "cholesky: shipped factor refused an SPD matrix")
    for i in range(m):
        for j in range(i + 1):
            _require(_bits(got[i * m + j]) == _bits(want[i * m + j]), "cholesky: L[" + String(i) + "," + String(j) + "]")
    card.record_host("x_linear.seam5002.cholesky", _u(got), m * m)
    print("PASS 5002 cholesky: column-ascending, inner-ascending factor")


def check_jacobi(mut card: IdentityTrace) raises:
    """DEVIATION 5003: cyclic Jacobi, p ascending then q ascending."""
    var m = 6
    var a = _spd(m, 31)
    var wa = a.copy()
    var wv = List[Float32](length=m * m, fill=0)
    jacobi(wa, wv, m, 60, False)
    var xa = a.copy()
    var xv = List[Float32](length=m * m, fill=0)
    jacobi(xa, xv, m, 60, True)
    var sep = False
    for i in range(m * m):
        if _bits(wv[i]) != _bits(xv[i]):
            sep = True
    _require(sep, "VACUOUS jacobi fixture: q ascending and q descending sweeps agree")
    var ga = a.copy()
    var gv = List[Float32](length=m * m, fill=0)
    jacobi_eig(_p(ga), 0, _p(gv), 0, m, 60)
    for i in range(m * m):
        _require(_bits(ga[i]) == _bits(wa[i]), "jacobi: eigenvalue matrix entry " + String(i))
        _require(_bits(gv[i]) == _bits(wv[i]), "jacobi: eigenvector entry " + String(i))
    card.record_host("x_linear.seam5003.jacobi", _u(gv), m * m)
    print("PASS 5003 jacobi: p-ascending, q-ascending sweep")


def check_shuffle(mut card: IdentityTrace) raises:
    """DEVIATION 5004: splitmix64 and Fisher-Yates with j = draw mod (i + 1)."""
    var n = 97
    var want = shuffle_mod(n, 12345)
    var alt = shuffle_mulshift(n, 12345)
    var sep = False
    for i in range(n):
        if want[i] != alt[i]:
            sep = True
    _require(sep, "VACUOUS shuffle fixture: mod and multiply-shift agree")
    var idx = List[Int32](length=n, fill=0)
    for i in range(n):
        idx[i] = Int32(i)
    var s = UInt64(12345)
    shuffle(_pi(idx), n, s)
    var rec = List[Float32](length=n, fill=0)
    for i in range(n):
        _require(Int(idx[i]) == want[i], "shuffle: position " + String(i))
        rec[i] = Float32(Int(idx[i]))
    card.record_host("x_linear.seam5004.shuffle", _u(rec), n)
    print("PASS 5004 shuffle: splitmix64 Fisher-Yates, mod mapping")


def check_tie(mut card: IdentityTrace) raises:
    """DEVIATION 5005: an exact tie goes to the lowest index (LARS's
    argmax|Cov|; the CV selections and the class argmax use the same rule)."""
    var n = 40
    var d = 4
    var x = List[Float32](length=n * d, fill=0)
    var y = List[Float32](length=n, fill=0)
    var r = _values(n * 2, 41, False)
    for i in range(n):
        x[i * d + 0] = r[i] * Float32(0.25)
        x[i * d + 1] = r[n + i]
        x[i * d + 2] = r[i] * Float32(0.125)
        x[i * d + 3] = -r[n + i]
        y[i] = r[n + i]
    var xty = List[Float32](length=d, fill=0)
    for j in range(d):
        var acc = Float32(0)
        for i in range(n):
            acc = flush(acc + flush(x[i * d + j] * y[i]))
        xty[j] = acc
    _require(argmax_abs_first(xty) != argmax_abs_last(xty), "VACUOUS tie fixture: first and last max agree")
    var ip: List[Int32] = [Int32(1), Int32(0), Int32(0)]
    var fp: List[Float32] = [Float32(0)]
    var res = List[Float32](length=2 * d + 4, fill=0)
    var fw = List[Float32](length=2 * d * d + 8 * d, fill=0)
    var iw = List[Int32](length=2 * d, fill=0)
    var tw = List[Float32](length=team_work(n, 0, 0), fill=0)
    lars_fit(solo(_p(tw), n, 0, 0), _p(x), _p(y), n, d, _pi(ip), _p(fp), _p(res), _p(fw), _pi(iw))
    _require(Int(res[d + 4]) == argmax_abs_first(xty), "tie: LARS's first active feature is not the lowest tied index")
    card.record_host("x_linear.seam5005.tie", _u(res), 2 * d + 4)
    print("PASS 5005 tie-break: the lowest index among exact ties")


def check_nan(mut card: IdentityTrace) raises:
    """DEVIATION 5006: an out-of-bounds isotonic prediction is the constant
    word 0x7FC00000, never a computed NaN (IDENTITY_PATHS Clause B)."""
    # the unpinned spelling is whatever NaN the target computes: x86's 0/0
    # is 0xFFC00000, NVIDIA's 0x7FFFFFFF; an Arm host's IS the canonical word,
    # so the fixture's separation is stated against x86's word, a constant
    _require(CANONICAL_NAN_BITS != UInt32(0xFFC00000), "VACUOUS nan fixture")
    var zero = List[Float32](length=1, fill=0)
    var computed = zero[0] / zero[0]
    comptime if CompilationTarget.is_x86():
        _require(_bits(computed) != CANONICAL_NAN_BITS,
                 "VACUOUS nan fixture: this x86 host's computed NaN is the canonical word")
    var q: List[Float32] = [Float32(-5), Float32(0.5), Float32(9)]
    var th: List[Float32] = [Float32(0), Float32(1), Float32(0), Float32(2)]
    var ip: List[Int32] = [Int32(2), Int32(0)]
    var fp: List[Float32] = [Float32(0), Float32(1)]
    var res = List[Float32](length=3, fill=0)
    var fw = List[Float32](length=1, fill=0)
    var iw = List[Int32](length=1, fill=0)
    var tw = List[Float32](length=team_work(3, 0, 0), fill=0)
    isotonic_predict(solo(_p(tw), 3, 0, 0), _p(q), _p(th), 3, 1, _pi(ip), _p(fp), _p(res), _p(fw), _pi(iw))
    _require(_bits(res[0]) == CANONICAL_NAN_BITS and _bits(res[2]) == CANONICAL_NAN_BITS,
             "nan: an out-of-bounds prediction is not the canonical word")
    _require(res[1] == Float32(1), "nan: the in-range prediction moved")
    card.record_host("x_linear.seam5006.nan", _u(res), 3)
    print("PASS 5006 nan: out of bounds writes 0x7FC00000")


def check_intercept(mut card: IdentityTrace) raises:
    """DEVIATION 5007: a score is the fused fold of x.w, j ascending, and
    the intercept joins LAST."""
    var d = 13
    var found = False
    var x = List[Float32]()
    var w = List[Float32]()
    var b = Float32(0)
    for seed in range(1, 200):
        x = _values(d, UInt64(100 + seed), True)
        w = _values(d, UInt64(400 + seed), True)
        b = Float32(1000.37) + Float32(seed)
        if _bits(score_intercept_last(x, w, b)) != _bits(score_intercept_first(x, w, b)):
            found = True
            break
    _require(found, "VACUOUS intercept fixture: intercept-first and intercept-last folds agree")
    var wb = w.copy()
    wb.append(b)
    var got = List[Float32](length=1, fill=0)
    got[0] = decision_one(_p(x), 0, d, _p(wb), 0, 0)
    _require(_bits(got[0]) == _bits(score_intercept_last(x, w, b)), "intercept: shipped score != fold then intercept")
    card.record_host("x_linear.seam5007.intercept", _u(got), 1)
    print("PASS 5007 intercept: joins after the fold")


def check_transcendentals(mut card: IdentityTrace) raises:
    """DEVIATION 5008: exp and log in every fit are the portable spellings
    (checks/numerics.mojo), never the target's libm."""
    var xs = List[Float32]()
    var sep_e = False
    var sep_l = False
    var s = UInt64(77)
    for _ in range(4000):
        var r = splitmix(s)
        var u = Float32(Int(r >> 40)) / Float32(1 << 24)
        var xe = u * 60 - 30
        var xl = u * 1000 + Float32(1e-3)
        if _bits(exp_pinned(xe)) != _bits(exp_libm(xe)):
            sep_e = True
        if _bits(log_pinned(xl)) != _bits(log_libm(xl)):
            sep_l = True
        xs.append(xe)
        xs.append(xl)
    _require(sep_e, "VACUOUS exp fixture: portable and libm exp agree on 4000 inputs")
    _require(sep_l, "VACUOUS log fixture: portable and libm log agree on 4000 inputs")
    var got = List[Float32](length=len(xs), fill=0)
    for i in range(len(xs) // 2):
        var e = fexp(xs[2 * i])
        var l = flog(xs[2 * i + 1])
        _require(_bits(e) == _bits(exp_pinned(xs[2 * i])), "exp: shipped != portable at input " + String(i))
        _require(_bits(l) == _bits(log_pinned(xs[2 * i + 1])), "log: shipped != portable at input " + String(i))
        got[2 * i] = e
        got[2 * i + 1] = l
    card.record_host("x_linear.seam5008.transcendentals", _u(got), len(got))
    print("PASS 5008 transcendentals: portable exp and log (fixture separates libm)")


def check_grid(mut card: IdentityTrace) raises:
    """DEVIATION 5009: the CV alpha grid is alpha_max * exp(frac * log eps)."""
    var a_n = 100
    var amax = Float32(3.7123)
    var eps = Float32(1e-3)
    var sep = False
    var got = List[Float32](length=a_n, fill=0)
    for k in range(a_n):
        var want = grid_pinned(amax, eps, k, a_n)
        if _bits(want) != _bits(grid_logspace(amax, eps, k, a_n)):
            sep = True
        got[k] = alpha_grid_value(amax, eps, k, a_n)
        _require(_bits(got[k]) == _bits(want), "grid: alpha " + String(k))
    _require(sep, "VACUOUS grid fixture: the scaled-exp and log-interpolation spellings agree on all 100 alphas")
    card.record_host("x_linear.seam5009.grid", _u(got), a_n)
    print("PASS 5009 grid: exp(frac * log eps) spelling")


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var path = getenv("MOJOLEARN_XLINEAR_CARD", "/tmp/x_linear_seams.card")
    var card = IdentityTrace.to_path(path)
    card.header("x_linear seams (DEVIATIONS 5000-5009)")
    check_dot(card)
    check_flush(card)
    check_cholesky(card)
    check_jacobi(card)
    check_shuffle(card)
    check_tie(card)
    check_nan(card)
    check_intercept(card)
    check_transcendentals(card)
    check_grid(card)
    print("PASS x_linear seams: 10 seams, card", path)
