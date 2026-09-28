# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CURVE EPILOGUE ON THE HOST (lane metrics-apple2, 2026-09-28).

DEVIATION 6106 puts what follows the O(n) device work in IEEE binary64 on
the host: for the ROC and precision-recall scores that is O(n) too (one
term per curve point), and in Python it made a Python float per point,
several times over (the largest cost left in roc_auc_score, the curves and
average_precision_score on the Apple board). These functions are the SAME
binary64 operations, in the same order, as the Python they stand in for in
python/mojolearn/_expansion_metrics.py (`_drop_collinear`, `_trapezoid`,
`_binary_auc`, `_binary_ap`, `roc_curve`), run over the Float32 curve words
in the caller's arena:

- + - / are correctly rounded binary64 operations, as in Python; every
  product is `pinned_mul_f64` (never fused into a neighboring add or
  subtract, whatever the build's contraction mode), as Python never fuses;
- the Float32 words widen exactly to binary64, as Python's float() does;
- `fsum` is CPython's `math.fsum` (Modules/mathmodule.c, Shewchuk's
  partials with the half-even fix-up), step for step, and returns the
  correctly rounded sum, which is unique, so it is `math.fsum`'s value;
- the collinear-drop rule compares the same binary64 differences.

A case the Python path handles by a warning or an exception (an empty
class, a zero denominator) is never sent here: the Python checks it first,
and any doubt raises so the caller falls back to its Python path.
"""
from std.memory import bitcast
from checks.numerics import pinned_mul_f64
from x_metrics.common import FP


@always_inline
def _w(a: Int, i: Int) -> Float64:
    """Float32 word i of the arena at address a, widened exactly."""
    return Float64(FP(unsafe_from_address=a).unsafe_load(i))


@always_inline
def _word(a: Int, i: Int) -> Int32:
    return bitcast[DType.int32](FP(unsafe_from_address=a).unsafe_load(i))


@no_inline
def fsum(vals: List[Float64]) -> Float64:
    """CPython's math.fsum over finite values (and the module's `_fsum`
    wrapper: a zero sum is +0.0)."""
    var p = List[Float64]()
    for k in range(len(vals)):
        var x = vals[k]
        var i = 0
        for j in range(len(p)):
            var y = p[j]
            if abs(x) < abs(y):
                var t = x
                x = y
                y = t
            var hi = x + y
            var yr = hi - x
            var lo = y - yr
            if lo != 0.0:
                p[i] = lo
                i += 1
            x = hi
        while len(p) > i:
            _ = p.pop()
        if x != 0.0:
            p.append(x)
    var n = len(p)
    var hi: Float64 = 0.0
    var lo: Float64 = 0.0
    if n > 0:
        n -= 1
        hi = p[n]
        while n > 0:
            var x = hi
            n -= 1
            var y = p[n]
            hi = x + y
            var yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
            var y = lo * 2.0
            var x = hi + y
            var yr = x - hi
            if y == yr:
                hi = x
    if hi == 0.0:
        return 0.0
    return hi


def _trapezoid(x: List[Float64], y: List[Float64], L: Int) -> Float64:
    """fsum of ((x[i] - x[i-1]) * (y[i] + y[i-1])) / 2 over the first L."""
    var terms = List[Float64](capacity=max(L - 1, 0))
    for i in range(1, L):
        terms.append(pinned_mul_f64(x[i] - x[i - 1], y[i] + y[i - 1]) / 2.0)
    return fsum(terms)


def kept(a: Int, fps: Int, tps: Int, keep: Int, c: Int, drop: Bool) -> List[Int]:
    """The curve points `_drop_collinear` keeps: all of them when not
    `drop` or c <= 2; the device's flags (`keep` >= 0, an unweighted curve);
    else the first, the last and every point where either binary64 step
    changes."""
    var out = List[Int](capacity=c)
    if not drop or c <= 2:
        for i in range(c):
            out.append(i)
        return out^
    if keep >= 0:
        for i in range(c):
            if _word(a, keep + i) != 0:
                out.append(i)
        return out^
    out.append(0)
    for i in range(1, c - 1):
        var f0 = _w(a, fps + i - 1)
        var f1 = _w(a, fps + i)
        var f2 = _w(a, fps + i + 1)
        var t0 = _w(a, tps + i - 1)
        var t1 = _w(a, tps + i)
        var t2 = _w(a, tps + i + 1)
        if (f2 - f1) != (f1 - f0) or (t2 - t1) != (t1 - t0):
            out.append(i)
    out.append(c - 1)
    return out^


def binary_auc(a: Int, fps: Int, tps: Int, keep: Int, c: Int, max_fpr: Float64) raises -> Float64:
    """`_binary_auc` for fps[c-1] > 0 and tps[c-1] > 0 (the caller checks);
    max_fpr < 0 means None (or 1)."""
    var F = _w(a, fps + c - 1)
    var T = _w(a, tps + c - 1)
    if not (F > 0.0 and T > 0.0):
        raise Error("x_metrics epilogue: an empty class goes the Python way")
    var ks = kept(a, fps, tps, keep, c, True)
    var m = len(ks)
    var fpr = List[Float64](capacity=m + 1)
    var tpr = List[Float64](capacity=m + 1)
    fpr.append(0.0)
    tpr.append(0.0)
    for j in range(m):
        fpr.append(_w(a, fps + ks[j]) / F)
        tpr.append(_w(a, tps + ks[j]) / T)
    if max_fpr < 0.0:
        return _trapezoid(fpr, tpr, m + 1)
    # bisect_right(fpr, max_fpr)
    var lo = 0
    var hi = m + 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if max_fpr < fpr[mid]:
            hi = mid
        else:
            lo = mid + 1
    var stop = lo
    if stop < 1 or stop > m:
        raise Error("x_metrics epilogue: max_fpr outside the curve goes the Python way")
    var x0 = fpr[stop - 1]
    var x1 = fpr[stop]
    var y0 = tpr[stop - 1]
    var y1 = tpr[stop]
    var yi = y0
    if x1 != x0:
        yi = y0 + pinned_mul_f64(max_fpr - x0, y1 - y0) / (x1 - x0)
    fpr[stop] = max_fpr
    tpr[stop] = yi
    var part = _trapezoid(fpr, tpr, stop + 1)
    var min_area = pinned_mul_f64(pinned_mul_f64(0.5, max_fpr), max_fpr)
    return pinned_mul_f64(0.5, 1.0 + (part - min_area) / (max_fpr - min_area))


def binary_ap(a: Int, fps: Int, tps: Int, c: Int) raises -> Float64:
    """`_binary_ap` for tps[c-1] != 0 (the caller checks): max(0.0, fsum
    of (r - r_prev) * (t / (t + f))), r = t / T, r_prev of the first 0.0."""
    var T = _w(a, tps + c - 1)
    if not (T != 0.0):
        raise Error("x_metrics epilogue: no positives goes the Python way")
    var terms = List[Float64](capacity=c)
    var rp: Float64 = 0.0
    for i in range(c):
        var t = _w(a, tps + i)
        var d = t + _w(a, fps + i)
        if d == 0.0:
            raise Error("x_metrics epilogue: a zero denominator goes the Python way")
        var r = t / T
        terms.append(pinned_mul_f64(r - rp, t / d))
        rp = r
    var s = fsum(terms)
    if s > 0.0:
        return s
    return 0.0


def roc_arrays(a: Int, fps: Int, tps: Int, thr: Int, keep: Int, c: Int, drop: Bool,
               out_fpr: Int, out_tpr: Int, out_thr: Int) raises -> Int:
    """`roc_curve`'s three Float64 arrays for fps[c-1] > 0 and tps[c-1] > 0
    (the caller checks): the kept points after a leading (0, 0, +inf).
    Returns their length."""
    var F = _w(a, fps + c - 1)
    var T = _w(a, tps + c - 1)
    if not (F > 0.0 and T > 0.0):
        raise Error("x_metrics epilogue: an empty class goes the Python way")
    var ks = kept(a, fps, tps, keep, c, drop)
    var pf = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_fpr)
    var pt = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_tpr)
    var ph = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=out_thr)
    pf[0] = 0.0
    pt[0] = 0.0
    ph[0] = bitcast[DType.float64](UInt64(0x7FF0000000000000))
    for j in range(len(ks)):
        pf[j + 1] = _w(a, fps + ks[j]) / F
        pt[j + 1] = _w(a, tps + ks[j]) / T
        ph[j + 1] = _w(a, thr + ks[j])
    return len(ks) + 1


# ---------------------------------------------------------------------------
# The expected mutual information (adjusted_mutual_info_score)
# ---------------------------------------------------------------------------

@always_inline
def _fm(a: Float64, b: Float64, c: Float64) -> Float64:
    from std.math import fma
    return fma(a, b, c)


@always_inline
def _mul(a: Float64, b: Float64) -> Float64:
    return pinned_mul_f64(a, b)


def _log_fraction(x_in: Float64, mut e: Int, mut frac: Float64) -> Float64:
    """packaging/portable_math/portable_math.c `log_fraction`, operation
    for operation (its products unfused, its fm sites fma)."""
    var input = x_in
    var u = bitcast[DType.uint64](input)
    e = 0
    if (u >> 52) == UInt64(0):
        input = _mul(input, 18014398509481984.0)
        u = bitcast[DType.uint64](input)
        e = -54
    e += Int((u >> 52) & UInt64(0x7FF)) - 1022
    var m = bitcast[DType.float64]((u & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x3FE0000000000000))
    var z: Float64
    var y: Float64
    var x: Float64
    if e > 2 or e < -2:
        if m < 0.70710678118654752440:
            e -= 1
            z = m - 0.5
            y = _fm(0.5, z, 0.5)
        else:
            z = m - 0.5
            z = z - 0.5
            y = _fm(0.5, m, 0.5)
        x = z / y
        z = _mul(x, x)
        var r = _fm(-7.89580278884799154124e-1, z, 1.63866645699558079767e1)
        r = _fm(r, z, -6.41409952958715622951e1)
        var q = z + -3.56722798256324312549e1
        q = _fm(q, z, 3.12093766372244180303e2)
        q = _fm(q, z, -7.69691943550460008604e2)
        y = _mul(x, _mul(z, r) / q)
    else:
        if m < 0.70710678118654752440:
            e -= 1
            x = _fm(2.0, m, -1.0)
        else:
            x = m - 1.0
        z = _mul(x, x)
        var p = _fm(1.01875663804580931796e-4, x, 4.97494994976747001425e-1)
        p = _fm(p, x, 4.70579119878881725854e0)
        p = _fm(p, x, 1.44989225341610930846e1)
        p = _fm(p, x, 1.79368678507819816313e1)
        p = _fm(p, x, 7.70838733755885391666e0)
        var q = x + 1.12873587189167450590e1
        q = _fm(q, x, 4.52279145837532221105e1)
        q = _fm(q, x, 8.29875266912776603211e1)
        q = _fm(q, x, 7.11544750618563894466e1)
        q = _fm(q, x, 2.31251620126765340583e1)
        y = _mul(x, _mul(z, p) / q)
    frac = x
    return y


def portable_log_c(input: Float64) -> Float64:
    """packaging/portable_math/portable_math.c `mojolearn_log` (what
    `mojolearn._portable_math.log` calls), operation for operation, for a
    finite positive input (the only kind the callers below pass)."""
    var u = bitcast[DType.uint64](input)
    var raw_e = 0
    if (u >> 52) == UInt64(0):
        u = bitcast[DType.uint64](_mul(input, 18014398509481984.0))
        raw_e = -54
    raw_e += Int((u >> 52) & UInt64(0x7FF)) - 1022
    var e = 0
    var x: Float64 = 0.0
    var y = _log_fraction(input, e, x)
    y = _fm(Float64(e), -2.121944400546905827679e-4, y)
    if not (raw_e > 2 or raw_e < -2):
        y = _fm(_mul(x, x), -0.5, y)
    y = y + x
    return _fm(Float64(e), 0.693359375, y)


def expected_mi(a_addr: Int, na: Int, b_addr: Int, nb: Int, n: Int) raises -> Float64:
    """python/mojolearn/_expansion_metrics.py `_expected_mi` for na, nb >= 2
    (Int64 class counts at the two addresses): the same integer bounds and
    mode, the same binary64 ratio walks from the mode (each stops at the
    first exact 0), the same pmf normalization by fsum, the same terms
    (nij / n) * (log(n nij) - log a - log b) * pr, fsum-ed. Python's
    int -> float conversions and int / int division are correctly rounded,
    as Float64(Int) and Float64 division are for these magnitudes."""
    var A = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=a_addr)
    var B = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=b_addr)
    if n <= 0 or n >= (1 << 31):
        raise Error("x_metrics epilogue: expected MI size goes the Python way")
    var fn = Float64(n)
    var terms = List[Float64]()
    for i in range(na):
        var a = Int(A[i])
        var la = portable_log_c(Float64(a))
        for j in range(nb):
            var b = Int(B[j])
            var lb = portable_log_c(Float64(b))
            var lo = max(0, a + b - n)
            var hi = min(a, b)
            var mode = min(max(((a + 1) * (b + 1)) // (n + 2), lo), hi)
            var up = List[Float64]()
            up.append(1.0)
            var x = mode
            var v: Float64 = 1.0
            while x < hi:
                v = _mul(v, Float64((a - x) * (b - x))) / Float64((x + 1) * (n - a - b + x + 1))
                if v == 0.0:
                    break
                up.append(v)
                x += 1
            var down = List[Float64]()
            x = mode
            v = 1.0
            while x > lo:
                v = _mul(v, Float64(x * (n - a - b + x))) / Float64((a - x + 1) * (b - x + 1))
                if v == 0.0:
                    break
                down.append(v)
                x -= 1
            var all = List[Float64](capacity=len(up) + len(down))
            for k in range(len(up)):
                all.append(up[k])
            for k in range(len(down)):
                all.append(down[k])
            var z = fsum(all)
            var first = mode - len(down)
            var nd = len(down)
            for k in range(nd + len(up)):
                var nij = first + k
                var u = down[nd - 1 - k] if k < nd else up[k - nd]
                if nij < 1:
                    continue
                var pr = u / z
                if pr == 0.0:
                    continue
                var q = Float64(nij) / fn
                var d = (portable_log_c(Float64(n * nij)) - la) - lb
                terms.append(_mul(_mul(q, d), pr))
    return fsum(terms)
