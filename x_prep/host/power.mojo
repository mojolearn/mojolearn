# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host's `pt_fit` (op 44 of x_prep/units.mojo), lane prep-cpu
2026-09-28: `pt_fit_unit`'s words (x_prep/transform.mojo), fewer
transcendentals.

The device's unit evaluates `_neg_llf` fifty times per column, and every
evaluation computes, per non-NaN row, `power(x, lam)` twice (once for the
mean, once for the variance) and the Jacobian term once: five logarithms or
exponentials. Only the exponential depends on lambda. This path computes
each row's logarithm (`power_log`) and the Jacobian fold ONCE per column,
then per evaluation one `power_from_log` per row, kept for the variance
pass. Every value is the same function of the same operands, and every fold
(the sum of T, the sum of squares, the Jacobian sum) runs over the same rows
in the same ascending order, so every word matches the device's:
`power(x, lam, m)` IS `power_from_log(power_log(x, m), x >= 0, lam, m)`.
The lane check (CPU == GPU on x-prep-power-transformer and
x-prep-inverse-transforms) and `check_host_power` in x_prep/seams/
prep_check.mojo hold it there."""
from x_prep.common import FP, IP, p, ld, st, is_nan
from x_prep.prims import add, sub, mul, div, logf
from x_prep.transform import power_log, power_from_log, PT_LO, PT_HI, PT_ITERS, GOLDEN


struct _Column(Movable):
    """One column's non-NaN rows in ascending order: the logarithm, the
    sign, the Jacobian sum, and the scratch for one evaluation's T."""
    var lg: List[Float32]
    var nonneg: List[Bool]
    var tv: List[Float32]
    var sj: Float32

    def __init__(out self, f: FP, X: Int, n: Int, d: Int, c: Int, method: Int):
        self.lg = List[Float32](capacity=n)
        self.nonneg = List[Bool](capacity=n)
        self.sj = Float32(0)
        for i in range(n):
            var x = ld(f, X + i * d + c)
            if is_nan(x):
                continue
            var l = power_log(x, method)
            var nn = x >= Float32(0)
            if method == 1 or nn:
                self.sj = add(self.sj, l)
            else:
                self.sj = sub(self.sj, l)
            self.lg.append(l)
            self.nonneg.append(nn)
        self.tv = List[Float32](length=len(self.lg), fill=Float32(0))

    def neg_llf(mut self, lam: Float32, method: Int) -> Float32:
        """`_neg_llf(..., lam, method)`'s word."""
        var cnt = len(self.lg)
        if cnt == 0:
            return Float32(0)
        var s = Float32(0)
        for i in range(cnt):
            var v = power_from_log(self.lg[i], self.nonneg[i], lam, method)
            self.tv[i] = v
            s = add(s, v)
        var mean = div(s, Float32(cnt))
        var ss = Float32(0)
        for i in range(cnt):
            var e = sub(self.tv[i], mean)
            ss = add(ss, mul(e, e))
        var var_ = div(ss, Float32(cnt))
        return sub(mul(mul(Float32(0.5), Float32(cnt)), logf(var_)), mul(sub(lam, Float32(1)), self.sj))


def pt_fit_host_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, METHOD, ST, LAMBDA]; t = column: `pt_fit_unit`'s word."""
    var n = p(q, 1)
    var d = p(q, 2)
    var method = p(q, 3)
    var c = t
    if method == 0 and ld(f, p(q, 4) + 2 * d + c) == Float32(0):
        st(f, p(q, 5) + c, Float32(1))
        return
    var col = _Column(f, p(q, 0), n, d, c, method)
    var a = PT_LO
    var b = PT_HI
    var x1 = add(a, mul(GOLDEN, sub(b, a)))
    var x2 = sub(b, mul(GOLDEN, sub(b, a)))
    var f1 = col.neg_llf(x1, method)
    var f2 = col.neg_llf(x2, method)
    for _ in range(PT_ITERS):
        if f1 <= f2 or f2 != f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = add(a, mul(GOLDEN, sub(b, a)))
            f1 = col.neg_llf(x1, method)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = sub(b, mul(GOLDEN, sub(b, a)))
            f2 = col.neg_llf(x2, method)
    st(f, p(q, 5) + c, mul(add(a, b), Float32(0.5)))
