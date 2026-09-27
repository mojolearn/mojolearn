# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Nelder-Mead, statsforecast's (`include/statsforecast/nelder_mead.h`,
`nm::NelderMead`): box-clamped simplex, the initial simplex perturbed by
init_step (zero_pert at a zero coordinate), adaptive coefficients
gamma = 1 + 2/n, rho = 0.75 - 1/(2n), sigma = 1 - 1/n, stop when the
population standard deviation of the simplex values falls below tol_std,
reflection / expansion / outside and inside contraction / shrink exactly as
there, in float32. Runs inside one thread (one series).

The one difference: the reference orders the simplex with std::sort, whose
order among EQUAL values is unspecified; here ties keep the lower vertex
index first (a stable insertion sort), so the order is a function of the
values alone.

The objective is chosen at compile time: `Obj.eval(x)` of a struct
conforming to `Objective`."""
from sequence.ops import FP, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_sqrt


trait Objective:
    def eval(mut self, x: FP) -> Float32:
        ...


@always_inline
def _clamp(v: Float32, lo: Float32, hi: Float32) -> Float32:
    var r = v if v > lo else lo
    return r if r < hi else hi


def nelder_mead[O: Objective](
    mut obj: O, x0: FP, lower: FP, upper: FP, n: Int, scratch: FP,
    init_step: Float32, zero_pert: Float32, max_iter: Int, tol_std: Float32,
) -> Int:
    """Minimises obj over n <= 8 coordinates from x0; the best point is
    written back to x0. scratch holds (n + 1) n + (n + 1) + 4 n floats.
    Returns the iteration count."""
    var nf = Float32(n)
    var gamma = add(Float32(1.0), ftz(identical_div(Float32(2.0), nf)))
    var rho = sub(Float32(0.75), ftz(identical_div(Float32(1.0), mul(Float32(2.0), nf))))
    var sigma = sub(Float32(1.0), ftz(identical_div(Float32(1.0), nf)))
    var simplex = scratch
    var fs = scratch + (n + 1) * n
    var xo = fs + (n + 1)
    var xr = xo + n
    var xe = xr + n
    var xt = xe + n
    for i in range(n + 1):
        for j in range(n):
            st(simplex, i * n + j, _clamp(ld(x0, j), ld(lower, j), ld(upper, j)))
    for i in range(n):
        var v = ld(simplex, i * n + i)
        if v == Float32(0.0):
            v = zero_pert
        else:
            v = mul(v, add(Float32(1.0), init_step))
        st(simplex, i * n + i, _clamp(v, ld(lower, i), ld(upper, i)))
    for i in range(n + 1):
        st(fs, i, obj.eval(simplex + i * n))
    var order = InlineArray[Int, 9](fill=0)
    var it = 0
    var best = 0
    while it < max_iter:
        # stable argsort of fs (ties: lower index first)
        for i in range(n + 1):
            order[i] = i
        for i in range(1, n + 1):
            var k = order[i]
            var j = i - 1
            while j >= 0 and ld(fs, order[j]) > ld(fs, k):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = k
        best = order[0]
        var worst = order[n]
        var second = order[n - 1]
        # population standard deviation of fs
        var mean = Float32(0.0)
        for i in range(n + 1):
            mean = add(mean, ld(fs, i))
        mean = ftz(identical_div(mean, Float32(n + 1)))
        var ss = Float32(0.0)
        for i in range(n + 1):
            var d = sub(ld(fs, i), mean)
            ss = fma3(d, d, ss)
        if ftz(identical_sqrt(ftz(identical_div(ss, Float32(n + 1))))) < tol_std:
            break
        # centroid without the worst vertex
        for j in range(n):
            var s = Float32(0.0)
            for i in range(n + 1):
                s = add(s, ld(simplex, i * n + j))
            st(xo, j, ftz(identical_div(sub(s, ld(simplex, worst * n + j)), nf)))
        # reflection (alpha = 1)
        for j in range(n):
            var o = ld(xo, j)
            st(xr, j, _clamp(add(o, sub(o, ld(simplex, worst * n + j))), ld(lower, j), ld(upper, j)))
        var fr = obj.eval(xr)
        if ld(fs, best) <= fr and fr < ld(fs, second):
            for j in range(n):
                st(simplex, worst * n + j, ld(xr, j))
            st(fs, worst, fr)
            it += 1
            continue
        if fr < ld(fs, best):
            for j in range(n):
                var o = ld(xo, j)
                st(xe, j, _clamp(fma3(gamma, sub(ld(xr, j), o), o), ld(lower, j), ld(upper, j)))
            var fe = obj.eval(xe)
            if fe < fr:
                for j in range(n):
                    st(simplex, worst * n + j, ld(xe, j))
                st(fs, worst, fe)
            else:
                for j in range(n):
                    st(simplex, worst * n + j, ld(xr, j))
                st(fs, worst, fr)
            it += 1
            continue
        var accepted = False
        if ld(fs, second) <= fr and fr < ld(fs, worst):
            for j in range(n):
                var o = ld(xo, j)
                st(xt, j, _clamp(fma3(rho, sub(ld(xr, j), o), o), ld(lower, j), ld(upper, j)))
            var fc = obj.eval(xt)
            if fc <= fr:
                for j in range(n):
                    st(simplex, worst * n + j, ld(xt, j))
                st(fs, worst, fc)
                accepted = True
        else:
            for j in range(n):
                var o = ld(xo, j)
                st(xt, j, _clamp(sub(o, mul(rho, sub(ld(xr, j), o))), ld(lower, j), ld(upper, j)))
            var fc = obj.eval(xt)
            if fc < ld(fs, worst):
                for j in range(n):
                    st(simplex, worst * n + j, ld(xt, j))
                st(fs, worst, fc)
                accepted = True
        if not accepted:
            for i in range(n + 1):
                if i == best:
                    continue
                for j in range(n):
                    var b = ld(simplex, best * n + j)
                    st(simplex, i * n + j,
                       _clamp(fma3(sigma, sub(ld(simplex, i * n + j), b), b), ld(lower, j), ld(upper, j)))
                st(fs, i, obj.eval(simplex + i * n))
        it += 1
    for j in range(n):
        st(x0, j, ld(simplex, best * n + j))
    return it + 1
