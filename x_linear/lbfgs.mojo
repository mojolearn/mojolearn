# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A SEQUENTIAL L-BFGS for the linear lane (lane/algos-linear, 2026-09-27).

Nocedal & Wright, Numerical Optimization (2nd ed.), Algorithm 7.4 (the
two-loop recursion) inside Algorithm 7.5, with m = 10 pairs, the initial
scaling gamma = s'y / y'y (their eq. 7.20), and a backtracking Armijo line
search (c1 = 1e-4, step halved up to 40 times) in place of their Wolfe
search. A pair with s'y <= 1e-10 * |s|*|y| is skipped. Stops when
max|g| <= tol, when the line search cannot decrease f at float32 resolution,
or at max_iter. Every sum ascends the index; the objective is a comptime
function parameter, so each caller compiles its own copy.

Objective contract:
    f = obj(x, y, n, d, ip, fp, theta, toff, grad, goff) and grad written.
Work layout at fw[woff:]: tn P | g P | gn P | dir P | S m*P | Y m*P | rho m | al m.
"""
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, ld, st, copy, fill

comptime LBFGS_M = 10

comptime Objective = def(FP, FP, Int, Int, IP, FP, FP, Int, FP, Int) thin -> Float32


def lbfgs_work(p: Int) -> Int:
    return 4 * p + 2 * LBFGS_M * p + 2 * LBFGS_M


def _dot(a: FP, ia: Int, b: FP, ib: Int, p: Int) -> Float32:
    var acc = Float32(0)
    for j in range(p):
        acc = fmad(ld(a, ia + j), ld(b, ib + j), acc)
    return acc


def lbfgs[obj: Objective](
    x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP,
    theta: FP, toff: Int, p: Int, max_iter: Int, tol: Float32, fw: FP, woff: Int,
) -> Int:
    """Minimizes obj over theta[toff:toff+p] in place. Returns iterations
    run (negative when it stopped on max_iter without meeting tol)."""
    var tn = woff
    var g = tn + p
    var gn = g + p
    var dr = gn + p
    var sS = dr + p
    var sY = sS + LBFGS_M * p
    var rho = sY + LBFGS_M * p
    var al = rho + LBFGS_M
    var f = obj(x, y, n, d, ip, fp, theta, toff, fw, g)
    var count = 0
    var head = 0
    var it = 0
    while it < max_iter:
        var gmax = Float32(0)
        for j in range(p):
            gmax = fmax(gmax, fabs(ld(fw, g + j)))
        if gmax <= tol:
            return it
        # two-loop recursion: dir = -H g
        for j in range(p):
            st(fw, dr + j, ld(fw, g + j))
        for kk in range(count):
            var k = (head - 1 - kk + 2 * LBFGS_M) % LBFGS_M
            var a = fm(ld(fw, rho + k), _dot(fw, sS + k * p, fw, dr, p))
            st(fw, al + k, a)
            for j in range(p):
                st(fw, dr + j, fs(ld(fw, dr + j), fm(a, ld(fw, sY + k * p + j))))
        var gamma: Float32
        if count > 0:
            var k = (head - 1 + LBFGS_M) % LBFGS_M
            gamma = fd(_dot(fw, sS + k * p, fw, sY + k * p, p), _dot(fw, sY + k * p, fw, sY + k * p, p))
        else:
            gamma = fd(Float32(1), fmax(Float32(1), fsqrt(_dot(fw, g, fw, g, p))))
        for j in range(p):
            st(fw, dr + j, fm(gamma, ld(fw, dr + j)))
        for kk in range(count):
            var k = (head - count + kk + 2 * LBFGS_M) % LBFGS_M
            var b = fm(ld(fw, rho + k), _dot(fw, sY + k * p, fw, dr, p))
            var c = fs(ld(fw, al + k), b)
            for j in range(p):
                st(fw, dr + j, fmad(c, ld(fw, sS + k * p + j), ld(fw, dr + j)))
        for j in range(p):
            st(fw, dr + j, -ld(fw, dr + j))
        var slope = _dot(fw, g, fw, dr, p)
        if not (slope < 0):
            count = 0
            head = 0
            for j in range(p):
                st(fw, dr + j, -ld(fw, g + j))
            slope = _dot(fw, g, fw, dr, p)
            if not (slope < 0):
                return it
        var t = Float32(1)
        var accepted = False
        var fnew = f
        for _ in range(40):
            for j in range(p):
                st(fw, tn + j, fmad(t, ld(fw, dr + j), ld(theta, toff + j)))
            fnew = obj(x, y, n, d, ip, fp, fw, tn, fw, gn)
            if fnew == fnew and fnew <= fa(f, fm(fm(Float32(1e-4), t), slope)):
                accepted = True
                break
            t = fm(t, Float32(0.5))
        if not accepted:
            return it
        it += 1
        # the new pair
        var k = head
        for j in range(p):
            st(fw, sS + k * p + j, fs(ld(fw, tn + j), ld(theta, toff + j)))
            st(fw, sY + k * p + j, fs(ld(fw, gn + j), ld(fw, g + j)))
        var sy = _dot(fw, sS + k * p, fw, sY + k * p, p)
        var ss = _dot(fw, sS + k * p, fw, sS + k * p, p)
        var yy = _dot(fw, sY + k * p, fw, sY + k * p, p)
        if sy > fm(Float32(1e-10), fsqrt(fm(ss, yy))) and sy > 0:
            st(fw, rho + k, fd(Float32(1), sy))
            head = (head + 1) % LBFGS_M
            if count < LBFGS_M:
                count += 1
        copy(theta, toff, fw, tn, p)
        copy(fw, g, fw, gn, p)
        f = fnew
    return -it
