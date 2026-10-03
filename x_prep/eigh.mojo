# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Symmetric eigendecomposition as ONE unit: cyclic Jacobi, float32.

Reference: Golub & Van Loan, Matrix Computations 4th ed., Algorithm 8.5.3
(cyclic-by-row Jacobi) with the Rutishauser rotation (Numerical Recipes 3rd
ed. section 11.1). One unit per matrix, so a batch of class covariances (QDA)
runs one matrix per thread. The sweep order is fixed (p ascending, then q),
a rotation is skipped only on a value test that reads the same bits
everywhere, and the sweep count is capped, so the result is a function of
the input bits alone. Eigenpairs come out sorted by eigenvalue DESCENDING
(index order on a tie) and each eigenvector's largest-magnitude component
(first on a tie) is made positive: the sign convention the tests can name
(DEVIATION 5405).
"""
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div, sqrtf

comptime MAX_SWEEPS = 64


def eigh_unit(t: Int, f: FP, q: IP):
    """q = [A, m, astride, EVAL, EVEC, cyclic] (cyclic: x_prep/device.mojo EIGH_CYCLIC_Q); t = batch index. A[t*astride :] is an
    m x m symmetric matrix, DESTROYED. EVAL[t*m + r] the eigenvalues
    descending; EVEC[t*m*m + c*m + r] the component c of eigenvector r
    (column r, as numpy's eigh returns it)."""
    var m = p(q, 1)
    var A = p(q, 0) + t * p(q, 2)
    var W = p(q, 3) + t * m
    var V = p(q, 4) + t * m * m
    for r in range(m):
        for c in range(m):
            st(f, V + r * m + c, Float32(1) if r == c else Float32(0))
    for _ in range(MAX_SWEEPS):
        var rotated = False
        for pp in range(m - 1):
            for qq in range(pp + 1, m):
                var apq = ld(f, A + pp * m + qq)
                if apq == Float32(0):
                    continue
                var app = ld(f, A + pp * m + pp)
                var aqq = ld(f, A + qq * m + qq)
                # negligible: |apq| below float32 resolution of the diagonal
                var scale = add(abs(app), abs(aqq))
                if add(scale, mul(abs(apq), Float32(64))) == scale:
                    st(f, A + pp * m + qq, Float32(0))
                    st(f, A + qq * m + pp, Float32(0))
                    continue
                rotated = True
                var theta = div(sub(aqq, app), mul(Float32(2), apq))
                var tt: Float32
                if abs(theta) > Float32(1.0e18):
                    tt = div(Float32(0.5), theta)
                else:
                    tt = div(Float32(1), add(abs(theta), sqrtf(add(mul(theta, theta), Float32(1)))))
                    if theta < Float32(0):
                        tt = sub(Float32(0), tt)
                var cc = div(Float32(1), sqrtf(add(mul(tt, tt), Float32(1))))
                var ss = mul(tt, cc)
                st(f, A + pp * m + pp, sub(app, mul(tt, apq)))
                st(f, A + qq * m + qq, add(aqq, mul(tt, apq)))
                st(f, A + pp * m + qq, Float32(0))
                st(f, A + qq * m + pp, Float32(0))
                for r in range(m):
                    if r == pp or r == qq:
                        continue
                    var arp = ld(f, A + r * m + pp)
                    var arq = ld(f, A + r * m + qq)
                    var nrp = sub(mul(cc, arp), mul(ss, arq))
                    var nrq = add(mul(ss, arp), mul(cc, arq))
                    st(f, A + r * m + pp, nrp)
                    st(f, A + pp * m + r, nrp)
                    st(f, A + r * m + qq, nrq)
                    st(f, A + qq * m + r, nrq)
                for r in range(m):
                    var vrp = ld(f, V + r * m + pp)
                    var vrq = ld(f, V + r * m + qq)
                    st(f, V + r * m + pp, sub(mul(cc, vrp), mul(ss, vrq)))
                    st(f, V + r * m + qq, add(mul(ss, vrp), mul(cc, vrq)))
        if not rotated:
            break
    # selection sort, descending, stable (a later equal value never moves ahead)
    for r in range(m):
        st(f, W + r, ld(f, A + r * m + r))
    for r in range(m):
        var best = r
        for c in range(r + 1, m):
            if ld(f, W + c) > ld(f, W + best):
                best = c
        if best != r:
            var tw = ld(f, W + r)
            st(f, W + r, ld(f, W + best))
            st(f, W + best, tw)
            for c in range(m):
                var tv = ld(f, V + c * m + r)
                st(f, V + c * m + r, ld(f, V + c * m + best))
                st(f, V + c * m + best, tv)
    for r in range(m):
        var big = 0
        for c in range(1, m):
            if abs(ld(f, V + c * m + r)) > abs(ld(f, V + big * m + r)):
                big = c
        if ld(f, V + big * m + r) < Float32(0):
            for c in range(m):
                st(f, V + c * m + r, sub(Float32(0), ld(f, V + c * m + r)))
