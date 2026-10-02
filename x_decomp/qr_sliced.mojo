# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SLICED geqrf / orgqr (lane hr-qr, 2026-10-02): the order of every fold of
the unblocked Householder QR (DEVIATION 5320) and the scalar steps shared by
the device kernels (x_decomp/qr_sliced_device.mojo) and their host replay
(x_decomp/qr_sliced_host.mojo). No GPU import: the CPU binding compiles
this file.

WHY. `geqrf_serial` / `orgqr_col` folded every reflector norm and every
reflector product w = v^T x as ONE dependent chain over all the rows below
the diagonal: on the device one thread per column per step, m dependent
multiply-adds long (17 s at 200,000 x 220 on the M4, 16.7 s on the MI325X
at the board's istella shape), which is why AMD and Apple handed the whole
factorization to a host walk for m >= 65,536.

THE ORDER, a pure function of the shape (m, k) and of nothing on the
machine. At step k the rows below the diagonal, k + 1 .. m - 1, are cut
into ns = ceil((m - k - 1) / QS_ROWS) SLICES of QS_ROWS rows (the last
takes the remainder); slice c is rows [k + 1 + c QS_ROWS, ...):

  * a reflector product w = x[k] + sum_{i > k} v_i x[i]: slice c's partial
    is the chain p_c = ftz(fma(ftz(v_i), ftz(x_i), p_c)) from 0, rows
    ascending; the ns partials are folded by `qs_tree` (a fixed binary
    tree, below); then w = ftz(ftz(x[k]) + tree). (`x` is a column of A for
    geqrf, of Q for orgqr; v_i the stored reflector below the diagonal.)
  * a reflector norm ||(alpha, a[k+1:, k])||: slice c's partial is the
    pair (s_c, q_c), s_c = its largest |ftz(a_i)| and q_c the chain of
    (ftz(a_i / s_c))^2 ascending from 0 (q_c = 0 when s_c = 0); the pairs
    are folded by the same tree with `qs_pair` (LAPACK dlassq's rescaled
    combine, the larger scale kept, a tie keeps the left); alpha's pair
    (|alpha|, 1) is combined last; nrm = ftz(sqrt(q) * s). The tree's s is
    also the sub-diagonal's largest |entry|: 0 means H_k = I (tau = 0), as
    dlarfg.
  * `qs_tree` over p[0 .. ns): for stride h = 1, 2, 4, ... while h < ns,
    p[a] = ftz(p[a] + p[a + h]) for a = 0, 2h, 4h, ... with a + h < ns; the
    result is p[0]. No slice count changes a bit except through the shape.

Everything else is DEVIATION 5320's cells verbatim: dlarfg's sign, tau,
the multipliers v_i = a_i / (alpha - beta) (`geqrf_scale_elem`), and the
updates (`geqrf_update_elem`, `orgqr_update_elem`), every cell its own
thread on the device. Reflectors go to A in order k ascending, to Q's
columns in order k descending.
"""
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from std.sys.compile import is_defined
from x_decomp.cells import F32Ptr, div0, sqrt0, sub

#: rows per slice of a reflector fold
comptime QS_ROWS = 512
#: multiply-adds per device launch (macOS silently cuts a long Metal
#: command buffer; every launch is sliced to about this much work)
comptime QS_LAUNCH_CELLS = 1 << 27
#: The A/B arm during measurement: -D MOJOLEARN_XD_QR_CHAIN keeps the old
#: one-chain-per-column order (and its routes); the default is the sliced
#: order on every column.
comptime XD_QR_SLICED = not is_defined["MOJOLEARN_XD_QR_CHAIN"]()


@always_inline
def qs_slices(rows: Int) -> Int:
    """The slice count of a fold over `rows` rows (0 when there are none)."""
    return (rows + QS_ROWS - 1) // QS_ROWS if rows > 0 else 0


@always_inline
def qs_slice_lo(k: Int, c: Int) -> Int:
    """Slice c's first row at step k."""
    return k + 1 + c * QS_ROWS


@always_inline
def qs_slice_hi(k: Int, c: Int, m: Int) -> Int:
    var hi = k + 1 + (c + 1) * QS_ROWS
    return hi if hi < m else m


def qs_tree(p: F32Ptr, ns: Int) -> Float32:
    """The fixed binary tree over p[0 .. ns) (in place); 0 when ns is 0."""
    if ns <= 0:
        return Float32(0)
    var h = 1
    while h < ns:
        var a = 0
        while a + h < ns:
            p.unsafe_store(a, ftz(p.unsafe_load(a) + p.unsafe_load(a + h)))
            a += 2 * h
        h *= 2
    return p.unsafe_load(0)


@always_inline
def qs_pair(s1: Float32, q1: Float32, s2: Float32, q2: Float32) -> SIMD[DType.float32, 2]:
    """(s, q) of two scaled sums of squares s1^2 q1 and s2^2 q2: the larger
    scale kept (a tie keeps the left), the other's q rescaled by the square
    of the ratio; a zero scale contributes nothing."""
    var bs = s1
    var bq = q1
    var ss = s2
    var sq = q2
    if s2 > s1:
        bs = s2
        bq = q2
        ss = s1
        sq = q1
    if ss == Float32(0):
        return SIMD[DType.float32, 2](bs, bq)
    var r = ftz(identical_div(ss, bs))
    return SIMD[DType.float32, 2](bs, ftz(identical_mul_add(ftz(identical_mul(r, r)), sq, bq)))


def qs_pair_tree(s: F32Ptr, q: F32Ptr, ns: Int) -> SIMD[DType.float32, 2]:
    """`qs_tree`'s shape over the pairs (s[c], q[c]) with `qs_pair` (in
    place); (0, 0) when ns is 0."""
    if ns <= 0:
        return SIMD[DType.float32, 2](0, 0)
    var h = 1
    while h < ns:
        var a = 0
        while a + h < ns:
            var r = qs_pair(s.unsafe_load(a), q.unsafe_load(a), s.unsafe_load(a + h), q.unsafe_load(a + h))
            s.unsafe_store(a, r[0])
            q.unsafe_store(a, r[1])
            a += 2 * h
        h *= 2
    return SIMD[DType.float32, 2](s.unsafe_load(0), q.unsafe_load(0))


def qs_slice_ssq(a: F32Ptr, n: Int, col: Int, lo: Int, hi: Int) -> SIMD[DType.float32, 2]:
    """Slice [lo, hi)'s (s, q) of column `col` of the row-major m x n a: s
    its largest |ftz(a_i)| (seeded 0, `v > s`, so a NaN never enters the
    scale), q the chain of (ftz(a_i / s))^2 ascending from 0."""
    var mx = Float32(0)
    for i in range(lo, hi):
        var v = abs(ftz(a.unsafe_load(i * n + col)))
        if v > mx:
            mx = v
    if mx == Float32(0):
        return SIMD[DType.float32, 2](0, 0)
    var acc = Float32(0)
    for i in range(lo, hi):
        var v = ftz(identical_div(ftz(a.unsafe_load(i * n + col)), mx))
        acc = ftz(identical_mul_add(v, v, acc))
    return SIMD[DType.float32, 2](mx, acc)


def qs_head(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, ps: F32Ptr, pq: F32Ptr, k: Int, n: Int, ns: Int):
    """Step k's reflector (dlarfg) from the ns slice pairs (ps, pq), folded
    in place: tau[k], beta on the diagonal, and in `scal` [the divisor
    alpha - beta, 1 when the step acts else 0] (`geqrf_head`'s outputs)."""
    var alpha = ftz(a.unsafe_load(k * n + k))
    var t = qs_pair_tree(ps, pq, ns)
    if t[0] == Float32(0):
        tau.unsafe_store(k, Float32(0))
        scal.unsafe_store(0, Float32(1))
        scal.unsafe_store(1, Float32(0))
        return
    var aa = abs(alpha)
    var f = qs_pair(t[0], t[1], aa, Float32(1) if aa != Float32(0) else Float32(0))
    var nrm = ftz(identical_mul(sqrt0(f[1]), f[0]))
    var beta = -nrm if alpha >= Float32(0) else nrm
    tau.unsafe_store(k, div0(sub(beta, alpha), beta))
    scal.unsafe_store(0, sub(alpha, beta))
    scal.unsafe_store(1, Float32(1))
    a.unsafe_store(k * n + k, beta)


@always_inline
def qs_dot_finish(seed: Float32, p: F32Ptr, ns: Int) -> Float32:
    """w = ftz(ftz(seed) + the tree of the ns slice partials at p)."""
    return ftz(ftz(seed) + qs_tree(p, ns))
