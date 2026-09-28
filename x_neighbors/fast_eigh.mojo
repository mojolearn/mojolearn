# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`symmetric_eig_host[float32]` with its rotations as vectors (lane
neighbors-apple3, 2026-09-28). Host code.

spectral/checks/symmetric_eig_host.mojo is Numerical Recipes' cyclic Jacobi
over the UPPER triangle of a row-major matrix: a rotation (p, q) walks
column p and column q above the pair (a stride-n walk), and columns p and q
of the basis (another stride-n walk), one element at a time. KernelPCA.fit
at 500 rows spent its whole 0.4 to 0.6 s (FAST, M3 Ultra) there.

Here the matrix is kept FULL and symmetric and the basis TRANSPOSED, so the
pairs a rotation touches are two contiguous rows:

  * NR's three loops rotate the pairs (A(j, p), A(j, q)) for every j that is
    neither p nor q, the p-side value first. By symmetry those are
    (row p [j], row q [j]), so one vector pass over rows p and q rotates them
    all, each pair by the same two multiply-adds (`_rotate`'s) on the same
    two operands;
  * the rotated rows are then copied into columns p and q (a copy, no
    arithmetic), which keeps the matrix symmetric for the next pair;
  * cells (p, p), (q, q), (p, q), (q, p) of the two rows are not pairs NR
    rotates: the vector pass runs over them and they are reset after it
    (the diagonal lives in `d`, as in NR, and (p, q) is zero after its
    rotation);
  * the basis pairs (V(j, p), V(j, q)) are rows p and q of the transposed
    basis.

The off-diagonal sum, the thresholds, the angle, the diagonal updates, the
ascending (value, index) order and the sign pin (DEVIATION 770) are the
reference routine's statements in its order. A vector's lanes are different
cells, never pieces of one cell's arithmetic.

Used where `x_neighbors/eigh.mojo` and kernel_methods' Nystroem select it
(FAST on Apple). Under FAST the products and sums are the compiler's
(`a * b + c`), so a FAST word may differ from the scalar routine's by a
rounding; the paired quality check is bench/x_neighbors_fast_quality.py.
"""
from checks.numerics import (
    ftz,
    ftz_simd,
    identical_mul_add,
    identical_mul_add_simd,
    identical_sqrt,
)
from std.sys.info import simd_width_of

comptime EIG_W = simd_width_of[DType.float32]()
comptime EigV = SIMD[DType.float32, EIG_W]
comptime EigP = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _rotate_rows(g_row: EigP, h_row: EigP, n: Int, s: Float32, tau: Float32):
    """`_rotate` on the pairs (g_row[j], h_row[j]), j in [0, n)."""
    var sv = EigV(s)
    var nsv = EigV(-s)
    var tv = EigV(tau)
    var j = 0
    while j + EIG_W <= n:
        var g = g_row.unsafe_load[width=EIG_W](j)
        var h = h_row.unsafe_load[width=EIG_W](j)
        var t1 = ftz_simd[EIG_W](identical_mul_add_simd[EIG_W](g, tv, h))
        var ng = ftz_simd[EIG_W](identical_mul_add_simd[EIG_W](nsv, t1, g))
        var t2 = ftz_simd[EIG_W](identical_mul_add_simd[EIG_W](-h, tv, g))
        var nh = ftz_simd[EIG_W](identical_mul_add_simd[EIG_W](sv, t2, h))
        g_row.unsafe_store(j, ng)
        h_row.unsafe_store(j, nh)
        j += EIG_W
    while j < n:
        var g1 = g_row.unsafe_load(j)
        var h1 = h_row.unsafe_load(j)
        var u1 = ftz(identical_mul_add(g1, tau, h1))
        var ng1 = ftz(identical_mul_add(-s, u1, g1))
        var u2 = ftz(identical_mul_add(-h1, tau, g1))
        var nh1 = ftz(identical_mul_add(s, u2, h1))
        g_row.unsafe_store(j, ng1)
        h_row.unsafe_store(j, nh1)
        j += 1


def symmetric_eig_rows(
    src: EigP, n: Int, evals: EigP, evecs: EigP, max_sweeps: Int = 60
) raises -> Int:
    """The symmetric `n x n` row-major `src` (its upper triangle and its
    diagonal are read, as the reference routine reads them): `evals` gets
    the n eigenvalues ascending by (value, index), `evecs` the eigenvectors,
    n x n row-major, eigenvector c in COLUMN c, signs pinned. Returns the
    sweeps used."""
    if n <= 0:
        raise Error("symmetric_eig_rows: n must be positive")
    var zero = Float32(0)
    var one = Float32(1)
    var am = List[Float32](length=n * n, fill=zero)
    var vt = List[Float32](length=n * n, fill=zero)
    var d = List[Float32](length=n, fill=zero)
    var b = List[Float32](length=n, fill=zero)
    var z = List[Float32](length=n, fill=zero)
    var pa = EigP(unsafe_from_address=Int(am.unsafe_ptr()))
    var pv = EigP(unsafe_from_address=Int(vt.unsafe_ptr()))
    for i in range(n):
        for j in range(i + 1, n):
            var x = src.unsafe_load(i * n + j)
            pa.unsafe_store(i * n + j, x)
            pa.unsafe_store(j * n + i, x)
        var dii = src.unsafe_load(i * n + i)
        d[i] = dii
        b[i] = dii
        pv.unsafe_store(i * n + i, one)

    var sweeps_used = 0
    for sweep in range(1, max_sweeps + 1):
        sweeps_used = sweep
        var sm = zero
        for p in range(n - 1):
            for q in range(p + 1, n):
                sm = ftz(sm + abs(pa.unsafe_load(p * n + q)))
        if sm == zero:
            break
        var tresh = zero
        if sweep < 4:
            var t0 = ftz(Float32(0.2) * sm)
            tresh = ftz(t0 / Float32(n * n))
        for p in range(n - 1):
            var rp = pa.unsafe_offset(p * n)
            var vp = pv.unsafe_offset(p * n)
            for q in range(p + 1, n):
                var apq = rp.unsafe_load(q)
                var g = ftz(Float32(100) * abs(apq))
                var dp_abs = abs(d[p])
                var dq_abs = abs(d[q])
                if (
                    sweep > 4
                    and ftz(dp_abs + g) == dp_abs
                    and ftz(dq_abs + g) == dq_abs
                ):
                    rp.unsafe_store(q, zero)
                    pa.unsafe_store(q * n + p, zero)
                elif abs(apq) > tresh:
                    var h = ftz(d[q] - d[p])
                    var t: Float32
                    var h_abs = abs(h)
                    if ftz(h_abs + g) == h_abs:
                        t = ftz(apq / h)
                    else:
                        var theta = ftz(ftz(Float32(0.5) * h) / apq)
                        var th2 = ftz(identical_mul_add(theta, theta, one))
                        var den = ftz(abs(theta) + identical_sqrt(th2))
                        t = ftz(one / den)
                        if theta < zero:
                            t = -t
                    var t2 = ftz(identical_mul_add(t, t, one))
                    var c = ftz(one / identical_sqrt(t2))
                    var s = ftz(t * c)
                    var tau = ftz(s / ftz(one + c))
                    var hh = ftz(t * apq)
                    z[p] = ftz(z[p] - hh)
                    z[q] = ftz(z[q] + hh)
                    d[p] = ftz(d[p] - hh)
                    d[q] = ftz(d[q] + hh)
                    var rq = pa.unsafe_offset(q * n)
                    _rotate_rows(rp, rq, n, s, tau)
                    # the four cells of the pair are not rotated pairs
                    rp.unsafe_store(p, zero)
                    rp.unsafe_store(q, zero)
                    rq.unsafe_store(p, zero)
                    rq.unsafe_store(q, zero)
                    # columns p and q are rows p and q (a copy)
                    for j in range(n):
                        if j != p and j != q:
                            pa.unsafe_store(j * n + p, rp.unsafe_load(j))
                            pa.unsafe_store(j * n + q, rq.unsafe_load(j))
                    _rotate_rows(vp, pv.unsafe_offset(q * n), n, s, tau)
        for p in range(n):
            b[p] = ftz(b[p] + z[p])
            d[p] = b[p]
            z[p] = zero

    # ascending by (value, index): the reference's stable insertion sort
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var key = order[i]
        var j = i - 1
        while j >= 0:
            if not (d[order[j]] > d[key]):
                break
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    for c in range(n):
        var col = order[c]
        evals.unsafe_store(c, d[col])
        var row = pv.unsafe_offset(col * n)
        # DEVIATION 770: the first component that is not a zero is positive
        var r = 0
        while r < n and row.unsafe_load(r) == zero:
            r += 1
        var negate = r < n and row.unsafe_load(r) < zero
        for rr in range(n):
            var e = row.unsafe_load(rr)
            evecs.unsafe_store(rr * n + c, -e if negate else e)
    _ = am^
    _ = vt^
    return sweeps_used
