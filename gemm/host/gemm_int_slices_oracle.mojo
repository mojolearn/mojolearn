# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE HOST COLUMN of the research profile `mojolearn.identical.gemm.fp32.v2-int`
(prototype, lane/gemm-int-slices, 2026-10-28; `docs/plans/GEMM_INT_SLICES.md`).

The answer every device plan of `gemm/int_slices.mojo` must equal bit for
bit. It reaches the integers by a DIFFERENT route from the device plans (codes
straight from the floats, the four half products as Int64 sums of Int32
products, `p` ascending; no planes, no matrix unit) and through the SAME seams
(`gemm/int_slices_seams.mojo`) from there, so a device plan that stages,
tiles, splits or orders its integer sums differently cannot agree with it by
accident, and one that agrees has the same integers.

Host Mojo on host memory: this is the CPU column's computation (the CPU
column runs no kernel, `checks/kernel_matrix.mojo::lib_int8_matrix_unit_for`),
not Python and not a host step inside a GPU route.
"""

from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.int_slices_seams import (
    INT_SLICES_MAX_K,
    int_slices_abs_ftz,
    int_slices_cell,
    int_slices_code,
    int_slices_half_hi,
    int_slices_half_lo,
    int_slices_row_exponent,
)


def int_slices_strides(op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int]:
    """`(a_row_stride, a_k_stride, b_row_stride, b_k_stride)`: element `(r, p)`
    of a "row along k" of op(A) (r < m) or of op(B) (r < n) sits at
    `r * row_stride + p * k_stride` of the stored row-major operand
    (contract section 0.1's three orientations)."""
    var ars = k
    var aps = 1
    if op == OP_TN:
        ars = 1
        aps = m
    var brs = 1
    var bps = n
    if op == OP_NT:
        brs = k
        bps = 1
    return (ars, aps, brs, bps)


def _row_codes(
    x: MutPointer[Float32, MutUntrackedOrigin], r: Int, k: Int, rs: Int, ps: Int, mut q: List[Int32]
) -> Int:
    """Clauses S1 and S2 for one row along k: fills `q` (length k), returns `e`."""
    # driver-loop: over the k values of one row (host column, the oracle itself)
    var best = Float32(0.0)
    for p in range(k):
        var v = int_slices_abs_ftz(x.unsafe_load(r * rs + p * ps))
        if v > best:
            best = v
    var e = int_slices_row_exponent(best)
    for p in range(k):
        q[p] = int_slices_code(x.unsafe_load(r * rs + p * ps), e)
    return e


def gemm_int_slices_oracle(
    c: MutPointer[Float32, MutUntrackedOrigin],
    a: MutPointer[Float32, MutUntrackedOrigin],
    b: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`C[m x n] = op(A) . op(B)` under the prototype profile, on the host.
    Refuses by name a non-positive extent, an unknown `op` and `k` above
    `INT_SLICES_MAX_K`."""
    if m <= 0 or n <= 0 or k <= 0 or k > INT_SLICES_MAX_K:
        raise Error(
            "gemm_int_slices_oracle: m, n, k must be positive and k at most "
            + String(INT_SLICES_MAX_K) + ", got m=" + String(m) + " n=" + String(n) + " k=" + String(k)
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("gemm_int_slices_oracle: op must be 0, 1 or 2, got " + String(op))
    var st = int_slices_strides(op, m, n, k)
    var qa = List[Int32](length=m * k, fill=Int32(0))
    var qb = List[Int32](length=n * k, fill=Int32(0))
    var ea = List[Int](length=m, fill=0)
    var eb = List[Int](length=n, fill=0)
    var row = List[Int32](length=k, fill=Int32(0))
    # driver-loop: over the rows of op(A) (host column)
    for i in range(m):
        ea[i] = _row_codes(a, i, k, st[0], st[1], row)
        for p in range(k):
            qa[i * k + p] = row[p]
    # driver-loop: over the rows of op(B) (host column)
    for j in range(n):
        eb[j] = _row_codes(b, j, k, st[2], st[3], row)
        for p in range(k):
            qb[j * k + p] = row[p]
    # driver-loop: over the output cells (host column)
    for i in range(m):
        for j in range(n):
            var p_hh = Int64(0)
            var p_hl = Int64(0)
            var p_lh = Int64(0)
            var p_ll = Int64(0)
            for p in range(k):
                var x = qa[i * k + p]
                var y = qb[j * k + p]
                var xh = int_slices_half_hi(x)
                var xl = int_slices_half_lo(x)
                var yh = int_slices_half_hi(y)
                var yl = int_slices_half_lo(y)
                p_hh += Int64(xh * yh)
                p_hl += Int64(xh * yl)
                p_lh += Int64(xl * yh)
                p_ll += Int64(xl * yl)
            c.unsafe_store(i * n + j, int_slices_cell(p_hh, p_hl, p_lh, p_ll, ea[i] + eb[j]))
