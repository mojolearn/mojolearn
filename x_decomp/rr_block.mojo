# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The BLOCK Jacobi eigh for large n (lane fam2-decomp, 2026-10-04), IDENTICAL
builds: the schedule, the gate and the pinned cells shared by the device
kernels (x_decomp/rr_block_device.mojo) and the host solver
(x_decomp/rr_solve.mojo `host_eigh_rb_sorted`). No GPU imports.

The rotation solver (x_decomp/rr.mojo) is two launches a round and m - 1
rounds a sweep, each launch rewriting all of A and V: at the board's n = 4096
that is 8,190 launches a sweep, each a pass over 128 MB with four scattered
rows per thread. Here the columns are cut into blocks of RB_B; the blocks
play the same round-robin tournament (`pj_first` / `pj_second` on M blocks,
M even, the matrix zero-padded to N = M RB_B), and a block round is:

  1. gather the H = M / 2 pivot problems [[A_II, A_IJ], [A_JI, A_JJ]]
     (RB_W x RB_W, RB_W = 2 RB_B);
  2. solve every one completely with the batched round-robin Jacobi
     (x_decomp/rr_batch.mojo, one thread block a problem: values ascending,
     vectors W_g in columns), to the tightened tolerance `rb_tol`;
  3. A = W^T A W: T = A W on the block pairs gi < gj (`rb_row_dot`), then
     W^T T (`rb_left_cell`) stored to the cell and its mirror, the pivot
     blocks set to diag(w_g) (the closed form, as `rr_block` zeroes a_pq);
  4. V = V W (`rb_row_dot`), into the other V buffer.

M - 1 block rounds a sweep (255 at n = 4096, RB_B = 16), six launches each,
every product a chain of RB_W terms read along rows. The convergence test
is the rotation solver's own (`rr_off_fold` on the padded matrix,
`rr_converged`, `rr_fro_kept`), before every sweep.

THE LOCAL TOLERANCE. A pivot problem that passes its own test returns a
permutation, so the outer iteration is at a fixed point when every pivot
passes. Summing the pivots' tests over a sweep bounds the global
off-diagonal mass by (M - 1) tol_local^2 ||A||_F^2, so tol_local = tol 2^-k
with 4^k >= M (`rb_tol`, exact scaling) makes a fixed point pass the global
test: the iteration cannot stall between the two tolerances.

PADDING. Pad rows and columns are exact zeros with a zero diagonal: every
rotation against a pad index is (1, 0), every chain term through a pad cell
is an exact zero, so a pad eigenvector stays a unit vector on a pad row and
a real one stays zero there. The pivot solves sort, so pads wander among
the columns; the tail finds them by their pad-row cell and ranks the real
columns only (`rb_rank_real`).

EXPERIMENTAL ONLY: the default is disabled after the IDENTICAL 4096 rank-one
quality fixture produced an intrinsically unconverged local pivot (outer
sweep 0, round 183, group 71). Exact-word host replay also refuses after
60/120/180/240 sweeps at the required local tolerance. Do not relax that
tolerance or the final convergence/accuracy gates to enable this candidate.

BITS: a different solver, so different words from the rotation solver at
n >= RB_MIN_N, on NVIDIA, AMD, Apple and the host column together (the gate
below is one comptime constant every one of them imports; the cells here
are the only arithmetic). -D MOJOLEARN_IDN_EIGH_BLOCK_ON explicitly enables
the experimental candidate; -D MOJOLEARN_IDN_EIGH_BLOCK_OFF (or
-D MOJOLEARN_IDN_ALL_OFF) restores the rotation solver at every size.

CANDIDATE ARMS (default off, each moves the words on all four columns):
  -D MOJOLEARN_IDN_EIGH_BLOCK_B8      RB_B = 8  (pivots 16 x 16)
  -D MOJOLEARN_IDN_EIGH_BLOCK_B32     RB_B = 32 (pivots 64 x 64)
  -D MOJOLEARN_IDN_EIGH_BLOCK_MIN128  the block form from n = 128 (default 512)
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul, identical_mul_add
from std.memory import bitcast
from x_decomp.cells import F32Ptr
from x_decomp.rr import pj_first, pj_second

comptime IDN_EIGH_BLOCK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_EIGH_BLOCK_ON"]() and not (is_defined["MOJOLEARN_IDN_EIGH_BLOCK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())

comptime _RB_B_WIDE = 32 if is_defined["MOJOLEARN_IDN_EIGH_BLOCK_B32"]() else 16
#: columns a block
comptime RB_B = 8 if is_defined["MOJOLEARN_IDN_EIGH_BLOCK_B8"]() else _RB_B_WIDE
#: order of a pivot problem
comptime RB_W = 2 * RB_B
#: the block form's least n
comptime RB_MIN_N = 128 if is_defined["MOJOLEARN_IDN_EIGH_BLOCK_MIN128"]() else 512


@always_inline
def rb_key(v: Float32) -> UInt32:
    """decomposition/spectrum_order_device.mojo `spectrum_key` (kept here so
    the host solver imports no device module): monotone in the float order,
    -0.0 keyed as +0.0."""
    var b = bitcast[DType.uint32](v)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    return b ^ UInt32(0xFFFFFFFF) if (b >> 31) == 1 else b | UInt32(0x80000000)


@always_inline
def rb_use(n: Int) -> Bool:
    """The block solver is THE eigh at this n (device and host alike)."""
    var use = False
    comptime if IDN_EIGH_BLOCK:
        use = n >= RB_MIN_N
    return use


@always_inline
def rb_blocks(n: Int) -> Int:
    """M: blocks of RB_B covering n, made even (the odd one out is a block
    of padding)."""
    var mb = (n + RB_B - 1) // RB_B
    return mb + (mb % 2)


@always_inline
def rb_tol(tol: Float32, m: Int) -> Float32:
    """tol 2^-k, the least k with 4^k >= m (see THE LOCAL TOLERANCE)."""
    var t = tol
    var c = 1
    while c < m:
        t = ftz(identical_mul(t, Float32(0.5)))
        c *= 4
    return t


@always_inline
def rb_lo(m: Int, r: Int, g: Int) -> Int:
    """The lower block of pair g in block round r."""
    return min(pj_first(r, g, m), pj_second(r, g, m))


@always_inline
def rb_hi(m: Int, r: Int, g: Int) -> Int:
    return max(pj_first(r, g, m), pj_second(r, g, m))


@always_inline
def rb_idx(lo: Int, hi: Int, x: Int) -> Int:
    """The matrix index of local index x (0 <= x < RB_W) of the pair of
    blocks lo < hi: increasing in x."""
    if x < RB_B:
        return lo * RB_B + x
    return hi * RB_B + (x - RB_B)


@always_inline
def rb_pivot_shift(a: F32Ptr, nn: Int, lo: Int, hi: Int) -> Float32:
    """Common spectral translation with a nonincreasing rounded local norm.

    Captured near-repeated diagonal-2 pivot: original60/120/180/240 sweeps
    refused; centered solve converged4, residual4.2e-10, orthogonality2.4e-6.
    No tolerance relaxation: each diagonal magnitude is checked separately.
    The shift MUST be restored to local eigenvalues before outer updates.
    """
    var first = rb_idx(lo, hi, 0)
    var shift = a.unsafe_load(first * nn + first)
    for k in range(RB_W):
        var index = rb_idx(lo, hi, k)
        var diagonal = a.unsafe_load(index * nn + index)
        if abs(ftz(diagonal - shift)) > abs(diagonal):
            return Float32(0.0)
    return shift


@always_inline
def rb_gather_cell(a: F32Ptr, nn: Int, lo: Int, hi: Int, x: Int, y: Int) -> Float32:
    """Symmetric block pivot, centered only on its diagonal.

    Subtracting one scalar preserves eigenvectors. Off-diagonals stay exact;
    the guarded smaller norm makes the same relative stopping test at least
    as strict. Sweep budget and outer convergence/Frobenius gates unchanged.
    """
    var i = rb_idx(lo, hi, min(x, y))
    var j = rb_idx(lo, hi, max(x, y))
    var value = a.unsafe_load(i * nn + j)
    if x != y:
        return value
    return ftz(value - rb_pivot_shift(a, nn, lo, hi))


@always_inline
def rb_row_dot(row: F32Ptr, wg: F32Ptr, lo: Int, hi: Int, lc: Int) -> Float32:
    """sum over y ascending of row[idx(y)] W[y, lc]: a cell of A W or of V W
    (`row` the matrix row's first cell, `wg` the pair's RB_W x RB_W vectors,
    row major, vector lc in column lc)."""
    var acc = Float32(0.0)
    for y in range(RB_W):
        acc = ftz(identical_mul_add(row.unsafe_load(rb_idx(lo, hi, y)), wg.unsafe_load(y * RB_W + lc), acc))
    return acc


@always_inline
def rb_left_cell(t: F32Ptr, wg: F32Ptr, nn: Int, gi: Int, lr: Int, v: Int) -> Float32:
    """sum over x ascending of W_gi[x, lr] T[gi RB_W + x, v]: a cell of
    W^T (A W). T is in PAIR coordinates (row gi RB_W + x, column v)."""
    var acc = Float32(0.0)
    var base = t + (gi * RB_W) * nn + v
    for x in range(RB_W):
        acc = ftz(identical_mul_add(wg.unsafe_load(x * RB_W + lr), base.unsafe_load(x * nn), acc))
    return acc


@always_inline
def rb_is_pad(v: F32Ptr, nn: Int, n: Int, c: Int) -> Bool:
    """Column c of the N x N basis is a pad vector: a nonzero cell on a pad
    row (rows n .. N - 1)."""
    for k in range(n, nn):
        if v.unsafe_load(k * nn + c) != Float32(0.0):
            return True
    return False


@always_inline
def rb_rank_real(key: F32Ptr, pad: F32Ptr, nn: Int, i: Int) -> Int:
    """`spectrum_rank_desc` among the real (pad[j] == 0) columns: the real
    values ahead of i descending, ties to the lower index."""
    var ki = rb_key(key.unsafe_load(i))
    var r = 0
    for j in range(nn):
        if pad.unsafe_load(j) == Float32(0.0):
            var kj = rb_key(key.unsafe_load(j))
            if kj > ki or (kj == ki and j < i):
                r += 1
    return r
