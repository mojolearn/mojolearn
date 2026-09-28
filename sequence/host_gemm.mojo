# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE HOST GEMM OF THE SEQUENCE LANE: `sequence/ops.mojo::op_gemm`'s cells,
many at once (lane sequence-cpu, 2026-09-28). HOST ONLY.

`op_gemm` computes ONE output cell per element:

    acc = accumulate ? ftz(C[m, n]) : 0
    for k ascending:  acc = ftz(identical_mul_add(ftz(A(m, k)), ftz(B(k, n)), acc))
    C[m, n] = ftz(acc)

This file computes the same cells, W columns and RB rows at a time: lane
`j` of accumulator `r` is cell `(m + r, n + j)`, and it goes through
exactly that sequence of operations, one fused multiply-add per term, k in
the same order, each result through the same flush. A vector lane is not a
different arithmetic: an IEEE fused multiply-add rounds once per lane, and
`_ftz_v` is `checks/numerics.mojo::ftz` lane by lane (a subnormal becomes
its signed zero, every other word is returned unchanged). No sum crosses a
cell, so which lane, which row block and which thread computes a cell moves
no bit: the cells equal `op_gemm`'s, and so the device's, at every width and
every thread count. The seam is IDENTITY_PATHS row 9 (the contraction pin)
and row 10 (the denormal pin), restated; the lane check's CPU == GPU diff
and the host sabotage (k descending, `SEQUENCE_HOST_SABOTAGE`, honored here
too) cover it.

B is read along n. When its n stride is not 1 (a weight stored [N, K], the
recurrent `x W^T` products) the launch first copies B into a contiguous
[K, N] panel (a copy, no arithmetic: `ftz` of every value, which `op_gemm`
applies on read anyway). Columns past the last full vector and rows past the
last full row block run `op_gemm` itself.
"""
from std.memory import bitcast
from std.sys.info import simd_width_of

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add_simd,
)
from sequence.ops import FP, Args, SEQUENCE_HOST_SABOTAGE, op_gemm

#: Columns per vector: the host's native float32 width (16 on AVX-512, 8 on
#: AVX2, 4 on NEON). A width moves no bit (one cell per lane).
comptime HG_W = simd_width_of[DType.float32]()
#: Rows per block: each B vector load feeds RB rows.
comptime HG_RB = 4


@always_inline
def _ftz_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`ftz`, lane by lane, by bits: under IDENTICAL a word whose exponent
    field is zero and whose mantissa is not becomes its sign bit alone;
    under FAST it is the identity, as `ftz` is."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var b = bitcast[DType.uint32, w](x)
        var sub = (b & SIMD[DType.uint32, w](0x7F800000)).eq(SIMD[DType.uint32, w](0)) & (
            b & SIMD[DType.uint32, w](0x007FFFFF)
        ).ne(SIMD[DType.uint32, w](0))
        return sub.select(bitcast[DType.float32, w](b & SIMD[DType.uint32, w](0x80000000)), x)
    return x


@always_inline
def _fma_v[w: Int](
    a: SIMD[DType.float32, w], b: SIMD[DType.float32, w], c: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    """`fma3` lane by lane: `ftz(identical_mul_add(a, b, c))`."""
    return _ftz_v[w](identical_mul_add_simd[w](a, b, c))


def host_gemm_pack(a: Args, panel: FP):
    """B(k, n) = p1[k*i5 + n*i6] -> panel[k * N + n], flushed (op_gemm's
    `ld`); a copy, no arithmetic."""
    var N = a.i1
    var K = a.i2
    for k in range(K):
        for n in range(N):
            panel.unsafe_store(k * N + n, ftz(a.p1.unsafe_load(k * a.i5 + n * a.i6)))


@always_inline
def _block[R: Int](a: Args, bp: FP, ldb: Int, m0: Int, n0: Int):
    """Cells (m0 .. m0+R-1, n0 .. n0+HG_W-1): op_gemm's statement, R rows
    by HG_W columns of lanes. `bp` is B with row stride `ldb` and unit n
    stride."""
    comptime W = HG_W
    var K = a.i2
    var acc = InlineArray[SIMD[DType.float32, W], R](fill=SIMD[DType.float32, W](0.0))
    if a.i7 != 0:
        comptime for r in range(R):
            acc[r] = _ftz_v[W](a.p2.load[width=W]((m0 + r) * a.i8 + n0))
    var abase = InlineArray[Int, R](fill=0)
    comptime for r in range(R):
        abase[r] = (m0 + r) * a.i3

    for q in range(K):
        var k = q
        comptime if SEQUENCE_HOST_SABOTAGE:
            # THE SABOTAGE ARM, as op_gemm's: k DESCENDING.
            k = K - 1 - q
        var bv = _ftz_v[W](bp.load[width=W](k * ldb + n0))
        comptime for r in range(R):
            var av = SIMD[DType.float32, W](ftz(a.p0.unsafe_load(abase[r] + k * a.i4)))
            acc[r] = _fma_v[W](av, bv, acc[r])
    comptime for r in range(R):
        a.p2.store((m0 + r) * a.i8 + n0, _ftz_v[W](acc[r]))


def host_gemm_rows(a: Args, bp: FP, ldb: Int, lo: Int, hi: Int):
    """Rows [lo, hi) of the product, every cell `op_gemm`'s bits."""
    comptime W = HG_W
    var N = a.i1
    var n_vec = (N // W) * W
    var m = lo
    while m + HG_RB <= hi:
        var n = 0
        while n < n_vec:
            _block[HG_RB](a, bp, ldb, m, n)
            n += W
        for r in range(HG_RB):
            for nn in range(n_vec, N):
                op_gemm((m + r) * N + nn, a)
        m += HG_RB
    while m < hi:
        var n = 0
        while n < n_vec:
            _block[1](a, bp, ldb, m, n)
            n += W
        for nn in range(n_vec, N):
            op_gemm(m * N + nn, a)
        m += 1
