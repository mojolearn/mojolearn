# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SPECULATIVE FLUSHED CHAIN (lane/cluster-apple, 2026-09-28).

The IDENTICAL folds that must stay one register chain in a fixed order
(`acc = ftz(acc + t)`, rows ascending: the mixture's nk and mean
log-likelihood, the cluster lane's moments) spend most of each step on the
`ftz` of the running sum: an integer test and select sit ON the chain, after
every add. `ftz` only changes a word whose exponent field is zero and whose
mantissa is not (a subnormal). A sum `y + t` has a zero exponent field only
when it is a subnormal or a zero; a zero from two zero operands is exact and
`ftz` returns it unchanged. So: run the adds plain, and beside the chain (it
never feeds back) note every sum whose exponent field is zero while an
operand is nonzero. If no sum of a block was noted, every `ftz` of the block
was the identity and the plain result IS the flushed chain's, bit for bit, on
every vendor and in both modes. If one was noted, the block is re-added
through the flushed spelling from its saved start. Only integer tests on the
bits: never a float compare on the running sum (the Apple compiler pitfall).
"""
from std.memory import bitcast

from checks.numerics import ftz


@always_inline
def suspect_sum(y: Float32, t: Float32, s: Float32) -> UInt32:
    """1 when `s = y + t` has a zero exponent field (zero or subnormal) and
    an operand is nonzero: the only sums `ftz` could change."""
    var sb = bitcast[DType.uint32](s)
    var ops_nz = (bitcast[DType.uint32](y) | bitcast[DType.uint32](t)) & UInt32(0x7FFFFFFF)
    var zero_exp = UInt32(1) if (sb & UInt32(0x7F800000)) == UInt32(0) else UInt32(0)
    var nz = UInt32(1) if ops_nz != UInt32(0) else UInt32(0)
    return zero_exp & nz


@always_inline
def ftz_chain_block[U: Int](acc: Float32, v: SIMD[DType.float32, U]) -> Float32:
    """`for u in range(U): acc = ftz(acc + ftz(v[u]))`, bit for bit, with
    the result flush off the chain (see the header)."""
    var y = acc
    var flag = UInt32(0)
    comptime for u in range(U):
        var t = ftz(v[u])
        var nx = y + t
        flag |= suspect_sum(y, t, nx)
        y = nx
    if flag != UInt32(0):
        y = acc
        comptime for u in range(U):
            y = ftz(y + ftz(v[u]))
    return y
