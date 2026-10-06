# SPDX-License-Identifier: Apache-2.0
"""C20/C22 canonical kernel cells, shared numerical profile with GEMM v1.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_exp, identical_tanh
from gemm.contract import contract_leaf_size, leaf_count, leaf_begin, leaf_end
from svm.impl.svm_parameter import KernelParams, KERNEL_LINEAR, KERNEL_RBF, KERNEL_POLYNOMIAL, KERNEL_TANH
comptime CFP = MutPointer[Float32, MutAnyOrigin]


def classical_dot(a: CFP, b: CFP, a0: Int, b0: Int, sa: Int, sb: Int, k: Int) -> Float32:
    """GEMM v1 leaves and adjacent balanced tree, streamed in 11 words."""
    var leaf = contract_leaf_size(k)
    var count = leaf_count(k, leaf)
    var levels = InlineArray[Float32, 11](fill=Float32(0))
    var occupied = UInt32(0)
    for t in range(count):
        var acc = Float32(0)
        for q in range(leaf_begin(t, leaf), leaf_end(t, leaf, k)):
            acc = ftz(identical_mul_add(ftz(a.unsafe_load(a0 + q * sa)), ftz(b.unsafe_load(b0 + q * sb)), acc))
        var level = 0
        while (occupied & (UInt32(1) << UInt32(level))) != 0:
            acc = ftz(levels[level] + acc)
            occupied = occupied & ~(UInt32(1) << UInt32(level))
            level += 1
        levels[level] = acc
        occupied = occupied | (UInt32(1) << UInt32(level))
    var acc = Float32(0)
    var have = False
    for level in range(11):
        if (occupied & (UInt32(1) << UInt32(level))) != 0:
            acc = ftz(levels[level] + acc) if have else levels[level]
            have = True
    return acc


def classical_kernel_cell(x: CFP, norms: CFP, i: Int, j: Int, d: Int, kind: Int, gain: Float32, offset: Float32, degree: Int) -> Float32:
    var dot = classical_dot(x, x, i * d, j * d, 1, 1, d)
    if kind == KERNEL_LINEAR:
        return dot
    if kind == KERNEL_RBF:
        var distance = ftz(ftz(ftz(norms.unsafe_load(i)) + ftz(norms.unsafe_load(j))) - ftz(Float32(2) * ftz(dot)))
        return ftz(identical_exp(ftz((-gain) * distance)))
    var shifted = ftz(identical_mul_add(gain, ftz(dot), offset))
    if kind == KERNEL_TANH:
        return ftz(identical_tanh(shifted))
    var acc = Float32(1)
    for _ in range(degree):
        acc = ftz(identical_mul(acc, shifted))
    return acc
