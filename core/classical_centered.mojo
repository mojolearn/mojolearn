# SPDX-License-Identifier: Apache-2.0
"""C04 centered loads with the incumbent non-split GEMM v1 fold.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
No intermediate data matrix: subtraction has exactly the stored FTZ seam.
The scalar is shared by host and device, independently of GPU geometry.
"""
from checks.numerics import ftz, identical_mul_add
from gemm.contract import contract_leaf_size


def centered_gram_v1_cell(
    x: MutPointer[Float32, MutAnyOrigin], means: MutPointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int, i: Int, j: Int,
) -> Float32:
    var leaf = contract_leaf_size(rows)
    var stack = InlineArray[Float32, 32](fill=Float32(0))
    var depth = 0
    var leaves = 0
    for start in range(0, rows, leaf):
        var acc = Float32(0)
        for row in range(start, min(start+leaf, rows)):
            var a = ftz(ftz(x.unsafe_load(row*cols+i))-ftz(means.unsafe_load(i)))
            var b = ftz(ftz(x.unsafe_load(row*cols+j))-ftz(means.unsafe_load(j)))
            acc = ftz(identical_mul_add(a, b, acc))
        var carry = leaves
        while (carry & 1) != 0:
            depth -= 1
            acc = ftz(ftz(stack[depth])+ftz(acc))
            carry >>= 1
        stack[depth] = acc
        depth += 1
        leaves += 1
    if depth == 0:
        return Float32(0)
    depth -= 1
    var result = stack[depth]
    while depth > 0:
        depth -= 1
        result = ftz(ftz(stack[depth])+ftz(result))
    return ftz(result)
