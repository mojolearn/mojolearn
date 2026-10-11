# SPDX-License-Identifier: Apache-2.0
"""C04 centered loads with the incumbent non-split GEMM v1 fold.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
No intermediate data matrix: subtraction has exactly the stored FTZ seam.
The scalar is shared by host and device, independently of GPU geometry.
"""
from checks.numerics import ftz, identical_mul_add
from gemm.contract import contract_leaf_size
from core.blocked_moments_ops import BM_TPB, bm_add, bm_div, bm_mean_leaf_rows, bm_sub_chains


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


def centered_cross_v1_cell(
    x: MutPointer[Float32, MutAnyOrigin], means: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin], y_mean: Float32,
    rows: Int, cols: Int, j: Int,
) -> Float32:
    """`centered_gram_v1_cell` for the cross moment (X^T y)_j = sum over rows
    of (x_rj - mean_j)(y_r - y_mean): the same leaves of `contract_leaf_size(rows)`
    rows, the same chain and the same binary-counter fold (the host column
    of core/blocked_moments.mojo::bm_centered_cross_panels)."""
    var leaf = contract_leaf_size(rows)
    var stack = InlineArray[Float32, 32](fill=Float32(0))
    var depth = 0
    var leaves = 0
    var ym = ftz(y_mean)
    var mj = ftz(means.unsafe_load(j))
    for start in range(0, rows, leaf):
        var acc = Float32(0)
        for row in range(start, min(start+leaf, rows)):
            var a = ftz(ftz(x.unsafe_load(row*cols+j))-mj)
            var b = ftz(ftz(y.unsafe_load(row))-ym)
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


def bm_column_mean_cell(x: MutPointer[Float32, MutAnyOrigin], rows: Int, cols: Int, j: Int) -> Float32:
    """One column of core/blocked_moments.mojo::bm_column_mean as a scalar
    cell (lane lars-grid-gram, 2026-10-11; the statements of
    core/blocked_moments_host.mojo::host_bm_column_mean for column j): leaves
    of `bm_mean_leaf_rows(rows)` rows, each leaf `bm_sub_chains(cols)`
    interleaved chains added in order (one chain when cols >= BM_TPB), the
    leaves folded by the binary counter (the stack form of `bm_sum_fold`'s
    levels, as `centered_gram_v1_cell`), one division by rows."""
    if rows < 1:
        return Float32(0)
    var leaf = bm_mean_leaf_rows(rows)
    var chains = bm_sub_chains(cols) if cols <= BM_TPB else 1
    var stack = InlineArray[Float32, 32](fill=Float32(0))
    var depth = 0
    var leaves = 0
    for start in range(0, rows, leaf):
        var stop = min(start + leaf, rows)
        var acc = Float32(0)
        for s in range(chains):
            var c = Float32(0)
            var r = start + s
            while r < stop:
                c = bm_add(c, x.unsafe_load(r * cols + j))
                r += chains
            acc = c if s == 0 else bm_add(acc, c)
        var carry = leaves
        while (carry & 1) != 0:
            depth -= 1
            acc = bm_add(stack[depth], acc)
            carry >>= 1
        stack[depth] = acc
        depth += 1
        leaves += 1
    depth -= 1
    var result = stack[depth]
    while depth > 0:
        depth -= 1
        result = bm_add(stack[depth], result)
    return bm_div(result, Float32(rows))
