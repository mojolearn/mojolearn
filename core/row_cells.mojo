# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Row and cell bodies that more than one family's kernels compute with the
same words (lane fx-dup-next, 2026-11-19). One body each; the families'
kernels, item functions and launches stay where they are and call it.

- `softmax_row_from_max_f32`: the float32 softmax of one row given its
  maximum: each exponential once (parked in the output row), the sum
  ascending, then the division. It was written three times
  (`x_neighbors/items.mojo::softmax_item`, `sequence/ops.mojo::op_softmax`,
  the softmax branch of `autograd/ops.mojo::op_ag_softmax_fwd`) with three
  spellings of the same flushes. Each caller keeps its own maximum loop
  (they compare raw or flushed words, which only differ in the sign of a
  zero maximum) and passes the result.
- `sqdist_diff_cell`: the squared euclidean distance of two rows by
  difference, features ascending, one fused multiply-add per term. It was
  written twice (`x_neighbors/items.mojo::sqdist_item`,
  `x_decomp/cells.mojo::sqdist_cell`). `x_cluster/bodies.mojo::sq_dist_rows`
  is NOT this body: its product is rounded before the add, so its words
  differ.

`@always_inline` bodies, called from kernels and from the host columns'
item loops: no launch, no allocation. No bit moves: every flush of the
replaced statements is here.
"""
from checks.numerics import ftz, identical_div, identical_exp, identical_mul_add


@always_inline
def softmax_row_from_max_f32[xo: MutOrigin, yo: MutOrigin, //](
    x: MutPointer[Float32, xo], y: MutPointer[Float32, yo], base: Int, c: Int, m: Float32,
):
    """`y[base + j] = exp(x[base + j] - m) / sum_j exp(x[base + j] - m)` for
    `j < c`, `m` the row's maximum: every operand and result flushed, the
    sum ascending from 0. `y` may be `x` (element j is read before it is
    written); otherwise the rows must not overlap."""
    var mf = ftz(m)
    var s = Float32(0.0)
    for j in range(c):
        var e = ftz(identical_exp(ftz(ftz(x.unsafe_load(base + j)) - mf)))
        y.unsafe_store(base + j, e)
        s = ftz(s + e)
    for j in range(c):
        y.unsafe_store(base + j, ftz(identical_div(ftz(y.unsafe_load(base + j)), s)))


@always_inline
def sqdist_diff_cell(
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin], i: Int, j: Int, d: Int,
) -> Float32:
    """`||a_i - b_j||^2` over `d` features, ascending: `t = a - b` on flushed
    operands, flushed; `acc = fma(t, t, acc)`, flushed."""
    var acc = Float32(0)
    for p in range(d):
        var t = ftz(ftz(a.unsafe_load(i * d + p)) - ftz(b.unsafe_load(j * d + p)))
        acc = ftz(identical_mul_add(t, t, acc))
    return acc
