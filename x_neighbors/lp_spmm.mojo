# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Label propagation's sparse graph product (lane neighbors-apple2's kernel,
moved to its own module by lane neighbors-apple3 so that
x_neighbors/iter_device.mojo and x_neighbors/lp_batched.mojo both import
it)."""
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_mul_add
from x_neighbors.items import FP, IP


def lp_spmm_kernel(
    indptr: IP, cols: IP, vals: FP, x: FP, res: FP, n_: Int64, c_: Int64,
):
    """`matmul_item` (G x, cell t = i*c + j, p ascending) over G's NONZERO
    entries only, columns ascending. Exact when every x is finite: a
    skipped term is fma(+-0, x, acc) with x finite, whose product is a zero
    and whose sum is acc unchanged, because acc starts at +0.0 and an fma
    returns -0.0 only from (-0) + (-0), so acc is never -0.0. The caller
    checks x's finiteness each iteration and runs the dense kernel when it
    fails."""
    var n = Int(n_)
    var c = Int(c_)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * c:
        return
    var i = t // c
    var j = t - i * c
    var acc = Float32(0)
    for e in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
        acc = ftz(identical_mul_add(ftz(vals.unsafe_load(e)), ftz(x.unsafe_load(Int(cols.unsafe_load(e)) * c + j)), acc))
    res.unsafe_store(t, acc)
