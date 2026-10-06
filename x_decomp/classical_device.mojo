# SPDX-License-Identifier: Apache-2.0
"""Classical-only fused kernels. All candidates remain unqualified."""
from std.gpu import block_idx, block_dim, thread_idx
from x_decomp.cells import F32Ptr
from x_decomp.classical_cells import contrast_pair, centered_gram_cell


def contrast_kernel(y: F32Ptr, gx: F32Ptr, gp: F32Ptr, n: Int32, fun: Int32, alpha: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var pair = contrast_pair(y.unsafe_load(i), Int(fun), alpha)
        gx.unsafe_store(i, pair[0])
        gp.unsafe_store(i, pair[1])


def centered_gram_kernel(x: F32Ptr, means: F32Ptr, out: F32Ptr, n: Int32, d: Int32):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dd = Int(d)
    if c < dd * dd:
        var i = c // dd
        var j = c % dd
        if j >= i:
            var v = centered_gram_cell(x, means, Int(n), dd, i, j)
            out.unsafe_store(c, v)
            if i != j:
                out.unsafe_store(j * dd + i, v)
