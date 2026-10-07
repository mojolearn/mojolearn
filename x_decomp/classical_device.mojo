# SPDX-License-Identifier: Apache-2.0
"""Classical-only fused kernels. All candidates remain unqualified."""
from std.gpu import block_idx, block_dim, thread_idx
from x_decomp.cells import F32Ptr
from x_decomp.classical_cells import contrast_pair


def contrast_kernel(y: F32Ptr, gx: F32Ptr, gp: F32Ptr, n: Int32, fun: Int32, alpha: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var pair = contrast_pair(y.unsafe_load(i), Int(fun), alpha)
        gx.unsafe_store(i, pair[0])
        gp.unsafe_store(i, pair[1])
