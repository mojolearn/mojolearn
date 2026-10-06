# SPDX-License-Identifier: Apache-2.0
"""C37 production centroid update, opt-in, source-only unverified."""
from std.gpu import block_idx, block_dim, thread_idx
from core.classical_centroid import classical_centroid_cell

def classical_centroid_kernel[GATED: Bool](
    gate: MutPointer[Int32,MutAnyOrigin], output: MutPointer[Float32,MutAnyOrigin],
    old: MutPointer[Float32,MutAnyOrigin], x: MutPointer[Float32,MutAnyOrigin],
    labels: MutPointer[UInt32,MutAnyOrigin], weights: MutPointer[Float32,MutAnyOrigin],
    n: Int32, k: Int32, d: Int32,
):
    comptime if GATED:
        if gate[0] != 0:
            return
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell < Int(k)*Int(d):
        output[cell] = classical_centroid_cell(x,labels,weights,old[cell],Int(n),Int(d),cell//Int(d),cell%Int(d))
