# SPDX-License-Identifier: Apache-2.0
"""Opt-in Apple FAST row gather, recovered independently of old resample bundle."""
from max.gpu import block_idx, block_dim, thread_idx


def gather_rows_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], count: Int, d: Int,
):
    var o = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if o < count * d:
        var r = o // d
        dst.unsafe_store(o, src.unsafe_load(Int(rows.unsafe_load(r)) * d + o % d))
