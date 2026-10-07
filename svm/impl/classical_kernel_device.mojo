# SPDX-License-Identifier: Apache-2.0
"""Classical C20 pair-load gather and C22 symmetric orientation kernels.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from std.gpu import block_idx, block_dim, thread_idx
from svm.impl.classical_kernel_cells import CFP, classical_kernel_cell
comptime CIP = MutPointer[Int32, MutAnyOrigin]


def classical_triangle_kernel(x: CFP, norms: CFP, output: CFP, n: Int32, d: Int32, kind: Int32, gain: Float32, offset: Float32, degree: Int32):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    if c < nn * nn:
        var i = c // nn
        var j = c % nn
        if j >= i:
            var v = classical_kernel_cell(x, norms, i, j, Int(d), Int(kind), gain, offset, Int(degree))
            output.unsafe_store(i * nn + j, v)
            if i != j:
                output.unsafe_store(j * nn + i, v)


def classical_gather_rows_norms_kernel(dst: CFP, dst_norm: CFP, x: CFP, norms: CFP, rows: CIP,
                                      n: Int32, d: Int32, with_norm: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n) * Int(d):
        var row = t // Int(d)
        var col = t % Int(d)
        var source = Int(rows.unsafe_load(row))
        dst.unsafe_store(t, x.unsafe_load(source * Int(d) + col))
        if col == 0 and with_norm != 0:
            dst_norm.unsafe_store(row, norms.unsafe_load(source))
