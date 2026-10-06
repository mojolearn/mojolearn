# SPDX-License-Identifier: Apache-2.0
"""Classical C20 row storage and C22 symmetric orientation kernels.
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


def classical_cache_row_kernel(x: CFP, norms: CFP, rows: CIP, q: Int32, cache: CFP, keys: CIP,
                               n: Int32, d: Int32, slots: Int32, kind: Int32, gain: Float32, offset: Float32, degree: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n):
        var row = Int(rows.unsafe_load(Int(q)))
        var slot = row % Int(slots)
        if Int(keys.unsafe_load(slot)) != row:
            cache.unsafe_store(slot * Int(n) + j, classical_kernel_cell(x, norms, row, j, Int(d), Int(kind), gain, offset, Int(degree)))


def classical_cache_publish_kernel(rows: CIP, q: Int32, keys: CIP, slots: Int32):
    var row = rows.unsafe_load(Int(q))
    keys.unsafe_store(Int(row) % Int(slots), row)


def classical_cache_gather_kernel(cache: CFP, rows: CIP, q: Int32, cols: CIP, output: CFP, n: Int32, ncols: Int32, slots: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(ncols):
        var row = Int(rows.unsafe_load(Int(q)))
        output.unsafe_store(Int(q) * Int(ncols) + j, cache.unsafe_load((row % Int(slots)) * Int(n) + Int(cols.unsafe_load(j))))


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
