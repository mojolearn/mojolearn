# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""MCD MMA preparation/finalization: parallel candidates, main's cells.

Host scheduling in mcd_fast calls the existing Apple GEMM launcher with
main's per-candidate shapes. These kernels gather/center and reduce exactly
as DKit does; they do not replace the matrix multiplication's arithmetic.
"""
from std.gpu import block_idx, block_dim, thread_idx
from x_decomp.cells import F32Ptr, I32Ptr, sub, mul, add


def mc_center_kernel(
    x: F32Ptr, rows: I32Ptr, selected: I32Ptr, loc0: F32Ptr, loc1: F32Ptr,
    out: F32Ptr, active: I32Ptr, fin: I32Ptr, nc: Int32, r: Int32, d: Int32,
    per: Int32, ident: Int32, count: Int32, compact: Int32, par: Int32, use_fin: Int32,
):
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var dd = Int(d)
    var rr = Int(r)
    var take = Int(count)
    if t >= Int(nc)*take*dd:
        return
    var j = t % dd
    var z = t // dd
    var i = z % take
    var c = z // take
    var target = c*rr*dd+i*dd+j
    # MMA calls for inactive candidates may still run; zero their source
    # and guard destination publication, retaining frozen fit state.
    if use_fin == 0 and active.unsafe_load(c) == 0:
        out.unsafe_store(target, Float32(0))
        return
    var source_i = Int(selected.unsafe_load(c*rr+i)) if compact != 0 else i
    var source = source_i if ident != 0 else Int(rows.unsafe_load((c//Int(per))*rr+source_i))
    var parity = Int(fin.unsafe_load(c)) if use_fin != 0 else Int(par)
    var mean = loc1.unsafe_load(c*dd+j) if parity == 1 else loc0.unsafe_load(c*dd+j)
    out.unsafe_store(target, sub(x.unsafe_load(source*dd+j), mean))


def mc_publish_matrix_kernel(src: F32Ptr, dst: F32Ptr, active: I32Ptr, nc: Int32, d: Int32, scale: Float32):
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var cells = Int(d)*Int(d)
    if t < Int(nc)*cells:
        if active.unsafe_load(t//cells) != 0:
            dst.unsafe_store(t, mul(src.unsafe_load(t), scale))


def mc_mahal_reduce_kernel(centered: F32Ptr, product: F32Ptr, dst: F32Ptr,
                           active: I32Ptr, err: I32Ptr, nc: Int32, r: Int32, d: Int32, use_fin: Int32):
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if t < Int(nc)*Int(r):
        if use_fin != 0 or active.unsafe_load(t//Int(r)) != 0:
            var acc = Float32(0)
            for j in range(Int(d)):
                acc = add(acc, mul(product.unsafe_load(t*Int(d)+j), centered.unsafe_load(t*Int(d)+j)))
            if acc != acc:
                err.unsafe_store(0, Int32(1))
            dst.unsafe_store(t, acc)
