# SPDX-License-Identifier: Apache-2.0
"""T39 combined apply, mean-prefix and final prediction production boundary.
Default OFF. NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from std.atomic import Atomic
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.forest_auxiliary_units import AuxI32, AuxF32, forest_auxiliary_row


def _auxiliary_kernel[RF_INPUT: Bool](offsets: AuxI32, colid: AuxI32, threshold: AuxF32, left: AuxI32, leaves: AuxF32,
    x: AuxF32, leaf_out: AuxI32, prefix_out: AuxF32, pred_out: AuxF32, totals: AuxF32, bad: AuxI32,
    rows: Int32, features: Int32, trees: Int32, outputs: Int32, emit_leaf: Bool, emit_prefix: Bool, emit_pred: Bool):
    var row = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if row < Int(rows):
        if not forest_auxiliary_row[RF_INPUT](offsets, colid, threshold, left, leaves, x, leaf_out, prefix_out, pred_out,
            totals, row, Int(features), Int(trees), Int(outputs), emit_leaf, emit_prefix, emit_pred):
            _ = Atomic.max(bad, Int32(1))


def forest_auxiliary_device(forest: List[Int], x: Int, leaf_out: Int, prefix_out: Int, pred_out: Int,
                            rows: Int, features: Int, trees: Int, outputs: Int, rf_input: Bool) raises:
    if rows == 0:
        return
    var ctx = process_ctx["MojoTreesT39Auxiliary"]()
    var offsets = AuxI32(unsafe_from_address=forest[0])
    var nodes = Int(offsets.unsafe_load(trees))
    var doff = ctx.enqueue_create_buffer[DType.int32](trees+1)
    var dcol = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dthr = ctx.enqueue_create_buffer[DType.float32](nodes)
    var dleft = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dleaf = ctx.enqueue_create_buffer[DType.float32](nodes*outputs if prefix_out!=0 or pred_out!=0 else 1)
    var dx = ctx.enqueue_create_buffer[DType.float32](rows*features)
    var did = ctx.enqueue_create_buffer[DType.int32](rows*trees if leaf_out!=0 else 1)
    var dpre = ctx.enqueue_create_buffer[DType.float32](rows*trees*outputs if prefix_out!=0 else 1)
    var dpred = ctx.enqueue_create_buffer[DType.float32](rows*outputs if pred_out!=0 else 1)
    var sums = ctx.enqueue_create_buffer[DType.float32](rows*outputs if prefix_out!=0 or pred_out!=0 else 1)
    var bad = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(bad, Int32(0))
    ctx.enqueue_copy(dst_buf=doff, src_ptr=offsets)
    ctx.enqueue_copy(dst_buf=dcol, src_ptr=AuxI32(unsafe_from_address=forest[1]))
    ctx.enqueue_copy(dst_buf=dthr, src_ptr=AuxF32(unsafe_from_address=forest[2]))
    ctx.enqueue_copy(dst_buf=dleft, src_ptr=AuxI32(unsafe_from_address=forest[3]))
    if prefix_out!=0 or pred_out!=0:
        ctx.enqueue_copy(dst_buf=dleaf, src_ptr=AuxF32(unsafe_from_address=forest[4]))
    ctx.enqueue_copy(dst_buf=dx, src_ptr=AuxF32(unsafe_from_address=x))
    if rf_input:
        ctx.enqueue_function[_auxiliary_kernel[True]](doff.unsafe_ptr(), dcol.unsafe_ptr(), dthr.unsafe_ptr(), dleft.unsafe_ptr(), dleaf.unsafe_ptr(),
            dx.unsafe_ptr(), did.unsafe_ptr(), dpre.unsafe_ptr(), dpred.unsafe_ptr(), sums.unsafe_ptr(), bad.unsafe_ptr(),
            Int32(rows), Int32(features), Int32(trees), Int32(outputs), leaf_out!=0, prefix_out!=0, pred_out!=0,
            grid_dim=(rows+127)//128, block_dim=128)
    else:
        ctx.enqueue_function[_auxiliary_kernel[False]](doff.unsafe_ptr(), dcol.unsafe_ptr(), dthr.unsafe_ptr(), dleft.unsafe_ptr(), dleaf.unsafe_ptr(),
            dx.unsafe_ptr(), did.unsafe_ptr(), dpre.unsafe_ptr(), dpred.unsafe_ptr(), sums.unsafe_ptr(), bad.unsafe_ptr(),
            Int32(rows), Int32(features), Int32(trees), Int32(outputs), leaf_out!=0, prefix_out!=0, pred_out!=0,
            grid_dim=(rows+127)//128, block_dim=128)
    var hbad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=hbad.unsafe_ptr(), src_buf=bad)
    ctx.synchronize()
    if hbad.unsafe_ptr().unsafe_load(0) != 0:
        raise Error("forest auxiliary outputs require valid finite trees and input")
    if leaf_out!=0:
        ctx.enqueue_copy(dst_ptr=AuxI32(unsafe_from_address=leaf_out), src_buf=did)
    if prefix_out!=0:
        ctx.enqueue_copy(dst_ptr=AuxF32(unsafe_from_address=prefix_out), src_buf=dpre)
    if pred_out!=0:
        ctx.enqueue_copy(dst_ptr=AuxF32(unsafe_from_address=pred_out), src_buf=dpred)
    ctx.synchronize()
