# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`x_linear_class_prep` on the device (lane cpu2-l10-linear, 2026-10-04).

The four steps of x_linear/class_prep.mojo as four grid launches, one
witness-guarded unit (x_linear/witness.mojo): every output is rebuilt from
inputs the launches do not write, so a cut Apple launch reruns.

  parts    one thread per (row block, class): nb * k threads
  counts   one thread per class (and the largest count, thread 0 after them)
  weights  one thread per class ('balanced' only)
  rows     one thread per row (when the caller asks for the row weights)

Only the k class weights (when 'balanced'), the n row weights (when asked)
and the largest class count come back to the host.
"""

from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from x_linear.ops import FP, IP, ld, st, ldi, sti
from x_linear.tops import fold_blocks
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.class_prep import (
    cp_part_count, cp_part_wsum, cp_class_count, cp_class_wsum, cp_balanced_weight, cp_largest, cp_row_weight,
)

comptime CP_TPB = 256


def _cp_blocks(count: Int) -> Int:
    return max((count + CP_TPB - 1) // CP_TPB, 1)


def cp_parts_kernel(codes: IP, w: FP, n: Int32, k: Int32, nb: Int32, weighted: Int32, icp: IP, wcp: FP,
                    wf: IP, woff: Int32, nonce: Int32):
    var t = Int(block_idx.x) * CP_TPB + Int(thread_idx.x)
    var kk = Int(k)
    if t < Int(nb) * kk:
        var b = t // kk
        var c = t % kk
        sti(icp, t, cp_part_count(codes, Int(n), b, c))
        if weighted != 0:
            st(wcp, t, cp_part_wsum(codes, w, Int(n), b, c))
    witness_end(wf, woff, nonce)


def cp_counts_kernel(icp: IP, wcp: FP, nb: Int32, k: Int32, weighted: Int32, ic: IP, wc: FP,
                     wf: IP, woff: Int32, nonce: Int32):
    var c = Int(block_idx.x) * CP_TPB + Int(thread_idx.x)
    if c < Int(k):
        sti(ic, c, cp_class_count(icp, Int(nb), Int(k), c))
        if weighted != 0:
            st(wc, c, cp_class_wsum(wcp, Int(nb), Int(k), c))
    witness_end(wf, woff, nonce)


def cp_weights_kernel(ic: IP, wc: FP, n: Int32, k: Int32, weighted: Int32, balanced: Int32, cw: FP, largest: IP,
                      wf: IP, woff: Int32, nonce: Int32):
    var c = Int(block_idx.x) * CP_TPB + Int(thread_idx.x)
    if c < Int(k):
        if balanced != 0:
            st(cw, c, cp_balanced_weight(ic, wc, weighted != 0, Int(n), Int(k), c))
        if c == 0:
            sti(largest, 0, cp_largest(ic, Int(k)))
    witness_end(wf, woff, nonce)


def cp_rows_kernel(codes: IP, sw: FP, has_sw: Int32, cw: FP, n: Int32, dst: FP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * CP_TPB + Int(thread_idx.x)
    if i < Int(n):
        st(dst, i, cp_row_weight(codes, sw, has_sw != 0, cw, i))
    witness_end(wf, woff, nonce)


def class_prep_device(var ctx: DeviceContext, codes: IP, sw: FP, has_sw: Bool, cw: FP, n: Int, k: Int,
                      balanced: Bool, weighted: Bool, rows_out: FP, has_rows: Bool) raises -> Int:
    """The GPU column of `class_prep_host` (x_linear/class_prep.mojo); the
    same arguments (host addresses), the same outputs. Returns the largest
    unweighted class count."""
    var nb = fold_blocks(n)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dsw = ctx.enqueue_create_buffer[DType.float32](max(n if has_sw else 1, 1))
    var dicp = ctx.enqueue_create_buffer[DType.int32](max(nb * k, 1))
    var dwcp = ctx.enqueue_create_buffer[DType.float32](max(nb * k, 1))
    var dic = ctx.enqueue_create_buffer[DType.int32](max(k, 1))
    var dwc = ctx.enqueue_create_buffer[DType.float32](max(k, 1))
    var dcw = ctx.enqueue_create_buffer[DType.float32](max(k, 1))
    var dlg = ctx.enqueue_create_buffer[DType.int32](1)
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n if has_rows else 1, 1))
    ctx.enqueue_copy(dst_buf=dcodes.create_sub_buffer[DType.int32](0, n), src_ptr=codes)
    if has_sw:
        ctx.enqueue_copy(dst_buf=dsw.create_sub_buffer[DType.float32](0, n), src_ptr=sw)
    if not balanced:
        # the caller's (dict) weights, read only
        ctx.enqueue_copy(dst_buf=dcw.create_sub_buffer[DType.float32](0, k), src_ptr=cw)
    var g_parts = _cp_blocks(nb * k)
    var g_class = _cp_blocks(k)
    var g_rows = _cp_blocks(n) if has_rows else 0
    var wit = Witness(ctx, g_parts + 2 * g_class + g_rows + 1)
    var wgt = Int32(1) if weighted else Int32(0)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[cp_parts_kernel](dcodes.unsafe_ptr(), dsw.unsafe_ptr(), Int32(n), Int32(k), Int32(nb), wgt,
                                              dicp.unsafe_ptr(), dwcp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                              grid_dim=g_parts, block_dim=CP_TPB)
        wo += g_parts
        ctx.enqueue_function[cp_counts_kernel](dicp.unsafe_ptr(), dwcp.unsafe_ptr(), Int32(nb), Int32(k), wgt,
                                               dic.unsafe_ptr(), dwc.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                               grid_dim=g_class, block_dim=CP_TPB)
        wo += g_class
        ctx.enqueue_function[cp_weights_kernel](dic.unsafe_ptr(), dwc.unsafe_ptr(), Int32(n), Int32(k), wgt,
                                                Int32(1) if balanced else Int32(0), dcw.unsafe_ptr(), dlg.unsafe_ptr(),
                                                wit.p(), Int32(wo), nonce, grid_dim=g_class, block_dim=CP_TPB)
        wo += g_class
        if has_rows:
            ctx.enqueue_function[cp_rows_kernel](dcodes.unsafe_ptr(), dsw.unsafe_ptr(), Int32(1) if has_sw else Int32(0),
                                                 dcw.unsafe_ptr(), Int32(n), dout.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                 grid_dim=g_rows, block_dim=CP_TPB)
            wo += g_rows
        if wit.ok(ctx, wo, "class prep"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    var hlg = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=hlg.unsafe_ptr(), src_buf=dlg)
    if balanced:
        ctx.enqueue_copy(dst_ptr=cw, src_buf=dcw.create_sub_buffer[DType.float32](0, k))
    if has_rows:
        ctx.enqueue_copy(dst_ptr=rows_out, src_buf=dout.create_sub_buffer[DType.float32](0, n))
    ctx.synchronize()
    var largest = Int(hlg[0])
    _ = hlg^
    _ = dcodes^
    _ = dsw^
    _ = dicp^
    _ = dwcp^
    _ = dic^
    _ = dwc^
    _ = dcw^
    _ = dlg^
    _ = dout^
    _ = wit^
    return largest
