# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host x_cnn image's operations as typed native runs, and their one
native entry (lane calltable-xcnn, 2026-11-11). HOST ONLY.

Each run is the body the Python entry `x_cnn_<name>` of
`bindings/_mojolearn_x_cnn_host.mojo` ran (and, for the graph builders, of
the former `providers/python/x_cnn_host_ops_host_pyapi.mojo`), moved word for
word: the same checks, refusal texts, host bodies and CPU-column output seams
(`out_f32`). The Python entries now only unpack into plain words and call
`x_cnn_host_run`; the CNN layer doors reach the same runs through
`x_cnn_host_entry_v1`, published in `core/xcnn_entry_table.mojo` (vendor
word 0) at module init. No kernel, fold order or route changed.

The host image has no `_m` operation and no CSR_UPLOAD: they refuse with the
text a missing Python entry raised. RES_ALLOC_NOFILL is the zero-filled
allocation here (the host image never had an unfilled one; the doors ask
for it only on a mixed binding).
"""
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32, copy_f32
from std.memory import alloc, unsafe_memcpy, unsafe_memset_zero
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from core.shared_slot import SharedSlot
from core.xcnn_entry_table import (
    XcnnEntryKey, xcnn_table_publish, xcnn_op_name, XC_ENTRY_ABI, XC_OP_COUNT,
    XC_RES_ALLOC, XC_RES_ALLOC_NOFILL, XC_RES_FREE, XC_RES_UPLOAD, XC_RES_DOWNLOAD, XC_GEMM, XC_CONV2D_FORWARD,
    XC_CONV2D_BACKWARD, XC_MAXPOOL2D_FORWARD, XC_MAXPOOL2D_BACKWARD, XC_AVGPOOL2D_FORWARD, XC_AVGPOOL2D_BACKWARD,
    XC_RELU_FORWARD, XC_RELU_BACKWARD, XC_ADD, XC_MUL, XC_LINEAR_FORWARD, XC_LINEAR_BACKWARD, XC_BATCHNORM_FORWARD,
    XC_BATCHNORM_BACKWARD, XC_DROPOUT2D, XC_SPMM, XC_GCN_NORM, XC_PAD2D_FORWARD, XC_PAD2D_BACKWARD,
    XC_ADAPTIVE_POOL, XC_GRAPH_OP, XC_CSR_BUILD, XC_PERM_GATHER, XC_VALS_T, XC_GCN_LOOPS, XC_GCN_BACKWARD_T,
    XC_SAGE_BACKWARD_T,
)
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, conv_params
from x_cnn.ops import pool_params
from x_cnn.ops import FP, IP
from x_cnn.ops import relu_fwd_at, relu_bwd_at, add_at, mul_at
from x_cnn.host.gemm_host import gemm_host_into
from x_cnn.host.ops_host import conv2d_forward_into, conv2d_backward_into
from x_cnn.host.ops_host import linear_forward_into, linear_backward_into
from x_cnn.host.ops_host import maxpool2d_forward_into, maxpool2d_backward_into
from x_cnn.host.ops_host import avgpool2d_forward_into, avgpool2d_backward_into
from x_cnn.host.ops_host import map2_into, batchnorm_forward_into, batchnorm_backward_into, dropout2d_into
from x_cnn.host.ops_host import graph_op_host as graph_op_impl
from x_cnn.host.ops_host import adaptive_pool_host as adaptive_pool_impl
from x_cnn.host.ops_host import pad2d_forward_host as pad2d_forward_impl
from x_cnn.host.ops_host import pad2d_backward_host as pad2d_backward_impl
from x_cnn.host.ops_host import spmm_host as spmm_impl
from x_cnn.host.ops_host import gcn_norm_host as gcn_norm_impl
from x_cnn.host.ops_host import csr_build_host, gcn_loops_host, perm_gather_host, vals_t_host
from x_cnn.host.binding_entries_impl import out_f32, pool_counts as _pool_counts, csr_checked
from autograd.rules_graph_host import gcn_backward_host_autograd, sage_backward_host_autograd


def fpw(a: Int) raises -> FP:
    """The caller's float32 array at `a`, read and written in place
    (DEVIATION 5719: the address-in/address-out entries)."""
    if a == 0:
        raise Error("x_cnn: null float32 buffer address")
    return FP(unsafe_from_address=a)


def ipw(a: Int) raises -> IP:
    if a == 0:
        raise Error("x_cnn: null int32 buffer address")
    return IP(unsafe_from_address=a)


def _need(p: List[Int], k: Int, what: String) raises:
    if len(p) < k:
        raise Error("x_cnn " + what + ": " + String(k) + " parameter words required")


def _addrs(a: List[Int], k: Int) raises:
    if len(a) != k:
        raise Error("x_cnn: this entry takes " + String(k) + " addresses")


def _count(p: List[Int]) raises -> Int:
    _need(p, 1, "count")
    var n = p[0]
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n


def _lin(p: List[Int]) raises -> Tuple[Int, Int, Int]:
    _need(p, 3, "linear")
    var n = p[0]
    var d_in = p[1]
    var d_out = p[2]
    if n <= 0 or d_in <= 0 or d_out <= 0:
        raise Error("x_cnn linear: positive rows, in and out features required")
    return (n, d_in, d_out)


def _bn_prm(p: List[Int]) raises -> List[Int32]:
    _need(p, 4, "batchnorm")
    var N = p[0]
    var C = p[1]
    var HW = p[2]
    if N <= 0 or C <= 0 or HW <= 0:
        raise Error("x_cnn batchnorm: positive N, C and H*W required")
    var out: List[Int32] = [Int32(N), Int32(C), Int32(HW)]
    return out^


def _pad_prm(p: List[Int]) raises -> List[Int32]:
    _need(p, 9, "pad")
    var out = List[Int32]()
    for k in range(9):  # small-loop(the nine pad parameter words)
        var v = p[k]
        if v < 0:
            raise Error("x_cnn pad: negative parameter")
        out.append(Int32(v))
    var N = Int(out[0]); var C = Int(out[1]); var H = Int(out[2]); var W = Int(out[3])
    if N <= 0 or C <= 0 or H <= 0 or W <= 0 or Int(out[8]) > 3:
        raise Error("x_cnn pad: positive N, C, H, W and a mode in 0..3 required")
    if Int(out[8]) == 1 and (Int(out[4]) >= H or Int(out[5]) >= H or Int(out[6]) >= W or Int(out[7]) >= W):
        raise Error("x_cnn pad: reflect padding must be smaller than the input dimension")
    if Int(out[8]) == 3 and (Int(out[4]) > H or Int(out[5]) > H or Int(out[6]) > W or Int(out[7]) > W):
        raise Error("x_cnn pad: circular padding can wrap around at most once")
    return out^


# ------------------------------------------------------------ resident arrays
# DEVIATION 5718: on the CPU a resident array is a host allocation and its
# handle is its address.

def res_alloc_run(p: List[Int]) raises -> Int:
    _need(p, 1, "res_alloc")
    var nn = p[0]
    var ptr = alloc[Float32](nn if nn > 0 else 1)
    unsafe_memset_zero(ptr, nn if nn > 0 else 1)
    return Int(ptr)


def res_free_run(a: List[Int]) raises -> Int:
    _addrs(a, 1)
    f32_ptr(a[0]).free()
    return 0


def res_upload_run(a: List[Int], p: List[Int]) raises -> Int:
    """n 4-byte words (float32 or int32) from a host array into the resident array."""
    _addrs(a, 2)
    _need(p, 1, "res_upload")
    var nn = p[0]
    if nn > 0:
        unsafe_memcpy(dest=f32_ptr(a[0]), src=f32_ptr(a[1]), count=nn)
    return nn


def res_download_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    _need(p, 1, "res_download")
    var nn = p[0]
    if nn > 0:
        unsafe_memcpy(dest=f32_ptr(a[1]), src=f32_ptr(a[0]), count=nn)
    return nn


# ------------------------------------------------------------ host-address entries

def gemm_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    _need(p, 4, "gemm")
    var m = p[0]; var n = p[1]; var k = p[2]; var op = p[3]
    if m <= 0 or n <= 0 or k <= 0 or op < 0 or op > 2:
        raise Error("x_cnn gemm: positive m, n, k and op in {0, 1, 2} required")
    var pc = fpw(a[2])
    gemm_host_into(fpw(a[0]), fpw(a[1]), pc, op, m, n, k)
    out_f32(pc, m * n)
    return m * n


def conv2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 4)
    var prm = conv_params(p)
    var total = Int(prm[CP_N]) * Int(prm[CP_OC]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var po = fpw(a[3])
    conv2d_forward_into(fpw(a[0]), fpw(a[1]), fpw(a[2]), po, prm)
    out_f32(po, total)
    return total


def conv2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 6)
    var prm = conv_params(p)
    var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var nx = Int(prm[CP_N]) * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var pdx = fpw(a[3])
    var pdw = fpw(a[4])
    var pdb = fpw(a[5])
    conv2d_backward_into(fpw(a[0]), fpw(a[1]), fpw(a[2]), pdx, pdw, pdb, prm, True)
    out_f32(pdx, nx)
    out_f32(pdw, OC * ckk)
    out_f32(pdb, OC)
    return nx


def maxpool2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var prm = pool_params(p)
    var c = _pool_counts(prm)
    var po = fpw(a[1])
    maxpool2d_forward_into(fpw(a[0]), po, ipw(a[2]), prm)
    out_f32(po, c[1])
    return c[1]


def maxpool2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var prm = pool_params(p)
    var c = _pool_counts(prm)
    var pd = fpw(a[2])
    maxpool2d_backward_into(fpw(a[0]), ipw(a[1]), pd, prm)
    out_f32(pd, c[0])
    return c[0]


def avgpool2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = pool_params(p)
    var c = _pool_counts(prm)
    var po = fpw(a[1])
    avgpool2d_forward_into(fpw(a[0]), po, prm)
    out_f32(po, c[1])
    return c[1]


def avgpool2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = pool_params(p)
    var c = _pool_counts(prm)
    var pd = fpw(a[1])
    avgpool2d_backward_into(fpw(a[0]), pd, prm)
    out_f32(pd, c[0])
    return c[0]


def relu_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var n = _count(p)
    var px = fpw(a[0])
    var po = fpw(a[1])
    map2_into[relu_fwd_at](px, px, po, n)
    out_f32(po, n)
    return n


def relu_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var n = _count(p)
    var po = fpw(a[2])
    map2_into[relu_bwd_at](fpw(a[0]), fpw(a[1]), po, n)
    out_f32(po, n)
    return n


def add_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var n = _count(p)
    var po = fpw(a[2])
    map2_into[add_at](fpw(a[0]), fpw(a[1]), po, n)
    out_f32(po, n)
    return n


def mul_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var n = _count(p)
    var po = fpw(a[2])
    map2_into[mul_at](fpw(a[0]), fpw(a[1]), po, n)
    out_f32(po, n)
    return n


def linear_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 4)
    var s = _lin(p)
    var po = fpw(a[3])
    linear_forward_into(fpw(a[0]), fpw(a[1]), fpw(a[2]), po, s[0], s[1], s[2])
    out_f32(po, s[0] * s[2])
    return s[0] * s[2]


def linear_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 6)
    var s = _lin(p)
    var n = s[0]
    var d_in = s[1]
    var d_out = s[2]
    var pdx = fpw(a[3])
    var pdw = fpw(a[4])
    var pdb = fpw(a[5])
    linear_backward_into(fpw(a[0]), fpw(a[1]), fpw(a[2]), pdx, pdw, pdb, n, d_in, d_out)
    out_f32(pdx, n * d_in)
    out_f32(pdw, d_out * d_in)
    out_f32(pdb, d_out)
    return n


def batchnorm_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [x, y, running, aux]; params = [N, C, HW, training]; running (2C)
    and aux (2 + 7C) update in place."""
    _addrs(a, 4)
    var prm = _bn_prm(p)
    var training = p[3] != 0
    var N = Int(prm[0]); var C = Int(prm[1]); var HW = Int(prm[2])
    if training and N * HW < 2:
        raise Error("x_cnn batchnorm: expected more than 1 value per channel when training")
    var total = N * C * HW
    var pyo = fpw(a[1])
    var pr = fpw(a[2])
    var pa = fpw(a[3])
    batchnorm_forward_into(fpw(a[0]), pr, pa, pyo, prm, training)
    out_f32(pyo, total)
    out_f32(pr, 2 * C)
    out_f32(pa, 2 + 7 * C)
    return total


def batchnorm_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [x, g, dx, aux]; params = [N, C, HW, training]; aux updates in place."""
    _addrs(a, 4)
    var prm = _bn_prm(p)
    var training = p[3] != 0
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var pd = fpw(a[2])
    var pa = fpw(a[3])
    batchnorm_backward_into(fpw(a[0]), fpw(a[1]), pa, pd, prm, training)
    out_f32(pd, total)
    out_f32(pa, 2 + 7 * C)
    return total


def dropout2d_run(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """a = [x, y, mask]; params = [N, C, HW, seed_lo, seed_hi, thresh_hi,
    thresh_lo]; floats = [p]."""
    _addrs(a, 3)
    _need(p, 7, "dropout2d")
    if p[0] <= 0 or p[1] <= 0 or p[2] <= 0:
        raise Error("x_cnn dropout2d: positive N, C and H*W required")
    var prm = List[Int32]()
    for k in range(7):  # small-loop(the seven dropout parameter words)
        var v = p[k]
        if v < 0:
            raise Error("x_cnn dropout2d: negative parameter")
        prm.append(Int32(Int64(v) - Int64(4294967296)) if v > 2147483647 else Int32(v))
    if len(f) < 1:
        raise Error("x_cnn dropout2d: the drop probability is required")
    var h: List[Float32] = [Float32(f[0])]
    var total = p[0] * p[1] * p[2]
    var pyo = fpw(a[1])
    var pm = fpw(a[2])
    dropout2d_into(fpw(a[0]), pyo, pm, prm, h, total)
    out_f32(pyo, total)
    out_f32(pm, total)
    return total


def spmm_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [vals, h, out, csr]: out (n x F) = the CSR fold of h (n x F); see
    x_cnn/ops.mojo spmm_at."""
    _addrs(a, 4)
    _need(p, 4, "spmm")
    var t = csr_checked(a[3], p[0], p[1], p[2], p[3])
    var n = Int(t[1][0])
    var F = Int(t[1][1])
    var nnz = Int(t[1][2])
    var vals = read_f32(a[0], nnz if nnz > 0 else 1) if nnz > 0 else List[Float32](length=1, fill=Float32(0))
    var hv = read_f32(a[1], n * F)
    var po = f32_ptr(a[2])
    var y = spmm_impl(vals, hv, t[0], t[1])
    copy_f32(y.unsafe_ptr(), po, n * F)
    return n * F


def gcn_norm_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [w, vals, csr]: vals[e] = PyG gcn_norm of the CSR."""
    _addrs(a, 3)
    _need(p, 4, "gcn_norm")
    var t = csr_checked(a[2], p[0], p[1], p[2], p[3])
    var nnz = Int(t[1][2])
    if nnz <= 0:
        raise Error("x_cnn gcn_norm: at least one edge (the self loops) is required")
    var w = read_f32(a[0], nnz)
    var pv = f32_ptr(a[1])
    var y = gcn_norm_impl(w, t[0], t[1])
    copy_f32(y.unsafe_ptr(), pv, nnz)
    return nnz


def pad2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = _pad_prm(p)
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var no = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var x = read_f32(a[0], nx)
    var po = f32_ptr(a[1])
    var y = pad2d_forward_impl(x, prm)
    copy_f32(y.unsafe_ptr(), po, no)
    return no


def pad2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = _pad_prm(p)
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var ng = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var g = read_f32(a[0], ng)
    var po = f32_ptr(a[1])
    var y = pad2d_backward_impl(g, prm)
    copy_f32(y.unsafe_ptr(), po, nx)
    return nx


def adaptive_pool_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [in, out, idx]; params = [N, C, H, W, OH, OW, kind] (kinds as the
    GPU image's)."""
    _addrs(a, 3)
    _need(p, 7, "adaptive pool")
    var prm = List[Int32]()
    for k in range(6):  # small-loop(the six adaptive pool shape words)
        var v = p[k]
        if v <= 0:
            raise Error("x_cnn adaptive pool: positive N, C, H, W, OH, OW required")
        prm.append(Int32(v))
    var kind = p[6]
    if kind < 0 or kind > 3:
        raise Error("x_cnn adaptive pool: kind in 0..3")
    var nc = Int(prm[0]) * Int(prm[1])
    var nin = nc * Int(prm[2]) * Int(prm[3])
    var nout = nc * Int(prm[4]) * Int(prm[5])
    var n_read = nin if kind == 0 or kind == 2 else nout
    var n_write = nout if kind == 0 or kind == 2 else nin
    var src = read_f32(a[0], n_read)
    var idx = read_i32(a[2], nout) if kind == 3 else List[Int32](length=nout, fill=Int32(0))
    var po = f32_ptr(a[1])
    var pi = i32_ptr(a[2])
    var idx_out = List[Int32]()
    var y = adaptive_pool_impl(src, idx, prm, kind, idx_out)
    copy_f32(y.unsafe_ptr(), po, n_write)
    if kind == 2:
        unsafe_memcpy(dest=pi, src=idx_out.unsafe_ptr(), count=nout)
        _ = len(idx_out)
    return n_write


def graph_op_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [a, b, aux, out, csr]; params = [n, F, nnz, kind] (kinds as the
    GPU image's)."""
    _addrs(a, 5)
    _need(p, 4, "graph op")
    var kind = p[3]
    if kind < 0 or kind > 3:
        raise Error("x_cnn graph op: kind in 0..3")
    var n = p[0]
    var F = p[1]
    var nnz = p[2]
    var t = csr_checked(a[4], n, F, nnz, 0)
    var naux = 2 * n * F if kind < 2 else n
    var va = read_f32(a[0], n * F)
    var vb = read_f32(a[1], n * F) if kind == 1 or kind == 3 else va.copy()
    var aux = read_f32(a[2], naux)
    var po = f32_ptr(a[3])
    var pa = f32_ptr(a[2])
    var r = graph_op_impl(va, vb, aux, t[0], t[1], kind)
    copy_f32(r.unsafe_ptr(), po, n * F)
    copy_f32(r.unsafe_ptr() + n * F, pa, naux)
    return n * F


def csr_build_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [rows, cols, csr, order]; params = [n, nnz]. Returns n + 1 + 2 * nnz."""
    _addrs(a, 4)
    _need(p, 2, "csr_build")
    var n = p[0]
    var nnz = p[1]
    if n <= 0 or nnz < 0:
        raise Error("x_cnn csr_build: positive n and nnz >= 0 required")
    var pc = i32_ptr(a[2]).unsafe_origin_cast[MutAnyOrigin]()
    var rows = i32_ptr(a[0]).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else pc
    var cols = i32_ptr(a[1]).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else pc
    var order = i32_ptr(a[3]).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else pc
    csr_build_host(rows, cols, nnz, n, pc, order)
    return n + 1 + 2 * nnz


def gcn_loops_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [src, dst, w, src_out, dst_out, w_out]; params = [n, nnz,
    improved]. Returns K."""
    _need(p, 3, "gcn_loops")
    var n = p[0]
    var nnz = p[1]
    var fill = Float32(2.0) if p[2] != 0 else Float32(1.0)
    if n <= 0 or nnz < 0 or len(a) != 6:
        raise Error("x_cnn gcn_loops: addrs [src, dst, w, src_out, dst_out, w_out], positive n and nnz >= 0 required")
    var so = i32_ptr(a[3]).unsafe_origin_cast[MutAnyOrigin]()
    var dso = i32_ptr(a[4]).unsafe_origin_cast[MutAnyOrigin]()
    var wo = f32_ptr(a[5]).unsafe_origin_cast[MutAnyOrigin]()
    var src = i32_ptr(a[0]).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else so
    var dst = i32_ptr(a[1]).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else so
    var w = f32_ptr(a[2]).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else wo
    return gcn_loops_host(src, dst, w, nnz, n, fill, so, dso, wo)


def perm_gather_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [src f32, idx i32, out f32], params = [n_src, n]; out[i] = src[idx[i]]."""
    _need(p, 2, "perm_gather")
    var n_src = p[0]
    var n = p[1]
    if n < 0 or n_src < 0:
        raise Error("x_cnn perm_gather: n >= 0 required")
    if n == 0:
        return 0
    _addrs(a, 3)
    perm_gather_host(f32_ptr(a[0]).unsafe_origin_cast[MutAnyOrigin](), i32_ptr(a[1]).unsafe_origin_cast[MutAnyOrigin](),
                     f32_ptr(a[2]).unsafe_origin_cast[MutAnyOrigin](), n)
    return n


def vals_t_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [vals_f f32, order_f i32, order_t i32, out f32], params = [nnz]."""
    _need(p, 1, "vals_t")
    var nnz = p[0]
    if nnz < 0:
        raise Error("x_cnn vals_t: nnz >= 0 required")
    if nnz == 0:
        return 0
    _addrs(a, 4)
    vals_t_host(f32_ptr(a[0]).unsafe_origin_cast[MutAnyOrigin](), i32_ptr(a[1]).unsafe_origin_cast[MutAnyOrigin](),
                i32_ptr(a[2]).unsafe_origin_cast[MutAnyOrigin](), f32_ptr(a[3]).unsafe_origin_cast[MutAnyOrigin](), nnz)
    return nnz


def gcn_backward_t_run(a: List[Int], p: List[Int]) raises -> Int:
    """GCNConv's backward on the tape (autograd/rules_graph_host.mojo, the
    host form): 13 host addresses; params = [n, d, f, nnz, bias, 0, 0, 0]."""
    _addrs(a, 13)
    _need(p, 2, "gcn backward")
    gcn_backward_host_autograd(a, p)
    return p[0] * p[1]


def sage_backward_t_run(a: List[Int], p: List[Int]) raises -> Int:
    """SAGEConv's backward on the tape (the host form): 33 host addresses,
    params = [n, d, f, nnz, aggr, normalize, root, 0, 0, project]."""
    _need(p, 2, "sage backward")
    sage_backward_host_autograd(a, p)
    return p[0] * p[1]


# ------------------------------------------------------------ dispatch and entry

def x_cnn_host_run(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """One operation of this image (the Python entries and `x_cnn_host_entry_v1`)."""
    if op == XC_RES_ALLOC or op == XC_RES_ALLOC_NOFILL:
        return res_alloc_run(p)
    if op == XC_RES_FREE:
        return res_free_run(a)
    if op == XC_RES_UPLOAD:
        return res_upload_run(a, p)
    if op == XC_RES_DOWNLOAD:
        return res_download_run(a, p)
    if op == XC_GEMM:
        return gemm_run(a, p)
    if op == XC_CONV2D_FORWARD:
        return conv2d_forward_run(a, p)
    if op == XC_CONV2D_BACKWARD:
        return conv2d_backward_run(a, p)
    if op == XC_MAXPOOL2D_FORWARD:
        return maxpool2d_forward_run(a, p)
    if op == XC_MAXPOOL2D_BACKWARD:
        return maxpool2d_backward_run(a, p)
    if op == XC_AVGPOOL2D_FORWARD:
        return avgpool2d_forward_run(a, p)
    if op == XC_AVGPOOL2D_BACKWARD:
        return avgpool2d_backward_run(a, p)
    if op == XC_RELU_FORWARD:
        return relu_forward_run(a, p)
    if op == XC_RELU_BACKWARD:
        return relu_backward_run(a, p)
    if op == XC_ADD:
        return add_run(a, p)
    if op == XC_MUL:
        return mul_run(a, p)
    if op == XC_LINEAR_FORWARD:
        return linear_forward_run(a, p)
    if op == XC_LINEAR_BACKWARD:
        return linear_backward_run(a, p)
    if op == XC_BATCHNORM_FORWARD:
        return batchnorm_forward_run(a, p)
    if op == XC_BATCHNORM_BACKWARD:
        return batchnorm_backward_run(a, p)
    if op == XC_DROPOUT2D:
        return dropout2d_run(a, p, f)
    if op == XC_SPMM:
        return spmm_run(a, p)
    if op == XC_GCN_NORM:
        return gcn_norm_run(a, p)
    if op == XC_PAD2D_FORWARD:
        return pad2d_forward_run(a, p)
    if op == XC_PAD2D_BACKWARD:
        return pad2d_backward_run(a, p)
    if op == XC_ADAPTIVE_POOL:
        return adaptive_pool_run(a, p)
    if op == XC_GRAPH_OP:
        return graph_op_run(a, p)
    if op == XC_CSR_BUILD:
        return csr_build_run(a, p)
    if op == XC_PERM_GATHER:
        return perm_gather_run(a, p)
    if op == XC_VALS_T:
        return vals_t_run(a, p)
    if op == XC_GCN_LOOPS:
        return gcn_loops_run(a, p)
    if op == XC_GCN_BACKWARD_T:
        return gcn_backward_t_run(a, p)
    if op == XC_SAGE_BACKWARD_T:
        return sage_backward_t_run(a, p)
    raise Error("AttributeError: the CNN binding has no " + xcnn_op_name(op))


def _float_words(address: Int, count: Int) raises -> List[Float64]:
    if count < 0 or (count > 0 and address == 0):
        raise Error("native descriptor: null or invalid scalar span")
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Float64](capacity=count)
    for i in range(count):  # small-loop(validated native launch scalar words)
        out.append(source[i])
    return out^


def x_cnn_host_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The host image's x_cnn entry (`XC_ENTRY_ABI`, vendor word 0)."""
    try:
        if operation < 0 or operation >= XC_OP_COUNT:
            raise Error("x_cnn entry: unknown operation " + String(operation))
        if not result:
            raise Error("x_cnn entry: invalid request spans")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = _float_words(ff, nf)
        var value = x_cnn_host_run(operation, a, p, f)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XCnnHostEntryOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the host x_cnn table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _HOST_OWNER_NAME = "MojolearnXCnnHostEntryOwnerV1"
comptime XCNN_HOST_ENTRY_OWNER = SharedSlot[XCnnHostEntryOwner, _HOST_OWNER_NAME]


def publish_x_cnn_host_entries() raises:
    """Publish the host image's entry at module init (vendor word 0)."""
    var owner = XCNN_HOST_ENTRY_OWNER.get_or_create_ptr()
    var key = XcnnEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    xcnn_table_publish(key, Int(owner), [String(XC_ENTRY_ABI)], [XC_OP_COUNT], [x_cnn_host_entry_v1])
