# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU x_cnn image's operations as typed native runs, and their one
native entry (lane calltable-xcnn, 2026-11-11).

Each run is the body the Python entry `x_cnn_<name>` of
`bindings/_mojolearn_x_cnn.mojo` ran, moved word for word: the same
parameter checks and refusal texts, the same device bodies, residency bits
and launch words. The Python entries now only unpack their arguments into
plain words and call `x_cnn_run`; the CNN layer doors in other extension
modules reach the same runs through `x_cnn_entry_v1`, published in
`core/xcnn_entry_table.mojo` at module init (`publish_x_cnn_entries`), with
no Python in between. No kernel, fold order or route changed.

Argument layout per operation: `core/xcnn_entry_table.mojo`. The runs are
synchronous launches over caller-owned buffers; each kernel parallelizes
inside. The dispatch is a fixed comparison chain on the operation code, no
data work.
"""
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32, copy_f32
from std.memory import unsafe_memcpy
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from core.shared_slot import SharedSlot
from core.xcnn_entry_table import (
    XcnnEntryKey, xcnn_table_publish, xcnn_entry_vendor, xcnn_op_name, XC_ENTRY_ABI, XC_OP_COUNT,
    XC_RES_ALLOC, XC_RES_ALLOC_NOFILL, XC_RES_FREE, XC_RES_UPLOAD, XC_RES_DOWNLOAD, XC_GEMM, XC_CONV2D_FORWARD,
    XC_CONV2D_BACKWARD, XC_MAXPOOL2D_FORWARD, XC_MAXPOOL2D_BACKWARD, XC_AVGPOOL2D_FORWARD, XC_AVGPOOL2D_BACKWARD,
    XC_RELU_FORWARD, XC_RELU_BACKWARD, XC_ADD, XC_MUL, XC_LINEAR_FORWARD, XC_LINEAR_BACKWARD, XC_BATCHNORM_FORWARD,
    XC_BATCHNORM_BACKWARD, XC_DROPOUT2D, XC_SPMM, XC_GCN_NORM, XC_PAD2D_FORWARD, XC_PAD2D_BACKWARD,
    XC_ADAPTIVE_POOL, XC_GRAPH_OP, XC_CSR_BUILD, XC_CSR_UPLOAD, XC_PERM_GATHER, XC_VALS_T, XC_GCN_LOOPS,
    XC_GCN_BACKWARD_T, XC_SAGE_BACKWARD_T, XC_GEMM_M, XC_CONV2D_FORWARD_M, XC_CONV2D_BACKWARD_M,
    XC_MAXPOOL2D_FORWARD_M, XC_MAXPOOL2D_BACKWARD_M, XC_AVGPOOL2D_FORWARD_M, XC_AVGPOOL2D_BACKWARD_M, XC_MAP2_M,
    XC_LINEAR_FORWARD_M, XC_BATCHNORM_FORWARD_M, XC_BATCHNORM_BACKWARD_M, XC_DROPOUT2D_M, XC_SPMM_M,
    XC_ADAPTIVE_POOL_M, XC_GRAPH_OP_M, XC_GCN_NORM_M, XC_PAD2D_M, XC_CHAN_COPY_M,
)
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_OH, CP_OW, conv_params
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, pool_params
from x_cnn.ops import FP, IP
from x_cnn.ops import relu_fwd_at, relu_bwd_at, add_at, mul_at, bias_rows_at
from x_cnn.device import (
    gemm_into, conv2d_forward_into, maxpool2d_forward_into, maxpool2d_backward_into, avgpool2d_forward_into,
    avgpool2d_backward_into, relu_forward_into, relu_backward_into, add_into, mul_into, linear_forward_into,
    batchnorm_forward_into, batchnorm_backward_into, dropout2d_into, spmm_into, pad2d_forward_into,
    pad2d_backward_into, res_alloc, res_alloc_nofill, res_free, res_upload, res_download, gemm_m,
    conv2d_forward_m, maxpool2d_forward_m, maxpool2d_backward_m, avgpool2d_forward_m, avgpool2d_backward_m, map2_m,
    linear_forward_m, batchnorm_forward_m, batchnorm_backward_m, dropout2d_m, spmm_m,
)
from x_cnn.device import graph_op_device as graph_op_impl
from x_cnn.device import adaptive_pool_device as adaptive_pool_impl
from x_cnn.device import gcn_norm_device as gcn_norm_impl
from x_cnn.device import csr_build_device as csr_build_impl
from x_cnn.device import perm_gather_device, vals_t_device, gcn_loops_device
from x_cnn.device import adaptive_pool_m, graph_op_m, gcn_norm_m, pad2d_m, chan_copy_m
from x_cnn.device_entries_impl import csr_checked_device
from autograd.kernels.cnn import conv2d_backward_into, linear_backward_into, conv2d_backward_m
from autograd.rules_graph import gcn_backward_autograd, sage_backward_autograd


# ------------------------------------------------------------ word helpers

def _fp(addr: Int) raises -> FP:
    """The caller's float32 array, by address (DEVIATION 5716: the device
    entries copy straight to and from it)."""
    if addr == 0:
        raise Error("x_cnn: null float32 buffer address")
    return f32_ptr(addr).unsafe_origin_cast[MutAnyOrigin]()


def _ip(addr: Int) raises -> IP:
    if addr == 0:
        raise Error("x_cnn: null int32 buffer address")
    return i32_ptr(addr).unsafe_origin_cast[MutAnyOrigin]()


def _need(p: List[Int], k: Int, what: String) raises:
    """At least `k` parameter words (the Python entries raised IndexError)."""
    if len(p) < k:
        raise Error("x_cnn " + what + ": " + String(k) + " parameter words required")


def _addrs(a: List[Int], k: Int) raises:
    if len(a) != k:
        raise Error("x_cnn: this entry takes " + String(k) + " addresses")


def _tail(p: List[Int], what: String) raises -> List[Int]:
    """The parameter words after a `_m` operation's residency word."""
    _need(p, 1, what)
    var out = List[Int](capacity=len(p))
    for k in range(1, len(p)):  # small-loop(the caller's int parameter words)
        out.append(p[k])
    return out^


def _count(p: List[Int]) raises -> Int:
    _need(p, 1, "count")
    var n = p[0]
    if n <= 0:
        raise Error("x_cnn: a positive element count is required")
    return n


def _pool_counts(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


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


def _dropout_prm(p: List[Int]) raises -> List[Int32]:
    _need(p, 7, "dropout2d")
    if p[0] <= 0 or p[1] <= 0 or p[2] <= 0:
        raise Error("x_cnn dropout2d: positive N, C and H*W required")
    var prm = List[Int32]()
    for k in range(7):  # small-loop(the seven dropout parameter words)
        var v = p[k]
        if v < 0:
            raise Error("x_cnn dropout2d: negative parameter")
        prm.append(Int32(Int64(v) - Int64(4294967296)) if v > 2147483647 else Int32(v))
    return prm^


def _drop_p(f: List[Float64]) raises -> List[Float32]:
    if len(f) < 1:
        raise Error("x_cnn dropout2d: the drop probability is required")
    var h: List[Float32] = [Float32(f[0])]
    return h^


def _csr(addr: Int, n: Int, F: Int, nnz: Int, mode: Int) raises -> Tuple[List[Int32], List[Int32]]:
    """Validate a CSR block [rowptr | col | row] against [n, F, nnz, mode]."""
    return csr_checked_device(addr, n, F, nnz, mode)


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


def _adaptive_prm(p: List[Int]) raises -> Tuple[List[Int32], Int]:
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
    return (prm^, kind)


# ------------------------------------------------------------ resident arrays
# DEVIATION 5718: a handle is a device address the caller keeps between calls.

def res_alloc_run(p: List[Int]) raises -> Int:
    _need(p, 1, "res_alloc")
    return res_alloc(p[0])


def res_alloc_nofill_run(p: List[Int]) raises -> Int:
    """lane neural-io-2 (XCNN_RES_NEW_NOZERO, x_cnn/device.mojo): a resident
    array for an output whose every word the op stores, unfilled."""
    _need(p, 1, "res_alloc_nofill")
    return res_alloc_nofill(p[0])


def res_free_run(a: List[Int]) raises -> Int:
    _addrs(a, 1)
    res_free(a[0])
    return 0


def res_upload_run(a: List[Int], p: List[Int]) raises -> Int:
    """n 4-byte words (float32 or int32) from a host array into the resident array."""
    _addrs(a, 2)
    _need(p, 1, "res_upload")
    res_upload(a[0], f32_ptr(a[1]).unsafe_origin_cast[MutAnyOrigin](), p[0])
    return p[0]


def res_download_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    _need(p, 1, "res_download")
    res_download(a[0], f32_ptr(a[1]).unsafe_origin_cast[MutAnyOrigin](), p[0])
    return p[0]


def csr_upload_run(a: List[Int], p: List[Int]) raises -> Int:
    """A resident copy of a CSR block [rowptr | col | row] (params [n, F,
    nnz, mode]), checked ONCE here as `x_cnn_spmm` checks it on every call;
    its handle (free with `x_cnn_res_free`)."""
    _addrs(a, 1)
    _need(p, 4, "csr_upload")
    var t = _csr(a[0], p[0], p[1], p[2], p[3])
    var csr = t[0].copy()
    var words = len(csr)
    var h = res_alloc(words)
    var src = FP(unsafe_from_address=Int(csr.unsafe_ptr()))
    res_upload(h, src, words)
    _ = csr^
    return h


# ------------------------------------------------------------ host-address entries

def gemm_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    _need(p, 4, "gemm")
    var m = p[0]; var n = p[1]; var k = p[2]; var op = p[3]
    if m <= 0 or n <= 0 or k <= 0 or op < 0 or op > 2:
        raise Error("x_cnn gemm: positive m, n, k and op in {0, 1, 2} required")
    gemm_into(_fp(a[0]), _fp(a[1]), _fp(a[2]), m, n, k, op)
    return m * n


def conv2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 4)
    var prm = conv_params(p)
    var total = Int(prm[CP_N]) * Int(prm[CP_OC]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    conv2d_forward_into(_fp(a[0]), _fp(a[1]), _fp(a[2]), prm, _fp(a[3]))
    return total


def conv2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 6)
    var prm = conv_params(p)
    var nx = Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W])
    conv2d_backward_into(_fp(a[0]), _fp(a[1]), _fp(a[2]), prm, _fp(a[3]), _fp(a[4]), _fp(a[5]))
    return nx


def maxpool2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var prm = pool_params(p)
    maxpool2d_forward_into(_fp(a[0]), prm, _fp(a[1]), _ip(a[2]))
    return _pool_counts(prm)[1]


def maxpool2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var prm = pool_params(p)
    maxpool2d_backward_into(_fp(a[0]), _ip(a[1]), prm, _fp(a[2]))
    return _pool_counts(prm)[0]


def avgpool2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = pool_params(p)
    avgpool2d_forward_into(_fp(a[0]), prm, _fp(a[1]))
    return _pool_counts(prm)[1]


def avgpool2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = pool_params(p)
    avgpool2d_backward_into(_fp(a[0]), prm, _fp(a[1]))
    return _pool_counts(prm)[0]


def relu_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var n = _count(p)
    relu_forward_into(_fp(a[0]), n, _fp(a[1]))
    return n


def relu_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var n = _count(p)
    relu_backward_into(_fp(a[0]), _fp(a[1]), n, _fp(a[2]))
    return n


def add_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var n = _count(p)
    add_into(_fp(a[0]), _fp(a[1]), n, _fp(a[2]))
    return n


def mul_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 3)
    var n = _count(p)
    mul_into(_fp(a[0]), _fp(a[1]), n, _fp(a[2]))
    return n


def linear_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 4)
    var s = _lin(p)
    linear_forward_into[False](_fp(a[0]), _fp(a[1]), _fp(a[2]), s[0], s[1], s[2], _fp(a[3]))
    return s[0] * s[2]


def linear_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 6)
    var s = _lin(p)
    linear_backward_into[False](_fp(a[0]), _fp(a[1]), _fp(a[2]), s[0], s[1], s[2], _fp(a[3]), _fp(a[4]),
                                _fp(a[5]))
    return s[0]


def batchnorm_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [x, y, running, aux]; params = [N, C, HW, training]; running (2C)
    and aux (2 + 7C) update in place."""
    _addrs(a, 4)
    var prm = _bn_prm(p)
    var training = p[3] != 0
    var N = Int(prm[0]); var C = Int(prm[1]); var HW = Int(prm[2])
    if training and N * HW < 2:
        raise Error("x_cnn batchnorm: expected more than 1 value per channel when training")
    batchnorm_forward_into(_fp(a[0]), _fp(a[2]), _fp(a[3]), prm, training, _fp(a[1]))
    return N * C * HW


def batchnorm_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [x, g, dx, aux]; params = [N, C, HW, training]; aux updates in place."""
    _addrs(a, 4)
    var prm = _bn_prm(p)
    var training = p[3] != 0
    batchnorm_backward_into(_fp(a[0]), _fp(a[1]), _fp(a[3]), prm, training, _fp(a[2]))
    return Int(prm[0]) * Int(prm[1]) * Int(prm[2])


def dropout2d_run(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """a = [x, y, mask]; params = [N, C, HW, seed_lo, seed_hi, thresh_hi,
    thresh_lo] (the seed words and the threshold halves as non-negative
    ints); floats = [p]."""
    _addrs(a, 3)
    var prm = _dropout_prm(p)
    var h = _drop_p(f)
    var total = p[0] * p[1] * p[2]
    dropout2d_into(_fp(a[0]), total, prm, h, _fp(a[1]), _fp(a[2]))
    return total


def spmm_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [vals, h, out, csr]: out (n x F) = the CSR fold of h (n x F); see
    x_cnn/ops.mojo spmm_at."""
    _addrs(a, 4)
    _need(p, 4, "spmm")
    var t = _csr(a[3], p[0], p[1], p[2], p[3])
    var n = Int(t[1][0])
    var F = Int(t[1][1])
    var nnz = Int(t[1][2])
    var vals = _fp(a[0]) if nnz > 0 else _fp(a[1])
    spmm_into(vals, nnz, _fp(a[1]), t[0], t[1], _fp(a[2]))
    return n * F


def gcn_norm_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [w, vals, csr]: vals[e] = PyG gcn_norm of the CSR (rows = targets,
    self loops included)."""
    _addrs(a, 3)
    _need(p, 4, "gcn_norm")
    var t = _csr(a[2], p[0], p[1], p[2], p[3])
    var nnz = Int(t[1][2])
    if nnz <= 0:
        raise Error("x_cnn gcn_norm: at least one edge (the self loops) is required")
    var w = read_f32(a[0], nnz)
    var pv = f32_ptr(a[1])
    var y = gcn_norm_impl(w, t[0], t[1])
    copy_f32(y.unsafe_ptr(), pv, nnz)
    return nnz


def pad2d_forward_run(a: List[Int], p: List[Int]) raises -> Int:
    """params = [N, C, H, W, top, bottom, left, right, mode (0 zeros, 1
    reflect, 2 replicate, 3 circular)]."""
    _addrs(a, 2)
    var prm = _pad_prm(p)
    var nc = Int(prm[0]) * Int(prm[1])
    var no = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    pad2d_forward_into(_fp(a[0]), prm, _fp(a[1]))
    return no


def pad2d_backward_run(a: List[Int], p: List[Int]) raises -> Int:
    _addrs(a, 2)
    var prm = _pad_prm(p)
    var nx = Int(prm[0]) * Int(prm[1]) * Int(prm[2]) * Int(prm[3])
    pad2d_backward_into(_fp(a[0]), prm, _fp(a[1]))
    return nx


def adaptive_pool_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [in, out, idx]; params = [N, C, H, W, OH, OW, kind]: kind 0 avg
    forward (in x, out y), 1 avg backward (in g, out dx), 2 max forward (in
    x, out y, idx written), 3 max backward (in g, out dx, idx read)."""
    _addrs(a, 3)
    var t = _adaptive_prm(p)
    var kind = t[1]
    var nc = Int(t[0][0]) * Int(t[0][1])
    var nin = nc * Int(t[0][2]) * Int(t[0][3])
    var nout = nc * Int(t[0][4]) * Int(t[0][5])
    var n_read = nin if kind == 0 or kind == 2 else nout
    var n_write = nout if kind == 0 or kind == 2 else nin
    var src = read_f32(a[0], n_read)
    var idx = read_i32(a[2], nout) if kind == 3 else List[Int32](length=nout, fill=Int32(0))
    var po = f32_ptr(a[1])
    var pi = i32_ptr(a[2])
    var idx_out = List[Int32]()
    var y = adaptive_pool_impl(src, idx, t[0], kind, idx_out)
    copy_f32(y.unsafe_ptr(), po, n_write)
    if kind == 2:
        unsafe_memcpy(dest=pi, src=idx_out.unsafe_ptr(), count=nout)
        _ = len(idx_out)
    return n_write


def graph_op_run(a: List[Int], p: List[Int]) raises -> Int:
    """a = [a, b, aux, out, csr]; params = [n, F, nnz, kind]: kind 0 SAGE max
    forward (a = h; aux 2nF written), 1 its backward (a = h, b = g, aux read;
    the TRANSPOSED csr), 2 row L2 normalize forward (a = x; aux n written), 3
    its backward (a = y, b = g, aux read). A kind with no graph passes nnz 0
    and a rowptr of zeros."""
    _addrs(a, 5)
    _need(p, 4, "graph op")
    var kind = p[3]
    if kind < 0 or kind > 3:
        raise Error("x_cnn graph op: kind in 0..3")
    var n = p[0]
    var F = p[1]
    var nnz = p[2]
    var t = _csr(a[4], n, F, nnz, 0)
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
    """a = [rows, cols, csr, order]: the CSR view of an edge list, on the
    device (x_cnn/device.mojo csr_build_device). params = [n, nnz]. Returns
    n + 1 + 2 * nnz."""
    _addrs(a, 4)
    _need(p, 2, "csr_build")
    var n = p[0]
    var nnz = p[1]
    if n <= 0 or nnz < 0:
        raise Error("x_cnn csr_build: positive n and nnz >= 0 required")
    var pc = _ip(a[2])
    var rows = _ip(a[0]) if nnz > 0 else pc
    var cols = _ip(a[1]) if nnz > 0 else pc
    var order = _ip(a[3]) if nnz > 0 else pc
    csr_build_impl(rows, cols, nnz, n, pc, order)
    return n + 1 + 2 * nnz


def perm_gather_run(a: List[Int], p: List[Int]) raises -> Int:
    """out[i] = src[idx[i]] on the device: a = [src f32, idx i32, out f32],
    params = [n_src, n]. Returns n."""
    _need(p, 2, "perm_gather")
    var n_src = p[0]
    var n = p[1]
    if n < 0 or n_src < 0:
        raise Error("x_cnn perm_gather: n >= 0 required")
    if n == 0:
        return 0
    _addrs(a, 3)
    perm_gather_device(_fp(a[0]), _ip(a[1]), _fp(a[2]), n_src, n)
    return n


def vals_t_run(a: List[Int], p: List[Int]) raises -> Int:
    """`_Graph.vals_t` on the device: a = [vals_f f32, order_f i32, order_t
    i32, out f32], params = [nnz]. Returns nnz."""
    _need(p, 1, "vals_t")
    var nnz = p[0]
    if nnz < 0:
        raise Error("x_cnn vals_t: nnz >= 0 required")
    if nnz == 0:
        return 0
    _addrs(a, 4)
    vals_t_device(_fp(a[0]), _ip(a[1]), _ip(a[2]), _fp(a[3]), nnz)
    return nnz


def gcn_loops_run(a: List[Int], p: List[Int]) raises -> Int:
    """GCN's remaining self loops on the device: a = [src, dst (int32 [nnz]),
    w (float32 [nnz]), src_out, dst_out (int32 [nnz + n]), w_out (float32
    [nnz + n])]; params = [n, nnz, improved]. Returns K: the outputs' first
    K + n slots are the edge list."""
    _need(p, 3, "gcn_loops")
    var n = p[0]
    var nnz = p[1]
    var fill = Float32(2.0) if p[2] != 0 else Float32(1.0)
    if n <= 0 or nnz < 0 or len(a) != 6:
        raise Error("x_cnn gcn_loops: addrs [src, dst, w, src_out, dst_out, w_out], positive n and nnz >= 0 required")
    var so = _ip(a[3])
    var dso = _ip(a[4])
    var wo = _fp(a[5])
    var src = _ip(a[0]) if nnz > 0 else so
    var dst = _ip(a[1]) if nnz > 0 else so
    var w = _fp(a[2]) if nnz > 0 else wo
    return gcn_loops_device(src, dst, w, nnz, n, fill, so, dso, wo)


def gcn_backward_t_run(a: List[Int], p: List[Int]) raises -> Int:
    """GCNConv's backward on the tape in one call (autograd/rules_graph.mojo
    `gcn_backward_autograd`): 13 addresses; params = [n, d, f, nnz, bias, dx
    resident, ones resident(, resident)]."""
    _addrs(a, 13)
    _need(p, 2, "gcn backward")
    gcn_backward_autograd(a, p)
    return p[0] * p[1]


def sage_backward_t_run(a: List[Int], p: List[Int]) raises -> Int:
    """SAGEConv's backward on the tape in one call (autograd/rules_graph.mojo
    `sage_backward_autograd`): 23 or 33 addresses (autograd/graph_tape.mojo
    `sage_cfg`)."""
    _need(p, 2, "sage backward")
    sage_backward_autograd(a, p)
    return p[0] * p[1]


# ------------------------------------------------------------ mixed residency
# lane gap-neural-overhead2 (2026-10-02): bit k of `dev` says a[k] is a
# resident device address (`x_cnn_res_alloc`'s) instead of a host one.

def gemm_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [A, B, C]; params = [m, n, k, op]."""
    _need(p, 4, "gemm")
    var m = p[0]; var n = p[1]; var k = p[2]; var op = p[3]
    if m <= 0 or n <= 0 or k <= 0 or op < 0 or op > 2:
        raise Error("x_cnn gemm: positive m, n, k and op in {0, 1, 2} required")
    _addrs(a, 3)
    gemm_m(a, d, m, n, k, op)
    return m * n


def conv2d_forward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [x, w, bias, out]."""
    var prm = conv_params(p)
    _addrs(a, 4)
    conv2d_forward_m(a, d, prm)
    return Int(prm[CP_N]) * Int(prm[CP_OC]) * Int(prm[CP_OH]) * Int(prm[CP_OW])


def conv2d_backward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [x, w, dout, dx, dW, db]."""
    var prm = conv_params(p)
    _addrs(a, 6)
    conv2d_backward_m(a, d, prm)
    return Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W])


def maxpool2d_forward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [x, out, idx (int32)]."""
    var prm = pool_params(p)
    _addrs(a, 3)
    maxpool2d_forward_m(a, d, prm)
    return _pool_counts(prm)[1]


def maxpool2d_backward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [dout, idx (int32), dx]."""
    var prm = pool_params(p)
    _addrs(a, 3)
    maxpool2d_backward_m(a, d, prm)
    return _pool_counts(prm)[0]


def avgpool2d_forward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [x, out]."""
    var prm = pool_params(p)
    _addrs(a, 2)
    avgpool2d_forward_m(a, d, prm)
    return _pool_counts(prm)[1]


def avgpool2d_backward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [dout, dx]."""
    var prm = pool_params(p)
    _addrs(a, 2)
    avgpool2d_backward_m(a, d, prm)
    return _pool_counts(prm)[0]


def map2_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [a, b, out]; params = [kind, n] or [4, n, cols]: kind 0 ReLU
    forward (a = x; b unused), 1 ReLU backward (a = x, b = g), 2 add, 3
    multiply, 4 the row bias add (a = y (n = rows * cols), b = bias (cols);
    `bias_rows_at`, the linear forward's)."""
    _need(p, 2, "map2")
    var kind = p[0]
    var n = p[1]
    if n <= 0 or kind < 0 or kind > 4:
        raise Error("x_cnn map2: positive n and kind in 0..4")
    _addrs(a, 3)
    if kind == 4:
        _need(p, 3, "map2")
        var cols = p[2]
        if cols <= 0 or n % cols != 0:
            raise Error("x_cnn map2: the row bias needs cols dividing n")
        var prm: List[Int32] = [Int32(n // cols), Int32(0), Int32(cols)]
        map2_m[bias_rows_at](a, d, n, cols, n, prm)
        return n
    var prm: List[Int32] = [0, 0, 0]
    if kind == 0:
        map2_m[relu_fwd_at](a, d & ~2, n, 0, n, prm)
    elif kind == 1:
        map2_m[relu_bwd_at](a, d, n, n, n, prm)
    elif kind == 2:
        map2_m[add_at](a, d, n, n, n, prm)
    else:
        map2_m[mul_at](a, d, n, n, n, prm)
    return n


def linear_forward_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [x, w, b, y]."""
    var s = _lin(p)
    _addrs(a, 4)
    linear_forward_m(a, d, s[0], s[1], s[2])
    return s[0] * s[2]


def batchnorm_forward_m_run(h: List[Int], d0: Int, p: List[Int]) raises -> Int:
    """h = [x, y, running, aux] (the host entry's order); params = [N, C, HW, training]."""
    var prm = _bn_prm(p)
    var training = p[3] != 0
    if training and Int(prm[0]) * Int(prm[2]) < 2:
        raise Error("x_cnn batchnorm: expected more than 1 value per channel when training")
    _addrs(h, 4)
    var a: List[Int] = [h[0], h[2], h[3], h[1]]
    # host order bits (x, y, running, aux) to the body's (x, running, aux, y)
    var d = (d0 & 1) | (((d0 >> 2) & 1) << 1) | (((d0 >> 3) & 1) << 2) | (((d0 >> 1) & 1) << 3)
    batchnorm_forward_m(a, d, prm, training)
    return Int(prm[0]) * Int(prm[1]) * Int(prm[2])


def batchnorm_backward_m_run(h: List[Int], d0: Int, p: List[Int]) raises -> Int:
    """h = [x, g, dx, aux] (the host entry's order); params = [N, C, HW, training]."""
    var prm = _bn_prm(p)
    var training = p[3] != 0
    _addrs(h, 4)
    var a: List[Int] = [h[0], h[1], h[3], h[2]]
    var d = (d0 & 3) | (((d0 >> 3) & 1) << 2) | (((d0 >> 2) & 1) << 3)
    batchnorm_backward_m(a, d, prm, training)
    return Int(prm[0]) * Int(prm[1]) * Int(prm[2])


def dropout2d_m_run(a: List[Int], d: Int, p: List[Int], f: List[Float64]) raises -> Int:
    """a = [x, y, mask]; params as `x_cnn_dropout2d`'s."""
    var prm = _dropout_prm(p)
    var h = _drop_p(f)
    _addrs(a, 3)
    var total = p[0] * p[1] * p[2]
    dropout2d_m(a, d, total, prm, h)
    return total


def spmm_m_run(a0: List[Int], d0: Int, p: List[Int]) raises -> Int:
    """a = [vals, h, csr, out]; params = [n, F, nnz, mode]. A resident csr
    (bit 2) is a `x_cnn_csr_upload` handle, checked when it was made; a host
    one is checked here as `x_cnn_spmm` checks it."""
    _need(p, 4, "spmm")
    var n = p[0]; var F = p[1]; var nnz = p[2]; var mode = p[3]
    _addrs(a0, 4)
    var a = a0.copy()
    var d = d0
    if (d >> 2) & 1 == 0:
        _ = _csr(a[2], n, F, nnz, mode)
    elif n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    if nnz <= 0:
        # the host entry's: no values are read, h stands in for them
        a[0] = a[1]
        d = (d & ~1) | ((d >> 1) & 1)
    spmm_m(a, d, nnz, n + 1 + 2 * nnz, prm)
    return n * F


def adaptive_pool_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [in, out, idx (int32)]; params as `x_cnn_adaptive_pool`'s. The
    average kinds (0, 1) never touch idx."""
    var t = _adaptive_prm(p)
    var kind = t[1]
    var nc = Int(t[0][0]) * Int(t[0][1])
    var nin = nc * Int(t[0][2]) * Int(t[0][3])
    var nout = nc * Int(t[0][4]) * Int(t[0][5])
    _addrs(a, 3)
    adaptive_pool_m(a, d, t[0], kind)
    return nout if kind == 0 or kind == 2 else nin


def graph_op_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [a, b, aux, out, csr]; params = [n, F, nnz, kind] as
    `x_cnn_graph_op`'s. A resident csr (bit 4) is a `x_cnn_csr_upload`
    handle, checked when it was made; a host one is checked here. b is read
    only by the backward kinds (1, 3), csr only by the SAGE kinds (0, 1)."""
    _need(p, 4, "graph op")
    var kind = p[3]
    if kind < 0 or kind > 3:
        raise Error("x_cnn graph op: kind in 0..3")
    var n = p[0]; var F = p[1]; var nnz = p[2]
    _addrs(a, 5)
    if kind < 2 and (d >> 4) & 1 == 0:
        _ = _csr(a[4], n, F, nnz, 0)
    elif n <= 0 or F <= 0 or nnz < 0:
        raise Error("x_cnn graph op: positive n and F, nnz >= 0")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(0)]
    graph_op_m(a, d, prm, kind)
    return n * F


def gcn_norm_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [w, vals, csr]; params = [n, F, nnz, mode] as `x_cnn_gcn_norm`'s."""
    _need(p, 4, "gcn_norm")
    var n = p[0]; var F = p[1]; var nnz = p[2]; var mode = p[3]
    _addrs(a, 3)
    if (d >> 2) & 1 == 0:
        _ = _csr(a[2], n, F, nnz, mode)
    elif n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    if nnz <= 0:
        raise Error("x_cnn gcn_norm: at least one edge (the self loops) is required")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    gcn_norm_m(a, d, prm)
    return nnz


def pad2d_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [in, out]; params = `x_cnn_pad2d_forward`'s nine and a tenth, 0 the
    forward (in x, out padded) or 1 the backward (in the padded gradient, out
    dx)."""
    var prm = _pad_prm(p)
    _need(p, 10, "pad")
    var back = p[9] != 0
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var no = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    _addrs(a, 2)
    pad2d_m(a, d, prm, back)
    return nx if back else no


def chan_copy_m_run(a: List[Int], d: Int, p: List[Int]) raises -> Int:
    """a = [src, dst]; params = [N, C, HW, cg, c0, place]: place 0 copies
    channels [c0, c0 + cg) of the (N, C, HW) src into the (N, cg, HW) dst;
    place 1 copies the (N, cg, HW) src into those channels of the resident
    (N, C, HW) dst."""
    _need(p, 6, "chan copy")
    var N = p[0]; var C = p[1]; var HW = p[2]; var cg = p[3]; var c0 = p[4]
    var place = p[5] != 0
    if N <= 0 or C <= 0 or HW <= 0 or cg <= 0 or c0 < 0 or c0 + cg > C:
        raise Error("x_cnn chan copy: positive N, C, HW, cg and a group inside [0, C)")
    var prm: List[Int32] = [Int32(N), Int32(C), Int32(HW), Int32(cg), Int32(c0)]
    _addrs(a, 2)
    chan_copy_m(a, d, prm, place)
    return N * cg * HW


# ------------------------------------------------------------ dispatch and entry

def x_cnn_run(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """One operation of this image (the Python entries and `x_cnn_entry_v1`)."""
    if op == XC_RES_ALLOC:
        return res_alloc_run(p)
    if op == XC_RES_ALLOC_NOFILL:
        return res_alloc_nofill_run(p)
    if op == XC_RES_FREE:
        return res_free_run(a)
    if op == XC_RES_UPLOAD:
        return res_upload_run(a, p)
    if op == XC_RES_DOWNLOAD:
        return res_download_run(a, p)
    if op == XC_CSR_UPLOAD:
        return csr_upload_run(a, p)
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
    var name = xcnn_op_name(op)
    var q = _tail(p, name)
    var d = p[0]
    if op == XC_GEMM_M:
        return gemm_m_run(a, d, q)
    if op == XC_CONV2D_FORWARD_M:
        return conv2d_forward_m_run(a, d, q)
    if op == XC_CONV2D_BACKWARD_M:
        return conv2d_backward_m_run(a, d, q)
    if op == XC_MAXPOOL2D_FORWARD_M:
        return maxpool2d_forward_m_run(a, d, q)
    if op == XC_MAXPOOL2D_BACKWARD_M:
        return maxpool2d_backward_m_run(a, d, q)
    if op == XC_AVGPOOL2D_FORWARD_M:
        return avgpool2d_forward_m_run(a, d, q)
    if op == XC_AVGPOOL2D_BACKWARD_M:
        return avgpool2d_backward_m_run(a, d, q)
    if op == XC_MAP2_M:
        return map2_m_run(a, d, q)
    if op == XC_LINEAR_FORWARD_M:
        return linear_forward_m_run(a, d, q)
    if op == XC_BATCHNORM_FORWARD_M:
        return batchnorm_forward_m_run(a, d, q)
    if op == XC_BATCHNORM_BACKWARD_M:
        return batchnorm_backward_m_run(a, d, q)
    if op == XC_DROPOUT2D_M:
        return dropout2d_m_run(a, d, q, f)
    if op == XC_SPMM_M:
        return spmm_m_run(a, d, q)
    if op == XC_ADAPTIVE_POOL_M:
        return adaptive_pool_m_run(a, d, q)
    if op == XC_GRAPH_OP_M:
        return graph_op_m_run(a, d, q)
    if op == XC_GCN_NORM_M:
        return gcn_norm_m_run(a, d, q)
    if op == XC_PAD2D_M:
        return pad2d_m_run(a, d, q)
    if op == XC_CHAN_COPY_M:
        return chan_copy_m_run(a, d, q)
    raise Error("AttributeError: the CNN binding has no " + name)


def _float_words(address: Int, count: Int) raises -> List[Float64]:
    if count < 0 or (count > 0 and address == 0):
        raise Error("native descriptor: null or invalid scalar span")
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Float64](capacity=count)
    for i in range(count):  # small-loop(validated native launch scalar words)
        out.append(source[i])
    return out^


def x_cnn_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The x_cnn entry of the convolution table (`XC_ENTRY_ABI`): addresses
    aa[0:na], integers pp[0:np], scalars ff[0:nf]. Writes the run's count to
    result; a refusal goes to sink."""
    try:
        if operation < 0 or operation >= XC_OP_COUNT:
            raise Error("x_cnn entry: unknown operation " + String(operation))
        if not result:
            raise Error("x_cnn entry: invalid request spans")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = _float_words(ff, nf)
        var value = x_cnn_run(operation, a, p, f)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XCnnEntryOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the x_cnn table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _OWNER_NAME = "MojolearnXCnnEntryOwnerV1"
comptime XCNN_ENTRY_OWNER = SharedSlot[XCnnEntryOwner, _OWNER_NAME]


def publish_x_cnn_entries() raises:
    """Publish this image's entry at module init. A build with no GPU vendor
    ("none") publishes nothing: its callers refuse with the table's
    "provider is not published". A repeated init publishes the same pointer;
    a second image with the same key and another pointer refuses
    (core/entry_table.mojo)."""
    var word = xcnn_entry_vendor(COMPILED_VENDOR)
    if word < 0:
        return
    var owner = XCNN_ENTRY_OWNER.get_or_create_ptr()
    var key = XcnnEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    xcnn_table_publish(key, Int(owner), [String(XC_ENTRY_ABI)], [XC_OP_COUNT], [x_cnn_entry_v1])
