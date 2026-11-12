# SPDX-License-Identifier: Apache-2.0
"""The convolution family's native entry table, reached natively (lane
calltable-xcnn, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publication `_mojolearn_x_cnn_host`
makes at module init (`publish_x_cnn_host_entries`), then an unknown
operation refuses through the entry's sink. Then one case per published
operation code (`XC_OP_COUNT`), each through `xcnn_entry_call` and the
published pointer, with literal expected words, the run's own refusal text,
or, for the GPU image's operations, the host image's refusal ("the CNN
binding has no x_cnn_<name>", the text the missing Python entry raised). No
Python participates.

The cases are independent units: on the host column they run across the
native host worker pool; on a device column one after another. Run in a
fresh process (the first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.xcnn_entry_table import (
    XcnnEntryKey, xcnn_entry_ref, xcnn_entry_call, xcnn_table_entry, XC_ENTRY_ABI, XC_OP_COUNT,
    XC_RES_ALLOC, XC_RES_ALLOC_NOFILL, XC_RES_FREE, XC_RES_UPLOAD, XC_RES_DOWNLOAD, XC_GEMM, XC_CONV2D_FORWARD, XC_CONV2D_BACKWARD, XC_MAXPOOL2D_FORWARD, XC_MAXPOOL2D_BACKWARD, XC_AVGPOOL2D_FORWARD, XC_AVGPOOL2D_BACKWARD, XC_RELU_FORWARD, XC_RELU_BACKWARD, XC_ADD, XC_MUL, XC_LINEAR_FORWARD, XC_LINEAR_BACKWARD, XC_BATCHNORM_FORWARD, XC_BATCHNORM_BACKWARD, XC_DROPOUT2D, XC_SPMM, XC_GCN_NORM, XC_PAD2D_FORWARD, XC_PAD2D_BACKWARD, XC_ADAPTIVE_POOL, XC_GRAPH_OP, XC_CSR_BUILD, XC_CSR_UPLOAD, XC_PERM_GATHER, XC_VALS_T, XC_GCN_LOOPS, XC_GCN_BACKWARD_T, XC_SAGE_BACKWARD_T, XC_GEMM_M, XC_CONV2D_FORWARD_M, XC_CONV2D_BACKWARD_M, XC_MAXPOOL2D_FORWARD_M, XC_MAXPOOL2D_BACKWARD_M, XC_AVGPOOL2D_FORWARD_M, XC_AVGPOOL2D_BACKWARD_M, XC_MAP2_M, XC_LINEAR_FORWARD_M, XC_BATCHNORM_FORWARD_M, XC_BATCHNORM_BACKWARD_M, XC_DROPOUT2D_M, XC_SPMM_M, XC_ADAPTIVE_POOL_M, XC_GRAPH_OP_M, XC_GCN_NORM_M, XC_PAD2D_M, XC_CHAN_COPY_M,
)
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries

comptime XCNN_ENTRY_CASES = 52


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> XcnnEntryKey:
    return XcnnEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _f32(mut l: List[Float32]) -> Int:
    return Int(l.unsafe_ptr())


def _i32(mut l: List[Int32]) -> Int:
    return Int(l.unsafe_ptr())


def _call(op: Int, addrs: List[Int], ints: List[Int]) raises -> Int:
    var floats = List[Float64]()
    return xcnn_entry_call(xcnn_entry_ref(_key()), op, addrs, ints, floats)


def _same(got: List[Float32], expected: List[Float32], what: String) raises:
    require(len(got) == len(expected), what + ": length differs")
    for i in range(len(got)):  # small-loop(a case's literal output words)
        require(got[i] == expected[i], what + ": word " + String(i) + " is " + String(got[i])
                + ", expected " + String(expected[i]))


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _refuses_call(op: Int, addrs: List[Int], ints: List[Int], text: String) raises:
    try:
        _ = _call(op, addrs, ints)
    except problem:
        _refuses(problem, text)
        return
    raise Error("operation " + String(op) + " did not refuse: " + text)


def _conv_words() -> List[Int]:
    """1x1x2x2 input, one 1x1 filter, stride 1, no pad, bias on."""
    var w: List[Int] = [1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    return w^


def _pool_words() -> List[Int]:
    """1x1x2x2 input, one 2x2 window, stride 2, no pad, floor mode."""
    var w: List[Int] = [1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    return w^


def precondition_unpublished() raises:
    try:
        _ = _call(XC_RES_ALLOC, List[Int](), [1])
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished x_cnn table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = xcnn_entry_ref(_key())
    var second = xcnn_entry_ref(_key())
    require(entry_ref_same(first, second), "x_cnn: a second resolution returned another entry")
    try:
        _ = xcnn_table_entry(_key(), 0, String(XC_ENTRY_ABI), XC_OP_COUNT + 1)
        raise Error("x_cnn: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "x_cnn: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    try:
        _ = _call(XC_OP_COUNT + 7, List[Int](), List[Int]())
    except problem:
        _refuses(problem, "x_cnn entry: unknown operation")
        return
    raise Error("an unknown x_cnn operation did not refuse")


def case_res_alloc() raises:
    var h = _call(XC_RES_ALLOC, List[Int](), [3])
    require(h != 0, "res_alloc returned a null handle")
    var out: List[Float32] = [9.0, 9.0, 9.0]
    require(_call(XC_RES_DOWNLOAD, [h, _f32(out)], [3]) == 3, "res_download count")
    _same(out, [0.0, 0.0, 0.0], "res_alloc zero fill")
    _ = _call(XC_RES_FREE, [h], List[Int]())

def case_res_alloc_nofill() raises:
    # the host image's RES_ALLOC_NOFILL is its zero-filled allocation
    var h = _call(XC_RES_ALLOC_NOFILL, List[Int](), [2])
    var out: List[Float32] = [9.0, 9.0]
    _ = _call(XC_RES_DOWNLOAD, [h, _f32(out)], [2])
    _same(out, [0.0, 0.0], "res_alloc_nofill on the host")
    _ = _call(XC_RES_FREE, [h], List[Int]())

def case_res_free() raises:
    var h = _call(XC_RES_ALLOC, List[Int](), [1])
    require(_call(XC_RES_FREE, [h], List[Int]()) == 0, "res_free result")

def case_res_upload() raises:
    var h = _call(XC_RES_ALLOC, List[Int](), [3])
    var src: List[Float32] = [1.0, 2.0, 3.0]
    require(_call(XC_RES_UPLOAD, [h, _f32(src)], [3]) == 3, "res_upload count")
    var out: List[Float32] = [0.0, 0.0, 0.0]
    _ = _call(XC_RES_DOWNLOAD, [h, _f32(out)], [3])
    _same(out, [1.0, 2.0, 3.0], "res_upload words")
    _ = _call(XC_RES_FREE, [h], List[Int]())
    _ = src

def case_res_download() raises:
    var h = _call(XC_RES_ALLOC, List[Int](), [2])
    var src: List[Float32] = [5.0, 6.0]
    _ = _call(XC_RES_UPLOAD, [h, _f32(src)], [2])
    var out: List[Float32] = [0.0, 0.0]
    require(_call(XC_RES_DOWNLOAD, [h, _f32(out)], [2]) == 2, "res_download count")
    _same(out, [5.0, 6.0], "res_download words")
    _ = _call(XC_RES_FREE, [h], List[Int]())
    _ = src

def case_gemm() raises:
    var a: List[Float32] = [3.0]
    var b: List[Float32] = [4.0]
    var c: List[Float32] = [0.0]
    require(_call(XC_GEMM, [_f32(a), _f32(b), _f32(c)], [1, 1, 1, 0]) == 1, "gemm count")
    _same(c, [12.0], "gemm 1x1x1")
    _ = a
    _ = b

def case_conv2d_forward() raises:
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var w: List[Float32] = [2.0]
    var b: List[Float32] = [1.0]
    var y: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    require(_call(XC_CONV2D_FORWARD, [_f32(x), _f32(w), _f32(b), _f32(y)], _conv_words()) == 4, "conv forward count")
    _same(y, [3.0, 5.0, 7.0, 9.0], "conv 1x1 forward")
    _ = x
    _ = w
    _ = b

def case_conv2d_backward() raises:
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var w: List[Float32] = [2.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var dx: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    var dw: List[Float32] = [0.0]
    var db: List[Float32] = [0.0]
    require(_call(XC_CONV2D_BACKWARD, [_f32(x), _f32(w), _f32(g), _f32(dx), _f32(dw), _f32(db)], _conv_words()) == 4,
            "conv backward count")
    _same(dx, [2.0, 2.0, 2.0, 2.0], "conv 1x1 dx")
    _same(dw, [10.0], "conv 1x1 dW")
    _same(db, [4.0], "conv 1x1 db")
    _ = x
    _ = w
    _ = g

def case_maxpool2d_forward() raises:
    var x: List[Float32] = [1.0, 4.0, 2.0, 3.0]
    var y: List[Float32] = [0.0]
    var idx: List[Int32] = [-1]
    require(_call(XC_MAXPOOL2D_FORWARD, [_f32(x), _f32(y), _i32(idx)], _pool_words()) == 1, "maxpool forward count")
    _same(y, [4.0], "maxpool 2x2 forward")
    _ = x

def case_maxpool2d_backward() raises:
    var x: List[Float32] = [1.0, 4.0, 2.0, 3.0]
    var y: List[Float32] = [0.0]
    var idx: List[Int32] = [-1]
    _ = _call(XC_MAXPOOL2D_FORWARD, [_f32(x), _f32(y), _i32(idx)], _pool_words())
    var g: List[Float32] = [5.0]
    var dx: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    require(_call(XC_MAXPOOL2D_BACKWARD, [_f32(g), _i32(idx), _f32(dx)], _pool_words()) == 4, "maxpool backward count")
    _same(dx, [0.0, 5.0, 0.0, 0.0], "maxpool 2x2 backward")
    _ = x
    _ = y
    _ = g

def case_avgpool2d_forward() raises:
    var x: List[Float32] = [1.0, 4.0, 2.0, 3.0]
    var y: List[Float32] = [0.0]
    require(_call(XC_AVGPOOL2D_FORWARD, [_f32(x), _f32(y)], _pool_words()) == 1, "avgpool forward count")
    _same(y, [2.5], "avgpool 2x2 forward")
    _ = x

def case_avgpool2d_backward() raises:
    var g: List[Float32] = [4.0]
    var dx: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    require(_call(XC_AVGPOOL2D_BACKWARD, [_f32(g), _f32(dx)], _pool_words()) == 4, "avgpool backward count")
    _same(dx, [1.0, 1.0, 1.0, 1.0], "avgpool 2x2 backward")
    _ = g

def case_relu_forward() raises:
    var x: List[Float32] = [-1.0, 2.0]
    var y: List[Float32] = [9.0, 9.0]
    require(_call(XC_RELU_FORWARD, [_f32(x), _f32(y)], [2]) == 2, "relu forward count")
    _same(y, [0.0, 2.0], "relu forward")
    _ = x

def case_relu_backward() raises:
    var x: List[Float32] = [-1.0, 2.0]
    var g: List[Float32] = [5.0, 7.0]
    var dx: List[Float32] = [9.0, 9.0]
    require(_call(XC_RELU_BACKWARD, [_f32(x), _f32(g), _f32(dx)], [2]) == 2, "relu backward count")
    _same(dx, [0.0, 7.0], "relu backward")
    _ = x
    _ = g

def case_add() raises:
    var a: List[Float32] = [1.0, 2.0]
    var b: List[Float32] = [3.0, 4.0]
    var y: List[Float32] = [0.0, 0.0]
    require(_call(XC_ADD, [_f32(a), _f32(b), _f32(y)], [2]) == 2, "add count")
    _same(y, [4.0, 6.0], "add")
    _ = a
    _ = b

def case_mul() raises:
    var a: List[Float32] = [1.0, 2.0]
    var b: List[Float32] = [3.0, 4.0]
    var y: List[Float32] = [0.0, 0.0]
    require(_call(XC_MUL, [_f32(a), _f32(b), _f32(y)], [2]) == 2, "mul count")
    _same(y, [3.0, 8.0], "mul")
    _ = a
    _ = b

def case_linear_forward() raises:
    var x: List[Float32] = [1.0, 2.0]
    var w: List[Float32] = [3.0, 4.0]
    var b: List[Float32] = [5.0]
    var y: List[Float32] = [0.0]
    require(_call(XC_LINEAR_FORWARD, [_f32(x), _f32(w), _f32(b), _f32(y)], [1, 2, 1]) == 1, "linear forward count")
    _same(y, [16.0], "linear forward")
    _ = x
    _ = w
    _ = b

def case_linear_backward() raises:
    var x: List[Float32] = [1.0, 2.0]
    var w: List[Float32] = [3.0, 4.0]
    var g: List[Float32] = [1.0]
    var dx: List[Float32] = [0.0, 0.0]
    var dw: List[Float32] = [0.0, 0.0]
    var db: List[Float32] = [0.0]
    require(_call(XC_LINEAR_BACKWARD, [_f32(x), _f32(w), _f32(g), _f32(dx), _f32(dw), _f32(db)], [1, 2, 1]) == 1,
            "linear backward count")
    _same(dx, [3.0, 4.0], "linear dx")
    _same(dw, [1.0, 2.0], "linear dW")
    _same(db, [1.0], "linear db")
    _ = x
    _ = w
    _ = g

def case_batchnorm_forward() raises:
    _refuses_call(XC_BATCHNORM_FORWARD, [0, 0, 0, 0], [0, 1, 1, 0], "x_cnn batchnorm: positive N, C and H*W required")

def case_batchnorm_backward() raises:
    _refuses_call(XC_BATCHNORM_BACKWARD, [0, 0, 0, 0], [1, 0, 1, 0], "x_cnn batchnorm: positive N, C and H*W required")

def case_dropout2d() raises:
    var floats: List[Float64] = [0.5]
    try:
        _ = xcnn_entry_call(xcnn_entry_ref(_key()), XC_DROPOUT2D, [0, 0, 0], [1, 1, 1, -1, 0, 0, 0], floats)
    except problem:
        _refuses(problem, "x_cnn dropout2d: negative parameter")
        return
    raise Error("dropout2d accepted a negative parameter")

def case_spmm() raises:
    _refuses_call(XC_SPMM, [0, 0, 0, 0], List[Int](), "x_cnn spmm: 4 parameter words required")

def case_gcn_norm() raises:
    _refuses_call(XC_GCN_NORM, [0, 0, 0], List[Int](), "x_cnn gcn_norm: 4 parameter words required")

def case_pad2d_forward() raises:
    var x: List[Float32] = [7.0]
    var y: List[Float32] = [9.0, 9.0]
    require(_call(XC_PAD2D_FORWARD, [_f32(x), _f32(y)], [1, 1, 1, 1, 1, 0, 0, 0, 0]) == 2, "pad forward count")
    _same(y, [0.0, 7.0], "zero pad top")
    _ = x

def case_pad2d_backward() raises:
    var g: List[Float32] = [5.0, 7.0]
    var dx: List[Float32] = [0.0]
    require(_call(XC_PAD2D_BACKWARD, [_f32(g), _f32(dx)], [1, 1, 1, 1, 1, 0, 0, 0, 0]) == 1, "pad backward count")
    _same(dx, [7.0], "zero pad top backward")
    _ = g

def case_adaptive_pool() raises:
    _refuses_call(XC_ADAPTIVE_POOL, [0, 0, 0], [1, 1, 1, 1, 1, 1, 5], "x_cnn adaptive pool: kind in 0..3")

def case_graph_op() raises:
    _refuses_call(XC_GRAPH_OP, [0, 0, 0, 0, 0], [1, 1, 0, 4], "x_cnn graph op: kind in 0..3")

def case_csr_build() raises:
    var rows: List[Int32] = [1]
    var cols: List[Int32] = [0]
    var csr: List[Int32] = [-1, -1, -1, -1, -1]
    var order: List[Int32] = [-1]
    require(_call(XC_CSR_BUILD, [_i32(rows), _i32(cols), _i32(csr), _i32(order)], [2, 1]) == 5, "csr_build count")
    require(csr[0] == 0 and csr[1] == 0 and csr[2] == 1 and csr[3] == 0 and csr[4] == 1, "csr_build words")
    require(order[0] == 0, "csr_build order")
    _ = rows
    _ = cols

def case_csr_upload() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_CSR_UPLOAD, List[Int](), [0], "the CNN binding has no x_cnn_csr_upload")

def case_perm_gather() raises:
    var src: List[Float32] = [10.0, 20.0, 30.0]
    var idx: List[Int32] = [2, 0]
    var y: List[Float32] = [0.0, 0.0]
    require(_call(XC_PERM_GATHER, [_f32(src), _i32(idx), _f32(y)], [3, 2]) == 2, "perm_gather count")
    _same(y, [30.0, 10.0], "perm_gather")
    _ = src
    _ = idx

def case_vals_t() raises:
    require(_call(XC_VALS_T, List[Int](), [0]) == 0, "vals_t of no edges")
    _refuses_call(XC_VALS_T, List[Int](), [-1], "x_cnn vals_t: nnz >= 0 required")

def case_gcn_loops() raises:
    _refuses_call(XC_GCN_LOOPS, [0, 0, 0, 0, 0, 0], [0, 0, 0],
                  "x_cnn gcn_loops: addrs [src, dst, w, src_out, dst_out, w_out], positive n and nnz >= 0 required")

def case_gcn_backward_t() raises:
    _refuses_call(XC_GCN_BACKWARD_T, [0], [1, 1], "x_cnn: this entry takes 13 addresses")

def case_sage_backward_t() raises:
    _refuses_call(XC_SAGE_BACKWARD_T, List[Int](), List[Int](), "x_cnn sage backward: 2 parameter words required")

def case_gemm_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_GEMM_M, List[Int](), [0], "the CNN binding has no x_cnn_gemm_m")

def case_conv2d_forward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_CONV2D_FORWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_conv2d_forward_m")

def case_conv2d_backward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_CONV2D_BACKWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_conv2d_backward_m")

def case_maxpool2d_forward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_MAXPOOL2D_FORWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_maxpool2d_forward_m")

def case_maxpool2d_backward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_MAXPOOL2D_BACKWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_maxpool2d_backward_m")

def case_avgpool2d_forward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_AVGPOOL2D_FORWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_avgpool2d_forward_m")

def case_avgpool2d_backward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_AVGPOOL2D_BACKWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_avgpool2d_backward_m")

def case_map2_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_MAP2_M, List[Int](), [0], "the CNN binding has no x_cnn_map2_m")

def case_linear_forward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_LINEAR_FORWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_linear_forward_m")

def case_batchnorm_forward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_BATCHNORM_FORWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_batchnorm_forward_m")

def case_batchnorm_backward_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_BATCHNORM_BACKWARD_M, List[Int](), [0], "the CNN binding has no x_cnn_batchnorm_backward_m")

def case_dropout2d_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_DROPOUT2D_M, List[Int](), [0], "the CNN binding has no x_cnn_dropout2d_m")

def case_spmm_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_SPMM_M, List[Int](), [0], "the CNN binding has no x_cnn_spmm_m")

def case_adaptive_pool_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_ADAPTIVE_POOL_M, List[Int](), [0], "the CNN binding has no x_cnn_adaptive_pool_m")

def case_graph_op_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_GRAPH_OP_M, List[Int](), [0], "the CNN binding has no x_cnn_graph_op_m")

def case_gcn_norm_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_GCN_NORM_M, List[Int](), [0], "the CNN binding has no x_cnn_gcn_norm_m")

def case_pad2d_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_PAD2D_M, List[Int](), [0], "the CNN binding has no x_cnn_pad2d_m")

def case_chan_copy_m() raises:
    # device only: the host image refuses with the missing Python entry's text
    _refuses_call(XC_CHAN_COPY_M, List[Int](), [0], "the CNN binding has no x_cnn_chan_copy_m")


def case(unit: Int) raises:
    if unit == 0:
        case_res_alloc()
        return
    if unit == 1:
        case_res_alloc_nofill()
        return
    if unit == 2:
        case_res_free()
        return
    if unit == 3:
        case_res_upload()
        return
    if unit == 4:
        case_res_download()
        return
    if unit == 5:
        case_gemm()
        return
    if unit == 6:
        case_conv2d_forward()
        return
    if unit == 7:
        case_conv2d_backward()
        return
    if unit == 8:
        case_maxpool2d_forward()
        return
    if unit == 9:
        case_maxpool2d_backward()
        return
    if unit == 10:
        case_avgpool2d_forward()
        return
    if unit == 11:
        case_avgpool2d_backward()
        return
    if unit == 12:
        case_relu_forward()
        return
    if unit == 13:
        case_relu_backward()
        return
    if unit == 14:
        case_add()
        return
    if unit == 15:
        case_mul()
        return
    if unit == 16:
        case_linear_forward()
        return
    if unit == 17:
        case_linear_backward()
        return
    if unit == 18:
        case_batchnorm_forward()
        return
    if unit == 19:
        case_batchnorm_backward()
        return
    if unit == 20:
        case_dropout2d()
        return
    if unit == 21:
        case_spmm()
        return
    if unit == 22:
        case_gcn_norm()
        return
    if unit == 23:
        case_pad2d_forward()
        return
    if unit == 24:
        case_pad2d_backward()
        return
    if unit == 25:
        case_adaptive_pool()
        return
    if unit == 26:
        case_graph_op()
        return
    if unit == 27:
        case_csr_build()
        return
    if unit == 28:
        case_csr_upload()
        return
    if unit == 29:
        case_perm_gather()
        return
    if unit == 30:
        case_vals_t()
        return
    if unit == 31:
        case_gcn_loops()
        return
    if unit == 32:
        case_gcn_backward_t()
        return
    if unit == 33:
        case_sage_backward_t()
        return
    if unit == 34:
        case_gemm_m()
        return
    if unit == 35:
        case_conv2d_forward_m()
        return
    if unit == 36:
        case_conv2d_backward_m()
        return
    if unit == 37:
        case_maxpool2d_forward_m()
        return
    if unit == 38:
        case_maxpool2d_backward_m()
        return
    if unit == 39:
        case_avgpool2d_forward_m()
        return
    if unit == 40:
        case_avgpool2d_backward_m()
        return
    if unit == 41:
        case_map2_m()
        return
    if unit == 42:
        case_linear_forward_m()
        return
    if unit == 43:
        case_batchnorm_forward_m()
        return
    if unit == 44:
        case_batchnorm_backward_m()
        return
    if unit == 45:
        case_dropout2d_m()
        return
    if unit == 46:
        case_spmm_m()
        return
    if unit == 47:
        case_adaptive_pool_m()
        return
    if unit == 48:
        case_graph_op_m()
        return
    if unit == 49:
        case_gcn_norm_m()
        return
    if unit == 50:
        case_pad2d_m()
        return
    if unit == 51:
        case_chan_copy_m()
        return
    raise Error("unknown x_cnn entry case " + String(unit))


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    preconditions_published()
    var failures = List[Int](length=XCNN_ENTRY_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn entry case", unit, String(problem))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        host_parallelize(task, XCNN_ENTRY_CASES)
    else:
        for unit in range(XCNN_ENTRY_CASES):  # driver-loop: device cases share one device context
            task(unit)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn entry cases failed: " + String(failed))
    print("PASS x_cnn entry", XCNN_ENTRY_CASES)
