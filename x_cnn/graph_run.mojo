# SPDX-License-Identifier: Apache-2.0
"""RUN half of GCNConv / SAGEConv and their linear and ReLU layers
(`bindings/xcnn_graph_native.mojo` is the UNPACK/PACK boundary; lane
doorsplit-cnn-xseq, 2026-11-11; docs/plans/PYTHON_AT_THE_BOUNDARY.md
section 2.2).

Typed values only: address and handle words of buffers the boundary owns,
shape words, the layer's configuration words, the layer's refusals (texts
unchanged) and every x_cnn native entry the graph layers call (CSR build and
upload, value gathers, GEMM / SpMM / add / map2, the GCN self loops and
normalization, the graph operation, the linear and ReLU entries and the
tape backward entries), with the argument words and residency bits the door
used. The boundary releases the GIL around each entry call here, where the
door released it before the split.
"""
from std.math import sqrt
from std.memory import unsafe_memcpy
from core.xcnn_entry_table import (
    XcnnEntryKey, XC_CSR_BUILD, XC_PERM_GATHER, XC_VALS_T, XC_GEMM, XC_SPMM, XC_ADD, XC_LINEAR_FORWARD,
    XC_RELU_FORWARD, XC_RES_FREE, XC_CSR_UPLOAD, XC_RES_DOWNLOAD, XC_GCN_LOOPS, XC_GCN_NORM, XC_GCN_NORM_M,
    XC_GEMM_M, XC_SPMM_M, XC_MAP2_M, XC_GRAPH_OP, XC_GRAPH_OP_M, XC_LINEAR_FORWARD_M, XC_LINEAR_BACKWARD,
    XC_GCN_BACKWARD_T, XC_SAGE_BACKWARD_T,
)
from core.xcnn_graph_impl import graph_bias_rows, graph_admit_edges
from core.xseq_marshal import xseq_fill_f32
from x_cnn.door_run import XcnnF32P, xcnn_run_call

comptime GraphU8P = MutPointer[UInt8, MutUntrackedOrigin]


# ---------------------------------------------------------------- shape words and refusals

def graph_run_same_shape(left: List[Int], right: List[Int]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):  # small-loop(tensor rank)
        if left[i] != right[i]:
            return False
    return True


def graph_run_csr_words(n: Int, nnz: Int) -> Int:
    """int32 words of one CSR view: n + 1 row offsets, then the column and
    order words of every entry."""
    return n + 1 + 2 * nnz


def graph_run_check_edge_layout(shape: List[Int]) raises:
    if len(shape) != 2 or shape[0] != 2:
        raise Error("mojolearn: edge_index must be (2, E)")


def graph_run_check_cache_dims(n: Int, count: Int) raises:
    if n < 0 or count < 0:
        raise Error("graph cache dimensions must be nonnegative")


def graph_run_check_index_layout(shape: List[Int], expected: Int) raises:
    if not graph_run_same_shape(shape, [expected]):
        raise Error("cached graph index layout differs from its node/edge metadata")


def graph_run_check_gemm(size_a: Int, size_b: Int, m: Int, n: Int, k: Int) raises:
    if size_a != m * k or size_b != k * n:
        raise Error("GEMM inputs differ from the declared graph operation dimensions")


def graph_run_check_graph_x(shape: List[Int], in_channels: Int) raises:
    """The public forward's input check."""
    if len(shape) != 2 or shape[1] != in_channels:
        raise Error("mojolearn: graph x must be (n, in_channels)")


def graph_run_check_x(shape: List[Int], d_in: Int) raises:
    """The resident forward's input check."""
    if len(shape) != 2 or shape[1] != d_in:
        raise Error("ValueError: mojolearn: x must be (n, " + String(d_in) + ")")


def graph_run_check_edge_weight(weights: Int, edges: Int) raises:
    if weights != edges:
        raise Error("ValueError: mojolearn: edge_weight must have one entry per edge")


def graph_run_check_bias(size: Int, f: Int) raises:
    if size != f:
        raise Error("ValueError: mojolearn: bias_ has the wrong size")


def graph_run_check_gcn_weight(shape: List[Int], f: Int, d: Int) raises:
    if not graph_run_same_shape(shape, [f, d]):
        raise Error("GCN weight shape differs from configured channels")


def graph_run_check_root_weight(shape: List[Int], f: Int, d: Int) raises:
    if not graph_run_same_shape(shape, [f, d]):
        raise Error("SAGE root weight shape differs from configured channels")


def graph_run_check_linear_input(shape: List[Int], in_features: Int) raises:
    if len(shape) != 2 or shape[1] != in_features:
        raise Error("linear input shape differs from in_features")


def graph_run_check_linear_weights(weight: List[Int], bias: List[Int], f: Int, d: Int) raises:
    if not graph_run_same_shape(weight, [f, d]) or not graph_run_same_shape(bias, [f]):
        raise Error("linear weights differ from configured dimensions")


def graph_run_check_linear_gradient(x: List[Int], g: List[Int], d: Int, f: Int) raises:
    if len(x) != 2 or x[1] != d or not graph_run_same_shape(g, [x[0], f]):
        raise Error("linear gradient shape differs from forward output")


def graph_run_check_gradient(x: List[Int], g: List[Int], f: Int) raises:
    if len(x) != 2 or not graph_run_same_shape(g, [x[0], f]):
        raise Error("mojolearn: graph grad_out shape differs from forward output")


def graph_run_check_channels(d: Int, f: Int) raises:
    if d <= 0 or f <= 0:
        raise Error("graph layer channels must be positive")


def graph_run_check_features(d: Int, f: Int) raises:
    if d <= 0 or f <= 0:
        raise Error("linear feature dimensions must be positive")


def graph_run_check_aggr(aggr: String) raises:
    if aggr != "mean" and aggr != "sum" and aggr != "add" and aggr != "max":
        raise Error("NotImplementedError: SAGE aggregation must be mean, sum, add or max")


def graph_run_aggr_kind(aggr: String) -> Int:
    """The tape's aggregation word: 2 max, 1 mean, 0 sum / add."""
    return 2 if aggr == "max" else (1 if aggr == "mean" else 0)


def graph_run_sage_res_form(mixed: Bool, project: Bool, graph_route: Bool, normalize: Bool, aggr: String) -> Bool:
    """`SAGEConv._res_form(b)`: whether the forward runs on resident buffers."""
    if not mixed or project:
        return False
    if graph_route:
        return True
    return not normalize and (aggr == "mean" or aggr == "sum" or aggr == "add")


def graph_run_gcn_bound(d: Int, f: Int) -> Float64:
    """GCN's Glorot-uniform weight bound."""
    return sqrt(6.0 / Float64(d + f))


def graph_run_fan_in_bound(d: Int) -> Float64:
    """The linear and SAGE root weights' fan-in bound."""
    return 1.0 / sqrt(Float64(d))


# ---------------------------------------------------------------- host words

def graph_run_ones(dst: Int, count: Int):
    """`count` float32 ones at address `dst`."""
    xseq_fill_f32(XcnnF32P(unsafe_from_address=dst), count, 1)


def graph_run_slice(dst: Int, src: Int, offset: Int, count: Int):
    """`count` four-byte words from word `offset` of `src` to `dst`."""
    if count:
        unsafe_memcpy(dest=GraphU8P(unsafe_from_address=dst), src=GraphU8P(unsafe_from_address=src) + offset * 4,
                      count=count * 4)


def graph_run_edges(base: Int, row_step: Int, edge_step: Int, count: Int, item: Int, kind: Int, little: Bool,
                    n: Int, src: Int, dst: Int) raises:
    """The (2, E) edge index at `base` as int32 source and target rows, with
    the node range refusal."""
    if not graph_admit_edges(base, row_step, edge_step, count, item, kind, little, n, src, dst):
        raise Error("mojolearn: edge_index refers to a node that does not exist")


def graph_run_bias_rows(bias: Int, dst: Int, rows: Int, cols: Int):
    """`np.broadcast_to(bias, (rows, cols))` written C-contiguous."""
    graph_bias_rows(bias, dst, rows, cols)


# ---------------------------------------------------------------- native entries (GIL released by the caller)

def graph_run_csr_build(key: XcnnEntryKey, rows: Int, cols: Int, csr: Int, order: Int, n: Int, nnz: Int) raises:
    _ = xcnn_run_call(key, XC_CSR_BUILD, [rows, cols, csr, order], [n, nnz])


def graph_run_gather(key: XcnnEntryKey, values: Int, order: Int, dst: Int, size: Int, n: Int) raises:
    _ = xcnn_run_call(key, XC_PERM_GATHER, [values, order, dst], [size, n])


def graph_run_vals_t(key: XcnnEntryKey, vals_f: Int, order_f: Int, order_t: Int, dst: Int, nnz: Int) raises:
    _ = xcnn_run_call(key, XC_VALS_T, [vals_f, order_f, order_t, dst], [nnz])


def graph_run_gemm(key: XcnnEntryKey, a: Int, b: Int, dst: Int, m: Int, n: Int, k: Int, op: Int) raises:
    _ = xcnn_run_call(key, XC_GEMM, [a, b, dst], [m, n, k, op])


def graph_run_spmm(key: XcnnEntryKey, vals: Int, h: Int, dst: Int, csr: Int, n: Int, cols: Int, nnz: Int,
                   mode: Int) raises:
    _ = xcnn_run_call(key, XC_SPMM, [vals, h, dst, csr], [n, cols, nnz, mode])


def graph_run_add(key: XcnnEntryKey, a: Int, b: Int, dst: Int, count: Int) raises:
    _ = xcnn_run_call(key, XC_ADD, [a, b, dst], [count])


def graph_run_linear_forward(key: XcnnEntryKey, x: Int, w: Int, b: Int, dst: Int, rows: Int, d: Int, f: Int) raises:
    _ = xcnn_run_call(key, XC_LINEAR_FORWARD, [x, w, b, dst], [rows, d, f])


def graph_run_linear_forward_m(key: XcnnEntryKey, x: Int, w: Int, b: Int, dst: Int, flags: Int, n: Int, d: Int,
                               f: Int) raises:
    _ = xcnn_run_call(key, XC_LINEAR_FORWARD_M, [x, w, b, dst], [flags, n, d, f])


def graph_run_linear_backward(key: XcnnEntryKey, x: Int, w: Int, g: Int, dx: Int, dw: Int, db: Int, rows: Int,
                              d: Int, f: Int) raises:
    _ = xcnn_run_call(key, XC_LINEAR_BACKWARD, [x, w, g, dx, dw, db], [rows, d, f])


def graph_run_relu_forward(key: XcnnEntryKey, x: Int, dst: Int, count: Int) raises:
    _ = xcnn_run_call(key, XC_RELU_FORWARD, [x, dst], [count])


def graph_run_res_free(key: XcnnEntryKey, h: Int) raises:
    _ = xcnn_run_call(key, XC_RES_FREE, [h], List[Int]())


def graph_run_csr_upload(key: XcnnEntryKey, csr: Int, n: Int, nnz: Int) raises -> Int:
    """One CSR view to a resident buffer; returns its handle."""
    return xcnn_run_call(key, XC_CSR_UPLOAD, [csr], [n, 1, nnz, 0])


def graph_run_download(key: XcnnEntryKey, h: Int, dst: Int, count: Int) raises:
    _ = xcnn_run_call(key, XC_RES_DOWNLOAD, [h, dst], [count])


def graph_run_gcn_loops(key: XcnnEntryKey, src: Int, dst: Int, w: Int, src_out: Int, dst_out: Int, w_out: Int,
                        n: Int, edges: Int, improved: Bool) raises -> Int:
    """GCN's self loops; returns the kept edge count (the loops follow)."""
    return xcnn_run_call(key, XC_GCN_LOOPS, [src, dst, w, src_out, dst_out, w_out], [n, edges, 1 if improved else 0])


def graph_run_gcn_norm(key: XcnnEntryKey, wf: Int, vals: Int, csr: Int, n: Int, nnz: Int, graph_route: Bool) raises:
    """GCN's symmetric normalization: the `_m` entry on the graph lane
    switch, the host entry otherwise."""
    if graph_route:
        _ = xcnn_run_call(key, XC_GCN_NORM_M, [wf, vals, csr], [0, n, 1, nnz, 0])
    else:
        _ = xcnn_run_call(key, XC_GCN_NORM, [wf, vals, csr], [n, 1, nnz, 0])


def graph_run_gemm_m(key: XcnnEntryKey, a: Int, b: Int, dst: Int, flags: Int, m: Int, n: Int, k: Int,
                     op: Int) raises:
    _ = xcnn_run_call(key, XC_GEMM_M, [a, b, dst], [flags, m, n, k, op])


def graph_run_spmm_m(key: XcnnEntryKey, vals: Int, h: Int, csr: Int, dst: Int, flags: Int, n: Int, cols: Int,
                     nnz: Int, mode: Int) raises:
    _ = xcnn_run_call(key, XC_SPMM_M, [vals, h, csr, dst], [flags, n, cols, nnz, mode])


def graph_run_bias_m(key: XcnnEntryKey, a: Int, bias: Int, dst: Int, flags: Int, count: Int, cols: Int) raises:
    """map2 kind 4: the bias row added to every row."""
    _ = xcnn_run_call(key, XC_MAP2_M, [a, bias, dst], [flags, 4, count, cols])


def graph_run_add_m(key: XcnnEntryKey, a: Int, b: Int, dst: Int, flags: Int, count: Int) raises:
    """map2 kind 2: element-wise sum."""
    _ = xcnn_run_call(key, XC_MAP2_M, [a, b, dst], [flags, 2, count])


def graph_run_graph_op(key: XcnnEntryKey, a: Int, b: Int, aux: Int, dst: Int, csr: Int, n: Int, cols: Int,
                       nnz: Int, kind: Int) raises:
    _ = xcnn_run_call(key, XC_GRAPH_OP, [a, b, aux, dst, csr], [n, cols, nnz, kind])


def graph_run_graph_op_m(key: XcnnEntryKey, a: Int, b: Int, aux: Int, dst: Int, csr: Int, flags: Int, n: Int,
                         cols: Int, nnz: Int, kind: Int) raises:
    _ = xcnn_run_call(key, XC_GRAPH_OP_M, [a, b, aux, dst, csr], [flags, n, cols, nnz, kind])


def graph_run_gcn_backward(key: XcnnEntryKey, addrs: List[Int], n: Int, d: Int, f: Int, nnz: Int, bias: Bool,
                           resident_grad: Bool, ones_dev: Int) raises:
    """GCNConv's backward tape (13 address words, autograd/rules_graph.mojo)."""
    _ = xcnn_run_call(key, XC_GCN_BACKWARD_T, addrs,
                      [n, d, f, nnz, 1 if bias else 0, 1 if resident_grad else 0, ones_dev])


def graph_run_sage_backward(key: XcnnEntryKey, addrs: List[Int], n: Int, d: Int, f: Int, nnz: Int, kind: Int,
                            norm: Bool, root: Bool, resident_grad: Bool, host_form: Bool, project: Bool) raises:
    """SAGEConv's backward tape: the resident form (23 address words, eight
    integers) or the host form (33 address words, ten integers;
    autograd/graph_tape.mojo `sage_cfg`)."""
    if host_form:
        _ = xcnn_run_call(key, XC_SAGE_BACKWARD_T, addrs,
                          [n, d, f, nnz, kind, 1 if norm else 0, 1 if root else 0, 0, 0, 1 if project else 0])
    else:
        _ = xcnn_run_call(key, XC_SAGE_BACKWARD_T, addrs,
                          [n, d, f, nnz, kind, 1 if norm else 0, 1 if root else 0, 1 if resident_grad else 0])
