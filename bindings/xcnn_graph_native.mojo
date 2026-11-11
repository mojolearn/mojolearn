# SPDX-License-Identifier: Apache-2.0
"""`GCNConv.forward` and `SAGEConv.forward` (python/mojolearn/_expansion_cnn.py)
as one native call each, through `xcnn_native_call` (bindings/xcnn_native.mojo):
the edge-index admission, the identity graph key and the cached graph, the
CSR views (`x_cnn_csr_build`), the per-entry value moves on the device
(`x_cnn_perm_gather`, `x_cnn_vals_t`; they were NumPy fancy indexing), GCN's
self loops and normalization, and the host or resident propagation, with
the same entries, argument words and residency bits as the Python bodies.
The fields the Python `backward` methods read (`_x`, `_g`, `_vals`, `_xs`,
`_xdev`, `_max_aux`, `_pre`, `_den`, `_nograph`, `_y`, `_ydev`) are written
under the same names; the graph is the module's `_Graph` record.

Native buffers own allocation, dtype admission and integer edge conversion;
NumPy supplies exact native ndarray containers for completed results.
"""
from std.math import sqrt
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.mode_metadata_native import _fields
from bindings.native_buffer_owner import native_buffer_view
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from bindings.neural_block_admission import neural_plain_value, neural_string_keys
from bindings.native_loaded_module import NativeModules
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.xseq_native_common import XseqStream
from bindings.xcnn_conv_native import _snapshot, _dimension, _same_shape, _relu_backward, xcnn_relu_forward
from core.cast_in import cast_in_kind
from core.xseq_marshal import xseq_fill_f32
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import (
    xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_mixed, xcnn_idn, XCNN_F_GRAPH, xcnn_is_t, xcnn_on_dev,
    xcnn_new, xcnn_wrap, xcnn_host_of, xcnn_host_f32, xcnn_f32, xcnn_addr, xcnn_dev, xcnn_dev_get,
    xcnn_dev_upload, xcnn_shape, xcnn_size, xcnn_tuple, xcnn_reshape, xcnn_host_i32, xcnn_slice_f32, xcnn_download,
    xcnn_device_io_off, xcnn_dev_fields, xcnn_reshape_t,
)
from core.xcnn_graph_impl import graph_edges_in_range, graph_bias_rows, graph_admit_edges


def _ones(count: Int, context: PythonObject) raises -> PythonObject:
    var out = xcnn_host_f32([count], context)
    xseq_fill_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(out))), count, 1)
    _ = out
    return out


def _slice(source: PythonObject, offset: Int, count: Int, integer: Bool, context: PythonObject) raises -> PythonObject:
    var out = xcnn_host_i32([count], context) if integer else xcnn_host_f32([count], context)
    if count:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(source))) + offset * 4,
               count=count * 4)
    _ = source
    _ = out
    return out


def _graph_key(n: Int, arrays: PythonObject, views: PythonObject, context: PythonObject) raises -> PythonObject:
    # Original references establish graph identity; metadata is read only from
    # admitted native views retained for this call. No input protocol lookup.
    # p3-cnn-sequence-tsa (hazard A, caller-aliased input): the key also
    # holds an owned byte snapshot of each admitted buffer (one C-level copy
    # through the buffer protocol, compared by memcmp in `_keys_equal`), so an
    # in-place edit of edge_index / edge_weight (same object, same address)
    # misses and rebuilds the graph and its resident copy. The copy and the
    # compare are far cheaper than the CSR build they guard.
    var bi = xcnn_builtins()
    var sig = Python.list()
    for i in range(len(views)):  # small-loop(one or two graph inputs)
        var a = views[i]
        var item = PythonObject(None)
        if a is not None:
            item = (Python.tuple(a[1], a[6], a[3], py_call(bi.bytes, a[0])) if i == 0
                    else Python.tuple(xcnn_tuple(xcnn_shape(a)), String(a.dtype.str), xcnn_addr(a), py_call(bi.bytes, a)))
        _ = py_call(sig.append, item)
    return Python.tuple(n, arrays, py_call(xcnn_builtins().tuple, sig))


def _graph_sig_admit(sig: PythonObject) raises:
    # The cached signature `_graph_key` built: per graph input None or an
    # exact 4-tuple (shape, dtype string, address, owned byte snapshot). The
    # snapshot is an exact `bytes` (compared by its builtin ==, no Python
    # hook), which the generic neural_plain_value scan does not admit: a
    # cache hit on the second forward raised "Python conversion hooks are
    # not supported" (0.8.66 freeze, identity lane x-cnn-gcn infer-model).
    var bi = xcnn_builtins()
    if py_type(sig) is not bi.tuple:
        raise Error("TypeError: graph cache key requires native metadata")
    for i in range(len(sig)):  # small-loop(one or two graph inputs)
        var item = sig[i]
        if item is None:
            continue
        if py_type(item) is not bi.tuple or len(item) != 4 or py_type(item[3]) is not bi.bytes:
            raise Error("TypeError: graph cache key requires native metadata")
        for j in range(3):  # small-loop(three layout words of one graph input)
            neural_plain_value(item[j], 0, PythonObject(None))


def _keys_equal(a: PythonObject, b: PythonObject) raises -> Bool:
    var bi = xcnn_builtins()
    if py_type(a) is not bi.tuple or len(a) != 3 or py_type(a[1]) is not bi.tuple:
        raise Error("TypeError: graph cache key requires native metadata")
    neural_plain_value(a[0], 0, PythonObject(None))
    _graph_sig_admit(a[2])
    if Int(py=a[0]) != Int(py=b[0]) or len(a[1]) != len(b[1]):
        return False
    for i in range(len(a[1])):  # small-loop(original graph buffer owners)
        if a[1][i] is not b[1][i]:
            return False
    return Bool(py=a[2] == b[2])


def _edge_view(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = xcnn_builtins()
    var cls = py_type(source)
    var a = PythonObject(None)
    if cls is b.tuple or cls is b.list:
        if len(source) == 2 and (py_type(source[0]) is b.list or py_type(source[0]) is b.tuple) and (py_type(source[1]) is b.list or py_type(source[1]) is b.tuple):
            if len(source[0]) == 0 and len(source[1]) == 0:
                a = xcnn_host_i32([2, 0], context)
    if a is None:
        sequential_admit_value(source, context)
        a = sequential_input(source, context)
    # Retain one real buffer export and immutable layout words for the whole
    # public operation, including signed strides and non-native byte order.
    var pin = native_buffer_view(a)
    var borrowed = AdmittedBorrowedBuffer(pin)
    var layout = borrowed.layout.value().copy()
    if len(layout.shape) != 2 or layout.shape[0] != 2:
        raise Error("mojolearn: edge_index must be (2, E)")
    var dtype = a.dtype
    var item = layout.itemsize
    var kind = cast_in_kind(Int(py=dtype.num), item, "edge_index")
    return Python.tuple(pin, xcnn_tuple(layout.shape), xcnn_tuple(layout.strides), layout.address, item, kind,
                        String(dtype.str), String(dtype.byteorder) != ">")


def _edges(ei: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    var e = Int(py=ei[1][1])
    var src = xcnn_host_i32([e], context)
    var dst = xcnn_host_i32([e], context)
    if not graph_admit_edges(Int(py=ei[3]), Int(py=ei[2][0]), Int(py=ei[2][1]), e, Int(py=ei[4]), Int(py=ei[5]),
                             Bool(py=ei[7]), n, Int(py=xcnn_addr(src)), Int(py=xcnn_addr(dst))):
        raise Error("mojolearn: edge_index refers to a node that does not exist")
    _ = ei
    return Python.tuple(src, dst)


def _csr(entries: PythonObject, rows: PythonObject, cols: PythonObject, n: Int, nnz: Int,
         context: PythonObject) raises -> PythonObject:
    var csr = xcnn_host_i32([n + 1 + 2 * nnz], context)
    var order = xcnn_host_i32([max(nnz, 1)], context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_csr_build"), xcnn_addr(rows), xcnn_addr(cols),
                      xcnn_addr(csr), xcnn_addr(order), Python.list(n, nnz))
    _ = rows
    _ = cols
    return Python.tuple(csr, _slice(order, 0, nnz, True, context))


def _graph_new(src: PythonObject, dst: PythonObject, n: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var nnz = xcnn_size(xcnn_shape(src))
    var f = _csr(entries, dst, src, n, nnz, context)
    var t = _csr(entries, src, dst, n, nnz, context)
    return Python.dict(n=n, nnz=nnz, src=src, dst=dst, csr_f=f[0], order_f=f[1], csr_t=t[0], order_t=t[1])


def _graph_admit(g: PythonObject, context: PythonObject) raises:
    neural_string_keys(g)
    neural_plain_value(g["n"], 0, PythonObject(None))
    neural_plain_value(g["nnz"], 0, PythonObject(None))
    var n = Int(py=g["n"])
    var count = Int(py=g["nnz"])
    if n < 0 or count < 0:
        raise Error("graph cache dimensions must be nonnegative")
    for name in ["src", "dst", "csr_f", "csr_t", "order_f", "order_t"]:  # small-loop(cached CSR buffer metadata)
        var array = g[PythonObject(name)]
        if py_type(array) is not context["ndarray"]:
            raise Error("TypeError: cached graph storage requires exact native arrays")
        var dtype = array.dtype
        if Int(py=dtype.num) != 5 or not Bool(py=dtype.isnative):
            raise Error("TypeError: cached graph indices require native int32 storage")
        var expected = n + 1 + 2 * count if name.startswith("csr_") else count
        if not _same_shape(xcnn_shape(array), [expected]):
            raise Error("cached graph index layout differs from its node/edge metadata")
        _ = xcnn_addr(array)


def _gather(entries: PythonObject, values: PythonObject, order: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = xcnn_size(xcnn_shape(order))
    var out = xcnn_host_f32([n], context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_perm_gather"), Python.list(xcnn_addr(values), xcnn_addr(order), xcnn_addr(out)),
                      Python.list(xcnn_size(xcnn_shape(values)), n))
    _ = values
    _ = order
    return out


def _vals_t(entries: PythonObject, g: PythonObject, vals_f: PythonObject, context: PythonObject) raises -> PythonObject:
    var nnz = Int(py=g["nnz"])
    var out = xcnn_host_f32([nnz], context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_vals_t"),
        Python.list(xcnn_addr(vals_f), xcnn_addr(g["order_f"]), xcnn_addr(g["order_t"]), xcnn_addr(out)), Python.list(nnz))
    _ = vals_f
    _ = g
    return out


# ---------------------------------------------------------------- host operations

def _gemm(entries: PythonObject, a: PythonObject, b2: PythonObject, m: Int, n: Int, k: Int, op: Int,
          context: PythonObject) raises -> PythonObject:
    """`_gemm(binding, a, b, m, n, k, op)`."""
    var out = xcnn_host_f32([m, n], context)
    var fa = xcnn_f32(a, "a", context)
    var fb = xcnn_f32(b2, "b", context)
    if xcnn_size(xcnn_shape(fa)) != m * k or xcnn_size(xcnn_shape(fb)) != k * n:
        raise Error("GEMM inputs differ from the declared graph operation dimensions")
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_gemm"), xcnn_addr(fa), xcnn_addr(fb), xcnn_addr(out), Python.list(m, n, k, op))
    _ = fb  # keep the Python object alive through the native call above
    _ = fa  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _spmm(entries: PythonObject, g: PythonObject, vals: PythonObject, h: PythonObject, mode: Int,
          context: PythonObject, transposed: Bool = False) raises -> PythonObject:
    """`g.spmm(binding, vals, h, mode)` (the forward view)."""
    var hh = xcnn_f32(h, "h", context)
    var out = xcnn_host_f32(xcnn_shape(hh), context)
    var v = xcnn_host_f32([1], context) if vals is None or len(vals) == 0 else xcnn_f32(vals, "values", context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_spmm"), xcnn_addr(v), xcnn_addr(hh), xcnn_addr(out), xcnn_addr(g["csr_t"] if transposed else g["csr_f"]),
              Python.list(Int(py=g["n"]), Int(py=hh.shape[1]), Int(py=g["nnz"]), mode))
    _ = v  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = hh  # keep the Python object alive through the native call above
    return out


def _add(entries: PythonObject, a: PythonObject, b2: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_add(binding, a, b)`."""
    var fa = xcnn_f32(a, "a", context)
    var fb = xcnn_f32(b2, "b", context)
    var out = xcnn_host_f32(xcnn_shape(fa), context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_add"), xcnn_addr(fa), xcnn_addr(fb), xcnn_addr(out), Python.list(Int(py=fa.size)))
    _ = out  # keep the Python object alive through the native call above
    _ = fb  # keep the Python object alive through the native call above
    _ = fa  # keep the Python object alive through the native call above
    return out


def _bias_rows(bias: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """`np.ascontiguousarray(np.broadcast_to(bias, (rows, cols)))`."""
    var fb = xcnn_f32(bias, "b", context)
    var out = xcnn_host_f32([rows, cols], context)
    graph_bias_rows(Int(py=xcnn_addr(fb)), Int(py=xcnn_addr(out)), rows, cols)
    _ = out  # keep the Python object alive through the native call above
    _ = fb  # keep the Python object alive through the native call above
    return out


def _linear_forward(lin: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Linear.forward(x)`."""
    var fields = _linear_fields(lin, context)
    var entries = xcnn_entries(fields, context)
    var fx = _snapshot(x, "x", context)
    var shape = xcnn_shape(fx)
    if len(shape) != 2 or shape[1] != Int(py=fields["in_features"]):
        raise Error("linear input shape differs from in_features")
    var values = _linear_weights(fields, context)
    var rows = shape[0]
    var out = xcnn_host_f32([rows, Int(py=fields["out_features"])], context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_linear_forward"), xcnn_addr(fx), xcnn_addr(values[0]), xcnn_addr(values[1]),
              xcnn_addr(out), Python.list(rows, fields["in_features"], fields["out_features"]))
    _ = fx  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    py_setitem(fields, "_x", fx)
    _ = values
    return out


def _relu_forward(relu: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ReLU.forward(x)` of a host array."""
    if py_type(relu) is not context["xcnn_relu_class"]:
        raise Error("TypeError: graph projection requires a native ReLU layer")
    var fields = _fields(relu)
    var entries = xcnn_entries(fields, context)
    var fx = _snapshot(x, "x", context)
    var out = xcnn_host_f32(xcnn_shape(fx), context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_relu_forward"), xcnn_addr(fx), xcnn_addr(out), Python.list(Int(py=fx.size)))
    _ = out  # keep the Python object alive through the native call above
    _ = fx  # keep the Python object alive through the native call above
    py_setitem(fields, "_x", fx)
    return out


# ---------------------------------------------------------------- resident graph

def _adopt(dev: PythonObject, entries: PythonObject, name: String, h: PythonObject, n: Int, context: PythonObject) raises:
    """`_Dev.adopt(name, h, n)`."""
    var table = xcnn_dev_fields(dev, context)["h"]
    var cur = py_call(table.pop, name, PythonObject(None))
    if cur is not None:
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_res_free"), cur[0])
    py_setitem(table, name, Python.tuple(Int(py=py_call(xcnn_builtins().int, h)), max(n, 1)))


def _graph_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, dev: PythonObject, g: PythonObject,
               vals_f: PythonObject, vals_t: PythonObject, context: PythonObject) raises:
    """`_graph_dev(layer, b, dev, g, vals_f, vals_t)`: the resident CSR views
    and values of graph `g`, uploaded once per graph."""
    var b = xcnn_builtins()
    neural_plain_value(py_get(fields, "_gdev_for"), 0, PythonObject(None))
    var tag = Python.tuple(py_call(b.id, g), py_call(b.id, dev))
    if Bool(py=py_get(fields, "_gdev_for") == tag) and py_get(fields, "_gdev_g") is g:
        return
    var n = Int(py=g["n"])
    var nnz = Int(py=g["nnz"])
    var prm = Python.list(n, 1, nnz, 0)
    _adopt(dev, entries, "csr_f", py_call_typed(xcnn_entry(entries, "x_cnn_csr_upload"), xcnn_addr(g["csr_f"]), prm), Int(py=g["csr_f"].size), context)
    _adopt(dev, entries, "csr_t", py_call_typed(xcnn_entry(entries, "x_cnn_csr_upload"), xcnn_addr(g["csr_t"]), prm), Int(py=g["csr_t"].size), context)
    var vf = xcnn_f32(vals_f, "values", context)
    var vt = _vals_t(entries, g, vf, context) if vals_t is None else xcnn_f32(vals_t, "values", context)
    _ = xcnn_dev_upload(layer, entries, "vals_f", vf if Int(py=vf.size) else xcnn_host_f32([1], context), context)
    _ = xcnn_dev_upload(layer, entries, "vals_t", vt if Int(py=vt.size) else xcnn_host_f32([1], context), context)
    py_setitem(fields, "_gdev_for", tag)
    py_setitem(fields, "_gdev_g", g)


def _dev_h(dev: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return xcnn_dev_fields(dev, context)["h"][PythonObject(name)][0]


def _dev_pair(dev: PythonObject, first: String, second: String, context: PythonObject) raises -> PythonObject:
    """(`_dev_h(dev, first)`, `_dev_h(dev, second)`) with the owner and its
    handle table admitted once for the launch, not once per handle."""
    var table = xcnn_dev_fields(dev, context)["h"]
    return Python.tuple(table[PythonObject(first)][0], table[PythonObject(second)][0])


def _download(entries: PythonObject, h: PythonObject, out_buf: PythonObject) raises:
    """`dev.download(h, out)`."""
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_res_download"), h, xcnn_addr(out_buf), out_buf.size)


# ---------------------------------------------------------------- GCNConv

def _gcn_graph(fields: PythonObject, entries: PythonObject, n: Int, edge_index: PythonObject, edge_weight: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`GCNConv._graph(n, edge_index, edge_weight)`: (g, vals)."""
    var edges = _edges(edge_index, n, context)
    var src = edges[0]
    var dst = edges[1]
    var w = PythonObject(None)
    if edge_weight is None:
        w = _ones(len(src), context)
    else:
        w = xcnn_slice_f32(edge_weight, 0, xcnn_size(xcnn_shape(edge_weight)), context)
    if len(w) != len(src):
        raise Error("ValueError: mojolearn: edge_weight must have one entry per edge")
    if Bool(py=fields["normalize"]) and Bool(py=fields["add_self_loops"]):
        if not py_contains(entries["symbols"], "x_cnn_gcn_loops"):
            raise Error("RuntimeError: mojolearn: this x_cnn binding has no x_cnn_gcn_loops (an older binary); rebuild it")
        var s32 = src
        var d32 = dst
        var w32 = w
        var t = len(s32) + n
        var so = xcnn_host_i32([t], context)
        var dso = xcnn_host_i32([t], context)
        var wo = xcnn_host_f32([t], context)
        var k = Int(py=py_call(xcnn_builtins().int, py_call_typed(xcnn_entry(entries, "x_cnn_gcn_loops"), Python.list(xcnn_addr(s32), xcnn_addr(d32), xcnn_addr(w32), xcnn_addr(so), xcnn_addr(dso), xcnn_addr(wo)),
            Python.list(n, len(s32), 1 if Bool(py=fields["improved"]) else 0))))
        _ = wo  # keep the Python object alive through the native call above
        _ = dso  # keep the Python object alive through the native call above
        _ = so  # keep the Python object alive through the native call above
        _ = w32  # keep the Python object alive through the native call above
        _ = d32  # keep the Python object alive through the native call above
        _ = s32  # keep the Python object alive through the native call above
        _ = wo  # keep the Python object alive through the native call above
        _ = dso  # keep the Python object alive through the native call above
        _ = so  # keep the Python object alive through the native call above
        _ = w32  # keep the Python object alive through the native call above
        _ = d32  # keep the Python object alive through the native call above
        _ = s32  # keep the Python object alive through the native call above
        src = _slice(so, 0, k + n, True, context)
        dst = _slice(dso, 0, k + n, True, context)
        w = _slice(wo, 0, k + n, False, context)
    var g = _graph_new(src, dst, n, entries, context)
    var wf = _gather(entries, w, g["order_f"], context)
    var vals = wf
    if Bool(py=fields["normalize"]):
        var nnz = Int(py=g["nnz"])
        vals = xcnn_host_f32([nnz], context)
        if xcnn_idn(entries, XCNN_F_GRAPH):
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_gcn_norm_m"), Python.list(xcnn_addr(wf), xcnn_addr(vals), xcnn_addr(g["csr_f"])),
                      PythonObject(0), Python.list(n, 1, nnz, 0))
            _ = vals  # keep the Python object alive through the native call above
            _ = wf  # keep the Python object alive through the native call above
        else:
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_gcn_norm"), xcnn_addr(wf), xcnn_addr(vals), xcnn_addr(g["csr_f"]), Python.list(n, 1, nnz, 0))
            _ = vals  # keep the Python object alive through the native call above
            _ = wf  # keep the Python object alive through the native call above
    return Python.tuple(g, vals)


def _gcn_forward_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, x: PythonObject, g: PythonObject,
                     vals: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = Int(py=x.shape[0])
    var f_out = Int(py=fields["out_channels"])
    var d_in = Int(py=fields["in_channels"])
    if len(xcnn_shape(x)) != 2 or Int(py=x.shape[1]) != d_in:
        raise Error("ValueError: mojolearn: x must be (n, " + String(d_in) + ")")
    var dev = xcnn_dev(layer, entries, context)
    _graph_dev(layer, fields, entries, dev, g, vals, PythonObject(None), context)
    var w = _snapshot(fields["weight_"], "weight_", context)
    if not _same_shape(xcnn_shape(w), [f_out, d_in]):
        raise Error("GCN weight shape differs from configured channels")
    var xt = xcnn_is_t(x, context)
    var xh = x.h if xt else xcnn_dev_upload(layer, entries, "x", x, context)
    var hh = xcnn_dev_get(layer, entries, "h", n * f_out, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_gemm_m"), Python.list(xh, xcnn_addr(w), hh), PythonObject(0b101), Python.list(n, f_out, d_in, 1))
    _ = w  # keep the Python object alive through the native call above
    var bias = Bool(py=fields["bias"])
    var out = xcnn_new(entries, [n, f_out], False, context) if xt else xcnn_host_f32([n, f_out], context)
    var oh = out.h if (xt and not bias) else xcnn_dev_get(layer, entries, "o", n * f_out, context)
    var _hv1 = _dev_pair(dev, "vals_f", "csr_f", context)   # one owner admission for both handles
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_spmm_m"), Python.list(_hv1[0], hh, _hv1[1], oh), PythonObject(0b1111),
              Python.list(n, f_out, Int(py=g["nnz"]), 0))
    if bias:
        var bv = _snapshot(fields["bias_"], "bias_", context)
        if Int(py=bv.size) != f_out:
            raise Error("ValueError: mojolearn: bias_ has the wrong size")
        if xt:
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(oh, xcnn_addr(bv), out.h), PythonObject(0b101), Python.list(4, n * f_out, f_out))
            _ = bv  # keep the Python object alive through the native call above
        else:
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(oh, xcnn_addr(bv), xcnn_addr(out)), PythonObject(0b001),
                      Python.list(4, n * f_out, f_out))
            _ = bv  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
    elif not xt:
        _download(entries, oh, out)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    py_setitem(fields, "_vals", vals)
    py_setitem(fields, "_xdev", Python.tuple(dev, xh))
    return out


def xcnn_gcn_forward(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _graph_layer_fields(layer, context)
    var entries = xcnn_entries(fields, context)
    var x = args[0]
    var edge_index = _edge_view(args[1], context)
    var edge_weight = PythonObject(None) if args[2] is None else _snapshot(args[2], "edge_weight", context)
    if xcnn_is_t(x, context):
        if not xcnn_on_dev(entries, x, context):
            var host = xcnn_host_of(x, context)
            return xcnn_wrap(entries, xcnn_gcn_forward(layer, Python.tuple(host, args[1], args[2]), context), context)
    else:
        x = _snapshot(x, "x", context)
    var shape = xcnn_shape(x)
    if len(shape) != 2 or shape[1] != Int(py=fields["in_channels"]):
        raise Error("mojolearn: graph x must be (n, in_channels)")
    if xcnn_is_t(x, context):
        x = xcnn_reshape_t(x, shape, context)
    var n = shape[0]
    var key = Python.tuple(_graph_key(n, Python.tuple(args[1], args[2]), Python.tuple(edge_index, edge_weight), context), fields["improved"],
                           fields["add_self_loops"], fields["normalize"])
    var hit = py_get(fields, "_gcache")
    if hit is not None and (py_type(hit) is not xcnn_builtins().tuple or len(hit) != 2):
        raise Error("TypeError: graph cache requires a native pair")
    var g = PythonObject(None)
    var vals = PythonObject(None)
    if hit is not None:
        if py_type(hit[0]) is not xcnn_builtins().tuple or len(hit[0]) != 4 or py_type(hit[1]) is not xcnn_builtins().tuple or len(hit[1]) != 2:
            raise Error("TypeError: GCN cache requires native graph metadata")
        for i in range(1, 4):  # small-loop(three cached graph normalization flags)
            neural_plain_value(hit[0][i], 0, PythonObject(None))
    if (hit is not None and _keys_equal(hit[0][0], key[0]) and Bool(py=hit[0][1] == key[1])
            and Bool(py=hit[0][2] == key[2]) and Bool(py=hit[0][3] == key[3])):
        g = hit[1][0]
        vals = hit[1][1]
    else:
        var gv = _gcn_graph(fields, entries, n, edge_index, edge_weight, context)
        g = gv[0]
        vals = gv[1]
        py_setitem(fields, "_gcache", Python.tuple(key, Python.tuple(g, vals)))
    _graph_admit(g, context)
    if xcnn_mixed(entries):
        return _gcn_forward_dev(layer, fields, entries, x, g, vals, context)
    py_setitem(fields, "_xdev", PythonObject(None))
    var f_out = Int(py=fields["out_channels"])
    var d_in = Int(py=fields["in_channels"])
    var h = _gemm(entries, x, fields["weight_"], n, f_out, d_in, 1, context)
    var out = _spmm(entries, g, vals, h, 0, context)
    if Bool(py=fields["bias"]):
        out = _add(entries, out, _bias_rows(fields["bias_"], Int(py=out.shape[0]), Int(py=out.shape[1]), context), context)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    py_setitem(fields, "_vals", vals)
    return out


# ---------------------------------------------------------------- SAGEConv

def _sage_res_form(fields: PythonObject, entries: PythonObject) raises -> Bool:
    """`SAGEConv._res_form(b)`."""
    if not xcnn_mixed(entries) or Bool(py=fields["project"]):
        return False
    if xcnn_idn(entries, XCNN_F_GRAPH):
        return True
    var aggr = String(py=fields["aggr"])
    return not Bool(py=fields["normalize"]) and (aggr == "mean" or aggr == "sum" or aggr == "add")


def _sage_forward_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, x: PythonObject, g: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    var b = xcnn_builtins()
    var n = Int(py=x.shape[0])
    var d_in = Int(py=fields["in_channels"])
    var f_out = Int(py=fields["out_channels"])
    if len(xcnn_shape(x)) != 2 or Int(py=x.shape[1]) != d_in:
        raise Error("ValueError: mojolearn: x must be (n, " + String(d_in) + ")")
    var dev = xcnn_dev(layer, entries, context)
    var aggr = String(py=fields["aggr"])
    var mean = aggr == "mean"
    neural_plain_value(py_get(fields, "_gdev_for"), 0, PythonObject(None))
    var tag = Python.tuple(py_call(b.id, g), py_call(b.id, dev))
    if py_get(fields, "_gdev_g") is not g or not Bool(py=py_get(fields, "_gdev_for") == tag):
        var ones = _ones(Int(py=g["nnz"]), context)
        _graph_dev(layer, fields, entries, dev, g, ones, ones, context)
    var xt = xcnn_is_t(x, context)
    var xh = x.h if xt else xcnn_dev_upload(layer, entries, "x", x, context)
    var aggh = xcnn_dev_get(layer, entries, "agg", n * d_in, context)
    var nnz = Int(py=g["nnz"])
    if aggr == "max":
        var auxh = xcnn_dev_get(layer, entries, "maxaux", 2 * n * d_in, context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_graph_op_m"), Python.list(xh, xh, auxh, aggh, _dev_h(dev, "csr_f", context)), PythonObject(0b11111),
                  Python.list(n, d_in, nnz, 0))
    else:
        var _hv2 = _dev_pair(dev, "vals_f", "csr_f", context)   # one owner admission for both handles
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_spmm_m"), Python.list(_hv2[0], xh, _hv2[1], aggh),
                  PythonObject(0b1111), Python.list(n, d_in, nnz, 1 if mean else 0))
    var lin = fields["lin_l"]
    var lin_fields = _linear_fields(lin, context)
    var lin_values = _linear_weights(lin_fields, context)
    var norm = Bool(py=fields["normalize"])
    var root = Bool(py=fields["root_weight"])
    var out = xcnn_new(entries, [n, f_out], False, context) if xt else xcnn_host_f32([n, f_out], context)
    var preh = xcnn_dev_get(layer, entries, "pre", n * f_out, context) if norm else PythonObject(None)
    var olh = PythonObject(None)
    if norm and not root:
        olh = preh
    else:
        olh = out.h if (xt and not root) else xcnn_dev_get(layer, entries, "ol", n * f_out, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_linear_forward_m"), Python.list(aggh, xcnn_addr(lin_values[0]), xcnn_addr(lin_values[1]), olh), PythonObject(0b1001),
              Python.list(n, d_in, f_out))
    if root:
        var wr = _snapshot(fields["weight_r_"], "weight_r_", context)
        if not _same_shape(xcnn_shape(wr), [f_out, d_in]):
            raise Error("SAGE root weight shape differs from configured channels")
        var rh = xcnn_dev_get(layer, entries, "r", n * f_out, context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_gemm_m"), Python.list(xh, xcnn_addr(wr), rh), PythonObject(0b101), Python.list(n, f_out, d_in, 1))
        _ = wr  # keep the Python object alive through the native call above
        if norm:
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(olh, rh, preh), PythonObject(0b111), Python.list(2, n * f_out))
        elif xt:
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(olh, rh, out.h), PythonObject(0b111), Python.list(2, n * f_out))
        else:
            _ = py_call_typed(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(olh, rh, xcnn_addr(out)), PythonObject(0b011), Python.list(2, n * f_out))
            _ = out  # keep the Python object alive through the native call above
    elif not xt and not norm:
        _download(entries, olh, out)
    py_setitem(fields, "_ydev", PythonObject(None))
    if norm:
        var denh = xcnn_dev_get(layer, entries, "den", n, context)
        var yh = out.h if xt else xcnn_dev_get(layer, entries, "y", n * f_out, context)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_graph_op_m"), Python.list(preh, preh, denh, yh, PythonObject(0)), PythonObject(0b01111),
                  Python.list(n, f_out, 0, 2))
        if not xt:
            _download(entries, yh, out)
        py_setitem(fields, "_ydev", Python.tuple(yh, denh, out if xt else PythonObject(None)))
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    py_setitem(fields, "_xs", x)
    py_setitem(fields, "_xdev", Python.tuple(dev, xh, aggh))
    _ = lin_values
    return out


def xcnn_sage_forward(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _graph_layer_fields(layer, context)
    var entries = xcnn_entries(fields, context)
    var x = args[0]
    var edge_index = _edge_view(args[1], context)
    if xcnn_is_t(x, context):
        if not (_sage_res_form(fields, entries) and xcnn_on_dev(entries, x, context)):
            var host = xcnn_host_of(x, context)
            return xcnn_wrap(entries, xcnn_sage_forward(layer, Python.tuple(host, args[1]), context), context)
    else:
        x = _snapshot(x, "x", context)
    var shape = xcnn_shape(x)
    if len(shape) != 2 or shape[1] != Int(py=fields["in_channels"]):
        raise Error("mojolearn: graph x must be (n, in_channels)")
    if xcnn_is_t(x, context):
        x = xcnn_reshape_t(x, shape, context)
    var n = shape[0]
    var key = _graph_key(n, Python.tuple(args[1]), Python.tuple(edge_index), context)
    var hit = py_get(fields, "_gcache")
    if hit is not None and (py_type(hit) is not xcnn_builtins().tuple or len(hit) != 2):
        raise Error("TypeError: graph cache requires a native pair")
    var g = PythonObject(None)
    if hit is not None and _keys_equal(hit[0], key):
        g = hit[1]
    else:
        var edges = _edges(edge_index, n, context)
        g = _graph_new(edges[0], edges[1], n, entries, context)
        py_setitem(fields, "_gcache", Python.tuple(key, g))
    _graph_admit(g, context)
    if _sage_res_form(fields, entries):
        return _sage_forward_dev(layer, fields, entries, x, g, context)
    py_setitem(fields, "_xdev", PythonObject(None))
    var xs = x
    if Bool(py=fields["project"]):
        xs = _relu_forward(fields["_relu_p"], _linear_forward(fields["lin"], x, context), context)
    py_setitem(fields, "_xs", xs)
    var aggr = String(py=fields["aggr"])
    var agg = PythonObject(None)
    var nnz = Int(py=g["nnz"])
    if aggr == "max":
        agg = xcnn_host_f32(xcnn_shape(xs), context)
        var aux = xcnn_host_f32([2 * xcnn_size(xcnn_shape(xs))], context)
        py_setitem(fields, "_max_aux", aux)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_graph_op"), xcnn_addr(xs), xcnn_addr(xs), xcnn_addr(aux), xcnn_addr(agg), xcnn_addr(g["csr_f"]),
                  Python.list(n, Int(py=xs.shape[1]), nnz, 0))
        _ = aux  # keep the Python object alive through the native call above
        _ = agg  # keep the Python object alive through the native call above
        _ = xs  # keep the Python object alive through the native call above
    else:
        agg = _spmm(entries, g, _ones(nnz, context), xs, 1 if aggr == "mean" else 0, context)
    var out = _linear_forward(fields["lin_l"], agg, context)
    if Bool(py=fields["root_weight"]):
        out = _add(entries, out, _gemm(entries, x, fields["weight_r_"], n, Int(py=fields["out_channels"]),
                                       Int(py=fields["in_channels"]), 1, context), context)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    if Bool(py=fields["normalize"]):
        py_setitem(fields, "_pre", out)
        var y = xcnn_host_f32(xcnn_shape(out), context)
        var den = xcnn_host_f32([n], context)
        var nograph = xcnn_host_i32([n + 1], context)
        py_setitem(fields, "_den", den)
        py_setitem(fields, "_nograph", nograph)
        _ = py_call_typed(xcnn_entry(entries, "x_cnn_graph_op"), xcnn_addr(out), xcnn_addr(out), xcnn_addr(den), xcnn_addr(y), xcnn_addr(nograph),
                  Python.list(n, Int(py=out.shape[1]), 0, 2))
        _ = nograph  # keep the Python object alive through the native call above
        _ = den  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        py_setitem(fields, "_y", y)
        out = y
    return out
# ---------------------------------------------------------------- lifecycle and backward owners

def _graph_layer_fields(layer: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(layer)
    # One admission dictionary and module view for the whole field list.
    var mods = NativeModules()
    var admission = Python.dict(admission_context=context)
    for name in ["in_channels", "out_channels", "improved", "add_self_loops", "normalize", "bias", "numeric_mode", "project", "root_weight", "aggr"]:  # small-loop(graph layer configuration)
        neural_plain_value(mods, py_get(fields, PythonObject(name)), 0, admission)
    return fields


def _linear_fields(layer: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(layer) is not context["xcnn_linear_class"]:
        raise Error("TypeError: graph projection requires a native linear layer")
    var fields = _fields(layer)
    var mods = NativeModules()
    var admission = Python.dict(admission_context=context)
    for name in ["in_features", "out_features", "numeric_mode"]:  # small-loop(linear configuration)
        neural_plain_value(mods, fields[PythonObject(name)], 0, admission)
    return fields


def _linear_init(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, Python.dict(admission_context=context))
    var fields = _fields(layer)
    var d = _dimension(args[0], "in_features", context)
    var f = _dimension(args[1], "out_features", context)
    if d <= 0 or f <= 0:
        raise Error("linear feature dimensions must be positive")
    py_setitem(fields, "in_features", d)
    py_setitem(fields, "out_features", f)
    py_setitem(fields, "numeric_mode", args[3])
    var seed = PythonObject(None) if args[2] is None else py_call(xcnn_builtins().int, args[2])
    var stream = XseqStream(seed, context)
    var bound = 1.0 / sqrt(Float64(d))
    var w = xcnn_host_f32([f, d], context)
    var bias = xcnn_host_f32([f], context)
    stream.fill_uniform(context, xcnn_addr(w), f * d, -bound, bound)
    stream.fill_uniform(context, xcnn_addr(bias), f, -bound, bound)
    py_setitem(fields, "weight_", w)
    py_setitem(fields, "bias_", bias)
    return PythonObject(None)


def _linear_new(d: Int, f: Int, seed: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var layer = py_call(xcnn_builtins().object.__new__, context["xcnn_linear_class"])
    _ = _linear_init(layer, Python.tuple(d, f, seed, mode), context)
    return layer


def _linear_weights(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var d = Int(py=fields["in_features"])
    var f = Int(py=fields["out_features"])
    var w = _snapshot(fields["weight_"], "weight_", context)
    var bias = _snapshot(fields["bias_"], "bias_", context)
    if not _same_shape(xcnn_shape(w), [f, d]) or not _same_shape(xcnn_shape(bias), [f]):
        raise Error("linear weights differ from configured dimensions")
    return Python.tuple(w, bias)


def _linear_backward(lin: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _linear_fields(lin, context)
    if py_get(fields, "_x") is None:
        raise Error("RuntimeError: linear backward requires a prior forward")
    var x = _snapshot(fields["_x"], "x", context)
    var g = _snapshot(gradient, "grad_out", context)
    var d = Int(py=fields["in_features"])
    var f = Int(py=fields["out_features"])
    var shape = xcnn_shape(x)
    if len(shape) != 2 or shape[1] != d or not _same_shape(xcnn_shape(g), [shape[0], f]):
        raise Error("linear gradient shape differs from forward output")
    var values = _linear_weights(fields, context)
    var w = values[0]
    var dx = xcnn_host_f32(shape, context)
    var dw = xcnn_host_f32([f, d], context)
    var db = xcnn_host_f32([f], context)
    _ = py_call_typed(xcnn_entry(xcnn_entries(fields, context), "x_cnn_linear_backward"), xcnn_addr(x), xcnn_addr(w),
                      xcnn_addr(g), xcnn_addr(dx), xcnn_addr(dw), xcnn_addr(db), Python.list(shape[0], d, f))
    py_setitem(fields, "grad_weight_", dw)
    py_setitem(fields, "grad_bias_", db)
    _ = x
    _ = w
    _ = g
    return dx


def _graph_init(layer: PythonObject, args: PythonObject, sage: Bool, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, Python.dict(admission_context=context))
    var fields = _fields(layer)
    var d = _dimension(args[0], "in_channels", context)
    var f = _dimension(args[1], "out_channels", context)
    if d <= 0 or f <= 0:
        raise Error("graph layer channels must be positive")
    var b = xcnn_builtins()
    var mode = args[8] if sage else args[7]
    var seed0 = args[7] if sage else args[6]
    var seed = PythonObject(None) if seed0 is None else py_call(b.int, seed0)
    var stream = XseqStream(seed, context)
    py_setitem(fields, "in_channels", d)
    py_setitem(fields, "out_channels", f)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "bias", Bool(py=py_call(b.bool, args[5])))
    if not sage:
        py_setitem(fields, "improved", Bool(py=py_call(b.bool, args[2])))
        py_setitem(fields, "add_self_loops", Bool(py=py_call(b.bool, args[3])))
        py_setitem(fields, "normalize", Bool(py=py_call(b.bool, args[4])))
        var bound = sqrt(6.0 / Float64(d + f))
        var w = xcnn_host_f32([f, d], context)
        stream.fill_uniform(context, xcnn_addr(w), d * f, -bound, bound)
        py_setitem(fields, "weight_", w)
        py_setitem(fields, "bias_", xcnn_host_f32([f], context))
        return PythonObject(None)
    if py_type(args[2]) is not b.str:
        raise Error("TypeError: SAGE aggregation requires a native string")
    var aggr = String(args[2])
    if aggr != "mean" and aggr != "sum" and aggr != "add" and aggr != "max":
        raise Error("NotImplementedError: SAGE aggregation must be mean, sum, add or max")
    py_setitem(fields, "aggr", args[2])
    py_setitem(fields, "normalize", Bool(py=py_call(b.bool, args[3])))
    py_setitem(fields, "root_weight", Bool(py=py_call(b.bool, args[4])))
    var lin_l = _linear_new(d, f, PythonObject(Int(stream.child_seed())), mode, context)
    if not Bool(py=fields["bias"]):
        py_setitem(_fields(lin_l), "bias_", xcnn_host_f32([f], context))
    py_setitem(fields, "lin_l", lin_l)
    var wr = xcnn_host_f32([f, d], context)
    var bound = 1.0 / sqrt(Float64(d))
    stream.fill_uniform(context, xcnn_addr(wr), d * f, -bound, bound)
    py_setitem(fields, "weight_r_", wr)
    var project = Bool(py=py_call(b.bool, args[6]))
    py_setitem(fields, "project", project)
    py_setitem(fields, "lin", PythonObject(None))
    if project:
        py_setitem(fields, "lin", _linear_new(d, d, PythonObject(Int(stream.child_seed())), mode, context))
        var relu = py_call(b.object.__new__, context["xcnn_relu_class"])
        py_setitem(_fields(relu), "numeric_mode", mode)
        py_setitem(fields, "_relu_p", relu)
    return PythonObject(None)


def _graph_gradient(layer: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _graph_layer_fields(layer, context)
    if py_get(fields, "_x") is None or py_get(fields, "_g") is None:
        raise Error("RuntimeError: graph backward requires a prior forward")
    _graph_admit(fields["_g"], context)
    var g = gradient if xcnn_is_t(gradient, context) else _snapshot(gradient, "grad_out", context)
    var xs = xcnn_shape(fields["_x"])
    if len(xs) != 2 or not _same_shape(xcnn_shape(g), [xs[0], Int(py=fields["out_channels"])]):
        raise Error("mojolearn: graph grad_out shape differs from forward output")
    return g


def _resident_saved(fields: PythonObject, entries: PythonObject, context: PythonObject) raises -> Bool:
    var saved = py_get(fields, "_xdev")
    if saved is None:
        return False
    var b = xcnn_builtins()
    if py_type(saved) is not b.tuple or len(saved) < 2 or py_type(saved[0]) is not context["xcnn_dev_class"]:
        raise Error("TypeError: graph resident state requires native storage metadata")
    for i in range(1, len(saved)):  # small-loop(saved native buffer handles)
        if py_type(saved[i]) is not b.int:
            raise Error("TypeError: graph resident state requires native handle words")
    return _fields(saved[0])["b"] is entries["module"]


def _resident_ones(layer: PythonObject, entries: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    # Preserve the existing resident bias-gradient operand: upload only when
    # its allocation size changes, rather than adding an upload per backward.
    var n = max(count, 1)
    var dev = xcnn_dev(layer, entries, context)
    var cached = py_get(xcnn_dev_fields(dev, context)["h"], "ones")
    if cached is not None and Int(py=cached[1]) == n:
        return cached[0]
    return xcnn_dev_upload(layer, entries, "ones", _ones(n, context), context)


# TOMBSTONE: the hand chain of GCNConv's resident backward (x_cnn_gemm_m bias, x_cnn_spmm_m, x_cnn_gemm_m dW, x_cnn_gemm_m dX:
# four entry crossings, removed: superseded by the tape) deleted 2026-11-09 by lane/x9-moe-gcn-rules; code recoverable at origin/integ/pre-0867-w2 f59d913b0b.
# Restore: git apply experiments/removed/XCNN_GRAPH_HAND_BACKWARD.patch
def _gcn_backward_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, grad: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """The resident backward as one native call: the tape over the forward's
    resident buffers (autograd/rules_graph.mojo `gcn_backward_autograd`),
    the same entries' bodies on the same arguments."""
    var saved = fields["_xdev"]
    var dev = saved[0]
    var xh = saved[1]
    var graph = fields["_g"]
    var n = xcnn_shape(grad)[0]
    var d = Int(py=fields["in_channels"])
    var f = Int(py=fields["out_channels"])
    var gt = xcnn_is_t(grad, context)
    var gh = grad.h if gt else xcnn_dev_upload(layer, entries, "G", grad, context)
    var bias = Bool(py=fields["bias"])
    var gb = xcnn_host_f32([f], context)
    var ones = PythonObject(None)
    var ones_addr = PythonObject(0)
    var ones_dev = 0
    var bv = PythonObject(None)
    var b_addr = PythonObject(0)
    var gb_addr = PythonObject(0)
    if bias:
        var resident = not xcnn_device_io_off(entries, context)
        if resident:
            ones_addr = _resident_ones(layer, entries, n, context)
            ones_dev = 1
        else:
            ones = _ones(n, context)
            ones_addr = xcnn_addr(ones)
        bv = _snapshot(fields["bias_"], "bias_", context)
        if Int(py=bv.size) != f:
            raise Error("ValueError: mojolearn: bias_ has the wrong size")
        b_addr = xcnn_addr(bv)
        gb_addr = xcnn_addr(gb)
    var w = _snapshot(fields["weight_"], "weight_", context)
    if not _same_shape(xcnn_shape(w), [f, d]):
        raise Error("GCN weight shape differs from configured channels")
    var gw = xcnn_host_f32([f, d], context)
    # the forward's h and output arrays (the same sizes: no reallocation;
    # recorded as the tape's values, never read)
    var hh = xcnn_dev_get(layer, entries, "h", n * f, context)
    var oh = xcnn_dev_get(layer, entries, "o", n * f, context)
    var dh = xcnn_dev_get(layer, entries, "dh", n * f, context)
    var dx = xcnn_new(entries, [n, d], False, context) if gt else xcnn_host_f32([n, d], context)
    var _hv = _dev_pair(dev, "vals_t", "csr_t", context)   # one owner admission for both handles
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_gcn_backward_t"),
                      Python.list(gh, xh, hh, oh, xcnn_addr(w), xcnn_addr(gw), dx.h if gt else xcnn_addr(dx), dh,
                                  _hv[0], _hv[1], b_addr, gb_addr, ones_addr),
                      Python.list(n, d, f, Int(py=graph["nnz"]), 1 if bias else 0, 1 if gt else 0, ones_dev))
    py_setitem(fields, "grad_bias_", gb)
    py_setitem(fields, "grad_weight_", gw)
    _ = ones
    _ = bv
    _ = w
    _ = grad
    return dx


def _gcn_backward(layer: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _graph_layer_fields(layer, context)
    var entries = xcnn_entries(fields, context)
    var g = _graph_gradient(layer, gradient, context)
    if xcnn_is_t(g, context):
        if _resident_saved(fields, entries, context) and xcnn_on_dev(entries, g, context):
            return _gcn_backward_dev(layer, fields, entries, g, context)
        return xcnn_wrap(entries, _gcn_backward(layer, xcnn_host_of(g, context), context), context)
    if _resident_saved(fields, entries, context):
        return _gcn_backward_dev(layer, fields, entries, g, context)
    return _gcn_backward_host(fields, entries, g, context)


# TOMBSTONE: the host-form hand chain of GCNConv's backward (x_cnn_gemm bias, x_cnn_spmm transposed, x_cnn_gemm dW,
# x_cnn_gemm dX: four entry crossings, removed: superseded by the tape) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/XCNN_GRAPH_HOST_HAND_BACKWARD.patch
def _gcn_backward_host(fields: PythonObject, entries: PythonObject, g: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """The host-form backward (the CPU binding, or a forward kept on the
    host) as one native call after the transposed values: the tape over
    host arrays (autograd/rules_graph.mojo `gcn_backward_autograd` on the
    GPU binding, autograd/rules_graph_host.mojo on the CPU binding)."""
    var n = xcnn_shape(g)[0]
    var d = Int(py=fields["in_channels"])
    var f = Int(py=fields["out_channels"])
    var graph = fields["_g"]
    var bias = Bool(py=fields["bias"])
    var vt = _vals_t(entries, graph, fields["_vals"], context)
    var x = xcnn_f32(fields["_x"], "x", context)
    var w = _snapshot(fields["weight_"], "weight_", context)
    if not _same_shape(xcnn_shape(w), [f, d]):
        raise Error("GCN weight shape differs from configured channels")
    var gw = xcnn_host_f32([f, d], context)
    var gb = xcnn_host_f32([f], context)
    var dx = xcnn_host_f32([n, d], context)
    var dh = xcnn_host_f32([n, f], context)
    # the forward's h and output (recorded as the tape's values, never read)
    var hv = xcnn_host_f32([n, f], context)
    var ones = PythonObject(None)
    var bv = PythonObject(None)
    var b_addr = PythonObject(0)
    var gb_addr = PythonObject(0)
    var ones_addr = PythonObject(0)
    if bias:
        ones = _ones(n, context)
        ones_addr = xcnn_addr(ones)
        bv = _snapshot(fields["bias_"], "bias_", context)
        if Int(py=bv.size) != f:
            raise Error("ValueError: mojolearn: bias_ has the wrong size")
        b_addr = xcnn_addr(bv)
        gb_addr = xcnn_addr(gb)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_gcn_backward_t"),
                      Python.list(xcnn_addr(g), xcnn_addr(x), xcnn_addr(hv), xcnn_addr(hv), xcnn_addr(w), xcnn_addr(gw),
                                  xcnn_addr(dx), xcnn_addr(dh), xcnn_addr(vt), xcnn_addr(graph["csr_t"]), b_addr, gb_addr,
                                  ones_addr),
                      Python.list(n, d, f, Int(py=graph["nnz"]), 1 if bias else 0, 0, 0, 0))
    py_setitem(fields, "grad_bias_", gb)
    py_setitem(fields, "grad_weight_", gw)
    _ = vt
    _ = x
    _ = w
    _ = dh
    _ = hv
    _ = ones
    _ = bv
    _ = g
    return dx


# TOMBSTONE: the hand chain of SAGEConv's resident backward (x_cnn_graph_op_m normalisation, x_cnn_linear_backward_m,
# x_cnn_spmm_m mean scale, x_cnn_spmm_m / x_cnn_graph_op_m aggregate, x_cnn_gemm_m x2 + x_cnn_map2_m root, x_cnn_res_download:
# up to nine entry crossings, removed: superseded by the tape) deleted 2026-11-09 by lane/x9-moe-gcn-rules; code recoverable at origin/integ/pre-0867-w2 f59d913b0b.
# Restore: git apply experiments/removed/XCNN_GRAPH_HAND_BACKWARD.patch
def _sage_backward_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, grad: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """The resident backward (no projection) as one native call: the tape
    over the forward's resident buffers (autograd/rules_graph.mojo
    `sage_backward_autograd`), the same entries' bodies on the same
    arguments."""
    var saved = fields["_xdev"]
    var dev = saved[0]
    var xh = saved[1]
    var aggh = saved[2]
    var graph = fields["_g"]
    var n = xcnn_shape(grad)[0]
    var d = Int(py=fields["in_channels"])
    var f = Int(py=fields["out_channels"])
    var gt = xcnn_is_t(grad, context)
    var gh = grad.h if gt else xcnn_dev_upload(layer, entries, "G", grad, context)
    var norm = Bool(py=fields["normalize"])
    var root = Bool(py=fields["root_weight"])
    var aggr = String(fields["aggr"])
    var kind = 2 if aggr == "max" else (1 if aggr == "mean" else 0)
    var outh = PythonObject(0)
    var preh = PythonObject(0)
    var gpre = PythonObject(0)
    var denh = PythonObject(0)
    if norm:
        var yd = fields["_ydev"]
        var b = xcnn_builtins()
        if py_type(yd) is not b.tuple or len(yd) != 3 or py_type(yd[0]) is not b.int or py_type(yd[1]) is not b.int:
            raise Error("TypeError: SAGE normalization requires native saved handle metadata")
        if yd[2] is not None and not xcnn_is_t(yd[2], context):
            raise Error("TypeError: SAGE normalization requires a native retained tensor")
        outh = yd[0]
        denh = yd[1]
        preh = xcnn_dev_get(layer, entries, "pre", n * f, context)
        gpre = xcnn_dev_get(layer, entries, "G2", n * f, context)
    else:
        # the forward's lin_l output array (recorded as the tape's value, never read)
        outh = xcnn_dev_get(layer, entries, "ol", n * f, context)
    var lin = _linear_fields(fields["lin_l"], context)
    var values = _linear_weights(lin, context)
    var dagg = xcnn_dev_get(layer, entries, "dagg", n * d, context)
    var dw = xcnn_host_f32([f, d], context)
    var db = xcnn_host_f32([f], context)
    var dx = xcnn_new(entries, [n, d], False, context) if gt else xcnn_host_f32([n, d], context)
    var dxa = dx.h if gt else xcnn_addr(dx)
    var dxs = dx.h if gt and not root else xcnn_dev_get(layer, entries, "dxs", n * d, context)
    var dsc = xcnn_dev_get(layer, entries, "dsc", n * d, context) if kind == 1 else PythonObject(0)
    var maxaux = _dev_h(dev, "maxaux", context) if kind == 2 else PythonObject(0)
    var _hf = _dev_pair(dev, "vals_f", "csr_f", context)   # one owner admission for both handles
    var _ht = _dev_pair(dev, "vals_t", "csr_t", context)
    var wr = PythonObject(None)
    var wr_addr = PythonObject(0)
    var gwr = PythonObject(None)
    var gwr_addr = PythonObject(0)
    var grh = PythonObject(0)
    if root:
        wr = _snapshot(fields["weight_r_"], "weight_r_", context)
        if not _same_shape(xcnn_shape(wr), [f, d]):
            raise Error("SAGE root weight shape differs from configured channels")
        wr_addr = xcnn_addr(wr)
        gwr = xcnn_host_f32([f, d], context)
        gwr_addr = xcnn_addr(gwr)
        grh = xcnn_dev_get(layer, entries, "gr", n * d, context)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_sage_backward_t"),
                      Python.list(gh, xh, aggh, outh, xcnn_addr(values[0]), xcnn_addr(values[1]), xcnn_addr(dw),
                                  xcnn_addr(db), dagg, dxa, dxs, _hf[0], _hf[1], _ht[0], _ht[1], maxaux, dsc, wr_addr,
                                  gwr_addr, grh, preh, gpre, denh),
                      Python.list(n, d, f, Int(py=graph["nnz"]), kind, 1 if norm else 0, 1 if root else 0, 1 if gt else 0))
    py_setitem(lin, "grad_weight_", dw)
    py_setitem(lin, "grad_bias_", db if Bool(py=fields["bias"]) else xcnn_host_f32([f], context))
    if root:
        py_setitem(fields, "grad_weight_r_", gwr)
    _ = values
    _ = wr
    _ = grad
    return dx


def _sage_backward(layer: PythonObject, gradient: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _graph_layer_fields(layer, context)
    var entries = xcnn_entries(fields, context)
    var g = _graph_gradient(layer, gradient, context)
    if xcnn_is_t(g, context):
        if _resident_saved(fields, entries, context) and xcnn_on_dev(entries, g, context):
            return _sage_backward_dev(layer, fields, entries, g, context)
        return xcnn_wrap(entries, _sage_backward(layer, xcnn_host_of(g, context), context), context)
    if _resident_saved(fields, entries, context):
        return _sage_backward_dev(layer, fields, entries, g, context)
    return _sage_backward_host(fields, entries, g, context)


# TOMBSTONE: the host-form hand chain of SAGEConv's backward (x_cnn_graph_op normalisation, x_cnn_linear_backward,
# x_cnn_spmm mean scale, x_cnn_spmm / x_cnn_graph_op aggregate, the projection's x_cnn_relu_backward +
# x_cnn_linear_backward, x_cnn_gemm x2 + x_cnn_add root: up to ten entry crossings, removed: superseded by the tape)
# deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/XCNN_GRAPH_HOST_HAND_BACKWARD.patch
def _sage_backward_host(fields: PythonObject, entries: PythonObject, g: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """The host-form backward (the CPU binding, project=True, or a forward
    kept on the host) as one native call: the tape over host arrays
    (autograd/graph_tape.mojo `sage_cfg`, 33 addresses and 10 integers)."""
    var n = xcnn_shape(g)[0]
    var d = Int(py=fields["in_channels"])
    var f = Int(py=fields["out_channels"])
    var graph = fields["_g"]
    var nnz = Int(py=graph["nnz"])
    var norm = Bool(py=fields["normalize"])
    var root = Bool(py=fields["root_weight"])
    var project = Bool(py=fields["project"])
    var aggr = String(fields["aggr"])
    var kind = 2 if aggr == "max" else (1 if aggr == "mean" else 0)
    var keep = Python.list()   # every array an address below points into, alive through the call
    var x = xcnn_f32(fields["_x"], "x", context)
    keep.append(x)
    var lin = _linear_fields(fields["lin_l"], context)
    if py_get(lin, "_x") is None:
        raise Error("RuntimeError: linear backward requires a prior forward")
    var agg = xcnn_f32(lin["_x"], "x", context)
    keep.append(agg)
    var values = _linear_weights(lin, context)
    keep.append(values)
    var dw = xcnn_host_f32([f, d], context)
    var db = xcnn_host_f32([f], context)
    var dagg = xcnn_host_f32([n, d], context)
    var dx = xcnn_host_f32([n, d], context)
    keep.append(dagg)
    var ones = _ones(max(nnz, 1), context)
    keep.append(ones)
    var outv = PythonObject(0)
    var prev = PythonObject(0)
    var gpre = PythonObject(0)
    var den = PythonObject(0)
    var nograph = PythonObject(0)
    if norm:
        var y = xcnn_f32(fields["_y"], "y", context)
        var pre = xcnn_f32(fields["_pre"], "pre", context)
        var gp = xcnn_host_f32([n, f], context)
        keep.append(y)
        keep.append(pre)
        keep.append(gp)
        keep.append(fields["_den"])
        keep.append(fields["_nograph"])
        outv = xcnn_addr(y)
        prev = xcnn_addr(pre)
        gpre = xcnn_addr(gp)
        den = xcnn_addr(fields["_den"])
        nograph = xcnn_addr(fields["_nograph"])
    else:
        # the layer output (recorded as the tape's value, never read)
        var ov = xcnn_host_f32([n, f], context)
        keep.append(ov)
        outv = xcnn_addr(ov)
    var maxaux = PythonObject(0)
    if kind == 2:
        keep.append(fields["_max_aux"])
        maxaux = xcnn_addr(fields["_max_aux"])
    var dsc = PythonObject(0)
    if kind == 1:
        var t = xcnn_host_f32([n, d], context)
        keep.append(t)
        dsc = xcnn_addr(t)
    var wr_addr = PythonObject(0)
    var gwr = PythonObject(None)
    var gwr_addr = PythonObject(0)
    var gr = PythonObject(0)
    var dxs = PythonObject(0)
    var dxp = PythonObject(0)
    if root:
        var wr = _snapshot(fields["weight_r_"], "weight_r_", context)
        if not _same_shape(xcnn_shape(wr), [f, d]):
            raise Error("SAGE root weight shape differs from configured channels")
        keep.append(wr)
        wr_addr = xcnn_addr(wr)
        gwr = xcnn_host_f32([f, d], context)
        gwr_addr = xcnn_addr(gwr)
        var t1 = xcnn_host_f32([n, d], context)
        var t2 = xcnn_host_f32([n, d], context)
        var t3 = xcnn_host_f32([n, d], context)
        keep.append(t1)
        keep.append(t2)
        keep.append(t3)
        gr = xcnn_addr(t1)
        dxs = xcnn_addr(t2)
        dxp = xcnn_addr(t3)
    var s_addr = PythonObject(0)
    var prep = PythonObject(0)
    var wp = PythonObject(0)
    var bp = PythonObject(0)
    var plin = PythonObject(None)
    var dwp = PythonObject(None)
    var dbp = PythonObject(None)
    var dwp_addr = PythonObject(0)
    var dbp_addr = PythonObject(0)
    var dsp = PythonObject(0)
    var dpre = PythonObject(0)
    if project:
        if py_type(fields["_relu_p"]) is not context["xcnn_relu_class"]:
            raise Error("TypeError: graph projection requires a native ReLU layer")
        var relu_x = py_get(_fields(fields["_relu_p"]), "_x")
        if relu_x is None:
            raise Error("RuntimeError: ReLU backward requires a prior forward")
        plin = _linear_fields(fields["lin"], context)
        var pvalues = _linear_weights(plin, context)
        var sv = xcnn_f32(fields["_xs"], "x", context)
        var pv = xcnn_f32(relu_x, "x", context)
        dwp = xcnn_host_f32([d, d], context)
        dbp = xcnn_host_f32([d], context)
        var t4 = xcnn_host_f32([n, d], context)
        var t5 = xcnn_host_f32([n, d], context)
        keep.append(pvalues)
        keep.append(sv)
        keep.append(pv)
        keep.append(t4)
        keep.append(t5)
        s_addr = xcnn_addr(sv)
        prep = xcnn_addr(pv)
        wp = xcnn_addr(pvalues[0])
        bp = xcnn_addr(pvalues[1])
        dwp_addr = xcnn_addr(dwp)
        dbp_addr = xcnn_addr(dbp)
        dsp = xcnn_addr(t4)
        dpre = xcnn_addr(t5)
    var csr_f = xcnn_addr(graph["csr_f"])
    var csr_t = xcnn_addr(graph["csr_t"])
    var addrs = Python.list(
        xcnn_addr(g), xcnn_addr(x), xcnn_addr(agg), outv, xcnn_addr(values[0]), xcnn_addr(values[1]), xcnn_addr(dw),
        xcnn_addr(db), xcnn_addr(dagg), xcnn_addr(dx), dxs, xcnn_addr(ones), csr_f, xcnn_addr(ones), csr_t, maxaux,
        dsc, wr_addr, gwr_addr, gr, prev, gpre, den,
    )
    # the host form's ten extra words (autograd/graph_tape.mojo sage_cfg)
    addrs.append(s_addr)
    addrs.append(prep)
    addrs.append(wp)
    addrs.append(bp)
    addrs.append(dwp_addr)
    addrs.append(dbp_addr)
    addrs.append(dsp)
    addrs.append(dpre)
    addrs.append(nograph)
    addrs.append(dxp)
    _ = py_call_typed(xcnn_entry(entries, "x_cnn_sage_backward_t"), addrs,
                      Python.list(n, d, f, nnz, kind, 1 if norm else 0, 1 if root else 0, 0, 0, 1 if project else 0))
    py_setitem(lin, "grad_weight_", dw)
    py_setitem(lin, "grad_bias_", db if Bool(py=fields["bias"]) else xcnn_host_f32([f], context))
    if project:
        py_setitem(plin, "grad_weight_", dwp)
        py_setitem(plin, "grad_bias_", dbp)
    if root:
        py_setitem(fields, "grad_weight_r_", gwr)
    _ = keep
    _ = g
    return dx


def xcnn_graph_layers(layer: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "gcn_init" or operation == "sage_init":
        return _graph_init(layer, args, operation == "sage_init", context)
    if operation == "gcn_forward":
        return xcnn_gcn_forward(layer, args, context)
    if operation == "sage_forward":
        return xcnn_sage_forward(layer, args, context)
    if operation == "gcn_backward":
        return _gcn_backward(layer, args[0], context)
    if operation == "sage_backward":
        return _sage_backward(layer, args[0], context)
    if operation == "linear_init":
        return _linear_init(layer, args, context)
    if operation == "linear_forward":
        return _linear_forward(layer, args[0], context)
    if operation == "linear_backward":
        return _linear_backward(layer, args[0], context)
    if operation == "relu_init":
        neural_plain_value(args, 0, Python.dict(admission_context=context))
        py_setitem(_fields(layer), "numeric_mode", args[0])
        return PythonObject(None)
    if operation == "relu_forward":
        return xcnn_relu_forward(layer, args[0], context)
    if operation == "relu_backward":
        return _relu_backward(layer, args[0], context)
    raise Error("RuntimeError: unknown native graph layer operation " + operation)
