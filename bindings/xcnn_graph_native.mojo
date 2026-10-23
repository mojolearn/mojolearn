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

NumPy is called for allocation and dtype conversion only.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import (
    xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_mixed, xcnn_idn, XCNN_F_GRAPH, xcnn_is_t, xcnn_on_dev,
    xcnn_new, xcnn_wrap, xcnn_host_of, xcnn_host_f32, xcnn_f32, xcnn_addr, xcnn_dev, xcnn_dev_get,
    xcnn_dev_upload,
)
from core.xcnn_graph_impl import graph_edges_in_range, graph_bias_rows


def _np() raises -> PythonObject:
    return Python.import_module("numpy")


# ---------------------------------------------------------------- the graph

def _data_addr(a: PythonObject) raises -> PythonObject:
    """`_data_addr(a)`: a NumPy array's or a tensor's data address, else None."""
    try:
        return a.__array_interface__["data"][0]
    except:
        pass
    try:
        return py_call(xcnn_builtins().int, py_call(a.data_ptr))
    except:
        return PythonObject(None)


def _graph_key(n: Int, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GraphKey(n, *arrays)`: the node count, the arrays themselves and their
    shape, dtype and address."""
    var b = xcnn_builtins()
    var key = py_call(b.object.__new__, context["xcnn_graph_key_class"])
    var sig = Python.list()
    for i in range(len(arrays)):  # small-loop(the graph input arrays, one or two)
        var a = arrays[i]
        if a is None:
            _ = py_call(sig.append, PythonObject(None))
        else:
            _ = py_call(sig.append, Python.tuple(py_call(b.tuple, py_call(b.getattr, a, "shape", Python.tuple())),
                                                py_call(b.str, py_call(b.getattr, a, "dtype", "")), _data_addr(a)))
    py_setattr(key, "n", n)
    py_setattr(key, "refs", py_call(b.tuple, arrays))
    py_setattr(key, "sig", py_call(b.tuple, sig))
    return key


def _keys_equal(a: PythonObject, b_key: PythonObject) raises -> Bool:
    """`_GraphKey.__eq__`: the same node count, the same array objects, the
    same signature."""
    var b = xcnn_builtins()
    if py_type(a) is not py_type(b_key):
        return False
    if Int(py=a.n) != Int(py=b_key.n) or len(a.refs) != len(b_key.refs):
        return False
    for i in range(len(a.refs)):  # small-loop(the graph input arrays, one or two)
        if a.refs[i] is not b_key.refs[i]:
            return False
    return Bool(py=a.sig == b_key.sig)


def _edges(edge_index: PythonObject, n: Int) raises -> PythonObject:
    """`_edges(edge_index, n)`: (src, dst) int64 rows, the shape and the
    node-range refusals."""
    var np = _np()
    var ei = py_call(np.asarray, edge_index)
    if Int(py=ei.ndim) != 2 or Int(py=ei.shape[0]) != 2:
        raise Error("ValueError: mojolearn: edge_index must be (2, E)")
    var e64 = py_call(np.ascontiguousarray, py_call(ei.astype, np.int64))
    var size = Int(py=e64.size)
    if size and not graph_edges_in_range(Int(py=xcnn_addr(e64)), size, n):
        raise Error("ValueError: mojolearn: edge_index refers to a node that does not exist")
    return Python.tuple(e64[0], e64[1])


def _csr(entries: PythonObject, rows: PythonObject, cols: PythonObject, n: Int, nnz: Int) raises -> PythonObject:
    """`_Graph._build(binding, rows, cols)`: (csr, order[:nnz])."""
    var np = _np()
    var csr = py_call(np.empty, n + 1 + 2 * nnz, np.int32)
    var order = py_call(np.empty, max(nnz, 1), np.int32)
    _ = py_call(xcnn_entry(entries, "x_cnn_csr_build"), xcnn_addr(rows), xcnn_addr(cols), xcnn_addr(csr), xcnn_addr(order),
              Python.list(n, nnz))
    return Python.tuple(csr, order.__getitem__(py_call(xcnn_builtins().slice, PythonObject(None), nnz)))


def _graph_new(src: PythonObject, dst: PythonObject, n: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Graph(src, dst, n, binding)`: the record and its two CSR views."""
    var np = _np()
    var g = py_call(xcnn_builtins().object.__new__, context["xcnn_graph_class"])
    var fields = py_fields(g)
    var s64 = py_call(np.asarray, src, np.int64)
    var d64 = py_call(np.asarray, dst, np.int64)
    var nnz = len(s64)
    py_setitem(fields, "n", n)
    py_setitem(fields, "src", s64)
    py_setitem(fields, "dst", d64)
    py_setitem(fields, "nnz", nnz)
    var s32 = py_call(np.ascontiguousarray, s64, np.int32)
    var d32 = py_call(np.ascontiguousarray, d64, np.int32)
    var f = _csr(entries, d32, s32, n, nnz)
    var t = _csr(entries, s32, d32, n, nnz)
    py_setitem(fields, "csr_f", f[0])
    py_setitem(fields, "order_f", f[1])
    py_setitem(fields, "csr_t", t[0])
    py_setitem(fields, "order_t", t[1])
    return g


def _gather(entries: PythonObject, values: PythonObject, order: PythonObject) raises -> PythonObject:
    """`np.ascontiguousarray(values[order])` (float32), on the device."""
    var np = _np()
    var src = py_call(np.ascontiguousarray, values, np.float32)
    var idx = py_call(np.ascontiguousarray, order, np.int32)
    var n = len(idx)
    var out = py_call(np.empty, n, np.float32)
    _ = py_call(xcnn_entry(entries, "x_cnn_perm_gather"), Python.list(xcnn_addr(src), xcnn_addr(idx), xcnn_addr(out)),
              Python.list(Int(py=src.size), n))
    return out


def _vals_t(entries: PythonObject, g: PythonObject, vals_f: PythonObject) raises -> PythonObject:
    """`g.vals_t(vals_f)`, on the device."""
    var np = _np()
    var nnz = Int(py=g.nnz)
    var vf = py_call(np.ascontiguousarray, vals_f, np.float32)
    var out = py_call(np.empty, nnz, np.float32)
    _ = py_call(xcnn_entry(entries, "x_cnn_vals_t"), Python.list(xcnn_addr(vf), xcnn_addr(py_call(np.ascontiguousarray, g.order_f)),
                                                   xcnn_addr(py_call(np.ascontiguousarray, g.order_t)), xcnn_addr(out)),
              Python.list(nnz))
    return out


# ---------------------------------------------------------------- host operations

def _gemm(entries: PythonObject, a: PythonObject, b2: PythonObject, m: Int, n: Int, k: Int, op: Int,
          context: PythonObject) raises -> PythonObject:
    """`_gemm(binding, a, b, m, n, k, op)`."""
    var out = xcnn_host_f32([m, n], context)
    var fa = xcnn_f32(a, "a", context)
    var fb = xcnn_f32(b2, "b", context)
    _ = py_call(xcnn_entry(entries, "x_cnn_gemm"), xcnn_addr(fa), xcnn_addr(fb), xcnn_addr(out), Python.list(m, n, k, op))
    return out


def _spmm(entries: PythonObject, g: PythonObject, vals: PythonObject, h: PythonObject, mode: Int,
          context: PythonObject) raises -> PythonObject:
    """`g.spmm(binding, vals, h, mode)` (the forward view)."""
    var np = _np()
    var hh = xcnn_f32(h, "h", context)
    var out = py_call(np.empty_like, hh)
    var v = py_call(np.zeros, 1, np.float32) if vals is None or len(vals) == 0 else py_call(np.ascontiguousarray, vals, np.float32)
    _ = py_call(xcnn_entry(entries, "x_cnn_spmm"), xcnn_addr(v), xcnn_addr(hh), xcnn_addr(out), xcnn_addr(g.csr_f),
              Python.list(Int(py=g.n), Int(py=hh.shape[1]), Int(py=g.nnz), mode))
    return out


def _add(entries: PythonObject, a: PythonObject, b2: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_add(binding, a, b)`."""
    var fa = xcnn_f32(a, "a", context)
    var fb = xcnn_f32(b2, "b", context)
    var out = py_call(_np().empty_like, fa)
    _ = py_call(xcnn_entry(entries, "x_cnn_add"), xcnn_addr(fa), xcnn_addr(fb), xcnn_addr(out), Python.list(Int(py=fa.size)))
    return out


def _bias_rows(bias: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """`np.ascontiguousarray(np.broadcast_to(bias, (rows, cols)))`."""
    var fb = xcnn_f32(bias, "b", context)
    var out = xcnn_host_f32([rows, cols], context)
    graph_bias_rows(Int(py=xcnn_addr(fb)), Int(py=xcnn_addr(out)), rows, cols)
    return out


def _linear_forward(lin: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Linear.forward(x)`."""
    var fields = py_fields(lin)
    var entries = xcnn_entries(fields, context)
    var fx = xcnn_f32(x, "x", context)
    var rows = Int(py=fx.shape[0])
    var out = xcnn_host_f32([rows, Int(py=fields["out_features"])], context)
    _ = py_call(xcnn_entry(entries, "x_cnn_linear_forward"), xcnn_addr(fx), xcnn_addr(fields["weight_"]), xcnn_addr(fields["bias_"]),
              xcnn_addr(out), Python.list(rows, fields["in_features"], fields["out_features"]))
    py_setitem(fields, "_x", fx)
    return out


def _relu_forward(relu: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ReLU.forward(x)` of a host array."""
    var fields = py_fields(relu)
    var entries = xcnn_entries(fields, context)
    var fx = xcnn_f32(x, "x", context)
    var out = py_call(_np().empty_like, fx)
    _ = py_call(xcnn_entry(entries, "x_cnn_relu_forward"), xcnn_addr(fx), xcnn_addr(out), Python.list(Int(py=fx.size)))
    py_setitem(fields, "_x", fx)
    return out


# ---------------------------------------------------------------- resident graph

def _adopt(dev: PythonObject, entries: PythonObject, name: String, h: PythonObject, n: Int) raises:
    """`_Dev.adopt(name, h, n)`."""
    var table = py_fields(dev)["h"]
    var cur = py_call(table.pop, name, PythonObject(None))
    if cur is not None:
        _ = py_call(xcnn_entry(entries, "x_cnn_res_free"), cur[0])
    py_setitem(table, name, Python.tuple(Int(py=py_call(xcnn_builtins().int, h)), max(n, 1)))


def _graph_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, dev: PythonObject, g: PythonObject,
               vals_f: PythonObject, vals_t: PythonObject, context: PythonObject) raises:
    """`_graph_dev(layer, b, dev, g, vals_f, vals_t)`: the resident CSR views
    and values of graph `g`, uploaded once per graph."""
    var np = _np()
    var b = xcnn_builtins()
    var tag = Python.tuple(py_call(b.id, g), py_call(b.id, dev))
    if Bool(py=py_get(fields, "_gdev_for") == tag) and py_get(fields, "_gdev_g") is g:
        return
    var n = Int(py=g.n)
    var nnz = Int(py=g.nnz)
    var prm = Python.list(n, 1, nnz, 0)
    _adopt(dev, entries, "csr_f", py_call(xcnn_entry(entries, "x_cnn_csr_upload"), xcnn_addr(g.csr_f), prm), Int(py=g.csr_f.size))
    _adopt(dev, entries, "csr_t", py_call(xcnn_entry(entries, "x_cnn_csr_upload"), xcnn_addr(g.csr_t), prm), Int(py=g.csr_t.size))
    var vf = py_call(np.ascontiguousarray, vals_f, np.float32)
    var vt = _vals_t(entries, g, vf) if vals_t is None else py_call(np.ascontiguousarray, vals_t, np.float32)
    _ = xcnn_dev_upload(layer, entries, "vals_f", vf if Int(py=vf.size) else py_call(np.zeros, 1, np.float32), context)
    _ = xcnn_dev_upload(layer, entries, "vals_t", vt if Int(py=vt.size) else py_call(np.zeros, 1, np.float32), context)
    py_setitem(fields, "_gdev_for", tag)
    py_setitem(fields, "_gdev_g", g)


def _dev_h(dev: PythonObject, name: String) raises -> PythonObject:
    return py_fields(dev)["h"][name][0]


def _download(entries: PythonObject, h: PythonObject, out_buf: PythonObject) raises:
    """`dev.download(h, out)`."""
    _ = py_call(xcnn_entry(entries, "x_cnn_res_download"), h, xcnn_addr(out_buf), out_buf.size)


# ---------------------------------------------------------------- GCNConv

def _gcn_graph(fields: PythonObject, entries: PythonObject, n: Int, edge_index: PythonObject, edge_weight: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`GCNConv._graph(n, edge_index, edge_weight)`: (g, vals)."""
    var np = _np()
    var edges = _edges(edge_index, n)
    var src = edges[0]
    var dst = edges[1]
    var w = PythonObject(None)
    if edge_weight is None:
        w = py_call(np.ones, len(src), np.float32)
    else:
        w = py_call(py_call(xcnn_f32(edge_weight, "edge_weight", context).reshape, -1).copy)
    if len(w) != len(src):
        raise Error("ValueError: mojolearn: edge_weight must have one entry per edge")
    if Bool(py=fields["normalize"]) and Bool(py=fields["add_self_loops"]):
        if not py_contains(entries["symbols"], "x_cnn_gcn_loops"):
            raise Error("RuntimeError: mojolearn: this x_cnn binding has no x_cnn_gcn_loops (an older binary); rebuild it")
        var s32 = py_call(np.ascontiguousarray, src, np.int32)
        var d32 = py_call(np.ascontiguousarray, dst, np.int32)
        var w32 = py_call(np.ascontiguousarray, w, np.float32)
        var t = len(s32) + n
        var so = py_call(np.empty, t, np.int32)
        var dso = py_call(np.empty, t, np.int32)
        var wo = py_call(np.empty, t, np.float32)
        var k = Int(py=py_call(xcnn_builtins().int, py_call(xcnn_entry(entries, "x_cnn_gcn_loops"), Python.list(xcnn_addr(s32), xcnn_addr(d32), xcnn_addr(w32), xcnn_addr(so), xcnn_addr(dso), xcnn_addr(wo)),
            Python.list(n, len(s32), 1 if Bool(py=fields["improved"]) else 0))))
        var cut = py_call(xcnn_builtins().slice, PythonObject(None), k + n)
        src = so.__getitem__(cut)
        dst = dso.__getitem__(cut)
        w = wo.__getitem__(cut)
    var g = _graph_new(src, dst, n, entries, context)
    var wf = _gather(entries, w, g.order_f)
    var vals = wf
    if Bool(py=fields["normalize"]):
        var nnz = Int(py=g.nnz)
        vals = py_call(np.empty, nnz, np.float32)
        if xcnn_idn(entries, XCNN_F_GRAPH):
            _ = py_call(xcnn_entry(entries, "x_cnn_gcn_norm_m"), Python.list(xcnn_addr(wf), xcnn_addr(vals), xcnn_addr(g.csr_f)),
                      PythonObject(0), Python.list(n, 1, nnz, 0))
        else:
            _ = py_call(xcnn_entry(entries, "x_cnn_gcn_norm"), xcnn_addr(wf), xcnn_addr(vals), xcnn_addr(g.csr_f), Python.list(n, 1, nnz, 0))
    return Python.tuple(g, vals)


def _gcn_forward_dev(layer: PythonObject, fields: PythonObject, entries: PythonObject, x: PythonObject, g: PythonObject,
                     vals: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = Int(py=x.shape[0])
    var f_out = Int(py=fields["out_channels"])
    var d_in = Int(py=fields["in_channels"])
    if Int(py=x.ndim) != 2 or Int(py=x.shape[1]) != d_in:
        raise Error("ValueError: mojolearn: x must be (n, " + String(d_in) + ")")
    var dev = xcnn_dev(layer, entries, context)
    _graph_dev(layer, fields, entries, dev, g, vals, PythonObject(None), context)
    var w = xcnn_f32(fields["weight_"], "b", context)
    var xt = xcnn_is_t(x, context)
    var xh = x.h if xt else xcnn_dev_upload(layer, entries, "x", x, context)
    var hh = xcnn_dev_get(layer, entries, "h", n * f_out, context)
    _ = py_call(xcnn_entry(entries, "x_cnn_gemm_m"), Python.list(xh, xcnn_addr(w), hh), PythonObject(0b101), Python.list(n, f_out, d_in, 1))
    var bias = Bool(py=fields["bias"])
    var out = xcnn_new(entries, [n, f_out], False, context) if xt else xcnn_host_f32([n, f_out], context)
    var oh = out.h if (xt and not bias) else xcnn_dev_get(layer, entries, "o", n * f_out, context)
    _ = py_call(xcnn_entry(entries, "x_cnn_spmm_m"), Python.list(_dev_h(dev, "vals_f"), hh, _dev_h(dev, "csr_f"), oh), PythonObject(0b1111),
              Python.list(n, f_out, Int(py=g.nnz), 0))
    if bias:
        var bv = py_call(xcnn_f32(fields["bias_"], "b", context).reshape, -1)
        if Int(py=bv.size) != f_out:
            raise Error("ValueError: mojolearn: bias_ has the wrong size")
        if xt:
            _ = py_call(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(oh, xcnn_addr(bv), out.h), PythonObject(0b101), Python.list(4, n * f_out, f_out))
        else:
            _ = py_call(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(oh, xcnn_addr(bv), xcnn_addr(out)), PythonObject(0b001),
                      Python.list(4, n * f_out, f_out))
    elif not xt:
        _download(entries, oh, out)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    py_setitem(fields, "_vals", vals)
    py_setitem(fields, "_xdev", Python.tuple(dev, xh))
    return out


def xcnn_gcn_forward(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(layer)
    var entries = xcnn_entries(fields, context)
    var x = args[0]
    var edge_index = args[1]
    var edge_weight = args[2]
    if xcnn_is_t(x, context):
        if not xcnn_on_dev(entries, x, context):
            var host = xcnn_host_of(x, context)
            return xcnn_wrap(entries, xcnn_gcn_forward(layer, Python.tuple(host, edge_index, edge_weight), context), context)
    else:
        x = xcnn_f32(x, "x", context)
    var n = Int(py=x.shape[0])
    var key = Python.tuple(_graph_key(n, Python.list(edge_index, edge_weight), context), fields["improved"],
                           fields["add_self_loops"], fields["normalize"])
    var hit = py_get(fields, "_gcache")
    var g = PythonObject(None)
    var vals = PythonObject(None)
    if (hit is not None and _keys_equal(hit[0][0], key[0]) and Bool(py=hit[0][1] == key[1])
            and Bool(py=hit[0][2] == key[2]) and Bool(py=hit[0][3] == key[3])):
        g = hit[1][0]
        vals = hit[1][1]
    else:
        var gv = _gcn_graph(fields, entries, n, edge_index, edge_weight, context)
        g = gv[0]
        vals = gv[1]
        py_setitem(fields, "_gcache", Python.tuple(key, Python.tuple(g, vals)))
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
    var np = _np()
    var b = xcnn_builtins()
    var n = Int(py=x.shape[0])
    var d_in = Int(py=fields["in_channels"])
    var f_out = Int(py=fields["out_channels"])
    if Int(py=x.ndim) != 2 or Int(py=x.shape[1]) != d_in:
        raise Error("ValueError: mojolearn: x must be (n, " + String(d_in) + ")")
    var dev = xcnn_dev(layer, entries, context)
    var aggr = String(py=fields["aggr"])
    var mean = aggr == "mean"
    var tag = Python.tuple(py_call(b.id, g), py_call(b.id, dev))
    if py_get(fields, "_gdev_g") is not g or not Bool(py=py_get(fields, "_gdev_for") == tag):
        var ones = py_call(np.ones, Int(py=g.nnz), np.float32)
        _graph_dev(layer, fields, entries, dev, g, ones, ones, context)
    var xt = xcnn_is_t(x, context)
    var xh = x.h if xt else xcnn_dev_upload(layer, entries, "x", x, context)
    var aggh = xcnn_dev_get(layer, entries, "agg", n * d_in, context)
    var nnz = Int(py=g.nnz)
    if aggr == "max":
        var auxh = xcnn_dev_get(layer, entries, "maxaux", 2 * n * d_in, context)
        _ = py_call(xcnn_entry(entries, "x_cnn_graph_op_m"), Python.list(xh, xh, auxh, aggh, _dev_h(dev, "csr_f")), PythonObject(0b11111),
                  Python.list(n, d_in, nnz, 0))
    else:
        _ = py_call(xcnn_entry(entries, "x_cnn_spmm_m"), Python.list(_dev_h(dev, "vals_f"), xh, _dev_h(dev, "csr_f"), aggh),
                  PythonObject(0b1111), Python.list(n, d_in, nnz, 1 if mean else 0))
    var lin = fields["lin_l"]
    var lin_fields = py_fields(lin)
    var norm = Bool(py=fields["normalize"])
    var root = Bool(py=fields["root_weight"])
    var out = xcnn_new(entries, [n, f_out], False, context) if xt else xcnn_host_f32([n, f_out], context)
    var preh = xcnn_dev_get(layer, entries, "pre", n * f_out, context) if norm else PythonObject(None)
    var olh = PythonObject(None)
    if norm and not root:
        olh = preh
    else:
        olh = out.h if (xt and not root) else xcnn_dev_get(layer, entries, "ol", n * f_out, context)
    _ = py_call(xcnn_entry(entries, "x_cnn_linear_forward_m"), Python.list(aggh, xcnn_addr(lin_fields["weight_"]), xcnn_addr(lin_fields["bias_"]), olh), PythonObject(0b1001),
              Python.list(n, d_in, f_out))
    if root:
        var wr = xcnn_f32(fields["weight_r_"], "b", context)
        var rh = xcnn_dev_get(layer, entries, "r", n * f_out, context)
        _ = py_call(xcnn_entry(entries, "x_cnn_gemm_m"), Python.list(xh, xcnn_addr(wr), rh), PythonObject(0b101), Python.list(n, f_out, d_in, 1))
        if norm:
            _ = py_call(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(olh, rh, preh), PythonObject(0b111), Python.list(2, n * f_out))
        elif xt:
            _ = py_call(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(olh, rh, out.h), PythonObject(0b111), Python.list(2, n * f_out))
        else:
            _ = py_call(xcnn_entry(entries, "x_cnn_map2_m"), Python.list(olh, rh, xcnn_addr(out)), PythonObject(0b011), Python.list(2, n * f_out))
    elif not xt and not norm:
        _download(entries, olh, out)
    py_setitem(fields, "_ydev", PythonObject(None))
    if norm:
        var denh = xcnn_dev_get(layer, entries, "den", n, context)
        var yh = out.h if xt else xcnn_dev_get(layer, entries, "y", n * f_out, context)
        _ = py_call(xcnn_entry(entries, "x_cnn_graph_op_m"), Python.list(preh, preh, denh, yh, PythonObject(0)), PythonObject(0b01111),
                  Python.list(n, f_out, 0, 2))
        if not xt:
            _download(entries, yh, out)
        py_setitem(fields, "_ydev", Python.tuple(yh, denh, out if xt else PythonObject(None)))
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    py_setitem(fields, "_xs", x)
    py_setitem(fields, "_xdev", Python.tuple(dev, xh, aggh))
    return out


def xcnn_sage_forward(layer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var np = _np()
    var fields = py_fields(layer)
    var entries = xcnn_entries(fields, context)
    var x = args[0]
    var edge_index = args[1]
    if xcnn_is_t(x, context):
        if not (_sage_res_form(fields, entries) and xcnn_on_dev(entries, x, context)):
            var host = xcnn_host_of(x, context)
            return xcnn_wrap(entries, xcnn_sage_forward(layer, Python.tuple(host, edge_index), context), context)
    else:
        x = xcnn_f32(x, "x", context)
    var n = Int(py=x.shape[0])
    var key = _graph_key(n, Python.list(edge_index), context)
    var hit = py_get(fields, "_gcache")
    var g = PythonObject(None)
    if hit is not None and _keys_equal(hit[0], key):
        g = hit[1]
    else:
        var edges = _edges(edge_index, n)
        g = _graph_new(edges[0], edges[1], n, entries, context)
        py_setitem(fields, "_gcache", Python.tuple(key, g))
    if _sage_res_form(fields, entries):
        return _sage_forward_dev(layer, fields, entries, x, g, context)
    py_setitem(fields, "_xdev", PythonObject(None))
    var xs = x
    if Bool(py=fields["project"]):
        xs = _relu_forward(fields["_relu_p"], _linear_forward(fields["lin"], x, context), context)
    py_setitem(fields, "_xs", xs)
    var aggr = String(py=fields["aggr"])
    var agg = PythonObject(None)
    var nnz = Int(py=g.nnz)
    if aggr == "max":
        agg = py_call(np.empty_like, xs)
        var aux = py_call(np.zeros, 2 * Int(py=xs.size), np.float32)
        py_setitem(fields, "_max_aux", aux)
        _ = py_call(xcnn_entry(entries, "x_cnn_graph_op"), xcnn_addr(xs), xcnn_addr(xs), xcnn_addr(aux), xcnn_addr(agg), xcnn_addr(g.csr_f),
                  Python.list(n, Int(py=xs.shape[1]), nnz, 0))
    else:
        agg = _spmm(entries, g, py_call(np.ones, nnz, np.float32), xs, 1 if aggr == "mean" else 0, context)
    var out = _linear_forward(fields["lin_l"], agg, context)
    if Bool(py=fields["root_weight"]):
        out = _add(entries, out, _gemm(entries, x, fields["weight_r_"], n, Int(py=fields["out_channels"]),
                                       Int(py=fields["in_channels"]), 1, context), context)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "_g", g)
    if Bool(py=fields["normalize"]):
        py_setitem(fields, "_pre", out)
        var y = py_call(np.empty_like, out)
        var den = py_call(np.zeros, n, np.float32)
        var nograph = py_call(np.zeros, n + 1, np.int32)
        py_setitem(fields, "_den", den)
        py_setitem(fields, "_nograph", nograph)
        _ = py_call(xcnn_entry(entries, "x_cnn_graph_op"), xcnn_addr(out), xcnn_addr(out), xcnn_addr(den), xcnn_addr(y), xcnn_addr(nograph),
                  Python.list(n, Int(py=out.shape[1]), 0, 2))
        py_setitem(fields, "_y", y)
        out = y
    return out
