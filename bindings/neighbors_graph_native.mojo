# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): PageRank,
# Louvain, connected_components and SVGP public methods
# (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_graph_*` operation validates, prepares, runs the family's
# compiled kernels and returns, with the former Python method's refusals,
# operations and order, so the bits do not move.
#
# The shell hands over the x_neighbors binding `_bind` resolved (args[0]; for
# connected_components the one `_backend.binding` resolves for its
# numeric_mode). Every kernel call is `binding.xn_<op>(addresses, ints,
# floats)` exactly as `_XNeighbors._op` made it, with the same buffers, the
# same integers and the same float32-rounded scalars, in the same order. The
# graph inputs are read as before: an `(indptr, indices, n)` tuple or a
# scipy.sparse matrix read natively from its stored buffers (no `tocsr()`), a
# dense matrix as float32. Outputs are new mojolearn Arrays (the Python
# shells' `empty` / `_empty_out`, zero-filled: every element is written by
# the kernel). MOJOLEARN_XN_LAZY_OUT (an opt-in memory-map A/B arm for
# outputs of 64 MB or more, same zeros and the same bits) is not consulted.
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_module, xn_is_module, xn_capabilities
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from x_neighbors.descriptor_schema import xn_operation_id
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.admitted_family import admit_family
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import family_call, family_isinstance, family_str, family_array
from bindings.cfb_spectral import _as_i32
from bindings.host_sparse_native import host_sparse_format, _sparse_fields, host_sparse_dense
from bindings.cfb_sparse_read import cfb_sparse_triples
from bindings.neighbors_ann_native import ann_keep_array
from bindings.mcls_cluster import mcls_cluster_operation
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, PENDING_PYTHON_ERROR
from neighbors.neighbors_graph_run import (
    XnOpRequest, xn_rows_match_check, xn_relabel_request, xn_relabel_decode, xn_fill_request,
    xn_iota_request, xn_graph_square_check, xn_graph_triples_check, xn_graph_weights_check,
    xn_graph_merge_triples, xn_graph_merged_copy, xn_graph_dense_counts, xn_graph_dense_fill, xn_graph_csr_check,
    xn_unit_length_check, xn_unit_request, xn_unit_check, xn_unit_missing, xn_pagerank_csr_request,
    xn_pagerank_dense_request, xn_pagerank_decode, xn_pagerank_converged_check, xn_cc_csr_request,
    xn_cc_dense_request, xn_cc_csr_components, xn_cc_relabel_check, xn_louvain_csr_request, xn_louvain_request,
    xn_graph_symmetry_request, xn_louvain_symmetry_check, xn_louvain_device_message, xn_louvain_relabel_check,
    xn_louvain_decode, xn_svgp_zero_denominator, xn_svgp_gamma, xn_svgp_even_rows, xn_take_rows_request, xn_svgp_fit_request,
    xn_svgp_fit_decode, xn_svgp_definite_check, xn_svgp_predict_request, xn_kernel_rbf_request,
    xn_unary_scale_request, xn_unary_shift_request, xn_matmul_request, xn_svgp_var_request,
)
from providers.python.estimator_binding_pyapi import estimator_binding, estimator_binding_mode
from providers.python.family_public_lifecycle_pyapi import family_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal

# The typed checks, the operation requests (their word order and float32
# rounding) and the info-word decoding are RUN in
# neighbors/neighbors_graph_run.mojo; this file is UNPACK/PACK: Python reads,
# admission, array allocation, publication and the binding call.


# ---------------------------------------------------------------- glue


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(_b().bool, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _ne(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a != b)`."""
    return _truth(family_call(native_loaded_module("operator").ne, a, b))


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`."""
    return family_call(_b().int, value)


def _float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=family_call(_b().float, value))


def _attr(model: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` of a stored attribute, refused as Python refused it."""
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[PythonObject(name)]


def _addr(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))


def _float_of(model: PythonObject, name: String) raises -> Float64:
    """`float(model.<name>)`."""
    return _float(_attr(model, name))


def _py_ints(values: List[Int]) raises -> PythonObject:
    var result = Python.list()
    for value in values:  # small-loop(named operation's fixed integer schema)
        _ = py_call(result.append, PythonObject(value))
    return result


def _py_floats(values: List[Float64]) raises -> PythonObject:
    var result = Python.list()
    for value in values:  # small-loop(named operation's fixed scalar schema)
        _ = py_call(result.append, PythonObject(value))
    return result


def _op(binding: PythonObject, request: XnOpRequest) raises:
    """`_XNeighbors._op(name, bufs, ints, floats)` of a RUN request. A
    compiled-binding snapshot's native descriptor provider takes the typed
    words (the call `xn_execute` makes for it, with its GIL release and call
    count); a loaded module's `xn_<name>` export takes them as Python lists,
    the entry looked up as `getattr` looked it up."""
    if xn_is_module(binding):
        xn_execute(binding, request.name, _py_ints(request.a), _py_ints(request.p), _py_floats(request.f))
        return
    var snapshot = binding.downcast_value_ptr[CompiledBindingSnapshot]()
    _ = neural_descriptor_execute(snapshot[].provider, xn_operation_id(request.name), request.a, request.p,
                                  request.f)


def _pending_value() raises -> PythonObject:
    """A new reference to the value of the Python exception `family_call`
    stashed, or None. The stash is kept."""
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    var address = pending[].words[1]
    if address == 0:
        return PythonObject(None)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("Py_NewRef")(address)
    return PythonObject(from_owned=pointer)


def _drop_pending() raises:
    """Release the stashed exception: the caller raises its own instead."""
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[NoneType]("Py_DecRef")
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    for t in range(3):  # small-loop(the stashed exception's three references)
        var address = pending[].words[t]
        if address:
            release(address)
            pending[].words[t] = 0


def _p2m_relabel(binding: PythonObject, lab: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._p2m_relabel(est, lab)`: `(count, labels)`
    renumbered by first occurrence (`xn_p2m_relabel`), or None when a label
    falls outside [0, n)."""
    var n = Int(py=lab.shape[0])
    var out = family_array(Python.tuple(max(n, 1)), "<i4", context)
    var info = family_array(Python.tuple(2), "<i4", context)
    _op(binding, xn_relabel_request(_addr(lab), _addr(out), _addr(info), n))
    _ = lab  # keep the Python object alive through the native call above
    var got = xn_relabel_decode(_addr(info))
    _ = info  # keep the Python object alive through the read above
    if not got[1]:
        return PythonObject(None)
    return Python.tuple(PythonObject(got[0]), out)


def _adjacency(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._adjacency`: float32 (`_f32(A, "adjacency")`),
    then the square refusal."""
    var a = admit_family(source, 2, "adjacency", context, dense=True).array
    xn_graph_square_check(Int(py=a.shape[0]), Int(py=a.shape[1]))
    return a


struct _Csr(Movable):
    """`_csr_of` / `_csr_graph_of`'s answer: the int32 lists, the float32
    weights (None for an (indptr, indices, n) tuple or for `_csr_of`) and
    n; `present` False for a dense input."""
    var present: Bool
    var indptr: PythonObject
    var indices: PythonObject
    var vals: PythonObject
    var n: Int

    def __init__(out self, present: Bool, indptr: PythonObject, indices: PythonObject, vals: PythonObject, n: Int):
        self.present = present
        self.indptr = indptr
        self.indices = indices
        self.vals = vals
        self.n = n


def _csr_of_triples(triples: PythonObject, n: Int, graph: Bool, context: PythonObject) raises -> PythonObject:
    """`(indptr, indices, data)` of the square `n` x `n` matrix whose
    stored `(row, col, data, shape)` triples are given, as SciPy's
    `tocsr()` builds it: the entries grouped by row in ascending column and
    repeated (row, col) entries merged, their weights summed in stored order
    (core/sparse_coo_csr_impl.mojo `sparse_coo_to_csr_summed`, parallel);
    `data` is None unless `graph`."""
    var rows = _as_i32(triples[0], "rows", context)
    var cols = _as_i32(triples[1], "cols", context)
    var nnz = Int(py=rows.shape[0])
    xn_graph_triples_check(nnz, Int(py=cols.shape[0]))
    var src_vals = PythonObject(None)
    if graph:
        src_vals = admit_family(triples[2], 1, "data", context).array
        xn_graph_weights_check(Int(py=src_vals.shape[0]), nnz)
    var indptr = family_array(Python.tuple(n + 1), "<i4", context)
    # The merged entries land in native scratch sized for the stored count,
    # then the m that remain are copied into exact-size arrays.
    var merged = xn_graph_merge_triples(n, nnz, _addr(rows), _addr(cols), _addr(src_vals) if graph else 0,
                                        _addr(indptr), graph)
    _ = rows  # keep the Python object alive through the native call above
    _ = cols  # keep the Python object alive through the native call above
    _ = src_vals  # keep the Python object alive through the native call above
    var m = merged.count
    var indices = family_array(Python.tuple(m), "<i4", context)
    var vals = family_array(Python.tuple(m), "<f4", context) if graph else PythonObject(None)
    xn_graph_merged_copy(merged, _addr(indices), _addr(vals) if graph else 0)
    _ = merged^  # keep the scratch alive through the copies above
    return Python.tuple(indptr, indices, vals)


def _csr_of_dense(dense: PythonObject, n: Int, graph: Bool, context: PythonObject) raises -> PythonObject:
    """`(indptr, indices, data)` of the nonzero entries of a natively
    densified square float32 adjacency (a BSR / DIA / DOK / LIL container,
    `host_sparse_dense`, which sums duplicates as `tocsr()` does), row by
    row in ascending column (core/sparse_coo_csr_impl.mojo, parallel);
    `data` is None unless `graph`."""
    var indptr = family_array(Python.tuple(n + 1), "<i4", context)
    var address = _addr(dense)
    var m = xn_graph_dense_counts(n, address, _addr(indptr))
    var indices = family_array(Python.tuple(m), "<i4", context)
    var vals = family_array(Python.tuple(m), "<f4", context) if graph else PythonObject(None)
    xn_graph_dense_fill(n, address, _addr(indptr), _addr(indices), _addr(vals) if graph else 0, m)
    _ = dense  # keep the Python object alive through the native calls above
    return Python.tuple(indptr, indices, vals)


def _duck_attr(source: PythonObject, name: String) raises -> PythonObject:
    """`getattr(source, name, None)`: one CPython attribute read (no method
    of the object is called)."""
    return py_call(_b().getattr, source, PythonObject(name), PythonObject(None))


def _duck_array(value: PythonObject, context: PythonObject) raises -> Bool:
    """Whether `value` is exactly a NumPy ndarray or a mojolearn Array."""
    if value is None:
        return False
    var cls = py_type(value)
    var ndarray = py_get(context, "ndarray")
    var array_class = py_get(context, "array_class")
    return (ndarray is not None and cls is ndarray) or (array_class is not None and cls is array_class)


def _duck_kind(source: PythonObject, context: PythonObject) raises -> Int:
    """1 when `source` (not a SciPy container) exposes `shape` and an
    `indptr` array (CSR), 2 when it exposes `shape` and `row` and `col`
    arrays (COO), else 0 (a dense input). Attribute reads only; the arrays
    must be exact NumPy ndarrays or mojolearn Arrays."""
    if _duck_array(source, context) or _duck_attr(source, "shape") is None:
        return 0
    if _duck_array(_duck_attr(source, "indptr"), context):
        return 1
    if _duck_array(_duck_attr(source, "row"), context) and _duck_array(_duck_attr(source, "col"), context):
        return 2
    return 0


def _csr_lists(source: PythonObject, graph: Bool, context: PythonObject) raises -> _Csr:
    """`_csr_of(A)` (graph False) or `_csr_graph_of(A)` (graph True): a
    tuple `(indptr, indices, n)`, an exact SciPy sparse matrix of any format
    read from its stored buffers (CSR as stored; CSC / COO grouped by row
    with duplicates summed, `_csr_of_triples`; BSR / DIA / DOK / LIL
    densified natively, `_csr_of_dense`), or any other object exposing
    `shape` with CSR (`indptr`, `indices`, `data`) or COO (`row`, `col`,
    `data`) arrays, read as attributes; the lists as int32 (`as_i32_c`), the
    weights as float32 (`as_f32_c(M.data, ndim=1, name="data")`). No method
    of the input (`tocsr()`) is called (P1, audit A #3)."""
    var b = _b()
    var indptr = PythonObject(None)
    var indices = PythonObject(None)
    var vals = PythonObject(None)
    var n = 0
    # An exact SciPy CSR container is read from its own stored buffers
    # (the instance dictionary, bindings/host_sparse_native.mojo
    # _sparse_fields): its `tocsr()` returns itself and `shape` is the
    # stored `_shape`, so the same arrays reach the kernels without running
    # SciPy's Python methods inside the native call. A CSC or COO container
    # is grouped by row natively from its stored triples; BSR / DIA / DOK /
    # LIL are densified natively from their stored buffers.
    var stored_csr = False
    var format = host_sparse_format(source)
    if format == "csr":
        var fields = _sparse_fields(source)
        var stored_shape = py_get(fields, "_shape")
        if (stored_shape is not None and py_type(stored_shape) is b.tuple and len(stored_shape) == 2
                and py_contains(fields, PythonObject("indptr")) and py_contains(fields, PythonObject("indices"))
                and py_contains(fields, PythonObject("data"))):
            if _ne(stored_shape[0], stored_shape[1]):
                raise Error("the adjacency matrix must be square")
            stored_csr = True
            indptr = fields["indptr"]
            indices = fields["indices"]
            n = Int(py=_int(stored_shape[0]))
            if graph:
                vals = admit_family(fields["data"], 1, "data", context).array
    if stored_csr:
        pass
    elif family_isinstance(source, b.tuple) and Int(py=family_call(b.len, source)) == 3:
        indptr = source[0]
        indices = source[1]
        n = Int(py=_int(source[2]))
    elif format.byte_length() > 0:
        # TOMBSTONE: P1_NEIGHBORS_GRAPH_TOCSR (removed: the caller's own tocsr(), SciPy Python code, inside the native graph call) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
        # Restore: experiments/removed/P1_NEIGHBORS_GRAPH_TOCSR.patch.
        # i2 correction: CSC / COO duplicates are summed as `tocsr()` summed
        # them, and BSR / DIA / DOK / LIL (or a CSR / CSC / COO whose stored
        # buffers are not well formed) are converted natively, not refused:
        # `host_sparse_dense` (the path spectral takes for them) sums
        # duplicates from the stored buffers, then the nonzeros by row.
        var triples = cfb_sparse_triples(source, context)
        var grouped: PythonObject
        if triples is not None:
            var shape = triples[3]
            if _ne(shape[0], shape[1]):
                raise Error("the adjacency matrix must be square")
            n = Int(py=_int(shape[0]))
            grouped = _csr_of_triples(triples, n, graph, context)
        else:
            var dense = host_sparse_dense(source, format, context)
            var shape = dense.shape
            if _ne(shape[0], shape[1]):
                raise Error("the adjacency matrix must be square")
            n = Int(py=_int(shape[0]))
            grouped = _csr_of_dense(dense, n, graph, context)
        indptr = grouped[0]
        indices = grouped[1]
        vals = grouped[2]
    elif _duck_kind(source, context) != 0:
        # i2 correction: a sparse object that is not a SciPy container (it
        # used to reach its own `tocsr()`) is read from its CSR or COO arrays
        # by attribute; a COO is grouped by row with duplicates summed.
        var shape = _duck_attr(source, "shape")
        if (py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int
                or py_type(shape[1]) is not b.int):
            raise Error("TypeError: a sparse adjacency's shape must be a tuple of two ints")
        if _ne(shape[0], shape[1]):
            raise Error("the adjacency matrix must be square")
        n = Int(py=_int(shape[0]))
        if _duck_kind(source, context) == 1:
            indptr = _duck_attr(source, "indptr")
            indices = _duck_attr(source, "indices")
            if indices is None:
                raise Error("TypeError: a CSR adjacency must expose indptr, indices and data arrays")
            if graph:
                var data = _duck_attr(source, "data")
                if data is None:
                    raise Error("TypeError: a CSR adjacency must expose indptr, indices and data arrays")
                vals = admit_family(data, 1, "data", context).array
        else:
            var data = _duck_attr(source, "data") if graph else PythonObject(None)
            if graph and data is None:
                raise Error("TypeError: a COO adjacency must expose row, col and data arrays")
            var grouped = _csr_of_triples(Python.tuple(_duck_attr(source, "row"), _duck_attr(source, "col"), data,
                                                       shape), n, graph, context)
            indptr = grouped[0]
            indices = grouped[1]
            vals = grouped[2]
    else:
        return _Csr(False, PythonObject(None), PythonObject(None), PythonObject(None), 0)
    var ip = _as_i32(indptr, "indptr", context)
    var ix = _as_i32(indices, "indices", context)
    var indptr_count = Int(py=ip.shape[0])
    # indptr[n] is read only once indptr holds n + 1 entries.
    xn_graph_csr_check(indptr_count, n, graph, _addr(ip) if indptr_count == n + 1 and graph else 0,
                       Int(py=ix.shape[0]) if graph else 0,
                       Int(py=vals.shape[0]) if graph and vals is not None else -1)
    _ = ip  # keep the Python object alive through the read above
    return _Csr(True, ip, ix, vals, n)


# ---------------------------------------------------------------- PageRank


def _unit(binding: PythonObject, v: PythonObject, n: Int, what: String,
          context: PythonObject) raises -> PythonObject:
    """`PageRank._unit(v, n, what)`: a caller's length-n non-negative vector
    divided by its sum on the device (`x_neighbors_unit_ff`)."""
    var b = _b()
    var unit_fn = PythonObject(None)
    var available = False
    if xn_is_module(binding):
        unit_fn = py_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_unit_ff"), PythonObject(None))
        available = unit_fn is not None
    else:
        available = (xn_capabilities(binding) & 4) != 0
    if not available:
        xn_unit_missing(what)
    var wv = admit_family(v, 1, what, context).array
    xn_unit_length_check(Int(py=wv.shape[0]), n, what)
    var out = family_array(Python.tuple(n), "<f4", context)
    var uinfo = family_array(Python.tuple(2), "<i4", context)
    var request = xn_unit_request(_addr(wv), _addr(out), _addr(uinfo), n)
    if xn_is_module(binding):
        _ = family_call(unit_fn, _py_ints(request.a), _py_ints(request.p), _py_floats(request.f))
    else:
        _op(binding, request)
    _ = wv  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    xn_unit_check(_addr(uinfo), what)
    _ = uinfo  # keep the Python object alive through the read above
    return out


def _uniform(binding: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """PageRank.fit's `uniform()`: [1/n] * n, 1/n in IEEE double handed to
    `xn_p2m_fill` (which rounds it once to float32)."""
    var u = family_array(Python.tuple(n), "<f4", context)
    if n != 0:
        _op(binding, xn_fill_request(_addr(u), n))
        _ = u  # keep the Python object alive through the native call above
    return u


def _pagerank_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`PageRank.fit(A)`."""
    classical_connect_numpy(context)
    var b = _b()
    var binding = args[0]
    var fields = py_fields(model)
    var weight = _attr(model, "weight")
    var binary = weight is None or (py_type(weight) is b.bool and not Bool(py=weight))
    var csr = _csr_lists(args[1], True, context)
    var a = PythonObject(None)
    var n: Int
    if csr.present:
        n = csr.n
    else:
        a = _adjacency(args[1], context)
        n = Int(py=a.shape[0])
    var info = family_array(Python.tuple(2), "<i4", context)
    # `n * float(tol)`: its IEEE words are RUN's (xn_pagerank_threshold).
    var tol = _float_of(model, "tol")
    var personalization = _attr(model, "personalization")
    var dangling = _attr(model, "dangling")
    var nstart = _attr(model, "nstart")
    var x: PythonObject
    if csr.present:
        var p_uniform = personalization is None
        var p = family_array(Python.tuple(n), "<f4", context) if p_uniform else _unit(
            binding, personalization, n, "personalization", context)
        var dw_uniform = dangling is None and p_uniform
        var dw = p if dangling is None else _unit(binding, dangling, n, "dangling", context)
        var x_uniform = nstart is None
        x = family_array(Python.tuple(n), "<f4", context) if x_uniform else _unit(
            binding, nstart, n, "nstart", context)
        var has_vals = csr.vals is not None
        var vals = csr.vals if has_vals else family_array(Python.tuple(1), "<f4", context)
        var max_iter = Int(py=_int(_attr(model, "max_iter")))
        var alpha = _float_of(model, "alpha")
        _op(binding, xn_pagerank_csr_request(_addr(csr.indptr), _addr(csr.indices), _addr(vals), _addr(x), _addr(p),
                                             _addr(dw), _addr(info), n, Int(py=csr.indices.shape[0]), has_vals,
                                             max_iter, tol, binary, x_uniform, p_uniform, dw_uniform, alpha))
        _ = vals  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        _ = dw  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
    else:
        var p = _uniform(binding, n, context) if personalization is None else _unit(
            binding, personalization, n, "personalization", context)
        var dw = p if dangling is None else _unit(binding, dangling, n, "dangling", context)
        # `x.copy()` of a vector this call made: a fresh Array either way.
        x = _uniform(binding, n, context) if nstart is None else _unit(binding, nstart, n, "nstart", context)
        var max_iter = Int(py=_int(_attr(model, "max_iter")))
        var alpha = _float_of(model, "alpha")
        _op(binding, xn_pagerank_dense_request(_addr(a), _addr(x), _addr(p), _addr(dw), _addr(info), n, max_iter,
                                               tol, binary, alpha))
        _ = a  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        _ = dw  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
    _ = csr^
    var got = xn_pagerank_decode(_addr(info))
    _ = info  # keep the Python object alive through the read above
    if got[1]:
        py_setitem(fields, "pagerank_", x)
        py_setitem(fields, "n_iter_", PythonObject(got[0]))
        return
    xn_pagerank_converged_check(got[1], family_str(_attr(model, "max_iter")))


# ---------------------------------------------------------------- connected components


def _cc_relabel(binding: PythonObject, lab: PythonObject, return_labels: Bool,
                context: PythonObject) raises -> PythonObject:
    """`_cc_relabel(lab, return_labels, est)`."""
    if Int(py=lab.shape[0]) == 0:
        return Python.tuple(PythonObject(0), lab) if return_labels else PythonObject(0)
    var got = _p2m_relabel(binding, lab, context)
    xn_cc_relabel_check(got is not None)
    return got if return_labels else got[0]


def _p2m_iota(binding: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_p2m_iota(est, n)`: [0, 1, ..., n - 1] as int32 (`xn_p2m_iota`)."""
    var lab = family_array(Python.tuple(max(n, 0)), "<i4", context)
    if n > 0:
        _op(binding, xn_iota_request(_addr(lab), n))
        _ = lab  # keep the Python object alive through the native call above
    return lab


def _connected_components(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`connected_components(A, directed, connection, return_labels,
    numeric_mode)`: args = (binding, A, directed, connection,
    return_labels)."""
    classical_connect_numpy(context)
    var binding = args[0]
    var connection = args[3]
    if _truth(args[2]) and _eq(connection, PythonObject("strong")):
        raise Error("NotImplementedError: connected_components: connection='strong' is not implemented")
    if not py_contains(Python.tuple(PythonObject("weak"), PythonObject("strong")), connection):
        raise Error("connection must be 'weak' or 'strong'")
    var csr = _csr_lists(args[1], False, context)
    if csr.present:
        var n = csr.n
        var lab = _p2m_iota(binding, n, context)
        var info = family_array(Python.tuple(2), "<i4", context)
        _op(binding, xn_cc_csr_request(_addr(csr.indptr), _addr(csr.indices), _addr(lab), _addr(info), n,
                                       Int(py=csr.indices.shape[0])))
        _ = lab  # keep the Python object alive through the native call above
        _ = csr^
        var components = xn_cc_csr_components(_addr(info))
        _ = info  # keep the Python object alive through the read above
        if components >= 0:
            if _truth(args[4]):
                return Python.tuple(PythonObject(components), lab)
            return PythonObject(components)
        return _cc_relabel(binding, lab, _truth(args[4]), context)
    _ = csr^
    var a = _adjacency(args[1], context)
    var n = Int(py=a.shape[0])
    var lab = _p2m_iota(binding, n, context)
    var info = family_array(Python.tuple(1), "<i4", context)
    _op(binding, xn_cc_dense_request(_addr(a), _addr(lab), _addr(info), n))
    _ = a  # keep the Python object alive through the native call above
    _ = lab  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    return _cc_relabel(binding, lab, _truth(args[4]), context)


# ---------------------------------------------------------------- Louvain


def _louvain_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`Louvain.fit(A)`."""
    classical_connect_numpy(context)
    var binding = args[0]
    var fields = py_fields(model)
    var max_level = _attr(model, "max_level")
    var ml = PythonObject(0) if max_level is None else _int(max_level)
    if max_level is not None and _truth(family_call(native_loaded_module("operator").lt, ml, PythonObject(1))):
        raise Error("max_level must be a positive integer or None")
    var csr = _csr_lists(args[1], True, context)
    var n: Int
    var labels: PythonObject
    var info: PythonObject
    if csr.present:
        n = csr.n
        var has_vals = csr.vals is not None
        var vals = csr.vals if has_vals else family_array(Python.tuple(1), "<f4", context)
        labels = family_array(Python.tuple(n), "<i4", context)
        info = family_array(Python.tuple(2), "<f4", context)
        try:
            var resolution = _float_of(model, "resolution")
            var threshold = _float_of(model, "threshold")
            _op(binding, xn_louvain_csr_request(_addr(csr.indptr), _addr(csr.indices), _addr(vals), _addr(labels),
                                                _addr(info), n, Int(py=csr.indices.shape[0]), has_vals,
                                                Int(py=ml), resolution, threshold))
            _ = vals  # keep the Python object alive through the native call above
            _ = labels  # keep the Python object alive through the native call above
            _ = info  # keep the Python object alive through the native call above
        except error:
            # `except Exception as e`: the device checks' messages as
            # ValueErrors, from "Louvain:" on; anything else unchanged.
            if String(error) != PYTHON_ERROR_PENDING:
                raise error^
            var value = _pending_value()
            var message = String("") if value is None else family_str(value)
            var refusal = xn_louvain_device_message(message)
            if refusal.byte_length() == 0:
                raise error^
            _drop_pending()
            raise Error(refusal)
    else:
        var a = _adjacency(args[1], context)
        n = Int(py=a.shape[0])
        # The symmetry and no-edge checks run natively (`xn_graph_symmetry`).
        var flags = family_array(Python.tuple(2), "<i4", context)
        _op(binding, xn_graph_symmetry_request(_addr(a), _addr(flags), n))
        _ = a  # keep the Python object alive through the native call above
        xn_louvain_symmetry_check(_addr(flags))
        _ = flags  # keep the Python object alive through the read above
        labels = family_array(Python.tuple(n), "<i4", context)
        info = family_array(Python.tuple(2), "<f4", context)
        var resolution = _float_of(model, "resolution")
        var threshold = _float_of(model, "threshold")
        _op(binding, xn_louvain_request(_addr(a), _addr(labels), _addr(info), n, Int(py=ml), resolution, threshold))
        _ = a  # keep the Python object alive through the native call above
        _ = labels  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
    _ = csr^
    var got = _p2m_relabel(binding, labels, context)
    xn_louvain_relabel_check(got is not None)
    var decoded = xn_louvain_decode(_addr(info))
    _ = info  # keep the Python object alive through the read above
    py_setitem(fields, "n_communities_", got[0])
    py_setitem(fields, "labels_", got[1])
    py_setitem(fields, "modularity_", PythonObject(decoded[0]))
    py_setitem(fields, "n_levels_", _int(PythonObject(decoded[1])))


# ---------------------------------------------------------------- SVGP


def _svgp_gamma(model: PythonObject) raises -> Float64:
    """`SVGP._gamma_value()`: 1 / (2 * ls * ls) in IEEE double, ls ** 2 as
    one product."""
    var ls = _float(_attr(model, "lengthscale"))
    # The interpreter's minor version picks its own wording for the float
    # division by zero the former `1.0 / (2 * ls * ls)` raised; it is read
    # only on that refusal path, as before the door split.
    var python_minor = 0
    if xn_svgp_zero_denominator(ls):
        python_minor = Int(py=native_loaded_module("sys").version_info[1])
    return xn_svgp_gamma(ls, python_minor)


def _svgp_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`SVGP.fit(X, y)`."""
    classical_connect_numpy(context)
    var b = _b()
    var binding = args[0]
    var fields = py_fields(model)
    var x = admit_family(args[1], 2, "X", context, dense=True).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var yv = admit_family(args[2], 1, "y", context).array
    xn_rows_match_check(Int(py=yv.shape[0]), n)
    var inducing = _attr(model, "inducing_points")
    var z: PythonObject
    if inducing is not None:
        z = ann_keep_array(inducing, admit_family(inducing, 2, "inducing_points", context, dense=True).array, context)
    else:
        var m_obj = family_call(b.min, _int(_attr(model, "n_inducing")), PythonObject(n))
        if _truth(family_call(native_loaded_module("operator").le, m_obj, PythonObject(0))):
            # `as_i32_c` of the empty row list.
            raise Error("mojolearn: rows is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) "
                        + "while a minimum of 1 is required.")
        var m = Int(py=m_obj)
        var rows = family_array(Python.tuple(m), "<i4", context)
        xn_svgp_even_rows(_addr(rows), n, m)
        _ = rows  # keep the Python object alive through the native call above
        z = family_array(Python.tuple(m, d), "<f4", context)
        _op(binding, xn_take_rows_request(_addr(x), _addr(rows), _addr(z), m, d, n))
        _ = x  # keep the Python object alive through the native call above
        _ = rows  # keep the Python object alive through the native call above
        _ = z  # keep the Python object alive through the native call above
    var mz = Int(py=z.shape[0])
    var alpha = family_array(Python.tuple(mz), "<f4", context)
    var c = family_array(Python.tuple(mz, mz), "<f4", context)
    var qmu = family_array(Python.tuple(mz), "<f4", context)
    var qsqrt = family_array(Python.tuple(mz, mz), "<f4", context)
    var info = family_array(Python.tuple(2), "<f4", context)
    var gamma = _svgp_gamma(model)
    var kernel_variance = _float_of(model, "kernel_variance")
    var noise_variance = _float_of(model, "noise_variance")
    var jitter = _float_of(model, "jitter")
    _op(binding, xn_svgp_fit_request(_addr(x), _addr(z), _addr(yv), _addr(alpha), _addr(c), _addr(qmu), _addr(qsqrt),
                                     _addr(info), n, mz, d, gamma, kernel_variance, noise_variance, jitter))
    _ = x  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = yv  # keep the Python object alive through the native call above
    _ = alpha  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = qmu  # keep the Python object alive through the native call above
    _ = qsqrt  # keep the Python object alive through the native call above
    var fitted = xn_svgp_fit_decode(_addr(info))
    _ = info  # keep the Python object alive through the read above
    var elbo = fitted[0]
    py_setitem(fields, "precision_", PythonObject("float-float"))
    xn_svgp_definite_check(fitted[1])
    py_setitem(fields, "Z_", z)
    py_setitem(fields, "_alpha", alpha)
    py_setitem(fields, "_C", c)
    py_setitem(fields, "q_mu_", qmu)
    py_setitem(fields, "q_sqrt_", qsqrt)
    py_setitem(fields, "elbo_", PythonObject(elbo))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _svgp_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict(X)`: the mean of `predict_f(X)` (its variance is computed
    as `predict_f` computed it)."""
    return _svgp_predict_f(model, args, context)[0]


def _svgp_predict_y(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict_y(X)`: `predict_f`'s mean, and its variance plus
    `noise_variance` (`_unary(var, _U_IDENTITY, 1.0, f32(noise_variance))`)."""
    var latent = _svgp_predict_f(model, args, context)
    var var_ = latent[1]
    var size = Int(py=var_.size)
    var out = family_array(Python.tuple(size), "<f4", context)
    _op(args[0], xn_unary_shift_request(_addr(var_), _addr(out), size, _float_of(model, "noise_variance")))
    _ = var_  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return Python.tuple(latent[0], out)


def _svgp_predict_f(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict_f(X)`: (mean, var) of the latent function."""
    classical_connect_numpy(context)
    var binding = args[0]
    var q = admit_family(args[1], 2, "X", context, dense=True).array
    var z = _attr(model, "Z_")
    var m = Int(py=z.shape[0])
    var nq = Int(py=q.shape[0])
    var d = Int(py=q.shape[1])
    var old_items = _truth(context["family_svc_xn_old_items"])
    var alpha = _attr(model, "_alpha")
    var c = _attr(model, "_C")
    if not old_items and nq != 0:
        # the fused chain: Ksu, the mean and the variance per row tile
        # (`xn_svgp_predict`)
        var mean = family_array(Python.tuple(nq), "<f4", context)
        var var_ = family_array(Python.tuple(nq), "<f4", context)
        var gamma = _svgp_gamma(model)
        var kernel_variance = _float_of(model, "kernel_variance")
        _op(binding, xn_svgp_predict_request(_addr(q), _addr(z), _addr(alpha), _addr(c), _addr(mean), _addr(var_),
                                             nq, m, d, gamma, kernel_variance))
        _ = q  # keep the Python object alive through the native call above
        _ = z  # keep the Python object alive through the native call above
        _ = alpha  # keep the Python object alive through the native call above
        _ = c  # keep the Python object alive through the native call above
        _ = var_  # keep the Python object alive through the native call above
        _ = mean  # keep the Python object alive through the native call above
        return Python.tuple(mean, var_)
    # `_k(Q, Z_)`: the rbf kernel, then the kernel variance scale.
    var g = _svgp_gamma(model)
    var k = family_array(Python.tuple(nq, m), "<f4", context)
    _op(binding, xn_kernel_rbf_request(_addr(q), _addr(z), _addr(k), nq, m, d, g))
    _ = q  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = k  # keep the Python object alive through the native call above
    var ksu = family_array(Python.tuple(nq, m), "<f4", context)
    _op(binding, xn_unary_scale_request(_addr(k), _addr(ksu), nq * m, _float_of(model, "kernel_variance")))
    _ = k  # keep the Python object alive through the native call above
    _ = ksu  # keep the Python object alive through the native call above
    # `matmul(Ksu, alpha.reshape((M, 1))).reshape((nq,))`: an (nq, 1) product
    # written into the (nq,) vector it is reshaped to.
    var mean = family_array(Python.tuple(nq), "<f4", context)
    _op(binding, xn_matmul_request(_addr(ksu), _addr(alpha), _addr(mean), nq, m, 1))
    _ = ksu  # keep the Python object alive through the native call above
    _ = alpha  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    var var_ = family_array(Python.tuple(nq), "<f4", context)
    _op(binding, xn_svgp_var_request(_addr(ksu), _addr(c), _addr(var_), nq, m, _float_of(model, "kernel_variance")))
    _ = ksu  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = var_  # keep the Python object alive through the native call above
    return Python.tuple(mean, var_)


def _svgp_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`RegressorScoreMixin.score(X, y, sample_weight)` as SVGP inherited
    it: y as float32 (`as_f32_c(y, ndim=None, name="y")`), the prediction,
    then `metrics.r2_score(y, pred, sample_weight=..., numeric_mode=...)`
    through the metrics owner; args = (binding, X, y, sample_weight[, the
    metrics binding]). Without a metrics binding the model's numeric_mode
    selects the import-connected one here, as `_get_binding` selected it."""
    classical_connect_numpy(context)
    var truth = admit_family(args[2], -1, "y", context).array
    var prediction = _svgp_predict(model, args, context)
    var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
    var metrics = args[4] if len(args) > 4 and args[4] is not None else estimator_binding(
        model, "_mojolearn_metrics", context)
    return mcls_cluster_operation("family_mcls_r2", Python.tuple(metrics, truth, prediction, args[3],
                                  PythonObject("uniform_average"), PythonObject(True), mode), context)


def _neighbors_graph_operation_route(model: PythonObject, operation: String, var args: PythonObject,
                                     context: PythonObject) raises -> PythonObject:
    if (operation == "family_xn_graph_connected_components" and len(args) > 0
            and py_type(args[0]) is _b().str):
        # connected_components(..., numeric_mode=<str>): the shell hands over
        # the requested mode in the binding slot (None was resolved to the
        # default-mode binding by the family door); select its binding here.
        var selected = Python.list(estimator_binding_mode(args[0], "_mojolearn_x_neighbors", context))
        for i in range(1, len(args)):  # driver-loop: over fixed public function arguments
            _ = py_call(selected.append, args[i])
        args = selected
    args = xn_admit_arguments(args, context)
    if operation == "family_xn_graph_pagerank_fit":
        _pagerank_fit(model, args, context)
        return model
    if operation == "family_xn_graph_louvain_fit":
        _louvain_fit(model, args, context)
        return model
    if operation == "family_xn_graph_louvain_fit_predict":
        _louvain_fit(model, args, context)
        return _attr(model, "labels_")
    if operation == "family_xn_graph_connected_components":
        return _connected_components(args, context)
    if operation == "family_xn_graph_svgp_fit":
        _svgp_fit(model, args, context)
        return model
    if operation == "family_xn_graph_svgp_predict":
        return _svgp_predict(model, args, context)
    if operation == "family_xn_graph_svgp_predict_f":
        return _svgp_predict_f(model, args, context)
    if operation == "family_xn_graph_svgp_predict_y":
        return _svgp_predict_y(model, args, context)
    if operation == "family_xn_graph_svgp_score":
        return _svgp_score(model, args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_graph_operation(model: PythonObject, operation: String, args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner."""
    var lifecycle = family_lifecycle_spec(operation)
    if lifecycle.method.byte_length() == 0:
        return _neighbors_graph_operation_route(model, operation, args, context)
    var call = family_lifecycle_begin(model, operation, lifecycle, args, 1, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_graph_operation_route(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
