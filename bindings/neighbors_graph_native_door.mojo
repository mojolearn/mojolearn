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
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit_arguments
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_unpack import family_isinstance, family_str
from bindings.family_common_pack import family_call, family_array
from bindings.cfb_spectral_door import _as_i32
from bindings.host_sparse_native_door import host_sparse_format, _sparse_fields, host_sparse_dense
from bindings.cfb_sparse_read_unpack import cfb_sparse_triples
from bindings.neighbors_ann_native_door import ann_keep_array
from bindings.mcls_cluster_door import mcls_cluster_operation
from bindings.operation_names_mcls import MCLS_OPN_FAMILY_MCLS_R2
from neighbors.neighbors_graph_run import (
    xn_rows_match_check, xn_graph_square_check, xn_graph_triples_check, xn_graph_weights_check,
    xn_graph_merge_triples, xn_graph_merged_copy, xn_graph_dense_counts, xn_graph_dense_fill, xn_graph_csr_check,
    xn_unit_length_check, xn_unit_missing, xn_pagerank_converged_check, xn_svgp_zero_denominator, xn_svgp_gamma,
    xn_svgp_definite_check,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_device
from providers.python.function_device_boundary_door import function_device_admit
from core.function_device import function_device_default
from providers.python.family_public_lifecycle_pyapi_door import family_lifecycle_begin, family_lifecycle_finish
from providers.python.family_public_lifecycle_run import family_lifecycle_spec
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_refusal
from bindings.py_optional import py_or_none, py_present
from std.python._cpython import GILReleased
from bindings.lof_provider_unpack import LofConnection
from neighbors.svgp_run import (
    SVGP_VARIANCE_NONE, SVGP_VARIANCE_LATENT, SVGP_VARIANCE_NOISY, SvgpFitRequest, SvgpQueryRequest, svgp_fit_run,
    svgp_query_run,
)
from neighbors.graph_fit_run import (
    XN_CAPABILITY_UNIT, PageRankRequest, ComponentsRequest, LouvainRequest, LouvainOutcome, xn_capabilities_run,
    pagerank_fit_run, components_run, louvain_fit_run,
)

# Each operation is UNPACK (this file: Python reads, admission, the published
# arrays), RUN (neighbors/graph_fit_run.mojo, neighbors/svgp_run.mojo: typed
# requests on the image's native descriptor entry, no Python object, the GIL
# released once) and PACK (this file: publication). The typed checks, the
# operation requests and the info-word decoding are
# neighbors/neighbors_graph_run.mojo's.


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
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var indices = Optional[PythonObject]()
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var vals = Optional[PythonObject]()
    var n = 0
    # An exact SciPy CSR container is read from its own stored buffers
    # (the instance dictionary, bindings/host_sparse_native_door.mojo
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
            indices = py_present(fields["indices"])
            n = Int(py=_int(stored_shape[0]))
            if graph:
                vals = admit_family(fields["data"], 1, "data", context).array
    if stored_csr:
        pass
    elif family_isinstance(source, b.tuple) and Int(py=family_call(b.len, source)) == 3:
        indptr = source[0]
        indices = py_present(source[1])
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
        indices = py_present(grouped[1])
        vals = py_present(grouped[2])
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
            indices = py_present(_duck_attr(source, "indices"))
            if not indices:
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
            indices = py_present(grouped[1])
            vals = py_present(grouped[2])
    else:
        return _Csr(False, PythonObject(None), PythonObject(None), PythonObject(None), 0)
    var ip = _as_i32(indptr, "indptr", context)
    var ix = _as_i32(py_or_none(indices), "indices", context)
    var indptr_count = Int(py=ip.shape[0])
    # indptr[n] is read only once indptr holds n + 1 entries.
    xn_graph_csr_check(indptr_count, n, graph, _addr(ip) if indptr_count == n + 1 and graph else 0,
                       Int(py=ix.shape[0]) if graph else 0,
                       Int(py=vals.value().shape[0]) if graph and Bool(vals) else -1)
    _ = ip  # keep the Python object alive through the read above
    return _Csr(True, ip, ix, py_or_none(vals), n)


# ---------------------------------------------------------------- PageRank
#
# Door split (lane dw-neighbors-b, 2026-11-21) of PageRank, Louvain and
# connected_components: UNPACK reads the fields, admits the adjacency and
# the caller's vectors and allocates the published array; RUN is
# neighbors/graph_fit_run.mojo (no Python object; one GIL release around
# it); PACK publishes. The image is reached through its native descriptor
# entry, never through a Python export.


def _unit_admit(connection: LofConnection, v: PythonObject, n: Int, what: String,
                context: PythonObject) raises -> PythonObject:
    """UNPACK of `PageRank._unit(v, n, what)`: the image must carry the
    `unit_ff` operation; the caller's vector as float32, of length n."""
    if (xn_capabilities_run(connection.driver) & XN_CAPABILITY_UNIT) == 0:
        xn_unit_missing(what)
    var wv = admit_family(v, 1, what, context).array
    xn_unit_length_check(Int(py=wv.shape[0]), n, what)
    return wv


def _pagerank_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`PageRank.fit(A)`."""
    # UNPACK
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
    # `n * float(tol)`: its IEEE words are RUN's (xn_pagerank_threshold).
    var tol = _float_of(model, "tol")
    var personalization = _attr(model, "personalization")
    var dangling = _attr(model, "dangling")
    var nstart = _attr(model, "nstart")
    var connection = LofConnection(binding)
    var p = PythonObject(None)
    var dw = PythonObject(None)
    var start = PythonObject(None)
    if personalization is not None:
        p = _unit_admit(connection, personalization, n, "personalization", context)
    if dangling is not None:
        dw = _unit_admit(connection, dangling, n, "dangling", context)
    if nstart is not None:
        start = _unit_admit(connection, nstart, n, "nstart", context)
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var alpha = _float_of(model, "alpha")
    var x = family_array(Python.tuple(n), "<f4", context)
    var has_vals = csr.present and csr.vals is not None
    var request = PageRankRequest(csr.present, _addr(csr.indptr) if csr.present else 0,
                                  _addr(csr.indices) if csr.present else 0, _addr(csr.vals) if has_vals else 0,
                                  Int(py=csr.indices.shape[0]) if csr.present else 0,
                                  0 if csr.present else _addr(a), _addr(x), n,
                                  0 if p is None else _addr(p), 0 if dw is None else _addr(dw),
                                  0 if start is None else _addr(start), max_iter, tol, binary, alpha)
    var got = (0, False)
    # RUN
    with GILReleased(Python()):
        got = pagerank_fit_run(connection.driver, request)
    _ = csr^  # keep the admitted lists alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = p  # keep the Python object alive through the native call above
    _ = dw  # keep the Python object alive through the native call above
    _ = start  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    # PACK
    if got[1]:
        py_setitem(fields, "pagerank_", x)
        py_setitem(fields, "n_iter_", PythonObject(got[0]))
        return
    xn_pagerank_converged_check(got[1], family_str(_attr(model, "max_iter")))


# ---------------------------------------------------------------- connected components


def _connected_components(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`connected_components(A, directed, connection, return_labels,
    numeric_mode)`: args = (binding, A, directed, connection,
    return_labels)."""
    # UNPACK
    classical_connect_numpy(context)
    var binding = args[0]
    var connection = args[3]
    if _truth(args[2]) and _eq(connection, PythonObject("strong")):
        raise Error("NotImplementedError: connected_components: connection='strong' is not implemented")
    if not py_contains(Python.tuple(PythonObject("weak"), PythonObject("strong")), connection):
        raise Error("connection must be 'weak' or 'strong'")
    var csr = _csr_lists(args[1], False, context)
    var a = PythonObject(None)
    var n: Int
    if csr.present:
        n = csr.n
    else:
        a = _adjacency(args[1], context)
        n = Int(py=a.shape[0])
    var labels = family_array(Python.tuple(max(n, 0)), "<i4", context)
    var request = ComponentsRequest(csr.present, _addr(csr.indptr) if csr.present else 0,
                                    _addr(csr.indices) if csr.present else 0,
                                    Int(py=csr.indices.shape[0]) if csr.present else 0,
                                    0 if csr.present else _addr(a), _addr(labels), n)
    var image = LofConnection(binding)
    var components = 0
    # RUN
    with GILReleased(Python()):
        components = components_run(image.driver, request)
    _ = csr^  # keep the admitted lists alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    # PACK
    if _truth(args[4]):
        return Python.tuple(PythonObject(components), labels)
    return PythonObject(components)


# ---------------------------------------------------------------- Louvain


def _louvain_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`Louvain.fit(A)`."""
    # UNPACK
    classical_connect_numpy(context)
    var binding = args[0]
    var fields = py_fields(model)
    var max_level = _attr(model, "max_level")
    var ml = PythonObject(0) if max_level is None else _int(max_level)
    if max_level is not None and _truth(family_call(native_loaded_module("operator").lt, ml, PythonObject(1))):
        raise Error("max_level must be a positive integer or None")
    var csr = _csr_lists(args[1], True, context)
    var a = PythonObject(None)
    var n: Int
    if csr.present:
        n = csr.n
    else:
        a = _adjacency(args[1], context)
        n = Int(py=a.shape[0])
    var resolution = _float_of(model, "resolution")
    var threshold = _float_of(model, "threshold")
    # `_p2m_relabel`'s answer: one word for an empty graph, as before.
    var labels = family_array(Python.tuple(max(n, 1)), "<i4", context)
    var has_vals = csr.present and csr.vals is not None
    var request = LouvainRequest(csr.present, _addr(csr.indptr) if csr.present else 0,
                                 _addr(csr.indices) if csr.present else 0, _addr(csr.vals) if has_vals else 0,
                                 Int(py=csr.indices.shape[0]) if csr.present else 0,
                                 0 if csr.present else _addr(a), _addr(labels), n, Int(py=ml), resolution,
                                 threshold)
    var connection = LofConnection(binding)
    var outcome = LouvainOutcome()
    # RUN
    with GILReleased(Python()):
        outcome = louvain_fit_run(connection.driver, request)
    _ = csr^  # keep the admitted lists alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    # PACK
    py_setitem(fields, "n_communities_", PythonObject(outcome.communities))
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "modularity_", PythonObject(outcome.modularity))
    py_setitem(fields, "n_levels_", _int(PythonObject(outcome.levels)))


# ---------------------------------------------------------------- SVGP
#
# Door split (lane dw-neighbors-b, 2026-11-21): UNPACK reads the fields,
# admits X, y and the caller's inducing points and allocates the published
# arrays; RUN is neighbors/svgp_run.mojo (no Python object; one GIL release
# around it); PACK publishes. The kernel helper (`_k(Q, Z_)`: the rbf kernel,
# then the kernel variance scale) is RUN's, on the estimator's image.


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
    # UNPACK
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
    var pick_rows = inducing is None
    if not pick_rows:
        z = ann_keep_array(inducing, admit_family(inducing, 2, "inducing_points", context, dense=True).array, context)
    else:
        var m_obj = family_call(b.min, _int(_attr(model, "n_inducing")), PythonObject(n))
        if _truth(family_call(native_loaded_module("operator").le, m_obj, PythonObject(0))):
            # `as_i32_c` of the empty row list.
            raise Error("mojolearn: rows is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) "
                        + "while a minimum of 1 is required.")
        z = family_array(Python.tuple(Int(py=m_obj), d), "<f4", context)
    var mz = Int(py=z.shape[0])
    var alpha = family_array(Python.tuple(mz), "<f4", context)
    var c = family_array(Python.tuple(mz, mz), "<f4", context)
    var qmu = family_array(Python.tuple(mz), "<f4", context)
    var qsqrt = family_array(Python.tuple(mz, mz), "<f4", context)
    var gamma = _svgp_gamma(model)
    var kernel_variance = _float_of(model, "kernel_variance")
    var noise_variance = _float_of(model, "noise_variance")
    var jitter = _float_of(model, "jitter")
    var request = SvgpFitRequest(_addr(x), _addr(yv), _addr(z), _addr(alpha), _addr(c), _addr(qmu), _addr(qsqrt),
                                 n, mz, d, pick_rows, gamma, kernel_variance, noise_variance, jitter)
    var connection = LofConnection(binding)
    var fitted = (Float64(0.0), False)
    # RUN
    with GILReleased(Python()):
        fitted = svgp_fit_run(connection.driver, request)
    _ = x  # keep the Python object alive through the native call above
    _ = yv  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = alpha  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = qmu  # keep the Python object alive through the native call above
    _ = qsqrt  # keep the Python object alive through the native call above
    # PACK
    py_setitem(fields, "precision_", PythonObject("float-float"))
    xn_svgp_definite_check(fitted[1])
    py_setitem(fields, "Z_", z)
    py_setitem(fields, "_alpha", alpha)
    py_setitem(fields, "_C", c)
    py_setitem(fields, "q_mu_", qmu)
    py_setitem(fields, "q_sqrt_", qsqrt)
    py_setitem(fields, "elbo_", PythonObject(fitted[0]))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _svgp_query(model: PythonObject, args: PythonObject, kind: Int,
                context: PythonObject) raises -> PythonObject:
    """The mean (SVGP_VARIANCE_NONE: the variance is computed as `predict_f`
    computed it and dropped), or `(mean, variance)`: the latent function's,
    or with `noise_variance` added (`predict_y`)."""
    # UNPACK
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
    var gamma = _svgp_gamma(model)
    var kernel_variance = _float_of(model, "kernel_variance")
    var noise_variance = Float64(0.0)
    if kind == SVGP_VARIANCE_NOISY:
        noise_variance = _float_of(model, "noise_variance")
    var mean = family_array(Python.tuple(nq), "<f4", context)
    var variance = PythonObject(None)
    var variance_at = 0
    if kind != SVGP_VARIANCE_NONE:
        variance = family_array(Python.tuple(nq), "<f4", context)
        variance_at = _addr(variance)
    var request = SvgpQueryRequest(_addr(q), _addr(z), _addr(alpha), _addr(c), _addr(mean), variance_at, nq, m, d,
                                   gamma, kernel_variance, noise_variance, kind, old_items)
    var connection = LofConnection(binding)
    # RUN
    with GILReleased(Python()):
        svgp_query_run(connection.driver, request)
    _ = q  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = alpha  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    # PACK
    if kind == SVGP_VARIANCE_NONE:
        return mean
    return Python.tuple(mean, variance)


def _svgp_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict(X)`: the mean of `predict_f(X)`."""
    return _svgp_query(model, args, SVGP_VARIANCE_NONE, context)


def _svgp_predict_y(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict_y(X)`: `predict_f`'s mean, and its variance plus
    `noise_variance`."""
    return _svgp_query(model, args, SVGP_VARIANCE_NOISY, context)


def _svgp_predict_f(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict_f(X)`: (mean, var) of the latent function."""
    return _svgp_query(model, args, SVGP_VARIANCE_LATENT, context)


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
    return mcls_cluster_operation(MCLS_OPN_FAMILY_MCLS_R2, Python.tuple(metrics, truth, prediction, args[3],
                                  PythonObject("uniform_average"), PythonObject(True), mode), context)


# connected_components has a device image (`_mojolearn_x_neighbors` of the GPU
# set) and its CPU identical image connected beside it at import
# (python/mojolearn/_expansion_neighbors.py `connect_bindings`).
comptime _CC_HAS_GPU = True
comptime _CC_HAS_CPU = True


def _neighbors_graph_operation_route(model: PythonObject, operation: String, var args: PythonObject,
                                     context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_graph_connected_components":
        # connected_components(numeric_mode, A, directed, connection,
        # return_labels, device): the family door resolves no binding for it
        # (bindings/native_operation_codes.mojo `family_route`), so the one
        # lookup is here, after the call's `device` (lane vdf-misc) is
        # admitted in the shared words. None (device and mode) is the same
        # estimator_binding_mode(None, ...) call the family door made; a mode
        # word selects its binding as before; an explicit "cpu" beside a GPU
        # set selects the CPU identical module (estimator_binding_device).
        var device = function_device_default()
        if args[5] is not None:
            device = function_device_admit(args[5], "connected_components", _CC_HAS_GPU, _CC_HAS_CPU, context, _b())
        var selected = Python.list(estimator_binding_device(device, args[0], "_mojolearn_x_neighbors", context))
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
