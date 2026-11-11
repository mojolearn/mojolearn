# SPDX-License-Identifier: Apache-2.0
# Spectral public UNPACK/PACK and staged lifecycle metadata. Fit/embedding
# execute typed native requests through an image-pinned provider. Public
# Arrays remain the output and fitted-state owners. Affinity preparation,
# alternative label assignment and query/cache plumbing are still staged
# below and migrate independently without changing their numerical routes.
from std.math import isinf
from core.spectral_native_api import (
    SPECTRAL_CAP_DENSE, SPECTRAL_CAP_RESIDENT, SPECTRAL_DATASET, SPECTRAL_GRAPH, SPECTRAL_DENSE,
    SpectralInput, SpectralFitOptions, SpectralFitRequest,
    SpectralEmbeddingOptions, SpectralEmbeddingRequest, SpectralQueryShape,
    SpectralQueryRequest, SpectralPrepareRequest, SpectralResidentQueryRequest,
)
from bindings.spectral_native_boundary import (
    SpectralConnection, SpectralStatePins, spectral_fit_run, spectral_embedding_run, spectral_query_run,
)
from std.python import Python, PythonObject
from providers.python.host_estimator_binding_pyapi import host_estimator_binding
from bindings.admitted_family import admit_family, admit_family_f64, admit_family_input
from bindings.admitted_input import admit_typed
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_fields, py_get
from bindings.classical_arrays import classical_address
from core.host_finite_scan import host_all_finite_f32
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_str, family_call, family_array,
    family_field, family_is_str, family_min,
)
from bindings.cluster_native import _reference_queries, _xc_array
from bindings.cluster_resident_table import CRES_SPECTRAL, cres_spectral_lookup, cres_spectral_store, cres_forget
from bindings.spectral_resident_boundary import (
    SpectralResidentLease, spectral_resident_prepare, spectral_resident_query, spectral_resident_retire,
)
from bindings.p3nc_fitted_freeze import fitted_is_freezable, fitted_freeze
from bindings.spectral_umap_routes import (
    spectral_umap_route, spectral_umap_entry as family_entry,
    spectral_umap_optional_entry as cfb_optional_entry,
)
from bindings.cfb_dbscan import cfb_borrow_copy
from bindings.cfb_sparse_read import cfb_sparse_triples
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.native_loaded_module import native_loaded_module
from bindings.family_raw_attr import family_raw_get, family_raw_get_or, family_raw_has
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_estimator_name, classical_check_width
from bindings.classical_exceptions import classical_require_fitted
from bindings.native_array_view import native_array_ndarray_view
from bindings.native_owner_lifetime import native_release_admit
from providers.python.spectral_lifecycle_pyapi import spectral_admit, spectral_input
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal

comptime _DENSE_EMPTY = -1
comptime _DENSE_NONFINITE = -2
comptime _DENSE_NEGATIVE = -3
#: x_decomp/cells.mojo op codes (`_expansion_decomp._OP`).
comptime _OP_EXP = 9
comptime _OP_SCALE = 14
#: `x_cluster/entries.mojo::ENTRY_SPECTRAL_ASSIGN`.
comptime _E_SPECTRAL_ASSIGN = 12


def _builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def _int(value: PythonObject) raises -> Int:
    """`int(value)`, the value's own conversion."""
    return Int(py=family_call(_builtins().int, value))


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`, the value's own conversion."""
    return family_call(_builtins().float, value)


def _shape_of(obj: PythonObject) raises -> PythonObject:
    """`_buffer.shape_of`: `.shape` when the object has one, the nesting of
    a list or tuple otherwise."""
    var b = _builtins()
    var shape = py_call(b.getattr, obj, PythonObject("shape"), PythonObject(None))
    if shape is not None:
        return family_call(b.tuple, shape)
    if Bool(py=family_call(b.isinstance, obj, Python.tuple(b.list, b.tuple))):
        if Bool(py=family_call(b.bool, obj)) and Bool(py=family_call(b.isinstance, obj[0], Python.tuple(b.list, b.tuple))):
            return Python.tuple(family_call(b.len, obj), family_call(b.len, obj[0]))
        return Python.tuple(family_call(b.len, obj))
    if Bool(py=family_call(b.hasattr, obj, PythonObject("__len__"))):
        return Python.tuple(family_call(b.len, obj))
    return Python.tuple()


def _square(shape: PythonObject) raises -> Bool:
    """`len(shape) == 2 and shape[0] == shape[1]`."""
    if len(shape) != 2:
        return False
    return Bool(py=family_call(_builtins().bool, family_call(native_loaded_module("operator").eq, shape[0], shape[1])))


def _as_i32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_i32_c(source, ndim=1, name=name)`: the rank and emptiness
    refusals, then the vector itself when it is already C-order int32, else
    one int32 conversion by Array.astype's rules (truncation toward zero, a
    value outside int32 refused), split across the host's cores."""
    # TOMBSTONE: the family_view + float64 widen + serial cfb_f64_to_i32 cast (core/cfb_impl.mojo) deleted 2026-11-09 by lane a6-device-x-family-classical; code recoverable at f59d913b0b.
    # Restore: experiments/removed/A6_FAMILY_VIEW.patch. The multi-dtype
    # admission (bindings/admitted_input.mojo admit_typed) casts once in parallel.
    return admit_typed(source, "<i4", 1, name, context).keep


# TOMBSTONE: P1_SPECTRAL_TOCOO (removed: the caller's own tocoo() / toarray() inside spectral fit and predict) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_SPECTRAL_TOCOO.patch.
# P1 (audit A #4): _tocoo removed. It looked up and called the caller's
# own `tocoo()` (SciPy's Python conversion, or any object's method) inside
# fit. A sparse input is now an exact SciPy container read natively from its
# stored buffers (_is_sparse, _sparse_triples, _densified); any other object
# takes the dense admission, which refuses what it cannot read.


def _is_sparse(obj: PythonObject) raises -> Bool:
    """Whether `obj` is an exact SciPy sparse container of any format (a
    type identity check; no attribute of the object is read)."""
    return host_sparse_format(obj).byte_length() > 0


def _densified(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """An exact SciPy container whose stored coordinates were not taken
    (BSR, DIA, DOK, LIL, or a COO/CSR/CSC whose storage was not read)
    densified natively to float32 from its stored buffers, duplicates
    summed (`host_sparse_dense`); any other input unchanged."""
    var format = host_sparse_format(source)
    if format.byte_length() == 0:
        return source
    return host_sparse_dense(source, format, context)


struct _Coo(Movable):
    var rows: PythonObject
    var cols: PythonObject
    var vals: PythonObject
    var n: Int

    def __init__(out self, rows: PythonObject, cols: PythonObject, vals: PythonObject, n: Int):
        self.rows = rows
        self.cols = cols
        self.vals = vals
        self.n = n


def _coo_of_triples(rows: PythonObject, cols: PythonObject, vals: PythonObject, shape: PythonObject,
                    who: String, context: PythonObject) raises -> _Coo:
    """`_coo_triples`' sparse arm on (row, col, data) and the shape."""
    if not _square(shape):
        raise Error("mojolearn " + who + ": a precomputed affinity matrix "
                    + "must be square, got shape " + family_repr(shape))
    var r = _as_i32(rows, "rows", context)
    var c = _as_i32(cols, "cols", context)
    var v = admit_family(vals, 1, "vals", context).array
    return _Coo(r, c, v, _int(shape[0]))


def _sparse_triples(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(row, col, data, shape)` of an exact SciPy COO/CSR/CSC input read
    from its stored buffers (bindings/cfb_sparse_read.mojo); None for any
    other input (the caller densifies another SciPy format natively with
    `_densified`). No method of the input is called."""
    return cfb_sparse_triples(source, context)


def _coo_triples(source: PythonObject, who: String, base: PythonObject, context: PythonObject) raises -> _Coo:
    """`_coo_triples(A, who)`: an exact SciPy COO/CSR/CSC input's stored
    triples (read natively), another SciPy format densified natively, a
    dense one through the base
    binding's float64 nonzero scan (exact zeros dropped, as
    `sp.coo_matrix(dense)` drops them)."""
    var triples = _sparse_triples(source, context)
    if triples is not None:
        return _coo_of_triples(triples[0], triples[1], triples[2], triples[3], who, context)
    var dense_source = _densified(source, context)
    var shape = _shape_of(dense_source)
    if not _square(shape):
        raise Error("mojolearn " + who + ": with affinity='precomputed', X "
                    + "must be a square affinity matrix or a SciPy sparse matrix, "
                    + "got shape " + family_repr(shape))
    # The metadata step of the one admission path (rank and emptiness
    # refusals), then the float64 widen of the same view (family_view deleted).
    var snapshot = admit_family_input(dense_source, 2, "X", context)
    var dense = admit_family_f64(snapshot.keep, "X", context)
    var n = _int(shape[0])
    var size = Int(py=dense.size)
    var nnz = 0
    if size:
        nnz = Int(py=family_call(family_entry(base, "nonzero_f64_count", context), classical_address(dense),
                                 PythonObject(size)))
        _ = dense  # keep the Python object alive through the native call above
    var rows = family_array(Python.tuple(nnz), "<i4", context)
    var cols = family_array(Python.tuple(nnz), "<i4", context)
    var vals = family_array(Python.tuple(nnz), "<f4", context)
    if nnz == 0:
        return _Coo(rows, cols, vals, n)
    var wrote = Int(py=family_call(family_entry(base, "nonzero_f64_fill", context), classical_address(dense),
                                   PythonObject(n), PythonObject(n),
                                   Python.list(classical_address(rows), classical_address(cols), classical_address(vals)),
                                   PythonObject(nnz)))
    _ = dense  # keep the Python object alive through the native call above
    _ = vals  # keep the Python object alive through the native call above
    _ = cols  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = dense^
    if wrote != nnz:
        raise Error("RuntimeError: mojolearn " + who + ": nonzero_f64_fill wrote "
                    + String(wrote) + " of " + String(nnz) + " entries")
        _ = vals  # keep the Python object alive through the native call above
        _ = cols  # keep the Python object alive through the native call above
        _ = rows  # keep the Python object alive through the native call above
    _ = vals  # keep the Python object alive through the native calls above
    _ = cols  # keep the Python object alive through the native calls above
    _ = rows  # keep the Python object alive through the native calls above
    return _Coo(rows, cols, vals, n)


def _dense_coo(m: PythonObject, who: String, base: PythonObject, context: PythonObject) raises -> _Coo:
    """`_coo_triples(_DenseCOO(m))`: the dense float32 matrix through the
    base binding's `v != 0.0` scan, row major, then the sparse arm's
    checks."""
    var n_r = Int(py=m.shape[0])
    var n_c = Int(py=m.shape[1])
    var nnz = 0
    if n_r * n_c:
        nnz = Int(py=family_call(family_entry(base, "nonzero_f32_count", context), classical_address(m),
                                 PythonObject(n_r * n_c)))
    var rows = family_array(Python.tuple(nnz), "<i4", context)
    var cols = family_array(Python.tuple(nnz), "<i4", context)
    var data = family_array(Python.tuple(nnz), "<f4", context)
    if nnz:
        var wrote = Int(py=family_call(family_entry(base, "nonzero_f32_fill", context), classical_address(m),
                                       PythonObject(n_r), PythonObject(n_c),
                                       Python.list(classical_address(rows), classical_address(cols), classical_address(data)),
                                       PythonObject(nnz)))
        _ = data  # keep the Python object alive through the native call above
        _ = cols  # keep the Python object alive through the native call above
        _ = rows  # keep the Python object alive through the native call above
        if wrote != nnz:
            raise Error("RuntimeError: mojolearn: nonzero_f32_fill wrote " + String(wrote) + " of "
                        + String(nnz) + " entries")
            _ = data  # keep the Python object alive through the native call above
            _ = cols  # keep the Python object alive through the native call above
            _ = rows  # keep the Python object alive through the native call above
        _ = data  # keep the Python object alive through the native calls above
        _ = cols  # keep the Python object alive through the native calls above
        _ = rows  # keep the Python object alive through the native calls above
    return _coo_of_triples(rows, cols, data, Python.tuple(n_r, n_c), who, context)


def _x_decomp_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_M.from_input(X)`: a scipy.sparse matrix densified, `as_f32_c`, and
    the finiteness refusal. An exact SciPy container is densified natively
    from its stored buffers straight to float32 (duplicates summed, as
    `toarray()` sums them; `host_sparse_dense`, which refuses an empty
    matrix). No method of the input is called (P1, audit A #4: the
    `toarray()` call on other sparse-like inputs is removed; such an object
    takes the dense admission)."""
    var x = _densified(source, context)
    # TOMBSTONE: cfb_spectral.mojo second host scans after admission (four sites) deleted 2026-11-09 by lane a8d-device-x-decomp-manifold; code recoverable at 0ce88411d3.
    # Restore: experiments/removed/A8D_DECOMP_MANIFOLD_HOST_FINITE.patch.
    # Lane a8d: the finiteness refusal is the admission's own (tested in the
    # cast pass, or one parallel scan of a borrowed X, or in a fold's
    # gather), not a second scan after it; the same sentence.
    var a = admit_family(x, 2, "X", context, dense=True, finite=True,
                         refusal="X: input must be finite; NaN/inf are unsupported").array
    return a


def _cells_free(free: PythonObject, ids: PythonObject):
    """Return the affinity cells' device buffers to the binding's pool
    (`_DevBuf.__del__`); quiet, as there."""
    var count = 0
    try:
        count = len(ids)
    except:
        return
    for i in range(count):  # driver-loop: over the device buffers of the affinity cells
        try:
            _ = py_call(free, ids[i])
        except:
            pass


def _rbf_affinity(x: PythonObject, gamma: Float64, xdecomp: PythonObject, context: PythonObject) raises -> PythonObject:
    """`kit.ew("exp", kit.ew("scale", kit.sqdist(xm, xm), s=-gamma)).out()`:
    the decomp lane's cells, exp(-gamma ||x_i - x_j||^2), as an n x n
    float32 Array. The resident cells where the binding has them (`_Kit._use`
    with `_RES_MIN` = 1: every nonempty call), else the host cells, as the
    Python kit chose."""
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var out = family_array(Python.tuple(n, n), "<f4", context)
    var nn = n * n
    var scale = PythonObject(-gamma)
    var resident = cfb_optional_entry(xdecomp, "x_decomp_dev_ew", context) is not None
    if resident:
        var alloc = family_entry(xdecomp, "x_decomp_dev_alloc", context)
        var free = family_entry(xdecomp, "x_decomp_dev_free", context)
        var upload = family_entry(xdecomp, "x_decomp_dev_upload", context)
        var ew = family_entry(xdecomp, "x_decomp_dev_ew", context)
        var ids = Python.list()
        try:
            # sqdist: the output, then X uploaded once (A and B are one matrix).
            var dsq = family_call(alloc, PythonObject(max(nn, 1)))
            _ = py_call(ids.append, dsq)
            var dx = family_call(alloc, PythonObject(max(n * d, 1)))
            _ = py_call(ids.append, dx)
            _ = family_call(upload, dx, classical_address(x), PythonObject(n * d))
            _ = family_call(family_entry(xdecomp, "x_decomp_dev_sqdist", context), dx, dx, dsq,
                            Python.list(PythonObject(n), PythonObject(n), PythonObject(d)))
            # the unused broadcast operand: one 0.0 on the device (_M._dev_one)
            var zero = family_array(Python.tuple(1), "<f4", context)
            var done = family_call(alloc, PythonObject(1))
            _ = py_call(ids.append, done)
            _ = family_call(upload, done, classical_address(zero), PythonObject(1))
            _ = zero  # keep the Python object alive through the native call above
            var params_scale = Python.list(PythonObject(_OP_SCALE), PythonObject(nn), PythonObject(n), PythonObject(1),
                                           PythonObject(3), PythonObject(1), PythonObject(3))
            var dscaled = family_call(alloc, PythonObject(max(nn, 1)))
            _ = py_call(ids.append, dscaled)
            _ = family_call(ew, dsq, done, done, dscaled, params_scale, scale)
            var params_exp = Python.list(PythonObject(_OP_EXP), PythonObject(nn), PythonObject(n), PythonObject(1),
                                         PythonObject(3), PythonObject(1), PythonObject(3))
            var dexp = family_call(alloc, PythonObject(max(nn, 1)))
            _ = py_call(ids.append, dexp)
            _ = family_call(ew, dscaled, done, done, dexp, params_exp, PythonObject(0.0))
            _ = family_call(family_entry(xdecomp, "x_decomp_dev_download", context), dexp, classical_address(out),
                            PythonObject(nn))
            _ = out  # keep the Python object alive through the native call above
        except error:
            _cells_free(free, ids)
            raise error^
        _cells_free(free, ids)
        return out
    var sq = family_array(Python.tuple(n, n), "<f4", context)
    _ = family_call(family_entry(xdecomp, "x_decomp_sqdist", context), classical_address(x), classical_address(x),
                    classical_address(sq), Python.list(PythonObject(n), PythonObject(n), PythonObject(d)))
    _ = sq  # keep the Python object alive through the native call above
    var one = family_array(Python.tuple(1), "<f4", context)
    var host_ew = family_entry(xdecomp, "x_decomp_ew", context)
    var scaled = family_array(Python.tuple(n, n), "<f4", context)
    _ = family_call(host_ew, classical_address(sq), classical_address(one), classical_address(one),
                    classical_address(scaled),
                    Python.list(PythonObject(_OP_SCALE), PythonObject(nn), PythonObject(n), PythonObject(1),
                                PythonObject(3), PythonObject(1), PythonObject(3)), scale)
    _ = one  # keep the Python object alive through the native call above
    _ = sq  # keep the Python object alive through the native call above
    _ = scaled  # keep the Python object alive through the native call above
    _ = sq^
    _ = family_call(host_ew, classical_address(scaled), classical_address(one), classical_address(one),
                    classical_address(out),
                    Python.list(PythonObject(_OP_EXP), PythonObject(nn), PythonObject(n), PythonObject(1),
                                PythonObject(3), PythonObject(1), PythonObject(3)), PythonObject(0.0))
    _ = scaled  # keep the Python object alive through the native call above
    _ = one  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = scaled^
    _ = one^
    return out


def _knn_affinity(fields: PythonObject, source: PythonObject, clustering: Bool, base: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`SpectralEmbedding._precomputed_knn_affinity`: scikit-learn's
    `kneighbors_graph(mode='connectivity')` of a precomputed distance matrix,
    symmetrized 0.5 (C + C^T), by the base binding's `knn_affinity_f32`.
    Sets and returns `affinity_matrix_`."""
    var b = _builtins()
    var triples = _sparse_triples(source, context)
    var n: Int
    var nnz = 0
    var dense = PythonObject(None)
    var rows = PythonObject(None)
    var cols = PythonObject(None)
    var vals = PythonObject(None)
    if triples is not None:
        var shape = triples[3]
        n = _int(shape[0])
        if not Bool(py=family_call(b.bool, family_call(native_loaded_module("operator").eq,
                                                        shape, Python.tuple(n, n)))):
            raise Error("mojolearn SpectralEmbedding: the precomputed distance graph must be square")
        rows = _as_i32(triples[0], "rows", context)
        cols = _as_i32(triples[1], "cols", context)
        vals = admit_family(triples[2], 1, "data", context).array
        if not (Int(py=rows.shape[0]) == Int(py=cols.shape[0]) and Int(py=cols.shape[0]) == Int(py=vals.shape[0])):
            raise Error("mojolearn SpectralEmbedding: the precomputed graph's row, col and data differ in length")
        nnz = Int(py=vals.shape[0])
    else:
        dense = admit_family(_densified(source, context), 2, "X", context).array
        n = Int(py=dense.shape[0])
        if Int(py=dense.shape[1]) != n:
            raise Error("mojolearn SpectralEmbedding: the precomputed distance matrix must be square")
    var k = _resolved_neighbors(fields, n, clustering)
    var a = family_array(Python.tuple(n, n), "<f4", context)
    if n == 0:
        py_setitem(fields, "affinity_matrix_", a)
        return a
    var status = family_array(Python.tuple(3), "<i4", context)
    var sp = dense is None
    var zero = PythonObject(0)
    _ = family_call(family_entry(base, "knn_affinity_f32", context),
                    Python.list(zero if sp else classical_address(dense),
                                classical_address(rows) if sp and nnz else zero,
                                classical_address(cols) if sp and nnz else zero,
                                classical_address(vals) if sp and nnz else zero,
                                classical_address(a), classical_address(status)),
                    Python.list(PythonObject(n), PythonObject(k), PythonObject(nnz), PythonObject(1 if sp else 0)))
    _ = dense  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = cols  # keep the Python object alive through the native call above
    _ = vals  # keep the Python object alive through the native call above
    _ = status  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    # Read all three status words while `status` is still named: Mojo
    # releases a local at its last named use, and a read after that (the
    # refusal's row and count, formerly read inside the raise) saw freed
    # words (a garbage row on the CPU host and on CUDA).
    var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(status)))
    var code = Int(words[0])
    var bad_row = Int(words[1])
    var stored = Int(words[2])
    _ = status  # keep the Python object alive through the reads above
    if code == 1:
        raise Error("mojolearn SpectralEmbedding: a precomputed distance is negative or NaN")
    if code == 2:
        raise Error("mojolearn SpectralEmbedding: row " + String(bad_row) + " of the precomputed graph has "
                    + String(stored) + " stored distances, fewer than n_neighbors=" + String(k))
    py_setitem(fields, "affinity_matrix_", a)
    return a


def _resolved_neighbors(fields: PythonObject, n: Int, clustering: Bool) raises -> Int:
    """SpectralClustering: `_n_neighbors`; SpectralEmbedding: `_n_neighbors`,
    or `max(n // 10, 1)` for None."""
    var value = family_field(fields, "_n_neighbors")
    if clustering or value is not None:
        return Int(py=value)
    return max(n // 10, 1)


def _seed(random_state: PythonObject, who: String, clustering: Bool) raises -> Int:
    """`random_state` None is seed 0 (DEVIATION 891); else 0 <= seed < 2**32."""
    if random_state is None:
        return 0
    var seed = _int(random_state)
    if seed < 0 or seed >= 4294967296:
        var tail = String(" (cuML's check_random_seed refuses the same range)") if clustering else String("")
        raise Error("mojolearn " + who + ": random_state must satisfy "
                    + "0 <= random_state < 2**32, got " + family_repr(random_state) + tail)
    return seed


def _sc_validate(fields: PythonObject, context: PythonObject) raises:
    """`SpectralClustering._validate_params`."""
    var b = _builtins()
    var n_clusters = family_field(fields, "n_clusters")
    var n_components = family_field(fields, "n_components")
    var random_state = family_field(fields, "random_state")
    var n_neighbors = family_field(fields, "n_neighbors")
    var n_init = family_field(fields, "n_init")
    var eigen_tol = family_field(fields, "eigen_tol")
    var affinity = family_field(fields, "affinity")
    var assign_labels = family_field(fields, "assign_labels")
    var eigen_solver = family_field(fields, "eigen_solver")
    var gamma = family_field(fields, "gamma")
    var degree = family_field(fields, "degree")
    var coef0 = family_field(fields, "coef0")
    var kernel_params = family_field(fields, "kernel_params")
    var n_jobs = family_field(fields, "n_jobs")
    var verbose = family_field(fields, "verbose")
    var prediction_data = family_field(fields, "prediction_data")
    var affinities = context["family_cfb_sc_affinities"]
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralClustering: affinity=" + family_repr(affinity) + " is "
                    + "refused; it must be one of " + family_repr(py_call(b.list, affinities)) + ". A "
                    + "callable or another kernel name builds the affinity outside "
                    + "every identity column.")
    var assigns = context["family_cfb_assign_labels"]
    if not py_contains(assigns, assign_labels):
        raise Error("mojolearn SpectralClustering: assign_labels="
                    + family_repr(assign_labels) + " must be one of " + family_repr(py_call(b.list, assigns)))
    var predicting = Bool(py=family_call(b.bool, prediction_data))
    var operator = native_loaded_module("operator")
    if predicting and Bool(py=family_call(b.bool, family_call(operator.ne, assign_labels, PythonObject("kmeans")))):
        raise Error("NotImplementedError: mojolearn SpectralClustering: prediction_data with "
                    + "assign_labels=" + family_repr(assign_labels) + " is refused; the Nystrom "
                    + "predict (DEVIATION 2860) assigns by the k-means centroids")
    if eigen_solver is not None:
        raise Error("NotImplementedError: mojolearn SpectralClustering: eigen_solver=" + family_repr(eigen_solver) + " "
                    + "is refused; there is one solver here, RAFT's thick-restart "
                    + "Lanczos, and it is the one the identity profile pins")
    var rbf = Bool(py=family_call(b.bool, family_call(operator.eq, affinity, PythonObject("rbf"))))
    if gamma is not None and not rbf:
        raise Error("NotImplementedError: mojolearn SpectralClustering: gamma is refused; it "
                    + "parameterizes the RBF affinity, and affinity is not 'rbf'")
    if predicting and py_contains(Python.tuple("rbf", "precomputed_nearest_neighbors"), affinity):
        raise Error("NotImplementedError: mojolearn SpectralClustering: prediction_data with affinity="
                    + family_repr(affinity) + " "
                    + "is refused; the Nystrom predict (DEVIATION 2860) carries the "
                    + "'nearest_neighbors' and 'precomputed' affinities only")
    var names: List[String] = ["degree", "coef0", "kernel_params"]
    var values = Python.list(degree, coef0, kernel_params)
    for i in range(3):  # small-loop(parameters: degree, coef0, kernel_params): 3
        if values[i] is not None:
            raise Error("NotImplementedError: mojolearn SpectralClustering: " + names[i] + " is refused; it "
                        + "parameterizes an RBF or polynomial affinity, and "
                        + "affinity is restricted to 'nearest_neighbors' and "
                        + "'precomputed'")
    if n_jobs is not None:
        raise Error("NotImplementedError: mojolearn SpectralClustering: n_jobs is refused; the work "
                    + "is on the GPU and there is no host thread pool to size")
    if Bool(py=family_call(b.bool, verbose)):
        raise Error("NotImplementedError: mojolearn SpectralClustering: verbose is refused; nothing "
                    + "in this path prints. Set MOJOLEARN_IDENTITY_TRACE=<path> "
                    + "for the identity card instead")
    var default_tol = family_str(context["family_cfb_default_eigen_tol"])
    if family_is_str(eigen_tol):
        raise Error("NotImplementedError: mojolearn SpectralClustering: eigen_tol=" + family_repr(eigen_tol) + " is "
                    + "refused (DEVIATION 890). cuML 26.08's Python maps 'auto' to "
                    + "0.0, which is passed straight through as the eigensolver "
                    + "tolerance and disables the convergence test (the loop "
                    + "condition is res > tol), so the solve runs to "
                    + "max_iterations = 10 * n_samples and an exactly converged "
                    + "problem hits the restart-breakdown refusal. Pass a positive "
                    + "float; " + default_tol + " is cuVS's own struct default "
                    + "and is this class's default.")
    var tol = _float(eigen_tol)
    var t = Float64(py=tol)
    if not (t > 0.0) or t != t:
        raise Error("mojolearn SpectralClustering: eigen_tol must be positive, "
                    + "got " + family_repr(tol) + " (DEVIATION 890; see the 'auto' note)")
    var seed = _seed(random_state, "SpectralClustering", True)
    var clusters = _int(n_clusters)
    if clusters < 1:
        raise Error("mojolearn SpectralClustering: n_clusters must be at least 1")
    var inits = _int(n_init)
    if inits < 1:
        raise Error("mojolearn SpectralClustering: n_init must be at least 1")
    var neighbors = _int(n_neighbors)
    if neighbors < 1:
        raise Error("mojolearn SpectralClustering: n_neighbors must be at least 1")
    if n_components is not None and _int(n_components) < 1:
        raise Error("mojolearn SpectralClustering: n_components must be at least "
                    + "1, or None to follow n_clusters")
    py_setitem(fields, "_n_clusters", PythonObject(clusters))
    py_setitem(fields, "_n_neighbors", PythonObject(neighbors))
    py_setitem(fields, "_n_init", PythonObject(inits))
    py_setitem(fields, "_eigen_tol", tol)
    # scikit-learn's SpectralClustering default gamma is 1.0.
    var g = PythonObject(None)
    if rbf:
        g = PythonObject(1.0) if gamma is None else _float(gamma)
    py_setitem(fields, "_gamma", g)
    py_setitem(fields, "_seed", PythonObject(seed))


def _sc_check_shape(fields: PythonObject, n: Int, k: Int) raises:
    var clusters = Int(py=family_field(fields, "_n_clusters"))
    if clusters > n:
        raise Error("mojolearn SpectralClustering: n_clusters=" + String(clusters) + " "
                    + "exceeds n_samples=" + String(n))
    if k >= n:
        raise Error("mojolearn SpectralClustering: n_components=" + String(k) + " must be "
                    + "below n_samples=" + String(n) + "; the Lanczos requires "
                    + "1 <= n_components < n_samples and n_samples - n_components "
                    + "> 0 (cuVS's own RAFT_EXPECTS)")


def _sc_refusal(n_out: Int, who: String) raises:
    """The dense entries' refusal codes, in the COO route's wording."""
    if n_out == _DENSE_EMPTY:
        raise Error("mojolearn " + who + ": the precomputed affinity "
                    + "matrix has no nonzero entries")
    if n_out == _DENSE_NONFINITE:
        if who == "SpectralClustering":
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a non-finite entry (the implemented path refuses "
                        + "it by name, because sqrt of a non-finite degree is a "
                        + "NaN and no NaN may reach a recorded value)")
        raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                    + "matrix has a non-finite entry")
    if n_out == _DENSE_NEGATIVE:
        if who == "SpectralClustering":
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a negative entry (refused by name: sqrt of a "
                        + "negative degree is a NaN in cuVS too)")
        raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                    + "matrix has a negative entry (refused by name: sqrt of a "
                    + "negative degree is a NaN)")


def _sc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`SpectralClustering.fit(X)`. args: (metrics, x_decomp, base,
    x_cluster, X)."""
    var b = _builtins()
    var fields = py_fields(model)
    var metrics = args[0]
    var source = args[4]
    _sc_validate(fields, context)
    var prediction_data = family_field(fields, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn SpectralClustering: prediction_data must be a bool, "
                    + "got " + String(py_type(prediction_data).__name__))
    var predicting = Bool(py=prediction_data)
    var dropped = Python.tuple("_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X")
    for i in range(5):  # small-loop(prediction data: the five cached attribute names): 5
        _ = py_call(fields.pop, dropped[i], PythonObject(None))
    var clusters = Int(py=family_field(fields, "_n_clusters"))
    var n_components = family_field(fields, "n_components")
    var k = clusters if n_components is None else _int(n_components)
    _ = py_call(fields.pop, PythonObject("affinity_matrix_"), PythonObject(None))
    var affinity = String(py=family_field(fields, "affinity"))
    var graph_affinity = affinity == "precomputed" or affinity == "rbf" or affinity == "precomputed_nearest_neighbors"
    var connection: Optional[SpectralConnection] = None
    var use_dense = False
    if graph_affinity:
        connection = SpectralConnection(metrics, context)
        use_dense = (connection.value().capabilities & SPECTRAL_CAP_DENSE) != 0
    # `dense` is the matrix the dense entry reads (an Array); `coo_source`
    # the input of the COO route.
    var dense = PythonObject(None)
    var from_matrix = False
    var coo_source = source
    if affinity == "precomputed":
        py_setitem(fields, "affinity_matrix_", source)
        if use_dense and not _is_sparse(source):
            var shape = _shape_of(source)
            if not _square(shape):
                raise Error("mojolearn SpectralClustering: with affinity='precomputed', X "
                            + "must be a square affinity matrix or a SciPy sparse matrix, "
                            + "got shape " + family_repr(shape))
            dense = admit_family(source, 2, "X", context).array
    elif affinity == "rbf":
        var x = _x_decomp_input(source, context)
        var aff = _rbf_affinity(x, Float64(py=family_field(fields, "_gamma")), args[1], context)
        py_setitem(fields, "affinity_matrix_", aff)
        py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))
        dense = aff
        from_matrix = True
    elif affinity == "precomputed_nearest_neighbors":
        dense = _knn_affinity(fields, source, True, args[2], context)
        from_matrix = True
    var labels = PythonObject(None)
    var embedding = PythonObject(None)
    var state = SpectralStatePins()
    var n_out: Int
    var n: Int
    var options = SpectralFitOptions(clusters, k, Int(py=family_field(fields, "_n_init")),
        Int(py=family_field(fields, "_n_neighbors")), Float64(py=family_field(fields, "_eigen_tol")),
        Int(py=family_field(fields, "_seed")), predicting)
    if use_dense and dense is not None:
        # the dense matrix straight to the device entry
        n = Int(py=dense.shape[0])
        _sc_check_shape(fields, n, k)
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        if predicting:
            state.allocate(n, k, clusters, context)
        var request = SpectralFitRequest(
            SpectralInput(Int(py=classical_address(dense)), 0, 0, 0, SPECTRAL_DENSE, n, 0, 0),
            Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(), options)
        n_out = spectral_fit_run(connection.value(), request)
        _ = state
        _ = embedding
        _ = labels
        _ = dense
        _sc_refusal(n_out, "SpectralClustering")
    elif graph_affinity:
        # the COO route: a sparse precomputed input, or a binding without
        # the dense entry (the CPU-only host binding)
        var coo: _Coo
        if from_matrix:
            coo = _dense_coo(dense, "SpectralClustering", args[2], context)
        else:
            coo = _coo_triples(coo_source, "SpectralClustering", args[2], context)
        n = coo.n
        var nnz = Int(py=coo.vals.size)
        if nnz == 0:
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has no nonzero entries")
        var vp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coo.vals)))
        if not host_all_finite_f32(vp, nnz):
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a non-finite entry (the implemented path refuses "
                        + "it by name, because sqrt of a non-finite degree is a "
                        + "NaN and no NaN may reach a recorded value)")
        if family_min(coo.vals) < 0:
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a negative entry (refused by name: sqrt of a "
                        + "negative degree is a NaN in cuVS too)")
        _sc_check_shape(fields, n, k)
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        if predicting:
            state.allocate(n, k, clusters, context)
        var request = SpectralFitRequest(
            SpectralInput(0, Int(py=classical_address(coo.rows)), Int(py=classical_address(coo.cols)),
                Int(py=classical_address(coo.vals)), SPECTRAL_GRAPH, n, 0, nnz),
            Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(), options)
        n_out = spectral_fit_run(connection.value(), request)
        _ = state
        _ = labels
        _ = embedding
        _ = coo.rows
        _ = coo.cols
        _ = coo.vals
    else:
        # Lane a8d: the admission's finite test (see `_x_decomp_input`).
        var x = admit_family(source, 2, "X", context, finite=True,
                             refusal="mojolearn SpectralClustering: X contains NaN or "
                             + "infinity (the implemented path refuses it by name before any "
                             + "recorded stage)").array
        py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
        n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        _sc_check_shape(fields, n, k)
        var neighbors = Int(py=family_field(fields, "_n_neighbors"))
        if neighbors > n:
            raise Error("mojolearn SpectralClustering: n_neighbors="
                        + String(neighbors) + " exceeds n_samples=" + String(n))
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        if predicting:
            state.allocate(n, k, clusters, context)
        connection = SpectralConnection(metrics, context)
        var request = SpectralFitRequest(
            SpectralInput(Int(py=classical_address(x)), 0, 0, 0, SPECTRAL_DATASET, n, d, 0),
            Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(), options)
        n_out = spectral_fit_run(connection.value(), request)
        _ = state
        _ = x
        _ = labels
        _ = embedding
        if predicting:
            # Preserve the fitted copy of a borrowed input for later queries.
            py_setitem(fields, "_fit_X", cfb_borrow_copy(x, source, context))
        py_setitem(fields, "n_features_in_", PythonObject(d))
        _ = x^
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralClustering: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    py_setitem(fields, "embedding_", embedding)
    py_setitem(fields, "n_components_", PythonObject(k))
    _ = py_call(fields.pop, PythonObject("n_iter_assign_"), PythonObject(None))
    var assign = family_field(fields, "assign_labels")
    if String(py=assign) != "kmeans":
        # scikit-learn's discretize / cluster_qr on the fit's embedding, the
        # x_cluster binding's host code (ENTRY_SPECTRAL_ASSIGN).
        var ip = Python.list(PythonObject(n), PythonObject(k), context["family_cfb_assign_codes"][assign],
                             PythonObject(_int(family_field(fields, "_seed"))), PythonObject(30), PythonObject(20))
        var result = family_call(family_entry(args[3], "x_cluster_call", context), PythonObject(_E_SPECTRAL_ASSIGN),
                                 classical_address(embedding), PythonObject(Int(py=embedding.size)), PythonObject(0),
                                 PythonObject(0), ip, Python.list(PythonObject(0.0)))
        _ = embedding  # keep the Python object alive through the native call above
        if String(py=assign) == "discretize":
            py_setitem(fields, "n_iter_assign_", PythonObject(_int(result[2][0])))
            _ = embedding  # keep the Python object alive through the native call above
        labels = _xc_array(result[1][0], Python.tuple(n), "<i4", "i", context)
        _ = embedding  # keep the Python object alive through the native call above
    # A fit releases the last fit's resident prediction state at once (N326).
    cres_forget(model, CRES_SPECTRAL)
    py_setitem(fields, "labels_", labels)
    if predicting:
        py_setitem(fields, "_pd_eigenvalues", state.eigenvalues)
        py_setitem(fields, "_pd_eigenvectors", state.eigenvectors)
        py_setitem(fields, "_pd_diag", state.diagonal)
        py_setitem(fields, "_pd_centroids", state.centroids)


def _sc_resident(model: PythonObject, binding: PythonObject, state: SpectralStatePins,
                 training: PythonObject, shape: SpectralQueryShape,
                 mut connection: Optional[SpectralConnection],
                 context: PythonObject) raises -> Optional[SpectralResidentLease]:
    """Take a typed native lease while this generation/storage/provider matches."""
    if (not fitted_is_freezable(state.eigenvalues, context)
            or not fitted_is_freezable(state.eigenvectors, context)
            or not fitted_is_freezable(state.diagonal, context)
            or not fitted_is_freezable(state.centroids, context)):
        return None
    if shape.affinity == 0 and not fitted_is_freezable(training, context):
        return None
    var dims: List[Int] = [shape.training_rows, shape.features, shape.components, shape.clusters, shape.affinity]
    connection = SpectralConnection(binding, context)
    var held = cres_spectral_lookup(model, binding, connection.value().provider, state, training, dims)
    if held:
        return held^
    if (connection.value().capabilities & SPECTRAL_CAP_RESIDENT) == 0:
        return None
    # Freeze before the same native resident copies; writable or rebound public
    # buffers retain the former per-call/rebuild behavior.
    _ = fitted_freeze(state.eigenvalues, context)
    _ = fitted_freeze(state.eigenvectors, context)
    _ = fitted_freeze(state.diagonal, context)
    _ = fitted_freeze(state.centroids, context)
    var training_address = 0
    if shape.affinity == 0:
        _ = fitted_freeze(training, context)
        training_address = Int(py=classical_address(training))
    var request = SpectralPrepareRequest(training_address, state.view(), shape)
    var lease = spectral_resident_prepare(connection.value(), state, training, request)
    var nbytes = 4 * (shape.training_rows * shape.components + shape.training_rows + shape.components
        + shape.clusters * shape.components + (shape.training_rows * shape.features if shape.affinity == 0 else 0))
    cres_spectral_store(model, binding, dims^, lease.copy(), nbytes)
    return lease^


def _sc_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`SpectralClustering._predict_embedding`: (labels, embedding). The Nystrom extension of the fitted
    eigenvectors, then the nearest fitted k-means centroid."""
    var b = _builtins()
    if not family_raw_has(model, "labels_"):
        raise Error("mojolearn SpectralClustering.predict: this instance is not "
                    + "fitted yet; call fit first")
    var eigenvalues = family_raw_get_or(model, "_pd_eigenvalues", PythonObject(None))
    if eigenvalues is None:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not "
                    + "stored. Fit with SpectralClustering(prediction_data=True), which "
                    + "keeps the eigenpairs, degrees and centroids the Nystrom extension "
                    + "needs (DEVIATION 2860)")
    var affinity = family_raw_get(model, "affinity")
    var affinities = context["family_cfb_predict_affinities"]
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralClustering.predict: affinity=" + family_repr(affinity) + " "
                    + "has no out-of-sample rule; it must be one of "
                    + family_repr(py_call(b.list, affinities)))
    var k = _int(family_raw_get(model, "n_components_"))
    var diag = family_raw_get(model, "_pd_diag")
    var n_train = Int(py=diag.shape[0])
    var q = PythonObject(None)
    var training = PythonObject(None)
    var train_addr = 0
    var n_features = 0
    var code = 0
    if String(py=affinity) == "precomputed":
        # P1 (audit A #4): an exact SciPy container is densified natively;
        # the caller's `toarray()` is never called.
        var a = _densified(source, context)
        var shape = _shape_of(a)
        if len(shape) != 2 or not Bool(py=family_call(b.bool, family_call(native_loaded_module("operator").eq,
                                                                           shape[1], PythonObject(n_train)))):
            raise Error("mojolearn SpectralClustering.predict: with affinity="
                        + "'precomputed', X must be the affinity between the new rows "
                        + "and the " + String(n_train) + " training rows, shape (n_new, " + String(n_train) + "); "
                        + "got shape " + family_repr(shape))
        # Lane a8d: the admission's finite test (see `_x_decomp_input`).
        q = admit_family(a, 2, "X", context, finite=True,
                         refusal="mojolearn SpectralClustering.predict: the affinity has a "
                         + "non-finite entry; refused by name").array
        if Int(py=q.shape[0]) < 1:
            raise Error("mojolearn SpectralClustering.predict: X has no rows; refused by name")
        if family_min(q) < 0:
            raise Error("mojolearn SpectralClustering.predict: the affinity has a "
                        + "negative entry; refused by name, as the fit refuses one")
        code = 1
    else:
        var fitted = Int(py=family_raw_get(model, "n_features_in_"))
        q = _reference_queries(source, fitted, "SpectralClustering.predict", context)
        training = family_raw_get(model, "_fit_X")
        train_addr = Int(py=classical_address(training))
        n_features = fitted
    var nq = Int(py=q.shape[0])
    var labels = family_array(Python.tuple(nq), "<i4", context)
    var emb = family_array(Python.tuple(nq, k), "<f4", context)
    var n_clusters = _int(family_raw_get(model, "_n_clusters"))
    var n_neighbors = _int(family_raw_get(model, "_n_neighbors"))
    var shape = SpectralQueryShape(n_train, nq, n_features, k, n_clusters, n_neighbors, code)
    var state = SpectralStatePins()
    state.eigenvalues = eigenvalues
    state.eigenvectors = family_raw_get(model, "_pd_eigenvectors")
    state.diagonal = diag
    state.centroids = family_raw_get(model, "_pd_centroids")
    var connection: Optional[SpectralConnection] = None
    var held = _sc_resident(model, binding, state, training, shape, connection, context)
    if held:
        # The resident state (N326): only the queries cross. The lease owns
        # the handle until the synchronous query finishes, even after eviction.
        try:
            var request = SpectralResidentQueryRequest(held.value().native[].handle,
                Int(py=classical_address(q)), Int(py=classical_address(labels)),
                Int(py=classical_address(emb)), shape)
            spectral_resident_query(held.value(), request)
        finally:
            spectral_resident_retire(held.take())
            _ = q
            _ = emb
            _ = labels
        return Python.tuple(labels, emb)
    if not connection:
        connection = SpectralConnection(binding, context)
    var request = SpectralQueryRequest(Int(py=classical_address(q)), train_addr, state.view(),
        Int(py=classical_address(labels)), Int(py=classical_address(emb)), shape)
    spectral_query_run(connection.value(), request)
    _ = q
    _ = state
    _ = training
    _ = emb
    _ = labels
    return Python.tuple(labels, emb)


def _se_validate(fields: PythonObject, context: PythonObject) raises:
    """`SpectralEmbedding._validate_params`."""
    var b = _builtins()
    var n_components = family_field(fields, "n_components")
    var affinity = family_field(fields, "affinity")
    var random_state = family_field(fields, "random_state")
    var n_neighbors = family_field(fields, "n_neighbors")
    var gamma = family_field(fields, "gamma")
    var eigen_solver = family_field(fields, "eigen_solver")
    var eigen_tol = family_field(fields, "eigen_tol")
    var affinities = context["family_cfb_se_affinities"]
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralEmbedding: affinity=" + family_repr(affinity) + " is "
                    + "refused; it must be one of " + family_repr(py_call(b.list, affinities)) + ". "
                    + "A callable affinity is Python code that no identity column "
                    + "can run.")
    var rbf = Bool(py=family_call(b.bool, family_call(native_loaded_module("operator").eq, affinity, PythonObject("rbf"))))
    if gamma is not None and not rbf:
        raise Error("NotImplementedError: mojolearn SpectralEmbedding: gamma is refused; it "
                    + "parameterizes an RBF affinity, and affinity is not 'rbf'")
    if gamma is not None and not (Float64(py=_float(gamma)) > 0):
        raise Error("mojolearn SpectralEmbedding: gamma must be positive")
    if eigen_solver is not None:
        raise Error("NotImplementedError: mojolearn SpectralEmbedding: eigen_solver=" + family_repr(eigen_solver) + " "
                    + "is refused; there is one solver here, RAFT's thick-restart "
                    + "Lanczos. scikit-learn's 'arpack', 'lobpcg' and 'amg' are "
                    + "other algorithms, and running the Lanczos under their names "
                    + "would be a silent substitution (spectral/NOT_IMPLEMENTED.tsv)")
    if family_is_str(eigen_tol):
        if String(py=eigen_tol) != "auto":
            raise Error("mojolearn SpectralEmbedding: eigen_tol=" + family_repr(eigen_tol) + " "
                        + "must be 'auto' or a positive float")
    elif eigen_tol is not None:
        var tol = Float64(py=_float(eigen_tol))
        if not (tol > 0.0) or tol != tol or isinf(tol):
            raise Error("mojolearn SpectralEmbedding: eigen_tol must be a positive "
                        + "finite float or 'auto', got " + family_repr(eigen_tol) + " (0 disables the "
                        + "Lanczos convergence test, DEVIATION 890)")
    if _int(n_components) < 1:
        raise Error("mojolearn SpectralEmbedding: n_components must be at least 1")
    if n_neighbors is not None and _int(n_neighbors) < 1:
        raise Error("mojolearn SpectralEmbedding: n_neighbors must be at least 1, "
                    + "or None for max(n_samples // 10, 1)")
    var seed = _seed(random_state, "SpectralEmbedding", False)
    py_setitem(fields, "_n_components", PythonObject(_int(n_components)))
    py_setitem(fields, "_n_neighbors", PythonObject(None) if n_neighbors is None else PythonObject(_int(n_neighbors)))
    py_setitem(fields, "_seed", PythonObject(seed))
    py_setitem(fields, "_gamma", PythonObject(None) if gamma is None else _float(gamma))


def _se_fit(model: PythonObject, args: PythonObject, source: PythonObject, context: PythonObject) raises:
    """`SpectralEmbedding.fit(X)`. args: (metrics, x_decomp, base, ...)."""
    var b = _builtins()
    var fields = py_fields(model)
    var metrics = args[0]
    _se_validate(fields, context)
    var k = Int(py=family_field(fields, "_n_components"))
    # `_drop_first` / `_norm_laplacian`: class attributes, or the instance's
    # (spectral_embedding sets them).
    var drop_first = Bool(py=family_call(b.bool, family_raw_get(model, "_drop_first")))
    var norm_laplacian = Bool(py=family_call(b.bool, family_raw_get(model, "_norm_laplacian")))
    # The one piece of arithmetic the reference keeps in Python
    # (spectral_embedding.pyx:294): the transform receives this number.
    var n_lanczos = k + 1 if drop_first else k
    var affinity = String(py=family_field(fields, "affinity"))
    var graph_affinity = affinity == "precomputed" or affinity == "rbf" or affinity == "precomputed_nearest_neighbors"
    var connection: Optional[SpectralConnection] = None
    var use_dense = False
    if graph_affinity:
        connection = SpectralConnection(metrics, context)
        use_dense = (connection.value().capabilities & SPECTRAL_CAP_DENSE) != 0
    var dense = PythonObject(None)
    var from_matrix = False
    if affinity == "precomputed_nearest_neighbors":
        dense = _knn_affinity(fields, source, False, args[2], context)
        from_matrix = True
    if affinity == "precomputed" and use_dense and not _is_sparse(source):
        var shape = _shape_of(source)
        if not _square(shape):
            raise Error("mojolearn SpectralEmbedding: with affinity='precomputed', X "
                        + "must be a square affinity matrix or a SciPy sparse matrix, "
                        + "got shape " + family_repr(shape))
        dense = admit_family(source, 2, "X", context).array
    if affinity == "rbf":
        var x = _x_decomp_input(source, context)
        var d = Int(py=x.shape[1])
        var gamma_field = family_field(fields, "_gamma")
        var gamma = Float64(py=gamma_field) if gamma_field is not None else 1.0 / Float64(d)
        py_setitem(fields, "gamma_", PythonObject(gamma))
        var aff = _rbf_affinity(x, gamma, args[1], context)
        py_setitem(fields, "affinity_matrix_", aff)
        py_setitem(fields, "n_features_in_", PythonObject(d))
        dense = aff
        from_matrix = True
    var tolerance = Float64(Float32(1e-5))
    var eigen_tol = family_field(fields, "eigen_tol")
    if eigen_tol is not None and not family_is_str(eigen_tol):
        tolerance = Float64(py=_float(eigen_tol))
    var seed = Int(py=family_field(fields, "_seed"))
    var embedding = PythonObject(None)
    var n_out: Int
    if use_dense and dense is not None:
        var n = Int(py=dense.shape[0])
        _se_check_shape(fields, n, n_lanczos)
        py_setitem(fields, "n_neighbors_", PythonObject(_resolved_neighbors(fields, n, False)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        var request = SpectralEmbeddingRequest(
            SpectralInput(Int(py=classical_address(dense)), 0, 0, 0, SPECTRAL_DENSE, n, 0, 0),
            Int(py=classical_address(embedding)), SpectralEmbeddingOptions(n_lanczos, k, 0,
                norm_laplacian, drop_first, seed, tolerance))
        n_out = spectral_embedding_run(connection.value(), request)
        _ = embedding
        _ = dense
        _sc_refusal(n_out, "SpectralEmbedding")
    elif graph_affinity:
        var coo: _Coo
        if from_matrix:
            coo = _dense_coo(dense, "SpectralEmbedding", args[2], context)
        else:
            coo = _coo_triples(source, "SpectralEmbedding", args[2], context)
        var n = coo.n
        var nnz = Int(py=coo.vals.size)
        if nnz == 0:
            raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                        + "matrix has no nonzero entries")
        # the rbf affinity is exp of a finite value, finite and >= 0 by
        # construction (`exp_c` clamps), so its two scans are skipped
        if affinity == "precomputed":
            if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coo.vals))), nnz):
                raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                            + "matrix has a non-finite entry")
            if family_min(coo.vals) < 0:
                raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                            + "matrix has a negative entry (refused by name: sqrt of a "
                            + "negative degree is a NaN)")
        _se_check_shape(fields, n, n_lanczos)
        py_setitem(fields, "n_neighbors_", PythonObject(_resolved_neighbors(fields, n, False)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        var request = SpectralEmbeddingRequest(
            SpectralInput(0, Int(py=classical_address(coo.rows)), Int(py=classical_address(coo.cols)),
                Int(py=classical_address(coo.vals)), SPECTRAL_GRAPH, n, 0, nnz),
            Int(py=classical_address(embedding)), SpectralEmbeddingOptions(n_lanczos, k, 0,
                norm_laplacian, drop_first, seed, tolerance))
        n_out = spectral_embedding_run(connection.value(), request)
        _ = embedding
        _ = coo.rows
        _ = coo.cols
        _ = coo.vals
    else:
        # Lane a8d: the admission's finite test (see `_x_decomp_input`).
        var x = admit_family(source, 2, "X", context, finite=True,
                             refusal="mojolearn SpectralEmbedding: X contains NaN or infinity").array
        py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        _se_check_shape(fields, n, n_lanczos)
        var neighbors = _resolved_neighbors(fields, n, False)
        if neighbors > n:
            raise Error("mojolearn SpectralEmbedding: n_neighbors=" + String(neighbors) + " "
                        + "exceeds n_samples=" + String(n))
        py_setitem(fields, "n_neighbors_", PythonObject(neighbors))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        connection = SpectralConnection(metrics, context)
        var request = SpectralEmbeddingRequest(
            SpectralInput(Int(py=classical_address(x)), 0, 0, 0, SPECTRAL_DATASET, n, d, 0),
            Int(py=classical_address(embedding)), SpectralEmbeddingOptions(n_lanczos, k, neighbors,
                norm_laplacian, drop_first, seed, tolerance))
        n_out = spectral_embedding_run(connection.value(), request)
        _ = x
        _ = embedding
        py_setitem(fields, "n_features_in_", PythonObject(d))
        _ = x^
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralEmbedding: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    py_setitem(fields, "embedding_", embedding)


def _se_check_shape(fields: PythonObject, n: Int, n_lanczos: Int) raises:
    if n_lanczos >= n:
        raise Error("mojolearn SpectralEmbedding: n_components=" + String(Int(py=family_field(fields, "_n_components"))) + " "
                    + "needs " + String(n_lanczos) + " eigenpairs, which must be below "
                    + "n_samples=" + String(n))


def _spectral_embedding(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`spectral_embedding(A, ...)`: `SpectralEmbedding(...).fit_transform(A)`
    with the two switches the class fixes. args: (SpectralEmbedding,
    metrics, x_decomp, base, A, n_components, affinity, random_state,
    n_neighbors, norm_laplacian, drop_first)."""
    var b = _builtins()
    var names: List[String] = ["norm_laplacian", "drop_first"]
    for i in range(2):  # small-loop(flags: norm_laplacian, drop_first): 2
        var value = args[9 + i]
        if py_type(value) is not b.bool:
            raise Error("TypeError: mojolearn spectral_embedding: " + names[i] + " must be a bool, got "
                        + String(py_type(value).__name__))
    # The model the function's Python built: the constructor's fields, in
    # its order, then the two switches.
    var model = py_call(b.object.__new__, args[0])
    var fields = py_fields(model)
    py_setitem(fields, "n_components", args[5])
    py_setitem(fields, "affinity", args[6])
    py_setitem(fields, "random_state", args[7])
    py_setitem(fields, "n_neighbors", args[8])
    py_setitem(fields, "gamma", PythonObject(None))
    py_setitem(fields, "eigen_solver", PythonObject(None))
    py_setitem(fields, "eigen_tol", PythonObject(None))
    py_setitem(fields, "n_jobs", PythonObject(None))
    py_setitem(fields, "verbose", PythonObject(False))
    py_setitem(fields, "_drop_first", args[10])
    py_setitem(fields, "_norm_laplacian", args[9])
    _se_fit(model, Python.tuple(args[1], args[2], args[3]), args[4], context)
    return fields["embedding_"]


def _cfb_spectral_algorithm_operation(model: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject, supplied_binding: PythonObject) raises -> PythonObject:
    """Fit/predict shells pass (X,); route selection belongs to this owner."""
    if operation == "family_cfb_sc_fit" or operation == "family_cfb_sc_fit_predict":
        var routed = Python.tuple(
            spectral_umap_route(context, "metrics", PythonObject(None)),
            spectral_umap_route(context, "decomp", PythonObject(None)),
            spectral_umap_route(context, "base", PythonObject("identical")),
            spectral_umap_route(context, "cluster", py_get(py_fields(model), "numeric_mode")), args[0])
        _sc_fit(model, routed, context)
        if operation == "family_cfb_sc_fit":
            return model
        return py_fields(model)["labels_"]
    if operation == "family_cfb_sc_predict":
        var mode = py_get(py_fields(model), "numeric_mode")
        var binding = supplied_binding
        if binding is None:
            binding = host_estimator_binding(model, "_mojolearn_metrics", mode, context)
        if binding is None:
            binding = spectral_umap_route(context, "metrics", mode)
        return _sc_predict(model, binding, args[0], context)[0]
    if operation == "family_cfb_sc_predict_embedding":
        var mode = py_get(py_fields(model), "numeric_mode")
        var binding = supplied_binding
        if binding is None:
            binding = host_estimator_binding(model, "_mojolearn_metrics", mode, context)
        if binding is None:
            binding = spectral_umap_route(context, "metrics", mode)
        return _sc_predict(model, binding, args[0], context)
    if operation == "family_cfb_se_validate":
        _se_validate(py_fields(model), context)
        return PythonObject(None)
    if operation == "family_cfb_se_knn_affinity":
        var base = spectral_umap_route(context, "base", PythonObject("identical"))
        return _knn_affinity(py_fields(model), args[0], False, base, context)
    if operation == "family_cfb_coo_triples":
        var base = spectral_umap_route(context, "base", PythonObject("identical"))
        var coo = _coo_triples(args[0], String(py=args[1]), base, context)
        return Python.tuple(coo.rows, coo.cols, coo.vals, PythonObject(coo.n))
    if operation == "family_cfb_se_fit" or operation == "family_cfb_se_fit_transform":
        var routed = Python.tuple(
            spectral_umap_route(context, "metrics", PythonObject(None)),
            spectral_umap_route(context, "decomp", PythonObject(None)),
            spectral_umap_route(context, "base", PythonObject("identical")))
        _se_fit(model, routed, args[0], context)
        if operation == "family_cfb_se_fit":
            return model
        return py_fields(model)["embedding_"]
    if operation == "family_cfb_spectral_embedding":
        var routed = Python.tuple(args[0],
            spectral_umap_route(context, "metrics", PythonObject(None)),
            spectral_umap_route(context, "decomp", PythonObject(None)),
            spectral_umap_route(context, "base", PythonObject("identical")),
            args[1], args[2], args[3], args[4], args[5], args[6], args[7])
        return _spectral_embedding(routed, context)
    raise Error("RuntimeError: unknown spectral operation " + operation)


def cfb_spectral_operation(model: PythonObject, operation: String, var args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = _builtins()
    if py_type(args) is not b.tuple:
        raise Error("TypeError: spectral public arguments require an exact tuple")
    var supplied_binding = PythonObject(None)
    if operation == "family_cfb_sc_predict" and len(args) == 2:
        # The archive recipe already carries its admitted CPU owner.
        supplied_binding = args[0]
        args = Python.tuple(args[1])
    var clustering = operation.startswith("family_cfb_sc_")
    if model is None:
        return _cfb_spectral_algorithm_operation(model, operation, args, context, supplied_binding)
    var fitting = operation == "family_cfb_sc_fit" or operation == "family_cfb_sc_fit_predict" or operation == "family_cfb_se_fit" or operation == "family_cfb_se_fit_transform"
    var fields = spectral_admit(model, clustering, fitting, context)
    if operation == "family_cfb_se_validate":
        for name in ["_n_components", "_n_neighbors", "_seed", "_gamma"]:  # small-loop(private validation output fields)
            if py_contains(fields, name):
                native_release_admit(fields[name], context)
        return _cfb_spectral_algorithm_operation(model, operation, args, context, supplied_binding)
    var method = String("fit_predict") if operation.endswith("_fit_predict") else (String("fit_transform") if operation.endswith("_fit_transform") else (String("fit") if fitting else String("predict")))
    var public_prediction = operation == "family_cfb_sc_predict"
    if public_prediction:
        classical_require_fitted(model, False, method, context)
    if operation == "family_cfb_se_knn_affinity" and py_contains(fields, "affinity_matrix_"):
        native_release_admit(fields["affinity_matrix_"], context)
    var source = args[0]
    var name = classical_estimator_name(model)
    var prepared = spectral_input(source, name, context)
    var rows = Int(py=prepared.shape[0])
    var columns = Int(py=prepared.shape[1])
    var depth = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0)))
    if depth == 0:
        if fitting:
            _target(args[1], rows, name, context)
        elif public_prediction:
            var width = py_get(fields, "n_features_in_")
            if py_type(width) is b.int:
                classical_check_width(columns, Int(py=width), name)
            estimator_feature_check(model, fields, source, context)
    var initial_depth = estimator_fit_begin(fields) if fitting else 0
    var outside = estimator_outside()
    var result = PythonObject(None)
    try:
        result = _cfb_spectral_algorithm_operation(model, operation, Python.tuple(prepared), context, supplied_binding)
    except error:
        var refusal = _refusal(error, fitting, depth > 0, rows, columns, name)
        raise refusal^
    finally:
        if fitting:
            estimator_fit_end(model, initial_depth)
    if fitting:
        if not py_contains(fields, "n_features_in_"):
            py_setitem(fields, "n_features_in_", PythonObject(columns))
        estimator_fit_complete(model, source, False, context)
    if (outside and (fitting or public_prediction)
            and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None):
        return native_array_ndarray_view(result, context)
    return result
