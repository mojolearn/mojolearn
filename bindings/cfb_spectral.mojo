# SPDX-License-Identifier: Apache-2.0
# SpectralClustering `fit`, `fit_predict` and `predict`, SpectralEmbedding
# `fit` and `fit_transform`, and `spectral_embedding`, one native call each
# (python/mojolearn/_spectral_impl.py). The shells hand over the metrics
# binding (`_get_binding()`: the dense, graph and dataset entries), the
# x_decomp binding of the default mode (the RBF affinity cells), the
# IDENTICAL base binding (the host COO scans and the precomputed kNN
# affinity, `_buffer._native`'s entries) and, for SpectralClustering, the
# x_cluster binding (discretize / cluster_qr). This owner makes
# `_validate_params`' and the fits' refusals in their order and wording and
# calls those entries with the argument lists, in the order, the Python
# made (the RBF affinity takes the `_Kit` route the Python took: the
# resident cells where the binding has them, else the host cells). Same
# entries, same inputs: the bits do not move.
from std.math import isinf
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address, classical_dtype_str, classical_c_order
from core.family_cast_impl import family_all_finite_f32
from core.cfb_impl import cfb_f64_to_i32
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, family_repr, family_str, family_call, family_entry,
    family_array, family_as_f32, family_view, family_cast_view_f64, family_field, family_is_str, family_min,
)
from bindings.cluster_native import _reference_queries, _xc_array
from bindings.cfb_resident import cfb_optional_entry
from bindings.cfb_dbscan import cfb_borrow_copy

comptime _DENSE_EMPTY = -1
comptime _DENSE_NONFINITE = -2
comptime _DENSE_NEGATIVE = -3
#: x_decomp/cells.mojo op codes (`_expansion_decomp._OP`).
comptime _OP_EXP = 9
comptime _OP_SCALE = 14
#: `x_cluster/entries.mojo::ENTRY_SPECTRAL_ASSIGN`.
comptime _E_SPECTRAL_ASSIGN = 12


def _builtins() raises -> PythonObject:
    return Python.import_module("builtins")


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
    return Bool(py=family_call(_builtins().bool, family_call(Python.import_module("operator").eq, shape[0], shape[1])))


def _as_i32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_i32_c(source, ndim=1, name=name)`: the rank and emptiness
    refusals, then the vector itself when it is already C-order int32, else
    one int32 copy."""
    var view = family_view(source, 1, name, context)
    if classical_dtype_str(view, context) == "<i4" and classical_c_order(view, context):
        return view
    var wide = family_cast_view_f64(view, name, context)
    var n = Int(py=wide.size)
    var out = family_array(Python.tuple(n), "<i4", context)
    cfb_f64_to_i32(FAMILY_F64P(unsafe_from_address=Int(py=classical_address(wide))),
                   FAMILY_I32P(unsafe_from_address=Int(py=classical_address(out))), n)
    _ = wide^
    _ = view^
    return out


def _tocoo(obj: PythonObject) raises -> PythonObject:
    """`getattr(obj, "tocoo", None)` when it is callable, else None."""
    var b = _builtins()
    var tocoo = py_call(b.getattr, obj, PythonObject("tocoo"), PythonObject(None))
    if tocoo is None or not Bool(py=family_call(b.callable, tocoo)):
        return PythonObject(None)
    return tocoo


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
    var v = family_as_f32(vals, 1, "vals", context)
    return _Coo(r, c, v, _int(shape[0]))


def _coo_triples(source: PythonObject, who: String, base: PythonObject, context: PythonObject) raises -> _Coo:
    """`_coo_triples(A, who)`: a sparse input through its `tocoo()`, a dense
    one through the base binding's float64 nonzero scan (exact zeros
    dropped, as `sp.coo_matrix(dense)` drops them)."""
    var b = _builtins()
    var tocoo = _tocoo(source)
    if tocoo is not None:
        var coo = family_call(tocoo)
        var shape = family_call(b.tuple, coo.shape)
        return _coo_of_triples(coo.row, coo.col, coo.data, shape, who, context)
    var shape = _shape_of(source)
    if not _square(shape):
        raise Error("mojolearn " + who + ": with affinity='precomputed', X "
                    + "must be a square affinity matrix or a sparse matrix with "
                    + "`tocoo()`, got shape " + family_repr(shape))
    var view = family_view(source, 2, "X", context)
    var dense = family_cast_view_f64(view, "X", context)
    _ = view^
    var n = _int(shape[0])
    var size = Int(py=dense.size)
    var nnz = 0
    if size:
        nnz = Int(py=family_call(family_entry(base, "nonzero_f64_count", context), classical_address(dense),
                                 PythonObject(size)))
    var rows = family_array(Python.tuple(nnz), "<i4", context)
    var cols = family_array(Python.tuple(nnz), "<i4", context)
    var vals = family_array(Python.tuple(nnz), "<f4", context)
    if nnz == 0:
        return _Coo(rows, cols, vals, n)
    var wrote = Int(py=family_call(family_entry(base, "nonzero_f64_fill", context), classical_address(dense),
                                   PythonObject(n), PythonObject(n),
                                   Python.list(classical_address(rows), classical_address(cols), classical_address(vals)),
                                   PythonObject(nnz)))
    _ = dense^
    if wrote != nnz:
        raise Error("RuntimeError: mojolearn " + who + ": nonzero_f64_fill wrote "
                    + String(wrote) + " of " + String(nnz) + " entries")
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
        if wrote != nnz:
            raise Error("RuntimeError: mojolearn: nonzero_f32_fill wrote " + String(wrote) + " of "
                        + String(nnz) + " entries")
    return _coo_of_triples(rows, cols, data, Python.tuple(n_r, n_c), who, context)


def _x_decomp_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_M.from_input(X)`: a scipy.sparse matrix densified (its own
    `toarray()`), `as_f32_c`, and the finiteness refusal."""
    var b = _builtins()
    var x = source
    if (Bool(py=family_call(b.hasattr, x, PythonObject("toarray"))) and Bool(py=family_call(b.hasattr, x, PythonObject("nnz")))
            and Bool(py=family_call(b.hasattr, x, PythonObject("format")))):
        x = family_call(py_call(b.getattr, x, PythonObject("toarray")))
    var a = family_as_f32(x, 2, "X", context)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(a))), Int(py=a.size)):
        raise Error("X: input must be finite; NaN/inf are unsupported")
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
        except error:
            _cells_free(free, ids)
            raise error^
        _cells_free(free, ids)
        return out
    var sq = family_array(Python.tuple(n, n), "<f4", context)
    _ = family_call(family_entry(xdecomp, "x_decomp_sqdist", context), classical_address(x), classical_address(x),
                    classical_address(sq), Python.list(PythonObject(n), PythonObject(n), PythonObject(d)))
    var one = family_array(Python.tuple(1), "<f4", context)
    var host_ew = family_entry(xdecomp, "x_decomp_ew", context)
    var scaled = family_array(Python.tuple(n, n), "<f4", context)
    _ = family_call(host_ew, classical_address(sq), classical_address(one), classical_address(one),
                    classical_address(scaled),
                    Python.list(PythonObject(_OP_SCALE), PythonObject(nn), PythonObject(n), PythonObject(1),
                                PythonObject(3), PythonObject(1), PythonObject(3)), scale)
    _ = sq^
    _ = family_call(host_ew, classical_address(scaled), classical_address(one), classical_address(one),
                    classical_address(out),
                    Python.list(PythonObject(_OP_EXP), PythonObject(nn), PythonObject(n), PythonObject(1),
                                PythonObject(3), PythonObject(1), PythonObject(3)), PythonObject(0.0))
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
    var tocoo = _tocoo(source)
    var n: Int
    var nnz = 0
    var dense = PythonObject(None)
    var rows = PythonObject(None)
    var cols = PythonObject(None)
    var vals = PythonObject(None)
    if tocoo is not None:
        var coo = family_call(tocoo)
        n = _int(coo.shape[0])
        if not Bool(py=family_call(b.bool, family_call(Python.import_module("operator").eq,
                                                        family_call(b.tuple, coo.shape), Python.tuple(n, n)))):
            raise Error("mojolearn SpectralEmbedding: the precomputed distance graph must be square")
        rows = _as_i32(coo.row, "rows", context)
        cols = _as_i32(coo.col, "cols", context)
        vals = family_as_f32(coo.data, 1, "data", context)
        if not (Int(py=rows.shape[0]) == Int(py=cols.shape[0]) and Int(py=cols.shape[0]) == Int(py=vals.shape[0])):
            raise Error("mojolearn SpectralEmbedding: the precomputed graph's row, col and data differ in length")
        nnz = Int(py=vals.shape[0])
    else:
        dense = family_as_f32(source, 2, "X", context)
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
    _ = status  # keep the Python object alive through the native call above
    var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(status)))
    _ = status  # keep the Python object alive through the native call above
    var code = Int(words[0])
    _ = status  # keep the Python object alive through the native call above
    if code == 1:
        raise Error("mojolearn SpectralEmbedding: a precomputed distance is negative or NaN")
    if code == 2:
        raise Error("mojolearn SpectralEmbedding: row " + String(Int(words[1])) + " of the precomputed graph has "
                    + String(Int(words[2])) + " stored distances, fewer than n_neighbors=" + String(k))
        _ = status  # keep the Python object alive through the native call above
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
    var operator = Python.import_module("operator")
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


def _state_arrays(n: Int, k: Int, clusters: Int, context: PythonObject) raises -> PythonObject:
    """`_state_arrays`: eigenvalues (k,), eigenvectors (n, k), diag (n,),
    centroids (n_clusters, k), float32."""
    return Python.tuple(family_array(Python.tuple(k), "<f4", context), family_array(Python.tuple(n, k), "<f4", context),
                        family_array(Python.tuple(n), "<f4", context),
                        family_array(Python.tuple(clusters, k), "<f4", context))


def _addresses(head: PythonObject, state: PythonObject) raises -> PythonObject:
    """`head + [addr(a) for a in state]` (the prediction-state buffers)."""
    var out = py_call(_builtins().list, head)
    for i in range(len(state)):  # small-loop(state: the four prediction-state buffers): 4
        _ = py_call(out.append, classical_address(state[i]))
    return out


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
    var dense_fn = PythonObject(None)
    if graph_affinity:
        dense_fn = cfb_optional_entry(metrics, "spectral_fit_predict_dense", context)
    # `dense` is the matrix the dense entry reads (an Array); `coo_source`
    # the input of the COO route.
    var dense = PythonObject(None)
    var from_matrix = False
    var coo_source = source
    if affinity == "precomputed":
        py_setitem(fields, "affinity_matrix_", source)
        if dense_fn is not None and _tocoo(source) is None:
            var shape = _shape_of(source)
            if not _square(shape):
                raise Error("mojolearn SpectralClustering: with affinity='precomputed', X "
                            + "must be a square affinity matrix or a sparse matrix with "
                            + "`tocoo()`, got shape " + family_repr(shape))
            dense = family_as_f32(source, 2, "X", context)
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
    var state = PythonObject(None)
    var n_out: Int
    var n: Int
    var params_tail = Python.list(PythonObject(clusters), PythonObject(k), family_field(fields, "_n_init"),
                                  family_field(fields, "_n_neighbors"), family_field(fields, "_eigen_tol"),
                                  family_field(fields, "_seed"))
    if dense_fn is not None and dense is not None:
        # the dense matrix straight to the device entry
        n = Int(py=dense.shape[0])
        _sc_check_shape(fields, n, k)
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_fit_predict_dense_binding.
        var params = Python.list(PythonObject(n))
        _ = py_call(params.extend, params_tail)
        var addrs = Python.list(classical_address(dense), classical_address(labels), classical_address(embedding))
        if predicting:
            state = _state_arrays(n, k, clusters, context)
            addrs = _addresses(addrs, state)
        n_out = Int(py=family_call(dense_fn, addrs, params))
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
        if not family_all_finite_f32(vp, nnz):
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
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_fit_predict_graph_binding:
        # n_samples, nnz, n_clusters, n_components, n_init, n_neighbors, eigen_tol, seed.
        var params = Python.list(PythonObject(n), PythonObject(nnz))
        _ = py_call(params.extend, params_tail)
        if predicting:
            state = _state_arrays(n, k, clusters, context)
            n_out = Int(py=family_call(family_entry(metrics, "spectral_fit_predict_graph_state", context),
                                       _addresses(Python.list(classical_address(coo.rows), classical_address(coo.cols),
                                                              classical_address(coo.vals), classical_address(labels),
                                                              classical_address(embedding)), state), params))
        else:
            n_out = Int(py=family_call(family_entry(metrics, "spectral_fit_predict_graph", context),
                                       classical_address(coo.rows), classical_address(coo.cols),
                                       classical_address(coo.vals), classical_address(labels),
                                       classical_address(embedding), params))
    else:
        var x = family_as_f32(source, 2, "X", context)
        py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
        if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
            raise Error("mojolearn SpectralClustering: X contains NaN or "
                        + "infinity (the implemented path refuses it by name before any "
                        + "recorded stage)")
        n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        _sc_check_shape(fields, n, k)
        var neighbors = Int(py=family_field(fields, "_n_neighbors"))
        if neighbors > n:
            raise Error("mojolearn SpectralClustering: n_neighbors="
                        + String(neighbors) + " exceeds n_samples=" + String(n))
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_fit_predict_dataset_binding:
        # n_samples, n_features, n_clusters, n_components, n_init, n_neighbors, eigen_tol, seed.
        var params = Python.list(PythonObject(n), PythonObject(d))
        _ = py_call(params.extend, params_tail)
        if predicting:
            state = _state_arrays(n, k, clusters, context)
            n_out = Int(py=family_call(family_entry(metrics, "spectral_fit_predict_dataset_state", context),
                                       _addresses(Python.list(classical_address(x), classical_address(labels),
                                                              classical_address(embedding)), state), params))
            # A borrowed input is copied so a caller's later write cannot
            # move what predict reads.
            py_setitem(fields, "_fit_X", cfb_borrow_copy(x, source, context))
        else:
            n_out = Int(py=family_call(family_entry(metrics, "spectral_fit_predict_dataset", context),
                                       classical_address(x), classical_address(labels), classical_address(embedding),
                                       params))
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
        if String(py=assign) == "discretize":
            py_setitem(fields, "n_iter_assign_", PythonObject(_int(result[2][0])))
        labels = _xc_array(result[1][0], Python.tuple(n), "<i4", "i", context)
    py_setitem(fields, "labels_", labels)
    if state is not None:
        py_setitem(fields, "_pd_eigenvalues", state[0])
        py_setitem(fields, "_pd_eigenvectors", state[1])
        py_setitem(fields, "_pd_diag", state[2])
        py_setitem(fields, "_pd_centroids", state[3])


def _sc_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`SpectralClustering._predict_embedding`: (labels, embedding). The Nystrom extension of the fitted
    eigenvectors, then the nearest fitted k-means centroid."""
    var b = _builtins()
    if not Bool(py=family_call(b.hasattr, model, PythonObject("labels_"))):
        raise Error("mojolearn SpectralClustering.predict: this instance is not "
                    + "fitted yet; call fit first")
    var eigenvalues = py_call(b.getattr, model, PythonObject("_pd_eigenvalues"), PythonObject(None))
    if eigenvalues is None:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not "
                    + "stored. Fit with SpectralClustering(prediction_data=True), which "
                    + "keeps the eigenpairs, degrees and centroids the Nystrom extension "
                    + "needs (DEVIATION 2860)")
    var affinity = py_call(b.getattr, model, PythonObject("affinity"))
    var affinities = context["family_cfb_predict_affinities"]
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralClustering.predict: affinity=" + family_repr(affinity) + " "
                    + "has no out-of-sample rule; it must be one of "
                    + family_repr(py_call(b.list, affinities)))
    var k = _int(py_call(b.getattr, model, PythonObject("n_components_")))
    var diag = py_call(b.getattr, model, PythonObject("_pd_diag"))
    var n_train = Int(py=diag.shape[0])
    var q = PythonObject(None)
    var train_addr = PythonObject(0)
    var n_features = 0
    var code = 0
    if String(py=affinity) == "precomputed":
        var a = source
        var toarray = py_call(b.getattr, source, PythonObject("toarray"), PythonObject(None))
        if toarray is not None and Bool(py=family_call(b.callable, toarray)):
            a = family_call(toarray)
        var shape = _shape_of(a)
        if len(shape) != 2 or not Bool(py=family_call(b.bool, family_call(Python.import_module("operator").eq,
                                                                           shape[1], PythonObject(n_train)))):
            raise Error("mojolearn SpectralClustering.predict: with affinity="
                        + "'precomputed', X must be the affinity between the new rows "
                        + "and the " + String(n_train) + " training rows, shape (n_new, " + String(n_train) + "); "
                        + "got shape " + family_repr(shape))
        q = family_as_f32(a, 2, "X", context)
        if Int(py=q.shape[0]) < 1:
            raise Error("mojolearn SpectralClustering.predict: X has no rows; refused by name")
        if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(q))), Int(py=q.size)):
            raise Error("mojolearn SpectralClustering.predict: the affinity has a "
                        + "non-finite entry; refused by name")
        if family_min(q) < 0:
            raise Error("mojolearn SpectralClustering.predict: the affinity has a "
                        + "negative entry; refused by name, as the fit refuses one")
        code = 1
    else:
        var fitted = Int(py=py_call(b.getattr, model, PythonObject("n_features_in_")))
        q = _reference_queries(source, fitted, "SpectralClustering.predict", context)
        train_addr = classical_address(py_call(b.getattr, model, PythonObject("_fit_X")))
        n_features = fitted
    var nq = Int(py=q.shape[0])
    var labels = family_array(Python.tuple(nq), "<i4", context)
    var emb = family_array(Python.tuple(nq, k), "<f4", context)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_predict_binding:
    # n_train, n_queries, n_features, n_components, n_clusters, n_neighbors, affinity.
    _ = family_call(family_entry(binding, "spectral_predict", context),
                    Python.list(classical_address(q), train_addr, classical_address(eigenvalues),
                                classical_address(py_call(b.getattr, model, PythonObject("_pd_eigenvectors"))),
                                classical_address(diag),
                                classical_address(py_call(b.getattr, model, PythonObject("_pd_centroids"))),
                                classical_address(labels), classical_address(emb)),
                    Python.list(PythonObject(n_train), PythonObject(nq), PythonObject(n_features), PythonObject(k),
                                PythonObject(_int(py_call(b.getattr, model, PythonObject("_n_clusters")))),
                                PythonObject(_int(py_call(b.getattr, model, PythonObject("_n_neighbors")))),
                                PythonObject(code)))
    _ = q^
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
    var rbf = Bool(py=family_call(b.bool, family_call(Python.import_module("operator").eq, affinity, PythonObject("rbf"))))
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
    var drop_first = Bool(py=family_call(b.bool, py_call(b.getattr, model, PythonObject("_drop_first"))))
    var norm_laplacian = Bool(py=family_call(b.bool, py_call(b.getattr, model, PythonObject("_norm_laplacian"))))
    # The one piece of arithmetic the reference keeps in Python
    # (spectral_embedding.pyx:294): the transform receives this number.
    var n_lanczos = k + 1 if drop_first else k
    var affinity = String(py=family_field(fields, "affinity"))
    var graph_affinity = affinity == "precomputed" or affinity == "rbf" or affinity == "precomputed_nearest_neighbors"
    var dense_fn = PythonObject(None)
    if graph_affinity:
        dense_fn = cfb_optional_entry(metrics, "spectral_embedding_dense", context)
    var dense = PythonObject(None)
    var from_matrix = False
    if affinity == "precomputed_nearest_neighbors":
        dense = _knn_affinity(fields, source, False, args[2], context)
        from_matrix = True
    if affinity == "precomputed" and dense_fn is not None and _tocoo(source) is None:
        var shape = _shape_of(source)
        if not _square(shape):
            raise Error("mojolearn SpectralEmbedding: with affinity='precomputed', X "
                        + "must be a square affinity matrix or a sparse matrix with "
                        + "`tocoo()`, got shape " + family_repr(shape))
        dense = family_as_f32(source, 2, "X", context)
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
    var tol_tail = Python.list()
    var eigen_tol = family_field(fields, "eigen_tol")
    if eigen_tol is not None and not family_is_str(eigen_tol):
        _ = py_call(tol_tail.append, _float(eigen_tol))
    var seed = family_field(fields, "_seed")
    var embedding = PythonObject(None)
    var n_out: Int
    if dense_fn is not None and dense is not None:
        var n = Int(py=dense.shape[0])
        _se_check_shape(fields, n, n_lanczos)
        py_setitem(fields, "n_neighbors_", PythonObject(_resolved_neighbors(fields, n, False)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_embedding_dense_binding.
        var params = Python.list(PythonObject(n), PythonObject(n_lanczos), PythonObject(k), PythonObject(Int(norm_laplacian)),
                                 PythonObject(Int(drop_first)), seed)
        _ = py_call(params.extend, tol_tail)
        n_out = Int(py=family_call(dense_fn, classical_address(dense), classical_address(embedding), params))
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
            if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coo.vals))), nnz):
                raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                            + "matrix has a non-finite entry")
            if family_min(coo.vals) < 0:
                raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                            + "matrix has a negative entry (refused by name: sqrt of a "
                            + "negative degree is a NaN)")
        _se_check_shape(fields, n, n_lanczos)
        py_setitem(fields, "n_neighbors_", PythonObject(_resolved_neighbors(fields, n, False)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_embedding_graph_binding.
        var params = Python.list(PythonObject(n), PythonObject(nnz), PythonObject(n_lanczos), PythonObject(k),
                                 PythonObject(Int(norm_laplacian)), PythonObject(Int(drop_first)), seed)
        _ = py_call(params.extend, tol_tail)
        n_out = Int(py=family_call(family_entry(metrics, "spectral_embedding_graph", context),
                                   classical_address(coo.rows), classical_address(coo.cols), classical_address(coo.vals),
                                   classical_address(embedding), params))
    else:
        var x = family_as_f32(source, 2, "X", context)
        py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
        if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
            raise Error("mojolearn SpectralEmbedding: X contains NaN or infinity")
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        _se_check_shape(fields, n, n_lanczos)
        var neighbors = _resolved_neighbors(fields, n, False)
        if neighbors > n:
            raise Error("mojolearn SpectralEmbedding: n_neighbors=" + String(neighbors) + " "
                        + "exceeds n_samples=" + String(n))
        py_setitem(fields, "n_neighbors_", PythonObject(neighbors))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_embedding_dataset_binding.
        var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(n_lanczos), PythonObject(k),
                                 PythonObject(neighbors), PythonObject(Int(norm_laplacian)), PythonObject(Int(drop_first)),
                                 seed)
        _ = py_call(params.extend, tol_tail)
        n_out = Int(py=family_call(family_entry(metrics, "spectral_embedding_dataset", context),
                                   classical_address(x), classical_address(embedding), params))
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


def cfb_spectral_operation(model: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    """SpectralClustering fit args: (metrics, x_decomp, base, x_cluster, X);
    predict args: (binding, X). SpectralEmbedding fit args: (metrics,
    x_decomp, base, X)."""
    if operation == "family_cfb_sc_fit":
        _sc_fit(model, args, context)
        return model
    if operation == "family_cfb_sc_fit_predict":
        _sc_fit(model, args, context)
        return py_fields(model)["labels_"]
    if operation == "family_cfb_sc_predict":
        return _sc_predict(model, args[0], args[1], context)[0]
    if operation == "family_cfb_sc_predict_embedding":
        return _sc_predict(model, args[0], args[1], context)
    if operation == "family_cfb_se_validate":
        _se_validate(py_fields(model), context)
        return PythonObject(None)
    if operation == "family_cfb_se_knn_affinity":
        return _knn_affinity(py_fields(model), args[1], False, args[0], context)
    if operation == "family_cfb_coo_triples":
        var coo = _coo_triples(args[1], String(py=args[2]), args[0], context)
        return Python.tuple(coo.rows, coo.cols, coo.vals, PythonObject(coo.n))
    if operation == "family_cfb_se_fit":
        _se_fit(model, args, args[3], context)
        return model
    if operation == "family_cfb_se_fit_transform":
        _se_fit(model, args, args[3], context)
        return py_fields(model)["embedding_"]
    if operation == "family_cfb_spectral_embedding":
        return _spectral_embedding(args, context)
    raise Error("RuntimeError: unknown spectral operation " + operation)
