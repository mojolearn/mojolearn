# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): LocalOutlierFactor, NearestCentroid and KNNImputer public methods
# (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_loc_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
#
# The shell hands over the binding `_bind()` resolved (the x_neighbors
# binding of the estimator's numeric mode) first in `args`. Every `xn_*`
# primitive is called as `_XNeighbors._op` called it: one list of buffer
# addresses, one list of ints, one list of floats. The A/B arms the module
# read at import (MOJOLEARN_XN_UNFUSED_KNN, MOJOLEARN_XN_UNCOMPACT_IMPUTE,
# MOJOLEARN_XN_OLD_ITEMS) are the import-time values in the context
# (`family_xn_loc_flags`); MOJOLEARN_NC_SPLIT_OPS is read at each fit, as the
# Python fit read it. The opt-in lazy output map (MOJOLEARN_XN_LAZY_OUT) is a
# memory arm only: outputs are zero-filled Arrays either way, same values.
# Scalars derived on the host (the contamination percentile, the prior
# checks) are the same binary64 operations in the same order; products that
# feed a sum are pinned (`pinned_mul_f64`), since Python never fuses.
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_module, xn_is_module
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.math import floor, isinf
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.host_sparse_native import host_sparse_format
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_call, family_entry,
    family_as_f32_dense, family_field, family_isinstance,
)
from bindings.classical_labels import (
    labels_empty, labels_vector, labels_cast, labels_view, labels_check_rank, labels_helper, classical_encode_labels,
    classical_decode_labels, classical_accuracy,
)
from checks.numerics import pinned_mul_f64

# x_neighbors/items.mojo's unary codes (`_U_SQRT`, `_U_IDENTITY`).
comptime _U_SQRT = 2
comptime _U_IDENTITY = 6


# ------------------------------------------------------------------ glue


def _admit_local_state(model: PythonObject, context: PythonObject) raises:
    # Mutable settings are re-admitted before comparison, coercion or truth
    # slots. Input metadata is captured by the consuming native buffer helper.
    classical_check_layout(model, context)
    var fields = py_fields(model)
    var scalars = Python.tuple("metric", "p", "contamination", "n_neighbors", "novelty", "shrink_threshold",
        "missing_values", "weights", "add_indicator", "keep_empty_features", "n_features_in_",
        "n_neighbors_", "n_samples_fit_", "offset_")
    for i in range(len(scalars)):  # small-loop(fixed scalar parameter and fitted-state schema)
        var value = py_get(fields, scalars[i])
        if value is not None:
            host_inference_scalar(value, context, True)
    var values = Python.tuple("metric_params", "priors", "classes_", "_fit_X", "_fit_dist", "_lrd",
        "centroids_", "class_prior_", "within_class_std_dev_", "_valid", "_miss_cols",
        "negative_outlier_factor_")
    for i in range(len(values)):  # small-loop(fixed native container and buffer owners)
        var value = py_get(fields, values[i])
        if value is not None:
            classical_admit_value(classifier_label_list(value, context) if String(values[i]) == "classes_" else value, context)


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`, as an `if` or `and`/`or` took it."""
    return Bool(py=family_call(_b().bool, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _ne(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a != b)`."""
    return _truth(family_call(native_loaded_module("operator").ne, a, b))


def _flag(context: PythonObject, name: String) raises -> Bool:
    """An A/B arm the module read at import."""
    return Bool(py=context["family_xn_loc_flags"][PythonObject(name)])


def _rows(a: PythonObject) raises -> Int:
    return Int(py=a.shape[0])


def _cols(a: PythonObject) raises -> Int:
    return Int(py=a.shape[1])


def _size(a: PythonObject) raises -> Int:
    return Int(py=a.size)


def _f32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_f32(X)`: `as_f32_c(_dense(X), ndim=2, name="X")`."""
    return family_as_f32_dense(source, 2, "X", context)


def _out(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`empty(shape, typestr)` / `_empty_out(shape)`: a new zero-filled Array."""
    return labels_empty(shape, typestr, context)


def _op(binding: PythonObject, name: String, addrs: PythonObject, ints: PythonObject, floats: PythonObject,
        context: PythonObject) raises:
    """`_XNeighbors._op(name, bufs, ints, floats)`: the binding's `xn_<name>`
    with the address, int and float lists. The caller keeps every buffer
    alive across the call."""
    xn_execute(binding, name, addrs, ints, floats)


def _i(v: Int) -> PythonObject:
    return PythonObject(v)


def _pow2_steps(count: Int) -> Tuple[Int, Int]:
    """`(p, lg * (lg + 1) // 2)`: the power-of-two pad of `count` (p = 1,
    doubled while p < count) and the bitonic network's step count, with
    lg = p.bit_length() - 1."""
    var p = 1
    var lg = 0
    while p < count:  # small-loop(bits: doublings to the next power of two of one count): at most 63
        p *= 2
        lg += 1
    return (p, lg * (lg + 1) // 2)


def _f32_scalar(value: PythonObject) raises -> Float64:
    """`_f32_scalar(v)`: `float(v)` rounded once to float32."""
    return Float64(Float32(Float64(py=family_call(_b().float, value))))


def _isclose(a: Float64, b: Float64, rel_tol: Float64, abs_tol: Float64) -> Bool:
    """`_portable_math.isclose` (finite tolerances, already floats)."""
    if a == b:
        return True
    if isinf(a) or isinf(b):
        return False
    var difference = abs(b - a)
    return difference <= abs(rel_tol * b) or difference <= abs(rel_tol * a) or difference <= abs_tol


def _astype_i64(lab: PythonObject, context: PythonObject) raises -> PythonObject:
    """`lab.astype("<i8")` of an int32 vector Array."""
    var n = _size(lab)
    var out = labels_vector(n, "<i8", context)
    labels_cast(Int(py=classical_address(lab)), "<i4", Int(py=classical_address(out)), "<i8", n, context)
    _ = lab  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _copy_f32(x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`X.copy()` of a C-order float32 array: a new Array of its shape."""
    var out = _out(py_call(_b().tuple, x.shape), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=_size(x))
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _sqdist(binding: PythonObject, a: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_XNeighbors._sqdist(A, B)`."""
    var n = _rows(a)
    var d = _cols(a)
    var m = _rows(b)
    var out = _out(Python.tuple(n, m), "<f4", context)
    _op(binding, "sqdist", Python.list(classical_address(a), classical_address(b), classical_address(out)),
        Python.list(_i(n), _i(m), _i(d)), Python.list(), context)
    _ = a  # keep the Python object alive through the native call above
    _ = b  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _l1dist(binding: PythonObject, a: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_XNeighbors._l1dist(A, B)`."""
    var n = _rows(a)
    var d = _cols(a)
    var m = _rows(b)
    var out = _out(Python.tuple(n, m), "<f4", context)
    _op(binding, "l1dist", Python.list(classical_address(a), classical_address(b), classical_address(out)),
        Python.list(_i(n), _i(m), _i(d)), Python.list(), context)
    _ = a  # keep the Python object alive through the native call above
    _ = b  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _knn_select(binding: PythonObject, dmat: PythonObject, k: Int, exclude_self: Bool,
                context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_XNeighbors._knn_select(D, k, exclude_self)`: (dist, idx)."""
    var n = _rows(dmat)
    var m = _cols(dmat)
    var dist = _out(Python.tuple(n, k), "<f4", context)
    var idx = _out(Python.tuple(n, k), "<i4", context)
    _op(binding, "knn_select", Python.list(classical_address(dmat), classical_address(dist), classical_address(idx)),
        Python.list(_i(n), _i(m), _i(k), _i(1 if exclude_self else 0)), Python.list(), context)
    _ = dmat  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = idx  # keep the Python object alive through the native call above
    return (dist, idx)


def _unary(binding: PythonObject, x: PythonObject, code: Int, a: Float64, bias: Float64,
           context: PythonObject) raises -> PythonObject:
    """`_XNeighbors._unary(X, op, a, b)`."""
    var out = _out(py_call(_b().tuple, x.shape), "<f4", context)
    _op(binding, "unary", Python.list(classical_address(x), classical_address(out)),
        Python.list(_i(_size(x)), _i(code)), Python.list(PythonObject(a), PythonObject(bias)), context)
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _knn(binding: PythonObject, q: PythonObject, r: PythonObject, k: Int, exclude_self: Bool,
         context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_XNeighbors._knn(Q, R, k, exclude_self)`: `_knn_sq` (the fused
    `knn_sq_tiled`, `knn_sq` under MOJOLEARN_XN_OLD_ITEMS, sqdist +
    knn_select under MOJOLEARN_XN_UNFUSED_KNN), then the square root."""
    var sq: PythonObject
    var idx: PythonObject
    if _flag(context, "unfused_knn"):
        var pair = _knn_select(binding, _sqdist(binding, q, r, context), k, exclude_self, context)
        sq = pair[0]
        idx = pair[1]
    else:
        var n = _rows(q)
        var d = _cols(q)
        var m = _rows(r)
        sq = _out(Python.tuple(n, k), "<f4", context)
        idx = _out(Python.tuple(n, k), "<i4", context)
        _op(binding, "knn_sq" if _flag(context, "old_items") else "knn_sq_tiled",
            Python.list(classical_address(q), classical_address(r), classical_address(sq), classical_address(idx)),
            Python.list(_i(n), _i(m), _i(d), _i(k), _i(1 if exclude_self else 0)), Python.list(), context)
        _ = q  # keep the Python object alive through the native call above
        _ = r  # keep the Python object alive through the native call above
    var root = _unary(binding, sq, _U_SQRT, 1.0, 0.0, context)
    _ = sq  # keep the Python object alive through the native call above
    return (root, idx)


def _i32_from_ints(values: List[Int], context: PythonObject) raises -> PythonObject:
    """`_i32(list)`: an int32 vector Array of a short int list (class
    offsets, kept columns, flags)."""
    var out = labels_vector(len(values), "<i4", context)
    var p = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(out)))
    for i in range(len(values)):  # driver-loop: over a class- or feature-sized int list of model structure
        p[i] = Int32(values[i])
    _ = out  # keep the Python object alive through the writes above
    return out


# ------------------------------------------------------------------ LOF


def _lof_check(fields: PythonObject) raises:
    """`LocalOutlierFactor._check_params`."""
    var b = _b()
    var metric = family_field(fields, "metric")
    var p = family_field(fields, "p")
    if not (_eq(metric, PythonObject("euclidean")) or (_eq(metric, PythonObject("minkowski")) and _eq(p, PythonObject(2)))):
        raise Error("NotImplementedError: LocalOutlierFactor: metric=" + family_repr(metric) + " (p=" + family_repr(p)
                    + ") is not implemented; euclidean (or minkowski with p=2) only")
    if _truth(family_field(fields, "metric_params")):
        raise Error("NotImplementedError: LocalOutlierFactor: metric_params is not implemented")
    var contamination = family_field(fields, "contamination")
    if _ne(contamination, PythonObject("auto")):
        var c = Float64(py=family_call(b.float, contamination))
        if not (0.0 < c and c <= 0.5):
            raise Error("contamination must be 'auto' or in (0, 0.5]")
    if Int(py=family_call(b.int, family_field(fields, "n_neighbors"))) < 1:
        raise Error("n_neighbors must be >= 1")


def _lof_percentile(binding: PythonObject, score: PythonObject, q: Float64, context: PythonObject) raises -> Float64:
    """`LocalOutlierFactor._sorted_percentile(score, q)`: numpy.percentile
    (method 'linear') with the sort on the device (`_p2m_sort_rows` over one
    row: `xn_p2m_row_sort`, ascending by value, ties by position); only the
    two order statistics it reads come back."""
    var n = _rows(score)
    var indptr = _i32_from_ints([0, n], context)
    var cols = _out(Python.tuple(n), "<i4", context)
    var nnz = _rows(score)
    var steps = _pow2_steps(nnz)
    var oc = _out(Python.tuple(nnz), "<i4", context)
    var od = _out(Python.tuple(nnz), "<f4", context)
    xn_execute(binding, "p2m_row_sort", Python.list(classical_address(indptr), classical_address(cols), classical_address(score),
                                       classical_address(oc), classical_address(od)),
                    Python.list(_i(_rows(indptr) - 1), _i(nnz), _i(steps[0]), _i(steps[1])), Python.list())
    _ = indptr  # keep the Python object alive through the native call above
    _ = cols  # keep the Python object alive through the native call above
    _ = score  # keep the Python object alive through the native call above
    _ = oc  # keep the Python object alive through the native call above
    var pos = pinned_mul_f64(q / 100.0, Float64(n - 1))
    var lo = Int(floor(pos))
    var hi = min(lo + 1, n - 1)
    var t = pos - Float64(lo)
    var srt = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(od)))
    var a = Float64(srt[lo])
    var bv = Float64(srt[hi])
    _ = od  # keep the Python object alive through the reads above
    var diff = bv - a
    if t >= 0.5:
        return bv - pinned_mul_f64(diff, 1.0 - t)
    return a + pinned_mul_f64(diff, t)


def _lof_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LocalOutlierFactor.fit(X)`."""
    var b = _b()
    var fields = py_fields(model)
    _lof_check(fields)
    var binding = args[0]
    var x = _f32(args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    if n < 2:
        # scikit-learn's check_array wording (ensure_min_samples=2)
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required.")
    var k = min(Int(py=family_call(b.int, family_field(fields, "n_neighbors"))), n - 1)
    var pair = _knn(binding, x, x, k, True, context)
    var dist = pair[0]
    var idx = pair[1]
    var lrd = _out(Python.tuple(n), "<f4", context)
    _op(binding, "lof_lrd", Python.list(classical_address(dist), classical_address(idx), classical_address(dist),
                                        classical_address(lrd)),
        Python.list(_i(k), _i(n), _i(n)), Python.list(), context)
    var score = _out(Python.tuple(n), "<f4", context)
    _op(binding, "lof_score", Python.list(classical_address(idx), classical_address(lrd), classical_address(lrd),
                                          classical_address(score)),
        Python.list(_i(k), _i(n), _i(n)), Python.list(), context)
    _ = idx  # keep the Python object alive through the native calls above
    _ = dist  # keep the Python object alive through the native calls above
    _ = lrd  # keep the Python object alive through the native calls above
    _ = score  # keep the Python object alive through the native calls above
    py_setitem(fields, "_fit_X", x)
    py_setitem(fields, "_fit_dist", dist)
    py_setitem(fields, "_lrd", lrd)
    py_setitem(fields, "n_neighbors_", _i(k))
    py_setitem(fields, "n_features_in_", _i(d))
    py_setitem(fields, "n_samples_fit_", _i(n))
    py_setitem(fields, "negative_outlier_factor_", score)
    var contamination = family_field(fields, "contamination")
    if _eq(contamination, PythonObject("auto")):
        py_setitem(fields, "offset_", PythonObject(-1.5))
    else:
        var q = pinned_mul_f64(100.0, Float64(py=family_call(b.float, contamination)))
        var value = _lof_percentile(binding, score, q, context)
        py_setitem(fields, "offset_", PythonObject(Float64(Float32(value))))


def _lof_fit_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LocalOutlierFactor.fit_predict(X)`."""
    var fields = py_fields(model)
    if _truth(family_field(fields, "novelty")):
        raise Error("AttributeError: fit_predict is not available when novelty=True; use predict on new data")
    _lof_fit(model, args, context)
    var off = family_field(fields, "offset_")
    var score = family_field(fields, "negative_outlier_factor_")
    var size = _size(score)
    var lab = _out(Python.tuple(size), "<i4", context)
    if size != 0:
        _op(args[0], "p2m_sign_label", Python.list(classical_address(score), classical_address(lab)),
            Python.list(_i(size), _i(0)), Python.list(family_call(_b().float, off)), context)
        _ = score  # keep the Python object alive through the native call above
    return _astype_i64(lab, context)


def _lof_novelty(fields: PythonObject, what: String) raises:
    """`LocalOutlierFactor._novelty(what)`."""
    if not _truth(family_field(fields, "novelty")):
        raise Error("AttributeError: " + what + " is not available when novelty=False; use fit_predict")


def _lof_score_samples(binding: PythonObject, fields: PythonObject, source: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`LocalOutlierFactor.score_samples(X)` after its novelty check."""
    var b = _b()
    var q = _f32(source, context)
    var k = Int(py=family_call(b.int, family_field(fields, "n_neighbors_")))
    var nf = Int(py=family_call(b.int, family_field(fields, "n_samples_fit_")))
    var fit_x = family_field(fields, "_fit_X")
    var pair = _knn(binding, q, fit_x, k, False, context)
    var dist = pair[0]
    var idx = pair[1]
    var nq = _rows(q)
    var fit_dist = family_field(fields, "_fit_dist")
    var fit_lrd = family_field(fields, "_lrd")
    var lrd = _out(Python.tuple(nq), "<f4", context)
    _op(binding, "lof_lrd", Python.list(classical_address(dist), classical_address(idx), classical_address(fit_dist),
                                        classical_address(lrd)),
        Python.list(_i(k), _i(nq), _i(nf)), Python.list(), context)
    var score = _out(Python.tuple(nq), "<f4", context)
    _op(binding, "lof_score", Python.list(classical_address(idx), classical_address(fit_lrd), classical_address(lrd),
                                          classical_address(score)),
        Python.list(_i(k), _i(nq), _i(nf)), Python.list(), context)
    _ = dist  # keep the Python object alive through the native calls above
    _ = idx  # keep the Python object alive through the native calls above
    _ = fit_dist  # keep the Python object alive through the native calls above
    _ = fit_lrd  # keep the Python object alive through the native calls above
    _ = lrd  # keep the Python object alive through the native calls above
    _ = score  # keep the Python object alive through the native calls above
    return score


def _lof_decision(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LocalOutlierFactor.decision_function(X)`: `score_samples(X) - offset_`
    through the identity unary."""
    var fields = py_fields(model)
    _lof_novelty(fields, "decision_function")
    var score = _lof_score_samples(args[0], fields, args[1], context)
    var op = native_loaded_module("operator")
    var shift = Float64(py=family_call(_b().float, family_call(op.neg, family_field(fields, "offset_"))))
    return _unary(args[0], score, _U_IDENTITY, 1.0, shift, context)


def _lof_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LocalOutlierFactor.predict(X)`."""
    var fields = py_fields(model)
    _lof_novelty(fields, "predict")
    var dec = _lof_decision(model, args, context)
    var size = _size(dec)
    var lab = _out(Python.tuple(size), "<i4", context)
    if size != 0:
        _op(args[0], "p2m_sign_label", Python.list(classical_address(dec), classical_address(lab)),
            Python.list(_i(size), _i(1)), Python.list(PythonObject(0.0)), context)
        _ = dec  # keep the Python object alive through the native call above
    return _astype_i64(lab, context)


# ------------------------------------------------------------ NearestCentroid


def _class_array(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_class_array(classes, codes)`: int or bool classes an int64 Array,
    real classes a float64 Array, other labels a list (`decode_labels`)."""
    var b = _b()
    var k = len(classes)
    var all_int = True
    for i in range(k):  # driver-loop: over the k class labels (kind test)
        if not family_isinstance(classes[i], Python.tuple(b.bool, b.int)):
            all_int = False
            break
    if all_int:
        var ints = Python.list()
        for i in range(k):  # driver-loop: over the k class values
            _ = py_call(ints.append, family_call(b.int, classes[i]))
        return classical_decode_labels(ints, codes, context)
    var all_real = True
    for i in range(k):  # driver-loop: over the k class labels (kind test)
        if not family_isinstance(classes[i], Python.tuple(b.int, b.float)):
            all_real = False
            break
    if all_real:
        var reals = Python.list()
        for i in range(k):  # driver-loop: over the k class labels as floats
            _ = py_call(reals.append, family_call(b.float, classes[i]))
        return classical_decode_labels(reals, codes, context)
    return classical_decode_labels(classes, codes, context)


def _nc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`NearestCentroid.fit(X, y)`."""
    var b = _b()
    var fields = py_fields(model)
    var binding = args[0]
    var metric = family_field(fields, "metric")
    if not py_contains(Python.tuple(PythonObject("euclidean"), PythonObject("manhattan")), metric):
        raise Error("NearestCentroid: metric must be 'euclidean' or 'manhattan'")
    var x = _f32(args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    # the native encoder's int32 codes, the class counts from the device
    var encoded = classical_encode_labels(args[2], context)
    var classes = encoded.classes
    var lab = encoded.codes
    if _size(lab) != n:
        raise Error("X and y have different numbers of rows")
    var C = len(classes)
    if C < 2:
        raise Error("The number of classes has to be greater than one; got " + String(C) + " class")
    var nk = _out(Python.tuple(C), "<f4", context)
    var info = _out(Python.tuple(1), "<i4", context)
    _op(binding, "p2m_class_counts", Python.list(classical_address(lab), classical_address(nk), classical_address(info)),
        Python.list(_i(n), _i(C)), Python.list(), context)
    _ = info  # keep the Python object alive through the native call above
    var nkp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(nk)))
    var counts = List[Int]()
    for c in range(C):  # driver-loop: over the k class counts (priors, offsets)
        counts.append(Int(nkp[c]))
    _ = nk  # keep the Python object alive through the reads above
    var priors = family_field(fields, "priors")
    var prior = labels_vector(C, "<f8", context)
    var pp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(prior)))
    if py_type(priors) is b.str and String(priors) == "empirical":
        # count / n per class in the base binding (`class_ratio_f64`)
        var ca = labels_vector(C, "<i8", context)
        var cap = FAMILY_I64P(unsafe_from_address=Int(py=classical_address(ca)))
        for c in range(C):  # driver-loop: over the k class counts
            cap[c] = Int64(counts[c])
        _ = family_call(labels_helper(context, "class_ratio_f64"), classical_address(ca), _i(C), _i(n), _i(1),
                        classical_address(prior))
        _ = ca  # keep the Python object alive through the native call above
    elif py_type(priors) is b.str and String(priors) == "uniform":
        for c in range(C):  # driver-loop: over the k uniform class priors
            pp[c] = 1.0 / Float64(C)
    else:
        # Parse admitted primitive storage once; ndarray.tolist (including
        # object-element conversions) is not part of native model execution.
        var given = labels_view(priors, "priors", context)
        labels_check_rank(given, 1, "priors")
        if given.size() != C:
            raise Error("priors must have one entry per class")
        var packed = labels_vector(C, "<f8", context)
        var packed_address = Int(py=classical_address(packed))
        labels_cast(given.addr, given.typestr, packed_address, "<f8", C, context)
        var source = FAMILY_F64P(unsafe_from_address=packed_address)
        var values = List[Float64]()
        for i in range(C):  # driver-loop: over admitted native class priors
            values.append(source[i])
        _ = given
        _ = packed
        for i in range(C):  # driver-loop: over the user priors argument checks
            if values[i] < 0:
                raise Error("priors must be non-negative")
        # `_portable_math.fsum`: the core `fsum_f64` (Shewchuk's partials)
        var src = labels_vector(C, "<f8", context)
        var sp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(src)))
        for i in range(C):  # driver-loop: over the user priors argument (k sized)
            sp[i] = values[i]
        var total = labels_vector(1, "<f8", context)
        var status = Int(py=family_call(labels_helper(context, "fsum_f64"), classical_address(src), _i(C),
                                        classical_address(total)))
        _ = src  # keep the Python object alive through the native call above
        var tot = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(total)))[0]
        _ = total  # keep the Python object alive through the read above
        if status != 0:
            if tot != 0.0:
                raise Error("OverflowError: math range error")
            raise Error("-inf + inf in fsum")
        var rescale = not _isclose(tot, 1.0, 1e-5, 1e-8)
        if rescale and tot == 0.0:
            raise Error("ZeroDivisionError: float division by zero")
        for i in range(C):  # driver-loop: over the user priors argument (rescaled to sum one)
            pp[i] = values[i] / tot if rescale else values[i]
    _ = prior  # keep the Python object alive through the writes above
    py_setitem(fields, "class_prior_", prior)
    var split = getenv("MOJOLEARN_NC_SPLIT_OPS", "") == "1"
    var euclidean = _eq(metric, PythonObject("euclidean"))
    var cent = _out(Python.tuple(C, d), "<f4", context)
    if euclidean:
        if split:
            _op(binding, "group_mean", Python.list(classical_address(x), classical_address(lab), classical_address(cent)),
                Python.list(_i(n), _i(d), _i(C)), Python.list(), context)
    else:
        # the per-class medians on the device: offsets are the class counts'
        # prefix (integer bookkeeping over the classes)
        var start = List[Int](length=C + 1, fill=0)
        for c in range(C):  # driver-loop: over the class offsets prefix (k classes)
            start[c + 1] = start[c] + counts[c]
        var steps = _pow2_steps(n)
        var starts = _i32_from_ints(start, context)
        _op(binding, "nc_median", Python.list(classical_address(x), classical_address(lab), classical_address(starts),
                                              classical_address(cent)),
            Python.list(_i(n), _i(d), _i(C), _i(steps[0]), _i(steps[1])), Python.list(), context)
        _ = starts  # keep the Python object alive through the native call above
    var stats = _out(Python.tuple(d), "<f4", context)
    var new_cent = _out(Python.tuple(C, d), "<f4", context)
    var devs = _out(Python.tuple(C, d), "<f4", context)
    var dsc = PythonObject(None)
    if euclidean and not split:
        # the class means, the within-class std and the dataset centroid in
        # one op (nc_stats_item)
        dsc = _out(Python.tuple(d), "<f4", context)
        _op(binding, "nc_stats", Python.list(classical_address(x), classical_address(lab), classical_address(nk),
                                             classical_address(cent), classical_address(stats), classical_address(dsc)),
            Python.list(_i(n), _i(d), _i(C)), Python.list(), context)
    else:
        _op(binding, "nc_std", Python.list(classical_address(x), classical_address(lab), classical_address(cent),
                                           classical_address(stats)),
            Python.list(_i(n), _i(d), _i(C)), Python.list(), context)
    # the median of the d stds and the all-zero test on the device (nc_med_std)
    var dsteps = _pow2_steps(d)
    var ms = _out(Python.tuple(2), "<f4", context)
    _op(binding, "nc_med_std", Python.list(classical_address(stats), classical_address(ms)),
        Python.list(_i(d), _i(dsteps[0]), _i(dsteps[1])), Python.list(), context)
    var msp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(ms)))
    var med_std = Float64(msp[0])
    var all_zero = Float64(msp[1])
    _ = ms  # keep the Python object alive through the reads above
    if all_zero != 0.0:
        # `_ptp_zero`: every feature constant (`p2m_const_cols`)
        var flag = _out(Python.tuple(1), "<i4", context)
        _op(binding, "p2m_const_cols", Python.list(classical_address(x), classical_address(flag)),
            Python.list(_i(n), _i(d)), Python.list(), context)
        var constant = Int(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(flag)))[0]) == 0
        _ = flag  # keep the Python object alive through the read above
        if constant:
            raise Error("All features have zero variance. Division by zero.")
    var threshold = family_field(fields, "shrink_threshold")
    var shrink = Float64(py=family_call(b.float, threshold)) if _truth(threshold) else 0.0
    var shrinking = 1 if shrink != 0.0 else 0
    if dsc is not None:
        _op(binding, "nc_shrink_d", Python.list(classical_address(dsc), classical_address(cent), classical_address(nk),
                                                classical_address(stats), classical_address(new_cent),
                                                classical_address(devs)),
            Python.list(_i(n), _i(d), _i(C), _i(shrinking)), Python.list(PythonObject(med_std), PythonObject(shrink)),
            context)
    else:
        _op(binding, "nc_shrink", Python.list(classical_address(x), classical_address(cent), classical_address(nk),
                                              classical_address(stats), classical_address(new_cent),
                                              classical_address(devs)),
            Python.list(_i(n), _i(d), _i(C), _i(shrinking)), Python.list(PythonObject(med_std), PythonObject(shrink)),
            context)
    _ = x  # keep the Python object alive through the native calls above
    _ = lab  # keep the Python object alive through the native calls above
    _ = nk  # keep the Python object alive through the native calls above
    _ = cent  # keep the Python object alive through the native calls above
    _ = dsc  # keep the Python object alive through the native calls above
    py_setitem(fields, "centroids_", new_cent)
    py_setitem(fields, "deviations_", devs)
    py_setitem(fields, "within_class_std_dev_", stats)
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "_codes_classes", classes)
    py_setitem(fields, "n_features_in_", _i(d))


def _nc_uniform(fields: PythonObject, context: PythonObject) raises -> Bool:
    """`NearestCentroid._uniform`: every prior within isclose of 1/C."""
    var C = len(classifier_label_list(family_field(fields, "classes_"), context))
    var prior = family_field(fields, "class_prior_")
    var count = _size(prior)
    var pp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(prior)))
    var uniform = True
    for i in range(count):  # driver-loop: over the uniform prior check (k classes)
        if not _isclose(pp[i], 1.0 / Float64(C), 1e-5, 1e-8):
            uniform = False
            break
    _ = prior  # keep the Python object alive through the reads above
    return uniform


def _nc_decision_matrix(binding: PythonObject, fields: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`NearestCentroid._decision_matrix(X)`."""
    if _ne(family_field(fields, "metric"), PythonObject("euclidean")):
        raise Error("AttributeError: decision_function is available for metric='euclidean' only")
    var q = _f32(source, context)
    var C = len(classifier_label_list(family_field(fields, "classes_"), context))
    # `Array.from_list(class_prior_.tolist(), "<f4")`: each prior rounded once
    var given = family_field(fields, "class_prior_")
    var count = _size(given)
    var prior = labels_vector(count, "<f4", context)
    var gp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(given)))
    var fp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(prior)))
    for i in range(count):  # driver-loop: over the k class priors
        fp[i] = Float32(gp[i])
    _ = given  # keep the Python object alive through the reads above
    var cent = family_field(fields, "centroids_")
    var std = family_field(fields, "within_class_std_dev_")
    var nq = _rows(q)
    var out = _out(Python.tuple(nq, C), "<f4", context)
    _op(binding, "nc_decision", Python.list(classical_address(q), classical_address(cent), classical_address(std),
                                            classical_address(prior), classical_address(out)),
        Python.list(_i(nq), _i(_cols(q)), _i(C)), Python.list(), context)
    _ = q  # keep the Python object alive through the native call above
    _ = cent  # keep the Python object alive through the native call above
    _ = std  # keep the Python object alive through the native call above
    _ = prior  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _nc_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NearestCentroid.predict(X)`."""
    var fields = py_fields(model)
    var binding = args[0]
    var q = _f32(args[1], context)
    var classes = classifier_label_list(family_field(fields, "classes_"), context)
    if _nc_uniform(fields, context):
        var cent = family_field(fields, "centroids_")
        var dmat: PythonObject
        if _eq(family_field(fields, "metric"), PythonObject("euclidean")):
            dmat = _sqdist(binding, q, cent, context)
        else:
            dmat = _l1dist(binding, q, cent, context)
        var pair = _knn_select(binding, dmat, 1, False, context)
        var idx = pair[1]
        var codes = labels_empty(Python.tuple(_rows(idx)), "<i4", context)
        memcpy(dest=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(codes))),
               src=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(idx))), count=_rows(idx))
        _ = idx  # keep the Python object alive through the copy above
        return _class_array(classes, codes, context)
    # `_argmax_labels`: the first maximum of each row on the device
    var mat = _nc_decision_matrix(binding, fields, q, context)
    var n = _rows(mat)
    var C = _cols(mat)
    var codes = _out(Python.tuple(n), "<i4", context)
    if n > 0:
        _op(binding, "row_argmax", Python.list(classical_address(mat), classical_address(codes)),
            Python.list(_i(n), _i(C)), Python.list(), context)
        _ = mat  # keep the Python object alive through the native call above
    return _class_array(classes, codes, context)


def _nc_decision(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NearestCentroid.decision_function(X)`: the score matrix, or for two
    classes `scores[:, 1] - scores[:, 0]` as one product with (-1, 1)."""
    var binding = args[0]
    var dec = _nc_decision_matrix(binding, py_fields(model), args[1], context)
    if _cols(dec) != 2:
        return dec
    var signs = labels_vector(2, "<f4", context)
    var sp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(signs)))
    sp[0] = -1.0
    sp[1] = 1.0
    var n = _rows(dec)
    # `_matmul(dec, signs.reshape((2, 1)))` then `.reshape((n,))`
    var diff = _out(Python.tuple(n), "<f4", context)
    _op(binding, "matmul", Python.list(classical_address(dec), classical_address(signs), classical_address(diff)),
        Python.list(_i(n), _i(2), _i(1)), Python.list(), context)
    _ = dec  # keep the Python object alive through the native call above
    _ = signs  # keep the Python object alive through the native call above
    return diff


def _nc_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NearestCentroid.predict_proba(X)`: the softmax of the decision
    matrix, float64 where the binary has `xn_softmax64`."""
    var b = _b()
    var binding = args[0]
    var dec = _nc_decision_matrix(binding, py_fields(model), args[1], context)
    var shape = py_call(b.tuple, dec.shape)
    # A shared native descriptor does not imply optional kernel availability.
    # Query its admitted compiled owner just as for a direct module binding;
    # identical bindings correctly omit the optional float64 softmax export.
    var wide = neural_descriptor_entry(xn_module(binding), "xn_softmax64", False) is not None
    var out = _out(shape, "<f8" if wide else "<f4", context)
    _op(binding, "softmax64" if wide else "softmax", Python.list(classical_address(dec), classical_address(out)),
        Python.list(_i(_rows(dec)), _i(_cols(dec))), Python.list(), context)
    _ = dec  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _nc_predict_log_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # The public result is the existing float32 log-softmax, including the
    # metric/fitted checks and native decision matrix preparation.
    var binding = args[0]
    var dec = _nc_decision_matrix(binding, py_fields(model), args[1], context)
    var rows = _rows(dec)
    var cols = _cols(dec)
    var out = _out(Python.tuple(rows, cols), "<f4", context)
    _op(binding, "log_softmax", Python.list(classical_address(dec), classical_address(out)),
        Python.list(_i(rows), _i(cols)), Python.list(), context)
    _ = dec
    return out


def _nc_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NearestCentroid.score(X, y, sample_weight)`: accuracy_fraction of
    `predict(X)` (the process-default metrics mode)."""
    var prediction = _nc_predict(model, Python.tuple(args[0], args[1]), context)
    return PythonObject(classical_accuracy(args[2], prediction, args[3], PythonObject(None), context))


# ------------------------------------------------------------- KNNImputer


def _imp_check(fields: PythonObject, context: PythonObject) raises:
    """`KNNImputer._check`."""
    var b = _b()
    var mv = family_field(fields, "missing_values")
    if (mv is None or family_isinstance(mv, Python.tuple(b.str, b.bool))
            or not family_isinstance(mv, context["labels_real_type"])):
        raise Error("NotImplementedError: KNNImputer: missing_values must be a number or NaN (float input)")
    if _ne(family_field(fields, "metric"), PythonObject("nan_euclidean")):
        raise Error("NotImplementedError: KNNImputer: metric must be 'nan_euclidean'")
    if not py_contains(Python.tuple(PythonObject("uniform"), PythonObject("distance")), family_field(fields, "weights")):
        raise Error("NotImplementedError: KNNImputer: weights must be 'uniform' or 'distance'")


def _imp_masked(binding: PythonObject, fields: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`KNNImputer._masked(X)`: a number's cells become NaN."""
    var x = _f32(source, context)
    var mv = family_field(fields, "missing_values")
    if _eq(mv, mv) and _size(x) != 0:
        var out = _out(py_call(_b().tuple, x.shape), "<f4", context)
        _op(binding, "p2m_mask_value", Python.list(classical_address(x), classical_address(out)),
            Python.list(_i(_size(x))), Python.list(PythonObject(_f32_scalar(mv))), context)
        _ = x  # keep the Python object alive through the native call above
        return out
    return x


def _imp_nan_cells(binding: PythonObject, x: PythonObject, colmiss_only: Int,
                   context: PythonObject) raises -> Tuple[PythonObject, PythonObject, Int]:
    """`KNNImputer._nan_cells(X, colmiss_only)`: (cells, colmiss, count),
    the per-column NaN counts as the int32 Array `xn_nan_cells` filled."""
    var b = _b()
    var n = _rows(x)
    var d = _cols(x)
    var lean = False
    if colmiss_only != 0:
        var probe = py_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_nc_fit_lean"), PythonObject(None))
        if probe is not None:
            lean = Int(py=family_call(b.int, family_call(probe))) != 0
    var cells = _out(Python.tuple(1 if lean else max(n * d, 1)), "<i4", context)
    var colmiss = _out(Python.tuple(max(d, 1)), "<i4", context)
    var info = _out(Python.tuple(1), "<i4", context)
    _op(binding, "nan_cells", Python.list(classical_address(x), classical_address(cells), classical_address(colmiss),
                                          classical_address(info)),
        Python.list(_i(n), _i(d), _i(colmiss_only)), Python.list(), context)
    _ = x  # keep the Python object alive through the native call above
    var count = Int(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(info)))[0])
    _ = info  # keep the Python object alive through the read above
    return (cells, colmiss, count)


def _imp_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KNNImputer.fit(X)`."""
    var fields = py_fields(model)
    _imp_check(fields, context)
    var binding = args[0]
    var x = _imp_masked(binding, fields, args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    var found = _imp_nan_cells(binding, x, 1, context)
    var colmiss = found[1]
    var cm = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(colmiss)))
    var valid = Python.list()
    var miss = Python.list()
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks)
        _ = py_call(valid.append, PythonObject(Int(cm[f]) < n))
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks)
        if Int(cm[f]) > 0:
            _ = py_call(miss.append, _i(f))
    _ = colmiss  # keep the Python object alive through the reads above
    py_setitem(fields, "_valid", valid)
    py_setitem(fields, "_miss_cols", miss)
    py_setitem(fields, "_fit_X", x)
    py_setitem(fields, "n_features_in_", _i(d))


def _imp_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KNNImputer.transform(X)`."""
    var b = _b()
    var fields = py_fields(model)
    var binding = args[0]
    var x = _imp_masked(binding, fields, args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    if d != Int(py=family_field(fields, "n_features_in_")):
        raise Error("X has a different number of features than during fit")
    var fit_x = family_field(fields, "_fit_X")
    var m = _rows(fit_x)
    var out = _out(Python.tuple(n, d), "<f4", context)
    var k = Int(py=family_call(b.int, family_field(fields, "n_neighbors")))
    if k < 1:
        raise Error("n_neighbors must be >= 1")
    var distance = 1 if _eq(family_field(fields, "weights"), PythonObject("distance")) else 0
    if _flag(context, "uncompact_impute"):
        _op(binding, "knn_impute", Python.list(classical_address(x), classical_address(fit_x), classical_address(out)),
            Python.list(_i(n), _i(m), _i(d), _i(k), _i(distance)), Python.list(), context)
    else:
        # one GPU thread per MISSING cell; a present cell keeps x
        var found = _imp_nan_cells(binding, x, 0, context)
        var cells = found[0]
        var nc = found[2]
        out = _copy_f32(x, context)
        if nc != 0:
            _op(binding, "knn_impute_cells" if _flag(context, "old_items") else "knn_impute_tiled",
                Python.list(classical_address(cells), classical_address(x), classical_address(fit_x),
                            classical_address(out)),
                Python.list(_i(n), _i(m), _i(d), _i(k), _i(distance), _i(nc)), Python.list(), context)
        _ = cells  # keep the Python object alive through the native call above
    _ = fit_x  # keep the Python object alive through the native calls above
    var valid = family_field(fields, "_valid")
    var keep = List[Int]()
    var all_valid = True
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks)
        if _truth(valid[f]):
            keep.append(f)
        else:
            all_valid = False
    if _truth(family_field(fields, "keep_empty_features")):
        if n != 0 and not all_valid:
            var flags = List[Int]()
            for f in range(d):  # driver-loop: over the d fit flags
                flags.append(0 if _truth(valid[f]) else 1)
            var flag_array = _i32_from_ints(flags, context)
            var z = _out(Python.tuple(n, d), "<f4", context)
            _op(binding, "p2m_zero_cols", Python.list(classical_address(out), classical_address(flag_array),
                                                      classical_address(z)),
                Python.list(_i(n), _i(d)), Python.list(), context)
            _ = out  # keep the Python object alive through the native call above
            _ = flag_array  # keep the Python object alive through the native call above
            out = z
    elif len(keep) != d:
        # `_take_cols(out, keep)`
        var cols = _i32_from_ints(keep, context)
        var rows = _rows(out)
        var taken = _out(Python.tuple(rows, len(keep)), "<f4", context)
        _op(binding, "take_cols", Python.list(classical_address(out), classical_address(cols), classical_address(taken)),
            Python.list(_i(rows), _i(_cols(out)), _i(len(keep))), Python.list(), context)
        _ = out  # keep the Python object alive through the native call above
        _ = cols  # keep the Python object alive through the native call above
        out = taken
    var miss = family_field(fields, "_miss_cols")
    if _truth(family_field(fields, "add_indicator")) and _truth(miss) and n != 0:
        var c = _cols(out)
        var q = len(miss)
        var miss_cols = List[Int]()
        for i in range(q):  # driver-loop: over the indicator columns (features missing at fit)
            miss_cols.append(Int(py=miss[i]))
        var miss_array = _i32_from_ints(miss_cols, context)
        var res = _out(Python.tuple(n, c + q), "<f4", context)
        var second = out if c != 0 else x
        _op(binding, "p2m_nan_indicator", Python.list(classical_address(x), classical_address(second),
                                                      classical_address(miss_array), classical_address(res)),
            Python.list(_i(n), _i(d), _i(c), _i(q)), Python.list(), context)
        _ = second  # keep the Python object alive through the native call above
        _ = miss_array  # keep the Python object alive through the native call above
        out = res
    _ = x  # keep the Python object alive through the native calls above
    return out


# ------------------------------------------------------------------ door


def _neighbors_local_operation(model: PythonObject, operation: String, var args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    _admit_local_state(model, context)
    for i in range(1, len(args)):  # small-loop(public data arguments, before native preparation)
        if host_sparse_format(args[i]).byte_length() == 0:
            classical_admit_value(args[i], context)
    args = xn_admit_arguments(args, context)
    if operation == "family_xn_loc_lof_fit":
        _lof_fit(model, args, context)
        return model
    if operation == "family_xn_loc_lof_fit_predict":
        return _lof_fit_predict(model, args, context)
    if operation == "family_xn_loc_lof_score_samples":
        var fields = py_fields(model)
        _lof_novelty(fields, "score_samples")
        return _lof_score_samples(args[0], fields, args[1], context)
    if operation == "family_xn_loc_lof_decision_function":
        return _lof_decision(model, args, context)
    if operation == "family_xn_loc_lof_predict":
        return _lof_predict(model, args, context)
    if operation == "family_xn_loc_nc_fit":
        _nc_fit(model, args, context)
        return model
    if operation == "family_xn_loc_nc_predict":
        return _nc_predict(model, args, context)
    if operation == "family_xn_loc_nc_decision_function":
        return _nc_decision(model, args, context)
    if operation == "family_xn_loc_nc_predict_proba":
        return _nc_predict_proba(model, args, context)
    if operation == "family_xn_loc_nc_predict_log_proba":
        return _nc_predict_log_proba(model, args, context)
    if operation == "family_xn_loc_nc_score":
        return _nc_score(model, args, context)
    if operation == "family_xn_loc_imp_fit":
        _imp_fit(model, args, context)
        return model
    if operation == "family_xn_loc_imp_transform":
        return _imp_transform(model, args, context)
    if operation == "family_xn_loc_imp_fit_transform":
        _imp_fit(model, args, context)
        return _imp_transform(model, args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_local_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    var call = neighbors_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_local_operation(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return neighbors_public_finish(model, operation, result, call, context)
