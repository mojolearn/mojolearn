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
from providers.python.x_neighbors_admission_pyapi import xn_capabilities
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.math import floor
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family import admit_family
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop, prepared_key_add, prepared_key_same
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.label_admission_native import labels_primitive_kind
from bindings.host_sparse_native import host_sparse_format
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_call, family_entry, family_field,
    family_isinstance,
)
from bindings.classical_labels import (
    labels_cast, classical_encode_labels,
    classical_decode_labels, classical_accuracy,
)
from bindings.labels_descriptor_boundary import labels_class_ratio_f64
from bindings.nearest_centroid_provider import NcConnection
from core.nearest_centroid_plan import (
    NcFitPlan, NcQueryPlan, nc_uniform, nc_normalize_priors, nc_narrow_priors, nc_binary_decision,
)
from core.admitted_weights import admitted_fill
from bindings.admitted_input import admitted_array, admitted_vector, admitted_storage_native, admit_c_view, admitted_dtype
from bindings.admitted_input import admitted_finite_refuse
from checks.numerics import pinned_mul_f64

# x_neighbors/items.mojo's unary codes (`_U_SQRT`, `_U_IDENTITY`).
comptime _U_SQRT = 2
comptime _U_IDENTITY = 6


# ------------------------------------------------------------------ glue


comptime _LOCAL_STATE_SLOT = "neighbors_local_state_v1"


def _admit_local_state(model: PythonObject, context: PythonObject, fitting: Bool = False) raises -> PythonObject:
    # Mutable settings are re-admitted before comparison, coercion or truth
    # slots. Input metadata is captured by the consuming native buffer helper.
    # The admitted state's identity stamp (every slot object and, for exact
    # list/tuple/dict containers, their items) lives in the native prepared
    # registry, outside `__dict__`, weakly keyed by the model: while it is
    # unchanged a query skips the re-admission, as knn's resident stamp does.
    # `classes_` is re-admitted on every call (an object-dtype label array can
    # change in place without an identity change), like knn's class lists.
    # Perf (perf2-neighbors): returns that admitted label list (None when the
    # model has no `classes_` or at fit), so NearestCentroid predict/score
    # reuse it instead of building it a second time per call.
    var labels = PythonObject(None)
    classical_check_layout(model, context)
    var fields = py_fields(model)
    var scalars = Python.tuple("metric", "p", "contamination", "n_neighbors", "novelty", "shrink_threshold",
        "missing_values", "weights", "add_indicator", "keep_empty_features", "n_features_in_",
        "n_neighbors_", "n_samples_fit_", "offset_")
    var values = Python.tuple("metric_params", "priors", "classes_", "_fit_X", "_fit_dist", "_lrd",
        "centroids_", "class_prior_", "within_class_std_dev_", "_valid", "_miss_cols",
        "negative_outlier_factor_")
    var mods = NativeModules()
    var stamp = Python.list()
    for i in range(len(scalars)):  # small-loop(fixed scalar parameter and fitted-state schema)
        prepared_key_add(mods, stamp, py_get(fields, scalars[i]))
    for i in range(len(values)):  # small-loop(fixed native container and buffer owners)
        if String(values[i]) != "classes_":
            prepared_key_add(mods, stamp, py_get(fields, values[i]))
    var record = prepared_get(mods, _LOCAL_STATE_SLOT, model)
    if record is not None and prepared_key_same(record, stamp):
        var classes = py_get(fields, "classes_")
        if classes is not None:
            labels = classifier_label_list(classes, context)
            classical_admit_value(labels, context)
    else:
        for i in range(len(scalars)):  # small-loop(fixed scalar parameter and fitted-state schema)
            var value = py_get(fields, scalars[i])
            if value is not None:
                host_inference_scalar(value, context, True)
        for i in range(len(values)):  # small-loop(fixed native container and buffer owners)
            var value = py_get(fields, values[i])
            if value is not None:
                if String(values[i]) == "classes_":
                    labels = classifier_label_list(value, context)
                    classical_admit_value(labels, context)
                else:
                    classical_admit_value(value, context)
    if fitting:
        # Fit replaces the fitted slots: never retain the old arrays.
        prepared_drop(mods, _LOCAL_STATE_SLOT, model)
        return PythonObject(None)
    if record is None or not prepared_key_same(record, stamp):
        for i in range(len(stamp)):  # driver-loop: over the stamp objects, refusing a self-reference
            if stamp[i] is model:
                return labels
        prepared_put(mods, _LOCAL_STATE_SLOT, model, stamp)
    return labels


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`, as an `if` or `and`/`or` took it."""
    # Perf (perf2-neighbors): an exact bool or None has no truth protocol to
    # run; read it by identity instead of a Python `bool()` call.
    if value is None:
        return False
    if py_type(value) is _b().bool:
        return value is PythonObject(True)
    return Bool(py=family_call(_b().bool, value))


def _int_word(value: PythonObject) raises -> Int:
    """`int(value)`; an exact int is read without the Python call
    (perf2-neighbors)."""
    if py_type(value) is _b().int:
        return Int(py=value)
    return Int(py=family_call(_b().int, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    # Perf (perf2-neighbors): two exact strs compare natively (str equality
    # runs no user protocol); anything else keeps `operator.eq`.
    var bi = _b()
    if py_type(a) is bi.str and py_type(b) is bi.str:
        return String(py=a) == String(py=b)
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _ne(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a != b)`."""
    var bi = _b()
    if py_type(a) is bi.str and py_type(b) is bi.str:
        return String(py=a) != String(py=b)
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
    return admit_family(source, 2, "X", context, dense=True).array


def _out(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`empty(shape, typestr)` / `_empty_out(shape)`: a new zero-filled Array."""
    return admitted_array(shape, typestr, context)


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


def _astype_i64(lab: PythonObject, context: PythonObject) raises -> PythonObject:
    """`lab.astype("<i8")` of an int32 vector Array."""
    var n = _size(lab)
    var out = admitted_vector(n, "<i8", context)
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


def _sqdist(binding: PythonObject, a: PythonObject, b: PythonObject, context: PythonObject,
            status: Int = 0) raises -> PythonObject:
    """`_XNeighbors._sqdist(A, B)`; `status`: A's finite status word (its
    status slot; 0 not asked)."""
    var n = _rows(a)
    var d = _cols(a)
    var m = _rows(b)
    var out = _out(Python.tuple(n, m), "<f4", context)
    var ints = Python.list(_i(n), _i(m), _i(d))
    if status != 0:
        _ = py_call(ints.append, _i(status))
    _op(binding, "sqdist", Python.list(classical_address(a), classical_address(b), classical_address(out)),
        ints, Python.list(), context)
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
         context: PythonObject, status: Int = 0) raises -> Tuple[PythonObject, PythonObject]:
    """`_XNeighbors._knn(Q, R, k, exclude_self)`: `_knn_sq` (the fused
    `knn_sq_tiled`, `knn_sq` under MOJOLEARN_XN_OLD_ITEMS, sqdist +
    knn_select under MOJOLEARN_XN_UNFUSED_KNN), then the square root.
    `status` (lane a8b-device-x-neighbors-rest; 0: not asked): the host
    address of the int32 word that receives Q's finite status, from the
    first op that uploads Q (its status slot)."""
    var sq: PythonObject
    var idx: PythonObject
    if _flag(context, "unfused_knn"):
        var pair = _knn_select(binding, _sqdist(binding, q, r, context, status), k, exclude_self, context)
        sq = pair[0]
        idx = pair[1]
    else:
        var n = _rows(q)
        var d = _cols(q)
        var m = _rows(r)
        sq = _out(Python.tuple(n, k), "<f4", context)
        idx = _out(Python.tuple(n, k), "<i4", context)
        var ints = Python.list(_i(n), _i(m), _i(d), _i(k), _i(1 if exclude_self else 0))
        if status != 0:
            _ = py_call(ints.append, _i(status))
        _op(binding, "knn_sq" if _flag(context, "old_items") else "knn_sq_tiled",
            Python.list(classical_address(q), classical_address(r), classical_address(sq), classical_address(idx)),
            ints, Python.list(), context)
        _ = q  # keep the Python object alive through the native call above
        _ = r  # keep the Python object alive through the native call above
    var root = _unary(binding, sq, _U_SQRT, 1.0, 0.0, context)
    _ = sq  # keep the Python object alive through the native call above
    return (root, idx)


def _i32_from_ints(values: List[Int], context: PythonObject) raises -> PythonObject:
    """`_i32(list)`: an int32 vector Array of a short int list (class
    offsets, kept columns, flags)."""
    var out = admitted_vector(len(values), "<i4", context)
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
    # Lane a8b-device-x-neighbors-rest: X's finite test rides behind the
    # k-NN op's upload (the public admission deferred it; a cast X was
    # refused by its cast).
    var status = List[Int32](length=1, fill=Int32(0))
    var pair = _knn(binding, x, x, k, True, context, Int(status.unsafe_ptr()))
    admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
    _ = status^
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
                       context: PythonObject, name: String) raises -> PythonObject:
    """`LocalOutlierFactor.score_samples(X)` after its novelty check. The
    queries' finite test rides behind the k-NN op's upload; `name` words
    its refusal."""
    var q = _f32(source, context)
    var k = _int_word(family_field(fields, "n_neighbors_"))
    var nf = _int_word(family_field(fields, "n_samples_fit_"))
    var fit_x = family_field(fields, "_fit_X")
    var status = List[Int32](length=1, fill=Int32(0))
    var pair = _knn(binding, q, fit_x, k, False, context, Int(status.unsafe_ptr()))
    admitted_finite_refuse(Int(status[0]), name)
    _ = status^
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
    var score = _lof_score_samples(args[0], fields, args[1], context, String(py_type(model).__name__))
    var offset = family_field(fields, "offset_")
    var shift: Float64
    if py_type(offset) is _b().float:
        # Exact float: float(-x) is -x exactly (perf2-neighbors).
        shift = -Float64(py=offset)
    else:
        var op = native_loaded_module("operator")
        shift = Float64(py=family_call(_b().float, family_call(op.neg, offset)))
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
    # Perf (perf2-neighbors): an exact int/bool/float label answers the kind
    # tests by its type, and `int()`/`float()` of an exact int/float is the
    # object itself, so only other labels pay the Python calls.
    var b = _b()
    var k = len(classes)
    var all_int = True
    for i in range(k):  # driver-loop: over the k class labels (kind test)
        var cls = py_type(classes[i])
        if cls is not b.int and cls is not b.bool and not family_isinstance(classes[i], Python.tuple(b.bool, b.int)):
            all_int = False
            break
    if all_int:
        var ints = Python.list()
        for i in range(k):  # driver-loop: over the k class values
            var label = classes[i]
            _ = py_call(ints.append, label if py_type(label) is b.int else family_call(b.int, label))
        return classical_decode_labels(ints, codes, context)
    var all_real = True
    for i in range(k):  # driver-loop: over the k class labels (kind test)
        var cls = py_type(classes[i])
        if cls is not b.int and cls is not b.float and not family_isinstance(classes[i], Python.tuple(b.int, b.float)):
            all_real = False
            break
    if all_real:
        var reals = Python.list()
        for i in range(k):  # driver-loop: over the k class labels as floats
            var label = classes[i]
            _ = py_call(reals.append, label if py_type(label) is b.float else family_call(b.float, label))
        return classical_decode_labels(reals, codes, context)
    return classical_decode_labels(classes, codes, context)


def _nc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """NearestCentroid admission/publication around the native fit plan."""
    var b = _b()
    var fields = py_fields(model)
    var metric = family_field(fields, "metric")
    if not py_contains(Python.tuple(PythonObject("euclidean"), PythonObject("manhattan")), metric):
        raise Error("NearestCentroid: metric must be 'euclidean' or 'manhattan'")
    var x = _f32(args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    var encoded = classical_encode_labels(args[2], context)
    var classes = encoded.classes
    var lab = encoded.codes
    if _size(lab) != n:
        raise Error("X and y have different numbers of rows")
    var C = len(classes)
    if C < 2:
        raise Error("The number of classes has to be greater than one; got " + String(C) + " class")
    var connection = NcConnection(args[0])
    # This existing switch selects the same measured fused/split kernels.
    var plan = NcFitPlan(n, d, C, _eq(metric, PythonObject("euclidean")),
                        getenv("MOJOLEARN_NC_SPLIT_OPS", "") == "1")
    var label_address = Int(py=classical_address(lab))
    with GILReleased(Python()):
        plan.count(connection.driver, label_address)
    var priors = family_field(fields, "priors")
    var prior = admitted_storage_native([C], "<f8", context, as_array=True)
    var prior_address = Int(py=classical_address(prior))
    if py_type(priors) is b.str and String(priors) == "empirical":
        # Empirical priors retain mode 1: Float64(count) / Float64(n).
        _ = labels_class_ratio_f64(context, Int(plan.wide_counts.unsafe_ptr()), C, n, 1, prior_address)
    elif py_type(priors) is b.str and String(priors) == "uniform":
        with GILReleased(Python()):
            admitted_fill[DType.float64](FAMILY_F64P(unsafe_from_address=prior_address), C, 1.0 / Float64(C))
    else:
        var given = admit_c_view(priors, "priors", context)
        if Int(py=given.ndim) != 1:
            raise Error("mojolearn: priors must be 1-D, got " + String(Int(py=given.ndim)) + "-D")
        if Int(py=given.size) != C:
            raise Error("priors must have one entry per class")
        labels_cast(Int(py=classical_address(given)), admitted_dtype(given, context), prior_address, "<f8", C, context)
        with GILReleased(Python()):
            nc_normalize_priors(prior_address, C)
        _ = given
    # Preserve publication before the variance/shrinkage checks, including
    # the established partial fitted-state behavior after a later refusal.
    py_setitem(fields, "class_prior_", prior)
    var stats = admitted_storage_native([d], "<f4", context, as_array=True)
    var new_cent = admitted_storage_native([C, d], "<f4", context, as_array=True)
    var devs = admitted_storage_native([C, d], "<f4", context, as_array=True)
    var x_address = Int(py=classical_address(x))
    var stats_address = Int(py=classical_address(stats))
    var cent_address = Int(py=classical_address(new_cent))
    var dev_address = Int(py=classical_address(devs))
    var status = 0
    with GILReleased(Python()):
        status = plan.statistics(connection.driver, x_address, label_address, stats_address)
    admitted_finite_refuse(status, String(py_type(model).__name__))
    var median = Float64(0)
    var all_zero = False
    with GILReleased(Python()):
        var summary = plan.median_std(connection.driver, stats_address)
        median = summary[0]
        all_zero = summary[1]
    if all_zero:
        var constant = False
        with GILReleased(Python()):
            constant = plan.constant(connection.driver, x_address)
        if constant:
            raise Error("All features have zero variance. Division by zero.")
    var threshold = family_field(fields, "shrink_threshold")
    var shrink = Float64(py=family_call(b.float, threshold)) if _truth(threshold) else 0.0
    with GILReleased(Python()):
        plan.shrink(connection.driver, x_address, stats_address, cent_address, dev_address, median, shrink)
    _ = plan^
    _ = connection^
    _ = x
    _ = lab
    py_setitem(fields, "centroids_", new_cent)
    py_setitem(fields, "deviations_", devs)
    py_setitem(fields, "within_class_std_dev_", stats)
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "_codes_classes", classes)
    py_setitem(fields, "n_features_in_", _i(d))


def _nc_uniform(fields: PythonObject, context: PythonObject) raises -> Bool:
    var prior = family_field(fields, "class_prior_")
    var count = _size(prior)
    var address = Int(py=classical_address(prior))
    var result = False
    with GILReleased(Python()):
        result = nc_uniform(address, count)
    _ = prior
    return result


struct _NcOperands(Movable):
    # Boundary owners only. Native execution consumes the typed request and
    # the native narrowed-prior buffer; public objects supply lifetime pins.
    var query: PythonObject
    var centroids: PythonObject
    var std: PythonObject
    var prior: List[Float32]
    var request: NcQueryPlan

    def __init__(out self, query: PythonObject, centroids: PythonObject, std: PythonObject,
                 var prior: List[Float32], request: NcQueryPlan):
        self.query = query
        self.centroids = centroids
        self.std = std
        self.prior = prior^
        self.request = request.copy()


def _nc_decision_inputs(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> _NcOperands:
    if _ne(family_field(fields, "metric"), PythonObject("euclidean")):
        raise Error("AttributeError: decision_function is available for metric='euclidean' only")
    var q = _f32(source, context)
    var given = family_field(fields, "class_prior_")
    var count = _size(given)
    var prior = List[Float32](length=count, fill=0)
    var source_address = Int(py=classical_address(given))
    var prior_address = Int(prior.unsafe_ptr())
    with GILReleased(Python()):
        nc_narrow_priors(source_address, prior_address, count)
    _ = given
    var cent = family_field(fields, "centroids_")
    var std = family_field(fields, "within_class_std_dev_")
    var request = NcQueryPlan(Int(py=classical_address(q)), Int(py=classical_address(cent)),
        Int(py=classical_address(std)), prior_address, _rows(q), _cols(q), count)
    return _NcOperands(q, cent, std, prior^, request)


def _nc_predict_values(model: PythonObject, binding: PythonObject, source: PythonObject,
                       context: PythonObject, labels: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var q = _f32(source, context)
    var classes = labels if labels is not None else classifier_label_list(family_field(fields, "classes_"), context)
    var connection = NcConnection(binding)
    var rows = _rows(q)
    var fused = rows > 0 and connection.fused
    var codes = admitted_storage_native([rows], "<i4", context, as_array=True)
    var code_address = Int(py=classical_address(codes))
    if _nc_uniform(fields, context):
        var cent = family_field(fields, "centroids_")
        var query_address = Int(py=classical_address(q))
        var request = NcQueryPlan(query_address, Int(py=classical_address(cent)), query_address,
                                  query_address, rows, _cols(q), _rows(cent))
        var euclidean = _eq(family_field(fields, "metric"), PythonObject("euclidean"))
        var status = 0
        with GILReleased(Python()):
            if fused:
                status = request.fused(connection.driver, "nc_labels", code_address, 0 if euclidean else 1)
            else:
                request.distance_labels(connection.driver, code_address, euclidean)
        if fused:
            admitted_finite_refuse(status, String(py_type(model).__name__))
        _ = cent
    else:
        var inputs = _nc_decision_inputs(fields, q, context)
        var status = 0
        with GILReleased(Python()):
            if fused:
                status = inputs.request.fused(connection.driver, "nc_labels", code_address, 2)
            else:
                inputs.request.score_labels(connection.driver, code_address)
        if fused:
            admitted_finite_refuse(status, String(py_type(model).__name__))
        _ = inputs^
    _ = connection^
    _ = q
    return _class_array(classes, codes, context)


def _nc_predict(model: PythonObject, args: PythonObject, context: PythonObject,
                labels: PythonObject) raises -> PythonObject:
    return _nc_predict_values(model, args[0], args[1], context, labels)


def _nc_decision(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var inputs = _nc_decision_inputs(py_fields(model), args[1], context)
    var connection = NcConnection(args[0])
    var n = inputs.request.rows
    var C = inputs.request.classes
    var status = List[Int32](length=1, fill=0)
    var output: PythonObject
    if C == 2:
        var scores = List[Float32](length=n * C, fill=0)
        with GILReleased(Python()):
            inputs.request.decision(connection.driver, Int(scores.unsafe_ptr()), Int(status.unsafe_ptr()))
        admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
        output = admitted_storage_native([n], "<f4", context, as_array=True)
        var dst = Int(py=classical_address(output))
        with GILReleased(Python()):
            nc_binary_decision(connection.driver, Int(scores.unsafe_ptr()), dst, n)
        _ = scores^
    else:
        output = admitted_storage_native([n, C], "<f4", context, as_array=True)
        var dst = Int(py=classical_address(output))
        with GILReleased(Python()):
            inputs.request.decision(connection.driver, dst, Int(status.unsafe_ptr()))
        admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
    _ = status^
    _ = inputs^
    _ = connection^
    return output


def _nc_probabilities(model: PythonObject, args: PythonObject, context: PythonObject,
                      logarithmic: Bool) raises -> PythonObject:
    var connection = NcConnection(args[0])
    var mode = 1 if logarithmic else (2 if connection.wide else 0)
    var inputs = _nc_decision_inputs(py_fields(model), args[1], context)
    var rows = inputs.request.rows
    var C = inputs.request.classes
    var output = admitted_storage_native([rows, C], "<f8" if mode == 2 else "<f4", context, as_array=True)
    var dst = Int(py=classical_address(output))
    var fused = connection.fused and rows > 0
    var status = 0
    with GILReleased(Python()):
        if fused:
            status = inputs.request.fused(connection.driver, "nc_proba", dst, mode)
        else:
            inputs.request.probabilities(connection.driver, dst, mode)
    if fused:
        admitted_finite_refuse(status, String(py_type(model).__name__))
    _ = inputs^
    _ = connection^
    return output


def _nc_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _nc_probabilities(model, args, context, False)


def _nc_predict_log_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _nc_probabilities(model, args, context, True)


def _nc_score(model: PythonObject, args: PythonObject, context: PythonObject,
              labels: PythonObject) raises -> PythonObject:
    var prediction = _nc_predict_values(model, args[0], args[1], context, labels)
    return PythonObject(classical_accuracy(args[2], prediction, args[3], PythonObject(None), context))


# ------------------------------------------------------------- KNNImputer


def _imp_check(fields: PythonObject, context: PythonObject) raises:
    """`KNNImputer._check`."""
    var b = _b()
    var mv = family_field(fields, "missing_values")
    # The former numbers.Real test, by exact class identity: isinstance with
    # the numbers ABC runs ABCMeta.__instancecheck__ (a Python frame) on every
    # fit. Kinds 3 and 4 are the exact native integer and real scalar classes
    # (Python and connected NumPy); bool and NumPy bool_ (kind 2) are refused
    # as before.
    if (mv is None or family_isinstance(mv, Python.tuple(b.str, b.bool))
            or labels_primitive_kind(py_type(mv), context) < 3):
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
    var k = _int_word(family_field(fields, "n_neighbors"))
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
    var flags = List[Int]()
    var all_valid = True
    var true_word = PythonObject(True)
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks, read once)
        # Perf (perf2-neighbors): fit stores exact bools; read those by
        # identity instead of one Python `bool()` call per feature.
        var item = valid[f]
        var present = (item is true_word) if py_type(item) is b.bool else _truth(item)
        if present:
            keep.append(f)
            flags.append(0)
        else:
            all_valid = False
            flags.append(1)
    if _truth(family_field(fields, "keep_empty_features")):
        if n != 0 and not all_valid:
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
    var labels = _admit_local_state(model, context, operation.endswith("_fit") or "_fit_" in operation)
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
        return _lof_score_samples(args[0], fields, args[1], context, String(py_type(model).__name__))
    if operation == "family_xn_loc_lof_decision_function":
        return _lof_decision(model, args, context)
    if operation == "family_xn_loc_lof_predict":
        return _lof_predict(model, args, context)
    if operation == "family_xn_loc_nc_fit":
        _nc_fit(model, args, context)
        return model
    if operation == "family_xn_loc_nc_predict":
        return _nc_predict(model, args, context, labels)
    if operation == "family_xn_loc_nc_decision_function":
        return _nc_decision(model, args, context)
    if operation == "family_xn_loc_nc_predict_proba":
        return _nc_predict_proba(model, args, context)
    if operation == "family_xn_loc_nc_predict_log_proba":
        return _nc_predict_log_proba(model, args, context)
    if operation == "family_xn_loc_nc_score":
        return _nc_score(model, args, context, labels)
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
