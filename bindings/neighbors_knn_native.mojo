# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024):
# NearestNeighbors.fit and the k-NN classifier/regressor `fit`, `predict`,
# `predict_proba` and `score` (python/mojolearn/neighbors.py). Each
# `family_xn_knn_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python methods' refusals (wording,
# class and order), operations and order, so the bits do not move:
#
#   _check_refusals      `NearestNeighbors._check_refusals` and the k-NN
#                        classes' weight resolution (`_resolve_metric`,
#                        `_resolve_rbc_metric`, `_resolve_weights`,
#                        `_refuse_inert_p`)
#   fit                  the float32 index (`as_f32_dense_c`), the
#                        jensenshannon input check (`Array.min`, the core
#                        helper `reduce_stat`), the label columns (int32 for
#                        the classifier, float32 for the regressor; several
#                        outputs transposed by the x_neighbors binding's
#                        `xn_p2m_transpose[_i]`) and the classes (the native
#                        encoder `prepare_targets`)
#   predict              the device-resident index (`knn_index_prepare`,
#                        `knn_classify_resident` / `knn_regress_resident`) or
#                        the per-call entry, the class-set check (POLICY 7)
#                        and the label decoding
#   score                `ClassifierScoreMixin.score` (the accuracy owner) and
#                        `RegressorScoreMixin.score` (the metrics owner's R^2)
#
# Public methods pass None in the binding slot. The operation owner resolves
# the import-connected provider after entry: `_mojolearn` for prediction and
# x_neighbors for fit. Explicit bindings remain supported for native drivers.
from providers.python.estimator_binding_pyapi import estimator_arguments, estimator_binding
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_buffer_owner import native_buffer_view
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_is_str, family_isinstance, family_call, family_array,
    family_as_f32, family_as_f32_dense, family_owned, family_field,
)
from bindings.classical_labels import (
    LabelsView, labels_view, labels_check_rank, labels_cast, labels_all_finite, labels_helper,
    labels_encode_labels, labels_shape_of, labels_reshape, classical_decode_labels, classical_accuracy,
    _native_labels,
)
from bindings.mcls_common import mc_is_integer_labels, mc_dtype_name
from bindings.mcls_cluster import mcls_cluster_operation
from bindings.host_archive_native import host_archive_rbc_metric_object

#: cuVS `DistanceType` values the refusals and the input check read
#: (python/mojolearn/neighbors.py `_DIST_LP_UNEXPANDED`, `_DIST_JENSEN_SHANNON`).
comptime _DIST_LP = 9
comptime _DIST_JS = 15
comptime _FLT_MIN = 1.1754943508222875e-38
#: `_array._NATIVE_CODE` and `_REDUCE_*` of the core helper `reduce_stat`.
comptime _CODE_F32 = 0
comptime _CODE_F64 = 1
comptime _CODE_I32 = 2
comptime _CODE_I64 = 3
comptime _REDUCE_MIN = 0
comptime _REDUCE_MAX = 1
comptime _REDUCE_INTEGRAL = 4


# ------------------------------------------------------------------ Python glue


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`, as an `if` reads a comparison's result."""
    return Bool(py=family_call(_b().bool, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _ne(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(family_call(native_loaded_module("operator").ne, a, b))


def _lt(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(family_call(native_loaded_module("operator").lt, a, b))


def _gt(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(family_call(native_loaded_module("operator").gt, a, b))


def _class_name(model: PythonObject) raises -> String:
    """`type(self).__name__`."""
    return String(py_type(model).__name__)


def _reduce(addr: Int, code: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` / `max()` / the integral test: the core helper
    `reduce_stat` over a contiguous block of `count` elements."""
    return family_call(labels_helper(context, "reduce_stat"), PythonObject(addr), PythonObject(code),
                       PythonObject(count), PythonObject(what))


def knn_door(binding: PythonObject, name: String) raises -> PythonObject:
    """`NearestNeighbors._resident_door`: the binding's entry `name`, or None
    where the loaded binding has no such door (an absent name is ImportError
    on a host binding, AttributeError on a module); any other exception the
    lookup raises reaches the caller unchanged."""
    ref cpy = Python().cpython()
    var b = _b()
    var import_error = b.ImportError
    var attribute_error = b.AttributeError
    var key = PythonObject(name)
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetAttr")(binding._obj_ptr, key._obj_ptr)
    _ = key^
    if not pointer:
        if (cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(import_error._obj_ptr) != 0
                or cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(attribute_error._obj_ptr) != 0):
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            return PythonObject(None)
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def _holder(fields: PythonObject) raises -> PythonObject:
    """The instance's `_resident_holder` dict, created in `__dict__` when it
    is missing (the `_resident` / query-report property setters)."""
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    if holder is None:
        holder = Python.dict()
        py_setitem(fields, "_resident_holder", holder)
    return holder


def _report(fields: PythonObject, name: String, value: PythonObject) raises:
    """A query report (`used_query_tile_`, `n_candidate_distances_`): set in
    the holder, never in `__dict__` itself."""
    py_setitem(_holder(fields), PythonObject(name), value)


def knn_release(model: PythonObject) raises:
    """`NearestNeighbors._release_resident_index`: drop the device copy of
    the index, if one is held, through the minting binding's
    `knn_index_release`. Quiet on a binding that cannot be reached any more
    and on a handle already released."""
    var b = _b()
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    try:
        _ = py_call(py_call(b.getattr, held[2], PythonObject("knn_index_release")), held[1])
    except:
        pass


def _resident_handle(model: PythonObject, binding: PythonObject, idx: PythonObject) raises -> PythonObject:
    """`NearestNeighbors._resident_index_handle(binding, idx)`: the handle of
    the device-resident copy of `idx`, prepared on the first call and reused
    while the index keeps its address and shape; None where the binding has
    no `knn_index_prepare`."""
    var b = _b()
    var prepare = knn_door(binding, "knn_index_prepare")
    if prepare is None:
        return PythonObject(None)
    var fields = py_fields(model)
    var key = Python.tuple(classical_address(idx), family_call(b.tuple, idx.shape))
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    var cached = PythonObject(None)
    if holder is not None and _truth(holder):
        cached = py_get(holder, PythonObject("index"), PythonObject(None))
    if cached is not None and _eq(cached[0], key):
        return cached[1]
    if cached is not None:
        knn_release(model)
    var handle = family_call(b.int, family_call(prepare, key[0],
                                                Python.list(family_call(b.int, idx.shape[0]),
                                                            family_call(b.int, idx.shape[1]))))
    _ = idx  # keep the Python object alive through the native call above
    # The minting binding travels with the handle: the release (here, and
    # the native finalizer when the instance is dropped) is its own
    # `knn_index_release`.
    py_setitem(_holder(fields), "index", Python.tuple(key, handle, binding))
    return handle


# ------------------------------------------------------------------ refusals


def _refuse_inert_p(cls_name: String, metric: PythonObject, p: PythonObject) raises:
    """`neighbors._refuse_inert_p`: only metric='minkowski'/'lp' reads p; any
    value but the default 2 under another metric is refused."""
    var b = _b()
    var is_default = False
    if not family_isinstance(p, b.bool):
        # `float(p)`; only its TypeError or ValueError reads as "not the
        # default", any other exception reaches the caller unchanged.
        ref cpy = Python().cpython()
        var type_error = b.TypeError
        var value_error = b.ValueError
        var envelope = Python.tuple(p)
        var result = cpy.PyObject_Call(b.float._obj_ptr, envelope._obj_ptr, PyObjectPtr())
        _ = envelope^
        if result:
            is_default = Float64(py=PythonObject(from_owned=result)) == 2.0
        elif (cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(type_error._obj_ptr) != 0
                or cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(value_error._obj_ptr) != 0):
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
        else:
            var failure = stash_python_error()
            raise failure^
    if not is_default:
        raise Error("mojolearn " + cls_name + ": p is read only by metric='minkowski' "
                    + "(or 'lp'); metric=" + family_repr(metric) + " never reads it, so p=" + family_repr(p)
                    + " would change nothing. Leave p at its default (2) or ask for metric='minkowski'.")


def knn_resolve_metric(cls_name: String, metric: PythonObject, p: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`neighbors._resolve_metric(cls_name, metric, p)`: `(metric_value,
    metric_arg)` for the brute-force table, or the same refusal, by name and
    in the same order."""
    var b = _b()
    var head = "mojolearn " + cls_name + ": "
    if not family_is_str(metric):
        raise Error(head + "metric must be a name, got " + family_repr(metric) + ". "
                    + "A callable metric is not implemented: it would have to run inside a GPU kernel.")
    var key = family_call(py_call(b.getattr, metric, PythonObject("lower")))
    var table = context["host_archive_neighbor_metrics"]
    var value = 0
    if py_contains(table, key):
        value = Int(py=table[key])
    elif py_contains(context["host_archive_unsupported_metrics"], key):
        raise Error(head + "metric=" + family_repr(metric) + " is in cuML's "
                    + "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
                    + "(neighbors/NOT_IMPLEMENTED.tsv). Implemented: "
                    + String(py_call(PythonObject(", ").join, py_call(b.sorted, table))))
    else:
        raise Error(head + "unknown metric " + family_repr(metric) + ". Implemented: "
                    + String(py_call(PythonObject(", ").join, py_call(b.sorted, table))))
    if value != _DIST_LP:
        _refuse_inert_p(cls_name, metric, p)
    var arg = Float64(2.0)
    if value == _DIST_LP:
        arg = Float64(py=family_call(b.float, p))
        # `not (arg > 0.0) or arg != arg or arg == float("inf")`
        if not (arg > 0.0) or arg != arg or (arg > 0.0 and not isfinite(arg)):
            raise Error(head + "metric='minkowski' needs a finite "
                        + "p > 0, got " + family_repr(p) + ". p = 0 makes 1/p infinite, p < 0 is not "
                        + "a metric, and p = infinity is metric='chebyshev'.")
        if arg < _FLT_MIN:
            raise Error(head + "metric='minkowski' p=" + family_repr(p) + " is "
                        + "subnormal in float32; the flush policy differs by vendor "
                        + "so this cannot be one arithmetic (DEVIATION 552)")
    return Python.tuple(PythonObject(value), PythonObject(arg))


def knn_resolve_weights(cls_name: String, weights: PythonObject) raises -> Int:
    """`neighbors._resolve_weights(cls_name, weights)`: 0 uniform (or None),
    1 distance; a callable or any other value is refused by name."""
    var b = _b()
    if weights is None:
        return 0
    if _truth(family_call(b.callable, weights)):
        raise Error("mojolearn " + cls_name + ": weights=<callable> is NOT IMPLEMENTED. "
                    + "scikit-learn calls it on the distance matrix in Python; the "
                    + "vote here runs in a GPU kernel and there is no portable way "
                    + "to lift a Python function into one. Use 'uniform' or "
                    + "'distance'.")
    var table = Python.dict()
    py_setitem(table, "uniform", PythonObject(0))
    py_setitem(table, "distance", PythonObject(1))
    if not family_is_str(weights) or not py_contains(table, weights):
        raise Error("mojolearn " + cls_name + ": weights=" + family_repr(weights) + " is not a "
                    + "weighting; use 'uniform' or 'distance'")
    return Int(py=table[weights])


def _is_knn(model: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(self, (KNeighborsClassifier, KNeighborsRegressor))`."""
    return family_isinstance(model, context["family_xn_knn_classes"])


def _check_refusals(model: PythonObject, context: PythonObject) raises:
    """`NearestNeighbors._check_refusals` (and, for the k-NN classes, their
    `_resolve_weights` after it): every refusal by name, at fit."""
    var fields = py_fields(model)
    var cls_name = _class_name(model)
    var knn = _is_knn(model, context)
    if _eq(family_field(fields, "algorithm"), PythonObject("rbc")) and knn:
        raise Error("mojolearn " + cls_name + ": algorithm='rbc' is "
                    + "refused. The ball cover's k-NN query is wired into "
                    + "NearestNeighbors.kneighbors only; this class runs cuML's "
                    + "knn_classify / knn_regress, which perform their own "
                    + "brute-force search inside one call and have no seam to "
                    + "hand an index to. Accepting 'rbc' here would be accepting "
                    + "a parameter that changes nothing. Use "
                    + "NearestNeighbors(algorithm='rbc').kneighbors and vote on "
                    + "the result, or leave this at 'brute'.")
    var metric = family_field(fields, "metric")
    if (_eq(py_get(fields, PythonObject("weights"), PythonObject(None)), PythonObject("distance"))
            and family_is_str(metric)
            and _eq(family_call(py_call(_b().getattr, metric, PythonObject("lower"))), PythonObject("inner_product"))):
        raise Error("mojolearn " + cls_name + ": weights='distance' is refused "
                    + "with metric='inner_product': it weighs a neighbour by 1 / d, "
                    + "and an inner product is a similarity (larger is nearer, and "
                    + "it can be zero or negative), not a distance")
    var algorithm = family_field(fields, "algorithm")
    if _eq(algorithm, PythonObject("rbc")):
        _ = host_archive_rbc_metric_object(cls_name, family_field(fields, "metric"), family_field(fields, "p"), context)
    elif py_contains(Python.tuple(PythonObject("brute"), PythonObject("auto")), algorithm):
        _ = knn_resolve_metric(cls_name, family_field(fields, "metric"), family_field(fields, "p"), context)
    else:
        raise Error("mojolearn NearestNeighbors: algorithm=" + family_repr(algorithm) + " is "
                    + "refused. Three are accepted: 'brute' (exact brute force), "
                    + "'auto' (which means 'brute' here), and 'rbc' (exact k-NN "
                    + "over a RANDOM BALL COVER, an index). "
                    + "'kd_tree' and 'ball_tree' are refused, and it is an "
                    + "ENGINEERING refusal rather than an attribution one: a "
                    + "kd-tree query is a per-query stack walk with "
                    + "data-dependent branching and divergent memory access, "
                    + "which is the shape a GPU is worst at, and above roughly "
                    + "15 dimensions the pruning bound stops firing and it "
                    + "degenerates to a full scan, so it would lose to 'brute' "
                    + "on exactly the workloads it claims to help. If what you "
                    + "want is an index instead of a full scan, that is 'rbc': "
                    + "it is EXACT, not approximate, it has no per-query stack, "
                    + "and it is the same structure RadiusNeighbors uses "
                    + "(neighbors/impl/ball_cover/knn.mojo).")
    if knn:
        _ = knn_resolve_weights(cls_name, family_field(fields, "weights"))


def _dist_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`self._dist_params()`: `[metric, metric_arg, weights]`, the list
    `bindings/_mojolearn.mojo::_dist_triple` reads (the metric resolved
    first, then the weights; the base class never weights)."""
    var fields = py_fields(model)
    var cls_name = _class_name(model)
    var resolved = knn_resolve_metric(cls_name, family_field(fields, "metric"), family_field(fields, "p"), context)
    var weights = 0
    if _is_knn(model, context):
        weights = knn_resolve_weights(cls_name, family_field(fields, "weights"))
    return Python.list(resolved[0], resolved[1], PythonObject(weights))


def _check_metric_input(cls_name: String, value: Int, x: PythonObject, context: PythonObject) raises:
    """`neighbors._check_metric_input`: jensenshannon takes logarithms, so a
    negative coordinate is refused before any device work (`x.min()` is
    `Array.min`, the core helper `reduce_stat`)."""
    if value != _DIST_JS:
        return
    var size = Int(py=x.size)
    if size == 0:
        return
    var low = Float64(py=family_call(_b().float, _reduce(Int(py=classical_address(x)), _CODE_F32, size, _REDUCE_MIN,
                                                           context)))
    _ = x  # keep the Python object alive through the native call above
    if low < 0.0:
        raise Error("mojolearn " + cls_name + ": metric='jensenshannon' needs X >= 0 "
                    + "(it takes logarithms of the coordinates)")


# ------------------------------------------------------------------ fit


def _nn_fit(model: PythonObject, x_source: PythonObject, context: PythonObject) raises:
    """`NearestNeighbors.fit(X)`: the refusals, the release of the last
    fit's device index, then the float32 points held on the instance."""
    var fields = py_fields(model)
    var cls_name = _class_name(model)
    _check_refusals(model, context)
    # A refit drops the device copy of the previous index HERE, not at the
    # next call: the copy is keyed on the array's address and shape.
    knn_release(model)
    # Held on the instance so the memory outlives this call: the search
    # borrows the address at query time (an owned float32 Array: the cast
    # result itself, or one copy of a caller's float32 block).
    var idx = family_owned(family_as_f32_dense(x_source, 2, "X", context), context)
    if not _eq(family_field(fields, "algorithm"), PythonObject("rbc")):
        var resolved = knn_resolve_metric(cls_name, family_field(fields, "metric"), family_field(fields, "p"), context)
        _check_metric_input(cls_name, Int(py=resolved[0]), idx, context)
    py_setitem(fields, "_index", idx)
    py_setitem(fields, "n_samples_fit_", idx.shape[0])
    py_setitem(fields, "n_features_in_", idx.shape[1])


def _y_shape(y: PythonObject, fields: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_shape_of(y)` with the k-NN fits' rank and row-count refusals. An
    object with neither a `shape` nor list/tuple nesting reads `(len(y),)`
    when it has a length and `()` otherwise, as `_buffer.shape_of` did."""
    var b = _b()
    var shape = List[Int]()
    if (y is not None and (family_isinstance(y, Python.tuple(b.list, b.tuple))
                           or _truth(py_call(b.hasattr, y, PythonObject("shape"))))):
        shape = labels_shape_of(y, context)
    elif y is not None and _truth(py_call(b.hasattr, y, PythonObject("__len__"))):
        shape.append(Int(py=family_call(b.len, y)))
    if len(shape) != 1 and len(shape) != 2:
        raise Error("mojolearn: y must be 1-D or 2-D, got " + String(len(shape)) + "-D")
    var n_fit = family_field(fields, "n_samples_fit_")
    if _ne(PythonObject(shape[0]), n_fit):
        raise Error("mojolearn: y has " + String(shape[0]) + " rows, X has " + family_str(n_fit))
    return shape^


def _is_float_buffer(y: PythonObject, context: PythonObject) raises -> Bool:
    """`_sk_validate.describe(y)` is `("buffer", shape, "float")`: a float
    buffer export (or a float Array), never a list, a sparse matrix, a
    frame or a str/bytes/dict."""
    var b = _b()
    if y is None:
        return False
    if (_truth(py_call(b.hasattr, y, PythonObject("tocsr"))) and _truth(py_call(b.hasattr, y, PythonObject("nnz")))
            and _truth(py_call(b.hasattr, y, PythonObject("shape")))):
        return False
    if family_isinstance(y, Python.tuple(b.list, b.tuple, b.str, b.bytes, b.dict)):
        return False
    var fmt: String
    if py_type(y) is context["array_class"]:
        fmt = String(y.dtype)
    else:
        try:
            fmt = String(native_buffer_view(y).format)
        except:
            return False  # no buffer export: a frame-like shape, or nothing
    var f = String(py_call(PythonObject(fmt).lstrip, PythonObject("@=<>!|")))
    return f == "f" or f == "d" or f == "e" or f == "g" or f == "f2" or f == "f4" or f == "f8"


def _whole_float_labels(y: PythonObject, context: PythonObject) raises -> Bool:
    """`_sk_validate.whole_float_labels(y)`: a float buffer whose every value
    is a finite whole number. The finiteness scan is the base binding's
    `all_finite`; the integral test is `reduce_stat` over the distinct values
    `prepare_targets` found (`_all_finite`, then `_integral`)."""
    if not _is_float_buffer(y, context):
        return False
    var finite_view = labels_view(y, "X", context)
    if finite_view.typestr == "<f4" or finite_view.typestr == "<f8":
        var finite = labels_all_finite(finite_view.addr, finite_view.size(), finite_view.typestr == "<f8", context)
        _ = finite_view  # keep the Python object alive through the native call above
        if not finite:
            return False
    _ = finite_view^
    var view = labels_view(y, "y", context)
    var n = view.size()
    var big = 0
    for k in range(view.ndim()):  # driver-loop: over array rank metadata
        if view.shape[k] > big:
            big = view.shape[k]
    if (view.typestr != "<f4" and view.typestr != "<f8") or n == 0 or n != big:
        return True
    var classes_store = family_array(Python.tuple(n), "<f8", context)
    var codes = family_array(Python.tuple(n), "<i4", context)
    var n_distinct = Int(py=family_call(labels_helper(context, "prepare_targets"), PythonObject(view.addr),
                               PythonObject(_CODE_F32 if view.typestr == "<f4" else _CODE_F64), PythonObject(n),
                               classical_address(classes_store), classical_address(codes),
                               PythonObject(0), PythonObject(0), PythonObject(0)))
    _ = view  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    var integral = _truth(_reduce(Int(py=classical_address(classes_store)), _CODE_F64, n_distinct, _REDUCE_INTEGRAL, context))
    _ = classes_store  # keep the Python object alive through the native call above
    return integral


def _encode_column(y_cols: PythonObject, row: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """`encode_labels(y_cols[row])[0]`: the sorted classes of one int32
    label column (the native encoder over the row of the column block)."""
    var shape = List[Int]()
    shape.append(n)
    var strides = List[Int]()
    strides.append(4)
    var view = LabelsView(y_cols, Int(py=classical_address(y_cols)) + 4 * row * n, shape^, strides^, "<i4", 4)
    var labels = _native_labels(view, False, context)
    _ = y_cols  # keep the Python object alive through the native call above
    return labels.classes


def _clf_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KNeighborsClassifier.fit(X, y)`: the index, then int32 label columns
    (POLICY 6: `n_outputs` contiguous columns) and the classes per column."""
    var b = _b()
    var xn = args[0]
    _nn_fit(model, args[1], context)
    var fields = py_fields(model)
    var y = args[2]
    var shape = _y_shape(y, fields, context)
    # whole-number float labels convert exactly to the integer labels below
    py_setitem(fields, "_label_names", PythonObject(None))
    if len(shape) == 1 and not mc_is_integer_labels(y, context) and not _whole_float_labels(y, context):
        # Labels of any other kind (str labels) vote as their dense codes in
        # `classes_` order; `predict` decodes the voted codes.
        var encoded = labels_encode_labels(y, context)
        var names = encoded.classes
        for i in range(len(names)):  # driver-loop: over kind test of the class labels (classes-sized: distinct classes)
            if not family_is_str(names[i]):
                raise Error("mojolearn: y must be integer or str class labels, got "
                            + mc_dtype_name(y, context) + "; cast your labels")
        py_setitem(fields, "_label_names", names)
        y = encoded.codes
    if not mc_is_integer_labels(y, context) and not _whole_float_labels(y, context):
        raise Error("mojolearn: y must be integer class labels, got "
                    + mc_dtype_name(y, context) + "; cuML converts to int32 "
                    + "(check_dtype=np.int32) and so does this surface -- cast "
                    + "your labels")
    # `as_i64_c(y, ndim=len(shape), name="y")`: the labels as int64 in the
    # view's storage order (the input itself when it already is int64).
    var view = labels_view(y, "y", context)
    labels_check_rank(view, len(shape), "y")
    var count = view.size()
    var wide = PythonObject(None)
    var wide_addr = view.addr
    if view.typestr != "<i8":
        wide = family_array(Python.tuple(count), "<i8", context)
        wide_addr = Int(py=classical_address(wide))
        labels_cast(view.addr, view.typestr, wide_addr, "<i8", count, context)
        _ = wide  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    var low = _reduce(wide_addr, _CODE_I64, count, _REDUCE_MIN, context)
    _ = wide  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    var high = PythonObject(None)
    if not _lt(low, PythonObject(-(1 << 31))):
        high = _reduce(wide_addr, _CODE_I64, count, _REDUCE_MAX, context)
        _ = wide  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native call above
    if _lt(low, PythonObject(-(1 << 31))) or _gt(high, PythonObject((1 << 31) - 1)):
        raise Error("mojolearn: y does not fit int32")
    var n = shape[0]
    var n_out = 1 if len(shape) == 1 else shape[1]
    py_setitem(fields, "outputs_2d_", PythonObject(len(shape) == 2 and shape[1] != 1))
    if n_out == 1:
        # One output is already its own contiguous column: the int64 ->
        # int32 narrowing (exact after the range check) and the classes from
        # the native encoder on that buffer.
        var y32 = family_array(Python.tuple(n), "<i4", context)
        labels_cast(wide_addr, "<i8", Int(py=classical_address(y32)), "<i4", n, context)
        _ = y32  # keep the Python object alive through the native call above
        _ = wide  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native call above
        py_setitem(fields, "_y_cols", labels_reshape(y32, Python.tuple(1, n), context))
        py_setitem(fields, "_classes_list", Python.list(labels_encode_labels(y32, context).classes))
        return
    var yc = family_array(Python.tuple(n_out, n), "<i4", context)
    if view.c_order():
        # The narrowing is a C-level copy (exact after the range check) and
        # the transpose a device op (`xn_p2m_transpose_i`; the host binding's
        # same op on a CPU-only install).
        var y32 = family_array(Python.tuple(n, n_out), "<i4", context)
        labels_cast(wide_addr, "<i8", Int(py=classical_address(y32)), "<i4", count, context)
        _ = y32  # keep the Python object alive through the native call above
        _ = wide  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native call above
        if n:
            _ = family_call(family_call(b.getattr, xn, PythonObject("xn_p2m_transpose_i")),
                            Python.list(classical_address(y32), classical_address(yc)),
                            Python.list(PythonObject(n), PythonObject(n_out)), Python.list())
            _ = y32  # keep the Python object alive through the native call above
            _ = yc  # keep the Python object alive through the native call above
    else:
        # A column-major y already stores the `n_outputs` contiguous columns:
        # its narrowing in storage order IS the column block the transpose of
        # its C-order copy produced (integers, so the same words).
        labels_cast(wide_addr, "<i8", Int(py=classical_address(yc)), "<i4", count, context)
        _ = yc  # keep the Python object alive through the native call above
        _ = wide  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native call above
    py_setitem(fields, "_y_cols", yc)
    var classes = Python.list()
    for j in range(n_out):  # driver-loop: over output columns, one native encoding each
        _ = py_call(classes.append, _encode_column(yc, j, n, context))
    py_setitem(fields, "_classes_list", classes)


def _reg_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KNeighborsRegressor.fit(X, y)`: the index, then float32 target
    columns (POLICY 6)."""
    var b = _b()
    var xn = args[0]
    _nn_fit(model, args[1], context)
    var fields = py_fields(model)
    var shape = _y_shape(args[2], fields, context)
    var ya = family_as_f32(args[2], len(shape), "y", context)
    var n = shape[0]
    var n_out = 1 if len(shape) == 1 else shape[1]
    if n_out == 1:
        # One column IS one contiguous row of the transposed layout.
        py_setitem(fields, "_y_cols", labels_reshape(family_owned(ya, context), Python.tuple(1, n), context))
    else:
        # The transpose on the device (`xn_p2m_transpose`; the host binding's
        # same op on a CPU-only install).
        var yc = family_array(Python.tuple(n_out, n), "<f4", context)
        if n:
            _ = family_call(family_call(b.getattr, xn, PythonObject("xn_p2m_transpose")),
                            Python.list(classical_address(ya), classical_address(yc)),
                            Python.list(PythonObject(n), PythonObject(n_out)), Python.list())
            _ = ya  # keep the Python object alive through the native call above
            _ = yc  # keep the Python object alive through the native call above
        py_setitem(fields, "_y_cols", yc)
    py_setitem(fields, "outputs_2d_", PythonObject(len(shape) == 2 and shape[1] != 1))


# ------------------------------------------------------------------ predict


def _queries(model: PythonObject, x_source: PythonObject, context: PythonObject) raises -> PythonObject:
    """The prediction preamble: fitted check, float32 queries, the metric
    input check (with `_dist_params`' refusals), width and k checks."""
    var fields = py_fields(model)
    if (py_get(fields, PythonObject("_index"), PythonObject(None)) is None
            or py_get(fields, PythonObject("_y_cols"), PythonObject(None)) is None):
        raise Error("mojolearn: call fit before predict")
    var q = family_as_f32_dense(x_source, 2, "X", context)
    _check_metric_input(_class_name(model), Int(py=_dist_params(model, context)[0]), q, context)
    var features = family_field(fields, "n_features_in_")
    if _ne(q.shape[1], features):
        raise Error("mojolearn: X has " + family_str(q.shape[1]) + " features, index has " + family_str(features))
    var k = family_field(fields, "n_neighbors")
    var n_fit = family_field(fields, "n_samples_fit_")
    if _lt(k, PythonObject(1)) or _gt(k, n_fit):
        raise Error("mojolearn: n_neighbors must be in [1, " + family_str(n_fit) + "], got " + family_str(k))
    return q


def _classify(model: PythonObject, binding: PythonObject, x_source: PythonObject, want_proba: Bool,
              context: PythonObject) raises -> PythonObject:
    """`KNeighborsClassifier._predict(X, want_proba)`: `(labels, proba,
    n_classes)`; the search and the vote are one compiled call."""
    var b = _b()
    var fields = py_fields(model)
    var q = _queries(model, x_source, context)
    var k = family_field(fields, "n_neighbors")
    var nq = Int(py=q.shape[0])
    var y_cols = family_field(fields, "_y_cols")
    var n_out = Int(py=y_cols.shape[0])
    var classes_list = family_field(fields, "_classes_list")
    var n_classes = Python.list()
    var total = 0
    for i in range(len(classes_list)):  # driver-loop: over class counts per output column
        var count = len(classes_list[i])
        total += count
        _ = py_call(n_classes.append, PythonObject(count))
    # The native contract writes exactly one of these two outputs; the
    # unselected one is a one-element sentinel it never reads or writes.
    var labels = family_array(Python.tuple(1), "<i4", context) if want_proba else family_array(
        Python.tuple(nq, n_out), "<i4", context)
    var proba = family_array(Python.tuple(nq * total), "<f4", context) if want_proba else family_array(
        Python.tuple(1), "<f4", context)
    var uniq = family_array(Python.tuple(total), "<i4", context)
    var idx = family_field(fields, "_index")
    # THE INDEX STAYS ON THE DEVICE (DEVIATION 3002): the first call uploads
    # it, every later call classifies against the device copy; a binding
    # without that entry takes the per-call path.
    var resident = knn_door(binding, "knn_classify_resident")
    var handle = PythonObject(None)
    if resident is not None:
        handle = _resident_handle(model, binding, idx)
    # n_index, n_queries, n_features, k, query_tile, n_outputs, want_proba,
    # then n_classes per output (bindings/_mojolearn.mojo::knn_classify_binding).
    var params = Python.list(idx.shape[0], PythonObject(nq), idx.shape[1], k, family_field(fields, "query_tile"),
                             PythonObject(n_out), PythonObject(1 if want_proba else 0))
    _ = py_call(params.extend, n_classes)
    var dist = _dist_params(model, context)
    var used: PythonObject
    if handle is not None:
        # The handle rides at the front of `params`: a binding takes at most
        # eight arguments and the classifier uses all eight.
        var front = Python.list(handle)
        _ = py_call(front.extend, params)
        used = family_call(resident, classical_address(idx), classical_address(q), classical_address(y_cols),
                           classical_address(labels), classical_address(proba), classical_address(uniq), front, dist)
    else:
        used = family_call(family_call(b.getattr, binding, PythonObject("knn_classify")), classical_address(idx),
                           classical_address(q), classical_address(y_cols), classical_address(labels),
                           classical_address(proba), classical_address(uniq), params, dist)
    _ = idx  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = y_cols  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = proba  # keep the Python object alive through the native call above
    _ = uniq  # keep the Python object alive through the native call above
    _report(fields, "used_query_tile_", used)
    # POLICY 7: the implementation's class set against ours, made visible:
    # `uniq` split at the running class counts.
    var flat = py_call(native_buffer_view(uniq._mv).tolist)
    var list_item = py_call(b.getattr, b.list, "__getitem__")
    var off = 0
    for i in range(len(classes_list)):  # driver-loop: over the class set check per output column
        var count = len(classes_list[i])
        var found = py_call(list_item, flat, py_call(b.slice, PythonObject(off), PythonObject(off + count)))
        off += count
        var expected = classes_list[i]
        if _ne(found, expected):
            var eight = py_call(b.slice, PythonObject(8))
            raise Error("RuntimeError: mojolearn: output " + String(i) + ": the implemented getUniquelabels found "
                        + "classes " + family_str(py_call(list_item, found, eight)) + "..., the host sort found "
                        + family_str(family_call(py_call(b.getattr, expected, PythonObject("__getitem__")), eight))
                        + "...; the two class sets disagree")
    return Python.tuple(labels, proba, n_classes)


def _clf_predict(model: PythonObject, binding: PythonObject, x_source: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """`KNeighborsClassifier.predict(X)`: the voted labels, int64 (decoded
    to the str classes when `y` had them)."""
    var fields = py_fields(model)
    var result = _classify(model, binding, x_source, False, context)
    var labels = result[0]
    var size = Int(py=labels.size)
    if _truth(family_field(fields, "outputs_2d_")):
        var wide2 = family_array(family_call(_b().tuple, labels.shape), "<i8", context)
        labels_cast(Int(py=classical_address(labels)), "<i4", Int(py=classical_address(wide2)), "<i8", size, context)
        _ = labels  # keep the Python object alive through the native call above
        _ = wide2  # keep the Python object alive through the native call above
        return wide2
    # (nq, 1) -> (nq,) is a C-order reshape; int32 -> int64 is exact.
    var codes = family_array(Python.tuple(Int(py=labels.shape[0])), "<i8", context)
    labels_cast(Int(py=classical_address(labels)), "<i4", Int(py=classical_address(codes)), "<i8", size, context)
    _ = labels  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    var names = py_get(fields, PythonObject("_label_names"), PythonObject(None))
    if names is not None:
        return classical_decode_labels(names, codes, context)
    return codes


def _clf_predict_proba(model: PythonObject, binding: PythonObject, x_source: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`KNeighborsClassifier.predict_proba(X)`: each output's vote fractions
    as a (n_queries, n_classes) float32 block (a copy of its slice of the
    flat buffer); a list of them for a 2-D `y`."""
    var fields = py_fields(model)
    var result = _classify(model, binding, x_source, True, context)
    var proba = result[1]
    var n_classes = result[2]
    var total = 0
    for i in range(len(n_classes)):  # driver-loop: over class counts per output column
        total += Int(py=n_classes[i])
    var nq = Int(py=proba.shape[0]) // total
    var out = Python.list()
    var off = 0
    var src = Int(py=classical_address(proba))
    for i in range(len(n_classes)):  # driver-loop: over probability block slices per output
        var count = Int(py=n_classes[i])
        var block = family_array(Python.tuple(nq, count), "<f4", context)
        memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(block))),
               src=FAMILY_F32P(unsafe_from_address=src + 4 * off), count=nq * count)
        _ = block  # keep the Python object alive through the native call above
        _ = proba  # keep the Python object alive through the native call above
        _ = py_call(out.append, block)
        off += nq * count
    if _truth(family_field(fields, "outputs_2d_")):
        return out
    return out[0]


def _reg_predict(model: PythonObject, binding: PythonObject, x_source: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """`KNeighborsRegressor.predict(X)`: the (weighted) mean target per
    query, float32; one compiled call for the search and the mean."""
    var b = _b()
    var fields = py_fields(model)
    var q = _queries(model, x_source, context)
    var k = family_field(fields, "n_neighbors")
    var nq = Int(py=q.shape[0])
    var y_cols = family_field(fields, "_y_cols")
    var n_out = Int(py=y_cols.shape[0])
    var out = family_array(Python.tuple(nq, n_out), "<f4", context)
    var idx = family_field(fields, "_index")
    # DEVIATION 3002: the regressor's resident door; see the classifier.
    var resident = knn_door(binding, "knn_regress_resident")
    var handle = PythonObject(None)
    if resident is not None:
        handle = _resident_handle(model, binding, idx)
    # n_index, n_queries, n_features, k, query_tile, n_outputs
    # (bindings/_mojolearn.mojo::knn_regress_binding).
    var params = Python.list(idx.shape[0], PythonObject(nq), idx.shape[1], k, family_field(fields, "query_tile"),
                             PythonObject(n_out))
    var dist = _dist_params(model, context)
    var used: PythonObject
    if handle is not None:
        used = family_call(resident, handle, classical_address(idx), classical_address(q), classical_address(y_cols),
                           classical_address(out), params, dist)
    else:
        used = family_call(family_call(b.getattr, binding, PythonObject("knn_regress")), classical_address(idx),
                           classical_address(q), classical_address(y_cols), classical_address(out), params, dist)
    _ = idx  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = y_cols  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _report(fields, "used_query_tile_", used)
    if _truth(family_field(fields, "outputs_2d_")):
        return out
    return labels_reshape(out, Python.tuple(nq), context)


# ------------------------------------------------------------------ score


def _clf_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ClassifierScoreMixin.score(X, y, sample_weight)`: the accuracy of
    `predict(X)` against `y` (the x_metrics binding's grouped sums)."""
    var prediction = _clf_predict(model, args[0], args[1], context)
    var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
    return PythonObject(classical_accuracy(args[2], prediction, args[3], mode, context))


def _reg_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`RegressorScoreMixin.score(X, y, sample_weight)`: y as float32
    (`as_f32_c(y, ndim=None)`), the prediction as float32, then
    `metrics.r2_score(..., numeric_mode=...)` through the metrics owner;
    args[4] is the metrics binding `_get_binding` resolved."""
    var truth = family_as_f32(args[2], -1, "y", context)
    var prediction = family_as_f32(_reg_predict(model, args[0], args[1], context), -1, "prediction", context)
    var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
    var metrics = args[4] if args[4] is not None else estimator_binding(model, "_mojolearn_metrics", context)
    return mcls_cluster_operation("family_mcls_r2", Python.tuple(metrics, truth, prediction, args[3],
                                  PythonObject("uniform_average"), PythonObject(True), mode), context)


def neighbors_knn_operation(model: PythonObject, operation: String, var args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if operation == "family_xn_knn_clf_fit" or operation == "family_xn_knn_reg_fit":
        args = estimator_arguments(model, "_mojolearn_x_neighbors", args, context)
    elif operation != "family_xn_knn_nn_fit":
        args = estimator_arguments(model, "_mojolearn", args, context)
    if operation == "family_xn_knn_nn_fit":
        # args = (X,)
        _nn_fit(model, args[0], context)
        return model
    if operation == "family_xn_knn_clf_fit":
        # args = (x_neighbors binding, X, y)
        _clf_fit(model, args, context)
        return model
    if operation == "family_xn_knn_reg_fit":
        # args = (x_neighbors binding, X, y)
        _reg_fit(model, args, context)
        return model
    if operation == "family_xn_knn_clf_predict":
        # args = (_mojolearn binding, X)
        return _clf_predict(model, args[0], args[1], context)
    if operation == "family_xn_knn_clf_predict_proba":
        return _clf_predict_proba(model, args[0], args[1], context)
    if operation == "family_xn_knn_clf_score":
        # args = (_mojolearn binding, X, y, sample_weight)
        return _clf_score(model, args, context)
    if operation == "family_xn_knn_reg_predict":
        return _reg_predict(model, args[0], args[1], context)
    if operation == "family_xn_knn_reg_score":
        # args = (_mojolearn binding, X, y, sample_weight, metrics binding)
        return _reg_score(model, args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)
