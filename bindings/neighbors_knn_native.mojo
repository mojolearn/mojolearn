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
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.ffi import _Global
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.admitted_family import admit_family, admit_family_meta, admit_family_owned
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_buffer_owner import native_buffer_view
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.host_sparse_native import host_sparse_format
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_is_str, family_isinstance, family_call, family_array,
    family_field, family_snapshot_min,
)
from bindings.classical_labels import (
    labels_cast, labels_helper,
    classical_encode_labels, labels_shape_of, labels_reshape, classical_decode_labels, classical_accuracy,
    classical_encode_words,
)
from bindings.admitted_input import admit_c_view, admitted_dtype
from core.host_finite_scan import host_all_finite_f32, host_all_finite_f64
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


# Fitted state is admitted once, not on every query (lane perf-neighbors-
# resident, report item 4: the per-query re-admission of the fitted index and
# its scalars came with the native query owners, e230964ecf and later). The
# stamp is the exact tuple of the admitted objects behind a per-process native
# token: a query whose fitted slots still hold the very same objects (`is`)
# skips their admission; any replaced slot, a refit (a fresh resident holder),
# a pickle or a deepcopy (a different token) admits everything again. The
# stamp holds its objects, so an identity is never reused while it is stored.
struct _AdmittedToken(Defaultable, Movable):
    var token: PythonObject

    def __init__(out self):
        self.token = PythonObject(None)


comptime _ADMITTED_TOKEN = _Global[StorageType=_AdmittedToken, name="MojolearnNeighborAdmittedTokenV1",
    init_fn=_AdmittedToken.__init__]


def _admitted_token() raises -> PythonObject:
    var state = _ADMITTED_TOKEN.get_or_create_ptr()
    if state[].token is None:
        state[].token = py_call(_b().object)
    return state[].token


def knn_state_values(fields: PythonObject, names: PythonObject) raises -> PythonObject:
    """The admission stamp of `names` in `fields`: the token, then each slot's
    current object (None where it is absent)."""
    var values = Python.list(_admitted_token())
    for i in range(len(names)):  # small-loop(fixed fitted-state slot names)
        _ = py_call(values.append, py_get(fields, names[i]))
    return py_call(_b().tuple, values)


def knn_state_unchanged(stored: PythonObject, values: PythonObject) raises -> Bool:
    """True when `stored` is a stamp of exactly the objects in `values`."""
    if stored is None or py_type(stored) is not _b().tuple or len(stored) != len(values):
        return False
    for i in range(len(values)):  # small-loop(fixed fitted-state identities)
        if stored[i] is not values[i]:
            return False
    return True


def _knn_list_items_same(value: PythonObject, snapshot: PythonObject) raises -> Bool:
    """True when the exact list `value` holds, slot by slot, the very objects
    of the tuple `snapshot` (pointer comparison, no Python temporaries)."""
    var b = _b()
    if py_type(value) is not b.list or py_type(snapshot) is not b.tuple:
        return False
    var count = len(value)
    if count != len(snapshot):
        return False
    var table = cpython_table()
    for i in range(count):  # driver-loop: over retained class-list item identities
        var live = PythonObject(from_borrowed=table[].PyList_GetItem(value._obj_ptr, i))
        var kept = PythonObject(from_borrowed=table[].PyTuple_GetItem(snapshot._obj_ptr, i))
        if live is not kept:
            return False
    return True


def _knn_scalar_item(b: PythonObject, item: PythonObject) raises -> Bool:
    var cls = py_type(item)
    return cls is b.str or cls is b.int or cls is b.float or cls is b.bool


def _knn_lists_record(fields: PythonObject, lists: PythonObject) raises -> PythonObject:
    """The admitted class lists' record: the token, then (list, tuple of its
    items, tuple of nested item tuples) per name, or None when a list holds
    anything but exact immutable scalars (str, int, float, bool) or exact
    lists of them, whose in-place edits identity cannot see.

    p3-neighbors-cluster: a classifier's `_classes_list` is a list of the
    per-output class lists, so the record now reaches one level down (the
    nested tuple holds, per item, None for a scalar or the tuple of the inner
    list's items); before, it was never stored for a classifier and every
    predict re-admitted the class lists."""
    var b = _b()
    var record = Python.list(_admitted_token())
    for i in range(len(lists)):  # small-loop(fixed neighbor class owners)
        var value = py_get(fields, lists[i])
        if value is None:
            _ = py_call(record.append, PythonObject(None))
            _ = py_call(record.append, PythonObject(None))
            _ = py_call(record.append, PythonObject(None))
            continue
        if py_type(value) is not b.list:
            return PythonObject(None)
        var snapshot = py_call(b.tuple, value)
        var nested = Python.list()
        for j in range(len(snapshot)):  # driver-loop: over class-list items, once per recorded fit state
            var item = snapshot[j]
            if _knn_scalar_item(b, item):
                _ = py_call(nested.append, PythonObject(None))
                continue
            if py_type(item) is not b.list:
                return PythonObject(None)
            var inner = py_call(b.tuple, item)
            for t in range(len(inner)):  # driver-loop: over one output's class items, once per recorded fit state
                if not _knn_scalar_item(b, inner[t]):
                    return PythonObject(None)
            _ = py_call(nested.append, inner)
        _ = py_call(record.append, value)
        _ = py_call(record.append, snapshot)
        _ = py_call(record.append, py_call(b.tuple, nested))
    return py_call(b.tuple, record)


def _knn_lists_unchanged(fields: PythonObject, lists: PythonObject, stored: PythonObject) raises -> Bool:
    """True when `stored` is a record of the class lists `fields` holds now:
    the same list objects holding the same item objects (one level of nested
    lists included)."""
    if stored is None or py_type(stored) is not _b().tuple or len(stored) != 1 + 3 * len(lists):
        return False
    if stored[0] is not _admitted_token():
        return False
    for i in range(len(lists)):  # small-loop(fixed neighbor class owners)
        var value = py_get(fields, lists[i])
        if stored[1 + 3 * i] is not value:
            return False
        if value is None:
            continue
        var snapshot = stored[2 + 3 * i]
        if not _knn_list_items_same(value, snapshot):
            return False
        var nested = stored[3 + 3 * i]
        for j in range(len(nested)):  # driver-loop: over nested class lists (pointer compares only)
            var inner = nested[j]
            if inner is None:
                continue
            if not _knn_list_items_same(snapshot[j], inner):
                return False
    return True


def _admit_knn_state(model: PythonObject, context: PythonObject) raises:
    classical_check_layout(model, context)
    var fields = py_fields(model)
    var holder = PythonObject(None)
    if py_get(fields, "_resident_holder") is not None:
        holder = _holder(fields)
    # Class lists are mutable containers. Perf (perf3-nbr-msel-trees): they
    # are admitted once per state; the holder keeps the admitted list objects
    # and a tuple of their items, and a later call compares item identities
    # natively (O(classes) pointer reads, no per-item admission). Lists of
    # anything but exact str/int/float/bool are admitted on every call.
    var lists = Python.tuple("_classes_list", "_label_names")
    if holder is None or not _knn_lists_unchanged(fields, lists, py_get(holder, "admitted_lists")):
        for i in range(len(lists)):  # small-loop(fixed neighbor class owners)
            var value = py_get(fields, lists[i])
            if value is not None:
                classical_admit_value(value, context)
        if holder is not None:
            py_setitem(holder, "admitted_lists", _knn_lists_record(fields, lists))
    var scalars = Python.tuple("algorithm", "metric", "p", "weights", "n_neighbors", "query_tile",
        "n_features_in_", "n_samples_fit_", "outputs_2d_")
    var buffers = Python.tuple("_index", "_y_cols")
    var stamp = knn_state_values(fields, scalars + buffers)
    if holder is not None and knn_state_unchanged(py_get(holder, "admitted_predict"), stamp):
        return
    for i in range(len(scalars)):  # small-loop(fixed neighbor parameter and fitted scalar schema)
        var value = py_get(fields, scalars[i])
        if value is not None:
            host_inference_scalar(value, context, True)
    for i in range(len(buffers)):  # small-loop(fixed neighbor buffer owners)
        var value = py_get(fields, buffers[i])
        if value is not None:
            classical_admit_value(value, context)
    if holder is not None:
        py_setitem(holder, "admitted_predict", stamp)


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
    # Both real extensions and the import-connected query export tables are
    # exact modules. Read their dictionaries directly: a missing entry never
    # invokes module __getattr__, and replaced Python callables are refused.
    var b = _b()
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: neighbor execution requires its connected native module")
    var entry = py_get(py_fields(binding), name)
    if entry is not None and (py_type(entry) is not py_type(b.len) or py_type(entry.__self__) is not py_type(b)):
        raise Error("TypeError: neighbor execution requires compiled entry " + name)
    return entry


def _holder(fields: PythonObject) raises -> PythonObject:
    """The instance's `_resident_holder` dict, created in `__dict__` when it
    is missing (the `_resident` / query-report property setters)."""
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    if holder is None:
        holder = Python.dict()
        py_setitem(fields, "_resident_holder", holder)
    var str_type = _b().str
    if py_type(holder) is not _b().dict:
        raise Error("TypeError: neighbor resident holder must be an exact dictionary")
    # Perf (perf3-nbr-msel-trees): the key check walks the dict natively
    # (PyDict_Next, borrowed keys) instead of building a key tuple per call.
    ref cpy = Python().cpython()
    var pos = Int(0)
    var key = PyObjectPtr()
    var item = PyObjectPtr()
    # p3-neighbors-cluster: the symbol is resolved once per call, not once per
    # holder key (the py_lists.mojo form: a typed function value, then calls).
    var dict_next = cpy.lib.get_function[Int32]("PyDict_Next")
    for _ in range(len(holder)):  # driver-loop: resident metadata keys before dictionary comparisons
        if dict_next(holder._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)) == 0:
            break
        if py_type(PythonObject(from_borrowed=key)) is not str_type:
            raise Error("TypeError: neighbor resident keys must be exact strings")
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
    if holder is None:
        return
    holder = _holder(fields)
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    if py_type(held) is not b.tuple or len(held) != 3 or py_type(held[1]) is not b.int:
        raise Error("TypeError: neighbor resident state must retain its native handle and module")
    var release = knn_door(held[2], "knn_index_release")
    py_setitem(holder, "index", PythonObject(None))
    try:
        if release is not None:
            _ = py_call(release, held[1])
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
    if holder is not None:
        cached = py_get(_holder(fields), PythonObject("index"), PythonObject(None))
    if cached is not None:
        if py_type(cached) is not b.tuple or len(cached) != 3 or py_type(cached[1]) is not b.int:
            raise Error("TypeError: neighbor resident state must retain its native handle and module")
        var previous = cached[0]
        if (py_type(previous) is not b.tuple or len(previous) != 2 or py_type(previous[0]) is not b.int
                or py_type(previous[1]) is not b.tuple or len(previous[1]) != 2
                or py_type(previous[1][0]) is not b.int or py_type(previous[1][1]) is not b.int):
            raise Error("TypeError: neighbor resident key must contain native address and shape words")
        _ = knn_door(cached[2], "knn_index_release")
    # Hazard H4 (lane pass2-cluster-nbr): `identical` and `fast` keep
    # separate handle registries, so a handle is reused only by the binding
    # that minted it; another binding releases it through its minter below
    # and prepares its own.
    if cached is not None and cached[2] is binding and _eq(cached[0], key):
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
        elif (cpython_table()[].PyErr_ExceptionMatches(type_error._obj_ptr) != 0
                or cpython_table()[].PyErr_ExceptionMatches(value_error._obj_ptr) != 0):
            cpython_table()[].PyErr_Clear()
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
    # p3-neighbors-cluster: the resolved triple is kept in the resident holder
    # (never `__dict__`) with the class and the metric, p and weights objects
    # it was resolved from; while they are the same immutable scalars
    # (exact str/int/float/bool/None), a query reuses it instead of resolving
    # the metric again. Any other value (a callable, a container) resolves
    # every call, as before.
    var b = _b()
    var metric = family_field(fields, "metric")
    var p = family_field(fields, "p")
    var weights_value = py_get(fields, "weights")
    var cls = py_type(model)
    var holder = py_get(fields, "_resident_holder")
    var cacheable = (holder is not None and py_type(holder) is b.dict
                     and _knn_immutable_word(b, metric) and _knn_immutable_word(b, p)
                     and _knn_immutable_word(b, weights_value))
    if cacheable:
        var held = py_get(holder, "dist_params")
        if (held is not None and py_type(held) is b.tuple and len(held) == 7 and held[0] is cls
                and held[1] is metric and held[2] is p and held[3] is weights_value):
            return Python.list(held[4], held[5], held[6])
    var cls_name = _class_name(model)
    var resolved = knn_resolve_metric(cls_name, metric, p, context)
    var weights = 0
    if _is_knn(model, context):
        weights = knn_resolve_weights(cls_name, family_field(fields, "weights"))
    if cacheable:
        py_setitem(holder, "dist_params", Python.tuple(cls, metric, p, weights_value,
                                                       resolved[0], resolved[1], PythonObject(weights)))
    return Python.list(resolved[0], resolved[1], PythonObject(weights))


def _knn_immutable_word(b: PythonObject, value: PythonObject) raises -> Bool:
    if value is None:
        return True
    var cls = py_type(value)
    return cls is b.str or cls is b.int or cls is b.float or cls is b.bool


def _check_metric_input(cls_name: String, value: Int, x: ArrayMetadata) raises:
    # The same captured shape/address/owner is used for validation and the
    # subsequent call. The minimum is native, with no Python reduction door.
    if value == _DIST_JS and x.count > 0 and Float64(family_snapshot_min(x)) < 0.0:
        raise Error("mojolearn " + cls_name + ": metric='jensenshannon' needs X >= 0 "
                    + "(it takes logarithms of the coordinates)")


def _check_metric_input(cls_name: String, value: Int, x: PythonObject, context: PythonObject) raises:
    # Existing native shard workers supply their prepared array, not the
    # public operation snapshot. Keep that internal ABI while their owner
    # migrates; plain operations above consume the retained snapshot directly.
    if value == _DIST_JS:
        var snapshot = admit_family_meta(x, 2, "X", context)
        _check_metric_input(cls_name, value, snapshot)


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
    # result itself, or the caller's float32 C-order block by reference,
    # never frozen or changed; any other layout is cast into an owned Array).
    # 0.8.66 regression C3 (knn fit 8.5x on istella): 0.8.55 and
    # scikit-learn's brute NearestNeighbors keep the caller's block; the
    # zero-filled copy is gone.
    var input = admit_family_meta(x_source, 2, "X", context, True)
    var idx: PythonObject
    if input.f32_c() and not input.copied:
        idx = input.keep
    else:
        idx = admit_family_owned(input, context)
    if not _eq(family_field(fields, "algorithm"), PythonObject("rbc")):
        var resolved = knn_resolve_metric(cls_name, family_field(fields, "metric"), family_field(fields, "p"), context)
        _check_metric_input(cls_name, Int(py=resolved[0]), input)
    py_setitem(fields, "_index", idx)
    py_setitem(fields, "n_samples_fit_", PythonObject(input.shape[0]))
    py_setitem(fields, "n_features_in_", PythonObject(input.shape[1]))


def _y_shape(y: PythonObject, fields: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_shape_of(y)` with the k-NN fits' rank and row-count refusals. An
    object with neither a `shape` nor list/tuple nesting reads `(len(y),)`
    when it has a length and `()` otherwise, as `_buffer.shape_of` did."""
    classical_admit_value(y, context)
    var shape = List[Int]() if y is None else labels_shape_of(y, context)
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
    if host_sparse_format(y).byte_length() > 0:
        return False
    classical_admit_value(y, context)
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes or cls is b.dict:
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
    # The labels in their own dtype, C order (bindings/admitted_input.mojo
    # admit_c_view; lane a1b: was two labels_view reads of the same y).
    var view = admit_c_view(y, "y", context)
    var typestr = admitted_dtype(view, context)
    var n = Int(py=view.size)
    var addr = Int(py=classical_address(view))
    if typestr == "<f4" or typestr == "<f8":
        var finite = (host_all_finite_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr), n)
                      if typestr == "<f8" else
                      host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr), n))
        if n > 0 and not finite:
            return False
    var big = 0
    for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
        if Int(py=view.shape[k]) > big:
            big = Int(py=view.shape[k])
    if (typestr != "<f4" and typestr != "<f8") or n == 0 or n != big:
        return True
    var classes_store = family_array(Python.tuple(n), "<f8", context)
    var codes = family_array(Python.tuple(n), "<i4", context)
    var n_distinct = Int(py=family_call(labels_helper(context, "prepare_targets"), PythonObject(addr),
                               PythonObject(_CODE_F32 if typestr == "<f4" else _CODE_F64), PythonObject(n),
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
    var labels = classical_encode_words(y_cols, Int(py=classical_address(y_cols)) + 4 * row * n, n, "<i4", False,
                                        context)
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
        var encoded = classical_encode_labels(y, context)
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
    var view = admit_c_view(y, "y", context)
    if Int(py=view.ndim) != len(shape):
        raise Error("mojolearn: y must be " + String(len(shape)) + "-D, got " + String(Int(py=view.ndim)) + "-D")
    var count = Int(py=view.size)
    var y_typestr = admitted_dtype(view, context)
    var wide = PythonObject(None)
    var wide_addr = Int(py=classical_address(view))
    if y_typestr != "<i8":
        wide = family_array(Python.tuple(count), "<i8", context)
        wide_addr = Int(py=classical_address(wide))
        labels_cast(Int(py=classical_address(view)), y_typestr, wide_addr, "<i8", count, context)
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
        py_setitem(fields, "_classes_list", Python.list(classical_encode_labels(y32, context).classes))
        return
    var yc = family_array(Python.tuple(n_out, n), "<i4", context)
    # admit_c_view hands the labels in C order (an F-ordered y is one
    # C-order copy), so the narrowing is a C-level copy (exact after the
    # range check) and the transpose a device op (`xn_p2m_transpose_i`; the
    # host binding's same op on a CPU-only install). Integers: the same
    # column words as the former storage-order narrowing of an F-ordered y.
    var y32 = family_array(Python.tuple(n, n_out), "<i4", context)
    labels_cast(wide_addr, "<i8", Int(py=classical_address(y32)), "<i4", count, context)
    _ = wide  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    if n:
        xn_execute(xn, "p2m_transpose_i", Python.list(classical_address(y32), classical_address(yc)),
                   Python.list(PythonObject(n), PythonObject(n_out)), Python.list())
        _ = y32  # keep the Python object alive through the native call above
        _ = yc  # keep the Python object alive through the native call above
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
    var target = admit_family_meta(args[2], -1, "y", context)
    var shape = target.shape.copy()
    if len(shape) != 1 and len(shape) != 2:
        raise Error("mojolearn: y must be 1-D or 2-D, got " + String(len(shape)) + "-D")
    var n_fit = Int(py=family_field(fields, "n_samples_fit_"))
    if shape[0] != n_fit:
        raise Error("mojolearn: y has " + String(shape[0]) + " rows, X has " + String(n_fit))
    var ya = target.keep
    var n = shape[0]
    var n_out = 1 if len(shape) == 1 else shape[1]
    if n_out == 1:
        # One column IS one contiguous row of the transposed layout.
        py_setitem(fields, "_y_cols", labels_reshape(admit_family_owned(target, context), Python.tuple(1, n), context))
    else:
        # The transpose on the device (`xn_p2m_transpose`; the host binding's
        # same op on a CPU-only install).
        var yc = family_array(Python.tuple(n_out, n), "<f4", context)
        if n:
            xn_execute(xn, "p2m_transpose", Python.list(PythonObject(target.source.addr), classical_address(yc)),
                       Python.list(PythonObject(n), PythonObject(n_out)), Python.list())
            _ = ya  # keep the Python object alive through the native call above
            _ = yc  # keep the Python object alive through the native call above
        py_setitem(fields, "_y_cols", yc)
    py_setitem(fields, "outputs_2d_", PythonObject(len(shape) == 2 and shape[1] != 1))
    _ = target


# ------------------------------------------------------------------ predict


def _queries(model: PythonObject, x_source: PythonObject, mut dist: PythonObject,
             context: PythonObject) raises -> ArrayMetadata:
    """The prediction preamble: fitted check, float32 queries, the metric
    input check (with `_dist_params`' refusals), width and k checks. `dist`
    receives the resolved `_dist_params`, read once per call."""
    var fields = py_fields(model)
    if (py_get(fields, PythonObject("_index"), PythonObject(None)) is None
            or py_get(fields, PythonObject("_y_cols"), PythonObject(None)) is None):
        raise Error("mojolearn: call fit before predict")
    var q = admit_family_meta(x_source, 2, "X", context, True)
    dist = _dist_params(model, context)
    _check_metric_input(_class_name(model), Int(py=dist[0]), q)
    var features = family_field(fields, "n_features_in_")
    if q.shape[1] != Int(py=features):
        raise Error("mojolearn: X has " + String(q.shape[1]) + " features, index has " + family_str(features))
    var k = family_field(fields, "n_neighbors")
    var n_fit = family_field(fields, "n_samples_fit_")
    if _lt(k, PythonObject(1)) or _gt(k, n_fit):
        raise Error("mojolearn: n_neighbors must be in [1, " + family_str(n_fit) + "], got " + family_str(k))
    return q^


def _classify(model: PythonObject, binding: PythonObject, x_source: PythonObject, want_proba: Bool,
              context: PythonObject) raises -> PythonObject:
    """`KNeighborsClassifier._predict(X, want_proba)`: `(labels, proba,
    n_classes)`; the search and the vote are one compiled call."""
    var b = _b()
    var fields = py_fields(model)
    var dist = PythonObject(None)
    var query = _queries(model, x_source, dist, context)
    var q = query.keep
    var k = family_field(fields, "n_neighbors")
    var nq = query.shape[0]
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
    var used: PythonObject
    if handle is not None:
        # The handle rides at the front of `params`: a binding takes at most
        # eight arguments and the classifier uses all eight.
        var front = Python.list(handle)
        _ = py_call(front.extend, params)
        used = family_call(resident, classical_address(idx), PythonObject(query.source.addr), classical_address(y_cols),
                           classical_address(labels), classical_address(proba), classical_address(uniq), front, dist)
    else:
        used = family_call(knn_door(binding, "knn_classify"), classical_address(idx),
                           PythonObject(query.source.addr), classical_address(y_cols), classical_address(labels),
                           classical_address(proba), classical_address(uniq), params, dist)
    _ = idx  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = query  # retain snapshot owners through execution
    _ = y_cols  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = proba  # keep the Python object alive through the native call above
    _ = uniq  # keep the Python object alive through the native call above
    _report(fields, "used_query_tile_", used)
    # p3-neighbors-cluster: `uniq` is the kernel's class set of the fitted
    # `_y_cols`, the same for every query, so POLICY 7's check runs once per
    # fitted state: skipped while `_y_cols` and `_classes_list` are the objects
    # it passed for and the class lists' admission record still holds.
    var class_holder = py_get(fields, "_resident_holder")
    if class_holder is not None and py_type(class_holder) is b.dict:
        var checked = py_get(class_holder, "class_set_checked")
        if (checked is not None and py_type(checked) is b.tuple and len(checked) == 2
                and checked[0] is y_cols and checked[1] is classes_list
                and _knn_lists_unchanged(fields, Python.tuple("_classes_list", "_label_names"),
                                         py_get(class_holder, "admitted_lists"))):
            return Python.tuple(labels, proba, n_classes)
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
    if class_holder is not None and py_type(class_holder) is b.dict:
        py_setitem(class_holder, "class_set_checked", Python.tuple(y_cols, classes_list))
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
    var dist = PythonObject(None)
    var query = _queries(model, x_source, dist, context)
    var q = query.keep
    var k = family_field(fields, "n_neighbors")
    var nq = query.shape[0]
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
    var used: PythonObject
    if handle is not None:
        used = family_call(resident, handle, classical_address(idx), PythonObject(query.source.addr), classical_address(y_cols),
                           classical_address(out), params, dist)
    else:
        used = family_call(knn_door(binding, "knn_regress"), classical_address(idx),
                           PythonObject(query.source.addr), classical_address(y_cols), classical_address(out), params, dist)
    _ = idx  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = query  # retain snapshot owners through execution
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
    var truth = admit_family(args[2], -1, "y", context).array
    var prediction = admit_family(_reg_predict(model, args[0], args[1], context), -1, "prediction", context).array
    var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
    var metrics = args[4] if args[4] is not None else estimator_binding(model, "_mojolearn_metrics", context)
    return mcls_cluster_operation("family_mcls_r2", Python.tuple(metrics, truth, prediction, args[3],
                                  PythonObject("uniform_average"), PythonObject(True), mode), context)


def _neighbors_knn_operation(model: PythonObject, operation: String, var args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    _admit_knn_state(model, context)
    var data_start = 0 if operation == "family_xn_knn_nn_fit" else 1
    for i in range(data_start, min(len(args), 4)):  # small-loop(public data arguments; metrics binding is not data)
        if host_sparse_format(args[i]).byte_length() == 0:
            classical_admit_value(args[i], context)
    if operation == "family_xn_knn_clf_fit" or operation == "family_xn_knn_reg_fit":
        args = xn_admit_arguments(estimator_arguments(model, "_mojolearn_x_neighbors", args, context), context)
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


def neighbors_knn_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_knn_clf_classes":
        # The public `classes_` property: sorted unique labels seen at fit
        # (the saved label names when fit kept them), a list per column for a
        # 2-D `y`. Reads the fit's own stored owners only.
        classical_connect_numpy(context)
        var fields = _fields(model)
        if py_get(fields, "_y_cols") is None:
            raise Error("AttributeError: classes_ is set by fit")
        var names = py_get(fields, "_label_names")
        if names is not None:
            return names
        if not py_contains(fields, "_classes_list") or not py_contains(fields, "outputs_2d_"):
            raise Error("AttributeError: classes_ is set by fit")
        var classes = fields["_classes_list"]
        var two_d = fields["outputs_2d_"]
        host_inference_scalar(two_d, context, True)
        if Bool(py=two_d):
            return classes
        if py_type(classes) is not _b().list:
            raise Error("TypeError: neighbor classes must be the native list fit stored")
        return classes[0]
    if operation == "family_xn_knn_getstate":
        var fields = _fields(model)
        var state = py_call(native_loaded_module("builtins").dict, fields)
        _ = py_call(state.pop, "_resident", PythonObject(None))
        _ = py_call(state.pop, "_resident_holder", PythonObject(None))
        return state
    if operation == "family_xn_knn_setstate":
        var b = _b()
        var state = args[0]
        if py_type(state) is not b.dict:
            raise Error("TypeError: neighbor pickle state must be an exact dictionary")
        var keys = py_call(b.tuple, state.keys())
        for i in range(len(keys)):  # driver-loop: state keys before native dictionary update
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: neighbor pickle state keys must be exact strings")
        var fields = _fields(model)
        knn_release(model)
        _ = py_call(fields.update, state)
        py_setitem(fields, "_resident_holder", Python.dict())
        return PythonObject(None)
    var call = neighbors_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_knn_operation(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return neighbors_public_finish(model, operation, result, call, context)
