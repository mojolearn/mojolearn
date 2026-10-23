# SPDX-License-Identifier: Apache-2.0
# IsolationForest's public operations in one native call each
# (python/mojolearn/_iforest_impl.py): `fit`, `score_samples`,
# `decision_function`, `predict` and `fit_predict`. This is the Python
# orchestration those methods ran, step for step: `_validate_params` with its
# wording, the float32 X, the device finite scan probe and its refusal, the
# resident forest token (IDN_IF_RESIDENT), the `iforest_run` launch with its
# sixteen (or seventeen) params, and `offset_` / `max_samples_` read back.
# The kernel and its arguments are the ones the Python path launched, so the
# forest, the scores and the labels keep their bits.
#
# The resident forest's release (formerly `weakref.finalize` plus a Python
# `__del__`) is a TreesResident lifetime held in the instance dictionary
# (`_if_lifetime`): the binding's `iforest_resident_release` runs when the
# estimator drops it (a refit, a new token, the estimator's end).
#
# Round trips removed against the Python path: one native call replaces each
# method's Python sequence (validation, conversion, probes, buffers, launch,
# read-back); `fit_predict` is ONE `iforest_run` (fit a fresh forest and
# predict X with it) instead of the fit's one-row run and a second run for
# predict, and X is converted and handed over once.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.native_buffer_owner import native_storage_allocate
from bindings.trees_public_inputs import (
    TreesMatrix, trees_builtins, trees_table, trees_entry, trees_vector, trees_address, trees_f32,
    trees_all_finite, trees_call,
)
from bindings.trees_lifetime import trees_resident_new

comptime _WANT_SCORE_SAMPLES = 0
comptime _WANT_DECISION_FUNCTION = 1
comptime _WANT_PREDICT = 2
comptime _PREFIX = "mojolearn IsolationForest: "
comptime _NONFINITE = "mojolearn IsolationForest: X contains NaN or infinity"


def _repr(value: PythonObject) raises -> String:
    return String(py_call(trees_builtins().repr, value))


def _is_bool(value: PythonObject) raises -> Bool:
    var t = py_type(value)
    return t is trees_builtins().bool or String(t.__name__) == "bool_"


def _field(fields: PythonObject, model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`self.<name>` of a stored constructor parameter."""
    var value = py_get(fields, PythonObject(name), context["trees_missing"])
    if value is context["trees_missing"]:
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return value


def iforest_validate(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """`IsolationForest._validate_params()`: the parameter checks, in order,
    with their wording; each resolved knob is stored as it is checked."""
    var b = trees_builtins()
    var integral = context["trees_integral_type"]
    var n_estimators = py_call_typed(b.int, _field(fields, model, "n_estimators", context))
    if Bool(py=n_estimators < PythonObject(1)):
        raise Error(_PREFIX + "n_estimators must be at least 1, got " + _repr(n_estimators))
    var max_samples = _field(fields, model, "max_samples", context)
    if Bool(py=py_call(b.isinstance, max_samples, b.str)):
        if String(py_call(max_samples.lower)) != "auto":
            raise Error(_PREFIX + "max_samples=" + _repr(max_samples) + " is not a name; it is 'auto', a positive int, "
                        + "or a float in (0.0, 1.0]")
        py_setitem(fields, "_max_samples_mode", PythonObject(0))
        py_setitem(fields, "_max_samples_int", PythonObject(256))
        py_setitem(fields, "_max_samples_frac", PythonObject(1.0))
    elif _is_bool(max_samples):
        raise Error(_PREFIX + "max_samples must be 'auto', an int, or a float, not a bool")
    elif Bool(py=py_call(b.isinstance, max_samples, integral)):
        var count = py_call_typed(b.int, max_samples)
        if Bool(py=count <= PythonObject(0)):
            raise Error(_PREFIX + "max_samples must be a positive integer, got " + _repr(max_samples))
        py_setitem(fields, "_max_samples_mode", PythonObject(1))
        py_setitem(fields, "_max_samples_int", count)
        py_setitem(fields, "_max_samples_frac", PythonObject(1.0))
    else:
        var v = Float64(py=py_call_typed(b.float, max_samples))
        if not (v > 0.0 and v <= 1.0):
            raise Error(_PREFIX + "float max_samples must be in (0.0, 1.0], got " + _repr(max_samples))
        py_setitem(fields, "_max_samples_mode", PythonObject(2))
        py_setitem(fields, "_max_samples_int", PythonObject(256))
        py_setitem(fields, "_max_samples_frac", PythonObject(v))
    var max_depth = _field(fields, model, "max_depth", context)
    if max_depth is None:
        py_setitem(fields, "_max_depth", PythonObject(-1))
    else:
        var depth = py_call_typed(b.int, max_depth)
        py_setitem(fields, "_max_depth", depth)
        if Bool(py=depth < PythonObject(1)):
            raise Error(_PREFIX + "max_depth is None (their auto) or a positive int, got " + _repr(max_depth))
    var max_features = _field(fields, model, "max_features", context)
    if _is_bool(max_features):
        raise Error(_PREFIX + "max_features must be an int or a float, not a bool")
    var features_error = String(_PREFIX + "max_features must be an int in [1, n_features] or a float in (0.0, 1.0], got ")
    if Bool(py=py_call(b.isinstance, max_features, integral)):
        var count = py_call_typed(b.int, max_features)
        if Bool(py=count < PythonObject(1)):
            raise Error(features_error + _repr(max_features))
        py_setitem(fields, "_max_features_mode", PythonObject(1))
        py_setitem(fields, "_max_features_int", count)
        py_setitem(fields, "_max_features_frac", PythonObject(1.0))
    else:
        var v = Float64(py=py_call_typed(b.float, max_features))
        if not (v > 0.0 and v <= 1.0):
            raise Error(features_error + _repr(max_features))
        py_setitem(fields, "_max_features_mode", PythonObject(0))
        py_setitem(fields, "_max_features_int", PythonObject(0))
        py_setitem(fields, "_max_features_frac", PythonObject(v))
    var contamination = _field(fields, model, "contamination", context)
    var contamination_error = String(_PREFIX + "contamination must be 'auto' or a float in the range (0.0, 0.5], got ")
    if Bool(py=py_call(b.isinstance, contamination, b.str)):
        if String(py_call(contamination.lower)) != "auto":
            raise Error(contamination_error + _repr(contamination))
        py_setitem(fields, "_contamination_auto", PythonObject(True))
        py_setitem(fields, "_contamination", PythonObject(0.0))
    else:
        var v = Float64(py=py_call_typed(b.float, contamination))
        if not (v > 0.0 and v <= 0.5):
            raise Error(contamination_error + _repr(contamination))
        py_setitem(fields, "_contamination_auto", PythonObject(False))
        py_setitem(fields, "_contamination", PythonObject(v))
    var random_state = _field(fields, model, "random_state", context)
    var seed = PythonObject(0)
    if random_state is not None:
        seed = py_call_typed(b.int, random_state)
        if Bool(py=seed < PythonObject(0)) or Bool(py=seed > PythonObject(4294967295)):
            raise Error(_PREFIX + "expected 0 <= random_state <= 2**32 - 1, got " + _repr(random_state))
    if Bool(py=py_call(b.bool, _field(fields, model, "warm_start", context))):
        raise Error("NotImplementedError: " + _PREFIX + "warm_start=True is not supported "
                    + "(cuML raises UnsupportedOnGPU for it too)")
    if _field(fields, model, "n_jobs", context) is not None:
        raise Error("NotImplementedError: " + _PREFIX + "n_jobs is refused; it is "
                    + "scikit-learn's CPU thread count and there is no CPU path in "
                    + "this library to spread")
    if Bool(py=py_call(b.bool, _field(fields, model, "verbose", context))):
        raise Error("NotImplementedError: " + _PREFIX + "verbose is refused; this implementation "
                    + "prints no log lines, so accepting it would be accepting-"
                    + "and-ignoring")
    if _field(fields, model, "output_type", context) is not None:
        raise Error("NotImplementedError: " + _PREFIX + "output_type is a cuML-internal "
                    + "array-type selector; this package returns mojolearn Arrays")
    py_setitem(fields, "_n_estimators", n_estimators)
    py_setitem(fields, "_bootstrap", PythonObject(Bool(py=py_call(b.bool, _field(fields, model, "bootstrap", context)))))
    py_setitem(fields, "_seed", seed)


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    """The tier `NumericModeMixin._bind` resolves: `numeric_mode`, or the
    process default."""
    var b = trees_builtins()
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var clean = String(py_call(py_call(mode.strip).lower))
    if clean != "fast" and clean != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    return clean


def _params(model: PythonObject, fields: PythonObject, n_train: Int, n_features: Int, n_query: Int, want: Int,
            context: PythonObject) raises -> PythonObject:
    """`IsolationForest._params`: the sixteen `iforest_run` slots."""
    if not py_contains(fields, "_seed"):  # an instance assembled without fit or load
        iforest_validate(model, fields, context)
    var b = trees_builtins()
    return Python.list(
        PythonObject(n_train), PythonObject(n_features), PythonObject(n_query), fields["_n_estimators"],
        fields["_max_samples_mode"], fields["_max_samples_int"], fields["_max_samples_frac"], fields["_max_depth"],
        fields["_max_features_mode"], fields["_max_features_int"], fields["_max_features_frac"],
        PythonObject(1 if Bool(py=py_call(b.bool, fields["_bootstrap"])) else 0), fields["_seed"],
        PythonObject(1 if Bool(py=py_call(b.bool, fields["_contamination_auto"])) else 0), fields["_contamination"],
        PythonObject(want),
    )


def _release(fields: PythonObject) raises:
    """`_release_resident()`: drop the resident forest's token and its
    lifetime (which releases it in the binding)."""
    _ = py_call(fields.pop, "_if_token", PythonObject(0))
    _ = py_call(fields.pop, "_if_lifetime", PythonObject(None))


def _run(model: PythonObject, fields: PythonObject, x: TreesMatrix, train: PythonObject, want: Int, fresh: Bool,
         context: PythonObject) raises -> PythonObject:
    """`IsolationForest._run` on a converted query: one fit-and-score
    `iforest_run`, the resident token kept or renewed, `offset_` and
    `max_samples_` read back. Returns the float32 scores or int32 labels."""
    var n_query = x.rows
    var svm = trees_table(context, _mode(fields, context), "_mojolearn_svm")
    var values = PythonObject(None) if want == _WANT_PREDICT else trees_vector(n_query, "<f4", context)
    var labels = trees_vector(n_query, "<i4", context) if want == _WANT_PREDICT else PythonObject(None)
    var probe = trees_entry(svm, "iforest_resident")
    var resident = probe is not None and Int(py=py_call_typed(probe)) == 1
    var info = trees_vector(4 if resident else 3, "<f8", context)
    var params = _params(model, fields, Int(py=train.shape[0]), Int(py=train.shape[1]), n_query, want, context)
    var current = py_get(fields, "_if_token", PythonObject(0))
    if resident:
        _ = py_call(params.append, PythonObject(0) if fresh else py_call(trees_builtins().int, current))
    var run = py_get(svm, "iforest_run")
    if run is None:
        raise Error("ImportError: mojolearn: the svm binding has no iforest_run")
    _ = trees_call(run, Python.list(PythonObject(trees_address(train)), PythonObject(x.addr),
                                    PythonObject(0 if values is None else trees_address(values)),
                                    PythonObject(0 if labels is None else trees_address(labels)),
                                    PythonObject(trees_address(info)), params))
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(info))
    # `array.array("f", [float(info[0])])[0]`: the float32 offset as a float.
    py_setitem(fields, "offset_", PythonObject(Float64(Float32(words[0]))))
    py_setitem(fields, "max_samples_", PythonObject(Int(words[1])))
    if resident:
        var token = Int(words[3])
        if not Bool(py=PythonObject(token) == current):
            _release(fields)
            var release = py_get(svm, "iforest_resident_release")
            if release is not None:
                py_setitem(fields, "_if_lifetime", trees_resident_new(release, token))
        py_setitem(fields, "_if_token", PythonObject(token))
    _ = x.keep
    return labels if want == _WANT_PREDICT else values


def _as_array(x: TreesMatrix, context: PythonObject) raises -> PythonObject:
    """The float32 C-order X as a mojolearn Array (what `as_f32_c`
    returned): the converted copy, the caller's Array, or a zero-copy Array
    view of the caller's buffer (`Array.from_buffer`)."""
    var b = trees_builtins()
    if py_type(x.keep) is context["array_class"]:
        return x.keep
    var view = x.keep
    var array = native_storage_allocate(context["array_class"])
    py_setattr(array, "_store", view.obj)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", view)
    py_setattr(array, "_mv", py_call(py_call(view.cast, "B").cast, "f"))
    py_setattr(array, "_addr", PythonObject(x.addr))
    py_setattr(array, "_readonly", view.readonly)
    py_setattr(array, "shape", Python.tuple(x.rows, x.cols))
    py_setattr(array, "dtype", PythonObject("<f4"))
    py_setattr(array, "ndim", PythonObject(2))
    py_setattr(array, "size", PythonObject(x.rows * x.cols))
    py_setattr(array, "nbytes", PythonObject(x.rows * x.cols * 4))
    py_setattr(array, "itemsize", PythonObject(4))
    py_setattr(array, "strides", Python.tuple(x.cols * 4, 4))
    py_setattr(array, "order", PythonObject("C"))
    return array


def _copied(x: TreesMatrix, source: PythonObject) raises -> Bool:
    """`as_f32_c`'s copy flag: False when X was borrowed in place."""
    if x.keep is source:
        return False
    if py_type(x.keep) is trees_builtins().memoryview and x.keep.obj is source:
        return False
    return True


def _device_scan(fields: PythonObject, context: PythonObject) raises -> Bool:
    """A binary that scans the uploaded X on the device
    (`iforest_device_scan` or `iforest_device_finite_scan`)."""
    var svm = trees_table(context, _mode(fields, context), "_mojolearn_svm")
    var names = Python.tuple("iforest_device_scan", "iforest_device_finite_scan")
    for i in range(2):  # small-loop(names): the two device scan probes
        var probe = trees_entry(svm, String(names[i]))
        if probe is not None and Int(py=py_call_typed(probe)) == 1:
            return True
    return False


def _fit(model: PythonObject, X: PythonObject, want: Int, context: PythonObject) raises -> PythonObject:
    """`fit` (want < 0: the one-row fresh run) or `fit_predict` (one fresh
    run that predicts X)."""
    var fields = py_fields(model)
    iforest_validate(model, fields, context)
    var x = trees_f32(X, 2, "C", "X", context)
    py_setitem(fields, "input_copied_", PythonObject(_copied(x, X)))
    var device_scan = _device_scan(fields, context)
    if not device_scan and not trees_all_finite(x.addr, x.rows * x.cols, False, context):
        raise Error(_NONFINITE)
    var had = Python.dict()
    var kept = Python.tuple("_x", "n_features_in_")
    for i in range(2):  # small-loop(kept): the two fitted field names restored on a refusal
        if py_contains(fields, kept[i]):
            py_setitem(had, kept[i], fields[kept[i]])
    _release(fields)
    var train = _as_array(x, context)
    py_setitem(fields, "_x", train)
    py_setitem(fields, "n_features_in_", PythonObject(x.cols))
    # `fit` scores the training matrix's first row (the Python `x[:1]`
    # copy: the same bytes, read in place); `fit_predict` predicts X.
    var query = TreesMatrix(train, x.addr, 1 if want < 0 else x.rows, x.cols, True)
    var out = PythonObject(None)
    try:
        out = _run(model, fields, query, train, _WANT_SCORE_SAMPLES if want < 0 else want, True, context)
    except error:
        var message = String(error)
        if device_scan and ("Input X contains" in message or "does not accept non-finite" in message):
            for i in range(2):  # small-loop(kept): the two fitted field names restored on a refusal
                _ = py_call(fields.pop, kept[i], PythonObject(None))
            _ = py_call(fields.update, had)
            raise Error(_NONFINITE)
        raise error^
    return model if want < 0 else out


def _score(model: PythonObject, X: PythonObject, want: Int, context: PythonObject) raises -> PythonObject:
    """`score_samples` / `decision_function` / `predict`: `_run(X, want)`."""
    var fields = py_fields(model)
    if not py_contains(fields, "_x"):
        raise Error(_PREFIX + "call fit() first")
    var x = trees_f32(X, 2, "C", "X", context)
    var fitted = Int(py=py_call(trees_builtins().int, fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error(_PREFIX + "X has " + String(x.cols) + " features, fit saw " + String(fitted))
    return _run(model, fields, x, fields["_x"], want, False, context)


def iforest_operation(model: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    if operation == "trees_iforest_fit":
        return _fit(model, args[0], -1, context)
    if operation == "trees_iforest_fit_predict":
        return _fit(model, args[0], _WANT_PREDICT, context)
    if operation == "trees_iforest_score_samples":
        return _score(model, args[0], _WANT_SCORE_SAMPLES, context)
    if operation == "trees_iforest_decision_function":
        return _score(model, args[0], _WANT_DECISION_FUNCTION, context)
    if operation == "trees_iforest_predict":
        return _score(model, args[0], _WANT_PREDICT, context)
    raise Error("unknown IsolationForest operation " + operation)
