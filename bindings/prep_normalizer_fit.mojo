# SPDX-License-Identifier: Apache-2.0
# Native Normalizer fit/fit_transform envelope, including the former _mode
# validation and fitting-depth wrapper. The optional target is validated but
# never used in normalization, and fit_params remain intentionally ignored.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.mode_metadata_native import _fields, _truth
from bindings.prep_normalizer_lifecycle import _setting, _class_setting, _tags, _native_entries, _normalizer_norm, _normalizer_apply
from bindings.prep_fit_error import prep_fit_error
from bindings.parallel_public_result import parallel_public_result
from providers.python.prep_transform_pyapi import prep_transform_snapshot, prep_transform_values
from providers.python.prep_target_pyapi import prep_ignored_target


def _target_tags(model: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var tags = _class_setting(model, "_sklearn_target_tags", Python.dict())
    if py_type(tags) is not b.dict:
        raise Error("TypeError: native Normalizer target tags require an ordinary dictionary")
    var keys = py_call(b.list, tags)
    for i in range(len(keys)):  # driver-loop: admit static target names before lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native Normalizer target tags require exact string keys")
    return py_call(b.dict, tags)


def _process_mode(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = context["backend_fields"]
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native Normalizer mode settings require an ordinary dictionary")
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: native namespace keys before default-mode lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native Normalizer mode keys require exact strings")
    return fields["_DEFAULT_MODE"]


def _publication(state: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, state)
    for i in range(len(keys)):  # driver-loop: refuse concurrent destination-key hooks before publication
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native Normalizer state keys must remain exact strings")


def _finish_depth(state: PythonObject, remaining: PythonObject) raises:
    # remaining comes from exact native bool/int/float arithmetic before execution.
    # Reuse the snapshot, not a mutable model setting reread after a kernel.
    _publication(state)
    if Bool(py=remaining > PythonObject(0)):
        py_setitem(state, "_mojolearn_fitting", remaining)
    else:
        _ = py_call(state.pop, "_mojolearn_fitting", PythonObject(None))


def normalizer_fit(model: PythonObject, source: PythonObject, target: PythonObject,
                   project: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var state = _fields(model)
    var fields = py_call(b.dict, state)
    var mode = _process_mode(context)
    var missing = Python.list()
    var norm = py_get(fields, "norm", missing)
    var previous = py_get(fields, "_mojolearn_fitting", 0)
    var nested = _truth(previous)
    var name = String(py_type(model).__name__)
    var tags = _tags(model, fields)
    var targets = _target_tags(model)
    var validate = not nested and _truth(py_get(tags, "two_d_array", True)) and not _truth(py_get(tags, "one_d_array", False))
    var scan = not _truth(py_get(tags, "allow_nan", False))
    var sparse = _truth(py_get(tags, "sparse", False))
    # _sk_validate._check_y's required default follows _estimator_type.
    # Only exact strings are compared natively; other values would run
    # Python equality hooks.
    var estimator_type = _setting(model, fields, "_estimator_type", PythonObject(None))
    if estimator_type is not None and py_type(estimator_type) is not b.str:
        raise Error("TypeError: native Normalizer fit requires an exact string estimator type")
    var classifier = estimator_type is not None and String(estimator_type) == "classifier"
    var regressor = estimator_type is not None and String(estimator_type) == "regressor"
    var required = _truth(py_get(targets, "required", PythonObject(classifier or regressor)))
    var multi = _truth(py_get(targets, "multi_output", False))
    # Hooks are captured/admitted before any input conversion or native entry
    # can release the GIL. This operation has no dynamic Python metadata work.
    if not nested and scan and _setting(model, fields, "_mojolearn_allows_nan", PythonObject(None)) is not None:
        raise Error("TypeError: native Normalizer validation requires static input tags; Python allow-NaN hooks cannot execute")
    if not nested and _setting(model, fields, "_sklearn_instance_target_tags", PythonObject(None)) is not None:
        raise Error("TypeError: native Normalizer validation requires static target tags; Python target-tag hooks cannot execute")
    if _setting(model, fields, "_mojolearn_after_fit", PythonObject(None)) is not None:
        raise Error("TypeError: native Normalizer fit cannot invoke Python after-fit hooks")
    var snapshot = prep_transform_snapshot(source, name, validate, scan, context, sparse, not nested)
    # _check_X's object/string buffers have no numeric Check.info. Sparse
    # descriptions retain their length even when static rank tags opt out.
    var numeric = (snapshot.dtype_number >= 0 and snapshot.dtype_number <= 13) or snapshot.dtype_number == 23 or snapshot.dtype_number == -103
    var described = (validate and numeric) or (not nested and snapshot.dtype_number == -101)
    var rows = snapshot.shape[0] if described and len(snapshot.shape) > 0 else -1
    var small_rows = rows if described and len(snapshot.shape) == 2 else -1
    var small_cols = snapshot.shape[1] if described and len(snapshot.shape) == 2 else -1
    var target_keep = target
    if not nested and classifier and target is not None:
        # The old classifier label path (integral float labels, classes_
        # wrapping) is not part of this unsupervised operation.
        raise Error("TypeError: native Normalizer fit does not run classifier label validation")
    if not nested:
        target_keep = prep_ignored_target(target, rows, required, multi, name, context)
    if py_type(previous) is not b.int and py_type(previous) is not b.bool and py_type(previous) is not b.float:
        raise Error("TypeError: native Normalizer fitting depth requires a native number")
    var started = previous + PythonObject(1)
    var remaining = started - PythonObject(1)
    _publication(state)
    py_setitem(state, "_mojolearn_fitting", started)
    var result = model
    try:
        var entries = _native_entries(mode, context)
        var values = prep_transform_values(snapshot, context)
        if norm is missing:
            norm = fields["norm"]
        var code = _normalizer_norm(norm)
        var n = snapshot.source.rows
        var d = snapshot.source.cols
        # Preserve the old _fit publication point: combined-transform failure
        # retains these learned fields, but does not set a new success flag.
        _publication(state)
        py_setitem(state, "n_features_in_", d)
        py_setitem(state, "numeric_mode_", mode)
        if project:
            result = _normalizer_apply(values, n, d, code, entries, context)
            # PublicOutputsMixin installs this conversion before _track_fit
            # wraps fit_transform, so output failure is still a failed fit.
            result = parallel_public_result(result, "fit_transform", PythonObject(None), context)
        _ = values^
    except error:
        if not nested:
            prep_fit_error(error^, name, small_rows, small_cols)
        else:
            raise error^
    finally:
        _finish_depth(state, remaining)
    py_setitem(state, "_mojolearn_fitted", True)
    if not nested:
        # before_fit's old _array_like had already removed DataFrame column
        # labels. Preserve that behavior, including removal on successful refit.
        _ = py_call(state.pop, "feature_names_in_", PythonObject(None))
    _ = target_keep^
    _ = snapshot^
    return result
