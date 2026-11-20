# SPDX-License-Identifier: Apache-2.0
# The native owner of QNRegressor's public fit, predict and score. Each is one
# call into qn_c, which converts and checks the inputs, selects the numeric-mode
# binding, runs the qn_fit / qn_decision_function bodies through the native
# estimators QN table (core/estimators_entry_table.mojo; the same request the
# Python exports built, so the same bits; no CPython between the modules) and
# stores the fitted attributes. score predicts and scores in the same call: the prediction stays
# a native buffer handed straight to the R^2 program.
# Split (lane doorsplit-linear, 2026-11-11): fit and predict are UNPACK and
# PACK in bindings/qn_boundary.mojo around a Python-free RUN in
# glm/qn_door_run.mojo; this file keeps the registered entry, the public
# lifecycle admission and the composition.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_exceptions_door import classical_require_fitted
from providers.python.linear_metadata_boundary_door import linear_fitted_unpack
from bindings.classical_attributes_door import _type_mro
from bindings.mode_metadata_native_door import _class_fields
from bindings.family_raw_attr_pack import family_raw_get_or
from bindings.classical_labels_door import labels_cast
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry, _neural_descriptor_module_fields, _neural_descriptor_function_owner
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_fit_begin, estimator_fit_end, estimator_fit_complete, estimator_feature_check
from providers.python.estimator_tabular_boundary_pyapi_unpack import _refusal
from std.python._cpython import PyObjectPtr, GILReleased
from glm.qn_door_run import QnDoorFitResult, qn_door_fit_run, qn_door_predict_run
from bindings.qn_boundary_unpack import qn_fit_unpack, qn_predict_unpack, qn_predict_pack
from bindings.qn_boundary_pack import qn_fit_pack
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_address, classical_typed
from bindings.classical_validate_unpack import classical_check_x, classical_check_y, classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from providers.python.linear_score_boundary_pack import classical_r2_score
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit, class_record, class_admit
from bindings.qn_keys_unpack import qn_keys, QN_OPN_INITIALIZE, QN_OPN_FIT, QN_OPN_PREDICT, QN_OPN_SCORE
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime _QN_AFTER_FIT_SLOT = "qn_no_after_fit_v1"
comptime _QN_KNOWN_SLOT = "qn_connected_owner_v1"


def _metric_table(metric: PythonObject, context: PythonObject) raises:
    var nk = qn_keys()
    # `_entry_table(metric, True)` once per connected metrics table while it
    # is unchanged (dictionary watchers, Python >= 3.12; older Pythons check
    # every call).
    var seen = py_get(context, nk[].k_qn_metric_admitted)
    if seen is not None and seen is metric and dict_keys_admitted(metric):
        return
    _entry_table(metric, True)
    dict_keys_admit(metric)
    py_setitem(context, nk[].k_qn_metric_admitted, metric)


def qn_regressor_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK (GIL held), RUN (GIL released, glm/qn_door_run.mojo), PACK.
    var fit = qn_fit_unpack(model, args, context)
    var result = QnDoorFitResult(0.0, 0.0, 0)
    var py = Python()
    with GILReleased(py):
        result = qn_door_fit_run(fit.run)
    return qn_fit_pack(model, fit^, result)


def qn_regressor_predict(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var predict = qn_predict_unpack(model, source, context)
    var py = Python()
    with GILReleased(py):
        qn_door_predict_run(predict.run)
    return qn_predict_pack(predict^)


def qn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var nk = qn_keys()
        var operation = py_op_code(nk[].ops, call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        _ = classical_family_admit(model)
        var result = qn_public_operation(model, operation, call[1], args, context)
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:") or String(error).startswith("DataConversionWarning:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 3)
                var warning = String(error).startswith("DataConversionWarning:")
                var cls = causal_borrowed_item(context, "warning_category" if warning else "active_not_fitted_class")
                return causal_set_exception(error_without_class(error, "DataConversionWarning" if warning else "NotFittedError"), cls)
            except nested:
                return raise_python_exception(nested)
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


# Native public lifecycle, replacing NumericModeMixin's Python decorators.
def _entry_table(entries: PythonObject, metrics: Bool) raises:
    var nk = qn_keys()
    var b = native_loaded_module("builtins")
    if py_type(entries) is not b.dict:
        raise Error("TypeError: QN providers require their native entry dictionaries")
    var keys = py_call(b.list, entries)
    for i in range(len(keys)):  # driver-loop: fixed provider metadata keys
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: QN provider names require exact strings")
    var module = py_item(entries, nk[].k__module)
    # neural_descriptor_entry's checks, with the module dictionary read and
    # its keys admitted ONCE for the whole table (it re-admitted every module
    # key for every entry on every call).
    if py_type(module) is not py_type(b):
        raise Error("TypeError: native provider requires an exact compiled module")
    var module_fields = _neural_descriptor_module_fields(module, b)
    var builtin_function = py_type(b.len)
    for i in range(len(keys)):  # driver-loop: every called entry belongs to the retained compiled provider
        var name = String(keys[i])
        if name == "_module":
            continue
        var entry = py_get(module_fields, keys[i])
        if entry is None or py_type(entry) is not builtin_function or _neural_descriptor_function_owner(entry) != module._obj_ptr:
            raise Error("TypeError: native provider requires compiled entry " + name)
        if entries[keys[i]] is not entry:
            raise Error("TypeError: QN provider entry changed after connection")


def _qn_schema(context: PythonObject) raises -> PythonObject:
    var nk = qn_keys()
    var b = native_loaded_module("builtins")
    var names = py_item(context, nk[].k_qn_parameters)
    if py_type(names) is not b.tuple:
        raise Error("TypeError: QN parameter schema requires its native tuple")
    for i in range(len(names)):  # driver-loop: exact constructor parameter names
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: QN parameter names require exact strings")
    return names


def _qn_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = qn_keys()
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    var names = _qn_schema(context)
    if len(names) != len(args):
        raise Error("TypeError: QN constructor metadata differs")
    var values = Python.dict()
    for i in range(len(names)):  # driver-loop: scalar metadata admission precedes every conversion/comparison
        host_inference_scalar(args[i], context, True)
        py_setitem(values, names[i], args[i])
    if not py_contains(py_item(context, nk[].k_qn_losses), py_item(values, nk[].k_loss)):
        raise Error("Expected loss to be one of " + String(py_call(b.repr, py_call(b.list, py_item(context, nk[].k_qn_losses)))) + ", got " + String(py_call(b.repr, py_item(values, nk[].k_loss))))
    if Bool(py=py_item(values, nk[].k_warm_start)):
        raise Error("NotImplementedError: mojolearn QNRegressor: warm_start is not implemented (the solver starts from w0 = 0)")
    if Bool(py=py_call(op.lt, py_item(values, nk[].k_l1_strength), 0)) or Bool(py=py_call(op.lt, py_item(values, nk[].k_l2_strength), 0)):
        raise Error("mojolearn QNRegressor: l1_strength and l2_strength must be non-negative, got " + String(py_call(b.str, py_item(values, nk[].k_l1_strength))) + " and " + String(py_call(b.str, py_item(values, nk[].k_l2_strength))))
    if py_item(values, nk[].k_delta) is not None and Bool(py=py_call(op.lt, py_item(values, nk[].k_delta), 0)):
        raise Error("mojolearn QNRegressor: delta must be non-negative, got " + String(py_call(b.str, py_item(values, nk[].k_delta))))
    if not Bool(py=py_call(op.gt, py_item(values, nk[].k_tol), 0)):
        raise Error("mojolearn QNRegressor: tol must be positive, got " + String(py_call(b.str, py_item(values, nk[].k_tol))))
    for name in ["max_iter", "linesearch_max_iter", "lbfgs_memory"]:  # driver-loop: three solver settings
        var value = values[name]
        if Bool(py=py_call(op.ne, py_call(b.int, value), value)) or Bool(py=py_call(op.lt, value, 1)):
            raise Error("mojolearn QNRegressor: " + name + " must be a positive integer, got " + String(py_call(b.str, value)))
    py_setitem(values, nk[].k_warm_start, PythonObject(False))
    var fields = py_fields(model)
    if py_item(values, nk[].k_numeric_mode) is None:
        py_setitem(values, nk[].k_numeric_mode, family_raw_get_or(model, "numeric_mode", PythonObject(None)))
    for i in range(len(names)):  # driver-loop: full release admission before publishing constructor values
        if py_contains(fields, names[i]) and fields[names[i]] is not values[names[i]]:
            native_release_admit(fields[names[i]], context)
    for i in range(len(names)):  # driver-loop: preserve verbatim public parameter values
        py_setitem(fields, names[i], values[names[i]])
    return PythonObject(None)


def _qn_admit(model: PythonObject, context: PythonObject, fitting: Bool) raises -> PythonObject:
    var nk = qn_keys()
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    # The class rows' verdict is kept per class state (native_class_admission:
    # any class or ancestor modification invalidates it); only a pass is kept.
    var mods = NativeModules()
    var cls = py_type(model)
    if class_record(mods, _QN_AFTER_FIT_SLOT, cls) is None:
        var ancestry = _type_mro(cls)
        for i in range(len(ancestry)):  # driver-loop: never invoke a Python post-fit hook
            if py_contains(_class_fields(ancestry[i]), nk[].k__mojolearn_after_fit):
                raise Error("TypeError: QNRegressor cannot invoke Python fit callbacks")
        class_admit(mods, _QN_AFTER_FIT_SLOT, cls, PythonObject(True))
    if py_contains(fields, nk[].k__mojolearn_after_fit):
        raise Error("TypeError: QNRegressor cannot invoke Python fit callbacks")
    var names = _qn_schema(context)
    for i in range(len(names)):  # driver-loop: fields used by the numerical owner
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context, True)
    var scalars = Python.tuple("n_features_in_", "input_copied_", "intercept_", "objective_", "retcode_", "_mojolearn_fitted", "_mojolearn_fitting")
    var arrays = Python.tuple("_w", "coef_", "n_iter_", "feature_names_in_")
    for i in range(len(scalars)):  # driver-loop: scalar fitted metadata before native conversion
        if py_contains(fields, scalars[i]):
            host_inference_scalar(fields[scalars[i]], context)
            if fitting:
                native_release_admit(fields[scalars[i]], context)
    for i in range(len(arrays)):  # driver-loop: retained fitted buffers and released prior owners
        if py_contains(fields, arrays[i]):
            classical_admit_value(fields[arrays[i]], context)
            if fitting:
                native_release_admit(fields[arrays[i]], context)
    var fitted = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags require exact booleans or integers")
    if py_type(py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    return fields


def qn_public_operation(model: PythonObject, operation: Int, operation_name: PythonObject, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var nk = qn_keys()
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: QN arguments require a native tuple")
    # Whether the model's class descends from the connected owner depends on
    # the class rows only: kept per class state, with the owner it matched
    # (a reconnection installs a new owner class, so the identity differs).
    var mods = NativeModules()
    var cls = py_type(model)
    var owner = py_item(context, nk[].k_qn_public_class)
    # The owner class itself needs no walk; a subclass's verdict is recorded
    # with the owner as payload (a record must not retain its own class).
    var record = PythonObject(None) if cls is owner else class_record(mods, _QN_KNOWN_SLOT, cls)
    if not (cls is owner) and (record is None or record is not owner):
        var ancestry = _type_mro(cls)
        var known = False
        for i in range(len(ancestry)):  # driver-loop: preserve passive subclasses of the connected owner
            known = known or ancestry[i] is owner
        if not known:
            raise Error("TypeError: QNRegressor requires its connected native class")
        class_admit(mods, _QN_KNOWN_SLOT, cls, owner)
    if operation == QN_OPN_INITIALIZE:
        return _qn_initialize(model, args, context)
    var fitting = operation == QN_OPN_FIT
    var fields = _qn_admit(model, context, fitting)
    # The operation's name String is built only for the not-fitted message
    # (`classical_require_fitted` re-checks and raises it unchanged).
    if not fitting and not linear_fitted_unpack(model):
        classical_require_fitted(model, False, String(py=operation_name), context)
    var name = classical_estimator_name(model)
    var x = classical_check_x(args[0], name, context)
    var rows = Int(py=x.shape[0])
    var columns = Int(py=x.shape[1])
    if not fitting:
        classical_check_width(columns, Int(py=py_item(fields, nk[].k_n_features_in_)), name)
        estimator_feature_check(model, fields, args[0], context)
    var y = classical_check_y(args[1], rows, name, context) if fitting else PythonObject(None)
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        if fitting:
            result = qn_regressor_fit(model, Python.tuple(x, y, args[2]), context)
        elif operation == QN_OPN_PREDICT:
            result = qn_regressor_predict(model, x, context)
        elif operation == QN_OPN_SCORE:
            var metric = py_item(context, nk[].k_classical_metrics)[py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)]
            if py_type(metric) is not b.str:
                _metric_table(metric, context)
            result = classical_r2_score(qn_regressor_predict(model, x, context), args[1], context)
        else:
            raise Error("RuntimeError: unknown native QN operation")
    except problem:
        var refused = _refusal(problem, fitting, depth > 0, rows, columns, name)
        raise refused^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, args[0], False, context)
    if operation == QN_OPN_PREDICT and estimator_outside() and py_get(context, nk[].k_ndarray) is not None:
        var wide = classical_typed(result.shape, "<f8", context)
        labels_cast(Int(py=classical_address(result)), "<f4", Int(py=classical_address(wide)), "<f8", Int(py=result.size), context)
        _ = result^
        return wide
    return result
