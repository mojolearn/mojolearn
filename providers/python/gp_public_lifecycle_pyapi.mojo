# SPDX-License-Identifier: Apache-2.0
# Whole public GP lifecycle previously supplied by _mode's Python decorators.
# The public sentinel distinguishes this owner from existing native archive,
# parallel and model-selection callers that already prepare their arguments.
from std.math import isfinite, floor
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.admitted_call import AdmittedCall
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.mode_metadata_native import _fields, _class_fields, _plan, _configuration_value
from bindings.classical_attributes import _type_mro, classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_width, classical_estimator_name
from bindings.classical_labels import _flatten_labels, labels_shape_of
from bindings.label_integer_storage import labels_integer_storage
from bindings.label_input_sources import labels_storage_source, labels_list_integer_storage
from bindings.gp_label_snapshot_native import gp_label_list
from bindings.classical_warnings import classical_warn_column
from bindings.classical_exceptions import classical_require_fitted
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.host_inference_scalars import host_inference_scalar, host_inference_primitive_number
from bindings.family_common import family_cast_view_f64
from bindings.native_array_view import native_array_ndarray_view
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.gp_kernel_public_native import gp_admit_kernel
from bindings.gp_regressor_native import gp_regressor_operation
from bindings.gp_classifier_native import gp_classifier_operation
from bindings.gp_state_admission_native import _gp_scalar_fields, _gp_array_fields, _gp_admit_binary_class
from bindings.gp_svm_records import gpsvm_no_fit_callbacks
from bindings.gp_svm_gp_state import gp_state_forget, gp_state_admitted, gp_state_remember, gp_x_mark, gp_x_release
from bindings.optimizer_public_words import stash_python_error
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.estimator_binding_pyapi import estimator_arguments, estimator_binding_mode
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_public_lifecycle_pyapi import _refusal


def _gp_admit_state(model: PythonObject, context: PythonObject, classifier: Bool, fitting: Bool) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    if classifier or py_contains(fields, "estimators_"):
        _gp_admit_binary_class(context)
    # The ancestry scan runs once per class state (bindings/gp_svm_records.mojo).
    gpsvm_no_fit_callbacks(model, fields, "TypeError: native GP operations cannot invoke Python fit callbacks")
    var scalar_names = Python.tuple("numeric_mode", "_alpha", "_optimizer", "_n_restarts_optimizer", "_normalize_y", "_random_state", "_max_iter_predict", "n_features_in_", "n_classes_", "info_", "normalize_y_", "_y_train_mean", "_y_train_std", "log_marginal_likelihood_value_", "_mojolearn_fitted", "_mojolearn_fitting")
    var array_names = Python.tuple("X_train_", "y_train_", "L_", "alpha_", "pi_", "W_sr_", "feature_names_in_")
    var member_arrays = Python.tuple("y_train_", "L_", "pi_", "W_sr_")
    var member_scalars = Python.tuple("log_marginal_likelihood_value_", "n_iter_", "nb_")
    if fitting:
        # A refit replaces the fitted state: release its record now.
        gp_state_forget(model)
    elif gp_state_admitted(model, fields, scalar_names, array_names, member_scalars, member_arrays, context):
        # The fitted state is the one a full admission accepted (identity
        # record per model); only identity-insufficient values were admitted
        # again. Every predict-side call used to re-admit it all.
        return fields
    # Constructor settings are consumed by fit. Prediction uses fitted state;
    # changing an unused constructor parameter must not change its behavior.
    if fitting:
        _gp_scalar_fields(fields, Python.tuple("alpha", "optimizer", "n_restarts_optimizer", "normalize_y", "copy_X_train", "n_targets", "max_iter_predict", "warm_start", "multi_class", "n_jobs"), context)
        if not classifier:
            _gp_scalar_fields(fields, Python.tuple("random_state"), context)
    _gp_scalar_fields(fields, scalar_names, context)
    # GPC stores random_state unchanged and never consumes it; RandomState
    # objects remain accepted without asking them for a scalar protocol.
    _gp_array_fields(fields, array_names, context)
    for name in ["kernel", "_kernel", "kernel_"]:  # small-loop(fixed kernel state fields)
        if name != "kernel" or fitting:
            var value = py_get(fields, name)
            if value is not None:
                gp_admit_kernel(value, context)
    if py_contains(fields, "classes_"):
        classical_admit_value(classifier_label_list(fields["classes_"], context), context)
    if py_contains(fields, "estimators_"):
        var fits = fields["estimators_"]
        if py_type(fits) is not b.list and py_type(fits) is not b.tuple:
            raise Error("TypeError: GP fitted estimators require native list or tuple storage")
        for i in range(len(fits)):  # driver-loop: retained one-vs-rest binary fit owners
            if py_type(fits[i]) is not context["family_gp_binary_laplace"]:
                raise Error("TypeError: GP binary state requires its native-connected storage class")
            var own = _fields(fits[i])
            _gp_array_fields(own, member_arrays, context)
            _gp_scalar_fields(own, member_scalars, context)
    if not fitting:
        gp_state_remember(model, fields, context)
    return fields


def _gp_label_admit(value: PythonObject, name: String, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: GP target nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.complex:
        raise Error("Complex data not supported: " + name + " got a complex y.")
    if kind is b.list or kind is b.tuple:
        for i in range(len(value)):  # driver-loop: passive label admission before encoder operations
            _gp_label_admit(value[i], name, context, depth + 1)
    else:
        classical_admit_value(value, context)


def _gp_classifier_target(source: PythonObject, rows: Int, name: String, context: PythonObject) raises -> PythonObject:
    if source is None:
        raise Error(name + " requires y to be passed, but the target y is None.")
    var b = native_loaded_module("builtins")
    var kind = py_type(source)
    # An exact flat list / tuple of exact ints first (lane p3-svm-gp-kernel,
    # row 80): every item is an exact int, which the per-label admission walk
    # below passes unchanged, so the walk is skipped (bools keep it).
    if (kind is b.list or kind is b.tuple) and labels_list_integer_storage(source, False):
        if len(source) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(source)) + "]")
        return py_call(b.list, source)
    _gp_label_admit(source, name, context)
    if kind is b.str or kind is b.bytes:
        raise Error("mojolearn: y must be a sequence of labels")
    # A rank-1 integer ndarray or Array goes to the encoder unchanged:
    # gp_encode_labels reads it with one native prepare_targets call. The
    # snapshot below boxes, admits and converts every label as a Python object
    # (about 2.3 s per 20000-label fit); it cannot refuse an integer label and
    # yields the same Python-int classes and codes, so no bits change. Bool,
    # float, 2-D, list, torch and object targets keep the snapshot.
    # perf3-label-doors: a numpy-backed pandas Series takes the same door
    # through its stored exact ndarray (the labels its snapshot lists, in the
    # same order), which the encoder then reads natively.
    var stored = labels_storage_source(source, context)
    if stored is not None and labels_integer_storage(stored, context, False):
        var target_shape = labels_shape_of(stored, context)
        if len(target_shape) == 1:
            if target_shape[0] != rows:
                raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(target_shape[0]) + "]")
            return stored
    # An exact flat list / tuple of exact ints (bools keep the snapshot): the
    # snapshot admits every label and lists the same ints; one native list
    # copy (no per-label call) stands in for its flattened list.
    if labels_list_integer_storage(source, False):
        if len(source) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(source)) + "]")
        return py_call(b.list, source)
    # A single native container snapshot retains label identities and integer
    # precision. All following shape/finite/integral checks use that snapshot;
    # the numerical owner sees the finished native list, never the user input.
    var values = source
    if _ordinary_torch(kind, context):
        values = gp_label_list(native_torch_numpy(source), context)
    elif kind is not b.list and kind is not b.tuple:
        values = gp_label_list(source, context)
    # A list / tuple source was admitted label by label above; only a new
    # snapshot is admitted here (one walk per label, not two).
    if not (values is source):
        _gp_label_admit(values, name, context)
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        raise Error("mojolearn: y must be a sequence of labels")
    if len(values) != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(values)) + "]")
    if len(values) > 0 and kind is not b.list and kind is not b.tuple and kind is not b.memoryview:
        var first = values[0]
        if (py_type(first) is b.list or py_type(first) is b.tuple) and len(first) == 1:
            classical_warn_column(context)
    var flat = _flatten_labels(values, context)
    for i in range(len(flat)):  # driver-loop: admitted scalar labels before native encoder/comparison
        var item = host_inference_primitive_number(flat[i], context)
        if py_type(item) is b.float:
            var number = Float64(py=item)
            if not isfinite(number):
                raise Error("Input y contains NaN or infinity (" + name + ").")
            if floor(number) != number:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
        py_setitem(flat, i, item)
    return flat


def _gp_finished(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is context["array_class"]:
        return native_array_ndarray_view(value, context)
    if py_type(value) is b.tuple:
        var out = Python.list()
        for i in range(len(value)):  # driver-loop: fixed result tuple members
            _ = py_call(out.append, _gp_finished(value[i], context))
        return py_call(b.tuple, out)
    if py_type(value) is b.list and len(value) > 0 and py_type(value[0]) is context["array_class"]:
        var out = Python.list()
        for i in range(len(value)):  # driver-loop: one native array result per output
            _ = py_call(out.append, _gp_finished(value[i], context))
        return out
    return value



def _gp_public_arguments(mut call: AdmittedCall, model: PythonObject, fields: PythonObject, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # `call`: the door's admission record (modules, fields, the resolved
    # binding), so the binding is resolved once per public call (N219).
    var b = call.builtins()
    var metadata = py_fields(call.module("mojolearn._mode"))["_MODE_METADATA_CONTEXT"]
    var plan = _plan(call.modules, model, fields, metadata)
    var route = String(plan["route"])
    if route != "host" and route != "host_gp":
        return estimator_arguments(call, model, "_mojolearn_gp", args, context)
    # cpu-route: inherited methods of explicit HostGaussianProcess* classes
    # retain their import-connected host route, even in a GPU installation.
    # This is never a fallback from a GPU estimator or process-selected mode.
    var requested = _configuration_value(model, fields, "numeric_mode")
    if requested is not None and (py_type(requested) is not b.str or String(requested) != "identical"):
        raise Error("ValueError: this host estimator runs identical only")
    var connected = plan["modes"]["identical"]
    var error = py_get(connected, "error")
    if error is not None:
        ref cpy = Python().cpython()
        var kind = py_type(error)
        cpy.lib.get_function[NoneType]("PyErr_SetObject")(kind._obj_ptr, error._obj_ptr)
        var failure = stash_python_error()
        raise failure^
    var reader = py_get(connected, "mode")
    if reader is None:
        reader = py_get(connected, "vendor")
    if reader is None or py_type(reader) is not py_type(b.len):
        raise Error("TypeError: GP host execution requires its connected native module readback")
    var module = reader.__self__
    if py_type(module) is not py_type(b):
        raise Error("TypeError: GP host execution requires an exact native module")
    var module_name = py_get(py_fields(module), "__name__")
    if py_type(module_name) is not b.str:
        raise Error("TypeError: GP host module name must be an exact string")
    var live = py_get(call.module("sys").modules, module_name)
    if live is None:
        raise Error("ImportError: GP host binding was removed after import-time connection")
    var live_reader = neural_descriptor_entry(live, String(reader.__name__))
    if py_get(connected, "mode") is not None and Int(py=py_call(live_reader)) != 1:
        raise Error("RuntimeError: GP host binding must report identical mode")
    var prepared = Python.list(live)
    for i in range(1, len(args)):  # driver-loop: fixed inherited public argument envelope
        _ = py_call(prepared.append, args[i])
    return py_call(b.tuple, prepared)

def gp_public_operation(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var classifier = operation.startswith("family_gp_gpc_")
    var fields = _fields(model)
    var fitting = operation.endswith("_fit")
    var method = String("fit") if fitting else (String("predict_proba") if operation.endswith("_predict_proba") else (String("predict") if operation.endswith("_predict") else (String("score") if operation.endswith("_score") else String("other"))))
    var guarded = method != "other"
    var double_binding = classifier and (fitting or method == "predict" or method == "score")
    var x_index = 2 if double_binding else 1
    var depth_value = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth_value) is not py_type(PythonObject(0)):
        raise Error("TypeError: native GP fit depth must be an exact integer")
    var nested = Int(py=depth_value) > 0
    if guarded and not fitting:
        classical_require_fitted(model, False, method, context)
    fields = _gp_admit_state(model, context, classifier, fitting)
    # One admission record for this public call (ne1-e-admitted-call): the
    # binding resolutions below (the GP binding and, for a classifier, the
    # base binding) read it instead of resolving per helper (N219, N256).
    var call = AdmittedCall()
    call.admit_fields(fields)
    var name = classical_estimator_name(model)
    var source = args[x_index] if guarded else PythonObject(None)
    var values = py_call(call.builtins().list, args)
    var rows = 0
    var columns = 0
    var input_copied = False
    var admitted_x = PythonObject(None)
    if guarded:
        var x = classical_check_x(source, name, context)
        admitted_x = x
        input_copied = not (x is source)
        rows = Int(py=x.shape[0])
        columns = Int(py=x.shape[1])
        py_setitem(values, x_index, x)
        if not nested:
            if fitting:
                var target = _gp_classifier_target(args[x_index + 1], rows, name, context) if classifier else classical_check_y(args[x_index + 1], rows, name, context)
                py_setitem(values, x_index + 1, target)
            else:
                var width = py_get(fields, "n_features_in_")
                if py_type(width) is py_type(PythonObject(0)):
                    classical_check_width(columns, Int(py=width), name)
                estimator_feature_check(model, fields, source, context)
    # Public option scalars must never reach Python truth/int protocols. Score
    # and fit y are separately admitted as labels/numeric buffers by owners.
    if operation == "family_gp_gpr_predict" or operation == "family_gp_gpr_sample_y":
        host_inference_scalar(args[2], context, True)
        host_inference_scalar(args[3], context, True)
    elif operation == "family_gp_gpr_lml":
        host_inference_scalar(args[2], context, True)
        classical_admit_value(args[1], context)
    if guarded and method == "score":
        classical_admit_value(args[x_index + 1], context)
    # The X / y admissions above may run in GIL release windows and take no
    # record: re-admit the record's modules before resolving the bindings.
    call.released()
    var prepared = _gp_public_arguments(call, model, fields, values, context)
    if double_binding and prepared[1] is None:
        var base = estimator_binding_mode(call, PythonObject("identical"), "_mojolearn", context)
        values = py_call(call.builtins().list, prepared)
        py_setitem(values, 1, base)
        prepared = py_call(call.builtins().tuple, values)
    var outside = estimator_outside()
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    if admitted_x is not None:
        # The owner reuses this admitted C-order float32 X instead of
        # converting it a second time (bindings/gp_svm_records.mojo).
        gp_x_mark(admitted_x)
    try:
        if classifier:
            result = gp_classifier_operation(model, operation, prepared, context)
        else:
            result = gp_regressor_operation(model, operation, prepared, context)
    except error:
        if guarded:
            var refused = _refusal(error, fitting, nested, rows, columns, name)
            raise refused^
        raise error^
    finally:
        if admitted_x is not None:
            gp_x_release(admitted_x)
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        # Include the public boundary conversion in the recorded copy fact.
        # The numerical owner only sees our finished native input.
        if input_copied:
            py_setitem(fields, "input_copied_", PythonObject(True))
        estimator_fit_complete(model, source, classifier, context)
    if outside and (method == "predict" or method == "predict_proba") and py_get(context, "ndarray") is not None:
        # Existing semantics widen only a standalone regressor prediction.
        # A (mean, std/cov) tuple keeps each kernel's float32 output dtype.
        if not classifier and method == "predict" and py_type(result) is context["array_class"]:
            result = family_cast_view_f64(result, "prediction", context)
        return _gp_finished(result, context)
    return result
