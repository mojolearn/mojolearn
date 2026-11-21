# SPDX-License-Identifier: Apache-2.0
# Whole public GP lifecycle previously supplied by _mode's Python decorators.
# The public sentinel distinguishes this owner from existing native archive,
# parallel and model-selection callers that already prepare their arguments.
from bindings.admitted_finite_wording import admitted_finite_wording
from std.math import isfinite, floor
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.admitted_family_door import admit_family_f64
from bindings.native_loaded_module_door import native_loaded_module
from bindings.admitted_call_door import AdmittedCall
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.mode_metadata_native_door import _fields, _class_fields, _plan, _configuration_value
from bindings.gp_native_boundary_door import gp_entry_key
from bindings.classical_attributes_door import _type_mro, classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.admitted_input_unpack import admit_x_unrecorded
from bindings.device_rows_door import door_is_device_rows
from bindings.classical_labels_door import _flatten_labels, labels_shape_of
from bindings.label_integer_storage_unpack import labels_integer_storage
from bindings.label_input_sources_unpack import labels_storage_source, labels_list_integer_storage
from bindings.gp_label_snapshot_native_unpack import gp_label_list
from bindings.gp_argument_pins_unpack import GPArgumentPins
from bindings.classical_warnings_unpack import classical_warn_column
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.host_inference_scalars_door import host_inference_scalar, host_inference_primitive_number
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.sequential_admission_door import _ordinary_torch
from bindings.native_torch_admission_door import native_torch_numpy
from bindings.gp_kernel_public_native_door import gp_admit_kernel
from bindings.gp_regressor_native_door import gp_regressor_operation
from bindings.gp_classifier_native_door import gp_classifier_operation
from bindings.gp_state_admission_native_unpack import _gp_scalar_fields, _gp_array_fields, _gp_admit_binary_class
from bindings.gp_svm_records_door import gpsvm_no_fit_callbacks
from bindings.gp_svm_gp_state_door import gp_state_forget, gp_state_admitted, gp_state_remember, gp_x_mark, gp_x_release
from bindings.optimizer_public_words_unpack import stash_python_error
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_routed
from providers.python.device_route_boundary_door import DeviceRouteOwner, device_route_held, device_route_keep
from core.device_route import DEVICE_CPU, DEVICE_GPU
from core.gp_entry_table import gp_table_resolve
from core.gp_native_api import GP_CAPABILITIES, GPDriver
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_public_lifecycle_pyapi_door import _refusal
from bindings.py_optional import py_present
from bindings.gp_keys_unpack import gp_keys
from bindings.operation_names_gp import (
    GP_OPN_FAMILY_GP_GPR_FIT, GP_OPN_FAMILY_GP_GPR_PREDICT, GP_OPN_FAMILY_GP_GPR_SAMPLE_Y,
    GP_OPN_FAMILY_GP_GPR_LML, GP_OPN_FAMILY_GP_GPR_SCORE, GP_OPN_FAMILY_GP_GPC_LML, GP_OPN_FAMILY_GP_GPC_FIT,
    GP_OPN_FAMILY_GP_GPC_PREDICT, GP_OPN_FAMILY_GP_GPC_PREDICT_PROBA, GP_OPN_FAMILY_GP_GPC_SCORE)
from bindings.py_balanced import py_item


def _gp_admit_state(model: PythonObject, context: PythonObject, classifier: Bool, fitting: Bool) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    var nk = gp_keys()
    if classifier or py_contains(fields, nk[].k_estimators_):
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
    if py_contains(fields, nk[].k_classes_):
        classical_admit_value(classifier_label_list(py_item(fields, nk[].k_classes_), context), context)
    if py_contains(fields, nk[].k_estimators_):
        var fits = py_item(fields, nk[].k_estimators_)
        if py_type(fits) is not b.list and py_type(fits) is not b.tuple:
            raise Error("TypeError: GP fitted estimators require native list or tuple storage")
        for i in range(len(fits)):  # driver-loop: retained one-vs-rest binary fit owners
            if py_type(fits[i]) is not py_item(context, nk[].k_family_gp_binary_laplace):
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
    var nk = gp_keys()
    for i in range(len(flat)):  # driver-loop: admitted scalar labels before native encoder/comparison
        var item = host_inference_primitive_number(flat[i], context)
        if py_type(item) is b.float:
            var number = Float64(py=item)
            if not isfinite(number):
                raise Error(admitted_finite_wording(name, "y"))
            if floor(number) != number:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
        # The primitive number is for the checks above only: the encoder keeps
        # the first ORIGINAL label object of each class (gp_mixed_classes,
        # 48664c4c59); writing the converted number back replaced NumPy scalar
        # labels with new Python numbers (1ec92cef40; smoke 2026-11-15). A 0-d
        # ndarray label is not a scalar the encoder admits: it keeps the number.
        if py_type(flat[i]) is py_get(context, nk[].k_ndarray):
            py_setitem(flat, i, item)
    return flat


def _gp_finished(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var nk = gp_keys()
    if py_type(value) is py_item(context, nk[].k_array_class):
        return native_array_ndarray_view(value, context)
    var tuple_output = py_type(value) is b.tuple
    if not tuple_output and not (py_type(value) is b.list and len(value) > 0
                                 and py_type(value[0]) is py_item(context, nk[].k_array_class)):
        return value
    ref cpy = Python().cpython()
    var api = cpython_table()
    var raw: PyObjectPtr
    if tuple_output:
        raw = cpy.PyTuple_New(len(value))
    else:
        raw = api[].PyList_New(len(value))
    if not raw:
        raise cpy.unsafe_get_error()
    # The exact, private NULL-filled output owns initialized slots even when
    # a later view conversion raises. It is published only when complete.
    var result = PythonObject(from_owned=raw)
    for i in range(len(value)):  # PACK fixed result members in their original order
        var item = _gp_finished(value[i], context)
        var status = Int32(0)
        if tuple_output:
            status = cpy.PyTuple_SetItem(result._obj_ptr, i, item.steal_data())
        else:
            status = api[].PyList_SetItem(result._obj_ptr, i, item.steal_data())
        # Both setters consume the item reference on success and failure.
        if status != 0:
            raise cpy.unsafe_get_error()
    _ = value
    return result


def _gp_host_class_binding(mut call: AdmittedCall, model: PythonObject, fields: PythonObject, plan: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var b = call.builtins()
    var nk = gp_keys()
    # cpu-route: inherited methods of explicit HostGaussianProcess* classes
    # retain their import-connected host route, even in a GPU installation.
    # This is never a fallback from a GPU estimator or process-selected mode.
    var requested = _configuration_value(model, fields, "numeric_mode")
    if requested is not None and (py_type(requested) is not b.str or String(requested) != "identical"):
        raise Error("ValueError: this host estimator runs identical only")
    var connected = py_item(py_item(plan, nk[].k_modes), nk[].k_identical)
    var error = py_get(connected, nk[].k_error)
    if error is not None:
        ref cpy = Python().cpython()
        var kind = py_type(error)
        cpy.lib.get_function[NoneType]("PyErr_SetObject")(kind._obj_ptr, error._obj_ptr)
        var failure = stash_python_error()
        raise failure^
    var reader = py_get(connected, nk[].k_mode)
    if reader is None:
        reader = py_get(connected, nk[].k_vendor)
    if reader is None or py_type(reader) is not py_type(b.len):
        raise Error("TypeError: GP host execution requires its connected native module readback")
    var module = reader.__self__
    if py_type(module) is not py_type(b):
        raise Error("TypeError: GP host execution requires an exact native module")
    var module_name = py_get(py_fields(module), nk[].k___name__)
    if py_type(module_name) is not b.str:
        raise Error("TypeError: GP host module name must be an exact string")
    var live = py_get(call.module("sys").modules, module_name)
    if live is None:
        raise Error("ImportError: GP host binding was removed after import-time connection")
    var live_reader = neural_descriptor_entry(live, String(reader.__name__))
    if py_get(connected, nk[].k_mode) is not None and Int(py=py_call(live_reader)) != 1:
        raise Error("RuntimeError: GP host binding must report identical mode")
    return live


def _gp_route(mut call: AdmittedCall, model: PythonObject, fields: PythonObject, need_base: Bool,
              context: PythonObject) raises -> PythonObject:
    """The estimator's route owner for this operation (the one
    `DeviceRouteOwner`, providers/python/device_route_boundary_door.mojo,
    slot `device_route_v1`). A hit is the mode read (and the default-mode
    read when it is None), one record read and three pointer compares; a
    miss (first operation, a changed mode, after `set_params`, a classifier's
    first operation that needs its base image) resolves and stores it.

    The owner holds, resolved ONCE per estimator: the GP image of the
    estimator's device (`images`; the shared estimator door reads `device`
    on its own miss), that image's table entry, tier and capability word
    (`route.entry`, `route.mode`, `route.flags`: what every
    `GPConnection` of the operation is built from, with no key read, slot
    lookup or capability call), the classifier's base image (`scope`), and
    for a CPU route beside a GPU set the program factory of the CPU metrics
    image (`table`, `route.has_second`). No public call reads the device.
    `call` is the door's admission record (read on a miss only)."""
    var nk = gp_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var held = device_route_held(model, mode, default_mode, context)
    if held is not None:
        if not need_base or held.downcast_value_ptr[DeviceRouteOwner]()[].scope is not None:
            return held
    var metadata = py_item(py_fields(call.module("mojolearn._mode")), nk[].k__MODE_METADATA_CONTEXT)
    var plan = _plan(call.modules, model, fields, metadata)
    var route = String(py_item(plan, nk[].k_route))
    var host_class = route == "host" or route == "host_gp"
    var binding = PythonObject(None)
    if host_class:
        # The explicit HostGaussianProcess* classes: CPU identical by class,
        # whatever `device` their inherited constructor stored.
        binding = _gp_host_class_binding(call, model, fields, plan, context)
    else:
        binding = estimator_binding(call, model, "_mojolearn_gp", context)
    var key = gp_entry_key(binding)
    var entry = gp_table_resolve(key)
    var driver = GPDriver(entry.call)
    var built = DeviceRouteOwner(DEVICE_CPU if key.vendor == 0 else DEVICE_GPU)
    built.route.mode = key.mode
    built.route.vendor = key.vendor
    built.route.entry = entry
    built.route.flags = driver.metadata(GP_CAPABILITIES)
    built.mode = mode
    built.default_mode = default_mode
    built.context = context
    built.images = binding
    if need_base:
        built.scope = estimator_binding_routed(model, PythonObject("identical"), "_mojolearn", context)
    if key.vendor == 0 and not host_class:
        var cpu_only = py_item(py_item(context, nk[].k_backend_fields), nk[].k__CPU_ONLY) is not None
        if not cpu_only:
            # A CPU route beside a GPU set: `score` runs the CPU metrics
            # image of the same route. One that is not connected is said by
            # the `score` that needs it (bindings/gp_native_boundary_door.mojo
            # `gp_route_cpu_metrics`), never by a fit or a predict.
            built.route.has_second = True
            try:
                var metrics = estimator_binding(call, model, "_mojolearn_x_metrics", context, False)
                built.table = py_get(py_fields(metrics), nk[].k_metrics_program_native_api_v1)
            except problem:
                built.refusal = String(problem)
    built.resolved = True
    return device_route_keep(model, built^)


def _gp_query_operation(operation: Int) -> Bool:
    """The public queries whose owner admits X through gp_query (lane a7a):
    their kernel entries test the query (or the owner does)."""
    return (operation == GP_OPN_FAMILY_GP_GPR_PREDICT or operation == GP_OPN_FAMILY_GP_GPR_SCORE
            or operation == GP_OPN_FAMILY_GP_GPC_PREDICT or operation == GP_OPN_FAMILY_GP_GPC_PREDICT_PROBA
            or operation == GP_OPN_FAMILY_GP_GPC_SCORE)


def gp_public_operation(model: PythonObject, operation: Int, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # bw2-core: `operation` is the leaf code of bindings/operation_names_gp.mojo
    # (a regressor or classifier code; the kernel codes never reach this
    # door). The classifier test is a range test and the method tests are
    # integer compares (they were a prefix test and four suffix tests on the
    # name String per call).
    classical_connect_numpy(context)
    var classifier = operation >= GP_OPN_FAMILY_GP_GPC_LML
    var fields = _fields(model)
    var fitting = operation == GP_OPN_FAMILY_GP_GPR_FIT or operation == GP_OPN_FAMILY_GP_GPC_FIT
    var predicting = operation == GP_OPN_FAMILY_GP_GPR_PREDICT or operation == GP_OPN_FAMILY_GP_GPC_PREDICT
    var scoring = operation == GP_OPN_FAMILY_GP_GPR_SCORE or operation == GP_OPN_FAMILY_GP_GPC_SCORE
    var proba = operation == GP_OPN_FAMILY_GP_GPC_PREDICT_PROBA
    var guarded = fitting or proba or predicting or scoring
    var double_binding = classifier and (fitting or predicting or scoring)
    var x_index = 2 if double_binding else 1
    var nk = gp_keys()
    var depth_value = py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))
    if py_type(depth_value) is not py_type(PythonObject(0)):
        raise Error("TypeError: native GP fit depth must be an exact integer")
    var nested = Int(py=depth_value) > 0
    if guarded and not fitting:
        # The method name is built only here, for the not-fitted refusal.
        var method = String("predict_proba") if proba else (String("predict") if predicting else String("score"))
        classical_require_fitted(model, False, method, context)
    fields = _gp_admit_state(model, context, classifier, fitting)
    # One admission record for this public call (ne1-e-admitted-call): the
    # binding resolutions below (the GP binding and, for a classifier, the
    # base binding) read it instead of resolving per helper (N219, N256).
    var call = AdmittedCall()
    call.admit_fields(fields)
    # The estimator's route: the images it runs on, resolved once per
    # estimator. Read before X is admitted: a CPU route takes host arrays.
    var routed = _gp_route(call, model, fields, double_binding, context)
    var name = classical_estimator_name(model)
    var source = args[x_index] if guarded else PythonObject(None)
    var values = GPArgumentPins(args)
    var rows = 0
    var columns = 0
    var input_copied = False
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var admitted_x = Optional[PythonObject]()
    var deferred = False
    if guarded:
        # Lane a7a: a predict / score defers the finite test of a borrowed
        # X to its kernel entry (defer_finite: the entry tests the words it
        # uploads and returns the status word; an owner whose route returns
        # none tests it itself, bindings/gp_query_x_door.mojo), and a
        # cross-validation fold stays on the device (device_x: the owner
        # hands its store ids to the entry, which gathers the rows). A fit
        # keeps the scan here: it keeps X as X_train_.
        var query = not fitting and _gp_query_operation(operation)
        if query and door_is_device_rows(source, context):
            if routed.downcast_value_ptr[DeviceRouteOwner]()[].route.has_second:
                raise Error("TypeError: " + name + "(device='cpu') takes host arrays; this X is rows held on the GPU")
            var fx = admit_x_unrecorded(source, name, context, device_x=True, finite=False)
            rows = fx.rows
            columns = fx.cols
            if not fx.store.active():
                # (an active store: values[x_index] stays the DeviceRows and
                # the owner resolves its store ids; no host words exist)
                admitted_x = py_present(fx.array)
                deferred = True
                values.set(x_index, fx.array)
        else:
            var ax = admit_x_unrecorded(source, name, context, defer_finite=query)
            admitted_x = py_present(ax.array)
            deferred = not ax.finite
            input_copied = not (ax.array is source)
            rows = ax.rows
            columns = ax.cols
            values.set(x_index, ax.array)
        if not nested:
            if fitting:
                var target = _gp_classifier_target(args[x_index + 1], rows, name, context) if classifier else classical_check_y(args[x_index + 1], rows, name, context)
                values.set(x_index + 1, target)
            else:
                var width = py_get(fields, nk[].k_n_features_in_)
                if py_type(width) is py_type(PythonObject(0)):
                    classical_check_width(columns, Int(py=width), name)
                estimator_feature_check(model, fields, source, context)
    # Public option scalars must never reach Python truth/int protocols. Score
    # and fit y are separately admitted as labels/numeric buffers by owners.
    if operation == GP_OPN_FAMILY_GP_GPR_PREDICT or operation == GP_OPN_FAMILY_GP_GPR_SAMPLE_Y:
        host_inference_scalar(args[2], context, True)
        host_inference_scalar(args[3], context, True)
    elif operation == GP_OPN_FAMILY_GP_GPR_LML:
        host_inference_scalar(args[2], context, True)
        classical_admit_value(args[1], context)
    if guarded and scoring:
        classical_admit_value(args[x_index + 1], context)
    # The X / y admissions above may run in GIL release windows and take no
    # record: re-admit the record's modules before resolving the bindings.
    call.released()
    var prepared = values^
    # args[0] is the route owner: every `GPConnection` of the operation is
    # built from its resolved entry (bindings/gp_native_boundary_door.mojo).
    prepared.binding(routed)
    if double_binding and prepared[1] is None:
        var base = routed.downcast_value_ptr[DeviceRouteOwner]()[].scope
        prepared.set(1, base)
    var outside = estimator_outside()
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    if Bool(admitted_x):
        # The owner reuses this admitted C-order float32 X instead of
        # converting it a second time (bindings/gp_svm_records.mojo), and
        # tests it when its finite test was deferred and its entry returns
        # no status word.
        gp_x_mark(admitted_x.value(), deferred)
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
        if Bool(admitted_x):
            gp_x_release(admitted_x.value())
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        # Include the public boundary conversion in the recorded copy fact.
        # The numerical owner only sees our finished native input.
        if input_copied:
            py_setitem(fields, nk[].k_input_copied_, PythonObject(True))
        estimator_fit_complete(model, source, classifier, context)
    if outside and (predicting or proba) and py_get(context, nk[].k_ndarray) is not None:
        # Existing semantics widen only a standalone regressor prediction.
        # A (mean, std/cov) tuple keeps each kernel's float32 output dtype.
        if not classifier and predicting and py_type(result) is py_item(context, nk[].k_array_class):
            result = admit_family_f64(result, "prediction", context)
        return _gp_finished(result, context)
    return result
