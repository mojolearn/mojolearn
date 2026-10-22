# SPDX-License-Identifier: Apache-2.0
# Explicit CPU density checkpoints retain the existing deterministic schemas.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_connect_numpy, classical_getattribute
from bindings.classical_arrays import classical_wrap_array
from bindings.classical_pandas import _pd_type_fields
from bindings.classical_archive import classical_archive_write
from bindings.host_tree_native import host_tree_constructor_fields
from bindings.host_density_constructor_native import _density_option, _density_member, _density_descriptor, _density_get_slot
from bindings.host_kde_native import _kde_option_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_iforest_fit_native import _if_fit_options, _if_fit_number, _if_fit_truth, _if_fit_repr
from bindings.native_owner_lifetime import native_release_admit
from bindings.optimizer_public_words import stash_python_error


def _density_save_array_layout(context: PythonObject) raises:
    # Metadata vectors use the connected Array allocation and slots. Inspect
    # every assignment target before creating a result whose initialization or
    # destruction could enter a subsequently replaced Python class hook.
    var cls = context["array_class"]
    var native_type = py_type(py_type(PythonObject(0)))
    if py_type(cls) is not native_type:
        raise Error("TypeError: density checkpoint storage requires its native Array class")
    var ancestry = py_call(native_type.__dict__["__mro__"].__get__, cls)
    var setter = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: fixed native Array class ancestry
        var fields = _pd_type_fields(ancestry[i])
        if py_contains(fields, "__del__"):
            raise Error("TypeError: density checkpoint storage cannot invoke Python finalizers")
        if setter is None:
            setter = py_get(fields, "__setattr__")
    if setter is not context["host_density_object_setattr"]:
        raise Error("TypeError: density checkpoint storage cannot invoke Python assignment hooks")
    var fields = _pd_type_fields(cls)
    var slots = context["owner_array_slots"]
    var names = py_call(py_type(Python.list()), slots)
    for i in range(len(names)):  # driver-loop: retained native storage descriptors
        if py_get(fields, names[i]) is not slots[names[i]]:
            raise Error("TypeError: density checkpoint storage requires its connected native Array slots")


def _density_save_attribute(model: PythonObject, name: String, context: PythonObject,
                            optional: Bool = False) raises -> PythonObject:
    try:
        return classical_getattribute(model, PythonObject(name), context)
    except error:
        if optional and String(error).startswith("AttributeError:"):
            return PythonObject(None)
        raise error^


def _density_save_text(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Valid kernel/metric names include native NumPy Unicode and benign
    # Unicode subclasses. Their contents are copied without __str__ hooks.
    return _kde_option_value(value, context)


def _density_save_mode(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = _density_save_attribute(model, "numeric_mode", context, True)
    if not _if_fit_truth(mode, context):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    var text = PythonObject(None)
    try:
        text = _density_save_text(mode, context)
    except error:
        if not String(error).startswith("TypeError:"):
            raise error^
    if text is not None and py_type(mode) is not context["archive_bytes_type"] and py_type(mode) is not context["archive_bytearray_type"]:
        text = py_call(py_call(text.strip).lower)
        if String(text) == "identical" or String(text) == "fast":
            return text
    raise Error("mojolearn: cannot save invalid numeric_mode " + _if_fit_repr(mode, context))


def _density_save_vector(values: PythonObject, integer: Bool, context: PythonObject) raises -> PythonObject:
    # array.array is the incumbent Array.from_list storage constructor. Its
    # native overflow/index behavior is retained after admitting every item.
    _density_save_array_layout(context)
    for i in range(len(values)):  # driver-loop: fixed checkpoint metadata vector
        _density_option(values[i], context)
        if py_type(values[i]) is py_get(context, "ndarray"):
            host_inference_scalar(values[i], context)
        if py_type(values[i]) is context["array_class"]:
            raise Error("TypeError: checkpoint scalar metadata requires native scalar values")
    var envelope = Python.tuple("q" if integer else "d", values)
    var constructor = context["array_type"]
    ref cpy = Python().cpython()
    var pointer = cpy.PyObject_Call(constructor._obj_ptr, envelope._obj_ptr, PyObjectPtr())
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var store = PythonObject(from_owned=pointer)
    return classical_wrap_array(store, Python.tuple(len(values)), "<i8" if integer else "<f8", "C", context)


def _density_save_resolve_iforest(model: PythonObject, fields: PythonObject,
                                 plan: PythonObject, context: PythonObject) raises:
    var names = Python.tuple("_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth",
        "_max_features_mode", "_max_features_int", "_max_features_frac", "_contamination_auto",
        "_contamination", "_n_estimators", "_bootstrap", "_seed")
    var ancestry = plan["mro"]
    for i in range(len(ancestry)):  # driver-loop: parameter resolution uses the native assignment contract
        var setter = py_get(_pd_type_fields(ancestry[i]), "__setattr__")
        if setter is not None:
            if setter is not context["host_density_object_setattr"]:
                raise Error("TypeError: density parameter resolution cannot invoke Python assignment hooks")
            break
    for i in range(len(names)):  # driver-loop: admit every resolved owner before any replacement
        var pair = _density_member(names[i], plan)
        var data = _density_descriptor(names[i], pair[0], Bool(py=pair[1]), plan, context)
        var previous = _density_get_slot(model, names[i], context) if data else Python.tuple(py_get(fields, names[i]), py_contains(fields, names[i]))
        if Bool(py=previous[1]):
            native_release_admit(previous[0], context)
    # The existing resolver records each completed assignment even when a
    # later option fails. Publish precisely those writes through admitted
    # native dict/member slots, preserving ordinary slotted subclasses too.
    var staged = Python.dict()
    try:
        _if_fit_options(model, staged, context)
    finally:
        ref cpy = Python().cpython()
        for i in range(len(names)):  # driver-loop: resolved parameter assignments and partial failure state
            if py_contains(staged, names[i]):
                if cpy.lib.get_function[Int32]("PyObject_GenericSetAttr")(model._obj_ptr, names[i]._obj_ptr, staged[names[i]]._obj_ptr) != 0:
                    var failure = stash_python_error()
                    raise failure^


def host_density_save(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = host_tree_constructor_fields(model, context)
    var family = String(args[0])
    var plan = context["classical_public_layouts"][py_type(model)]
    var known = False
    var mro = plan["mro"]
    for i in range(len(mro)):  # driver-loop: connected explicit CPU estimator ancestry
        var kind = py_get(context["host_inference_kinds"], mro[i])
        known = known or (kind is not None and String(kind) == family)
    if not known or (family != "kde" and family != "iforest"):
        raise Error("TypeError: density checkpoint requires its connected native host class")
    var x = PythonObject(None)
    try:
        x = _density_save_attribute(model, "_x", context)
    except error:
        if String(error).startswith("AttributeError:"):
            raise Error("RuntimeError: this estimator is not fitted yet")
        raise error^
    if family == "iforest" and not py_contains(fields, "_seed"):
        _density_save_resolve_iforest(model, fields, plan, context)
    var values = Python.dict()
    py_setitem(values, "format", "mojolearn-kde-1" if family == "kde" else "mojolearn-iforest-1")
    # Canonical family names let both host_model and class.load consume files
    # saved from explicit CPU classes or their ordinary registered subclasses.
    py_setitem(values, "estimator", "KernelDensity" if family == "kde" else "IsolationForest")
    py_setitem(values, "numeric_mode", _density_save_mode(model, context))
    py_setitem(values, "x", x)
    if family == "kde":
        py_setitem(values, "kernel", _density_save_text(_density_save_attribute(model, "kernel", context), context))
        py_setitem(values, "metric", _density_save_text(_density_save_attribute(model, "metric", context), context))
        var bandwidth = _if_fit_number(_density_save_attribute(model, "bandwidth", context), context)
        py_setitem(values, "bandwidth", _density_save_vector(Python.list(bandwidth), False, context))
        var features = _if_fit_number(_density_save_attribute(model, "n_features_in_", context), context, True)
        var rows = _if_fit_number(_density_save_attribute(model, "n_samples_fit_", context), context, True)
        var weights = _density_save_attribute(model, "_w", context)
        py_setitem(values, "meta", _density_save_vector(Python.list(features, rows, 0 if weights is None else 1), True, context))
        if weights is not None:
            py_setitem(values, "weights", weights)
    else:
        var integers = Python.list()
        var names = Python.tuple("_n_estimators", "_max_samples_mode", "_max_samples_int", "_max_depth",
            "_max_features_mode", "_max_features_int", "_bootstrap", "_seed", "_contamination_auto",
            "n_features_in_", "max_samples_")
        for i in range(len(names)):  # driver-loop: the eleven incumbent integer archive members
            var value = _density_save_attribute(model, String(names[i]), context)
            if i == 0 or i == 9 or i == 10:
                value = _if_fit_number(value, context, True)
            elif i == 6 or i == 8:
                value = PythonObject(1 if _if_fit_truth(value, context) else 0)
            _ = py_call(integers.append, value)
        py_setitem(values, "ints", _density_save_vector(integers, True, context))
        var reals = Python.list()
        var real_names = Python.tuple("_max_samples_frac", "_max_features_frac", "_contamination", "offset_")
        for i in range(len(real_names)):  # driver-loop: the four incumbent real archive members
            _ = py_call(reals.append, _if_fit_number(_density_save_attribute(model, String(real_names[i]), context), context))
        py_setitem(values, "reals", _density_save_vector(reals, False, context))
    var result = classical_archive_write(args[1], values, context)
    _ = x^
    _ = values^
    return result
