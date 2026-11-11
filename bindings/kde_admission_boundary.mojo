# SPDX-License-Identifier: Apache-2.0
"""Native admission records; Python owners here are boundary lifetime pins."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get, py_contains, py_fields, py_call
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_address
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from bindings.native_model_records import (
    ContentRecord, content_value, records_stamped_get, records_stamped_put, records_drop, RECORDS_PREPARED,
)
from core.kde_native_state import KDEAdmissionWords


struct KDEAdmissionOwner(Movable, Writable):
    var state: KDEAdmissionWords
    var pins: List[PythonObject]

    def __init__(out self, var words: List[Int], var pins: List[PythonObject]):
        self.state = KDEAdmissionWords(words^)
        self.pins = pins^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("KDEAdmissionOwner(", len(self.state.words), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _kde_metadata(mut words: List[Int], mut pins: List[PythonObject], value: PythonObject,
                  context: PythonObject) raises:
    var cls = py_type(value)
    var ndarray = py_get(context, "ndarray")
    var is_ndarray = ndarray is not None and cls is ndarray
    if not is_ndarray and cls is not context["array_class"]:
        return
    # Exact Array storage is admitted by classical_address before slot reads;
    # exact ndarray metadata is supplied by native NumPy descriptors.
    words.append(Int(py=classical_address(value)))
    var shape = value.shape
    var b = native_loaded_module("builtins")
    if py_type(shape) is not b.tuple:
        raise Error("TypeError: KDE fitted shapes require an exact tuple")
    words.append(len(shape))
    for axis in range(len(shape)):  # driver-loop: array metadata, never elements
        if py_type(shape[axis]) is not b.int:
            raise Error("TypeError: KDE fitted shape dimensions require exact integers")
        words.append(Int(py=shape[axis]))
    pins.append(value)
    if is_ndarray:
        var dtype = value.dtype
        pins.append(dtype)
        words.append(Int(dtype._obj_ptr))
        words.append(Int(Bool(py=value.flags.c_contiguous)))


def _kde_names(required: PythonObject, outer: Bool) raises -> List[String]:
    var names = List[String]()
    if outer:
        for i in range(len(required)):  # driver-loop: import-connected constructor schema
            names.append(String(required[i]))
        for name in ["input_copied_", "n_features_in_", "n_samples_fit_", "n_iter_", "_mojolearn_fitted",
                     "_mojolearn_fitting", "labels_", "components_", "core_sample_indices_", "_core_labels",
                     "_x", "_w", "feature_names_in_", "_resident"]:
            names.append(name)
    else:
        for name in ["bandwidth", "algorithm", "kernel", "metric", "atol", "rtol", "breadth_first",
                     "leaf_size", "n_features_in_", "n_samples_fit_", "metric_params", "_x", "_w"]:
            names.append(name)
    return names^


def _kde_words(model: PythonObject, names: List[String], context: PythonObject,
               mut words: List[Int], mut pins: List[PythonObject]) raises:
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var cls = py_type(model)
    words.append(Int(cls._obj_ptr))
    pins.append(cls)
    for i in range(len(names)):  # driver-loop: selected model metadata only
        var value = py_get(fields, names[i])
        _kde_metadata(words, pins, value, context)
        if names[i] == "metric_params" and py_type(value) is native_loaded_module("builtins").dict:
            var items = py_call(native_loaded_module("builtins").tuple, value.items())
            for j in range(len(items)):  # driver-loop: metric-parameter owner metadata, never numerical data
                _kde_metadata(words, pins, items[j][0], context)
                _kde_metadata(words, pins, items[j][1], context)


def kde_admission_hit(model: PythonObject, required: PythonObject, outer: Bool,
                      slot: String, context: PythonObject) raises -> Bool:
    var saved = records_stamped_get(RECORDS_PREPARED, model, slot)
    if saved is None:
        return False
    var words = List[Int]()
    var pins = List[PythonObject]()
    var names = _kde_names(required, outer)
    _kde_words(model, names, context, words, pins)
    var result = saved.downcast_value_ptr[KDEAdmissionOwner]()[].state.matches(words)
    _ = pins^
    return result


def kde_admission_remember(model: PythonObject, required: PythonObject, outer: Bool,
                           slot: String, context: PythonObject) raises:
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    if outer:
        for i in range(len(required)):  # driver-loop: inherited defaults cannot be stamped by an instance generation
            if not py_contains(fields, required[i]):
                return
    var names = _kde_names(required, outer)
    var content = ContentRecord()
    for i in range(len(names)):  # driver-loop: selected setting identities and nested native containers
        # The resident owner is native, immutable from Python, and its
        # top-level identity is proven by the model generation. Do not pin it
        # in an admission cache: explicit release must drop idle device data.
        if names[i] != "_resident":
            content_value(content, py_get(fields, names[i]), 0, 4, False)
    if not content.ok:
        return
    for i in range(len(content.held)):  # driver-loop: no model-to-record strong reference cycle
        if content.held[i] is model:
            return
    var words = List[Int]()
    var pins = List[PythonObject]()
    _kde_words(model, names, context, words, pins)
    var owner = KDEAdmissionOwner(words^, pins^)
    _ = records_stamped_put(RECORDS_PREPARED, model, slot, content^, PythonObject(alloc=owner^))


def kde_admission_drop(model: PythonObject, slot: String) raises:
    records_drop(RECORDS_PREPARED, model, slot)
