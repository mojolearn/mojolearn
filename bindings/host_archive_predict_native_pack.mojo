# SPDX-License-Identifier: Apache-2.0
# `host_predict(path, X)` / `host_predict_proba(path, X)`: the saved model
# rebuilt by the native archive reader, then the compiled owner entry its
# own public method reaches, selected from fixed import-time recipes
# (python/mojolearn/_native_host_archive_predict.py). No public Python
# method, method wrapper, availability lambda or `_bind` runs here.
# The PACK half of the former `bindings/host_archive_predict_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/host_archive_predict_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, _loaded_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.family_common_unpack import family_repr
from bindings.family_common_pack import family_call
from bindings.host_archive_native_door import host_archive_build
from bindings.host_svc_native_door import host_svc_probability
from bindings.native_loaded_module_door import NativeModules
from bindings.native_class_admission_door import class_record, class_admit

from bindings.host_archive_predict_native_unpack import (
    _check_module,
    _class_name,
    _unavailable,
)


# Per class state and method: the verdict that the class ancestry is the
# connected one and the public method resolves to the connected descriptor
# (payload True, so the record never retains the class). The type watcher
# drops it on any change of the class or an ancestor and the full walk runs
# again (lane ne1-e-table, row N229). Python 3.10 / 3.11 keep the full walk.
comptime _ARCHIVE_PREDICT_SLOT = "host_archive_predict_descriptor_v1_"


def _archive_prediction_record(model: PythonObject, method: Int, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: saved host prediction requires an ordinary native class")
    var records = context["host_archive_predict_records"]
    for i in range(len(records)):  # driver-loop: over the fixed saved-model class inventory
        var record = records[i]
        if cls is not record[0]:
            continue
        var name = String("predict") if method == 0 else String("predict_proba")
        var key = PythonObject(name)
        var mods = NativeModules()
        var slot = String(_ARCHIVE_PREDICT_SLOT) + String(method)
        if class_record(mods, slot, cls) is None:
            var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
            if ancestry is not record[1]:
                raise Error("TypeError: saved host prediction class ancestry changed")
            # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
            var descriptor = Optional[PythonObject]()
            for k in range(len(ancestry)):  # driver-loop: over the class's method resolution order, without descriptors
                var fields = py_call(b.type.__dict__["__dict__"].__get__, ancestry[k])
                var keys = py_call(b.list, fields)
                for j in range(len(keys)):  # driver-loop: over class field names, admitted before any keyed lookup
                    if py_type(keys[j]) is not b.str:
                        raise Error("TypeError: saved host prediction requires native class field names")
                if py_contains(fields, key):
                    descriptor = fields[key]
                    break
            var expected = record[3][method]
            var replaced = (descriptor.value() is not expected) if descriptor else (expected is not None)
            if replaced:
                raise Error("TypeError: saved host prediction cannot invoke a replaced public method or descriptor")
            class_admit(mods, slot, cls, PythonObject(True))
        if record[4][method] is None:
            raise Error("AttributeError: '" + _class_name(record) + "' object has no attribute '" + name + "'")
        # Method availability formerly ran during attribute lookup, before
        # any argument preparation (available_if lambdas, the native SVC
        # descriptor). Keep that order and those refusals.
        var fields = py_fields(model)
        var cls_name = _class_name(record)
        if cls_name == "HostDBSCAN":
            var data = py_get(fields, "prediction_data")
            if data is None or not Bool(py=data):
                var unavailable = _unavailable(record, name)
                raise unavailable
        elif cls_name == "HostAgglomerativeClustering" or cls_name == "HostSpectralClustering":
            var data = py_get(fields, "prediction_data")
            var fit_x = py_get(fields, "_fit_X")
            if (data is None or not Bool(py=data)) and fit_x is None:
                var unavailable = _unavailable(record, name)
                raise unavailable
        elif cls_name == "HostSVC" and method == 1:
            var svc = context["host_archive_predict_svc_context"]
            host_svc_probability(model, fields, key, svc)
        return record
    raise Error("TypeError: saved host class has no connected native prediction schema")


def host_archive_predict(path: PythonObject, source: PythonObject, probability: Bool,
                         context: PythonObject) raises -> PythonObject:
    var model = host_archive_build(path, context, PythonObject(None), False)
    var method = 1 if probability else 0
    var record = _archive_prediction_record(model, method, context)
    var recipe = record[4][method]
    var b = native_loaded_module("builtins")
    if py_contains(context["host_archive_predict_mode_checked"], record[2]):
        # _HostBound._bind's numeric-mode refusal, which the saved class's
        # public method reached before reading X.
        var mode = py_get(py_fields(model), "numeric_mode")
        if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
            raise Error("mojolearn: " + _class_name(record) + " runs IDENTICAL only on the host; this model was saved " + family_repr(mode))
    if py_type(recipe[0]) is b.str:
        raise Error(String(recipe[0]))
    if py_type(recipe[0]) is not context["host_archive_predict_entry_type"]:
        raise Error("TypeError: saved host prediction requires a compiled native owner")
    var arguments = Python.list()
    var words = recipe[3]
    for i in range(len(words)):  # driver-loop: over the fixed positional argument recipe
        var word = words[i]
        var kind = String(word[0])
        if kind == "input":
            _ = py_call(arguments.append, source)
        elif kind == "constant":
            var value = word[1]
            if py_type(value) is b.str and String(value).startswith("ImportError:"):
                # A host binding that is not built or failed its read-back
                # at import: the refusal the old lazy `_bind` raised.
                raise Error(String(value))
            if py_type(value) is context["host_archive_predict_module_type"]:
                _check_module(value, context)
            _ = py_call(arguments.append, value)
        else:
            raise Error("TypeError: malformed saved host prediction recipe")
    var knn = py_get(context, "host_archive_predict_knn_binding")
    if String(recipe[1]).startswith("archive_knn_") and py_type(knn) is context["host_archive_predict_module_type"]:
        _check_module(knn, context)
    var result = family_call(recipe[0], model, recipe[1], py_call(b.tuple, arguments), recipe[2])
    _ = model^
    return result
