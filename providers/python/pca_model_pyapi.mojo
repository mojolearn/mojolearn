# SPDX-License-Identifier: Apache-2.0
# Operation-local PCA query settings. Copies hold admitted references; scalar
# storage is decoded before execution, without NumPy/user conversion slots.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission import sequential_admit_value, _one_of
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.native_scalar_decode import native_scalar_buffer_dtype
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.native_loaded_module import native_loaded_module
from providers.python.classical_family_admission_pyapi import classical_family_admit


def _native_number(value: PythonObject, context: PythonObject, truth: Bool) raises -> PythonObject:
    var cls = py_type(value)
    if truth and (cls is context["native_bytearray_type"] or cls is py_type(Python.list()) or cls is py_type(Python.tuple()) or cls is py_type(Python.dict())):
        # Native container truth is its length, not its contents. Freeze it
        # before any compiled reader/input work can release the GIL; the
        # original reference remains in the operation's scalar owner ledger.
        return PythonObject(len(value) != 0)
    # Exact builtin values have no application conversion hooks. Retain
    # their established Int/Bool behavior, including primitive error cases.
    if value is None or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject("")) or cls is context["native_bytes_type"] or cls is context["native_bytearray_type"]:
        return value
    var array = cls is py_get(context, "ndarray")
    var scalar = _one_of(cls, context["scalar_types"])
    if not array and not scalar and py_get(context, "ndarray") is not None:
        var constructor = native_numpy_dtype(context)
        for code in ["q", "Q", "g", "F", "D", "G"]:  # small-loop(native scalar ABI aliases)
            scalar = scalar or cls is py_call(constructor, code).type
    if not array and not scalar:
        raise Error("TypeError: PCA settings require native values without Python conversion hooks")
    var dtype = value.dtype
    var kind = String(dtype.kind)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "c":
        raise Error("TypeError: PCA scalar settings require primitive numeric storage without Python element hooks")
    if array:
        if truth:
            var count = Int(py=value.size)
            if count == 0:
                raise Error("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")
            if count != 1:
                raise Error("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
        elif Int(py=value.ndim) != 0 or (kind != "i" and kind != "u"):
            # Int(py=...) consumes an index, not float()/int() coercion of
            # an arbitrary array. Preserve its integer scalar-array rule.
            raise Error("only integer scalar arrays can be converted to a scalar index")
    var decoded = native_scalar_buffer_dtype(value, dtype)
    if truth:
        return PythonObject(decoded.truth())
    if kind == "c":
        raise Error("complex scalar cannot be interpreted as an integer")
    return decoded.to_python_number(kind == "i" or kind == "u" or kind == "b")


def pca_query_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = classical_family_admit(model)
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
    var scalar_owners = Python.list()
    # Preserve the PCA owner's existing admission of all declared settings
    # and fitted fields, including fields this particular query will not use.
    var names = context["pca_parameters"] + context["pca_state"]
    for index in range(len(names)):  # driver-loop: connected field inventory, private snapshot only
        if py_contains(fields, names[index]):
            sequential_admit_value(fields[names[index]], context)
    for name in ["_mojolearn_fitted", "_mojolearn_fitting"]:  # small-loop(fitted lifecycle flags)
        if py_contains(fields, name):
            _ = py_call(scalar_owners.append, fields[name])
            py_setitem(fields, name, _native_number(fields[name], context, True))
    var fitted = Bool(py=py_get(fields, "_mojolearn_fitted", False)) or Bool(py=py_get(fields, "_mojolearn_fitting", False))
    if not fitted:
        var keys = py_call(py_type(Python.list()), fields)
        for index in range(len(keys)):  # driver-loop: existing public fitted-attribute compatibility rule
            var key = String(keys[index])
            if key.endswith("_") and not key.startswith("_") and fields[key] is not None:
                fitted = True
                break
    if not fitted:
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This PCA instance is not fitted yet. Call 'fit' with appropriate arguments first.")
    for name in ["n_samples_", "n_features_in_", "n_components_", "whiten"]:  # small-loop(query scalar snapshot)
        if py_contains(fields, name):
            _ = py_call(scalar_owners.append, fields[name])
            py_setitem(fields, name, _native_number(fields[name], context, name == "whiten"))
    # PCA queries historically select the current numeric_mode, rather than
    # numeric_mode_. Capture the process default now when it is requested.
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_pca_query_scalar_owners", scalar_owners)
    return fields


def _entry_dictionary(value: PythonObject) raises -> PythonObject:
    if py_type(value) is not py_type(Python.dict()):
        raise Error("TypeError: PCA binding tables require native dictionaries")
    var keys = py_call(py_type(Python.list()), value)
    for index in range(len(keys)):  # driver-loop: string-key admission before keyed table access
        if py_type(keys[index]) is not py_type(PythonObject("")):
            raise Error("TypeError: PCA binding tables require native string keys")
    return py_call(py_type(Python.dict()).__dict__["copy"], value)


def pca_query_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var mode = py_call(py_call(fields["numeric_mode"].strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    var bindings = _entry_dictionary(context["pca_bindings"])
    var selected = bindings[mode]
    if py_type(selected) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(selected))
    var entries = _entry_dictionary(selected)
    var records = context["pca_query_connections"]
    if py_type(records) is not py_type(Python.tuple()):
        raise Error("TypeError: PCA query connections require their import-time tuple")
    var connected = PythonObject(None)
    for index in range(len(records)):  # driver-loop: the two import-connected numeric modes
        var record = records[index]
        if py_type(record) is not py_type(Python.tuple()) or len(record) != 2 or py_type(record[0]) is not py_type(PythonObject("")) or py_type(record[1]) is not py_type(Python.tuple()) or len(record[1]) != 6:
            raise Error("TypeError: PCA query connections require native mode/entry records")
        if String(record[0]) == String(mode):
            connected = record[1]
    if connected is None:
        raise Error("TypeError: PCA query mode was not connected at import")
    var names = Python.tuple("estimators_vendor", "estimators_numeric_mode", "pca_transform", "inverse_transform",
                             "pca_whiten_transform", "pca_whiten_inverse_transform")
    var builtin_type = py_type(native_loaded_module("builtins").len)
    for index in range(6):  # small-loop(all consumed compiled entries, before either reader executes)
        var entry = py_get(entries, names[index])
        if index >= 4 and entry is None:
            # Missing or explicitly None whitening exports keep the existing
            # optional-pair refusal at pca_whiten_entries when whitening runs.
            continue
        var registration = connected[index]
        if py_type(registration) is not py_type(Python.tuple()) or len(registration) != 3:
            raise Error("TypeError: PCA query requires its import-connected entry record")
        if entry is None or py_type(entry) is not builtin_type or entry is not registration[0]:
            raise Error("TypeError: PCA query requires its connected compiled entry " + String(names[index]))
        # Preserve the exact native function's import-time owner/name. Some
        # legitimate bindings may use None owners or exported aliases; neither
        # requires the universal ModuleType/export-key assumptions.
        if entry.__self__ is not registration[1] or py_type(registration[2]) is not py_type(PythonObject("")) or String(entry.__name__) != String(registration[2]):
            raise Error("TypeError: PCA query compiled entry metadata changed after import")
    if String(py_call(entries["estimators_vendor"])) != String(context["vendor"]):
        raise Error("RuntimeError: PCA native vendor differs from selected execution")
    if Int(py=py_call(entries["estimators_numeric_mode"])) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: PCA native mode differs from selected execution")
    return entries
