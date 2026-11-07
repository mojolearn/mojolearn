# SPDX-License-Identifier: Apache-2.0
# Operation-local PCA query settings. Copies hold admitted references; scalar
# storage is decoded before execution, without NumPy/user conversion slots.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission import sequential_admit_value, _one_of
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.native_scalar_decode import native_scalar_buffer_dtype
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.native_loaded_module import native_loaded_module, NativeModules
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.pca_fit_scalar_pyapi import PCAFitScalar, pca_fit_scalar_admit
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_key_same
from bindings.native_model_records import records_open, model_generation, records_key_append


def _native_number(value: PythonObject, context: PythonObject, truth: Bool,
                   operation: String = "transform") raises -> PythonObject:
    if truth:
        var captured = PCAFitScalar(value, context, True, operation)
        return PythonObject(captured.truth(operation, context))
    var cls = py_type(value)
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
    if array and (Int(py=value.ndim) != 0 or (kind != "i" and kind != "u")):
        # Query dimensions use the integer-index contract, not float()/int()
        # coercion of an arbitrary scalar array.
        raise Error("only integer scalar arrays can be converted to a scalar index")
    var decoded = native_scalar_buffer_dtype(value, dtype)
    if kind == "c":
        raise Error("complex scalar cannot be interpreted as an integer")
    return decoded.to_python_number(kind == "i" or kind == "u" or kind == "b")


def pca_query_fields(model: PythonObject, context: PythonObject, operation: String = "transform") raises -> PythonObject:
    return _pca_query_fields_from(classical_family_admit(model), context, operation)


def _pca_query_fields_from(state: PythonObject, context: PythonObject, operation: String) raises -> PythonObject:
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
    var scalar_owners = Python.list()
    # Preserve the PCA owner's existing admission of all declared settings
    # and fitted fields, including fields this particular query will not use.
    var names = context["pca_parameters"] + context["pca_state"]
    # Row N262: one module view for the field admissions of this query, not
    # one NativeModules per value; a scalar admission (NumPy's scalar
    # protocol) is followed by released(), so the next value re-checks.
    var mods = NativeModules()
    for index in range(len(names)):  # driver-loop: connected field inventory, private snapshot only
        if py_contains(fields, names[index]):
            var name = String(names[index])
            if index < len(context["pca_parameters"]) or name == "_mojolearn_fitted" or name == "_mojolearn_fitting":
                pca_fit_scalar_admit(fields[names[index]], context)
                mods.released()
            else:
                sequential_admit_value(mods, fields[names[index]], context)
    for name in ["_mojolearn_fitted", "_mojolearn_fitting"]:  # small-loop(fitted lifecycle flags)
        if py_contains(fields, name):
            _ = py_call(scalar_owners.append, fields[name])
            py_setitem(fields, name, _native_number(fields[name], context, True, operation))
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
            py_setitem(fields, name, _native_number(fields[name], context, name == "whiten", operation))
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


# Per-model query record (lane perf2-pca-decomp). A transform used to re-admit
# all 22 connected settings and fitted fields (scalar parser, NumPy dtype
# builds, generic admission), decode six scalars, copy and re-check both
# binding tables and call the two compiled getters on every call. Those steps
# read only the identities and exact types recorded below, so while the record
# matches they would accept again and produce the same decoded values and
# entries. Class admission (classical_family_admit), the X snapshot and the
# fitted-array shape/dtype/finite snapshots still run on every call.
# The record notices: set_params or any assignment or deletion of one of the
# 22 fields (refit replaces them), a changed process default mode while
# numeric_mode is None, replaced connected tables or a replaced selected entry.
# It does not notice in-place edits of the private connected context objects
# other than the selected mode's entry table, which nothing public reaches.
comptime _PCA_QUERY_SLOT = "pca_query_v1"


def _pca_query_decoded(name: String) -> Bool:
    # The six fields whose value (not only type) the query decodes.
    return (name == "_mojolearn_fitted" or name == "_mojolearn_fitting" or name == "n_samples_"
            or name == "n_features_in_" or name == "n_components_" or name == "whiten")


def _pca_query_plain(value: PythonObject, ndarray: PythonObject, decoded: Bool) raises -> Bool:
    # Admission and decoding of an exact builtin scalar depend on its identity
    # only (immutable). An exact ndarray is admitted by type alone; its shape,
    # dtype and values are read again by the per-call state snapshots. An
    # ndarray among the decoded scalars can change value in place: never
    # recorded. Anything else (NumPy scalars, user types) takes the full path.
    var cls = py_type(value)
    if (value is None or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0))
            or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(""))):
        return True
    return not decoded and cls is ndarray


def _pca_query_key(state: PythonObject, context: PythonObject) raises -> PythonObject:
    var ndarray = py_get(context, "ndarray")
    if ndarray is None or py_type(state) is not py_type(Python.dict()):
        return PythonObject(None)
    var parameters = context["pca_parameters"]
    var stored = context["pca_state"]
    var key = Python.list(context, ndarray, parameters, stored, py_get(context, "vendor"),
                          context["pca_bindings"], context["pca_query_connections"])
    for group in range(2):  # small-loop(connected settings, then fitted state)
        var names = parameters if group == 0 else stored
        for index in range(len(names)):  # driver-loop: connected PCA field inventory, identity only
            var name = names[index]
            if not py_contains(state, name):
                _ = py_call(key.append, False)
                continue
            var value = state[name]
            if not _pca_query_plain(value, ndarray, _pca_query_decoded(String(name))):
                return PythonObject(None)
            _ = py_call(key.append, True)
            _ = py_call(key.append, value)
    if py_get(state, "numeric_mode") is None:
        # The query then selects the process default mode, read per call.
        var backend = context["backend_fields"]
        _ = py_call(key.append, backend)
        _ = py_call(key.append, backend["_DEFAULT_MODE"])
    return key


def _pca_entries_key(context: PythonObject, mode: PythonObject) raises -> PythonObject:
    # The selected mode's binding table and the six entries the query
    # consumes (pca_query_entries' names), by identity (N316: every other
    # entry of the table was listed into the key before; none is read).
    var selected = py_get(context["pca_bindings"], mode)
    var key = Python.list(selected)
    if py_type(selected) is py_type(Python.dict()):
        var names = Python.tuple("estimators_vendor", "estimators_numeric_mode", "pca_transform", "inverse_transform",
                                 "pca_whiten_transform", "pca_whiten_inverse_transform")
        for index in range(6):  # small-loop(the six consumed compiled entry names)
            records_key_append(key, py_get(selected, names[index]))
    return key


def _pca_context_key(state: PythonObject, context: PythonObject) raises -> PythonObject:
    # The connected-context part of _pca_query_key (no per-field items):
    # what a generation word of the instance dictionary does not cover.
    var key = Python.list(context, py_get(context, "ndarray"), context["pca_parameters"], context["pca_state"],
                          py_get(context, "vendor"), context["pca_bindings"], context["pca_query_connections"])
    if py_get(state, "numeric_mode") is None:
        var backend = context["backend_fields"]
        records_key_append(key, backend)
        records_key_append(key, backend["_DEFAULT_MODE"])
    return key


def _pca_generation(model: PythonObject) raises -> Int:
    # The model's generation word (bindings/native_model_records.mojo), its
    # record entry opened first; 0 proves nothing.
    _ = records_open(model)
    return model_generation(model)


def pca_query_admitted(model: PythonObject, context: PythonObject, operation: String) raises -> PythonObject:
    """(fields, entries) of a PCA transform or inverse_transform: the full
    admission on first use and after any recorded change, else the record."""
    var state = classical_family_admit(model)
    var mods = NativeModules()
    # Generation path (N315, N316): the record was taken (or last matched)
    # at the model's generation word; an unchanged word proves the instance
    # dictionary, and so every per-field item of the key, unchanged. Only
    # the connected-context items and the six consumed entries are compared
    # then; the 22-field key is not built. A word of 0 proves nothing.
    var generation = _pca_generation(model)
    if generation != 0:
        var stamped = prepared_get(mods, _PCA_QUERY_SLOT, model)
        if (stamped is not None and len(stamped) == 7 and Int(py=stamped[5]) == generation
                and prepared_key_same(stamped[6], _pca_context_key(state, context))
                and prepared_key_same(stamped[3], _pca_entries_key(context, stamped[2]))):
            var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
            _ = py_call(py_type(Python.dict()).__dict__["update"], fields, stamped[1])
            return Python.tuple(fields, stamped[4])
    var key = _pca_query_key(state, context)
    if key is not None:
        var held = prepared_get(mods, _PCA_QUERY_SLOT, model)
        if held is not None and prepared_key_same(held[0], key) and prepared_key_same(held[3], _pca_entries_key(context, held[2])):
            if len(held) == 7 and generation != 0 and Int(py=held[5]) != generation:
                # Re-stamp at the current word (the dictionary changed and
                # changed back, or our own bookkeeping wrote it).
                prepared_put(mods, _PCA_QUERY_SLOT, model,
                             Python.tuple(held[0], held[1], held[2], held[3], held[4], PythonObject(generation),
                                          _pca_context_key(state, context)))
            var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
            _ = py_call(py_type(Python.dict()).__dict__["update"], fields, held[1])
            return Python.tuple(fields, held[4])
    var fields = _pca_query_fields_from(state, context, operation)
    if py_get(context, "ndarray") is None:
        raise Error("ImportError: mojolearn: PCA requires optional NumPy support. Install it with: "
                    + 'python -m pip install "mojolearn[numpy]" ("mojolearn[verify]" also includes NumPy).')
    var entries = pca_query_entries(fields, context)
    if key is not None:
        # Only a model fitted by its lifecycle flags is recorded: the
        # attribute-name fallback reads fields outside the key.
        var flagged = Bool(py=py_get(fields, "_mojolearn_fitted", False)) or Bool(py=py_get(fields, "_mojolearn_fitting", False))
        if flagged:
            var overlay = Python.dict()
            var decoded = Python.tuple("_mojolearn_fitted", "_mojolearn_fitting", "n_samples_", "n_features_in_",
                                       "n_components_", "whiten", "numeric_mode", "_pca_query_scalar_owners")
            for index in range(len(decoded)):  # small-loop(eight decoded query field names)
                if py_contains(fields, decoded[index]):
                    py_setitem(overlay, decoded[index], fields[decoded[index]])
            var mode = py_call(py_call(fields["numeric_mode"].strip).lower)
            prepared_put(mods, _PCA_QUERY_SLOT, model,
                         Python.tuple(key, overlay, mode, _pca_entries_key(context, mode), entries,
                                      PythonObject(generation), _pca_context_key(state, context)))
    return Python.tuple(fields, entries)
