# SPDX-License-Identifier: Apache-2.0
# Normalizer operations own their complete previously decorated public calls.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields, _truth
from bindings.params_metadata_native import _resolved, _value
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import class_record, class_admit, class_watch_ready
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.optimizer_public_words import stash_python_error
from bindings.metrics_native_support import metric_warn
from core.preprocessing_program import prep_program_stage as _stage
from bindings.x_prep_native_owner import xprep_storage, xprep_take, xprep_run
from bindings.parallel_public_result import parallel_public_result
from bindings.prep_entry_cache import prep_mode_checked
from providers.python.prep_transform_pyapi import prep_transform_snapshot, prep_transform_values
from core.shared_slot import SharedSlot


def _setting(model: PythonObject, fields: PythonObject, name: String, fallback: PythonObject) raises -> PythonObject:
    var missing = Python.list()
    var key = PythonObject(name)
    if not py_contains(fields, key) and _resolved(py_type(model), key, missing) is missing:
        return fallback
    return _value(model, fields, key, Python.dict(missing=missing), "AttributeError")


def _class_setting(model: PythonObject, name: String, fallback: PythonObject) raises -> PythonObject:
    var missing = Python.list()
    var value = _resolved(py_type(model), PythonObject(name), missing)
    return fallback if value is missing else value


def _fitted(fields: PythonObject) raises -> Bool:
    if _truth(py_get(fields, "_mojolearn_fitted")) or _truth(py_get(fields, "_mojolearn_fitting")):
        return True
    var b = native_loaded_module("builtins")
    var names = py_get(fields, "_mojolearn_params_reset")
    if names is None:
        names = py_call(b.list, fields)
    elif py_type(names) is not b.tuple:
        raise Error("TypeError: native parameter reset metadata requires an ordinary tuple")
    for i in range(len(names)):  # driver-loop: native fitted field names
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native parameter reset metadata requires exact string names")
        var key = String(names[i])
        if key.endswith("_") and not key.startswith("_") and py_contains(fields, names[i]) and fields[names[i]] is not None:
            return True
    return False


def _repr_admit(value: PythonObject, depth: Int = 0) raises:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex:
        return
    if depth < 64 and (cls is b.list or cls is b.tuple):
        for i in range(len(value)):  # driver-loop: admit native invalid-setting repr before formatting
            _repr_admit(value[i], depth + 1)
        return
    if depth < 64 and cls is b.dict:
        var pairs = py_call(b.list, value.items())
        for i in range(len(pairs)):  # driver-loop: native invalid-setting keys/values before repr
            _repr_admit(pairs[i][0], depth + 1)
            _repr_admit(pairs[i][1], depth + 1)
        return
    raise Error("TypeError: native Normalizer settings cannot invoke Python representation or conversion hooks")


def _safe_repr(value: PythonObject) raises -> String:
    _repr_admit(value)
    return String(py_call(native_loaded_module("builtins").repr, value))


def _tags(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var tags = _class_setting(model, "_sklearn_input_tags", Python.dict())
    if py_type(tags) is not b.dict:
        raise Error("TypeError: native Normalizer input tags require an ordinary dictionary")
    var names = py_call(b.list, tags)
    for i in range(len(names)):  # driver-loop: static native tag names before lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native Normalizer input tags require exact string keys")
    return py_call(b.dict, tags)


def _tags_of(value: PythonObject, unset: PythonObject) raises -> PythonObject:
    # `_tags` on a class value already resolved by `_class_settings`: the
    # same exact-dict and exact-string-key admission (native key iteration)
    # and the same copy, read from the live class dict every call.
    var b = native_loaded_module("builtins")
    var tags = Python.dict() if value is unset else value
    if py_type(tags) is not b.dict:
        raise Error("TypeError: native Normalizer input tags require an ordinary dictionary")
    if not native_namespace_string_keys(tags._obj_ptr, b.str):
        raise Error("TypeError: native Normalizer input tags require exact string keys")
    return py_call(b.dict, tags)


comptime _CLASS_SETTINGS_SLOT = "prep_normalizer_class_settings_v1"
comptime _CLASS_SETTING_NAMES = 7


def _class_settings_names() raises -> PythonObject:
    # The class-level names one Normalizer transform reads, in this order.
    return Python.tuple("_sklearn_requires_fit", "_sklearn_input_tags", "_mojolearn_allows_nan",
                        "metric", "kernel", "affinity", "dissimilarity")


def _plain_setting(value: PythonObject, b: PythonObject) raises -> Bool:
    # A resolved class value a record may hold: plain values only, so the
    # record never retains a function (its globals reach the class) or a
    # descriptor.
    var cls = py_type(value)
    return value is None or cls is b.bool or cls is b.int or cls is b.str or cls is b.dict


def _class_settings(model: PythonObject) raises -> PythonObject:
    """(missing, the seven class-resolved values of `_class_settings_names`,
    each the `missing` object when no ancestry row defines it).

    The MRO walks behind `_class_setting` / `_setting` ran seven times per
    transform. Their class half depends only on the class rows, so it is
    kept per class in the class-admission registry (bindings/
    native_class_admission.mojo, type-watcher epoch: any assignment to the
    class or an ancestor drops it; Python 3.10/3.11 have no record and walk
    every call, as before) (perf2-prep). Only plain values are kept; any
    other resolved value is not recorded, and the walk runs every call."""
    var mods = NativeModules()
    var cls = py_type(model)
    var record = class_record(mods, _CLASS_SETTINGS_SLOT, cls)
    if record is not None:
        return record
    var b = mods.module("builtins")
    var missing = Python.list()
    var names = _class_settings_names()
    var out = Python.list()
    _ = py_call(out.append, missing)
    var plain = True
    for i in range(len(names)):  # small-loop(metadata: the seven fixed class setting names)
        var value = _resolved(mods, cls, names[i], missing)
        plain = plain and (value is missing or _plain_setting(value, b))
        _ = py_call(out.append, value)
    var payload = py_call(b.tuple, out)
    # The payload is the result either way; the record is kept only where
    # type watchers exist (not on Python 3.10/3.11).
    if plain and class_watch_ready():
        class_admit(mods, _CLASS_SETTINGS_SLOT, cls, payload)
    return payload


def _instance_setting(model: PythonObject, fields: PythonObject, settings: PythonObject, names: PythonObject,
                      index: Int, fallback: PythonObject) raises -> PythonObject:
    # `_setting` on a recorded class value: the instance field wins, then the
    # class value through `_value` (descriptor checks), else `fallback`.
    # `names`: `_class_settings_names()`, built once per transform by the
    # caller instead of once per setting (pass2-prep-metrics).
    var key = names[index]
    if not py_contains(fields, key) and settings[index + 1] is settings[0]:
        return fallback
    return _value(model, fields, key, Python.dict(missing=Python.list()), "AttributeError")


struct _PinnedEntries(Defaultable, Movable):
    # The import-time entry identities of one numeric mode as a name -> entry
    # dict, built once from the immutable pins tuple it came from.
    var pins: PythonObject
    var mode: String
    var expected: PythonObject

    def __init__(out self):
        self.pins = PythonObject(None)
        self.mode = String("")
        self.expected = PythonObject(None)


comptime _PINNED_ENTRIES = SharedSlot[_PinnedEntries, "MojolearnPrepNormalizerPinnedEntriesV2"]


def _expected_entries(pins: PythonObject, mode: PythonObject, b: PythonObject) raises -> PythonObject:
    # `pins` is the immutable tuple of immutable (mode, ((name, entry), ...))
    # records the bridge froze at import. The name -> entry dict built from
    # it is kept with a strong reference to that tuple and reused while the
    # context still holds the same tuple object (an identity compare), so
    # the record walk and the dict build run once per process and mode, not
    # once per operation (perf2-prep). The dict never leaves this module.
    var g = _PINNED_ENTRIES.get_or_create_ptr()
    var text = String(mode)
    if g[].pins is pins and g[].mode == text and g[].expected is not None:
        return g[].expected
    if py_type(pins) is not b.tuple:
        raise Error("TypeError: native Normalizer requires its immutable import-time entry identities")
    var selected = PythonObject(None)
    for i in range(len(pins)):  # small-loop(ABI: connected numeric modes)
        var record = pins[i]
        if py_type(record) is not b.tuple or len(record) != 2 or py_type(record[0]) is not b.str:
            raise Error("TypeError: native Normalizer binding identity metadata is invalid")
        if String(record[0]) == text:
            selected = record[1]
    if py_type(selected) is not b.tuple:
        raise Error("TypeError: native Normalizer requires the originally connected compiled entries")
    var expected_entries = Python.dict()
    for i in range(len(selected)):  # driver-loop: immutable imported entry references, never name/module inference
        var pair = selected[i]
        if py_type(pair) is not b.tuple or len(pair) != 2 or py_type(pair[0]) is not b.str:
            raise Error("TypeError: native Normalizer entry identity metadata is invalid")
        py_setitem(expected_entries, pair[0], pair[1])
    g[].pins = pins
    g[].mode = text
    g[].expected = expected_entries
    return expected_entries


def _native_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(mode) is not b.str or (String(mode) != "identical" and String(mode) != "fast"):
        raise Error("numeric_mode must be fast, identical or None")
    var tables = context["scaler_bindings"]
    if py_type(tables) is not b.dict:
        raise Error("TypeError: native Normalizer bindings require an ordinary dictionary")
    # Exact string keys are admitted by native dict iteration (no Python key
    # list, no per-key call), before any keyed lookup can compare a key.
    if not native_namespace_string_keys(tables._obj_ptr, b.str):
        raise Error("TypeError: native Normalizer binding keys require exact strings")
    var entries = tables[mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    if py_type(entries) is not b.dict:
        raise Error("TypeError: native Normalizer entries require an ordinary dictionary")
    if not native_namespace_string_keys(entries._obj_ptr, b.str):
        raise Error("TypeError: native Normalizer entry keys require exact strings")
    # Retain the compiled entries once before any entry executes: one C-level
    # copy of the exact dict (it neither hashes nor compares keys).
    var captured = py_call(b.dict, entries)
    var expected_entries = _expected_entries(context["prep_stateless_entry_identities"], mode, b)
    var names = Python.tuple("preprocessing_numeric_mode", "x_prep_numeric_mode", "x_prep_run",
        "x_prep_run_out", "x_prep_run_scratch", "x_prep_run_ranges", "x_prep_dev_put", "x_prep_dev_free",
        "x_prep_dev_live", "x_prep_dev_take_rows", "arena_binding", "arena_binding_id")
    var builtin_type = py_type(b.len)
    for i in range(len(names)):  # small-loop(ABI: fixed connected numerical entry points)
        var entry = captured[names[i]] if i < 3 else py_get(captured, names[i])
        if entry is not py_get(expected_entries, names[i]):
            raise Error("TypeError: native Normalizer compiled entry identity changed after import")
        if i < 10 and entry is not None and py_type(entry) is not builtin_type:
            raise Error("TypeError: native Normalizer execution requires compiled binding entries; Python callbacks cannot execute")
    if py_type(captured["arena_binding_id"]) is not b.int:
        raise Error("TypeError: native Normalizer arena binding identity must remain a native integer")
    if _truth(py_get(context, "scaler_explicit_host", False)) and String(mode) != "identical":
        raise Error("HostStandardScaler runs identical only on the host")
    var expected = 1 if String(mode) == "identical" else 0
    # The entries were matched to the import-time identities above; their two
    # mode probes run once per binding (prep_entry_cache).
    if not prep_mode_checked(entries, expected):
        raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
    return captured


def _normalizer_norm(norm: PythonObject) raises -> Int:
    var b = native_loaded_module("builtins")
    var code = -1
    if py_type(norm) is b.str:
        var word = String(norm)
        code = 0 if word == "l1" else (1 if word == "l2" else (2 if word == "max" else -1))
    if code < 0:
        raise Error("mojolearn: invalid norm " + _safe_repr(norm))
    return code


def _normalizer_apply(values: PythonObject, n: Int, d: Int, code: Int,
                      entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # Same normalization stage, arena operands, large-output rule and native
    # transport as the preexisting _transform branch. No model math moves.
    var words = n * d
    var dev_out = words >= 134217728 and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
    var host = words + (0 if dev_out else words)
    if host + (words if dev_out else 0) > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var arena = xprep_storage(host, context)
    var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
    var program = List[Int32]()
    _stage(program, 47, n, 0, n, d, code, host if dev_out else words)
    var inputs = Python.list(Python.tuple(0, Python.tuple(0, values, PythonObject(None), n, d)))
    xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
    var result = xprep_take(output_store[0] if dev_out else arena[0], 0 if dev_out else words, n, d, context["output_context"])
    _ = values
    _ = arena^
    return result


def normalizer_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    # _fields admits an exact native dictionary and exact string keys. Its
    # native copy pins operation-local state before input preparation or a
    # compiled readback can release the GIL. Transform never writes the model.
    var fields = py_call(b.dict, _fields(model))
    var missing = Python.list()
    var mode = py_get(fields, "numeric_mode_", missing)
    var norm = py_get(fields, "norm", missing)
    var fitted = py_get(fields, "n_features_in_")
    var feature_names = py_get(fields, "feature_names_in_")
    var name = String(py_type(model).__name__)
    var settings = _class_settings(model)
    var unset = settings[0]
    var requires_fit = _truth(PythonObject(True) if settings[1] is unset else settings[1])
    if requires_fit and not _fitted(fields):
        var category = classical_not_fitted_class(context)
        var message = "This " + name + " instance is not fitted yet. Call 'fit' with appropriate arguments before using transform."
        ref cpy = Python().cpython()
        cpy.PyErr_SetString(category._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
        var failure = stash_python_error()
        raise failure^
    var nested = _truth(py_get(fields, "_mojolearn_fitting"))
    var tags = _tags_of(settings[2], unset)
    var validate = not nested and _truth(py_get(tags, "two_d_array", True)) and not _truth(py_get(tags, "one_d_array", False))
    var scan = not _truth(py_get(tags, "allow_nan", False))
    var setting_names = _class_settings_names()
    # A Python hook used solely to decide validation is still runtime work.
    if not nested and scan and _instance_setting(model, fields, settings, setting_names, 2,
                                                 PythonObject(None)) is not None:
        raise Error("TypeError: native Normalizer validation requires static input tags; Python allow-NaN hooks cannot execute")
    var precomputed = False
    for k in range(3, 7):  # small-loop(metadata: existing width-check exemptions metric/kernel/affinity/dissimilarity)
        var option = _instance_setting(model, fields, settings, setting_names, k, PythonObject(None))
        precomputed = precomputed or (py_type(option) is b.str and String(option) == "precomputed")
    var sparse = _truth(py_get(tags, "sparse", False))
    var snapshot = prep_transform_snapshot(source, name, validate, scan, context, sparse, not nested)
    var numeric = (snapshot.dtype_number >= 0 and snapshot.dtype_number <= 13) or snapshot.dtype_number == 23 or snapshot.dtype_number == -103
    if validate and numeric and not precomputed and len(snapshot.shape) == 2:
        if py_type(fitted) is b.int and Int(py=fitted) != snapshot.source.cols:
            raise Error("X has " + String(snapshot.source.cols) + " features, but " + name + " is expecting " + String(Int(py=fitted)) + " features as input.")
        # The former _array_like conversion removed pandas column labels before
        # before_call's _column_names check. Preserve that public behavior.
        if feature_names is not None:
            metric_warn("X does not have valid feature names, but " + name + " was fitted with feature names", b.UserWarning, context)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + name + " instance is not fitted yet")
    if mode is missing:
        # Preserve the former missing-field error at its original stage.
        mode = fields["numeric_mode_"]
    var entries = _native_entries(mode, context)
    var values = prep_transform_values(snapshot, context)
    var n = snapshot.source.rows
    var d = snapshot.source.cols
    if py_type(fitted) is not b.int or Int(py=fitted) != d:
        _repr_admit(fitted)
        var text = String(py_call(b.str, fitted))
        raise Error("mojolearn: X has " + String(d) + " features, " + name + " was fitted with " + text)
    if norm is missing:
        norm = fields["norm"]
    var result = _normalizer_apply(values, n, d, _normalizer_norm(norm), entries, context)
    _ = values^
    _ = snapshot^
    return parallel_public_result(result, "transform", PythonObject(None), context)
