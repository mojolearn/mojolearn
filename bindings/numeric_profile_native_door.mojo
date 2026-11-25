# SPDX-License-Identifier: Apache-2.0
# Complete public profile metadata/default ownership. Import time connects exact
# module dictionaries and the existing shared provider defaults. Product calls
# invoke no Python profile helpers, environment methods or input protocols.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_item
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from bindings.optimizer_public_words_unpack import optimizer_python_error, stash_python_error, PYTHON_ERROR_PENDING
from bindings.backend_public_native_door import backend_public_operation
from core.shared_slot import SharedSlot
from std.sys import is_defined

# SABOTAGE ARM MOJOLEARN_NUMERIC_PROFILE_SABOTAGE_DROP_SAVED (default off; verify's sabotage registry
# lists it with lane numeric_profile_parts, case "save and reload with a per-part spec"): the
# saved-state reader (`_saved`) keeps only the "preset" of a saved per-part dict, so a reload that
# names no profile loses the overrides and its digest differs from the saved model's.
comptime NP_SABOTAGE_DROP_SAVED = is_defined["MOJOLEARN_NUMERIC_PROFILE_SABOTAGE_DROP_SAVED"]()


def _dict(value: PythonObject, key_bytes: Bool = False) raises:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: numeric profile metadata requires ordinary dictionaries")
    var keys = py_call(b.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit metadata keys before any hashed access
        if py_type(keys[i]) is not (b.bytes if key_bytes else b.str):
            raise Error("TypeError: numeric profile metadata requires native string keys")


def _passive(value: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: numeric profile metadata nesting is too deep")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bool or cls is b.int or cls is b.float:
        return
    if cls is b.tuple or cls is b.list:
        for i in range(len(value)):  # driver-loop: admit ordinary metadata sequences
            _passive(value[i], depth + 1)
        return
    if cls is b.dict:
        _dict(value)
        var values = py_call(b.list, value.values())
        for i in range(len(values)):  # driver-loop: admit ordinary profile registry records
            _passive(values[i], depth + 1)
        return
    raise Error("TypeError: numeric profile metadata cannot invoke Python protocols")


def _str(value: PythonObject, what: String) raises -> String:
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("TypeError: mojolearn: " + what + " must be a profile name (an ordinary str)")
    return String(value)


def _repr(value: PythonObject) raises -> String:
    # All callers have admitted exact built-in values; no user __repr__ runs.
    _passive(value)
    return String(py_call(native_loaded_module("builtins").repr, value))


struct ProfileAdmission(Defaultable, Movable):
    # The connected context admitted last, and every dictionary its full
    # admission walked. Valid while none of them changed (dict watchers).
    var context: PythonObject
    var dicts: List[PythonObject]

    def __init__(out self):
        self.context = PythonObject(None)
        self.dicts = List[PythonObject]()


comptime PROFILE_ADMISSION = SharedSlot[ProfileAdmission, "MojolearnNumericProfileAdmissionV2"]


def _collect(value: PythonObject, mut dicts: List[PythonObject], depth: Int = 0) raises -> Bool:
    # The dictionaries of an admitted registry. False when it holds a list:
    # a list edit is not observable, so such a registry is walked per call.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or depth > 64:
        return False
    if cls is b.tuple:
        for i in range(len(value)):  # driver-loop: over admitted metadata sequence items
            if not _collect(value[i], dicts, depth + 1):
                return False
        return True
    if cls is b.dict:
        dicts.append(value)
        var values = py_call(b.list, value.values())
        for i in range(len(values)):  # driver-loop: over admitted profile registry records
            if not _collect(values[i], dicts, depth + 1):
                return False
    return True


def _state(context: PythonObject) raises -> PythonObject:
    # The full admission below reads only exact dictionaries, immutable
    # scalars and tuples. Reuse it while the same context and every
    # dictionary it walked are unchanged (each one watched).
    var g = PROFILE_ADMISSION.get_or_create_ptr()
    if g[].context is context and len(g[].dicts) > 0:
        var same = True
        for i in range(len(g[].dicts)):  # driver-loop: over the admitted registry dictionaries
            if not dict_keys_admitted(g[].dicts[i]):
                same = False
                break
        if same:
            return context["fields"]
    var fields = _state_full(context)
    # Exactly the containers _state_full walked: the context and module
    # dictionaries themselves (keys and the bindings it read) and the four
    # registries and USES recursively.
    var dicts = List[PythonObject]()
    dicts.append(context)
    dicts.append(fields)
    var cacheable = True
    var names = Python.tuple("PROFILES", "REJECTED", "MEASURED", "_defaults", "USES", "FAMILIES",
                             "PARTS", "PART_FORMATS", "FAMILY_PARTS")
    for i in range(len(names)):  # driver-loop: fixed connected profile registries
        if cacheable and not _collect(fields[names[i]], dicts):
            cacheable = False
    if cacheable:
        for i in range(len(dicts)):  # driver-loop: over the admitted registry dictionaries
            dict_keys_admit(dicts[i])
        g[].context = context
        g[].dicts = dicts^
    return fields


def _state_full(context: PythonObject) raises -> PythonObject:
    _dict(context)
    var fields = context["fields"]
    _dict(fields)
    var names = Python.tuple("PROFILES", "REJECTED", "MEASURED", "_defaults", "PART_FORMATS", "FAMILY_PARTS")
    for i in range(len(names)):  # driver-loop: fixed connected profile registries
        _dict(fields[names[i]])
        _passive(fields[names[i]])
    if py_type(fields["PARTS"]) is not native_loaded_module("builtins").tuple:
        raise Error("TypeError: numeric profile parts must be an ordinary tuple")
    _passive(fields["PARTS"])
    if fields["_defaults"] is not context["defaults"]:
        raise Error("TypeError: numeric profile defaults must retain their connected provider storage")
    var strings = Python.tuple("DEFAULT", "TRAINING_DEFAULT", "BASELINE", "ENV", "STATE_KEY")
    for i in range(len(strings)):  # driver-loop: fixed profile constant admission
        _ = _str(fields[strings[i]], "profile metadata")
    if py_type(fields["USES"]) is not native_loaded_module("builtins").tuple:
        raise Error("TypeError: numeric profile uses must be an ordinary tuple")
    _passive(fields["USES"])
    if py_type(fields["FAMILIES"]) is not native_loaded_module("builtins").tuple:
        raise Error("TypeError: numeric profile families must be an ordinary tuple")
    _passive(fields["FAMILIES"])
    return fields


def _registered(fields: PythonObject) raises -> String:
    var keys = py_call(native_loaded_module("builtins").list, fields["PROFILES"].keys())
    var result = String()
    for i in range(len(keys)):  # driver-loop: registered profile names in insertion order
        if i > 0:
            result += ", "
        result += String(keys[i])
    return result


def _canonical(name: PythonObject, what: String, fields: PythonObject) raises -> PythonObject:
    _ = _str(name, what)
    # Exact str's native strip preserves Python Unicode whitespace semantics.
    var key = py_call(name.strip)
    if py_contains(fields["PROFILES"], key):
        return key
    var registered = _registered(fields)
    if py_contains(fields["REJECTED"], key):
        raise Error("ValueError: mojolearn: " + what + "=" + _repr(name) + " is not offered: "
                    + _str(fields["REJECTED"][key], "rejection reason") + ". Registered: " + registered)
    raise Error("ValueError: mojolearn: " + what + "=" + _repr(name)
                + " is not a registered numeric profile; registered: " + registered)


def _use(use: PythonObject, fields: PythonObject) raises:
    _passive(use)
    if not py_contains(fields["USES"], use):
        raise Error("ValueError: mojolearn: use must be one of " + _repr(fields["USES"]) + ", got " + _repr(use))
    if py_type(use) is not native_loaded_module("builtins").str:
        raise Error("TypeError: numeric profile use requires an ordinary string")


def _available(key: PythonObject, what: String, use: PythonObject, fields: PythonObject) raises -> PythonObject:
    _use(use, fields)
    var row = fields["PROFILES"][key]
    _dict(row)
    if not Bool(py=row[use]):
        var other = "inference" if String(use) == "training" else "training"
        var also = String()
        if Bool(py=row[other]):
            also = " It is available for " + other + "."
        raise Error("NotImplementedError: mojolearn: " + what + "=" + _repr(key) + " is registered ("
                    + _str(row["note"], "profile note") + ") and is refused for " + String(use)
                    + "; it is never replaced by " + _repr(fields["BASELINE"]) + " silently." + also
                    + " Use numeric_profile=" + _repr(fields["BASELINE"]) + ".")
    return key


# ---------------------------------------------------------------- per-part specs
# A spec is a profile name or an exact dict {"preset": name, part: name, ...}
# (docs/plans/NUMERIC_PROFILE_PARTS.md section 2). `_spec` validates one and
# returns its CANONICAL form: the preset string when no override differs from
# the preset's own format for that part, else a new dict with "preset" first
# and the differing overrides in PARTS order. It runs once per model, at
# construction or load (the native plan keeps the parts mask, never a name).


def _joined(names: PythonObject) raises -> String:
    var result = String()
    for i in range(len(names)):  # driver-loop: over a few admitted names
        if i > 0:
            result += ", "
        result += _repr(names[i])
    return result


def _preset_format(preset: PythonObject, part: PythonObject, fields: PythonObject) raises -> PythonObject:
    # The format the preset itself gives `part`: the preset when the part can
    # use it, else the baseline (fixed15_v1 leaves attn_context on fp32_v1).
    if py_contains(py_item(fields["PART_FORMATS"], part), preset):
        return preset
    return fields["BASELINE"]


def _spec(spec: PythonObject, what: String, family: PythonObject, fields: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(spec) is not b.dict:
        return _canonical(spec, what, fields)
    var keys = py_call(b.list, py_call(spec.keys))
    for i in range(len(keys)):  # driver-loop: admit the few spec keys before any hashed access
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: mojolearn: " + what + " part names must be ordinary str")
    var preset_key = PythonObject("preset")
    if not py_contains(spec, preset_key):
        raise Error("ValueError: mojolearn: a " + what + " dict needs 'preset', a profile name, for example "
                    + "{'preset': 'fixed15_v1', 'lm_head': 'fp32_v1'}")
    var preset = _canonical(py_item(spec, preset_key), what + "['preset']", fields)
    _family(family, fields)
    var family_parts = PythonObject(None)
    if family is not None:
        family_parts = py_get(fields["FAMILY_PARTS"], family)
        if family_parts is None:
            family_parts = Python.tuple()
    var parts = fields["PARTS"]
    var overrides = Python.dict()
    for i in range(len(keys)):  # driver-loop: over the few part overrides of one spec
        var key = keys[i]
        if String(key) == "preset":
            continue
        if not py_contains(parts, key):
            raise Error("ValueError: mojolearn: " + what + " has no part " + _repr(key) + "; parts: "
                        + _joined(parts))
        if family is not None and not py_contains(family_parts, key):
            raise Error("ValueError: mojolearn: " + what + " part " + _repr(key) + " is not a part of the "
                        + _repr(family) + " family (its parts: " + _joined(family_parts) + ")")
        var value = py_item(spec, key)
        var label = what + "[" + _repr(key) + "]"
        var fmt = _canonical(value, label, fields)
        var allowed = py_item(fields["PART_FORMATS"], key)
        if not py_contains(allowed, fmt):
            raise Error("ValueError: mojolearn: " + what + " part " + _repr(key) + " cannot use " + _repr(fmt)
                        + "; it takes " + _joined(allowed))
        py_setitem(overrides, key, fmt)
    var result = Python.dict()
    py_setitem(result, preset_key, preset)
    var kept = 0
    for i in range(len(parts)):  # driver-loop: over the ten parts, in table order
        var part = parts[i]
        var fmt = py_get(overrides, part)
        if fmt is None:
            continue
        if String(fmt) == String(_preset_format(preset, part, fields)):
            continue
        py_setitem(result, part, fmt)
        kept += 1
    if kept == 0:
        return preset
    return result


def _table(spec: PythonObject, fields: PythonObject) raises -> PythonObject:
    # part -> profile for every part of a CANONICAL spec.
    var is_dict = py_type(spec) is native_loaded_module("builtins").dict
    var preset = py_item(spec, PythonObject("preset")) if is_dict else spec
    var parts = fields["PARTS"]
    var table = Python.dict()
    for i in range(len(parts)):  # driver-loop: over the ten parts, in table order
        var part = parts[i]
        var fmt = py_get(spec, part) if is_dict else PythonObject(None)
        if fmt is None:
            fmt = _preset_format(preset, part, fields)
        py_setitem(table, part, fmt)
    return table


def _used(spec: PythonObject, fields: PythonObject) raises -> PythonObject:
    # The distinct profiles a CANONICAL spec names: the name itself, or the
    # preset and every profile its parts compute under.
    var b = native_loaded_module("builtins")
    var used = Python.list()
    if py_type(spec) is not b.dict:
        _ = py_call(used.append, spec)
        return used
    _ = py_call(used.append, py_item(spec, PythonObject("preset")))
    var values = py_call(b.list, py_call(_table(spec, fields).values))
    for i in range(len(values)):  # driver-loop: over the ten parts' profiles
        if not py_contains(used, values[i]):
            _ = py_call(used.append, values[i])
    return used


def _available_spec(spec: PythonObject, what: String, use: PythonObject, fields: PythonObject) raises -> PythonObject:
    # `_available` for a canonical spec: every profile it names must be
    # available for `use`; a per-part dict is inference only.
    if py_type(spec) is not native_loaded_module("builtins").dict:
        return _available(spec, what, use, fields)
    _use(use, fields)
    if String(use) != "inference":
        raise Error("ValueError: mojolearn: " + what + " is a per-part dict, which is for inference only; a trainer "
                    + "takes a profile name (" + _repr(fields["TRAINING_DEFAULT"]) + ")")
    var used = _used(spec, fields)
    for i in range(len(used)):  # driver-loop: over the spec's one or two profiles
        _ = _available(used[i], what, use, fields)
    return spec


def _environment(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # os._Environ methods are Python. Read only its connected native backing
    # dictionary, matching os.environ's filesystem decoding on supported POSIX
    # installations without invoking __getitem__ or any mapping callback.
    var os_fields = context["os_fields"]
    _dict(os_fields)
    if os_fields["environ"] is not context["environ"]:
        raise Error("TypeError: numeric profiles require the import-connected process environment")
    var env_fields = context["environ_fields"]
    _dict(env_fields)
    var data = env_fields["_data"]
    _dict(data, True)
    ref cpy = Python().cpython()
    var key_raw = cpy.lib.get_function[PyObjectPtr]("PyUnicode_EncodeFSDefault")(fields["ENV"]._obj_ptr)
    if not key_raw:
        raise cpy.unsafe_get_error()
    var key = PythonObject(from_owned=key_raw)
    var raw = py_get(data, key)
    if raw is None:
        return PythonObject("")
    if py_type(raw) is not native_loaded_module("builtins").bytes:
        raise Error("TypeError: numeric profile environment storage requires native bytes")
    var address = Int(0)
    var size = Int(0)
    if cpy.lib.get_function[Int32]("PyBytes_AsStringAndSize")(raw._obj_ptr, Pointer(to=address), Pointer(to=size)) != 0:
        raise cpy.unsafe_get_error()
    var decoded = cpy.lib.get_function[PyObjectPtr]("PyUnicode_DecodeFSDefaultAndSize")(address, size)
    if not decoded:
        raise cpy.unsafe_get_error()
    var value = PythonObject(from_owned=decoded)
    return py_call(value.strip)


def _default(use: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    _use(use, fields)
    var defaults = fields["_defaults"]
    if not py_contains(defaults, use):
        var value = fields["TRAINING_DEFAULT"]
        if String(use) == "inference":
            var env = _environment(fields, context)
            value = fields["DEFAULT"]
            if len(env) > 0:
                var what = String(fields["ENV"])
                value = _available(_canonical(env, what, fields), what, PythonObject("inference"), fields)
        py_setitem(defaults, use, value)
    return py_item(defaults, use)  # tail-call fix (2026-11-12): a std subscript returned directly is emitted as an LLVM tail call; LLVM may drop the borrowed key's stack store (Mojo 1.2.0). Read through the C API.


def _profiles(fields: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = Python.list()
    var names = py_call(b.list, fields["PROFILES"].keys())
    for i in range(len(names)):  # driver-loop: publish the small registered profile metadata table
        var name = names[i]
        var row = fields["PROFILES"][name]
        _dict(row)
        var record = Python.dict()
        py_setitem(record, "name", name)
        py_setitem(record, "default", PythonObject(Bool(py=name == fields["DEFAULT"])))
        py_setitem(record, "training_default", PythonObject(Bool(py=name == fields["TRAINING_DEFAULT"])))
        _ = py_call(record.update, row)
        _dict(row["products"])
        _dict(row["quality"])
        py_setitem(record, "products", py_call(b.dict, row["products"]))
        py_setitem(record, "quality", py_call(b.dict, row["quality"]))
        var columns = Python.tuple("computes", "default_for")
        for k in range(len(columns)):  # driver-loop: two family metadata columns
            var value = row[columns[k]]
            py_setitem(record, columns[k], value if value is None else py_call(b.tuple, value))
        var measured = py_get(fields["MEASURED"], name, Python.dict())
        _dict(measured)
        py_setitem(record, "measured", py_call(b.dict, measured))
        _ = py_call(result.append, record)
    return py_call(b.tuple, result)


def _label(value: PythonObject, what: String) raises -> String:
    # The caller's own label for its keyword (`what`): an exact str only.
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("TypeError: numeric profile " + what + " requires an ordinary str")
    return String(value)


def _family(family: PythonObject, fields: PythonObject) raises:
    # Was _numeric_profile._family (Python, until 2026-11-15).
    if family is None:
        return
    _passive(family)
    if not py_contains(fields["FAMILIES"], family):
        raise Error("ValueError: mojolearn: family must be one of " + _repr(fields["FAMILIES"])
                    + " or None, got " + _repr(family))


def _row_families(key: PythonObject, column: String, fields: PythonObject) raises -> PythonObject:
    var row = fields["PROFILES"][key]
    _dict(row)
    return py_item(row, PythonObject(column))


def _covers(profile: PythonObject, family: PythonObject, column: String, fields: PythonObject) raises -> Bool:
    # Was _numeric_profile.computes / default_reaches: `fams is None or
    # _family(family) in fams`, the family checked only when fams is a tuple.
    var key = _canonical(profile, "numeric_profile", fields)
    var families = _row_families(key, column, fields)
    if families is None:
        return True
    _family(family, fields)
    _passive(families)
    return py_contains(families, family)


def _format_g(value: PythonObject) raises -> String:
    # Python's f"{value:g}" on an admitted exact float or int.
    _passive(value)
    return String(py_call(native_loaded_module("builtins").format, value, PythonObject("g")))


def _warn_if_slow(key: PythonObject, vendor: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # Was _numeric_profile._warn_if_slow_here: once per (profile, vendor),
    # NumericProfileSpeedWarning when the profile was measured slower than
    # the baseline on `vendor`. Returns the message, or None.
    if vendor is None:
        return PythonObject(None)
    _passive(vendor)
    var rows = py_get(fields["MEASURED"], key, Python.dict())
    _dict(rows)
    var row = py_get(rows, vendor)
    if row is None:
        return PythonObject(None)
    _dict(row)
    var over = row["over"]
    _passive(over)
    if Bool(py=over[0] <= PythonObject(1.0)):
        return PythonObject(None)
    var warned = fields["_warned"]
    if py_type(warned) is not native_loaded_module("builtins").set:
        raise Error("TypeError: numeric profile warning record must be an ordinary set")
    var seen = Python.tuple(key, vendor)
    if py_contains(warned, seen):
        return PythonObject(None)
    _ = py_call(warned.add, seen)
    var baseline = _str(fields["BASELINE"], "profile metadata")
    var env = _str(fields["ENV"], "profile metadata")
    var message = ("mojolearn: numeric profile " + _repr(key) + " was measured on "
                   + _str(row["box"], "measured box") + " to take " + _format_g(over[0]) + " to "
                   + _format_g(over[1]) + " times " + baseline + "'s time ("
                   + _str(row["what"], "measured operation") + "; " + _str(row["source"], "measured source")
                   + "). Its bits are the same on every vendor; on this one it costs time. numeric_profile="
                   + _repr(fields["BASELINE"]) + " (or " + env + "=" + baseline
                   + ") gives the previous arithmetic.")
    var category = fields["NumericProfileSpeedWarning"]
    if category is not context["speed_warning"]:
        raise Error("TypeError: numeric profile warnings require the connected warning category")
    # stacklevel 2: the frame above the one-call Python wrapper, as the old
    # stacklevel=3 from _warn_if_slow_here inside resolve named.
    ref cpy = Python().cpython()
    if cpy.lib.get_function[Int32]("PyErr_WarnEx")(category._obj_ptr, message.as_c_string_slice().unsafe_ptr(), Int(2)) != 0:
        var failure = stash_python_error()
        raise failure^
    var published = PythonObject(message)
    _ = message^
    return published


def _this_vendor(context: PythonObject) raises -> PythonObject:
    # Was _numeric_profile._this_vendor: the native backend vendor read, None
    # when the backend cannot answer (the old `except Exception`).
    try:
        return backend_public_operation("vendor", Python.tuple(), context["backend_context"])
    except error:
        _ = error^
        return PythonObject(None)


def _resolve(name: PythonObject, what: String, use: PythonObject, family: PythonObject,
             fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # Was _numeric_profile.resolve (Python, until 2026-11-15).
    _family(family, fields)
    var key: PythonObject
    if name is None:
        key = _default(use, fields, context)
        if family is not None and not (_covers(key, family, "computes", fields)
                                       and _covers(key, family, "default_for", fields)):
            key = fields["BASELINE"]
    else:
        if py_type(name) is native_loaded_module("builtins").dict:
            # A per-part dict is inference only: refused for a trainer before
            # anything in it is read.
            _use(use, fields)
            if String(use) != "inference":
                raise Error("ValueError: mojolearn: " + what + " is a per-part dict, which is for inference only; "
                            + "a trainer takes a profile name (" + _repr(fields["TRAINING_DEFAULT"]) + ")")
        key = _spec(name, what, family, fields)
        var named = _used(key, fields)
        for n in range(len(named)):  # driver-loop: over the spec's one or two profiles
            var profile = named[n]
            if family is not None and not _covers(profile, family, "computes", fields):
                var computes = _row_families(profile, "computes", fields)
                var listed = String()
                for i in range(len(computes)):  # driver-loop: over the row's few family names
                    if i > 0:
                        listed += ", "
                    listed += _str(computes[i], "profile family")
                raise Error("NotImplementedError: mojolearn: " + what + "=" + _repr(profile)
                            + " does not compute the " + _repr(family) + " family (it computes " + listed
                            + "); it is never replaced by " + _repr(fields["BASELINE"])
                            + " silently. Use numeric_profile=" + _repr(fields["BASELINE"]) + ".")
    _ = _available_spec(key, what, use, fields)
    var baseline = _str(fields["BASELINE"], "profile metadata")
    var used = _used(key, fields)
    var vendor = PythonObject(None)
    var asked = False
    for i in range(len(used)):  # driver-loop: over the spec's one or two profiles
        if String(used[i]) != baseline:
            if not asked:
                vendor = _this_vendor(context)
                asked = True
            _ = _warn_if_slow(used[i], vendor, fields, context)
    return key


def _saved(state: PythonObject, what: String, fields: PythonObject) raises -> PythonObject:
    # Was _numeric_profile._saved: `state.get(STATE_KEY)` for a mapping. An
    # exact dict is read natively; an admitted built-in value has no `get`
    # and reads as the baseline; any other object would run a Python `get`
    # and is refused.
    var b = native_loaded_module("builtins")
    var saved = PythonObject(None)
    if py_type(state) is b.dict:
        saved = py_get(state, fields["STATE_KEY"])
    else:
        try:
            _passive(state)
        except error:
            _ = error^
            raise Error("TypeError: mojolearn: " + what + " must be an ordinary dict to carry a numeric profile")
    if saved is None:
        return fields["BASELINE"]
    comptime if NP_SABOTAGE_DROP_SAVED:
        # The sabotage arm (top of file): a saved per-part dict keeps only its preset.
        if py_type(saved) is b.dict:
            var preset_only = py_get(saved, PythonObject("preset"))
            saved = preset_only
    # A saved spec is a profile name or a per-part dict; both are read back
    # in their canonical form.
    return _spec(saved, what + " " + _str(fields["STATE_KEY"], "profile metadata"), PythonObject(None), fields)


def _check_saved(state: PythonObject, profile: PythonObject, what: String, fields: PythonObject) raises -> PythonObject:
    # Was _numeric_profile.check_saved (Python, until 2026-11-15). Compares
    # canonical forms, so a dict that only restates its preset equals the
    # preset name.
    var saved = _saved(state, what, fields)
    var mine = _spec(profile, "numeric_profile", PythonObject(None), fields)
    if not Bool(py=saved == mine):
        raise Error("ValueError: mojolearn: " + what + " was written under numeric profile " + _repr(saved)
                    + " and this object computes under " + _repr(mine) + "; a run keeps one arithmetic, "
                    + "so build the object with numeric_profile=" + _repr(saved))
    return saved


def numeric_profile_operation(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(context)
    if py_type(values) is not native_loaded_module("builtins").tuple:
        raise Error("TypeError: numeric profile arguments require a native tuple")
    if operation == "profiles" and len(values) == 0:
        return _profiles(fields)
    if operation == "measured" and len(values) == 2:
        var key = _canonical(values[0], "numeric_profile", fields)
        var rows = py_get(fields["MEASURED"], key, Python.dict())
        _dict(rows)
        if values[1] is None:
            return py_call(native_loaded_module("builtins").dict, rows)
        _passive(values[1])
        return py_get(rows, values[1])
    if operation == "default" and len(values) == 1:
        return _default(values[0], fields, context)
    if operation == "set_default" and len(values) == 2:
        var previous = _default(values[1], fields, context)
        var value = _available(_canonical(values[0], "set_numeric_profile", fields), "set_numeric_profile", values[1], fields)
        py_setitem(fields["_defaults"], values[1], value)
        return previous
    if operation == "canonical" and len(values) == 2:
        return _spec(values[0], _label(values[1], "label"), PythonObject(None), fields)
    if operation == "parts" and len(values) == 2:
        return _table(_spec(values[0], "numeric_profile", values[1], fields), fields)
    if (operation == "computes" or operation == "default_reaches") and len(values) == 2:
        var column = "computes" if operation == "computes" else "default_for"
        return PythonObject(_covers(values[0], values[1], column, fields))
    if operation == "resolve" and len(values) == 4:
        return _resolve(values[0], _label(values[1], "label"), values[2], values[3], fields, context)
    if operation == "warn_if_slow" and len(values) == 2:
        var key = _canonical(values[0], "numeric_profile", fields)
        var vendor = _this_vendor(context) if values[1] is None else values[1]
        return _warn_if_slow(key, vendor, fields, context)
    if operation == "state_field" and len(values) == 1:
        # The canonical form is stored: a per-part dict as a fresh dict, a
        # name unless it is the baseline (an fp32_v1 state gains no bytes).
        var key = _spec(values[0], "numeric_profile", PythonObject(None), fields)
        var field = Python.dict()
        if py_type(key) is native_loaded_module("builtins").dict:
            py_setitem(field, fields["STATE_KEY"], key)
        elif String(key) != _str(fields["BASELINE"], "profile metadata"):
            py_setitem(field, fields["STATE_KEY"], key)
        return field
    if operation == "check_saved" and len(values) == 3:
        return _check_saved(values[0], values[1], _label(values[2], "label"), fields)
    if operation == "adopt_saved" and len(values) == 4:
        var what = _label(values[2], "label")
        var saved = _saved(values[0], what, fields)
        if values[1] is not None:
            _ = _check_saved(values[0], values[1], what, fields)
        return _available_spec(saved, what + " " + _str(fields["STATE_KEY"], "profile metadata"), values[3], fields)
    raise Error("ValueError: unknown numeric profile operation or argument count")


def numeric_profile_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: numeric profile entry expects operation, values and context")
        var operation = _str(args[0], "operation")
        return numeric_profile_operation(operation, args[1], args[2]).steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("ValueError:"):
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
