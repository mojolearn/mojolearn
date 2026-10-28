# SPDX-License-Identifier: Apache-2.0
# Complete public profile metadata/default ownership. Import time connects exact
# module dictionaries and the existing shared provider defaults. Product calls
# invoke no Python profile helpers, environment methods or input protocols.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


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


def _state(context: PythonObject) raises -> PythonObject:
    _dict(context)
    var fields = context["fields"]
    _dict(fields)
    var names = Python.tuple("PROFILES", "REJECTED", "MEASURED", "_defaults")
    for i in range(len(names)):  # driver-loop: fixed connected profile registries
        _dict(fields[names[i]])
        _passive(fields[names[i]])
    if fields["_defaults"] is not context["defaults"]:
        raise Error("TypeError: numeric profile defaults must retain their connected provider storage")
    var strings = Python.tuple("DEFAULT", "TRAINING_DEFAULT", "BASELINE", "ENV")
    for i in range(len(strings)):  # driver-loop: fixed profile constant admission
        _ = _str(fields[strings[i]], "profile metadata")
    if py_type(fields["USES"]) is not native_loaded_module("builtins").tuple:
        raise Error("TypeError: numeric profile uses must be an ordinary tuple")
    _passive(fields["USES"])
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
    return defaults[use]


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
