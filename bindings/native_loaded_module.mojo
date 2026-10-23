# SPDX-License-Identifier: Apache-2.0
# Read canonical import-time module/export connections without import callbacks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def _loaded_type(value: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyObject_Type(value._obj_ptr))


def _loaded_lookup(mapping: PyObjectPtr, var name: String) raises -> PyObjectPtr:
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemString")(mapping, name.as_c_string_slice().unsafe_ptr())
    _ = name^
    return result


def _loaded_fields(module: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
    if not fields:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("TypeError: native module connection is not a native module")
    var cls = _loaded_type(module)
    var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
    if (flags & UInt64(1 << 9)) != 0:
        raise Error("TypeError: native module connection cannot use a module subclass")
    # Keys are checked before keyed reads: foreign hash/equality hooks must
    # not run even if a module namespace acquired a non-string key.
    var raw_keys = cpy.lib.get_function[PyObjectPtr]("PyDict_Keys")(fields)
    if not raw_keys:
        raise cpy.unsafe_get_error()
    var keys = PythonObject(from_owned=raw_keys)
    var string_type = _loaded_type(PythonObject(""))
    # driver-loop: over module namespace field names, before metadata lookups
    for i in range(len(keys)):  # small-loop(keys: module field names): validates namespace metadata before attribute lookup
        if _loaded_type(keys[i]) is not string_type:
            raise Error("TypeError: native module connection requires ordinary string fields")
    if _loaded_lookup(fields, "__getattr__"):
        raise Error("TypeError: native module connection cannot invoke module attribute hooks")
    return PythonObject(from_owned=cpy.Py_NewRef(fields))


def native_loaded_module(name: String) raises -> PythonObject:
    # These stable public CPython APIs read the loaded registry directly;
    # they never invoke __import__, a finder, a loader or module attributes.
    ref cpy = Python().cpython()
    var modules = cpy.lib.get_function[PyObjectPtr]("PyImport_GetModuleDict")()
    if not modules:
        raise Error("ImportError: native execution has no loaded-module registry")
    var registry = _loaded_lookup(modules, "mojolearn._native_loaded_modules")
    if not registry:
        raise Error("ImportError: native execution requires import-time module connections")
    var metadata = PythonObject(from_owned=cpy.Py_NewRef(registry))
    var fields = _loaded_fields(metadata)
    var raw = _loaded_lookup(fields._obj_ptr, "CONNECTIONS")
    if not raw:
        raise Error("ImportError: native module connection metadata is absent")
    var connections = PythonObject(from_owned=cpy.Py_NewRef(raw))
    var tuple_type = _loaded_type(Python.tuple())
    var string_type = _loaded_type(PythonObject(""))
    if _loaded_type(connections) is not tuple_type:
        raise Error("TypeError: native module connection records require an exact tuple")
    # driver-loop: over the fixed import-time module inventory
    for i in range(len(connections)):  # small-loop(connections: registered modules): selects the fixed native setup record
        var record = connections[i]
        if _loaded_type(record) is not tuple_type or len(record) != 3 or _loaded_type(record[0]) is not string_type:
            raise Error("TypeError: malformed native module connection record")
        if String(record[0]) != name:
            continue
        var found = _loaded_lookup(modules, name)
        if not found:
            raise Error("ImportError: native execution requires an import-time connection for " + name)
        var module = PythonObject(from_owned=cpy.Py_NewRef(found))
        if module is not record[1]:
            raise Error("TypeError: native module connection was replaced: " + name)
        var module_fields = _loaded_fields(module)
        var exports = record[2]
        if _loaded_type(exports) is not tuple_type:
            raise Error("TypeError: native module export records require an exact tuple")
        # driver-loop: over exact identities of registered native exports
        for j in range(len(exports)):  # small-loop(exports: registered native names): validates fixed export identities before use
            var item = exports[j]
            if _loaded_type(item) is not tuple_type or len(item) != 2 or _loaded_type(item[0]) is not string_type:
                raise Error("TypeError: malformed native module export record")
            var current = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemWithError")(module_fields._obj_ptr, item[0]._obj_ptr)
            if not current or current != item[1]._obj_ptr:
                raise Error("TypeError: native module connection export changed: " + name + "." + String(item[0]))
        return module
    raise Error("ImportError: native execution requires an import-time connection for " + name)
