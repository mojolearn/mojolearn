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


def _loaded_fields(module: PythonObject, pinned_hook: Bool = False) raises -> PythonObject:
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
    # `pinned_hook`: an import-time connection that retained CPython's own
    # module-level __getattr__ as a pinned export (_native_loaded_modules.
    # stdlib_module_hook); its identity is checked with the other exports and
    # these dictionary reads never call it.
    if not pinned_hook and _loaded_lookup(fields, "__getattr__"):
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
    # CONNECTIONS is private import-only metadata, like the binding context
    # tables. Do not rescan its fixed records on every lookup; all mutable
    # target modules and the exports actually consumed remain checked below.
    var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(metadata._obj_ptr)
    if not fields:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("TypeError: native module connection metadata is not a module")
    var raw = _loaded_lookup(fields, "CONNECTIONS")
    if not raw:
        raise Error("ImportError: native module connection metadata is absent")
    var connections = PythonObject(from_owned=cpy.Py_NewRef(raw))
    # driver-loop: over the fixed import-time module inventory
    for i in range(len(connections)):  # small-loop(connections: registered modules): selects the fixed native setup record
        var record = connections[i]
        if String(record[0]) != name:
            continue
        var found = _loaded_lookup(modules, name)
        if not found:
            raise Error("ImportError: native execution requires an import-time connection for " + name)
        var module = PythonObject(from_owned=cpy.Py_NewRef(found))
        if module is not record[1]:
            raise Error("TypeError: native module connection was replaced: " + name)
        var exports = record[2]
        var pinned_hook = False
        # driver-loop: over the registered export names, for a pinned stdlib hook
        for j in range(len(exports)):  # small-loop(exports: registered native names): finds a pinned module hook record
            if String(exports[j][0]) == "__getattr__":
                pinned_hook = True
        var module_fields = _loaded_fields(module, pinned_hook)
        # driver-loop: over exact identities of registered native exports
        for j in range(len(exports)):  # small-loop(exports: registered native names): validates fixed export identities before use
            var item = exports[j]
            var current = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemWithError")(module_fields._obj_ptr, item[0]._obj_ptr)
            if not current or current != item[1]._obj_ptr:
                raise Error("TypeError: native module connection export changed: " + name + "." + String(item[0]))
        return module
    raise Error("ImportError: native execution requires an import-time connection for " + name)


struct NativeModules(Movable):
    # Admit-once view of the import-time module connections for ONE public
    # operation. The public C entry creates it, and the helpers of that
    # operation receive it as `mut mods` instead of calling
    # native_loaded_module() again. The first request for a module runs the
    # complete native_loaded_module() check (registry record, module identity,
    # namespace keys, export identities) and raises exactly what it raised
    # before; later requests in the same operation reuse that admitted handle,
    # so the namespace and export scans run once per module, not once per
    # helper or per layer.
    # GIL release windows: while the GIL is released (a GILReleased block, a
    # lock wait, a call into another native entry that may release it),
    # another thread can replace a module or an export. Every such window is
    # followed by released(), so the next request after it repeats the full
    # check: exactly one re-check per release window, never zero.
    # A handle never outlives its operation and is never shared across
    # threads. Callers outside the neural bindings keep native_loaded_module().
    var _names: List[String]
    var _modules: List[PythonObject]

    def __init__(out self):
        self._names = List[String]()
        self._modules = List[PythonObject]()

    def module(mut self, name: String) raises -> PythonObject:
        # driver-loop: over the few modules this operation already admitted
        for i in range(len(self._names)):  # small-loop(admitted: at most a handful of module names): reuses an admitted connection
            if self._names[i] == name:
                return self._modules[i]
        var module = native_loaded_module(name)
        self._names.append(name)
        self._modules.append(module)
        return module

    def released(mut self):
        # A GIL release window ended: forget every admission so the next
        # request re-runs the full native_loaded_module() check.
        self._names.clear()
        self._modules.clear()
