# SPDX-License-Identifier: Apache-2.0
# Read canonical import-time module/export connections without import callbacks.
from std.ffi import _Global
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


def native_namespace_string_keys(fields: PyObjectPtr, string_type: PythonObject) raises -> Bool:
    # GIL held; public dictionary iteration neither hashes keys nor executes
    # Python protocols. No temporary Python key/item list is constructed.
    ref cpy = Python().cpython()
    var position = Int(0)
    var key = PyObjectPtr()
    var value = PyObjectPtr()
    var next_item = cpy.lib.get_function[Int32]("PyDict_Next")
    while next_item(fields, Pointer(to=position), Pointer(to=key), Pointer(to=value)) != 0:  # driver-loop: admitted namespace keys
        var cls = cpy.PyObject_Type(key)
        var same = cls == string_type._obj_ptr
        cpy.Py_DecRef(cls)
        if not same:
            return False
    return True


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
    var string_type = _loaded_type(PythonObject(""))
    if not native_namespace_string_keys(fields, string_type):
        raise Error("TypeError: native module connection requires ordinary string fields")
    # `pinned_hook`: an import-time connection that retained CPython's own
    # module-level __getattr__ as a pinned export (_native_loaded_modules.
    # stdlib_module_hook); its identity is checked with the other exports and
    # these dictionary reads never call it.
    if not pinned_hook and _loaded_lookup(fields, "__getattr__"):
        raise Error("TypeError: native module connection cannot invoke module attribute hooks")
    return PythonObject(from_owned=cpy.Py_NewRef(fields))


def _native_loaded_module_full(name: String) raises -> PythonObject:
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


struct _AdmittedModule(Copyable, Movable):
    # One connection admitted by the full check, kept for the process.
    var name: String
    var module: PythonObject
    var module_type: PythonObject
    var fields: PythonObject
    var size: Int
    var keys: List[PythonObject]
    var values: List[PythonObject]
    var namespace_keys: List[PythonObject]
    var key_ptrs: List[PyObjectPtr]
    var value_ptrs: List[PyObjectPtr]

    def __init__(out self, name: String, module: PythonObject, module_type: PythonObject, fields: PythonObject,
                 size: Int, keys: List[PythonObject], values: List[PythonObject]) raises:
        self.name = name
        self.module = module
        self.module_type = module_type
        self.fields = fields
        self.size = size
        self.keys = keys.copy()
        self.values = values.copy()
        self.namespace_keys = List[PythonObject]()
        ref cpy = Python().cpython()
        var position = Int(0)
        var key = PyObjectPtr()
        var value = PyObjectPtr()
        var next_item = cpy.lib.get_function[Int32]("PyDict_Next")
        while next_item(fields._obj_ptr, Pointer(to=position), Pointer(to=key), Pointer(to=value)) != 0:  # driver-loop: capture admitted namespace keys
            self.namespace_keys.append(PythonObject(from_owned=cpy.Py_NewRef(key)))
        # Raw pointers of the retained lists' items (kept alive by them).
        self.key_ptrs = List[PyObjectPtr]()
        self.value_ptrs = List[PyObjectPtr]()
        # driver-loop: over the registered exports, once per admission
        for j in range(len(keys)):  # small-loop(keys: registered export names): retains admitted export pointers
            self.key_ptrs.append(keys[j]._obj_ptr)
            self.value_ptrs.append(values[j]._obj_ptr)


struct _AdmittedModules(Defaultable, Movable):
    var entries: List[_AdmittedModule]

    def __init__(out self):
        self.entries = List[_AdmittedModule]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever _AdmittedModule's layout changes.
comptime _ADMITTED_MODULES = _Global[
    StorageType=_AdmittedModules, name="MojolearnAdmittedModuleConnectionsV2",
    init_fn=_AdmittedModules.__init__,
]


def _admitted_stamp(entry: _AdmittedModule) raises -> Bool:
    # Identity stamp of an admitted connection: the registered module is
    # still the loaded one, with its type, its namespace dictionary and that
    # dictionary's size, every namespace key, and every registered export
    # still have their admitted identities. Native pointer comparisons only:
    # no registry scan or temporary Python containers.
    ref cpy = Python().cpython()
    var modules = cpy.lib.get_function[PyObjectPtr]("PyImport_GetModuleDict")()
    if not modules:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    var found = _loaded_lookup(modules, entry.name.copy())
    if not found or found != entry.module._obj_ptr:
        return False
    var cls = PythonObject(from_owned=cpy.PyObject_Type(entry.module._obj_ptr))
    if cls is not entry.module_type:
        return False
    var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(entry.module._obj_ptr)
    if not fields or fields != entry.fields._obj_ptr:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    if cpy.lib.get_function[Int]("PyDict_Size")(fields) != entry.size:
        return False
    # Size alone misses delete+insert edits. Compare every admitted key by
    # pointer before any hash lookup, including an added __getattr__ key.
    var position = Int(0)
    var key = PyObjectPtr()
    var value = PyObjectPtr()
    var index = 0
    var next_item = cpy.lib.get_function[Int32]("PyDict_Next")
    while next_item(fields, Pointer(to=position), Pointer(to=key), Pointer(to=value)) != 0:  # driver-loop: native namespace identity check
        if index >= len(entry.namespace_keys) or key != entry.namespace_keys[index]._obj_ptr:
            return False
        index += 1
    if index != len(entry.namespace_keys):
        return False
    var lookup = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemWithError")
    var count = len(entry.key_ptrs)
    # driver-loop: over the admitted export identities of one module (pointer compares)
    for j in range(count):  # small-loop(count: registered export names): pointer identity stamp, no namespace walk
        var current = lookup(fields, entry.key_ptrs[j])
        if not current or current != entry.value_ptrs[j]:
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            return False
    return True


def _admit_module(name: String, module: PythonObject) raises:
    # Called right after the full check passed: record what it verified.
    ref cpy = Python().cpython()
    var registry = _loaded_lookup(cpy.lib.get_function[PyObjectPtr]("PyImport_GetModuleDict")(), "mojolearn._native_loaded_modules")
    var metadata = PythonObject(from_owned=cpy.Py_NewRef(registry))
    var connections = PythonObject(from_owned=cpy.Py_NewRef(
        _loaded_lookup(cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(metadata._obj_ptr), "CONNECTIONS")))
    var fields = PythonObject(from_owned=cpy.Py_NewRef(cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)))
    var keys = List[PythonObject]()
    var values = List[PythonObject]()
    # driver-loop: over the fixed import-time module inventory, once per process
    for i in range(len(connections)):  # small-loop(connections: registered modules): finds the admitted record once
        var record = connections[i]
        if String(record[0]) != name or record[1] is not module:
            continue
        var exports = record[2]
        # driver-loop: over the registered exports, once per process
        for j in range(len(exports)):  # small-loop(exports: registered native names): retains admitted export identities
            keys.append(exports[j][0])
            values.append(exports[j][1])
        var size = cpy.lib.get_function[Int]("PyDict_Size")(fields._obj_ptr)
        var cls = PythonObject(from_owned=cpy.PyObject_Type(module._obj_ptr))
        var g = _ADMITTED_MODULES.get_or_create_ptr()
        # driver-loop: over the few admitted connections
        for k in range(len(g[].entries)):  # small-loop(entries: admitted connections): replaces a stale admission
            if g[].entries[k].name == name:
                g[].entries[k] = _AdmittedModule(name, module, cls, fields, size, keys, values)
                return
        g[].entries.append(_AdmittedModule(name, module, cls, fields, size, keys, values))
        return


def native_loaded_module(name: String) raises -> PythonObject:
    # Validate once per process, then stamp: the full check below (registry
    # record, module identity and type, namespace keys, module __getattr__,
    # export identities) runs on first use and again whenever the identity
    # stamp differs, so a refusal keeps its type, message and position. An
    # unchanged admitted connection costs one pointer-compare stamp.
    var g = _ADMITTED_MODULES.get_or_create_ptr()
    # driver-loop: over the few admitted connections
    for k in range(len(g[].entries)):  # small-loop(entries: admitted connections): finds this name's admission
        if g[].entries[k].name == name:
            if _admitted_stamp(g[].entries[k]):
                return g[].entries[k].module
            break
    var module = _native_loaded_module_full(name)
    _admit_module(name, module)
    return module


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
    # followed by released(), so the next request after it repeats the
    # check: exactly one re-check per release window, never zero. That check
    # is native_loaded_module(): the identity stamp of the connection admitted
    # once per process, and the full check only when the stamp differs.
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
