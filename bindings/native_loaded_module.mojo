# SPDX-License-Identifier: Apache-2.0
# Read canonical import-time module/export connections without import callbacks.
from std.ffi import _Global
from std.python import Python, PythonObject
from std.ffi import c_int
from std.python._cpython import ExternalFunction, PyObjectPtr
from bindings.cpython_table import cpython_table


# int Py_IsFinalizing(void): public from CPython 3.13; _Py_IsFinalizing is
# the same query, exported from 3.7 through 3.12.
comptime _Py_IsFinalizing = ExternalFunction["Py_IsFinalizing", def() thin abi("C") -> c_int]
comptime _Py_IsFinalizing_old = ExternalFunction["_Py_IsFinalizing", def() thin abi("C") -> c_int]


def native_interpreter_finalizing() raises -> Bool:
    """True once the interpreter has begun finalization. A native finalizer
    (a `__del__` descriptor) collected while the interpreter tears down
    `__main__` and the other modules must return before any loaded-module
    read: PyImport_GetModuleDict, which native_loaded_module reads, is a
    fatal error once the interpreter's modules are cleared ("get_modules_dict:
    interpreter has no modules dictionary"), not a NULL return. Resolved per
    call; only finalizers ask, never a public hot path."""
    ref cpy = Python().cpython()
    if cpy.lib.check_symbol("Py_IsFinalizing"):
        return _Py_IsFinalizing.load(cpy.lib.borrow())() != 0
    if cpy.lib.check_symbol("_Py_IsFinalizing"):
        return _Py_IsFinalizing_old.load(cpy.lib.borrow())() != 0
    return False


def _loaded_type(value: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyObject_Type(value._obj_ptr))


def _loaded_lookup(mapping: PyObjectPtr, var name: String) raises -> PyObjectPtr:
    # PyDict_GetItemString builds the same temporary str key and, like
    # PyDict_GetItem, suppresses lookup errors; the key object is ours here.
    var key = PythonObject(name)
    return cpython_table()[].PyDict_GetItem(mapping, key._obj_ptr)


def native_namespace_string_keys(fields: PyObjectPtr, string_type: PythonObject) raises -> Bool:
    # GIL held; public dictionary iteration neither hashes keys nor executes
    # Python protocols. No temporary Python key/item list is constructed.
    ref cpy = Python().cpython()
    var position = Int(0)
    var key = PyObjectPtr()
    var value = PyObjectPtr()
    var count = cpython_table()[].PyDict_Size(fields)
    # Resolved once per call, not per item; main's compiled form (metrics_plan_admission.mojo).
    var dict_next = cpy.lib.get_function[Int32]("PyDict_Next")
    for i in range(count):  # small-loop(count: namespace fields): native metadata key admission only
        if dict_next(fields, MutPointer(to=position), MutPointer(to=key), MutPointer(to=value)) == 0:
            return False
        var cls = cpy.PyObject_Type(key)
        var same = cls == string_type._obj_ptr
        cpy.Py_DecRef(cls)
        if not same:
            return False
    return True


def native_mapping_string_keys(fields: PythonObject, string_type: PythonObject) raises -> Bool:
    # Only native mappingproxy instances from type.__dict__ use this helper.
    # Python >=3.10 support precludes PyType_GetDict; the supported iterator
    # removes keys-view/list materialization without accessing private layout.
    ref cpy = Python().cpython()
    var iter_table = cpython_table()
    var raw = iter_table[].PyObject_GetIter(fields._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var iterator = PythonObject(from_owned=raw)
    while True:  # driver-loop: over exact native class mapping keys before keyed reads
        var key = iter_table[].PyIter_Next(iterator._obj_ptr)
        if not key:
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            return True
        var cls = cpy.PyObject_Type(key)
        var same = cls == string_type._obj_ptr
        cpy.Py_DecRef(cls)
        cpy.Py_DecRef(key)
        if not same:
            return False


def _loaded_fields(module: PythonObject, pinned_hook: Bool = False) raises -> PythonObject:
    ref cpy = Python().cpython()
    var fields = cpython_table()[].PyModule_GetDict(module._obj_ptr)
    if not fields:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("TypeError: native module connection is not a native module")
    var cls = _loaded_type(module)
    var flags = cpython_table()[].PyType_GetFlags(cls._obj_ptr)
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
    var modules = cpython_table()[].PyImport_GetModuleDict()
    if not modules:
        raise Error("ImportError: native execution has no loaded-module registry")
    var registry = _loaded_lookup(modules, "mojolearn._native_loaded_modules")
    if not registry:
        raise Error("ImportError: native execution requires import-time module connections")
    var metadata = PythonObject(from_owned=cpy.Py_NewRef(registry))
    # CONNECTIONS is private import-only metadata, like the binding context
    # tables. Do not rescan its fixed records on every lookup; all mutable
    # target modules and the exports actually consumed remain checked below.
    var fields = cpython_table()[].PyModule_GetDict(metadata._obj_ptr)
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
            var current = cpy.PyDict_GetItemWithError(module_fields._obj_ptr, item[0]._obj_ptr)
            if not current or current != item[1]._obj_ptr:
                raise Error("TypeError: native module connection export changed: " + name + "." + String(item[0]))
        return module
    raise Error("ImportError: native execution requires an import-time connection for " + name)


struct _AdmittedModule(Copyable, Movable):
    # One connection admitted by the full check, kept for the process.
    var name: String
    # The registry key as a str object: its cached hash serves every stamp
    # lookup (no C-string to str conversion per call).
    var name_key: PythonObject
    var module: PythonObject
    var module_type: PythonObject
    var fields: PythonObject
    var size: Int
    var keys: List[PythonObject]
    var values: List[PythonObject]
    var namespace_keys: List[PythonObject]
    var key_ptrs: List[PyObjectPtr]
    var value_ptrs: List[PyObjectPtr]
    # Per namespace slot (PyDict_Next order): the admitted export value at
    # that slot, or null for a key that is not a registered export. The stamp
    # compares keys and export values in ONE walk, with no hashed lookup.
    var slot_values: List[PyObjectPtr]
    # Registered exports whose key object is not the namespace's own key
    # object (an equal but distinct str): the stamp looks these up by key.
    var unslotted: List[Int]
    # Python >= 3.12: the namespace dictionary is watched (shared dictionary
    # watcher below) and `epoch` is its watch epoch at admission. Any
    # insertion, replacement, deletion or clear bumps the epoch, so the stamp
    # compares one integer instead of walking the namespace. False on 3.10 /
    # 3.11 or when the watch was refused: the stamp keeps the full walk.
    var watched: Bool
    var epoch: Int

    def __init__(out self, name: String, module: PythonObject, module_type: PythonObject, fields: PythonObject,
                 size: Int, keys: List[PythonObject], values: List[PythonObject]) raises:
        self.name = name
        self.name_key = PythonObject(name)
        self.module = module
        self.module_type = module_type
        self.fields = fields
        self.size = size
        self.keys = keys.copy()
        self.values = values.copy()
        self.namespace_keys = List[PythonObject]()
        self.slot_values = List[PyObjectPtr]()
        self.unslotted = List[Int]()
        self.watched = False
        self.epoch = 0
        # Raw pointers of the retained lists' items (kept alive by them).
        self.key_ptrs = List[PyObjectPtr]()
        self.value_ptrs = List[PyObjectPtr]()
        # driver-loop: over the registered exports, once per admission
        for j in range(len(keys)):  # small-loop(keys: registered export names): retains admitted export pointers
            self.key_ptrs.append(keys[j]._obj_ptr)
            self.value_ptrs.append(values[j]._obj_ptr)
        var placed = List[Bool](length=len(keys), fill=False)
        ref cpy = Python().cpython()
        var position = Int(0)
        var key = PyObjectPtr()
        var value = PyObjectPtr()
        # Resolved once per call, not per item; main's compiled form (metrics_plan_admission.mojo).
        var dict_next = cpy.lib.get_function[Int32]("PyDict_Next")
        for i in range(size):  # small-loop(size: namespace fields): capture metadata identities once
            if dict_next(fields._obj_ptr, MutPointer(to=position), MutPointer(to=key), MutPointer(to=value)) == 0:
                raise Error("native module namespace changed during admission")
            self.namespace_keys.append(PythonObject(from_owned=cpy.Py_NewRef(key)))
            var expected = PyObjectPtr()
            # driver-loop: over the registered exports, once per admission (slot of each export key)
            for j in range(len(keys)):  # small-loop(keys: registered export names): pointer match, once per process
                if not placed[j] and key == self.key_ptrs[j]:
                    expected = self.value_ptrs[j]
                    placed[j] = True
                    break
            self.slot_values.append(expected)
        # driver-loop: over the registered exports, once per admission
        for j in range(len(keys)):  # small-loop(keys: registered export names): exports needing a keyed stamp read
            if not placed[j]:
                self.unslotted.append(j)


struct _AdmittedModules(Defaultable, Movable):
    # name -> position in `entries`: one String-keyed lookup per request
    # (was a membership test plus two keyed reads, each hashing the name).
    var index: Dict[String, Int]
    var entries: List[_AdmittedModule]

    def __init__(out self):
        self.index = Dict[String, Int]()
        self.entries = List[_AdmittedModule]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever _AdmittedModule's layout changes.
comptime _ADMITTED_MODULES = _Global[
    StorageType=_AdmittedModules, name="MojolearnAdmittedModuleConnectionsV5",
    init_fn=_AdmittedModules.__init__,
]


# ------------------------------------------------- shared dictionary watcher
# CPython >= 3.12 dictionary watchers (`PyDict_AddWatcher` / `PyDict_Watch`)
# report every insertion, replacement, deletion, clear, clone and
# deallocation of a watched dictionary. Each event bumps that dictionary's
# epoch in a native map (the callback runs no Python). One watcher serves
# both the module-connection stamp here and the key admissions of
# native_class_admission (dict_keys_admitted), so the process spends one of
# CPython's eight dictionary-watcher slots, not two. Python 3.10 and 3.11
# have no dictionary watchers: every caller keeps its full check there.


struct SharedDictWatch(Defaultable, Movable):
    # watcher: -2 not set up yet, -1 unavailable (Python < 3.12 or no free
    # watcher slot), otherwise the interpreter's dictionary watcher ID.
    var watcher: Int
    var epochs: Dict[Int, Int]

    def __init__(out self):
        self.watcher = -2
        self.epochs = Dict[Int, Int]()


comptime SHARED_DICT_WATCH = _Global[
    StorageType=SharedDictWatch, name="MojolearnSharedDictWatchV1",
    init_fn=SharedDictWatch.__init__,
]


def _shared_dict_modified_c(event: Int32, fields: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr) abi("C") -> Int32:
    # Dictionary watcher callback (GIL held): native map update only; never
    # raises into CPython.
    try:
        var g = SHARED_DICT_WATCH.get_or_create_ptr()
        var address = Int(fields)
        if address in g[].epochs:
            g[].epochs[address] = g[].epochs[address] + 1
        else:
            g[].epochs[address] = 1
    except:
        pass
    return 0


def shared_dict_watcher() raises -> Int:
    """The process dictionary watcher ID, set up on first use; -1 where
    dictionary watchers are unavailable (Python 3.10 / 3.11, or every
    interpreter slot taken)."""
    var g = SHARED_DICT_WATCH.get_or_create_ptr()
    if g[].watcher != -2:
        return g[].watcher
    ref cpy = Python().cpython()
    var version_ptr = cpy.lib.get_function[Pointer[Int8, MutUntrackedOrigin]]("Py_GetVersion")()
    var version = String(unsafe_from_utf8_ptr=version_ptr)
    if version.startswith("3.10.") or version.startswith("3.11."):
        g[].watcher = -1
        return -1
    var identifier = cpy.lib.get_function[Int32]("PyDict_AddWatcher")(_shared_dict_modified_c)
    if identifier < 0:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        g[].watcher = -1
        return -1
    g[].watcher = Int(identifier)
    return g[].watcher


def shared_dict_epoch(address: Int) raises -> Int:
    """Watch epoch of the dictionary at `address` (0 until its first event)."""
    var g = SHARED_DICT_WATCH.get_or_create_ptr()
    if address in g[].epochs:
        return g[].epochs[address]
    return 0


def shared_dict_watch(fields: PyObjectPtr) raises -> Bool:
    """Watch the exact dictionary `fields`; False where watching is not
    available or was refused (the caller keeps its full check)."""
    var watcher = shared_dict_watcher()
    if watcher < 0:
        return False
    ref cpy = Python().cpython()
    if cpy.lib.get_function[Int32]("PyDict_Watch")(Int32(watcher), fields) < 0:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    return True


def _admitted_stamp(entry: _AdmittedModule) raises -> Bool:
    # Identity stamp of an admitted connection: the registered module is
    # still the loaded one, with its type, its namespace dictionary and that
    # dictionary's size, every namespace key, and every registered export
    # still have their admitted identities. Native pointer comparisons only:
    # no registry scan or temporary Python containers. Keys and export values
    # are compared in one PyDict_Next walk (the slot of each export key was
    # fixed at admission); only exports whose key object differs from the
    # namespace's own key take a hashed lookup.
    ref cpy = Python().cpython()
    var t = cpython_table()
    var modules = t[].PyImport_GetModuleDict()
    if not modules:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    var found = t[].PyDict_GetItem(modules, entry.name_key._obj_ptr)
    if not found or found != entry.module._obj_ptr:
        return False
    var cls = cpy.PyObject_Type(entry.module._obj_ptr)
    var same_type = cls == entry.module_type._obj_ptr
    cpy.Py_DecRef(cls)
    if not same_type:
        return False
    var fields = cpython_table()[].PyModule_GetDict(entry.module._obj_ptr)
    if not fields or fields != entry.fields._obj_ptr:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    if entry.watched:
        # Watched namespace (Python >= 3.12): any key or value edit since the
        # admission bumped the epoch. One integer compare replaces the walk;
        # the registry, module identity, type and namespace identity checks
        # above still run on every request.
        return shared_dict_epoch(Int(fields)) == entry.epoch
    if t[].PyDict_Size(fields) != entry.size:
        return False
    if len(entry.namespace_keys) != entry.size or len(entry.slot_values) != entry.size:
        return False
    # Size alone misses delete+insert edits. Compare every admitted key by
    # pointer before any hash lookup, including an added __getattr__ key.
    var position = Int(0)
    var key = PyObjectPtr()
    var value = PyObjectPtr()
    # Resolved once per call, not per item; main's compiled form (metrics_plan_admission.mojo).
    var dict_next = cpy.lib.get_function[Int32]("PyDict_Next")
    for i in range(entry.size):  # small-loop(entry.size: namespace fields): compare native metadata pointers only
        if dict_next(fields, MutPointer(to=position), MutPointer(to=key), MutPointer(to=value)) == 0:
            return False
        if key != entry.namespace_keys[i]._obj_ptr:
            return False
        var expected = entry.slot_values[i]
        if expected and value != expected:
            return False
    if len(entry.unslotted) > 0:
        # driver-loop: over the exports whose key object is not the namespace key (pointer compares)
        for k in range(len(entry.unslotted)):  # small-loop(unslotted: registered export names): keyed identity stamp
            var j = entry.unslotted[k]
            var current = cpy.PyDict_GetItemWithError(fields, entry.key_ptrs[j])
            if not current or current != entry.value_ptrs[j]:
                cpy.lib.get_function[NoneType]("PyErr_Clear")()
                return False
    return True


def _admit_module(name: String, module: PythonObject) raises:
    # Called right after the full check passed: record what it verified.
    ref cpy = Python().cpython()
    var registry = _loaded_lookup(cpython_table()[].PyImport_GetModuleDict(), "mojolearn._native_loaded_modules")
    var metadata = PythonObject(from_owned=cpy.Py_NewRef(registry))
    var connections = PythonObject(from_owned=cpy.Py_NewRef(
        _loaded_lookup(cpython_table()[].PyModule_GetDict(metadata._obj_ptr), "CONNECTIONS")))
    var fields = PythonObject(from_owned=cpy.Py_NewRef(cpython_table()[].PyModule_GetDict(module._obj_ptr)))
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
        var size = cpython_table()[].PyDict_Size(fields._obj_ptr)
        var cls = PythonObject(from_owned=cpy.PyObject_Type(module._obj_ptr))
        var admitted = _AdmittedModule(name, module, cls, fields, size, keys, values)
        # Watch first, then read the epoch: an edit after this point bumps it.
        # The full check just passed with the GIL held and no Python ran since.
        if shared_dict_watch(fields._obj_ptr):
            admitted.watched = True
            admitted.epoch = shared_dict_epoch(Int(fields._obj_ptr))
        var g = _ADMITTED_MODULES.get_or_create_ptr()
        # Native names index the retained connections directly. Replacing a
        # stale admission preserves the same complete module/export check.
        var slot = g[].index.find(name)
        if slot:
            g[].entries[slot.value()] = admitted^
        else:
            g[].index[name] = len(g[].entries)
            g[].entries.append(admitted^)
        return


def native_loaded_module(name: String) raises -> PythonObject:
    # Validate once per process, then stamp: the full check below (registry
    # record, module identity and type, namespace keys, module __getattr__,
    # export identities) runs on first use and again whenever the identity
    # stamp differs, so a refusal keeps its type, message and position. An
    # unchanged admitted connection costs one pointer-compare stamp.
    # One String-keyed lookup, then the stamp on the admitted record by
    # reference (no record copy). On Python >= 3.12 the stamp is pointer and
    # epoch compares only: no namespace walk, no by-name symbol lookup.
    var g = _ADMITTED_MODULES.get_or_create_ptr()
    var slot = g[].index.find(name)
    if slot:
        ref entry = g[].entries[slot.value()]
        if _admitted_stamp(entry):
            return entry.module
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
    var _modules: Dict[String, PythonObject]

    def __init__(out self):
        self._modules = Dict[String, PythonObject]()

    def module(mut self, name: String) raises -> PythonObject:
        # No repeated linear name search or Python-object dispatch. The map
        # belongs to this operation and is invalidated after every GIL window.
        if name in self._modules:
            return self._modules[name]
        var module = native_loaded_module(name)
        self._modules[name] = module
        return module

    def released(mut self):
        # A GIL release window ended: forget every admission so the next
        # request re-runs the full native_loaded_module() check.
        self._modules.clear()
