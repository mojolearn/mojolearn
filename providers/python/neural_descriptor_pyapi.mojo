# SPDX-License-Identifier: Apache-2.0
# Setup/ownership boundary for explicitly named native execution ABIs. No
# unwrapping of arbitrary Python functions, extension layouts or signatures.
from std.ffi import c_int, _Global
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.cpython_table import cpython_table
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit, class_watch_ready
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call_spans
from core.native_call_output import NativeCallWorkspace, NativeCallWorkspaceLease


struct NeuralDescriptorProvider(Movable, Writable):
    var module: PythonObject
    var capsule: PythonObject
    var factory: PythonObject
    var name: String
    var entry: NativeDescriptorCall
    var calls: List[Int]
    var workspace: List[NativeCallWorkspace]

    def __init__(out self, module: PythonObject, capsule: PythonObject, factory: PythonObject, var name: String, entry: NativeDescriptorCall, operation_count: Int = 6):
        self.module = module
        self.capsule = capsule
        self.factory = factory
        self.name = name^
        self.entry = entry
        self.calls = List[Int](length=operation_count, fill=0)
        # Retain heap-backed ABI output and session-prefix capacity with this
        # admitted provider. Its exclusive lease handles concurrent/reentrant
        # calls without sharing writable scratch or waiting while holding GIL.
        self.workspace = List[NativeCallWorkspace]()
        self.workspace.append(NativeCallWorkspace())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NeuralDescriptorProvider(", self.name, ", calls=", self.calls, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def neural_descriptor_capsule(mut name: String, entries: List[NativeDescriptorCall]) raises -> PythonObject:
    # Shared table export for every engine. The caller MUST retain the name
    # and entries in module-global native storage for the entire image lifetime;
    # PyCapsule borrows both addresses and does not own or free either allocation.
    if len(entries) != 1:
        raise Error("native provider: a v1 descriptor table has one dispatch entry")
    ref cpy = Python().cpython()
    var capsule = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(
        entries.unsafe_ptr(), name.as_c_string_slice().unsafe_ptr(), PyObjectPtr())
    if not capsule:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=capsule)


def _neural_descriptor_keys(fields: PythonObject, string_type: PythonObject) raises:
    # Admit ALL live keys before any keyed read. The shared native iterator
    # neither constructs dict.keys()/list objects nor hashes caller keys.
    if not native_namespace_string_keys(fields._obj_ptr, string_type):
        raise Error("TypeError: neural provider metadata keys require exact strings")


def _neural_descriptor_dictionary(fields: PythonObject, bi: PythonObject) raises:
    if py_type(fields) is not bi.dict:
        raise Error("TypeError: neural provider metadata requires an exact dictionary")
    _neural_descriptor_keys(fields, bi.str)


def _neural_descriptor_module_fields(module: PythonObject, bi: PythonObject) raises -> PythonObject:
    # The caller has already admitted an exact module, so these are its native
    # dictionary fields. Reuse its current builtins connection through this
    # GIL-held, callback-free admission rather than reconnecting per helper.
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyModule_GetDict(module._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_owned=cpy.Py_NewRef(raw))
    _neural_descriptor_keys(fields, bi.str)
    return fields


def _neural_descriptor_entry_fields(entries: PythonObject, bi: PythonObject) raises -> PythonObject:
    var cls = py_type(entries)
    if cls is py_type(bi):
        return _neural_descriptor_module_fields(entries, bi)
    elif cls is native_loaded_module("types").SimpleNamespace:
        ref cpy = Python().cpython()
        var raw = cpython_table()[].PyObject_GenericGetDict(entries._obj_ptr, PyObjectPtr())
        if not raw:
            raise cpy.unsafe_get_error()
        var fields = PythonObject(from_owned=raw)
        _neural_descriptor_keys(fields, bi.str)
        return fields
    else:
        raise Error("TypeError: neural provider entries require a native module or namespace")


def _neural_descriptor_function_owner(function: PythonObject) raises -> PyObjectPtr:
    # Callers first admit an exact native function (or its retained identity).
    # The public CPython API reads its borrowed owner directly: no __self__
    # attribute resolution or temporary owned PythonObject on the check path.
    # This neither inspects extension layout nor skips current owner checks.
    ref cpy = Python().cpython()
    var owner = cpython_table()[].PyCFunction_GetSelf(function._obj_ptr)
    if not owner and cpy.PyErr_Occurred():
        raise cpy.unsafe_get_error()
    _ = function
    return owner


struct NeuralDescriptorAdmitted(Defaultable, Movable):
    # Binding modules whose whole namespace was admitted (every key an exact
    # string), as (module address, dictionary address, dictionary size). Plain
    # integers only: no Python reference outlives the interpreter here.
    var modules: List[Int]
    var dicts: List[Int]
    var sizes: List[Int]

    def __init__(out self):
        self.modules = List[Int]()
        self.dicts = List[Int]()
        self.sizes = List[Int]()


comptime NEURAL_DESCRIPTOR_ADMITTED = _Global[StorageType=NeuralDescriptorAdmitted,
    name="MojolearnNeuralDescriptorAdmittedV1", init_fn=NeuralDescriptorAdmitted.__init__]


def _neural_descriptor_module_fields_once(module: PythonObject, bi: PythonObject) raises -> PythonObject:
    # The module dictionary, with its key walk done once per module identity:
    # a primitive operation resolves its entry on every call, and walking the
    # whole binding namespace each time repeated the same verdict. The walk
    # runs again whenever the module, its dictionary or the dictionary size
    # changes. An inserted key changes the size; only a same-step delete plus
    # insert keeps it, which no Mojolearn code does to a compiled binding's
    # namespace (filled once at module init). Exact per-edit detection would
    # need CPython dict watchers (3.12+), recorded as owed for the 3.10 floor.
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyModule_GetDict(module._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_owned=cpy.Py_NewRef(raw))
    # Python >= 3.12: the dictionary watcher reports every edit and the
    # deallocation of the namespace, so the verdict is exact and a recycled
    # address never matches (lane p3-neural-blocks). The integer record
    # below is the fallback only where watchers are unavailable.
    if dict_keys_admitted(fields):
        return fields
    if class_watch_ready():
        _neural_descriptor_keys(fields, bi.str)
        dict_keys_admit(fields)
        return fields
    var module_address = Int(module._obj_ptr)
    var dict_address = Int(raw)
    var size = len(fields)
    var record = NEURAL_DESCRIPTOR_ADMITTED.get_or_create_ptr()
    var slot = -1
    for i in range(len(record[].modules)):  # small-loop(record: admitted binding modules): a few compiled extension modules per process, never data
        if record[].modules[i] == module_address:
            if record[].dicts[i] == dict_address and record[].sizes[i] == size:
                return fields
            slot = i
            break
    _neural_descriptor_keys(fields, bi.str)
    if slot >= 0:
        record[].dicts[slot] = dict_address
        record[].sizes[slot] = size
    else:
        if len(record[].modules) >= 64:
            record[].modules.clear()
            record[].dicts.clear()
            record[].sizes.clear()
        record[].modules.append(module_address)
        record[].dicts.append(dict_address)
        record[].sizes.append(size)
    return fields


def neural_descriptor_entry(module: PythonObject, name: String, required: Bool = True) raises -> PythonObject:
    # Read an exact module dictionary, never its optional Python __getattr__.
    # Metadata/ownership doors must be compiled entries too; provider execution
    # alone would not prevent callbacks during admission or readback.
    var bi = native_loaded_module("builtins")
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: native provider requires an exact compiled module")
    # One borrowed keyed read of the admitted namespace (every key an exact
    # str, so no caller hash or equality hook runs), not a bound `dict.get`
    # method call per primitive operation (lane perf-provider-cache).
    var fields = _neural_descriptor_module_fields_once(module, bi)
    var key = PythonObject(name)
    ref cpy = Python().cpython()
    var raw = cpy.PyDict_GetItemWithError(fields._obj_ptr, key._obj_ptr)
    if not raw and cpy.PyErr_Occurred():
        raise cpy.unsafe_get_error()
    var entry = PythonObject(from_borrowed=raw) if raw else PythonObject(None)
    if entry is None and not required:
        return entry
    if entry is None or py_type(entry) is not py_type(bi.len) or _neural_descriptor_function_owner(entry) != module._obj_ptr:
        raise Error("TypeError: native provider requires compiled entry " + name)
    return entry


def neural_descriptor_connect(factory: PythonObject, var name: String, operation_count: Int = 6) raises -> PythonObject:
    if operation_count < 1:
        raise Error("native provider: operation_count must be positive")
    var bi = native_loaded_module("builtins")
    if py_type(factory) is not py_type(bi.len):
        raise Error("TypeError: neural native ABI factory must be a compiled extension entry")
    var raw_module = _neural_descriptor_function_owner(factory)
    # Cold connection owns the module reference; reuse/admission comparisons
    # above/below only need borrowed pointer identity while factory is pinned.
    var module = PythonObject(None)
    if raw_module:
        module = PythonObject(from_owned=Python().cpython().Py_NewRef(raw_module))
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: neural native ABI factory must retain its extension module")
    # Lifetime belongs to the execution owner, which can use
    # neural_descriptor_reuse. A global weak-key record containing this
    # provider would retain factory through provider.factory and never expire.
    return _neural_descriptor_connect_new(factory, module, name, operation_count)


struct NeuralDescriptorConnections(Defaultable, Movable):
    # Providers connected once per process (lane perf-provider-cache,
    # 2026-11-06), keyed by (module, factory entry, ABI name, operation
    # count). Each record holds its module, factory and provider strongly: a
    # loaded extension module is never unloaded, so the retention costs
    # nothing, and holding them makes the identity compare sound. A replaced
    # factory entry (read from the live module namespace on every call) or a
    # replaced module misses and connects again, with the same refusals.
    var modules: List[PythonObject]
    var factories: List[PythonObject]
    var names: List[String]
    var counts: List[Int]
    var providers: List[PythonObject]

    def __init__(out self):
        self.modules = List[PythonObject]()
        self.factories = List[PythonObject]()
        self.names = List[String]()
        self.counts = List[Int]()
        self.providers = List[PythonObject]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever NeuralDescriptorConnections' layout changes.
comptime NEURAL_DESCRIPTOR_CONNECTIONS = _Global[StorageType=NeuralDescriptorConnections,
    name="MojolearnNeuralDescriptorConnectionsV1", init_fn=NeuralDescriptorConnections.__init__]


def neural_descriptor_connect_once(module: PythonObject, factory_name: String, var name: String,
                                   operation_count: Int = 6) raises -> PythonObject:
    """The provider of `module`'s compiled factory `factory_name` for ABI
    `name`, connected on first use and reused while the module's live
    namespace still holds the same factory entry. The entry read
    (`neural_descriptor_entry`) runs every call, so a replaced entry or module
    reconnects and refuses exactly as `neural_descriptor_connect` does."""
    if operation_count < 1:
        raise Error("native provider: operation_count must be positive")
    var factory = neural_descriptor_entry(module, factory_name)
    var g = NEURAL_DESCRIPTOR_CONNECTIONS.get_or_create_ptr()
    for i in range(len(g[].modules)):  # small-loop(connected providers): a few compiled extension modules per process, never data
        if (g[].modules[i] is module and g[].factories[i] is factory and g[].names[i] == name
                and g[].counts[i] == operation_count):
            return g[].providers[i]
    var provider = neural_descriptor_connect(factory, name, operation_count)
    var slot = -1
    for i in range(len(g[].modules)):  # small-loop(connected providers): replace a stale record of the same module/ABI
        if g[].modules[i] is module and g[].names[i] == name and g[].counts[i] == operation_count:
            slot = i
            break
    if slot >= 0:
        g[].factories[slot] = factory
        g[].providers[slot] = provider
        return provider
    if len(g[].modules) >= 64:
        g[].modules.clear()
        g[].factories.clear()
        g[].names.clear()
        g[].counts.clear()
        g[].providers.clear()
    g[].modules.append(module)
    g[].factories.append(factory)
    g[].names.append(name)
    g[].counts.append(operation_count)
    g[].providers.append(provider)
    return provider


def _neural_descriptor_connect_new(factory: PythonObject, module: PythonObject, var name: String, operation_count: Int) raises -> PythonObject:
    var capsule = py_call(factory)
    ref cpy = Python().cpython()
    if cpy.lib.get_function[c_int]("PyCapsule_IsValid")(
        capsule._obj_ptr, name.as_c_string_slice().unsafe_ptr()) == 0:
        raise Error("neural native ABI: invalid capsule or version")
    var raw = cpy.lib.get_function[Optional[MutPointer[UInt8, MutUntrackedOrigin]]]("PyCapsule_GetPointer")(
        capsule._obj_ptr, name.as_c_string_slice().unsafe_ptr())
    if not raw:
        raise cpy.unsafe_get_error()
    var table = MutPointer[NativeDescriptorCall, MutUntrackedOrigin](unsafe_from_address=Int(raw.value()))
    return PythonObject(alloc=NeuralDescriptorProvider(module, capsule, factory, name^, table[0], operation_count))


def neural_descriptor_reuse(previous: PythonObject, factory: PythonObject, name: String, operation_count: Int = 6) raises -> PythonObject:
    if operation_count < 1:
        raise Error("native provider: operation_count must be positive")
    # Caller admits this exact compiled factory before entering here. Retain
    # factory identity as well as its module: swapping an ABI entry on the same
    # module must reconnect, even if the facade/canonical module did not change.
    if previous is not None:
        var held = previous.downcast_value_ptr[NeuralDescriptorProvider]()
        if held[].factory is factory and held[].module._obj_ptr == _neural_descriptor_function_owner(factory) and held[].name == name and len(held[].calls) == operation_count:
            return previous
    return neural_descriptor_connect(factory, name, operation_count)


def neural_descriptor_cached(
    fields: PythonObject, entries: PythonObject, slot: String, factory: String, name: String, eligible: Bool,
) raises -> PythonObject:
    # Construction/admission ONLY, never call this resolver from token/layer loops.
    # Capture the explicit candidate switch once per owner. A later native
    # binding replacement reconnects rather than reusing a stale table/image.
    var bi = native_loaded_module("builtins")
    _neural_descriptor_dictionary(fields, bi)
    var entry_fields = _neural_descriptor_entry_fields(entries, bi)
    var canonical = py_get(entry_fields, "canonical", entries)
    var binding_key = slot + "_binding"
    if py_contains(fields, slot) and py_get(fields, binding_key) is canonical:
        return fields[slot]
    var enabled_key = slot + "_enabled"
    if not py_contains(fields, enabled_key):
        py_setitem(fields, enabled_key, PythonObject(getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1"))
    if py_type(fields[enabled_key]) is not bi.bool:
        raise Error("TypeError: neural provider enable flag requires an exact bool")
    var result = PythonObject(None)
    if eligible and Bool(py=fields[enabled_key]):
        var mode_name = "training_numeric_mode" if factory == "neural_primitives_api_v1" else "mamba_numeric_mode"
        var mode_entry = py_get(entry_fields, mode_name)
        if py_type(mode_entry) is not py_type(bi.len):
            raise Error("TypeError: neural native ABI mode entry must be compiled")
        if Int(py=py_call(mode_entry)) == 1:
            result = neural_descriptor_connect(py_get(entry_fields, factory), name,
                29 if factory == "neural_primitives_api_v1" else 6)
    py_setitem(fields, slot, result)
    py_setitem(fields, binding_key, canonical)
    return result


def _neural_descriptor_admit_entry(provider: PythonObject, factory: PythonObject) raises:
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var bi = native_loaded_module("builtins")
    if py_type(factory) is not py_type(bi.len) or _neural_descriptor_function_owner(factory) != owner[].module._obj_ptr:
        raise Error("TypeError: native state admission requires the provider's compiled entry")


def neural_descriptor_construct(provider: PythonObject, factory: PythonObject) raises -> PythonObject:
    # A no-argument compiled state constructor is lifetime admission only;
    # subsequent computation uses the descriptor table rather than this door.
    _neural_descriptor_admit_entry(provider, factory)
    return py_call(factory)


def neural_descriptor_handle(provider: PythonObject, factory: PythonObject, value: PythonObject) raises -> Int:
    # The engine checks the exact value type. Retain value and provider for each
    # subsequent handle use; never unwrap a generic extension object layout.
    _neural_descriptor_admit_entry(provider, factory)
    var handle = Int(py=py_call(factory, value))
    if handle == 0:
        raise Error("native provider: state admission returned a null handle")
    return handle


def neural_descriptor_matches(provider: PythonObject, canonical: PythonObject) raises -> Bool:
    # Native owner identity only: no getattr, Python callback or cache lookup.
    return provider is not None and provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is canonical


def neural_descriptor_execute(
    provider: PythonObject, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64],
    release_gil: Bool = True,
) raises -> Int:
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if operation < 0 or operation >= len(owner[].calls):
        raise Error("neural native ABI: unknown operation")
    owner[].calls[operation] += 1
    var loan = NativeCallWorkspaceLease(Int(owner[].workspace.unsafe_ptr()))
    var scratch = loan.pointer()
    try:
        var result = 0
        if release_gil:
            with GILReleased(Python()):
                result = native_descriptor_call_spans(owner[].entry, operation,
                    Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
                    Int(floats.unsafe_ptr()), len(floats), scratch[].output)
        else:
            # Native resource registry admission/finalization whose ownership
            # requires the admission lock uses this branch; computation keeps True.
            result = native_descriptor_call_spans(owner[].entry, operation,
                Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
                Int(floats.unsafe_ptr()), len(floats), scratch[].output)
        return result
    finally:
        loan.release()
        # Span addresses and the workspace pointer do not retain these owners.
        _ = addresses
        _ = ints
        _ = floats
        _ = provider


def neural_descriptor_reserved_execute(
    provider: PythonObject, session_owner: PythonObject, handle: Int,
    reserve: Int, release: Int, operation: Int,
    addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    # Common provider ownership/GIL boundary for named tables with native
    # reservation operations. The caller obtained handle from that table's
    # checked native admission entry; no Python extension layout is inspected.
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var count = len(owner[].calls)
    if reserve < 0 or reserve >= count or release < 0 or release >= count or operation < 0 or operation >= count:
        raise Error("neural native ABI: unknown operation")
    var loan = NativeCallWorkspaceLease(Int(owner[].workspace.unsafe_ptr()))
    var scratch = loan.pointer()
    try:
        # One exclusive vector supplies both the session-only reserve/release
        # span and the full prefixed operation span. Refresh every launch word;
        # only allocation capacity survives calls, never argument values.
        scratch[].words.resize(len(ints) + 1, 0)
        scratch[].words[0] = handle
        if len(ints) > 0:
            memcpy(dest=scratch[].words.unsafe_ptr() + 1, src=ints.unsafe_ptr(), count=len(ints))
        var words = Int(scratch[].words.unsafe_ptr())
        owner[].calls[reserve] += 1
        _ = native_descriptor_call_spans(owner[].entry, reserve, 0, 0, words, 1, 0, 0, scratch[].output)
        var result = 0
        try:
            owner[].calls[operation] += 1
            with GILReleased(Python()):
                result = native_descriptor_call_spans(owner[].entry, operation,
                    Int(addresses.unsafe_ptr()), len(addresses), words, len(ints) + 1,
                    Int(floats.unsafe_ptr()), len(floats), scratch[].output)
        except problem:
            owner[].calls[release] += 1
            _ = native_descriptor_call_spans(owner[].entry, release, 0, 0, words, 1, 0, 0, scratch[].output)
            raise problem
        owner[].calls[release] += 1
        _ = native_descriptor_call_spans(owner[].entry, release, 0, 0, words, 1, 0, 0, scratch[].output)
        return result
    finally:
        loan.release()
        _ = addresses
        _ = ints
        _ = floats
        _ = provider
        _ = session_owner


def neural_descriptor_info(provider: PythonObject) raises -> PythonObject:
    # Explicit untimed diagnostic for coverage retention; never called by the
    # executor itself. Counts record attempts, including refused native calls.
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var counts = Python.list()
    for value in owner[].calls:  # small-loop(the admitted table's named native operations)
        _ = py_call(counts.append, value)
    return Python.tuple(PythonObject(owner[].name), counts)


struct NativeDescriptorResource(Movable, Writable):
    # Generic native registry resource. This provider and its compiled image
    # outlive the handle; context is opaque native metadata for that engine's
    # close operation, not a guessed extension object layout.
    var provider: PythonObject
    var handle: Int
    var close_operation: Int
    var context: List[Int]

    def __init__(out self, provider: PythonObject, handle: Int, close_operation: Int, var context: List[Int]):
        self.provider = provider
        self.handle = handle
        self.close_operation = close_operation
        self.context = context^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NativeDescriptorResource(open=", self.handle != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def close(mut self) raises:
        var handle = self.handle
        self.handle = 0
        if handle != 0:
            var addresses: List[Int] = [handle]
            _ = neural_descriptor_execute(self.provider, self.close_operation, addresses,
                self.context, List[Float64](), False)

    def __deinit__(deinit self):
        if self.handle != 0:
            try:
                var addresses: List[Int] = [self.handle]
                _ = neural_descriptor_execute(self.provider, self.close_operation, addresses,
                    self.context, List[Float64](), False)
            except:
                pass


def neural_descriptor_adopt(provider: PythonObject, handle: Int, close_operation: Int,
                             var context: List[Int]) raises -> PythonObject:
    var owner = NativeDescriptorResource(provider, handle, close_operation, context^)
    # RAII closes the native resource if boxing/registration fails.
    return PythonObject(alloc=owner^)


def neural_descriptor_resource_handle(value: PythonObject, context: List[Int]) raises -> Int:
    var owner = value.downcast_value_ptr[NativeDescriptorResource]()
    if len(owner[].context) != len(context):
        raise Error("TypeError: native resource belongs to another registry")
    for i in range(len(context)):  # driver-loop: over retained native resource identity words
        if owner[].context[i] != context[i]:
            raise Error("TypeError: native resource belongs to another registry")
    return owner[].handle


def neural_descriptor_resource_close(value: PythonObject, context: List[Int]) raises:
    _ = neural_descriptor_resource_handle(value, context)
    value.downcast_value_ptr[NativeDescriptorResource]()[].close()
