# SPDX-License-Identifier: Apache-2.0
# Setup/ownership boundary for explicitly named native execution ABIs. No
# unwrapping of arbitrary Python functions, extension layouts or signatures.
from std.ffi import c_int
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.native_loaded_module import native_loaded_module
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call


struct NeuralDescriptorProvider(Movable, Writable):
    var module: PythonObject
    var capsule: PythonObject
    var factory: PythonObject
    var name: String
    var entry: NativeDescriptorCall
    var calls: List[Int]

    def __init__(out self, module: PythonObject, capsule: PythonObject, factory: PythonObject, var name: String, entry: NativeDescriptorCall, operation_count: Int = 6):
        self.module = module
        self.capsule = capsule
        self.factory = factory
        self.name = name^
        self.entry = entry
        self.calls = List[Int](length=operation_count, fill=0)

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


def neural_descriptor_entry(module: PythonObject, name: String, required: Bool = True) raises -> PythonObject:
    # Read an exact module dictionary, never its optional Python __getattr__.
    # Metadata/ownership doors must be compiled entries too; provider execution
    # alone would not prevent callbacks during admission or readback.
    var bi = native_loaded_module("builtins")
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: native provider requires an exact compiled module")
    var entry = py_get(py_fields(module), name)
    if entry is None and not required:
        return entry
    if entry is None or py_type(entry) is not py_type(bi.len) or entry.__self__ is not module:
        raise Error("TypeError: native provider requires compiled entry " + name)
    return entry


def neural_descriptor_connect(factory: PythonObject, var name: String, operation_count: Int = 6) raises -> PythonObject:
    if operation_count < 1:
        raise Error("native provider: operation_count must be positive")
    var bi = native_loaded_module("builtins")
    if py_type(factory) is not py_type(bi.len):
        raise Error("TypeError: neural native ABI factory must be a compiled extension entry")
    var module = factory.__self__
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: neural native ABI factory must retain its extension module")
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
        if held[].factory is factory and held[].module is factory.__self__ and held[].name == name and len(held[].calls) == operation_count:
            return previous
    return neural_descriptor_connect(factory, name, operation_count)


def neural_descriptor_cached(
    fields: PythonObject, entries: PythonObject, slot: String, factory: String, name: String, eligible: Bool,
) raises -> PythonObject:
    # Construction/admission ONLY, never call this resolver from token/layer loops.
    # Capture the explicit candidate switch once per owner. A later native
    # binding replacement reconnects rather than reusing a stale table/image.
    var bi = native_loaded_module("builtins")
    var canonical = py_call(bi.getattr, entries, "canonical", entries)
    var binding_key = slot + "_binding"
    if py_contains(fields, slot) and py_get(fields, binding_key) is canonical:
        return fields[slot]
    var enabled_key = slot + "_enabled"
    if not py_contains(fields, enabled_key):
        py_setitem(fields, enabled_key, PythonObject(getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1"))
    var result = PythonObject(None)
    if eligible and Bool(py=fields[enabled_key]):
        var mode_name = "training_numeric_mode" if factory == "neural_primitives_api_v1" else "mamba_numeric_mode"
        var mode_entry = py_call(bi.getattr, entries, mode_name)
        if py_type(mode_entry) is not py_type(bi.len):
            raise Error("TypeError: neural native ABI mode entry must be compiled")
        if Int(py=py_call(mode_entry)) == 1:
            result = neural_descriptor_connect(py_call(bi.getattr, entries, factory), name,
                25 if factory == "neural_primitives_api_v1" else 6)
    py_setitem(fields, slot, result)
    py_setitem(fields, binding_key, canonical)
    return result


def _neural_descriptor_admit_entry(provider: PythonObject, factory: PythonObject) raises:
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var bi = native_loaded_module("builtins")
    if py_type(factory) is not py_type(bi.len) or factory.__self__ is not owner[].module:
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
    var result = 0
    if release_gil:
        with GILReleased(Python()):
            result = native_descriptor_call(owner[].entry, operation, addresses, ints, floats)
    else:
        # Native resource registry admission/finalization whose ownership
        # requires the admission lock uses this branch; computation keeps True.
        result = native_descriptor_call(owner[].entry, operation, addresses, ints, floats)
    _ = provider
    return result


def neural_descriptor_reserved_execute(
    provider: PythonObject, session_owner: PythonObject, handle: Int,
    reserve: Int, release: Int, operation: Int,
    addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    # Common provider ownership/GIL boundary for named tables with native
    # reservation operations. The caller obtained handle from that table's
    # checked native admission entry; no Python extension layout is inspected.
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    for op in [reserve, release, operation]:  # small-loop(three explicit table operation ids)
        if op < 0 or op >= len(owner[].calls):
            raise Error("neural native ABI: unknown operation")
    var words: List[Int] = [handle]
    for value in ints:  # small-loop(admitted operation scalar span)
        words.append(value)
    owner[].calls[reserve] += 1
    _ = native_descriptor_call(owner[].entry, reserve, [], [handle], [])
    var result = 0
    try:
        owner[].calls[operation] += 1
        with GILReleased(Python()):
            result = native_descriptor_call(owner[].entry, operation, addresses, words, floats)
    except problem:
        owner[].calls[release] += 1
        _ = native_descriptor_call(owner[].entry, release, [], [handle], [])
        raise problem
    owner[].calls[release] += 1
    _ = native_descriptor_call(owner[].entry, release, [], [handle], [])
    _ = provider
    _ = session_owner
    return result


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
