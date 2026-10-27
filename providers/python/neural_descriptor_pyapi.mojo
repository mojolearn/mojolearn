# SPDX-License-Identifier: Apache-2.0
# Setup/ownership boundary for explicitly named native execution ABIs. No
# unwrapping of arbitrary Python functions, extension layouts or signatures.
from std.ffi import c_int
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
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

    def __init__(out self, module: PythonObject, capsule: PythonObject, factory: PythonObject, var name: String, entry: NativeDescriptorCall):
        self.module = module
        self.capsule = capsule
        self.factory = factory
        self.name = name^
        self.entry = entry
        self.calls = List[Int](length=6, fill=0)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NeuralDescriptorProvider(", self.name, ", calls=", self.calls, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def neural_descriptor_connect(factory: PythonObject, var name: String) raises -> PythonObject:
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
    return PythonObject(alloc=NeuralDescriptorProvider(module, capsule, factory, name^, table[0]))


def neural_descriptor_reuse(previous: PythonObject, factory: PythonObject, name: String) raises -> PythonObject:
    # Caller admits this exact compiled factory before entering here. Retain
    # factory identity as well as its module: swapping an ABI entry on the same
    # module must reconnect, even if the facade/canonical module did not change.
    if previous is not None:
        var held = previous.downcast_value_ptr[NeuralDescriptorProvider]()
        if held[].factory is factory and held[].module is factory.__self__ and held[].name == name:
            return previous
    return neural_descriptor_connect(factory, name)


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
            result = neural_descriptor_connect(py_call(bi.getattr, entries, factory), name)
    py_setitem(fields, slot, result)
    py_setitem(fields, binding_key, canonical)
    return result


def neural_descriptor_matches(provider: PythonObject, canonical: PythonObject) raises -> Bool:
    # Native owner identity only: no getattr, Python callback or cache lookup.
    return provider is not None and provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is canonical


def neural_descriptor_execute(
    provider: PythonObject, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if operation < 0 or operation >= len(owner[].calls):
        raise Error("neural native ABI: unknown operation")
    owner[].calls[operation] += 1
    var result = 0
    with GILReleased(Python()):
        result = native_descriptor_call(owner[].entry, operation, addresses, ints, floats)
    _ = provider
    return result


def neural_descriptor_info(provider: PythonObject) raises -> PythonObject:
    # Explicit untimed diagnostic for coverage retention; never called by the
    # executor itself. Counts record attempts, including refused native calls.
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var counts = Python.list()
    for value in owner[].calls:  # small-loop(six named native operations)
        _ = py_call(counts.append, value)
    return Python.tuple(PythonObject(owner[].name), counts)
