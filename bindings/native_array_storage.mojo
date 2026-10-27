# SPDX-License-Identifier: Apache-2.0
# Allocate the existing array.array stores without calling its audited Python
# constructor. PySequence_Repeat is public CPython stable ABI; on an exact
# array.array its native array_repeat allocates a fresh ordinary array owner.
# Fixed one-word seeds are import-time metadata, never caller input or output.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_address import native_buffer_address
from bindings.py_balanced import py_call, py_type, py_get


def _storage_metadata() raises -> PythonObject:
    var module = native_loaded_module("mojolearn._native_array_storage")
    ref cpy = Python().cpython()
    var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
    if not fields:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=cpy.Py_NewRef(fields))
    _ = module  # retain the module until its dictionary reference is owned
    return result


def _storage_type(metadata: PythonObject) raises -> PythonObject:
    var cls = py_get(metadata, "ARRAY_TYPE")
    # Both metadata fields are Python-visible. A replacement class must be
    # refused before len/getitem/typecode or the sequence-repeat protocol.
    if py_type(cls) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native array storage requires its extension type")
    var module = native_loaded_module("array")
    ref cpy = Python().cpython()
    var owner = cpy.lib.get_function[PyObjectPtr]("PyType_GetModule")(cls._obj_ptr)
    if not owner:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("TypeError: native array storage requires its extension type")
    if owner != module._obj_ptr:
        raise Error("TypeError: native array storage type has a different extension owner")
    return cls


def native_array_zeros(code: String, count: Int) raises -> PythonObject:
    if count < 0:
        raise Error("ValueError: native array element count must be nonnegative")
    var metadata = _storage_metadata()
    var cls = _storage_type(metadata)
    var seeds = py_get(metadata, "ZERO_STORES")
    if py_type(seeds) is not py_type(Python.dict()):
        raise Error("TypeError: native array storage metadata is unavailable")
    var seed = py_get(seeds, PythonObject(code))
    if seed is None or py_type(seed) is not cls:
        raise Error("TypeError: unsupported native array storage typecode " + code)
    if len(seed) != 1 or String(seed.typecode) != code:
        raise Error("TypeError: native array storage seed changed")
    # Numerical equality would admit -0.0. Zero-filled buffers require every
    # raw byte to be zero, including the IEEE sign bit in float prototypes.
    var address = native_buffer_address(seed)
    var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(Int(py=seed.itemsize)):  # small-loop(seed.itemsize: scalar storage byte width): at most eight native storage bytes
        if bytes[i] != 0:
            raise Error("TypeError: native array storage seed changed")
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PySequence_Repeat")(seed._obj_ptr, count)
    _ = seed  # retain the original owner through the native repeat operation
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def native_array_empty(code: String) raises -> PythonObject:
    return native_array_zeros(code, 0)


def native_array_from_bytes(code: String, source: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    if cls is not b.bytes and cls is not b.bytearray and cls is not b.memoryview:
        raise Error("TypeError: native array bytes require an exact native byte buffer")
    var result = native_array_empty(code)
    _ = py_call(result.frombytes, source)
    return result


def native_array_from_values(code: String, source: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    var metadata = _storage_metadata()
    var native_array = _storage_type(metadata)
    if cls is not b.list and cls is not b.tuple and cls is not native_array:
        raise Error("TypeError: native array values require a native array or exact numeric sequence")
    # Callers normalize NumPy/buffer leaves through their existing native input
    # paths. Reject Python scalar conversion protocols before native append can
    # invoke them. array.array iteration itself yields exact Python primitives.
    for i in range(len(source)):  # driver-loop: over caller-provided scalar ABI types before native conversion
        var item_type = py_type(source[i])
        if item_type is not b.int and item_type is not b.float and item_type is not b.bool:
            raise Error("TypeError: native array values cannot invoke scalar conversion hooks")
    var result = native_array_empty(code)
    # array(code, another_array) converts each scalar when their typecodes
    # differ. array.extend(another_array) instead refuses that input. Preserve
    # constructor conversion through an exact native list, whose leaves were
    # admitted above; no Python iterator or scalar-conversion hook is called.
    var values = source
    if cls is native_array and String(source.typecode) != code:
        values = py_call(b.list, source)
    _ = py_call(result.extend, values)
    return result
