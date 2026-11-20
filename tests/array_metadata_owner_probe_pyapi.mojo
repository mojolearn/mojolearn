# SPDX-License-Identifier: Apache-2.0
# Test-only native ABI probe. It instantiates the real ArrayMetadata methods;
# fabricated addresses are metadata only and are never dereferenced.
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from std.python._cpython import PyObjectPtr
from core.cast_in import CastIn, CAST_KIND_FLOAT, NPY_HALF
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_python_shape


def replace_binding(source: PythonObject, original_owner: PythonObject,
                    converted_f32: PythonObject, converted_f64: PythonObject, route: Int) raises -> PythonObject:
    var metadata = ArrayMetadata(List[Int]([2, 3]),
        CastIn(256, CAST_KIND_FLOAT, 2, 2, 3, 6, 2), 6, NPY_HALF,
        True, False, True, False)
    # The pin is what the door holds; a conversion replaces only `keep`.
    var pin = ArrayPin(source, original_owner)
    if route == 32 or route == 3264:
        metadata.replace_f32(512)
        pin.keep = converted_f32
    if route == 64 or route == 3264 or route == 6432:
        metadata.replace_f64(768)
        pin.keep = converted_f64
    if route == 6432:
        metadata.replace_f32(512)
        pin.keep = converted_f32
    return Python.tuple(pin.owner, pin.keep, metadata.copied,
        metadata.owned_writable, metadata.source.addr, metadata.source.itemsize,
        metadata.source.s0, metadata.source.s1, metadata.count, array_python_shape(metadata))


def replace_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return replace_binding(args[0], args[1], args[2], args[3], Int(py=args[4])).steal_data()
    except error:
        return raise_python_exception(error)


@export
def PyInit_array_metadata_owner_probe() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("array_metadata_owner_probe")
        module.def_py_c_function(replace_c, "replace")
        return module.finalize()
    except error:
        abort(String(error))
