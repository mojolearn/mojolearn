# SPDX-License-Identifier: Apache-2.0
# Test-only native ABI probe. It instantiates the real ArrayMetadata methods;
# fabricated addresses are metadata only and are never dereferenced.
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from core.cast_in import CastIn, CAST_KIND_FLOAT, NPY_HALF
from providers.python.array_metadata_pyapi import ArrayMetadata


def replace_binding(source: PythonObject, original_owner: PythonObject,
                    converted_f32: PythonObject, converted_f64: PythonObject, route: Int) raises -> PythonObject:
    var metadata = ArrayMetadata(source, List[Int]([2, 3]),
        CastIn(256, CAST_KIND_FLOAT, 2, 2, 3, 6, 2), 6, NPY_HALF,
        True, False, True, original_owner, False)
    if route == 32 or route == 3264:
        metadata.replace_f32(converted_f32, 512)
    if route == 64 or route == 3264 or route == 6432:
        metadata.replace_f64(converted_f64, 768)
    if route == 6432:
        metadata.replace_f32(converted_f32, 512)
    return Python.tuple(metadata.owner, metadata.keep, metadata.copied,
        metadata.owned_writable, metadata.source.addr, metadata.source.itemsize,
        metadata.source.s0, metadata.source.s1, metadata.count, metadata.python_shape())


@export
def PyInit_array_metadata_owner_probe() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("array_metadata_owner_probe")
        module.def_function[replace_binding]("replace")
        return module.finalize()
    except error:
        abort(String(error))
