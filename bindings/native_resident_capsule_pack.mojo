# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A PyCapsule that owns one resident device handle of a compiled binding.

`resident_capsule(free, handle)` returns a capsule holding a strong reference
to the minting binding's release entry `free` and the handle word. When the
last reference to the capsule goes (a prepared-owner record replaced by a
refit or a new fitted array, or dropped by the model's death through the
`native_prepared_owner` weakref), its native destructor calls
`free(handle)` (no Python function runs) and so returns the device memory to
the binding that minted it. Moved out of bindings/decomp_kit_native.mojo
(lane perf2-pca-decomp) by lane ne1-a-decomp-tsa so the decomposition kit,
the TSA owners and the random-projection owner share one copy (plan section 4,
lesson 8). Nothing here touches data.
"""
# The PACK half of the former `bindings/native_resident_capsule.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/native_resident_capsule_unpack.mojo`.
from std.ffi import external_call, c_size_t
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call
from bindings.cpython_table import cpython_table
from bindings.native_resident_capsule_run import _RESIDENT_NAMES, _ResidentWords, _ResidentBytes


from bindings.native_resident_capsule_unpack import (
    _resident_destroy,
)


def resident_capsule(free: PythonObject, handle: Int) raises -> PythonObject:
    # Two words: a strong reference to the minting binding's free entry and
    # the device buffer id. On failure the buffer is returned at once.
    ref cpy = Python().cpython()
    var raw = external_call["malloc", Optional[_ResidentBytes]](c_size_t(16))
    if not raw:
        _ = py_call(free, handle)
        raise Error("resident device record allocation failed")
    var words = _ResidentWords(unsafe_from_address=Int(raw.value()))
    words[0] = Int(cpy.Py_NewRef(free._obj_ptr))
    words[1] = handle
    var names = _RESIDENT_NAMES.get_or_create_ptr()
    var result = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(
        raw.value(), names[].capsule_name.as_c_string_slice().unsafe_ptr(), _resident_destroy)
    if not result:
        cpy.lib.get_function[NoneType]("Py_DecRef")(words[0])
        external_call["free", NoneType](raw.value())
        var failure = cpy.unsafe_get_error()
        _ = py_call(free, handle)
        raise failure^
    return PythonObject(from_owned=result)
