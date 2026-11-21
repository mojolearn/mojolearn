# SPDX-License-Identifier: Apache-2.0
# `cfb_borrow_copy`, the one helper bindings/cfb_spectral_door.mojo imports
# from here. DBSCAN's public operations left this file for
# bindings/dbscan_public_native_unpack.mojo (UNPACK) and
# bindings/dbscan_public_native_pack.mojo (operators, PACK).
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.classical_arrays_door import classical_address, classical_is_array
from bindings.family_common_unpack import FAMILY_F32P
from bindings.family_common_pack import family_array
from bindings.native_loaded_module_door import native_loaded_module


def cfb_borrow_copy(x: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`x if input_copied_ else x.copy()` as a mojolearn Array: an Array the
    cast made is kept, anything else (the caller's own buffer) is copied so
    a later write by the caller cannot move what predict reads."""
    if classical_is_array(x, context) and not (x is source):
        return x
    var b = native_loaded_module("builtins")
    var out = family_array(py_call(b.tuple, x.shape), "<f4", context)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=Int(py=x.size))
    _ = out  # keep the Python object alive through the native call above
    return out
