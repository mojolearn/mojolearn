# SPDX-License-Identifier: Apache-2.0
# GP label transport from a single native Array metadata/ownership snapshot.
# Raw Array._mv can expose uint16 backing for float16, or physical storage
# order rather than logical shape. Decode the declared primitive dtype and
# captured byte strides; integer labels never pass through float32/float64.
# The PACK half of the former `bindings/gp_label_snapshot_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/gp_label_snapshot_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.classical_labels import _label_list, _flatten_label_pins
from bindings.native_scalar_decode_unpack import NativeScalarBuffer
from core.native_scalar_decode import NativeRealBits, native_scalar_real
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_BOOL
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once


def _gp_label_word(data: MutPointer[UInt8, MutUntrackedOrigin], offset: Int,
                   kind: Int, itemsize: Int) raises -> PythonObject:
    var code = String("q") if kind == CAST_KIND_SIGNED else String("Q")
    if kind == CAST_KIND_FLOAT:
        code = String("e") if itemsize == 2 else (String("f") if itemsize == 4 else String("d"))
    elif kind == CAST_KIND_BOOL:
        code = "?"
    var real = native_scalar_real(data, offset, code, itemsize, True)
    if kind == CAST_KIND_BOOL:
        return PythonObject(real.truth())
    var decoded = NativeScalarBuffer(real, NativeRealBits(), False, False)
    return decoded.to_python_number(kind != CAST_KIND_FLOAT)
