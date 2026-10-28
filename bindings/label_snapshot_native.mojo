# SPDX-License-Identifier: Apache-2.0
# Array label values from one retained logical dtype/shape/stride snapshot.
# Keep the GP-specific helper and its API intact; both use native scalar bits.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.native_scalar_decode import NativeScalarBuffer
from core.native_scalar_decode import NativeRealBits, native_scalar_real
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once


def _label_word(snapshot: ArrayMetadata, offset: Int) raises -> PythonObject:
    var letter = String(snapshot.dtype_name[byte=1:2])
    var width = snapshot.source.itemsize
    var code = String("q") if letter == "i" else String("Q")
    if letter == "f":
        code = String("e") if width == 2 else String("f") if width == 4 else String("d")
    elif letter == "b":
        code = "?"
    elif letter != "i" and letter != "u":
        raise Error("TypeError: Array labels require a real numeric dtype")
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=snapshot.source.addr)
    var real = native_scalar_real(data, offset, code, width, True)
    if letter == "b":
        return PythonObject(real.truth())
    var decoded = NativeScalarBuffer(real, NativeRealBits(), False, False)
    return decoded.to_python_number(letter != "f")


def _label_axis(snapshot: ArrayMetadata, axis: Int, offset: Int) raises -> PythonObject:
    if axis == len(snapshot.shape):
        return _label_word(snapshot, offset)
    var result = Python.list()
    for i in range(snapshot.shape[axis]):  # driver-loop: native values follow logical strides in each captured label dimension
        _ = py_call(result.append, _label_axis(snapshot, axis + 1, offset + i * snapshot.strides[axis]))
    return result


def label_array_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var shape = array_shape_once(value, context)
    var snapshot = array_metadata_once(value, shape^, "labels", context, allow_unsupported=True)
    # Higher rank metadata is admission-only for numerical CastIn. This label
    # decoder uses captured address/width/strides and declared dtype directly;
    # it never consumes the placeholder CastIn.kind when castable is false.
    var result = _label_axis(snapshot, 0, 0)
    _ = snapshot.keep
    _ = snapshot.owner
    return result
