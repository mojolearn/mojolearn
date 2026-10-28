# SPDX-License-Identifier: Apache-2.0
# GP label transport from a single native Array metadata/ownership snapshot.
# Raw Array._mv can expose uint16 backing for float16, or physical storage
# order rather than logical shape. Decode the declared primitive dtype and
# captured byte strides; integer labels never pass through float32/float64.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.classical_labels import _label_list
from bindings.native_scalar_decode import NativeScalarBuffer
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


def gp_label_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(value) is not context["array_class"]:
        return _label_list(value, context)
    # Exact class admission precedes every native slot read. The shared
    # metadata owner retains backing storage independently of mutable slots.
    var shape = array_shape_once(value, context)
    var snapshot = array_metadata_once(value, shape^, "GP labels", context)
    var source = snapshot.source
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.addr)
    var result = Python.list()
    if len(snapshot.shape) == 0:
        result = _gp_label_word(data, 0, source.kind, source.itemsize)
    else:
        for row in range(source.rows):  # driver-loop: native scalar boxing at the label boundary
            if len(snapshot.shape) == 1:
                _ = py_call(result.append, _gp_label_word(data, row * source.s0, source.kind, source.itemsize))
            else:
                var values = Python.list()
                for column in range(source.cols):  # driver-loop: preserve logical matrix label shape and strides
                    _ = py_call(values.append, _gp_label_word(data, row * source.s0 + column * source.s1, source.kind, source.itemsize))
                _ = py_call(result.append, values)
    _ = snapshot.keep
    _ = snapshot.owner
    return result
