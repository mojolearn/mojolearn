# SPDX-License-Identifier: Apache-2.0
# Buffer readings of the native linalg owner (bindings/linalg_ops_native.mojo):
# `bool(value)` and `dtype_name`'s refusal name. The admission itself (the
# metadata capture, the zero-copy borrow, the parallel conversion by
# Array.astype's rules, the zero-filled Arrays and the shape tuples) is the one
# multi-dtype admission path, bindings/admitted_input.mojo (`AdmittedTyped`,
# `admitted_typed_c`, `admit_typed`, `admitted_typed_output`, `admitted_zeros`,
# `AdmittedBuf`).
from std.python import PythonObject
from bindings.py_balanced import py_call
from bindings.native_loaded_module import native_loaded_module
from bindings.tensor_view import TensorView, base_format, _format_name, _typestr_name

# TOMBSTONE: LinAdmission, LinBuf, LinBufferPin, register_lin_buffer_pin,
# lin_admitted_c, lin_borrow_array, lin_convert, lin_typed, lin_admitted_output,
# lin_admit_exporter, lin_new, lin_array, lin_width, lin_tuple (moved onto the one
# admission path, bindings/admitted_input.mojo) and lin_view, lin_c_order (no
# caller) deleted 2026-11-09 by lane a6-device-x-family-classical; code
# recoverable at f59d913b0b.
# Restore: experiments/removed/A6_LIN_ADMISSION.patch.


def lin_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=py_call(native_loaded_module("builtins").bool, value))


def lin_is_f32(v: TensorView) -> Bool:
    """`_bufcheck.is_native_f32(pb.format)`."""
    return base_format(v.fmt) == "f"


def lin_dtype_name(obj: PythonObject, v: TensorView) raises -> String:
    # The admitted layout already carries dtype information. Reading an
    # arbitrary object's dtype/name properties would re-enter user metadata.
    if v.is_array:
        var name = _typestr_name(v.dtype)
        if name != "":
            return name
    return _format_name(base_format(v.fmt), v.itemsize, v.fmt)
