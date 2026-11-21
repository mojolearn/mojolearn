# SPDX-License-Identifier: Apache-2.0
# Shared strict float32 / int32 training-buffer admission into a native buffer.
from std.python import PythonObject
from bindings.py_balanced import py_type
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_arrays_door import classical_input, classical_address, classical_dtype_str
from core.training_input_copy import training_copy_f32, training_copy_words32


def training_words32_into(source: PythonObject, dims: List[Int], name: String,
                          context: PythonObject, integer: Bool, dest: Int) raises:
    """Admit `source` (exact dtype and shape; float32 also finite) and copy
    its `dims` words to the caller's buffer at `dest`, which holds the
    product of `dims` 4-byte words. The byte LM trainer's native state owner
    passes its own buffers; the Array-returning wrappers
    (`training_f32_owned`, `training_i32_owned`) had no other caller and were
    removed with the Python-held state (lane byte-state, 2026-11-21)."""
    classical_admit_value(source, context)
    var bi = native_loaded_module("builtins")
    var kind = String("int32" if integer else "float32")
    if py_type(source) is bi.list or py_type(source) is bi.tuple:
        raise Error("TypeError: " + name + " must be a " + kind + " buffer")
    var value = classical_input(source, context, name)
    var shape = value.shape
    var strides = value.strides
    var item = Int(py=value.itemsize)
    var dtype = String("<i4" if integer else "<f4")
    if classical_dtype_str(value, context) != dtype or item != 4:
        raise Error("TypeError: " + name + " must be a " + kind + " buffer")
    if len(shape) != len(dims):
        raise Error(name + " has an unexpected shape")
    var native_strides = List[Int]()
    for i in range(len(dims)):  # small-loop(the admitted buffer rank)
        if Int(py=shape[i]) != dims[i]:
            raise Error(name + " has an unexpected shape")
        native_strides.append(Int(py=strides[i]))
    var source_addr = Int(py=classical_address(value))
    if integer:
        training_copy_words32(source_addr, dest, dims, native_strides)
    else:
        training_copy_f32(source_addr, dest, dims, native_strides, name)
    _ = value^


# TOMBSTONE: training_f32_cast_owned (admission helper, merged into the one path) deleted 2026-11-08 by lane a4-admit-sequence-tsa-neural; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A4_ADMISSION_HELPERS.patch. It is
# `admit_nd_owned_f32` (bindings/admitted_sequence.mojo): the same cast and
# refusal, the finite test riding in the parallel cast.
