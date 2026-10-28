# SPDX-License-Identifier: Apache-2.0
# Shared strict float32 training-buffer snapshot and owned copy at admission.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_input, classical_address, classical_dtype_str, classical_cast_kind
from bindings.tsa_hw_native import tsa_array
from core.xseq_marshal import xseq_cast_nd, xseq_all_finite_f32
from core.training_input_copy import training_copy_f32, training_copy_words32


def training_f32_owned(source: PythonObject, dims: List[Int], name: String,
                       context: PythonObject) raises -> PythonObject:
    return training_words32_owned(source, dims, name, context, False)


def training_i32_owned(source: PythonObject, dims: List[Int], name: String,
                       context: PythonObject) raises -> PythonObject:
    return training_words32_owned(source, dims, name, context, True)


def training_words32_owned(source: PythonObject, dims: List[Int], name: String,
                           context: PythonObject, integer: Bool) raises -> PythonObject:
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
    var count = 1
    var native_strides = List[Int]()
    var output_shape = Python.list()
    for i in range(len(dims)):  # small-loop(the admitted buffer rank)
        if Int(py=shape[i]) != dims[i]:
            raise Error(name + " has an unexpected shape")
        count *= dims[i]
        native_strides.append(Int(py=strides[i]))
        _ = py_call(output_shape.append, dims[i])
    var source_addr = Int(py=classical_address(value))
    var output = tsa_array("i" if integer else "f", dtype, output_shape, count, 4, context)
    if integer:
        training_copy_words32(source_addr, Int(py=classical_address(output)), dims, native_strides)
    else:
        training_copy_f32(source_addr, Int(py=classical_address(output)), dims, native_strides, name)
    _ = value^
    return output


def training_f32_cast_owned(source: PythonObject, dims: List[Int], name: String,
                            context: PythonObject) raises -> PythonObject:
    # Host APIs historically accept native numeric dtypes and nested sequences,
    # unlike strict training state buffers. Preserve Array.astype's integer ->
    # double -> float rule through the existing shared native caster.
    classical_admit_value(source, context)
    var value = classical_input(source, context, name)
    var shape = value.shape
    var strides = value.strides
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, name, context)
    if len(shape) != len(dims):
        raise Error(name + " has an unexpected shape")
    var count = 1
    var native_strides = List[Int]()
    var output_shape = Python.list()
    for axis in range(len(dims)):  # driver-loop: over admitted array rank metadata
        if Int(py=shape[axis]) != dims[axis]:
            raise Error(name + " has an unexpected shape")
        count *= dims[axis]
        native_strides.append(Int(py=strides[axis]))
        _ = py_call(output_shape.append, dims[axis])
    var result = tsa_array("f", "<f4", output_shape, count, 4, context)
    var address = Int(py=classical_address(result))
    var pointer = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    xseq_cast_nd(Int(py=classical_address(value)), dims, native_strides, item, kind, True, pointer)
    if not xseq_all_finite_f32(pointer, count):
        raise Error(name + " must be finite")
    _ = value^
    return result
