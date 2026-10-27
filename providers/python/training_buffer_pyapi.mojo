# SPDX-License-Identifier: Apache-2.0
# Shared strict float32 training-buffer snapshot and owned copy at admission.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_input, classical_address, classical_dtype_str
from bindings.tsa_hw_native import tsa_array
from core.training_input_copy import training_copy_f32


def training_f32_owned(source: PythonObject, dims: List[Int], name: String,
                       context: PythonObject) raises -> PythonObject:
    classical_admit_value(source, context)
    var bi = native_loaded_module("builtins")
    if py_type(source) is bi.list or py_type(source) is bi.tuple:
        raise Error("TypeError: " + name + " must be a float32 buffer")
    var value = classical_input(source, context, name)
    var shape = value.shape
    var strides = value.strides
    var item = Int(py=value.itemsize)
    if classical_dtype_str(value, context) != "<f4" or item != 4:
        raise Error("TypeError: " + name + " must be a float32 buffer")
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
    var output = tsa_array("f", "<f4", output_shape, count, 4, context)
    training_copy_f32(source_addr, Int(py=classical_address(output)), dims, native_strides, name)
    _ = value^
    return output
