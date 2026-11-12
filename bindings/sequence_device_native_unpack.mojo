# SPDX-License-Identifier: Apache-2.0
# Native sequence tensor ownership, buffer IO and consumed float32 outputs.
from std.python import Python, PythonObject
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from std.python._cpython import PyObjectPtr
from bindings.optimizer_public_words_unpack import optimizer_python_error
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_admission import opt_scalar, opt_value
from bindings.optimizer_device_native import device_shape, shape_count
from bindings.optimizer_public_door import tensor_view
from bindings.tensor_view import TensorView, base_format
from training.optimizer_input_plan import same_shape
from providers.python.array_layout_boundary_pack import native_shape_pack
from bindings.sequence_optimizer_native_door import seq_binding, seq_device, seq_device_get, seq_cast
from bindings.classical_arrays_door import classical_input, classical_typed, classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequence_array_native_unpack import sequence_array_cast
from bindings.native_numpy_metadata_unpack import native_numpy_dtype


def seq_device_set(owner: PythonObject, binding: PythonObject, shape: List[Int], handle: PythonObject,
                   base: PythonObject, context: PythonObject) raises:
    var setters = context["sequence_device_setters"]
    _ = py_call(setters["shape"], owner, native_shape_pack(shape))
    _ = py_call(setters["h"], owner, handle)
    _ = py_call(setters["_base"], owner, base)
    _ = py_call(setters["b"], owner, binding)


def seq_device_make(binding: PythonObject, shape: List[Int], handle: PythonObject, base: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    var result = py_call(native_loaded_module("builtins").object.__new__, context["sequence_device_class"])
    seq_device_set(result, binding, shape, handle, base, context)
    return result


def seq_device_input(value: PythonObject, strict: Bool, context: PythonObject, mut buffer_pins: OptimizerInputOwners) raises -> PythonObject:
    opt_value(value, context)
    var input = context["input_context"]
    classical_connect_numpy(input)
    var source = classical_input(value, input)
    var view = TensorView()
    _ = tensor_view(source, view, buffer_pins)
    if strict and base_format(view.fmt) != "f":
        raise Error("TypeError: to_device takes float32; convert the input explicitly")
    return source if base_format(view.fmt) == "f" and view.c_contig else seq_cast(source, context)
