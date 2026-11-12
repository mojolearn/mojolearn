# SPDX-License-Identifier: Apache-2.0
# Public loss preparation, native routing and result ownership; kernels unchanged.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.optimizer_admission import opt_value, opt_numpy
from bindings.optimizer_public_door import tensor_view, new_array, refuse_kwargs
from bindings.tensor_view import TensorView, base_format
from training.optimizer_input_plan import optimizer_contiguous_address
from bindings.optimizer_device_native import device_entries, device_shape, device_allocate, shape_count
from bindings.training_embedding_native import embedding_indices, embedding_leaves, embedding_native_call, embedding_int_list
from bindings.training_scalar_native_door import training_scalar_number, training_scalar_float
from bindings.training_loss_options_unpack import loss_truth, loss_reduction
from bindings.native_numpy_metadata_pack import native_numpy_type
from bindings.sequence_dtype_native_door import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create


def loss_buffer_dtype(value: PythonObject, context: PythonObject) raises:
    if not opt_numpy(py_type(value), context):
        return
    var dtype = value.dtype
    var admitted = sequence_dtype_instance(dtype, sequence_dtype_classes(context))
    for code in ["q", "Q"]:  # driver-loop: native long-long descriptor aliases
        admitted = admitted or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
    if not admitted:
        raise Error("TypeError: cross_entropy buffers require builtin native dtype metadata")
