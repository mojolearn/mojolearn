# SPDX-License-Identifier: Apache-2.0
# Whole CNN DeviceTensor public operations. Slot identities are import-connected;
# shapes and buffer owners are admitted before conversion or native execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.neural_block_admission_door import neural_plain_value
from bindings.sequential_admission_door import sequential_admit_value, sequential_integer
from bindings.sequence_array_native_unpack import sequence_array_cast
from bindings.xcnn_common_door import (
    xcnn_builtins, xcnn_tensor, xcnn_tensor_slot, xcnn_tensor_set, xcnn_tensor_admit,
    xcnn_shape, xcnn_size, xcnn_tuple, xcnn_new, xcnn_out_new, xcnn_wrap,
    xcnn_host_of, xcnn_f32, xcnn_entries, xcnn_module_entries, xcnn_device_io_off,
    xcnn_is_t, xcnn_reshape_t, xcnn_tensor_layout, _xcnn_set_raw, _xcnn_slot_raw, xcnn_ref,
)
from std.python._cpython import GILReleased
from x_cnn.tensor_run import (
    xcnn_tensor_run_check_dim, xcnn_tensor_run_check_handle, xcnn_tensor_run_handle_live,
    xcnn_tensor_run_reshape_dims, xcnn_tensor_run_free,
)


def _shape(value: PythonObject, context: PythonObject, infer: Bool = False) raises -> List[Int]:
    var b = xcnn_builtins()
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        raise Error("TypeError: CNN tensor shape requires a native list or tuple")
    var out = List[Int]()
    for i in range(len(value)):  # driver-loop: passive shape words before any conversion
        if not sequential_integer(value[i], context):
            raise Error("TypeError: CNN tensor dimensions require native integers")
        var size = Int(py=value[i])
        xcnn_tensor_run_check_dim(size, infer)
        out.append(size)
    return out^
