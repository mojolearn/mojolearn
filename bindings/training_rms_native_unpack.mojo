# SPDX-License-Identifier: Apache-2.0
# Public RMS normalization preparation and output ownership; kernels unchanged.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from bindings.py_balanced import py_call, py_type
from bindings.optimizer_public_door import tensor_view, new_array
from bindings.tensor_view import TensorView, base_format
from training.optimizer_input_plan import optimizer_contiguous_address
from bindings.optimizer_device_native_door import shape_count
from bindings.training_scalar_native_door import training_scalar_float
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_RMS_NORM_BACKWARD
from bindings.py_balanced import py_item
from bindings.training_layers_device_unpack import rms_norm_device


def rms_host_view(value: PythonObject, name: String, where: PythonObject, mut view: TensorView, mut buffer_pins: OptimizerInputOwners) raises:
    if not tensor_view(value, view, buffer_pins) or base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn." + String(py=where) + ": " + name + " must be float32")
    if shape_count(view.shape) == 0:
        raise Error("ValueError: mojolearn." + String(py=where) + ": " + name + " must be nonempty")
    if len(view.shape) == 0 and not view.is_array:
        view.shape.append(1)
        view.strides.append(4)


def training_rms_operation(operation: Int, where: PythonObject, values: PythonObject, entries: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        var backward = operation == OPTIMIZER_OPN_RMS_NORM_BACKWARD
        # the arguments and the admitted `device` item (lane vdf-linalg; read by the router)
        if len(values) != (5 if backward else 4):
            raise Error("TypeError: invalid RMS-normalization argument envelope")
        var entry = py_item(entries, where)
        var nk = optimizer_keys()
        if py_type(entry) is not py_item(context, nk[].k_device_finalizer_types)[2]:
            raise Error("TypeError: RMS normalization requires its connected compiled entry")
        var x = values[1] if backward else values[0]
        var weight = values[2] if backward else values[1]
        var gradient = values[0] if backward else PythonObject(None)
        # lane vdf-linalg2: TrainDeviceArray inputs run the device entry in
        # place (bindings/training_layers_device_unpack.mojo); one class read
        # and identity compares on the host path
        var cls = py_item(context, nk[].k_device_class)
        if py_type(x) is cls or py_type(weight) is cls or (backward and py_type(gradient) is cls):
            var eps_device = training_scalar_float(values[3] if backward else values[2], where, context)
            return rms_norm_device(backward, where, x, weight, gradient, eps_device, entries, context)
        var xv = TensorView()
        var wv = TensorView()
        var gv = TensorView()
        rms_host_view(x, "x", where, xv, buffer_pins)
        if backward:
            rms_host_view(gradient, "dy", where, gv, buffer_pins)
        rms_host_view(weight, "weight", where, wv, buffer_pins)
        if backward:
            var same = len(xv.shape) == len(gv.shape)
            if same:
                for i in range(len(xv.shape)):  # driver-loop: gradient and input rank metadata
                    same = same and xv.shape[i] == gv.shape[i]
            if not same:
                raise Error("ValueError: mojolearn.rms_norm_backward: dy and x shapes differ")
        if len(xv.shape) == 0:
            raise Error("IndexError: tuple index out of range")
        var width = xv.shape[len(xv.shape) - 1]
        var rows = shape_count(xv.shape) // width
        if not backward and (len(wv.shape) != 1 or wv.shape[0] != width):
            raise Error("ValueError: mojolearn.rms_norm_forward: weight must be (D,)")
        if backward and shape_count(wv.shape) < width:
            raise Error("ValueError: RMS backward weight buffer is shorter than D")
        # Backward historically consumes the first D contiguous weight values;
        # preserve safe wider/rank-zero buffers and the returned(D,) gradient.
        var epsilon = training_scalar_float(values[3] if backward else values[2], where, context)
        var keep = List[List[Float32]]()
        var xa = 0
        var wa = 0
        with GILReleased(Python()):
            xa = optimizer_contiguous_address(xv, keep)
            wa = optimizer_contiguous_address(wv, keep)
        var params = Python.list(rows, width, epsilon)
        if not backward:
            var result = new_array(xv.shape, "<f4")
            _ = py_call(entry, Python.list(result._addr, xa, wa), params)
            _ = keep^
            _ = x^
            _ = weight^
            return result
        var ga = 0
        with GILReleased(Python()):
            ga = optimizer_contiguous_address(gv, keep)
        var dx = new_array(xv.shape, "<f4")
        var dw = new_array([width], "<f4")
        _ = py_call(entry, Python.list(dx._addr, dw._addr, ga, xa, wa), params)
        _ = keep^
        _ = x^
        _ = weight^
        _ = gradient^
        return Python.tuple(dx, dw)
    finally:
        _ = buffer_pins^
