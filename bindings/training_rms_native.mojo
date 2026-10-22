# SPDX-License-Identifier: Apache-2.0
# Public RMS normalization preparation and output ownership; kernels unchanged.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.optimizer_public_door import TensorView, tensor_view, base_format, contiguous_addr, new_array
from bindings.optimizer_device_native import shape_count
from bindings.training_scalar_native import training_scalar_float


def rms_host_view(value: PythonObject, name: String, operation: String, mut view: TensorView) raises:
    if not tensor_view(value, view) or base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn." + operation + ": " + name + " must be float32")
    if shape_count(view.shape) == 0:
        raise Error("ValueError: mojolearn." + operation + ": " + name + " must be nonempty")
    if len(view.shape) == 0 and not view.is_array:
        view.shape.append(1)
        view.strides.append(4)


def training_rms_operation(operation: String, values: PythonObject, entries: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var backward = operation == "rms_norm_backward"
    if len(values) != (4 if backward else 3):
        raise Error("TypeError: invalid RMS-normalization argument envelope")
    var entry = entries[PythonObject(operation)]
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: RMS normalization requires its connected compiled entry")
    var x = values[1] if backward else values[0]
    var weight = values[2] if backward else values[1]
    var gradient = values[0] if backward else PythonObject(None)
    var xv = TensorView()
    var wv = TensorView()
    var gv = TensorView()
    rms_host_view(x, "x", operation, xv)
    if backward:
        rms_host_view(gradient, "dy", operation, gv)
    rms_host_view(weight, "weight", operation, wv)
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
    var epsilon = training_scalar_float(values[3] if backward else values[2], operation, context)
    var keep = List[List[Float32]]()
    var xa = contiguous_addr(x, xv, keep)
    var wa = contiguous_addr(weight, wv, keep)
    var params = Python.list(rows, width, epsilon)
    if not backward:
        var result = new_array(xv.shape, "<f4")
        _ = py_call(entry, Python.list(result._addr, xa, wa), params)
        _ = keep^
        _ = x^
        _ = weight^
        return result
    var ga = contiguous_addr(gradient, gv, keep)
    var dx = new_array(xv.shape, "<f4")
    var dw = new_array([width], "<f4")
    _ = py_call(entry, Python.list(dx._addr, dw._addr, ga, xa, wa), params)
    _ = keep^
    _ = x^
    _ = weight^
    _ = gradient^
    return Python.tuple(dx, dw)
