# SPDX-License-Identifier: Apache-2.0
# Public linear-layer preparation, routing and output ownership; kernels unchanged.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from bindings.py_balanced import py_call, py_type, py_fields, py_get
from bindings.optimizer_admission_door import opt_value
from bindings.optimizer_public_door import tensor_view, new_array
from bindings.tensor_view import TensorView, base_format
from training.optimizer_input_plan import optimizer_contiguous_address
from providers.python.array_layout_boundary_pack import native_shape_pack
from bindings.optimizer_device_native_door import device_entries, device_shape, device_allocate, shape_count
from training.training_linear_run import linear_dimensions
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_LINEAR_BACKWARD
from bindings.py_balanced import py_item


def linear_host_view(value: PythonObject, name: String, where: PythonObject, mut view: TensorView, mut buffer_pins: OptimizerInputOwners) raises:
    if not tensor_view(value, view, buffer_pins) or base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn." + String(py=where) + ": " + name + " must be float32")
    if shape_count(view.shape) == 0:
        raise Error("ValueError: mojolearn." + String(py=where) + ": " + name + " must be nonempty")
    if len(view.shape) == 0 and not view.is_array:
        # _materialize promoted scalar buffer exports to one element, while
        # retaining existing Array shapes. Preserve that supported distinction.
        view.shape.append(1)
        view.strides.append(4)


def linear_device(operation: Int, where: PythonObject, a: PythonObject, weight: PythonObject, gradient: PythonObject,
                  entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var backward = operation == OPTIMIZER_OPN_LINEAR_BACKWARD
    var nk = optimizer_keys()
    var cls = py_item(context, nk[].k_device_class)
    if py_type(a) is not cls or py_type(weight) is not cls or (backward and py_type(gradient) is not cls):
        raise Error("TypeError: mojolearn." + String(py=where) + ": device and host arrays are mixed; pass every array through train_to_device, or none")
    var a_owner = a
    var weight_owner = weight
    var gradient_owner = gradient
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var af = py_fields(a)
    var wf = py_fields(weight)
    var gf = py_fields(gradient) if backward else Python.dict()
    if py_item(af, nk[].k__b) is not py_item(entries, nk[].k_module) or py_item(wf, nk[].k__b) is not py_item(entries, nk[].k_module):
        raise Error("ValueError: linear device inputs must belong to the selected native binding")
    if backward and py_item(gf, nk[].k__b) is not py_item(entries, nk[].k_module):
        raise Error("ValueError: linear device gradient must belong to the selected native binding")
    var ashape = device_shape(py_item(af, nk[].k_shape), context)
    var wshape = device_shape(py_item(wf, nk[].k_shape), context)
    var gshape = device_shape(py_item(gf, nk[].k_shape), context) if backward else List[Int]()
    var dims = linear_dimensions(ashape, wshape, gshape, backward)
    if shape_count(ashape) != Int(py=py_item(af, nk[].k_size)) or shape_count(wshape) != Int(py=py_item(wf, nk[].k_size)):
        raise Error("ValueError: linear device shape does not match its stored element count")
    if backward and shape_count(gshape) != Int(py=py_item(gf, nk[].k_size)):
        raise Error("ValueError: linear device gradient shape does not match its stored element count")
    var entry = py_get(native, nk[].k_linear_backward_dev if backward else nk[].k_linear_forward_dev)
    if entry is None:
        raise Error("RuntimeError: this build has no native device linear layer entry")
    if py_type(entry) is not py_item(context, nk[].k_device_finalizer_types)[2]:
        raise Error("TypeError: linear layer requires its connected compiled entry")
    if not backward:
        var shape = ashape.copy()
        shape[len(shape) - 1] = dims[1]
        var result = device_allocate(native, shape, context)
        _ = py_call(entry, Python.list(py_item(py_fields(result), nk[].k_h), py_item(af, nk[].k_h), py_item(wf, nk[].k_h)), native_shape_pack(dims))
        _ = a_owner^
        _ = weight_owner^
        _ = gradient_owner^
        return result
    var da = device_allocate(native, ashape, context)
    var dw = device_allocate(native, wshape, context)
    _ = py_call(entry, Python.list(py_item(py_fields(da), nk[].k_h), py_item(py_fields(dw), nk[].k_h), py_item(gf, nk[].k_h), py_item(af, nk[].k_h), py_item(wf, nk[].k_h)), native_shape_pack(dims))
    var result = Python.tuple(da, dw)
    _ = a_owner^
    _ = weight_owner^
    _ = gradient_owner^
    return result


def training_linear_operation(operation: Int, where: PythonObject, values: PythonObject, entries: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        var backward = operation == OPTIMIZER_OPN_LINEAR_BACKWARD
        # the arguments and the admitted `device` item (lane vdf-linalg; read by the router)
        if len(values) != (4 if backward else 3):
            raise Error("TypeError: invalid linear-layer argument envelope")
        opt_value(values, context)
        var entry = py_item(entries, where)
        var nk = optimizer_keys()
        if py_type(entry) is not py_item(context, nk[].k_device_finalizer_types)[2]:
            raise Error("TypeError: linear layer requires its connected compiled entry")
        var a = values[1] if backward else values[0]
        var weight = values[2] if backward else values[1]
        var gradient = values[0] if backward else PythonObject(None)
        var cls = py_item(context, nk[].k_device_class)
        if py_type(a) is cls or py_type(weight) is cls or (backward and py_type(gradient) is cls):
            return linear_device(operation, where, a, weight, gradient, entries, context)
        var av = TensorView()
        var wv = TensorView()
        var gv = TensorView()
        linear_host_view(a, "a", where, av, buffer_pins)
        linear_host_view(weight, "weight", where, wv, buffer_pins)
        if backward:
            linear_host_view(gradient, "dc", where, gv, buffer_pins)
        if len(av.shape) == 0:
            # Input indexing preceded weight unpacking in both host wrappers.
            raise Error("IndexError: tuple index out of range")
        var kernel_weight_shape = wv.shape.copy()
        if backward:
            if len(wv.shape) == 0 or len(gv.shape) == 0:
                raise Error("IndexError: tuple index out of range")
            # The old host backward wrapper read N from weight.shape[0] and
            # consumed its first N*K contiguous values; it did not unpack a
            # matrix shape. Preserve safe rank-one/oversized native buffers,
            # including promoted NumPy scalars when K=1. Refuse a short buffer
            # before the unchanged raw-pointer kernel can read beyond it.
            kernel_weight_shape = [wv.shape[0], av.shape[len(av.shape) - 1]]
            if shape_count(wv.shape) < shape_count(kernel_weight_shape):
                raise Error("ValueError: linear backward weight buffer is shorter than N*K")
        var dims = linear_dimensions(av.shape, kernel_weight_shape, gv.shape, backward)
        var keep = List[List[Float32]]()
        var aa = 0
        var wa = 0
        with GILReleased(Python()):
            aa = optimizer_contiguous_address(av, keep)
            wa = optimizer_contiguous_address(wv, keep)
        if not backward:
            var shape = av.shape.copy()
            shape[len(shape) - 1] = dims[1]
            var result = new_array(shape, "<f4")
            _ = py_call(entry, Python.list(result._addr, aa, wa), native_shape_pack(dims))
            _ = keep^
            _ = a^
            _ = weight^
            return result
        var ga = 0
        with GILReleased(Python()):
            ga = optimizer_contiguous_address(gv, keep)
        var da = new_array(av.shape, "<f4")
        var dw = new_array(kernel_weight_shape, "<f4")
        _ = py_call(entry, Python.list(da._addr, dw._addr, ga, aa, wa), native_shape_pack(dims))
        _ = keep^
        _ = a^
        _ = weight^
        _ = gradient^
        return Python.tuple(da, dw)
    finally:
        _ = buffer_pins^
