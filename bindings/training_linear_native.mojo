# SPDX-License-Identifier: Apache-2.0
# Public linear-layer preparation, routing and output ownership; kernels unchanged.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get
from bindings.optimizer_admission import opt_value
from bindings.optimizer_public_door import TensorView, tensor_view, contiguous_addr, base_format, new_array, py_ints
from bindings.optimizer_device_native import device_entries, device_shape, device_allocate, shape_count


def linear_dimensions(a: List[Int], weight: List[Int], grad: List[Int], backward: Bool) raises -> List[Int]:
    if len(a) < 1 or len(weight) != 2:
        raise Error("ValueError: linear layer input must have a last feature axis and weight must be a matrix")
    for i in range(len(a)):  # driver-loop: over input shape words before indexing/allocation
        if a[i] < 0:
            raise Error("ValueError: linear layer dimensions must be nonnegative")
    if weight[0] < 0 or weight[1] < 0:
        raise Error("ValueError: linear layer dimensions must be nonnegative")
    var k = a[len(a) - 1]
    if k == 0:
        raise Error("ValueError: linear layer cannot infer rows with a zero feature dimension")
    if weight[1] != k:
        var name = String("linear_backward") if backward else String("linear_forward")
        raise Error("ValueError: mojolearn." + name + ": a has K=" + String(k) + ", weight has K=" + String(weight[1]))
    var m = shape_count(a) // k
    var n = weight[0]
    if m == 0 or n == 0:
        raise Error("ValueError: linear layer inputs must be nonempty")
    _ = shape_count(weight)
    if backward:
        if len(grad) < 1:
            raise Error("ValueError: mojolearn.linear_backward: dc must be (M, N)")
        for i in range(len(grad)):  # driver-loop: over gradient shape words before counting
            if grad[i] < 0:
                raise Error("ValueError: linear gradient dimensions must be nonnegative")
        if grad[len(grad) - 1] != n or shape_count(grad) != shape_count(List[Int](m, n)):
            raise Error("ValueError: mojolearn.linear_backward: dc must be (M, N)")
    return List[Int](m, n, k)


def linear_host_view(value: PythonObject, name: String, where: String, mut view: TensorView) raises:
    if not tensor_view(value, view) or base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn." + where + ": " + name + " must be float32")
    if shape_count(view.shape) == 0:
        raise Error("ValueError: mojolearn." + where + ": " + name + " must be nonempty")
    if len(view.shape) == 0 and not view.is_array:
        # _materialize promoted scalar buffer exports to one element, while
        # retaining existing Array shapes. Preserve that supported distinction.
        view.shape.append(1)
        view.strides.append(4)


def linear_device(operation: String, a: PythonObject, weight: PythonObject, gradient: PythonObject,
                  entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var backward = operation == "linear_backward"
    var cls = context["device_class"]
    if py_type(a) is not cls or py_type(weight) is not cls or (backward and py_type(gradient) is not cls):
        raise Error("TypeError: mojolearn." + operation + ": device and host arrays are mixed; pass every array through train_to_device, or none")
    var a_owner = a
    var weight_owner = weight
    var gradient_owner = gradient
    var native = device_entries(entries["module"], context)
    var af = py_fields(a)
    var wf = py_fields(weight)
    var gf = py_fields(gradient) if backward else Python.dict()
    if af["_b"] is not entries["module"] or wf["_b"] is not entries["module"]:
        raise Error("ValueError: linear device inputs must belong to the selected native binding")
    if backward and gf["_b"] is not entries["module"]:
        raise Error("ValueError: linear device gradient must belong to the selected native binding")
    var ashape = device_shape(af["shape"], context)
    var wshape = device_shape(wf["shape"], context)
    var gshape = device_shape(gf["shape"], context) if backward else List[Int]()
    var dims = linear_dimensions(ashape, wshape, gshape, backward)
    if shape_count(ashape) != Int(py=af["size"]) or shape_count(wshape) != Int(py=wf["size"]):
        raise Error("ValueError: linear device shape does not match its stored element count")
    if backward and shape_count(gshape) != Int(py=gf["size"]):
        raise Error("ValueError: linear device gradient shape does not match its stored element count")
    var entry = py_get(native, PythonObject(operation + "_dev"))
    if entry is None:
        raise Error("RuntimeError: this build has no native device linear layer entry")
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: linear layer requires its connected compiled entry")
    if not backward:
        var shape = ashape.copy()
        shape[len(shape) - 1] = dims[1]
        var result = device_allocate(native, shape, context)
        _ = py_call(entry, Python.list(py_fields(result)["h"], af["h"], wf["h"]), py_ints(dims))
        _ = a_owner^
        _ = weight_owner^
        _ = gradient_owner^
        return result
    var da = device_allocate(native, ashape, context)
    var dw = device_allocate(native, wshape, context)
    _ = py_call(entry, Python.list(py_fields(da)["h"], py_fields(dw)["h"], gf["h"], af["h"], wf["h"]), py_ints(dims))
    var result = Python.tuple(da, dw)
    _ = a_owner^
    _ = weight_owner^
    _ = gradient_owner^
    return result


def training_linear_operation(operation: String, values: PythonObject, entries: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    var backward = operation == "linear_backward"
    if len(values) != (3 if backward else 2):
        raise Error("TypeError: invalid linear-layer argument envelope")
    opt_value(values, context)
    var entry = entries[PythonObject(operation)]
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: linear layer requires its connected compiled entry")
    var a = values[1] if backward else values[0]
    var weight = values[2] if backward else values[1]
    var gradient = values[0] if backward else PythonObject(None)
    var cls = context["device_class"]
    if py_type(a) is cls or py_type(weight) is cls or (backward and py_type(gradient) is cls):
        return linear_device(operation, a, weight, gradient, entries, context)
    var av = TensorView()
    var wv = TensorView()
    var gv = TensorView()
    linear_host_view(a, "a", operation, av)
    linear_host_view(weight, "weight", operation, wv)
    if backward:
        linear_host_view(gradient, "dc", operation, gv)
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
        kernel_weight_shape = List[Int](wv.shape[0], av.shape[len(av.shape) - 1])
        if shape_count(wv.shape) < shape_count(kernel_weight_shape):
            raise Error("ValueError: linear backward weight buffer is shorter than N*K")
    var dims = linear_dimensions(av.shape, kernel_weight_shape, gv.shape, backward)
    var keep = List[List[Float32]]()
    var aa = contiguous_addr(a, av, keep)
    var wa = contiguous_addr(weight, wv, keep)
    if not backward:
        var shape = av.shape.copy()
        shape[len(shape) - 1] = dims[1]
        var result = new_array(shape, "<f4")
        _ = py_call(entry, Python.list(result._addr, aa, wa), py_ints(dims))
        _ = keep^
        _ = a^
        _ = weight^
        return result
    var ga = contiguous_addr(gradient, gv, keep)
    var da = new_array(av.shape, "<f4")
    var dw = new_array(kernel_weight_shape, "<f4")
    _ = py_call(entry, Python.list(da._addr, dw._addr, ga, aa, wa), py_ints(dims))
    _ = keep^
    _ = a^
    _ = weight^
    _ = gradient^
    return Python.tuple(da, dw)
