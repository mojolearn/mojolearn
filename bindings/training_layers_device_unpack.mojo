# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of `training.rms_norm_forward/backward` and `embedding_forward/backward` on `TrainDeviceArray`
inputs (lane vdf-linalg2, 2026-11-24): the convention of `linear_forward` (bindings/training_linear_native_unpack.mojo
`linear_device`): every float32 array argument a TrainDeviceArray of the selected binding (mixed host and device arrays
refused in words), the device entry (`*_dev`, training/dev_tensors_layers.mojo) called on their handles, and the
results returned as new TrainDeviceArrays. Nothing is downloaded or re-uploaded; the embedding ids (int32, host) are
uploaded once by view inside the entry. A TrainDeviceArray with device="cpu" is refused before this by the router
(bindings/training_function_device_unpack.mojo).
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_item
from bindings.optimizer_device_native_door import device_entries, device_shape, device_allocate, shape_count
from bindings.optimizer_keys_unpack import optimizer_keys


def training_device_array(value: PythonObject, context: PythonObject) -> Bool:
    """True when `value` is exactly a TrainDeviceArray (one identity compare)."""
    try:
        var nk = optimizer_keys()
        return py_type(value) is py_item(context, nk[].k_device_class)
    except:
        return False


def _dev_fields(value: PythonObject, name: String, where: PythonObject, entries: PythonObject,
                context: PythonObject, mut shape: List[Int]) raises -> PythonObject:
    """The fields of a TrainDeviceArray argument of the selected binding, its
    shape checked against its stored element count."""
    var nk = optimizer_keys()
    if py_type(value) is not py_item(context, nk[].k_device_class):
        raise Error("TypeError: mojolearn." + String(py=where) + ": device and host arrays are mixed; pass every "
                    + "float32 array through train_to_device, or none")
    var fields = py_fields(value)
    if py_item(fields, nk[].k__b) is not py_item(entries, nk[].k_module):
        raise Error("ValueError: mojolearn." + String(py=where) + ": " + name
                    + " must belong to the selected native binding")
    shape = device_shape(py_item(fields, nk[].k_shape), context)
    if shape_count(shape) != Int(py=py_item(fields, nk[].k_size)):
        raise Error("ValueError: mojolearn." + String(py=where) + ": " + name
                    + " shape does not match its stored element count")
    if shape_count(shape) == 0:
        raise Error("ValueError: mojolearn." + String(py=where) + ": " + name + " must be nonempty")
    return fields


def _dev_entry(native: PythonObject, key: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = optimizer_keys()
    var entry = py_get(native, key)
    if entry is None:
        raise Error("RuntimeError: this build has no native device entry for this training layer")
    if py_type(entry) is not py_item(context, nk[].k_device_finalizer_types)[2]:
        raise Error("TypeError: training layer requires its connected compiled entry")
    return entry


def rms_norm_device(backward: Bool, where: PythonObject, x: PythonObject, weight: PythonObject,
                    gradient: PythonObject, epsilon: PythonObject, entries: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`rms_norm_forward` / `rms_norm_backward` on TrainDeviceArrays: y, or
    (dx, dw), as new TrainDeviceArrays."""
    var nk = optimizer_keys()
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var xs = List[Int]()
    var ws = List[Int]()
    var gs = List[Int]()
    var xf = _dev_fields(x, "x", where, entries, context, xs)
    var wf = _dev_fields(weight, "weight", where, entries, context, ws)
    var gf = _dev_fields(gradient, "dy", where, entries, context, gs) if backward else Python.dict()
    var width = xs[len(xs) - 1]
    var rows = shape_count(xs) // width
    if backward:
        var same = len(xs) == len(gs)
        if same:
            for i in range(len(xs)):  # driver-loop: gradient and input rank metadata
                same = same and xs[i] == gs[i]
        if not same:
            raise Error("ValueError: mojolearn.rms_norm_backward: dy and x shapes differ")
        if shape_count(ws) < width:
            raise Error("ValueError: RMS backward weight buffer is shorter than D")
    elif len(ws) != 1 or ws[0] != width:
        raise Error("ValueError: mojolearn.rms_norm_forward: weight must be (D,)")
    var params = Python.list(rows, width, epsilon)
    if not backward:
        var entry = _dev_entry(native, nk[].k_rms_norm_forward_dev, context)
        var y = device_allocate(native, xs, context)
        _ = py_call(entry, Python.list(py_item(py_fields(y), nk[].k_h), py_item(xf, nk[].k_h),
                                       py_item(wf, nk[].k_h)), params)
        _ = x
        _ = weight
        return y
    var entry = _dev_entry(native, nk[].k_rms_norm_backward_dev, context)
    var dx = device_allocate(native, xs, context)
    var dw = device_allocate(native, [width], context)
    _ = py_call(entry, Python.list(py_item(py_fields(dx), nk[].k_h), py_item(py_fields(dw), nk[].k_h),
                                   py_item(gf, nk[].k_h), py_item(xf, nk[].k_h), py_item(wf, nk[].k_h)), params)
    _ = x
    _ = weight
    _ = gradient
    return Python.tuple(dx, dw)


def embedding_device(backward: Bool, where: PythonObject, input: PythonObject, ids_address: Int, count: Int,
                     vocab_backward: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`embedding_forward(weight, ids)` / `embedding_backward(dy, ids,
    vocab)` with the float32 array a TrainDeviceArray: the result a new
    TrainDeviceArray. `ids_address`: the admitted int32 ids (count words)."""
    var nk = optimizer_keys()
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var shape = List[Int]()
    var fields = _dev_fields(input, "dy" if backward else "weight", where, entries, context, shape)
    if len(shape) != 2:
        raise Error("ValueError: mojolearn." + String(py=where) + ": "
                    + ("dy must be (N, D)" if backward else "weight must be (V, D)"))
    if backward and count != shape[0]:
        raise Error("ValueError: mojolearn.embedding_backward: ids and dy row counts differ")
    var width = shape[1]
    var vocab = vocab_backward if backward else shape[0]
    var result = device_allocate(native, [vocab if backward else count, width], context)
    var entry = _dev_entry(native, nk[].k_embedding_backward_dev if backward else nk[].k_embedding_forward_dev,
                           context)
    _ = py_call(entry, Python.list(py_item(py_fields(result), nk[].k_h), py_item(fields, nk[].k_h)),
                Python.list(ids_address, count, vocab, width))
    _ = input
    return result


def chunked_lm_head_device(hidden: PythonObject, weight: PythonObject, targets_address: Int, count: Int,
                           with_grad: Bool, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`chunked_lm_head_loss(hidden, weight, targets, return_grad)` with hidden and weight TrainDeviceArrays of the
    selected binding (lane vdf-linalg3, 2026-11-24): read in place by the device entry `chunked_lm_head_dev`
    (training/dev_tensors_layers.mojo); the targets are the admitted int32 host words (count of them), uploaded once
    inside the entry. Returns the loss as a Python float, or (loss, d_hidden, d_weight) with the gradients new
    TrainDeviceArrays: nothing but the loss scalar comes down."""
    var nk = optimizer_keys()
    var where = PythonObject("chunked_lm_head_loss")
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var hs = List[Int]()
    var ws = List[Int]()
    var hf = _dev_fields(hidden, "hidden", where, entries, context, hs)
    var wf = _dev_fields(weight, "weight", where, entries, context, ws)
    if len(hs) != 2 or len(ws) != 2:
        raise Error("ValueError: mojolearn.training.chunked_lm_head_loss: hidden must be (rows, width) and weight "
                    + "(vocab, width)")
    var rows = hs[0]
    var width = hs[1]
    var vocab = ws[0]
    if width != ws[1]:
        raise Error("ValueError: mojolearn.training.chunked_lm_head_loss: hidden has width " + String(width)
                    + ", weight has width " + String(ws[1]))
    if vocab < 2:
        raise Error("ValueError: mojolearn.training.chunked_lm_head_loss: vocab must be at least 2, got "
                    + String(vocab))
    if count != rows:
        raise Error("ValueError: mojolearn.training.chunked_lm_head_loss: " + String(count) + " targets for "
                    + String(rows) + " rows")
    var entry = _dev_entry(native, nk[].k_chunked_lm_head_dev, context)
    var loss = List[Float32](length=1, fill=Float32(0.0))
    var loss_address = Int(loss.unsafe_ptr())
    var params = Python.list(loss_address, targets_address, rows, vocab, width, 1 if with_grad else 0)
    if not with_grad:
        _ = py_call(entry, Python.list(-1, -1, py_item(hf, nk[].k_h), py_item(wf, nk[].k_h)), params)
        _ = hidden
        _ = weight
        var value = Float64(loss[0])
        _ = loss^
        return PythonObject(value)
    var dh = device_allocate(native, hs, context)
    var dw = device_allocate(native, ws, context)
    _ = py_call(entry, Python.list(py_item(py_fields(dh), nk[].k_h), py_item(py_fields(dw), nk[].k_h),
                                   py_item(hf, nk[].k_h), py_item(wf, nk[].k_h)), params)
    _ = hidden
    _ = weight
    var value = Float64(loss[0])
    _ = loss^
    return Python.tuple(PythonObject(value), dh, dw)
