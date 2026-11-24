# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of `training.clip_grad_norm_` and `training.accumulate_grads` on `TrainDeviceArray` gradients
(lane vdf-linalg3, 2026-11-24), the convention of `linear_forward` and the RMS / embedding layers
(bindings/training_layers_device_unpack.mojo): every gradient a TrainDeviceArray of the selected binding (mixed host
and device gradients refused in words), the device entry (`clip_dev` / `accumulate_dev`,
training/dev_tensors_layers.mojo) called on their handles. clip_grad_norm_ scales the device arrays in place and
returns the pre-clip total norm as a float; accumulate_grads returns new TrainDeviceArrays in the structure of one
entry. Nothing is downloaded or re-uploaded but the two clip info words. The scalar refusals are the host door's own
(bindings/optimizer_public_door.mojo `door_clip_options`, `door_clip_max_norm`, `door_accumulate_parts`,
`door_accumulate_tokens`). A TrainDeviceArray with device="cpu" is refused before this by the router
(bindings/training_function_device_unpack.mojo).
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_item
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_device_native_door import device_entries, device_allocate, shape_count
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.optimizer_container_boundary_pack import optimizer_sequence_items, optimizer_pack_list
from bindings.optimizer_public_door import (
    door_clip_options, door_clip_max_norm, door_accumulate_parts, door_accumulate_tokens,
)
from bindings.training_layers_device_unpack import _dev_fields, _dev_entry


def _is_sequence(value: PythonObject, b: PythonObject) raises -> Bool:
    var cls = py_type(value)
    return cls is b.list or cls is b.tuple


def grads_device_input(value: PythonObject, nested: Bool, context: PythonObject) raises -> Bool:
    """True when the gradients `value` are TrainDeviceArrays: the value itself, or the first item of a list or tuple
    (`nested`: of the first entry, for accumulate_grads' microbatch list). Identity compares only."""
    var nk = optimizer_keys()
    var cls = py_item(context, nk[].k_device_class)
    var b = native_loaded_module("builtins")
    var first = value
    var depth = 2 if nested else 1
    for _ in range(depth):  # driver-loop: over at most two nesting levels of the argument (not data)
        if py_type(first) is cls:
            return True
        if not _is_sequence(first, b) or len(first) == 0:
            return False
        first = first[0]
    return py_type(first) is cls


def _device_items(value: PythonObject, name: String, where: PythonObject, entries: PythonObject,
                  context: PythonObject, mut handles: List[Int], mut shapes: List[List[Int]]) raises -> Bool:
    """The device arrays of one gradient argument (one array, or a list / tuple of them): their handles and shapes
    appended; True for a single array."""
    var nk = optimizer_keys()
    var b = native_loaded_module("builtins")
    if py_type(value) is py_item(context, nk[].k_device_class):
        var shape = List[Int]()
        var fields = _dev_fields(value, name, where, entries, context, shape)
        handles.append(Int(py=py_item(fields, nk[].k_h)))
        shapes.append(shape^)
        return True
    if not _is_sequence(value, b) or len(value) == 0:
        raise Error("TypeError: mojolearn." + String(py=where) + ": device and host arrays are mixed; pass every "
                    + "float32 array through train_to_device, or none")
    for i in range(len(value)):  # driver-loop: over the caller's gradient tensors (handles, not data)
        var shape = List[Int]()
        var fields = _dev_fields(value[i], name + "[" + String(i) + "]", where, entries, context, shape)
        handles.append(Int(py=py_item(fields, nk[].k_h)))
        shapes.append(shape^)
    return False


def clip_device(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`clip_grad_norm_(grads, max_norm, norm_type, error_if_nonfinite, **kwargs)` on TrainDeviceArrays: scaled in
    place on the device; returns the pre-clip total norm as a float."""
    door_clip_options(values[2], values[3], values[4])
    var nk = optimizer_keys()
    var where = PythonObject("clip_grad_norm_")
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var handles = List[Int]()
    var shapes = List[List[Int]]()
    _ = _device_items(values[0], "grads", where, entries, context, handles, shapes)
    var mn = door_clip_max_norm(values[1])
    var entry = _dev_entry(native, nk[].k_clip_dev, context)
    var hs = Python.list()
    var ns = Python.list()
    for k in range(len(handles)):  # driver-loop: over gradient tensors (handles and sizes, not data)
        _ = py_call(hs.append, PythonObject(handles[k]))
        _ = py_call(ns.append, PythonObject(shape_count(shapes[k])))
    var info = List[Float32](length=2, fill=Float32(0.0))
    var info_address = Int(info.unsafe_ptr())
    _ = py_call(entry, hs, ns, Python.list(info_address, mn))
    _ = values
    var total = Float64(info[0])
    _ = info^
    return PythonObject(total)


def accumulate_device(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`accumulate_grads(microbatch_grads, tokens)` on TrainDeviceArrays: the combined gradient as new
    TrainDeviceArrays in the structure of one entry."""
    var parts = door_accumulate_parts(values[0], values[1])
    var nk = optimizer_keys()
    var where = PythonObject("accumulate_grads")
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var a = len(parts)
    var handles = List[Int]()
    var shapes = List[List[Int]]()
    var single = _device_items(parts[0], "microbatch_grads[0]", where, entries, context, handles, shapes)
    var j = len(handles)
    for i in range(1, a):  # driver-loop: over microbatches (handles and shapes, not data)
        var more = List[List[Int]]()
        var name = "microbatch_grads[" + String(i) + "]"
        _ = _device_items(parts[i], name, where, entries, context, handles, more)
        var same = len(more) == j
        if same:
            for k in range(j):  # driver-loop: over gradient tensors (shape words)
                var r = len(more[k]) == len(shapes[k])
                if r:
                    for d in range(len(more[k])):  # driver-loop: over one tensor's rank (shape words)
                        r = r and more[k][d] == shapes[k][d]
                same = same and r
        if not same:
            raise Error("ValueError: mojolearn.accumulate_grads: " + name
                        + " does not have the shapes of microbatch_grads[0]")
    var t_tokens = door_accumulate_tokens(values[1])
    var entry = _dev_entry(native, nk[].k_accumulate_dev, context)
    var results = List[PythonObject]()
    var outs = Python.list()
    var ins = Python.list()
    var ns = Python.list()
    for k in range(j):  # driver-loop: over gradient tensors: one result array each
        var r = device_allocate(native, shapes[k], context)
        _ = py_call(outs.append, py_item(py_fields(r), nk[].k_h))
        _ = py_call(ns.append, PythonObject(shape_count(shapes[k])))
        results.append(r)
    for i in range(len(handles)):  # driver-loop: over the A x J input handles (not data)
        _ = py_call(ins.append, PythonObject(handles[i]))
    _ = py_call(entry, outs, ins, ns, Python.list(a, t_tokens))
    _ = parts^
    if single:
        return results[0]
    return optimizer_pack_list(results)


def mlp_sessions_device(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`mlp_inference_sessions(inputs, weight1, bias1, weight2, bias2)` on TrainDeviceArrays (lane vdf-linalg3,
    2026-11-24): every session input and weight read in place by the device entry `mlp_sessions_dev`
    (training/dev_tensors_layers.mojo); the logits, (sum of rows, out_width), a new TrainDeviceArray. The host door's
    shape refusals in its words (bindings/training_extra_native_door.mojo `training_mlp_sessions`)."""
    var nk = optimizer_keys()
    var where = PythonObject("mlp_inference_sessions")
    var native = device_entries(py_item(entries, nk[].k_module), context)
    var s1 = List[Int]()
    var t1 = List[Int]()
    var s2 = List[Int]()
    var t2 = List[Int]()
    var w1 = _dev_fields(values[1], "weight1", where, entries, context, s1)
    var b1 = _dev_fields(values[2], "bias1", where, entries, context, t1)
    var w2 = _dev_fields(values[3], "weight2", where, entries, context, s2)
    var b2 = _dev_fields(values[4], "bias2", where, entries, context, t2)
    if len(s1) != 2 or len(s2) != 2:
        raise Error("ValueError: MLP session weights must be matrices")
    var hidden = s1[0]
    var width = s1[1]
    var out_width = s2[0]
    if s2[1] != hidden or len(t1) != 1 or t1[0] != hidden or len(t2) != 1 or t2[0] != out_width:
        raise Error("ValueError: MLP session weight/bias shapes do not agree")
    var handles = List[Int]()
    var shapes = List[List[Int]]()
    _ = _device_items(values[0], "session input", where, entries, context, handles, shapes)
    var total = 0
    for i in range(len(handles)):  # driver-loop: over the caller's sessions (shape words, not data)
        if len(shapes[i]) != 2 or shapes[i][1] != width:
            raise Error("ValueError: MLP sessions require at least one (rows, width) input")
        total += shapes[i][0]
    var output = device_allocate(native, [total, out_width], context)
    var hs = Python.list(py_item(py_fields(output), nk[].k_h), py_item(w1, nk[].k_h), py_item(b1, nk[].k_h),
                         py_item(w2, nk[].k_h), py_item(b2, nk[].k_h))
    var rs = Python.list()
    for i in range(len(handles)):  # driver-loop: over the caller's sessions (handles, not data)
        _ = py_call(hs.append, PythonObject(handles[i]))
        _ = py_call(rs.append, PythonObject(shapes[i][0]))
    var entry = _dev_entry(native, nk[].k_mlp_sessions_dev, context)
    var wrote = py_call(entry, hs, rs, Python.list(width, hidden, out_width))
    if Int(py=wrote) != total * out_width:
        raise Error("RuntimeError: MLP sessions returned an invalid output count")
    _ = values
    return output
