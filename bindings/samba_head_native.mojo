# SPDX-License-Identifier: Apache-2.0
"""`training.samba_head_loss` as one native operation.

The fused Samba head (`linear_forward` -> sum cross-entropy -> `linear_backward`
in one binding call, logits kept on the device) used to be prepared in
Python: buffer probes, float32/int32 conversion and the shape arithmetic.
This operation does that admission, sizing and output allocation natively and
calls the binding's `samba_head_loss` entry once, as `chunked_lm_head_loss`
does for the chunked head. Kernels and their operand order are unchanged.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.optimizer_admission import opt_value
from bindings.optimizer_public_door import TensorView, tensor_view, new_array, shape_text
from bindings.optimizer_device_native import shape_count
from bindings.training_embedding_native import embedding_indices
from bindings.training_extra_native import _c32


def training_samba_head_loss(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """values: (a, weight, targets, num_items). `(loss, da, dweight)`, or None
    when the selected binding has no fused `samba_head_loss` entry (the
    caller then runs the three separate calls)."""
    if len(values) != 4:
        raise Error("TypeError: invalid samba_head_loss argument envelope")
    opt_value(values, context)
    var bi = native_loaded_module("builtins")
    # The unfused entry is required before the optional fused one is resolved.
    _ = entries["linear_forward"]
    var entry = py_call(bi.getattr, entries["module"], PythonObject("samba_head_loss"), PythonObject(None))
    if entry is None:
        return PythonObject(None)
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: samba_head_loss requires its connected compiled entry")
    var site = String("samba_head_loss")
    var keep = List[List[Float32]]()
    var av = TensorView()
    var wv = TensorView()
    var aa = _c32(values[0], "a", site, av, keep)
    var wa = _c32(values[1], "weight", site, wv, keep)
    # a is read as (M, K) over its last axis; weight is (N, K).
    if len(av.shape) == 0:
        raise Error("IndexError: tuple index out of range")
    var k = av.shape[len(av.shape) - 1]
    var m = shape_count(av.shape) // k
    if len(wv.shape) < 2:
        raise Error("ValueError: not enough values to unpack (expected 2, got " + String(len(wv.shape)) + ")")
    if len(wv.shape) > 2:
        raise Error("ValueError: too many values to unpack (expected 2)")
    var n = wv.shape[0]
    if k != wv.shape[1]:
        raise Error("ValueError: mojolearn.linear_forward: a has K=" + String(k) + ", weight has K=" + String(wv.shape[1]))
    var tv = TensorView()
    if tensor_view(values[2], tv) and len(tv.shape) != 1:
        raise Error("ValueError: mojolearn: targets must be 1-D, got " + String(len(tv.shape)) + "-D shape "
                    + shape_text(tv.shape))
    var targets = embedding_indices(values[2], context)
    var count = Int(py=targets[2])
    if count != m:
        raise Error("ValueError: mojolearn.cross_entropy: logits has " + String(m) + " rows and targets has "
                    + String(count) + " (python/mojolearn/_training_impl.py)")
    var items = 0
    if values[3] is not None:
        items = Int(py=py_call(bi.int, values[3]))
    var loss = new_array([1], "<f4")
    var row_loss = new_array([m], "<f4")
    var da = new_array([m, k], "<f4")
    var dw = new_array([n, k], "<f4")
    # ignore_index -100 (the cross_entropy default), sum reduction, the
    # num_items divisor (0: this call's own target count), no smoothing.
    var params = Python.list(PythonObject(m), PythonObject(n), PythonObject(k), PythonObject(-100), PythonObject(1),
                             PythonObject(items), PythonObject(Float64(0.0)))
    _ = py_call(entry, Python.list(loss._addr, row_loss._addr, da._addr, dw._addr,
                                   PythonObject(aa), PythonObject(wa), targets[1]), params)
    var value = Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0])
    _ = loss  # keep the Python object alive through the native read above
    _ = row_loss^
    _ = keep^
    _ = targets^
    return Python.tuple(PythonObject(value), da, dw)
