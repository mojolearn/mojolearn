# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the native owner of the CNN layers
(python/mojolearn/_expansion_cnn.py, bindings/xcnn_native.mojo): the
binding route and its numerical-profile check, the lane switches the
binding reports (`x_cnn_idn_flags`, `x_cnn_idn2_flags`), the resident
`DeviceTensor` and the layer's resident-array table (`_Dev`), input
admission and the host outputs.

The entries are connected once at import (`_native_xcnn_bridge.py`); the
layer methods are one call each into `xcnn_native_call`, which runs the same
compiled entries with the same argument words and residency bits as the
Python bodies it replaces.
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_inputs import sequential_input
from bindings.classical_attributes import classical_connect_numpy
from bindings.xseq_native_common import xseq_f32

# lane fam-neural switch bits (`x_cnn_idn_flags`)
comptime XCNN_F_ADAPT = 1
comptime XCNN_F_GRAPH = 2
comptime XCNN_F_PAD = 4
comptime XCNN_F_GROUP = 8
comptime XCNN_F_LAYER_IO = 16
comptime XCNN_F_PIN = 32
comptime XCNN_F_RESIDENT = 128
# lane fam2-neural switch bits (`x_cnn_idn2_flags`)
comptime XCNN_F2_ALL_OFF = 16


def xcnn_builtins() raises -> PythonObject:
    return Python.import_module("builtins")


def xcnn_numpy(context: PythonObject) raises:
    """NumPy loaded and its native types connected: the layers return
    ndarrays, as `_np()` (`require_numpy`) made them."""
    _ = Python.import_module("numpy")
    classical_connect_numpy(context)


def xcnn_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Layer._binding()`: the connected `_mojolearn_x_cnn` entries of the
    layer's mode (or the process default), with the identical-mode
    numerical-profile check; records `numeric_mode_`."""
    var b = xcnn_builtins()
    var mode = py_get(fields, "numeric_mode")
    if mode is None or not Bool(py=py_call(b.bool, mode)):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var key = py_call(py_call(b.getattr, py_call(py_call(b.getattr, mode, "strip")), "lower"))
    if String(py=key) != "identical" and String(py=key) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["xcnn_bindings"][key]
    if Bool(py=py_call(b.isinstance, entries, b.str)):
        raise Error(String(py=entries))
    if String(py=mode) == "identical":
        # API-shell metadata only: the arithmetic version the binding names,
        # kept on the model so state persistence cannot switch versions
        var current = entries["profile"]
        var previous = py_get(fields, "numerical_profile_")
        if previous is not None and current is not None and not Bool(py=previous == current):
            raise Error("mojolearn: CNN numerical profile differs from this model's recorded "
                        + "profile; use a matching binding or a fresh model for the new version")
        if previous is not None and current is None:
            raise Error("mojolearn: CNN numerical profile differs from this model's recorded "
                        + "profile; use a matching binding or a fresh model for the new version")
        if current is not None:
            py_setitem(fields, "numerical_profile_", current)
    py_setitem(fields, "numeric_mode_", mode)
    return entries


def xcnn_entry(entries: PythonObject, name: String) raises -> PythonObject:
    """A connected compiled entry of the layer's binding."""
    var entry = py_get(entries["symbols"], PythonObject(name))
    if entry is None:
        raise Error("AttributeError: the CNN binding has no " + name)
    return entry


def xcnn_mixed(entries: PythonObject) raises -> Bool:
    """Whether the binding has the mixed-residency entries (`x_cnn_*_m`)."""
    return Bool(py=entries["mixed"])


def xcnn_idn(entries: PythonObject, bit: Int) raises -> Bool:
    return (Int(py=entries["idn"]) & bit) != 0


def xcnn_idn2(entries: PythonObject, bit: Int) raises -> Bool:
    return (Int(py=entries["idn2"]) & bit) != 0


def xcnn_device_io_off(entries: PythonObject, context: PythonObject) raises -> Bool:
    """The resident I/O route is off: the environment switches read at
    import, or a `-D MOJOLEARN_IDN_ALL_OFF` build of the binding."""
    return Bool(py=context["xcnn_device_io_off"]) or xcnn_idn2(entries, XCNN_F2_ALL_OFF)


def xcnn_resident(entries: PythonObject, context: PythonObject) raises -> Bool:
    """Whether the binding runs the G3 resident glue."""
    return xcnn_mixed(entries) and xcnn_idn(entries, XCNN_F_RESIDENT) and not xcnn_device_io_off(entries, context)


# ---------------------------------------------------------------- shapes

def xcnn_shape(value: PythonObject) raises -> List[Int]:
    """A shape tuple as ints (an ndarray's or a DeviceTensor's slot)."""
    var shape = value.shape
    var out = List[Int]()
    for i in range(len(shape)):  # driver-loop: over shape words
        out.append(Int(py=shape[i]))
    return out^


def xcnn_size(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        n *= shape[i]
    return n


def xcnn_tuple(shape: List[Int]) raises -> PythonObject:
    var b = xcnn_builtins()
    var out = Python.list()
    for i in range(len(shape)):  # driver-loop: over shape words
        _ = py_call(py_call(b.getattr, out, "append"), PythonObject(shape[i]))
    return py_call(b.tuple, out)


def xcnn_shape_text(shape: List[Int]) -> String:
    var text = String("(")
    for i in range(len(shape)):  # driver-loop: over shape words of a refusal message
        if i > 0:
            text += ", "
        text += String(shape[i])
    if len(shape) == 1:
        text += ","
    return text + ")"


# ---------------------------------------------------------------- resident tensors

def xcnn_is_t(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(value, DeviceTensor)`."""
    return py_type(value) is context["xcnn_tensor_class"]


def xcnn_on_dev(entries: PythonObject, value: PythonObject, context: PythonObject) raises -> Bool:
    """`_on_dev(b, value)`: a tensor resident on this mixed binding."""
    if not xcnn_mixed(entries) or not xcnn_is_t(value, context):
        return False
    return value.h is not None and value.b is entries["module"]


def xcnn_tensor(module: PythonObject, shape: PythonObject, h: PythonObject, host: PythonObject, base: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`DeviceTensor(binding, shape, h, host, base)`, its slots set here."""
    var b = xcnn_builtins()
    var t = py_call(py_call(b.getattr, b.object, "__new__"), context["xcnn_tensor_class"])
    py_setattr(t, "b", module)
    py_setattr(t, "shape", shape)
    py_setattr(t, "h", h)
    py_setattr(t, "_host", host)
    py_setattr(t, "_base", base)
    return t


def xcnn_new(entries: PythonObject, shape: List[Int], nofill: Bool, context: PythonObject) raises -> PythonObject:
    """`DeviceTensor._new` (or `_new_out` with `nofill`: no +0.0 fill, on a
    binding that has `x_cnn_res_alloc_nofill`)."""
    var symbols = entries["symbols"]
    var alloc = py_get(symbols, "x_cnn_res_alloc_nofill") if nofill else PythonObject(None)
    if alloc is None:
        alloc = symbols["x_cnn_res_alloc"]
    var h = Int(py=py_call_typed(alloc, PythonObject(max(xcnn_size(shape), 1))))
    return xcnn_tensor(entries["module"], xcnn_tuple(shape), PythonObject(h), PythonObject(None), PythonObject(None),
                       context)


def xcnn_out_new(entries: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`_out_new`: a resident output whose every word the entry stores."""
    return xcnn_new(entries, shape, xcnn_resident(entries, context), context)


def xcnn_reshape_t(value: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`DeviceTensor.reshape(shape)`: the same resident words under another
    shape, the view keeping `value` alive."""
    var old = xcnn_shape(value)
    if xcnn_size(old) != xcnn_size(shape):
        raise Error("mojolearn: cannot reshape " + xcnn_shape_text(old) + " to " + xcnn_shape_text(shape))
    return xcnn_tensor(value.b, xcnn_tuple(shape), value.h, value._host, value, context)


def xcnn_wrap(entries: PythonObject, array: PythonObject, context: PythonObject) raises -> PythonObject:
    """`DeviceTensor._wrap(b, a)`: host float32 array `a` as a tensor (one
    upload)."""
    var a = xseq_f32(array, "x", False, context)
    if not xcnn_mixed(entries):
        return xcnn_tensor(entries["module"], a.shape, PythonObject(None), a, PythonObject(None), context)
    var shape = xcnn_shape(a)
    var t = xcnn_new(entries, shape, False, context)
    var size = xcnn_size(shape)
    if size:
        _ = py_call_typed(entries["symbols"]["x_cnn_res_upload"], t.h, sequential_address(a), PythonObject(size))
        _ = a  # keep the Python object alive through the native call above
    _ = a^
    return t


def xcnn_module_entries(module: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected entries whose module is `module`."""
    var b = xcnn_builtins()
    var table = context["xcnn_bindings"]
    var keys = Python.tuple("identical", "fast")
    for i in range(2):  # driver-loop: over the two numeric modes
        var entries = py_get(table, keys[i])
        if entries is not None and not Bool(py=py_call(b.isinstance, entries, b.str)) and entries["module"] is module:
            return entries
    raise Error("RuntimeError: a DeviceTensor of a binding this process did not connect")


def xcnn_host_of(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`DeviceTensor.numpy()`: the tensor's words as a host float32 array."""
    var shape = xcnn_shape(value)
    if value.h is None:
        var b = xcnn_builtins()
        return py_call(py_call(b.getattr, py_call(py_call(b.getattr, value._host, "reshape"), value.shape), "copy"))
    var out = xcnn_host_f32(shape, context)
    var size = xcnn_size(shape)
    if size:
        var entries = xcnn_module_entries(value.b, context)
        _ = py_call_typed(entries["symbols"]["x_cnn_res_download"], value.h, sequential_address(out), PythonObject(size))
        _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- host arrays

def xcnn_host_f32(shape: List[Int], context: PythonObject) raises -> PythonObject:
    """A new float32 ndarray of `shape`."""
    return sequential_array(xcnn_tuple(shape), context["float32"], context)


def xcnn_host_i32(shape: List[Int], context: PythonObject) raises -> PythonObject:
    return sequential_array(xcnn_tuple(shape), context["int32"], context)


def xcnn_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_f32(x, name)`: `np.asarray(x)`, float32 only (refused by name,
    never converted), C-contiguous."""
    var b = xcnn_builtins()
    # a DeviceTensor reads back through its `__array__` (`numpy()`)
    var a = xcnn_host_of(value, context) if xcnn_is_t(value, context) else sequential_input(value, context)
    var dtype = a.dtype
    if Int(py=dtype.num) != 11 or not Bool(py=dtype.isnative):
        raise Error("TypeError: mojolearn: " + name + " must be float32 (got " + String(py=py_call(b.str, dtype))
                    + "); convert it yourself so the bits that ran are bits you made")
    return xseq_f32(a, name, False, context)


def xcnn_reshape(value: PythonObject, shape: List[Int]) raises -> PythonObject:
    """`a.reshape(shape)` of a host ndarray (its own C method)."""
    var b = xcnn_builtins()
    return py_call(py_call(b.getattr, value, "reshape"), xcnn_tuple(shape))


def xcnn_addr(value: PythonObject) raises -> PythonObject:
    return sequential_address(value)


# ---------------------------------------------------------------- the layer's resident arrays

def xcnn_dev(layer: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_dev_of(layer, b)`: the layer's `_Dev` on this binding, a new one
    (the old one's arrays freed) when the binding changed."""
    var b = xcnn_builtins()
    var fields = py_fields(layer)
    var dev = py_get(fields, "_dev")
    if dev is not None and dev.b is entries["module"]:
        return dev
    if dev is not None:
        # `_Dev.free()`: every handle of the old binding released, the table emptied
        var old = py_fields(dev)
        var table = old["h"]
        var owner = xcnn_module_entries(old["b"], context)
        var handles = py_call(b.list, py_call(py_call(b.getattr, table, "values")))
        py_setitem(old, "h", Python.dict())
        for i in range(len(handles)):  # driver-loop: over the layer's resident array table
            _ = py_call_typed(owner["symbols"]["x_cnn_res_free"], handles[i][0])
    dev = py_call(py_call(b.getattr, b.object, "__new__"), context["xcnn_dev_class"])
    var state = py_fields(dev)
    py_setitem(state, "b", entries["module"])
    py_setitem(state, "h", Python.dict())
    py_setitem(fields, "_dev", dev)
    return dev


def xcnn_dev_get(layer: PythonObject, entries: PythonObject, name: String, count: Int,
                 context: PythonObject) raises -> PythonObject:
    """`_Dev.get(name, n)`: the layer's resident array `name` of `n` words,
    reallocated only when its size changes."""
    var n = max(count, 1)
    var dev = xcnn_dev(layer, entries, context)
    var table = py_fields(dev)["h"]
    var cur = py_get(table, PythonObject(name))
    if cur is not None and Int(py=cur[1]) == n:
        return cur[0]
    var symbols = entries["symbols"]
    if cur is not None:
        _ = py_call_typed(symbols["x_cnn_res_free"], cur[0])
        _ = py_call(py_call(xcnn_builtins().getattr, table, "pop"), PythonObject(name))
    var h = py_call_typed(symbols["x_cnn_res_alloc"], PythonObject(n))
    var handle = PythonObject(Int(py=h))
    py_setitem(table, PythonObject(name), Python.tuple(handle, n))
    return handle


def xcnn_dev_upload(layer: PythonObject, entries: PythonObject, name: String, array: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`_Dev.upload(name, a)`: resident array `name` holding the 4-byte words
    of host array `a`."""
    var size = Int(py=array.size)
    var h = xcnn_dev_get(layer, entries, name, size, context)
    _ = py_call_typed(entries["symbols"]["x_cnn_res_upload"], h, sequential_address(array), PythonObject(size))
    return h


def xcnn_dev_has(dev: PythonObject, name: String) raises -> Bool:
    """`name in dev.h`."""
    return py_contains(py_fields(dev)["h"], PythonObject(name))


def xcnn_dev_handle(dev: PythonObject, name: String) raises -> PythonObject:
    """`dev.h[name][0]`."""
    return py_fields(dev)["h"][PythonObject(name)][0]


def xcnn_download(dev: PythonObject, h: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """`dev.download(h, np.empty(count, np.float32))`."""
    var out = xcnn_host_f32([count], context)
    if count:
        var owner = xcnn_module_entries(py_fields(dev)["b"], context)
        _ = py_call_typed(owner["symbols"]["x_cnn_res_download"], h, sequential_address(out), PythonObject(count))
        _ = out  # keep the Python object alive through the native call above
    return out


def xcnn_slice_f32(source: PythonObject, offset: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """`a[offset:offset + count].copy()` of a C-ordered float32 array."""
    var out = xcnn_host_f32([count], context)
    if count:
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(out))),
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(source))) + offset,
               count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def xcnn_put_f32(dest: PythonObject, offset: Int, source: PythonObject, count: Int, name: String,
                 context: PythonObject) raises:
    """`dest[offset:offset + count] = source` (float32 words; `source` cast
    as NumPy assignment casts it), refused unless it holds `count` values."""
    var a = xseq_f32(sequential_input(source, context), name, False, context)
    if Int(py=a.size) != count:
        raise Error("ValueError: could not broadcast " + name + " into shape (" + String(count) + ",)")
    if count:
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(dest))) + offset,
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(a))),
               count=count)
        _ = a  # keep the Python object alive through the native call above
