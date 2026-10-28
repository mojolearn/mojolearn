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
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_address import native_buffer_address
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_numpy_metadata import native_numpy_type
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.neural_block_admission import neural_plain_value
from bindings.sequential_admission import sequential_admit_value
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer

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
    return native_loaded_module("builtins")


def xcnn_numpy(context: PythonObject) raises:
    """NumPy loaded and its native types connected: the layers return
    ndarrays, as `_np()` (`require_numpy`) made them."""
    classical_connect_numpy(context)
    if py_get(context, "ndarray") is None:
        raise Error("ImportError: CNN operations require NumPy; install it before importing mojolearn")


def xcnn_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Layer._binding()`: the connected `_mojolearn_x_cnn` entries of the
    layer's mode (or the process default), with the identical-mode
    numerical-profile check; records `numeric_mode_`."""
    var b = xcnn_builtins()
    var mode = py_get(fields, "numeric_mode")
    neural_plain_value(mode, 0, context)
    if mode is None or not Bool(py=mode):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var key = PythonObject(String(mode).strip().lower())
    if String(py=key) != "identical" and String(py=key) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["xcnn_bindings"][key]
    if Bool(py=py_call(b.isinstance, entries, b.str)):
        raise Error(String(py=entries))
    if String(key) == "identical":
        # API-shell metadata only: the arithmetic version the binding names,
        # kept on the model so state persistence cannot switch versions
        var current = entries["profile"]
        var previous = py_get(fields, "numerical_profile_")
        neural_plain_value(previous, 0, context)
        if previous is not None and current is not None and not Bool(py=previous == current):
            raise Error("mojolearn: CNN numerical profile differs from this model's recorded "
                        + "profile; use a matching binding or a fresh model for the new version")
        if previous is not None and current is None:
            raise Error("mojolearn: CNN numerical profile differs from this model's recorded "
                        + "profile; use a matching binding or a fresh model for the new version")
        if current is not None:
            py_setitem(fields, "numerical_profile_", current)
    py_setitem(fields, "numeric_mode_", key)
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
    """Native shape metadata, with subclass/descriptor hooks refused first."""
    var cls = py_type(value)
    var namespace = _class_fields(cls)
    var descriptor = py_get(namespace, "shape")
    var types = native_loaded_module("types")
    if (py_type(descriptor) is not types.MemberDescriptorType
            and py_type(descriptor) is not py_type(xcnn_builtins().type.__dict__["__dict__"])):
        raise Error("TypeError: CNN shape requires native storage metadata")
    if descriptor.__objclass__ is not cls:
        raise Error("TypeError: CNN shape refuses subclasses")
    var shape = py_call(descriptor.__get__, value, cls)
    var b = xcnn_builtins()
    if py_type(shape) is not b.tuple:
        raise Error("TypeError: CNN shape requires an exact tuple")
    var out = List[Int]()
    for i in range(len(shape)):  # driver-loop: snapshot native shape words
        if py_type(shape[i]) is not b.int or Int(py=shape[i]) < 0:
            raise Error("TypeError: CNN dimensions must be nonnegative native integers")
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
    if py_type(value) is not context["xcnn_tensor_class"]:
        return False
    xcnn_tensor_layout(context)
    return True


def xcnn_tensor_layout(context: PythonObject) raises:
    var cls = context["xcnn_tensor_class"]
    var b = xcnn_builtins()
    if py_type(cls) is not b.type:
        raise Error("TypeError: CNN tensor refuses Python metaclass hooks")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: fixed tensor storage access ancestry
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, "__getattr__"):
            raise Error("TypeError: CNN tensor refuses Python attribute hooks")
        for name in ["__getattribute__", "__setattr__"]:  # driver-loop: native slot access protocols
            var own = py_get(namespace, PythonObject(name))
            if own is not None and own is not b.object.__dict__[name]:
                raise Error("TypeError: CNN tensor refuses Python attribute hooks")
    var slots = context["xcnn_tensor_slots"]
    var namespace = _class_fields(cls)
    var finalizer = py_get(context, "xcnn_tensor_finalizer")
    if finalizer is not None and py_get(namespace, "__del__") is not finalizer:
        raise Error("TypeError: CNN tensor requires its native lifetime descriptor")
    for name in ["b", "shape", "h", "_host", "_base"]:  # driver-loop: native connected tensor member descriptors
        if py_get(namespace, PythonObject(name)) is not slots[PythonObject(name)]:
            raise Error("TypeError: CNN tensor storage descriptors changed")


def xcnn_tensor_slot(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    xcnn_tensor_layout(context)
    var cls = context["xcnn_tensor_class"]
    if py_type(value) is not cls:
        raise Error("TypeError: CNN tensor requires the connected native storage class")
    var slots = context["xcnn_tensor_slots"]
    var slot = slots[PythonObject(name)]
    if py_get(_class_fields(cls), PythonObject(name)) is not slot:
        raise Error("TypeError: CNN tensor storage descriptors changed")
    return py_call(slot.__get__, value, cls)


def xcnn_tensor_set(value: PythonObject, name: String, item: PythonObject, context: PythonObject) raises:
    xcnn_tensor_layout(context)
    var cls = context["xcnn_tensor_class"]
    if py_type(value) is not cls:
        raise Error("TypeError: CNN tensor requires the connected native storage class")
    var slot = context["xcnn_tensor_slots"][PythonObject(name)]
    if py_get(_class_fields(cls), PythonObject(name)) is not slot:
        raise Error("TypeError: CNN tensor storage descriptors changed")
    _ = py_call(slot.__set__, value, item)


def xcnn_tensor_admit(value: PythonObject, context: PythonObject) raises:
    var b = xcnn_builtins()
    var shape = xcnn_tensor_slot(value, "shape", context)
    if py_type(shape) is not b.tuple:
        raise Error("TypeError: CNN tensor shape requires an exact tuple")
    for i in range(len(shape)):  # driver-loop: native tensor dimensions before arithmetic
        if py_type(shape[i]) is not b.int or Int(py=shape[i]) < 0:
            raise Error("TypeError: CNN tensor dimensions require nonnegative integers")
    var handle = xcnn_tensor_slot(value, "h", context)
    if handle is not None and (py_type(handle) is not b.int or Int(py=handle) < 1):
        raise Error("TypeError: CNN tensor handle requires a positive native integer")
    _ = xcnn_module_entries(xcnn_tensor_slot(value, "b", context), context)
    if handle is None:
        sequential_admit_value(xcnn_tensor_slot(value, "_host", context), context)


def xcnn_on_dev(entries: PythonObject, value: PythonObject, context: PythonObject) raises -> Bool:
    """`_on_dev(b, value)`: a tensor resident on this mixed binding."""
    if not xcnn_mixed(entries) or not xcnn_is_t(value, context):
        return False
    xcnn_tensor_admit(value, context)
    return xcnn_tensor_slot(value, "h", context) is not None and xcnn_tensor_slot(value, "b", context) is entries["module"]


def xcnn_tensor(module: PythonObject, shape: PythonObject, h: PythonObject, host: PythonObject, base: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`DeviceTensor(binding, shape, h, host, base)`, its slots set here."""
    var b = xcnn_builtins()
    var t = py_call(py_call(b.getattr, b.object, "__new__"), context["xcnn_tensor_class"])
    xcnn_tensor_set(t, "b", module, context)
    xcnn_tensor_set(t, "shape", shape, context)
    xcnn_tensor_set(t, "h", h, context)
    xcnn_tensor_set(t, "_host", host, context)
    xcnn_tensor_set(t, "_base", base, context)
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
    return xcnn_tensor(xcnn_tensor_slot(value, "b", context), xcnn_tuple(shape),
                       xcnn_tensor_slot(value, "h", context), xcnn_tensor_slot(value, "_host", context), value, context)


def xcnn_wrap(entries: PythonObject, array: PythonObject, context: PythonObject) raises -> PythonObject:
    """`DeviceTensor._wrap(b, a)`: host float32 array `a` as a tensor (one
    upload)."""
    sequential_admit_value(array, context)
    var a = xseq_f32(sequential_input(array, context), "x", False, context)
    if not xcnn_mixed(entries):
        return xcnn_tensor(entries["module"], a.shape, PythonObject(None), a, PythonObject(None), context)
    var shape = xcnn_shape(a)
    var t = xcnn_new(entries, shape, False, context)
    var size = xcnn_size(shape)
    if size:
        _ = py_call_typed(entries["symbols"]["x_cnn_res_upload"], t.h, xcnn_addr(a), PythonObject(size))
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
    xcnn_tensor_admit(value, context)
    var shape = xcnn_shape(value)
    var handle = xcnn_tensor_slot(value, "h", context)
    if handle is None:
        var source = xcnn_f32(xcnn_tensor_slot(value, "_host", context), "tensor host", context)
        if xcnn_size(shape) != Int(py=source.size):
            raise Error("ValueError: CNN tensor host storage has the wrong size")
        var out = xcnn_host_f32(shape, context)
        var size = xcnn_size(shape)
        if size:
            memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(out)),
                   src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(source)), count=size)
        return out
    var out = xcnn_host_f32(shape, context)
    var size = xcnn_size(shape)
    if size:
        var entries = xcnn_module_entries(xcnn_tensor_slot(value, "b", context), context)
        _ = py_call_typed(entries["symbols"]["x_cnn_res_download"], handle, xcnn_addr(out), PythonObject(size))
        _ = out
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
    if not xcnn_is_t(value, context):
        sequential_admit_value(value, context)
    var a = xcnn_host_of(value, context) if xcnn_is_t(value, context) else sequential_input(value, context)
    admit_native_buffer_exporter(a)
    var borrowed = AdmittedBorrowedBuffer(a)
    var layout = borrowed.layout.value().copy()
    if borrowed.format != "f" and borrowed.format != "@f" and borrowed.format != "=f":
        raise Error("TypeError: mojolearn: " + name + " must be native float32; convert it yourself so the bits that ran are bits you made")
    # The original exporter metadata is captured once. The internal view has
    # private shape metadata and retains the source owner for the operation.
    if layout.contiguous():
        return py_call(context["ndarray"], xcnn_tuple(layout.shape), dtype=context["float32"], buffer=a)
    var out = xcnn_host_f32(layout.shape, context)
    layout.copy_to(native_buffer_address(out))
    _ = a
    return out


def xcnn_reshape(value: PythonObject, shape: List[Int]) raises -> PythonObject:
    """A contiguous ndarray view built from native buffer metadata."""
    admit_native_buffer_exporter(value)
    var view = AdmittedBorrowedBuffer(value)
    var layout = view.layout.value().copy()
    if layout.count != xcnn_size(shape):
        raise Error("ValueError: CNN reshape changes the number of elements")
    var metadata = Python.dict(modules=native_loaded_module("sys").modules)
    var cls = native_numpy_type("ndarray", metadata)
    if py_type(value) is not cls:
        raise Error("TypeError: CNN reshape requires an exact ndarray")
    return py_call(cls, xcnn_tuple(shape), dtype=value.dtype, buffer=value)


def xcnn_addr(value: PythonObject) raises -> PythonObject:
    admit_native_buffer_exporter(value)
    return PythonObject(native_buffer_address(value))


# ---------------------------------------------------------------- the layer's resident arrays

def xcnn_dev_fields(dev: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(dev) is not context["xcnn_dev_class"]:
        raise Error("TypeError: CNN resident state requires its connected native owner")
    var fields = _fields(dev)
    _ = xcnn_module_entries(fields["b"], context)
    var table = fields["h"]
    var b = xcnn_builtins()
    if py_type(table) is not b.dict:
        raise Error("TypeError: CNN resident handles require an exact dictionary")
    var names = py_call(b.list, table.keys())
    for i in range(len(names)):  # driver-loop: native resident buffer allocation metadata
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: CNN resident buffer names require native strings")
    for i in range(len(names)):  # driver-loop: native resident handle and capacity metadata
        var item = table[names[i]]
        if (py_type(item) is not b.tuple or len(item) != 2 or py_type(item[0]) is not b.int
                or py_type(item[1]) is not b.int or Int(py=item[0]) < 1 or Int(py=item[1]) < 1):
            raise Error("TypeError: CNN resident buffers require native handle/capacity pairs")
    return fields


def xcnn_dev(layer: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_dev_of(layer, b)`: the layer's `_Dev` on this binding, a new one
    (the old one's arrays freed) when the binding changed."""
    var b = xcnn_builtins()
    var fields = _fields(layer)
    var dev = py_get(fields, "_dev")
    if dev is not None and xcnn_dev_fields(dev, context)["b"] is entries["module"]:
        return dev
    if dev is not None:
        # `_Dev.free()`: every handle of the old binding released, the table emptied
        var old = xcnn_dev_fields(dev, context)
        var table = old["h"]
        var owner = xcnn_module_entries(old["b"], context)
        var handles = py_call(b.list, py_call(py_call(b.getattr, table, "values")))
        py_setitem(old, "h", Python.dict())
        for i in range(len(handles)):  # driver-loop: over the layer's resident array table
            _ = py_call_typed(owner["symbols"]["x_cnn_res_free"], handles[i][0])
    dev = py_call(py_call(b.getattr, b.object, "__new__"), context["xcnn_dev_class"])
    var state = _fields(dev)
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
    var table = xcnn_dev_fields(dev, context)["h"]
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
    _ = py_call_typed(entries["symbols"]["x_cnn_res_upload"], h, xcnn_addr(array), PythonObject(size))
    return h


def xcnn_dev_has(dev: PythonObject, name: String) raises -> Bool:
    """`name in dev.h`."""
    return py_contains(_fields(dev)["h"], PythonObject(name))


def xcnn_dev_handle(dev: PythonObject, name: String) raises -> PythonObject:
    """`dev.h[name][0]`."""
    return _fields(dev)["h"][PythonObject(name)][0]


def xcnn_download(dev: PythonObject, h: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """`dev.download(h, np.empty(count, np.float32))`."""
    var out = xcnn_host_f32([count], context)
    if count:
        var owner = xcnn_module_entries(xcnn_dev_fields(dev, context)["b"], context)
        _ = py_call_typed(owner["symbols"]["x_cnn_res_download"], h, xcnn_addr(out), PythonObject(count))
        _ = out  # keep the Python object alive through the native call above
    return out


def xcnn_slice_f32(source: PythonObject, offset: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """`a[offset:offset + count].copy()` of a C-ordered float32 array."""
    var out = xcnn_host_f32([count], context)
    if count:
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(out))),
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(source))) + offset,
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
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(dest))) + offset,
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(a))),
               count=count)
        _ = a  # keep the Python object alive through the native call above
