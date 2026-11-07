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
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_class_admission import class_record, class_admit, dict_keys_admitted, dict_keys_admit
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


def xcnn_connect_numpy(context: PythonObject) raises:
    """Connect the optional NumPy metadata without requiring arrays."""
    classical_connect_numpy(context)
    if py_get(context, "ndarray") is None and py_get(context, "xcnn_numpy_import_tried") is None:
        # The CNN host buffers are ndarrays. Importing the package never loads
        # NumPy (NUMPY_FREE_CONTRACT.md; the former import-time `import numpy`
        # in _native_xcnn_bridge.connect is gone), so the first CNN operation
        # of a process imports an installed NumPy here, once, and connects it.
        # A NumPy-free install records the attempt and keeps running
        # scalar-only metadata operations; host arrays refuse in xcnn_numpy.
        py_setitem(context, "xcnn_numpy_import_tried", PythonObject(True))
        try:
            _ = Python.import_module("numpy")
        except:
            pass  # NumPy is optional; xcnn_numpy raises the ImportError when arrays are needed
        classical_connect_numpy(context)
    var admission = py_get(context, "admission_context")
    if admission is not None:
        classical_connect_numpy(admission)


def xcnn_numpy(context: PythonObject) raises:
    """Host-array creation and readback require the optional NumPy dependency."""
    xcnn_connect_numpy(context)
    if py_get(context, "ndarray") is None:
        raise Error("ImportError: CNN host arrays require NumPy; install NumPy to use this operation")


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
    # p3-cnn-sequence-tsa: an exact "identical" / "fast" is used as it is
    var text = String(py=mode)
    var key = mode
    if text != "identical" and text != "fast":
        key = PythonObject(String(mode).strip().lower())
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
        if current is not None and previous is None:
            # equal to `previous` otherwise (checked above): written once
            py_setitem(fields, "numerical_profile_", current)
    var recorded = py_get(fields, "numeric_mode_")
    if recorded is None or py_type(recorded) is not b.str or String(py=recorded) != String(py=key):
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
    # The bound `append` is looked up once per tuple, not once per word.
    var append = py_call(b.getattr, out, "append")
    for i in range(len(shape)):  # driver-loop: over shape words
        _ = py_call(append, PythonObject(shape[i]))
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


comptime _XCNN_TENSOR_LAYOUT_SLOT = "xcnn_tensor_layout_v1"


def xcnn_tensor_layout(context: PythonObject) raises:
    var cls = context["xcnn_tensor_class"]
    # p3-cnn-sequence-tsa: the verdict is kept per class under a type watcher
    # (`class_record`; any edit of the class or an ancestor misses), so the
    # MRO / namespace / slot walk runs once, not 3-8 times per layer call.
    # Python 3.10/3.11 keep the full walk on every call.
    var mods = NativeModules()
    if class_record(mods, _XCNN_TENSOR_LAYOUT_SLOT, cls) is not None:
        return
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
    class_admit(mods, _XCNN_TENSOR_LAYOUT_SLOT, cls, PythonObject(True))


def _xcnn_slot_raw(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """A tensor storage slot read after this operation ran
    `xcnn_tensor_layout` (which already compared every slot descriptor with
    its import-time identity): the exact-class check and the member
    descriptor read only. No Python code runs between that check and this
    read, so the class and its descriptors cannot have changed."""
    var cls = context["xcnn_tensor_class"]
    if py_type(value) is not cls:
        raise Error("TypeError: CNN tensor requires the connected native storage class")
    return py_call(context["xcnn_tensor_slots"][PythonObject(name)].__get__, value, cls)


def _xcnn_set_raw(value: PythonObject, name: String, item: PythonObject, context: PythonObject) raises:
    """A tensor storage slot write after `xcnn_tensor_layout` (see
    `_xcnn_slot_raw`)."""
    if py_type(value) is not context["xcnn_tensor_class"]:
        raise Error("TypeError: CNN tensor requires the connected native storage class")
    _ = py_call(context["xcnn_tensor_slots"][PythonObject(name)].__set__, value, item)


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
    # One layout check for the whole admission; the slot reads below are raw.
    xcnn_tensor_layout(context)
    _xcnn_tensor_admit_raw(value, context)


def _xcnn_tensor_admit_raw(value: PythonObject, context: PythonObject) raises:
    """`xcnn_tensor_admit` after this operation's `xcnn_tensor_layout`."""
    var b = xcnn_builtins()
    var shape = _xcnn_slot_raw(value, "shape", context)
    if py_type(shape) is not b.tuple:
        raise Error("TypeError: CNN tensor shape requires an exact tuple")
    for i in range(len(shape)):  # driver-loop: native tensor dimensions before arithmetic
        if py_type(shape[i]) is not b.int or Int(py=shape[i]) < 0:
            raise Error("TypeError: CNN tensor dimensions require nonnegative integers")
    var handle = _xcnn_slot_raw(value, "h", context)
    if handle is not None and (py_type(handle) is not b.int or Int(py=handle) < 1):
        raise Error("TypeError: CNN tensor handle requires a positive native integer")
    _ = xcnn_module_entries(_xcnn_slot_raw(value, "b", context), context)
    if handle is None:
        sequential_admit_value(_xcnn_slot_raw(value, "_host", context), context)


def xcnn_on_dev(entries: PythonObject, value: PythonObject, context: PythonObject) raises -> Bool:
    """`_on_dev(b, value)`: a tensor resident on this mixed binding."""
    if not xcnn_mixed(entries) or not xcnn_is_t(value, context):
        return False
    # `xcnn_is_t` ran the layout check; the admission and slot reads reuse it.
    _xcnn_tensor_admit_raw(value, context)
    return _xcnn_slot_raw(value, "h", context) is not None and _xcnn_slot_raw(value, "b", context) is entries["module"]


def xcnn_tensor(module: PythonObject, shape: PythonObject, h: PythonObject, host: PythonObject, base: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`DeviceTensor(binding, shape, h, host, base)`, its slots set here."""
    xcnn_tensor_layout(context)
    return _xcnn_tensor_raw(module, shape, h, host, base, context)


def _xcnn_tensor_raw(module: PythonObject, shape: PythonObject, h: PythonObject, host: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`xcnn_tensor` after this operation's `xcnn_tensor_layout`: the five
    slots written without re-walking the class layout once per slot."""
    var b = xcnn_builtins()
    var t = py_call(py_call(b.getattr, b.object, "__new__"), context["xcnn_tensor_class"])
    _xcnn_set_raw(t, "b", module, context)
    _xcnn_set_raw(t, "shape", shape, context)
    _xcnn_set_raw(t, "h", h, context)
    _xcnn_set_raw(t, "_host", host, context)
    _xcnn_set_raw(t, "_base", base, context)
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
    # One layout check covers the three reads and the five writes.
    xcnn_tensor_layout(context)
    var module = _xcnn_slot_raw(value, "b", context)
    var dims = xcnn_tuple(shape)
    var handle = _xcnn_slot_raw(value, "h", context)
    var host = _xcnn_slot_raw(value, "_host", context)
    return _xcnn_tensor_raw(module, dims, handle, host, value, context)


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
    xcnn_numpy(context)
    xcnn_tensor_admit(value, context)
    var shape = xcnn_shape(value)
    # The admission above ran the layout check; no Python code ran since.
    var handle = _xcnn_slot_raw(value, "h", context)
    if handle is None:
        var source = xcnn_f32(_xcnn_slot_raw(value, "_host", context), "tensor host", context)
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
        var entries = xcnn_module_entries(_xcnn_slot_raw(value, "b", context), context)
        _ = py_call_typed(entries["symbols"]["x_cnn_res_download"], handle, xcnn_addr(out), PythonObject(size))
        _ = out
    return out


# ---------------------------------------------------------------- host arrays

def xcnn_host_f32(shape: List[Int], context: PythonObject) raises -> PythonObject:
    """A new float32 ndarray of `shape`."""
    xcnn_numpy(context)
    return sequential_array(xcnn_tuple(shape), context["float32"], context)


def xcnn_host_i32(shape: List[Int], context: PythonObject) raises -> PythonObject:
    xcnn_numpy(context)
    return sequential_array(xcnn_tuple(shape), context["int32"], context)


def xcnn_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_f32(x, name)`: `np.asarray(x)`, float32 only (refused by name,
    never converted), C-contiguous."""
    xcnn_numpy(context)
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


def xcnn_readonly(value: PythonObject, original: PythonObject, context: PythonObject) raises -> PythonObject:
    """p3-cnn-sequence-tsa (hazard A, the read-only scheme): a read-only
    float32 ndarray over the words of `value` (a contiguous float32 ndarray,
    the snapshot just uploaded to a resident copy). Its base is a read-only
    memoryview, so `arr[:] = W` or `arr -= lr * g` raises NumPy's read-only
    error and `arr.flags.writeable = True` is refused: the device copy can
    never silently diverge from the host words. `original` (the array the
    layer held, which other references may share) is marked read-only too.
    Replacing the attribute stays allowed; callers key on its identity."""
    if py_type(original) is context["ndarray"]:
        py_setattr(original.flags, "writeable", PythonObject(False))
    if py_type(value) is not context["ndarray"]:
        raise Error("TypeError: CNN resident weights require an exact native ndarray")
    py_setattr(value.flags, "writeable", PythonObject(False))
    # memoryview of a non-writeable ndarray is itself read-only
    var view = native_buffer_view(value)
    return py_call(context["ndarray"], xcnn_tuple(xcnn_shape(value)), dtype=context["float32"], buffer=view)


def xcnn_writable_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A fresh writeable float32 copy of a contiguous float32 ndarray (the
    host weights handed back when a layer's resident copy is dropped)."""
    var shape = xcnn_shape(value)
    var out = xcnn_host_f32(shape, context)
    var count = xcnn_size(shape)
    if count:
        memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(out))),
               src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(value))), count=count)
    _ = value
    return out


def xcnn_reshape(value: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`xcnn_reshape` with the operation's connected `context["ndarray"]`
    (p3-cnn-sequence-tsa: the two-argument form builds a fresh module dict
    and resolves the ndarray type again on every call)."""
    var cls = py_get(context, "ndarray")
    if cls is None:
        return xcnn_reshape(value, shape)
    if py_type(value) is not cls:
        raise Error("TypeError: CNN reshape requires an exact ndarray")
    admit_native_buffer_exporter(value)
    var view = AdmittedBorrowedBuffer(value)
    var layout = view.layout.value().copy()
    if layout.count != xcnn_size(shape):
        raise Error("ValueError: CNN reshape changes the number of elements")
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
    # p3-cnn-sequence-tsa: a handle table that passed the full check below and
    # has not changed since (a dictionary watcher sees every insert, replace,
    # delete and deallocation) is not walked again; a BasicBlock forward read
    # it about ten times per call. Python 3.10/3.11 keep the walk.
    if dict_keys_admitted(table):
        return fields
    var names = py_call(b.list, table.keys())
    for i in range(len(names)):  # driver-loop: native resident buffer allocation metadata
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: CNN resident buffer names require native strings")
    for i in range(len(names)):  # driver-loop: native resident handle and capacity metadata
        var item = table[names[i]]
        if (py_type(item) is not b.tuple or len(item) != 2 or py_type(item[0]) is not b.int
                or py_type(item[1]) is not b.int or Int(py=item[0]) < 1 or Int(py=item[1]) < 1):
            raise Error("TypeError: CNN resident buffers require native handle/capacity pairs")
    dict_keys_admit(table)
    return fields


def _xcnn_settle_old_dev(fields: PythonObject, dev: PythonObject, context: PythonObject) raises:
    """Before layer owner `dev` (another binding's) frees its handles: a
    pending resident weight gradient (`_gw_dev` / `_gb_dev`, a (dev, handle)
    pair) becomes the host `grad_weight_` / `grad_bias_` it would have
    downloaded on first read (the `_conv_gradient` shapes), a resident
    dropout mask becomes `_mask_host`, and every other field holding an old
    handle number (`_idx_dev`, `_mask_dev`, `_chain`, `_gdev_for`) is
    cleared."""
    var b = xcnn_builtins()
    var names = Python.tuple("_gw_dev", "_gb_dev")
    for i in range(2):  # small-loop(the two pending gradient fields)
        var pending = py_get(fields, names[i])
        if pending is None or py_type(pending) is not b.tuple or len(pending) != 2 or pending[0] is not dev:
            continue
        var weight = i == 0
        var shape = xcnn_shape(fields["weight_"]) if weight else [Int(py=fields["out_channels"])]
        var value = xcnn_download(dev, pending[1], xcnn_size(shape), context)
        if weight and Bool(py=py_get(fields, "_gw_1d")):
            shape = [shape[0], shape[1], shape[3]]
        py_setitem(fields, "grad_weight_" if weight else "grad_bias_", xcnn_reshape(value, shape, context))
        py_setitem(fields, names[i], PythonObject(None))
        if weight:
            py_setitem(fields, "_gw_1d", PythonObject(False))
    var mask = py_get(fields, "_mask_dev")
    if mask is not None and py_get(fields, "_mask_host") is None and py_get(fields, "_mask_shape") is not None:
        var mshape = List[Int]()
        var words = fields["_mask_shape"]
        for i in range(len(words)):  # small-loop(mask shape words)
            mshape.append(Int(py=words[i]))
        var host = xcnn_download(dev, PythonObject(Int(py=mask)), xcnn_size(mshape), context)
        py_setitem(fields, "_mask_host", xcnn_reshape(host, mshape, context))
    for name in ["_idx_dev", "_mask_dev", "_chain", "_gdev_for", "_gdev_g"]:  # small-loop(fields naming old handles)
        if py_contains(fields, PythonObject(name)):
            py_setitem(fields, PythonObject(name), PythonObject(None))


def xcnn_dev(layer: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_dev_of(layer, b)`: the layer's `_Dev` on this binding, a new one
    (the old one's arrays freed) when the binding changed."""
    var b = xcnn_builtins()
    var fields = _fields(layer)
    var dev = py_get(fields, "_dev")
    if dev is not None and xcnn_dev_fields(dev, context)["b"] is entries["module"]:
        return dev
    if dev is not None:
        # p3-cnn-sequence-tsa (hazard C): fields that name a handle of the
        # old owner are settled before its handles are freed (pending weight
        # gradients and a dropout mask downloaded to the host, the rest
        # cleared), so no later read uses a freed or reused handle number.
        _xcnn_settle_old_dev(fields, dev, context)
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
    # `xcnn_dev` admitted an existing owner (or made a fresh one with an
    # empty table) and no Python code ran since: its table is read directly.
    var table = _fields(dev)["h"]
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
