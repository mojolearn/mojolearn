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
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains, py_item
from bindings.py_cpython_table import cpython_table
from std.python._cpython import GILReleased, PyObjectPtr
from core.shared_slot import SharedSlot
from core.entry_table import EntryRef
from core.xcnn_entry_table import (
    XcnnEntryKey, xcnn_entry_ref, XC_RES_ALLOC, XC_RES_ALLOC_NOFILL, XC_RES_FREE,
    XC_RES_UPLOAD, XC_RES_DOWNLOAD,
)
from bindings.sequential_arrays_door import sequential_array, sequential_address
from bindings.sequential_inputs_door import sequential_input
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.admitted_sequence_door import admit_nd_f32
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit, dict_keys_admitted, dict_keys_admit
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.native_buffer_owner_pack import native_buffer_view, native_storage_allocate
from bindings.native_numpy_metadata_pack import native_numpy_type
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.neural_block_admission_door import neural_plain_value
from bindings.sequential_admission_door import sequential_admit_value
from providers.python.borrowed_buffer_pyapi_unpack import admit_native_buffer_exporter, AdmittedBorrowedBuffer

# lane fam-neural switch bits (`x_cnn_idn_flags`)
# The lane switch bits, the size/shape-text words and the native call moved
# to the RUN half, x_cnn/door_run.mojo (lane doorsplit-cnn-xseq); they are
# imported here so the doors keep one import site.
from x_cnn.door_run import (
    XCNN_F_ADAPT, XCNN_F_GRAPH, XCNN_F_PAD, XCNN_F_GROUP, XCNN_F_LAYER_IO, XCNN_F_PIN, XCNN_F_RESIDENT,
    XCNN_F2_ALL_OFF, xcnn_run_call, xcnn_size, xcnn_shape_text, xcnn_run_reshape_check,
    xcnn_run_copy_f32,
)


# ---------------------------------------------------------------- held names and exact types
# Lane fx-cnn-layers (2026-11-18). One resident layer call ran 150 to 250
# CPython calls, most of them a `context["..."]`, `fields["..."]` or
# `b.<type>` read that first built a Python str from a literal (decode,
# allocate, hash, compare, free) and then looked it up; the call profile of
# global-avgpool infer on the L40S (ec1c4b5aff) put 22 of its 50 us in the
# door, 400k instructions per call, with the kernel at 12 us. The names the
# doors read on every call are now built once per process and held here, and
# the exact builtin types the doors compare against are the types of
# C-created values (the same identities `classical_connect_numpy` takes that
# way), read once. A read through a held name is the same dictionary lookup
# with the same hashing and equality; only the str is no longer rebuilt.
# Nothing here is keyed on a layer, a shape or a dataset.

#: index of each held name in `_XcnnNames.keys` (the fill list below is in this order)
comptime XK_TENSOR_CLASS = 0
comptime XK_BINDINGS = 1
comptime XK_BACKEND_FIELDS = 2
comptime XK_DEFAULT_MODE = 3
comptime XK_DEVICE_IO_OFF = 4
comptime XK_DEV_CLASS = 5
comptime XK_MODULE = 6
comptime XK_MIXED = 7
comptime XK_IDN = 8
comptime XK_IDN2 = 9
comptime XK_PROFILE = 10
comptime XK_NATIVE_KEY = 11
comptime XK_IDENTICAL = 12
comptime XK_FAST = 13
comptime XK_B = 14
comptime XK_SHAPE = 15
comptime XK_H = 16
comptime XK_HOST = 17
comptime XK_BASE = 18
comptime XK_NUMERIC_MODE = 19
comptime XK_PROFILE_REC = 20
comptime XK_MODE_REC = 21
comptime XK_DEV = 22
comptime XK_XSHAPE = 23
comptime XK_IDX_HOST = 24
comptime XK_IDX_DEV = 25
comptime XK_IDX_SHAPE = 26
comptime XK_LAYER = 27
comptime XK_POOL_CACHE = 28
comptime XK_OUTPUT_SIZE = 29
comptime XK_KERNEL_SIZE = 30
comptime XK_STRIDE = 31
comptime XK_PADDING = 32
comptime XK_DILATION = 33
comptime XK_COUNT_INCLUDE_PAD = 34
comptime XK_CEIL_MODE = 35
comptime XK_DIVISOR = 36
comptime XK_NDARRAY = 37
comptime XK_ADMISSION = 38

#: index of each exact builtin type in `_XcnnNames.types`
comptime XT_TUPLE = 0
comptime XT_INT = 1
comptime XT_STR = 2
comptime XT_DICT = 3
comptime XT_BOOL = 4
comptime XT_FLOAT = 5
comptime XT_LIST = 6


struct _XcnnNames(Defaultable, Movable):
    # `keys`: the held name objects, filled once under the GIL (empty until
    # then: the slot's creation builds no Python object, core/shared_slot.mojo).
    # `types`: addresses of CPython's static, immortal builtin types.
    var ready: Bool
    var keys: List[PythonObject]
    var types: List[Int]

    def __init__(out self):
        self.ready = False
        self.keys = List[PythonObject]()
        self.types = List[Int]()


comptime _XCNN_NAMES = SharedSlot[_XcnnNames, "MojolearnXcnnHeldNamesV1"]


def _xcnn_type_address(sample: PythonObject) raises -> Int:
    # The `_is_exact_dict` idiom of bindings/py_balanced.mojo.
    ref cpy = Python().cpython()
    var cls = cpy.PyObject_Type(sample._obj_ptr)
    var address = Int(cls)
    cpy.Py_DecRef(cls)
    return address


def _xcnn_names() raises -> UnsafePointer[_XcnnNames, MutAnyOrigin]:
    """The held names and exact types, filled at first use. Every caller
    holds the GIL (the doors run under it), which serializes the fill."""
    var g = _XCNN_NAMES.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if g[].ready:
        return g
    if len(g[].keys) == 0:
        # small-loop(the fixed names the doors read per call; order = the XK_ indexes)
        for name in ["xcnn_tensor_class", "xcnn_bindings", "backend_fields", "_DEFAULT_MODE", "xcnn_device_io_off",
                     "xcnn_dev_class", "module", "mixed", "idn", "idn2", "profile", "native_key", "identical", "fast",
                     "b", "shape", "h", "_host", "_base", "numeric_mode", "numerical_profile_", "numeric_mode_",
                     "_dev", "_xshape", "_idx_host", "_idx_dev", "_idx_shape", "_layer", "_pool_cache", "output_size",
                     "kernel_size", "stride", "padding", "dilation", "count_include_pad", "ceil_mode",
                     "divisor_override", "ndarray", "admission_context"]:
            g[].keys.append(PythonObject(name))
    if len(g[].types) == 0:
        g[].types.append(_xcnn_type_address(Python.tuple()))
        g[].types.append(_xcnn_type_address(PythonObject(0)))
        g[].types.append(_xcnn_type_address(PythonObject("")))
        g[].types.append(_xcnn_type_address(Python.dict()))
        g[].types.append(_xcnn_type_address(PythonObject(True)))
        g[].types.append(_xcnn_type_address(PythonObject(0.0)))
        g[].types.append(_xcnn_type_address(Python.list()))
    g[].ready = True
    return g


def xcnn_name(which: Int) raises -> PythonObject:
    """The held str of name `XK_<which>` (a new reference to the one object)."""
    var g = _xcnn_names()
    return g[].keys[which]


def _xcnn_exact_ptr(value: PyObjectPtr, kind: Int) raises -> Bool:
    var g = _xcnn_names()
    ref cpy = Python().cpython()
    var cls = cpy.PyObject_Type(value)
    var same = Int(cls) == g[].types[kind]
    cpy.Py_DecRef(cls)
    return same


def xcnn_exact(value: PythonObject, kind: Int) raises -> Bool:
    """`type(value) is <the builtin type XT_<kind>>`: an exact-type test
    against the type of a C-created value (no `builtins` attribute read)."""
    return _xcnn_exact_ptr(value._obj_ptr, kind)


def xcnn_item(mapping: PythonObject, which: Int) raises -> PythonObject:
    """`mapping["<name>"]` through the held name: one dictionary lookup on an
    exact dict; a missing key, or any other mapping, takes the subscript
    (`PyObject_GetItem`) and raises what it raised."""
    var g = _xcnn_names()
    if _xcnn_exact_ptr(mapping._obj_ptr, XT_DICT):
        var found = cpython_table()[].PyDict_GetItemWithError(mapping._obj_ptr, g[].keys[which]._obj_ptr)
        if found:
            return PythonObject(from_borrowed=found)
    return py_item(mapping, g[].keys[which])


def xcnn_get(mapping: PythonObject, which: Int) raises -> PythonObject:
    """`mapping.get("<name>")` through the held name."""
    var g = _xcnn_names()
    return py_get(mapping, g[].keys[which])


def xcnn_put(mapping: PythonObject, which: Int, value: PythonObject) raises:
    """`mapping["<name>"] = value` through the held name."""
    var g = _xcnn_names()
    py_setitem(mapping, g[].keys[which], value)


def xcnn_at(sequence: PythonObject, index: Int) raises -> PythonObject:
    """`sequence[index]`: a borrowed read of an exact tuple's item; anything
    else (and an index out of range) takes the subscript and raises what it
    raised."""
    if index >= 0 and _xcnn_exact_ptr(sequence._obj_ptr, XT_TUPLE):
        var t = cpython_table()
        if index < t[].PyTuple_Size(sequence._obj_ptr):
            return PythonObject(from_borrowed=t[].PyTuple_GetItem(sequence._obj_ptr, index))
    return py_item(sequence, PythonObject(index))


def xcnn_int_pair(value: PythonObject, mut first: Int, mut second: Int) raises -> Bool:
    """True, with the two words, when `value` is an exact tuple of two exact
    ints that fit a machine word. That is the stored form of every pool
    setting (`xcnn_pool_init` / `xcnn_adaptive_init` write such a pair), and
    for it the general parse returns an equal pair, so a caller that gets
    True skips the parse; False sends it to the general parse, which admits
    or refuses exactly as before."""
    if not _xcnn_exact_ptr(value._obj_ptr, XT_TUPLE):
        return False
    var t = cpython_table()
    if t[].PyTuple_Size(value._obj_ptr) != 2:
        return False
    var a = t[].PyTuple_GetItem(value._obj_ptr, 0)
    var b = t[].PyTuple_GetItem(value._obj_ptr, 1)
    if not _xcnn_exact_ptr(a, XT_INT) or not _xcnn_exact_ptr(b, XT_INT):
        return False
    ref cpy = Python().cpython()
    var wa = t[].PyLong_AsLongLong(a)
    if wa == -1 and cpy.PyErr_Occurred():
        # Out of machine range: the general parse raises as before.
        t[].PyErr_Clear()
        return False
    var wb = t[].PyLong_AsLongLong(b)
    if wb == -1 and cpy.PyErr_Occurred():
        t[].PyErr_Clear()
        return False
    first = Int(wa)
    second = Int(wb)
    return True


def xcnn_builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def xcnn_connect_numpy(context: PythonObject) raises:
    """Connect the optional NumPy metadata without requiring arrays."""
    # fx-cnn-layers (2026-11-18): the connected case first, through held
    # names. With `ndarray` connected on the context and on its admission
    # context (or no admission context) every statement below is a no-op
    # (`classical_connect_numpy` returns at its own probe), so this is the
    # same outcome with two or three dictionary reads; it ran two probes that
    # each built a dict and a str, and two more by-name reads, per layer call.
    if xcnn_get(context, XK_NDARRAY) is not None:
        var connected = xcnn_get(context, XK_ADMISSION)
        if connected is None or xcnn_get(connected, XK_NDARRAY) is not None:
            return
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
            # import-time: one-time cold connection of optional NumPy (xcnn_numpy_import_tried guards every later call)
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
    # fx-rows (2026-11-18): every layer call ran this. The mode text is now
    # decoded once (it was decoded three to four times), the connected
    # entries' type is an exact-type test before the `isinstance` call, and
    # the recorded profile and mode are compared by identity before any rich
    # compare or decode. Same refusals, same writes, same result. Timing:
    # global-avgpool / batchnorm2d infer (board), a Conv2d(3, 16, 3) on
    # 32 x 32 x 8 random input (non-board).
    # fx-cnn-layers (2026-11-18): the fields, the context and the connected
    # entries are read through held names, the exact-type tests use the held
    # builtin types, and the plain-value scan runs only for a value it could
    # refuse (it returns at once for None and an exact str). Same refusals in
    # the same order, same writes.
    var mode = xcnn_get(fields, XK_NUMERIC_MODE)
    if mode is not None and not xcnn_exact(mode, XT_STR):
        neural_plain_value(mode, 0, context)
    if mode is None or not Bool(py=mode):
        mode = xcnn_item(xcnn_item(context, XK_BACKEND_FIELDS), XK_DEFAULT_MODE)
    if not xcnn_exact(mode, XT_STR):
        raise Error("numeric_mode must be fast, identical or None")
    # p3-cnn-sequence-tsa: an exact "identical" / "fast" is used as it is
    var text = String(py=mode)
    var key = mode
    if text != "identical" and text != "fast":
        # a new String, then assigned (never a slice of `text` assigned back to it)
        var lowered = String(text.strip().lower())
        text = lowered
        key = PythonObject(text)
    if text != "identical" and text != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = py_item(xcnn_item(context, XK_BINDINGS), key)
    if xcnn_exact(entries, XT_STR):
        raise Error(String(py=entries))
    if not xcnn_exact(entries, XT_DICT):
        var b = xcnn_builtins()
        if Bool(py=py_call(b.isinstance, entries, b.str)):
            raise Error(String(py=entries))
    if text == "identical":
        # API-shell metadata only: the arithmetic version the binding names,
        # kept on the model so state persistence cannot switch versions
        var current = xcnn_item(entries, XK_PROFILE)
        var previous = xcnn_get(fields, XK_PROFILE_REC)
        if previous is not None and not xcnn_exact(previous, XT_STR):
            neural_plain_value(previous, 0, context)
        if (previous is not None and current is not None and previous is not current
                and not Bool(py=previous == current)):
            raise Error("mojolearn: CNN numerical profile differs from this model's recorded "
                        + "profile; use a matching binding or a fresh model for the new version")
        if previous is not None and current is None:
            raise Error("mojolearn: CNN numerical profile differs from this model's recorded "
                        + "profile; use a matching binding or a fresh model for the new version")
        if current is not None and previous is None:
            # equal to `previous` otherwise (checked above): written once
            xcnn_put(fields, XK_PROFILE_REC, current)
    var recorded = xcnn_get(fields, XK_MODE_REC)
    if recorded is not key and (recorded is None or not xcnn_exact(recorded, XT_STR) or String(py=recorded) != text):
        xcnn_put(fields, XK_MODE_REC, key)
    return entries


def xcnn_key(entries: PythonObject) raises -> XcnnEntryKey:
    """The native table key of the layer's connected x_cnn image (the bridge
    records it at import as `native_key`): the doors call that image's
    published entry (core/xcnn_entry_table.mojo), not its Python entries."""
    var key = xcnn_get(entries, XK_NATIVE_KEY)
    if key is None:
        raise Error("ImportError: mojolearn: the x_cnn binding publishes no native entry table")
    return XcnnEntryKey(Int(py=xcnn_at(key, 0)), Int(py=xcnn_at(key, 1)))


def xcnn_ref(entries: PythonObject) raises -> EntryRef:
    """UNPACK the layer's native table key and resolve the image's entry once
    per door operation (lane entry-table-zero-alloc): every RUN call of that
    operation calls through the returned ref; the table refusals raise here."""
    return xcnn_entry_ref(xcnn_key(entries))


# perf-audit-3 (pattern 14): the unused `xcnn_call_f` and `xcnn_route` (a
# second, eager-reading route beside the per-operation `xcnn_ref`; no door
# called them) are deleted. `XcnnRoute` stays in x_cnn/door_run.mojo for its
# native test suite.


def xcnn_call(xc: EntryRef, op: Int, addrs: List[Int], ints: List[Int]) raises -> Int:
    """One RUN call (`x_cnn/door_run.mojo` `xcnn_run_call`) through the
    resolved entry, with the GIL released. Plain words only: the caller
    converted every address and handle and keeps the buffers alive. An image
    without the operation refuses with the text a missing Python entry raised
    ("the CNN binding has no x_cnn_<name>")."""
    var result = 0
    var py = Python()
    with GILReleased(py):
        result = xcnn_run_call(xc, op, addrs, ints)
    return result


def xcnn_word(value: PythonObject) raises -> Int:
    """A handle or address word held in a Python int (UNPACK)."""
    return Int(py=value)


def xcnn_ptr(value: PythonObject) raises -> Int:
    """The admitted buffer address of a host array (UNPACK)."""
    admit_native_buffer_exporter(value)
    return native_buffer_address(value)


def xcnn_mixed(entries: PythonObject) raises -> Bool:
    """Whether the binding has the mixed-residency entries (`x_cnn_*_m`)."""
    return Bool(py=xcnn_item(entries, XK_MIXED))


def xcnn_idn(entries: PythonObject, bit: Int) raises -> Bool:
    return (Int(py=xcnn_item(entries, XK_IDN)) & bit) != 0


def xcnn_idn2(entries: PythonObject, bit: Int) raises -> Bool:
    return (Int(py=xcnn_item(entries, XK_IDN2)) & bit) != 0


def xcnn_device_io_off(entries: PythonObject, context: PythonObject) raises -> Bool:
    """The resident I/O route is off: the environment switches read at
    import, or a `-D MOJOLEARN_IDN_ALL_OFF` build of the binding."""
    return Bool(py=xcnn_item(context, XK_DEVICE_IO_OFF)) or xcnn_idn2(entries, XCNN_F2_ALL_OFF)


def xcnn_resident(entries: PythonObject, context: PythonObject) raises -> Bool:
    """Whether the binding runs the G3 resident glue."""
    return xcnn_mixed(entries) and xcnn_idn(entries, XCNN_F_RESIDENT) and not xcnn_device_io_off(entries, context)


# ---------------------------------------------------------------- shapes

comptime _XCNN_SHAPE_SLOT = "xcnn_shape_descriptor_v1"


def xcnn_shape(value: PythonObject) raises -> List[Int]:
    """Native shape metadata, with subclass/descriptor hooks refused first."""
    var cls = py_type(value)
    # fx-rows (2026-11-18): the descriptor verdict (a member or getset
    # descriptor owned by exactly this class) is kept per class under the
    # type watcher (`class_record`, as `xcnn_tensor_layout` keeps its
    # verdict): any edit of the class or an ancestor misses and re-runs the
    # full check. It loaded the `types` module, two descriptor types and
    # `__objclass__` on every shape read (several per layer call).
    # fx-cnn-layers (2026-11-18): with the verdict kept, the class's own
    # `shape` entry IS that admitted data descriptor, so the generic
    # attribute read (`PyObject_GenericGetAttr`: the type's MRO lookup, then
    # the data descriptor's getter) calls the same getter with the same
    # arguments as `descriptor.__get__(value, cls)` did. The class namespace
    # (a `type.__dict__` descriptor call and its key-scan record), the
    # by-name `shape` lookup, the bound `__get__` and its call envelope ran
    # per shape read although the verdict already covered them; they now run
    # at a miss only. The tuple is read with borrowed item reads.
    var mods = NativeModules()
    var shape = PythonObject(None)
    if class_record(mods, _XCNN_SHAPE_SLOT, cls) is not None:
        var g = _xcnn_names()
        var raw = cpython_table()[].PyObject_GenericGetAttr(value._obj_ptr, g[].keys[XK_SHAPE]._obj_ptr)
        if not raw:
            raise Python().cpython().unsafe_get_error()
        shape = PythonObject(from_owned=raw)
    else:
        var namespace = _class_fields(cls)
        var descriptor = py_get(namespace, "shape")
        var types = native_loaded_module("types")
        if (py_type(descriptor) is not types.MemberDescriptorType
                and py_type(descriptor) is not py_type(xcnn_builtins().type.__dict__["__dict__"])):
            raise Error("TypeError: CNN shape requires native storage metadata")
        if descriptor.__objclass__ is not cls:
            raise Error("TypeError: CNN shape refuses subclasses")
        class_admit(mods, _XCNN_SHAPE_SLOT, cls, PythonObject(True))
        shape = py_call(descriptor.__get__, value, cls)
    if not xcnn_exact(shape, XT_TUPLE):
        raise Error("TypeError: CNN shape requires an exact tuple")
    var out = List[Int]()
    var t = cpython_table()
    var count = t[].PyTuple_Size(shape._obj_ptr)
    for i in range(count):  # driver-loop: snapshot native shape words
        var item = PythonObject(from_borrowed=t[].PyTuple_GetItem(shape._obj_ptr, i))
        if not xcnn_exact(item, XT_INT) or Int(py=item) < 0:
            raise Error("TypeError: CNN dimensions must be nonnegative native integers")
        out.append(Int(py=item))
    _ = shape  # the borrowed items above are the tuple's: keep it past the last read
    return out^


def xcnn_tuple(shape: List[Int]) raises -> PythonObject:
    """`tuple(shape)` of Python ints, built through the C API (fx-rows,
    2026-11-18; the `_admission_int_tuple` idiom of
    bindings/labels_admission_unpack.mojo): it built a list, looked up its
    `append`, made one Python call per word, then called `tuple`."""
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(shape))
    if not envelope:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=envelope)
    var t = cpython_table()
    for i in range(len(shape)):  # driver-loop: over shape words
        var value = t[].PyLong_FromLongLong(Int64(shape[i]))
        if not value:
            raise cpy.unsafe_get_error()
        _ = cpy.PyTuple_SetItem(envelope, i, value)
    return result


def xcnn_is_t(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(value, DeviceTensor)`."""
    if py_type(value) is not xcnn_item(context, XK_TENSOR_CLASS):
        return False
    xcnn_tensor_layout(context)
    return True


comptime _XCNN_TENSOR_LAYOUT_SLOT = "xcnn_tensor_layout_v1"


def xcnn_tensor_layout(context: PythonObject) raises:
    var cls = xcnn_item(context, XK_TENSOR_CLASS)
    # p3-cnn-sequence-tsa: the verdict is kept per class under a type watcher
    # (`class_record`; any edit of the class or an ancestor misses), so the
    # MRO / namespace / slot walk runs once, not 3-8 times per layer call.
    # Python 3.10/3.11 key the record on the class's version tag
    # (bindings/py_version_tags.mojo).
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


def _xcnn_slot_name(name: String) raises -> Int:
    """The held-name index of a tensor storage slot."""
    if name == "h":
        return XK_H
    if name == "shape":
        return XK_SHAPE
    if name == "b":
        return XK_B
    if name == "_host":
        return XK_HOST
    if name == "_base":
        return XK_BASE
    raise Error("TypeError: CNN tensor has no storage slot " + name)


def _xcnn_slot_raw(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """A tensor storage slot read after this operation ran
    `xcnn_tensor_layout` (which already compared every slot descriptor with
    its import-time identity): the exact-class check and the member
    descriptor read only. No Python code runs between that check and this
    read, so the class and its descriptors cannot have changed."""
    # fx-cnn-layers (2026-11-18): the layout check proved the class's own
    # entry for each slot name is the import-time member descriptor and that
    # no ancestor overrides attribute access, so the generic attribute read
    # (`PyObject_GenericGetAttr`: MRO lookup of the held name, then that data
    # descriptor's getter) is the same getter call `slots[name].__get__(value,
    # cls)` made. Per read it took two context reads by a built str, a third
    # for the slot, a bound `__get__` object and a call envelope (about 4k
    # instructions; a resident pool forward did eight to ten of them).
    var g = _xcnn_names()
    if py_type(value) is not xcnn_item(context, XK_TENSOR_CLASS):
        raise Error("TypeError: CNN tensor requires the connected native storage class")
    var raw = cpython_table()[].PyObject_GenericGetAttr(value._obj_ptr, g[].keys[_xcnn_slot_name(name)]._obj_ptr)
    if not raw:
        raise Python().cpython().unsafe_get_error()
    return PythonObject(from_owned=raw)


def _xcnn_set_raw(value: PythonObject, name: String, item: PythonObject, context: PythonObject) raises:
    """A tensor storage slot write after `xcnn_tensor_layout` (see
    `_xcnn_slot_raw`): `PyObject_GenericSetAttr` finds the same member
    descriptor on the class and calls its setter, as `slots[name].__set__(
    value, item)` did."""
    var g = _xcnn_names()
    if py_type(value) is not xcnn_item(context, XK_TENSOR_CLASS):
        raise Error("TypeError: CNN tensor requires the connected native storage class")
    if cpython_table()[].PyObject_GenericSetAttr(value._obj_ptr, g[].keys[_xcnn_slot_name(name)]._obj_ptr,
                                                 item._obj_ptr) != 0:
        raise Python().cpython().unsafe_get_error()


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
    _ = _xcnn_tensor_admit_raw(value, context)


def _xcnn_tensor_admit_raw(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`xcnn_tensor_admit` after this operation's `xcnn_tensor_layout`;
    returns the admitted handle slot (None or a positive int) so a caller
    does not read it again."""
    # fx-cnn-layers (2026-11-18): exact-type tests against the held builtin
    # types and borrowed tuple item reads; it read `b.int` by name and built
    # two index objects per dimension.
    var shape = _xcnn_slot_raw(value, "shape", context)
    if not xcnn_exact(shape, XT_TUPLE):
        raise Error("TypeError: CNN tensor shape requires an exact tuple")
    var t = cpython_table()
    var count = t[].PyTuple_Size(shape._obj_ptr)
    for i in range(count):  # driver-loop: native tensor dimensions before arithmetic
        var item = PythonObject(from_borrowed=t[].PyTuple_GetItem(shape._obj_ptr, i))
        if not xcnn_exact(item, XT_INT) or Int(py=item) < 0:
            raise Error("TypeError: CNN tensor dimensions require nonnegative integers")
    _ = shape  # the borrowed items above are the tuple's: keep it past the last read
    var handle = _xcnn_slot_raw(value, "h", context)
    if handle is not None and (not xcnn_exact(handle, XT_INT) or Int(py=handle) < 1):
        raise Error("TypeError: CNN tensor handle requires a positive native integer")
    _ = xcnn_module_entries(_xcnn_slot_raw(value, "b", context), context)
    if handle is None:
        sequential_admit_value(_xcnn_slot_raw(value, "_host", context), context)
    return handle


def xcnn_on_dev(entries: PythonObject, value: PythonObject, context: PythonObject) raises -> Bool:
    """`_on_dev(b, value)`: a tensor resident on this mixed binding."""
    if not xcnn_mixed(entries) or not xcnn_is_t(value, context):
        return False
    # `xcnn_is_t` ran the layout check; the admission and slot reads reuse it.
    # The admitted handle is returned, not read a second time (fx-rows).
    var handle = _xcnn_tensor_admit_raw(value, context)
    return handle is not None and _xcnn_slot_raw(value, "b", context) is xcnn_item(entries, XK_MODULE)


def xcnn_dev_word(entries: PythonObject, value: PythonObject, context: PythonObject) raises -> Int:
    """`xcnn_on_dev` for a `value` this operation already found to be a
    tensor (`xcnn_is_t` returned True for it and no Python code ran since):
    its resident handle word when it is admitted and resident on this mixed
    binding, else 0 (an admitted handle is 1 or more). The same admission
    and the same verdict; the caller gets the word it would read from
    `value.h` next, and the class test and layout check `xcnn_is_t` already
    made for this value are not made a second time (fx-cnn-layers,
    2026-11-18)."""
    if not xcnn_mixed(entries):
        return 0
    var handle = _xcnn_tensor_admit_raw(value, context)
    if handle is None or _xcnn_slot_raw(value, "b", context) is not xcnn_item(entries, XK_MODULE):
        return 0
    return Int(py=handle)


def xcnn_tensor(module: PythonObject, shape: PythonObject, h: PythonObject, host: PythonObject, base: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`DeviceTensor(binding, shape, h, host, base)`, its slots set here."""
    xcnn_tensor_layout(context)
    return _xcnn_tensor_raw(module, shape, h, host, base, context)


def _xcnn_tensor_raw(module: PythonObject, shape: PythonObject, h: PythonObject, host: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`xcnn_tensor` after this operation's `xcnn_tensor_layout`: the five
    slots written without re-walking the class layout once per slot."""
    # fx-cnn-layers (2026-11-18): `object.__new__(cls)` is the class's
    # ordinary heap allocation (`native_storage_allocate`, the form
    # bindings/native_buffer_owner_pack.mojo uses for its own class); it was
    # found through `getattr(object, "__new__")` and called through two
    # envelopes per output tensor. No Python `__new__` or `__init__` runs in
    # either form; the five slots are written below.
    var t = native_storage_allocate(xcnn_item(context, XK_TENSOR_CLASS))
    _xcnn_set_raw(t, "b", module, context)
    _xcnn_set_raw(t, "shape", shape, context)
    _xcnn_set_raw(t, "h", h, context)
    _xcnn_set_raw(t, "_host", host, context)
    _xcnn_set_raw(t, "_base", base, context)
    return t


def xcnn_new(entries: PythonObject, shape: List[Int], nofill: Bool, context: PythonObject) raises -> PythonObject:
    """`DeviceTensor._new` (or `_new_out` with `nofill`: no +0.0 fill, on a
    binding that has `x_cnn_res_alloc_nofill`)."""
    # The host image runs RES_ALLOC_NOFILL as its zero-filled allocation
    # (x_cnn/host/entry_runs_host.mojo), the entry the Python lookup fell back to.
    var xc = xcnn_ref(entries)
    var h = xcnn_call(xc, XC_RES_ALLOC_NOFILL if nofill else XC_RES_ALLOC, List[Int](), [max(xcnn_size(shape), 1)])
    return xcnn_tensor(xcnn_item(entries, XK_MODULE), xcnn_tuple(shape), PythonObject(h), PythonObject(None),
                       PythonObject(None), context)


def xcnn_alloc_op(entries: PythonObject, context: PythonObject) raises -> Int:
    """The allocation operation `xcnn_out_new` runs for an output whose
    every word the entry stores: unfilled under the resident glue, the
    filled allocation otherwise. For a door that allocates the output in
    the same GIL window as the entry that fills it (`xcnn_adopt`)."""
    return XC_RES_ALLOC_NOFILL if xcnn_resident(entries, context) else XC_RES_ALLOC


def xcnn_adopt(entries: PythonObject, shape: List[Int], h: Int, context: PythonObject) raises -> PythonObject:
    """The `DeviceTensor` of `shape` that owns resident array `h`, which the
    caller just allocated on this binding (`xcnn_alloc_op`) in the GIL
    window of the entry that filled it. The object `xcnn_new` returns for
    the same handle. The caller ran `xcnn_tensor_layout` in this operation
    (it admitted a tensor input) and no Python code ran since."""
    return _xcnn_tensor_raw(xcnn_item(entries, XK_MODULE), xcnn_tuple(shape), PythonObject(h), PythonObject(None),
                            PythonObject(None), context)


def xcnn_out_new(entries: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`_out_new`: a resident output whose every word the entry stores."""
    return xcnn_new(entries, shape, xcnn_resident(entries, context), context)


def xcnn_reshape_t(value: PythonObject, shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`DeviceTensor.reshape(shape)`: the same resident words under another
    shape, the view keeping `value` alive."""
    xcnn_run_reshape_check(xcnn_shape(value), shape)
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
    var a = admit_nd_f32(sequential_input(array, context), "x", False, context)
    if not xcnn_mixed(entries):
        return xcnn_tensor(entries["module"], a.shape, PythonObject(None), a, PythonObject(None), context)
    var shape = xcnn_shape(a)
    var t = xcnn_new(entries, shape, False, context)
    var size = xcnn_size(shape)
    var xc = xcnn_ref(entries)
    if size:
        _ = xcnn_call(xc, XC_RES_UPLOAD, [xcnn_word(t.h), xcnn_ptr(a)], [size])
        _ = a  # keep the Python object alive through the native call above
    _ = a^
    return t


def xcnn_module_entries(module: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected entries whose module is `module`."""
    # fx-cnn-layers (2026-11-18): the two mode names are held (it built a
    # tuple of two fresh strs per call), and connected entries (an exact
    # dict) are told from a connection error text (a str) by an exact-type
    # test before the `isinstance` call, as `xcnn_entries` does. Every tensor
    # admission and every resident-owner admission runs this.
    var table = xcnn_item(context, XK_BINDINGS)
    for i in range(2):  # driver-loop: over the two numeric modes
        var entries = xcnn_get(table, XK_IDENTICAL + i)
        if entries is None:
            continue
        if not xcnn_exact(entries, XT_DICT):
            var b = xcnn_builtins()
            if Bool(py=py_call(b.isinstance, entries, b.str)):
                continue
        if xcnn_item(entries, XK_MODULE) is module:
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
        xcnn_run_copy_f32(native_buffer_address(out), native_buffer_address(source), size)
        _ = source  # keep the Python object alive through the copy above
        return out
    var out = xcnn_host_f32(shape, context)
    var size = xcnn_size(shape)
    if size:
        var entries = xcnn_module_entries(_xcnn_slot_raw(value, "b", context), context)
        var xc = xcnn_ref(entries)
        _ = xcnn_call(xc, XC_RES_DOWNLOAD, [xcnn_word(handle), xcnn_ptr(out)], [size])
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
    xcnn_run_copy_f32(Int(py=xcnn_addr(out)), Int(py=xcnn_addr(value)), count)
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
    if py_type(dev) is not xcnn_item(context, XK_DEV_CLASS):
        raise Error("TypeError: CNN resident state requires its connected native owner")
    var fields = _fields(dev)
    _ = xcnn_module_entries(xcnn_item(fields, XK_B), context)
    var table = xcnn_item(fields, XK_H)
    if not xcnn_exact(table, XT_DICT):
        raise Error("TypeError: CNN resident handles require an exact dictionary")
    # p3-cnn-sequence-tsa: a handle table that passed the full check below and
    # has not changed since (a dictionary watcher sees every insert, replace,
    # delete and deallocation) is not walked again; a BasicBlock forward read
    # it about ten times per call. Python 3.10/3.11 keep the walk.
    if dict_keys_admitted(table):
        return fields
    var b = xcnn_builtins()
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


def _xcnn_dev_admitted(fields: PythonObject, entries: PythonObject, context: PythonObject,
                       mut table: PythonObject) raises -> PythonObject:
    """`xcnn_dev` on the layer's instance fields (`_fields(layer)`, read by
    the caller); `table` receives the owner's handle table, admitted here
    (an existing owner's by `xcnn_dev_fields`, a fresh owner's empty one)."""
    var dev = xcnn_get(fields, XK_DEV)
    if dev is not None:
        var state = xcnn_dev_fields(dev, context)
        if xcnn_item(state, XK_B) is xcnn_item(entries, XK_MODULE):
            table = xcnn_item(state, XK_H)
            return dev
        var b = xcnn_builtins()
        # p3-cnn-sequence-tsa (hazard C): fields that name a handle of the
        # old owner are settled before its handles are freed (pending weight
        # gradients and a dropout mask downloaded to the host, the rest
        # cleared), so no later read uses a freed or reused handle number.
        _xcnn_settle_old_dev(fields, dev, context)
        # `_Dev.free()`: every handle of the old binding released, the table emptied
        var old = xcnn_dev_fields(dev, context)
        var old_table = old["h"]
        var owner = xcnn_module_entries(old["b"], context)
        var xco = xcnn_ref(owner)
        var handles = py_call(b.list, py_call(py_call(b.getattr, old_table, "values")))
        py_setitem(old, "h", Python.dict())
        for i in range(len(handles)):  # driver-loop: over the layer's resident array table
            _ = xcnn_call(xco, XC_RES_FREE, [xcnn_word(handles[i][0])], List[Int]())
    var bi = xcnn_builtins()
    dev = py_call(py_call(bi.getattr, bi.object, "__new__"), context["xcnn_dev_class"])
    var fresh = Python.dict()
    var made = _fields(dev)
    py_setitem(made, "b", xcnn_item(entries, XK_MODULE))
    py_setitem(made, "h", fresh)
    py_setitem(fields, "_dev", dev)
    table = fresh
    return dev


def xcnn_dev(layer: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_dev_of(layer, b)`: the layer's `_Dev` on this binding, a new one
    (the old one's arrays freed) when the binding changed."""
    var table = PythonObject(None)
    return _xcnn_dev_admitted(_fields(layer), entries, context, table)


def xcnn_dev_get_f(fields: PythonObject, entries: PythonObject, name: String, count: Int,
                   context: PythonObject) raises -> PythonObject:
    """`xcnn_dev_get` for a caller that already holds the layer's instance
    fields (`_fields(layer)`, read in this operation): the layer's resident
    array `name` of `count` words, reallocated only when its size changes."""
    # fx-cnn-layers (2026-11-18): one admission of the owner per call. It
    # read the layer's instance dictionary again, admitted the owner, then
    # read the owner's instance dictionary a second time for the table the
    # admission had just checked: three instance-dictionary admissions (each
    # a class record, a key scan and a module stamp) for one handle. Max
    # pools, dropout, batch norm and the graph layers call this one to eight
    # times per forward.
    var n = max(count, 1)
    var table = PythonObject(None)
    _ = _xcnn_dev_admitted(fields, entries, context, table)
    var key = PythonObject(name)
    var cur = py_get(table, key)
    if cur is not None and Int(py=xcnn_at(cur, 1)) == n:
        return xcnn_at(cur, 0)
    var xc = xcnn_ref(entries)
    if cur is not None:
        _ = xcnn_call(xc, XC_RES_FREE, [xcnn_word(xcnn_at(cur, 0))], List[Int]())
        _ = py_call(py_call(xcnn_builtins().getattr, table, "pop"), key)
    var handle = PythonObject(xcnn_call(xc, XC_RES_ALLOC, List[Int](), [n]))
    py_setitem(table, key, Python.tuple(handle, n))
    return handle


def xcnn_dev_get(layer: PythonObject, entries: PythonObject, name: String, count: Int,
                 context: PythonObject) raises -> PythonObject:
    """`_Dev.get(name, n)`: the layer's resident array `name` of `n` words,
    reallocated only when its size changes."""
    return xcnn_dev_get_f(_fields(layer), entries, name, count, context)


def xcnn_dev_upload(layer: PythonObject, entries: PythonObject, name: String, array: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`_Dev.upload(name, a)`: resident array `name` holding the 4-byte words
    of host array `a`."""
    var size = Int(py=array.size)
    var h = xcnn_dev_get(layer, entries, name, size, context)
    var xc = xcnn_ref(entries)
    _ = xcnn_call(xc, XC_RES_UPLOAD, [xcnn_word(h), xcnn_ptr(array)], [size])
    _ = array  # keep the Python object alive through the native call above
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
        var xco = xcnn_ref(owner)
        _ = xcnn_call(xco, XC_RES_DOWNLOAD, [xcnn_word(h), xcnn_ptr(out)], [count])
        _ = out  # keep the Python object alive through the native call above
    return out


def xcnn_slice_f32(source: PythonObject, offset: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """`a[offset:offset + count].copy()` of a C-ordered float32 array."""
    var out = xcnn_host_f32([count], context)
    if count:
        xcnn_run_copy_f32(Int(py=xcnn_addr(out)), Int(py=xcnn_addr(source)) + 4 * offset, count)
        _ = out  # keep the Python object alive through the native call above
    return out


def xcnn_put_f32(dest: PythonObject, offset: Int, source: PythonObject, count: Int, name: String,
                 context: PythonObject) raises:
    """`dest[offset:offset + count] = source` (float32 words; `source` cast
    as NumPy assignment casts it), refused unless it holds `count` values."""
    var a = admit_nd_f32(sequential_input(source, context), name, False, context)
    if Int(py=a.size) != count:
        raise Error("ValueError: could not broadcast " + name + " into shape (" + String(count) + ",)")
    if count:
        xcnn_run_copy_f32(Int(py=xcnn_addr(dest)) + 4 * offset, Int(py=xcnn_addr(a)), count)
        _ = a  # keep the Python object alive through the native call above
