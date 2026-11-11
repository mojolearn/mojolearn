# SPDX-License-Identifier: Apache-2.0
"""CPython container allocation used by the native generation owner.

Only C builtins are invoked. Array.__init__, reshape, __buffer__, and package
helpers never run. A foreign buffer is packed inside the operation's boundary.

Every call with an argument, attribute assignment and type query goes through
bindings/py_balanced.mojo: Mojo 1.0's PythonObject call and `__setattr__` keep
one extra reference on each argument or value (docs/MODULAR_ASKS.md), which
kept every array built here, and the caller's buffers, alive for ever.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_class_admission import class_record, class_admit
from bindings.py_balanced import py_call, py_setattr, py_type, py_setitem, py_contains
from bindings.native_array_snapshot import native_array_slots, native_array_snapshot
from bindings.native_array_storage import native_array_zeros
from bindings.native_buffer_address import native_buffer_address
from bindings.p3nc_fitted_freeze import fitted_freeze_foreign
from core.weight_epoch import weight_epoch_bump
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter
from core.generation_buffers_impl import generation_ids_into
from std.memory import unsafe_memcpy
from core.shared_slot import SharedSlot


struct GenerationBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var state_cls: PythonObject
    var layer_classes: PythonObject
    var stream_cls: PythonObject
    # The Array context dictionary, built once per connected Array class
    # (formerly a new dictionary per call).
    var array_context: PythonObject
    # The class tuple admitted last and its layer-registry size: a repeat
    # call with the same tuple object and an unchanged registry size skips
    # the full re-admission (generator draws call this per draw).
    var admitted_classes: PythonObject
    var admitted_layer_count: Int

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.state_cls = PythonObject(None)
        self.layer_classes = PythonObject(None)
        self.stream_cls = PythonObject(None)
        self.array_context = PythonObject(None)
        self.admitted_classes = PythonObject(None)
        self.admitted_layer_count = -1


comptime GENERATION_BOUNDARY = SharedSlot[GenerationBoundary, "MojolearnCausalGenerationBoundaryV1"]


def generation_initialize_binding(classes: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if (g[].admitted_classes is not None and classes is g[].admitted_classes
            and len(classes[2]) == g[].admitted_layer_count):
        return PythonObject(None)
    g[].bi = native_loaded_module("builtins")
    if py_type(classes) is not g[].bi.tuple or len(classes) != 4:
        raise Error("TypeError: generation requires its import-connected class tuple")
    for index in [0, 1, 3]:  # small-loop(generation owner class identities)
        if py_type(classes[index]) is not g[].bi.type:
            raise Error("TypeError: generation owner classes require native type metadata")
    if g[].array_cls is not None and g[].array_cls is not classes[0]:
        raise Error("TypeError: generation Array identity cannot be replaced")
    if ((g[].state_cls is not None and g[].state_cls is not classes[1])
            or (g[].stream_cls is not None and g[].stream_cls is not classes[3])):
        raise Error("TypeError: generation state and stream identities cannot be replaced")
    if py_type(classes[2]) is not g[].bi.dict:
        raise Error("TypeError: generation layer classes require an exact dictionary")
    var layer_names = py_call(g[].bi.list, classes[2].keys())
    for i in range(len(layer_names)):  # small-loop(import-connected layer class registry)
        if py_type(layer_names[i]) is not g[].bi.str:
            raise Error("TypeError: generation layer class names require native strings")
    var layer_classes = py_call(g[].bi.dict, classes[2])
    if g[].layer_classes is not None and len(g[].layer_classes) != len(layer_names):
        raise Error("TypeError: generation layer class registry cannot be resized")
    for i in range(len(layer_names)):  # small-loop(capture generation layer class identities)
        if py_type(layer_classes[layer_names[i]]) is not g[].bi.type:
            raise Error("TypeError: generation layer classes require native type metadata")
        if g[].layer_classes is not None:
            if (not py_contains(g[].layer_classes, layer_names[i])
                    or g[].layer_classes[layer_names[i]] is not layer_classes[layer_names[i]]):
                raise Error("TypeError: generation layer class identities cannot be replaced")
    g[].arrays = native_loaded_module("array").array
    g[].array_cls = classes[0]
    g[].state_cls = classes[1]
    g[].layer_classes = layer_classes
    g[].stream_cls = classes[3]
    g[].admitted_classes = classes
    g[].admitted_layer_count = len(layer_names)
    return PythonObject(None)


def generation_array_context() raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var cls = g[].array_cls
    if cls is None:
        raise Error("RuntimeError: generation requires its import-time Array connection")
    # The Array class identity can never be replaced after connection
    # (generation_initialize_binding refuses it), so one context serves
    # every call.
    if g[].array_context is None or g[].array_context["array_class"] is not cls:
        g[].array_context = Python.dict(array_class=cls)
    return g[].array_context


comptime _GENERATION_SLOTS = "generation_array_slots_v1"


def _generation_slots() raises -> PythonObject:
    # The admitted Array storage descriptors, kept per class epoch (type
    # watcher): a class change re-runs the full descriptor admission. The
    # descriptors reference the Array class, which the generation boundary
    # already holds for the process lifetime.
    var mods = NativeModules()
    var cls = GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls
    var cached = class_record(mods, _GENERATION_SLOTS, cls)
    if cached is not None:
        return cached
    var slots = native_array_slots(mods, generation_array_context())
    class_admit(mods, _GENERATION_SLOTS, cls, slots)
    return slots


def generation_array_snapshot(value: PythonObject) raises -> PythonObject:
    return native_array_snapshot(value, generation_array_context(), True)


def generation_buffer_view(value: PythonObject) raises -> PythonObject:
    # A fresh native memoryview owns the admitted export and freezes metadata.
    # Releasing a caller's original memoryview cannot invalidate this owner.
    admit_native_buffer_exporter(value)
    return py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].bi.memoryview, value)


def py_ints(values: List[Int]) raises -> PythonObject:
    var out = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.list()
    var append = out.append
    for i in range(len(values)):  # small-loop(argument words handed to a binding entry)
        _ = py_call(append, values[i])
    return out


def generation_array(dims: List[Int], code: String = "f") raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].array_cls is None:
        raise Error("RuntimeError: generation requires its import-time Array connection")
    var slots = _generation_slots()
    if code != "f" and code != "i" and code != "b":
        raise Error("generation: unsupported native allocation dtype")
    var n = 1
    var itemsize = 1 if code == "b" else 4
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        if dims[i] < 0 or (dims[i] != 0 and n > 2147483647 // dims[i]):
            raise Error("generation: invalid shape or signed 32-bit kernel indexing overflow")
        n *= dims[i]
    var store = native_array_zeros(code, n)
    var result = py_call(g[].bi.object.__new__, g[].array_cls)
    # Strides computed natively (row-major), then one tuple; each slot is set
    # directly (no intermediate dictionary or key list), in the former order.
    var stride_words = List[Int](length=len(dims), fill=0)
    var stride = itemsize
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        var axis = len(dims) - i - 1
        stride_words[axis] = stride
        stride *= dims[axis]
    var shape = py_call(g[].bi.tuple, py_ints(dims))
    var strides = py_call(g[].bi.tuple, py_ints(stride_words))
    _generation_slot_set(slots, result, "_store", store)
    _generation_slot_set(slots, result, "_base", PythonObject(None))
    _generation_slot_set(slots, result, "_pin", PythonObject(None))
    _generation_slot_set(slots, result, "_mv", py_call(g[].bi.memoryview, store))
    _generation_slot_set(slots, result, "_addr", store.buffer_info()[0])
    _generation_slot_set(slots, result, "_readonly", PythonObject(False))
    _generation_slot_set(slots, result, "shape", shape)
    _generation_slot_set(slots, result, "dtype", PythonObject("<f4" if code == "f" else ("<i4" if code == "i" else "<i1")))
    _generation_slot_set(slots, result, "ndim", PythonObject(len(dims)))
    _generation_slot_set(slots, result, "size", PythonObject(n))
    _generation_slot_set(slots, result, "nbytes", PythonObject(n * itemsize))
    _generation_slot_set(slots, result, "itemsize", PythonObject(itemsize))
    _generation_slot_set(slots, result, "order", PythonObject("C"))
    _generation_slot_set(slots, result, "strides", strides)
    return result


def _generation_slot_set(slots: PythonObject, target: PythonObject, name: String, value: PythonObject) raises:
    py_call(slots[name].__set__, target, value)


def generation_address(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    if value is None:
        return 0
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        var captured = generation_array_snapshot(value)
        if String(captured.order) != "C":
            raise Error("generation: internal model and state arrays must be C-contiguous")
        keep.append(captured)
        return Int(py=captured._addr)
    var view = generation_buffer_view(value)
    if not Bool(py=view.c_contiguous):
        raise Error("generation: internal model and state buffers must be C-contiguous")
    # Copy read-only model weights to a native owner. Mutable state is created
    # as Array by this owner; supplied state arrays are validated separately.
    var format = String(view.format)
    if format != "f" and format != "i" and format != "b":
        raise Error("generation: unsupported native model buffer format " + format)
    return _generation_pack_foreign(view, format, keep)


def _generation_pack_foreign(view: PythonObject, format: String, mut keep: List[PythonObject]) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var packed = native_array_zeros(format, Int(py=view.nbytes) // Int(py=view.itemsize))
    var destination = py_call(py_call(g[].bi.memoryview, packed).cast, "B")
    py_setitem(destination, py_call(g[].bi.slice, PythonObject(None)), py_call(view.cast, "B"))
    keep.append(view)
    keep.append(packed)
    return Int(py=packed.buffer_info()[0])


def generation_weight_address(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    # Row N200 (foreign weights): the address of a model weight that every
    # kernel only reads. A C-contiguous foreign buffer (ndarray, array.array,
    # memoryview) aligned for its item width is read in place through the
    # private memoryview kept in `keep` (formerly copied whole into a new
    # native store on every call) once the read-only rule holds for it
    # (lesson 10): `fitted_freeze_foreign` clears an exact ndarray's
    # writeable flag; a buffer that cannot be frozen keeps the per-call
    # copy. Never use this for an output or a state
    # buffer: a kernel write would then reach the caller's buffer, where the
    # copy used to absorb it. Array weights take the snapshot path as before.
    #
    # Lane x45c (core/weight_epoch.mojo): every weight this door hands to a
    # kernel is read-only from here on, so a resident copy keyed by its
    # address stays valid until the weight epoch moves. The epoch moves when
    # a weight that was writeable until now is frozen (a new or reassigned
    # array, at its first use) and on every per-call copy of storage that
    # cannot be frozen (its address names new bytes each call). The resident
    # copies compare no weight bytes.
    if value is None:
        return 0
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        if not Bool(py=value._readonly):
            py_setattr(value, "_readonly", PythonObject(True))
            weight_epoch_bump()
        return generation_address(value, keep)
    var view = generation_buffer_view(value)
    if not Bool(py=view.c_contiguous):
        raise Error("generation: internal model and state buffers must be C-contiguous")
    var format = String(view.format)
    if format != "f" and format != "i" and format != "b":
        raise Error("generation: unsupported native model buffer format " + format)
    var itemsize = Int(py=view.itemsize)
    var address = native_buffer_address(view)
    var was_writeable = not Bool(py=view.readonly)
    if address != 0 and (itemsize == 1 or address % itemsize == 0) and fitted_freeze_foreign(value):
        if was_writeable:
            weight_epoch_bump()
        keep.append(view)
        return address
    weight_epoch_bump()
    return _generation_pack_foreign(view, format, keep)


def generation_ids(value: PythonObject, vocab: Int, step: Bool = False) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var shape = PythonObject(None)
    var view = PythonObject(None)
    var captured_owner = PythonObject(None)
    var column_major = False
    if py_type(value) is g[].array_cls:
        var captured = generation_array_snapshot(value)
        column_major = String(captured.order) == "F"
        shape = captured.shape
        view = captured._mv
        # Retain the captured base export alongside the typed view while IDs
        # are copied, including ownerless physical views used by Array.
        captured_owner = captured
    else:
        view = generation_buffer_view(value)
        shape = view.shape
    var rank = len(shape)
    if (not step and rank != 2) or (step and rank != 1 and rank != 2):
        raise Error("generation: ids must have shape (B, L), or (B,) for step")
    var b = Int(py=shape[0])
    var l = 1 if rank == 1 else Int(py=shape[1])
    if b < 1 or l < 1 or (step and l != 1):
        raise Error("generation: ids must be nonempty, with one token per row for step")
    var fmt = String(view.format)
    if fmt != "i" and fmt != "q" and fmt != "l" and fmt != "h" and fmt != "b" and fmt != "I" and fmt != "Q" and fmt != "L" and fmt != "H" and fmt != "B":
        raise Error("TypeError: generation ids must be integer (a native-endian integer buffer)")
    var out = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    if Bool(py=view.c_contiguous):
        # A contiguous buffer is read in place: its storage-order bytes are
        # what `tobytes("C")` copied into the staging store (twice, through
        # the bytes object and the slice assignment) before this one native
        # conversion, which now reads the caller's buffer directly.
        generation_ids_into(native_buffer_address(view), Int(py=view.itemsize), signed, b, l, column_major, vocab, dst)
        _ = view
        _ = captured_owner
        return out
    # Pack strided buffer inputs with the CPython buffer implementation; all
    # validation and conversion below is a Mojo loop over native storage.
    var packed = native_array_zeros(fmt, Int(py=view.nbytes) // Int(py=view.itemsize))
    var destination = py_call(py_call(g[].bi.memoryview, packed).cast, "B")
    py_setitem(destination, py_call(g[].bi.slice, PythonObject(None)), py_call(view.tobytes, "C"))
    generation_ids_into(Int(py=packed.buffer_info()[0]), Int(py=packed.itemsize), signed, b, l, column_major, vocab, dst)
    _ = packed^
    _ = captured_owner
    return out


def generation_copy(value: PythonObject) raises -> PythonObject:
    var captured = generation_array_snapshot(value)
    var dims = List[Int]()
    for i in range(len(captured.shape)):  # small-loop(array shape dimension words)
        dims.append(Int(py=captured.shape[i]))
    var code = "f" if String(captured.dtype) == "<f4" else "i"
    if String(captured.dtype) != "<f4" and String(captured.dtype) != "<i4":
        raise Error("TypeError: generation state copy requires float32 or int32 storage")
    var out = generation_array(dims, code)
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=captured._addr))
    unsafe_memcpy(dest=dst, src=src, count=Int(py=captured.nbytes))
    _ = captured
    return out


def generation_mutable(
    value: PythonObject, expected: List[Int], mut keep: List[PythonObject], mut copies: List[PythonObject],
    name: String = "",
) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var label = String("state buffer")
    if name != "":
        label = "state buffer " + name
    var view = PythonObject(None)
    var shape = PythonObject(None)
    if value is None:
        raise Error("CausalLM: " + label + " is missing; allocate_state() makes a complete state")
    if py_type(value) is g[].array_cls:
        var captured = generation_array_snapshot(value)
        if String(captured.dtype) != "<f4" or String(captured.order) != "C" or Bool(py=captured._readonly):
            raise Error("CausalLM: " + label + " must be writable C-contiguous float32 (it is updated in place)")
        view = captured._mv
        shape = captured.shape
        keep.append(captured)
    else:
        view = generation_buffer_view(value)
        shape = view.shape
        if Bool(py=view.readonly) or not Bool(py=view.c_contiguous) or String(view.format) != "f":
            raise Error("CausalLM: " + label + " must be writable C-contiguous float32 (it is updated in place)")
    var want = String("(")
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if i > 0:
            want += ", "
        want += String(expected[i])
    want += ")"
    if len(shape) != len(expected):
        raise Error("CausalLM: " + label + " has shape " + String(shape) + ", want " + want)
    var size = 1
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if Int(py=shape[i]) != expected[i]:
            raise Error("CausalLM: " + label + " has shape " + String(shape) + ", want " + want)
        size *= expected[i]
    if Int(py=view.nbytes) != size * 4:
        raise Error("CausalLM: " + label + " span differs from its shape")
    var address: Int
    if py_type(value) is g[].array_cls:
        address = Int(py=keep[len(keep) - 1]._addr)
    else:
        # The immutable internal view is the already-captured metadata owner.
        address = generation_address(view, keep)
        var pair = g[].bi.list()
        _ = py_call(pair.append, view)
        _ = py_call(pair.append, keep[len(keep) - 1])
        copies.append(pair)
    return address
