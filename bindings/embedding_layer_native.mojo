# SPDX-License-Identifier: Apache-2.0
"""The native owner of `mojolearn.Embedding`'s public operations
(python/mojolearn/embedding.py), reached through the optimizer native door
(`emblayer_*` operations, bindings/optimizer_native_router.mojo):

    forward / __call__   the ids admitted (integers only, any rank), the
                         layer's resident table token, one
                         `embedding_forward` (or `embedding_forward_dev` for
                         device=True) call, the result in the ids' shape
    load                 the saved table read natively (the NPZ reader of
                         bindings/classical_archive_read.mojo), the checks of
                         the constructor, the layer built without running
                         Python
and its lifetime (`embedding_layer_lifetime_c`, installed at import as the
class's `__setattr__` and `__del__` through the native attribute descriptor):
assigning `weight` and finalizing release the resident device table.

The same compiled entries with the same argument words as the Python bodies
it replaces, so no bit moves.
"""
from std.ffi import _Global
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_attr import py_attr
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_door import new_array, same_shape, base_format, py_ints, door_ready
from bindings.optimizer_device_native import shape_count, device_shape
from bindings.training_embedding_native import embedding_indices, embedding_leaves, embedding_leaf_store
from bindings.sequence_device_native import seq_device_make, sequence_device_operation
from bindings.sequence_optimizer_native import seq_device_get
from bindings.optimizer_admission import opt_value, opt_scalar, opt_guards, opt_exact, opt_numpy
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.neural_block_admission import neural_record_fields
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequential_inputs import sequential_input
from bindings.native_loaded_module import native_loaded_module
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from core.xseq_marshal import xseq_cast_nd
from core.training_embedding_inputs import embedding_pack, embedding_cast_refusal
from core.borrowed_buffer import BorrowedBufferLayout
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_FLOAT, CAST_KIND_BOOL
from bindings.classical_archive_read import classical_archive_read, _archive_string, _archive_exact
from bindings.optimizer_public_words import optimizer_python_error

comptime _NO_PADDING = -1
comptime _FORMAT = "mojolearn-embedding-1"


struct _EmbeddingTokens(Defaultable, Movable):
    """The resident-table token counter (one token per layer and per
    assignment of `weight`; never pickled or copied with the layer)."""
    var next: Int

    def __init__(out self):
        self.next = 1


comptime _EMBEDDING_TOKENS = _Global[
    StorageType=_EmbeddingTokens, name="MojoEmbeddingLayerTokens", init_fn=_EmbeddingTokens.__init__
]


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _repr(value: PythonObject) raises -> String:
    # Error formatting must not call caller-defined repr/str protocols.
    var b = _b()
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.int or cls is b.float or cls is b.bool:
        return String(py=py_call(b.repr, value))
    return String("<native value>")


def _class_name(cls: PythonObject) raises -> String:
    return String(py=py_call(_b().type.__dict__["__name__"].__get__, cls))


# ---------------------------------------------------------------- the route

def _want_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`getattr(self, "numeric_mode", None) or _backend.default_mode()`,
    checked as `_backend.binding` checks it."""
    var b = _b()
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var lowered = py_call(py_call(mode.strip).lower)
    if String(py=lowered) != "identical" and String(py=lowered) != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    return lowered


def _entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # An overload, not a `model = PythonObject(None)` default (a non-raising
    # caller compile-time evaluates a default; a PythonObject cannot be built then).
    return _entries(fields, context, PythonObject(None))


def _entries(fields: PythonObject, context: PythonObject, model: PythonObject) raises -> PythonObject:
    """`Embedding._extension()`: the embedding binding of the layer's mode,
    connected at import, with its compiled mode read back; for a host
    inference class (`_HostBound`, IDENTICAL only) the lookup host binding."""
    var b = _b()
    if model is not None and _class_flag(py_type(model), "_HOST_INFERENCE_ONLY", context):
        var saved = py_get(fields, "numeric_mode")
        if saved is not None and (py_type(saved) is not b.str or String(py=saved) != "identical"):
            raise Error("ValueError: host Embedding inference requires identical mode")
        var host = context["embedding_host_binding"]
        if py_type(host) is b.str:
            raise Error(String(py=host))
        return host
    var want = _want_mode(fields, context)
    var entries = context["embedding_bindings"][want]
    if py_type(entries) is _b().str:
        raise Error(String(py=entries))
    if py_contains(entries, "embedding_numeric_mode"):
        var code = Int(py=py_call_typed(entries["embedding_numeric_mode"]))
        var expected = 1 if String(py=want) == "identical" else 0
        if code != expected:
            raise Error("RuntimeError: mojolearn Embedding: numeric_mode=" + _repr(want) + " was requested but "
                        + String("the connected embedding binding")
                        + " reports another compile-time mode; rebuild it with bash bindings/build_embedding.sh")
    return entries


def _token(fields: PythonObject, entries: PythonObject, context: PythonObject) raises -> Int:
    """`Embedding._table_token`: 0 for a binding without the resident table
    (the host bindings), else this layer's token."""
    if not py_contains(entries, "embedding_table_release"):
        return 0
    var tok = py_get(fields, "_table_tok")
    var owner = py_get(fields, "_table_owner")
    if tok is not None and owner is not None and owner is not entries["module"]:
        _drop_table(fields, context)
        tok = py_get(fields, "_table_tok")
        if tok is not None:
            raise Error("RuntimeError: cannot release the previous Embedding table owner")
    if tok is None:
        var counter = _EMBEDDING_TOKENS.get_or_create_ptr()
        var fresh = counter[].next
        counter[].next += 1
        py_setitem(fields, "_table_tok", fresh)
        py_setitem(fields, "_table_owner", entries["module"])
        return fresh
    return _integer(tok, "resident table token")


def _drop_table(fields: PythonObject, context: PythonObject):
    """`Embedding._drop_table`: release this layer's resident device table,
    if it has one; every failure is swallowed, as it was."""
    try:
        var tok = py_get(fields, "_table_tok")
        if tok is None:
            return
        var owner = py_get(fields, "_table_owner")
        var entries = PythonObject(None)
        if owner is None:
            # Older in-process objects had no retained binding. New tokens
            # always record their native owner, including across mode changes.
            entries = _entries(fields, context, PythonObject(None))
        else:
            for name in ["identical", "fast"]:  # small-loop(two import-connected token owners)
                var candidate = context["embedding_bindings"][PythonObject(name)]
                if py_type(candidate) is _b().dict and candidate["module"] is owner:
                    entries = candidate
                    break
            if entries is None:
                raise Error("TypeError: resident Embedding table requires its original native binding")
        if py_contains(entries, "embedding_table_release"):
            _ = py_call_typed(entries["embedding_table_release"], PythonObject(_integer(tok, "resident table token")))
        _ = py_call(fields.pop, "_table_tok", PythonObject(None))
        _ = py_call(fields.pop, "_table_owner", PythonObject(None))
    except:
        pass


# ---------------------------------------------------------------- ids

def _ids(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Capture a native buffer once, retain its export through packing, and
    # feed the existing integer conversion ABI an owned compact snapshot.
    var pin = EmbeddingInput(value, context, True)
    var layout = pin.layout.copy()
    var fmt = pin.format
    var swapped = fmt.startswith(">") or fmt.startswith("!")
    if swapped:
        var trimmed = String(fmt[byte=1:])
        fmt = trimmed^
    fmt = base_format(fmt)
    var kind = _format_kind(fmt)
    if kind != CAST_KIND_SIGNED and kind != CAST_KIND_UNSIGNED:
        raise Error("TypeError: mojolearn Embedding: ids must be integers; float and bool ids are refused")
    if layout.count == 0:
        raise Error("ValueError: embedding indices must be nonempty")
    var code = -1
    if kind == CAST_KIND_SIGNED:
        code = 7 if layout.itemsize == 1 else (9 if layout.itemsize == 2 else (2 if layout.itemsize == 4 else 3))
    else:
        code = 5 if layout.itemsize == 1 else (6 if layout.itemsize == 2 else (4 if layout.itemsize == 4 else 10))
    var packed = embedding_pack(layout.address, layout.shape, layout.strides, layout.itemsize, swapped, code, layout.count)
    code = 2 if code == 9 else (3 if code == 10 else code)
    var out = new_array([layout.count], "<i4")
    var entry = context["embedding_index_cast"]
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: Embedding requires its compiled index caster")
    var address = Int(packed.unsafe_ptr())
    if Int(py=py_call_typed(entry, PythonObject(address), PythonObject(code), out._addr, PythonObject(2), PythonObject(layout.count))) != 0:
        embedding_cast_refusal(address, code, layout.count)
    _ = packed^
    _ = pin^
    return Python.tuple(out, out._addr, layout.count, py_ints(layout.shape))


# ---------------------------------------------------------------- forward

def _forward(model: PythonObject, ids: PythonObject, device: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _metadata(model, context)
    opt_scalar(device, context)
    var admitted = _ids(ids, context)
    var flat_addr = Int(py=admitted[1])
    var t = Int(py=admitted[2])
    var d = _integer(fields["embedding_dim"], "embedding_dim")
    var v = _integer(fields["num_embeddings"], "num_embeddings")
    if v < 1 or d < 1:
        raise Error("ValueError: Embedding dimensions must be positive")
    _ = shape_count([v, d])
    var dims = List[Int]()
    for k in range(len(admitted[3])):  # driver-loop: over the ids' shape words
        dims.append(Int(py=admitted[3][k]))
    dims.append(d)
    var weight = fields["weight"]
    var weight_pin = EmbeddingInput(weight, context)
    var weight_layout = weight_pin.layout.copy()
    if base_format(weight_pin.format) != "f" or weight_layout.itemsize != 4 or not weight_layout.contiguous() or not same_shape(weight_layout.shape, [v, d]):
        raise Error("TypeError: Embedding weight must be its contiguous float32 (V, D) table")
    var entries = _entries(fields, context, model)
    if Bool(py=device):
        for name in [String("seq_tensor_alloc"), String("embedding_forward_dev"), String("embedding_backward_dev")]:  # small-loop(the device I/O entry names)
            if not py_contains(entries, name):
                raise Error("RuntimeError: mojolearn Embedding: " + String("the connected embedding binding")
                            + " has no device I/O entries; use host arrays (device=False)")
        if not py_contains(context, "sequence_device_class"):
            # the device tensor class connects when its module is imported
            # (it needs NumPy); nothing is imported from here
            raise Error("RuntimeError: mojolearn Embedding: device=True returns a SequenceDeviceTensor; "
                        + "import mojolearn._x_sequence_device (or call to_device) first")
        var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(1, shape_count(dims))), PythonObject(0))
        var y = seq_device_make(entries["module"], dims, handle, PythonObject(None), context)
        var tok = _token(fields, entries, context)
        _ = py_call_typed(entries["embedding_forward_dev"],
                          Python.list(weight_layout.address, flat_addr), Python.list(handle),
                          Python.list(v, d, t, tok))
        _ = weight  # keep the Python object alive through the native call above
        _ = admitted^
        _ = weight_pin^
        _ = weight^
        return y
    var y = new_array(dims, "<f4")
    var tok = _token(fields, entries, context)
    var params = Python.list(v, d, t)
    if py_contains(entries, "embedding_table_release"):
        _ = py_call(params.append, tok)
    _ = py_call_typed(entries["embedding_forward"],
                      Python.list(weight_layout.address, flat_addr, y._addr), params)
    _ = y  # keep the Python object alive through the native call above
    _ = admitted^
    _ = weight_pin^
    _ = weight^
    return y


# ---------------------------------------------------------------- shared input capture and lifecycle

def _class_flag(cls: PythonObject, name: String, context: PythonObject) raises -> Bool:
    var ancestry = py_call(_b().type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native class metadata without descriptor calls
        var value = py_get(_class_fields(ancestry[i]), PythonObject(name))
        if value is not None:
            opt_scalar(value, context)
            return Bool(py=value)
    return False


def _owner(cls: PythonObject, context: PythonObject) raises:
    var ancestry = py_call(_b().type.__dict__["__mro__"].__get__, cls)
    var finalizer = _class_fields(context["embedding_class"])["__del__"]
    for i in range(len(ancestry)):  # driver-loop: require native lifetime before allocating or retaining a subclass
        var own_finalizer = py_get(_class_fields(ancestry[i]), "__del__")
        if own_finalizer is not None and own_finalizer is not finalizer:
            raise Error("TypeError: Embedding subclasses cannot install Python finalizer callbacks")
        if ancestry[i] is context["embedding_class"]:
            return
    raise Error("TypeError: Embedding requires its connected native class")


def _constructible(cls: PythonObject, context: PythonObject) raises:
    _owner(cls, context)
    var ancestry = py_call(_b().type.__dict__["__mro__"].__get__, cls)
    var schema = _class_fields(context["embedding_class"])
    for i in range(len(ancestry)):  # driver-loop: no implicit user constructor callback during native construction
        if ancestry[i] is context["embedding_class"]:
            return
        var namespace = _class_fields(ancestry[i])
        var init = py_get(namespace, "__init__")
        var new = py_get(namespace, "__new__")
        if (init is not None and init is not schema["__init__"]) or new is not None:
            raise Error("TypeError: native Embedding construction cannot call a Python subclass constructor")


def _metadata(model: PythonObject, context: PythonObject) raises -> PythonObject:
    _owner(py_type(model), context)
    var fields = neural_record_fields(model, Python.tuple("num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type",
        "scale_grad_by_freq", "sparse", "weight", "plan", "numeric_mode", "_table_tok", "_table_owner"))
    for name in ["num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse", "plan", "numeric_mode", "_table_tok"]:  # driver-loop: admit scalar state before native conversion
        opt_scalar(py_get(fields, PythonObject(name)), context)
    return fields


struct EmbeddingInput(Movable):
    # Shared buffer pins retain original storage for this complete public call.
    # Array metadata is captured through the import-connected native slots;
    # ordinary buffers use the public Py_buffer snapshot exactly once.
    var owner: PythonObject
    var pin: AdmittedBorrowedBuffer
    var layout: BorrowedBufferLayout
    var format: String

    def __init__(out self, source: PythonObject, context: PythonObject, indices: Bool = False) raises:
        self.owner = source
        var b = _b()
        var cls = py_type(source)
        var storage = source
        var shape = List[Int]()
        var strides = List[Int]()
        var address = 0
        var item = 0
        var dtype = String()
        var array = cls is context["array_class"]
        var sequence = cls is b.list or cls is b.tuple or cls is b.int or cls is b.float or cls is b.bool
        if array:
            var getters = context["array_getters"]
            var given_shape = py_call(getters["shape"], source)
            var given_strides = py_call(getters["strides"], source)
            shape = device_shape(given_shape, context)
            opt_scalar(given_strides, context)
            if py_type(given_strides) is not b.tuple or len(given_strides) != len(shape):
                raise Error("TypeError: Embedding Array requires native stride metadata")
            for i in range(len(shape)):  # driver-loop: capture each Array stride once
                if py_type(given_strides[i]) is not b.int:
                    raise Error("TypeError: Embedding Array strides must be integers")
                strides.append(Int(py=given_strides[i]))
            var raw_address = py_call(getters["_addr"], source)
            var raw_item = py_call(getters["itemsize"], source)
            var raw_dtype = py_call(getters["dtype"], source)
            if py_type(raw_address) is not b.int or py_type(raw_item) is not b.int or py_type(raw_dtype) is not b.str:
                raise Error("TypeError: Embedding Array needs native storage metadata")
            address = Int(py=raw_address)
            item = Int(py=raw_item)
            dtype = String(py=raw_dtype)
            storage = py_call(getters["_store"], source)
            if storage is None:
                storage = py_call(getters["_mv"], source)
                if py_type(storage) is not b.memoryview or storage.obj is None:
                    raise Error("TypeError: Embedding Array requires retained native storage")
        elif sequence:
            var leaves = Python.list()
            var depth = -1
            embedding_leaves(source, shape, 0, depth, leaves, context)
            if indices:
                for i in range(len(leaves)):  # driver-loop: index scalar kind admission before native storage
                    var kind = py_type(leaves[i])
                    if kind is b.float or kind is b.bool:
                        raise Error("TypeError: mojolearn Embedding: ids must be integers")
            var packed = embedding_leaf_store(leaves, False, context)
            storage = packed[0]
        else:
            opt_value(source, context)
        if not opt_exact(py_type(storage), context["buffer_types"]) and not opt_numpy(py_type(storage), context):
            admit_native_buffer_exporter(storage)
        self.pin = AdmittedBorrowedBuffer(storage)
        var raw = self.pin.layout.value().copy()
        for i in range(len(raw.suboffsets)):  # driver-loop: the numerical entries consume direct native storage
            if raw.suboffsets[i] >= 0:
                raise Error("TypeError: Embedding requires direct native buffer storage")
        self.format = self.pin.format
        if array:
            var expected_item = 0
            if dtype == "<f4":
                self.format = String("f")
                expected_item = 4
            elif dtype == "<f8":
                self.format = String("d")
                expected_item = 8
            elif dtype == "<f2":
                self.format = String("e")
                expected_item = 2
            elif dtype == "<i4":
                self.format = String("i")
                expected_item = 4
            elif dtype == "<i8":
                self.format = String("q")
                expected_item = 8
            elif dtype == "<u4":
                self.format = String("I")
                expected_item = 4
            elif dtype == "<u2":
                self.format = String("H")
                expected_item = 2
            elif dtype == "<i1" or dtype == "|i1":
                self.format = String("b")
                expected_item = 1
            elif dtype == "<u1" or dtype == "|u1":
                self.format = String("B")
                expected_item = 1
            else:
                raise Error("TypeError: Embedding Array has an unsupported dtype")
            if item != expected_item:
                raise Error("TypeError: Embedding Array item size differs from its dtype")
            var count = shape_count(shape)
            var storage_bytes = raw.count * raw.itemsize
            if not raw.contiguous():
                raise Error("TypeError: Embedding Array requires contiguous retained storage")
            if count != 0:
                if storage_bytes < item or address < raw.address or address > raw.address + storage_bytes - item:
                    raise Error("TypeError: Embedding Array address lies outside its retained storage")
                var room_before = address - raw.address
                var room_after = storage_bytes - room_before - item
                for i in range(len(shape)):  # driver-loop: bound captured byte strides before multiplication
                    var steps = max(shape[i] - 1, 0)
                    if steps == 0:
                        continue
                    if strides[i] >= 0:
                        if strides[i] > room_after // steps:
                            raise Error("TypeError: Embedding Array view lies outside its retained storage")
                        room_after -= steps * strides[i]
                    else:
                        if strides[i] < -(room_before // steps):
                            raise Error("TypeError: Embedding Array view lies outside its retained storage")
                        room_before += steps * strides[i]
            var rank = len(shape)
            self.layout = BorrowedBufferLayout(address, item, raw.readonly, shape^, strides^, List[Int](length=rank, fill=-1))
        elif sequence:
            var step = raw.itemsize
            strides = List[Int](length=len(shape), fill=0)
            for i in range(len(shape) - 1, -1, -1):  # driver-loop: native rectangular sequence shape
                strides[i] = step
                step *= shape[i]
            var rank = len(shape)
            self.layout = BorrowedBufferLayout(raw.address, raw.itemsize, raw.readonly, shape^, strides^, List[Int](length=rank, fill=-1))
        else:
            self.layout = raw^


def _format_kind(format: String) raises -> Int:
    var word = base_format(format)
    if word == "f" or word == "d" or word == "e":
        return CAST_KIND_FLOAT
    if word == "b" or word == "h" or word == "i" or word == "l" or word == "q" or word == "n":
        return CAST_KIND_SIGNED
    if word == "B" or word == "H" or word == "I" or word == "L" or word == "Q" or word == "N":
        return CAST_KIND_UNSIGNED
    if word == "?":
        return CAST_KIND_BOOL
    raise Error("TypeError: Embedding requires native real numeric buffers")


def _float_owned(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var pin = EmbeddingInput(source, context)
    var layout = pin.layout.copy()
    var format = pin.format
    var swapped = format.startswith(">") or format.startswith("!")
    if swapped:
        var trimmed = String(format[byte=1:])
        format = trimmed^
    var kind = _format_kind(format)
    var bytes = embedding_pack(layout.address, layout.shape, layout.strides, layout.itemsize, swapped, 0, layout.count)
    var strides = List[Int](length=len(layout.shape), fill=0)
    var step = layout.itemsize
    for i in range(len(strides) - 1, -1, -1):  # driver-loop: compact copied input rank metadata
        strides[i] = step
        step *= layout.shape[i]
    var out = new_array(layout.shape, "<f4")
    xseq_cast_nd(Int(bytes.unsafe_ptr()), layout.shape, strides, layout.itemsize, kind, True,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr)))
    _ = bytes^
    _ = pin^
    return out


def _dimensions(value: PythonObject, context: PythonObject) raises -> List[Int]:
    return device_shape(py_call(context["array_getters"]["shape"], value), context)


def _require_shape(value: PythonObject, expected: List[Int], name: String, context: PythonObject) raises:
    if not same_shape(_dimensions(value, context), expected):
        raise Error("ValueError: mojolearn Embedding: " + name + " has an unexpected shape")


def _integer(value: PythonObject, name: String) raises -> Int:
    if py_type(value) is not _b().int:
        raise Error("TypeError: mojolearn Embedding: " + name + " must be an int")
    return Int(py=value)


def _initialize(model: PythonObject, values: PythonObject, context: PythonObject, prepared: PythonObject) raises -> PythonObject:
    var fields = _metadata(model, context)
    var v = _integer(values[0], "num_embeddings")
    var d = _integer(values[1], "embedding_dim")
    if v < 1 or d < 1:
        raise Error("ValueError: mojolearn Embedding: num_embeddings and embedding_dim must be at least 1")
    _ = shape_count([v, d])
    if values[3] is not None:
        raise Error("ValueError: mojolearn Embedding: max_norm is REFUSED; in-place renormalization is outside the numerical profile")
    for i in [4, 5, 6]:  # small-loop(three passive constructor options)
        opt_scalar(values[i], context)
    if Float64(py=values[4]) != 2.0:
        raise Error("ValueError: mojolearn Embedding: norm_type is only 2.0")
    if Bool(py=values[5]) or Bool(py=values[6]):
        raise Error("ValueError: mojolearn Embedding: scale_grad_by_freq and sparse must be False")
    if py_type(values[8]) is not _b().str:
        raise Error("ValueError: mojolearn Embedding: plan must be auto, scan or sort")
    var plan = String(py=values[8])
    if plan != "auto" and plan != "scan" and plan != "sort":
        raise Error("ValueError: mojolearn Embedding: plan must be auto, scan or sort")
    var pad = values[2]
    if pad is not None:
        var index = _integer(pad, "padding_idx")
        if index < -v or index >= v:
            raise Error("ValueError: mojolearn Embedding: padding_idx must be within [-V, V)")
        pad = PythonObject(index + v if index < 0 else index)
    var mode = values[9]
    if mode is not None:
        if py_type(mode) is not _b().str:
            raise Error("ValueError: numeric_mode must be fast, identical or None")
        var normalized = String(py=mode).strip().lower()
        if normalized != "identical" and normalized != "fast":
            raise Error("ValueError: numeric_mode must be fast, identical or None")
        mode = PythonObject(normalized)
    if values[7] is None:
        raise Error("ValueError: mojolearn Embedding: weight is required; pass a table or use from_pretrained")
    var table = _float_owned(values[7], "weight", context) if prepared is None else prepared
    _require_shape(table, [v, d], "weight", context)
    _drop_table(fields, context)
    if py_get(fields, "_table_tok") is not None:
        raise Error("RuntimeError: cannot release the previous Embedding table owner")
    py_setitem(fields, "num_embeddings", v)
    py_setitem(fields, "embedding_dim", d)
    py_setitem(fields, "padding_idx", pad)
    py_setitem(fields, "max_norm", PythonObject(None))
    py_setitem(fields, "norm_type", values[4])
    py_setitem(fields, "scale_grad_by_freq", values[5])
    py_setitem(fields, "sparse", values[6])
    py_setitem(fields, "weight", table)
    py_setitem(fields, "plan", values[8])
    py_setitem(fields, "numeric_mode", mode)
    return PythonObject(None)


def _pretrained(cls: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    _constructible(cls, context)
    var table = _float_owned(values[0], "embeddings", context)
    var shape = _dimensions(table, context)
    if len(shape) != 2:
        raise Error("ValueError: mojolearn Embedding: embeddings must be two-dimensional")
    var model = py_call(_b().object.__new__, cls)
    _ = _initialize(model, Python.tuple(shape[0], shape[1], values[1], values[2], values[3], values[4], values[5], table, values[7], values[6]), context, table)
    return model


def _device_snapshot(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(value) is not context["sequence_device_class"]:
        raise Error("TypeError: Embedding requires a native SequenceDeviceTensor")
    var binding = seq_device_get(value, "b", context)
    var shape = device_shape(seq_device_get(value, "shape", context), context)
    var handle = seq_device_get(value, "h", context)
    if py_type(handle) is not _b().int:
        raise Error("TypeError: device tensor handle requires a native integer")
    _ = shape_count(shape)
    return Python.tuple(binding, py_ints(shape), handle)


def _is_device(value: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_get(context, "sequence_device_class")
    return cls is not None and py_type(value) is cls


def _device_entries(fields: PythonObject, model: PythonObject, context: PythonObject) raises -> PythonObject:
    var entries = _entries(fields, context, model)
    for name in ["seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download", "embedding_backward_dev"]:  # driver-loop: fixed resident embedding ABI
        if not py_contains(entries, name):
            raise Error("RuntimeError: mojolearn Embedding: this binding has no device I/O entries")
    if not py_contains(context, "sequence_device_class"):
        raise Error("RuntimeError: Embedding requires its import-connected SequenceDeviceTensor")
    return entries


def _to_device(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var entries = _device_entries(_metadata(model, context), model, context)
    var value = _float_owned(source, "a", context)
    var shape = _dimensions(value, context)
    var count = shape_count(shape)
    var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(count, 1)), PythonObject(0))
    try:
        if count:
            _ = py_call_typed(entries["seq_tensor_upload"], handle, value._addr, PythonObject(count))
    except error:
        _ = py_call_typed(entries["seq_tensor_free"], handle)
        raise error^
    _ = value^
    return seq_device_make(entries["module"], shape, handle, PythonObject(None), context)


def _backward(model: PythonObject, ids: PythonObject, dy: PythonObject, grad: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _metadata(model, context)
    var admitted = _ids(ids, context)
    var t = Int(py=admitted[2])
    var v = _integer(fields["num_embeddings"], "num_embeddings")
    var d = _integer(fields["embedding_dim"], "embedding_dim")
    if v < 1 or d < 1:
        raise Error("ValueError: Embedding dimensions must be positive")
    _ = shape_count([v, d])
    var expected = device_shape(admitted[3], context)
    expected.append(d)
    var pad = _NO_PADDING if fields["padding_idx"] is None else _integer(fields["padding_idx"], "padding_idx")
    var plan_name = fields["plan"]
    if py_type(plan_name) is not _b().str:
        raise Error("TypeError: Embedding plan requires a native string")
    var plan_text = String(py=plan_name)
    var plan = 2 if plan_text == "auto" else (0 if plan_text == "scan" else 1)
    if plan_text != "auto" and plan_text != "scan" and plan_text != "sort":
        raise Error("ValueError: Embedding plan must be auto, scan or sort")
    var entries = _entries(fields, context, model)
    var accumulate = 0 if grad is None else 1
    if _is_device(dy, context):
        entries = _device_entries(fields, model, context)
        var source = _device_snapshot(dy, context)
        if source[0] is not entries["module"] or not same_shape(device_shape(source[1], context), expected):
            raise Error("ValueError: Embedding dy must have ids.shape + (D,) on this layer's binding")
        var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(v * d, 1)), PythonObject(0))
        try:
            if grad is not None:
                if _is_device(grad, context):
                    var previous = _device_snapshot(grad, context)
                    if previous[0] is not entries["module"] or not same_shape(device_shape(previous[1], context), [v, d]):
                        raise Error("ValueError: Embedding grad must have (V, D) on this layer's binding")
                    _ = py_call_typed(entries["seq_tensor_copy"], handle, previous[2], PythonObject(v * d))
                else:
                    var previous = _float_owned(grad, "grad", context)
                    _require_shape(previous, [v, d], "grad", context)
                    _ = py_call_typed(entries["seq_tensor_upload"], handle, previous._addr, PythonObject(v * d))
                    _ = previous^
            _ = py_call_typed(entries["embedding_backward_dev"], Python.list(admitted[1]), Python.list(source[2], handle),
                              Python.list(v, d, t, pad, accumulate, plan))
        except error:
            _ = py_call_typed(entries["seq_tensor_free"], handle)
            raise error^
        _ = admitted^
        _ = dy
        _ = grad
        return seq_device_make(entries["module"], [v, d], handle, PythonObject(None), context)
    var gradient = _float_owned(dy, "dy", context)
    _require_shape(gradient, expected, "dy", context)
    var previous = grad
    if _is_device(previous, context):
        previous = sequence_device_operation("seqdev_host", previous, Python.tuple(), PythonObject(None), context)
    var out = new_array([v, d], "<f4") if previous is None else _float_owned(previous, "grad", context)
    _require_shape(out, [v, d], "grad", context)
    if plan == 2 and not py_contains(entries, "embedding_table_release"):
        plan = 0
    if not py_contains(entries, "embedding_backward"):
        raise Error("RuntimeError: this Embedding install has no training entry")
    _ = py_call_typed(entries["embedding_backward"], Python.list(gradient._addr, admitted[1], out._addr), Python.list(v, d, t, pad, accumulate, plan))
    _ = admitted^
    _ = gradient^
    return out


# ---------------------------------------------------------------- load

def _load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Embedding.load(path)`: read_npz with the format check,
    `_check_saved_by`, the meta and weight checks, the constructor's checks,
    then the layer built natively with its own copy of the table and the
    saved numeric mode."""
    var b = _b()
    _constructible(cls, context)
    var archive = context["embedding_archive_context"]
    var arrays = classical_archive_read(path, archive)
    var tag = String("")
    if py_contains(arrays, "format"):
        tag = String(py=_archive_string(arrays, "format", archive))
    if tag != _FORMAT:
        raise Error("ValueError: mojolearn: " + _repr(path) + " holds model format " + _repr(PythonObject(tag))
                    + ", this loader reads " + _repr(PythonObject(_FORMAT)))
    var saved_as = _archive_string(arrays, "estimator", archive)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var known = False
    for k in range(len(mro)):  # driver-loop: over the loading class's ancestry names
        known = known or _class_name(mro[k]) == String(py=saved_as)
    if not known:
        raise Error("ValueError: mojolearn: " + _repr(path) + " was saved by " + String(py=saved_as)
                    + ", not " + _class_name(cls))
    var meta = _archive_exact(arrays, "meta", "<i8", archive)
    if Int(py=meta.size) != 3:
        raise Error("ValueError: mojolearn: " + _repr(path) + " meta holds " + String(Int(py=meta.size))
                    + " fields, 3 are needed")
    var mp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=meta._addr))
    _ = meta  # keep the Python object alive through the native call above
    var v = Int(mp[0])
    _ = meta  # keep the Python object alive through the native call above
    var d = Int(mp[1])
    _ = meta  # keep the Python object alive through the native call above
    var pad = Int(mp[2])
    _ = meta  # keep the Python object alive through the native call above
    var weight = _archive_exact(arrays, "weight", "<f4", archive)
    var shape = weight.shape
    if len(shape) != 2 or Int(py=shape[0]) != v or Int(py=shape[1]) != d:
        raise Error("ValueError: mojolearn: " + _repr(path) + " weight has shape " + _repr(py_call(b.tuple, shape))
                    + ", not " + _repr(Python.tuple(v, d)))
    var plan = _archive_string(arrays, "plan", archive)
    var mode = _archive_string(arrays, "numeric_mode", archive)
    # the constructor's refusals, in its words
    if v < 1:
        raise Error("ValueError: mojolearn Embedding: num_embeddings must be at least 1, got " + String(v))
    if d < 1:
        raise Error("ValueError: mojolearn Embedding: embedding_dim must be at least 1, got " + String(d))
    if String(py=plan) != "auto" and String(py=plan) != "scan" and String(py=plan) != "sort":
        raise Error("ValueError: mojolearn Embedding: plan must be 'auto', 'scan' or 'sort' (contract section 6's "
                    + "PLAN_SCAN and PLAN_SORT), got " + _repr(plan))
    var padding = PythonObject(None)
    if pad != _NO_PADDING:
        if not (-v <= pad and pad < v):
            raise Error("ValueError: mojolearn Embedding: padding_idx must be within [-" + String(v) + ", " + String(v)
                        + "), got " + String(pad))
        padding = PythonObject(pad + v if pad < 0 else pad)
    if String(py=mode) != "fast" and String(py=mode) != "identical":
        raise Error("ValueError: mojolearn: invalid saved numeric_mode " + _repr(mode))
    # the layer's own bytes (the binding keeps a device copy of the table)
    var table = new_array([v, d], "<f4")
    memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=table._addr)),
           src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=weight._addr)), count=v * d)
    _ = table  # keep the Python object alive through the native call above
    _ = weight  # keep the Python object alive through the native call above
    var model = py_call(b.object.__new__, cls)
    var fields = _fields(model)
    py_setitem(fields, "num_embeddings", v)
    py_setitem(fields, "embedding_dim", d)
    py_setitem(fields, "padding_idx", padding)
    py_setitem(fields, "max_norm", PythonObject(None))
    py_setitem(fields, "norm_type", 2.0)
    py_setitem(fields, "scale_grad_by_freq", False)
    py_setitem(fields, "sparse", False)
    py_setitem(fields, "plan", plan)
    py_setitem(fields, "weight", table)
    py_setitem(fields, "numeric_mode", mode)
    _ = meta^
    _ = weight^
    return model


def emblayer_operation(operation: String, owner: PythonObject, values: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    if operation == "emblayer_init":
        return _initialize(owner, values, context, PythonObject(None))
    if operation == "emblayer_pretrained":
        return _pretrained(owner, values, context)
    if operation == "emblayer_to_device":
        return _to_device(owner, values[0], context)
    if operation == "emblayer_backward":
        return _backward(owner, values[0], values[1], values[2], context)
    if operation == "emblayer_state":
        var fields = _metadata(owner, context)
        var state = py_call(_b().dict, fields)
        _ = py_call(state.pop, "_table_tok", PythonObject(None))
        _ = py_call(state.pop, "_table_owner", PythonObject(None))
        return state
    if operation == "emblayer_set_params":
        var fields = _metadata(owner, context)
        var options = values[0]
        if py_type(options) is not _b().dict:
            raise Error("TypeError: Embedding settings require a native dictionary")
        var keys = py_call(_b().list, options)
        var names = Python.tuple("num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse", "weight", "plan", "numeric_mode")
        for i in range(len(keys)):  # driver-loop: validate parameter keys before dictionary reads
            if py_type(keys[i]) is not _b().str or not py_contains(names, keys[i]):
                raise Error("ValueError: unknown Embedding constructor parameter")
        if len(keys) == 0:
            return owner
        var args = Python.list()
        for i in range(len(names)):  # driver-loop: fixed constructor configuration
            _ = py_call(args.append, options[names[i]] if py_contains(options, names[i]) else fields[names[i]])
        _ = _initialize(owner, py_call(_b().tuple, args), context, PythonObject(None))
        return owner
    if operation == "emblayer_forward":
        return _forward(owner, values[0], values[1], context)
    if operation == "emblayer_load":
        return _load(owner, values[0], context)
    raise Error("RuntimeError: unknown Embedding operation " + operation)


# ---------------------------------------------------------------- lifetime

def embedding_layer_lifetime_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    """The class's `__setattr__` (args: context, layer, name, value) and
    `__del__` (args: context, layer), bound through the native attribute
    descriptor at import. Assigning `weight` releases the resident table
    first; finalizing releases it; nothing else changes."""
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var context = args[0]
        var model = args[1]
        if len(args) == 2:
            _drop_table(_fields(model), context)
            return PythonObject(None).steal_data()
        opt_guards(context)
        door_ready()
        var fields = _metadata(model, context)
        var name = args[2]
        if py_type(name) is not _b().str:
            raise Error("TypeError: Embedding attribute names must be native strings")
        if String(py=name) == "weight":
            var table = _float_owned(args[3], "weight", context)
            if py_get(fields, "num_embeddings") is not None and py_get(fields, "embedding_dim") is not None:
                _require_shape(table, [_integer(fields["num_embeddings"], "num_embeddings"), _integer(fields["embedding_dim"], "embedding_dim")], "weight", context)
            _drop_table(fields, context)
            if py_get(fields, "_table_tok") is not None:
                raise Error("RuntimeError: cannot release the previous Embedding table owner")
            py_setitem(fields, "weight", table)
        else:
            # Refuse descriptor callbacks before native setattr; native slots
            # keep their documented precedence over the instance dictionary.
            var b = _b()
            var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
            for i in range(len(ancestry)):  # driver-loop: descriptor precedence before attribute storage
                var member = py_get(_class_fields(ancestry[i]), name)
                if member is not None:
                    var member_mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(member))
                    for j in range(len(member_mro)):  # driver-loop: refuse Python descriptor writes
                        if py_contains(_class_fields(member_mro[j]), "__set__"):
                            var types = native_loaded_module("types")
                            if py_type(member) is not types.MemberDescriptorType and py_type(member) is not types.GetSetDescriptorType:
                                raise Error("TypeError: Embedding attribute assignment refuses Python descriptor callbacks")
                    break
            _ = py_call(py_attr(b.object, "__setattr__"), model, name, args[3])
        return PythonObject(None).steal_data()
    except error:
        var n_args = 0
        try:
            n_args = len(args)
        except:
            pass
        if n_args == 2:
            # a finalizer never raises (released handles, a shutting-down interpreter)
            return PythonObject(None).steal_data()
        return optimizer_python_error(error)
