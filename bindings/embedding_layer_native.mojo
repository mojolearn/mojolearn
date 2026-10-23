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
from bindings.optimizer_public_door import TensorView, tensor_view, new_array, contiguous_addr
from bindings.optimizer_device_native import shape_count
from bindings.training_embedding_native import embedding_indices, embedding_leaves
from bindings.sequence_device_native import seq_device_make
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
    return Python.import_module("builtins")


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


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


def _entries(fields: PythonObject, context: PythonObject, model: PythonObject = PythonObject(None)) raises -> PythonObject:
    """`Embedding._extension()`: the embedding binding of the layer's mode,
    connected at import, with its compiled mode read back; for a host
    inference class (`_HostBound`, IDENTICAL only) the lookup host binding."""
    var b = _b()
    if model is not None and Bool(py=py_call(b.getattr, py_type(model), "_HOST_INFERENCE_ONLY", False)):
        var saved = py_get(fields, "numeric_mode")
        if saved is not None and not Bool(py=saved == "identical"):
            raise Error("ValueError: mojolearn: " + String(py=py_type(model).__name__)
                        + " runs IDENTICAL only on the host; this model was saved " + _repr(saved))
        var module = py_get(context["modules"], context["embedding_host_module"])
        if module is None:
            raise Error("RuntimeError: mojolearn: the host " + String(py=py_type(model).__name__)
                        + " has no loaded lookup binding; load the model with its own load()")
        var host = b.dict()
        py_setitem(host, "module", module)
        var names = context["embedding_host_names"]
        for i in range(len(names)):  # driver-loop: over the fixed host entry names
            var entry = py_call(b.getattr, module, names[i], PythonObject(None))
            if entry is not None and py_type(entry) is context["device_finalizer_types"][2]:
                py_setitem(host, names[i], entry)
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
                        + String(py=entries["module"].__name__)
                        + " reports another compile-time mode; rebuild it with bash bindings/build_embedding.sh")
    return entries


def _token(fields: PythonObject, entries: PythonObject) raises -> Int:
    """`Embedding._table_token`: 0 for a binding without the resident table
    (the host bindings), else this layer's token."""
    if not py_contains(entries, "embedding_table_release"):
        return 0
    var tok = py_get(fields, "_table_tok")
    if tok is None:
        var counter = _EMBEDDING_TOKENS.get_or_create_ptr()
        var fresh = counter[].next
        counter[].next += 1
        py_setitem(fields, "_table_tok", fresh)
        return fresh
    return Int(py=tok)


def _drop_table(fields: PythonObject, context: PythonObject):
    """`Embedding._drop_table`: release this layer's resident device table,
    if it has one; every failure is swallowed, as it was."""
    try:
        var tok = py_get(fields, "_table_tok")
        if tok is None:
            return
        _ = py_call(fields.pop, "_table_tok", PythonObject(None))
        var entries = _entries(fields, context, PythonObject(None))
        if py_contains(entries, "embedding_table_release"):
            _ = py_call(entries["embedding_table_release"], tok)
    except:
        pass


# ---------------------------------------------------------------- ids

def _ids(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Embedding._ids`: integer ids of any rank (a float or bool id is
    refused by name, never truncated) as one C-ordered int32 vector;
    returns (vector, address, count, shape words)."""
    var view = TensorView()
    var dims = List[Int]()
    if tensor_view(value, view):
        var fmt = view.fmt
        if fmt.byte_length() > 1 and (fmt.startswith("<") or fmt.startswith(">") or fmt.startswith("=")
                                      or fmt.startswith("@") or fmt.startswith("!")):
            var trimmed = String(fmt[byte=1:])
            fmt = trimmed^
        if fmt == "f" or fmt == "d" or fmt == "e" or fmt == "?":
            raise Error("TypeError: mojolearn Embedding: ids must be integers, got dtype '" + view.dtype
                        + "'; a float id is refused rather than truncated")
        for k in range(len(view.shape)):  # driver-loop: over the ids' shape words
            dims.append(view.shape[k])
    else:
        # nested sequences and scalars: their leaves decide the dtype the
        # materialized Array would have had
        var depth = -1
        var leaves = Python.list()
        embedding_leaves(value, dims, 0, depth, leaves, context)
        var b = _b()
        for i in range(len(leaves)):  # driver-loop: over the leaf types (no value is read)
            if py_type(leaves[i]) is b.float:
                raise Error("TypeError: mojolearn Embedding: ids must be integers, got dtype '<f8'; "
                            + "a float id is refused rather than truncated")
            if py_type(leaves[i]) is b.bool:
                raise Error("TypeError: mojolearn Embedding: ids must be integers, got dtype '|b1'; "
                            + "a float id is refused rather than truncated")
    var packed = embedding_indices(value, context)
    var shape = Python.list()
    for k in range(len(dims)):  # driver-loop: over the ids' shape words
        _ = py_call(shape.append, dims[k])
    return Python.tuple(packed[0], packed[1], packed[2], shape)


def _weight_addr(weight: PythonObject, mut keep: List[List[Float32]], context: PythonObject) raises -> Int:
    """`addr_ro(self.weight, name="weight")`."""
    if py_type(weight) is context["array_class"]:
        return Int(py=weight._addr)
    var view = TensorView()
    if not tensor_view(weight, view):
        raise Error("TypeError: mojolearn: weight does not export a buffer")
    return contiguous_addr(weight, view, keep)


# ---------------------------------------------------------------- forward

def _forward(model: PythonObject, ids: PythonObject, device: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var admitted = _ids(ids, context)
    var flat_addr = Int(py=admitted[1])
    var t = Int(py=admitted[2])
    var d = Int(py=fields["embedding_dim"])
    var v = Int(py=fields["num_embeddings"])
    var dims = List[Int]()
    for k in range(len(admitted[3])):  # driver-loop: over the ids' shape words
        dims.append(Int(py=admitted[3][k]))
    dims.append(d)
    var keep = List[List[Float32]]()
    var weight = fields["weight"]
    var entries = _entries(fields, context, model)
    if Bool(py=device):
        for name in [String("seq_tensor_alloc"), String("embedding_forward_dev"), String("embedding_backward_dev")]:  # small-loop(the device I/O entry names)
            if not py_contains(entries, name):
                raise Error("RuntimeError: mojolearn Embedding: " + String(py=entries["module"].__name__)
                            + " has no device I/O entries; use host arrays (device=False)")
        if not py_contains(context, "sequence_device_class"):
            # the device tensor class connects when its module is imported
            # (it needs NumPy); nothing is imported from here
            raise Error("RuntimeError: mojolearn Embedding: device=True returns a SequenceDeviceTensor; "
                        + "import mojolearn._x_sequence_device (or call to_device) first")
        var handle = py_call_typed(entries["seq_tensor_alloc"], PythonObject(max(1, shape_count(dims))), PythonObject(0))
        var y = seq_device_make(entries["module"], dims, handle, PythonObject(None), context)
        var tok = _token(fields, entries)
        _ = py_call_typed(entries["embedding_forward_dev"],
                          Python.list(_weight_addr(weight, keep, context), flat_addr), Python.list(handle),
                          Python.list(v, d, t, tok))
        _ = keep^
        _ = admitted^
        _ = weight^
        return y
    var y = new_array(dims, "<f4")
    var tok = _token(fields, entries)
    var params = Python.list(v, d, t)
    if py_contains(entries, "embedding_table_release"):
        _ = py_call(params.append, tok)
    _ = py_call_typed(entries["embedding_forward"],
                      Python.list(_weight_addr(weight, keep, context), flat_addr, y._addr), params)
    _ = keep^
    _ = admitted^
    _ = weight^
    return y


# ---------------------------------------------------------------- load

def _load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Embedding.load(path)`: read_npz with the format check,
    `_check_saved_by`, the meta and weight checks, the constructor's checks,
    then the layer built natively with its own copy of the table and the
    saved numeric mode."""
    var b = _b()
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
        known = known or Bool(py=mro[k].__name__ == saved_as)
    if not known:
        raise Error("ValueError: mojolearn: " + _repr(path) + " was saved by " + String(py=saved_as)
                    + ", not " + String(py=cls.__name__))
    var meta = _archive_exact(arrays, "meta", "<i8", archive)
    if Int(py=meta.size) != 3:
        raise Error("ValueError: mojolearn: " + _repr(path) + " meta holds " + String(Int(py=meta.size))
                    + " fields, 3 are needed")
    var mp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=meta._addr))
    var v = Int(mp[0])
    var d = Int(mp[1])
    var pad = Int(mp[2])
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
    var model = py_call(b.object.__new__, cls)
    var fields = py_fields(model)
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
        var fields = py_fields(model)
        if len(args) == 2:
            _drop_table(fields, context)
            return PythonObject(None).steal_data()
        var name = args[2]
        if String(py=name) == "weight":
            _drop_table(fields, context)
        _ = py_call(py_attr(_b().object, "__setattr__"), model, name, args[3])
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
