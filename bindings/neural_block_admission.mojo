# SPDX-License-Identifier: Apache-2.0
"""Admission and retained metadata for standalone neural public operations."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_class_admission import class_record, class_admit, class_watch_ready
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_array_snapshot import native_array_slots as neural_array_slots, native_array_snapshot as neural_array_snapshot
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from bindings.generation_buffers import generation_array, py_ints
from bindings.native_array_storage import native_scratch_bytes, native_scratch_address
from transformer.causal_lm_construct_impl import causal_lm_gather_f32
from core.hostptr import f32_ptr
from std.memory import memcpy
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequential_admission import _one_of

# The uses of admit_block_f32 (below): read in place when possible, kept for
# the operation, or retained by the model.
comptime BLOCK_READ = 0
comptime BLOCK_OPERATION = 1
comptime BLOCK_RETAINED = 2


def neural_native_scalar(mut mods: NativeModules, value: PythonObject, context: PythonObject, integer: Bool = False) raises -> Bool:
    var b = mods.module("builtins")
    var cls = py_type(value)
    if cls is b.int or (not integer and (cls is b.float or cls is b.bool)):
        return True
    var admission = py_get(context, "admission_context")
    if admission is not None:
        classical_connect_numpy(admission)
        return _one_of(cls, admission["integer_types" if integer else "scalar_types"])
    return False


def _neural_scalar_in_admission(
    mut mods: NativeModules, cls: PythonObject, admission: PythonObject, mut connected: Bool, integer: Bool = False
) raises -> Bool:
    # Same verdict as neural_native_scalar for a non-builtin class, with the
    # admission context resolved once per scan by the caller. The NumPy
    # connect runs at most once per scan (`connected`), at the first value
    # that needs it, so a scan without such values still never connects and
    # a failing connect raises at the same value as before. A scan never
    # releases the GIL and only walks exact list/tuple/dict containers, so
    # the connection cannot change between values of one scan.
    if admission is None:
        return False
    if not connected:
        classical_connect_numpy(admission)
        connected = True
    return _one_of(cls, admission["integer_types" if integer else "scalar_types"])


def neural_native_scalar(value: PythonObject, context: PythonObject, integer: Bool = False) raises -> Bool:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_native_scalar(mods, value, context, integer)


def neural_string_keys(mut mods: NativeModules, value: PythonObject) raises:
    var b = mods.module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: neural metadata requires an exact dictionary")
    var keys = py_call(b.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit every key before dictionary lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: neural metadata requires exact string keys")


def neural_string_keys(value: PythonObject) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    neural_string_keys(mods, value)


def _neural_plain_scan(
    mut mods: NativeModules, value: PythonObject, depth: Int, admission: PythonObject, mut connected: Bool
) raises:
    # One scan of one value tree: the admission context is resolved once by
    # neural_plain_value and the NumPy connect runs at most once per scan,
    # not once per non-plain scalar.
    if depth > 32:
        raise Error("TypeError: neural metadata nesting exceeds the supported depth")
    var b = mods.module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.int or cls is b.float or cls is b.bool or cls is b.str:
        return
    if _neural_scalar_in_admission(mods, cls, admission, connected):
        return
    if cls is b.list or cls is b.tuple:
        for i in range(len(value)):  # driver-loop: passive configuration and descriptor words
            _neural_plain_scan(mods, value[i], depth + 1, admission, connected)
        return
    if cls is b.dict:
        neural_string_keys(mods, value)
        var keys = py_call(b.list, value.keys())
        for i in range(len(keys)):  # driver-loop: passive option dictionaries only
            _neural_plain_scan(mods, value[keys[i]], depth + 1, admission, connected)
        return
    raise Error("TypeError: neural metadata requires native scalar/container values; Python conversion hooks are not supported")


def neural_plain_value(mut mods: NativeModules, value: PythonObject, depth: Int = 0, context: PythonObject = PythonObject(None)) raises:
    var admission = PythonObject(None)
    if context is not None:
        admission = py_get(context, "admission_context")
    var connected = False
    _neural_plain_scan(mods, value, depth, admission, connected)


def neural_plain_value(value: PythonObject, depth: Int = 0, context: PythonObject = PythonObject(None)) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    neural_plain_value(mods, value, depth, context)


def neural_record_fields(mut mods: NativeModules, value: PythonObject, names: PythonObject) raises -> PythonObject:
    # _fields rejects a replaced __dict__ descriptor before native dictionary
    # access, including on an otherwise ordinary subclass. No getattr is used.
    var fields = _fields(mods, value)
    var b = mods.module("builtins")
    # The descriptor refusal reads only the class rows and the names: kept
    # per (class, names) (native_class_admission; re-run when the class or an
    # ancestor changes; every call on Python < 3.12). Lane p3-neural-blocks:
    # block, state and session calls ran this MRO walk on every step.
    var slot = String()
    var cls = py_type(value)
    if class_watch_ready():
        slot = String("neural_record_fields_v1")
        for j in range(len(names)):  # small-loop(known operation field names): the record slot's exact name list
            slot += "|" + String(py=names[j])
        if class_record(mods, slot, cls) is not None:
            return fields
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: refuse field descriptors before later shared attribute access
        var namespace = _class_fields(mods, ancestry[i])
        for j in range(len(names)):  # small-loop(known operation field names)
            if py_contains(namespace, names[j]):
                var member = namespace[names[j]]
                var member_mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(member))
                for k in range(len(member_mro)):  # small-loop(descriptor protocol ancestry)
                    if py_contains(_class_fields(mods, member_mro[k]), "__get__"):
                        raise Error("TypeError: neural fields require native instance storage, not Python descriptors")
    if slot.byte_length() > 0:
        class_admit(mods, slot, cls, PythonObject(True))
    return fields


def neural_record_fields(value: PythonObject, names: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_record_fields(mods, value, names)



def neural_admit_weights(mut mods: NativeModules, values: PythonObject, context: PythonObject) raises:
    var b = mods.module("builtins")
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        raise Error("TypeError: neural weights require exact native slot lists")
    for i in range(len(values)):  # driver-loop: admit each tensor owner before numerical preparation
        if values[i] is None:
            continue
        if py_type(values[i]) is context["array_class"]:
            _ = neural_array_snapshot(mods, values[i], context)
        else:
            admit_native_buffer_exporter(values[i])


def neural_block_fields(mut mods: NativeModules, block: PythonObject, context: PythonObject, initialized: Bool = True) raises -> PythonObject:
    var b = mods.module("builtins")
    if py_type(py_type(block)) is not b.type:
        raise Error("TypeError: neural blocks cannot invoke Python metaclass protocols")
    var names = Python.tuple("_native_kind", "_native_route", "numeric_mode", "numeric_profile", "weight_format",
        "d_model", "d_inner", "nheads", "conv_dim", "d_in_proj", "dt_rank", "_owned_generation", "_owned_restore_pending", "_discard_reports", "dt_limit", "n_heads", "n_kv_heads", "head_dim",
        "intermediate", "window", "max_positions", "_opts_tail", "_extended", "_w", "_wopt", "_int15",
        "_arithmetic_profile", "_arithmetic_profile_mode", "_arithmetic_profile_changed", "_W_NAMES",
        "norm", "mlp", "qkv_bias", "o_bias", "norm_bias", "mlp_bias", "qk_norm", "rope_scaling",
        "norm_eps", "rope_theta", "rope_dim", "attn_softcap")
    var fields = neural_record_fields(mods, block, names)
    # A subclass of a connected block class: the answer depends only on the
    # class ancestry and the import-connected classes, kept per class.
    var block_cls = py_type(block)
    if class_record(mods, _NEURAL_CONNECTED_SLOT, block_cls) is None:
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__, block_cls)
        var admitted = False
        # The eight connected block classes are read once, not once per ancestor.
        var block_classes = context["block_classes"]
        var connected = List[PythonObject]()
        for route in ["cpu", "gpu"]:  # driver-loop: native admission metadata and retained owner registry
            var by_kind = block_classes[route]
            for kind in ["transformer", "mamba1", "mamba2", "mamba3"]:  # driver-loop: native admission metadata and retained owner registry
                connected.append(by_kind[kind])
        for i in range(len(ancestry)):  # driver-loop: passive subclasses of connected public block classes
            var ancestor = ancestry[i]
            for j in range(len(connected)):  # small-loop(eight connected block classes)
                admitted = admitted or ancestor is connected[j]
        if not admitted:
            raise Error("TypeError: neural operation requires a connected native block")
        class_admit(mods, _NEURAL_CONNECTED_SLOT, block_cls, PythonObject(True))
    if not initialized:
        return fields
    for i in range(len(names)):  # small-loop(admit mutable scalar words before conversion or comparisons)
        var name = String(names[i])
        if name != "_w" and name != "_wopt" and name != "_int15" and py_contains(fields, names[i]):
            neural_plain_value(mods, fields[names[i]], 0, context)
    for name in ["_native_kind", "_native_route"]:  # driver-loop: native admission metadata and retained owner registry
        if py_type(fields[name]) is not b.str:
            raise Error("TypeError: neural route metadata requires exact strings")
    if py_type(fields["d_model"]) is not b.int or Int(py=fields["d_model"]) < 1:
        raise Error("TypeError: neural d_model must be a positive native integer")
    return fields


comptime _NEURAL_CONNECTED_SLOT = "neural_block_connected_class_v1"


def neural_block_fields(block: PythonObject, context: PythonObject, initialized: Bool = True) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_block_fields(mods, block, context, initialized)


def neural_block_weight_names(mut mods: NativeModules, block: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    var fields = _fields(mods, block)
    var names = py_get(fields, "_W_NAMES")
    if names is None:
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(block))
        for i in range(len(ancestry)):  # driver-loop: native admission metadata and retained owner registry
            names = py_get(_class_fields(mods, ancestry[i]), "_W_NAMES")
            if names is not None:
                break
    if py_type(names) is not b.tuple and py_type(names) is not b.list:
        raise Error("TypeError: neural weight registry requires native names")
    for i in range(len(names)):  # driver-loop: native admission metadata and retained owner registry
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: neural weight names require exact strings")
    return py_call(b.tuple, names)


def neural_weights_snapshot(mut mods: NativeModules, values: PythonObject, context: PythonObject, f32: Bool = True) raises -> PythonObject:
    var b = mods.module("builtins")
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        raise Error("TypeError: neural weights require exact native slot lists")
    var result = Python.list()
    for i in range(len(values)):  # driver-loop: capture each weight's storage metadata exactly once
        var value = values[i]
        if value is None:
            py_call(result.append, value)
        elif py_type(value) is context["array_class"]:
            py_call(result.append, admit_block_f32(mods, value, "weight", context, BLOCK_OPERATION) if f32 else neural_array_snapshot(mods, value, context))
        else:
            # Foreign model buffers historically copy to native storage on
            # use. Capture that copy once and retain it for this operation.
            py_call(result.append, admit_block_f32(mods, value, "weight", context, BLOCK_OPERATION))
    return result


def neural_metadata_snapshot(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var b = mods.module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        var values = Python.list()
        for i in range(len(value)):  # small-loop(copy already admitted option words)
            py_call(values.append, neural_metadata_snapshot(mods, value[i]))
        return py_call(b.tuple, values) if py_type(value) is b.tuple else values
    if py_type(value) is b.dict:
        var values = Python.dict()
        var keys = py_call(b.list, value.keys())
        for i in range(len(keys)):  # driver-loop: native admission metadata and retained owner registry
            py_setitem(values, keys[i], neural_metadata_snapshot(mods, value[keys[i]]))
        return values
    return value


def neural_metadata_snapshot(value: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_metadata_snapshot(mods, value)


def neural_block_snapshot(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = neural_block_fields(mods, block, context)
    var b = mods.module("builtins")
    var snapshot = py_call(b.dict, fields)
    py_setitem(snapshot, "_neural_live_fields", fields)
    py_setitem(snapshot, "_neural_weight_names", neural_block_weight_names(mods, block))
    for name in ["dt_limit", "_opts_tail", "rope_scaling"]:  # driver-loop: native admission metadata and retained owner registry
        if py_contains(fields, name):
            py_setitem(snapshot, name, neural_metadata_snapshot(mods, fields[name]))
    for name in ["_w", "_wopt"]:  # driver-loop: native admission metadata and retained owner registry
        if py_contains(fields, name):
            var values = fields[name]
            if py_type(values) is not b.list and py_type(values) is not b.tuple:
                raise Error("TypeError: neural weights require exact native slot lists")
            var originals = py_call(b.list, values)
            # Public parameters expose the original live buffers. Retain those
            # identities now rather than rereading caller slots after kernels.
            py_setitem(snapshot, PythonObject("_neural_original" + name), originals)
            py_setitem(snapshot, name, neural_weights_snapshot(mods, originals, context))
    var packed = py_get(fields, "_int15")
    if packed is not None:
        if py_type(packed) is not b.list and py_type(packed) is not b.tuple:
            raise Error("TypeError: neural packed weights require native plane lists")
        var planes = Python.list()
        for i in range(len(packed)):  # driver-loop: native admission metadata and retained owner registry
            py_call(planes.append, PythonObject(None) if packed[i] is None else neural_weights_snapshot(mods, packed[i], context, False))
        py_setitem(snapshot, "_int15", planes)
    return snapshot


def neural_block_snapshot(block: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_block_snapshot(mods, block, context)


def neural_model_weights(mut mods: NativeModules, values: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_string_keys(mods, values)
    var b = mods.module("builtins")
    var keys = py_call(b.list, values.keys())
    var result = Python.dict()
    for i in range(len(keys)):  # driver-loop: named constructor weights, before any unpacking/conversion
        var value = values[keys[i]]
        if py_type(value) is context["bf16_class"] or py_type(value) is context["int8_class"]:
            var fields = neural_record_fields(mods, value, Python.tuple("shape", "bits", "codes", "exponents"))
            neural_plain_value(mods, fields["shape"], 0, context)
            var captured = py_call(b.object.__new__, py_type(value))
            var saved = _fields(mods, captured)
            py_setitem(saved, "shape", neural_metadata_snapshot(mods, fields["shape"]))
            for name in ["bits", "codes", "exponents"]:  # small-loop(packed weight storage fields)
                if py_contains(fields, name):
                    var storage = fields[name]
                    if py_type(storage) is context["array_class"]:
                        storage = neural_array_snapshot(mods, storage, context)
                    else:
                        admit_native_buffer_exporter(storage)
                        # A retained native memoryview captures foreign metadata
                        # once for the existing packed-weight unpack operation.
                        storage = py_call(b.memoryview, storage)
                    py_setitem(saved, name, storage)
            py_setitem(result, keys[i], captured)
        else:
            py_setitem(result, keys[i], admit_block_f32(mods, value, String(keys[i]), context, BLOCK_RETAINED))
    return result



# THE ONE NEURAL INPUT DOOR (plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md
# section 3.3 row A4, lane a4-admit-sequence-tsa-neural, 2026-11-08): every
# float32 tensor a neural block, model or session operation admits goes
# through `admit_block_f32` with the USE that decides borrow versus copy.
#
#   BLOCK_READ       the operation only reads it (a forward / step / backward
#                    x): a C-order float32 Array or a C-contiguous, four-byte
#                    aligned float32 export is read in place (a read-only
#                    Array view); anything else is ONE copy into
#                    uninitialised storage (0.8.66 regression C5: a fresh
#                    zero-filled Array per call before the copy).
#   BLOCK_OPERATION  kept for this operation (weights, grad_output, session
#                    pieces): a C-order float32 Array is used as it is, any
#                    other input is the operation's own copy (a native Array).
#   BLOCK_RETAINED   kept by the model (constructor weights): always the
#                    model's own C-order copy, so a later in-place write into
#                    the caller's array cannot reach a resident session.
#
# Numerics: none; the same float32 words, in C order, reach the same kernels.
# TOMBSTONE: neural_block_f32, neural_block_f32_owned and neural_block_f32_input (admission helpers, merged into admit_block_f32) deleted 2026-11-08 by lane a4-admit-sequence-tsa-neural; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A4_ADMISSION_HELPERS.patch.
# (BLOCK_READ / BLOCK_OPERATION / BLOCK_RETAINED are defined after the imports.)


def _neural_dtype_refusal(name: String, code: String) -> String:
    # code is captured native Array metadata or the admitted buffer format,
    # never an arbitrary object's dtype attribute or a coercion callback.
    var label = String(code)
    if code == "<f8" or code == ">f8" or code == "d" or code == "<d" or code == ">d" or code == "=d" or code == "@d":
        label = "float64"
    elif code == "<f2" or code == ">f2" or code == "e" or code == "<e" or code == ">e" or code == "=e" or code == "@e":
        label = "float16"
    elif code == "<u2" or code == ">u2" or code == "H" or code == "<H" or code == ">H" or code == "=H" or code == "@H":
        label = "uint16 (bfloat16 bits)"
    return ("TypeError: neural " + name + " has dtype " + label + "; float32 only: bfloat16, "
            + "float16 and float64 are refused rather than converted. Convert to float32.")


def _block_view(mut mods: NativeModules, store: PythonObject, address: Int, dims: List[Int], count: Int,
                context: PythonObject) raises -> PythonObject:
    """A read-only C-order float32 Array over `store` (a native memoryview
    of `count` float32 words at `address`), assembled from its slots: the
    view holds the one export for the call, no Array callback runs."""
    var b = mods.module("builtins")
    var slots = neural_array_slots(mods, context)
    var result = py_call(b.object.__new__, context["array_class"])
    var steps = List[Int](length=len(dims), fill=0)
    var step = 4
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        var axis = len(dims) - i - 1
        steps[axis] = step
        step *= dims[axis]
    var typed = py_call(py_call(store.cast, "B").cast, "f")
    py_call(slots["_store"].__set__, result, store)
    py_call(slots["_base"].__set__, result, PythonObject(None))
    py_call(slots["_pin"].__set__, result, PythonObject(None))
    py_call(slots["_mv"].__set__, result, typed)
    py_call(slots["_addr"].__set__, result, PythonObject(address))
    py_call(slots["_readonly"].__set__, result, PythonObject(True))
    py_call(slots["shape"].__set__, result, py_call(b.tuple, py_ints(dims)))
    py_call(slots["dtype"].__set__, result, PythonObject("<f4"))
    py_call(slots["ndim"].__set__, result, PythonObject(len(dims)))
    py_call(slots["size"].__set__, result, PythonObject(count))
    py_call(slots["nbytes"].__set__, result, PythonObject(count * 4))
    py_call(slots["itemsize"].__set__, result, PythonObject(4))
    py_call(slots["strides"].__set__, result, py_call(b.tuple, py_ints(steps)))
    py_call(slots["order"].__set__, result, PythonObject("C"))
    return result


def _block_destination(mut mods: NativeModules, dims: List[Int], count: Int, use: Int,
                       context: PythonObject) raises -> PythonObject:
    """The C-order float32 destination of one copy of `count` words. A
    BLOCK_READ copy never outlives the operation and is written in full
    before any kernel reads it, so it is UNINITIALISED storage (a read-only
    Array view over a native scratch store); a kept copy is a native Array
    (its array.array store has no uninitialised constructor)."""
    if use != BLOCK_READ or count == 0:
        return generation_array(dims)
    var scratch = native_scratch_bytes(count * 4)
    var store = py_call(mods.module("builtins").memoryview, scratch)
    return _block_view(mods, store, native_scratch_address(scratch), dims, count, context)


def admit_block_f32(mut mods: NativeModules, value: PythonObject, name: String, context: PythonObject,
                    use: Int) raises -> PythonObject:
    """The float32 tensor `value` (a mojolearn Array or a native float32
    buffer export) as a C-order float32 Array for `use` (BLOCK_READ,
    BLOCK_OPERATION or BLOCK_RETAINED, above). Other dtypes are refused by
    name, never converted."""
    if py_type(value) is context["array_class"]:
        var captured = neural_array_snapshot(mods, value, context)
        if String(captured.dtype) != "<f4":
            raise Error(_neural_dtype_refusal(name, String(captured.dtype)))
        var dims = List[Int]()
        for i in range(len(captured.shape)):  # driver-loop: over the tensor's shape words
            dims.append(Int(py=captured.shape[i]))
        var count = Int(py=captured.size)
        if String(captured.order) == "C":
            if use != BLOCK_RETAINED:
                return captured
            var owned = generation_array(dims)
            if count > 0:
                memcpy(dest=f32_ptr(Int(py=owned._addr)), src=f32_ptr(Int(py=captured._addr)), count=count)
            _ = captured^
            return owned
        # F-order: the byte-preserving C gather, metadata captured once above
        # and the original owner retained.
        var steps = List[Int]()
        for i in range(len(captured.shape)):  # driver-loop: native admission metadata and retained owner registry
            steps.append(Int(py=captured.strides[i]))
        var result = _block_destination(mods, dims, count, use, context)
        causal_lm_gather_f32(f32_ptr(Int(py=captured._addr)), f32_ptr(Int(py=result._addr)), count, dims, steps)
        _ = captured^
        return result
    admit_native_buffer_exporter(value)
    # One export of the caller's buffer; every word below is read from it.
    var store = py_call(mods.module("builtins").memoryview, value)
    var pin = AdmittedBorrowedBuffer(store)
    if pin.format != "f" and pin.format != "<f" and pin.format != "=f" and pin.format != "@f":
        raise Error(_neural_dtype_refusal(name, pin.format))
    var layout = pin.layout.value().copy()
    if layout.itemsize != 4:
        raise Error("TypeError: neural float32 input requires four-byte elements")
    # The Array class's storage descriptors are admitted before any Array is
    # assembled or allocated for this input.
    _ = neural_array_slots(mods, context)
    if use == BLOCK_READ and layout.contiguous() and layout.count > 0 and layout.address % 4 == 0:
        # 0.8.66 regression C5 (samba-forward 1.55x, mamba1-forward 1.51x,
        # mamba3-forward 1.35x, mamba2-forward 1.27x): read in place, as
        # 0.8.55 did, through a read-only view holding the one export.
        var view = _block_view(mods, store, layout.address, layout.shape, layout.count, context)
        _ = pin^
        return view
    var result = _block_destination(mods, layout.shape, layout.count, use, context)
    layout.copy_to(Int(py=result._addr))
    _ = pin^
    _ = store^
    return result


def admit_block_f32(value: PythonObject, name: String, context: PythonObject, use: Int) raises -> PythonObject:
    # Callers without an operation handle keep their per-call module checks.
    var mods = NativeModules()
    return admit_block_f32(mods, value, name, context, use)
