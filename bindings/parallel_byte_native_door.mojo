# SPDX-License-Identifier: Apache-2.0
"""Native public owner for the existing byte-LM multi-device sessions.

The GPU session remains the numerical implementation. This owner admits and
packs buffers, preserves the portable checkpoint registry, and drives the
byte-LM binding's private schedule descriptor table
(`mojolearn.training.byte_lm_schedule.v1`, bindings/_mojolearn_byte_lm.mojo)
with integer descriptors. Trainer classes are admitted once at import; each
trainer keeps one native owner record, so no per-instance Python callable,
schedule string or mutable attribute selects what runs.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from std.memory import unsafe_memcpy
from std.math import isfinite
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_array, py_ints
from bindings.py_balanced import py_call, py_contains, py_fields, py_get, py_setattr, py_setitem, py_type
from bindings.native_json_door import native_json_parse
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.native_transport_door import _transport_json, transport_initialize
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_connect, neural_descriptor_provider_id
from bindings.parallel_byte_native_run import (
    ParallelByteSession,
    PARALLEL_SCHEDULE_ABI_NAME, PARALLEL_SCHEDULE_OPERATIONS,
    PARALLEL_REPLICA, PARALLEL_MODEL_POOL, PARALLEL_OFFLOAD, PARALLEL_SCALAR_LR,
    parallel_session_apply_gradient, parallel_session_close, parallel_session_export,
    parallel_session_fold, parallel_session_fold_export, parallel_session_open,
    parallel_session_openable, parallel_session_ownership, parallel_session_ownership_capacity,
    parallel_session_ownership_width, parallel_session_require_split, parallel_session_rollback,
    parallel_session_schedule_name, parallel_session_set_lr, parallel_session_shard_gradient,
    parallel_session_step_begin, parallel_session_step_commit,
)
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from training.parallel_byte_impl import (
    _parallel_span,
    parallel_copy_words,
    parallel_require_binary_i32,
    parallel_require_finite_f32,
    parallel_require_nonnegative_f32,
)
from core.shared_slot import SharedSlot
from core.entry_table import EntryRef
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_published, byte_lm_vendor_text_word, BYTE_LM_TABLE_TRAINER
from training.byte_lm_glue_abi import (
    BYTE_LM_GLUE_ABI, BG_OP_COUNT, BG_NUMERIC_MODE, BG_VENDOR, BG_PROFILE, BG_SUFFIX, BG_CONFIG_PROFILE,
    byte_lm_glue_ref, byte_lm_glue_read_text, byte_lm_glue_read_int,
)
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_BYTE_OPN_ADMITTED_STATE,
    PARALLEL_BYTE_OPN_ADMIT_CLASS,
    PARALLEL_BYTE_OPN_APPLY_GRADIENT,
    PARALLEL_BYTE_OPN_CHECKPOINT,
    PARALLEL_BYTE_OPN_CLOSE,
    PARALLEL_BYTE_OPN_ENTER,
    PARALLEL_BYTE_OPN_EXPORT_GRADIENTS,
    PARALLEL_BYTE_OPN_EXPORT_RAW,
    PARALLEL_BYTE_OPN_FOLD_ADD,
    PARALLEL_BYTE_OPN_FOLD_EXPORT,
    PARALLEL_BYTE_OPN_FOLD_RESET,
    PARALLEL_BYTE_OPN_FROM_CHECKPOINT,
    PARALLEL_BYTE_OPN_HAS_DEVICE_FOLD,
    PARALLEL_BYTE_OPN_INIT,
    PARALLEL_BYTE_OPN_INITIALIZE,
    PARALLEL_BYTE_OPN_MODEL_OWNERSHIP,
    PARALLEL_BYTE_OPN_OPEN,
    PARALLEL_BYTE_OPN_OWNERSHIP,
    PARALLEL_BYTE_OPN_REQUIRE_FOLD,
    PARALLEL_BYTE_OPN_SET_LR,
    PARALLEL_BYTE_OPN_SHARD_GRADIENT,
    PARALLEL_BYTE_OPN_SHARD_GRADIENT_FOLD,
    PARALLEL_BYTE_OPN_STATE,
    PARALLEL_BYTE_OPN_STEP,
    PARALLEL_BYTE_OPN_TRAIN_STEP,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


# The schedule descriptor ABI words and the typed session are in
# bindings/parallel_byte_native_run.mojo (RUN); this file is UNPACK and PACK.


struct ParallelByteBoundary(Defaultable, Movable):
    var bi: PythonObject
    var array_cls: PythonObject
    var shape_cls: PythonObject
    var modes: PythonObject
    var address: PythonObject
    var char: PythonObject
    var index: PythonObject
    var vendor: String
    # Import-time admitted trainer classes, one per schedule (replica,
    # model_pool, offload); None until that module's class is admitted.
    var classes: List[PythonObject]

    def __init__(out self):
        self.bi = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.shape_cls = PythonObject(None)
        self.modes = PythonObject(None)
        self.address = PythonObject(None)
        self.char = PythonObject(None)
        self.index = PythonObject(None)
        self.vendor = String()
        self.classes = List[PythonObject]()
        for _ in range(3):  # driver-loop: over 3: replica, model_pool and offload schedules
            self.classes.append(PythonObject(None))


comptime PARALLEL_BYTE_BOUNDARY = SharedSlot[ParallelByteBoundary, "MojolearnParallelByteBoundaryV1"]


struct ParallelByteOwner(Movable, Writable):
    """One trainer's native owner. Stored once in the instance dictionary as
    `_native`; every operation reads this record, never instance attributes.
    `native` is the typed state RUN works on (bindings/parallel_byte_native_run.mojo).
    The Python objects here are boundary pins only: the schedule provider and
    its factory entries, the live session object, the caller's device values
    and the admitted state dictionary (a boundary copy of the step and the
    learning rate, refreshed by `_parallel_publish` before it is read).
    `devices`, `logical_shards`, `pool_optimizer`, `_shape` and `_session`
    on the instance are read-only mirrors for callers."""
    var owner_id: Int
    var native: ParallelByteSession
    var provider: PythonObject
    var create: PythonObject
    var handle_entry: PythonObject
    var session: PythonObject
    var device_objects: PythonObject
    var state: PythonObject
    var state_addresses: List[Int]
    var shape: PythonObject

    def __init__(out self, owner_id: Int, schedule: Int, table: PythonObject, pool_optimizer: Bool,
                 var devices: List[Int], device_objects: PythonObject, logical_shards: Int,
                 shape: PythonObject, state: PythonObject) raises:
        self.owner_id = owner_id
        var nk = parallel_keys()
        var provider = py_item(table, nk[].k_provider)
        # UNPACK, once per trainer: the provider id word, the shape words, the
        # optimizer scalars and the committed step leave their Python objects
        # here; no operation reads them from Python again.
        var provider_id = 0
        if provider is not None:
            provider_id = neural_descriptor_provider_id(provider)
        var dims = List[Int]()
        var shape_dims = py_item(shape, nk[].k_dims)
        for i in range(len(shape_dims)):  # driver-loop: over dims: nine or ten admitted shape words
            dims.append(Int(py=shape_dims[i]))
        var config = py_item(state, nk[].k_config)
        var scalars = List[Float64]()
        for key in [String("lr"), String("beta1"), String("beta2"), String("eps"),  # driver-loop: over keys: eight optimizer scalars
                    String("weight_decay"), String("momentum"), String("dampening"), String("max_norm")]:
            scalars.append(Float64(py=config[key]))
        self.native = ParallelByteSession(
            provider_id, schedule, pool_optimizer, devices^, logical_shards, dims^,
            Int(py=py_item(shape, nk[].k_n_total)), Int(py=py_item(shape, nk[].k_n_tensors)),
            Int(py=py_item(state, nk[].k_completed_steps)),
            Int(py=py_item(config, nk[].k_kind)), Bool(py=py_item(config, nk[].k_nesterov)), scalars^)
        self.provider = provider
        self.create = py_item(table, nk[].k_create)
        self.handle_entry = py_item(table, nk[].k_handle)
        self.session = PythonObject(None)
        self.device_objects = device_objects
        self.state = state
        # Captured once from the freshly admitted native copies.
        self.state_addresses = List[Int]()
        for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # driver-loop: over keys: four state arrays
            self.state_addresses.append(Int(py=state[key]._addr))
        self.shape = shape

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParallelByteOwner(schedule=", self.native.schedule, ", open=", self.native.handle != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _parallel_compiled(binding: PythonObject, name: String) raises -> PythonObject:
    var bi = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()[].bi
    var entry = PythonObject(None)
    try:
        entry = py_call(bi.getattr, binding, name, PythonObject(None))
    except:
        # _HostBinding refuses GPU names by NoCpuEntry; resolve that
        # connection-time proxy once and keep the native refusal at open.
        entry = PythonObject(None)
    if entry is not None and py_type(entry) is not py_type(bi.len):
        raise Error("parallel: session entry must be compiled native code")
    return entry


def _parallel_glue(mode: String, vendor: String) raises -> EntryRef:
    """The glue entry of the byte LM image of `mode` and `vendor` (the trainer
    table key core/byte_lm_entry_table.mojo publishes); an image without a
    published table refuses as the former Python suffix probe did."""
    var vendor_word = byte_lm_vendor_text_word(vendor)
    if vendor_word < 0 or (mode != "identical" and mode != "fast"):
        raise Error("parallel: binding has no native arithmetic profile")
    var key = ByteLmEntryKey(1 if mode == "identical" else 0, vendor_word)
    if not byte_lm_table_published[BYTE_LM_TABLE_TRAINER](key, 1, String(BYTE_LM_GLUE_ABI), BG_OP_COUNT):
        raise Error("parallel: binding has no native arithmetic profile")
    return byte_lm_glue_ref(key)


def _parallel_glue_text(gl: EntryRef, op: Int, var words: List[Int]) raises -> String:
    var text = byte_lm_glue_read_text(gl, op, words^)
    if not text:
        raise Error("parallel: binding has no native arithmetic profile")
    return text.value().copy()


def parallel_byte_initialize(array_cls: PythonObject, shape_cls: PythonObject, modes: PythonObject, expected_vendor: String) raises:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    p[].bi = Python.import_module("builtins")
    p[].array_cls = array_cls
    p[].shape_cls = shape_cls
    p[].modes = p[].bi.dict()
    var ctypes = Python.import_module("ctypes")
    p[].address = ctypes.addressof
    p[].char = ctypes.c_char
    p[].index = Python.import_module("_operator").index
    p[].vendor = expected_vendor
    transport_initialize()
    if py_type(p[].address) is not py_type(p[].bi.len):
        raise Error("parallel: buffer address must be a native C entry")
    if py_type(modes) is not p[].bi.dict:
        raise Error("TypeError: parallel: installed modes must be a dict")
    # Share the native Array allocator without resetting any generation model
    # classes already admitted by the generation boundary.
    var arrays = GENERATION_BOUNDARY.get_or_create_ptr()
    arrays[].bi = p[].bi
    arrays[].arrays = Python.import_module("array").array
    arrays[].array_cls = array_cls
    var mode_keys = py_call(p[].bi.list, modes)
    var nk = parallel_keys()
    for k in range(len(mode_keys)):  # driver-loop: over installed numerical-mode ABI tables
        var mode = mode_keys[k]
        if py_type(mode) is not p[].bi.str:
            raise Error("TypeError: parallel: mode names must be str")
        var binding = modes[mode]
        var table = p[].bi.dict()
        # The mode/vendor/profile witness is read through the trainer table's
        # glue entry (core/byte_lm_entry_table.mojo; published by both byte LM
        # images at module init), the same read-back the trainer door's _load
        # makes. The Python metadata entries it replaces were deleted from the
        # device image by 1bdaa5e00f / 6890e9658f (lane calltable-byte-lm), so
        # this import-time probe of `byte_lm_arithmetic_suffix` refused every
        # GPU import ("parallel: binding has no native arithmetic profile").
        var gl = _parallel_glue(String(mode), expected_vendor)
        var suffix = _parallel_glue_text(gl, BG_SUFFIX, List[Int]())
        py_setitem(table, nk[].k_suffix, PythonObject(suffix))
        if byte_lm_glue_read_int(gl, BG_NUMERIC_MODE, List[Int]()) != (1 if String(mode) == "identical" else 0) or _parallel_glue_text(gl, BG_VENDOR, List[Int]()) != expected_vendor:
            raise Error("parallel: native mode/vendor mismatch")
        if _parallel_glue_text(gl, BG_PROFILE, List[Int]()) != "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1" + suffix:
            raise Error("parallel: native default profile mismatch")
        # The private schedule descriptor table and its session factory. A
        # binding without them (the CPU host binding) keeps provider None and
        # refuses at open, so import never fails and never touches a device.
        var factory = _parallel_compiled(binding, "byte_lm_schedule_api_v1")
        var create = _parallel_compiled(binding, "byte_lm_schedule_create_v1")
        var handle = _parallel_compiled(binding, "byte_lm_schedule_handle_v1")
        var provider = PythonObject(None)
        if factory is not None:
            if create is None or handle is None or create.__self__ is not factory.__self__ or handle.__self__ is not factory.__self__:
                raise Error("parallel: schedule factory entries must come from one compiled image")
            provider = neural_descriptor_connect(factory, String(PARALLEL_SCHEDULE_ABI_NAME), PARALLEL_SCHEDULE_OPERATIONS)
        else:
            create = PythonObject(None)
            handle = PythonObject(None)
        py_setitem(table, nk[].k_provider, provider)
        py_setitem(table, nk[].k_create, create)
        py_setitem(table, nk[].k_handle, handle)
        py_setitem(p[].modes, mode, table)


def _parallel_schedule_index(name: String) raises -> Int:
    if name == "replica":
        return PARALLEL_REPLICA
    if name == "model_pool":
        return PARALLEL_MODEL_POOL
    if name == "offload":
        return PARALLEL_OFFLOAD
    raise Error("parallel: unknown byte-LM schedule " + name)


def parallel_byte_admit_class(cls: PythonObject, name: String) raises:
    """Import-time admission of one trainer class for one schedule: an exact
    `type` whose hierarchy has no Python attribute, storage or finalizer
    hooks. The replica class derives from object; the layer-pool and offload
    classes derive directly from the admitted replica class."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if p[].bi is None:
        raise Error("RuntimeError: parallel: ABI is not initialized")
    var schedule = _parallel_schedule_index(name)
    if py_type(cls) is not p[].bi.type:
        raise Error("TypeError: parallel: trainer class must be an exact type")
    var bases = py_call(p[].bi.object.__getattribute__, cls, "__bases__")
    if len(bases) != 1:
        raise Error("TypeError: parallel: trainer class must have one base")
    if schedule == PARALLEL_REPLICA:
        if bases[0] is not p[].bi.object:
            raise Error("TypeError: parallel: replica trainer class must derive from object")
    elif p[].classes[PARALLEL_REPLICA] is None or bases[0] is not p[].classes[PARALLEL_REPLICA]:
        raise Error("TypeError: parallel: schedule trainer class must derive from the admitted replica class")
    var forbidden: List[String] = ["__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__", "__slots__", "_native"]
    var mro = py_call(p[].bi.object.__getattribute__, cls, "__mro__")
    for i in range(len(mro) - 1):  # driver-loop: over the admitted trainer class hierarchy (object excluded)
        var namespace = py_call(p[].bi.object.__getattribute__, mro[i], "__dict__")
        for hook in forbidden:  # driver-loop: over forbidden: seven storage and lifecycle hook names
            if py_contains(namespace, PythonObject(hook)):
                raise Error("TypeError: parallel: trainer class refuses Python storage or finalizer hook " + hook)
    for i in range(3):  # small-loop(3: admitted schedule classes)
        if i != schedule and p[].classes[i] is cls:
            raise Error("TypeError: parallel: trainer class is admitted for another schedule")
    # A module reload admits its new class object in place of the old one.
    p[].classes[schedule] = cls


def _parallel_class_schedule(cls: PythonObject) raises -> Int:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    for i in range(3):  # small-loop(3: admitted schedule classes)
        if p[].classes[i] is not None and p[].classes[i] is cls:
            return i
    raise Error("TypeError: parallel: trainer must be an exact admitted byte-LM trainer class")


def _parallel_record(owner: PythonObject) raises -> PythonObject:
    """The exact native owner record of an admitted trainer instance."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var schedule = _parallel_class_schedule(py_type(owner))
    var nk = parallel_keys()
    var record = py_get(py_fields(owner), nk[].k__native)
    if record is None:
        raise Error("RuntimeError: parallel trainer is not initialized")
    var owner_id = Int(py=py_call(p[].bi.id, owner))
    try:
        var rec = record.downcast_value_ptr[ParallelByteOwner]()
        if rec[].owner_id != owner_id or rec[].native.schedule != schedule:
            raise Error("foreign")
    except:
        raise Error("TypeError: parallel trainer storage does not belong to this trainer")
    return record


def _parallel_int(value: PythonObject, minimum: Int, maximum: Int, label: String, strict: Bool = True) raises -> Int:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    if py_type(value) is bi.bool or String(py_type(value).__name__) == "bool_" or (strict and py_type(value) is not bi.int):
        raise Error(label + " must be an integer")
    var n = Int(py=value) if strict else Int(py=py_call(p[].index, value))
    if n < minimum or n > maximum:
        raise Error(label + " outside admitted range")
    return n


def _parallel_float(value: PythonObject, label: String) raises -> Float32:
    var bi = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()[].bi
    if py_type(value) is bi.bool or String(py_type(value).__name__) == "bool_":
        raise Error(label + " must be a finite float32 scalar")
    var number = Float64(py=value)
    if not isfinite(number) or abs(number) > Float64(3.4028234663852886e38):
        raise Error(label + " must be finite float32")
    return Float32(number)


def _parallel_view(value: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if py_type(value) is p[].array_cls:
        return value._mv
    var view = PythonObject(None)
    try:
        view = py_call(p[].bi.memoryview, value)
    except error:
        # An object without the buffer protocol is the wrong type; keep
        # Python's TypeError instead of the boundary's default ValueError.
        raise Error("TypeError: " + String(error))
    return view


def _parallel_array(value: PythonObject, dims: List[Int], integer: Bool = False, finite: Bool = True) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var own_array = py_type(value) is p[].array_cls
    var view = _parallel_view(value)
    var shape = value.shape if own_array else view.shape
    var fmt = String(view.format)
    if (integer and fmt != "i" and fmt != "=i" and fmt != "<i") or (
        not integer and fmt != "f" and fmt != "=f" and fmt != "<f") or Int(py=view.itemsize) != 4:
        raise Error("TypeError: parallel input requires exact float32 or int32 buffer dtype")
    if len(shape) != len(dims):
        raise Error("parallel input has wrong rank")
    for i in range(len(dims)):  # driver-loop: over dims: array rank words
        if Int(py=shape[i]) != dims[i]:
            raise Error("parallel input has wrong shape")
    var output = generation_array(dims, "i" if integer else "f")
    var n = Int(py=output.size)
    var dst_addr = Int(py=output._addr)
    var transposed = own_array and String(value.order) == "F" and len(dims) > 1
    if Bool(py=view.c_contiguous):
        # A contiguous buffer is read in place: its bytes in storage order
        # are what `tobytes("C")` copied into the staging store, so the one
        # native copy below reads them from the caller's buffer directly
        # (it was tobytes, frombytes, then this copy).
        if Int(py=view.nbytes) != n * 4:
            raise Error("parallel input byte count differs from shape")
        if n > 0:
            parallel_copy_words(native_buffer_address(view), dst_addr, dims, n, transposed)
        _ = view
        _ = value
    else:
        var store = py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].arrays, "i" if integer else "f")
        _ = py_call(store.frombytes, py_call(view.tobytes, "C"))
        if len(store) != n:
            raise Error("parallel input byte count differs from shape")
        var src_addr = Int(py=store.buffer_info()[0])
        parallel_copy_words(src_addr, dst_addr, dims, n, transposed)
        _ = store^
    if not integer and finite:
        parallel_require_finite_f32(Int(py=output._addr), n)
    return output


def _parallel_word_format(format: String, integer: Bool) -> Bool:
    if integer:
        return format == "i" or format == "=i" or format == "<i"
    return format == "f" or format == "=f" or format == "<f"


def _parallel_pin_words(value: PythonObject, dims: List[Int], n: Int, integer: Bool, raw: Bool,
                        mut pins: List[AdmittedBorrowedBuffer]) raises -> Int:
    # One PyObject_GetBuffer of an exact native exporter (the shared
    # borrowed-buffer admission): format, item size, shape and strides come
    # from the Py_buffer words, not from memoryview attributes. Returns the
    # caller's address when the storage already is the C-contiguous,
    # four-aligned int32/float32 cells the session reads, else -1 (the copy
    # route then admits it, with its own refusal messages).
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var exporter = value
    var own_array = py_type(value) is p[].array_cls
    if own_array:
        # An Array in Fortran order is transposed by the copy route.
        if len(dims) > 1 and String(value.order) == "F":
            return -1
        exporter = value._mv
    admit_native_buffer_exporter(exporter)
    var pin = AdmittedBorrowedBuffer(exporter)
    var layout = pin.layout.value().copy()
    var fits = layout.contiguous() and layout.address % 4 == 0 and layout.address != 0
    if raw:
        # Raw float32 bytes (the transport payload): n words of four bytes.
        fits = fits and layout.itemsize == 1 and layout.count == n * 4
    else:
        fits = fits and layout.itemsize == 4 and layout.count == n and _parallel_word_format(pin.format, integer)
        if own_array:
            var shape = value.shape
            fits = fits and len(shape) == len(dims)
            if fits:
                for i in range(len(dims)):  # driver-loop: over dims: array rank words
                    fits = fits and Int(py=shape[i]) == dims[i]
        else:
            fits = fits and len(layout.shape) == len(dims)
            if fits:
                for i in range(len(dims)):  # small-loop(dims: array rank words)
                    fits = fits and layout.shape[i] == dims[i]
    if not fits:
        pin.release()
        return -1
    var address = layout.address
    pins.append(pin^)
    return address


def _parallel_borrow(value: PythonObject, dims: List[Int], integer: Bool,
                     mut pins: List[AdmittedBorrowedBuffer], mut copies: List[PythonObject],
                     finite: Bool = True) raises -> Int:
    """The address of `value`'s int32/float32 cells for one session call.

    pass 3 (N102, N111, N157): a C-contiguous buffer of the exact dtype and
    shape is read in place: the caller's storage stays pinned in `pins`
    until the caller releases it after the call, and the session's one
    memcpy/upload reads it directly (it was a memoryview, six attribute
    reads, a new Array and a host copy per buffer). Anything else (strided,
    Fortran-ordered, misaligned, wrong dtype or shape, Python buffer hooks)
    takes the copy route `_parallel_array`, which keeps its refusal
    messages; the copy stays alive in `copies`. A float input also admits
    raw float32 bytes (`bytes`/`bytearray`), as the gradient doors did.
    With `finite` False the copy route skips the host scan: the session's
    device scan refuses non-finite values (N157). Same bytes reach the
    same kernels on every route."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var n = 1
    for d in dims:  # small-loop(dims: array rank words)
        n *= d
    var cls = py_type(value)
    var raw = not integer and (cls is p[].bi.bytes or cls is p[].bi.bytearray)
    if n > 0:
        var address = -1
        try:
            address = _parallel_pin_words(value, dims, n, integer, raw, pins)
        except:
            # Not an exact native exporter: the copy route admits it (or
            # refuses it) as before.
            address = -1
        if address > 0:
            return address
    var copied = _parallel_array(_parallel_float_bytes(value) if raw else value, dims, integer, finite)
    var copy_address = Int(py=copied._addr)
    copies.append(copied)
    return copy_address


def _parallel_release(mut pins: List[AdmittedBorrowedBuffer]) raises:
    for i in range(len(pins)):  # small-loop(pins: one per borrowed input buffer of the call)
        pins[i].release()


def _parallel_target(value: PythonObject, count: Int, integer: Bool, mut pins: List[PythonObject]) raises -> Int:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var view = _parallel_view(value)
    var shape = value.shape if py_type(value) is p[].array_cls else view.shape
    var fmt = String(view.format)
    if (integer and fmt != "i") or (not integer and fmt != "f") or Int(py=view.itemsize) != 4:
        raise Error("TypeError: export target has wrong dtype")
    if len(shape) != 1 or Int(py=shape[0]) != count or not Bool(py=view.c_contiguous):
        raise Error("export target must have the exact contiguous one-dimensional shape")
    if Bool(py=view.readonly):
        raise Error("export target is read-only")
    if py_type(value) is p[].array_cls:
        if Bool(py=value._readonly):
            raise Error("export target is read-only")
        pins.append(value)
        return Int(py=value._addr)
    # Both calls are CPython C operations cached at ABI setup, not ctypes.py
    # helpers, ndarray.ctypes, Array.__buffer__ or any package conversion.
    var pin = py_call(p[].char.from_buffer, view)
    var address = Int(py=py_call(p[].address, pin))
    pins.append(pin)
    return address


def _parallel_schedule_check(value: PythonObject, depth: Int, mut budget: Int) raises:
    var bi = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()[].bi
    budget -= 1
    if budget < 0 or depth > 8:
        raise Error("Byte-LM schedule is too complex")
    var cls = py_type(value)
    if cls is bi.dict:
        if len(value) > 128:
            raise Error("Byte-LM schedule has too many keys")
        var keys = py_call(bi.list, value)
        for i in range(len(keys)):  # small-loop(keys: schedule JSON keys, at most 128)
            if py_type(keys[i]) is not bi.str or len(keys[i]) > 2048:
                raise Error("Byte-LM schedule keys exceed bounds")
            _parallel_schedule_check(value[keys[i]], depth + 1, budget)
    elif cls is bi.list:
        if len(value) > 128:
            raise Error("Byte-LM schedule list exceeds its bound")
        for i in range(len(value)):  # small-loop(value: schedule JSON list, at most 128)
            _parallel_schedule_check(value[i], depth + 1, budget)
    elif cls is bi.str:
        if len(value) > 2048:
            raise Error("Byte-LM schedule string exceeds its bound")
    elif cls is bi.int:
        if Int(py=value.bit_length()) > 4096:
            raise Error("Byte-LM schedule integer exceeds its bound")
    elif value is not None and cls is not bi.float and cls is not bi.bool:
        raise Error("Byte-LM schedule requires JSON values")


def _parallel_schedule(value: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()[].bi
    if py_type(value) is not bi.dict or len(value) == 0:
        raise Error("Byte-LM data_schedule must be a nonempty JSON object")
    var budget = 256
    _parallel_schedule_check(value, 0, budget)
    var raw = _transport_json(value)
    if raw.byte_length() > 16384:
        raise Error("Byte-LM schedule exceeds 16384 bytes")
    return native_json_parse(PythonObject(raw))


def _parallel_shape(state: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var names = [String("batch"), String("length"), String("d_model"), String("n_heads"),
                 String("n_kv"), String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]
    var defaults: List[Int] = [2, 32, 32, 4, 2, 8, 64, 2, 256]
    var nk = parallel_keys()
    var shape = py_get(state, nk[].k_model_shape, PythonObject(None))
    if shape is not None:
        if py_type(shape) is not p[].bi.dict:
            raise Error("Byte-LM model_shape must be a dict")
        var keys = py_call(p[].bi.list, shape)
        for i in range(len(keys)):  # small-loop(keys: model_shape dimension names)
            var key = String(keys[i])
            if key != "chunked_lm_head_v2" and key not in names:
                raise Error("Byte-LM model_shape has unknown dimensions")
        for i in range(7):  # driver-loop: over names: seven required dimension names
            if not py_contains(shape, PythonObject(names[i])):
                raise Error("Byte-LM model_shape has missing dimensions")
        if py_contains(shape, nk[].k_n_layers) != py_contains(shape, nk[].k_vocab_size):
            raise Error("Byte-LM model_shape must specify both layers and vocabulary")
    var dims = p[].bi.list()
    var normalized = p[].bi.dict()
    for i in range(len(names)):  # driver-loop: over names: nine dimension names
        var value = PythonObject(defaults[i]) if shape is None else py_get(shape, PythonObject(names[i]), PythonObject(defaults[i]))
        var n = _parallel_int(value, 1, 1048576, names[i], False)
        _ = py_call(dims.append, n)
        py_setitem(normalized, names[i], PythonObject(n))
    var chunked = PythonObject(False) if shape is None else py_get(shape, nk[].k_chunked_lm_head_v2, PythonObject(False))
    if py_type(chunked) is not p[].bi.bool:
        raise Error("chunked_lm_head_v2 must be bool")
    py_setitem(normalized, nk[].k_chunked_lm_head_v2, chunked)
    var b = Int(py=dims[0])
    var l = Int(py=dims[1])
    var dm = Int(py=dims[2])
    var h = Int(py=dims[3])
    var kv = Int(py=dims[4])
    var hd = Int(py=dims[5])
    var ff = Int(py=dims[6])
    var layers = Int(py=dims[7])
    var vocab = Int(py=dims[8])
    if l > 8192 or dm != h * hd or h % kv != 0 or hd % 2 != 0:
        raise Error("Byte-LM requires L <= 8192, DM = H*HD, H divisible by KV and even HD")
    if Bool(py=chunked) and vocab < 2:
        raise Error("chunked_lm_head_v2 requires vocabulary size >= 2")
    _ = _parallel_span([b, l, vocab])
    _ = _parallel_span([b, l, dm])
    _ = _parallel_span([b, l, ff])
    _ = _parallel_span([b, h, l, l])
    _ = _parallel_span([b, kv, l, hd])
    _ = _parallel_span([b, l + 1])
    var names_out = p[].bi.list()
    var shapes_out = p[].bi.list()
    var offsets = p[].bi.list()
    var total = 0
    _ = py_call(offsets.append, 0)
    var block_names = [String("norm1_w"), String("w_q"), String("w_k"), String("w_v"), String("w_o"),
                       String("norm2_w"), String("w_gate"), String("w_up"), String("w_down")]
    for tensor in range(2 + layers * 9):  # driver-loop: over the portable model parameter registry
        var key = String()
        var tensor_shape = List[Int]()
        if tensor == 0 or tensor == 1 + layers * 9:
            key = "embed" if tensor == 0 else "lm_head"
            tensor_shape.append(vocab)
            tensor_shape.append(dm)
        else:
            var local = (tensor - 1) % 9
            key = "block" + String((tensor - 1) // 9) + "." + block_names[local]
            tensor_shape.append(dm if local <= 5 else (dm if local == 8 else ff))
            if local != 0 and local != 5:
                tensor_shape.append(ff if local == 8 else dm)
                if local == 2 or local == 3:
                    tensor_shape[0] = kv * hd
        var count = _parallel_span(tensor_shape)
        if count > 2147483647 or total > 2147483647 - count:
            raise Error("Byte-LM parameter registry exceeds signed 32-bit indexing")
        total += count
        _ = py_call(names_out.append, key)
        _ = py_call(shapes_out.append, py_ints(tensor_shape))
        _ = py_call(offsets.append, total)
    var base = String("mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1")
    if Bool(py=chunked) or b != 2 or l != 32 or dm != 32 or h != 4 or kv != 2 or hd != 8 or ff != 64 or layers != 2 or vocab != 256:
        var suffix = "-v256-blocks2.fp32.v2" if layers == 2 and vocab == 256 else "-v" + String(vocab) + "-blocks" + String(layers) + ".fp32.v3"
        base = "mojolearn.byte-lm.b" + String(b) + "-l" + String(l) + "-d" + String(dm) + "-h" + String(h) + "-kv" + String(kv) + "-hd" + String(hd) + "-ff" + String(ff) + suffix
        if Bool(py=chunked):
            base += "-lmhead-chunk256-v2"
    if Bool(py=chunked):
        _ = py_call(dims.append, 1)
    var result = p[].bi.dict()
    py_setitem(result, nk[].k_dims, dims)
    py_setitem(result, nk[].k_model_shape, normalized)
    py_setitem(result, nk[].k_names, names_out)
    py_setitem(result, nk[].k_shapes, shapes_out)
    py_setitem(result, nk[].k_offsets, offsets)
    py_setitem(result, nk[].k_n_total, PythonObject(total))
    py_setitem(result, nk[].k_n_tensors, PythonObject(2 + layers * 9))
    py_setitem(result, nk[].k_profile_base, PythonObject(base))
    return result


def _parallel_validate_state(value: PythonObject, shape: PythonObject, exported: Bool = False) raises -> PythonObject:
    # exported: the four state arrays are the exact contiguous float32/int32
    # Arrays _parallel_export just allocated and filled. They are kept as
    # they are (no second copy each); the value checks still run on them.
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if py_type(value) is not p[].bi.dict:
        raise Error("Byte-LM state must be a dict")
    var names = [String("schema"), String("profile"), String("numeric_mode"), String("parameter_names"),
                 String("parameter_shapes"), String("parameter_offsets"), String("parameters"), String("m"),
                 String("v"), String("flags"), String("completed_steps"), String("next_batch_index"),
                 String("config"), String("data_schedule")]
    var nk = parallel_keys()
    var has_shape = py_contains(value, nk[].k_model_shape)
    if len(value) != len(names) + Int(has_shape):
        raise Error("Byte-LM state has missing or unknown fields")
    for name in names:  # driver-loop: over names: fourteen state field names
        if not py_contains(value, PythonObject(name)):
            raise Error("Byte-LM state is missing " + name)
    var mode = String(py_item(value, nk[].k_numeric_mode))
    if not py_contains(p[].modes, PythonObject(mode)):
        raise Error("Byte-LM numeric mode is not installed")
    var table = p[].modes[mode]
    var shape_words = List[Int]()
    var dims = py_item(shape, nk[].k_dims)
    for i in range(len(dims)):  # small-loop(dims: nine or ten admitted shape words)
        shape_words.append(Int(py=dims[i]))
    if len(shape_words) == 9:
        shape_words.append(0)  # the glue's tenth word: chunked head off
    var gl = _parallel_glue(mode, p[].vendor)
    if _parallel_glue_text(gl, BG_CONFIG_PROFILE, shape_words^) != String(py_item(shape, nk[].k_profile_base)) + String(py_item(table, nk[].k_suffix)):
        raise Error("Byte-LM native runtime shape/profile mismatch")
    if String(py_item(value, nk[].k_schema)) != "mojolearn.small-byte-lm-state.v1" or String(py_item(value, nk[].k_profile)) != String(py_item(shape, nk[].k_profile_base)) + String(py_item(table, nk[].k_suffix)):
        raise Error("Byte-LM state profile/registry/mode mismatch")
    if not Bool(py=py_item(value, nk[].k_parameter_names) == py_item(shape, nk[].k_names)) or not Bool(py=py_item(value, nk[].k_parameter_shapes) == py_item(shape, nk[].k_shapes)) or not Bool(py=py_item(value, nk[].k_parameter_offsets) == py_item(shape, nk[].k_offsets)):
        raise Error("Byte-LM state parameter registry mismatch")
    var cfg = py_item(value, nk[].k_config)
    var cfg_names = [String("kind"), String("lr"), String("beta1"), String("beta2"), String("eps"),
                     String("weight_decay"), String("momentum"), String("dampening"), String("nesterov"), String("max_norm")]
    if py_type(cfg) is not p[].bi.dict or len(cfg) != len(cfg_names):
        raise Error("Byte-LM optimizer configuration mismatch")
    for name in cfg_names:  # driver-loop: over cfg_names: ten optimizer field names
        if not py_contains(cfg, PythonObject(name)):
            raise Error("Byte-LM optimizer configuration is missing " + name)
    if _parallel_int(py_item(cfg, nk[].k_kind), 2, 2, "optimizer kind") != 2 or py_type(py_item(cfg, nk[].k_nesterov)) is not p[].bi.bool or Bool(py=py_item(cfg, nk[].k_nesterov)):
        raise Error("Byte-LM first profile requires AdamW without Nesterov")
    var config = p[].bi.dict()
    py_setitem(config, nk[].k_kind, PythonObject(2))
    py_setitem(config, nk[].k_nesterov, PythonObject(False))
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay"),  # driver-loop: over names: eight optimizer scalars
                 String("momentum"), String("dampening"), String("max_norm")]:
        var number = _parallel_float(cfg[name], name)
        if ((name == "lr" or name == "eps") and number <= 0) or (name == "weight_decay" and number < 0):
            raise Error("Byte-LM requires lr > 0, eps > 0 and weight_decay >= 0")
        if (name == "beta1" or name == "beta2") and (number < 0 or number >= 1):
            raise Error("Byte-LM betas must remain in [0, 1) in float32")
        if (name == "momentum" or name == "dampening" or name == "max_norm") and number != 0:
            raise Error("Byte-LM first profile requires AdamW without clipping or SGD options")
        py_setitem(config, name, PythonObject(Float64(number)))
    var completed = _parallel_int(py_item(value, nk[].k_completed_steps), 0, 999999, "completed_steps", False)
    if _parallel_int(py_item(value, nk[].k_next_batch_index), 0, 999999, "next_batch_index", False) != completed:
        raise Error("Byte-LM schedule cursor must match completed_steps")
    var result = py_call(p[].bi.dict, value)
    var n = Int(py=py_item(shape, nk[].k_n_total))
    var flags = PythonObject(None)
    if exported:
        for name in [String("parameters"), String("m"), String("v")]:  # driver-loop: over names: three state arrays
            parallel_require_finite_f32(Int(py=value[name]._addr), n)
            py_setitem(result, name, value[name])
        flags = py_item(value, nk[].k_flags)
    else:
        for name in [String("parameters"), String("m"), String("v")]:  # small-loop(names: three state arrays)
            py_setitem(result, name, _parallel_array(value[name], [n]))
        flags = _parallel_array(py_item(value, nk[].k_flags), [Int(py=py_item(shape, nk[].k_n_tensors))], True)
    parallel_require_nonnegative_f32(Int(py=py_item(result, nk[].k_v)._addr), n)
    parallel_require_binary_i32(Int(py=flags._addr), Int(py=py_item(shape, nk[].k_n_tensors)))
    py_setitem(result, nk[].k_flags, flags)
    py_setitem(result, nk[].k_config, config)
    py_setitem(result, nk[].k_completed_steps, PythonObject(completed))
    py_setitem(result, nk[].k_next_batch_index, PythonObject(completed))
    py_setitem(result, nk[].k_data_schedule, _parallel_schedule(py_item(value, nk[].k_data_schedule)))
    var vocabulary = py_get(py_item(result, nk[].k_data_schedule), nk[].k_vocabulary, PythonObject(None))
    if vocabulary is not None:
        if py_type(vocabulary) is not p[].bi.dict or py_type(py_get(vocabulary, nk[].k_n_vocab)) is not p[].bi.int or py_type(py_get(vocabulary, nk[].k_sha256)) is not p[].bi.str:
            raise Error("Byte-LM data schedule vocabulary descriptor is invalid")
        if Int(py=py_item(vocabulary, nk[].k_n_vocab)) != Int(py=py_item(shape, nk[].k_dims)[8]):
            raise Error("Byte-LM data schedule vocabulary does not match model")
    py_setitem(result, nk[].k_parameter_names, py_call(p[].bi.list, py_item(shape, nk[].k_names)))
    py_setitem(result, nk[].k_parameter_shapes, native_json_parse(PythonObject(_transport_json(py_item(shape, nk[].k_shapes)))))
    py_setitem(result, nk[].k_parameter_offsets, py_call(p[].bi.list, py_item(shape, nk[].k_offsets)))
    if has_shape:
        py_setitem(result, nk[].k_model_shape, py_call(p[].bi.dict, py_item(shape, nk[].k_model_shape)))
    return result


def parallel_byte_init(owner: PythonObject, state: PythonObject, devices: PythonObject,
                       logical_shards: PythonObject, pool_optimizer: PythonObject) raises -> PythonObject:
    """Admit a byte-LM multi-device trainer. The owner's exact admitted class
    selects the session: ParallelByteLanguageModelTrainer (resident
    replicas), PooledByteLanguageModelTrainer (decoder layers owned by
    devices) or OffloadedByteLanguageModelTrainer (one device, one decoder
    layer resident at a time)."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var schedule = _parallel_class_schedule(py_type(owner))
    var count = _parallel_int(logical_shards, 1, 1024, "logical_shards")
    if py_type(devices) is not p[].bi.tuple and py_type(devices) is not p[].bi.list:
        raise Error("TypeError: devices must be a tuple or list of device indices")
    var ids = py_call(p[].bi.tuple, devices)
    if schedule == PARALLEL_REPLICA and (len(ids) < 1 or len(ids) > count):
        raise Error("require 1 <= device count <= logical_shards <= 1024")
    if schedule == PARALLEL_OFFLOAD and len(ids) != 1:
        raise Error("offloaded replay requires exactly one device")
    var device_ids = List[Int]()
    for i in range(len(ids)):  # driver-loop: over ids: physical device indices
        var id = _parallel_int(ids[i], 0, 2147483647, "device index")
        for j in range(len(device_ids)):  # small-loop(device_ids: admitted device indices)
            if device_ids[j] == id:
                raise Error("devices must be distinct nonnegative integer indices")
        device_ids.append(id)
    if py_type(pool_optimizer) is not p[].bi.bool:
        raise Error("pool_optimizer must be bool")
    var pooled = Bool(py=pool_optimizer)
    if schedule != PARALLEL_REPLICA and pooled:
        raise Error("pool_optimizer belongs to ParallelByteLanguageModelTrainer")
    if py_type(state) is not p[].bi.dict:
        raise Error("Byte-LM state must be a dict")
    var shape = _parallel_shape(state)
    var admitted = _parallel_validate_state(state, shape)
    var nk = parallel_keys()
    if schedule == PARALLEL_MODEL_POOL:
        # Each decoder layer has one owner and the embedding/head share
        # devices[0], so more than n_layers + 1 devices would leave one empty.
        var owners = min(64, Int(py=py_item(shape, nk[].k_dims)[7]) + 1)
        if len(ids) < 1 or len(ids) > owners:
            raise Error("require 1 <= device count <= min(64, n_layers + 1)")
    var table = p[].modes[py_item(admitted, nk[].k_numeric_mode)]
    var owner_id = Int(py=py_call(p[].bi.id, owner))
    var record = PythonObject(alloc=ParallelByteOwner(owner_id, schedule, table, pooled, device_ids^, ids,
                                                      count, shape, admitted))
    var shape_object = py_call(p[].bi.object.__new__, p[].shape_cls)
    # Initialize frozen dataclass storage directly, without invoking the
    # Python-generated __setattr__ or colliding with PythonObject's setter.
    _ = py_call(py_fields(shape_object).update, py_item(shape, nk[].k_model_shape))
    var fields = py_fields(owner)
    # Replacing a previous record releases its session through the native
    # session value's deallocation, as reinitialization always did.
    py_setitem(fields, nk[].k__native, record)
    py_setitem(fields, nk[].k_pool_optimizer, pool_optimizer)
    py_setitem(fields, nk[].k_devices, ids)
    py_setitem(fields, nk[].k_logical_shards, PythonObject(count))
    py_setitem(fields, nk[].k__shape, shape_object)
    py_setitem(fields, nk[].k__session, PythonObject(None))
    return PythonObject(None)


def _parallel_mirror_session(owner: PythonObject, session: PythonObject) raises:
    var nk = parallel_keys()
    py_setitem(py_fields(owner), nk[].k__session, session)


def parallel_byte_open(owner: PythonObject) raises:
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if not parallel_session_openable(rec[].native):
        return
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if rec[].provider is None:
        if p[].vendor == "cpu":
            # The declared by-design sentence (python/mojolearn/_parallel_pool.py
            # `_cpu_refusal`): the identity harness reads a par-* host cell
            # N/A by it (tools/identity_break.py CPU_BY_DESIGN_REFUSAL).
            raise Error("NotImplementedError: no CPU implementation of the cooperative multi-GPU driver byte_lm_"
                        + parallel_session_schedule_name(rec[].native) + " yet: its sessions live inside the GPU byte-LM binding, "
                        + "which no host binding restates")
        raise Error("NotImplementedError: installed binding has no cooperative GPU byte-LM session")
    # UNPACK: the schedule factory makes the session object and names its
    # handle word. RUN admits the four state arrays by address.
    var session = py_call(rec[].create, PythonObject(rec[].native.schedule))
    var handle = Int(py=py_call(rec[].handle_entry, session))
    rec[].session = session
    var state_addresses = rec[].state_addresses.copy()
    try:
        parallel_session_open(rec[].native, handle, state_addresses)
    except error:
        rec[].session = PythonObject(None)
        raise error
    # PACK: the session mirror, and the host arrays leave the retained state.
    _parallel_mirror_session(owner, session)
    var state = rec[].state
    var nk = parallel_keys()
    var retained = py_call(p[].bi.dict, state)
    py_setitem(retained, nk[].k_parameters, PythonObject(None))
    py_setitem(retained, nk[].k_m, PythonObject(None))
    py_setitem(retained, nk[].k_v, PythonObject(None))
    rec[].state = retained
    rec[].state_addresses = List[Int]()
    _ = record


def _parallel_publish(record: PythonObject) raises:
    """PACK: refresh the boundary state dictionary from the typed session.
    Called where the dictionary is read (state, checkpoint, admitted_state)
    and after set_lr; a step no longer writes it."""
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    var nk = parallel_keys()
    var completed = rec[].native.completed
    if rec[].native.mirror_completed != completed:
        py_setitem(rec[].state, nk[].k_completed_steps, PythonObject(completed))
        py_setitem(rec[].state, nk[].k_next_batch_index, PythonObject(completed))
        rec[].native.mirror_completed = completed
    if rec[].native.lr_stale:
        var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
        var config = py_call(p[].bi.dict, py_item(rec[].state, nk[].k_config))
        py_setitem(config, nk[].k_lr, PythonObject(rec[].native.scalars[PARALLEL_SCALAR_LR]))
        py_setitem(rec[].state, nk[].k_config, config)
        rec[].native.lr_stale = False
    _ = record


def _parallel_export(owner: PythonObject, rank: PythonObject, into: PythonObject, gradients: Bool = False) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    var n = rec[].native.n_total
    var nt = rec[].native.n_tensors
    var replica_rank = _parallel_int(rank, 0, 2147483647, "rank")
    var output = into
    var addresses = List[Int]()
    var pins = List[PythonObject]()
    if gradients:
        if output is None:
            output = generation_array([n])
        addresses.append(_parallel_target(output, n, False, pins))
    else:
        var keys = [String("parameters"), String("m"), String("v"), String("flags")]
        if output is None:
            output = p[].bi.dict()
            for key in keys:  # small-loop(keys: four state arrays)
                py_setitem(output, key, generation_array([nt if key == "flags" else n], "i" if key == "flags" else "f"))
        if py_type(output) is not p[].bi.dict or len(output) != 4:
            raise Error("into must be a dict with parameters, m, v and flags")
        for key in keys:  # driver-loop: over keys: four state arrays
            if not py_contains(output, PythonObject(key)):
                raise Error("into is missing " + key)
            addresses.append(_parallel_target(output[key], nt if key == "flags" else n, key == "flags", pins))
    parallel_byte_open(owner)
    parallel_session_export(rec[].native, replica_rank, gradients, addresses)
    _ = pins^
    _ = record
    return output


def parallel_byte_state(owner: PythonObject, rank: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var arrays = _parallel_export(owner, rank, PythonObject(None))
    var rec = _parallel_record(owner)
    _parallel_publish(rec)
    var state = py_call(p[].bi.dict, rec.downcast_value_ptr[ParallelByteOwner]()[].state)
    _ = py_call(state.update, arrays)
    return _parallel_validate_state(state, rec.downcast_value_ptr[ParallelByteOwner]()[].shape, True)


def parallel_byte_shard_words(owner: PythonObject) raises -> List[Int]:
    """[logical_shards, batch, length] of an admitted trainer: what one step
    consumes (logical_shards buffers of int32[batch, length + 1])."""
    var rec = _parallel_record(owner).downcast_value_ptr[ParallelByteOwner]()
    var words: List[Int] = [rec[].native.logical_shards, rec[].native.dims[0], rec[].native.dims[1]]
    return words^


def parallel_byte_step_shards(owner: PythonObject, shard_addresses: List[Int]) raises -> PythonObject:
    """One committed step over `logical_shards` int32[batch, length + 1]
    shard buffers, given by address. The caller keeps every buffer alive
    and unchanged through this call (the session copies the ids at its
    start, with the GIL held). Shared by `train_step` (caller buffers) and
    the corpus door (`lm_corpus_public_native`, rows gathered natively)."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    var count = rec[].native.logical_shards
    if len(shard_addresses) != count:
        raise Error("logical shard count mismatch")
    var losses = List[Float64](length=count, fill=0)
    parallel_byte_open(owner)
    # RUN. An entry failure already rolled back inside the native GPU
    # session; a wrong loss count is rolled back by the step itself.
    # Do not roll back a prior committed step on an admission refusal.
    var before = parallel_session_step_begin(rec[].native, shard_addresses, losses)
    # PACK. A result that cannot be built returns the session to `before`.
    var result = p[].bi.dict()
    var nk = parallel_keys()
    try:
        var values = p[].bi.list()
        for i in range(count):  # driver-loop: over count: one loss per logical shard
            _ = py_call(values.append, PythonObject(losses[i]))
        py_setitem(result, nk[].k_losses, py_call(p[].bi.tuple, values))
        py_setitem(result, nk[].k_completed_steps, PythonObject(before + 1))
        py_setitem(result, nk[].k_logical_shards, PythonObject(count))
        py_setitem(result, nk[].k_reduction, PythonObject("ordered_sum"))
    except error:
        parallel_session_rollback(rec[].native, before)
        raise error
    parallel_session_step_commit(rec[].native, before)
    _ = losses^
    _ = record
    return result


def parallel_byte_train_step(owner: PythonObject, shards: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if py_type(shards) is not p[].bi.list and py_type(shards) is not p[].bi.tuple:
        raise Error("TypeError: shards must be a list or tuple of microbatches")
    var count = rec[].native.logical_shards
    if len(shards) != count:
        raise Error("logical shard count mismatch")
    var b = rec[].native.dims[0]
    var length = rec[].native.dims[1]
    # pass 3 (N102): each shard is read in place when it already is
    # C-contiguous int32 (pinned for the step); only other layouts copy.
    var pins = List[AdmittedBorrowedBuffer]()
    var copies = List[PythonObject]()
    var shard_addresses = List[Int]()
    for k in range(count):  # driver-loop: over immutable logical training shards
        shard_addresses.append(_parallel_borrow(shards[k], [b, length + 1], True, pins, copies))
    var result = parallel_byte_step_shards(owner, shard_addresses)
    _parallel_release(pins)
    _ = copies^
    return result


def _parallel_float_bytes(value: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if py_type(value) is p[].bi.bytes or py_type(value) is p[].bi.bytearray:
        var store = py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].arrays, "f")
        _ = py_call(store.frombytes, value)
        return store
    return value


def _parallel_ownership(owner: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if rec[].native.schedule == PARALLEL_OFFLOAD:
        raise Error("NotImplementedError: offloaded state lives on the host; no resident GPU optimizer owners")
    parallel_byte_open(owner)
    var width = parallel_session_ownership_width(rec[].native)
    var words = List[Int](length=parallel_session_ownership_capacity(rec[].native), fill=0)
    var rows = parallel_session_ownership(rec[].native, words)
    var device_count = len(rec[].native.devices)
    var result = p[].bi.list()
    var nk = parallel_keys()
    if rec[].native.schedule == PARALLEL_MODEL_POOL:
        # Rows are [owner index, first, count, parameter, moment, rollback,
        # gradient bytes] (byte_lm_model_pool ownership).
        var pool_fields = [String("first"), String("count"), String("parameter_bytes"),
                           String("moment_bytes"), String("rollback_bytes"), String("gradient_bytes")]
        for i in range(rows):  # driver-loop: over the model chunks each device owns
            var device = words[i * width]
            if device < 0 or device >= device_count:
                raise Error("RuntimeError: native ownership names an unknown device owner")
            var row = p[].bi.dict()
            py_setitem(row, nk[].k_device, rec[].device_objects[device])
            for j in range(len(pool_fields)):  # driver-loop: over pool_fields: six ownership record words
                py_setitem(row, pool_fields[j], PythonObject(words[i * width + j + 1]))
            _ = py_call(result.append, row)
        _ = words^
        _ = record
        return py_call(p[].bi.tuple, result)
    var fields = [String("first"), String("count"), String("moment_bytes"), String("rollback_bytes"), String("reduction_bytes")]
    if rows != device_count:
        raise Error("RuntimeError: native ownership row count differs from devices")
    for i in range(rows):  # driver-loop: over actual physical device ownership records
        var row = p[].bi.dict()
        py_setitem(row, nk[].k_device, rec[].device_objects[i])
        for j in range(len(fields)):  # driver-loop: over fields: five ownership record words
            py_setitem(row, fields[j], PythonObject(words[i * width + j]))
        _ = py_call(result.append, row)
    _ = words^
    _ = record
    return py_call(p[].bi.tuple, result)


def _parallel_state_snapshot(record: PythonObject) raises -> PythonObject:
    """A detached copy of the admitted state metadata and any host arrays
    still held before the session opens; mutating it changes nothing."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    _parallel_publish(record)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    var state = rec[].state
    var result = py_call(p[].bi.dict, state)
    var nk = parallel_keys()
    py_setitem(result, nk[].k_config, py_call(p[].bi.dict, py_item(state, nk[].k_config)))
    py_setitem(result, nk[].k_data_schedule, native_json_parse(PythonObject(_transport_json(py_item(state, nk[].k_data_schedule)))))
    py_setitem(result, nk[].k_parameter_names, py_call(p[].bi.list, py_item(state, nk[].k_parameter_names)))
    py_setitem(result, nk[].k_parameter_shapes, native_json_parse(PythonObject(_transport_json(py_item(state, nk[].k_parameter_shapes)))))
    py_setitem(result, nk[].k_parameter_offsets, py_call(p[].bi.list, py_item(state, nk[].k_parameter_offsets)))
    for key in [String("parameters"), String("m"), String("v")]:  # small-loop(keys: three state arrays)
        if state[key] is not None:
            py_setitem(result, key, _parallel_array(state[key], [rec[].native.n_total]))
    if py_item(state, nk[].k_flags) is not None:
        py_setitem(result, nk[].k_flags, _parallel_array(py_item(state, nk[].k_flags), [rec[].native.n_tensors], True))
    return result


def parallel_byte_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_BYTE_OPN_*` code (-1 for a name the table
    does not hold); `name` is the caller's str, read only for the unknown
    operation's text."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if operation == PARALLEL_BYTE_OPN_INITIALIZE:
        parallel_byte_initialize(args[0], args[1], args[2], String(args[3]))
        return PythonObject(None)
    if operation == PARALLEL_BYTE_OPN_ADMIT_CLASS:
        parallel_byte_admit_class(args[0], String(args[1]))
        return PythonObject(None)
    if operation == PARALLEL_BYTE_OPN_INIT:
        if len(args) != 5:
            raise Error("TypeError: parallel init takes owner, state, devices, logical_shards, pool_optimizer")
        return parallel_byte_init(args[0], args[1], args[2], args[3], args[4])
    var nk = parallel_keys()
    if operation == PARALLEL_BYTE_OPN_FROM_CHECKPOINT:
        if len(args) != 4:
            raise Error("TypeError: parallel from_checkpoint takes cls, checkpoint, devices, pool_optimizer")
        var cls = args[0]
        _ = _parallel_class_schedule(cls)
        var checkpoint = args[1]
        if py_type(checkpoint) is not p[].bi.dict:
            raise Error("unsupported parallel checkpoint contract")
        if String(py_get(checkpoint, nk[].k_schema)) != "mojolearn.parallel-byte-lm.v1" or String(py_get(checkpoint, nk[].k_reduction)) != "ordered_sum":
            raise Error("unsupported parallel checkpoint contract")
        var restored = py_call(p[].bi.object.__new__, cls)
        _ = parallel_byte_init(restored, py_item(checkpoint, nk[].k_state), args[2], py_item(checkpoint, nk[].k_logical_shards), args[3])
        return restored
    var owner = args[0]
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if operation == PARALLEL_BYTE_OPN_STEP:
        return PythonObject(rec[].native.completed)
    if operation == PARALLEL_BYTE_OPN_ENTER:
        return owner
    if operation == PARALLEL_BYTE_OPN_ADMITTED_STATE:
        return _parallel_state_snapshot(record)
    if operation == PARALLEL_BYTE_OPN_CLOSE:
        var was_open = rec[].native.handle != 0
        var failure = parallel_session_close(rec[].native)
        if was_open:
            rec[].session = PythonObject(None)
            _parallel_mirror_session(owner, PythonObject(None))
            if failure.byte_length() > 0:
                raise Error(failure)
        return PythonObject(None)
    if operation == PARALLEL_BYTE_OPN_OPEN:
        parallel_byte_open(owner)
        return PythonObject(None)
    if operation == PARALLEL_BYTE_OPN_TRAIN_STEP:
        return parallel_byte_train_step(owner, args[1])
    if operation == PARALLEL_BYTE_OPN_STATE:
        return parallel_byte_state(owner, args[1])
    if operation == PARALLEL_BYTE_OPN_EXPORT_RAW:
        return _parallel_export(owner, args[1], args[2])
    if operation == PARALLEL_BYTE_OPN_EXPORT_GRADIENTS:
        return _parallel_export(owner, args[1], args[2], True)
    if operation == PARALLEL_BYTE_OPN_CHECKPOINT:
        var result = p[].bi.dict()
        py_setitem(result, nk[].k_schema, PythonObject("mojolearn.parallel-byte-lm.v1"))
        py_setitem(result, nk[].k_logical_shards, PythonObject(rec[].native.logical_shards))
        py_setitem(result, nk[].k_reduction, PythonObject("ordered_sum"))
        py_setitem(result, nk[].k_state, parallel_byte_state(owner, PythonObject(0)))
        return result
    if operation == PARALLEL_BYTE_OPN_SET_LR:
        var lr = _parallel_float(args[1], "lr")
        parallel_session_set_lr(rec[].native, lr)
        # PACK: the retained configuration carries the new rate at once.
        _parallel_publish(record)
        return PythonObject(Float64(lr))
    if operation == PARALLEL_BYTE_OPN_REQUIRE_FOLD:
        parallel_session_require_split(rec[].native, "shard_gradient")
        return PythonObject(None)
    if operation == PARALLEL_BYTE_OPN_HAS_DEVICE_FOLD:
        parallel_byte_open(owner)
        return PythonObject(rec[].native.schedule == PARALLEL_REPLICA)
    if operation == PARALLEL_BYTE_OPN_OWNERSHIP:
        return _parallel_ownership(owner)
    if operation == PARALLEL_BYTE_OPN_MODEL_OWNERSHIP:
        if rec[].native.schedule != PARALLEL_MODEL_POOL:
            raise Error("model_ownership requires PooledByteLanguageModelTrainer")
        return _parallel_ownership(owner)
    var n = rec[].native.n_total
    if operation == PARALLEL_BYTE_OPN_SHARD_GRADIENT or operation == PARALLEL_BYTE_OPN_SHARD_GRADIENT_FOLD:
        var fold = operation == PARALLEL_BYTE_OPN_SHARD_GRADIENT_FOLD
        # UNPACK. pass 3 (N102): the shard is read in place when it already is
        # C-contiguous int32 (pinned for the call); only other layouts copy.
        var pins = List[AdmittedBorrowedBuffer]()
        var copies = List[PythonObject]()
        var ids_address = _parallel_borrow(args[1], [rec[].native.dims[0], rec[].native.dims[1] + 1], True, pins, copies)
        parallel_byte_open(owner)
        parallel_session_require_split(rec[].native, "shard_gradient_fold" if fold else "shard_gradient")
        var gradient = PythonObject(None)
        var gradient_address = 0
        if not fold:
            gradient = generation_array([n])
            gradient_address = Int(py=gradient._addr)
        # RUN
        var loss = parallel_session_shard_gradient(rec[].native, ids_address, gradient_address, fold)
        _parallel_release(pins)
        _ = copies^
        # PACK
        var value = PythonObject(loss)
        if fold:
            return value
        var result = p[].bi.list()
        _ = py_call(result.append, value)
        _ = py_call(result.append, gradient)
        return py_call(p[].bi.tuple, result)
    if operation == PARALLEL_BYTE_OPN_APPLY_GRADIENT:
        # `total` is documented as the bytes `cross_vendor.ordered_fold`
        # returns; admit raw float32 bytes the way fold_reset/fold_add do.
        # pass 3 (N111, N157): the caller's buffer (the received payload)
        # is read in place and uploaded once; the session's device scan of
        # the summed gradient refuses non-finite values (no host copy, no
        # host scan).
        var pins = List[AdmittedBorrowedBuffer]()
        var copies = List[PythonObject]()
        var total_address = _parallel_borrow(args[1], [n], False, pins, copies, False)
        parallel_byte_open(owner)
        parallel_session_require_split(rec[].native, "apply_gradient")
        var completed = 0
        try:
            completed = parallel_session_apply_gradient(rec[].native, total_address)
        except error:
            _parallel_release(pins)
            raise error
        _parallel_release(pins)
        _ = copies^
        return PythonObject(completed)
    if operation == PARALLEL_BYTE_OPN_FOLD_RESET or operation == PARALLEL_BYTE_OPN_FOLD_ADD:
        # pass 3 (N111, N157): a held gradient or received prefix is read in
        # place; the device scan in fold_reset/fold_add refuses non-finite
        # values before the fold changes (no host copy, no host scan).
        var add = operation == PARALLEL_BYTE_OPN_FOLD_ADD
        var pins = List[AdmittedBorrowedBuffer]()
        var copies = List[PythonObject]()
        var address = 0
        if args[1] is not None:
            address = _parallel_borrow(args[1], [n], False, pins, copies, False)
        elif add:
            raise Error("fold_add requires a gradient")
        parallel_byte_open(owner)
        parallel_session_require_split(rec[].native, "fold_add" if add else "fold_reset")
        parallel_session_fold(rec[].native, add, address)
        _parallel_release(pins)
        _ = copies^
        return PythonObject(None)
    if operation == PARALLEL_BYTE_OPN_FOLD_EXPORT:
        var into = args[1]
        var output = generation_array([n]) if into is None else into
        var pins = List[PythonObject]()
        var address = _parallel_target(output, n, False, pins)
        parallel_byte_open(owner)
        parallel_session_require_split(rec[].native, "fold_export")
        parallel_session_fold_export(rec[].native, address)
        var result = py_call(output._mv.cast, "B").tobytes() if into is None else output
        _ = pins^
        return result
    raise Error("parallel: unknown native operation " + String(name))


def parallel_byte_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = parallel_keys()
        var result = parallel_byte_call(py_op_code(nk[].ops_byte, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
