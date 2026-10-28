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
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from std.memory import bitcast, memcpy
from std.math import isfinite
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array, py_ints
from bindings.py_balanced import py_call, py_contains, py_fields, py_get, py_setattr, py_setitem, py_type
from bindings.native_json import native_json_parse
from bindings.native_transport import _transport_json, transport_initialize
from providers.python.neural_descriptor_pyapi import neural_descriptor_connect, neural_descriptor_execute
from training.parallel_byte_impl import (
    _parallel_span,
    parallel_copy_words,
    parallel_require_binary_i32,
    parallel_require_finite_f32,
    parallel_require_nonnegative_f32,
)


# The byte-LM schedule descriptor ABI (bindings/_mojolearn_byte_lm.mojo).
comptime PARALLEL_SCHEDULE_ABI_NAME = "mojolearn.training.byte_lm_schedule.v1"
comptime PARALLEL_SCHEDULE_OPERATIONS = 15
comptime PARALLEL_REPLICA = 0
comptime PARALLEL_MODEL_POOL = 1
comptime PARALLEL_OFFLOAD = 2
comptime PARALLEL_OP_CLOSE = 0
comptime PARALLEL_OP_OPEN = 1
comptime PARALLEL_OP_STEP = 2
comptime PARALLEL_OP_EXPORT = 3
comptime PARALLEL_OP_ROLLBACK = 4
comptime PARALLEL_OP_OWNERSHIP = 5
comptime PARALLEL_OP_SET_LR = 6
comptime PARALLEL_OP_SHARD_GRADIENT = 7
comptime PARALLEL_OP_APPLY_GRADIENT = 8
comptime PARALLEL_OP_FOLD_RESET = 9
comptime PARALLEL_OP_SHARD_GRADIENT_FOLD = 10
comptime PARALLEL_OP_FOLD_ADD = 11
comptime PARALLEL_OP_FOLD_EXPORT = 12


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
        for _ in range(3):  # small-loop(3: replica, model_pool and offload schedules)
            self.classes.append(PythonObject(None))


comptime PARALLEL_BYTE_BOUNDARY = _Global[
    StorageType=ParallelByteBoundary, name="MojoParallelByteBoundary",
    init_fn=ParallelByteBoundary.__init__,
]


struct ParallelByteOwner(Movable, Writable):
    """One trainer's native state. Stored once in the instance dictionary as
    `_native`; every operation reads this record, never instance attributes.
    `devices`, `logical_shards`, `pool_optimizer`, `_shape` and `_session`
    on the instance are read-only mirrors for callers."""
    var owner_id: Int
    var schedule: Int
    var provider: PythonObject
    var create: PythonObject
    var handle_entry: PythonObject
    var session: PythonObject
    var handle: Int
    var closed: Bool
    var lost: Bool
    var pool_optimizer: Bool
    var devices: List[Int]
    var device_objects: PythonObject
    var logical_shards: Int
    var dims: List[Int]
    var n_total: Int
    var n_tensors: Int
    var state: PythonObject
    var state_addresses: List[Int]
    var shape: PythonObject

    def __init__(out self, owner_id: Int, schedule: Int, table: PythonObject, pool_optimizer: Bool,
                 var devices: List[Int], device_objects: PythonObject, logical_shards: Int,
                 shape: PythonObject, state: PythonObject) raises:
        self.owner_id = owner_id
        self.schedule = schedule
        self.provider = table["provider"]
        self.create = table["create"]
        self.handle_entry = table["handle"]
        self.session = PythonObject(None)
        self.handle = 0
        self.closed = False
        self.lost = False
        self.pool_optimizer = pool_optimizer
        self.devices = devices^
        self.device_objects = device_objects
        self.logical_shards = logical_shards
        self.dims = List[Int]()
        var dims = shape["dims"]
        for i in range(len(dims)):  # small-loop(dims: nine or ten admitted shape words)
            self.dims.append(Int(py=dims[i]))
        self.n_total = Int(py=shape["n_total"])
        self.n_tensors = Int(py=shape["n_tensors"])
        self.state = state
        # Captured once from the freshly admitted native copies.
        self.state_addresses = List[Int]()
        for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(keys: four state arrays)
            self.state_addresses.append(Int(py=state[key]._addr))
        self.shape = shape

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParallelByteOwner(schedule=", self.schedule, ", open=", self.handle != 0, ")")

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
    for k in range(len(mode_keys)):  # driver-loop: over installed numerical-mode ABI tables
        var mode = mode_keys[k]
        if py_type(mode) is not p[].bi.str:
            raise Error("TypeError: parallel: mode names must be str")
        var binding = modes[mode]
        var table = p[].bi.dict()
        var suffix = _parallel_compiled(binding, "byte_lm_arithmetic_suffix")
        if suffix is None:
            suffix = _parallel_compiled(binding, "byte_lm_host_arithmetic_suffix")
        if suffix is None:
            raise Error("parallel: binding has no native arithmetic profile")
        py_setitem(table, "suffix", suffix())
        # Resolve and prove the same mode/vendor/profile witness as _load;
        # the connection-time host proxy is never retained as a runtime call.
        for name in [String("numeric_mode"), String("vendor"), String("profile"), String("config_profile")]:  # small-loop(names: four profile entry names)
            var native_name = "byte_lm_" + name
            if expected_vendor == "cpu":
                native_name = "byte_lm_host_" + ("profile" if name == "config_profile" else name)
            var entry = py_call(p[].bi.getattr, binding, native_name)
            if py_type(entry) is not py_type(p[].bi.len):
                raise Error("parallel: profile entry must be compiled native code")
            py_setitem(table, "profile_" + name, entry)
        if Int(py=table["profile_numeric_mode"]()) != (1 if String(mode) == "identical" else 0) or String(table["profile_vendor"]()) != expected_vendor:
            raise Error("parallel: native mode/vendor mismatch")
        var default_profile = py_call(table["profile_profile"], py_ints([2, 32, 32, 4, 2, 8, 64, 2, 256])) if expected_vendor == "cpu" else table["profile_profile"]()
        if String(default_profile) != "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1" + String(table["suffix"]):
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
        py_setitem(table, "provider", provider)
        py_setitem(table, "create", create)
        py_setitem(table, "handle", handle)
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
        for hook in forbidden:  # small-loop(forbidden: seven storage and lifecycle hook names)
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
    var record = py_get(py_fields(owner), "_native")
    if record is None:
        raise Error("RuntimeError: parallel trainer is not initialized")
    var owner_id = Int(py=py_call(p[].bi.id, owner))
    try:
        var rec = record.downcast_value_ptr[ParallelByteOwner]()
        if rec[].owner_id != owner_id or rec[].schedule != schedule:
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
    for i in range(len(dims)):  # small-loop(dims: array rank words)
        if Int(py=shape[i]) != dims[i]:
            raise Error("parallel input has wrong shape")
    var output = generation_array(dims, "i" if integer else "f")
    var store = py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].arrays, "i" if integer else "f")
    _ = py_call(store.frombytes, py_call(view.tobytes, "C"))
    var n = Int(py=output.size)
    if len(store) != n:
        raise Error("parallel input byte count differs from shape")
    var src_addr = Int(py=store.buffer_info()[0])
    var dst_addr = Int(py=output._addr)
    parallel_copy_words(src_addr, dst_addr, dims, n,
                        own_array and String(value.order) == "F" and len(dims) > 1)
    if not integer and finite:
        parallel_require_finite_f32(Int(py=output._addr), n)
    _ = store^
    return output


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
    var shape = py_get(state, "model_shape", PythonObject(None))
    if shape is not None:
        if py_type(shape) is not p[].bi.dict:
            raise Error("Byte-LM model_shape must be a dict")
        var keys = py_call(p[].bi.list, shape)
        for i in range(len(keys)):  # small-loop(keys: model_shape dimension names)
            var key = String(keys[i])
            if key != "chunked_lm_head_v2" and key not in names:
                raise Error("Byte-LM model_shape has unknown dimensions")
        for i in range(7):  # small-loop(names: seven required dimension names)
            if not py_contains(shape, PythonObject(names[i])):
                raise Error("Byte-LM model_shape has missing dimensions")
        if py_contains(shape, "n_layers") != py_contains(shape, "vocab_size"):
            raise Error("Byte-LM model_shape must specify both layers and vocabulary")
    var dims = p[].bi.list()
    var normalized = p[].bi.dict()
    for i in range(len(names)):  # small-loop(names: nine dimension names)
        var value = PythonObject(defaults[i]) if shape is None else py_get(shape, PythonObject(names[i]), PythonObject(defaults[i]))
        var n = _parallel_int(value, 1, 1048576, names[i], False)
        _ = py_call(dims.append, n)
        py_setitem(normalized, names[i], PythonObject(n))
    var chunked = PythonObject(False) if shape is None else py_get(shape, "chunked_lm_head_v2", PythonObject(False))
    if py_type(chunked) is not p[].bi.bool:
        raise Error("chunked_lm_head_v2 must be bool")
    py_setitem(normalized, "chunked_lm_head_v2", chunked)
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
    py_setitem(result, "dims", dims)
    py_setitem(result, "model_shape", normalized)
    py_setitem(result, "names", names_out)
    py_setitem(result, "shapes", shapes_out)
    py_setitem(result, "offsets", offsets)
    py_setitem(result, "n_total", PythonObject(total))
    py_setitem(result, "n_tensors", PythonObject(2 + layers * 9))
    py_setitem(result, "profile_base", PythonObject(base))
    return result


def _parallel_validate_state(value: PythonObject, shape: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if py_type(value) is not p[].bi.dict:
        raise Error("Byte-LM state must be a dict")
    var names = [String("schema"), String("profile"), String("numeric_mode"), String("parameter_names"),
                 String("parameter_shapes"), String("parameter_offsets"), String("parameters"), String("m"),
                 String("v"), String("flags"), String("completed_steps"), String("next_batch_index"),
                 String("config"), String("data_schedule")]
    var has_shape = py_contains(value, "model_shape")
    if len(value) != len(names) + Int(has_shape):
        raise Error("Byte-LM state has missing or unknown fields")
    for name in names:  # small-loop(names: fourteen state field names)
        if not py_contains(value, PythonObject(name)):
            raise Error("Byte-LM state is missing " + name)
    var mode = String(value["numeric_mode"])
    if not py_contains(p[].modes, PythonObject(mode)):
        raise Error("Byte-LM numeric mode is not installed")
    var table = p[].modes[mode]
    if String(py_call(table["profile_config_profile"], shape["dims"])) != String(shape["profile_base"]) + String(table["suffix"]):
        raise Error("Byte-LM native runtime shape/profile mismatch")
    if String(value["schema"]) != "mojolearn.small-byte-lm-state.v1" or String(value["profile"]) != String(shape["profile_base"]) + String(table["suffix"]):
        raise Error("Byte-LM state profile/registry/mode mismatch")
    if not Bool(py=value["parameter_names"] == shape["names"]) or not Bool(py=value["parameter_shapes"] == shape["shapes"]) or not Bool(py=value["parameter_offsets"] == shape["offsets"]):
        raise Error("Byte-LM state parameter registry mismatch")
    var cfg = value["config"]
    var cfg_names = [String("kind"), String("lr"), String("beta1"), String("beta2"), String("eps"),
                     String("weight_decay"), String("momentum"), String("dampening"), String("nesterov"), String("max_norm")]
    if py_type(cfg) is not p[].bi.dict or len(cfg) != len(cfg_names):
        raise Error("Byte-LM optimizer configuration mismatch")
    for name in cfg_names:  # small-loop(cfg_names: ten optimizer field names)
        if not py_contains(cfg, PythonObject(name)):
            raise Error("Byte-LM optimizer configuration is missing " + name)
    if _parallel_int(cfg["kind"], 2, 2, "optimizer kind") != 2 or py_type(cfg["nesterov"]) is not p[].bi.bool or Bool(py=cfg["nesterov"]):
        raise Error("Byte-LM first profile requires AdamW without Nesterov")
    var config = p[].bi.dict()
    py_setitem(config, "kind", PythonObject(2))
    py_setitem(config, "nesterov", PythonObject(False))
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay"),  # small-loop(names: eight optimizer scalars)
                 String("momentum"), String("dampening"), String("max_norm")]:
        var number = _parallel_float(cfg[name], name)
        if ((name == "lr" or name == "eps") and number <= 0) or (name == "weight_decay" and number < 0):
            raise Error("Byte-LM requires lr > 0, eps > 0 and weight_decay >= 0")
        if (name == "beta1" or name == "beta2") and (number < 0 or number >= 1):
            raise Error("Byte-LM betas must remain in [0, 1) in float32")
        if (name == "momentum" or name == "dampening" or name == "max_norm") and number != 0:
            raise Error("Byte-LM first profile requires AdamW without clipping or SGD options")
        py_setitem(config, name, PythonObject(Float64(number)))
    var completed = _parallel_int(value["completed_steps"], 0, 999999, "completed_steps", False)
    if _parallel_int(value["next_batch_index"], 0, 999999, "next_batch_index", False) != completed:
        raise Error("Byte-LM schedule cursor must match completed_steps")
    var result = py_call(p[].bi.dict, value)
    var n = Int(py=shape["n_total"])
    for name in [String("parameters"), String("m"), String("v")]:  # small-loop(names: three state arrays)
        py_setitem(result, name, _parallel_array(value[name], [n]))
    var flags = _parallel_array(value["flags"], [Int(py=shape["n_tensors"])], True)
    parallel_require_nonnegative_f32(Int(py=result["v"]._addr), n)
    parallel_require_binary_i32(Int(py=flags._addr), Int(py=shape["n_tensors"]))
    py_setitem(result, "flags", flags)
    py_setitem(result, "config", config)
    py_setitem(result, "completed_steps", PythonObject(completed))
    py_setitem(result, "next_batch_index", PythonObject(completed))
    py_setitem(result, "data_schedule", _parallel_schedule(value["data_schedule"]))
    var vocabulary = py_get(result["data_schedule"], "vocabulary", PythonObject(None))
    if vocabulary is not None:
        if py_type(vocabulary) is not p[].bi.dict or py_type(py_get(vocabulary, "n_vocab")) is not p[].bi.int or py_type(py_get(vocabulary, "sha256")) is not p[].bi.str:
            raise Error("Byte-LM data schedule vocabulary descriptor is invalid")
        if Int(py=vocabulary["n_vocab"]) != Int(py=shape["dims"][8]):
            raise Error("Byte-LM data schedule vocabulary does not match model")
    py_setitem(result, "parameter_names", py_call(p[].bi.list, shape["names"]))
    py_setitem(result, "parameter_shapes", native_json_parse(PythonObject(_transport_json(shape["shapes"]))))
    py_setitem(result, "parameter_offsets", py_call(p[].bi.list, shape["offsets"]))
    if has_shape:
        py_setitem(result, "model_shape", py_call(p[].bi.dict, shape["model_shape"]))
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
    for i in range(len(ids)):  # small-loop(ids: physical device indices)
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
    if schedule == PARALLEL_MODEL_POOL:
        # Each decoder layer has one owner and the embedding/head share
        # devices[0], so more than n_layers + 1 devices would leave one empty.
        var owners = min(64, Int(py=shape["dims"][7]) + 1)
        if len(ids) < 1 or len(ids) > owners:
            raise Error("require 1 <= device count <= min(64, n_layers + 1)")
    var table = p[].modes[admitted["numeric_mode"]]
    var owner_id = Int(py=py_call(p[].bi.id, owner))
    var record = PythonObject(alloc=ParallelByteOwner(owner_id, schedule, table, pooled, device_ids^, ids,
                                                      count, shape, admitted))
    var shape_object = py_call(p[].bi.object.__new__, p[].shape_cls)
    # Initialize frozen dataclass storage directly, without invoking the
    # Python-generated __setattr__ or colliding with PythonObject's setter.
    _ = py_call(py_fields(shape_object).update, shape["model_shape"])
    var fields = py_fields(owner)
    # Replacing a previous record releases its session through the native
    # session value's deallocation, as reinitialization always did.
    py_setitem(fields, "_native", record)
    py_setitem(fields, "pool_optimizer", pool_optimizer)
    py_setitem(fields, "devices", ids)
    py_setitem(fields, "logical_shards", PythonObject(count))
    py_setitem(fields, "_shape", shape_object)
    py_setitem(fields, "_session", PythonObject(None))
    return PythonObject(None)


def _parallel_mirror_session(owner: PythonObject, session: PythonObject) raises:
    py_setitem(py_fields(owner), "_session", session)


def _parallel_execute(record: PythonObject, op: Int, var addresses: List[Int], ints: List[Int],
                      floats: List[Float64]) raises -> Int:
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if rec[].handle == 0:
        raise Error("RuntimeError: parallel trainer has no open session")
    var words: List[Int] = [rec[].handle]
    for value in addresses:  # small-loop(addresses: the operation's admitted buffer descriptors)
        words.append(value)
    # Retain the GIL: it is also the native session's exclusion lock.
    var result = neural_descriptor_execute(rec[].provider, op, words, ints, floats, False)
    _ = record
    return result


def parallel_byte_open(owner: PythonObject) raises:
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if rec[].closed or rec[].lost:
        raise Error("RuntimeError: parallel trainer is closed or lost; restore a state export")
    if rec[].handle != 0:
        return
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if rec[].provider is None:
        if p[].vendor == "cpu":
            # The declared by-design sentence (python/mojolearn/_parallel_pool.py
            # `_cpu_refusal`): the identity harness reads a par-* host cell
            # N/A by it (tools/identity_break.py CPU_BY_DESIGN_REFUSAL).
            var name = String("parallel") if rec[].schedule == PARALLEL_REPLICA else (
                String("model_pool") if rec[].schedule == PARALLEL_MODEL_POOL else String("offload"))
            raise Error("NotImplementedError: no CPU implementation of the cooperative multi-GPU driver byte_lm_"
                        + name + " yet: its sessions live inside the GPU byte-LM binding, "
                        + "which no host binding restates")
        raise Error("NotImplementedError: installed binding has no cooperative GPU byte-LM session")
    var state = rec[].state
    var config = state["config"]
    var completed = Int(py=state["completed_steps"])
    var ints: List[Int] = [completed, Int(py=config["kind"]), 1 if Bool(py=config["nesterov"]) else 0,
                           1 if rec[].pool_optimizer else 0, len(rec[].dims)]
    for value in rec[].dims:  # small-loop(dims: nine or ten admitted shape words)
        ints.append(value)
    ints.append(rec[].logical_shards)
    ints.append(len(rec[].devices))
    for value in rec[].devices:  # small-loop(devices: admitted device indices)
        ints.append(value)
    var floats = List[Float64]()
    for key in [String("lr"), String("beta1"), String("beta2"), String("eps"),  # small-loop(keys: eight optimizer scalars)
                String("weight_decay"), String("momentum"), String("dampening"), String("max_norm")]:
        floats.append(Float64(py=config[key]))
    var session = py_call(rec[].create, PythonObject(rec[].schedule))
    var handle = Int(py=py_call(rec[].handle_entry, session))
    if handle == 0:
        raise Error("RuntimeError: byte-LM schedule factory returned a null session")
    rec[].session = session
    rec[].handle = handle
    try:
        var opened = _parallel_execute(record, PARALLEL_OP_OPEN, rec[].state_addresses.copy(), ints, floats)
        if opened != completed:
            raise Error("RuntimeError: parallel admission returned wrong step")
    except error:
        try:
            _ = _parallel_execute(record, PARALLEL_OP_CLOSE, List[Int](), List[Int](), List[Float64]())
        except:
            pass
        rec[].session = PythonObject(None)
        rec[].handle = 0
        raise error
    _parallel_mirror_session(owner, session)
    var retained = py_call(p[].bi.dict, state)
    py_setitem(retained, "parameters", PythonObject(None))
    py_setitem(retained, "m", PythonObject(None))
    py_setitem(retained, "v", PythonObject(None))
    rec[].state = retained
    rec[].state_addresses = List[Int]()


def _parallel_require_split(record: PythonObject, entry: String) raises:
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if rec[].schedule != PARALLEL_REPLICA:
        raise Error(entry + " requires ParallelByteLanguageModelTrainer")
    if rec[].pool_optimizer or len(rec[].devices) != 1:
        raise Error(entry + " requires devices=(d,) and pool_optimizer=False")
    if rec[].provider is None:
        raise Error("RuntimeError: installed byte-LM binding lacks " + entry)


def _parallel_completed(record: PythonObject) raises -> Int:
    return Int(py=record.downcast_value_ptr[ParallelByteOwner]()[].state["completed_steps"])


def _parallel_commit(record: PythonObject, completed: Int) raises:
    var state = record.downcast_value_ptr[ParallelByteOwner]()[].state
    py_setitem(state, "completed_steps", PythonObject(completed))
    py_setitem(state, "next_batch_index", PythonObject(completed))


def _parallel_export(owner: PythonObject, rank: PythonObject, into: PythonObject, gradients: Bool = False) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    var n = rec[].n_total
    var nt = rec[].n_tensors
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
        for key in keys:  # small-loop(keys: four state arrays)
            if not py_contains(output, PythonObject(key)):
                raise Error("into is missing " + key)
            addresses.append(_parallel_target(output[key], nt if key == "flags" else n, key == "flags", pins))
    parallel_byte_open(owner)
    var ints: List[Int] = [replica_rank, 1 if gradients else 0]
    var step = _parallel_execute(record, PARALLEL_OP_EXPORT, addresses^, ints, List[Float64]())
    if step != _parallel_completed(record):
        raise Error("RuntimeError: parallel export returned wrong step")
    _ = pins^
    return output


def parallel_byte_state(owner: PythonObject, rank: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var arrays = _parallel_export(owner, rank, PythonObject(None))
    var rec = _parallel_record(owner)
    var state = py_call(p[].bi.dict, rec.downcast_value_ptr[ParallelByteOwner]()[].state)
    _ = py_call(state.update, arrays)
    return _parallel_validate_state(state, rec.downcast_value_ptr[ParallelByteOwner]()[].shape)


def parallel_byte_train_step(owner: PythonObject, shards: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if py_type(shards) is not p[].bi.list and py_type(shards) is not p[].bi.tuple:
        raise Error("TypeError: shards must be a list or tuple of microbatches")
    var count = rec[].logical_shards
    if len(shards) != count:
        raise Error("logical shard count mismatch")
    var tokens = List[PythonObject]()
    var b = rec[].dims[0]
    var length = rec[].dims[1]
    var losses = List[Float64](length=count, fill=0)
    var addresses: List[Int] = [Int(losses.unsafe_ptr())]
    for k in range(count):  # driver-loop: over immutable logical training shards
        var ids = _parallel_array(shards[k], [b, length + 1], True)
        addresses.append(Int(py=ids._addr))
        tokens.append(ids)
    parallel_byte_open(owner)
    var before = _parallel_completed(record)
    # An entry failure already rolled back inside the native GPU session.
    # Do not roll back a prior committed step on an admission refusal.
    var ints: List[Int] = [before, count]
    var written = _parallel_execute(record, PARALLEL_OP_STEP, addresses^, ints, List[Float64]())
    var result = p[].bi.dict()
    try:
        if written != count:
            raise Error("RuntimeError: parallel step returned wrong loss count")
        var values = p[].bi.list()
        for i in range(count):  # small-loop(count: one loss per logical shard)
            _ = py_call(values.append, PythonObject(losses[i]))
        py_setitem(result, "losses", py_call(p[].bi.tuple, values))
        py_setitem(result, "completed_steps", PythonObject(before + 1))
        py_setitem(result, "logical_shards", PythonObject(count))
        py_setitem(result, "reduction", PythonObject("ordered_sum"))
    except error:
        try:
            if _parallel_execute(record, PARALLEL_OP_ROLLBACK, List[Int](), List[Int](), List[Float64]()) != before:
                rec[].lost = True
        except:
            rec[].lost = True
        raise error
    _parallel_commit(record, before + 1)
    _ = tokens^
    _ = losses^
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
    if rec[].schedule == PARALLEL_OFFLOAD:
        raise Error("NotImplementedError: offloaded state lives on the host; no resident GPU optimizer owners")
    parallel_byte_open(owner)
    var width = 5 if rec[].schedule == PARALLEL_REPLICA else 7
    var capacity = width * (rec[].dims[7] + 2 + len(rec[].devices))
    var words = List[Int](length=capacity, fill=0)
    var addresses: List[Int] = [Int(words.unsafe_ptr())]
    var ints: List[Int] = [capacity]
    var written = _parallel_execute(record, PARALLEL_OP_OWNERSHIP, addresses^, ints, List[Float64]())
    if written < 0 or written % width != 0 or written > capacity:
        raise Error("RuntimeError: native ownership returned a malformed record")
    var rows = written // width
    var result = p[].bi.list()
    if rec[].schedule == PARALLEL_MODEL_POOL:
        # Rows are [owner index, first, count, parameter, moment, rollback,
        # gradient bytes] (byte_lm_model_pool ownership).
        var pool_fields = [String("first"), String("count"), String("parameter_bytes"),
                           String("moment_bytes"), String("rollback_bytes"), String("gradient_bytes")]
        for i in range(rows):  # driver-loop: over the model chunks each device owns
            var device = words[i * width]
            if device < 0 or device >= len(rec[].devices):
                raise Error("RuntimeError: native ownership names an unknown device owner")
            var row = p[].bi.dict()
            py_setitem(row, "device", rec[].device_objects[device])
            for j in range(len(pool_fields)):  # small-loop(pool_fields: six ownership record words)
                py_setitem(row, pool_fields[j], PythonObject(words[i * width + j + 1]))
            _ = py_call(result.append, row)
        _ = words^
        return py_call(p[].bi.tuple, result)
    var fields = [String("first"), String("count"), String("moment_bytes"), String("rollback_bytes"), String("reduction_bytes")]
    if rows != len(rec[].devices):
        raise Error("RuntimeError: native ownership row count differs from devices")
    for i in range(rows):  # driver-loop: over actual physical device ownership records
        var row = p[].bi.dict()
        py_setitem(row, "device", rec[].device_objects[i])
        for j in range(len(fields)):  # small-loop(fields: five ownership record words)
            py_setitem(row, fields[j], PythonObject(words[i * width + j]))
        _ = py_call(result.append, row)
    _ = words^
    return py_call(p[].bi.tuple, result)


def _parallel_state_snapshot(record: PythonObject) raises -> PythonObject:
    """A detached copy of the admitted state metadata and any host arrays
    still held before the session opens; mutating it changes nothing."""
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    var state = rec[].state
    var result = py_call(p[].bi.dict, state)
    py_setitem(result, "config", py_call(p[].bi.dict, state["config"]))
    py_setitem(result, "data_schedule", native_json_parse(PythonObject(_transport_json(state["data_schedule"]))))
    py_setitem(result, "parameter_names", py_call(p[].bi.list, state["parameter_names"]))
    py_setitem(result, "parameter_shapes", native_json_parse(PythonObject(_transport_json(state["parameter_shapes"]))))
    py_setitem(result, "parameter_offsets", py_call(p[].bi.list, state["parameter_offsets"]))
    for key in [String("parameters"), String("m"), String("v")]:  # small-loop(keys: three state arrays)
        if state[key] is not None:
            py_setitem(result, key, _parallel_array(state[key], [rec[].n_total]))
    if state["flags"] is not None:
        py_setitem(result, "flags", _parallel_array(state["flags"], [rec[].n_tensors], True))
    return result


def parallel_byte_call(operation: String, args: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if operation == "initialize":
        parallel_byte_initialize(args[0], args[1], args[2], String(args[3]))
        return PythonObject(None)
    if operation == "admit_class":
        parallel_byte_admit_class(args[0], String(args[1]))
        return PythonObject(None)
    if operation == "init":
        if len(args) != 5:
            raise Error("TypeError: parallel init takes owner, state, devices, logical_shards, pool_optimizer")
        return parallel_byte_init(args[0], args[1], args[2], args[3], args[4])
    if operation == "from_checkpoint":
        if len(args) != 4:
            raise Error("TypeError: parallel from_checkpoint takes cls, checkpoint, devices, pool_optimizer")
        var cls = args[0]
        _ = _parallel_class_schedule(cls)
        var checkpoint = args[1]
        if py_type(checkpoint) is not p[].bi.dict:
            raise Error("unsupported parallel checkpoint contract")
        if String(py_get(checkpoint, "schema")) != "mojolearn.parallel-byte-lm.v1" or String(py_get(checkpoint, "reduction")) != "ordered_sum":
            raise Error("unsupported parallel checkpoint contract")
        var restored = py_call(p[].bi.object.__new__, cls)
        _ = parallel_byte_init(restored, checkpoint["state"], args[2], checkpoint["logical_shards"], args[3])
        return restored
    var owner = args[0]
    var record = _parallel_record(owner)
    var rec = record.downcast_value_ptr[ParallelByteOwner]()
    if operation == "step":
        return PythonObject(_parallel_completed(record))
    if operation == "enter":
        return owner
    if operation == "admitted_state":
        return _parallel_state_snapshot(record)
    if operation == "close":
        rec[].closed = True
        if rec[].handle != 0:
            var failure = String()
            try:
                _ = _parallel_execute(record, PARALLEL_OP_CLOSE, List[Int](), List[Int](), List[Float64]())
            except error:
                failure = String(error)
            rec[].session = PythonObject(None)
            rec[].handle = 0
            _parallel_mirror_session(owner, PythonObject(None))
            if failure.byte_length() > 0:
                raise Error(failure)
        return PythonObject(None)
    if operation == "open":
        parallel_byte_open(owner)
        return PythonObject(None)
    if operation == "train_step":
        return parallel_byte_train_step(owner, args[1])
    if operation == "state":
        return parallel_byte_state(owner, args[1])
    if operation == "export_raw":
        return _parallel_export(owner, args[1], args[2])
    if operation == "export_gradients":
        return _parallel_export(owner, args[1], args[2], True)
    if operation == "checkpoint":
        var result = p[].bi.dict()
        py_setitem(result, "schema", PythonObject("mojolearn.parallel-byte-lm.v1"))
        py_setitem(result, "logical_shards", PythonObject(rec[].logical_shards))
        py_setitem(result, "reduction", PythonObject("ordered_sum"))
        py_setitem(result, "state", parallel_byte_state(owner, PythonObject(0)))
        return result
    if operation == "set_lr":
        var lr = _parallel_float(args[1], "lr")
        if lr <= 0:
            raise Error("Byte-LM requires lr > 0")
        if rec[].handle != 0:
            if rec[].schedule != PARALLEL_REPLICA:
                raise Error("RuntimeError: installed byte-LM binding lacks set_lr")
            var floats: List[Float64] = [Float64(lr)]
            var stored = _parallel_execute(record, PARALLEL_OP_SET_LR, List[Int](), List[Int](), floats)
            if UInt32(stored) != bitcast[DType.uint32](lr):
                raise Error("RuntimeError: native set_lr stored different float32 bits")
        var config = py_call(p[].bi.dict, rec[].state["config"])
        py_setitem(config, "lr", PythonObject(Float64(lr)))
        py_setitem(rec[].state, "config", config)
        return PythonObject(Float64(lr))
    if operation == "require_split":
        _parallel_require_split(record, String(args[1]))
        return PythonObject(None)
    if operation == "require_fold":
        _parallel_require_split(record, "shard_gradient")
        return PythonObject(None)
    if operation == "has_device_fold":
        parallel_byte_open(owner)
        return PythonObject(rec[].schedule == PARALLEL_REPLICA)
    if operation == "ownership":
        return _parallel_ownership(owner)
    if operation == "model_ownership":
        if rec[].schedule != PARALLEL_MODEL_POOL:
            raise Error("model_ownership requires PooledByteLanguageModelTrainer")
        return _parallel_ownership(owner)
    var n = rec[].n_total
    if operation == "shard_gradient" or operation == "shard_gradient_fold":
        var fold = operation == "shard_gradient_fold"
        var ids = _parallel_array(args[1], [rec[].dims[0], rec[].dims[1] + 1], True)
        parallel_byte_open(owner)
        _parallel_require_split(record, operation)
        var loss = List[Float64](length=1, fill=0)
        var gradient = PythonObject(None)
        var addresses: List[Int] = [Int(py=ids._addr)]
        if not fold:
            gradient = generation_array([n])
            addresses.append(Int(py=gradient._addr))
        addresses.append(Int(loss.unsafe_ptr()))
        var ints: List[Int] = [_parallel_completed(record)]
        _ = _parallel_execute(record, PARALLEL_OP_SHARD_GRADIENT_FOLD if fold else PARALLEL_OP_SHARD_GRADIENT,
                              addresses^, ints, List[Float64]())
        _ = ids^
        var value = PythonObject(loss[0])
        if fold:
            return value
        var result = p[].bi.list()
        _ = py_call(result.append, value)
        _ = py_call(result.append, gradient)
        return py_call(p[].bi.tuple, result)
    if operation == "apply_gradient":
        # `total` is documented as the bytes `cross_vendor.ordered_fold`
        # returns; admit raw float32 bytes the way fold_reset/fold_add do.
        var total = _parallel_array(_parallel_float_bytes(args[1]), [n])
        parallel_byte_open(owner)
        _parallel_require_split(record, "apply_gradient")
        var before = _parallel_completed(record)
        var addresses: List[Int] = [Int(py=total._addr)]
        var ints: List[Int] = [before]
        var completed = _parallel_execute(record, PARALLEL_OP_APPLY_GRADIENT, addresses^, ints, List[Float64]())
        if completed != before + 1:
            rec[].lost = True
            raise Error("RuntimeError: apply_gradient returned wrong step")
        _parallel_commit(record, completed)
        _ = total^
        return PythonObject(completed)
    if operation == "fold_reset" or operation == "fold_add":
        var total = PythonObject(None)
        var addresses = List[Int]()
        if args[1] is not None:
            total = _parallel_array(_parallel_float_bytes(args[1]), [n])
            addresses.append(Int(py=total._addr))
        elif operation == "fold_add":
            raise Error("fold_add requires a gradient")
        parallel_byte_open(owner)
        _parallel_require_split(record, operation)
        _ = _parallel_execute(record, PARALLEL_OP_FOLD_RESET if operation == "fold_reset" else PARALLEL_OP_FOLD_ADD,
                              addresses^, List[Int](), List[Float64]())
        _ = total^
        return PythonObject(None)
    if operation == "fold_export":
        var into = args[1]
        var output = generation_array([n]) if into is None else into
        var pins = List[PythonObject]()
        var addresses: List[Int] = [_parallel_target(output, n, False, pins)]
        parallel_byte_open(owner)
        _parallel_require_split(record, "fold_export")
        _ = _parallel_execute(record, PARALLEL_OP_FOLD_EXPORT, addresses^, List[Int](), List[Float64]())
        var result = py_call(output._mv.cast, "B").tobytes() if into is None else output
        _ = pins^
        return result
    raise Error("parallel: unknown native operation " + operation)


def parallel_byte_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = parallel_byte_call(String(args[0]), args[1])
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
