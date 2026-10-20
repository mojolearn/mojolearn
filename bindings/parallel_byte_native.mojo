# SPDX-License-Identifier: Apache-2.0
"""Native public owner for the existing byte-LM multi-device sessions.

The GPU session remains the numerical implementation. This owner admits and
packs buffers, preserves the portable checkpoint registry, and drives those
compiled entries without calling package Array/config/trainer Python methods.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import bitcast, memcpy
from std.math import isfinite
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array, py_ints
from bindings.native_json import native_json_parse
from bindings.native_transport import _transport_json, transport_initialize


struct ParallelByteBoundary(Defaultable, Movable):
    var bi: PythonObject
    var array_cls: PythonObject
    var shape_cls: PythonObject
    var modes: PythonObject
    var address: PythonObject
    var char: PythonObject
    var index: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.shape_cls = PythonObject(None)
        self.modes = PythonObject(None)
        self.address = PythonObject(None)
        self.char = PythonObject(None)
        self.index = PythonObject(None)


comptime PARALLEL_BYTE_BOUNDARY = _Global[
    StorageType=ParallelByteBoundary, name="MojoParallelByteBoundary",
    init_fn=ParallelByteBoundary.__init__,
]


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
    transport_initialize()
    if p[].bi.type(p[].address) is not p[].bi.type(p[].bi.len):
        raise Error("parallel: buffer address must be a native C entry")
    # Share the native Array allocator without resetting any generation model
    # classes already admitted by the generation boundary.
    var arrays = GENERATION_BOUNDARY.get_or_create_ptr()
    arrays[].bi = p[].bi
    arrays[].arrays = Python.import_module("array").array
    arrays[].array_cls = array_cls
    var mode_keys = p[].bi.list(modes)
    for k in range(len(mode_keys)):  # driver-loop: over installed numerical-mode ABI tables
        var mode = mode_keys[k]
        var binding = modes[mode]
        var table = p[].bi.dict()
        for name in [String("create"), String("open"), String("open_pooled"), String("close"),
                     String("step"), String("export"), String("rollback"), String("set_lr"),
                     String("ownership"), String("shard_gradient"), String("apply_gradient"),
                     String("fold_reset"), String("shard_gradient_fold"), String("fold_add"),
                     String("fold_export"), String("reduction_pool_available")]:
            var entry = PythonObject(None)
            try:
                entry = p[].bi.getattr(binding, "byte_lm_parallel_" + name, PythonObject(None))
            except:
                # _HostBinding refuses unsupported GPU names by NoCpuEntry;
                # resolve that connection-time proxy once and preserve a
                # native missing-entry refusal when the session is opened.
                entry = PythonObject(None)
            if entry is not None and p[].bi.type(entry) is not p[].bi.type(p[].bi.len):
                raise Error("parallel: session entry must be compiled native code")
            table[name] = entry
        var suffix = PythonObject(None)
        try:
            suffix = p[].bi.getattr(binding, "byte_lm_arithmetic_suffix", PythonObject(None))
        except:
            suffix = PythonObject(None)
        if suffix is None:
            suffix = p[].bi.getattr(binding, "byte_lm_host_arithmetic_suffix", PythonObject(None))
        if suffix is None or p[].bi.type(suffix) is not p[].bi.type(p[].bi.len):
            raise Error("parallel: binding has no native arithmetic profile")
        table["suffix"] = suffix()
        # Resolve and prove the same mode/vendor/profile witness as _load;
        # the connection-time host proxy is never retained as a runtime call.
        for name in [String("numeric_mode"), String("vendor"), String("profile"), String("config_profile")]:
            var native_name = "byte_lm_" + name
            if expected_vendor == "cpu":
                native_name = "byte_lm_host_" + ("profile" if name == "config_profile" else name)
            var entry = p[].bi.getattr(binding, native_name)
            if p[].bi.type(entry) is not p[].bi.type(p[].bi.len):
                raise Error("parallel: profile entry must be compiled native code")
            table["profile_" + name] = entry
        if Int(py=table["profile_numeric_mode"]()) != (1 if String(mode) == "identical" else 0) or String(table["profile_vendor"]()) != expected_vendor:
            raise Error("parallel: native mode/vendor mismatch")
        var default_profile = table["profile_profile"](py_ints([2, 32, 32, 4, 2, 8, 64, 2, 256])) if expected_vendor == "cpu" else table["profile_profile"]()
        if String(default_profile) != "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1" + String(table["suffix"]):
            raise Error("parallel: native default profile mismatch")
        table["binding"] = binding
        p[].modes[mode] = table


def _parallel_int(value: PythonObject, minimum: Int, maximum: Int, label: String, strict: Bool = True) raises -> Int:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    if bi.type(value) is bi.bool or String(bi.type(value).__name__) == "bool_" or (strict and bi.type(value) is not bi.int):
        raise Error(label + " must be an integer")
    var n = Int(py=value) if strict else Int(py=p[].index(value))
    if n < minimum or n > maximum:
        raise Error(label + " outside admitted range")
    return n


def _parallel_span(dims: List[Int]) raises -> Int:
    var count = 1
    for d in dims:
        if d < 0 or (d != 0 and count > 2147483647 // d):
            raise Error("Byte-LM shape exceeds signed 32-bit indexing")
        count *= d
    return count


def _parallel_float(value: PythonObject, label: String) raises -> Float32:
    var bi = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()[].bi
    if bi.type(value) is bi.bool or String(bi.type(value).__name__) == "bool_":
        raise Error(label + " must be a finite float32 scalar")
    var number = Float64(py=value)
    if not isfinite(number) or abs(number) > Float64(3.4028234663852886e38):
        raise Error(label + " must be finite float32")
    return Float32(number)


def _parallel_view(value: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    return value._mv if p[].bi.type(value) is p[].array_cls else p[].bi.memoryview(value)


def _parallel_array(value: PythonObject, dims: List[Int], integer: Bool = False) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var own_array = p[].bi.type(value) is p[].array_cls
    var view = _parallel_view(value)
    var shape = value.shape if own_array else view.shape
    var fmt = String(view.format)
    if (integer and fmt != "i" and fmt != "=i" and fmt != "<i") or (
        not integer and fmt != "f" and fmt != "=f" and fmt != "<f") or Int(py=view.itemsize) != 4:
        raise Error("TypeError: parallel input requires exact float32 or int32 buffer dtype")
    if len(shape) != len(dims):
        raise Error("parallel input has wrong rank")
    for i in range(len(dims)):
        if Int(py=shape[i]) != dims[i]:
            raise Error("parallel input has wrong shape")
    var output = generation_array(dims, "i" if integer else "f")
    var store = GENERATION_BOUNDARY.get_or_create_ptr()[].arrays("i" if integer else "f")
    store.frombytes(view.tobytes("C"))
    var n = Int(py=output.size)
    if len(store) != n:
        raise Error("parallel input byte count differs from shape")
    var src = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=store.buffer_info()[0]))
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    if own_array and String(value.order) == "F" and len(dims) > 1:
        var strides = List[Int]()
        var stride = 1
        for axis in range(len(dims)):
            strides.append(stride)
            stride *= dims[axis]
        for i in range(n):
            var remaining = i
            var source = 0
            for reverse in range(len(dims)):
                var axis = len(dims) - reverse - 1
                source += (remaining % dims[axis]) * strides[axis]
                remaining //= dims[axis]
            dst[i] = src[source]
    else:
        memcpy(dest=dst, src=src, count=n)
    if not integer:
        var fp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
        for i in range(n):
            if not isfinite(fp[i]):
                raise Error("parallel input requires finite float32 values")
    _ = store^
    return output


def _parallel_target(value: PythonObject, count: Int, integer: Bool, mut pins: List[PythonObject]) raises -> Int:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var view = _parallel_view(value)
    var shape = value.shape if p[].bi.type(value) is p[].array_cls else view.shape
    var fmt = String(view.format)
    if (integer and fmt != "i") or (not integer and fmt != "f") or Int(py=view.itemsize) != 4:
        raise Error("TypeError: export target has wrong dtype")
    if len(shape) != 1 or Int(py=shape[0]) != count or not Bool(py=view.c_contiguous):
        raise Error("export target must have the exact contiguous one-dimensional shape")
    if Bool(py=view.readonly):
        raise Error("export target is read-only")
    if p[].bi.type(value) is p[].array_cls:
        if Bool(py=value._readonly):
            raise Error("export target is read-only")
        pins.append(value)
        return Int(py=value._addr)
    # Both calls are CPython C operations cached at ABI setup, not ctypes.py
    # helpers, ndarray.ctypes, Array.__buffer__ or any package conversion.
    var pin = p[].char.from_buffer(view)
    var address = Int(py=p[].address(pin))
    pins.append(pin)
    return address


def _parallel_schedule_check(value: PythonObject, depth: Int, mut budget: Int) raises:
    var bi = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()[].bi
    budget -= 1
    if budget < 0 or depth > 8:
        raise Error("Byte-LM schedule is too complex")
    var cls = bi.type(value)
    if cls is bi.dict:
        if len(value) > 128:
            raise Error("Byte-LM schedule has too many keys")
        var keys = bi.list(value)
        for i in range(len(keys)):
            if bi.type(keys[i]) is not bi.str or len(keys[i]) > 2048:
                raise Error("Byte-LM schedule keys exceed bounds")
            _parallel_schedule_check(value[keys[i]], depth + 1, budget)
    elif cls is bi.list:
        if len(value) > 128:
            raise Error("Byte-LM schedule list exceeds its bound")
        for i in range(len(value)):
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
    if bi.type(value) is not bi.dict or len(value) == 0:
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
    var shape = state.get("model_shape", PythonObject(None))
    if shape is not None:
        if p[].bi.type(shape) is not p[].bi.dict:
            raise Error("Byte-LM model_shape must be a dict")
        var keys = p[].bi.list(shape)
        for i in range(len(keys)):
            var key = String(keys[i])
            if key != "chunked_lm_head_v2" and key not in names:
                raise Error("Byte-LM model_shape has unknown dimensions")
        for i in range(7):
            if not Bool(py=shape.__contains__(PythonObject(names[i]))):
                raise Error("Byte-LM model_shape has missing dimensions")
        if Bool(py=shape.__contains__("n_layers")) != Bool(py=shape.__contains__("vocab_size")):
            raise Error("Byte-LM model_shape must specify both layers and vocabulary")
    var dims = p[].bi.list()
    var normalized = p[].bi.dict()
    for i in range(len(names)):
        var value = PythonObject(defaults[i]) if shape is None else shape.get(PythonObject(names[i]), PythonObject(defaults[i]))
        var n = _parallel_int(value, 1, 1048576, names[i], False)
        dims.append(n)
        normalized[names[i]] = PythonObject(n)
    var chunked = PythonObject(False) if shape is None else shape.get("chunked_lm_head_v2", PythonObject(False))
    if p[].bi.type(chunked) is not p[].bi.bool:
        raise Error("chunked_lm_head_v2 must be bool")
    normalized["chunked_lm_head_v2"] = chunked
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
    offsets.append(0)
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
        names_out.append(key)
        shapes_out.append(py_ints(tensor_shape))
        offsets.append(total)
    var base = String("mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1")
    if Bool(py=chunked) or b != 2 or l != 32 or dm != 32 or h != 4 or kv != 2 or hd != 8 or ff != 64 or layers != 2 or vocab != 256:
        var suffix = "-v256-blocks2.fp32.v2" if layers == 2 and vocab == 256 else "-v" + String(vocab) + "-blocks" + String(layers) + ".fp32.v3"
        base = "mojolearn.byte-lm.b" + String(b) + "-l" + String(l) + "-d" + String(dm) + "-h" + String(h) + "-kv" + String(kv) + "-hd" + String(hd) + "-ff" + String(ff) + suffix
        if Bool(py=chunked):
            base += "-lmhead-chunk256-v2"
    if Bool(py=chunked):
        dims.append(1)
    var result = p[].bi.dict()
    result["dims"] = dims
    result["model_shape"] = normalized
    result["names"] = names_out
    result["shapes"] = shapes_out
    result["offsets"] = offsets
    result["n_total"] = PythonObject(total)
    result["n_tensors"] = PythonObject(2 + layers * 9)
    result["profile_base"] = PythonObject(base)
    return result


def _parallel_validate_state(value: PythonObject, shape: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if p[].bi.type(value) is not p[].bi.dict:
        raise Error("Byte-LM state must be a dict")
    var names = [String("schema"), String("profile"), String("numeric_mode"), String("parameter_names"),
                 String("parameter_shapes"), String("parameter_offsets"), String("parameters"), String("m"),
                 String("v"), String("flags"), String("completed_steps"), String("next_batch_index"),
                 String("config"), String("data_schedule")]
    var has_shape = Bool(py=value.__contains__("model_shape"))
    if len(value) != len(names) + Int(has_shape):
        raise Error("Byte-LM state has missing or unknown fields")
    for name in names:
        if not Bool(py=value.__contains__(PythonObject(name))):
            raise Error("Byte-LM state is missing " + name)
    var mode = String(value["numeric_mode"])
    if not Bool(py=p[].modes.__contains__(PythonObject(mode))):
        raise Error("Byte-LM numeric mode is not installed")
    var table = p[].modes[mode]
    if String(table["profile_config_profile"](shape["dims"])) != String(shape["profile_base"]) + String(table["suffix"]):
        raise Error("Byte-LM native runtime shape/profile mismatch")
    if String(value["schema"]) != "mojolearn.small-byte-lm-state.v1" or String(value["profile"]) != String(shape["profile_base"]) + String(table["suffix"]):
        raise Error("Byte-LM state profile/registry/mode mismatch")
    if not Bool(py=value["parameter_names"] == shape["names"]) or not Bool(py=value["parameter_shapes"] == shape["shapes"]) or not Bool(py=value["parameter_offsets"] == shape["offsets"]):
        raise Error("Byte-LM state parameter registry mismatch")
    var cfg = value["config"]
    var cfg_names = [String("kind"), String("lr"), String("beta1"), String("beta2"), String("eps"),
                     String("weight_decay"), String("momentum"), String("dampening"), String("nesterov"), String("max_norm")]
    if p[].bi.type(cfg) is not p[].bi.dict or len(cfg) != len(cfg_names):
        raise Error("Byte-LM optimizer configuration mismatch")
    for name in cfg_names:
        if not Bool(py=cfg.__contains__(PythonObject(name))):
            raise Error("Byte-LM optimizer configuration is missing " + name)
    if _parallel_int(cfg["kind"], 2, 2, "optimizer kind") != 2 or p[].bi.type(cfg["nesterov"]) is not p[].bi.bool or Bool(py=cfg["nesterov"]):
        raise Error("Byte-LM first profile requires AdamW without Nesterov")
    var config = p[].bi.dict()
    config["kind"] = PythonObject(2)
    config["nesterov"] = PythonObject(False)
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay"),
                 String("momentum"), String("dampening"), String("max_norm")]:
        var number = _parallel_float(cfg[name], name)
        if ((name == "lr" or name == "eps") and number <= 0) or (name == "weight_decay" and number < 0):
            raise Error("Byte-LM requires lr > 0, eps > 0 and weight_decay >= 0")
        if (name == "beta1" or name == "beta2") and (number < 0 or number >= 1):
            raise Error("Byte-LM betas must remain in [0, 1) in float32")
        if (name == "momentum" or name == "dampening" or name == "max_norm") and number != 0:
            raise Error("Byte-LM first profile requires AdamW without clipping or SGD options")
        config[name] = PythonObject(Float64(number))
    var completed = _parallel_int(value["completed_steps"], 0, 999999, "completed_steps", False)
    if _parallel_int(value["next_batch_index"], 0, 999999, "next_batch_index", False) != completed:
        raise Error("Byte-LM schedule cursor must match completed_steps")
    var result = p[].bi.dict(value)
    var n = Int(py=shape["n_total"])
    for name in [String("parameters"), String("m"), String("v")]:
        result[name] = _parallel_array(value[name], [n])
    var flags = _parallel_array(value["flags"], [Int(py=shape["n_tensors"])], True)
    var vp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=result["v"]._addr))
    for i in range(n):
        if vp[i] < 0:
            raise Error("Byte-LM requires nonnegative second moments")
    var fp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(py=flags._addr))
    for i in range(Int(py=shape["n_tensors"])):
        if fp[i] < 0 or fp[i] > 1:
            raise Error("Byte-LM requires binary optimizer flags")
    result["flags"] = flags
    result["config"] = config
    result["completed_steps"] = PythonObject(completed)
    result["next_batch_index"] = PythonObject(completed)
    result["data_schedule"] = _parallel_schedule(value["data_schedule"])
    var vocabulary = result["data_schedule"].get("vocabulary", PythonObject(None))
    if vocabulary is not None:
        if p[].bi.type(vocabulary) is not p[].bi.dict or p[].bi.type(vocabulary.get("n_vocab")) is not p[].bi.int or p[].bi.type(vocabulary.get("sha256")) is not p[].bi.str:
            raise Error("Byte-LM data schedule vocabulary descriptor is invalid")
        if Int(py=vocabulary["n_vocab"]) != Int(py=shape["dims"][8]):
            raise Error("Byte-LM data schedule vocabulary does not match model")
    result["parameter_names"] = p[].bi.list(shape["names"])
    result["parameter_shapes"] = native_json_parse(PythonObject(_transport_json(shape["shapes"])))
    result["parameter_offsets"] = p[].bi.list(shape["offsets"])
    if has_shape:
        result["model_shape"] = p[].bi.dict(shape["model_shape"])
    return result


def parallel_byte_init(owner: PythonObject, state: PythonObject, devices: PythonObject,
                       logical_shards: PythonObject, pool_optimizer: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var count = _parallel_int(logical_shards, 1, 1024, "logical_shards")
    var ids = p[].bi.tuple(devices)
    if len(ids) < 1 or len(ids) > count:
        raise Error("require 1 <= device count <= logical_shards <= 1024")
    var unique = p[].bi.set()
    for i in range(len(ids)):
        var id = _parallel_int(ids[i], 0, 2147483647, "device index")
        if Bool(py=unique.__contains__(id)):
            raise Error("devices must be distinct nonnegative integer indices")
        unique.add(id)
    if p[].bi.type(pool_optimizer) is not p[].bi.bool:
        raise Error("pool_optimizer must be bool")
    if p[].bi.type(state) is not p[].bi.dict:
        raise Error("Byte-LM state must be a dict")
    var shape = _parallel_shape(state)
    var admitted = _parallel_validate_state(state, shape)
    owner.pool_optimizer = pool_optimizer
    owner.devices = ids
    owner.logical_shards = PythonObject(count)
    owner._state = admitted
    owner._native_shape = shape
    var shape_object = p[].bi.object.__new__(p[].shape_cls)
    # Initialize frozen dataclass storage directly, without invoking the
    # Python-generated __setattr__ or colliding with PythonObject's setter.
    shape_object.__dict__.update(shape["model_shape"])
    owner._shape = shape_object
    owner._session = PythonObject(None)
    owner._binding = PythonObject(None)
    owner._entries = p[].modes[admitted["numeric_mode"]]
    owner._closed = PythonObject(False)
    owner._lost = PythonObject(False)
    return PythonObject(None)


def parallel_byte_open(owner: PythonObject) raises:
    if Bool(py=owner._closed) or Bool(py=owner._lost):
        raise Error("RuntimeError: parallel trainer is closed or lost; restore a state export")
    if owner._session is not None:
        return
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var entries = owner._entries
    var open_name = "open_pooled" if Bool(py=owner.pool_optimizer) else "open"
    for name in [String("create"), String(open_name), String("close"), String("step"), String("export"), String("rollback"), String("reduction_pool_available")]:
        if entries[name] is None:
            raise Error("NotImplementedError: installed binding has no cooperative GPU byte-LM session " + name)
    if Int(py=entries["reduction_pool_available"]()) != 1:
        raise Error("RuntimeError: binding refused distributed reduction availability")
    var state = owner._state
    var config = state["config"]
    var params = p[].bi.list()
    params.append(0)
    params.append(state["completed_steps"])
    for key in [String("kind"), String("lr"), String("beta1"), String("beta2"), String("eps"),
                String("weight_decay"), String("momentum"), String("dampening"), String("nesterov"), String("max_norm")]:
        params.append(config[key])
    var addresses = p[].bi.list()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:
        addresses.append(state[key]._addr)
    var session = entries["create"]()
    try:
        var completed = entries[open_name](session, addresses, params, owner._native_shape["dims"], owner.devices, owner.logical_shards)
        if Int(py=completed) != Int(py=state["completed_steps"]):
            raise Error("RuntimeError: parallel admission returned wrong step")
    except error:
        _ = entries["close"](session)
        raise error
    owner._binding = entries["binding"]
    owner._session = session
    var retained = p[].bi.dict(state)
    retained["parameters"] = PythonObject(None)
    retained["m"] = PythonObject(None)
    retained["v"] = PythonObject(None)
    owner._state = retained


def _parallel_require_split(owner: PythonObject, entry: String) raises:
    if Bool(py=owner.pool_optimizer) or len(owner.devices) != 1:
        raise Error(entry + " requires devices=(d,) and pool_optimizer=False")
    if owner._entries[entry] is None:
        raise Error("RuntimeError: installed byte-LM binding lacks " + entry)


def _parallel_export(owner: PythonObject, rank: PythonObject, into: PythonObject, gradients: Bool = False) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var n = Int(py=owner._native_shape["n_total"])
    var nt = Int(py=owner._native_shape["n_tensors"])
    var output = into
    var addresses = p[].bi.list()
    var pins = List[PythonObject]()
    if gradients:
        if output is None:
            output = generation_array([n])
        addresses.append(_parallel_target(output, n, False, pins))
    else:
        var keys = [String("parameters"), String("m"), String("v"), String("flags")]
        if output is None:
            output = p[].bi.dict()
            for key in keys:
                output[key] = generation_array([nt if key == "flags" else n], "i" if key == "flags" else "f")
        if p[].bi.type(output) is not p[].bi.dict or len(output) != 4:
            raise Error("into must be a dict with parameters, m, v and flags")
        for key in keys:
            if not Bool(py=output.__contains__(PythonObject(key))):
                raise Error("into is missing " + key)
            addresses.append(_parallel_target(output[key], nt if key == "flags" else n, key == "flags", pins))
    parallel_byte_open(owner)
    var step = owner._entries["export"](owner._session, addresses, rank, PythonObject(gradients))
    if Int(py=step) != Int(py=owner._state["completed_steps"]):
        raise Error("RuntimeError: parallel export returned wrong step")
    _ = pins^
    return output


def parallel_byte_state(owner: PythonObject, rank: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var arrays = _parallel_export(owner, rank, PythonObject(None))
    var state = p[].bi.dict(owner._state)
    state.update(arrays)
    return _parallel_validate_state(state, owner._native_shape)


def parallel_byte_train_step(owner: PythonObject, shards: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    var parts = p[].bi.list(shards)
    if len(parts) != Int(py=owner.logical_shards):
        raise Error("logical shard count mismatch")
    var addresses = p[].bi.list()
    var tokens = List[PythonObject]()
    var b = Int(py=owner._native_shape["dims"][0])
    var length = Int(py=owner._native_shape["dims"][1])
    for k in range(len(parts)):  # driver-loop: over immutable logical training shards
        var ids = _parallel_array(parts[k], [b, length + 1], True)
        addresses.append(ids._addr)
        tokens.append(ids)
    parallel_byte_open(owner)
    var before = Int(py=owner._state["completed_steps"])
    # An entry failure already rolled back inside the native GPU session.
    # Do not roll back a prior committed step on an admission refusal.
    var losses = owner._entries["step"](owner._session, addresses, before)
    var result = p[].bi.dict()
    try:
        if len(losses) != Int(py=owner.logical_shards):
            raise Error("RuntimeError: parallel step returned wrong loss count")
        result["losses"] = p[].bi.tuple(losses)
        result["completed_steps"] = PythonObject(before + 1)
        result["logical_shards"] = owner.logical_shards
        result["reduction"] = PythonObject("ordered_sum")
    except error:
        try:
            if Int(py=owner._entries["rollback"](owner._session)) != before:
                owner._lost = PythonObject(True)
        except:
            owner._lost = PythonObject(True)
        raise error
    owner._state["completed_steps"] = PythonObject(before + 1)
    owner._state["next_batch_index"] = PythonObject(before + 1)
    _ = tokens^
    return result


def _parallel_float_bytes(value: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if p[].bi.type(value) is p[].bi.bytes or p[].bi.type(value) is p[].bi.bytearray:
        var store = GENERATION_BOUNDARY.get_or_create_ptr()[].arrays("f")
        store.frombytes(value)
        return store
    return value


def parallel_byte_call(operation: String, args: PythonObject) raises -> PythonObject:
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if operation == "initialize":
        parallel_byte_initialize(args[0], args[1], args[2], String(args[3]))
        return PythonObject(None)
    if operation == "init":
        return parallel_byte_init(args[0], args[1], args[2], args[3], args[4])
    if operation == "from_checkpoint":
        var checkpoint = args[1]
        if String(checkpoint.get("schema")) != "mojolearn.parallel-byte-lm.v1" or String(checkpoint.get("reduction")) != "ordered_sum":
            raise Error("unsupported parallel checkpoint contract")
        var restored = p[].bi.object.__new__(args[0])
        _ = parallel_byte_init(restored, checkpoint["state"], args[2], checkpoint["logical_shards"], args[3])
        return restored
    var owner = args[0]
    if operation == "step":
        return owner._state["completed_steps"]
    if operation == "enter":
        return owner
    if operation == "close":
        owner._closed = PythonObject(True)
        if owner._session is not None:
            _ = owner._entries["close"](owner._session)
            owner._session = PythonObject(None)
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
        result["schema"] = PythonObject("mojolearn.parallel-byte-lm.v1")
        result["logical_shards"] = owner.logical_shards
        result["reduction"] = PythonObject("ordered_sum")
        result["state"] = parallel_byte_state(owner, PythonObject(0))
        return result
    if operation == "set_lr":
        var lr = _parallel_float(args[1], "lr")
        if lr <= 0:
            raise Error("Byte-LM requires lr > 0")
        if owner._session is not None:
            if owner._entries["set_lr"] is None:
                raise Error("RuntimeError: installed byte-LM binding lacks set_lr")
            var stored = Int(py=owner._entries["set_lr"](owner._session, PythonObject(Float64(lr))))
            if UInt32(stored) != bitcast[DType.uint32](lr):
                raise Error("RuntimeError: native set_lr stored different float32 bits")
        var config = p[].bi.dict(owner._state["config"])
        config["lr"] = PythonObject(Float64(lr))
        owner._state["config"] = config
        return PythonObject(Float64(lr))
    if operation == "require_split":
        _parallel_require_split(owner, String(args[1]))
        return PythonObject(None)
    if operation == "require_fold":
        _parallel_require_split(owner, "shard_gradient")
        if owner._entries["fold_export"] is None:
            raise Error("RuntimeError: installed byte-LM binding lacks device fold")
        return PythonObject(None)
    if operation == "has_device_fold":
        parallel_byte_open(owner)
        return PythonObject(owner._entries["fold_export"] is not None)
    if operation == "ownership":
        parallel_byte_open(owner)
        var rows = owner._entries["ownership"](owner._session)
        var result = p[].bi.list()
        var fields = [String("first"), String("count"), String("moment_bytes"), String("rollback_bytes"), String("reduction_bytes")]
        if len(rows) != len(owner.devices):
            raise Error("RuntimeError: native ownership row count differs from devices")
        for i in range(len(rows)):  # driver-loop: over actual physical device ownership records
            var row = p[].bi.dict()
            row["device"] = owner.devices[i]
            for j in range(len(fields)):
                row[fields[j]] = rows[i][j]
            result.append(row)
        return p[].bi.tuple(result)
    var n = Int(py=owner._native_shape["n_total"])
    var addresses = p[].bi.list()
    if operation == "shard_gradient" or operation == "shard_gradient_fold":
        var ids = _parallel_array(args[1], [Int(py=owner._native_shape["dims"][0]), Int(py=owner._native_shape["dims"][1]) + 1], True)
        addresses.append(ids._addr)
        parallel_byte_open(owner)
        _parallel_require_split(owner, operation)
        var gradient = PythonObject(None)
        if operation == "shard_gradient":
            gradient = generation_array([n])
            addresses.append(gradient._addr)
        var loss = owner._entries[operation](owner._session, addresses, owner._state["completed_steps"])
        _ = ids^
        if operation == "shard_gradient_fold":
            return loss
        var result = p[].bi.list()
        result.append(loss)
        result.append(gradient)
        return p[].bi.tuple(result)
    if operation == "apply_gradient":
        var total = _parallel_array(args[1], [n])
        addresses.append(total._addr)
        parallel_byte_open(owner)
        _parallel_require_split(owner, "apply_gradient")
        var before = Int(py=owner._state["completed_steps"])
        var completed = owner._entries["apply_gradient"](owner._session, addresses, before)
        if Int(py=completed) != before + 1:
            owner._lost = PythonObject(True)
            raise Error("RuntimeError: apply_gradient returned wrong step")
        owner._state["completed_steps"] = completed
        owner._state["next_batch_index"] = completed
        _ = total^
        return completed
    if operation == "fold_reset" or operation == "fold_add":
        var total = PythonObject(None)
        if args[1] is not None:
            total = _parallel_array(_parallel_float_bytes(args[1]), [n])
            addresses.append(total._addr)
        elif operation == "fold_add":
            raise Error("fold_add requires a gradient")
        parallel_byte_open(owner)
        _parallel_require_split(owner, operation)
        _ = owner._entries[operation](owner._session, addresses)
        _ = total^
        return PythonObject(None)
    if operation == "fold_export":
        var into = args[1]
        var output = generation_array([n]) if into is None else into
        var pins = List[PythonObject]()
        addresses.append(_parallel_target(output, n, False, pins))
        parallel_byte_open(owner)
        _parallel_require_split(owner, "fold_export")
        _ = owner._entries["fold_export"](owner._session, addresses)
        var result = output._mv.cast("B").tobytes() if into is None else output
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
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error, ExceptionType.ValueError)
