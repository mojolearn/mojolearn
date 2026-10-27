# SPDX-License-Identifier: Apache-2.0
"""`byte_lm_trainer_call(trainer, "train_step", (ids,), context)`: the one
native call of `LanguageModelTrainer.train_step`
(python/mojolearn/_byte_lm_impl.py `SmallByteLanguageModelTrainer`).

Under the trainer's own lock: the model shape read from the state (the
`state_shape` refusals, the profile and the tensor registry computed here),
the ids admitted and copied (int32, the exact batch shape), the binding
resolved and checked as `_load` checks it (a binding change suspends the
open session first), and the resident step (`byte_lm_trainer_step`, with
the session's create / admit / adopt / failure lifecycle) or the stateless
step (`byte_lm_trainer_run`). The same compiled entries with the same
argument words as the Python bodies, so the step keeps its bits.
"""
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_handle, neural_descriptor_info
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind, classical_is_array, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.neural_block_public import _neural_lengths
from core.neural_block_public_impl import neural_copy_row_prefixes
from core.host_helpers_impl import argmax_rows_f32
from providers.python.training_metadata_pyapi import training_f32_scalar, training_schedule_json
from providers.python.training_buffer_pyapi import training_f32_owned
from bindings.native_json import native_json_parse
from bindings.tsa_hw_native import tsa_array
from core.xseq_marshal import neural_cast_i32_2d

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _TOKEN_REFUSAL = "token ID outside configured vocabulary"
comptime _MAX_STEP = 999999


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _text(value: PythonObject) raises -> String:
    return String(py=py_call(_b().str, value))


# ---------------------------------------------------------------- the shape

struct _Shape(Copyable, Movable):
    """`ByteLanguageModelConfig`'s numbers: (batch, length, d_model, n_heads,
    n_kv, head_dim, intermediate, n_layers, vocab_size) and the chunked head
    flag."""
    var d: List[Int]
    var chunked: Bool

    def __init__(out self, var d: List[Int], chunked: Bool):
        self.d = d^
        self.chunked = chunked

    def batch(self) -> Int:
        return self.d[0]

    def length(self) -> Int:
        return self.d[1]

    def vocab(self) -> Int:
        return self.d[8]

    def layers(self) -> Int:
        return self.d[7]

    def n_tensors(self) -> Int:
        return 2 + 9 * self.d[7]

    def shapes(self) -> List[List[Int]]:
        """`parameter_shapes`, in packing order."""
        var dm = self.d[2]
        var kd = self.d[4] * self.d[5]
        var ff = self.d[6]
        var out = List[List[Int]]()
        out.append([self.d[8], dm])
        for _ in range(self.d[7]):  # driver-loop: over the decoder layers (model structure)
            out.append([dm])
            out.append([dm, dm])
            out.append([kd, dm])
            out.append([kd, dm])
            out.append([dm, dm])
            out.append([dm])
            out.append([ff, dm])
            out.append([ff, dm])
            out.append([dm, ff])
        out.append([self.d[8], dm])
        return out^

    def names(self) -> List[String]:
        """`parameter_names`, in packing order."""
        var block = [String("norm1_w"), String("w_q"), String("w_k"), String("w_v"), String("w_o"),
                     String("norm2_w"), String("w_gate"), String("w_up"), String("w_down")]
        var out = List[String]()
        out.append(String("embed"))
        for layer in range(self.d[7]):  # driver-loop: over the decoder layers (model structure)
            for k in range(9):  # small-loop(the nine tensors of one block)
                out.append(String("block") + String(layer) + "." + block[k])
        out.append(String("lm_head"))
        return out^

    def n_total(self) -> Int:
        var total = 0
        var shapes = self.shapes()
        for i in range(len(shapes)):  # driver-loop: over the parameter tensors (model structure)
            var count = 1
            for j in range(len(shapes[i])):  # small-loop(one tensor's shape words)
                count *= shapes[i][j]
            total += count
        return total

    def native_shape(self) raises -> PythonObject:
        """`list(shape.native_shape)`: the nine dimensions, and 1 with the
        chunked head."""
        var out = Python.list()
        for i in range(9):  # small-loop(the nine dimensions)
            _ = py_call(out.append, self.d[i])
        if self.chunked:
            _ = py_call(out.append, 1)
        return out

    def profile(self, default_profile: String) -> String:
        """`shape.profile`."""
        var b = self.d[0]
        var l = self.d[1]
        var dm = self.d[2]
        var h = self.d[3]
        var kv = self.d[4]
        var hd = self.d[5]
        var ff = self.d[6]
        var layers = self.d[7]
        var vocab = self.d[8]
        if (not self.chunked and b == 2 and l == 32 and dm == 32 and h == 4 and kv == 2 and hd == 8 and ff == 64
                and layers == 2 and vocab == 256):
            return default_profile
        var suffix = String("-v256-blocks2.fp32.v2") if (layers == 2 and vocab == 256) else (
            String("-v") + String(vocab) + "-blocks" + String(layers) + ".fp32.v3")
        var head = String("-lmhead-chunk256-v2") if self.chunked else String("")
        return (String("mojolearn.byte-lm.b") + String(b) + "-l" + String(l) + "-d" + String(dm) + "-h" + String(h)
                + "-kv" + String(kv) + "-hd" + String(hd) + "-ff" + String(ff) + suffix + head)


def _shape_of(state: PythonObject) raises -> _Shape:
    """`state_shape(state)` (the config's dimension refusals included)."""
    var b = _b()
    var dims: List[Int] = [2, 32, 32, 4, 2, 8, 64, 2, 256]
    if not py_contains(state, "model_shape"):
        return _Shape(dims^, False)
    var value = state["model_shape"]
    var names = [String("batch"), String("length"), String("d_model"), String("n_heads"), String("n_kv"),
                 String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]
    if py_type(value) is not b.dict:
        raise Error("ValueError: Byte-LM model_shape has missing or unknown dimensions")
    var keys = py_call(b.set, value)
    var all_keys = b.set()
    for i in range(9):  # small-loop(the nine dimension names)
        _ = py_call(all_keys.add, names[i])
    _ = py_call(all_keys.add, "chunked_lm_head_v2")
    var no_chunk = py_call(all_keys.difference, Python.list("chunked_lm_head_v2"))
    var no_layers = py_call(all_keys.difference, Python.list("n_layers", "vocab_size"))
    var no_both = py_call(all_keys.difference, Python.list("n_layers", "vocab_size", "chunked_lm_head_v2"))
    if not (Bool(py=keys == all_keys) or Bool(py=keys == no_chunk) or Bool(py=keys == no_layers) or Bool(py=keys == no_both)):
        raise Error("ValueError: Byte-LM model_shape has missing or unknown dimensions")
    for i in range(9):  # small-loop(the nine dimension names)
        if py_contains(value, names[i]):
            var v = value[names[i]]
            if py_type(v) is b.bool or String(py=py_type(v).__name__) == "bool_":
                raise Error("ValueError: Byte-LM shape requires integer dimensions")
            var n = 0
            try:
                n = Int(py=py_call(native_loaded_module("operator").index, v))
            except:
                raise Error("ValueError: Byte-LM shape requires integer dimensions")
            if not (0 < n and n <= (1 << 20)):
                raise Error("ValueError: Byte-LM dimensions must be in [1, 2**20]")
            dims[i] = n
    var chunked = False
    if py_contains(value, "chunked_lm_head_v2"):
        if py_type(value["chunked_lm_head_v2"]) is not b.bool:
            raise Error("ValueError: chunked_lm_head_v2 must be a boolean")
        chunked = Bool(py=value["chunked_lm_head_v2"])
    return _Shape(dims^, chunked)


# ---------------------------------------------------------------- inputs and outputs

def _ids(source: PythonObject, shape: _Shape, context: PythonObject) raises -> PythonObject:
    """`_array(ids, (batch, length + 1), "ids", "<i4")`: an owned C-ordered
    int32 copy of an int32 buffer of exactly that shape."""
    var b = _b()
    var refusal = String("TypeError: Byte-LM ids must be a int32 array (a NumPy array, an array.array or a mojolearn Array)")
    classical_admit_value(source, context)
    var buffer = classical_is_array(source, context)
    if not buffer and py_type(source) is not b.list and py_type(source) is not b.tuple:
        try:
            _ = native_buffer_view(source)
            buffer = True
        except:
            buffer = False
    if not buffer:
        raise Error(refusal)
    var value = classical_input(source, context, "ids")
    var dtype = String(py=py_call(b.str, value.dtype))
    var item = Int(py=value.itemsize)
    if not ((dtype == "<i4" or dtype == "int32") and item == 4):
        raise Error(refusal)
    var rows = shape.batch()
    var cols = shape.length() + 1
    if Int(py=value.ndim) != 2 or Int(py=value.shape[0]) != rows or Int(py=value.shape[1]) != cols:
        raise Error("ValueError: Byte-LM ids requires finite shape (" + String(rows) + ", " + String(cols) + ")")
    var kind = classical_cast_kind(value, item, "ids", context)
    var tokens = tsa_array("i", "<i4", Python.tuple(rows, cols), rows * cols, 4, context)
    neural_cast_i32_2d(Int(py=classical_address(value)), Int(py=value.strides[0]), Int(py=value.strides[1]),
                       rows, cols, item, kind, _I32P(unsafe_from_address=Int(py=classical_address(tokens))))
    _ = value  # keep the Python object alive through the native call above
    _ = tokens  # keep the Python object alive through the native call above
    _ = value^
    return tokens


def _f32(n: Int, context: PythonObject) raises -> PythonObject:
    return tsa_array("f", "<f4", Python.tuple(n), n, 4, context)


def _i32(n: Int, context: PythonObject) raises -> PythonObject:
    return tsa_array("i", "<i4", Python.tuple(n), n, 4, context)


def _gradient_outputs(shape: _Shape, named: Bool, context: PythonObject) raises -> PythonObject:
    """`_gradient_outputs(shape, named)`: (flat, {name: Array})."""
    var flat = _f32(shape.n_total(), context)
    var by_name = _b().dict()
    if named:
        var names = shape.names()
        var shapes = shape.shapes()
        for i in range(len(names)):  # driver-loop: one output array per model parameter tensor
            var dims = Python.list()
            var count = 1
            for j in range(len(shapes[i])):  # small-loop(one tensor's shape words)
                _ = py_call(dims.append, shapes[i][j])
                count *= shapes[i][j]
            py_setitem(by_name, names[i], tsa_array("f", "<f4", py_call(_b().tuple, dims), count, 4, context))
    return Python.tuple(flat, by_name)


def _loss(out_loss: PythonObject) raises -> PythonObject:
    """`float(out_loss[0])`."""
    return PythonObject(Float64(_F32P(unsafe_from_address=Int(py=classical_address(out_loss)))[0]))


def _copy_i32(source: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    var out = _i32(n, context)
    if n > 0:
        memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(out))),
               src=_I32P(unsafe_from_address=Int(py=classical_address(source))), count=n)
        _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- the binding and the session

def _mode(context: PythonObject) raises -> PythonObject:
    """`_mode()`: the process-selected tier."""
    # the backend's own answers, as `_backend.default_mode()` /
    # `_backend.numeric_mode()` gave them (a setting changed after import
    # is seen; the bare `_DEFAULT_MODE` constant is not that answer)
    var backend = context["backend"]
    # Import-time connection resolves the initial default. Read its native
    # configuration once here; a product operation never invokes backend.py.
    var mode = backend["_DEFAULT_MODE"]
    if mode is None or py_type(mode) is not _b().str or (String(py=mode) != "identical" and String(py=mode) != "fast"):
        raise Error("RuntimeError: SmallByteLanguageModelTrainer requires process-selected IDENTICAL or FAST mode")
    return mode


def _has(binding: PythonObject, name: String) raises -> Bool:
    return Bool(py=py_call(_b().callable, py_call(_b().getattr, binding, name, PythonObject(None))))


def _load_snapshot(shape: _Shape, context: PythonObject) raises -> Tuple[PythonObject, PythonObject, String]:
    """`_load(shape)`: the byte LM binding of the process tier, its profile,
    mode and vendor read back."""
    var mode = _mode(context)
    var binding = context["byte_lm_bindings"][mode]
    if py_type(binding) is _b().str:
        raise Error(String(py=binding))
    var suffix = String(py=py_call(_b().str, py_call(binding.byte_lm_arithmetic_suffix)))
    var code = Int(py=py_call(binding.byte_lm_numeric_mode))
    var expected = 1 if String(py=mode) == "identical" else 0
    var vendor = String(py=py_call(_b().str, py_call(binding.byte_lm_vendor)))
    var vendors = context["byte_lm_vendors"]
    var vendor_ok = False
    for i in range(len(vendors)):  # small-loop(the admitted vendor names)
        vendor_ok = vendor_ok or String(py=vendors[i]) == vendor
    var profile = String(py=context["byte_lm_profile"])
    if code != expected or String(py=py_call(_b().str, py_call(binding.byte_lm_profile))) != profile + suffix or not vendor_ok:
        raise Error("RuntimeError: Byte-LM requires the exact native profile, the selected IDENTICAL/FAST mode and "
                    + "CUDA/HIP/Metal vendor (or, on a CPU-only install, the CPU byte LM binding)")
    if not _has(binding, "byte_lm_run"):
        raise Error("ImportError: Byte-LM binding is missing byte_lm_run; rebuild bindings/build_byte_lm.sh")
    var shape_profile = shape.profile(profile)
    if shape_profile != profile:
        if not _has(binding, "byte_lm_run_configured") or not _has(binding, "byte_lm_config_profile"):
            raise Error("ImportError: Byte-LM binding lacks runtime shapes; rebuild bindings/build_byte_lm.sh")
        if String(py=py_call(_b().str, py_call(binding.byte_lm_config_profile, shape.native_shape()))) != shape_profile + suffix:
            raise Error("RuntimeError: Byte-LM native runtime shape/profile mismatch")
    return (binding, mode, suffix)


def _load(shape: _Shape, context: PythonObject) raises -> PythonObject:
    var snapshot = _load_snapshot(shape, context)
    return snapshot[0]


def _provider(fields: PythonObject, binding: PythonObject) raises -> PythonObject:
    var factory = py_call(_b().getattr, binding, "byte_lm_trainer_api_v1", PythonObject(None))
    if factory is None:
        return PythonObject(None)
    var provider = neural_descriptor_reuse(py_get(fields, "_native_provider"), factory,
        "mojolearn.training.byte_lm.v1", 8)
    py_setitem(fields, "_native_provider", provider)
    return provider


def _session_handle(fields: PythonObject, binding: PythonObject, provider: PythonObject) raises -> Int:
    var session = fields["_native_session"]
    var factory = binding.byte_lm_native_session_handle_v1
    if py_get(fields, "_provider_session") is session and py_get(fields, "_provider_handle_factory") is factory and py_get(fields, "_provider_handle_owner") is provider:
        return Int(py=fields["_provider_handle"])
    var handle = neural_descriptor_handle(provider, factory, session)
    py_setitem(fields, "_provider_session", session)
    py_setitem(fields, "_provider_handle_factory", factory)
    py_setitem(fields, "_provider_handle_owner", provider)
    py_setitem(fields, "_provider_handle", handle)
    return handle


def _native_plan(shape: _Shape, state: PythonObject, train: Bool, batch: Int = 0,
                 length: Int = 0, release: Bool = False) raises -> Tuple[List[Int], List[Float64]]:
    var cfg = state["config"]
    var ints = List[Int]()
    for i in range(9):  # small-loop(the admitted model shape)
        ints.append(shape.d[i])
    ints.append(Int(shape.chunked))
    ints.append(Int(train))
    ints.append(Int(py=state["completed_steps"]))
    ints.append(Int(py=cfg["kind"]))
    ints.append(Int(py=cfg["nesterov"]))
    ints.append(batch)
    ints.append(length)
    ints.append(Int(release))
    var floats = List[Float64]()
    for key in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay"), String("momentum"), String("dampening"), String("max_norm")]:  # small-loop(the eight optimizer scalars)
        floats.append(Float64(py=cfg[key]))
    return (ints^, floats^)


def _native_execute(fields: PythonObject, binding: PythonObject, provider: PythonObject,
                    operation: Int, addresses: PythonObject, shape: _Shape, state: PythonObject,
                    train: Bool, batch: Int = 0, length: Int = 0, release: Bool = False) raises -> Int:
    var session = fields["_native_session"]
    var a = List[Int]()
    a.append(_session_handle(fields, binding, provider))
    for i in range(len(addresses)):  # small-loop(the admitted operation buffer descriptors)
        a.append(Int(py=addresses[i]))
    var plan = _native_plan(shape, state, train, batch, length, release)
    var result = neural_descriptor_execute(provider, operation, a, plan[0], plan[1])
    _ = session  # owner is live through the synchronous native call
    _ = binding
    return result


def _forget(fields: PythonObject, context: PythonObject) raises:
    """`_forget_session`."""
    py_setitem(fields, "_provider_session", PythonObject(None))
    py_setitem(fields, "_provider_handle_owner", PythonObject(None))
    py_setitem(fields, "_provider_handle_factory", PythonObject(None))
    py_setitem(fields, "_provider_handle", 0)
    py_setitem(fields, "_native_session", PythonObject(None))
    py_setitem(fields, "_session_binding", PythonObject(None))
    py_setitem(fields, "_session_open", False)
    var pool = py_call(_b().object.__new__, context["byte_lm_logits_pool"])
    var pool_fields = py_fields(pool)
    py_setitem(pool_fields, "keep", 2)
    py_setitem(pool_fields, "free", _b().list())
    py_setitem(fields, "_logits_pool", pool)
    py_setitem(fields, "_grad_step", -1)


def _mark_lost(fields: PythonObject) raises:
    if fields["_lost_at"] is None:
        py_setitem(fields, "_lost_at", Python.tuple(fields["_state"]["completed_steps"], fields["_last_export_step"]))


def _export_release(fields: PythonObject, shape: _Shape, context: PythonObject) raises -> PythonObject:
    """`_export_state_impl(release=True)`."""
    var state = fields["_state"]
    var binding = fields["_session_binding"]
    var n = shape.n_total()
    var out_p = _f32(n, context)
    var out_m = _f32(n, context)
    var out_v = _f32(n, context)
    var out_flags = _i32(shape.n_tensors(), context)
    var completed = state["completed_steps"]
    var addresses = Python.list(classical_address(state["flags"]), classical_address(out_p), classical_address(out_m),
                                classical_address(out_v), classical_address(out_flags))
    _ = out_m  # keep the Python object alive through the native call above
    _ = out_p  # keep the Python object alive through the native call above
    _ = out_flags  # keep the Python object alive through the native call above
    _ = out_v  # keep the Python object alive through the native call above
    var provider = _provider(fields, binding)
    var returned = PythonObject(_native_execute(fields, binding, provider, 3, addresses, shape, state, False, release=True)) if provider is not None else py_call_typed(binding.byte_lm_trainer_export_state, fields["_native_session"], addresses,
                                 shape.native_shape(), completed, PythonObject(True))
    _ = out_flags  # keep the Python object alive through the native call above
    _ = out_v  # keep the Python object alive through the native call above
    _ = out_m  # keep the Python object alive through the native call above
    _ = out_p  # keep the Python object alive through the native call above
    if py_type(returned) is not _b().int or not Bool(py=returned == completed):
        raise Error("RuntimeError: Byte-LM state export returned a different completed step")
    py_setitem(fields, "_last_export_step", completed)
    var exported = py_call(_b().dict, state)
    py_setitem(exported, "parameters", out_p)
    py_setitem(exported, "m", out_m)
    py_setitem(exported, "v", out_v)
    py_setitem(exported, "flags", out_flags)
    return exported


def _suspend(fields: PythonObject, shape: _Shape, context: PythonObject) raises:
    """`_suspend_session`."""
    if Bool(py=fields["_session_open"]) and fields["_lost_at"] is None:
        var exported = PythonObject(None)
        try:
            exported = _export_release(fields, shape, context)
        except error:
            # the native export closed the session on its way out
            _mark_lost(fields)
            _forget(fields, context)
            raise error^
        py_setitem(fields, "_state", exported)
        _forget(fields, context)
        return
    var session = fields["_native_session"]
    var binding = fields["_session_binding"]
    var provider = _provider(fields, binding) if session is not None else PythonObject(None)
    var handle = _session_handle(fields, binding, provider) if provider is not None else 0
    _forget(fields, context)
    if session is not None:
        if provider is not None:
            _ = neural_descriptor_execute(provider, 4, [handle], List[Int](), List[Float64]())
            _ = session
        else:
            _ = py_call(binding.byte_lm_session_close, session)


def _binding(fields: PythonObject, shape: _Shape, context: PythonObject) raises -> PythonObject:
    """`_binding()`: a binding change suspends the open session first; the
    run metadata is then computed on its next read (`run_metadata`)."""
    var binding = _load(shape, context)
    if binding is not fields["_runtime_binding"]:
        _suspend(fields, shape, context)
        py_setitem(fields, "_runtime", PythonObject(None))
        py_setitem(fields, "_runtime_binding", binding)
    return binding


def _is_resident(fields: PythonObject, binding: PythonObject, context: PythonObject) raises -> Bool:
    if fields["_resident"] is None:
        var names = context["byte_lm_session_entries"]
        var all_there = True
        for i in range(len(names)):  # small-loop(the session entry names)
            all_there = all_there and _has(binding, String(py=names[i]))
        py_setitem(fields, "_resident", all_there)
    return Bool(py=fields["_resident"])


def _parameters(state: PythonObject, train: Bool) raises -> PythonObject:
    """`_parameters(train)`: the twelve scalars every native step reads."""
    var cfg = state["config"]
    var b = _b()
    return Python.list(1 if train else 0, state["completed_steps"], cfg["kind"], cfg["lr"], cfg["beta1"], cfg["beta2"],
                       cfg["eps"], cfg["weight_decay"], cfg["momentum"], cfg["dampening"],
                       Int(py=py_call(b.int, cfg["nesterov"])), cfg["max_norm"])


def _adopt_open(fields: PythonObject) raises:
    """`_adopt_open_session`: the device holds the only copy now."""
    if not Bool(py=fields["_session_open"]):
        py_setitem(fields, "_session_open", True)
        py_setitem(fields, "_grad_step", -1)
        var state = fields["_state"]
        py_setitem(fields, "_last_export_step", state["completed_steps"])
        var dropped = py_call(_b().dict, state)
        py_setitem(dropped, "parameters", PythonObject(None))
        py_setitem(dropped, "m", PythonObject(None))
        py_setitem(dropped, "v", PythonObject(None))
        py_setitem(fields, "_state", dropped)


def _after_failure(fields: PythonObject) raises:
    """`_after_native_failure`: the native call's verdict read back."""
    py_setitem(fields, "_grad_step", -1)
    var session = fields["_native_session"]
    var binding = fields["_session_binding"]
    if session is None or fields["_lost_at"] is not None:
        return
    var is_open = Bool(py=fields["_session_open"])
    var usable = False
    var completed = PythonObject(None)
    try:
        var provider = _provider(fields, binding)
        if provider is not None:
            var info = List[Int](length=4, fill=0)
            _ = neural_descriptor_execute(provider, 5, [_session_handle(fields, binding, provider), Int(info.unsafe_ptr())], List[Int](), List[Float64]())
            is_open = info[3] != 0
            usable = info[2] != 0
            completed = PythonObject(info[0])
            _ = info^
            _ = session
        else:
            var info = py_call(binding.byte_lm_session_info, session)
            is_open = Bool(py=info[3])
            usable = Bool(py=info[2])
            completed = info[0]
    except:
        is_open = Bool(py=fields["_session_open"])
        usable = False
        completed = PythonObject(None)
    if is_open:
        _adopt_open(fields)
    if Bool(py=fields["_session_open"]) and (not usable or not Bool(py=completed == fields["_state"]["completed_steps"])):
        _mark_lost(fields)


def _run_resident(fields: PythonObject, tokens: PythonObject, shape: _Shape, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var lost = fields["_lost_at"]
    if lost is not None:
        raise Error("RuntimeError: Byte-LM session lost at step " + _text(lost[0]) + "; last exported state is step "
                    + _text(lost[1]))
    var names = context["byte_lm_session_entries"]
    var missing = String("")
    for i in range(len(names)):  # small-loop(the session entry names)
        if not _has(binding, String(py=names[i])):
            missing += (", " if missing != "" else "") + String(py=names[i])
    if missing != "":
        raise Error("ImportError: Byte-LM binding lacks owned sessions (" + missing + "); rebuild bindings/build_byte_lm.sh")
    var state = fields["_state"]
    var completed = Int(py=state["completed_steps"])
    if completed >= _MAX_STEP:
        raise Error("ValueError: Byte-LM native call step counter is exhausted")
    if fields["_native_session"] is None:
        py_setitem(fields, "_native_session", py_call(binding.byte_lm_session_create))
        py_setitem(fields, "_session_binding", binding)
        py_setitem(fields, "_session_open", False)
        py_setitem(fields, "_grad_step", -1)
    var session = fields["_native_session"]
    var full = String(py=fields["_step_result"]) == "full"
    var out_loss = _f32(1, context)
    var out_flags = _i32(shape.n_tensors(), context)
    var grads = _gradient_outputs(shape, True, context) if full else Python.tuple(PythonObject(None), b.dict())
    var addresses = Python.list(classical_address(tokens), classical_address(state["flags"]))
    if Bool(py=fields["_session_open"]):
        for _ in range(3):  # small-loop(the three admission slots)
            _ = py_call(addresses.append, 0)
    else:
        for key in [String("parameters"), String("m"), String("v")]:  # small-loop(the three named state arrays)
            _ = py_call(addresses.append, classical_address(state[key]))
    _ = py_call(addresses.append, classical_address(out_loss))
    _ = out_loss  # keep the Python object alive through the native call above
    _ = py_call(addresses.append, classical_address(out_flags))
    _ = out_flags  # keep the Python object alive through the native call above
    _ = out_loss  # keep the Python object alive through the native call above
    _ = py_call(addresses.append, classical_address(grads[0]) if full else PythonObject(0))
    _ = out_flags  # keep the Python object alive through the native call above
    _ = out_loss  # keep the Python object alive through the native call above
    var by_name = grads[1]
    var keys = py_call(b.list, by_name)
    for i in range(len(keys)):  # driver-loop: one address per model parameter tensor
        _ = py_call(addresses.append, classical_address(by_name[keys[i]]))
        _ = out_flags  # keep the Python object alive through the native call above
        _ = out_loss  # keep the Python object alive through the native call above
    py_setitem(fields, "_grad_step", -1)
    var returned = PythonObject(None)
    try:
        var provider = _provider(fields, binding)
        if provider is not None:
            returned = PythonObject(_native_execute(fields, binding, provider, 1, addresses, shape, state, True))
        else:
            returned = py_call_typed(binding.byte_lm_trainer_step, session, addresses, _parameters(state, True),
                                     shape.native_shape())
        _ = out_flags  # keep the Python object alive through the native call above
        _ = out_loss  # keep the Python object alive through the native call above
    except error:
        _after_failure(fields)
        raise error^
    _adopt_open(fields)
    var expected = completed + 1
    if py_type(returned) is not b.int or Int(py=returned) != expected:
        _mark_lost(fields)
        raise Error("RuntimeError: Byte-LM returned an invalid completed-step counter")
    var result = b.dict()
    py_setitem(result, "loss", _loss(out_loss))
    py_setitem(result, "step", expected)
    py_setitem(result, "completed_steps", expected)
    py_setitem(result, "next_batch_index", expected)
    if full:
        py_setitem(result, "flat_gradients", grads[0])
        py_setitem(result, "gradients", by_name)
    else:
        py_setitem(result, "flags", _copy_i32(out_flags, shape.n_tensors(), context))
    var committed = fields["_state"]
    py_setitem(committed, "flags", out_flags)
    py_setitem(committed, "completed_steps", expected)
    py_setitem(committed, "next_batch_index", expected)
    py_setitem(fields, "_grad_step", expected)
    return result


def _run_stateless(fields: PythonObject, tokens: PythonObject, shape: _Shape, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var state = fields["_state"]
    if Int(py=state["completed_steps"]) >= _MAX_STEP:
        raise Error("ValueError: Byte-LM native call step counter is exhausted")
    var entry = String(py=context["byte_lm_run_entry"])
    if not _has(binding, entry):
        raise Error("ImportError: Byte-LM binding lacks " + entry + "; rebuild bindings/build_byte_lm.sh")
    var n = shape.n_total()
    var out_p = _f32(n, context)
    var out_m = _f32(n, context)
    var out_v = _f32(n, context)
    var out_flags = _i32(shape.n_tensors(), context)
    var out_loss = _f32(1, context)
    var grads = _gradient_outputs(shape, True, context)
    var flat = grads[0]
    var by_name = grads[1]
    var addresses = Python.list()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(the four named state arrays)
        _ = py_call(addresses.append, classical_address(state[key]))
    for value in [tokens, out_p, out_m, out_v, flat, out_flags, out_loss]:  # small-loop(the step's seven buffers)
        _ = py_call(addresses.append, classical_address(value))
    var named = Python.list()
    var keys = py_call(b.list, by_name)
    for i in range(len(keys)):  # driver-loop: one address per model parameter tensor
        _ = py_call(named.append, classical_address(by_name[keys[i]]))
    var completed = py_call_typed(py_call(b.getattr, binding, entry), addresses, named, _parameters(state, True),
                                  shape.native_shape())
    var expected = Int(py=state["completed_steps"]) + 1
    if py_type(completed) is not b.int or Int(py=completed) != expected:
        raise Error("RuntimeError: Byte-LM returned an invalid completed-step counter")
    var committed = py_call(b.dict, state)
    py_setitem(committed, "parameters", out_p)
    py_setitem(committed, "m", out_m)
    py_setitem(committed, "v", out_v)
    py_setitem(committed, "flags", out_flags)
    py_setitem(committed, "completed_steps", expected)
    py_setitem(committed, "next_batch_index", expected)
    py_setitem(fields, "_state", committed)
    var result = b.dict()
    py_setitem(result, "loss", _loss(out_loss))
    py_setitem(result, "step", expected)
    py_setitem(result, "completed_steps", expected)
    py_setitem(result, "next_batch_index", expected)
    py_setitem(result, "flat_gradients", flat)
    py_setitem(result, "gradients", by_name)
    return result


def byte_lm_construct(trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var bi = _b()
    if py_type(trainer) is not context["byte_lm_trainer_class"]:
        raise Error("TypeError: Byte-LM requires its native trainer class without Python subclass hooks")
    var profile = py_get(context["byte_lm_profile_defaults"], "training", PythonObject("fp32_v1"))
    if py_type(profile) is not bi.str or not Bool(py=context["byte_lm_profiles"][profile]["training"]):
        raise Error("NotImplementedError: the selected numeric profile is unavailable for Byte-LM training")
    # Every external value is admitted before scalar, buffer or JSON protocols.
    for i in range(6):  # small-loop(parameter data, schedule and five optimizer arguments)
        classical_admit_value(args[i], context)
    var resident = args[7]
    if resident is not None and py_type(resident) is not bi.bool:
        raise Error("TypeError: resident must be a bool")
    var step_result = args[8]
    if step_result is None:
        step_result = PythonObject("lean" if resident is not None and Bool(py=resident) else "full")
    if py_type(step_result) is not bi.str or (String(py=step_result) != "full" and String(py=step_result) != "lean"):
        raise Error("ValueError: step_result must be 'full' or 'lean'")
    if String(py=step_result) == "lean" and (resident is None or not Bool(py=resident)):
        raise Error("ValueError: step_result='lean' requires resident=True: the stateless path destroys its context before returning, so there is nothing to export from")
    var config_value = args[6]
    var shape_state = bi.dict()
    if config_value is not None:
        if py_type(config_value) is not context["byte_lm_config_class"]:
            raise Error("TypeError: shape must be a ByteLanguageModelConfig")
        var config_fields = py_fields(config_value)
        classical_admit_value(config_fields, context)
        py_setitem(shape_state, "model_shape", config_fields)
    var shape = _shape_of(shape_state)
    # The selected native binding validates configured dimensions before any
    # allocation; a host-only glue binary must not impose its own numeric mode.
    var loaded = _load_snapshot(shape, context)
    var names = shape.names()
    var shapes = shape.shapes()
    var total = shape.n_total()
    var parameters = args[0]
    var flat: PythonObject
    if py_type(parameters) is bi.dict:
        if len(parameters) != len(names):
            raise Error("ValueError: Byte-LM named parameters must contain the exact configured tensor registry")
        flat = _f32(total, context)
        var dest = _F32P(unsafe_from_address=Int(py=classical_address(flat)))
        var offset = 0
        for i in range(len(names)):  # driver-loop: admit each configured parameter tensor once
            if not py_contains(parameters, names[i]):
                raise Error("ValueError: Byte-LM named parameters must contain the exact configured tensor registry")
            var owned = training_f32_owned(parameters[names[i]], shapes[i], "Byte-LM " + names[i], context)
            var count = 1
            for width in shapes[i]:  # small-loop(parameter rank)
                count *= width
            memcpy(dest=dest + offset, src=_F32P(unsafe_from_address=Int(py=classical_address(owned))), count=count)
            _ = owned^
            offset += count
    else:
        flat = training_f32_owned(parameters, [total], "Byte-LM parameters", context)
    var betas: PythonObject
    try:
        # Only already-admitted native containers/buffers reach iteration.
        betas = py_call(bi.tuple, args[3])
    except:
        raise Error("ValueError: Byte-LM betas must contain two scalars")
    if len(betas) != 2:
        raise Error("ValueError: Byte-LM betas must contain two scalars")
    var config = bi.dict()
    var keys = [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]
    var scalars: List[PythonObject] = [args[2], betas[0], betas[1], args[4], args[5]]
    var numeric = List[Float64]()
    for i in range(5):  # small-loop(the five optimizer configuration scalars)
        numeric.append(training_f32_scalar(scalars[i], "Byte-LM " + keys[i], context, True))
        py_setitem(config, keys[i], numeric[i])
    if numeric[0] <= 0 or numeric[3] <= 0 or numeric[4] < 0:
        raise Error("ValueError: Byte-LM requires lr > 0, eps > 0 and weight_decay >= 0")
    if numeric[1] < 0 or numeric[1] >= 1 or numeric[2] < 0 or numeric[2] >= 1:
        raise Error("ValueError: Byte-LM betas must remain in [0, 1) in float32")
    py_setitem(config, "kind", 2)
    py_setitem(config, "momentum", 0.0)
    py_setitem(config, "dampening", 0.0)
    py_setitem(config, "nesterov", False)
    py_setitem(config, "max_norm", 0.0)
    var schedule_text = training_schedule_json(args[1], 256, 16384, "Byte-LM data_schedule", context)
    var schedule = native_json_parse(PythonObject(schedule_text))
    var vocabulary = py_get(schedule, "vocabulary")
    if vocabulary is not None:
        if py_type(vocabulary) is not bi.dict or py_type(py_get(vocabulary, "n_vocab")) is not bi.int or py_type(py_get(vocabulary, "sha256")) is not bi.str:
            raise Error("ValueError: Byte-LM data_schedule vocabulary must carry an int n_vocab and a str sha256")
        if Int(py=vocabulary["n_vocab"]) != shape.vocab():
            raise Error("ValueError: Byte-LM data_schedule vocabulary does not match the configured vocabulary")
    var binding = loaded[0]
    var mode = loaded[1]
    var state = bi.dict()
    py_setitem(state, "schema", "mojolearn.small-byte-lm-state.v1")
    py_setitem(state, "profile", PythonObject(shape.profile(String(py=context["byte_lm_profile"])) + loaded[2]))
    py_setitem(state, "numeric_mode", mode)
    var named = Python.list()
    var shaped = Python.list()
    var offsets = Python.list(0)
    var offset = 0
    for i in range(len(names)):  # driver-loop: publish native registry metadata once
        _ = py_call(named.append, names[i])
        var dims = Python.list()
        var count = 1
        for width in shapes[i]:  # small-loop(parameter rank)
            _ = py_call(dims.append, width)
            count *= width
        _ = py_call(shaped.append, dims)
        offset += count
        _ = py_call(offsets.append, offset)
    py_setitem(state, "parameter_names", named)
    py_setitem(state, "parameter_shapes", shaped)
    py_setitem(state, "parameter_offsets", offsets)
    py_setitem(state, "parameters", flat)
    py_setitem(state, "m", _f32(total, context))
    py_setitem(state, "v", _f32(total, context))
    py_setitem(state, "flags", _i32(shape.n_tensors(), context))
    py_setitem(state, "completed_steps", 0)
    py_setitem(state, "next_batch_index", 0)
    py_setitem(state, "config", config)
    py_setitem(state, "data_schedule", schedule)
    if shape.profile(String(py=context["byte_lm_profile"])) != String(py=context["byte_lm_profile"]):
        py_setitem(state, "model_shape", py_call(bi.dict, shape_state["model_shape"]))
    var fields = py_fields(trainer)
    py_setitem(fields, "_resident", resident)
    py_setitem(fields, "_step_result", step_result)
    py_setitem(fields, "_lock", py_call(context["byte_lm_lock"]))
    py_setitem(fields, "_runtime", PythonObject(None))
    py_setitem(fields, "_runtime_binding", PythonObject(None))
    py_setitem(fields, "_last_export_step", 0)
    py_setitem(fields, "_lost_at", PythonObject(None))
    py_setitem(fields, "_state", state)
    _forget(fields, context)
    _ = binding
    return PythonObject(None)


def byte_lm_train_step(trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`train_step(ids)`: `with self._lock: return self._run(ids, True)`."""
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    var result = PythonObject(None)
    try:
        classical_admit_value(fields["_state"], context)
        var shape = _shape_of(fields["_state"])
        var tokens = _ids(args[0], shape, context)
        var binding = _binding(fields, shape, context)
        try:
            if _is_resident(fields, binding, context):
                result = _run_resident(fields, tokens, shape, binding, context)
            else:
                result = _run_stateless(fields, tokens, shape, binding, context)
        except error:
            var text = String(error)
            if not text.startswith("ValueError:") and text.find(_TOKEN_REFUSAL) >= 0:
                raise Error("ValueError: Byte-LM IDs must be in [0, " + String(shape.vocab()) + ")")
            raise error^
    except error:
        _ = py_call(lock.release)
        raise error^
    _ = py_call(lock.release)
    return result


def _logits_input(source: PythonObject, lengths: PythonObject, shape: _Shape,
                  context: PythonObject) raises -> Tuple[PythonObject, Int, Int, List[Int]]:
    classical_admit_value(source, context)
    classical_admit_value(lengths, context)
    var value = classical_input(source, context, "ids")
    var dims = value.shape
    if len(dims) != 2:
        raise Error("ValueError: ids must be [batch, 1.." + String(shape.length()) + "]")
    var batch = Int(py=dims[0])
    var length = Int(py=dims[1])
    if batch < 1 or batch > 1024 or length < 1 or length > shape.length() or batch * length > 268435456 // shape.vocab():
        raise Error("ValueError: Byte-LM logits input shape exceeds its admitted bounds")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "ids", context)
    var strides = value.strides
    var source_addr = Int(py=classical_address(value))
    var tokens = tsa_array("i", "<i4", Python.tuple(batch, length), batch * length, 4, context)
    var token_addr = Int(py=classical_address(tokens))
    neural_cast_i32_2d(source_addr, Int(py=strides[0]), Int(py=strides[1]), batch, length,
        item, kind, _I32P(unsafe_from_address=token_addr))
    _ = value^
    var sizes = List[Int](length=batch, fill=length)
    if lengths is not None:
        sizes = _neural_lengths(lengths, batch, length, context)
        var padded = tsa_array("i", "<i4", Python.tuple(batch, length), batch * length, 4, context)
        neural_copy_row_prefixes(Int(py=classical_address(padded)), token_addr, sizes, batch, length, 1)
        _ = tokens^
        tokens = padded
    return (tokens, batch, length, sizes^)


def _pooled_logits(fields: PythonObject, batch: Int, length: Int, vocab: Int,
                   context: PythonObject) raises -> PythonObject:
    var pool = py_get(fields, "_logits_pool")
    if pool is None:
        pool = native_storage_allocate(context["byte_lm_logits_pool"])
        py_setitem(py_fields(pool), "keep", 2)
        py_setitem(py_fields(pool), "free", Python.list())
        py_setitem(fields, "_logits_pool", pool)
    if py_type(pool) is not context["byte_lm_logits_pool"]:
        raise Error("TypeError: Byte-LM logits requires its native output pool")
    var free = py_fields(pool)["free"]
    if py_type(free) is not _b().list:
        raise Error("TypeError: Byte-LM logits pool requires a native list")
    var count = batch * length * vocab
    var store = PythonObject(None)
    for i in range(len(free)):  # small-loop(the bounded reusable host-buffer pool)
        if py_type(free[i]) is not context["array_type"]:
            raise Error("TypeError: Byte-LM logits pool contains a foreign buffer")
        if len(free[i]) == count:
            store = py_call(free.pop, i)
            break
    if store is None:
        store = py_call(context["array_type"], "f", py_call(context["native_bytes_type"], count * 4))
    var out = classical_wrap_array(store, Python.tuple(batch, length, vocab), "<f4", "C", context["byte_lm_pooled_context"])
    py_setitem(py_fields(out), "_pool", pool)
    return out


def byte_lm_pooled_finalize(owner: PythonObject, context: PythonObject) raises:
    # Installed as a compiled instance-method descriptor. A decref during a
    # native operation never invokes _PooledLogits.__del__ or pool.give Python.
    if py_type(owner) is not context["byte_lm_pooled_class"]:
        return
    var pool = py_get(py_fields(owner), "_pool")
    if pool is None or py_type(pool) is not context["byte_lm_logits_pool"]:
        return
    var store = py_call(context["byte_lm_store_getter"], owner, context["byte_lm_pooled_class"])
    if py_type(store) is not context["array_type"]:
        return
    var pool_fields = py_fields(pool)
    var free = py_get(pool_fields, "free")
    var keep = py_get(pool_fields, "keep")
    if py_type(free) is not _b().list or py_type(keep) is not _b().int or Int(py=keep) < 1:
        return
    for i in range(len(free)):  # small-loop(the bounded reusable host-buffer pool)
        if free[i] is store:
            return
    if len(free) >= Int(py=keep):
        _ = py_call(free.pop, 0)
    _ = py_call(free.append, store)


def byte_lm_pooled_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        byte_lm_pooled_finalize(args[1], args[0])
    except:
        # Finalization is best effort, as the previous Python __del__; a
        # malformed/expired pool must not lose the Array's owned storage.
        pass
    return PythonObject(None).steal_data()


def byte_lm_logits(trainer: PythonObject, args: PythonObject, context: PythonObject,
                   next_bytes: Bool = False) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    var result = PythonObject(None)
    try:
        classical_admit_value(fields["_state"], context)
        var shape = _shape_of(fields["_state"])
        var admitted = _logits_input(args[0], args[1], shape, context)
        var tokens = admitted[0]
        var batch = admitted[1]
        var length = admitted[2]
        var binding = _binding(fields, shape, context)
        var provider = _provider(fields, binding)
        var resident = _is_resident(fields, binding, context)
        var direct_next = next_bytes and resident and args[1] is None
        var expected = batch if direct_next else batch * length * shape.vocab()
        var out = _i32(batch, context) if direct_next else (_pooled_logits(fields, batch, length, shape.vocab(), context) if resident else tsa_array("f", "<f4", Python.tuple(batch, length, shape.vocab()), expected, 4, context))
        var written = 0
        if resident:
            var lost = fields["_lost_at"]
            if lost is not None:
                raise Error("RuntimeError: Byte-LM session lost at step " + _text(lost[0]) + "; last exported state is step " + _text(lost[1]))
            if fields["_native_session"] is None:
                py_setitem(fields, "_native_session", py_call(binding.byte_lm_session_create))
                py_setitem(fields, "_session_binding", binding)
                py_setitem(fields, "_session_open", False)
                py_setitem(fields, "_grad_step", -1)
            var state = fields["_state"]
            var addresses = Python.list(classical_address(tokens), classical_address(out), classical_address(state["flags"]))
            for key in [String("parameters"), String("m"), String("v")]:  # small-loop(the three session admission buffers)
                _ = py_call(addresses.append, 0 if Bool(py=fields["_session_open"]) else Int(py=classical_address(state[key])))
            try:
                if provider is not None:
                    written = _native_execute(fields, binding, provider, 6 if direct_next else 2, addresses, shape, state, False, batch, length)
                else:
                    var entry = binding.byte_lm_trainer_next_bytes if direct_next else binding.byte_lm_trainer_logits
                    written = Int(py=py_call_typed(entry, fields["_native_session"], addresses, Python.list(batch, length), _parameters(state, False), shape.native_shape()))
            except problem:
                _after_failure(fields)
                raise problem
            _adopt_open(fields)
        else:
            var state = fields["_state"]
            var addresses: List[Int] = [Int(py=classical_address(state["parameters"])), Int(py=classical_address(tokens)), Int(py=classical_address(out))]
            if provider is not None:
                var plan = _native_plan(shape, state, False, batch, length)
                written = neural_descriptor_execute(provider, 7, addresses, plan[0], plan[1])
            else:
                written = Int(py=py_call_typed(binding.byte_lm_logits, Python.list(addresses[0], addresses[1], addresses[2]), Python.list(batch, length), shape.native_shape()))
        _ = tokens
        if written != expected:
            raise Error("RuntimeError: byte LM returned an unexpected number of logits")
        if next_bytes:
            var indices = List[Int64](length=batch, fill=0)
            if direct_next:
                var source = _I32P(unsafe_from_address=Int(py=classical_address(out)))
                for row in range(batch):  # cpu-route: consume the native greedy output
                    indices[row] = Int64(source[row])
            else:
                var source = _F32P(unsafe_from_address=Int(py=classical_address(out)))
                for row in range(batch):  # cpu-route: same strict first-max owner on each last real row
                    argmax_rows_f32(source + (row * length + admitted[3][row] - 1) * shape.vocab(), 1, shape.vocab(), MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(indices.unsafe_ptr()) + row * 8))
            result = Python.list()
            for row in range(batch):  # driver-loop: publish the completed native token outputs
                _ = py_call(result.append, Int(indices[row]))
        elif args[1] is not None:
            result = tsa_array("f", "<f4", Python.tuple(batch, length, shape.vocab()), expected, 4, context)
            neural_copy_row_prefixes(Int(py=classical_address(result)), Int(py=classical_address(out)), admitted[3], batch, length, shape.vocab())
        else:
            result = out
        _ = out
    except problem:
        _ = py_call(lock.release)
        raise problem
    _ = py_call(lock.release)
    return result


def byte_lm_trainer_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        if operation == "construct":
            return byte_lm_construct(call[0], call[2], call[3]).steal_data()
        if operation == "train_step":
            return byte_lm_train_step(call[0], call[2], call[3]).steal_data()
        if operation == "logits" or operation == "next_bytes":
            return byte_lm_logits(call[0], call[2], call[3], operation == "next_bytes").steal_data()
        if operation == "native_provider_info":
            var fields = py_fields(call[0])
            return neural_descriptor_info(fields["_native_provider"]).steal_data()
        raise Error("RuntimeError: unknown byte LM trainer operation " + operation)
    except error:
        var text = String(error)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if text.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
