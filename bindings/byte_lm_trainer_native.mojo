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
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind, classical_is_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.native_buffer_owner import native_buffer_view
from bindings.tsa_hw_native import tsa_array
from core.xseq_marshal import neural_cast_i32_2d

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _TOKEN_REFUSAL = "token ID outside configured vocabulary"
comptime _MAX_STEP = 999999


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


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
                n = Int(py=py_call(Python.import_module("operator").index, v))
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
    var buffer = classical_is_array(source, context)
    if not buffer and py_type(source) is not b.list and py_type(source) is not b.tuple:
        try:
            _ = native_buffer_view(source)
            buffer = True
        except:
            buffer = False
    if not buffer:
        raise Error(refusal)
    classical_admit_value(source, context)
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
    var mode = py_call_typed(backend["default_mode"])
    if mode is None or py_type(mode) is not _b().str or (String(py=mode) != "identical" and String(py=mode) != "fast"):
        raise Error("RuntimeError: SmallByteLanguageModelTrainer requires process-selected IDENTICAL or FAST mode")
    if not Bool(py=py_call_typed(backend["numeric_mode"]) == mode):
        raise Error("RuntimeError: SmallByteLanguageModelTrainer requires process-selected IDENTICAL or FAST mode")
    return mode


def _has(binding: PythonObject, name: String) raises -> Bool:
    return Bool(py=py_call(_b().callable, py_call(_b().getattr, binding, name, PythonObject(None))))


def _load(shape: _Shape, context: PythonObject) raises -> PythonObject:
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
    return binding


def _forget(fields: PythonObject, context: PythonObject) raises:
    """`_forget_session`."""
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
    var returned = py_call_typed(binding.byte_lm_trainer_export_state, fields["_native_session"], addresses,
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
    _forget(fields, context)
    if session is not None:
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


def _is_resident(fields: PythonObject, shape: _Shape, context: PythonObject) raises -> Bool:
    if fields["_resident"] is None:
        var binding = _binding(fields, shape, context)
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


def _run_resident(fields: PythonObject, tokens: PythonObject, shape: _Shape, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var lost = fields["_lost_at"]
    if lost is not None:
        raise Error("RuntimeError: Byte-LM session lost at step " + _text(lost[0]) + "; last exported state is step "
                    + _text(lost[1]))
    var binding = _binding(fields, shape, context)
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
    _ = _mode(context)
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


def _run_stateless(fields: PythonObject, tokens: PythonObject, shape: _Shape, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var state = fields["_state"]
    if Int(py=state["completed_steps"]) >= _MAX_STEP:
        raise Error("ValueError: Byte-LM native call step counter is exhausted")
    var binding = _binding(fields, shape, context)
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


def byte_lm_train_step(trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`train_step(ids)`: `with self._lock: return self._run(ids, True)`."""
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    var result = PythonObject(None)
    try:
        var shape = _shape_of(fields["_state"])
        var tokens = _ids(args[0], shape, context)
        try:
            if _is_resident(fields, shape, context):
                result = _run_resident(fields, tokens, shape, context)
            else:
                result = _run_stateless(fields, tokens, shape, context)
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


def byte_lm_trainer_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        if operation != "train_step":
            raise Error("RuntimeError: unknown byte LM trainer operation " + operation)
        return byte_lm_train_step(call[0], call[2], call[3]).steal_data()
    except error:
        var text = String(error)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
