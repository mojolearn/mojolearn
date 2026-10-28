# SPDX-License-Identifier: Apache-2.0
# Public Transformer decode-session admission/ownership. The Python class is a
# thin connection; GPU execution uses the common native provider and CPU copies
# stay native. No Python session methods or block hooks are invoked here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from std.memory import memcpy
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_setitem, py_setattr, py_type, py_fields
from bindings.generation_buffers import generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.causal_lm_construct import _model_bindings, _model_f32
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.neural_block_public import _neural_input
from providers.python.generation_metadata_pyapi import GenerationLayerSnapshot
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse
from providers.python.transformer_resident_pyapi import transformer_resident_decode_call
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME
from core.hostptr import f32_ptr


struct TransformerDecodeHandle(Movable, Writable):
    var binding: PythonObject
    var provider: PythonObject
    var device: PythonObject
    var open: Bool
    var weights: List[List[Float32]]
    var optional: List[List[Float32]]
    var k_cache: List[Float32]
    var v_cache: List[Float32]

    def __init__(out self, binding: PythonObject):
        self.binding = binding
        self.provider = PythonObject(None)
        self.device = PythonObject(None)
        self.open = False
        self.weights = List[List[Float32]]()
        self.optional = List[List[Float32]]()
        self.k_cache = List[Float32]()
        self.v_cache = List[Float32]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerDecodeHandle(open=", self.open, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _transformer_decode_profile(fields: PythonObject, slots: PythonObject) raises:
    var profile = py_get(fields, "_arithmetic_profile")
    var saved = py_get(slots, "arithmetic_profile")
    if saved is None and Int(py=slots["cached_tokens"]) != 0 and Bool(py=py_get(fields, "_arithmetic_profile_changed", PythonObject(False))):
        raise Error("TransformerState with cached tokens needs arithmetic profile metadata under an experimental version")
    if saved is not None and profile is not None and Bool(py=saved != profile):
        raise Error("TransformerState arithmetic profile differs from this block")
    if profile is not None:
        py_setitem(slots, "arithmetic_profile", profile)


def _transformer_decode_params(fields: PythonObject, slots: PythonObject, batch: Int, length: Int) raises -> List[Int]:
    var p: List[Int] = [batch, length, Int(py=fields["d_model"]), Int(py=fields["n_heads"]),
        Int(py=fields["n_kv_heads"]), Int(py=fields["head_dim"]), Int(py=fields["intermediate"]),
        Int(py=slots["max_tokens"]), Int(py=slots["cached_tokens"]), Int(py=fields["window"])]
    var tail = fields["_opts_tail"]
    for i in range(len(tail)):  # driver-loop: admitted fixed block-option words
        p.append(Int(py=tail[i]))
    return p^


def _transformer_decode_pieces(fields: PythonObject, slots: PythonObject,
                               mut keep: List[PythonObject], mut copies: List[PythonObject]) raises -> List[Int]:
    var window = Int(py=py_get(slots, "window", PythonObject(0)))
    var capacity = window if window > 0 else Int(py=slots["max_tokens"])
    var n = Int(py=slots["batch_size"]) * Int(py=fields["n_kv_heads"]) * capacity * Int(py=fields["head_dim"])
    return [_mutable(slots["k_cache"], [n], keep, copies, "k_cache"),
            _mutable(slots["v_cache"], [n], keep, copies, "v_cache"), n]


def _transformer_decode_copy(value: PythonObject, context: PythonObject) raises -> List[Float32]:
    if value is None:
        return List[Float32]()
    var array = _model_f32(value, "Transformer decode session weight", context)
    var result = List[Float32](length=Int(py=array.size), fill=0)
    if len(result) != 0:
        memcpy(dest=result.unsafe_ptr(), src=f32_ptr(Int(py=array._addr)), count=len(result))
    _ = array
    return result^


def transformer_decode_init(session: PythonObject, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = py_fields(block)
    var binding = _model_bindings(fields["_native_route"], "transformer", context, fields["numeric_mode"])["transformer"]
    var device = Bool(py=py_call(bi.hasattr, binding, "transformer_decode_session_create"))
    var packed = py_get(fields, "_int15") is not None
    if packed and (not device or not Bool(py=py_call(bi.hasattr, binding, "transformer_decode_session_open_int15"))):
        raise Error("NotImplementedError: TransformerDecodeSession fixed15_v1 needs its GPU resident session entry")
    if not device and not Bool(py=py_call(bi.hasattr, binding, "transformer_decode_step")):
        raise Error("NotImplementedError: TransformerDecodeSession binding has neither resident nor per-call decode")
    if state is None:
        raise Error("TransformerDecodeSession: state is required (allocate_state)")
    var slots = py_fields(state)
    _transformer_decode_profile(fields, slots)
    if py_get(slots, "_resident_session") is not None:
        raise Error("TransformerDecodeSession: the state belongs to an open resident session")
    var batch = Int(py=slots["batch_size"])
    if batch < 1:
        raise Error("TransformerDecodeSession: the state holds no rows")
    if Int(py=py_get(slots, "window", PythonObject(0))) != Int(py=fields["window"]):
        raise Error("TransformerDecodeSession: state window differs from this block")
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _transformer_decode_pieces(fields, slots, keep, copies)
    var handle = TransformerDecodeHandle(binding)
    var cached = Int(py=slots["cached_tokens"])
    py_setattr(session, "_block", block)
    py_setattr(session, "_state", state)
    if device:
        var snapshot = GenerationLayerSnapshot(fields, "transformer", packed)
        var a = snapshot.weights.resolve(keep)
        a.append(pieces[0]); a.append(pieces[1])
        var optional = snapshot.optional.resolve(keep)
        for address in optional:  # driver-loop: optional resident weight addresses
            a.append(address)
        var planes = snapshot.packed.resolve(keep)
        for address in planes:  # driver-loop: packed projection plane addresses
            a.append(address)
        var p = snapshot.words.transformer_params(batch, 0, Int(py=slots["max_tokens"]), cached)
        var open_p = List[Int]()
        for i in range(len(p)):  # driver-loop: resident shape words omit sequence length
            if i != 1:
                open_p.append(p[i])
        handle.device = _native_function(binding, "transformer_decode_session_create")()
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1" and String(fields["numeric_mode"]) == "identical":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_block_provider_cache"),
                _native_function(binding, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 25)
            cached = transformer_resident_decode_call(handle.provider, handle.device, 19 if packed else 18, a, open_p)
        else:
            cached = Int(py=py_call(_native_function(binding, "transformer_decode_session_open_int15" if packed else "transformer_decode_session_open"),
                handle.device, py_ints(a), py_ints(open_p)))
    else:
        var weights = fields["_w"]
        var optional = fields["_wopt"]
        for i in range(len(weights)):  # driver-loop: native session-owned weight copies
            handle.weights.append(_transformer_decode_copy(weights[i], context))
        for i in range(len(optional)):  # driver-loop: native session-owned optional weight copies
            handle.optional.append(_transformer_decode_copy(optional[i], context))
        handle.k_cache = List[Float32](length=pieces[2], fill=0)
        handle.v_cache = List[Float32](length=pieces[2], fill=0)
        memcpy(dest=handle.k_cache.unsafe_ptr(), src=f32_ptr(pieces[0]), count=pieces[2])
        memcpy(dest=handle.v_cache.unsafe_ptr(), src=f32_ptr(pieces[1]), count=pieces[2])
    handle.open = True
    py_setattr(session, "_handle", PythonObject(alloc=handle^))
    py_setitem(slots, "cached_tokens", PythonObject(cached))
    py_setitem(slots, "_resident_session", session)
    return PythonObject(None)


def transformer_decode_run(session: PythonObject, input: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    var hobj = py_fields(session)["_handle"]
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        raise Error("TransformerDecodeSession: the session is closed")
    var fields = py_fields(py_fields(session)["_block"])
    var slots = py_fields(py_fields(session)["_state"])
    var width = Int(py=fields["d_model"])
    var x = _neural_input(input, width, step, context)
    var shape = x.shape
    var batch = Int(py=shape[0])
    var length = Int(py=shape[1])
    if batch != Int(py=slots["batch_size"]):
        raise Error("TransformerDecodeSession: input rows differ from the state")
    var y = generation_array([batch, length, width])
    var keep = List[PythonObject]()
    var px = generation_address(x, keep)
    var py = generation_address(y, keep)
    var cached = Int(py=slots["cached_tokens"])
    if h[].device is not None:
        var p: List[Int] = [cached] if step else [length, cached]
        if h[].provider is not None:
            cached = transformer_resident_decode_call(h[].provider, h[].device, 20 if step else 21, [px, py], p)
        else:
            cached = Int(py=py_call(_native_function(h[].binding, "transformer_decode_session_step" if step else "transformer_decode_session_forward"),
                h[].device, py_ints([px, py]), py_ints(p)))
    else:
        var a: List[Int] = [px]
        for i in range(len(h[].weights)):  # driver-loop: native session-owned weight addresses
            a.append(0 if len(h[].weights[i]) == 0 else Int(h[].weights[i].unsafe_ptr()))
        a.append(Int(h[].k_cache.unsafe_ptr())); a.append(Int(h[].v_cache.unsafe_ptr())); a.append(py)
        var p = _transformer_decode_params(fields, slots, batch, length)
        for i in range(len(h[].optional)):  # driver-loop: native optional-weight address slots
            a.append(0 if len(h[].optional[i]) == 0 else Int(h[].optional[i].unsafe_ptr()))
        if step:
            var shortened = List[Int]()
            for i in range(len(p)):  # driver-loop: host decode shape words omit sequence length
                if i != 1:
                    shortened.append(p[i])
            p = shortened^
        cached = Int(py=py_call(_native_function(h[].binding, "transformer_decode_step" if step else "transformer_forward"), py_ints(a), py_ints(p)))
    py_setitem(slots, "cached_tokens", PythonObject(cached))
    return y


def transformer_decode_state(session: PythonObject, load: Bool) raises -> PythonObject:
    var values = py_fields(session)
    var hobj = values["_handle"]
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        raise Error("TransformerDecodeSession: the session is closed")
    var fields = py_fields(values["_block"])
    var state = values["_state"]
    var slots = py_fields(state)
    if load:
        _transformer_decode_profile(fields, slots)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _transformer_decode_pieces(fields, slots, keep, copies)
    var cached = Int(py=slots["cached_tokens"])
    if h[].device is not None:
        var a: List[Int] = [pieces[0], pieces[1]]
        var p: List[Int] = [cached] if load else []
        if h[].provider is not None:
            cached = transformer_resident_decode_call(h[].provider, h[].device, 23 if load else 22, a, p)
        elif load:
            cached = Int(py=py_call(_native_function(h[].binding, "transformer_decode_session_load_state"), h[].device, py_ints(a), py_ints(p)))
        else:
            cached = Int(py=py_call(_native_function(h[].binding, "transformer_decode_session_export_state"), h[].device, py_ints(a)))
    else:
        if pieces[2] != len(h[].k_cache) or pieces[2] != len(h[].v_cache):
            raise Error("TransformerDecodeSession: cache dimensions differ from the retained native session")
        if load:
            memcpy(dest=h[].k_cache.unsafe_ptr(), src=f32_ptr(pieces[0]), count=pieces[2])
            memcpy(dest=h[].v_cache.unsafe_ptr(), src=f32_ptr(pieces[1]), count=pieces[2])
        else:
            memcpy(dest=f32_ptr(pieces[0]), src=h[].k_cache.unsafe_ptr(), count=pieces[2])
            memcpy(dest=f32_ptr(pieces[1]), src=h[].v_cache.unsafe_ptr(), count=pieces[2])
    if not load:
        _writeback(copies)
    py_setitem(slots, "cached_tokens", PythonObject(cached))
    return state


def transformer_decode_release(session: PythonObject) raises:
    var values = py_fields(session)
    var hobj = py_get(values, "_handle")
    if hobj is None:
        return
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        return
    h[].open = False
    py_setitem(py_fields(values["_state"]), "_resident_session", PythonObject(None))
    if h[].device is not None:
        if h[].provider is not None:
            _ = transformer_resident_decode_call(h[].provider, h[].device, 24, [], [])
        else:
            _ = py_call(_native_function(h[].binding, "transformer_decode_session_close"), h[].device)
    h[].weights.clear(); h[].optional.clear(); h[].k_cache.clear(); h[].v_cache.clear()


def transformer_decode_native_call(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    if operation == "new":
        var session = py_call(Python.import_module("builtins").object.__new__, context["transformer_session_class"])
        _ = transformer_decode_init(session, values[0], values[1], context)
        return session
    var session = values[0]
    if py_type(session) is not context["transformer_session_class"]:
        raise Error("TypeError: TransformerDecodeSession requires its exact native public owner")
    if operation == "init":
        return transformer_decode_init(session, values[1], values[2], context)
    if operation == "step" or operation == "forward":
        return transformer_decode_run(session, values[1], operation == "step", context)
    if operation == "sync" or operation == "load":
        return transformer_decode_state(session, operation == "load")
    if operation == "state":
        return py_fields(session)["_state"]
    if operation == "enter":
        return session
    if operation == "close" or operation == "exit":
        var hobj = py_get(py_fields(session), "_handle")
        if hobj is None or not hobj.downcast_value_ptr[TransformerDecodeHandle]()[].open:
            return PythonObject(False) if operation == "exit" else PythonObject(None)
        try:
            _ = transformer_decode_state(session, False)
        finally:
            transformer_decode_release(session)
        return PythonObject(False) if operation == "exit" else PythonObject(None)
    if operation == "discard" or operation == "drop":
        transformer_decode_release(session)
        return PythonObject(None)
    var hobj = py_fields(session)["_handle"]
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if operation == "is_open":
        return PythonObject(h[].open)
    if operation == "native":
        return h[].device
    if operation == "repr":
        var cached = Int(py=py_fields(py_fields(session)["_state"])["cached_tokens"])
        return PythonObject("TransformerDecodeSession(open=" + String("True" if h[].open else "False") + ", cached_tokens=" + String(cached) + ")")
    raise Error("Transformer decode: unknown operation " + operation)


def transformer_decode_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = transformer_decode_native_call(String(args[0]), args[1], args[2])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
