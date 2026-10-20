# SPDX-License-Identifier: Apache-2.0
"""Whole-model runtime owned by Mojo, over existing compiled kernel entries.

The selected CPU/GPU extensions are bound once at model construction. This
module never invokes model, block, Array or backend Python methods. State,
allocation, layer traversal, token traversal and selection are native. Calling
a compiled sibling extension through CPython's native function ABI changes no
arithmetic and uses no interpreter callback; targets are checked at admission.
"""
from std.python import Python, PythonObject
from std.memory import memcpy
from bindings.generation_buffers import (
    GENERATION_BOUNDARY, generation_array, generation_address, generation_copy,
    generation_ids, py_ints,
)
from bindings.generation_policy import generation_pick
from bindings.hostptr import f32_ptr, i32_ptr
from std.math import isfinite


def _native_function(module: PythonObject, name: String) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var fn = g[].bi.getattr(module, name)
    if g[].bi.type(fn) is not g[].bi.type(g[].bi.len):
        raise Error("CausalLM: " + name + " must be a compiled native extension entry")
    return fn


def causal_lm_admit_binding(
    model: PythonObject, block_binding: PythonObject, primitives: PythonObject,
    int15_binding: PythonObject,
) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var entries = g[].bi.dict()
    for name in [String("embedding_forward"), String("rms_norm_forward"), String("linear_forward")]:
        entries[name] = _native_function(primitives, name)
    var kind = String(model.kind)
    if kind == "transformer":
        entries["forward"] = _native_function(block_binding, "transformer_forward")
        if model._head_int15 is not None:
            if Bool(py=g[].bi.hasattr(block_binding, "transformer_session_forward_int15")):
                entries["create"] = _native_function(block_binding, "transformer_session_create")
                entries["forward15"] = _native_function(block_binding, "transformer_session_forward_int15")
            else:
                entries["forward15_host"] = _native_function(block_binding, "transformer_forward_int15")
            entries["quantize15"] = _native_function(int15_binding, "quantize_int15")
            entries["gemm15"] = _native_function(int15_binding, "gemm_int15")
        # Preserve the existing resident GPU route for fixed-length greedy
        # generation. No performance default changes with this migration.
        if Bool(py=g[].bi.hasattr(block_binding, "causal_lm_session_run")):
            for name in [String("transformer_decode_session_create"), String("transformer_decode_session_open"),
                         String("transformer_decode_session_close"), String("causal_lm_session_create"),
                         String("causal_lm_session_open"), String("causal_lm_session_run"), String("causal_lm_session_close")]:
                entries[name] = _native_function(block_binding, name)
            if model._head_int15 is not None:
                entries["transformer_decode_session_open_int15"] = _native_function(block_binding, "transformer_decode_session_open_int15")
                entries["causal_lm_session_open_int15"] = _native_function(block_binding, "causal_lm_session_open_int15")
    elif kind == "mamba1":
        entries["forward"] = _native_function(block_binding, "mamba1_forward")
        var profile = 1
        if Bool(py=g[].bi.hasattr(block_binding, "mamba1_profile")):
            profile = Int(py=_native_function(block_binding, "mamba1_profile")())
        entries["mamba1_profile"] = PythonObject(profile)
    elif kind == "mamba2":
        entries["forward"] = _native_function(block_binding, "mamba2_forward")
    else:
        raise Error("CausalLM: unsupported model kind " + kind)
    model._native_entries = entries
    return PythonObject(None)


def _positions(model: PythonObject, state: PythonObject, length: Int) raises:
    var have = 0 if state is None else Int(py=state.positions)
    if length < 0 or have < 0:
        raise Error("CausalLM: negative sequence position")
    if model.max_positions is not None and have + length > Int(py=model.max_positions):
        raise Error("CausalLM: sequence exceeds max_positions")
    if state is not None and have + length > Int(py=state.max_tokens):
        raise Error("CausalLM: sequence exceeds allocated state capacity")


def causal_lm_allocate_binding(model: PythonObject, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    var b = Int(py=batch)
    var cap = Int(py=capacity)
    if b < 1 or cap < 1:
        raise Error("CausalLM: batch_size and max_tokens must be positive")
    _positions(model, PythonObject(None), cap)
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var state = g[].bi.object.__new__(g[].state_cls)
    state.batch_size = PythonObject(b)
    state.max_tokens = PythonObject(cap)
    state.positions = PythonObject(0)
    state._owner = model
    state._usable = PythonObject(True)
    state.layers = g[].bi.list()
    var kind = String(model.kind)
    for i in range(len(model._blocks)):
        var blk = model._blocks[i]
        var st = g[].bi.object.__new__(g[].layer_classes[kind])
        if kind == "transformer":
            var window = Int(py=blk.window)
            var nkv = Int(py=blk.n_kv_heads)
            var hd = Int(py=blk.head_dim)
            var n = b * nkv * (cap if window == 0 else window) * hd
            st.batch_size = PythonObject(b)
            st.n_kv_heads = PythonObject(nkv)
            st.head_dim = PythonObject(hd)
            st.max_tokens = PythonObject(cap)
            st.window = PythonObject(window)
            st.k_cache = generation_array([n])
            st.v_cache = generation_array([n])
            st.cached_tokens = PythonObject(0)
            st._resident_session = PythonObject(None)
            st._numeric_profile = model.numeric_profile
        elif kind == "mamba1":
            var di = Int(py=blk.d_inner)
            var profile = Int(py=model._native_entries["mamba1_profile"])
            st.conv_window = generation_array([b, di, 4])
            st.h = generation_array([b, di, 16])
            st.profile_id = PythonObject(profile)
            st.absolute_position = PythonObject(0)
            st._nn34_usable = PythonObject(True)
            st.affine_boundary = PythonObject(None)
            st.affine_a = PythonObject(None)
            st.affine_b = PythonObject(None)
            if profile == 2:
                st.affine_boundary = generation_array([b, di, 16])
                st.affine_a = generation_array([b, di, 16, 6])
                st.affine_b = generation_array([b, di, 16, 6])
        else:
            var cd = Int(py=blk.conv_dim)
            var nh = Int(py=blk.nheads)
            st.conv_window = generation_array([b, cd, 4])
            st.h = generation_array([b, nh, 64, 128])
            st.buffer_xbc = generation_array([b, 256, cd])
            st.buffer_dtraw = generation_array([b, 256, nh])
            st.buffered_tokens = PythonObject(0)
        state.layers.append(st)
    return state


def _state(model: PythonObject, state: PythonObject, b: Int, l: Int) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].bi.type(state) is not g[].state_cls or state._owner is not model:
        raise Error("CausalLM: state belongs to another model")
    if not Bool(py=g[].bi.getattr(state, "_usable", True)):
        raise Error("CausalLM: state lost after a failed call; reset or restore it")
    if Int(py=state.batch_size) != b or len(state.layers) != len(model._blocks):
        raise Error("CausalLM: state batch/layer counts disagree with model and input")
    _positions(model, state, l)


def causal_lm_reset_binding(model: PythonObject, state: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].bi.type(state) is not g[].state_cls or state._owner is not model:
        raise Error("CausalLM: state belongs to another model")
    var fresh = causal_lm_allocate_binding(model, state.batch_size, state.max_tokens)
    state.layers = fresh.layers
    state.positions = PythonObject(0)
    state._usable = PythonObject(True)
    return state


def _addr(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    return generation_address(value, keep)


def _transformer_args(
    model: PythonObject, blk: PythonObject, st: PythonObject, x: PythonObject, y: PythonObject,
    b: Int, l: Int, planes: Bool, mut keep: List[PythonObject],
) raises -> PythonObject:
    var addresses = py_ints([_addr(x, keep)])
    for j in range(len(blk._w)):
        addresses.append(0 if planes and j >= 2 else _addr(blk._w[j], keep))
    addresses.append(_addr(st.k_cache, keep))
    addresses.append(_addr(st.v_cache, keep))
    addresses.append(_addr(y, keep))
    for j in range(len(blk._wopt)):
        addresses.append(_addr(blk._wopt[j], keep))
    if planes:
        for j in range(len(blk._int15)):
            var trip = blk._int15[j]
            for k in range(3):
                addresses.append(0 if trip is None else _addr(trip[k], keep))
    return addresses


def _transformer_params(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> PythonObject:
    var p = py_ints([b, l, Int(py=blk.d_model), Int(py=blk.n_heads), Int(py=blk.n_kv_heads),
                    Int(py=blk.head_dim), Int(py=blk.intermediate), Int(py=st.max_tokens),
                    Int(py=st.cached_tokens), Int(py=blk.window)])
    p.extend(blk._opts_tail)
    return p


def _run(model: PythonObject, ids: PythonObject, state: PythonObject) raises -> PythonObject:
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var d = Int(py=model.d_model)
    var v = Int(py=model.vocab_size)
    _state(model, state, b, l)
    var e = model._native_entries
    var keep = List[PythonObject]()
    var x = generation_array([b, l, d])
    e["embedding_forward"](py_ints([_addr(x, keep), _addr(model._embed, keep), _addr(ids, keep)]), py_ints([b * l, v, d]))
    var kind = String(model.kind)
    try:
        for i in range(len(model._blocks)):
            var blk = model._blocks[i]
            var st = state.layers[i]
            var y = generation_array([b, l, d])
            var addresses = PythonObject(None)
            var params = PythonObject(None)
            if kind == "transformer":
                if st._resident_session is not None:
                    raise Error("CausalLM: layer state is owned by a decode session")
                params = _transformer_params(blk, st, b, l)
                var profile = model._head_int15 is not None
                var device15 = profile and Bool(py=e.__contains__("forward15"))
                addresses = _transformer_args(model, blk, st, x, y, b, l, device15, keep)
                if device15:
                    if blk._native_session is None:
                        blk._native_session = e["create"]()
                    st.cached_tokens = e["forward15"](blk._native_session, addresses, params)
                elif profile:
                    st.cached_tokens = e["forward15_host"](addresses, params)
                else:
                    st.cached_tokens = e["forward"](addresses, params)
            else:
                addresses = py_ints([_addr(x, keep)])
                for j in range(len(blk._w)):
                    addresses.append(_addr(blk._w[j], keep))
                addresses.append(_addr(st.conv_window, keep))
                addresses.append(_addr(st.h, keep))
                if kind == "mamba1":
                    addresses.append(_addr(y, keep))
                    params = py_ints([b, l, d])
                    var profile = Int(py=e["mamba1_profile"])
                    if Int(py=st.profile_id) != profile or not Bool(py=st._nn34_usable):
                        raise Error("CausalLM: incompatible or failed Mamba1 state")
                    if profile == 2:
                        addresses.append(_addr(st.affine_boundary, keep))
                        addresses.append(_addr(st.affine_a, keep))
                        addresses.append(_addr(st.affine_b, keep))
                        params.append(st.absolute_position)
                    e["forward"](addresses, params)
                    st.absolute_position = PythonObject(Int(py=st.absolute_position) + l)
                else:
                    addresses.append(_addr(st.buffer_xbc, keep))
                    addresses.append(_addr(st.buffer_dtraw, keep))
                    addresses.append(_addr(y, keep))
                    var last = generation_array([b, Int(py=blk.nheads), 64, 128])
                    addresses.append(_addr(last, keep))
                    params = py_ints([b, l, d, Int(py=st.buffered_tokens)])
                    params.extend(blk.dt_limit)
                    st.buffered_tokens = e["forward"](addresses, params)
                    blk.h_last_ = last
            x = y
        var hn = x
        if model._norm is not None:
            hn = generation_array([b, l, d])
            var np = py_ints([b * l, d])
            np.append(model.norm_eps)
            e["rms_norm_forward"](py_ints([_addr(hn, keep), _addr(x, keep), _addr(model._norm, keep)]), np)
        var out = generation_array([b, l, v])
        if model._head_int15 is None:
            e["linear_forward"](py_ints([_addr(out, keep), _addr(hn, keep), _addr(model._head, keep)]), py_ints([b * l, v, d]))
        else:
            var hi = generation_array([b * l, d], "b")
            var lo = generation_array([b * l, d], "b")
            var ex = generation_array([b * l], "i")
            e["quantize15"](py_ints([_addr(hi, keep), _addr(lo, keep), _addr(ex, keep), _addr(hn, keep)]), py_ints([b * l, d]))
            var hp = model._head_int15
            e["gemm15"](py_ints([_addr(out, keep), _addr(hi, keep), _addr(lo, keep), _addr(ex, keep),
                _addr(hp[0], keep), _addr(hp[1], keep), _addr(hp[2], keep)]), py_ints([b * l, v, d]))
        state.positions = PythonObject(Int(py=state.positions) + l)
        _ = keep^
        return out
    except error:
        state._usable = PythonObject(False)
        raise error


def causal_lm_forward_binding(model: PythonObject, value: PythonObject, var state: PythonObject, step: PythonObject) raises -> PythonObject:
    var one = Bool(py=step)
    var ids = generation_ids(value, Int(py=model.vocab_size), one)
    if state is None:
        if one:
            raise Error("CausalLM.step: state is required")
        state = causal_lm_allocate_binding(model, ids.shape[0], ids.shape[1])
    var out = _run(model, ids, state)
    if one:
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        out.shape = g[].bi.tuple(py_ints([Int(py=ids.shape[0]), Int(py=model.vocab_size)]))
        out.strides = g[].bi.tuple(py_ints([Int(py=model.vocab_size) * 4, 4]))
        out.ndim = PythonObject(2)
    return out


struct GenerationHandle(Movable, Writable):
    var model: PythonObject
    var state: PythonObject
    var prompt: PythonObject
    var current: PythonObject
    var tokenizer: PythonObject
    var pending_bytes: PythonObject
    var max_new: Int
    var produced: Int
    var greedy: Bool
    var temperature: Float32
    var top_k: Int
    var top_p: Float64
    var seed: UInt64
    var eos: List[Int]
    var pad: Int
    var lengths: List[Int]
    var finished: List[Bool]
    var busy: Bool
    var usable: Bool

    def __init__(out self, model: PythonObject, prompt: PythonObject, maximum: Int) raises:
        self.model = model
        self.prompt = prompt
        self.current = prompt
        self.max_new = maximum
        self.produced = 0
        self.state = causal_lm_allocate_binding(model, prompt.shape[0], PythonObject(Int(py=prompt.shape[1]) + maximum))
        self.greedy = True
        self.temperature = Float32(1)
        self.top_k = 0
        self.top_p = Float64(1)
        self.seed = UInt64(0)
        self.eos = List[Int]()
        self.pad = 0
        self.lengths = List[Int](length=Int(py=prompt.shape[0]), fill=0)
        self.finished = List[Bool](length=Int(py=prompt.shape[0]), fill=False)
        self.busy = False
        self.usable = True
        self.tokenizer = PythonObject(None)
        self.pending_bytes = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.bytearray()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GenerationHandle(produced=", self.produced, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def done(self) -> Bool:
        if self.produced >= self.max_new:
            return True
        for i in range(len(self.finished)):
            if not self.finished[i]:
                return False
        return True


def _options(mut s: GenerationHandle, options: PythonObject) raises:
    # The Python veneer forwards these scalar objects unchanged. Explicit
    # sampling requires an explicit seed; greedy consumes no RNG state.
    if len(options) != 7:
        raise Error("generation: expected greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].bi.type(options[0]) is not g[].bi.bool:
        raise Error("generation: greedy must be bool")
    s.greedy = Bool(py=options[0])
    if not s.greedy and options[1] is None:
        raise Error("generation: sampling requires an explicit seed")
    if options[1] is not None:
        if g[].bi.type(options[1]) is not g[].bi.int or Bool(py=options[1] < 0) or Bool(py=options[1] >= g[].bi.int("18446744073709551616")):
            raise Error("generation: seed must be an integer in [0, 2**64)")
        var halves = g[].bi.divmod(options[1], g[].bi.int("4294967296"))
        s.seed = (UInt64(Int(py=halves[0])) << 32) | UInt64(Int(py=halves[1]))
    s.temperature = Float32(Float64(py=options[2]))
    s.top_k = Int(py=options[3])
    s.top_p = Float64(py=options[4])
    var v = Int(py=s.model.vocab_size)
    if s.temperature <= 0 or not isfinite(s.temperature) or s.top_k < 0 or s.top_k > v or s.top_p <= 0 or s.top_p > 1 or not isfinite(s.top_p):
        raise Error("generation: invalid temperature, top_k or top_p")
    if options[5] is not None:
        var values = options[5]
        if g[].bi.type(values) is g[].bi.int:
            values = py_ints([Int(py=values)])
        for i in range(len(values)):
            if g[].bi.type(values[i]) is not g[].bi.int:
                raise Error("generation: eos_token_id must be integer or sequence of integers")
            var id = Int(py=values[i])
            if id < 0 or id >= v:
                raise Error("generation: EOS id is outside vocabulary")
            s.eos.append(id)
    s.pad = (s.eos[0] if len(s.eos) else 0) if options[6] is None else Int(py=options[6])
    if s.pad < 0 or s.pad >= v:
        raise Error("generation: pad id is outside vocabulary")


def causal_lm_stream_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var n = Int(py=maximum)
    if n < 0:
        raise Error("generation: max_new_tokens must be nonnegative")
    var prompt = generation_ids(value, Int(py=model.vocab_size))
    var handle = GenerationHandle(model, prompt, n)
    _options(handle, options)
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var stream = g[].bi.object.__new__(g[].stream_cls)
    stream._native = PythonObject(alloc=handle^)
    stream._runtime = model._runtime
    return stream


def _read(mut s: GenerationHandle, count: Int) raises -> PythonObject:
    if count < 1:
        raise Error("generation: read count must be positive")
    if not s.usable:
        raise Error("generation: stream lost after a failed call; restore a checkpoint")
    if s.busy:
        raise Error("generation: stream is busy")
    var b = len(s.finished)
    var v = Int(py=s.model.vocab_size)
    var capacity = min(count, s.max_new - s.produced)
    var tokens = List[Int32](capacity=b * capacity)
    var wrote = 0
    s.busy = True
    try:
        while wrote < capacity and not s.done():
            var l = Int(py=s.current.shape[1])
            var logits = _run(s.model, s.current, s.state)
            var raw = f32_ptr(Int(py=logits._addr))
            var next = generation_array([b, 1], "i")
            var dst = i32_ptr(Int(py=next._addr))
            for r in range(b):
                var pick = s.pad
                if not s.finished[r]:
                    var row = List[Float32](capacity=v)
                    for j in range(v):
                        row.append(raw[(r * l + l - 1) * v + j])
                    pick = generation_pick(row, s.greedy, s.temperature, s.top_k, s.top_p, s.seed, r, s.produced)
                    s.lengths[r] += 1
                    for i in range(len(s.eos)):
                        if pick == s.eos[i]:
                            s.finished[r] = True
                dst[r] = Int32(pick)
                tokens.append(Int32(pick))
            s.current = next
            s.produced += 1
            wrote += 1
        var out = generation_array([b, wrote], "i")
        var dst = i32_ptr(Int(py=out._addr))
        for r in range(b):
            for t in range(wrote):
                dst[r * wrote + t] = tokens[t * b + r]
        s.busy = False
        return out
    except error:
        s.busy = False
        s.usable = False
        raise error


def _decode_chunk(mut s: GenerationHandle, ids: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var n = Int(py=ids.shape[1])
    var values = g[].bi.list()
    var raw = i32_ptr(Int(py=ids._addr))
    for i in range(n):
        var special = False
        for j in range(len(s.eos)):
            special = special or Int(raw[i]) == s.eos[j]
        if not special:
            values.append(Int(raw[i]))
    s.pending_bytes.extend(_native_function(s.tokenizer._m, "vocab_decode_public")(s.tokenizer, values, PythonObject(True), PythonObject("replace")))
    var size = len(s.pending_bytes)
    var cut = size
    # Preserve an incomplete trailing UTF-8 sequence across tokens. Invalid
    # complete sequences are handled by the normal CPython UTF-8 decoder.
    if not s.done() and size > 0:
        var lead = size - 1
        while lead > max(-1, size - 4) and (Int(py=s.pending_bytes[lead]) & 0xC0) == 0x80:
            lead -= 1
        if lead >= 0:
            var first = Int(py=s.pending_bytes[lead])
            var need = 1
            if first >= 0xC2 and first <= 0xDF:
                need = 2
            elif first >= 0xE0 and first <= 0xEF:
                need = 3
            elif first >= 0xF0 and first <= 0xF4:
                need = 4
            if size - lead < need:
                cut = lead
    var prefix = g[].bi.bytearray()
    var remaining = g[].bi.bytearray()
    for i in range(size):
        if i < cut:
            prefix.append(s.pending_bytes[i])
        else:
            remaining.append(s.pending_bytes[i])
    s.pending_bytes = remaining
    return prefix.decode("utf-8", "replace")


def causal_lm_read_binding(stream: PythonObject, count: PythonObject) raises -> PythonObject:
    var owner = stream._native.downcast_value_ptr[GenerationHandle]()
    var ids = _read(owner[], Int(py=count))
    if owner[].tokenizer is not None:
        return _decode_chunk(owner[], ids)
    return ids


def causal_lm_info_binding(stream: PythonObject) raises -> PythonObject:
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    var result = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.dict()
    result["done"] = PythonObject(s[].done())
    result["generated_tokens"] = PythonObject(s[].produced)
    result["lengths"] = py_ints(s[].lengths)
    result["usable"] = PythonObject(s[].usable)
    return result


def causal_lm_generate_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_stream_binding(model, value, maximum, options)
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    var continuation = _read(s[], max(1, s[].max_new))
    var b = len(s[].lengths)
    var l = Int(py=s[].prompt.shape[1])
    var n = Int(py=continuation.shape[1])
    var result = generation_array([b, l + n], "i")
    var dest = i32_ptr(Int(py=result._addr))
    var prompt = i32_ptr(Int(py=s[].prompt._addr))
    var tail = i32_ptr(Int(py=continuation._addr))
    for r in range(b):
        memcpy(dest=dest + r * (l + n), src=prompt + r * l, count=l)
        memcpy(dest=dest + r * (l + n) + l, src=tail + r * n, count=n)
    return result


def causal_lm_text_stream_binding(model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var encode = _native_function(tokenizer._m, "vocab_encode_public")
    var ids = encode(tokenizer, text, PythonObject(True), PythonObject(False), PythonObject(False))
    var prompt = generation_array([1, len(ids)], "i")
    var ptr = i32_ptr(Int(py=prompt._addr))
    for i in range(len(ids)):
        ptr[i] = Int32(Int(py=ids[i]))
    var stream = causal_lm_stream_binding(model, prompt, maximum, options)
    var owner = stream._native.downcast_value_ptr[GenerationHandle]()
    owner[].tokenizer = tokenizer
    return stream


def causal_lm_generate_text_binding(model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_text_stream_binding(model, tokenizer, text, maximum, options)
    var owner = stream._native.downcast_value_ptr[GenerationHandle]()
    var ids = _read(owner[], max(1, owner[].max_new))
    return _decode_chunk(owner[], ids)
