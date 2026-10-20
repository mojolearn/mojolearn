# SPDX-License-Identifier: Apache-2.0
"""Whole-model runtime owned by Mojo, over existing compiled kernel entries.

The selected CPU/GPU extensions are bound once at model construction. This
module never invokes model, block, Array or backend Python methods. State,
allocation, layer traversal, token traversal and selection are native. Calling
a compiled sibling extension through CPython's native function ABI changes no
arithmetic and uses no interpreter callback; targets are checked at admission.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from bindings.generation_buffers import (
    GENERATION_BOUNDARY, generation_array, generation_address, generation_copy,
    generation_ids, py_ints,
)
from bindings.generation_policy import generation_pick
from bindings.hostptr import f32_ptr, i32_ptr
from std.math import isfinite
from std.os import getenv


def _native_function(module: PythonObject, name: String) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var entry = g[].bi.getattr(module, name)
    if g[].bi.type(entry) is not g[].bi.type(g[].bi.len):
        raise Error("CausalLM: " + name + " must be a compiled native extension entry")
    return entry


def causal_lm_admit_binding(
    model: PythonObject, block_binding: PythonObject, primitives: PythonObject,
    int15_binding: PythonObject,
) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var entries = g[].bi.dict()
    var requested_gpu = String(model.device) == "gpu"
    var kind = String(model.kind)
    var block_prefix = ("transformer" if kind == "transformer" else "mamba") if requested_gpu else "neural_host"
    var primitive_prefix = "training" if requested_gpu else "neural_host"
    var vendor = String(_native_function(block_binding, block_prefix + "_vendor")())
    var primitive_vendor = String(_native_function(primitives, primitive_prefix + "_vendor")())
    var mode = Int(py=_native_function(block_binding, block_prefix + "_numeric_mode")())
    var primitive_mode = Int(py=_native_function(primitives, primitive_prefix + "_numeric_mode")())
    if requested_gpu:
        if vendor != "metal" and vendor != "cuda" and vendor != "hip":
            raise Error("CausalLM: requested GPU execution but native block reports " + vendor)
    elif vendor != "cpu":
        raise Error("CausalLM: requested CPU execution but native block reports " + vendor)
    if primitive_vendor != vendor or primitive_mode != mode:
        raise Error("CausalLM: native block and model primitives differ in vendor or numeric mode")
    if int15_binding is not None:
        var packed_vendor = String(_native_function(int15_binding, "linalg_vendor")())
        var packed_mode = Int(py=_native_function(int15_binding, "linalg_numeric_mode")())
        if packed_vendor != vendor or packed_mode != mode:
            raise Error("CausalLM: native packed arithmetic differs in vendor or numeric mode")
    model._native_vendor = PythonObject(vendor)
    model._native_numeric_mode = PythonObject(mode)
    entries["weight_names"] = g[].bi.object.__getattribute__(model._blocks[0], "_W_NAMES")
    for name in [String("embedding_forward"), String("rms_norm_forward"), String("linear_forward")]:
        entries[name] = _native_function(primitives, name)
    if kind == "transformer":
        entries["forward"] = _native_function(block_binding, "transformer_forward")
        entries["decode"] = _native_function(block_binding, "transformer_decode_step")
        entries["block_binding"] = block_binding
        for name in [String("transformer_forward_fresh"), String("transformer_session_forward_fresh")]:
            if Bool(py=g[].bi.hasattr(block_binding, name)):
                entries[name] = _native_function(block_binding, name)
        if Bool(py=g[].bi.hasattr(block_binding, "transformer_session_forward")):
            entries["create"] = _native_function(block_binding, "transformer_session_create")
            entries["forward_session"] = _native_function(block_binding, "transformer_session_forward")
            entries["block_binding"] = block_binding
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
            entries["resident_vendor"] = _native_function(block_binding, "transformer_vendor")()
            for name in [String("transformer_decode_session_create"), String("transformer_decode_session_open"),
                         String("transformer_decode_session_close"), String("causal_lm_session_create"),
                         String("causal_lm_session_open"), String("causal_lm_session_run"), String("causal_lm_session_close")]:
                entries[name] = _native_function(block_binding, name)
            if model._head_int15 is not None:
                entries["transformer_decode_session_open_int15"] = _native_function(block_binding, "transformer_decode_session_open_int15")
                entries["causal_lm_session_open_int15"] = _native_function(block_binding, "causal_lm_session_open_int15")
    elif kind == "mamba1":
        entries["forward"] = _native_function(block_binding, "mamba1_forward")
        entries["decode"] = _native_function(block_binding, "mamba1_decode_step")
        var profile = 1
        if Bool(py=g[].bi.hasattr(block_binding, "mamba1_profile")):
            profile = Int(py=_native_function(block_binding, "mamba1_profile")())
        entries["mamba1_profile"] = PythonObject(profile)
    elif kind == "mamba2":
        entries["forward"] = _native_function(block_binding, "mamba2_forward")
        entries["decode"] = _native_function(block_binding, "mamba2_decode_step")
    else:
        raise Error("CausalLM: unsupported model kind " + kind)
    model._native_entries = entries
    return PythonObject(None)


def _block(model: PythonObject, index: Int) raises -> PythonObject:
    # PublicOutputsMixin.__getattribute__ is Python even for private metadata.
    # Access the object's plain dictionary with CPython's base implementation,
    # bypassing that overridden getter; every field below is stored metadata.
    var parallel = model._native_entries.get("parallel_call")
    if parallel is not None:
        return parallel("block_fields", Python.list(model, index))
    return GENERATION_BOUNDARY.get_or_create_ptr()[].bi.object.__getattribute__(model._blocks[index], "__dict__")


def _positions(model: PythonObject, state: PythonObject, length: Int) raises:
    var have = 0 if state is None else Int(py=state.positions)
    if length < 0 or have < 0:
        raise Error("CausalLM: negative sequence position")
    if model.max_positions is not None and have + length > Int(py=model.max_positions):
        raise Error("CausalLM: sequence exceeds max_positions")
    if state is not None and have + length > Int(py=state.max_tokens):
        raise Error("CausalLM: sequence exceeds allocated state capacity")


def causal_lm_parameters_binding(model: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    var p = model.plan
    result[p.embed_name] = model._embed
    if p.norm_name is not None:
        result[p.norm_name] = model._norm
    if p.head_name is not None:
        result[p.head_name] = model._head
    var weight_names = model._native_entries["weight_names"]
    # Same canonical optional slot order as the transformer native ABI.
    var optional_names = [String("q_proj.bias"), String("k_proj.bias"), String("v_proj.bias"),
                          String("o_proj.bias"), String("input_layernorm.bias"),
                          String("post_attention_layernorm.bias"), String("up_proj.bias"),
                          String("down_proj.bias"), String("gate_proj.bias"),
                          String("q_norm.weight"), String("k_norm.weight")]
    var keep = List[PythonObject]()
    for i in range(len(model._blocks)):  # driver-loop: over model layer metadata
        var names = g[].bi.dict()
        var slices = g[].bi.dict()
        for j in range(len(p._layer_map)):  # driver-loop: over checkpoint tensor mappings
            var entry = p._layer_map[j]
            names[entry[0]] = PythonObject(String(entry[1]).replace("{i}", String(i)))
            slices[entry[0]] = entry[2]
        var blk = _block(model, i)
        var values = g[].bi.dict()
        for j in range(min(len(weight_names), len(blk["_w"]))):  # driver-loop: over block weight slots
            if blk["_w"][j] is not None:
                values[weight_names[j]] = blk["_w"][j]
        if String(model.kind) == "transformer":
            for j in range(len(optional_names)):  # driver-loop: over optional block weight slots
                if blk["_wopt"][j] is not None:
                    values[optional_names[j]] = blk["_wopt"][j]
        var keys = g[].bi.list(values)
        var fused = g[].bi.dict()
        for j in range(len(keys)):  # driver-loop: over checkpoint tensor names
            var key = keys[j]
            var name = names.get(key, PythonObject("layers." + String(i) + "." + String(key)))
            var rows = slices.get(key)
            if rows is None:
                result[name] = values[key]
            else:
                # A fused checkpoint tensor (Phi-3 QKV / gate-up) is split
                # for execution. Return every row under its checkpoint name.
                var piece = g[].bi.list(rows)
                piece.append(values[key])
                fused.setdefault(name, g[].bi.list()).append(piece)
        var fused_names = g[].bi.list(fused)
        for j in range(len(fused_names)):  # driver-loop: over fused checkpoint tensor names
            var name = fused_names[j]
            var pieces = fused[name]
            var rows = 0
            var cols = Int(py=pieces[0][2].shape[1])
            for k in range(len(pieces)):  # driver-loop: over projection slice metadata
                rows = max(rows, Int(py=pieces[k][1]))
            var joined = generation_array([rows, cols])
            for k in range(len(pieces)):  # driver-loop: over contiguous projection buffers
                var begin = Int(py=pieces[k][0])
                var end = Int(py=pieces[k][1])
                if end > begin:
                    memcpy(dest=f32_ptr(Int(py=joined._addr)) + begin * cols, src=f32_ptr(_addr(pieces[k][2], keep)), count=(end - begin) * cols)
            result[name] = joined
    _ = keep^
    return result


def causal_lm_blocks_binding(model: PythonObject) raises -> PythonObject:
    return GENERATION_BOUNDARY.get_or_create_ptr()[].bi.list(model._blocks)


def causal_layer_allocate(model: PythonObject, i: Int, b: Int, cap: Int) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var kind = String(model.kind)
    var blk = _block(model, i)
    var st = g[].bi.object.__new__(g[].layer_classes[kind])
    if kind == "transformer":
        var window = Int(py=blk["window"])
        var nkv = Int(py=blk["n_kv_heads"])
        var hd = Int(py=blk["head_dim"])
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
        var di = Int(py=blk["d_inner"])
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
        var cd = Int(py=blk["conv_dim"])
        var nh = Int(py=blk["nheads"])
        st.conv_window = generation_array([b, cd, 4])
        st.h = generation_array([b, nh, 64, 128])
        st.buffer_xbc = generation_array([b, 256, cd])
        st.buffer_dtraw = generation_array([b, 256, nh])
        st.buffered_tokens = PythonObject(0)
    return st


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
    var parallel = model._native_entries.get("parallel_call")
    try:
        for i in range(len(model._blocks)):  # driver-loop: over model layer state owners
            state.layers.append(causal_layer_allocate(model, i, b, cap) if parallel is None else
                                parallel("allocate_layer", Python.list(model, i, b, cap)))
    except error:
        if parallel is not None:
            try:
                _ = parallel("release_layers", Python.list(model, state))
            except cleanup_error:
                raise Error("CausalLM: state allocation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
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
    var parallel = model._native_entries.get("parallel_call")
    if parallel is not None:
        try:
            _ = parallel("release_layers", Python.list(model, state))
        except error:
            state._usable = PythonObject(False)
            try:
                _ = parallel("release_layers", Python.list(model, fresh))
            except cleanup_error:
                raise Error("CausalLM: state reset failed: " + String(error) + "; fresh-state cleanup also failed: " + String(cleanup_error))
            raise error
    state.layers = fresh.layers
    state.positions = PythonObject(0)
    state._usable = PythonObject(True)
    return state


def _addr(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    return generation_address(value, keep)


def _mutable(
    value: PythonObject, expected: List[Int], mut keep: List[PythonObject], mut copies: List[PythonObject],
) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var view = PythonObject(None)
    var shape = PythonObject(None)
    if g[].bi.type(value) is g[].array_cls:
        if String(value.dtype) != "<f4" or String(value.order) != "C" or Bool(py=value._readonly):
            raise Error("CausalLM: state must be writable C-contiguous float32")
        view = value._mv
        shape = value.shape
    else:
        view = g[].bi.memoryview(value)
        shape = view.shape
        if Bool(py=view.readonly) or not Bool(py=view.c_contiguous) or String(view.format) != "f":
            raise Error("CausalLM: state must be writable C-contiguous float32")
    if len(shape) != len(expected):
        raise Error("CausalLM: state buffer rank differs")
    var size = 1
    for i in range(len(expected)):
        if Int(py=shape[i]) != expected[i]:
            raise Error("CausalLM: state buffer shape differs")
        size *= expected[i]
    if Int(py=view.nbytes) != size * 4:
        raise Error("CausalLM: state buffer span differs")
    var address = _addr(value, keep)
    if g[].bi.type(value) is not g[].array_cls:
        var pair = g[].bi.list()
        pair.append(view)
        pair.append(keep[len(keep) - 1])
        copies.append(pair)
    return address


def _writeback(mut copies: List[PythonObject]) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for i in range(len(copies)):
        var view = copies[i][0].cast("B")
        var source = g[].bi.memoryview(copies[i][1]).cast("B")
        view[g[].bi.slice(PythonObject(None))] = source
    copies.clear()


def _transformer_args(
    model: PythonObject, blk: PythonObject, st: PythonObject, x: PythonObject, y: PythonObject,
    b: Int, l: Int, planes: Bool, mut keep: List[PythonObject],
) raises -> PythonObject:
    var addresses = py_ints([_addr(x, keep)])
    for j in range(len(blk["_w"])):
        addresses.append(0 if planes and j >= 2 else _addr(blk["_w"][j], keep))
    addresses.append(_addr(st.k_cache, keep))
    addresses.append(_addr(st.v_cache, keep))
    addresses.append(_addr(y, keep))
    for j in range(len(blk["_wopt"])):
        addresses.append(_addr(blk["_wopt"][j], keep))
    if planes:
        for j in range(len(blk["_int15"])):
            var trip = blk["_int15"][j]
            for k in range(3):
                addresses.append(0 if trip is None else _addr(trip[k], keep))
    return addresses


def _transformer_params(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> PythonObject:
    var p = py_ints([b, l, Int(py=blk["d_model"]), Int(py=blk["n_heads"]), Int(py=blk["n_kv_heads"]),
                    Int(py=blk["head_dim"]), Int(py=blk["intermediate"]), Int(py=st.max_tokens),
                    Int(py=st.cached_tokens), Int(py=blk["window"])])
    p.extend(blk["_opts_tail"])
    return p


def causal_layer_run(model: PythonObject, i: Int, x: PythonObject, supplied_state: PythonObject, step: Bool = False) raises -> PythonObject:
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=model.d_model)
    var kind = String(model.kind)
    var e = model._native_entries
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var blk = _block(model, i)
    var st = supplied_state
    var y = generation_array([b, l, d])
    var addresses = PythonObject(None)
    var params = PythonObject(None)
    if kind == "transformer":
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        if g[].bi.getattr(st, "_resident_session", PythonObject(None)) is not None:
            raise Error("CausalLM: layer state is owned by a decode session")
        var old_profile = g[].bi.getattr(st, "arithmetic_profile", PythonObject(None))
        var profile_id = blk.get("_arithmetic_profile", PythonObject(None))
        if old_profile is None and Int(py=st.cached_tokens) != 0 and Bool(py=blk.get("_arithmetic_profile_changed", PythonObject(False))):
            raise Error("CausalLM: cached state requires arithmetic profile metadata")
        if old_profile is not None and profile_id is not None and not Bool(py=old_profile == profile_id):
            raise Error("CausalLM: state arithmetic profile differs")
        if profile_id is not None:
            st.arithmetic_profile = profile_id
        var cap = Int(py=st.max_tokens) if Int(py=blk["window"]) == 0 else Int(py=blk["window"])
        var cache_n = b * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
        params = _transformer_params(blk, st, b, l)
        var profile = model._head_int15 is not None
        var device15 = profile and Bool(py=e.__contains__("forward15"))
        addresses = _transformer_args(model, blk, st, x, y, b, l, device15, keep)
        addresses[10] = PythonObject(_mutable(st.k_cache, [cache_n], keep, copies))
        addresses[11] = PythonObject(_mutable(st.v_cache, [cache_n], keep, copies))
        if device15:
            if blk["_native_session"] is None:
                blk["_native_session"] = e["create"]()
                blk["_session_binding"] = e["block_binding"]
            st.cached_tokens = e["forward15"](blk["_native_session"], addresses, params)
        elif profile:
            st.cached_tokens = e["forward15_host"](addresses, params)
        elif Bool(py=e.__contains__("forward_session")) and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1":
            if blk["_native_session"] is None:
                blk["_native_session"] = e["create"]()
                blk["_session_binding"] = e["block_binding"]
            st.cached_tokens = e["forward_session"](blk["_native_session"], addresses, params)
        else:
            if step:
                params.pop(1)
            st.cached_tokens = e["decode" if step else "forward"](addresses, params)
    else:
        if GENERATION_BOUNDARY.get_or_create_ptr()[].bi.getattr(st, "_resident_session", PythonObject(None)) is not None:
            raise Error("neural block: state is owned by a resident decode session")
        addresses = py_ints([_addr(x, keep)])
        for j in range(len(blk["_w"])):
            addresses.append(_addr(blk["_w"][j], keep))
        if kind == "mamba1":
            var di = Int(py=blk["d_inner"])
            addresses.append(_mutable(st.conv_window, [b, di, 4], keep, copies))
            addresses.append(_mutable(st.h, [b, di, 16], keep, copies))
            addresses.append(_addr(y, keep))
            params = py_ints([b, d]) if step else py_ints([b, l, d])
            var profile = Int(py=e["mamba1_profile"])
            if Int(py=st.profile_id) != profile or not Bool(py=st._nn34_usable):
                raise Error("CausalLM: incompatible or failed Mamba1 state")
            if profile == 2:
                addresses.append(_mutable(st.affine_boundary, [b, di, 16], keep, copies))
                addresses.append(_mutable(st.affine_a, [b, di, 16, 6], keep, copies))
                addresses.append(_mutable(st.affine_b, [b, di, 16, 6], keep, copies))
                params.append(st.absolute_position)
            e["decode" if step else "forward"](addresses, params)
            st.absolute_position = PythonObject(Int(py=st.absolute_position) + l)
        else:
            var nh = Int(py=blk["nheads"])
            var cd = Int(py=blk["conv_dim"])
            addresses.append(_mutable(st.conv_window, [b, cd, 4], keep, copies))
            addresses.append(_mutable(st.h, [b, nh, 64, 128], keep, copies))
            addresses.append(_mutable(st.buffer_xbc, [b, 256, cd], keep, copies))
            addresses.append(_mutable(st.buffer_dtraw, [b, 256, nh], keep, copies))
            addresses.append(_addr(y, keep))
            var last = generation_array([b, Int(py=blk["nheads"]), 64, 128])
            addresses.append(_addr(last, keep))
            params = py_ints([b, d, Int(py=st.buffered_tokens)]) if step else py_ints([b, l, d, Int(py=st.buffered_tokens)])
            params.extend(blk["dt_limit"])
            st.buffered_tokens = e["decode" if step else "forward"](addresses, params)
            blk["h_last_"] = last
    _writeback(copies)
    _ = keep^
    return y


def causal_model_embedding(model: PythonObject, ids: PythonObject) raises -> PythonObject:
    var flat = Int(py=ids.ndim) == 1
    var b = 1 if flat else Int(py=ids.shape[0])
    var l = Int(py=ids.shape[0]) if flat else Int(py=ids.shape[1])
    var d = Int(py=model.d_model)
    var v = Int(py=model.vocab_size)
    var e = model._native_entries
    var keep = List[PythonObject]()
    var x = generation_array([b, l, d])
    e["embedding_forward"](py_ints([_addr(x, keep), _addr(model._embed, keep), _addr(ids, keep)]), py_ints([b * l, v, d]))
    if flat:
        var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
        x.shape = bi.tuple(py_ints([l, d]))
        x.strides = bi.tuple(py_ints([d * 4, 4]))
        x.ndim = PythonObject(2)
    _ = keep^
    return x


def causal_model_head(model: PythonObject, x: PythonObject, normalize: Bool = True, packed: Bool = True) raises -> PythonObject:
    var flat = Int(py=x.ndim) == 2
    var b = 1 if flat else Int(py=x.shape[0])
    var l = Int(py=x.shape[0]) if flat else Int(py=x.shape[1])
    var d = Int(py=model.d_model)
    var v = Int(py=model.vocab_size)
    var e = model._native_entries
    var keep = List[PythonObject]()
    var hn = x
    if normalize and model._norm is not None:
        hn = generation_array([b, l, d])
        var np = py_ints([b * l, d])
        np.append(model.norm_eps)
        e["rms_norm_forward"](py_ints([_addr(hn, keep), _addr(x, keep), _addr(model._norm, keep)]), np)
    var out = generation_array([b, l, v])
    if model._head_int15 is None or not packed:
        e["linear_forward"](py_ints([_addr(out, keep), _addr(hn, keep), _addr(model._head, keep)]), py_ints([b * l, v, d]))
    else:
        var hi = generation_array([b * l, d], "b")
        var lo = generation_array([b * l, d], "b")
        var ex = generation_array([b * l], "i")
        e["quantize15"](py_ints([_addr(hi, keep), _addr(lo, keep), _addr(ex, keep), _addr(hn, keep)]), py_ints([b * l, d]))
        var hp = model._head_int15
        e["gemm15"](py_ints([_addr(out, keep), _addr(hi, keep), _addr(lo, keep), _addr(ex, keep),
            _addr(hp[0], keep), _addr(hp[1], keep), _addr(hp[2], keep)]), py_ints([b * l, v, d]))
    if flat:
        var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
        out.shape = bi.tuple(py_ints([l, v]))
        out.strides = bi.tuple(py_ints([v * 4, 4]))
        out.ndim = PythonObject(2)
    _ = keep^
    return out


def _run(model: PythonObject, ids: PythonObject, state: PythonObject) raises -> PythonObject:
    var l = Int(py=ids.shape[1])
    _state(model, state, Int(py=ids.shape[0]), l)
    var parallel = model._native_entries.get("parallel_call")
    try:
        if parallel is not None:
            return parallel("run", Python.list(model, ids, state))
        var x = causal_model_embedding(model, ids)
        for i in range(len(model._blocks)):  # driver-loop: over ordered model layers
            x = causal_layer_run(model, i, x, state.layers[i])
        var out = causal_model_head(model, x)
        state.positions = PythonObject(Int(py=state.positions) + l)
        return out
    except error:
        state._usable = PythonObject(False)
        raise error


def causal_lm_forward_binding(model: PythonObject, value: PythonObject, supplied_state: PythonObject, step: PythonObject) raises -> PythonObject:
    var state = supplied_state
    var one = Bool(py=step)
    var ids = generation_ids(value, Int(py=model.vocab_size), one)
    if state is None:
        if one:
            raise Error("CausalLM.step: state is required")
        state = causal_lm_allocate_binding(model, ids.shape[0], ids.shape[1])
    var out = PythonObject(None)
    var parallel = model._native_entries.get("parallel_call")
    try:
        out = _run(model, ids, state)
    except error:
        if supplied_state is None and parallel is not None:
            try:
                _ = parallel("release_layers", Python.list(model, state))
            except cleanup_error:
                raise Error("CausalLM: forward failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if supplied_state is None and parallel is not None:
        _ = parallel("release_layers", Python.list(model, state))
    if one:
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        out.shape = g[].bi.tuple(py_ints([Int(py=ids.shape[0]), Int(py=model.vocab_size)]))
        out.strides = g[].bi.tuple(py_ints([Int(py=model.vocab_size) * 4, 4]))
        out.ndim = PythonObject(2)
    return out


def _resident_close(entries: PythonObject, lm: PythonObject, sessions: PythonObject) raises:
    entries["causal_lm_session_close"](lm)
    for i in range(len(sessions)):
        entries["transformer_decode_session_close"](sessions[i])


def causal_lm_resident_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, last_logits: PythonObject, supplied_state: PythonObject) raises -> PythonObject:
    var e = model._native_entries
    if not Bool(py=e.__contains__("causal_lm_session_run")) or model._norm is None:
        return PythonObject(None)
    var int15 = model._head_int15 is not None
    if int15 and String(e["resident_vendor"]) != "cuda":
        return PythonObject(None)
    var ids = generation_ids(value, Int(py=model.vocab_size))
    var n = Int(py=maximum)
    if n == 0:
        return ids
    if n < 0:
        raise Error("generation: max_new_tokens must be nonnegative")
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var state = supplied_state
    if state is None:
        state = causal_lm_allocate_binding(model, PythonObject(b), PythonObject(l + n))
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var sessions = g[].bi.list()
    var lm = e["causal_lm_session_create"]()
    var keep = List[PythonObject]()
    var out = generation_array([b, n], "i")
    try:
        for i in range(len(model._blocks)):
            var blk = _block(model, i)
            var st = state.layers[i]
            var a = _transformer_args(model, blk, st, PythonObject(None), PythonObject(None), b, l, int15, keep)
            var open_a = g[].bi.list()
            for j in range(1, 12):
                open_a.append(a[j])
            for j in range(13, len(a)):
                open_a.append(a[j])
            var p = _transformer_params(blk, st, b, l)
            var open_p = g[].bi.list()
            for j in range(len(p)):
                if j != 1:
                    open_p.append(p[j])
            var ss = e["transformer_decode_session_create"]()
            sessions.append(ss)
            e["transformer_decode_session_open_int15" if int15 else "transformer_decode_session_open"](ss, open_a, open_p)
        var top = py_ints([_addr(model._embed, keep), _addr(model._norm, keep)])
        if int15:
            for i in range(3):
                top.append(_addr(model._head_int15[i], keep))
        else:
            top.append(0 if model._head is model._embed else _addr(model._head, keep))
        var tp = py_ints([b, Int(py=model.vocab_size), Int(py=model.d_model)])
        tp.append(model.norm_eps)
        e["causal_lm_session_open_int15" if int15 else "causal_lm_session_open"](lm, top, tp)
        e["causal_lm_session_run"](lm, sessions, py_ints([_addr(ids, keep), _addr(out, keep), _addr(last_logits, keep)]), py_ints([l, n, 0]))
    except error:
        _resident_close(e, lm, sessions)
        raise error
    _resident_close(e, lm, sessions)
    var result = generation_array([b, l + n], "i")
    var dst = i32_ptr(Int(py=result._addr))
    var prompt = i32_ptr(Int(py=ids._addr))
    var generated = i32_ptr(Int(py=out._addr))
    for r in range(b):
        memcpy(dest=dst + r * (l + n), src=prompt + r * l, count=l)
        memcpy(dest=dst + r * (l + n) + l, src=generated + r * n, count=n)
    _ = keep^
    return result


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
        if wrote == 0:
            s.busy = False
            return out
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
    var raw = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
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


def _generate_complete(model: PythonObject, value: PythonObject, maximum: PythonObject, mut s: GenerationHandle) raises -> PythonObject:
    if s.greedy and len(s.eos) == 0:
        var resident = causal_lm_resident_binding(model, value, maximum, PythonObject(None), s.state)
        if resident is not None:
            return resident
    var continuation = _read(s, max(1, s.max_new))
    var b = len(s.lengths)
    var l = Int(py=s.prompt.shape[1])
    var n = Int(py=continuation.shape[1])
    var result = generation_array([b, l + n], "i")
    var dest = i32_ptr(Int(py=result._addr))
    var prompt = i32_ptr(Int(py=s.prompt._addr))
    var tail = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=continuation._addr))
    for r in range(b):
        memcpy(dest=dest + r * (l + n), src=prompt + r * l, count=l)
        if n > 0:
            memcpy(dest=dest + r * (l + n) + l, src=tail + r * n, count=n)
    _ = continuation^
    return result


def causal_lm_generate_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_stream_binding(model, value, maximum, options)
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    var parallel = model._native_entries.get("parallel_call")
    var result = PythonObject(None)
    try:
        result = _generate_complete(model, value, maximum, s[])
    except error:
        if parallel is not None:
            try:
                _ = parallel("release_layers", Python.list(model, s[].state))
            except cleanup_error:
                raise Error("CausalLM: generation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if parallel is not None:
        _ = parallel("release_layers", Python.list(model, s[].state))
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
    var parallel = model._native_entries.get("parallel_call")
    var result = PythonObject(None)
    try:
        var ids = _read(owner[], max(1, owner[].max_new))
        result = _decode_chunk(owner[], ids)
    except error:
        if parallel is not None:
            try:
                _ = parallel("release_layers", Python.list(model, owner[].state))
            except cleanup_error:
                raise Error("CausalLM: text generation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if parallel is not None:
        _ = parallel("release_layers", Python.list(model, owner[].state))
    return result


def _array_bytes(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].bi.type(value) is g[].array_cls:
        return value._mv.tobytes("C")
    return g[].bi.memoryview(value).tobytes("C")


def _hash() raises -> PythonObject:
    return GENERATION_BOUNDARY.get_or_create_ptr()[].hash_type()


def _hash_scalar(hash: PythonObject, value: PythonObject) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    hash.update(g[].bi.str(value).encode("utf-8"))
    hash.update(PythonObject("\n").encode("utf-8"))


def _fingerprint(model: PythonObject) raises -> PythonObject:
    var h = _hash()
    _hash_scalar(h, model.kind)
    _hash_scalar(h, model.numeric_profile)
    _hash_scalar(h, model.norm_eps)
    _hash_scalar(h, model.max_positions)
    _hash_scalar(h, model._embed.shape)
    h.update(_array_bytes(model._embed))
    if model._norm is not None:
        h.update(_array_bytes(model._norm))
    h.update(_array_bytes(model._head))
    for i in range(len(model._blocks)):
        var blk = _block(model, i)
        if String(model.kind) == "transformer":
            _hash_scalar(h, blk["_opts_tail"])
            _hash_scalar(h, blk["n_heads"])
            _hash_scalar(h, blk["n_kv_heads"])
            _hash_scalar(h, blk["window"])
            for j in range(len(blk["_wopt"])):
                if blk["_wopt"][j] is not None:
                    h.update(_array_bytes(blk["_wopt"][j]))
        elif String(model.kind) == "mamba2":
            _hash_scalar(h, blk["dt_limit"])
        else:
            _hash_scalar(h, model._native_entries["mamba1_profile"])
        for j in range(len(blk["_w"])):
            if blk["_w"][j] is not None:
                _hash_scalar(h, blk["_w"][j].shape)
                h.update(_array_bytes(blk["_w"][j]))
        if model._head_int15 is not None:
            for j in range(len(blk["_int15"])):
                if blk["_int15"][j] is not None:
                    for k in range(3):
                        h.update(_array_bytes(blk["_int15"][j][k]))
    if model._head_int15 is not None:
        for k in range(3):
            h.update(_array_bytes(model._head_int15[k]))
    return h.hexdigest()


def _state_arrays(kind: String, profile: Int = 1) -> List[String]:
    if kind == "transformer":
        return [String("k_cache"), String("v_cache")]
    if kind == "mamba2":
        return [String("conv_window"), String("h"), String("buffer_xbc"), String("buffer_dtraw")]
    if profile == 2:
        return [String("conv_window"), String("h"), String("affine_boundary"), String("affine_a"), String("affine_b")]
    return [String("conv_window"), String("h")]


def _snapshot_hash(checkpoint: PythonObject) raises -> PythonObject:
    var h = _hash()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for name in [String("schema"), String("model"), String("options"), String("maximum"),
                 String("produced"), String("positions"), String("lengths"), String("finished"), String("text"), String("tokenizer")]:
        _hash_scalar(h, checkpoint[name])
    h.update(checkpoint["pending_bytes"])
    h.update(_array_bytes(checkpoint["prompt"]))
    h.update(_array_bytes(checkpoint["current"]))
    for i in range(len(checkpoint["layers"])):
        var layer = checkpoint["layers"][i]
        var names = g[].bi.sorted(layer.keys())
        for j in range(len(names)):
            _hash_scalar(h, names[j])
            var value = layer[names[j]]
            if g[].bi.type(value) is g[].array_cls:
                _hash_scalar(h, value.shape)
                _hash_scalar(h, value.dtype)
                h.update(_array_bytes(value))
            else:
                _hash_scalar(h, value)
    return h.hexdigest()


def causal_lm_snapshot_binding(stream: PythonObject) raises -> PythonObject:
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    if s[].busy or not s[].usable:
        raise Error("generation: cannot checkpoint a busy or failed stream")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    result["schema"] = PythonObject("mojolearn.generation.v1")
    result["model"] = _fingerprint(s[].model)
    result["maximum"] = PythonObject(s[].max_new)
    result["produced"] = PythonObject(s[].produced)
    result["positions"] = s[].state.positions
    result["prompt"] = generation_copy(s[].prompt)
    result["current"] = generation_copy(s[].current)
    result["lengths"] = py_ints(s[].lengths)
    var finished = g[].bi.list()
    for i in range(len(s[].finished)):
        finished.append(s[].finished[i])
    result["finished"] = finished
    result["text"] = PythonObject(s[].tokenizer is not None)
    result["tokenizer"] = PythonObject(None)
    if s[].tokenizer is not None:
        result["tokenizer"] = _native_function(s[].tokenizer._m, "vocab_fingerprint")(s[].tokenizer)
    result["pending_bytes"] = g[].bi.bytes(s[].pending_bytes)
    var options = g[].bi.list()
    options.append(s[].greedy)
    # Decimal conversion avoids truncating UInt64 through the signed Python
    # bridge. Every unsigned seed is representable by Python's native int.
    options.append(g[].bi.int(String(s[].seed)))
    options.append(Float64(s[].temperature))
    options.append(s[].top_k)
    options.append(s[].top_p)
    options.append(py_ints(s[].eos))
    options.append(s[].pad)
    result["options"] = options
    var layers = g[].bi.list()
    var kind = String(s[].model.kind)
    var profile = Int(py=s[].model._native_entries["mamba1_profile"]) if kind == "mamba1" else 1
    var arrays = _state_arrays(kind, profile)
    for i in range(len(s[].state.layers)):
        var st = s[].state.layers[i]
        var parallel = s[].model._native_entries.get("parallel_call")
        if parallel is not None:
            st = parallel("state_snapshot", Python.list(st))
        var entry = g[].bi.dict()
        for j in range(len(arrays)):
            entry[arrays[j]] = generation_copy(g[].bi.getattr(st, arrays[j]))
        if kind == "transformer":
            entry["cached_tokens"] = st.cached_tokens
        elif kind == "mamba1":
            entry["absolute_position"] = st.absolute_position
            entry["profile_id"] = st.profile_id
        else:
            entry["buffered_tokens"] = st.buffered_tokens
        layers.append(entry)
    result["layers"] = layers
    result["checksum"] = _snapshot_hash(result)
    return result


def causal_lm_resume_binding(model: PythonObject, checkpoint: PythonObject, tokenizer: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if String(checkpoint["schema"]) != "mojolearn.generation.v1":
        raise Error("generation: unsupported checkpoint schema")
    if String(checkpoint["model"]) != String(_fingerprint(model)):
        raise Error("generation: checkpoint model weights/settings do not match")
    if String(checkpoint["checksum"]) != String(_snapshot_hash(checkpoint)):
        raise Error("generation: checkpoint checksum differs")
    if Bool(py=checkpoint["text"]) and tokenizer is None:
        raise Error("generation: a text checkpoint requires its tokenizer")
    if Bool(py=checkpoint["text"]):
        var fingerprint = _native_function(tokenizer._m, "vocab_fingerprint")(tokenizer)
        if not Bool(py=checkpoint["tokenizer"] == fingerprint):
            raise Error("generation: checkpoint tokenizer differs")
    var stream = causal_lm_stream_binding(model, checkpoint["prompt"], checkpoint["maximum"], checkpoint["options"])
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    s[].produced = Int(py=checkpoint["produced"])
    if s[].produced < 0 or s[].produced > s[].max_new:
        raise Error("generation: invalid checkpoint output position")
    s[].current = generation_ids(checkpoint["current"], Int(py=model.vocab_size))
    var b = len(s[].lengths)
    if Int(py=s[].current.shape[0]) != b or len(checkpoint["lengths"]) != b or len(checkpoint["finished"]) != b:
        raise Error("generation: checkpoint batch counts differ")
    if s[].produced > 0 and Int(py=s[].current.shape[1]) != 1:
        raise Error("generation: continuation checkpoint must hold one pending token per row")
    var position = 0 if s[].produced == 0 else Int(py=s[].prompt.shape[1]) + s[].produced - 1
    if Int(py=checkpoint["positions"]) != position:
        raise Error("generation: checkpoint cache position disagrees with output count")
    s[].state.positions = PythonObject(position)
    for i in range(b):
        s[].lengths[i] = Int(py=checkpoint["lengths"][i])
        s[].finished[i] = Bool(py=checkpoint["finished"][i])
        if s[].lengths[i] < 0 or s[].lengths[i] > s[].produced:
            raise Error("generation: invalid checkpoint row length")
    if len(checkpoint["layers"]) != len(s[].state.layers):
        raise Error("generation: checkpoint layer count differs")
    var kind = String(model.kind)
    var profile = Int(py=model._native_entries["mamba1_profile"]) if kind == "mamba1" else 1
    var arrays = _state_arrays(kind, profile)
    for i in range(len(s[].state.layers)):
        var dest = s[].state.layers[i]
        var parallel = model._native_entries.get("parallel_call")
        if parallel is not None:
            dest = parallel("state_snapshot", Python.list(dest))
        var source = checkpoint["layers"][i]
        for j in range(len(arrays)):
            var expected = g[].bi.getattr(dest, arrays[j])
            var supplied = source[arrays[j]]
            if g[].bi.type(supplied) is not g[].array_cls or String(supplied.dtype) != "<f4" or not Bool(py=supplied.shape == expected.shape):
                raise Error("generation: checkpoint state array shape or dtype differs")
            g[].bi.setattr(dest, arrays[j], generation_copy(supplied))
        if kind == "transformer":
            if Int(py=source["cached_tokens"]) != position:
                raise Error("generation: checkpoint layer cache position differs")
            dest.cached_tokens = PythonObject(position)
        elif kind == "mamba1":
            if Int(py=source["profile_id"]) != profile or Int(py=source["absolute_position"]) != position:
                raise Error("generation: checkpoint Mamba profile/position differs")
            dest.absolute_position = PythonObject(position)
        else:
            if Int(py=source["buffered_tokens"]) != position % 256:
                raise Error("generation: checkpoint Mamba2 buffered length differs")
            dest.buffered_tokens = source["buffered_tokens"]
        if parallel is not None:
            _ = parallel("state_publish", Python.list(s[].state.layers[i], dest))
    s[].tokenizer = tokenizer if Bool(py=checkpoint["text"]) else PythonObject(None)
    s[].pending_bytes = g[].bi.bytearray(checkpoint["pending_bytes"])
    return stream


def causal_state_init_binding(state: PythonObject, values: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    state.batch_size = bi.int(values[0])
    state.max_tokens = bi.int(values[1])
    state.layers = bi.list(values[2])
    state.positions = PythonObject(0)
    state._owner = values[3]
    state._usable = PythonObject(True)
    return PythonObject(None)


def causal_public_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if operation == 0:
            result = causal_lm_admit_binding(args[0], args[1], args[2], args[3])
        elif operation == 1:
            result = causal_lm_allocate_binding(args[0], args[1], args[2])
        elif operation == 2:
            result = causal_lm_reset_binding(args[0], args[1])
        elif operation == 3:
            result = causal_lm_forward_binding(args[0], args[1], args[2], args[3])
        elif operation == 4:
            result = causal_lm_stream_binding(args[0], args[1], args[2], args[3])
        elif operation == 5:
            result = causal_lm_read_binding(args[0], args[1])
        elif operation == 6:
            result = causal_lm_info_binding(args[0])
        elif operation == 7:
            result = causal_lm_generate_binding(args[0], args[1], args[2], args[3])
        elif operation == 8:
            result = causal_lm_text_stream_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 9:
            result = causal_lm_generate_text_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 10:
            result = causal_lm_resident_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 11:
            result = causal_lm_snapshot_binding(args[0])
        elif operation == 12:
            result = causal_lm_resume_binding(args[0], args[1], args[2])
        elif operation == 13:
            result = causal_lm_parameters_binding(args[0])
        elif operation == 14:
            result = causal_lm_blocks_binding(args[0])
        else:
            result = causal_state_init_binding(args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
