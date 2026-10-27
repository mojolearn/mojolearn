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
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from checks.numerics import NUMERIC_IDENTICAL
from providers.python.generation_session_pyapi import (
    generation_connect_native_provider, generation_connect_native_session,
    generation_native_session_matches, generation_native_forward,
)
from std.memory import memcpy
from bindings.generation_buffers import (
    GENERATION_BOUNDARY, generation_array, generation_address, generation_copy,
    generation_ids, py_ints,
)
from bindings.generation_policy import generation_pick
from core.hostptr import f32_ptr, i32_ptr
from core.sha256 import Sha256
from bindings.sha256_door import sha256_update, sha256_update_text
from std.math import isfinite
from std.os import getenv
from transformer.causal_lm_public_impl import (
    causal_lm_rows_done, causal_lm_join_resident, causal_lm_join_continuation,
    causal_lm_pick_step, causal_lm_tokens_row_major, causal_lm_drop_eos,
    causal_lm_utf8_cut, causal_lm_store_ids, causal_lm_flags_as_ints,
    causal_lm_restore_rows,
)


def _native_function(module: PythonObject, name: String) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var entry = py_call(g[].bi.getattr, module, name)
    if py_type(entry) is not py_type(g[].bi.len):
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
    py_setattr(model, "_native_vendor", PythonObject(vendor))
    py_setattr(model, "_native_numeric_mode", PythonObject(mode))
    py_setitem(entries, "weight_names", py_call(g[].bi.object.__getattribute__, model._blocks[0], "_W_NAMES"))
    for name in [String("embedding_forward"), String("rms_norm_forward"), String("linear_forward")]:  # small-loop(primitive entry names)
        py_setitem(entries, name, _native_function(primitives, name))
    if kind == "transformer":
        py_setitem(entries, "forward", _native_function(block_binding, "transformer_forward"))
        py_setitem(entries, "decode", _native_function(block_binding, "transformer_decode_step"))
        py_setitem(entries, "block_binding", block_binding)
        for name in [String("transformer_forward_fresh"), String("transformer_session_forward_fresh")]:  # small-loop(optional entry names)
            if Bool(py=py_call(g[].bi.hasattr, block_binding, name)):
                py_setitem(entries, name, _native_function(block_binding, name))
        if Bool(py=py_call(g[].bi.hasattr, block_binding, "transformer_session_forward")):
            py_setitem(entries, "create", _native_function(block_binding, "transformer_session_create"))
            py_setitem(entries, "forward_session", _native_function(block_binding, "transformer_session_forward"))
            py_setitem(entries, "block_binding", block_binding)
        if model._head_int15 is not None:
            if Bool(py=py_call(g[].bi.hasattr, block_binding, "transformer_session_forward_int15")):
                py_setitem(entries, "create", _native_function(block_binding, "transformer_session_create"))
                py_setitem(entries, "forward15", _native_function(block_binding, "transformer_session_forward_int15"))
            else:
                py_setitem(entries, "forward15_host", _native_function(block_binding, "transformer_forward_int15"))
            py_setitem(entries, "quantize15", _native_function(int15_binding, "quantize_int15"))
            py_setitem(entries, "gemm15", _native_function(int15_binding, "gemm_int15"))
        # Preserve the existing resident GPU route for fixed-length greedy
        # generation. No performance default changes with this migration.
        if Bool(py=py_call(g[].bi.hasattr, block_binding, "causal_lm_session_run")):
            py_setitem(entries, "resident_vendor", _native_function(block_binding, "transformer_vendor")())
            for name in [String("transformer_decode_session_create"), String("transformer_decode_session_open"),  # small-loop(resident entry names)
                         String("transformer_decode_session_close"), String("causal_lm_session_create"),
                         String("causal_lm_session_open"), String("causal_lm_session_run"), String("causal_lm_session_close")]:
                py_setitem(entries, name, _native_function(block_binding, name))
            for name in [String("transformer_decode_session_load_state"), String("transformer_decode_session_export_state"),  # small-loop(optional resident entry names)
                         String("causal_lm_session_forward")]:
                if Bool(py=py_call(g[].bi.hasattr, block_binding, name)):
                    py_setitem(entries, name, _native_function(block_binding, name))
            if model._head_int15 is not None:
                py_setitem(entries, "transformer_decode_session_open_int15", _native_function(block_binding, "transformer_decode_session_open_int15"))
                py_setitem(entries, "causal_lm_session_open_int15", _native_function(block_binding, "causal_lm_session_open_int15"))
    elif kind == "mamba1":
        py_setitem(entries, "forward", _native_function(block_binding, "mamba1_forward"))
        py_setitem(entries, "decode", _native_function(block_binding, "mamba1_decode_step"))
        var profile = 1
        if Bool(py=py_call(g[].bi.hasattr, block_binding, "mamba1_profile")):
            profile = Int(py=_native_function(block_binding, "mamba1_profile")())
        py_setitem(entries, "mamba1_profile", PythonObject(profile))
    elif kind == "mamba2":
        py_setitem(entries, "forward", _native_function(block_binding, "mamba2_forward"))
        py_setitem(entries, "decode", _native_function(block_binding, "mamba2_decode_step"))
    else:
        raise Error("CausalLM: unsupported model kind " + kind)
    # Private candidate: resolve its native table once at admission. Keep
    # the qualified default until whole-workload NVIDIA/AMD A/B and required
    # public runtime/identity checks are complete. No Apple fast campaign.
    if requested_gpu and kind == "transformer" and mode == NUMERIC_IDENTICAL and String(getenv("MOJOLEARN_GENERATION_NATIVE_SESSION")) == "1":
        if not py_contains(entries, "forward_session"):
            raise Error("CausalLM: native session candidate requires transformer_session_forward")
        var provider = generation_connect_native_provider(block_binding,
            _native_function(block_binding, "transformer_native_api_v1"),
            _native_function(block_binding, "transformer_native_session_handle_v1"))
        py_setitem(entries, "native_session_provider", provider)
    py_setattr(model, "_native_entries", entries)
    return PythonObject(None)


def _block(model: PythonObject, index: Int) raises -> PythonObject:
    # PublicOutputsMixin.__getattribute__ is Python even for private metadata.
    # Access the object's plain dictionary with CPython's base implementation,
    # bypassing that overridden getter; every field below is stored metadata.
    var parallel = py_get(model._native_entries, "parallel_call")
    if parallel is not None:
        return py_call(parallel, "block_fields", Python.list(model, index))
    return py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].bi.object.__getattribute__, model._blocks[index], "__dict__")


def _positions(model: PythonObject, state: PythonObject, length: Int) raises:
    var have = 0 if state is None else Int(py=state.positions)
    if length < 0 or have < 0:
        raise Error("CausalLM: negative sequence position")
    if model.max_positions is not None and have + length > Int(py=model.max_positions):
        raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
    if state is not None and have + length > Int(py=state.max_tokens):
        raise Error("CausalLM: sequence exceeds allocated state capacity")


def causal_lm_parameters_binding(model: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    # The live arrays leave the model: a kept resident session set could miss
    # an in-place edit, so it is dropped and generation opens per call again.
    var model_fields = py_call(g[].bi.object.__getattribute__, model, "__dict__")
    py_setitem(model_fields, "_parameters_exposed", True)
    py_setitem(model_fields, "_resident_sessions", PythonObject(None))
    var p = model.plan
    py_setitem(result, p.embed_name, model._embed)
    if p.norm_name is not None:
        py_setitem(result, p.norm_name, model._norm)
    if p.head_name is not None:
        py_setitem(result, p.head_name, model._head)
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
            py_setitem(names, entry[0], PythonObject(String(entry[1]).replace("{i}", String(i))))
            py_setitem(slices, entry[0], entry[2])
        var blk = _block(model, i)
        var values = g[].bi.dict()
        for j in range(min(len(weight_names), len(blk["_w"]))):  # driver-loop: over block weight slots
            if blk["_w"][j] is not None:
                py_setitem(values, weight_names[j], blk["_w"][j])
        if String(model.kind) == "transformer":
            for j in range(len(optional_names)):  # driver-loop: over optional block weight slots
                if blk["_wopt"][j] is not None:
                    py_setitem(values, optional_names[j], blk["_wopt"][j])
        var keys = py_call(g[].bi.list, values)
        var fused = g[].bi.dict()
        for j in range(len(keys)):  # driver-loop: over checkpoint tensor names
            var key = keys[j]
            var name = py_get(names, key, PythonObject("layers." + String(i) + "." + String(key)))
            var rows = py_get(slices, key)
            if rows is None:
                py_setitem(result, name, values[key])
            else:
                # A fused checkpoint tensor (Phi-3 QKV / gate-up) is split
                # for execution. Return every row under its checkpoint name.
                var piece = py_call(g[].bi.list, rows)
                _ = py_call(piece.append, values[key])
                _ = py_call(py_call(fused.setdefault, name, g[].bi.list()).append, piece)
        var fused_names = py_call(g[].bi.list, fused)
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
                    _ = joined  # keep the Python object alive through the native call above
            py_setitem(result, name, joined)
    _ = keep^
    return result


def causal_lm_blocks_binding(model: PythonObject) raises -> PythonObject:
    return py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].bi.list, model._blocks)


def causal_layer_allocate(model: PythonObject, i: Int, b: Int, cap: Int) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var kind = String(model.kind)
    var blk = _block(model, i)
    var st = py_call(g[].bi.object.__new__, g[].layer_classes[kind])
    if kind == "transformer":
        var window = Int(py=blk["window"])
        var nkv = Int(py=blk["n_kv_heads"])
        var hd = Int(py=blk["head_dim"])
        var n = b * nkv * (cap if window == 0 else window) * hd
        py_setattr(st, "batch_size", PythonObject(b))
        py_setattr(st, "n_kv_heads", PythonObject(nkv))
        py_setattr(st, "head_dim", PythonObject(hd))
        py_setattr(st, "max_tokens", PythonObject(cap))
        py_setattr(st, "window", PythonObject(window))
        py_setattr(st, "k_cache", generation_array([n]))
        py_setattr(st, "v_cache", generation_array([n]))
        py_setattr(st, "cached_tokens", PythonObject(0))
        py_setattr(st, "_resident_session", PythonObject(None))
        py_setattr(st, "_numeric_profile", model.numeric_profile)
    elif kind == "mamba1":
        var di = Int(py=blk["d_inner"])
        var profile = Int(py=model._native_entries["mamba1_profile"])
        py_setattr(st, "conv_window", generation_array([b, di, 4]))
        py_setattr(st, "h", generation_array([b, di, 16]))
        py_setattr(st, "profile_id", PythonObject(profile))
        py_setattr(st, "absolute_position", PythonObject(0))
        py_setattr(st, "_nn34_usable", PythonObject(True))
        py_setattr(st, "affine_boundary", PythonObject(None))
        py_setattr(st, "affine_a", PythonObject(None))
        py_setattr(st, "affine_b", PythonObject(None))
        if profile == 2:
            py_setattr(st, "affine_boundary", generation_array([b, di, 16]))
            py_setattr(st, "affine_a", generation_array([b, di, 16, 6]))
            py_setattr(st, "affine_b", generation_array([b, di, 16, 6]))
    else:
        var cd = Int(py=blk["conv_dim"])
        var nh = Int(py=blk["nheads"])
        py_setattr(st, "conv_window", generation_array([b, cd, 4]))
        py_setattr(st, "h", generation_array([b, nh, 64, 128]))
        py_setattr(st, "buffer_xbc", generation_array([b, 256, cd]))
        py_setattr(st, "buffer_dtraw", generation_array([b, 256, nh]))
        py_setattr(st, "buffered_tokens", PythonObject(0))
    return st


def _public_int(value: PythonObject, message: String) raises -> Int:
    # A public count is an integer (anything with __index__): a str is not
    # parsed and a bool is not counted as 0 or 1.
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].bi.bool:
        raise Error("TypeError: " + message)
    var index = PythonObject(None)
    try:
        index = py_call(Python.import_module("operator").index, value)
    except error:
        raise Error("TypeError: " + message)
    return Int(py=index)


def causal_lm_allocate_binding(model: PythonObject, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    var b = _public_int(batch, "CausalLM: batch_size and max_tokens must be integers")
    var cap = _public_int(capacity, "CausalLM: batch_size and max_tokens must be integers")
    if b < 1 or cap < 1:
        raise Error("CausalLM: batch_size and max_tokens must be positive")
    _positions(model, PythonObject(None), cap)
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var state = py_call(g[].bi.object.__new__, g[].state_cls)
    py_setattr(state, "batch_size", PythonObject(b))
    py_setattr(state, "max_tokens", PythonObject(cap))
    py_setattr(state, "positions", PythonObject(0))
    py_setattr(state, "_owner", model)
    py_setattr(state, "_usable", PythonObject(True))
    py_setattr(state, "layers", g[].bi.list())
    var kind = String(model.kind)
    var parallel = py_get(model._native_entries, "parallel_call")
    try:
        for i in range(len(model._blocks)):  # driver-loop: over model layer state owners
            _ = py_call(state.layers.append, causal_layer_allocate(model, i, b, cap) if parallel is None else
                        py_call(parallel, "allocate_layer", Python.list(model, i, b, cap)))
    except error:
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, state))
            except cleanup_error:
                raise Error("CausalLM: state allocation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    return state


def _state(model: PythonObject, state: PythonObject, b: Int, l: Int) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(state) is not g[].state_cls or state._owner is not model:
        raise Error("CausalLM: state belongs to another model")
    if not Bool(py=py_call(g[].bi.getattr, state, "_usable", True)):
        raise Error("CausalLM: state lost after a failed call; reset or restore it")
    if Int(py=state.batch_size) != b or len(state.layers) != len(model._blocks):
        raise Error("CausalLM: state batch/layer counts disagree with model and input")
    _positions(model, state, l)


def causal_lm_reset_binding(model: PythonObject, state: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(state) is not g[].state_cls or state._owner is not model:
        raise Error("CausalLM: state belongs to another model")
    var fresh = causal_lm_allocate_binding(model, state.batch_size, state.max_tokens)
    var parallel = py_get(model._native_entries, "parallel_call")
    if parallel is not None:
        try:
            _ = py_call(parallel, "release_layers", Python.list(model, state))
        except error:
            py_setattr(state, "_usable", PythonObject(False))
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, fresh))
            except cleanup_error:
                raise Error("CausalLM: state reset failed: " + String(error) + "; fresh-state cleanup also failed: " + String(cleanup_error))
            raise error
    py_setattr(state, "layers", fresh.layers)
    py_setattr(state, "positions", PythonObject(0))
    py_setattr(state, "_usable", PythonObject(True))
    return state


def _addr(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    return generation_address(value, keep)


def _mutable(
    value: PythonObject, expected: List[Int], mut keep: List[PythonObject], mut copies: List[PythonObject],
    name: String = "",
) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var label = String("state buffer")
    if name != "":
        label = "state buffer " + name
    var view = PythonObject(None)
    var shape = PythonObject(None)
    if value is None:
        raise Error("CausalLM: " + label + " is missing; allocate_state() makes a complete state")
    if py_type(value) is g[].array_cls:
        if String(value.dtype) != "<f4" or String(value.order) != "C" or Bool(py=value._readonly):
            raise Error("CausalLM: " + label + " must be writable C-contiguous float32 (it is updated in place)")
        view = value._mv
        shape = value.shape
    else:
        view = py_call(g[].bi.memoryview, value)
        shape = view.shape
        if Bool(py=view.readonly) or not Bool(py=view.c_contiguous) or String(view.format) != "f":
            raise Error("CausalLM: " + label + " must be writable C-contiguous float32 (it is updated in place)")
    var want = String("(")
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if i > 0:
            want += ", "
        want += String(expected[i])
    want += ")"
    if len(shape) != len(expected):
        raise Error("CausalLM: " + label + " has shape " + String(shape) + ", want " + want)
    var size = 1
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if Int(py=shape[i]) != expected[i]:
            raise Error("CausalLM: " + label + " has shape " + String(shape) + ", want " + want)
        size *= expected[i]
    if Int(py=view.nbytes) != size * 4:
        raise Error("CausalLM: " + label + " span differs from its shape")
    var address = _addr(value, keep)
    if py_type(value) is not g[].array_cls:
        var pair = g[].bi.list()
        _ = py_call(pair.append, view)
        _ = py_call(pair.append, keep[len(keep) - 1])
        copies.append(pair)
    return address


def _writeback(mut copies: List[PythonObject]) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for i in range(len(copies)):  # driver-loop: over copied state buffer handles
        var view = py_call(copies[i][0].cast, "B")
        var source = py_call(py_call(g[].bi.memoryview, copies[i][1]).cast, "B")
        py_setitem(view, py_call(g[].bi.slice, PythonObject(None)), source)
    copies.clear()


def _transformer_arg_words(
    model: PythonObject, blk: PythonObject, st: PythonObject, x: PythonObject, y: PythonObject,
    b: Int, l: Int, planes: Bool, mut keep: List[PythonObject],
) raises -> List[Int]:
    var addresses = List[Int]()
    addresses.append(_addr(x, keep))
    var weights = blk["_w"]
    for j in range(len(weights)):  # driver-loop: over block weight slots
        addresses.append(0 if planes and j >= 2 else _addr(weights[j], keep))
    addresses.append(_addr(st.k_cache, keep))
    addresses.append(_addr(st.v_cache, keep))
    addresses.append(_addr(y, keep))
    var optional = blk["_wopt"]
    for j in range(len(optional)):  # driver-loop: over optional block weight slots
        addresses.append(_addr(optional[j], keep))
    if planes:
        var packed = blk["_int15"]
        for j in range(len(packed)):  # driver-loop: over packed block weight slots
            var trip = packed[j]
            for k in range(3):  # small-loop(int15 plane triple)
                addresses.append(0 if trip is None else _addr(trip[k], keep))
    return addresses^


def _transformer_param_words(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> List[Int]:
    var p: List[Int] = [b, l, Int(py=blk["d_model"]), Int(py=blk["n_heads"]), Int(py=blk["n_kv_heads"]),
                        Int(py=blk["head_dim"]), Int(py=blk["intermediate"]), Int(py=st.max_tokens),
                        Int(py=st.cached_tokens), Int(py=blk["window"])]
    var tail = blk["_opts_tail"]
    for j in range(len(tail)):  # driver-loop: read the block's admitted options tail
        p.append(Int(py=tail[j]))
    return p^


def _transformer_args(
    model: PythonObject, blk: PythonObject, st: PythonObject, x: PythonObject, y: PythonObject,
    b: Int, l: Int, planes: Bool, mut keep: List[PythonObject],
) raises -> PythonObject:
    return py_ints(_transformer_arg_words(model, blk, st, x, y, b, l, planes, keep))


def _transformer_params(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> PythonObject:
    return py_ints(_transformer_param_words(blk, st, b, l))


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
    # every layer kind: a state a decode session holds resident is refused by name
    var gb = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_call(gb[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
        raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
    if kind == "transformer":
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        var old_profile = py_call(g[].bi.getattr, st, "arithmetic_profile", PythonObject(None))
        var profile_id = py_get(blk, "_arithmetic_profile", PythonObject(None))
        if old_profile is None and Int(py=st.cached_tokens) != 0 and Bool(py=py_get(blk, "_arithmetic_profile_changed", PythonObject(False))):
            raise Error("CausalLM: cached state requires arithmetic profile metadata")
        if old_profile is not None and profile_id is not None and not Bool(py=old_profile == profile_id):
            raise Error("CausalLM: state arithmetic profile differs")
        if profile_id is not None:
            py_setattr(st, "arithmetic_profile", profile_id)
        var cap = Int(py=st.max_tokens) if Int(py=blk["window"]) == 0 else Int(py=blk["window"])
        var cache_n = b * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
        var p_words = _transformer_param_words(blk, st, b, l)
        var profile = model._head_int15 is not None
        var device15 = profile and py_contains(e, "forward15")
        var a_words = _transformer_arg_words(model, blk, st, x, y, b, l, device15, keep)
        a_words[10] = _mutable(st.k_cache, [cache_n], keep, copies, "k_cache")
        a_words[11] = _mutable(st.v_cache, [cache_n], keep, copies, "v_cache")
        var native_provider = py_get(e, "native_session_provider")
        if native_provider is not None and not profile and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1":
            if blk["_native_session"] is None:
                py_setitem(blk, "_native_session", e["create"]())
                py_setitem(blk, "_session_binding", e["block_binding"])
            var session = blk["_native_session"]
            var linked = py_get(blk, "_native_forward_owner")
            if linked is None or not generation_native_session_matches(linked, native_provider, session):
                linked = generation_connect_native_session(native_provider, session)
                py_setitem(blk, "_native_forward_owner", linked)
            # Native vectors cross directly. No Python argument-list packing
            # or CPython callable dispatch occurs in this repeated layer call.
            py_setattr(st, "cached_tokens", PythonObject(generation_native_forward(linked, a_words, p_words)))
        else:
            addresses = py_ints(a_words)
            params = py_ints(p_words)
            if device15:
                if blk["_native_session"] is None:
                    py_setitem(blk, "_native_session", e["create"]())
                    py_setitem(blk, "_session_binding", e["block_binding"])
                py_setattr(st, "cached_tokens", py_call(e["forward15"], blk["_native_session"], addresses, params))
            elif profile:
                py_setattr(st, "cached_tokens", py_call(e["forward15_host"], addresses, params))
            elif py_contains(e, "forward_session") and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1":
                if blk["_native_session"] is None:
                    py_setitem(blk, "_native_session", e["create"]())
                    py_setitem(blk, "_session_binding", e["block_binding"])
                py_setattr(st, "cached_tokens", py_call(e["forward_session"], blk["_native_session"], addresses, params))
            else:
                if step:
                    _ = py_call(params.pop, 1)
                py_setattr(st, "cached_tokens", py_call(e["decode" if step else "forward"], addresses, params))
    else:
        if py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
            raise Error("neural block: state is owned by a resident decode session")
        addresses = py_ints([_addr(x, keep)])
        for j in range(len(blk["_w"])):  # driver-loop: over block weight slots
            _ = py_call(addresses.append, _addr(blk["_w"][j], keep))
        if kind == "mamba1":
            var di = Int(py=blk["d_inner"])
            _ = py_call(addresses.append, _mutable(st.conv_window, [b, di, 4], keep, copies, "conv_window"))
            _ = py_call(addresses.append, _mutable(st.h, [b, di, 16], keep, copies, "h"))
            _ = py_call(addresses.append, _addr(y, keep))
            _ = y  # keep the Python object alive through the native call above
            params = py_ints([b, d]) if step else py_ints([b, l, d])
            var profile = Int(py=e["mamba1_profile"])
            if Int(py=st.profile_id) != profile or not Bool(py=st._nn34_usable):
                raise Error("CausalLM: incompatible or failed Mamba1 state")
            if profile == 2:
                _ = py_call(addresses.append, _mutable(st.affine_boundary, [b, di, 16], keep, copies, "affine_boundary"))
                _ = y  # keep the Python object alive through the native call above
                _ = py_call(addresses.append, _mutable(st.affine_a, [b, di, 16, 6], keep, copies, "affine_a"))
                _ = y  # keep the Python object alive through the native call above
                _ = py_call(addresses.append, _mutable(st.affine_b, [b, di, 16, 6], keep, copies, "affine_b"))
                _ = y  # keep the Python object alive through the native call above
                _ = py_call(params.append, st.absolute_position)
            _ = py_call(e["decode" if step else "forward"], addresses, params)
            _ = y  # keep the Python object alive through the native call above
            py_setattr(st, "absolute_position", PythonObject(Int(py=st.absolute_position) + l))
        else:
            var nh = Int(py=blk["nheads"])
            var cd = Int(py=blk["conv_dim"])
            _ = py_call(addresses.append, _mutable(st.conv_window, [b, cd, 4], keep, copies, "conv_window"))
            _ = y  # keep the Python object alive through the native call above
            _ = py_call(addresses.append, _mutable(st.h, [b, nh, 64, 128], keep, copies, "h"))
            _ = y  # keep the Python object alive through the native call above
            _ = py_call(addresses.append, _mutable(st.buffer_xbc, [b, 256, cd], keep, copies, "buffer_xbc"))
            _ = y  # keep the Python object alive through the native call above
            _ = py_call(addresses.append, _mutable(st.buffer_dtraw, [b, 256, nh], keep, copies, "buffer_dtraw"))
            _ = y  # keep the Python object alive through the native call above
            _ = py_call(addresses.append, _addr(y, keep))
            _ = y  # keep the Python object alive through the native call above
            var last = generation_array([b, Int(py=blk["nheads"]), 64, 128])
            _ = py_call(addresses.append, _addr(last, keep))
            _ = last  # keep the Python object alive through the native call above
            _ = y  # keep the Python object alive through the native call above
            params = py_ints([b, d, Int(py=st.buffered_tokens)]) if step else py_ints([b, l, d, Int(py=st.buffered_tokens)])
            _ = py_call(params.extend, blk["dt_limit"])
            py_setattr(st, "buffered_tokens", py_call(e["decode" if step else "forward"], addresses, params))
            _ = last  # keep the Python object alive through the native call above
            _ = y  # keep the Python object alive through the native call above
            py_setitem(blk, "h_last_", last)
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
    _ = py_call(e["embedding_forward"], py_ints([_addr(x, keep), _addr(model._embed, keep), _addr(ids, keep)]), py_ints([b * l, v, d]))
    _ = x  # keep the Python object alive through the native call above
    if flat:
        var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
        py_setattr(x, "shape", py_call(bi.tuple, py_ints([l, d])))
        py_setattr(x, "strides", py_call(bi.tuple, py_ints([d * 4, 4])))
        py_setattr(x, "ndim", PythonObject(2))
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
        _ = py_call(np.append, model.norm_eps)
        _ = py_call(e["rms_norm_forward"], py_ints([_addr(hn, keep), _addr(x, keep), _addr(model._norm, keep)]), np)
        _ = hn  # keep the Python object alive through the native call above
    var out = generation_array([b, l, v])
    if model._head_int15 is None or not packed:
        _ = py_call(e["linear_forward"], py_ints([_addr(out, keep), _addr(hn, keep), _addr(model._head, keep)]), py_ints([b * l, v, d]))
        _ = out  # keep the Python object alive through the native call above
        _ = hn  # keep the Python object alive through the native call above
    else:
        var hi = generation_array([b * l, d], "b")
        var lo = generation_array([b * l, d], "b")
        var ex = generation_array([b * l], "i")
        _ = py_call(e["quantize15"], py_ints([_addr(hi, keep), _addr(lo, keep), _addr(ex, keep), _addr(hn, keep)]), py_ints([b * l, d]))
        _ = ex  # keep the Python object alive through the native call above
        _ = lo  # keep the Python object alive through the native call above
        _ = hi  # keep the Python object alive through the native call above
        _ = hn  # keep the Python object alive through the native call above
        var hp = model._head_int15
        e["gemm15"](py_ints([_addr(out, keep), _addr(hi, keep), _addr(lo, keep), _addr(ex, keep),
            _addr(hp[0], keep), _addr(hp[1], keep), _addr(hp[2], keep)]), py_ints([b * l, v, d]))
        _ = ex  # keep the Python object alive through the native call above
        _ = lo  # keep the Python object alive through the native call above
        _ = hi  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    if flat:
        var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
        py_setattr(out, "shape", py_call(bi.tuple, py_ints([l, v])))
        py_setattr(out, "strides", py_call(bi.tuple, py_ints([v * 4, 4])))
        py_setattr(out, "ndim", PythonObject(2))
    _ = keep^
    return out


def _resident_forward_ok(model: PythonObject, state: PythonObject) raises -> Bool:
    """Whether `forward` runs as the one resident program
    (`causal_lm_session_forward`): a transformer stack with a norm, the
    resident entries, a head the resident session takes, no arithmetic
    profile metadata to carry and no layer state held by a decode session."""
    var e = model._native_entries
    if String(model.kind) != "transformer" or model._norm is None:
        return False
    for name in [String("causal_lm_session_forward"), String("transformer_decode_session_load_state"),  # small-loop(required resident entry names)
                 String("transformer_decode_session_export_state"), String("causal_lm_session_run")]:
        if not py_contains(e, name):
            return False
    if String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) == "1" or String(getenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD")) == "0":
        return False
    if model._head_int15 is not None and String(e["resident_vendor"]) != "cuda":
        return False
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for i in range(len(model._blocks)):  # driver-loop: over model layers
        var blk = _block(model, i)
        if py_get(blk, "_arithmetic_profile", PythonObject(None)) is not None:
            return False
        var st = state.layers[i]
        if py_call(g[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
            return False
        if py_call(g[].bi.getattr, st, "arithmetic_profile", PythonObject(None)) is not None:
            return False
    return True


def _resident_forward(model: PythonObject, ids: PythonObject, state: PythonObject) raises -> PythonObject:
    """`(B, L, V)` logits of `forward` in one resident program: the kept (or
    newly opened) sessions take the state's caches, run the stack, and the
    advanced caches come back into the state."""
    var e = model._native_entries
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var v = Int(py=model.vocab_size)
    var int15 = model._head_int15 is not None
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var fields = py_call(g[].bi.object.__getattribute__, model, "__dict__")
    var persist = (not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                   and not Bool(py=py_get(fields, "_resident_busy", PythonObject(False))))
    var key = _resident_key(model, state, b, int15)
    var kept = py_get(fields, "_resident_sessions") if persist else PythonObject(None)
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    var out = generation_array([b, l, v])
    if persist:
        py_setitem(fields, "_resident_busy", True)
    try:
        if kept is not None and Bool(py=kept[0] == key):
            lm = kept[1]
            sessions = kept[2]
            _resident_load_states(e, sessions, state, keep)
        else:
            if kept is not None:
                py_setitem(fields, "_resident_sessions", PythonObject(None))
                _resident_close(e, kept[1], kept[2])
            var opened = _resident_open_all(model, e, state, b, l, int15, keep)
            lm = opened[0]
            sessions = opened[1]
        var cached = py_call(e["causal_lm_session_forward"], lm, sessions, py_ints([_addr(ids, keep), _addr(out, keep)]), py_ints([l]))
        _ = out  # keep the Python object alive through the native call above
        for i in range(len(sessions)):  # driver-loop: over layer decode sessions
            var blk = _block(model, i)
            var st = state.layers[i]
            var cap = Int(py=st.max_tokens) if Int(py=blk["window"]) == 0 else Int(py=blk["window"])
            var cache_n = b * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
            var pk = _mutable(st.k_cache, [cache_n], keep, copies, "k_cache")
            var pv = _mutable(st.v_cache, [cache_n], keep, copies, "v_cache")
            _ = py_call(e["transformer_decode_session_export_state"], sessions[i], py_ints([pk, pv]))
            py_setattr(st, "cached_tokens", cached)
            _ = out  # keep the Python object alive through the native call above
        _writeback(copies)
    except error:
        if persist:
            py_setitem(fields, "_resident_sessions", PythonObject(None))
            py_setitem(fields, "_resident_busy", False)
        if lm is not None:
            _resident_close(e, lm, sessions)
        raise error
    if persist:
        py_setitem(fields, "_resident_sessions", Python.tuple(key, lm, sessions))
        py_setitem(fields, "_resident_busy", False)
    else:
        _resident_close(e, lm, sessions)
    _ = keep^
    return out


def _run(model: PythonObject, ids: PythonObject, state: PythonObject, remote_step: Bool = False) raises -> PythonObject:
    var l = Int(py=ids.shape[1])
    _state(model, state, Int(py=ids.shape[0]), l)
    var parallel = py_get(model._native_entries, "parallel_call")
    try:
        if parallel is not None:
            return py_call(parallel, "run", Python.list(model, ids, state, remote_step))
        if _resident_forward_ok(model, state):
            var logits = _resident_forward(model, ids, state)
            py_setattr(state, "positions", PythonObject(Int(py=state.positions) + l))
            return logits
        var x = causal_model_embedding(model, ids)
        for i in range(len(model._blocks)):  # driver-loop: over ordered model layers
            x = causal_layer_run(model, i, x, state.layers[i])
        var out = causal_model_head(model, x)
        py_setattr(state, "positions", PythonObject(Int(py=state.positions) + l))
        return out
    except error:
        py_setattr(state, "_usable", PythonObject(False))
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
    var parallel = py_get(model._native_entries, "parallel_call")
    try:
        out = _run(model, ids, state, one)
    except error:
        if supplied_state is None and parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, state))
            except cleanup_error:
                raise Error("CausalLM: forward failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if supplied_state is None and parallel is not None:
        _ = py_call(parallel, "release_layers", Python.list(model, state))
    if one:
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        py_setattr(out, "shape", py_call(g[].bi.tuple, py_ints([Int(py=ids.shape[0]), Int(py=model.vocab_size)])))
        py_setattr(out, "strides", py_call(g[].bi.tuple, py_ints([Int(py=model.vocab_size) * 4, 4])))
        py_setattr(out, "ndim", PythonObject(2))
    return out


def _resident_close(entries: PythonObject, lm: PythonObject, sessions: PythonObject) raises:
    _ = py_call_typed(entries["causal_lm_session_close"], lm)
    for i in range(len(sessions)):  # driver-loop: over layer decode sessions
        _ = py_call_typed(entries["transformer_decode_session_close"], sessions[i])


def _resident_open_all(
    model: PythonObject, e: PythonObject, state: PythonObject, b: Int, l: Int, int15: Bool,
    mut keep: List[PythonObject],
) raises -> PythonObject:
    """Open the model's resident sessions on `state`: one decode session per
    layer (weights and the layer's caches copied to the device) and the
    model session (embedding, norm, head). Returns (lm, sessions)."""
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var sessions = g[].bi.list()
    var lm = e["causal_lm_session_create"]()
    try:
        for i in range(len(model._blocks)):  # driver-loop: over model layers
            var blk = _block(model, i)
            var st = state.layers[i]
            var a = _transformer_args(model, blk, st, PythonObject(None), PythonObject(None), b, l, int15, keep)
            var open_a = g[].bi.list()
            for j in range(1, 12):  # driver-loop: over layer argument spans
                _ = py_call(open_a.append, a[j])
            for j in range(13, len(a)):  # driver-loop: over layer argument spans
                _ = py_call(open_a.append, a[j])
            var p = _transformer_params(blk, st, b, l)
            var open_p = g[].bi.list()
            for j in range(len(p)):  # driver-loop: over layer shape parameters
                if j != 1:
                    _ = py_call(open_p.append, p[j])
            var ss = e["transformer_decode_session_create"]()
            _ = py_call(sessions.append, ss)
            _ = py_call(e["transformer_decode_session_open_int15" if int15 else "transformer_decode_session_open"], ss, open_a, open_p)
        var top = py_ints([_addr(model._embed, keep), _addr(model._norm, keep)])
        if int15:
            for i in range(3):  # small-loop(int15 head plane triple)
                _ = py_call(top.append, _addr(model._head_int15[i], keep))
        else:
            _ = py_call(top.append, 0 if model._head is model._embed else _addr(model._head, keep))
        var tp = py_ints([b, Int(py=model.vocab_size), Int(py=model.d_model)])
        _ = py_call(tp.append, model.norm_eps)
        _ = py_call(e["causal_lm_session_open_int15" if int15 else "causal_lm_session_open"], lm, top, tp)
    except error:
        _resident_close(e, lm, sessions)
        raise error
    return Python.tuple(lm, sessions)


def _resident_key(model: PythonObject, state: PythonObject, b: Int, int15: Bool) raises -> PythonObject:
    """What a kept resident session set must match: the rows, every layer's
    cache capacity, the head format and the entries it was opened with."""
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var caps = g[].bi.list()
    for i in range(len(state.layers)):  # driver-loop: over model layers
        _ = py_call(caps.append, state.layers[i].max_tokens)
    return Python.tuple(b, py_call(g[].bi.tuple, caps), int15, py_call(g[].bi.id, model._native_entries))


def _resident_load_states(e: PythonObject, sessions: PythonObject, state: PythonObject, mut keep: List[PythonObject]) raises:
    """Each kept layer session takes this call's caches (the state half of
    an open; the weights stay where they are)."""
    for i in range(len(sessions)):  # driver-loop: over layer decode sessions
        var st = state.layers[i]
        _ = py_call(e["transformer_decode_session_load_state"], sessions[i],
                    py_ints([_addr(st.k_cache, keep), _addr(st.v_cache, keep)]), py_ints([Int(py=st.cached_tokens)]))


def causal_lm_resident_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, last_logits: PythonObject, supplied_state: PythonObject) raises -> PythonObject:
    var e = model._native_entries
    if not py_contains(e, "causal_lm_session_run") or model._norm is None:
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
    var keep = List[PythonObject]()
    var out = generation_array([b, n], "i")
    # The resident sessions persist under the model handle (lane
    # migrate-neural-2: every call re-uploaded every layer's weights). A kept
    # set is reused when its rows and capacities match; the caches of this
    # call's state are loaded into it. The model's weights never change after
    # construction; once `parameters()` has handed the live arrays out, the
    # sessions are opened per call again, so an in-place edit is observed as
    # before. A concurrent call, or a binding without load_state, opens its
    # own set.
    var fields = py_call(g[].bi.object.__getattribute__, model, "__dict__")
    var persist = (py_contains(e, "transformer_decode_session_load_state")
                   and not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                   and not Bool(py=py_get(fields, "_resident_busy", PythonObject(False))))
    var key = _resident_key(model, state, b, int15)
    var kept = py_get(fields, "_resident_sessions") if persist else PythonObject(None)
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    if persist:
        py_setitem(fields, "_resident_busy", True)
    try:
        if kept is not None and Bool(py=kept[0] == key):
            lm = kept[1]
            sessions = kept[2]
            _resident_load_states(e, sessions, state, keep)
        else:
            if kept is not None:
                py_setitem(fields, "_resident_sessions", PythonObject(None))
                _resident_close(e, kept[1], kept[2])
            var opened = _resident_open_all(model, e, state, b, l, int15, keep)
            lm = opened[0]
            sessions = opened[1]
        _ = py_call(e["causal_lm_session_run"], lm, sessions, py_ints([_addr(ids, keep), _addr(out, keep), _addr(last_logits, keep)]), py_ints([l, n, 0]))
        _ = out  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
    except error:
        if persist:
            py_setitem(fields, "_resident_sessions", PythonObject(None))
            py_setitem(fields, "_resident_busy", False)
        if lm is not None:
            _resident_close(e, lm, sessions)
        raise error
    if persist:
        py_setitem(fields, "_resident_sessions", Python.tuple(key, lm, sessions))
        py_setitem(fields, "_resident_busy", False)
    else:
        _resident_close(e, lm, sessions)
    var result = generation_array([b, l + n], "i")
    var dst = i32_ptr(Int(py=result._addr))
    _ = result  # keep the Python object alive through the native call above
    var prompt = i32_ptr(Int(py=ids._addr))
    _ = ids  # keep the Python object alive through the native call above
    var generated = i32_ptr(Int(py=out._addr))
    _ = out  # keep the Python object alive through the native call above
    causal_lm_join_resident(dst, prompt, generated, b, l, n)
    _ = result  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
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
        return causal_lm_rows_done(self.finished)


def _options(mut s: GenerationHandle, options: PythonObject) raises:
    # The Python veneer forwards these scalar objects unchanged. Explicit
    # sampling requires an explicit seed; greedy consumes no RNG state.
    if len(options) != 7:
        raise Error("generation: expected greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(options[0]) is not g[].bi.bool:
        raise Error("generation: greedy must be bool")
    s.greedy = Bool(py=options[0])
    if not s.greedy and options[1] is None:
        raise Error("generation: greedy=True is the default decoding; greedy=False (sampling) requires an explicit seed")
    if options[1] is not None:
        if py_type(options[1]) is not g[].bi.int or Bool(py=options[1] < 0) or Bool(py=options[1] >= py_call(g[].bi.int, "18446744073709551616")):
            raise Error("generation: seed must be an integer in [0, 2**64)")
        var halves = py_call(g[].bi.divmod, options[1], py_call(g[].bi.int, "4294967296"))
        s.seed = (UInt64(Int(py=halves[0])) << 32) | UInt64(Int(py=halves[1]))
    s.temperature = Float32(Float64(py=options[2]))
    s.top_k = Int(py=options[3])
    s.top_p = Float64(py=options[4])
    var v = Int(py=s.model.vocab_size)
    if s.temperature <= 0 or not isfinite(s.temperature) or s.top_k < 0 or s.top_k > v or s.top_p <= 0 or s.top_p > 1 or not isfinite(s.top_p):
        raise Error("generation: invalid temperature, top_k or top_p")
    if options[5] is not None:
        var values = options[5]
        if py_type(values) is g[].bi.int:
            values = py_ints([Int(py=values)])
        for i in range(len(values)):  # small-loop(eos token id arguments)
            if py_type(values[i]) is not g[].bi.int:
                raise Error("generation: eos_token_id must be integer or sequence of integers")
            var id = Int(py=values[i])
            if id < 0 or id >= v:
                raise Error("generation: EOS id is outside vocabulary")
            s.eos.append(id)
    s.pad = (s.eos[0] if len(s.eos) else 0) if options[6] is None else Int(py=options[6])
    if s.pad < 0 or s.pad >= v:
        raise Error("generation: pad id is outside vocabulary")


def causal_lm_stream_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var n = _public_int(maximum, "generation: max_new_tokens must be an integer")
    if n < 0:
        raise Error("generation: max_new_tokens must be nonnegative")
    var prompt = generation_ids(value, Int(py=model.vocab_size))
    var handle = GenerationHandle(model, prompt, n)
    try:
        _options(handle, options)
    except error:
        var parallel = py_get(model._native_entries, "parallel_call")
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, handle.state))
            except cleanup_error:
                raise Error("generation: option validation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var stream = py_call(g[].bi.object.__new__, g[].stream_cls)
    py_setattr(stream, "_native", PythonObject(alloc=handle^))
    py_setattr(stream, "_runtime", model._runtime)
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
        while wrote < capacity and not s.done():  # driver-loop: over decode steps, one model forward each
            var l = Int(py=s.current.shape[1])
            var logits = _run(s.model, s.current, s.state, s.produced > 0)
            var raw = f32_ptr(Int(py=logits._addr))
            _ = logits  # keep the Python object alive through the native call above
            var next = generation_array([b, 1], "i")
            var dst = i32_ptr(Int(py=next._addr))
            _ = next  # keep the Python object alive through the native call above
            causal_lm_pick_step(raw, dst, b, l, v, s.finished, s.lengths, s.eos, s.pad, s.greedy,
                                s.temperature, s.top_k, s.top_p, s.seed, s.produced, tokens)
            _ = next  # keep the Python object alive through the native call above
            _ = logits  # keep the Python object alive through the native call above
            # `raw` points into `logits`: keep it alive through the pick (Mojo
            # destroys a local at its last use, and nothing else owns it).
            _ = logits^
            s.current = next
            s.produced += 1
            wrote += 1
        var out = generation_array([b, wrote], "i")
        if wrote == 0:
            s.busy = False
            return out
        var dst = i32_ptr(Int(py=out._addr))
        _ = out  # keep the Python object alive through the native call above
        causal_lm_tokens_row_major(tokens, dst, b, wrote)
        _ = out  # keep the Python object alive through the native call above
        s.busy = False
        return out
    except error:
        s.busy = False
        s.usable = False
        raise error


def _decode_chunk(mut s: GenerationHandle, ids: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var n = Int(py=ids.shape[1])
    var raw = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
    var values = py_ints(causal_lm_drop_eos(raw, n, s.eos))
    _ = py_call(s.pending_bytes.extend, py_call(_native_function(s.tokenizer._m, "vocab_decode_public"), s.tokenizer, values, PythonObject(True), PythonObject("replace")))
    var size = len(s.pending_bytes)
    # The CPython array constructor copies the pending bytes to native storage.
    var pending = py_call(g[].arrays, "B", s.pending_bytes)
    var cut = causal_lm_utf8_cut(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=pending.buffer_info()[0])), size, s.done())
    _ = pending  # keep the Python object alive through the native call above
    _ = pending^
    var prefix = s.pending_bytes[py_call(g[].bi.slice, PythonObject(0), PythonObject(cut))]
    var remaining = s.pending_bytes[py_call(g[].bi.slice, PythonObject(cut), PythonObject(None))]
    s.pending_bytes = remaining
    return py_call(prefix.decode, "utf-8", "replace")


def causal_lm_read_binding(stream: PythonObject, count: PythonObject) raises -> PythonObject:
    var owner = stream._native.downcast_value_ptr[GenerationHandle]()
    var ids = _read(owner[], _public_int(count, "generation: max_tokens must be an integer"))
    if owner[].tokenizer is not None:
        return _decode_chunk(owner[], ids)
    return ids


def causal_lm_info_binding(stream: PythonObject) raises -> PythonObject:
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    var result = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.dict()
    py_setitem(result, "done", PythonObject(s[].done()))
    py_setitem(result, "generated_tokens", PythonObject(s[].produced))
    py_setitem(result, "lengths", py_ints(s[].lengths))
    py_setitem(result, "usable", PythonObject(s[].usable))
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
    _ = result  # keep the Python object alive through the native call above
    var prompt = i32_ptr(Int(py=s.prompt._addr))
    var tail = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=continuation._addr))
    _ = continuation  # keep the Python object alive through the native call above
    causal_lm_join_continuation(dest, prompt, tail, b, l, n)
    _ = result  # keep the Python object alive through the native call above
    _ = continuation  # keep the Python object alive through the native call above
    _ = continuation^
    return result


def causal_lm_generate_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_stream_binding(model, value, maximum, options)
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    var parallel = py_get(model._native_entries, "parallel_call")
    var result = PythonObject(None)
    try:
        result = _generate_complete(model, value, maximum, s[])
    except error:
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, s[].state))
            except cleanup_error:
                raise Error("CausalLM: generation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if parallel is not None:
        _ = py_call(parallel, "release_layers", Python.list(model, s[].state))
    return result


def causal_lm_text_stream_binding(model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var encode = _native_function(tokenizer._m, "vocab_encode_public")
    var ids = py_call(encode, tokenizer, text, PythonObject(True), PythonObject(False), PythonObject(False))
    var prompt = generation_array([1, len(ids)], "i")
    var ptr = i32_ptr(Int(py=prompt._addr))
    _ = prompt  # keep the Python object alive through the native call above
    # The CPython array constructor unpacks the id list to native int64 storage.
    var packed = py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].arrays, "q", ids)
    causal_lm_store_ids(ptr, MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=packed.buffer_info()[0])), len(ids))
    _ = packed  # keep the Python object alive through the native call above
    _ = prompt  # keep the Python object alive through the native call above
    _ = packed^
    var stream = causal_lm_stream_binding(model, prompt, maximum, options)
    var owner = stream._native.downcast_value_ptr[GenerationHandle]()
    owner[].tokenizer = tokenizer
    return stream


def causal_lm_generate_text_binding(model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_text_stream_binding(model, tokenizer, text, maximum, options)
    var owner = stream._native.downcast_value_ptr[GenerationHandle]()
    var parallel = py_get(model._native_entries, "parallel_call")
    var result = PythonObject(None)
    try:
        var ids = _read(owner[], max(1, owner[].max_new))
        result = _decode_chunk(owner[], ids)
    except error:
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, owner[].state))
            except cleanup_error:
                raise Error("CausalLM: text generation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if parallel is not None:
        _ = py_call(parallel, "release_layers", Python.list(model, owner[].state))
    return result


def _array_bytes(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        return py_call(value._mv.tobytes, "C")
    return py_call(py_call(g[].bi.memoryview, value).tobytes, "C")


def _hash() -> Sha256:
    return Sha256()


def _hash_scalar(mut hash: Sha256, value: PythonObject) raises:
    sha256_update_text(hash, value)
    hash.update_text(String("\n"))


def _fingerprint(model: PythonObject) raises -> PythonObject:
    var h = _hash()
    _hash_scalar(h, model.kind)
    _hash_scalar(h, model.numeric_profile)
    _hash_scalar(h, model.norm_eps)
    _hash_scalar(h, model.max_positions)
    _hash_scalar(h, model._embed.shape)
    sha256_update(h, _array_bytes(model._embed))
    if model._norm is not None:
        sha256_update(h, _array_bytes(model._norm))
    sha256_update(h, _array_bytes(model._head))
    for i in range(len(model._blocks)):  # driver-loop: over model layers
        var blk = _block(model, i)
        if String(model.kind) == "transformer":
            _hash_scalar(h, blk["_opts_tail"])
            _hash_scalar(h, blk["n_heads"])
            _hash_scalar(h, blk["n_kv_heads"])
            _hash_scalar(h, blk["window"])
            for j in range(len(blk["_wopt"])):  # driver-loop: over optional block weight slots
                if blk["_wopt"][j] is not None:
                    sha256_update(h, _array_bytes(blk["_wopt"][j]))
        elif String(model.kind) == "mamba2":
            _hash_scalar(h, blk["dt_limit"])
        else:
            _hash_scalar(h, model._native_entries["mamba1_profile"])
        for j in range(len(blk["_w"])):  # driver-loop: over block weight slots
            if blk["_w"][j] is not None:
                _hash_scalar(h, blk["_w"][j].shape)
                sha256_update(h, _array_bytes(blk["_w"][j]))
        if model._head_int15 is not None:
            for j in range(len(blk["_int15"])):  # driver-loop: over packed block weight slots
                if blk["_int15"][j] is not None:
                    for k in range(3):  # small-loop(int15 plane triple)
                        sha256_update(h, _array_bytes(blk["_int15"][j][k]))
    if model._head_int15 is not None:
        for k in range(3):  # small-loop(int15 head plane triple)
            sha256_update(h, _array_bytes(model._head_int15[k]))
    return PythonObject(h.hexdigest())


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
    for name in [String("schema"), String("model"), String("options"), String("maximum"),  # small-loop(checkpoint scalar fields)
                 String("produced"), String("positions"), String("lengths"), String("finished"), String("text"), String("tokenizer")]:
        _hash_scalar(h, checkpoint[name])
    sha256_update(h, checkpoint["pending_bytes"])
    sha256_update(h, _array_bytes(checkpoint["prompt"]))
    sha256_update(h, _array_bytes(checkpoint["current"]))
    for i in range(len(checkpoint["layers"])):  # driver-loop: over checkpoint layers
        var layer = checkpoint["layers"][i]
        var names = py_call(g[].bi.sorted, layer.keys())
        for j in range(len(names)):  # driver-loop: over layer state field names
            _hash_scalar(h, names[j])
            var value = layer[names[j]]
            if py_type(value) is g[].array_cls:
                _hash_scalar(h, value.shape)
                _hash_scalar(h, value.dtype)
                sha256_update(h, _array_bytes(value))
            else:
                _hash_scalar(h, value)
    return PythonObject(h.hexdigest())


def causal_lm_snapshot_binding(stream: PythonObject) raises -> PythonObject:
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    if s[].busy or not s[].usable:
        raise Error("generation: cannot checkpoint a busy or failed stream")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    py_setitem(result, "schema", PythonObject("mojolearn.generation.v1"))
    py_setitem(result, "model", _fingerprint(s[].model))
    py_setitem(result, "maximum", PythonObject(s[].max_new))
    py_setitem(result, "produced", PythonObject(s[].produced))
    py_setitem(result, "positions", s[].state.positions)
    py_setitem(result, "prompt", generation_copy(s[].prompt))
    py_setitem(result, "current", generation_copy(s[].current))
    py_setitem(result, "lengths", py_ints(s[].lengths))
    py_setitem(result, "finished", py_call(g[].bi.list, py_call(g[].bi.map, g[].bi.bool, py_ints(causal_lm_flags_as_ints(s[].finished)))))
    py_setitem(result, "text", PythonObject(s[].tokenizer is not None))
    py_setitem(result, "tokenizer", PythonObject(None))
    if s[].tokenizer is not None:
        py_setitem(result, "tokenizer", py_call(_native_function(s[].tokenizer._m, "vocab_fingerprint"), s[].tokenizer))
    py_setitem(result, "pending_bytes", py_call(g[].bi.bytes, s[].pending_bytes))
    var options = g[].bi.list()
    _ = py_call(options.append, s[].greedy)
    # Decimal conversion avoids truncating UInt64 through the signed Python
    # bridge. Every unsigned seed is representable by Python's native int.
    _ = py_call(options.append, py_call(g[].bi.int, String(s[].seed)))
    _ = py_call(options.append, Float64(s[].temperature))
    _ = py_call(options.append, s[].top_k)
    _ = py_call(options.append, s[].top_p)
    _ = py_call(options.append, py_ints(s[].eos))
    _ = py_call(options.append, s[].pad)
    py_setitem(result, "options", options)
    var layers = g[].bi.list()
    var kind = String(s[].model.kind)
    var profile = Int(py=s[].model._native_entries["mamba1_profile"]) if kind == "mamba1" else 1
    var arrays = _state_arrays(kind, profile)
    for i in range(len(s[].state.layers)):  # driver-loop: over model layer states
        var st = s[].state.layers[i]
        var parallel = py_get(s[].model._native_entries, "parallel_call")
        if parallel is not None:
            st = py_call(parallel, "state_snapshot", Python.list(st))
        var entry = g[].bi.dict()
        for j in range(len(arrays)):  # driver-loop: over layer state arrays
            py_setitem(entry, arrays[j], generation_copy(py_call(g[].bi.getattr, st, arrays[j])))
        if kind == "transformer":
            py_setitem(entry, "cached_tokens", st.cached_tokens)
        elif kind == "mamba1":
            py_setitem(entry, "absolute_position", st.absolute_position)
            py_setitem(entry, "profile_id", st.profile_id)
        else:
            py_setitem(entry, "buffered_tokens", st.buffered_tokens)
        _ = py_call(layers.append, entry)
    py_setitem(result, "layers", layers)
    py_setitem(result, "checksum", _snapshot_hash(result))
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
        var fingerprint = py_call(_native_function(tokenizer._m, "vocab_fingerprint"), tokenizer)
        if not Bool(py=checkpoint["tokenizer"] == fingerprint):
            raise Error("generation: checkpoint tokenizer differs")
    var stream = causal_lm_stream_binding(model, checkpoint["prompt"], checkpoint["maximum"], checkpoint["options"])
    var s = stream._native.downcast_value_ptr[GenerationHandle]()
    try:
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
        py_setattr(s[].state, "positions", PythonObject(position))
        # The CPython array constructor unpacks the per-row lists to native storage.
        var packed_lengths = py_call(g[].arrays, "q", checkpoint["lengths"])
        var packed_finished = py_call(g[].arrays, "b", py_call(g[].bi.map, g[].bi.bool, checkpoint["finished"]))
        causal_lm_restore_rows(
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=packed_lengths.buffer_info()[0])),
            MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(py=packed_finished.buffer_info()[0])),
            b, s[].produced, s[].lengths, s[].finished,
        )
        _ = packed_lengths  # keep the Python object alive through the native call above
        _ = packed_finished  # keep the Python object alive through the native call above
        _ = packed_lengths^
        _ = packed_finished^
        if len(checkpoint["layers"]) != len(s[].state.layers):
            raise Error("generation: checkpoint layer count differs")
        var kind = String(model.kind)
        var profile = Int(py=model._native_entries["mamba1_profile"]) if kind == "mamba1" else 1
        var arrays = _state_arrays(kind, profile)
        for i in range(len(s[].state.layers)):  # driver-loop: over model layer states
            var dest = s[].state.layers[i]
            var parallel = py_get(model._native_entries, "parallel_call")
            if parallel is not None:
                dest = py_call(parallel, "state_snapshot", Python.list(dest))
            var source = checkpoint["layers"][i]
            for j in range(len(arrays)):  # driver-loop: over layer state arrays
                var expected = py_call(g[].bi.getattr, dest, arrays[j])
                var supplied = source[arrays[j]]
                if py_type(supplied) is not g[].array_cls or String(supplied.dtype) != "<f4" or not Bool(py=supplied.shape == expected.shape):
                    raise Error("generation: checkpoint state array shape or dtype differs")
                _ = py_call(g[].bi.setattr, dest, arrays[j], generation_copy(supplied))
            if kind == "transformer":
                if Int(py=source["cached_tokens"]) != position:
                    raise Error("generation: checkpoint layer cache position differs")
                py_setattr(dest, "cached_tokens", PythonObject(position))
            elif kind == "mamba1":
                if Int(py=source["profile_id"]) != profile or Int(py=source["absolute_position"]) != position:
                    raise Error("generation: checkpoint Mamba profile/position differs")
                py_setattr(dest, "absolute_position", PythonObject(position))
            else:
                if Int(py=source["buffered_tokens"]) != position % 256:
                    raise Error("generation: checkpoint Mamba2 buffered length differs")
                py_setattr(dest, "buffered_tokens", source["buffered_tokens"])
            if parallel is not None:
                _ = py_call(parallel, "state_publish", Python.list(s[].state.layers[i], dest))
        s[].tokenizer = tokenizer if Bool(py=checkpoint["text"]) else PythonObject(None)
        s[].pending_bytes = py_call(g[].bi.bytearray, checkpoint["pending_bytes"])
        return stream
    except error:
        # These caches belong only to this attempted restore. Existing live
        # streams retain their own worker tokens and are never touched.
        var parallel = py_get(model._native_entries, "parallel_call")
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(model, s[].state))
            except cleanup_error:
                raise Error("generation: checkpoint restore failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error


def causal_state_init_binding(state: PythonObject, values: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    py_setattr(state, "batch_size", py_call(bi.int, values[0]))
    py_setattr(state, "max_tokens", py_call(bi.int, values[1]))
    py_setattr(state, "layers", py_call(bi.list, values[2]))
    py_setattr(state, "positions", PythonObject(0))
    py_setattr(state, "_owner", values[3])
    py_setattr(state, "_usable", PythonObject(True))
    return PythonObject(None)


def _causal_owner_unusable(args_ptr: PyObjectPtr) -> Bool:
    """True when the call's model is closed or lost its worker state.

    A ParallelCausalLM refusal raised through `py_call` (the parallel entry)
    arrives here as a bare message: CPython's exception class does not cross
    that boundary. A closed model or a lost state is a RuntimeError, so the
    door classes it by the owner's state, not by the message text."""
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) == 0:
            return False
        var bi = Python.import_module("builtins")
        var owner = args[0]
        if Bool(py=py_call(bi.getattr, owner, PythonObject("_closed"), PythonObject(False))):
            return True
        return Bool(py=py_call(bi.getattr, owner, PythonObject("_failed"), PythonObject(False)))
    except:
        return False


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
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if _causal_owner_unusable(args_ptr):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
