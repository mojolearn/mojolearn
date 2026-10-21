# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native Mamba public operations: Mamba3Block and the Mamba decode sessions.

`Mamba3Block.forward` (plain, carried state and ragged `lengths=`), `step`,
`allocate_state`, the owned weight snapshot (`install_owned_weights`,
`release_owned_weights`, `export_owned_weights`, `session_info`), pickling,
`Mamba3State.set_input_states`, and every method of the Mamba-1/2/3 resident
decode sessions are each ONE call into this module (`mamba_native_call`). The
compiled mamba entries reached here are the ones the retired Python drivers in
`python/mojolearn/_mamba_impl.py` called, with the same addresses in the same
order and the same parameters, so no output bit moves. No Python callable of the
package is invoked: only builtins, buffer exporters and compiled entries.

Lifetime (docs/MODULAR_ASKS.md, PythonObject attribute assignment): where a
result is a new object the caller keeps (y, the Mamba3 reports, the prefill
session handle), this module RETURNS it and the Python glue assigns it. The
session object itself is created here because a decode session is opened by
one native call; its state marker is the immortal `True`, never the session.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.generation_buffers import (
    GENERATION_BOUNDARY, generation_initialize_binding, generation_array, generation_address, py_ints,
)
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.causal_lm_construct import _model_bindings, _model_block, _model_f32
from bindings.neural_block_public import neural_training_entries, _neural_input, _neural_lengths
from bindings.neural_prefill_native import _prefill_mamba3_reports
from bindings.samba_inference_native import _m3_shapes, _m3_names, _m3_state_call
from core.neural_block_public_impl import neural_copy_row_prefixes
from mamba.impl.mamba_session_impl import mamba_copy_f32, mamba_host_copy


# ---------------------------------------------------------------- Mamba3Block


def _m3_entries(block: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The binding the retired `Mamba3Block._extension()` returned: the shipped
    # host binding for the CPU inference class, the block's tier otherwise.
    var bi = Python.import_module("builtins")
    if Bool(py=bi.isinstance(block, context["block_classes"]["cpu"]["mamba3"])):
        return _model_bindings(PythonObject("cpu"), "mamba3", context, PythonObject("identical"))["mamba3"]
    return neural_training_entries(fields, context)


def _m3_canonical(entries: PythonObject) raises -> PythonObject:
    if Bool(py=Python.import_module("builtins").hasattr(entries, "canonical")):
        return entries.canonical
    return entries


def _m3_allocate(block: PythonObject, fields: PythonObject, batch: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var b = Int(py=bi.int(batch))
    if b < 1:
        raise Error("Mamba3Block: batch_size must be positive, got " + String(b))
    var state = bi.object.__new__(context["mamba3_state_class"])
    var slots = state.__dict__
    var shapes = _m3_shapes(b, Int(py=fields["nheads"]))
    var names = _m3_names()
    for j in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
        slots[names[j]] = generation_array(shapes[j])
    slots["buffered_tokens"] = 0
    slots["pending"] = False
    return state


def _m3_fresh(block: PythonObject, fields: PythonObject, x: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # The retired `_call`: the discard-only prefill where the binding has it,
    # else the carried-state entry from a fresh zero state.
    if Bool(py=Python.import_module("builtins").hasattr(entries, "mamba3_forward_fresh")):
        return _prefill_mamba3_reports(block, x, entries)
    var state = _m3_allocate(block, fields, x.shape[0], context)
    return _m3_state_call(block, x, state, False, entries)


def mamba3_forward(
    block: PythonObject, input: PythonObject, state: PythonObject, lengths: PythonObject, step: Bool,
    context: PythonObject,
) raises -> PythonObject:
    """`(y, h_last, k_last, v_last, theta_last)` of one forward or step."""
    var fields = block.__dict__
    var d = Int(py=fields["d_model"])
    if step and state is None:
        raise Error("Mamba3Block.step: state is required (a decode step continues a sequence; allocate_state(B) makes the fresh one)")
    var x = _neural_input(input, d, step, context)
    var entries = _m3_entries(block, fields, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    if lengths is not None:
        if state is not None:
            raise Error("Mamba3Block.forward: lengths= (a ragged, right-padded batch) cannot carry a state; run the ragged prefill with state=None, or continue each sequence as its own batch")
        var sizes = _neural_lengths(lengths, b, l, context)
        var padded = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=padded._addr), Int(py=x._addr), sizes, b, l, d)
        var ran = _m3_fresh(block, fields, padded, entries, context)
        var result = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=result._addr), Int(py=ran[0]._addr), sizes, b, l, d)
        return Python.tuple(result, ran[1], ran[2], ran[3], ran[4])
    if state is None:
        return _m3_fresh(block, fields, x, entries, context)
    return _m3_state_call(block, x, state, step, entries)


def mamba3_allocate(block: PythonObject, batch: PythonObject, context: PythonObject) raises -> PythonObject:
    return _m3_allocate(block, block.__dict__, batch, context)


def _m3_checked_weights(block: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The current weights through the constructor's own native checks, as the
    # retired `type(self)(...)` re-validation did, without a Python constructor.
    var bi = Python.import_module("builtins")
    var plan = bi.object.__new__(context["config_context"]["plan_class"])
    plan.kind = PythonObject("mamba3")
    plan.d_model = fields["d_model"]
    plan.block_kwargs = bi.dict()
    var named = bi.dict()
    var names = block._W_NAMES
    var weights = fields["_w"]
    for i in range(len(names)):  # driver-loop: over the fixed block weight registry
        named[names[i]] = weights[i]
    var route = fields["_native_route"]
    var checked = _model_block(named, plan, PythonObject("fp32_v1"), route,
                               _model_bindings(route, "mamba3", context, fields.get("numeric_mode")), context)
    return checked.__dict__["_w"]


def mamba3_install_owned(block: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(prefill session, binding, generation)`; the glue assigns all three."""
    var bi = Python.import_module("builtins")
    var fields = block.__dict__
    var entries = _m3_entries(block, fields, context)
    if not Bool(py=bi.hasattr(entries, "mamba3_owned_weights_enabled")) or not Bool(py=_native_function(entries, "mamba3_owned_weights_enabled")()):
        raise Error("RuntimeError: Mamba3 owned weights require the NN40 IDENTICAL arm")
    var weights = _m3_checked_weights(block, fields, context)
    var canonical = _m3_canonical(entries)
    var session = fields.get("_prefill_session")
    if session is None or fields.get("_prefill_binding") is not canonical:
        session = _native_function(entries, "mamba3_prefill_session_create")()
    var keep = List[PythonObject]()
    var addresses = py_ints([])
    for i in range(len(weights)):  # driver-loop: over the fixed owned-weight registry
        addresses.append(generation_address(weights[i], keep))
    var generation = _native_function(entries, "mamba3_prefill_session_install_weights")(session, addresses, fields["d_model"])
    _ = keep^
    return Python.tuple(session, canonical, generation)


def mamba3_release_owned(block: PythonObject) raises -> PythonObject:
    var fields = block.__dict__
    var session = fields.get("_prefill_session")
    var binding = fields.get("_prefill_binding")
    if session is not None and binding is not None:
        _native_function(binding, "mamba3_prefill_session_close")(session)
    fields["_prefill_session"] = PythonObject(None)
    fields["_prefill_binding"] = PythonObject(None)
    fields["_owned_generation"] = PythonObject(None)
    fields["_owned_restore_pending"] = PythonObject(False)
    return PythonObject(None)


def mamba3_session_info(block: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = block.__dict__
    var session = fields.get("_prefill_session")
    var binding = fields.get("_prefill_binding")
    if session is None or binding is None or not Bool(py=bi.hasattr(binding, "mamba3_prefill_session_info")):
        return PythonObject(None)
    var values = _native_function(binding, "mamba3_prefill_session_info")(session)
    var keys = [String("weight_uploads"), String("weight_recopies"), String("weight_reuses"),
                String("backward_reuses"), String("backward_recomputes"), String("stages_held")]
    var result = bi.dict()
    for i in range(len(keys)):  # small-loop(six named session counters)
        result[keys[i]] = Int(py=values[i])
    return result


def mamba3_export_owned(block: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = block.__dict__
    var generation = fields.get("_owned_generation")
    if generation is None:
        raise Error("RuntimeError: Mamba3 has no installed owned snapshot")
    var dm = Int(py=fields["d_model"])
    var di = Int(py=fields["d_inner"])
    var nh = Int(py=fields["nheads"])
    var dip = Int(py=fields["d_in_proj"])
    var shapes = List[List[Int]]()
    shapes.append([dm])
    shapes.append([dip, dm])
    shapes.append([nh])
    shapes.append([128])
    shapes.append([128])
    shapes.append([nh, 128])
    shapes.append([nh, 128])
    shapes.append([nh])
    shapes.append([dm, di])
    var arrays = bi.list()
    var keep = List[PythonObject]()
    var addresses = py_ints([])
    for i in range(len(shapes)):  # driver-loop: over the fixed parameter buffer list
        var array = generation_array(shapes[i])
        arrays.append(array)
        addresses.append(generation_address(array, keep))
    _native_function(fields["_prefill_binding"], "mamba3_prefill_session_export_weights")(fields["_prefill_session"], addresses, generation)
    var result = bi.dict()
    var names = block._W_NAMES
    for i in range(len(names)):  # driver-loop: over named parameter slots
        result[names[i]] = arrays[i]
    _ = keep^
    return result


def mamba3_getstate(block: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = block.__dict__
    var result = bi.dict(fields)
    if fields.get("_owned_generation") is not None:
        var snapshot = mamba3_export_owned(block)
        var names = block._W_NAMES
        var weights = bi.list()
        for i in range(len(names)):  # driver-loop: over the fixed parameter name list
            weights.append(snapshot[names[i]])
        result["_w"] = weights
        result["_owned_restore_pending"] = True
    result["_owned_generation"] = PythonObject(None)
    _ = result.pop("_prefill_session", PythonObject(None))
    _ = result.pop("_prefill_binding", PythonObject(None))
    return result


def mamba3_setstate(block: PythonObject, state: PythonObject) raises -> PythonObject:
    var fields = block.__dict__
    fields.update(state)
    fields["_prefill_session"] = PythonObject(None)
    fields["_prefill_binding"] = PythonObject(None)
    fields["_owned_generation"] = PythonObject(None)
    return PythonObject(None)


def _shape_of(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].bi.type(value) is g[].array_cls:
        return value.shape
    return g[].bi.memoryview(value).shape


def mamba3_set_input_states(
    state: PythonObject, theta: PythonObject, h: PythonObject, k: PythonObject, v: PythonObject,
    context: PythonObject,
) raises -> PythonObject:
    var slots = state.__dict__
    var names = [String("theta"), String("h"), String("k"), String("v")]
    var targets = [String("theta"), String("h"), String("pending_k"), String("pending_v")]
    var sources = Python.list(theta, h, k, v)
    for i in range(4):  # driver-loop: over the four named continuation pieces
        var source = _model_f32(sources[i], names[i], context)
        var target = slots[targets[i]]
        var want = _shape_of(target)
        if not Bool(py=source.shape == want):
            raise Error("Mamba3State.set_input_states: " + names[i] + " has shape " + String(source.shape) + ", want " + String(want))
        var dims = List[Int]()
        for axis in range(len(want)):  # small-loop(state piece shape dimension words)
            dims.append(Int(py=want[axis]))
        var keep = List[PythonObject]()
        var copies = List[PythonObject]()
        var address = _mutable(target, dims, keep, copies, names[i])
        mamba_copy_f32(address, Int(py=source._addr), Int(py=source.size))
        _writeback(copies)
        _ = keep^
    slots["pending"] = True
    return PythonObject(None)


# ---------------------------------------------------------------- decode sessions


struct MambaDecodeHandle(Movable, Writable):
    """The native owner of one open Mamba decode session.

    Device arm: the compiled `<kind>_session_*` handle keeps the weights and the
    state resident. Host arm: typed copies of the weights and the state, stepped
    through `<kind>_decode_step` in the per-call address order."""
    var kind: Int
    var binding: PythonObject
    var device: PythonObject
    var open: Bool
    var b: Int
    var d_model: Int
    var q: Int
    var pending: Bool
    var position: Int
    var profile: Int
    var dt_limit: PythonObject
    var names: List[String]
    var dims: List[List[Int]]
    var report_dims: List[List[Int]]
    var weights: List[List[Float32]]
    var parts: List[List[Float32]]

    def __init__(out self, kind: Int, binding: PythonObject):
        self.kind = kind
        self.binding = binding
        self.device = PythonObject(None)
        self.open = False
        self.b = 0
        self.d_model = 0
        self.q = 0
        self.pending = False
        self.position = 0
        self.profile = 1
        self.dt_limit = PythonObject(None)
        self.names = List[String]()
        self.dims = List[List[Int]]()
        self.report_dims = List[List[Int]]()
        self.weights = List[List[Float32]]()
        self.parts = List[List[Float32]]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MambaDecodeHandle(kind=", self.kind, ", open=", self.open, ", rows=", self.b, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _kind_name(kind: Int) -> String:
    if kind == 1:
        return "mamba1"
    if kind == 2:
        return "mamba2"
    return "mamba3"


def _session_name(kind: Int) -> String:
    return "Mamba" + String(kind) + "DecodeSession"


def _count(dims: List[Int]) -> Int:
    var n = 1
    for i in range(len(dims)):  # small-loop(state piece shape dimension words)
        n *= dims[i]
    return n


def _m1_check(slots: PythonObject, profile: Int, what: String) raises:
    # The retired `_m1_affine_buffers` metadata checks; the arithmetic is Mojo's.
    if Int(py=slots.get("profile_id", PythonObject(1))) != profile:
        raise Error(what + ": checkpoint numerical profile differs from the binding")
    if profile != 2:
        return
    if not Bool(py=slots.get("_nn34_usable", PythonObject(True))):
        raise Error(what + ": NN34 state was lost after a failed call; restore a checkpoint")
    var position = Int(py=slots["absolute_position"])
    if position < 0 or position > (1 << 62) - 32:
        raise Error(what + ": invalid absolute checkpoint position")


def _session_layout(mut handle: MambaDecodeHandle, fields: PythonObject) raises:
    var b = handle.b
    if handle.kind == 1:
        var di = Int(py=fields["d_inner"])
        handle.names = [String("conv_window"), String("h")]
        handle.dims.append([b, di, 4])
        handle.dims.append([b, di, 16])
        if handle.profile == 2:
            handle.names.append("affine_boundary")
            handle.names.append("affine_a")
            handle.names.append("affine_b")
            handle.dims.append([b, di, 16])
            handle.dims.append([b, di, 16, 6])
            handle.dims.append([b, di, 16, 6])
    elif handle.kind == 2:
        var cd = Int(py=fields["conv_dim"])
        var nh = Int(py=fields["nheads"])
        handle.names = [String("conv_window"), String("h"), String("buffer_xbc"), String("buffer_dtraw")]
        handle.dims.append([b, cd, 4])
        handle.dims.append([b, nh, 64, 128])
        handle.dims.append([b, 256, cd])
        handle.dims.append([b, 256, nh])
        handle.report_dims.append([b, nh, 64, 128])
    else:
        var nh = Int(py=fields["nheads"])
        handle.names = _m3_names()
        handle.dims = _m3_shapes(b, nh)
        handle.report_dims.append([b, nh, 64, 128])
        handle.report_dims.append([b, nh, 128])
        handle.report_dims.append([b, nh, 64])
        handle.report_dims.append([b, nh, 32])


def mamba_session_init(session: PythonObject, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = block.__dict__
    var kind_name = String(fields.get("_native_kind", PythonObject("")))
    var kind = 1 if kind_name == "mamba1" else (2 if kind_name == "mamba2" else (3 if kind_name == "mamba3" else 0))
    if kind == 0:
        raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
    var what = _session_name(kind)
    var binding = _model_bindings(fields["_native_route"], kind_name, context, fields.get("numeric_mode"))[kind_name]
    var device_arm = Bool(py=bi.hasattr(binding, kind_name + "_session_create"))
    if not device_arm and not Bool(py=bi.hasattr(binding, kind_name + "_decode_step")):
        raise Error("NotImplementedError: " + what + ": the loaded binding exports neither a resident decode session nor the per-call " + kind_name + "_decode_step entry")
    if state is None:
        raise Error(what + ": state is required (allocate_state)")
    var slots = state.__dict__
    if slots.get("_resident_session") is not None:
        raise Error(what + ": the state is owned by an open resident decode session (its pieces live in the session, "
                    + "on the device or on the host, and the caller's buffers are stale); call sync_state() or close() on the session first")
    var hshape = _shape_of(slots["h"])
    if len(hshape) != (3 if kind == 1 else 4) or Int(py=hshape[0]) < 1:
        raise Error(what + ": state.h has an invalid shape " + String(hshape))
    var handle = MambaDecodeHandle(kind, binding)
    handle.b = Int(py=hshape[0])
    handle.d_model = Int(py=fields["d_model"])
    if kind == 1:
        if Bool(py=bi.hasattr(binding, "mamba1_profile")):
            handle.profile = Int(py=_native_function(binding, "mamba1_profile")())
        _m1_check(slots, handle.profile, what)
        handle.position = Int(py=slots.get("absolute_position", PythonObject(0)))
    elif kind == 2:
        handle.q = Int(py=slots["buffered_tokens"])
        handle.dt_limit = fields["dt_limit"]
    else:
        handle.q = Int(py=slots["buffered_tokens"])
        handle.pending = Bool(py=slots["pending"])
    _session_layout(handle, fields)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = List[Int]()
    for i in range(len(handle.names)):  # driver-loop: over the state's named pieces
        pieces.append(_mutable(slots[handle.names[i]], handle.dims[i], keep, copies, handle.names[i]))
    var weights = fields["_w"]
    if device_arm:
        # The weights, then the state pieces, as the retired session opened them.
        var addresses = py_ints([])
        for i in range(len(weights)):  # driver-loop: over block weight slots
            addresses.append(generation_address(weights[i], keep))
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            addresses.append(pieces[i])
        var params = py_ints([handle.b, handle.d_model])
        if kind == 1:
            if handle.profile == 2:
                params.append(handle.position)
        elif kind == 2:
            params.append(handle.q)
            params.extend(handle.dt_limit)
        else:
            params.append(handle.q)
            params.append(1 if handle.pending else 0)
        handle.device = _native_function(binding, kind_name + "_session_create")()
        _native_function(binding, kind_name + "_session_open")(handle.device, addresses, params)
    else:
        # The host arm owns typed copies, the device arm's upload on the host.
        for i in range(len(weights)):  # driver-loop: over block weight slots
            handle.weights.append(mamba_host_copy(generation_address(weights[i], keep), Int(py=weights[i].size)))
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            handle.parts.append(mamba_host_copy(pieces[i], _count(handle.dims[i])))
    handle.open = True
    _ = keep^
    session._handle = PythonObject(alloc=handle^)
    session._block = block
    session._state = state
    slots["_resident_session"] = PythonObject(True)
    return PythonObject(None)


def mamba_session_open(block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var kind = String(block.__dict__.get("_native_kind", PythonObject("")))
    var classes = context["mamba_session_classes"]
    if not Bool(py=classes.__contains__(kind)):
        raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
    var session = bi.object.__new__(classes[kind])
    _ = mamba_session_init(session, block, state, context)
    return session


def _session_require_open(handle: PythonObject, what: String) raises:
    if not handle.downcast_value_ptr[MambaDecodeHandle]()[].open:
        raise Error(what + ": the session is closed")


def mamba_session_step(session: PythonObject, input: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].kind
    var name = _kind_name(kind)
    var what = _session_name(kind) + ".step"
    _session_require_open(hobj, what)
    var x = _neural_input(input, h[].d_model, True, context)
    var b = h[].b
    if Int(py=x.shape[0]) != b:
        raise Error(what + ": the session holds " + String(b) + " rows, x has B = " + String(x.shape[0]))
    var keep = List[PythonObject]()
    var y = generation_array([b, 1, h[].d_model])
    var reports = bi.list()
    for i in range(len(h[].report_dims)):  # driver-loop: over the fixed report buffers
        reports.append(generation_array(h[].report_dims[i]))
    if h[].device is not None:
        var addresses = py_ints([generation_address(x, keep), generation_address(y, keep)])
        for i in range(len(reports)):  # driver-loop: over the fixed report buffers
            addresses.append(generation_address(reports[i], keep))
        var updated = _native_function(h[].binding, name + "_session_step")(h[].device, addresses)
        if kind != 1:
            h[].q = Int(py=updated)
    else:
        # The per-call `<kind>_decode_step` address order with the session's
        # own copies standing where the caller's buffers would: x, weights,
        # state pieces, y, then (Mamba-1) the affine pieces or (Mamba-2/3) the reports.
        var addresses = py_ints([generation_address(x, keep)])
        for i in range(len(h[].weights)):  # driver-loop: over block weight slots
            addresses.append(Int(h[].weights[i].unsafe_ptr()))
        var lead = 2 if kind == 1 else len(h[].parts)
        for i in range(lead):  # driver-loop: over the state's named pieces
            addresses.append(Int(h[].parts[i].unsafe_ptr()))
        addresses.append(generation_address(y, keep))
        for i in range(lead, len(h[].parts)):  # driver-loop: over the Mamba-1 affine pieces
            addresses.append(Int(h[].parts[i].unsafe_ptr()))
        for i in range(len(reports)):  # driver-loop: over the fixed report buffers
            addresses.append(generation_address(reports[i], keep))
        var params = py_ints([b, h[].d_model])
        if kind == 1:
            if h[].profile == 2:
                params.append(h[].position)
        elif kind == 2:
            params.append(h[].q)
            params.extend(h[].dt_limit)
        else:
            params.append(h[].q)
            params.append(1 if h[].pending else 0)
        var updated = _native_function(h[].binding, name + "_decode_step")(addresses, params)
        if kind != 1:
            h[].q = Int(py=updated)
        if kind == 3:
            h[].pending = False
    if kind == 1 and h[].profile == 2:
        h[].position += 1
    _ = keep^
    if kind == 1:
        return y
    if kind == 2:
        return Python.tuple(y, reports[0])
    return Python.tuple(y, reports[0], reports[1], reports[2], reports[3])


def _session_pieces(
    hobj: PythonObject, slots: PythonObject, what: String, mut keep: List[PythonObject], mut copies: List[PythonObject],
) raises -> List[Int]:
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    if h[].kind == 1:
        _m1_check(slots, h[].profile, what)
    var pieces = List[Int]()
    for i in range(len(h[].names)):  # driver-loop: over the state's named pieces
        pieces.append(_mutable(slots[h[].names[i]], h[].dims[i], keep, copies, h[].names[i]))
    return pieces^


def mamba_session_sync(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].kind
    var what = _session_name(kind) + ".sync_state"
    _session_require_open(hobj, what)
    var state = session._state
    var slots = state.__dict__
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _session_pieces(hobj, slots, what, keep, copies)
    if h[].device is not None:
        var addresses = py_ints(pieces)
        var exported = _native_function(h[].binding, _kind_name(kind) + "_session_export_state")(h[].device, addresses)
        if kind == 1:
            if h[].profile == 2:
                h[].position = Int(py=exported)
        elif kind == 2:
            h[].q = Int(py=exported)
        else:
            h[].q = Int(py=exported[0])
            h[].pending = Bool(py=exported[1])
    else:
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            mamba_copy_f32(pieces[i], Int(h[].parts[i].unsafe_ptr()), _count(h[].dims[i]))
    _writeback(copies)
    _ = keep^
    if kind == 1:
        if h[].profile == 2:
            slots["absolute_position"] = h[].position
    else:
        slots["buffered_tokens"] = h[].q
        if kind == 3:
            slots["pending"] = PythonObject(h[].pending)
    return state


def mamba_session_load(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].kind
    var what = _session_name(kind) + ".load_state"
    _session_require_open(hobj, what)
    var state = session._state
    var slots = state.__dict__
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _session_pieces(hobj, slots, what, keep, copies)
    var q = 0
    var pending = False
    var position = h[].position
    if kind == 1:
        if h[].profile == 2:
            position = Int(py=slots["absolute_position"])
    else:
        q = Int(py=slots["buffered_tokens"])
        if kind == 3:
            pending = Bool(py=slots["pending"])
    if h[].device is not None:
        var addresses = py_ints(pieces)
        var name = _kind_name(kind) + "_session_load_state"
        if kind == 1:
            if h[].profile == 2:
                addresses.append(position)
            _native_function(h[].binding, name)(h[].device, addresses)
        elif kind == 2:
            _native_function(h[].binding, name)(h[].device, addresses, py_ints([q]))
        else:
            _native_function(h[].binding, name)(h[].device, addresses, py_ints([q, 1 if pending else 0]))
    else:
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            mamba_copy_f32(Int(h[].parts[i].unsafe_ptr()), pieces[i], _count(h[].dims[i]))
    _ = keep^
    h[].q = q
    h[].pending = pending
    h[].position = position
    return state


def _session_release(hobj: PythonObject, state: PythonObject) raises:
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    if not h[].open:
        return
    h[].open = False
    if state is not None:
        state.__dict__["_resident_session"] = PythonObject(None)
    if h[].device is not None:
        _native_function(h[].binding, _kind_name(h[].kind) + "_session_close")(h[].device)
    h[].weights.clear()
    h[].parts.clear()


def mamba_session_close(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    if not hobj.downcast_value_ptr[MambaDecodeHandle]()[].open:
        return PythonObject(None)
    try:
        _ = mamba_session_sync(session, context)
    finally:
        _session_release(hobj, session._state)
    return PythonObject(None)


def mamba_session_drop(session: PythonObject) raises -> PythonObject:
    # Finalizer: release an unclosed session without copying its state back.
    var values = session.__dict__
    var hobj = values.get("_handle")
    if hobj is None:
        return PythonObject(None)
    _session_release(hobj, values.get("_state"))
    return PythonObject(None)


def mamba_session_is_open(session: PythonObject) raises -> PythonObject:
    return PythonObject(session._handle.downcast_value_ptr[MambaDecodeHandle]()[].open)


def mamba_session_device(session: PythonObject) raises -> PythonObject:
    return session._handle.downcast_value_ptr[MambaDecodeHandle]()[].device


def mamba_session_repr(session: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var flag = String("True") if h[].open else String("False")
    return PythonObject(_session_name(h[].kind) + "(open=" + flag + ", rows=" + String(h[].b) + ")")


# ---------------------------------------------------------------- the C boundary


def mamba_native_call(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    if operation == "forward":
        return mamba3_forward(values[0], values[1], values[2], values[3], False, context)
    if operation == "step":
        return mamba3_forward(values[0], values[1], values[2], PythonObject(None), True, context)
    if operation == "allocate":
        return mamba3_allocate(values[0], values[1], context)
    if operation == "install_owned":
        return mamba3_install_owned(values[0], context)
    if operation == "release_owned":
        return mamba3_release_owned(values[0])
    if operation == "session_info":
        return mamba3_session_info(values[0])
    if operation == "export_owned":
        return mamba3_export_owned(values[0])
    if operation == "getstate":
        return mamba3_getstate(values[0])
    if operation == "setstate":
        return mamba3_setstate(values[0], values[1])
    if operation == "set_input_states":
        return mamba3_set_input_states(values[0], values[1], values[2], values[3], values[4], context)
    if operation == "session_open":
        return mamba_session_open(values[0], values[1], context)
    if operation == "session_init":
        return mamba_session_init(values[0], values[1], values[2], context)
    if operation == "session_step":
        return mamba_session_step(values[0], values[1], context)
    if operation == "session_sync":
        return mamba_session_sync(values[0], context)
    if operation == "session_load":
        return mamba_session_load(values[0], context)
    if operation == "session_close":
        return mamba_session_close(values[0], context)
    if operation == "session_drop":
        return mamba_session_drop(values[0])
    if operation == "session_is_open":
        return mamba_session_is_open(values[0])
    if operation == "session_device":
        return mamba_session_device(values[0])
    if operation == "session_repr":
        return mamba_session_repr(values[0])
    raise Error("mamba native: unknown operation " + operation)


def mamba_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = mamba_native_call(String(args[0]), args[1], args[2])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
