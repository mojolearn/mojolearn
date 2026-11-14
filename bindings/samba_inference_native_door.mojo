# SPDX-License-Identifier: Apache-2.0
"""Native Samba inference: `SambaStack.forward` / `step` and the loss.

A device build runs every call on the Samba stack program
(training/samba_program.mojo, lane x6-samba-on-executor): one executor
session, one step op per layer, the hidden state on the device. The host
route (CPU builds, the CPU inference model) runs the installed CPU kernels
layer by layer. Public classes and buffer exporters remain CPython-owned
interface records.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from core.samba_ids_impl import samba_inference_ids_into, samba_copy_row_prefixes
from std.memory import unsafe_memcpy
from bindings.generation_buffers_door import generation_initialize_binding, generation_array, generation_address, generation_weight_address, py_ints
from bindings.samba_state_native_door import samba_fields, samba_entries, samba_entry_key, samba_primitive
from core.entry_table import EntryRef, entry_ref_call
from core.mamba_state_entry_table import Mamba3StateRoute, MAMBA3_STATE_FORWARD, MAMBA3_STATE_DECODE
from core.neural_entry_table import NEURAL_MAMBA3_OWNED_STATE
from providers.python.mamba3_prefill_session_pyapi_door import mamba3_prefill_session_serves, mamba3_prefill_session_run
from core.samba_entry_table import (
    samba_ref,
    SambaEntryKey, SAMBA_TABLE_TRAINING, SAMBA_TABLE_NEURAL_HOST, SAMBA_PRIM_EMBEDDING_FORWARD,
    SAMBA_PRIM_RMS_NORM_FORWARD, SAMBA_PRIM_LINEAR_FORWARD, SAMBA_PRIM_CE_LOSS,
)
from bindings.samba_training_native_door import samba_ids
from std.python._cpython import GILReleased
from core.samba_entry_table_programs import (
    samba_programs_ref, samba_programs_call, samba_programs_available,
    SAMBA_PROGRAM_OPEN, SAMBA_PROGRAM_STATUS, SAMBA_PROGRAM_FORWARD,
)
from core.samba_program_handle import SambaProgramHandle
from bindings.neural_block_public_door import neural_training_entries, neural_training_selection, neural_block_allocate_binding, neural_block_forward_binding, _neural_lengths
from bindings.neural_prefill_native_door import _prefill_mamba3, _m3_output_array
from bindings.causal_lm_public_door import _mutable, _writeback
from bindings.causal_lm_construct_door import _model_bindings
from bindings.py_balanced import py_call, py_get, py_setattr, py_setitem, py_type
from bindings.byte_perf_helpers_unpack import byte_view_bytes
from bindings.samba_admission_door import samba_admit_model_fields
# RUN (lane rs-neural): typed pieces, no Python import.
from training.samba_inference_run import _m3_shapes, _m3_names, _samba_host_ref
# The stateless forward's route (lane restore-samba-resident-forward): the
# resident forward by default, the stack program under the B-arm define.
from training.neural_identical_experiments import IDN_SAMBA_PROGRAM_FORWARD
from providers.python.samba_resident_pyapi_door import samba_resident_native_forward


def samba_state_construct(state: PythonObject, b: PythonObject, maximum: PythonObject, layers: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(state)
    py_setitem(fields, "batch_size", py_call(bi.int, b))
    py_setitem(fields, "max_tokens", py_call(bi.int, maximum))
    py_setitem(fields, "layers", py_call(bi.list, layers))
    return PythonObject(None)


def samba_allocate(model: PythonObject, batch: PythonObject, maximum: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    # Integers only (anything with __index__): a str is not parsed and a
    # bool is not counted as 1.
    if py_type(batch) is bi.bool or py_type(maximum) is bi.bool:
        raise Error("TypeError: SambaStack.allocate_state: batch_size and max_tokens must be integers")
    var b = 0
    var cap = 0
    try:
        var operator = native_loaded_module("operator")
        b = Int(py=py_call(operator.index, batch))
        cap = Int(py=py_call(operator.index, maximum))
    except error:
        raise Error("TypeError: SambaStack.allocate_state: batch_size and max_tokens must be integers")
    if b < 1 or cap < 1:
        raise Error("SambaStack.allocate_state: batch_size and max_tokens must be positive")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var blocks = fields["_blocks"]
    var states = bi.list()
    for i in range(len(config["layers"])):  # driver-loop: over persistent layer state owners
        var block = blocks[i]
        if String(config["layers"][i]) == "attention":
            _ = py_call(states.append, neural_block_allocate_binding(block, PythonObject(b), PythonObject(cap), context["neural"]))
        else:
            var state = py_call(bi.object.__new__, context["mamba3_state_class"])
            var slots = samba_fields(state)
            var shapes = _m3_shapes(b, Int(py=samba_fields(block)["nheads"]))
            var names = _m3_names()
            for j in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
                py_setitem(slots, names[j], generation_array(shapes[j]))
            py_setitem(slots, "buffered_tokens", 0)
            py_setitem(slots, "pending", False)
            _ = py_call(states.append, state)
    var result = py_call(bi.object.__new__, context["state_class"])
    _ = samba_state_construct(result, PythonObject(b), PythonObject(cap), states)
    return result


def _samba_block_entries(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var route = Mamba3StateRoute(1, 0, False)
    return _samba_block_entries(block, context, route)


def _samba_block_entries(block: PythonObject, context: PythonObject, mut route: Mamba3StateRoute) raises -> PythonObject:
    # The block's binding entries and the key of the image that serves its
    # carried-state calls (core/mamba_state_entry_table.mojo): the CPU
    # inference class reads the neural host at (identical, cpu); a training
    # block reads its connected training selection's (tier, vendor), the CPU
    # training image's own slot when the vendor is the CPU.
    var bi = native_loaded_module("builtins")
    var nc = context["neural"]
    if Bool(py=py_call(bi.isinstance, block, nc["block_classes"]["cpu"]["mamba3"])):
        route = Mamba3StateRoute(1, 0, False)
        return _model_bindings(PythonObject("cpu"), "mamba3", nc, PythonObject("identical"))["mamba3"]
    var selection = neural_training_selection(samba_fields(block), nc)
    route = Mamba3StateRoute(selection.key.mode, selection.key.vendor, selection.key.vendor == 0)
    return selection.entries

def _m3_state_call(block: PythonObject, x: PythonObject, state: PythonObject, step: Bool, entries: PythonObject,
                   captured_fields: PythonObject, captured_slots: PythonObject, route: Mamba3StateRoute) raises -> PythonObject:
    """One carried-state call: `(y, h_last, k_last, v_last, theta_last)`. The
    state's pieces, `buffered_tokens` and `pending` are updated; the caller
    assigns the four reports."""
    # Standalone Mamba operations supply one retained metadata snapshot. Other
    # native owners preserve their established admission and call contract.
    var fields = samba_fields(block) if captured_fields is None else captured_fields
    var slots = samba_fields(state) if captured_slots is None else captured_slots
    if py_get(slots, "_resident_session") is not None:
        raise Error("Mamba3: state belongs to an open resident decode session; synchronize or close it first")
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=fields["d_model"])
    var h = Int(py=fields["nheads"])
    var shapes = _m3_shapes(b, h)
    var names = _m3_names()
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var addresses: List[Int] = [generation_address(x, keep)]
    # An installed owned snapshot is read on the device: op 16 on the
    # block's Mamba-3 session handle, open in the image the block's route
    # names (core/mamba3_prefill_session.mojo; lane mamba3-prefill-handle).
    var owned = py_get(fields, "_owned_generation")
    var session = py_get(fields, "_prefill_session")
    var on_session = (owned is not None and route.has_sessions()
                      and mamba3_prefill_session_serves(session, route.session_ref()))
    var weights = fields["_w"]
    for i in range(len(weights)):  # driver-loop: over existing Mamba3 parameter ABI slots
        if on_session:
            addresses.append(0)
        else:
            addresses.append(generation_weight_address(weights[i], keep))
    for i in range(len(names)):  # driver-loop: over mutable continuation buffers
        addresses.append(_mutable(slots[names[i]], shapes[i], keep, copies, names[i]))
    # y and the four reports are written whole by every route below (the
    # GPU and owned `mamba_block_call`, the host shells' `_write`): unfilled
    # unless MAMBA3_FORWARD_ZERO_FILL (bindings/neural_prefill_native_door.mojo).
    var output = _m3_output_array([b, l, d])
    var reports = Python.list(_m3_output_array([b, h, 64, 128]), _m3_output_array([b, h, 128]),
                              _m3_output_array([b, h, 64]), _m3_output_array([b, h, 32]))
    addresses.append(generation_address(output, keep))
    _ = output  # keep the Python object alive through the native call above
    for i in range(4):  # driver-loop: over the four reporting buffers
        addresses.append(generation_address(reports[i], keep))
        _ = output  # keep the Python object alive through the native call above
    var q = Int(py=slots["buffered_tokens"])
    var pending = 1 if Bool(py=slots["pending"]) else 0
    var updated = PythonObject(None)
    if on_session:
        # The former Python entry `mamba3_state_owned` and the env-gated
        # provider arm ran this operation with these words.
        updated = PythonObject(mamba3_prefill_session_run(session, NEURAL_MAMBA3_OWNED_STATE,
            addresses, [b, 1 if step else l, d, q, pending, Int(py=owned), 1 if step else 0]))
        _ = output  # keep the Python object alive through the native call above
    else:
        # The image that serves this block's carried state, by the block's key
        # (core/mamba_state_entry_table.mojo; lane transformer-session-native):
        # one table call, the GIL released around it. The former state
        # provider capsule and the Python entries `mamba3_forward` /
        # `mamba3_decode_step` of the block's binding module ran these ops
        # with these words.
        var ints: List[Int] = [b, d, q, pending] if step else [b, l, d, q, pending]
        var entry = route.state_ref()
        var buffered = 0
        with GILReleased(Python()):
            buffered = entry_ref_call(entry, MAMBA3_STATE_DECODE if step else MAMBA3_STATE_FORWARD,
                addresses, ints, List[Float64]())
        updated = PythonObject(buffered)
        _ = output  # keep the Python object alive through the native call above
    _writeback(copies)
    py_setitem(slots, "buffered_tokens", updated)
    py_setitem(slots, "pending", False)
    _ = keep^
    return Python.tuple(output, reports[0], reports[1], reports[2], reports[3])


def _m3_state_forward(block: PythonObject, x: PythonObject, state: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    # Captured fields/slots are passed explicitly: a PythonObject(None)
    # default is evaluated at compile time and needs a live interpreter.
    var fields = samba_fields(block)
    var route = Mamba3StateRoute(1, 0, False)
    var entries = _samba_block_entries(block, context, route)
    var result = _m3_state_call(block, x, state, step, entries, fields, samba_fields(state), route)
    var report_names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(4):  # driver-loop: over reporting result metadata
        py_setitem(fields, report_names[i], result[i + 1])
    return result[0]


# TOMBSTONE: SAMBA_THREE_TIER_FORWARD _samba_resident_forward + the provider tier (samba_provider_forward) + the device per-layer walk of samba_forward (the three-tier driver; one program now: SambaStackProgram, training/samba_program.mojo) deleted 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch


def samba_program_arm(fields: PythonObject, key: SambaEntryKey) raises -> Bool:
    """Whether this stack runs on the Samba stack program: its GPU training
    image published the program entry (core/samba_entry_table_programs.mojo)
    and it is not the CPU inference route. Otherwise the host per-layer route
    below runs (CPU builds). Lane samba-sessions-native (2026-11-12): the key
    decides, not `hasattr` on a Python module."""
    if Bool(py=py_get(fields, "_native_cpu_inference", PythonObject(False))):
        return False
    return samba_programs_available(key)


def samba_program_words(fields: PythonObject, config: PythonObject, b: Int, smax: Int) raises -> Tuple[List[Int], Float64]:
    """UNPACK, once per program open: SAMBA_PROGRAM_OPEN's integer words
    ([n_layers, layer codes, vocab, d_model, n_heads, n_kv_heads, head_dim,
    intermediate, tie, n_total, batch_size, max_tokens]) and norm_eps."""
    var layers = config["layers"]
    var n = len(layers)
    var words = List[Int](capacity=n + 11)
    words.append(n)
    for i in range(n):  # driver-loop: over model layer metadata at program creation
        words.append(0 if String(layers[i]) == "mamba3" else 1)
    words.append(Int(py=config["vocab"]))
    words.append(Int(py=config["d_model"]))
    for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over program shape metadata
        var value = config[name]
        words.append(0 if value is None else Int(py=value))
    words.append(1 if Bool(py=config["tie_embeddings"]) else 0)
    words.append(Int(py=fields["n_total"]))
    words.append(b)
    words.append(smax)
    return (words^, Float64(py=config["norm_eps"]))


def samba_program_open_owner(key: SambaEntryKey, addresses: List[Int], words: List[Int], eps: Float64,
                             b: Int, smax: Int) raises -> PythonObject:
    """Open a program through the table entry of `key` (resolved here, once)
    and PACK its native handle owner (core/samba_program_handle.mojo). The
    GIL is released around the native open, as the former Python entry
    released it around the state load."""
    var entry = samba_programs_ref(key)
    var floats: List[Float64] = [eps]
    var handle = 0
    with GILReleased(Python()):
        handle = samba_programs_call(entry, SAMBA_PROGRAM_OPEN, addresses, words, floats)
    return PythonObject(alloc=SambaProgramHandle(key, entry, handle, b, smax))


def samba_program_call(program: PythonObject, operation: Int, addresses: List[Int], words: List[Int]) raises -> Int:
    """One program operation through the handle owner `program` (its handle
    word and entry read once here), the GIL released around the native call.
    The caller keeps every addressed buffer alive."""
    var owner = program.downcast_value_ptr[SambaProgramHandle]()
    var result = 0
    with GILReleased(Python()):
        result = owner[].call(operation, addresses, words)
    _ = program
    return result


def samba_state_words(
    model: PythonObject, state: PythonObject, what: String,
    mut keep: List[PythonObject], mut copies: List[PythonObject],
) raises -> Tuple[List[Int], List[Int]]:
    """The state's piece addresses and cursor words, in the layout the
    program's state entries take: per layer in stack order, the ten Mamba-3
    pieces (`_m3_names()` order) and (buffered_tokens, pending), or k_cache,
    v_cache and (cached_tokens, 0)."""
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var slots = samba_fields(state)
    var b = Int(py=slots["batch_size"])
    var smax = Int(py=slots["max_tokens"])
    var layers = slots["layers"]
    if len(layers) != len(config["layers"]):
        raise Error(what + ": the state holds " + String(len(layers)) + " layers, the stack "
                    + String(len(config["layers"])))
    var blocks = fields["_blocks"]
    var names = _m3_names()
    var pieces = List[Int]()
    var cursors = List[Int]()
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layer state owners
        var lslots = samba_fields(layers[i])
        if String(config["layers"][i]) == "attention":
            if Int(py=py_get(lslots, "batch_size", PythonObject(b))) != b or Int(py=lslots["max_tokens"]) != smax:
                raise Error(what + ": layer " + String(i) + " state differs from the stack state's batch or capacity")
            if Int(py=py_get(lslots, "window", PythonObject(0))) != 0:
                raise Error(what + ": layer " + String(i) + " state has a sliding window")
            # The arithmetic-profile word the attention state carries
            # (`_causal_layer_run_bound`'s check and write).
            var pid = py_get(samba_fields(blocks[i]), "_arithmetic_profile")
            var old = py_get(lslots, "arithmetic_profile")
            if old is not None and pid is not None and not Bool(py=old == pid):
                raise Error(what + ": layer " + String(i) + " state arithmetic profile differs")
            if pid is not None:
                py_setitem(lslots, "arithmetic_profile", pid)
            var n = b * Int(py=lslots["n_kv_heads"]) * smax * Int(py=lslots["head_dim"])
            pieces.append(_mutable(lslots["k_cache"], [n], keep, copies, "k_cache"))
            pieces.append(_mutable(lslots["v_cache"], [n], keep, copies, "v_cache"))
            cursors.append(Int(py=lslots["cached_tokens"]))
            cursors.append(0)
        else:
            var shapes = _m3_shapes(b, Int(py=samba_fields(blocks[i])["nheads"]))
            for k in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
                pieces.append(_mutable(lslots[names[k]], shapes[k], keep, copies, names[k]))
            cursors.append(Int(py=lslots["buffered_tokens"]))
            cursors.append(1 if Bool(py=lslots["pending"]) else 0)
    return (pieces^, cursors^)


def samba_set_cursors(model: PythonObject, state: PythonObject, cursors: List[Int]) raises:
    """PACK: the program's post-call cursor words onto the state's layer
    records."""
    var config = samba_fields(samba_fields(model)["config"])
    var layers = samba_fields(state)["layers"]
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        var lslots = samba_fields(layers[i])
        if String(config["layers"][i]) == "attention":
            py_setitem(lslots, "cached_tokens", PythonObject(cursors[2 * i]))
        else:
            py_setitem(lslots, "buffered_tokens", PythonObject(cursors[2 * i]))
            py_setitem(lslots, "pending", PythonObject(cursors[2 * i + 1] != 0))


def _samba_program_reports(fields: PythonObject, config: PythonObject, b: Int, mut addresses: List[Int]) raises -> PythonObject:
    """Fresh report arrays for every Mamba-3 layer (four each, the shapes
    `_m3_state_call` gives them), their addresses appended to `addresses`
    in layer order; returns them per layer (None for an attention layer)."""
    var bi = native_loaded_module("builtins")
    var blocks = fields["_blocks"]
    var out = bi.list()
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layers (report owners)
        if String(config["layers"][i]) != "mamba3":
            _ = py_call(out.append, PythonObject(None))
            continue
        var h = Int(py=samba_fields(blocks[i])["nheads"])
        # Lane fix-wave2-leftovers (fix-mamba3-unfilled-rest leftover): the
        # four arrays are unfilled unless MAMBA3_FORWARD_ZERO_FILL (the one
        # Mamba-3 output switch, bindings/neural_prefill_native_door.mojo):
        # a run that asks for reports clears skip_reports
        # (training/samba_program_entry.mojo), every Mamba-3 step op copies
        # its four reports, and enqueue_reports downloads each whole
        # (block_step_reports_to_host, m3_report_floats = these shapes) or
        # raises (no resident block), behind the run's one wait.
        var four = Python.list(_m3_output_array([b, h, 64, 128]), _m3_output_array([b, h, 128]),
                               _m3_output_array([b, h, 64]), _m3_output_array([b, h, 32]))
        for k in range(4):  # small-loop(the four Mamba-3 reports)
            addresses.append(Int(py=four[k]._addr))
        _ = py_call(out.append, four)
    return out


def _samba_publish_reports(fields: PythonObject, config: PythonObject, reports: PythonObject) raises:
    """Assign each Mamba-3 block's four reports after the run wrote them."""
    var blocks = fields["_blocks"]
    var names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layers (report owners)
        if reports[i] is None:
            continue
        var block_fields = samba_fields(blocks[i])
        for k in range(4):  # small-loop(the four Mamba-3 reports)
            py_setitem(block_fields, names[k], reports[i][k])


def _samba_program_call(model: PythonObject, fields: PythonObject, config: PythonObject, key: SambaEntryKey,
                        ids: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    """One per-call run on the stack's program: (B, L) ids -> (B, L, vocab)
    logits, from a zero state (stateless) or from the caller's state, which
    holds the post-call state afterwards. The stateless and the stateful
    programs are kept apart (their attention capacity differs: L versus the
    state's max_tokens); each is reopened when its (B, capacity) changes or
    a failed call lost it. The model keeps each program as its native handle
    owner (core/samba_program_handle.mojo); RUN is native table calls."""
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var smax = l
    var slot = String("_samba_program")
    if state is not None:
        smax = Int(py=samba_fields(state)["max_tokens"])
        slot = String("_samba_state_program")
    var program = py_get(fields, slot)
    if program is not None:
        var owner = program.downcast_value_ptr[SambaProgramHandle]()
        if (owner[].b != b or owner[].smax != smax
                or samba_program_call(program, SAMBA_PROGRAM_STATUS, List[Int](), List[Int]()) == 0):
            # Dropping the owner releases its program.
            py_setitem(fields, slot, PythonObject(None))
            program = PythonObject(None)
    var registry = Int(py=fields["flat"]._addr)
    if program is None:
        var opened = samba_program_words(fields, config, b, smax)
        var open_addresses: List[Int] = [registry]
        program = samba_program_open_owner(key, open_addresses, opened[0], opened[1], b, smax)
        py_setitem(fields, slot, program)
    var output = generation_array([b, l, Int(py=config["vocab"])])
    # [registry, ids, logits, cursors_out], the Mamba-3 blocks' public
    # reports (h_last_, k_last_, v_last_, theta_last_), as the host route's
    # per-block calls assign them: the program's step ops keep the last
    # run's reports, read back with the run's one wait
    # (training/samba_program.mojo enqueue_reports); then the state pieces.
    var addresses: List[Int] = [registry, Int(py=ids._addr), Int(py=output._addr), 0]
    if state is None:
        # lane perf-neural-sessions (2026-11-13): the stateless forward
        # requests no Mamba-3 reports. A stack's Mamba-3 blocks discard them
        # (`_discard_reports`, set by samba_build_blocks on every one), and
        # the 0.8.55 resident forward (`samba_resident_forward`, the measured
        # winner: samba-forward/bytes NVIDIA 4.60 ms vs 10.54 candidate
        # 4c2340386abc, AMD 6.10 vs 8.36) published none. The program path
        # allocated four zero-filled host arrays per Mamba-3 layer per call
        # (2 x 12 x 64 x 128 floats for h_last alone at the board shape),
        # downloaded the reports into them and rewrote four block fields.
        # Reports stay on the stateful forward and the step, whose callers
        # read them. No kernel and no logits byte changes.
        var stateless_words: List[Int] = [b, l, 0]
        _ = samba_program_call(program, SAMBA_PROGRAM_FORWARD, addresses, stateless_words)
        _ = ids  # keep the Python object alive through the native call above
        _ = output
        return output
    var report_addresses = List[Int]()
    var reports = _samba_program_reports(fields, config, b, report_addresses)
    for i in range(len(report_addresses)):  # driver-loop: over the Mamba-3 layers' report addresses
        addresses.append(report_addresses[i])
    var words: List[Int] = [b, l, len(report_addresses)]
    # The state was admitted at the public door (samba_admit).
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var state_words = samba_state_words(model, state, "SambaStack", keep, copies)
    var cursors = List[Int](length=len(state_words[1]), fill=0)
    addresses[3] = Int(cursors.unsafe_ptr())
    for i in range(len(state_words[0])):  # driver-loop: over the state's piece addresses (per layer)
        addresses.append(state_words[0][i])
    for i in range(len(state_words[1])):  # driver-loop: over the state's cursor words (two per layer)
        words.append(state_words[1][i])
    _ = samba_program_call(program, SAMBA_PROGRAM_FORWARD, addresses, words)
    _samba_publish_reports(fields, config, reports)
    _writeback(copies)
    _ = keep^
    _ = ids  # keep the Python object alive through the native call above
    samba_set_cursors(model, state, cursors)
    _ = cursors
    return output


def _samba_inference_ids(value: PythonObject, vocab: Int, step: Bool, lengths: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, List[Int]]:
    var bi = native_loaded_module("builtins")
    var own = py_type(value) is context["neural"]["array_class"]
    var view = value._mv if own else py_call(bi.memoryview, value)
    var shape = value.shape if own else view.shape
    if (len(shape) != 2 and not (step and len(shape) == 1)):
        raise Error("Samba: inputs must be (B,L) integer ids, or (B,) for step")
    var b = Int(py=shape[0])
    var l = 1 if len(shape) == 1 else Int(py=shape[1])
    if b < 1 or l < 1 or (step and l != 1):
        raise Error("Samba: input batch/length must be positive; step takes one token per row")
    var sizes = List[Int]()
    if lengths is not None:
        sizes = _neural_lengths(lengths, b, l, context["neural"])
    var fmt = String(py_call(view.format.lstrip, "=@<"))
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("Samba: inputs must be native-endian integer ids")
    # One native C-order copy of the ids under a buffer pin (formerly
    # view.tobytes then array.frombytes: two Python copies per token step).
    var item = Int(py=view.itemsize)
    var packed = byte_view_bytes(view, b * l * item)
    var column_major = own and String(value.order) == "F"
    var ids = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
    _ = ids  # keep the Python object alive through the native call above
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    samba_inference_ids_into(Int(packed.unsafe_ptr()), item, signed, b, l,
                             column_major, vocab, lengths is not None, sizes, dst)
    _ = packed  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    _ = packed^
    # The admitted ragged lengths go back to samba_forward, which formerly
    # admitted them a second time for the output prefix copy.
    return (ids, sizes^)


def _samba_host_primitive(prim: EntryRef, op: Int, addresses: List[Int], ints: List[Int],
                          floats: List[Float64]) raises:
    """One host route primitive through the resolved entry."""
    _ = samba_primitive(prim, op, addresses, ints, floats)


def samba_forward(model: PythonObject, inputs: PythonObject, state: PythonObject, lengths: PythonObject,
                  step: Bool, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    if step and state is None:
        raise Error("SambaStack.step: state is required")
    if lengths is not None and state is not None:
        raise Error("SambaStack.forward: lengths applies to stateless calls only")
    if state is not None and not Bool(py=py_call(bi.isinstance, state, context["state_class"])):
        raise Error("TypeError: SambaStack: state must be a SambaState")
    if state is not None and py_get(samba_fields(state), "_resident_session") is not None:
        # A SambaDecodeSession holds this state on the device
        # (bindings/samba_decode_native.mojo); its arrays are stale.
        raise Error("SambaStack: the state is owned by an open decode session (its pieces live in the session and the "
                    + "caller's arrays are stale); call sync_state() or close() on the session first")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var cpu_inference = Bool(py=py_get(fields, "_native_cpu_inference", PythonObject(False)))
    # UNPACK: the Samba table key of the primitives this model runs on (the
    # identical `_mojolearn_neural_host` image for the CPU inference model).
    # The binding reads below load and admit the image that publishes the
    # key's entries; RUN reaches them through the tables only.
    var key = SambaEntryKey(1, 0)
    var entries = PythonObject(None)
    if cpu_inference:
        _ = _model_bindings(PythonObject("cpu"), "mamba3", context["neural"], PythonObject("identical"))["primitives"]
    else:
        entries = samba_entries(context, fields["numeric_mode"])
        key = samba_entry_key(context, fields["numeric_mode"])
    var d = Int(py=config["d_model"])
    var v = Int(py=config["vocab"])
    var admitted = _samba_inference_ids(inputs, v, step, lengths, context)
    var ids = admitted[0]
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var output = PythonObject(None)
    if samba_program_arm(fields, key):
        # A device build. The stateless forward runs the resident session's
        # forward (`samba_resident_forward`, descriptor op 16 on the stack's
        # primitive image; training/samba_resident.mojo), the measured winner
        # of samba-forward: IDN_SAMBA_RESIDENT_STEP PROMOTED NVIDIA 12.0 ->
        # 5.3 ms (0.44x), AMD 8.8 -> 5.8 ms (0.66x); 0.8.55 paper column
        # NVIDIA 4.60 / AMD 6.10 ms. Lane x6 (2026-11-09) put every call on
        # the stack program without a measurement (candidate 4c2340386abc:
        # NVIDIA 10.54 ms, MI325X 8.36 ms). -D MOJOLEARN_SAMBA_PROGRAM_FORWARD
        # is the program B arm of the owed A/B. The stateful forward and the
        # step stay on the program (the resident session holds no carried
        # state); an image without the resident route (op 24 == 0) or a
        # dropout stack falls through to the program as well.
        # The route taken is recorded on the model (`_samba_forward_route`)
        # so the A/B can prove which arm ran: 0.8.66 measured 10.58 ms while
        # the resident forward still existed, and whether it was selected
        # there was never established.
        comptime if not IDN_SAMBA_PROGRAM_FORWARD:
            if state is None:
                output = samba_resident_native_forward(fields, config, entries, ids)
        if output is None:
            output = _samba_program_call(model, fields, config, key, ids, state, context)
            if state is None:
                py_setitem(fields, "_samba_forward_route", PythonObject("program"))
        else:
            py_setitem(fields, "_samba_forward_route", PythonObject("resident"))
    else:
        # The host route (CPU builds and the CPU inference model): the
        # installed CPU kernels layer by layer. Device builds never come
        # here (samba_program_arm).
        # lane perf-neural-sessions: the door admits only the fields every
        # route reads (bindings/samba_admission_door.mojo
        # _samba_admit_read_only); this route also reads the registry views
        # and the layer blocks, admitted here before the reads.
        samba_admit_model_fields(model, [String("arrays"), String("_blocks")], context)
        var blocks = fields["_blocks"]
        var states = PythonObject(None)
        if state is not None:
            var slots = samba_fields(state)
            if Int(py=slots["batch_size"]) != b or len(slots["layers"]) != len(config["layers"]):
                raise Error("Samba: decode state batch or layer count differs")
            states = slots["layers"]
        var arrays = fields["arrays"]
        var hidden = generation_array([b, l, d])
        var prim = _samba_host_ref(cpu_inference, key)
        _samba_host_primitive(prim, SAMBA_PRIM_EMBEDDING_FORWARD,
            [Int(py=hidden._addr), Int(py=arrays["embed.weight"]._addr), Int(py=ids._addr)], [b * l, v, d], [])
        _ = hidden  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        for i in range(len(config["layers"])):  # driver-loop: over native model layer execution owners
            var block = blocks[i]
            var result = PythonObject(None)
            if String(config["layers"][i]) == "attention":
                result = neural_block_forward_binding(block, hidden, PythonObject(None) if states is None else states[i], PythonObject(step), PythonObject(None), context["neural"])
            elif states is None:
                var route = Mamba3StateRoute(1, 0, False)
                var entries = _samba_block_entries(block, context, route)
                result = _prefill_mamba3(block, hidden, entries, route)
            else:
                result = _m3_state_forward(block, hidden, states[i], step, context)
            hidden = result
        output = generation_array([b, l, v])
        var head = arrays["embed.weight"] if Bool(py=config["tie_embeddings"]) else arrays["lm_head.weight"]
        var normalized = generation_array([b, l, d])
        _samba_host_primitive(prim, SAMBA_PRIM_RMS_NORM_FORWARD,
            [Int(py=normalized._addr), Int(py=hidden._addr), Int(py=arrays["norm_f.weight"]._addr)], [b * l, d],
            [Float64(py=config["norm_eps"])])
        _ = normalized  # keep the Python object alive through the native call above
        _ = hidden  # keep the Python object alive through the native call above
        _samba_host_primitive(prim, SAMBA_PRIM_LINEAR_FORWARD,
            [Int(py=output._addr), Int(py=normalized._addr), Int(py=head._addr)], [b * l, v, d], [])
        _ = normalized  # keep the Python object alive through the native call above
        _ = head  # keep the Python object alive through the native call above
        _ = output  # keep the Python object alive through the native call above
        _ = normalized^
        _ = head^
        _ = hidden^
    if lengths is not None:
        var sizes = admitted[1].copy()
        var padded = generation_array([b, l, v])
        samba_copy_row_prefixes(Int(py=padded._addr), Int(py=output._addr), sizes, b, l, v)
        _ = padded  # keep the Python object alive through the native call above
        _ = output  # keep the Python object alive through the native call above
        _ = output^  # the copy above read `output` by address
        output = padded
    if step:
        py_setattr(output, "shape", py_call(bi.tuple, py_ints([b, v])))
        py_setattr(output, "strides", py_call(bi.tuple, py_ints([4 * v, 4])))
        py_setattr(output, "ndim", 2)
    _ = ids^
    return output


def samba_loss(model: PythonObject, inputs: PythonObject, targets: PythonObject, context: PythonObject) raises -> PythonObject:
    # Row N169: the logits come from `SambaStack.forward`'s one path (the
    # stack program on a device build, lane x6), so no layer loop moves the
    # hidden state through host arrays.
    # OWED: the cross entropy inside the program (logits never leave the
    # device) needs a loss op after the head in training/samba_program.mojo.
    var logits = samba_forward(model, inputs, PythonObject(None), PythonObject(None), False, context)
    var fields = samba_fields(model)
    var v = Int(py=samba_fields(fields["config"])["vocab"])
    var y = samba_ids(targets, v, True, context)
    var rows = Int(py=logits.shape[0]) * Int(py=logits.shape[1])
    if Int(py=y.size) != rows:
        raise Error("SambaStack.loss: targets must contain one class index per logit row")
    _ = samba_entries(context, fields["numeric_mode"])  # mode/vendor admission of the connected binding
    var loss = generation_array([1])
    var row_loss = generation_array([rows])
    var unused_gradient = generation_array([1])
    # ce_loss words: [n_rows, vocab, ignore_index, reduction (2 mean), num_items, want_grad], [label_smoothing].
    _ = samba_primitive(samba_ref[SAMBA_TABLE_TRAINING](samba_entry_key(context, fields["numeric_mode"])), SAMBA_PRIM_CE_LOSS,
        [Int(py=loss._addr), Int(py=row_loss._addr), Int(py=unused_gradient._addr), Int(py=logits._addr), Int(py=y._addr)],
        [rows, v, -100, 2, 0, 0], [0.0])
    _ = unused_gradient  # keep the Python object alive through the native call above
    _ = row_loss  # keep the Python object alive through the native call above
    _ = loss  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = logits  # keep the Python object alive through the native call above
    var result = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
    _ = loss  # keep the Python object alive through the native call above
    _ = loss^
    _ = row_loss^
    _ = unused_gradient^
    _ = logits^
    _ = y^
    return result
