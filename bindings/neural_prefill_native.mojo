# SPDX-License-Identifier: Apache-2.0
"""Native zero-state training prefill and immutable tape ownership."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_setitem
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.neural_block_public import neural_training_selection, neural_mamba3_restore, neural_block_forward_binding, _neural_input, _neural_int_words
from core.entry_table import entry_ref_unresolved
from core.mamba_state_entry_table import Mamba3StateRoute
from core.neural_entry_table import (
    NEURAL_MAMBA3_SESSION_FORWARD, NEURAL_MAMBA3_OWNED_FORWARD, MAMBA3_CAP_SESSION_PREFILL, MAMBA3_CAP_REPORTS_OPTIONAL,
)
from providers.python.mamba3_prefill_session_pyapi import (
    mamba3_prefill_caps, mamba3_prefill_session_ensure, mamba3_prefill_session_run,
)
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_execute, neural_descriptor_construct, neural_descriptor_handle,
    neural_descriptor_reserved_execute,
)
from bindings.causal_lm_construct import _model_f32, _model_shape
from bindings.causal_lm_public import _native_function
from core.shared_slot import SharedSlot


struct _PrefillVerdicts(Defaultable, Movable):
    # Row N187: the capability answers of one connected entries object
    # (hasattr probes and the compiled `*_forward_tape_enabled` /
    # `*_reports_optional` words, fixed for a binding), resolved once per
    # entries object instead of on every training step. -1 = not resolved.
    # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
    # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
    # (the transformer forward tape: a TransformerBlock prefill keeps no tape.)
    # The Mamba-3 prefill session answers (session forward present, reports
    # optional) are the training image's capability bits now, one table call
    # (lane mamba3-prefill-handle; core/neural_entry_table.mojo
    # NEURAL_MAMBA3_SESSION_CAPS): their fields and probes are gone.
    var entries: PythonObject
    var tape_mamba3: Int

    def __init__(out self):
        self.entries = PythonObject(None)
        self.tape_mamba3 = -1


comptime _PREFILL_VERDICTS = SharedSlot[_PrefillVerdicts, "MojolearnNeuralPrefillVerdictsV3"]


def _prefill_hold(entries: PythonObject) raises:
    # A different entries object (another binding, mode or vendor) drops
    # every held answer. The slot keeps that object alive for its identity.
    var v = _PREFILL_VERDICTS.get_or_create_ptr()
    if v[].entries is None or v[].entries is not entries:
        v[].entries = entries
        v[].tape_mamba3 = -1


def _prefill_tape_enabled(entries: PythonObject) raises -> Bool:
    """Whether the Mamba-3 binding keeps a forward tape (MOJOLEARN_IDN_ACT_RETAIN=3)."""
    _prefill_hold(entries)
    var v = _PREFILL_VERDICTS.get_or_create_ptr()
    if v[].tape_mamba3 >= 0:
        return v[].tape_mamba3 == 1
    var bi = native_loaded_module("builtins")
    var enabled = Bool(py=py_call(bi.hasattr, entries, "mamba3_forward_tape_enabled"))
    if enabled:
        enabled = Bool(py=_native_function(entries, "mamba3_forward_tape_enabled")())
    _prefill_hold(entries)
    v = _PREFILL_VERDICTS.get_or_create_ptr()
    v[].tape_mamba3 = 1 if enabled else 0
    return enabled


def _prefill_mamba3_reports(block: PythonObject, x: PythonObject, entries: PythonObject,
                            captured_fields: PythonObject, route: Mamba3StateRoute) raises -> PythonObject:
    """The zero-state prefill: `(y, h_last, k_last, v_last, theta_last)`, the
    reports None where the binding skips them. The caller assigns them.
    `route` is the block's Mamba-3 key: it names the training image holding
    the block's prefill session (core/mamba3_prefill_session.mojo)."""
    var bi = native_loaded_module("builtins")
    # A public Mamba operation captures model metadata and weight owners once;
    # its caller publishes session/cache changes back to the live dictionary.
    var fields = py_call(bi.object.__getattribute__, block, "__dict__") if captured_fields is None else captured_fields
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=fields["d_model"])
    var nh = Int(py=fields["nheads"])
    var weights = fields["_w"]
    neural_mamba3_restore(fields, entries, weights, route)
    var owned = py_get(fields, "_owned_generation")
    # The session image's capability bits (one table call), replacing the
    # former hasattr probes of `mamba3_prefill_session_forward` and the
    # `mamba3_prefill_session_reports_optional` Python entry.
    var caps = 0
    var session_entry = entry_ref_unresolved()
    if route.has_sessions():
        session_entry = route.session_ref()
        caps = mamba3_prefill_caps(session_entry)
    var session = ((caps & MAMBA3_CAP_SESSION_PREFILL) != 0 or owned is not None) and getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") != "1"
    var skip = False
    if session and Bool(py=py_get(fields, "_discard_reports", PythonObject(False))):
        skip = (caps & MAMBA3_CAP_REPORTS_OPTIONAL) != 0
    var reports = bi.list()
    var report_addresses = List[Int]()
    # Native report shapes; the former Python tuples were built only to be
    # read back word by word.
    var report_shapes: List[List[Int]] = [[b, nh, 64, 128], [b, nh, 128], [b, nh, 64], [b, nh, 32]]
    for i in range(4):  # driver-loop: over the four native Mamba3 report buffers
        var report = PythonObject(None)
        if not skip:
            report = generation_array(report_shapes[i])
        _ = py_call(reports.append, report)
        report_addresses.append(0 if skip else Int(py=report._addr))
    var out = generation_array([b, l, d])
    var params: List[Int] = [b, l, d]
    if session:
        # The block's session: a native registry handle of its training image
        # (the former Python entries mamba3_prefill_session_create,
        # mamba3_prefill_session_forward[_owned], the handle factory and the
        # env-gated provider arm ran the same operations with these words).
        var owner = mamba3_prefill_session_ensure(fields, session_entry)
        var addresses: List[Int] = [Int(py=x._addr)]
        if owned is not None:
            addresses.append(Int(py=out._addr))
            addresses.extend(report_addresses.copy())
            params.append(Int(py=owned))
            _ = mamba3_prefill_session_run(owner, NEURAL_MAMBA3_OWNED_FORWARD, addresses, params)
        else:
            for i in range(len(weights)):  # driver-loop: over fixed Mamba3 weight addresses
                addresses.append(Int(py=weights[i]._addr))
            addresses.append(Int(py=out._addr))
            addresses.extend(report_addresses.copy())
            _ = mamba3_prefill_session_run(owner, NEURAL_MAMBA3_SESSION_FORWARD, addresses, params)
        _ = owner^
    else:
        var addresses = Python.list(x._addr)
        for i in range(len(weights)):  # driver-loop: over fixed Mamba3 weight addresses
            _ = py_call(addresses.append, weights[i]._addr)
        _ = py_call(addresses.append, out._addr)
        for i in range(4):  # driver-loop: over the four report addresses
            _ = py_call(addresses.append, report_addresses[i])
        var native = py_get(fields, "_native_mamba_block_provider_cache")
        if native is not None:
            _ = neural_descriptor_execute(native, 5, _neural_int_words(addresses), params, [])
        else:
            _ = py_call(_native_function(entries, "mamba3_forward_fresh"), addresses, py_ints(params))
    _ = x
    _ = weights
    return Python.tuple(out, reports[0], reports[1], reports[2], reports[3])


def _prefill_mamba3(block: PythonObject, x: PythonObject, entries: PythonObject, route: Mamba3StateRoute) raises -> PythonObject:
    # Pass the captured fields explicitly: a PythonObject(None) default is
    # evaluated at compile time and needs a live interpreter.
    var fields = py_call(native_loaded_module("builtins").object.__getattribute__, block, "__dict__")
    var result = _prefill_mamba3_reports(block, x, entries, fields, route)
    var names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(4):  # driver-loop: over report result metadata slots
        py_setitem(fields, names[i], result[i + 1])
    return result[0]


def neural_prefill_forward(block: PythonObject, input: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = native_loaded_module("builtins")
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    var kind = String(fields["_native_kind"])
    if kind != "transformer" and kind != "mamba3":
        raise Error("neural prefill tape requires TransformerBlock or Mamba3Block")
    var selection = neural_training_selection(fields, context)
    var entries = selection.entries
    var route = Mamba3StateRoute(selection.key.mode, selection.key.vendor, selection.key.vendor == 0)
    var x = _neural_input(input, Int(py=fields["d_model"]), False, context)
    if kind == "transformer":
        # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
        # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
        # (`transformer_forward_tape`: a TransformerBlock prefill keeps no
        # tape; its backward is `TransformerBlock.backward` on the step op.)
        return Python.tuple(neural_block_forward_binding(block, x, PythonObject(None), False, PythonObject(None), context), PythonObject(None))
    if not _prefill_tape_enabled(entries):
        return Python.tuple(_prefill_mamba3(block, x, entries, route), PythonObject(None))
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=x.shape[2])
    var out = generation_array([b, l, d])
    var addresses = Python.list(x._addr)
    var weights = fields["_w"]
    var shapes = bi.list()
    for i in range(len(weights)):  # driver-loop: over immutable tape input weights
        _ = py_call(addresses.append, weights[i]._addr)
        _ = py_call(shapes.append, weights[i].shape)
    _ = py_call(addresses.append, out._addr)
    var params = py_ints([b, l, d])
    var native = py_get(fields, "_native_mamba_block_provider_cache")
    var tape_base = 10
    var handle = PythonObject(None)
    var opaque = 0
    if native is not None:
        handle = neural_descriptor_construct(native, _native_function(entries, "mamba3_tape_create_native_v1"))
        opaque = neural_descriptor_handle(native, _native_function(entries, "mamba3_tape_handle_native_v1"), handle)
        _ = neural_descriptor_reserved_execute(native, handle, opaque, tape_base, tape_base + 1, tape_base + 2,
            _neural_int_words(addresses), _neural_int_words(params), [])
    else:
        handle = py_call(_native_function(entries, "mamba3_forward_tape"), addresses, params)
    return Python.tuple(out, Python.tuple(entries, handle, x.shape, py_call(bi.tuple, shapes), native, PythonObject(opaque), PythonObject(tape_base)))


def neural_prefill_backward(block: PythonObject, tape: PythonObject, grad_output: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = native_loaded_module("builtins")
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    if String(fields["_native_kind"]) != "mamba3":
        raise Error("TypeError: immutable tape belongs to another block family")
    var dims = List[Int]()
    for i in range(len(tape[2])):  # driver-loop: over immutable input shape metadata
        dims.append(Int(py=tape[2][i]))
    var dy = _model_f32(grad_output, "grad_output", context)
    _model_shape(dy, dims, "grad_output")
    var gradients = Python.list(generation_array(dims))
    var shapes = tape[3]
    for i in range(len(shapes)):  # driver-loop: over immutable parameter gradient shapes
        var shape = List[Int]()
        for axis in range(len(shapes[i])):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=shapes[i][axis]))
        _ = py_call(gradients.append, generation_array(shape))
    var addresses = Python.list(dy._addr)
    for i in range(len(gradients)):  # driver-loop: over gradient output addresses
        _ = py_call(addresses.append, gradients[i]._addr)
    if len(tape) == 7 and tape[4] is not None:
        var base = Int(py=tape[6])
        if base != 10:
            raise Error("TypeError: immutable tape belongs to another block family")
        _ = neural_descriptor_reserved_execute(tape[4], tape[1], Int(py=tape[5]), base, base + 1, base + 3,
            _neural_int_words(addresses), [], [])
    else:
        _ = py_call(_native_function(tape[0], "mamba3_backward_tape"), tape[1], addresses)
    _ = dy^  # the backward entry read `dy` by address
    var result = bi.dict()
    py_setitem(result, "x", gradients[0])
    var names = py_call(bi.object.__getattribute__, block, "_W_NAMES")
    for i in range(len(names)):  # driver-loop: over named gradient result metadata
        py_setitem(result, names[i], gradients[i + 1])
    return result


def neural_prefill_close(tape: PythonObject) raises:
    if tape is None:
        return
    if len(tape) == 7 and tape[4] is not None:
        var base = Int(py=tape[6])
        _ = neural_descriptor_reserved_execute(tape[4], tape[1], Int(py=tape[5]), base, base + 1, base + 4,
            [], [], [])
    else:
        _ = py_call(_native_function(tape[0], "mamba3_close_tape"), tape[1])


# TOMBSTONE: bpe_decode, causal_lm_admit, neural_prefill_call, optimizer_finalize exports (dead native entries: no live caller of the export) deleted 2026-10-26 by 5206fc31b; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_tokenizer_host.patch

