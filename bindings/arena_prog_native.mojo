# SPDX-License-Identifier: Apache-2.0
# Native owner of the shared arena program runner: `_Prog.run` of the x_prep
# family (_expansion_prep) and the x_metrics programs the native metrics owners build.
#
# Python builds a program (its arena layout, its input buffers and its stage
# words) and makes ONE call here. This file owns everything the run does:
# the binding entries of the requested numeric mode (connected at import by
# python/mojolearn/_native_arena_prog.py), the host arena and output stores,
# the input copies, the resident-store and device-gathered inputs
# (core/device_store.mojo through the family's *_dev_put / *_dev_take_rows),
# the input/output range descriptors of the shared ranges runner
# (core/arena_io.mojo), the stage word packing, the run itself and the
# release of every device slot it acquired. It replaces the Python bodies of
# `_expansion_prep._Prog.run` and `_expansion_metrics._execute` step for
# step: the same entries are called with the same words, so no bit moves.
from bindings.native_array_storage import native_array_zeros
from std.memory import unsafe_memcpy
from core.metrics_program_plan import MetricProgramPlan
from std.os import getenv
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.native_error_class import error_without_class
from bindings.optimizer_public_words import optimizer_python_error, stash_python_error, PYTHON_ERROR_PENDING
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address
from bindings.x_prep_native_owner_run import xprep_r3
from bindings.x_prep_native_owner import xprep_address, _resident, _cached_slot, _resident_scope, _resident_cache, _cache_state, _cached_slot_in
from bindings.x_prep_native_inputs import _values, _rows_fields, _dense_values
from bindings.x_prep_native_owner import xprep_drop_resident
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.cpython_table import cpython_table
from bindings.p3_array_arena_records import P3_DICT_NATIVE_STRING_KEYS, p3_dict_admitted, p3_dict_admit
from bindings.arena_prog_run import _switch, _add_input, _merge_pairs, _pad, MetricArenaOwner, MetricProgramInput, metrics_plan_execute
from std.python._cpython import GILReleased
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime INT32_MAX = 2147483647
# x_prep: a host block of at least this many words is an anonymous mapping
# (lane prep-apple3 `mapped`); _expansion_prep._MAP_MIN_WORDS.
comptime PREP_MAP_MIN_WORDS = 262144
# x_prep: an input of at least this many words goes up from its own buffer
# into a slot freed after the run (MOJOLEARN_XPREP_DIRECT);
# _expansion_prep._DIRECT_MIN_WORDS.
comptime PREP_DIRECT_MIN_WORDS = 1048576
# Words of one stage record: op, total and _PARAMS (14) parameters.
comptime STAGE_WORDS = 16


def _entry_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    # Call a connected compiled entry and keep the exception class it raised
    # (ValueError, RuntimeError, ...) across this native boundary.
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _drop_pending(error: Error):
    # A cleanup failure replaces the run's own error (Python's `finally`):
    # discard a stashed caller exception so it never resurfaces later.
    if String(error) != PYTHON_ERROR_PENDING:
        return
    _ = optimizer_python_error(error)
    try:
        ref cpy = Python().cpython()
        cpython_table()[].PyErr_Clear()
    except ignored:
        _ = ignored^


def _family_dictionary(value: PythonObject, builtins: PythonObject) raises:
    # Exact dictionaries and native key iteration cannot enter __getitem__
    # or a colliding key's __eq__. The shared key helper caches unchanged maps.
    if py_type(value) is not builtins.dict:
        raise Error("TypeError: arena program metadata requires exact string-keyed dictionaries")
    if not p3_dict_admitted(P3_DICT_NATIVE_STRING_KEYS, value):
        if not native_namespace_string_keys(value._obj_ptr, builtins.str):
            raise Error("TypeError: arena program metadata requires exact string-keyed dictionaries")
        p3_dict_admit(P3_DICT_NATIVE_STRING_KEYS, value)


def _family_entries(mode: PythonObject, family: String, context: PythonObject) raises -> PythonObject:
    # Keep live context and mode selection, admitting each mapping before
    # its first keyed read. Unused modes retain their original error records.
    var b = native_loaded_module("builtins")
    _family_dictionary(context, b)
    var types = context["prog_types"]
    _family_dictionary(types, b)
    if types["str"] is not b.str:
        raise Error("TypeError: arena program string metadata requires the native str type")
    var selected = mode
    if selected is None:
        var backend = context["backend_fields"]
        _family_dictionary(backend, b)
        selected = backend["_DEFAULT_MODE"]
    if py_type(selected) is not b.str:
        var refusal = context["prog_mode_refusal"]
        if py_type(refusal) is not b.str:
            raise Error("TypeError: arena program mode refusal requires a native string")
        raise Error("ValueError: " + String(refusal))
    var word = String(py=selected)
    if word != "fast" and word != "identical":
        selected = py_call(py_call(selected.strip).lower)
        word = String(py=selected)
    if word != "fast" and word != "identical":
        var refusal = context["prog_mode_refusal"]
        if py_type(refusal) is not b.str:
            raise Error("TypeError: arena program mode refusal requires a native string")
        raise Error("ValueError: " + String(refusal))
    var bindings = context["prog_bindings"]
    _family_dictionary(bindings, b)
    var entries = bindings[selected]
    if py_type(entries) is b.str:
        raise Error(String(entries))
    _family_dictionary(entries, b)
    var connected_family = entries["prog_family"]
    if py_type(connected_family) is not b.str:
        raise Error("TypeError: arena program family metadata requires a native string")
    if String(connected_family) != family:
        raise Error("TypeError: arena program family does not match its connected binding")
    return entries



def _store(words: Int, code: String, mapped: Bool, context: PythonObject) raises -> PythonObject:
    # (store, address) of max(words, 1) zeroed 4-byte words of `code`; a
    # large block is a memoryview of an anonymous mapping when `mapped`.
    var n = max(words, 1)
    var types = context["prog_types"]
    var store = PythonObject(None)
    if mapped and n >= PREP_MAP_MIN_WORDS and xprep_r3("mapped"):
        var mapping = py_call(context["prog_mmap"], -1, 4 * n)
        store = py_call(py_call(types["memoryview"], mapping).cast, PythonObject(code))
    else:
        store = native_array_zeros(String(code), n)
    return Python.tuple(store, xprep_address(store))


def _int(value: PythonObject, context: PythonObject) raises -> Int:
    if py_type(value) is not context["prog_types"]["int"]:
        raise Error("TypeError: arena program layout words must be native integers")
    return Int(py=value)


def _is_rows(value: PythonObject, context: PythonObject) raises -> Bool:
    return py_type(value) is context["maxabs_rows_class"]


def _rows_gather(value: PythonObject, entries: PythonObject, prefix: String, mut owned: List[Int],
                 context: PythonObject) raises -> Int:
    # The store slot of a DeviceRows' rows gathered on the device out of its
    # base, put once through the active resident() scope (any size) or
    # directly into a slot this run owns (`_arena_io.rows_slot`).
    var fields = _rows_fields(value, context)
    var base = _dense_values(fields["base"], context)
    var indices = fields["indices"]
    var n = Int(py=indices.size)
    if n < 1:
        return -1
    var d = Int(py=fields["row_words"])
    var cache = _resident(entries, 0, True, context, prefix)
    var base_slot = -1
    if cache is not None:
        base_slot = _cached_slot(cache, base, entries, context, prefix)
    else:
        base_slot = Int(py=_entry_call(entries[PythonObject(prefix + "_dev_put")],
                                       Python.tuple(classical_address(base), base.size)))
        owned.append(base_slot)
    var slot = Int(py=_entry_call(entries[PythonObject(prefix + "_dev_take_rows")],
                                  Python.tuple(base_slot, d, classical_address(indices), n)))
    owned.append(slot)
    _ = base^
    return slot


def _can_gather(entries: PythonObject, prefix: String) raises -> Bool:
    return (py_get(entries, PythonObject(prefix + "_dev_take_rows")) is not None
            and py_get(entries, PythonObject(prefix + "_dev_put")) is not None
            and py_get(entries, PythonObject(prefix + "_dev_free")) is not None)


def _release(owned: List[Int], entries: PythonObject, prefix: String) -> String:
    # Free every slot this run acquired, newest first, even when one fails.
    var failures = String()
    for i in range(len(owned)):  # driver-loop: over the device slots one program run acquired
        var slot = owned[len(owned) - 1 - i]
        try:
            _ = py_call_typed(entries[PythonObject(prefix + "_dev_free")], slot)
        except error:
            failures += ("" if failures.byte_length() == 0 else "; ") + "slot " + String(slot) + ": " + String(error)
    return failures


# ---------------------------------------------------------------- x_prep


def _prep_scratch(value: PythonObject, sbase: Int, obase: Int, context: PythonObject) raises -> Int:
    # A stage parameter: a plain int, or a `_Scratch` offset past the host
    # arena (kind "s": device scratch at sbase, kind "o": the output at obase).
    if py_type(value) is context["prog_types"]["int"]:
        return Int(py=value)
    if py_type(value) is not context["prog_scratch_class"]:
        raise Error("TypeError: x_prep stage parameters must be integers or scratch offsets")
    var off = _int(py_call(context["prog_scratch_off"], value), context)
    var kind = py_call(context["prog_scratch_kind"], value)
    return off + (sbase if String(kind) == "s" else obase)


def _prep_run(prog: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # The stage records come from the program's `_stages` list of lists.
    return _prep_run_impl(prog, mode, context, List[Int32](), False)


def prep_run_words(prog: PythonObject, mode: PythonObject, context: PythonObject,
                   words: List[Int32]) raises -> PythonObject:
    """The same runner for a native owner whose stage records are already
    native words (16 per stage, scratch words already placed): the owner no
    longer builds a Python list of 16 ints per stage for this runner to
    type-check and convert word by word (perf3-prep-shared)."""
    return _prep_run_impl(prog, mode, context, words, True)


def _prep_run_impl(prog: PythonObject, mode: PythonObject, context: PythonObject,
                   native_words: List[Int32], has_native: Bool) raises -> PythonObject:
    # Classifier owners call this runner directly, bypassing arena_prog_c.
    # Connect already-loaded native NumPy metadata in this runner's own
    # context before a DeviceRows snapshot admits an ndarray-owned view.
    # This never imports NumPy and is a no-op in a NumPy-free process.
    classical_connect_numpy(context)
    var profile = String(getenv("MOJOLEARN_XPREP_PROFILE", "0")) == "1"
    var t_run = Int(perf_counter_ns()) if profile else 0
    var types = context["prog_types"]
    var entries = _family_entries(mode, "x_prep", context)
    var fields = py_fields(prog)
    var H = _int(fields["size"], context)
    var sc = _int(fields["scratch_size"], context)
    var on = 0 if fields["out_size"] is None else _int(fields["out_size"], context)
    var out_code = String(fields["_out_code"])
    var has_out = py_get(entries, "x_prep_run_out") is not None
    var dev_out = has_out and on > 0 and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
    var has_scratch_entry = sc > 0 and not has_out and py_get(entries, "x_prep_run_scratch") is not None
    var dev_scratch = sc > 0 and (has_out or has_scratch_entry)
    if H + sc + on > INT32_MAX:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    # layout: the host arena, then (host) the output unless the device keeps
    # it, then the scratch, then (device) the output
    var ha = H + (0 if dev_out else on)
    var sbase = ha
    var obase = ha + sc if dev_out else H
    var host_words = ha + (0 if dev_scratch else sc)
    var arena = _store(host_words, "f", True, context)
    var base = Int(py=arena[1])
    var t_alloc = Int(perf_counter_ns()) if profile else 0
    var ranges = (dev_scratch or sc == 0) and _switch("MOJOLEARN_ARENA_RANGES") and py_get(entries, "x_prep_run_ranges") is not None
    var direct_ok = (py_get(entries, "x_prep_dev_put") is not None and py_get(entries, "x_prep_dev_free") is not None
                     and _switch("MOJOLEARN_XPREP_DIRECT"))
    var inout_pairs = List[Int]()
    var inout = fields["_inout"]
    for k in range(len(inout)):  # driver-loop: over the program's in-out input ranges
        inout_pairs.append(_int(inout[k][0], context))
        inout_pairs.append(_int(inout[k][1], context))
    var inputs = fields["_inputs"]
    var ins = List[Int32]()
    var in_spans = Python.list()
    var owned = List[Int](capacity=2 * len(inputs))
    var stages = List[Int32]()
    var out = PythonObject(None)
    var nst = 0
    var t_in = 0
    # Per run, not per input (perf3-prep-shared, the helpers perf2-prep added
    # for xprep_run): the resident scope admission and the resident cache's
    # state admission are read once, at the first input that needs them,
    # with the same checks and refusals `_resident` / `_cached_slot` ran per
    # input. Within one run only this runner inserts native keys.
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    # The scope's presence is the read-once flag (its value may itself be None).
    var scope = Optional[PythonObject]()
    var cache_state = Optional[PythonObject]()
    # In-out inputs a stage rewrites on the host (p3-array-arena, hazard D):
    # after the run their resident device copies (taken by an earlier run in
    # the same scope) are stale and are dropped.
    var inout_values = List[PythonObject]()
    try:
        for k in range(len(inputs)):  # driver-loop: over the program's input buffers
            var record = inputs[k]
            var off = _int(record[0], context)
            var value = record[1]
            var size = Int(py=value.size)
            if size == 0:
                continue
            var is_inout = False
            for j in range(len(inout_pairs) // 2):  # driver-loop: over the program's in-out input ranges
                if inout_pairs[2 * j] == off and inout_pairs[2 * j + 1] == off + size:
                    is_inout = True
            if is_inout:
                inout_values.append(value)
            var src = -1
            if _is_rows(value, context):
                # device-rows input (lane cpu4-misc): the fold's rows gathered
                # on the device; a binding without the gather copies host rows
                if ranges and _can_gather(entries, "x_prep"):
                    src = _rows_gather(value, entries, "x_prep", owned, context)
                if src < 0:
                    value = _values(value, context, True)
            if src < 0 and ranges and not is_inout:
                if not scope:
                    scope = _resident_scope(entries, context, "x_prep")
                if scope.value() is not None and size >= Int(py=scope.value()[0]):
                    if not cache_state:
                        var cache = _resident_cache(entries, scope.value()[1], context, "x_prep")
                        cache_state = _cache_state(cache, entries, context, "x_prep")
                    src = _cached_slot_in(cache_state.value(), value, entries, "x_prep")
                elif direct_ok and size >= PREP_DIRECT_MIN_WORDS:
                    # TOMBSTONE: lane fg-knn-nb G3 (MOJOLEARN_XPREP_NO_SLOT_HOP, x_prep_run_ranges_host host spans) deleted 2026-10-09
                    # (lane postmerge-act-2): slower on the average, gaussian-nb istella NV 1.25x / AMD 1.00x, taxi 1.02x / 1.03x
                    # (nv n0668->n0630, amd a1066->a1090), accuracy SAME, same digests; code recoverable at main 5c137b55e.
                    src = Int(py=_entry_call(entries["x_prep_dev_put"], Python.tuple(classical_address(value), size)))
                    owned.append(src)
            if src < 0:
                unsafe_memcpy(dest=F32P(unsafe_from_address=base + 4 * off),
                       src=F32P(unsafe_from_address=Int(py=classical_address(value))), count=size)
            _add_input(ins, off, off + size, src)
            if not is_inout:
                _ = py_call(in_spans.append, Python.tuple(off, off + size))
        py_setitem(fields, "_in_spans", in_spans)
        if has_native:
            if len(native_words) % STAGE_WORDS != 0:
                raise Error("ValueError: x_prep: a stage record must hold 16 words")
            nst = len(native_words) // STAGE_WORDS
            stages = native_words.copy()
        else:
            var stage_list = fields["_stages"]
            nst = len(stage_list)
            for i in range(nst):  # driver-loop: over the program's stage records
                var stage = stage_list[i]
                if len(stage) != STAGE_WORDS:
                    raise Error("ValueError: x_prep: a stage record must hold 16 words")
                for j in range(STAGE_WORDS):  # driver-loop: over one stage record's fixed words
                    stages.append(Int32(_prep_scratch(stage[j], sbase, obase, context)))
        if nst == 0:
            stages.append(0)
        var pa = Int(stages.unsafe_ptr())
        py_setitem(fields, "_out", PythonObject(None))
        py_setitem(fields, "_out_at", obase)
        t_in = Int(perf_counter_ns()) if profile else 0
        if ranges:
            # the shared ranges runner (core/arena_io.mojo): the inputs go up,
            # the rest of the host arena starts zero on the device, and
            # everything but the inputs (and the in-out inputs) comes back
            var pairs = List[Int]()
            var at = 0
            for i in range(len(ins) // 3):  # driver-loop: complement of the input ranges over the host arena
                var lo = Int(ins[3 * i])
                if lo > at:
                    pairs.append(at)
                    pairs.append(lo)
                at = max(at, Int(ins[3 * i + 1]))
            if at < ha:
                pairs.append(at)
                pairs.append(ha)
            for j in range(len(inout_pairs)):  # driver-loop: the in-out input ranges come back too
                pairs.append(inout_pairs[j])
            var outs = _merge_pairs(pairs^)
            var n_in = len(ins) // 3
            var n_out = len(outs) // 4
            var ia = _pad(ins.copy(), [Int32(0), Int32(0), Int32(-1)])
            var oa = _pad(outs^, [Int32(0), Int32(0), Int32(-1), Int32(1)])
            var out_addr = 0
            if dev_out:
                var made = _store(on, out_code, True, context)
                out = made[0]
                out_addr = Int(py=made[1])
            _ = _entry_call(entries["x_prep_run_ranges"], Python.tuple(
                base, pa, out_addr, Python.tuple(ha, sc if dev_scratch else 0, on if dev_out else 0, nst),
                Python.tuple(Int(ia.unsafe_ptr()), n_in, Int(oa.unsafe_ptr()), n_out)))
            # The tuples hold raw addresses: keep both descriptor lists live
            # until the entry has returned.
            _ = ia^
            _ = oa^
            py_setitem(fields, "_out", out)
        elif dev_out:
            var made = _store(on, out_code, True, context)
            out = made[0]
            _ = _entry_call(entries["x_prep_run_out"], Python.tuple(
                base, pa, made[1], Python.tuple(ha, sc if dev_scratch else 0, on, nst)))
            py_setitem(fields, "_out", out)
        elif dev_scratch:
            if has_out:
                _ = _entry_call(entries["x_prep_run_out"], Python.tuple(base, pa, 0, Python.tuple(ha, sc, 0, nst)))
            else:
                _ = _entry_call(entries["x_prep_run_scratch"], Python.tuple(base, ha, sc, pa, nst))
        else:
            _ = _entry_call(entries["x_prep_run"], Python.tuple(base, host_words, pa, nst))
    except error:
        var cleanup = _release(owned, entries, "x_prep")
        var dropped = xprep_drop_resident(inout_values, entries, context, "x_prep")
        if dropped.byte_length() > 0:
            cleanup += ("" if cleanup.byte_length() == 0 else "; ") + dropped
        if cleanup.byte_length() > 0:
            _drop_pending(error)
            raise Error("RuntimeError: x_prep operation failed: " + String(error) + "; cleanup also failed: " + cleanup)
        raise error
    var cleanup = _release(owned, entries, "x_prep")
    var dropped = xprep_drop_resident(inout_values, entries, context, "x_prep")
    if dropped.byte_length() > 0:
        cleanup += ("" if cleanup.byte_length() == 0 else "; ") + dropped
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_prep slot cleanup failed: " + cleanup)
    py_setitem(fields, "arena", arena[0])
    if profile:
        # one line per program (the device's XPPHASE lines come in the same order)
        var names = context["prog_op_names"]
        var line = String("XPPROG ops=")
        for i in range(nst):  # driver-loop: stage names for the profiling line
            var name = py_get(names, Int(stages[i * STAGE_WORDS]))
            line += ("" if i == 0 else "+") + (String(name) if name is not None else String(Int(stages[i * STAGE_WORDS])))
        var done = Int(perf_counter_ns())
        print(line + " arena=" + String(ha) + " scratch=" + String(sc) + " out=" + String(on)
              + " dev_out=" + String(Int(dev_out)) + " ranges=" + String(Int(ranges))
              + " alloc_s=" + String(Float64(t_alloc - t_run) / 1.0e9)
              + " inputs_s=" + String(Float64(t_in - t_alloc) / 1.0e9)
              + " call_s=" + String(Float64(done - t_in) / 1.0e9))
    _ = stages^
    return prog


# ---------------------------------------------------------------- x_metrics


# TOMBSTONE: `_metrics_run`, `_metrics_run_fields`, `_metrics_download` and `_metrics_execute` (the Python-record
# metric runner and the Python `x_metrics_run*` entry calls; no caller since the "x_metrics_run" door arm was
# deleted by lane pyout-metrics-cnn) deleted 2026-11-12 by lane entangled-programs; the native plan run is the
# typed `metrics_plan_execute` (bindings/arena_prog_run.mojo); code recoverable at e561a66d6b.


def _metrics_run_plan(plan: MetricProgramPlan, inputs: List[PythonObject], mode: PythonObject,
                      context: PythonObject) raises -> MetricArenaOwner:
    """UNPACK + RUN of a native metric plan (lane entangled-programs,
    2026-11-12). UNPACK: the selected mode's x_metrics image, its program
    dispatch (connected once per factory), the route words and each input's
    address (a DeviceRows input materialized on the host: x_metrics connects
    no row gather or resident store). RUN: `metrics_plan_execute`, typed, the
    GIL released, every input owner held across it. The arena is native."""
    classical_connect_numpy(context)
    var entries = _family_entries(mode, "x_metrics", context)
    var provider = neural_descriptor_connect_factory_once(entries["metrics_program_native_api_v1"],
        String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var ranges = _switch("MOJOLEARN_ARENA_RANGES") and py_get(entries, "x_metrics_run_ranges") is not None
    var output_ranges = py_get(entries, "x_metrics_run_out") is not None
    if len(inputs) != len(plan.input_offsets):
        raise Error("ValueError: x_metrics: the program's inputs do not match its layout")
    var held = List[PythonObject](capacity=len(inputs))
    var records = List[MetricProgramInput](capacity=len(inputs))
    var scope_checked = False
    for k in range(len(inputs)):  # driver-loop: over the program's input buffers
        var value = inputs[k]
        var size_k = Int(py=value.size)
        if size_k == 0:
            continue
        if _is_rows(value, context):
            value = _values(value, context, True)
        if ranges and not scope_checked:
            # The thread's resident scope keeps its admission checks; an
            # x_metrics image has no resident store, so no scope applies.
            scope_checked = True
            if _resident_scope(entries, context, "x_metrics") is not None:
                raise Error("RuntimeError: arena family x_metrics has no resident store")
        records.append(MetricProgramInput(plan.input_offsets[k], Int(py=classical_address(value)), size_k))
        held.append(value)
    var arena = MetricArenaOwner(0)
    try:
        with GILReleased(Python()):
            arena = metrics_plan_execute(entry, plan, records, ranges, output_ranges)
    finally:
        _ = held^
        _ = records^
        _ = provider
        _ = entries
    return arena^


# ---------------------------------------------------------------- the door


def arena_prog_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var prog = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        if py_type(prog) is not context["prog_class"]:
            raise Error("TypeError: arena program run requires its connected program class")
        var result = PythonObject(None)
        if operation == "x_prep_run":
            result = _prep_run(prog, args[0], context)
        # TOMBSTONE: the "x_metrics_run" arm (Python `_expansion_metrics._Prog` programs; every x_metrics program
        # is now built and run natively through `_metrics_run_plan`/`_metrics_run`, and x_metrics connects no
        # Python program class) deleted 2026-11-11 by lane pyout-metrics-cnn; code recoverable at 6f82d0255d.
        else:
            raise Error("ValueError: unknown native arena program operation")
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if message.startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if message.startswith("AssertionError:"):
            return raise_python_exception(error_without_class(error, "AssertionError"), ExceptionType("PyExc_AssertionError"))
        if message.startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        if (message == PYTHON_ERROR_PENDING or message.startswith("ValueError:") or message.startswith("TypeError:")
                or message.startswith("RuntimeError:") or message.startswith("NotImplementedError:")
                or message.startswith("OverflowError:")):
            return optimizer_python_error(error)
        return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
