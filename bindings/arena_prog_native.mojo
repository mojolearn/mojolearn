# SPDX-License-Identifier: Apache-2.0
# Native owner of the shared arena program runner: `_Prog.run` of the x_prep
# family (_expansion_prep) and of the x_metrics family (_expansion_metrics).
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
from std.memory import memcpy
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
from bindings.x_prep_native_owner import xprep_r3, xprep_address, _resident, _cached_slot
from bindings.x_prep_native_inputs import _values, _rows_fields, _dense_values

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


def _switch(name: String) -> Bool:
    # Unset or anything but "0": on (the Python `_switch` / `ranges_enabled`).
    return String(getenv(name, "1").strip()) != "0"


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
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
    except ignored:
        _ = ignored^


def _family_entries(mode: PythonObject, family: String, context: PythonObject) raises -> PythonObject:
    # The family's entries for `mode` (None: the package default), connected
    # at import for both numeric modes. A mode whose binding did not connect
    # raises the error its connection raised.
    var types = context["prog_types"]
    var selected = mode
    if selected is None:
        selected = context["backend_fields"]["_DEFAULT_MODE"]
    var refusal = String(context["prog_mode_refusal"])
    if py_type(selected) is not types["str"]:
        raise Error("ValueError: " + refusal)
    selected = py_call(py_call(selected.strip).lower)
    if String(selected) != "fast" and String(selected) != "identical":
        raise Error("ValueError: " + refusal)
    var entries = context["prog_bindings"][selected]
    if py_type(entries) is types["str"]:
        raise Error(String(entries))
    if String(entries["prog_family"]) != family:
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


def _add_input(mut ins: List[Int32], lo: Int, hi: Int, src: Int) raises:
    # `_arena_io.input_ranges`: ascending disjoint spans, adjacent host spans
    # (src -1) merged; an overlap is a layout bug.
    var count = len(ins) // 3
    if count > 0:
        var last_hi = Int(ins[3 * count - 2])
        if lo < last_hi:
            raise Error("AssertionError: arena ranges: inputs overlap at [" + String(lo) + ", " + String(hi) + ")")
        if src < 0 and Int(ins[3 * count - 1]) < 0 and lo == last_hi:
            ins[3 * count - 2] = Int32(hi)
            return
    ins.append(Int32(lo))
    ins.append(Int32(hi))
    ins.append(Int32(src))


def _merge_pairs(var pairs: List[Int]) -> List[Int32]:
    # `_arena_io.output_ranges`: [lo, hi, -1, 1] quads of the sorted, merged
    # non-empty pairs.
    var n = len(pairs) // 2
    for i in range(1, n):  # driver-loop: insertion sort of the program's output range pairs
        var lo = pairs[2 * i]
        var hi = pairs[2 * i + 1]
        var j = i - 1
        while j >= 0 and (pairs[2 * j] > lo or (pairs[2 * j] == lo and pairs[2 * j + 1] > hi)):  # driver-loop: insertion shift over range pairs
            pairs[2 * j + 2] = pairs[2 * j]
            pairs[2 * j + 3] = pairs[2 * j + 1]
            j -= 1
        pairs[2 * j + 2] = lo
        pairs[2 * j + 3] = hi
    var out = List[Int32]()
    for i in range(n):  # driver-loop: merge of the sorted output range pairs
        var lo = pairs[2 * i]
        var hi = pairs[2 * i + 1]
        if hi <= lo:
            continue
        var m = len(out) // 4
        if m > 0 and lo <= Int(out[4 * m - 3]):
            out[4 * m - 3] = Int32(max(Int(out[4 * m - 3]), hi))
        else:
            out.append(Int32(lo))
            out.append(Int32(hi))
            out.append(-1)
            out.append(1)
    return out^


def _pad(var words: List[Int32], fill: List[Int32]) -> List[Int32]:
    # A packed descriptor list is never empty (the entry reads a pointer).
    if len(words) == 0:
        return fill.copy()
    return words^


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
            var src = -1
            if _is_rows(value, context):
                # device-rows input (lane cpu4-misc): the fold's rows gathered
                # on the device; a binding without the gather copies host rows
                if ranges and _can_gather(entries, "x_prep"):
                    src = _rows_gather(value, entries, "x_prep", owned, context)
                if src < 0:
                    value = _values(value, context, True)
            if src < 0 and ranges and not is_inout:
                var cache = _resident(entries, size, False, context, "x_prep")
                if cache is not None:
                    src = _cached_slot(cache, value, entries, context, "x_prep")
                elif direct_ok and size >= PREP_DIRECT_MIN_WORDS:
                    # TOMBSTONE: lane fg-knn-nb G3 (MOJOLEARN_XPREP_NO_SLOT_HOP, x_prep_run_ranges_host host spans) deleted 2026-10-09
                    # (lane postmerge-act-2): slower on the average, gaussian-nb istella NV 1.25x / AMD 1.00x, taxi 1.02x / 1.03x
                    # (nv n0668->n0630, amd a1066->a1090), accuracy SAME, same digests; code recoverable at main 5c137b55e.
                    src = Int(py=_entry_call(entries["x_prep_dev_put"], Python.tuple(classical_address(value), size)))
                    owned.append(src)
            if src < 0:
                memcpy(dest=F32P(unsafe_from_address=base + 4 * off),
                       src=F32P(unsafe_from_address=Int(py=classical_address(value))), count=size)
            _add_input(ins, off, off + size, src)
            if not is_inout:
                _ = py_call(in_spans.append, Python.tuple(off, off + size))
        py_setitem(fields, "_in_spans", in_spans)
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
        if cleanup.byte_length() > 0:
            _drop_pending(error)
            raise Error("RuntimeError: x_prep operation failed: " + String(error) + "; cleanup also failed: " + cleanup)
        raise error
    var cleanup = _release(owned, entries, "x_prep")
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


def _metrics_download(fields: PythonObject, size: Int, context: PythonObject, mut known: Bool) raises -> List[Int32]:
    # `_Prog._download`: the [lo, hi, count, mult] ranges the device brings
    # back; the declared outputs (uncounted ones merged), else the arena
    # less the inputs and scratch. `known` False: the whole arena.
    var outs = fields["_outs"]
    var result = List[Int32]()
    known = True
    if outs is not None:
        var plain = List[Int]()
        for i in range(len(outs)):  # driver-loop: over the program's declared output ranges
            if _int(outs[i][2], context) < 0:
                plain.append(_int(outs[i][0], context))
                plain.append(_int(outs[i][1], context))
        # Merged with `lo <= last hi`, as _download does (adjacent ranges join).
        result = _merge_pairs(plain^)
        for i in range(len(outs)):  # driver-loop: the counted output ranges, in declaration order
            if _int(outs[i][2], context) >= 0:
                for j in range(4):  # driver-loop: one range descriptor's four words
                    result.append(Int32(_int(outs[i][j], context)))
        return result^
    var skip = fields["_skip"]
    if len(skip) == 0:
        known = False
        return result^
    var pairs = List[Int]()
    for i in range(len(skip)):  # driver-loop: over the program's input and scratch ranges
        pairs.append(_int(skip[i][0], context))
        pairs.append(_int(skip[i][1], context))
    var n = len(pairs) // 2
    for i in range(1, n):  # driver-loop: insertion sort of the input and scratch ranges
        var lo = pairs[2 * i]
        var hi = pairs[2 * i + 1]
        var j = i - 1
        while j >= 0 and (pairs[2 * j] > lo or (pairs[2 * j] == lo and pairs[2 * j + 1] > hi)):  # driver-loop: insertion shift over range pairs
            pairs[2 * j + 2] = pairs[2 * j]
            pairs[2 * j + 3] = pairs[2 * j + 1]
            j -= 1
        pairs[2 * j + 2] = lo
        pairs[2 * j + 3] = hi
    var at = 0
    for i in range(n):  # driver-loop: complement of the input and scratch ranges
        if pairs[2 * i] > at:
            result.append(Int32(at))
            result.append(Int32(pairs[2 * i]))
            result.append(-1)
            result.append(1)
        at = max(at, pairs[2 * i + 1])
    if at < size:
        result.append(Int32(at))
        result.append(Int32(size))
        result.append(-1)
        result.append(1)
    return result^


def _metrics_run(prog: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve the public Prog API and its identity-return contract. Native
    # metric owners supply their already-built field dictionary directly.
    _ = _metrics_run_fields(py_fields(prog), mode, context)
    return prog


def _metrics_run_fields(fields: PythonObject, mode: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # Metric programs are native structural data, not Python class instances.
    # Admit dictionary/key storage before any layout read can perform lookup.
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: native metric program fields require an exact dictionary")
    var keys = py_call(py_type(Python.list()), fields)
    for i in range(len(keys)):  # driver-loop: passive native program layout keys, before dictionary lookup
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: native metric program fields require exact string keys")
    classical_connect_numpy(context)
    var entries = _family_entries(mode, "x_metrics", context)
    var size = _int(fields["size"], context)
    var arena = _store(size, "f", False, context)
    var base = Int(py=arena[1])
    var stage_list = fields["_stages"]
    var nst = len(stage_list)
    var stages = List[Int32]()
    for i in range(nst):  # driver-loop: over the program's stage records
        var stage = stage_list[i]
        if len(stage) != STAGE_WORDS:
            raise Error("ValueError: x_metrics: a stage record must hold 16 words")
        for j in range(STAGE_WORDS):  # driver-loop: over one stage record's fixed words
            stages.append(Int32(_int(stage[j], context)))
    if nst == 0:
        stages.append(0)
    var known = True
    var merged = _metrics_download(fields, size, context, known)
    return _metrics_execute[False](fields, List[PythonObject](), List[Int](), stages,
                                   nst, size, arena, base, entries, known, merged, context)


def _metrics_run_plan(plan: MetricProgramPlan, inputs: List[PythonObject], mode: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    # Private native builders supply words directly. The separately callable
    # public _Prog path above retains its dictionary/record admission and order.
    classical_connect_numpy(context)
    var entries = _family_entries(mode, "x_metrics", context)
    var arena = _store(plan.size, "f", False, context)
    var base = Int(py=arena[1])
    var known = True
    var merged = plan.download(known)
    return _metrics_execute[True](PythonObject(None), inputs, plan.input_offsets, plan.stages,
                                  len(plan.stages) // STAGE_WORDS, plan.size, arena, base,
                                  entries, known, merged, context)


def _metrics_execute[native_plan: Bool](
    fields: PythonObject, native_inputs: List[PythonObject], native_offsets: List[Int],
    stages: List[Int32], nst: Int, size: Int, arena: PythonObject, base: Int,
    entries: PythonObject, known: Bool, merged: List[Int32], context: PythonObject,
) raises -> PythonObject:
    var ranges = _switch("MOJOLEARN_ARENA_RANGES") and py_get(entries, "x_metrics_run_ranges") is not None
    var inputs = PythonObject(None)
    comptime if not native_plan:
        inputs = fields["_inputs"]
    var input_count = len(native_inputs) if native_plan else len(inputs)
    var ins = List[Int32]()
    var owned = List[Int](capacity=2 * input_count)
    try:
        for k in range(input_count):  # driver-loop: over the program's input buffers
            var off = 0
            var value = PythonObject(None)
            comptime if native_plan:
                off = native_offsets[k]
                value = native_inputs[k]
            else:
                var record = inputs[k]
                off = _int(record[0], context)
                value = record[1]
            var size_k = Int(py=value.size)
            if size_k == 0:
                continue
            var src = -1
            if _is_rows(value, context):
                # device-rows input (lane cpu4-misc): gathered on the device
                # out of the base put once into this binding's store
                if ranges and _can_gather(entries, "x_metrics"):
                    src = _rows_gather(value, entries, "x_metrics", owned, context)
                if src < 0:
                    value = _values(value, context, True)
            if src < 0 and ranges:
                # resident (lane py-shared): the device copies it from the
                # store; the host arena's words stay zero and are never read
                var cache = _resident(entries, size_k, False, context, "x_metrics")
                if cache is not None:
                    src = _cached_slot(cache, value, entries, context, "x_metrics")
            if src < 0:
                memcpy(dest=F32P(unsafe_from_address=base + 4 * off),
                       src=F32P(unsafe_from_address=Int(py=classical_address(value))), count=size_k)
            _add_input(ins, off, off + size_k, src)
        var pa = Int(stages.unsafe_ptr())
        if ranges:
            # the shared ranges runner (core/arena_io.mojo): only the inputs
            # go up, only the outputs come back
            var outs = merged.copy() if known else [Int32(0), Int32(size), Int32(-1), Int32(1)]
            var n_in = len(ins) // 3
            var n_out = len(outs) // 4
            var ia = _pad(ins.copy(), [Int32(0), Int32(0), Int32(-1)])
            var oa = _pad(outs^, [Int32(0), Int32(0), Int32(-1), Int32(1)])
            _ = _entry_call(entries["x_metrics_run_ranges"], Python.tuple(
                base, pa, Python.tuple(size, nst, n_in, n_out), Int(ia.unsafe_ptr()), Int(oa.unsafe_ptr())))
            _ = ia^
            _ = oa^
        elif known and py_get(entries, "x_metrics_run_out") is not None:
            var n_out = len(merged) // 4
            var oa = _pad(merged.copy(), [Int32(0), Int32(0), Int32(-1), Int32(1)])
            _ = _entry_call(entries["x_metrics_run_out"], Python.tuple(
                base, size, pa, nst, Int(oa.unsafe_ptr()), n_out))
            _ = oa^
        else:
            _ = _entry_call(entries["x_metrics_run"], Python.tuple(base, size, pa, nst))
    except error:
        var cleanup = _release(owned, entries, "x_metrics")
        if cleanup.byte_length() > 0:
            _drop_pending(error)
            raise Error("RuntimeError: x_metrics operation failed: " + String(error) + "; cleanup also failed: " + cleanup)
        raise error
    var cleanup = _release(owned, entries, "x_metrics")
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_metrics slot cleanup failed: " + cleanup)
    comptime if not native_plan:
        py_setitem(fields, "arena", arena[0])
    _ = stages  # The packed native record owner survives the synchronous run.
    return arena[0]


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
        elif operation == "x_metrics_run":
            result = _metrics_run(prog, args[0], context)
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
