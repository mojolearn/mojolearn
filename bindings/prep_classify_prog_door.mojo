# SPDX-License-Identifier: Apache-2.0
# The x_prep program of the native classifier owner (bindings/prep_classify_native.mojo),
# door side (lane entangled-programs, 2026-11-12). The program itself is the typed
# `ClsProgram` and its run the typed `cls_prog_run` (bindings/prep_classify_prog_run.mojo,
# on the image's x_prep transport: route words, input choices, the run, the release
# of its slots and the drop of stale resident copies). This door holds only the
# Python owners of the caller buffers for the call's duration (UNPACK: the selected
# image as a `ClsRunPlan`, addresses, DeviceRows fields, the thread's scope word, the
# pin of a buffer the scope's cache takes) and reads the arena back into public
# Arrays (PACK). The arena, inputs, ranges,
# resident and device-gathered inputs and the run are the ones the Python
# `_Prog.run` made: no bit moves.
from bindings.native_array_storage_door import native_array_empty, native_array_zeros
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address, classical_typed, classical_wrap_array, classical_is_array
from bindings.admitted_input_unpack import admit_x
from std.python._cpython import GILReleased
from std.os import getenv
from std.time import perf_counter_ns
from bindings.arena_prog_native_door import _family_entries, _store, _is_rows
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.arena_scope_door import arena_unpin_all
from bindings.arena_cache_run import ArenaCachePut
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.x_prep_native_owner_door import xprep_key, xprep_resident
from bindings.x_prep_native_inputs_door import _values, _rows_fields, _dense_values
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsRunInput, ClsRunPlan, ClsResident, ClsRunDone, CLS_IN_BUFFER, CLS_STAGE_WORDS,
    CLS_INPUT_RESIDENT, CLS_INPUT_DIRECT, cls_run_route, cls_host_words, cls_input_choice, cls_prog_run,
)
from bindings.x_prep_native_owner_door import xprep_take, xprep_address
from bindings.x_prep_native_owner_run import xprep_r3
from bindings.scaler_route_unpack import (
    scaler_selection_held, scaler_selection_keep, scaler_route_refuse_fast, scaler_route_held,
    SCALER_CAP_TRANSPORT, SCALER_CAP_RUN_OUT, SCALER_CAP_RUN_SCRATCH, SCALER_CAP_RUN_RANGES,
    SCALER_CAP_DEV_PUT, SCALER_CAP_DEV_FREE, SCALER_CAP_DEV_TAKE,
)

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
# `_expansion_prep._NONE`: an absent stage operand.
comptime NONE = -1
comptime INT32_MAX = 2147483647


def off_add(offset: Int, k: Int) -> Int:
    """`offset + k` for an arena offset or a scratch offset (`_Scratch.__add__`):
    scratch word s is the stage word -(s + 2)."""
    if offset < -1:
        return offset - k
    return offset + k


def f32_array(count: Int, context: PythonObject) raises -> PythonObject:
    """A zeroed float32 vector Array of `count` words (at least one)."""
    return classical_typed(Python.tuple(max(count, 1)), "<f4", context["output_context"])


def i32_array(count: Int, context: PythonObject) raises -> PythonObject:
    """A zeroed int32 vector Array of `count` words (at least one)."""
    var b = native_loaded_module("builtins")
    var n = max(count, 1)
    var store = native_array_zeros("i", n)
    return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context["output_context"])


def f32_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Prog.put`'s admission of a stored attribute: a float32 C-ordered Array
    (or a fold's DeviceRows) as it is, anything else converted once."""
    if py_type(value) is context["maxabs_rows_class"]:
        return value
    if classical_is_array(value, context) and String(value.dtype) == "<f4" and (String(value.order) == "C" or Int(py=value.ndim) < 2):
        return value
    var nd = Int(py=value.ndim)
    return admit_x(value, 1 if nd < 2 else 2, "input", context)


def _plan_by_name(entries: PythonObject, scratch: Int) raises -> ClsRunPlan:
    """UNPACK of a selection no estimator route holds (a program without a
    model, or an image that publishes no transport table in its route): the
    capability entries read by name, each only where the run can use it, as
    the run read them before."""
    var key = xprep_key(entries)
    var has_out = py_get(entries, "x_prep_run_out") is not None
    var has_scratch = scratch > 0 and not has_out and py_get(entries, "x_prep_run_scratch") is not None
    var dev_scratch = scratch > 0 and (has_out or has_scratch)
    var has_ranges = (dev_scratch or scratch == 0) and py_get(entries, "x_prep_run_ranges") is not None
    var dev_store = py_get(entries, "x_prep_dev_put") is not None and py_get(entries, "x_prep_dev_free") is not None
    var take_rows = has_ranges and dev_store and py_get(entries, "x_prep_dev_take_rows") is not None
    return ClsRunPlan(key, has_out, has_scratch, has_ranges, dev_store, take_rows)


def _plan_held(model: PythonObject, entries: PythonObject, scratch: Int) raises -> ClsRunPlan:
    """UNPACK of the estimator's held selection: the transport entry and the
    capability bits its device route resolved once (one record read, one
    pointer compare); by name when the route does not hold them."""
    var held = scaler_route_held(model, entries)
    if not held.held or (held.flags & SCALER_CAP_TRANSPORT) == 0:
        return _plan_by_name(entries, scratch)
    var flags = held.flags
    var dev_store = (flags & SCALER_CAP_DEV_PUT) != 0 and (flags & SCALER_CAP_DEV_FREE) != 0
    return ClsRunPlan(held.transport, (flags & SCALER_CAP_RUN_OUT) != 0, (flags & SCALER_CAP_RUN_SCRATCH) != 0,
                      (flags & SCALER_CAP_RUN_RANGES) != 0, dev_store, dev_store and (flags & SCALER_CAP_DEV_TAKE) != 0)


def _thread_resident(context: PythonObject, plan: ClsRunPlan) raises -> ClsResident:
    """The calling thread's `resident()` scope for one run (the thread-local
    read, once per run), as the typed run's value."""
    var found = xprep_resident(context, plan.key, plan.dev_store)
    return ClsResident(found.scope, found.min_words, found.store)


def _resident_slot(mut resident: ClsResident, source: PythonObject, address: Int, words: Int) raises -> Int:
    """The store id of `source`'s resident copy: found (a hit), or put once
    with the GIL released and published with `source` pinned by the cache
    (bindings/arena_scope_door.mojo `arena_cache_slot`, on the typed run's
    lease)."""
    var found = resident.find(address, words)
    if found >= 0:
        return found
    var pin = arena_pin(source)
    var put = ArenaCachePut(-1, False)
    try:
        with GILReleased(Python()):
            put = resident.put(address, words, pin)
    except error:
        arena_unpin(pin)
        raise error^
    if not put.pinned:
        arena_unpin(pin)
    return put.store_id


struct ClsProg(Movable):
    """The door of one x_prep program: the typed program, the Python owner
    of each input record (None for a native constant), and after the run
    the arena store and its base address. `model` is the estimator whose
    device route holds the selected image (its entries, transport entry and
    capability bits, resolved once per estimator); None for a program no
    estimator owns, whose run selects and reads them per run, as before."""
    var program: ClsProgram
    var owners: List[PythonObject]
    var store: PythonObject
    var base: Int
    var model: PythonObject

    def __init__(out self) raises:
        self.program = ClsProgram(xprep_r3("work"))
        self.owners = List[PythonObject]()
        self.store = PythonObject(None)
        self.base = 0
        self.model = PythonObject(None)

    def __init__(out self, model: PythonObject) raises:
        self.program = ClsProgram(xprep_r3("work"))
        self.owners = List[PythonObject]()
        self.store = PythonObject(None)
        self.base = 0
        self.model = model

    def alloc(mut self, n: Int) -> Int:
        return self.program.alloc(n)

    def scratch(mut self, n: Int) -> Int:
        return self.program.scratch(n)

    def work(mut self, n: Int) -> Int:
        return self.program.work(n)

    def put(mut self, value: PythonObject, in_out: Bool = False) raises -> Int:
        """UNPACK `_Prog.put` of an admitted float32 buffer (or DeviceRows):
        the record's layout, the owner held until the run."""
        var offset = self.program.input(Int(py=value.size), CLS_IN_BUFFER, -1, in_out)
        self.owners.append(value)
        return offset

    def put_words(mut self, words: PythonObject) raises -> Int:
        """`_Prog.put_words`: an int32 buffer's words as they are."""
        return self.put(words)

    def put_list(mut self, values: List[Float64], context: PythonObject, in_out: Bool = False) raises -> Int:
        """`_Prog.put_list`: float32 words of a parameter list ([0.0] when
        empty), a native constant of the program."""
        var offset = self.program.put_list(values, in_out)
        self.owners.append(PythonObject(None))
        return offset

    def put_scalar(mut self, value: Float64, context: PythonObject) raises -> Int:
        var one = List[Float64]()
        one.append(value)
        return self.put_list(one, context)

    def put_ints(mut self, values: List[Int], context: PythonObject) raises -> Int:
        """`_Prog.put_ints`: int32 words read by `ldi` ([0] when empty)."""
        var offset = self.program.put_ints(values)
        self.owners.append(PythonObject(None))
        return offset

    def put_codes(mut self, codes: PythonObject) raises -> Int:
        """`_Prog.put_codes`: int32 codes -> offset of their float values (i2f)."""
        var n = Int(py=codes.size)
        var bits = self.put(codes)
        var out = self.alloc(n)
        self.stage(10, n, bits, out)
        return out

    def stage(mut self, op: Int, total: Int, *params: Int):
        """`_Prog.stage`: op, total, the parameters, zero padding."""
        var words = List[Int](capacity=len(params))
        for i in range(len(params)):  # driver-loop: over at most fourteen stage parameters
            words.append(params[i])
        self.program.stage(op, total, words)

    def run(mut self, mode: PythonObject, context: PythonObject) raises:
        """`_Prog.run`. UNPACK: the selected image as a `ClsRunPlan`, the
        arena store, each input's address, the thread's scope word. RUN
        (bindings/prep_classify_prog_run.mojo): `cls_run_route`,
        `cls_input_choice` per input, `cls_prog_run` with the GIL released
        (the run, the release of its slots, the drop of stale resident
        copies of in-out inputs). PACK: the store the readers take."""
        var pc = context["prog_context"]
        classical_connect_numpy(pc)
        var profile = String(getenv("MOJOLEARN_XPREP_PROFILE", "0")) == "1"
        var t_run = Int(perf_counter_ns()) if profile else 0
        # The selection of the estimator's device route for `mode` in this
        # run context (one record read and pointer compares), or the shared
        # runner's own selection, kept in the route. `device="cpu"` with the
        # fast mode is refused in the shared words before the CPU table is read.
        var entries = PythonObject(None)
        var routed = self.model is not None
        if routed:
            entries = scaler_selection_held(self.model, mode, context)
        if entries is None:
            if routed:
                scaler_route_refuse_fast(self.model, mode, context)
            entries = _family_entries(mode, "x_prep", pc)
            if routed:
                scaler_selection_keep(self.model, mode, entries, context)
        var plan = _plan_by_name(entries, self.program.scratch_size) if not routed else _plan_held(self.model, entries, self.program.scratch_size)
        var route = cls_run_route(plan, self.program)
        var H = self.program.size
        var sc = self.program.scratch_size
        var arena = _store(cls_host_words(self.program, route), "f", True, pc)
        var base = Int(py=arena[1])
        var t_alloc = Int(perf_counter_ns()) if profile else 0
        # TOMBSTONE: MOJOLEARN_XPREP_FAST_DIRECT_OFF (B arm loses: host-arena staging of large inputs; A/B abn9 NVIDIA L40S, 48 cells 0.96-17.38 arm/base, median 1.75, 34 slower) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
        # Restore: git apply experiments/removed/MOJOLEARN_XPREP_FAST_DIRECT_OFF.patch; record in docs/TOMBSTONES.md.
        var gather_ok = route.ranges and plan.take_rows
        var records = List[ClsRunInput](capacity=len(self.program.inputs))
        var held = List[PythonObject](capacity=2 * len(self.program.inputs))
        var inout_addresses = List[Int]()
        var inout_words = List[Int]()
        # The thread's resident scope, resolved once per run at the first
        # input that could be kept resident (or, for a run with in-out
        # inputs, before their stale copies are dropped); its cache is
        # leased at the first input that is.
        var resident = ClsResident()
        var resident_read = False
        for k in range(len(self.program.inputs)):  # driver-loop: over the program's input records
            var record = self.program.inputs[k]
            if record.words == 0:
                continue
            if record.kind != CLS_IN_BUFFER:
                records.append(ClsRunInput(record.offset, record.words, self.program.const_address(record), -1,
                                           False, False, -1, 0, 0, 0, 0, 0))
                continue
            var value = self.owners[k]
            var gathered = False
            var base_slot = -1
            var base_address = 0
            var base_words = 0
            var indices_address = 0
            var n_indices = 0
            var row_words = 0
            if _is_rows(value, pc):
                # device-rows input (lane cpu4-misc): the fold's rows gathered
                # on the device; a binding without the gather copies host rows
                if gather_ok:
                    var fold = _rows_fields(value, pc)
                    var rows_base = _dense_values(fold.base, pc)
                    var indices = fold.indices
                    n_indices = fold.rows
                    if n_indices >= 1:
                        gathered = True
                        row_words = fold.row_words
                        base_address = Int(py=classical_address(rows_base))
                        base_words = Int(py=rows_base.size)
                        if not resident_read:
                            resident = _thread_resident(pc, plan)
                            resident_read = True
                        if resident.active():
                            # the scope keeps a DeviceRows base at any size
                            base_slot = _resident_slot(resident, rows_base, base_address, base_words)
                        indices_address = Int(py=classical_address(indices))
                        held.append(rows_base)
                        held.append(indices)
                if not gathered:
                    value = _values(value, pc, True)
            var source = -1
            var direct = False
            var address = 0
            if not gathered:
                address = Int(py=classical_address(value))
                if record.in_out:
                    # the buffer's identity is (address, value.size), as before
                    inout_addresses.append(address)
                    inout_words.append(Int(py=value.size))
                elif route.ranges:
                    if not resident_read:
                        resident = _thread_resident(pc, plan)
                        resident_read = True
                    var choice = cls_input_choice(record, route, plan, resident)
                    if choice == CLS_INPUT_RESIDENT:
                        source = _resident_slot(resident, value, address, Int(py=value.size))
                        address = 0
                    elif choice == CLS_INPUT_DIRECT:
                        direct = True
            records.append(ClsRunInput(record.offset, record.words, address, source, direct, gathered,
                                       base_slot, base_address, base_words, indices_address, n_indices, row_words))
            held.append(value)
        if len(inout_addresses) > 0 and not resident_read:
            resident = _thread_resident(pc, plan)
        var t_in = Int(perf_counter_ns()) if profile else 0
        var done = ClsRunDone()
        with GILReleased(Python()):
            done = cls_prog_run(plan, self.program, records, base, route, resident, inout_addresses, inout_words)
        # The sources of dropped resident copies: the last step, after every
        # free and publication.
        arena_unpin_all(done.pins)
        if done.failed:
            if done.cleanup.byte_length() > 0:
                raise Error("RuntimeError: x_prep operation failed: " + done.error + "; cleanup also failed: " + done.cleanup)
            raise Error(done.error)
        if done.cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep slot cleanup failed: " + done.cleanup)
        _ = held^
        _ = records^
        _ = entries  # the selection pins the image through the run
        self.store = arena[0]
        self.base = base
        if profile:
            # one line per program (the device's XPPHASE lines come in the same order)
            var line = String("XPPROG ops=")
            var nst = len(self.program.stages) // CLS_STAGE_WORDS
            for i in range(nst):  # driver-loop: stage op words for the profiling line
                line += ("" if i == 0 else "+") + String(self.program.stages[i * CLS_STAGE_WORDS])
            var finished = Int(perf_counter_ns())
            print(line + " arena=" + String(H) + " scratch=" + String(sc) + " out=0 dev_out=0 ranges=" + String(Int(route.ranges))
                  + " alloc_s=" + String(Float64(t_alloc - t_run) / 1.0e9)
                  + " inputs_s=" + String(Float64(t_in - t_alloc) / 1.0e9)
                  + " call_s=" + String(Float64(finished - t_in) / 1.0e9))

    def value(self, offset: Int) -> Float64:
        """One arena word after the run (`_Prog.values`)."""
        return Float64(F32P(unsafe_from_address=self.base)[offset])

    def get(self, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
        """`_Prog.get`: an (rows,) or (rows, cols) float32 Array of the words."""
        if rows * (1 if cols == 0 else cols) <= 0:
            var empty = native_array_empty("f")
            return classical_wrap_array(empty, Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols), "<f4", "C", context["output_context"])
        return xprep_take(self.store, offset, rows, cols, context["output_context"])

    def _bytes(self, offset: Int, words: Int) raises -> PythonObject:
        var b = native_loaded_module("builtins")
        var part = py_call(b.memoryview, self.store)[py_call(b.slice, offset, offset + words)]
        return py_call(part.cast, "B")

    def get_i32(self, offset: Int, n: Int, context: PythonObject) raises -> PythonObject:
        """`_Prog.get_i32`: n int32 words as an Array."""
        var store = native_array_empty("i")
        if n > 0:
            _ = py_call(store.frombytes, self._bytes(offset, n))
        return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context["output_context"])

    def get_f64(self, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
        """`_Prog.get_f64`: 2 * rows * cols words as a (rows, cols) float64 Array."""
        var store = native_array_empty("d")
        if rows * cols > 0:
            _ = py_call(store.frombytes, self._bytes(offset, 2 * rows * cols))
        return classical_wrap_array(store, Python.tuple(rows, cols), "<f8", "C", context["output_context"])
