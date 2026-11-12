# SPDX-License-Identifier: Apache-2.0
# The x_prep program of the native classifier owner (bindings/prep_classify_native.mojo),
# door side (lane entangled-programs, 2026-11-12). The program itself is the typed
# `ClsProgram` and its run the typed `cls_prog_execute` (bindings/prep_classify_prog_run.mojo,
# on the image's x_prep transport). This door holds only the Python owners of the
# caller buffers for the call's duration (UNPACK), resolves the selected image, its
# route words, the resident cache slots and DeviceRows fields before the run, and
# reads the arena back into public Arrays (PACK). The arena, inputs, ranges,
# resident and device-gathered inputs and the run are the ones the Python
# `_Prog.run` made: no bit moves.
from bindings.native_array_storage_door import native_array_empty, native_array_zeros
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address, classical_typed, classical_wrap_array, classical_is_array
from bindings.admitted_input import admit_x
from std.python._cpython import GILReleased
from std.os import getenv
from std.time import perf_counter_ns
from bindings.arena_prog_native_door import _family_entries, _store, _is_rows, _can_gather, _drop_pending
from bindings.arena_prog_run import _switch
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.x_prep_native_owner_door import (
    xprep_key, _resident, _cached_slot, _resident_scope, _resident_cache, _cache_state, _cached_slot_in,
    xprep_drop_resident,
)
from bindings.x_prep_native_inputs_door import _values, _rows_fields, _dense_values
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsRunInput, ClsRoute, cls_prog_execute, cls_prog_release, CLS_IN_BUFFER, CLS_STAGE_WORDS,
)
from bindings.x_prep_native_owner_door import xprep_take, xprep_address
from bindings.x_prep_native_owner_run import xprep_r3

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
# `_expansion_prep._NONE`: an absent stage operand.
comptime NONE = -1
comptime INT32_MAX = 2147483647
# x_prep: an input of at least this many words goes up from its own buffer
# into a slot freed after the run (MOJOLEARN_XPREP_DIRECT);
# _expansion_prep._DIRECT_MIN_WORDS (bindings/arena_prog_native.mojo).
comptime PREP_DIRECT_MIN_WORDS = 1048576


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


struct ClsProg(Movable):
    """The door of one x_prep program: the typed program, the Python owner
    of each input record (None for a native constant), and after the run
    the arena store and its base address."""
    var program: ClsProgram
    var owners: List[PythonObject]
    var store: PythonObject
    var base: Int

    def __init__(out self) raises:
        self.program = ClsProgram(xprep_r3("work"))
        self.owners = List[PythonObject]()
        self.store = PythonObject(None)
        self.base = 0

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
        """`_Prog.run`. UNPACK: the selected image's transport and route
        words, the arena store, each input's address, resident slot or
        device gather. RUN: `cls_prog_execute`, the GIL released. Then the
        slots this run owns are freed and stale resident copies of in-out
        inputs dropped, as the shared runner did."""
        var pc = context["prog_context"]
        classical_connect_numpy(pc)
        var profile = String(getenv("MOJOLEARN_XPREP_PROFILE", "0")) == "1"
        var t_run = Int(perf_counter_ns()) if profile else 0
        var entries = _family_entries(mode, "x_prep", pc)
        var key = xprep_key(entries)
        var H = self.program.size
        var sc = self.program.scratch_size
        var has_out = py_get(entries, "x_prep_run_out") is not None
        var has_scratch_entry = sc > 0 and not has_out and py_get(entries, "x_prep_run_scratch") is not None
        var dev_scratch = sc > 0 and (has_out or has_scratch_entry)
        if H + sc > INT32_MAX:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        var host_words = H + (0 if dev_scratch else sc)
        var arena = _store(host_words, "f", True, pc)
        var base = Int(py=arena[1])
        var t_alloc = Int(perf_counter_ns()) if profile else 0
        var ranges = (dev_scratch or sc == 0) and _switch("MOJOLEARN_ARENA_RANGES") and py_get(entries, "x_prep_run_ranges") is not None
        var direct_ok = (py_get(entries, "x_prep_dev_put") is not None and py_get(entries, "x_prep_dev_free") is not None
                         and _switch("MOJOLEARN_XPREP_DIRECT"))
        var gather_ok = ranges and _can_gather(entries, "x_prep")
        var records = List[ClsRunInput](capacity=len(self.program.inputs))
        var held = List[PythonObject](capacity=2 * len(self.program.inputs))
        var inout_values = List[PythonObject]()
        # Read once per run, at the first input that needs them (perf3-prep-shared).
        var scope = Optional[PythonObject]()
        var cache_state = Optional[PythonObject]()
        for k in range(len(self.program.inputs)):  # driver-loop: over the program's input records
            var record = self.program.inputs[k]
            if record.words == 0:
                continue
            if record.kind != CLS_IN_BUFFER:
                records.append(ClsRunInput(record.offset, record.words, self.program.const_address(record), -1,
                                           False, False, -1, 0, 0, 0, 0, 0))
                continue
            var value = self.owners[k]
            if record.in_out:
                inout_values.append(value)
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
                    var fields = _rows_fields(value, pc)
                    var rows_base = _dense_values(fields["base"], pc)
                    var indices = fields["indices"]
                    n_indices = Int(py=indices.size)
                    if n_indices >= 1:
                        gathered = True
                        row_words = Int(py=fields["row_words"])
                        var cache = _resident(entries, 0, True, pc, "x_prep")
                        if cache is not None:
                            base_slot = _cached_slot(cache, rows_base, entries, pc, "x_prep")
                        base_address = Int(py=classical_address(rows_base))
                        base_words = Int(py=rows_base.size)
                        indices_address = Int(py=classical_address(indices))
                        held.append(rows_base)
                        held.append(indices)
                if not gathered:
                    value = _values(value, pc, True)
            var source = -1
            var direct = False
            if not gathered and ranges and not record.in_out:
                if not scope:
                    scope = _resident_scope(entries, pc, "x_prep")
                if scope.value() is not None and record.words >= Int(py=scope.value()[0]):
                    if not cache_state:
                        var cache = _resident_cache(entries, scope.value()[1], pc, "x_prep")
                        cache_state = _cache_state(cache, entries, pc, "x_prep")
                    source = _cached_slot_in(cache_state.value(), value, entries, "x_prep")
                elif direct_ok and record.words >= PREP_DIRECT_MIN_WORDS:
                    direct = True
            var address = 0 if gathered or source >= 0 else Int(py=classical_address(value))
            records.append(ClsRunInput(record.offset, record.words, address, source, direct, gathered,
                                       base_slot, base_address, base_words, indices_address, n_indices, row_words))
            held.append(value)
        var t_in = Int(perf_counter_ns()) if profile else 0
        var owned = List[Int](capacity=2 * len(records))
        var route = ClsRoute(dev_scratch, has_out, ranges)
        try:
            with GILReleased(Python()):
                cls_prog_execute(key, self.program, records, base, route, owned)
        except error:
            var cleanup = String()
            with GILReleased(Python()):
                cleanup = cls_prog_release(key, owned)
            var dropped = xprep_drop_resident(inout_values, entries, pc, "x_prep")
            if dropped.byte_length() > 0:
                cleanup += ("" if cleanup.byte_length() == 0 else "; ") + dropped
            if cleanup.byte_length() > 0:
                _drop_pending(error)
                raise Error("RuntimeError: x_prep operation failed: " + String(error) + "; cleanup also failed: " + cleanup)
            raise error
        var cleanup = String()
        with GILReleased(Python()):
            cleanup = cls_prog_release(key, owned)
        var dropped = xprep_drop_resident(inout_values, entries, pc, "x_prep")
        if dropped.byte_length() > 0:
            cleanup += ("" if cleanup.byte_length() == 0 else "; ") + dropped
        if cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep slot cleanup failed: " + cleanup)
        _ = held^
        _ = records^
        self.store = arena[0]
        self.base = base
        if profile:
            # one line per program (the device's XPPHASE lines come in the same order)
            var line = String("XPPROG ops=")
            var nst = len(self.program.stages) // CLS_STAGE_WORDS
            for i in range(nst):  # driver-loop: stage op words for the profiling line
                line += ("" if i == 0 else "+") + String(self.program.stages[i * CLS_STAGE_WORDS])
            var done = Int(perf_counter_ns())
            print(line + " arena=" + String(H) + " scratch=" + String(sc) + " out=0 dev_out=0 ranges=" + String(Int(ranges))
                  + " alloc_s=" + String(Float64(t_alloc - t_run) / 1.0e9)
                  + " inputs_s=" + String(Float64(t_in - t_alloc) / 1.0e9)
                  + " call_s=" + String(Float64(done - t_in) / 1.0e9))

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
