# SPDX-License-Identifier: Apache-2.0
# The x_prep program of the native classifier owner (bindings/prep_classify_native.mojo):
# `_expansion_prep._Prog`'s layout rules (alloc, work, put, put_list, put_codes,
# put_words, put_ints, stage) kept natively, run by the shared arena program
# runner (bindings/arena_prog_native.mojo `_prep_run`) on a `_Prog` record built
# from these words, so the arena, inputs, ranges, resident and device-gathered
# inputs and the run are the ones the Python `_Prog.run` made: no bit moves.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_arrays import classical_address, classical_typed, classical_wrap_array, classical_is_array, classical_host_f32
from bindings.arena_prog_native import _prep_run
from bindings.x_prep_native_owner import xprep_r3, xprep_take, xprep_address

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
# `_expansion_prep._NONE`: an absent stage operand.
comptime NONE = -1
# Words of one stage record: op, total and the fourteen parameters.
comptime STAGE_WORDS = 16
comptime PARAMS = 14
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
    var b = Python.import_module("builtins")
    var n = max(count, 1)
    var store = py_call(context["array_type"], "i", py_call(b.bytes, 4 * n))
    return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context["output_context"])


def f32_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Prog.put`'s admission of a stored attribute: a float32 C-ordered Array
    (or a fold's DeviceRows) as it is, anything else converted once."""
    if py_type(value) is context["maxabs_rows_class"]:
        return value
    if classical_is_array(value, context) and String(value.dtype) == "<f4" and (String(value.order) == "C" or Int(py=value.ndim) < 2):
        return value
    var nd = Int(py=value.ndim)
    return classical_host_f32(value, 1 if nd < 2 else 2, "input", context)


struct ClsProg(Movable):
    """One x_prep program: its host words, its device scratch words, the
    input buffers at their offsets, the in-out input ranges and the stage
    words (a scratch word s is the stage word -(s + 2) until the run)."""
    var size: Int
    var scratch_size: Int
    var work_on: Bool
    var inputs: PythonObject
    var inout: PythonObject
    var stages: List[Int]
    var store: PythonObject
    var base: Int

    def __init__(out self) raises:
        self.size = 0
        self.scratch_size = 0
        self.work_on = xprep_r3("work")
        self.inputs = Python.list()
        self.inout = Python.list()
        self.stages = List[Int]()
        self.store = PythonObject(None)
        self.base = 0

    def alloc(mut self, n: Int) -> Int:
        var offset = self.size
        self.size += max(n, 0)
        return offset

    def scratch(mut self, n: Int) -> Int:
        var offset = self.scratch_size
        self.scratch_size += max(n, 0)
        return -(offset + 2)

    def work(mut self, n: Int) -> Int:
        """`_Prog.work`: device scratch under the prep-apple3 `work` change,
        else arena words."""
        if not self.work_on:
            return self.alloc(n)
        return self.scratch(n)

    def put(mut self, value: PythonObject, inout: Bool = False) raises -> Int:
        """`_Prog.put` of an admitted float32 buffer (or DeviceRows)."""
        var size = Int(py=value.size)
        var offset = self.alloc(size)
        _ = py_call(self.inputs.append, Python.tuple(offset, value, "f"))
        if inout and size > 0:
            _ = py_call(self.inout.append, Python.tuple(offset, offset + size))
        return offset

    def put_words(mut self, words: PythonObject) raises -> Int:
        """`_Prog.put_words`: an int32 buffer's words as they are."""
        var offset = self.alloc(Int(py=words.size))
        _ = py_call(self.inputs.append, Python.tuple(offset, words, "i"))
        return offset

    def put_list(mut self, values: List[Float64], context: PythonObject, inout: Bool = False) raises -> Int:
        """`_Prog.put_list`: float32 words of a parameter list ([0.0] when empty)."""
        var words = f32_array(len(values), context)
        var dest = F32P(unsafe_from_address=Int(py=classical_address(words)))
        for i in range(len(values)):  # driver-loop: over a parameter list (priors, scalars, model-structure words)
            dest[i] = Float32(values[i])
        return self.put(words, inout)

    def put_scalar(mut self, value: Float64, context: PythonObject) raises -> Int:
        var one = List[Float64]()
        one.append(value)
        return self.put_list(one, context)

    def put_ints(mut self, values: List[Int], context: PythonObject) raises -> Int:
        """`_Prog.put_ints`: int32 words read by `ldi` ([0] when empty)."""
        var words = i32_array(len(values), context)
        var dest = I32P(unsafe_from_address=Int(py=classical_address(words)))
        for i in range(len(values)):  # driver-loop: over a parameter list of int words
            dest[i] = Int32(values[i])
        return self.put_words(words)

    def put_codes(mut self, codes: PythonObject) raises -> Int:
        """`_Prog.put_codes`: int32 codes -> offset of their float values (i2f)."""
        var n = Int(py=codes.size)
        var bits = self.alloc(n)
        _ = py_call(self.inputs.append, Python.tuple(bits, codes, "i"))
        var out = self.alloc(n)
        self.stage(10, n, bits, out)
        return out

    def stage(mut self, op: Int, total: Int, *params: Int):
        """`_Prog.stage`: op, total, the parameters, zero padding."""
        self.stages.append(op)
        self.stages.append(total)
        for i in range(len(params)):  # small-loop(params: at most fourteen stage parameters)
            self.stages.append(params[i])
        for _ in range(PARAMS - len(params)):  # small-loop(params: pad the fixed fourteen-word stage ABI)
            self.stages.append(0)

    def run(mut self, mode: PythonObject, context: PythonObject) raises:
        """`_Prog.run`: the shared runner over this program's record."""
        var pc = context["prog_context"]
        var b = Python.import_module("builtins")
        var prog = py_call(b.object.__new__, pc["prog_class"])
        var fields = py_fields(prog)
        var host = self.size
        var records = Python.list()
        for i in range(len(self.stages) // STAGE_WORDS):  # driver-loop: over the program's stages
            var record = Python.list()
            for k in range(STAGE_WORDS):  # small-loop(params: the fixed sixteen-word stage record)
                var word = self.stages[i * STAGE_WORDS + k]
                if k >= 2 and word < -1:
                    # scratch word s sits past the host arena (no output region)
                    word = host - word - 2
                _ = py_call(record.append, word)
            _ = py_call(records.append, record)
        py_setitem(fields, "size", host)
        py_setitem(fields, "scratch_size", self.scratch_size)
        py_setitem(fields, "out_size", PythonObject(None))
        py_setitem(fields, "_inputs", self.inputs)
        py_setitem(fields, "_inout", self.inout)
        py_setitem(fields, "_stages", records)
        py_setitem(fields, "arena", PythonObject(None))
        py_setitem(fields, "_out", PythonObject(None))
        py_setitem(fields, "_out_at", PythonObject(None))
        py_setitem(fields, "_out_code", "f")
        py_setitem(fields, "_after", Python.list())
        _ = _prep_run(prog, mode, pc)
        self.store = fields["arena"]
        self.base = xprep_address(self.store)

    def value(self, offset: Int) -> Float64:
        """One arena word after the run (`_Prog.values`)."""
        return Float64(F32P(unsafe_from_address=self.base)[offset])

    def get(self, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
        """`_Prog.get`: an (rows,) or (rows, cols) float32 Array of the words."""
        if rows * (1 if cols == 0 else cols) <= 0:
            var empty = py_call(context["array_type"], "f")
            return classical_wrap_array(empty, Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols), "<f4", "C", context["output_context"])
        return xprep_take(self.store, offset, rows, cols, context["output_context"])

    def _bytes(self, offset: Int, words: Int) raises -> PythonObject:
        var b = Python.import_module("builtins")
        var part = py_call(b.memoryview, self.store)[py_call(b.slice, offset, offset + words)]
        return py_call(part.cast, "B")

    def get_i32(self, offset: Int, n: Int, context: PythonObject) raises -> PythonObject:
        """`_Prog.get_i32`: n int32 words as an Array."""
        var store = py_call(context["array_type"], "i")
        if n > 0:
            _ = py_call(store.frombytes, self._bytes(offset, n))
        return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context["output_context"])

    def get_f64(self, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
        """`_Prog.get_f64`: 2 * rows * cols words as a (rows, cols) float64 Array."""
        var store = py_call(context["array_type"], "d")
        if rows * cols > 0:
            _ = py_call(store.frombytes, self._bytes(offset, 2 * rows * cols))
        return classical_wrap_array(store, Python.tuple(rows, cols), "<f8", "C", context["output_context"])
