# SPDX-License-Identifier: Apache-2.0
# The native owner of the x_prep imputers, discretizer and encoders:
# SimpleImputer and KBinsDiscretizer. Each public fit, transform,
# inverse_transform and fit_transform is one call into prep_encode_c, which
# checks the parameters, builds the x_prep program the Python `_Prog` built
# (the same stages, operands and order, so the same bits), runs it through
# the shared transport (bindings/x_prep_native_owner.mojo `xprep_run`) and
# stores the fitted attributes under the names, types and dtypes the Python
# set. fit_transform converts X once and hands the converted X to the
# transform program.
from std.memory import memcpy, bitcast
from std.math import ceil
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_wrap_array, classical_c_order
from bindings.scaler_native import _stage, _ptr, _input
from bindings.scaler_lifecycle_native import _entries
from bindings.x_prep_native_owner import xprep_storage, xprep_take, xprep_values, xprep_run, xprep_r3
from bindings.prep_stateless_native import _fam, _int_like, _repr, _name, _take_f

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias I32P = MutPointer[Int32, MutUntrackedOrigin]
alias I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_SORT_COLS = 0
comptime OP_COL_STATS = 1
comptime OP_QUANTILE = 2
comptime OP_UNIQUE_COLS = 5
comptime OP_MODE_COLS = 6
comptime OP_LOOKUP = 7
comptime OP_COUNT_NEG = 8
comptime OP_ONEHOT = 9
comptime OP_WHERE_NEG = 19
comptime OP_TE_APPLY = 22
comptime OP_MARK_MISSING = 23
comptime OP_FILL = 24
comptime OP_KBINS_EDGES = 25
comptime OP_KBINS_CODES = 26
comptime OP_NAN_MASK = 60
comptime OP_BLOCK_ARGMAX = 76
comptime OP_ORD_INVERSE = 77
comptime OP_CAT_GATHER = 78
comptime OP_WHERE_CODE = 79
comptime OP_KBINS_INVERSE = 80
comptime OP_CODE_COUNTS = 88
comptime OP_REMAP_CODES = 89
comptime OP_KBINS_GW = 96
comptime OP_KBINS_WQ = 97
comptime OP_KBINS_WKM = 98
comptime OP_UNIQ_COUNT = 120
comptime OP_UNIQ_SCAN = 121
comptime OP_UNIQ_WRITE = 122
comptime OP_COLB_PART = 124
comptime OP_COLB_FOLD = 125
comptime OP_COLB_SS = 126
comptime OP_COLB_VAR = 127
comptime OP_SPLINE_KNOTS = 49
comptime OP_SPLINE_APPLY = 50
comptime OP_HCAT = 140
comptime OP_CAT_ZERO = 157
comptime OP_CAT_PRESENT = 158
comptime OP_PRES_COUNT = 159
comptime OP_PRES_WRITE = 160
comptime OP_CAT_PACK = 161
comptime OP_UNIQUE_INVERSE = 177
comptime OP_RGATHER = 204
comptime OP_SMROWS = 205
comptime OP_SEL_COUNT = 206
comptime OP_SEL_SCAN = 207
comptime OP_SEL_WRITE = 208
comptime OP_TRANSPOSE = 209
comptime OP_ROWFLAG = 210
comptime OP_WBLK = 230
comptime OP_WSCAN = 231
comptime OP_WDRAW = 232
comptime OP_INF_M0 = 249
comptime OP_INF_M1 = 250
comptime OP_INF_MAP = 251
comptime OP_IMP_STATS = 252
comptime OP_GRID = 263
# `_NONE`: a stage's "no operand" word.
comptime NONE = -1
# x_prep_idn_fam bit 1: IDN_STATS_BLOCKED.
comptime IDN_STATS_BLOCKED = 1
# Row blocks of the blocked column statistics (_expansion_prep._XB).
comptime ROW_BLOCK = 2048
# `_II_CH`: the most chunks of a chunk-parallel p2m selection.
comptime SEL_CHUNKS = 1024
# `_Prog.output`: below this many words the output is plain arena words.
comptime OUT_MIN_WORDS = 134217728
# The stage word of a program's one device output region, resolved when the
# program runs (past the host arena and the scratch, as `_Prog.output`).
comptime OUT_MARK = -2147483647
# Device scratch word k is the stage word SCRATCH_BASE + k (`_Scratch`, so an
# offset added to a scratch word stays one), resolved past the host arena when
# the program runs. Stage parameters are never below -3, and the span keeps
# the words clear of them; a program needing more scratch keeps the rest as
# arena words (where a word lives moves no bit).
comptime SCRATCH_BASE = -1073741824
comptime SCRATCH_SPAN = 1073740800
comptime INT32_MAX = 2147483647
# `_CAT_CAP`, `_CAT_R`, `_CAT_CH` (x_prep/cat_cls2.mojo): the packed distinct
# values' host region (words), the presence flags per column, flags per chunk.
comptime CAT_CAP = 65536
comptime CAT_R = 4096
comptime CAT_CH = 64


# ---------------------------------------------------------------- programs
struct EncProg(Movable):
    """One x_prep program: its arena layout (host words, then device
    scratch), the inputs copied into it, its stages and at most one device
    output region (`_Prog.output`)."""
    var host: Int
    var scratch_size: Int
    var work_on: Bool
    var program: List[Int32]
    var inputs: PythonObject
    var out_words: Int

    def __init__(out self) raises:
        self.host = 0
        self.scratch_size = 0
        self.work_on = xprep_r3("work")
        self.program = List[Int32]()
        self.inputs = Python.list()
        self.out_words = 0

    def alloc(mut self, words: Int) -> Int:
        # `_Prog.alloc`: host arena words.
        var offset = self.host
        self.host += max(words, 0)
        return offset

    def scratch(mut self, words: Int) -> Int:
        # `_Prog.scratch`: device-only words a stage writes before any reads.
        if self.scratch_size + max(words, 0) >= SCRATCH_SPAN:
            return self.alloc(words)
        var offset = self.scratch_size
        self.scratch_size += max(words, 0)
        return SCRATCH_BASE + offset

    def work(mut self, words: Int) -> Int:
        # `_Prog.work`: device scratch under the `work` change, else arena words.
        if not self.work_on:
            return self.alloc(words)
        return self.scratch(words)

    def output(mut self, words: Int, entries: PythonObject) raises -> Int:
        # Below OUT_MIN_WORDS, or on a binding without an output region
        # (the host column; MOJOLEARN_XPREP_OUT=0), the output is host words.
        if words < OUT_MIN_WORDS or py_get(entries, "x_prep_run_out") is None or String(getenv("MOJOLEARN_XPREP_OUT", "1")) == "0":
            return self.alloc(words)
        if self.out_words > 0:
            raise Error("x_prep: one output region per program")
        self.out_words = words
        return OUT_MARK

    def put(mut self, value: PythonObject, words: Int) raises -> Int:
        # A float32 (or int32-word) contiguous buffer of `words` words.
        var offset = self.alloc(words)
        if words > 0:
            _ = py_call(self.inputs.append, Python.tuple(offset, Python.tuple(0, value, PythonObject(None), words, 1)))
        return offset

    def put_x(mut self, values: PythonObject) raises -> Int:
        # X as `xprep_values` converted it (host words or device rows).
        var words = Int(py=values[3]) * Int(py=values[4])
        var offset = self.alloc(words)
        _ = py_call(self.inputs.append, Python.tuple(offset, values))
        return offset

    def put_array2(mut self, value: PythonObject) raises -> Int:
        # A float32 C-order (n, d) Array this owner produced (the C08 inverse).
        return self.put(value, Int(py=value.size))

    def put_array(mut self, value: PythonObject, context: PythonObject) raises -> Int:
        # A fitted float32 vector (`_Prog.put` of an Array).
        var flat = _input(value, 1, False, context)
        return self.put(flat, Int(py=flat.size))

    def put_floats(mut self, values: List[Float64], context: PythonObject) raises -> Int:
        # `_Prog.put_list`: the values rounded to float32 ([0.0] when empty).
        return self.put(_floats(values, context), max(len(values), 1))

    def put_ints(mut self, values: List[Int], context: PythonObject) raises -> Int:
        # `_Prog.put_ints`: int32 words ([0] when empty).
        return self.put(_int_words(values, context), max(len(values), 1))

    def run(mut self, entries: PythonObject, context: PythonObject) raises -> PythonObject:
        """`_Prog.run`: the output region (when there is one) past the host
        arena and the scratch, then the scratch past the host arena; returns
        (arena, output store)."""
        var host = self.host
        var scratch = self.scratch_size
        if host + scratch + self.out_words > INT32_MAX:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        var program = self.program.copy()
        for i in range(len(program) // 16):  # driver-loop: over the program's stages
            for k in range(2, 16):  # small-loop(params: the fixed fourteen-word stage ABI)
                var word = Int(program[i * 16 + k])
                if word == OUT_MARK:
                    program[i * 16 + k] = Int32(host + scratch)
                elif word >= SCRATCH_BASE and word < SCRATCH_BASE + SCRATCH_SPAN:
                    program[i * 16 + k] = Int32(host + word - SCRATCH_BASE)
        var device_scratch = scratch > 0 and (py_get(entries, "x_prep_run_out") is not None or py_get(entries, "x_prep_run_scratch") is not None)
        var arena = xprep_storage(host + (0 if device_scratch else scratch), context)
        var store = xprep_storage(self.out_words, context) if self.out_words > 0 else PythonObject(None)
        xprep_run(arena, host, scratch, store, self.out_words, program^, self.inputs, entries, context, self.out_words)
        return Python.tuple(arena, store)


def _floats(values: List[Float64], context: PythonObject) raises -> PythonObject:
    var out = classical_typed(Python.tuple(max(len(values), 1)), "<f4", context["output_context"])
    var dest = _ptr(out)
    for i in range(len(values)):  # driver-loop: over a program's parameter words
        dest[i] = Float32(values[i])
    return out


def _int_words(values: List[Int], context: PythonObject) raises -> PythonObject:
    # int32 words in a 4-byte store (the arena copies words as they are).
    var out = classical_typed(Python.tuple(max(len(values), 1)), "<f4", context["output_context"])
    var dest = I32P(unsafe_from_address=Int(py=classical_address(out)))
    for i in range(len(values)):  # driver-loop: over a program's int32 parameter words
        dest[i] = Int32(values[i])
    return out


def _word(ran: PythonObject, offset: Int) raises -> Float32:
    return F32P(unsafe_from_address=Int(py=ran[0][1]))[offset]


def _iword(ran: PythonObject, offset: Int) raises -> Int:
    return Int(I32P(unsafe_from_address=Int(py=ran[0][1]))[offset])


def _take1(ran: PythonObject, offset: Int, count: Int, context: PythonObject) raises -> PythonObject:
    # `_Prog.get(off, count)`: a 1-D float32 Array.
    if offset == OUT_MARK:
        return xprep_take(ran[1][0], 0, count, 0, context["output_context"])
    return xprep_take(ran[0][0], offset, count, 0, context["output_context"])


def _take2(ran: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_Prog.get(off, (rows, cols))`: a C-order float32 Array.
    if rows * cols == 0:
        return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(rows, cols), "<f4", "C", context["output_context"])
    if offset == OUT_MARK:
        return xprep_take(ran[1][0], 0, rows, cols, context["output_context"])
    return xprep_take(ran[0][0], offset, rows, cols, context["output_context"])


def _int_list(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over fitted per-column metadata
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _list_text(values: List[Int]) -> String:
    # A Python list's str of ints.
    var text = String("[")
    for i in range(len(values)):  # driver-loop: over reported row or column indices
        text += (", " if i > 0 else "") + String(values[i])
    return text + "]"


# ---------------------------------------------------------------- shared stages
def _cs(mut p: EncProg, mode: PythonObject, entries: PythonObject, xo: Int, n: Int, d: Int, out: Int, variance: Bool = True) raises:
    """`_cs`: col_stats' six rows of d at `out`, in x_prep/blocked.mojo's
    blocked order under IDN_STATS_BLOCKED (IDENTICAL), else one thread per
    column; variance False skips the blocked variance pass."""
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"
    if blocked and (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0:
        var nb = (n + ROW_BLOCK - 1) // ROW_BLOCK
        var part = p.work(5 * nb * d)
        _stage(p.program, OP_COLB_PART, nb * d, xo, n, d, part, nb)
        _stage(p.program, OP_COLB_FOLD, d, part, nb, d, out)
        if variance:
            var ss = p.work(nb * d)
            _stage(p.program, OP_COLB_SS, nb * d, xo, n, d, out, ss, nb)
            _stage(p.program, OP_COLB_VAR, d, ss, nb, d, out)
    else:
        _stage(p.program, OP_COL_STATS, d, xo, n, d, out)


def _sel_chunks(n: Int) -> Tuple[Int, Int]:
    # `_p2m_chunks(n)`: (rows a chunk, chunks).
    var ch = max(SEL_CHUNKS, (n + SEL_CHUNKS - 1) // SEL_CHUNKS, (n + 16777215) // 16777216, 1)
    return (ch, max(1, (n + ch - 1) // ch))


def _sel(mut p: EncProg, xo: Int, n: Int, d: Int, mode: Int, dst: Int) -> Int:
    """`_p2m_sel`: per column of the (n, d) block at xo, the selected rows
    (mode 0: the non-NaN words; 1: the positive rows' indices) ascending at
    dst + j*n. Returns the offset of the d int32 counts."""
    var chunks = _sel_chunks(n)
    var ch = chunks[0]
    var nch = chunks[1]
    var cnt = p.alloc(d * nch)
    var tot = p.alloc(d)
    _stage(p.program, OP_SEL_COUNT, d * nch, xo, n, d, ch, nch, mode, cnt)
    _stage(p.program, OP_SEL_SCAN, d, cnt, nch, tot)
    _stage(p.program, OP_SEL_WRITE, d * nch, xo, n, d, ch, nch, mode, cnt, dst)
    return tot


def _bad_rows_stages(mut p: EncProg, codes: Int, n: Int, d: Int, bad: Int, strict: Int) -> Tuple[Int, Int]:
    # `_bad_rows_stages`: (ROWS, TOT) of the rows holding a `bad` code (in a
    # column whose `strict` word is nonzero, when given).
    var flags = p.alloc(n)
    var rows = p.alloc(n)
    _stage(p.program, OP_ROWFLAG, n, codes, n, d, bad, strict, flags)
    return (rows, _sel(p, flags, n, 1, 1, rows))


def _bad_rows(ran: PythonObject, staged: Tuple[Int, Int], limit: Int) raises -> List[Int]:
    var k = _iword(ran, staged[1])
    var rows = List[Int]()
    for i in range(min(k, limit)):  # driver-loop: over at most `limit` reported rows
        rows.append(_iword(ran, staged[0] + i))
    return rows^


def _seed64(value: PythonObject) raises -> Tuple[Int, Int]:
    # (0 if None else int(value)) & 0xFFFFFFFFFFFFFFFF as its (low, high) words.
    var b = Python.import_module("builtins")
    var seed = PythonObject(0) if value is None else py_call_typed(b.int, value)
    var mask = PythonObject(0xFFFFFFFF)
    var lo = Int(py=py_call(b.int.__dict__["__and__"], seed, mask))
    var hi = Int(py=py_call(b.int.__dict__["__and__"], py_call(b.int.__dict__["__rshift__"], seed, PythonObject(32)), mask))
    return (lo, hi)


def _seed_words(lo: Int, hi: Int) -> List[Int]:
    # `_seed_words` / `put_ints`: the words as signed int32.
    var words = List[Int]()
    words.append(lo - (1 << 32) if lo >= (1 << 31) else lo)
    words.append(hi - (1 << 32) if hi >= (1 << 31) else hi)
    return words^


# ---------------------------------------------------------------- shared checks
def _truth(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("builtins").bool, value))


def _is_str(value: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.str))


def _is_float(value: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.float))


def _is_nan_value(value: PythonObject) raises -> Bool:
    # `_is_nan_value`: a Python float (or subclass) that is NaN.
    if not _is_float(value):
        return False
    var v = Float64(py=value)
    return v != v


def _real_like(value: PythonObject, context: PythonObject) raises -> Bool:
    # numbers.Real without its Python __instancecheck__: Python ints, bools
    # and floats, and NumPy integer and floating scalars when connected.
    var b = Python.import_module("builtins")
    if Bool(py=py_call(b.isinstance, value, b.int)) or Bool(py=py_call(b.isinstance, value, b.float)):
        return True
    var integer = py_get(context, "integer")
    if integer is not None and Bool(py=py_call(b.isinstance, value, integer)):
        return True
    var floats = py_get(context, "float_types")
    return floats is not None and Bool(py=py_call(b.isinstance, value, floats))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call_typed(Python.import_module("builtins").float, value))


def _str_text(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").str, value))


def _fitted(model: PythonObject) raises:
    # `_check_fitted`.
    if not py_contains(py_fields(model), "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")


def _check_width(model: PythonObject, d: Int) raises:
    # `_check_width`.
    var fitted = py_fields(model)["n_features_in_"]
    if py_type(fitted) is not Python.import_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + _str_text(fitted))


def _i64_values(value: PythonObject, context: PythonObject) raises -> List[Int]:
    # A fitted int64 vector's values (`n_bins_.tolist()`).
    if String(value.dtype) != "<i8" or Int(py=value.ndim) != 1 or not classical_c_order(value, context):
        raise Error("TypeError: a fitted int64 vector must remain a contiguous int64 Array")
    var count = Int(py=value.size)
    var source = I64P(unsafe_from_address=Int(py=classical_address(value)))
    var out = List[Int]()
    for i in range(count):  # driver-loop: over a fitted per-column vector
        out.append(Int(source[i]))
    return out^


def _default_mode(context: PythonObject) raises -> PythonObject:
    # `_mode()`: the process numeric mode.
    return context["backend_fields"]["_DEFAULT_MODE"]


# ---------------------------------------------------------------- SimpleImputer
def _mark_missing(mut p: EncProg, xo: Int, count: Int, missing: PythonObject, context: PythonObject) raises -> Int:
    """`_mark_missing`: a numeric missing_values marked NaN (mark_missing);
    the input itself when missing_values is None or NaN."""
    if missing is None or _is_nan_value(missing):
        return xo
    var words = List[Float64]()
    words.append(_float(missing))
    var value = p.put_floats(words, context)
    var out = p.alloc(count)
    _stage(p.program, OP_MARK_MISSING, count, xo, count, value, out)
    return out


def _qselect(mode: PythonObject, entries: PythonObject, context: PythonObject) raises -> Bool:
    # `_prep2_qselect(mode, 1)`: MOJOLEARN_X_PREP_FAST_QSELECT=1 (read at
    # import) on the FAST tier of a Metal device binding.
    if not Bool(py=context["prep_encode_qselect"]):
        return False
    var text = py_call(Python.import_module("builtins").str, mode)
    if String(py_call(py_call(text.strip).lower)) != "fast":
        return False
    if py_get(entries, "x_prep_host_column") is not None:
        return False
    var vendor = py_get(entries, "x_prep_vendor")
    return vendor is not None and String(py_call_typed(vendor)) == "metal"


def _imputer_fit(model: PythonObject, values: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var strategy = fields["strategy"]
    var is_callable = Bool(py=py_call(b.callable, strategy))
    var name = String(strategy) if _is_str(strategy) else String("")
    if not is_callable and name != "mean" and name != "median" and name != "most_frequent" and name != "constant":
        raise Error("mojolearn: SimpleImputer strategy " + _repr(strategy) + " is not valid")
    var konst = name == "constant"
    var fill_value = fields["fill_value"]
    if konst and fill_value is not None and not _real_like(fill_value, context):
        raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var p = EncProg()
    var x_in = p.put_x(values)
    var xo = _mark_missing(p, x_in, n * d, fields["missing_values"], context)
    # only the median and the mode read the sorted columns (`imputer_nosort`);
    # the median by radix select needs no sort (`_prep2_qselect`)
    var ordered = name == "median" or name == "most_frequent"
    var qsel = name == "median" and _qselect(mode, entries, context)
    var sorts = (ordered or not xprep_r3("imputer_nosort")) and not qsel
    var so = 0
    if sorts:
        so = p.work(n * d) if ordered else p.alloc(n * d)
    var st = p.alloc(6 * d)
    var med = p.alloc(d)
    var mf = p.alloc(d)
    var halves = List[Float64]()
    halves.append(0.5)
    var half = p.put_floats(halves, context)
    if sorts:
        _stage(p.program, OP_SORT_COLS, d, xo, n, d, so, 0)
    _cs(p, mode, entries, xo, n, d, st, False)
    if name == "median" and qsel:
        _stage(p.program, OP_QUANTILE, d, xo, n, d, half, 1, med, st, 1)
    elif name == "median":
        _stage(p.program, OP_QUANTILE, d, so, n, d, half, 1, med, st)
    if name == "most_frequent":
        _stage(p.program, OP_MODE_COLS, d, so, n, d, mf, NONE)
    var comp = 0
    var ctot = 0
    if is_callable:
        # each column's non-NaN words, ascending rows, at comp + j*n
        comp = p.alloc(n * d)
        ctot = _sel(p, xo, n, d, 0, comp)
    var src = 0
    var so2 = 0
    var fo2 = 0
    if not is_callable:
        var fv = NONE
        if konst:
            var fills = List[Float64]()
            fills.append(Float64(0.0) if fill_value is None else _float(fill_value))
            fv = p.put_floats(fills, context)
        if konst or name == "mean":
            src = st + d
        elif name == "median":
            src = med
        else:
            src = mf
        so2 = p.alloc(d)
        fo2 = p.alloc(d)
        _stage(p.program, OP_IMP_STATS, d, st, src, d, 1 if _truth(fields["keep_empty_features"]) else 0, 1 if konst else 0, fv, so2, fo2)
    var ran = p.run(entries, context)
    var counts = List[Int]()
    var any_empty = False
    for j in range(d):  # driver-loop: over the per-column non-missing counts (feature metadata)
        counts.append(Int(_word(ran, st + j)))
        any_empty = any_empty or counts[j] == 0
    var keep_empty = _truth(fields["keep_empty_features"])
    var keep = List[Int]()
    var indicator = List[Int]()
    var output = context["output_context"]
    if is_callable:
        # The user's strategy, called once per column on the column's
        # non-missing words, as the Python `_fit_callable` called it.
        var statistics = classical_typed(Python.tuple(d), "<f4", output)
        var dest = _ptr(statistics)
        for j in range(d):  # cpu-route: calls the user strategy callable per column
            var kept = _take1(ran, comp + j * n, _iword(ran, ctot + j), context)
            var stat = _float(py_call_typed(strategy, kept))
            dest[j] = Float32(stat)
            if keep_empty or stat == stat:
                keep.append(j)
        py_setitem(fields, "statistics_", statistics)
        py_setitem(fields, "_fill", statistics)
    else:
        if konst or any_empty:
            py_setitem(fields, "statistics_", _take1(ran, so2, d, context))
            py_setitem(fields, "_fill", _take1(ran, fo2, d, context))
        else:
            var statistics = _take1(ran, src, d, context)
            py_setitem(fields, "statistics_", statistics)
            py_setitem(fields, "_fill", statistics)
        for j in range(d):  # driver-loop: over the kept columns (feature metadata)
            if keep_empty or counts[j] != 0:
                keep.append(j)
    if _truth(fields["add_indicator"]):
        for j in range(d):  # driver-loop: over the indicator columns (feature metadata)
            if counts[j] < n:
                indicator.append(j)
    py_setitem(fields, "_keep", _int_list(keep))
    py_setitem(fields, "_indicator", _int_list(indicator))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = ran^


def _index_list(value: PythonObject) raises -> List[Int]:
    # A fitted list of Python ints (`_keep`, `_indicator`).
    var b = Python.import_module("builtins")
    var items = py_call(b.list, value)
    var out = List[Int]()
    for i in range(len(items)):  # driver-loop: over fitted column indices (feature metadata)
        out.append(Int(py=py_call(b.int, items[i])))
    return out^


def _to_floats(values: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(values)):  # driver-loop: over parameter words
        out.append(Float64(values[i]))
    return out^


def _imputer_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var entries = _entries(fields["numeric_mode_"], context)
    var keep = _index_list(fields["_keep"])
    var indicator = _index_list(fields["_indicator"])
    var dout = len(keep)
    var p = EncProg()
    var x_in = p.put_x(values)
    var xo = _mark_missing(p, x_in, n * d, fields["missing_values"], context)
    var so = p.put_array(fields["_fill"], context)
    var ko = p.put_floats(_to_floats(keep), context)
    var out = p.alloc(n * dout)
    _stage(p.program, OP_FILL, n * dout, xo, n, d, so, out, ko, dout)
    var m = len(indicator)
    var hc = 0
    if m > 0:
        var io = p.put_floats(_to_floats(indicator), context)
        var mo = p.alloc(n * m)
        _stage(p.program, OP_NAN_MASK, n * m, xo, n, d, io, m, mo)
        hc = p.output(n * (dout + m), entries)
        _stage(p.program, OP_HCAT, n * (dout + m), out, dout, mo, m, hc)
    var ran = p.run(entries, context)
    if m == 0:
        return _take2(ran, out, n, dout, context)
    return _take2(ran, hc, n, dout + m, context)


# ---------------------------------------------------------------- KBinsDiscretizer
def _kbins_strategy(fields: PythonObject) raises -> Int:
    var strategy = fields["strategy"]
    var name = String(strategy) if _is_str(strategy) else String("")
    if name == "uniform":
        return 0
    if name == "kmeans":
        return 3
    if name == "quantile":
        var method = fields["quantile_method"]
        var text = String(method) if _is_str(method) else String("")
        if text == "averaged_inverted_cdf":
            return 1
        if text == "linear":
            return 2
        if text == "inverted_cdf":
            return 4
        if text == "closest_observation":
            return 5
        if text == "interpolated_inverted_cdf":
            return 6
        if text == "hazen":
            return 7
        if text == "weibull":
            return 8
        if text == "median_unbiased":
            return 9
        if text == "normal_unbiased":
            return 10
        raise Error("mojolearn: invalid quantile_method " + _repr(method))
    raise Error("mojolearn: invalid strategy " + _repr(strategy))


def _check_weights(source: PythonObject, n: Int, who: String, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`_check_weights`: sample_weight as float32 words (nonnegative, length
    n) and its count of positive entries, from one device program
    (col_stats for the count and minimum, p2m_sel for the positives)."""
    var w = _input(source, 1, False, context)
    if Int(py=w.size) != n:
        raise Error("mojolearn: sample_weight has " + String(Int(py=w.size)) + " entries; X has " + String(n) + " rows")
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var p = EncProg()
    var wo = p.put(w, n)
    var st = p.alloc(6)
    _cs(p, mode, entries, wo, n, 1, st, False)
    var rows = p.alloc(n)
    var pos = _sel(p, wo, n, 1, 1, rows)
    var ran = p.run(entries, context)
    var count = _word(ran, st)
    var low = _word(ran, st + 3)
    var m = _iword(ran, pos)
    # a NaN is not counted by col_stats
    if Int(count) != n or low < 0:
        raise Error("mojolearn: " + who + " sample_weight must be nonnegative")
    if m == 0:
        raise Error("mojolearn: " + who + " sample_weight is all zero")
    _ = ran^
    return (w, m)


def _kbins_fit(model: PythonObject, values: PythonObject, weight_source: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var operator = Python.import_module("operator")
    var fields = py_fields(model)
    var encode = fields["encode"]
    var encode_text = String(encode) if _is_str(encode) else String("")
    if encode_text != "onehot" and encode_text != "onehot-dense" and encode_text != "ordinal":
        raise Error("mojolearn: invalid encode " + _repr(encode))
    var strat = _kbins_strategy(fields)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var w = PythonObject(None)
    var wl = 0
    if weight_source is not None:
        var checked = _check_weights(weight_source, n, "KBinsDiscretizer", context)
        w = checked[0]
        wl = checked[1]
    var subsample = fields["subsample"]
    var resample = subsample is not None and Bool(py=py_call(operator.gt, PythonObject(n), subsample))
    var weighted_draw = resample and w is not None
    var plain_draw = resample and w is None
    var draws = 0
    var seed_lo = 0
    var seed_hi = 0
    var wsub = PythonObject(None)
    if resample:
        var seed = _seed64(fields["random_state"])
        seed_lo = seed[0]
        seed_hi = seed[1]
        draws = Int(py=py_call_typed(b.int, subsample))
    if weighted_draw:
        # the weighted resample with replacement is drawn in the fit's program
        wsub = w
        w = PythonObject(None)
    var nb = List[Int]()
    var n_bins = fields["n_bins"]
    if _int_like(n_bins, context):
        var k = Int(py=py_call_typed(b.int, n_bins))
        for _ in range(d):  # driver-loop: over the per-column n_bins argument
            nb.append(k)
    else:
        var items = py_call_typed(b.list, n_bins)
        for i in range(len(items)):  # driver-loop: over the per-column n_bins argument
            nb.append(Int(py=py_call_typed(b.int, items[i])))
    var smallest = nb[0] if len(nb) > 0 else 0
    var nbmax = 0
    for i in range(len(nb)):  # driver-loop: over the per-column n_bins argument
        smallest = min(smallest, nb[i])
        nbmax = max(nbmax, nb[i])
    if len(nb) != d or smallest < 2:
        raise Error("mojolearn: n_bins must be >= 2 per feature")
    var weighted = w is not None
    if weighted and strat != 0 and strat != 1 and strat != 3 and strat != 4:
        raise Error("mojolearn: When fitting with strategy='quantile' and sample weights, quantile_method "
                    + "should either be set to 'averaged_inverted_cdf' or 'inverted_cdf', got "
                    + "quantile_method='" + _str_text(fields["quantile_method"]) + "' instead.")
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var p = EncProg()
    var xo = p.put_x(values)
    var x_orig = xo
    if weighted_draw:
        var nbk = (n + ROW_BLOCK - 1) // ROW_BLOCK
        var wo = p.put(wsub, n)
        var bs = p.work(nbk)
        var bp = p.work(nbk)
        var ro = p.alloc(draws)
        var xs = p.alloc(draws * d)
        _stage(p.program, OP_WBLK, nbk, wo, n, bs)
        _stage(p.program, OP_WSCAN, 1, bs, nbk, bp)
        var seed_at = p.put_ints(_seed_words(seed_lo, seed_hi), context)
        _stage(p.program, OP_WDRAW, draws, wo, n, bs, bp, nbk, seed_at, ro)
        _stage(p.program, OP_RGATHER, draws * d, xo, d, ro, xs)
        xo = xs
        n = draws
    elif plain_draw:
        var so_seed = p.put_ints(_seed_words(seed_lo, seed_hi), context)
        var ro = p.alloc(draws)
        var xs = p.alloc(draws * d)
        _stage(p.program, OP_SMROWS, draws, so_seed, n, ro)
        _stage(p.program, OP_RGATHER, draws * d, xo, d, ro, xs)
        xo = xs
        n = draws
    var so = p.work(n * d) if not weighted else p.alloc(n * d)
    var st = p.alloc(6 * d)
    var nbo = p.put_floats(_to_floats(nb), context)
    var edges = p.alloc(d * (nbmax + 1))
    var ne = p.alloc(d)
    var lab = p.alloc(n * d) if strat == 3 else 0
    var cen = p.alloc(d * nbmax) if strat == 3 else 0
    _cs(p, mode, entries, xo, n, d, st)
    if not weighted:
        _stage(p.program, OP_SORT_COLS, d, xo, n, d, so, 0)
        _stage(p.program, OP_KBINS_EDGES, d, so, n, d, nbo, nbmax, strat, st, edges, ne, lab, cen)
    else:
        # The weights go up once for every stage that reads them, and the
        # weighted groups read the X input already in the arena (the Python
        # put both a second time; the same words).
        var wo = -1
        var stw = st
        if strat == 0 or strat == 3:
            # the min / max over the rows of nonzero weight
            stw = p.alloc(6 * d)
            var m = wl
            if m < n:
                wo = p.put(w, n)
                var ro = p.alloc(n)
                _ = _sel(p, wo, n, 1, 1, ro)
                var xz = p.alloc(m * d)
                _stage(p.program, OP_RGATHER, m * d, xo, d, ro, xz)
                _cs(p, mode, entries, xz, m, d, stw)
            else:
                stw = st
        if strat == 0:
            _stage(p.program, OP_KBINS_EDGES, d, so, n, d, nbo, nbmax, 0, stw, edges, ne, 0, 0)
        else:
            # `_weighted_groups`: every column's distinct values and summed weights
            var gs = p.work(n * d)
            var ug = p.alloc(2 * n * d)
            var ucnt = p.alloc(d)
            var codes = p.alloc(n * d)
            _stage(p.program, OP_SORT_COLS, d, x_orig, n, d, gs, 1)
            _stage(p.program, OP_UNIQUE_COLS, d, gs, n, d, ug, ucnt)
            _stage(p.program, OP_LOOKUP, n * d, x_orig, n, d, ug, n, ucnt, codes)
            if wo < 0:
                wo = p.put(w, n)
            _stage(p.program, OP_KBINS_GW, d, codes, n, d, wo, ug + n * d)
            if strat == 3:
                _stage(p.program, OP_KBINS_WKM, d, ug, n, d, ucnt, nbo, nbmax, st, stw, edges, cen, lab)
            else:
                # `_weighted_levels`: the percent levels i * (100 / b) (c2_grid KIND 0)
                var stride = nbmax + 1
                var lv = p.alloc(d * stride)
                var grid_nb = p.put_floats(_to_floats(nb), context)
                _stage(p.program, OP_GRID, d * stride, grid_nb, stride, 0, lv)
                var wq_nb = p.put_floats(_to_floats(nb), context)
                _stage(p.program, OP_KBINS_WQ, d, ug, n, d, ucnt, wq_nb, nbmax, lv, 1 if strat == 1 else 0, edges)
            _stage(p.program, OP_KBINS_EDGES, d, so, n, d, nbo, nbmax, 11, stw, edges, ne, 0, 0)
    var ran = p.run(entries, context)
    var output = context["output_context"]
    var bin_edges = Python.list()
    var n_bins_ = classical_typed(Python.tuple(d), "<i8", output)
    var dest = I64P(unsafe_from_address=Int(py=classical_address(n_bins_)))
    for j in range(d):  # driver-loop: over the per-column edge counts (feature metadata)
        var count = Int(_word(ran, ne + j))
        _ = py_call(bin_edges.append, _take1(ran, edges + j * (nbmax + 1), count, context))
        dest[j] = Int64(count - 1)
    py_setitem(fields, "bin_edges_", bin_edges)
    py_setitem(fields, "n_bins_", n_bins_)
    py_setitem(fields, "_edges", _take1(ran, edges, d * (nbmax + 1), context))
    py_setitem(fields, "_ne", _take1(ran, ne, d, context))
    py_setitem(fields, "_stride", nbmax + 1)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = w^
    _ = wsub^
    _ = ran^


def _prefix(widths: List[Int]) -> List[Float64]:
    # [sum(widths[:j]) for j in range(d)]
    var out = List[Float64]()
    var total = 0
    for j in range(len(widths)):  # driver-loop: over per-column block widths
        out.append(Float64(total))
        total += widths[j]
    return out^


def _kbins_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var entries = _entries(fields["numeric_mode_"], context)
    var stride = Int(py=fields["_stride"])
    var p = EncProg()
    var xo = p.put_x(values)
    var eo = p.put_array(fields["_edges"], context)
    var no = p.put_array(fields["_ne"], context)
    var codes = p.alloc(n * d)
    _stage(p.program, OP_KBINS_CODES, n * d, xo, n, d, eo, stride, no, codes)
    var encode = fields["encode"]
    if _is_str(encode) and String(encode) == "ordinal":
        var ran = p.run(entries, context)
        return _take2(ran, codes, n, d, context)
    var widths = _i64_values(fields["n_bins_"], context)
    var W = 0
    for j in range(len(widths)):  # driver-loop: over the fitted per-column bin counts
        W += widths[j]
    var so = p.put_floats(_prefix(widths), context)
    var out = p.output(n * W, entries)
    _stage(p.program, OP_ONEHOT, n * d, codes, n, d, so, NONE, W, out)
    var ran = p.run(entries, context)
    return _take2(ran, out, n, W, context)


def _kbins_inverse(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var widths = _i64_values(fields["n_bins_"], context)
    var d = len(widths)
    var n = Int(py=values[3])
    var cols = Int(py=values[4])
    var entries = _entries(fields["numeric_mode_"], context)
    var p = EncProg()
    var codes = 0
    var bad_code = 0
    var why = String()
    var encode = fields["encode"]
    if _is_str(encode) and String(encode) == "ordinal":
        _check_width(model, cols)
        var xo = p.put_x(values)
        codes = p.alloc(n * d)
        var unknown = List[Float64]()
        for _ in range(d):  # driver-loop: over the per-column missing slots
            unknown.append(-1.0)
        var zeros = List[Float64]()
        zeros.append(0.0)
        var mo = p.put_floats(unknown, context)
        var eo = p.put_floats(zeros, context)
        var uo = p.put_floats(zeros, context)
        var wo = p.put_floats(_to_floats(widths), context)
        _stage(p.program, OP_ORD_INVERSE, n * d, xo, n, d, mo, eo, 0, uo, wo, codes)
        bad_code = -2
        why = "hold codes that name no bin"
    else:
        var W = 0
        for j in range(d):  # driver-loop: over the fitted per-column bin counts
            W += widths[j]
        if cols != W:
            raise Error("mojolearn: X has " + String(cols) + " columns, expected " + String(W))
        # `_block_argmax`
        var xo = p.put_x(values)
        var so = p.put_floats(_prefix(widths), context)
        var wo = p.put_floats(_to_floats(widths), context)
        codes = p.alloc(n * d)
        _stage(p.program, OP_BLOCK_ARGMAX, n * d, xo, n, W, d, so, wo, NONE, 1, codes)
        bad_code = -1
        why = "can not be inverted because they contain all zeros"
    var out = p.output(n * d, entries)
    var eo2 = p.put_array(fields["_edges"], context)
    _stage(p.program, OP_KBINS_INVERSE, n * d, codes, n, d, eo2, Int(py=fields["_stride"]), out)
    var staged = _bad_rows_stages(p, codes, n, d, bad_code, NONE)
    var ran = p.run(entries, context)
    var bad = _bad_rows(ran, staged, 10)
    if len(bad) > 0:
        raise Error("mojolearn: samples " + _list_text(bad) + " " + why)
    return _take2(ran, out, n, d, context)


# ---------------------------------------------------------------- categories
def _fast_unique(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_fast_on("UNIQUE", mode)`: the FAST binding exports x_prep_fast_unique.
    return String(mode) == "fast" and py_get(entries, "x_prep_fast_unique") is not None


def _cls2_cat(mode: PythonObject, entries: PythonObject) raises -> Int:
    # `_cls2_cat`: bit 1 PACK, bit 2 PRESENT on the FAST binding built with them.
    if String(mode) != "fast":
        return 0
    var entry = py_get(entries, "x_prep_cls2_cat")
    return Int(py=py_call_typed(entry)) if entry is not None else 0


def _label_chunk(n: Int) -> Int:
    # `_label_chunk`: max(1024, isqrt(n) + 1) by integer Newton steps.
    var r = n
    if n > 1:
        var y = (r + 1) // 2
        while y < r:  # driver-loop: over integer Newton steps of a row count
            r = y
            y = (y + n // y) // 2
    return max(1024, r + 1)


def _uniq_scans(mut p: EncProg, base: Int, rows: Int, d: Int, stride: Int, counts: Int, out: Int, presence: Bool):
    """One device run scan per column (uniq_count or pres_count, uniq_scan,
    uniq_write or pres_write) over the column at base + c * stride."""
    var ch = CAT_CH if presence else _label_chunk(rows)
    var nch = (rows + ch - 1) // ch
    for c in range(d):  # driver-loop: stages one device scan per column (feature count)
        var cnt = p.work(nch)
        var off = p.work(nch)
        _stage(p.program, OP_PRES_COUNT if presence else OP_UNIQ_COUNT, nch, base + c * stride, rows, ch, cnt)
        _stage(p.program, OP_UNIQ_SCAN, 1, cnt, nch, off, counts + c)
        _stage(p.program, OP_PRES_WRITE if presence else OP_UNIQ_WRITE, nch, base + c * stride, rows, ch, off, out + c * stride)


def _fit_categories_cls2(values: PythonObject, mode: PythonObject, entries: PythonObject, bits: Int, context: PythonObject) raises -> PythonObject:
    """`_fit_categories_cls2`: the lists packed into a small host region;
    None when a column does not suit PRESENT or the values do not fit."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var p = EncProg()
    var xo = p.put_x(values)
    var co = p.alloc(d)
    var cap = min(n * d, CAT_CAP)
    var pk = p.alloc(cap)
    var bad = -1
    if (bits & 2) != 0:
        bad = p.alloc(d)
        var fl = p.work(d * CAT_R)
        var uo = p.work(d * CAT_R)
        _stage(p.program, OP_CAT_ZERO, d * CAT_R, fl)
        _stage(p.program, OP_CAT_PRESENT, n * d, xo, n, d, CAT_R, fl, bad)
        _uniq_scans(p, fl, CAT_R, d, CAT_R, co, uo, True)
        _stage(p.program, OP_CAT_PACK, d * CAT_R, uo, CAT_R, d, co, cap, pk)
    else:
        var so = p.work(n * d)
        var uo = p.work(n * d)
        _stage(p.program, OP_SORT_COLS, d, xo, n, d, so, 1)
        _uniq_scans(p, so, n, d, n, co, uo, False)
        _stage(p.program, OP_CAT_PACK, n * d, uo, n, d, co, cap, pk)
    var ran = p.run(entries, context)
    if bad >= 0:
        for j in range(d):  # driver-loop: over the per-column overflow flags (feature metadata)
            if _iword(ran, bad + j) != 0:
                return PythonObject(None)
    var counts = List[Int]()
    var total = 0
    for c in range(d):  # driver-loop: over the per-column category counts (feature metadata)
        counts.append(Int(_word(ran, co + c)))
        total += counts[c]
    if total > cap:
        return PythonObject(None)
    var out = Python.list()
    var o = 0
    for c in range(d):  # driver-loop: slices the per-column category outputs (feature count)
        _ = py_call(out.append, _take1(ran, pk + o, counts[c], context))
        o += counts[c]
    return out


def _fit_categories(values: PythonObject, mode: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_fit_categories`: per column the sorted distinct values (-0.0 folded
    into 0.0), by a sort per column and a run scan."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var cls2 = _cls2_cat(mode, entries)
    if (cls2 & 1) != 0 and _fast_unique(mode, entries):
        var got = _fit_categories_cls2(values, mode, entries, cls2, context)
        if got is not None:
            return got
    var p = EncProg()
    var xo = p.put_x(values)
    var so = p.work(n * d)
    var uo = p.alloc(n * d)
    var co = p.alloc(d)
    _stage(p.program, OP_SORT_COLS, d, xo, n, d, so, 1)
    if _fast_unique(mode, entries):
        _uniq_scans(p, so, n, d, n, co, uo, False)
    else:
        _stage(p.program, OP_UNIQUE_COLS, d, so, n, d, uo, co)
    var ran = p.run(entries, context)
    var out = Python.list()
    for c in range(d):  # driver-loop: slices the per-column category outputs (feature count)
        _ = py_call(out.append, _take1(ran, uo + c * n, Int(_word(ran, co + c)), context))
    return out


def _fit_categories_with_codes(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_fit_categories_with_codes` (C08): the categories and the (n, d)
    inverse from one program (sort_cols, unique_inverse)."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var p = EncProg()
    var xo = p.put_x(values)
    var sorted = p.work(n * d)
    var dictionary = p.alloc(n * d)
    var counts = p.alloc(d)
    var inverse = p.alloc(n * d)
    _stage(p.program, OP_SORT_COLS, d, xo, n, d, sorted, 1)
    _stage(p.program, OP_UNIQUE_INVERSE, d, sorted, n, d, dictionary, counts, xo, inverse)
    var ran = p.run(entries, context)
    var out = Python.list()
    for c in range(d):  # small-loop(d: per-feature category dictionaries): wraps fitted metadata per feature
        _ = py_call(out.append, _take1(ran, dictionary + c * n, Int(_word(ran, counts + c)), context))
    return (out, _take2(ran, inverse, n, d, context))


def _c08_route(mode: PythonObject, entries: PythonObject, bit: Int) raises -> Bool:
    # `_c08_route`: the IDENTICAL binding's x_prep_c08_routes bit.
    if String(mode) != "identical":
        return False
    var entry = py_get(entries, "x_prep_c08_routes")
    return entry is not None and (Int(py=py_call_typed(entry)) & bit) != 0


def _user_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # A user list argument's values as a Python list (`v.tolist()` for an
    # array type, a mojolearn Array read as its float32 words).
    var b = Python.import_module("builtins")
    if py_type(value) is context["array_class"]:
        var out = Python.list()
        if Int(py=value.size) == 0:
            return out
        var flat = _input(value, 1, False, context)
        var source = _ptr(flat)
        for i in range(Int(py=flat.size)):  # driver-loop: over one user category or drop list
            _ = py_call(out.append, PythonObject(Float64(source[i])))
        return out
    if Bool(py=py_call(b.hasattr, value, "tolist")):
        return py_call(b.list, py_call(value.tolist))
    return py_call(b.list, value)


def _f32(value: PythonObject) raises -> Float32:
    # array.array("f", [float(v)])[0]
    return Float32(_float(value))


def _cat_values(cats: PythonObject, context: PythonObject) raises -> List[Float32]:
    # A fitted category Array's float32 words.
    var out = List[Float32]()
    var count = Int(py=cats.size)
    if count == 0:
        return out^
    var flat = _input(cats, 1, False, context)
    var source = _ptr(flat)
    for i in range(count):  # driver-loop: over one column's fitted categories
        out.append(source[i])
    return out^


def _f32_array(values: List[Float32], context: PythonObject) raises -> PythonObject:
    # Array.from_list(values, "<f4"); an empty list is an empty Array.
    if len(values) == 0:
        return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(0), "<f4", "C", context["output_context"])
    var out = classical_typed(Python.tuple(len(values)), "<f4", context["output_context"])
    var dest = _ptr(out)
    for i in range(len(values)):  # driver-loop: over one column's category words
        dest[i] = values[i]
    return out


def _category_block(mut p: EncProg, categories: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    # `_category_block`: every column's categories in one (d, kmax) block.
    var d = len(categories)
    var kmax = 0
    for j in range(d):  # driver-loop: over the per-column category arrays
        kmax = max(kmax, Int(py=categories[j].size))
    var block = classical_typed(Python.tuple(max(d * kmax, 1)), "<f4", context["output_context"])
    var dest = _ptr(block)
    for j in range(d):  # driver-loop: over the per-column category arrays
        var cats = _cat_values(categories[j], context)
        for i in range(len(cats)):  # driver-loop: over one column's categories
            dest[j * kmax + i] = cats[i]
    return (p.put(block, max(d * kmax, 1)), kmax)


def _sizes(categories: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for j in range(len(categories)):  # driver-loop: over the per-column category arrays
        out.append(Int(py=categories[j].size))
    return out^


def _codes(mut p: EncProg, values: PythonObject, categories: PythonObject, context: PythonObject, device_only: Bool = False) raises -> Tuple[Int, Int]:
    """`_codes`: each element's category index (or -1) and, unless
    device_only, each column's unknown count. Returns (codes, neg)."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var xo = p.put_x(values)
    var block = _category_block(p, categories, context)
    var co = p.put_floats(_to_floats(_sizes(categories)), context)
    var codes = p.scratch(n * d) if device_only else p.alloc(n * d)
    _stage(p.program, OP_LOOKUP, n * d, xo, n, d, block[0], block[1], co, codes)
    if device_only:
        return (codes, -1)
    var neg = p.alloc(d)
    _stage(p.program, OP_COUNT_NEG, d, codes, n, d, neg)
    return (codes, neg)


def _unknown_columns(ran: PythonObject, neg: Int, d: Int) raises -> List[Int]:
    # The columns whose unknown count the device set.
    var bad = List[Int]()
    for j in range(d):  # driver-loop: over the per-column unknown flags (feature metadata)
        if _word(ran, neg + j) > 0:
            bad.append(j)
    return bad^


def _given_categories(categories: PythonObject, values: PythonObject, mode: PythonObject, entries: PythonObject,
                      check_unknown: Bool, who: String, context: PythonObject) raises -> PythonObject:
    """`_given_categories`: one numeric list per column, sorted ascending with
    at most a NaN last, stored as float32; with check_unknown a training
    value outside its column's list is refused."""
    var b = Python.import_module("builtins")
    var d = Int(py=values[4])
    var lists = py_call(b.list, categories)
    if len(lists) != d:
        raise Error("mojolearn: " + who + " categories has " + String(len(lists)) + " lists; X has " + String(d) + " features")
    var out = Python.list()
    for j in range(d):  # driver-loop: validates the user categories argument (user category lists)
        var raw = _user_values(lists[j], context)
        var f32 = List[Float32]()
        for i in range(len(raw)):  # driver-loop: over one user category list
            f32.append(_f32(raw[i]))
        var nums = List[Float32]()
        for i in range(len(f32)):  # driver-loop: over one user category list
            if f32[i] == f32[i]:
                nums.append(f32[i])
        var total = len(f32)
        if len(nums) < total - 1 or (len(nums) < total and f32[total - 1] == f32[total - 1]):
            raise Error("mojolearn: " + who + " categories[" + String(j) + "]: nan must be the last category")
        for i in range(1, len(nums)):  # driver-loop: over one user category list
            if nums[i] < nums[i - 1]:
                raise Error("mojolearn: " + who + " unsorted categories are not supported for numerical categories")
        for i in range(1, len(nums)):  # driver-loop: over one user category list
            if nums[i] == nums[i - 1]:
                raise Error("mojolearn: " + who + " categories[" + String(j) + "] has values equal in float32")
        if total == 0:
            raise Error("mojolearn: " + who + " categories[" + String(j) + "] is empty")
        var canon = List[Float32]()
        for i in range(len(nums)):  # driver-loop: over one user category list
            canon.append(Float32(0.0) if nums[i] == 0 else nums[i])
        if len(nums) < total:
            canon.append(Float32(Float64(py=py_call(b.float, "nan"))))
        _ = py_call(out.append, _f32_array(canon, context))
    if check_unknown:
        var p = EncProg()
        var staged = _codes(p, values, out, context)
        var ran = p.run(entries, context)
        var bad = _unknown_columns(ran, staged[1], d)
        if len(bad) > 0:
            raise Error("mojolearn: " + who + " found unknown categories in column(s) " + _list_text(bad) + " during fit")
    return out


# ---------------------------------------------------------------- infrequent categories
def _integral(value: PythonObject, context: PythonObject) raises -> Bool:
    # numbers.Integral: Python ints (bool included), NumPy integers.
    return _int_like(value, context)


def _is_bool(value: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.bool))


def _cmp(name: String, a: PythonObject, b: PythonObject) raises -> Bool:
    # A Python comparison (operator.<name>) of two numbers.
    var operator = Python.import_module("operator")
    return _truth(py_call(py_get(operator.__dict__, PythonObject(name)), a, b))


def _check_infrequent_params(fields: PythonObject, context: PythonObject) raises -> Bool:
    """`_check_infrequent_params`: the reference's constraints; True when
    grouping is on."""
    var mf = fields["min_frequency"]
    var mc = fields["max_categories"]
    if mf is not None:
        var ok = (_integral(mf, context) and not _is_bool(mf) and _cmp("ge", mf, PythonObject(1))) or (
            _real_like(mf, context) and not _integral(mf, context) and _cmp("lt", PythonObject(0), mf) and _cmp("lt", mf, PythonObject(1)))
        if not ok:
            raise Error("mojolearn: min_frequency must be an int >= 1 or a float in (0, 1), got " + _repr(mf))
    if mc is not None and not (_integral(mc, context) and not _is_bool(mc) and _cmp("ge", mc, PythonObject(1))):
        raise Error("mojolearn: max_categories must be an int >= 1, got " + _repr(mc))
    return mc is not None or mf is not None


def _infrequent_threshold(n: Int, min_frequency: PythonObject, context: PythonObject) raises -> Int:
    """`_infrequent_threshold`: an int min_frequency as is; a fraction f,
    ceil(n * f) in binary64; none, 0."""
    if min_frequency is None:
        return 0
    var b = Python.import_module("builtins")
    if _integral(min_frequency, context):
        return Int(py=py_call(b.int, min_frequency))
    return Int(ceil(Float64(n) * _float(min_frequency)))


def _last_is_nan(cats: PythonObject, context: PythonObject) raises -> Bool:
    var values = _cat_values(cats, context)
    return len(values) > 0 and values[len(values) - 1] != values[len(values) - 1]


def _fit_infrequent(model: PythonObject, values: PythonObject, mode: PythonObject, entries: PythonObject,
                    ignore_missing: Bool, inverse: PythonObject, context: PythonObject) raises:
    """`_fit_infrequent`: one program (code_counts, c2_inf_m0, c2_inf_m1,
    c2_inf_map) whose masks and codes come back as the fitted per-column
    `_infrequent`, `_grouping` and `infrequent_categories_`."""
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var cats_all = fields["categories_"]
    var ks = List[Int]()
    var kmax = 0
    for j in range(len(cats_all)):  # driver-loop: one size per column
        var size = Int(py=cats_all[j].size)
        ks.append(size - 1 if (ignore_missing and size > 0 and _last_is_nan(cats_all[j], context)) else size)
        kmax = max(kmax, size)
    var p = EncProg()
    var codes = 0
    if inverse is None:
        codes = _codes(p, values, cats_all, context)[0]
    else:
        codes = p.put_array2(inverse)
    var cnt = p.alloc(d * kmax)
    var kc = p.put_floats(_to_floats(ks), context)
    var m0 = p.work(d * kmax)
    var m1 = p.alloc(d * kmax)
    var mp = p.alloc(d * kmax)
    var max_categories = fields["max_categories"]
    var maxc = -1 if max_categories is None else Int(py=py_call(Python.import_module("builtins").int, max_categories))
    _stage(p.program, OP_CODE_COUNTS, d, codes, n, d, kmax, cnt)
    _stage(p.program, OP_INF_M0, d * kmax, cnt, kc, kmax, _infrequent_threshold(n, fields["min_frequency"], context), m0)
    _stage(p.program, OP_INF_M1, d * kmax, cnt, kc, kmax, m0, maxc, m1)
    _stage(p.program, OP_INF_MAP, d * kmax, kc, kmax, m1, mp)
    var ran = p.run(entries, context)
    var infrequent = Python.list()
    var grouping = Python.list()
    var categories = Python.list()
    for j in range(len(ks)):  # driver-loop: the device's tables as the fitted per-column lists
        var inf = List[Int]()
        for i in range(ks[j]):  # driver-loop: over one column's categories
            if _iword(ran, m1 + j * kmax + i) != 0:
                inf.append(i)
        if len(inf) == 0:
            _ = py_call(infrequent.append, PythonObject(None))
            _ = py_call(grouping.append, PythonObject(None))
            _ = py_call(categories.append, PythonObject(None))
            continue
        _ = py_call(infrequent.append, _int_list(inf))
        var group = List[Int]()
        for i in range(ks[j]):  # driver-loop: over one column's categories
            group.append(_iword(ran, mp + j * kmax + i))
        _ = py_call(grouping.append, _int_list(group))
        var values_j = _cat_values(cats_all[j], context)
        var picked = List[Float32]()
        for i in range(len(inf)):  # driver-loop: over one column's infrequent categories
            picked.append(values_j[inf[i]])
        _ = py_call(categories.append, _f32_array(picked, context))
    py_setitem(fields, "_infrequent", infrequent)
    py_setitem(fields, "_grouping", grouping)
    py_setitem(fields, "infrequent_categories_", categories)
    _ = ran^


def _list_max(values: List[Int]) -> Int:
    var top = values[0] if len(values) > 0 else 0
    for i in range(len(values)):  # driver-loop: over one column grouping
        top = max(top, values[i])
    return top


def _grouping_table(mut p: EncProg, grouping: PythonObject, inverse: Bool, context: PythonObject) raises -> Tuple[Int, Int, Int]:
    """`_grouping_table`: (MAP, MSTRIDE, NMAP) for remap_codes: category ->
    grouped code, or (inverse) grouped code -> category index with the
    infrequent code -> -3."""
    var tables = List[List[Int]]()
    for j in range(len(grouping)):  # driver-loop: builds the category grouping table (per-column category groupings)
        if grouping[j] is None:
            tables.append(List[Int]())
            continue
        var g = _index_list(grouping[j])
        if not inverse:
            tables.append(g^)
            continue
        var nf = _list_max(g)
        var back = List[Int](length=nf + 1, fill=0)
        for i in range(len(g)):  # driver-loop: inverts one column grouping table
            if g[i] < nf:
                back[g[i]] = i
        back[nf] = -3
        tables.append(back^)
    var stride = 1
    for j in range(len(tables)):  # driver-loop: over the per-column grouping tables
        stride = max(stride, len(tables[j]))
    var flat = List[Float64]()
    var lens = List[Float64]()
    for j in range(len(tables)):  # driver-loop: pads the per-column grouping tables
        for i in range(stride):  # driver-loop: over one padded grouping table
            flat.append(Float64(tables[j][i]) if i < len(tables[j]) else Float64(0.0))
        lens.append(Float64(len(tables[j])))
    var mo = p.put_floats(flat, context)
    var no = p.put_floats(lens, context)
    return (mo, stride, no)


def _remap(mut p: EncProg, codes: Int, n: Int, d: Int, table: Tuple[Int, Int, Int], neg: Int) -> Int:
    var out = p.alloc(n * d)
    _stage(p.program, OP_REMAP_CODES, n * d, codes, n * d, d, table[0], table[1], table[2], neg, out)
    return out


def _gather_categories(mut p: EncProg, codes: Int, n: Int, d: Int, categories: PythonObject, context: PythonObject) raises -> Int:
    var block = _category_block(p, categories, context)
    var out = p.alloc(n * d)
    _stage(p.program, OP_CAT_GATHER, n * d, codes, n, d, block[0], block[1], out)
    return out


def _scalar(mut p: EncProg, value: Float64, context: PythonObject) raises -> Int:
    var words = List[Float64]()
    words.append(value)
    return p.put_floats(words, context)


def _num_text(value: PythonObject) raises -> String:
    # A number in an f-string (format(value, "")).
    return String(py_call(Python.import_module("builtins").format, value, ""))


# ---------------------------------------------------------------- OrdinalEncoder
def _ordinal_fit(model: PythonObject, values: PythonObject, capture: Bool, context: PythonObject) raises -> PythonObject:
    """OrdinalEncoder.fit; with `capture` (the C08 fit_transform) the fit's
    (n, d) inverse is returned for the transform, else None."""
    var fields = py_fields(model)
    var grouping = _check_infrequent_params(fields, context)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    if handle_text != "error" and handle_text != "use_encoded_value":
        raise Error("mojolearn: invalid handle_unknown " + _repr(handle))
    var unknown_value = fields["unknown_value"]
    if handle_text == "use_encoded_value" and not _real_like(unknown_value, context):
        raise Error("TypeError: mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
    var emv = fields["encoded_missing_value"]
    if not _real_like(emv, context):
        raise Error("TypeError: mojolearn: encoded_missing_value must be a number (or NaN)")
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var auto = _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
    var inverse = PythonObject(None)
    if capture and auto:
        var got = _fit_categories_with_codes(values, entries, context)
        py_setitem(fields, "categories_", got[0])
        inverse = got[1]
    elif auto:
        py_setitem(fields, "categories_", _fit_categories(values, mode, entries, context))
    else:
        py_setitem(fields, "categories_", _given_categories(fields["categories"], values, mode, entries, handle_text == "error", "OrdinalEncoder", context))
    var cats = fields["categories_"]
    var missing = List[Int]()
    var any_missing = False
    for j in range(len(cats)):  # driver-loop: per-column NaN category slot (fitted per-column category arrays)
        var size = Int(py=cats[j].size)
        missing.append(size - 1 if size > 0 and _last_is_nan(cats[j], context) else -1)
        any_missing = any_missing or missing[j] >= 0
    py_setitem(fields, "_missing", _int_list(missing))
    py_setitem(fields, "_infrequent", PythonObject(None))
    py_setitem(fields, "_grouping", PythonObject(None))
    if grouping:
        _fit_infrequent(model, values, mode, entries, True, inverse, context)
    var cards = List[Int]()
    for j in range(len(cats)):  # driver-loop: per-column category cardinalities (fitted per-column category arrays)
        cards.append(Int(py=cats[j].size) - (1 if missing[j] >= 0 else 0))
    if grouping:
        var groups = fields["_grouping"]
        for j in range(len(cards)):  # driver-loop: grouped per-column cardinalities
            if groups[j] is not None:
                cards[j] = _list_max(_index_list(groups[j])) + 1
    if handle_text == "use_encoded_value" and not _is_nan_value(unknown_value):
        for j in range(len(cards)):  # driver-loop: validates unknown_value against cardinalities
            if _cmp("le", PythonObject(0), unknown_value) and _cmp("lt", unknown_value, PythonObject(cards[j])):
                raise Error("mojolearn: the used value for unknown_value " + _num_text(unknown_value) + " is one of the "
                            + "values already used for encoding the seen categories.")
    if any_missing and not _is_nan_value(emv):
        var bad = List[Int]()
        for j in range(len(cards)):  # driver-loop: validates encoded_missing_value argument
            if missing[j] >= 0 and _cmp("le", PythonObject(0), emv) and _cmp("lt", emv, PythonObject(cards[j])):
                bad.append(j)
        if len(bad) > 0:
            raise Error("mojolearn: encoded_missing_value (" + _num_text(emv) + ") is already "
                        + "used to encode a known category in features: " + _list_text(bad))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    return inverse


def _ordinal_transform(model: PythonObject, values: PythonObject, inverse: PythonObject, context: PythonObject) raises -> PythonObject:
    """OrdinalEncoder `_transform_encoded`: lookup (or the C08 fit's
    inverse), the grouping remap, where_neg and where_code."""
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var entries = _entries(fields["numeric_mode_"], context)
    var p = EncProg()
    var codes = 0
    var neg = 0
    if inverse is None:
        var staged = _codes(p, values, fields["categories_"], context)
        codes = staged[0]
        neg = staged[1]
    else:
        codes = p.put_array2(inverse)
        neg = p.alloc(d)
        _stage(p.program, OP_COUNT_NEG, d, codes, n, d, neg)
    var out = codes
    if fields["_grouping"] is not None:
        var table = _grouping_table(p, fields["_grouping"], False, context)
        out = _remap(p, codes, n, d, table, NONE)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    if handle_text == "use_encoded_value":
        var val = _scalar(p, _float(fields["unknown_value"]), context)
        var src = out
        out = p.alloc(n * d)
        _stage(p.program, OP_WHERE_NEG, n * d, src, n * d, val, out)
    var missing = _index_list(fields["_missing"])
    var any_missing = False
    for j in range(len(missing)):  # driver-loop: checks for a NaN category slot (per-column NaN category slots)
        any_missing = any_missing or missing[j] >= 0
    if any_missing:
        var src = out
        out = p.alloc(n * d)
        var mo = p.put_floats(_to_floats(missing), context)
        var eo = _scalar(p, _float(fields["encoded_missing_value"]), context)
        _stage(p.program, OP_WHERE_CODE, n * d, codes, n, d, mo, eo, src, out)
    var ran = p.run(entries, context)
    if handle_text == "error":
        var bad = _unknown_columns(ran, neg, d)
        if len(bad) > 0:
            raise Error("mojolearn: OrdinalEncoder found unknown categories in column(s) " + _list_text(bad) + " during transform")
    return _take2(ran, out, n, d, context)


def _ordinal_inverse(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """OrdinalEncoder.inverse_transform: ord_inverse (`_inverse_codes`), the
    inverse grouping remap, cat_gather and the bad-row compaction."""
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var entries = _entries(fields["numeric_mode_"], context)
    var cats = fields["categories_"]
    var handle = fields["handle_unknown"]
    var encoded = _is_str(handle) and String(handle) == "use_encoded_value"
    var p = EncProg()
    var ncat = _sizes(cats)
    var grouped = fields["_grouping"] is not None
    var back = (0, 0, 0)
    if grouped:
        var groups = fields["_grouping"]
        for j in range(len(ncat)):  # driver-loop: per-column grouped category counts
            if groups[j] is not None:
                ncat[j] = _list_max(_index_list(groups[j])) + 1
        back = _grouping_table(p, groups, True, context)
    # `_inverse_codes`
    var xo = p.put_x(values)
    var mo = p.put_floats(_to_floats(_index_list(fields["_missing"])), context)
    var eo = _scalar(p, _float(fields["encoded_missing_value"]), context)
    var uo = _scalar(p, _float(fields["unknown_value"]) if encoded else Float64(0.0), context)
    var no = p.put_floats(_to_floats(ncat), context)
    var codes = p.alloc(n * d)
    _stage(p.program, OP_ORD_INVERSE, n * d, xo, n, d, mo, eo, 1 if encoded else 0, uo, no, codes)
    if grouped:
        codes = _remap(p, codes, n, d, back, NONE)
    var out = _gather_categories(p, codes, n, d, cats, context)
    var staged = _bad_rows_stages(p, codes, n, d, -2, NONE)
    var ran = p.run(entries, context)
    var bad = _bad_rows(ran, staged, 10)
    if len(bad) > 0:
        raise Error("mojolearn: rows " + _list_text(bad) + " hold codes that name no category")
    return _take2(ran, out, n, d, context)


# ---------------------------------------------------------------- OneHotEncoder
def _grouped_sizes(fields: PythonObject) raises -> List[Int]:
    var sizes = _sizes(fields["categories_"])
    var groups = fields["_grouping"]
    if groups is not None:
        for j in range(len(sizes)):  # driver-loop: per-column grouped category counts
            if groups[j] is not None:
                sizes[j] = _list_max(_index_list(groups[j])) + 1
    return sizes^


def _set_drop_idx(model: PythonObject, context: PythonObject) raises:
    """`_set_drop_idx`: `_drop_after` in grouped codes, drop_idx_ in category
    indices (None, or one int or None per column)."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var sizes = _grouped_sizes(fields)
    var d = len(sizes)
    var groups = fields["_grouping"]
    var drop = fields["drop"]
    if drop is None:
        py_setitem(fields, "_drop_after", PythonObject(None))
        py_setitem(fields, "drop_idx_", PythonObject(None))
        return
    var after = Python.list()
    var text = String(drop) if _is_str(drop) else String("")
    if text == "first":
        for _ in range(d):  # driver-loop: over the per-column drop indices
            _ = py_call(after.append, PythonObject(0))
    elif text == "if_binary":
        for j in range(d):  # driver-loop: drop index for binary columns
            _ = py_call(after.append, PythonObject(0) if sizes[j] == 2 else PythonObject(None))
    else:
        var vals = _user_values(drop, context)
        if len(vals) != d:
            raise Error("mojolearn: `drop` should have length equal to the number of features ("
                        + String(d) + "), got " + String(len(vals)))
        var missing = String()
        var any_missing = False
        var cats = fields["categories_"]
        for j in range(d):  # driver-loop: validates the user drop argument (per-column drop arguments)
            var v = vals[j]
            var cl = _cat_values(cats[j], context)
            var hit = -1
            if _is_nan_value(v):
                if len(cl) > 0 and cl[len(cl) - 1] != cl[len(cl) - 1]:
                    hit = len(cl) - 1
            else:
                var fv = _f32(v)
                for i in range(len(cl)):  # driver-loop: finds the user drop category in its column
                    if cl[i] == fv:
                        hit = i
                        break
            if hit < 0:
                missing += ("\n" if any_missing else "") + "Category: " + _str_text(v) + ", Feature: " + String(j)
                any_missing = True
                continue
            var i = hit
            if groups is not None and groups[j] is not None:
                var infrequent = _index_list(fields["_infrequent"][j])
                for k in range(len(infrequent)):  # driver-loop: over one column's infrequent categories
                    if infrequent[k] == i:
                        raise Error("mojolearn: Unable to drop category " + _repr(PythonObject(Float64(cl[i]))) + " from feature "
                                    + String(j) + " because it is infrequent")
                i = _index_list(groups[j])[i]
            _ = py_call(after.append, PythonObject(i))
        if any_missing:
            raise Error("mojolearn: The following categories were supposed to be dropped, but were not "
                        + "found in the training data.\n" + missing)
    py_setitem(fields, "_drop_after", after)
    var idx = Python.list()
    for j in range(d):  # driver-loop: maps drop indices through the grouping (per-column drop indices)
        var a = after[j]
        if a is None or groups is None or groups[j] is None:
            _ = py_call(idx.append, a)
            continue
        var g = _index_list(groups[j])
        var target = Int(py=a)
        var found = -1
        for i in range(len(g)):  # driver-loop: over one column grouping
            if g[i] == target:
                found = i
                break
        if found < 0:
            raise Error(String(target) + " is not in list")
        _ = py_call(idx.append, PythonObject(found))
    py_setitem(fields, "drop_idx_", idx)


def _onehot_drops(fields: PythonObject) raises -> List[Int]:
    # `_widths`' drop indices per column in grouped codes (-1: none).
    var d = len(fields["categories_"])
    var after = fields["_drop_after"]
    var drops = List[Int]()
    for j in range(d):  # driver-loop: per-column drop indices
        var dr = PythonObject(None) if after is None else after[j]
        drops.append(-1 if dr is None else Int(py=dr))
    return drops^


def _onehot_widths(fields: PythonObject) raises -> List[Int]:
    # `_widths`: per-column one-hot block widths.
    var sizes = _grouped_sizes(fields)
    var drops = _onehot_drops(fields)
    var widths = List[Int]()
    for j in range(len(sizes)):  # driver-loop: per-column one-hot block widths
        widths.append(sizes[j] - (0 if drops[j] < 0 else 1))
    return widths^


def _onehot_fit(model: PythonObject, values: PythonObject, capture: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var grouping = _check_infrequent_params(fields, context)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    if handle_text != "error" and handle_text != "ignore" and handle_text != "infrequent_if_exist" and handle_text != "warn":
        raise Error("mojolearn: OneHotEncoder handle_unknown=" + _repr(handle) + " is not valid")
    var drop = fields["drop"]
    if _is_str(drop) and String(drop) != "first" and String(drop) != "if_binary":
        raise Error("mojolearn: OneHotEncoder drop must be None, 'first', 'if_binary' or one category "
                    + "per feature")
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var auto = _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
    var inverse = PythonObject(None)
    if capture and auto:
        var got = _fit_categories_with_codes(values, entries, context)
        py_setitem(fields, "categories_", got[0])
        inverse = got[1]
    elif auto:
        py_setitem(fields, "categories_", _fit_categories(values, mode, entries, context))
    else:
        py_setitem(fields, "categories_", _given_categories(fields["categories"], values, mode, entries, handle_text == "error", "OneHotEncoder", context))
    py_setitem(fields, "_infrequent", PythonObject(None))
    py_setitem(fields, "_grouping", PythonObject(None))
    if grouping:
        _fit_infrequent(model, values, mode, entries, False, inverse, context)
    _set_drop_idx(model, context)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    return inverse


def _onehot_transform(model: PythonObject, values: PythonObject, inverse: PythonObject, context: PythonObject) raises -> PythonObject:
    """OneHotEncoder `_transform_encoded`: lookup (or the C08 fit's inverse),
    the grouping remap with the unknown slots, and onehot."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var entries = _entries(fields["numeric_mode_"], context)
    var widths = _onehot_widths(fields)
    var drops = _onehot_drops(fields)
    var W = 0
    for j in range(len(widths)):  # driver-loop: total one-hot output width
        W += widths[j]
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    var p = EncProg()
    var codes = 0
    var neg = 0
    if inverse is None:
        var staged = _codes(p, values, fields["categories_"], context)
        codes = staged[0]
        neg = staged[1]
    else:
        codes = p.put_array2(inverse)
        neg = p.alloc(d)
        _stage(p.program, OP_COUNT_NEG, d, codes, n, d, neg)
    var groups = fields["_grouping"]
    if groups is not None:
        # `_unknown_to`: the grouped code an unknown value takes
        var unk = NONE
        if handle_text == "infrequent_if_exist" or handle_text == "warn":
            var slots = List[Int]()
            for j in range(len(groups)):  # driver-loop: per-column infrequent slot
                slots.append(-1 if groups[j] is None else _list_max(_index_list(groups[j])))
            var table = _grouping_table(p, groups, False, context)
            unk = p.put_floats(_to_floats(slots), context)
            codes = _remap(p, codes, n, d, table, unk)
        else:
            var table = _grouping_table(p, groups, False, context)
            codes = _remap(p, codes, n, d, table, NONE)
    var so = p.put_floats(_prefix(widths), context)
    var do = p.put_floats(_to_floats(drops), context)
    var out = p.output(n * W, entries)
    _stage(p.program, OP_ONEHOT, n * d, codes, n, d, so, do, W, out)
    var ran = p.run(entries, context)
    var dropping = fields["drop"] is not None
    if handle_text == "error":
        var bad = _unknown_columns(ran, neg, d)
        if len(bad) > 0:
            raise Error("mojolearn: OneHotEncoder found unknown categories in column(s) " + _list_text(bad) + " during transform")
    elif handle_text == "warn" or (dropping and (handle_text == "ignore" or handle_text == "infrequent_if_exist")):
        var bad = _unknown_columns(ran, neg, d)
        if len(bad) > 0:
            var where = String("encoded as the infrequent category") if handle_text != "ignore" else String("encoded as all zeros")
            var warnings = Python.import_module("warnings")
            _ = py_call_typed(warnings.warn, PythonObject("Found unknown categories in columns " + _list_text(bad)
                              + " during transform. These unknown categories will be " + where + "."), b.UserWarning)
    return _take2(ran, out, n, W, context)


def _onehot_inverse(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """OneHotEncoder.inverse_transform: block_argmax, the inverse grouping
    remap, cat_gather and the strict bad-row compaction."""
    var fields = py_fields(model)
    var widths = _onehot_widths(fields)
    var drops = _onehot_drops(fields)
    var n = Int(py=values[3])
    var cols = Int(py=values[4])
    var d = len(widths)
    var W = 0
    for j in range(d):  # driver-loop: over the per-column block widths
        W += widths[j]
    if cols != W:
        raise Error("mojolearn: X has " + String(cols) + " columns, expected " + String(W))
    var entries = _entries(fields["numeric_mode_"], context)
    var p = EncProg()
    # `_block_argmax`
    var xo = p.put_x(values)
    var so = p.put_floats(_prefix(widths), context)
    var wo = p.put_floats(_to_floats(widths), context)
    var dro = p.put_floats(_to_floats(drops), context)
    var codes = p.alloc(n * d)
    _stage(p.program, OP_BLOCK_ARGMAX, n * d, xo, n, W, d, so, wo, dro, 1, codes)
    var grouped = codes
    var groups = fields["_grouping"]
    if groups is not None:
        var table = _grouping_table(p, groups, True, context)
        codes = _remap(p, codes, n, d, table, NONE)
    var out = _gather_categories(p, codes, n, d, fields["categories_"], context)
    # an all-zero block is unknown (NaN) under 'ignore', and under
    # 'infrequent_if_exist' / 'warn' for a column with no infrequent
    # category; anywhere else it cannot be inverted
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    var infrequent = fields["_infrequent"]
    var strict = List[Float64]()
    for j in range(d):  # driver-loop: over the per-column strictness flags
        var hard = handle_text == "error" or (handle_text != "ignore" and infrequent is not None and infrequent[j] is not None)
        strict.append(1.0 if hard else 0.0)
    var sto = p.put_floats(strict, context)
    var staged = _bad_rows_stages(p, grouped, n, d, -1, sto)
    var ran = p.run(entries, context)
    var bad = _bad_rows(ran, staged, 10)
    if len(bad) > 0:
        raise Error("mojolearn: samples " + _list_text(bad) + " can not be inverted when drop=None and "
                    + "handle_unknown='error' because they contain all zeros")
    return _take2(ran, out, n, d, context)


# ---------------------------------------------------------------- TargetEncoder
def _target_scratch(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_target_scratch`: the FAST binding's x_prep_target_scratch.
    if String(mode) != "fast":
        return False
    var entry = py_get(entries, "x_prep_target_scratch")
    return entry is not None and _truth(py_call_typed(entry))


def _target_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """TargetEncoder.transform: lookup (device-only codes when the binding
    keeps them in scratch) and te_apply over the fitted encodings."""
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var mode = fields["numeric_mode_"]
    var entries = _entries(mode, context)
    var T = Int(py=fields["_T"])
    var cmax = Int(py=fields["_cmax"])
    var p = EncProg()
    var codes = _codes(p, values, fields["categories_"], context, _target_scratch(mode, entries))[0]
    var enc = p.put_array(fields["_enc"], context)
    var meta = p.put_array(fields["_meta"], context)
    var out = p.output(n * d * T, entries)
    _stage(p.program, OP_TE_APPLY, n * d * T, codes, n, d, T, NONE, enc, cmax, meta, 0, out)
    var ran = p.run(entries, context)
    return _take2(ran, out, n, d * T, context)


# ---------------------------------------------------------------- SplineTransformer
def _spline_fused(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_spline_fused`: the FAST Apple build with -D MOJOLEARN_SPLINE_FAST_FUSED
    # (it alone exports x_prep_spline_fused).
    var text = py_call(Python.import_module("builtins").str, mode)
    if String(py_call(py_call(text.strip).lower)) != "fast":
        return False
    return py_get(entries, "x_prep_spline_fused") is not None


def _col_stats_blocked(mut p: EncProg, xo: Int, n: Int, d: Int, out: Int) raises:
    # `_col_stats(..., var=False)`: the blocked order whenever
    # MOJOLEARN_XPREP_BLOCKED is on, else one thread per column.
    if String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) == "0":
        _stage(p.program, OP_COL_STATS, d, xo, n, d, out)
        return
    var nb = (n + ROW_BLOCK - 1) // ROW_BLOCK
    var part = p.work(5 * nb * d)
    _stage(p.program, OP_COLB_PART, nb * d, xo, n, d, part, nb)
    _stage(p.program, OP_COLB_FOLD, d, part, nb, d, out)


def _extrapolation(value: PythonObject) raises -> Int:
    # `SplineTransformer._EXTRAP`.
    if _is_str(value):
        var text = String(value)
        if text == "constant":
            return 0
        if text == "continue" or text == "error":
            return 1
        if text == "linear":
            return 2
        if text == "periodic":
            return 3
    raise Error("mojolearn: invalid extrapolation " + _repr(value))


def _spline_checks(fields: PythonObject) raises:
    """The parameter refusals `_fit` raised before converting X."""
    _ = _extrapolation(fields["extrapolation"])
    var missing = fields["handle_missing"]
    if not (_is_str(missing) and (String(missing) == "error" or String(missing) == "zeros")):
        raise Error("mojolearn: invalid handle_missing " + _repr(missing))
    var order = fields["order"]
    if not (_is_str(order) and (String(order) == "C" or String(order) == "F")):
        raise Error("mojolearn: invalid order " + _repr(order))
    if _truth(fields["sparse_output"]):
        raise Error("NotImplementedError: mojolearn: SplineTransformer sparse_output is not implemented "
                    + "(there is no sparse Array)")
    var k = Int(py=py_call_typed(Python.import_module("builtins").int, fields["degree"]))
    if k < 0 or k > 7:
        raise Error("mojolearn: SplineTransformer needs 0 <= degree <= 7")


def _no_nan(fields: PythonObject, ran: PythonObject, st: Int, d: Int, n: Int) raises:
    # `_no_nan`: handle_missing='error' refuses a column with a NaN.
    var missing = fields["handle_missing"]
    if not (_is_str(missing) and String(missing) == "error"):
        return
    for j in range(d):  # driver-loop: over the per-column non-NaN counts (feature metadata)
        if Int(_word(ran, st + j)) != n:
            raise Error("mojolearn: Input X contains NaN values and `SplineTransformer` is configured to "
                        + "error in this case (handle_missing='error'). To avoid this error, set "
                        + "handle_missing='zeros' to encode missing values as splines with value 0 or ensure "
                        + "no missing values in X.")


def _knot_rows(knots: PythonObject, context: PythonObject) raises -> PythonObject:
    # The user knots argument's rows, each a Python list of numbers.
    var b = Python.import_module("builtins")
    if py_type(knots) is context["array_class"] and Int(py=knots.ndim) == 2:
        var rows = Python.list()
        var r = Int(py=knots.shape[0])
        var c = Int(py=knots.shape[1])
        if r * c == 0:
            for _ in range(r):  # driver-loop: over the user knots rows
                _ = py_call(rows.append, Python.list())
            return rows
        var flat = _input(knots, 2, False, context)
        var source = _ptr(flat)
        for i in range(r):  # driver-loop: over the user knots rows
            var row = Python.list()
            for j in range(c):  # driver-loop: over one user knots row
                _ = py_call(row.append, PythonObject(Float64(source[i * c + j])))
            _ = py_call(rows.append, row)
        return rows
    var outer = _user_values(knots, context)
    var rows = Python.list()
    for i in range(len(outer)):  # driver-loop: over the user knots rows
        _ = py_call(rows.append, _user_values(outer[i], context))
    return rows


def _spline_fit(model: PythonObject, values: PythonObject, weight_source: PythonObject, fuse: Bool,
                context: PythonObject) raises -> PythonObject:
    """SplineTransformer `_fit`: the knots program (col_stats, the given,
    quantile or uniform base knots, spline_knots) and, with `fuse`, the
    spline_apply of X in the same program; returns the output or None."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    _spline_checks(fields)
    var extrap = _extrapolation(fields["extrapolation"])
    var k = Int(py=py_call_typed(b.int, fields["degree"]))
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required by SplineTransformer.")
    var w = PythonObject(None)
    var wl = 0
    var weighted = weight_source is not None
    if weighted:
        var checked = _check_weights(weight_source, n, "SplineTransformer", context)
        w = checked[0]
        wl = checked[1]
    var knots_arg = fields["knots"]
    var given = not _is_str(knots_arg)
    var knot_kind = String(knots_arg) if not given else String("")
    if not given and knot_kind != "uniform" and knot_kind != "quantile":
        raise Error("mojolearn: invalid knots " + _repr(knots_arg))
    var nk = 0
    var cols = List[Float64]()
    if given:
        var rows = _knot_rows(knots_arg, context)
        nk = len(rows)
        if nk < 2:
            raise Error("mojolearn: Number of knots, knots.shape[0], must be >= 2.")
        var table = List[List[Float32]]()
        for i in range(nk):  # driver-loop: over the user knots rows
            if len(rows[i]) != d:
                raise Error("mojolearn: knots.shape[1] == n_features is violated.")
            var row = List[Float32]()
            for c in range(d):  # driver-loop: over one user knots row
                row.append(_f32(rows[i][c]))
            table.append(row^)
        for c in range(d):  # driver-loop: transposes the user knots argument (feature count)
            for i in range(nk):  # driver-loop: over one knots column
                if i > 0 and not (table[i][c] > table[i - 1][c]):
                    raise Error("mojolearn: knots must be sorted without duplicates.")
        for c in range(d):  # driver-loop: packs the user knots argument column-major
            for i in range(nk):  # driver-loop: over one knots column
                cols.append(Float64(table[i][c]))
    else:
        nk = Int(py=py_call_typed(b.int, fields["n_knots"]))
        if nk < 2:
            raise Error("mojolearn: SplineTransformer needs n_knots >= 2")
    var periodic = extrap == 3
    if periodic and nk <= k:
        raise Error("mojolearn: Periodic splines require degree < n_knots. Got n_knots=" + String(nk) + " and "
                    + "degree=" + String(k) + ".")
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var p = EncProg()
    var xo = p.put_x(values)
    var sorts = not given and knot_kind == "quantile" and not weighted
    # MOJOLEARN_SPLINE_FAST_FUSED: no n*d block when nothing sorts, and the
    # count / min / max rows by the blocked units
    var fused = _spline_fused(mode, entries)
    var so = 0
    if sorts:
        so = p.work(n * d)
    elif not fused:
        so = p.alloc(n * d)
    var st = p.alloc(6 * d)
    var wd = nk + 2 * k
    var knots = p.alloc(d * wd)
    if fused:
        _col_stats_blocked(p, xo, n, d, st)
    else:
        _cs(p, mode, entries, xo, n, d, st, False)
    var uniform = 0
    var kst = st
    var base = 0
    if given:
        base = p.put_floats(cols, context)
    elif knot_kind == "quantile":
        base = p.alloc(d * nk)
        if not weighted:
            # `_grid(pr, [nk - 1], nk, 1)`
            var qf = p.alloc(nk)
            var levels = List[Float64]()
            levels.append(Float64(nk - 1))
            var lo = p.put_floats(levels, context)
            _stage(p.program, OP_GRID, nk, lo, nk, 1, qf)
            _stage(p.program, OP_SORT_COLS, d, xo, n, d, so, 0)
            _stage(p.program, OP_QUANTILE, d * nk, so, n, d, qf, nk, base, st)
        else:
            # `_weighted_groups` over the X input already in the arena, then
            # `_weighted_levels` at the levels c2_grid KIND 2 writes
            var gs = p.work(n * d)
            var ug = p.alloc(2 * n * d)
            var ucnt = p.alloc(d)
            var codes = p.alloc(n * d)
            _stage(p.program, OP_SORT_COLS, d, xo, n, d, gs, 1)
            _stage(p.program, OP_UNIQUE_COLS, d, gs, n, d, ug, ucnt)
            _stage(p.program, OP_LOOKUP, n * d, xo, n, d, ug, n, ucnt, codes)
            var wo = p.put(w, n)
            _stage(p.program, OP_KBINS_GW, d, codes, n, d, wo, ug + n * d)
            var nb = List[Int]()
            for _ in range(d):  # driver-loop: one level row per column
                nb.append(nk - 1)
            var lv = p.alloc(d * nk)
            var grid_nb = p.put_floats(_to_floats(nb), context)
            _stage(p.program, OP_GRID, d * nk, grid_nb, nk, 2, lv)
            var wq_nb = p.put_floats(_to_floats(nb), context)
            _stage(p.program, OP_KBINS_WQ, d, ug, n, d, ucnt, wq_nb, nk - 1, lv, 0, base)
    else:
        base = p.alloc(d * nk)
        uniform = 1
        if weighted and wl < n:
            # the positive-weight rows on the device
            kst = p.alloc(6 * d)
            var wo = p.put(w, n)
            var ro = p.alloc(n)
            _ = _sel(p, wo, n, 1, 1, ro)
            var xz = p.alloc(wl * d)
            _stage(p.program, OP_RGATHER, wl * d, xo, d, ro, xz)
            _cs(p, mode, entries, xz, wl, d, kst, False)
    _stage(p.program, OP_SPLINE_KNOTS, d, base, nk, d, k, knots, uniform, kst, 1 if periodic else 0)
    var nspl = nk - 1 if periodic else nk + k - 1
    var bias = _truth(fields["include_bias"])
    var W = d * (nspl if bias else nspl - 1)
    var out = 0
    if fuse:
        out = p.output(n * W, entries)
        _stage(p.program, OP_SPLINE_APPLY, n * d, xo, n, d, knots, nk, k, extrap, W, 1 if bias else 0, out)
    var ran = p.run(entries, context)
    _no_nan(fields, ran, st, d, n)
    var output = context["output_context"]
    py_setitem(fields, "_knots", _take1(ran, knots, d * wd, context))
    var bsplines = Python.list()
    var lows = Python.list()
    var highs = Python.list()
    for c in range(d):  # driver-loop: per-column fitted spline knots (feature count)
        _ = py_call(bsplines.append, _take1(ran, knots + c * wd, wd, context))
        _ = py_call(lows.append, PythonObject(Float64(_word(ran, knots + c * wd + k))))
        _ = py_call(highs.append, PythonObject(Float64(_word(ran, knots + c * wd + k + nk - 1))))
    py_setitem(fields, "bsplines_", bsplines)
    py_setitem(fields, "_lo", lows)
    py_setitem(fields, "_hi", highs)
    py_setitem(fields, "_nk", nk)
    py_setitem(fields, "_k", k)
    py_setitem(fields, "n_features_out_", W)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = w^
    if fuse:
        return _take2(ran, out, n, W, context)
    return PythonObject(None)


def _spline_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """SplineTransformer.transform: spline_apply over the fitted knots (with
    the column statistics when a refusal reads them) and, for order 'F',
    p2m_transpose into the output."""
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var mode = fields["numeric_mode_"]
    var entries = _entries(mode, context)
    var W = Int(py=fields["n_features_out_"])
    var extrap = _extrapolation(fields["extrapolation"])
    var fortran = _is_str(fields["order"]) and String(fields["order"]) == "F"
    var missing_error = _is_str(fields["handle_missing"]) and String(fields["handle_missing"]) == "error"
    var extrap_error = _is_str(fields["extrapolation"]) and String(fields["extrapolation"]) == "error"
    var p = EncProg()
    var xo = p.put_x(values)
    var ko = p.put_array(fields["_knots"], context)
    var out = p.alloc(n * W) if fortran else p.output(n * W, entries)
    var st = p.alloc(6 * d)
    var check = extrap_error or missing_error
    if check:
        if _spline_fused(mode, entries):
            _col_stats_blocked(p, xo, n, d, st)
        else:
            _cs(p, mode, entries, xo, n, d, st, False)
    _stage(p.program, OP_SPLINE_APPLY, n * d, xo, n, d, ko, Int(py=fields["_nk"]), Int(py=fields["_k"]), extrap, W,
           1 if _truth(fields["include_bias"]) else 0, out)
    var fo = 0
    var transposed = fortran and n * W > 0
    if transposed:
        fo = p.output(n * W, entries)
        _stage(p.program, OP_TRANSPOSE, n * W, out, n, W, fo)
    var ran = p.run(entries, context)
    if check:
        _no_nan(fields, ran, st, d, n)
    if extrap_error:
        var lows = fields["_lo"]
        var highs = fields["_hi"]
        for j in range(d):  # driver-loop: over the per-column knot limits (feature metadata)
            if Float64(_word(ran, st + 3 * d + j)) < Float64(py=lows[j]) or Float64(_word(ran, st + 4 * d + j)) > Float64(py=highs[j]):
                raise Error("mojolearn: X contains values beyond the limits of the knots")
    if transposed:
        if fo == OUT_MARK:
            return _take_f(ran[1][0], 0, n, W, context)
        return _take_f(ran[0][0], fo, n, W, context)
    if fortran:
        # an empty block
        return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(n, W), "<f4", "F", context["output_context"])
    return _take2(ran, out, n, W, context)


# ---------------------------------------------------------------- the door
def _kind(model: PythonObject, context: PythonObject) raises -> String:
    # The admitted class this model is (or subclasses), by its MRO.
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var kinds = context["prep_encode_kinds"]
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native encode owner does not serve this class")


def _fit_weights(model: PythonObject, args: PythonObject, operation: String) raises -> PythonObject:
    """The fit's sample_weight: fit(X, y, sample_weight); fit_transform(X,
    y, fit_params) passes the keywords the Python fit would have taken."""
    if operation == "fit":
        return args[2] if len(args) > 2 else PythonObject(None)
    var params = args[2]
    var b = Python.import_module("builtins")
    var keys = py_call(b.list, params)
    var weights = PythonObject(None)
    for i in range(len(keys)):  # driver-loop: over the caller's fit keywords
        var key = keys[i]
        if _is_str(key) and String(key) == "sample_weight":
            weights = params[key]
        else:
            raise Error("TypeError: " + _name(model) + ".fit() got an unexpected keyword argument " + _repr(key))
    return weights


def _no_fit_params(model: PythonObject, args: PythonObject) raises:
    # fit(X, y=None) takes no keywords: fit_transform's refuse them as fit would.
    var keys = py_call(Python.import_module("builtins").list, args[2])
    if len(keys) > 0:
        raise Error("TypeError: " + _name(model) + ".fit() got an unexpected keyword argument " + _repr(keys[0]))


def _precheck(model: PythonObject, kind: String, context: PythonObject) raises:
    """The parameter refusals the Python fit raised before converting X."""
    var fields = py_fields(model)
    if kind == "simple_imputer":
        var b = Python.import_module("builtins")
        var strategy = fields["strategy"]
        var name = String(strategy) if _is_str(strategy) else String("")
        if not Bool(py=py_call(b.callable, strategy)) and name != "mean" and name != "median" and name != "most_frequent" and name != "constant":
            raise Error("mojolearn: SimpleImputer strategy " + _repr(strategy) + " is not valid")
        if name == "constant" and fields["fill_value"] is not None and not _real_like(fields["fill_value"], context):
            raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    elif kind == "ordinal":
        _ = _check_infrequent_params(fields, context)
        var handle = fields["handle_unknown"]
        if not (_is_str(handle) and (String(handle) == "error" or String(handle) == "use_encoded_value")):
            raise Error("mojolearn: invalid handle_unknown " + _repr(handle))
        if _is_str(handle) and String(handle) == "use_encoded_value" and not _real_like(fields["unknown_value"], context):
            raise Error("TypeError: mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
        if not _real_like(fields["encoded_missing_value"], context):
            raise Error("TypeError: mojolearn: encoded_missing_value must be a number (or NaN)")
    elif kind == "onehot":
        _ = _check_infrequent_params(fields, context)
        var handle = fields["handle_unknown"]
        var text = String(handle) if _is_str(handle) else String("")
        if text != "error" and text != "ignore" and text != "infrequent_if_exist" and text != "warn":
            raise Error("mojolearn: OneHotEncoder handle_unknown=" + _repr(handle) + " is not valid")
        var drop = fields["drop"]
        if _is_str(drop) and String(drop) != "first" and String(drop) != "if_binary":
            raise Error("mojolearn: OneHotEncoder drop must be None, 'first', 'if_binary' or one category "
                        + "per feature")
    elif kind == "spline":
        _spline_checks(fields)
    elif kind == "kbins":
        var encode = fields["encode"]
        var text = String(encode) if _is_str(encode) else String("")
        if text != "onehot" and text != "onehot-dense" and text != "ordinal":
            raise Error("mojolearn: invalid encode " + _repr(encode))
        _ = _kbins_strategy(fields)


def _fit(model: PythonObject, kind: String, values: PythonObject, weights: PythonObject, capture: Bool,
         context: PythonObject) raises -> PythonObject:
    # The fit; an encoder's C08 fit (`capture`) returns its (n, d) inverse.
    if kind == "simple_imputer":
        _imputer_fit(model, values, context)
    elif kind == "kbins":
        _kbins_fit(model, values, weights, context)
    elif kind == "ordinal":
        return _ordinal_fit(model, values, capture, context)
    elif kind == "onehot":
        return _onehot_fit(model, values, capture, context)
    elif kind == "spline":
        _ = _spline_fit(model, values, weights, False, context)
    else:
        raise Error("TypeError: the native encode owner does not fit this class")
    return PythonObject(None)


def _transform(model: PythonObject, kind: String, values: PythonObject, inverse: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "simple_imputer":
        return _imputer_transform(model, values, context)
    if kind == "kbins":
        return _kbins_transform(model, values, context)
    if kind == "ordinal":
        _check_width(model, Int(py=values[4]))
        return _ordinal_transform(model, values, inverse, context)
    if kind == "onehot":
        _check_width(model, Int(py=values[4]))
        return _onehot_transform(model, values, inverse, context)
    if kind == "target":
        return _target_transform(model, values, context)
    if kind == "spline":
        return _spline_transform(model, values, context)
    raise Error("TypeError: the native encode owner does not transform this class")


def prep_encode_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var kind = _kind(model, context)
        var result = PythonObject(None)
        if operation == "fit" or operation == "fit_transform":
            var weights = PythonObject(None)
            if kind == "kbins":
                weights = _fit_weights(model, args, operation)
            elif kind == "spline":
                # fit(X, y, sample_weight) and fit_transform(X, y, sample_weight)
                weights = args[2]
            elif operation == "fit_transform":
                _no_fit_params(model, args)
            _precheck(model, kind, context)
            var mode = _default_mode(context)
            var entries = _entries(mode, context)
            var values = xprep_values(args[0], entries, context)
            # The C08 fit_transform (OneHotEncoder bit 2, OrdinalEncoder bit
            # 4, categories='auto'): the fit's inverse is the transform's codes.
            var fields = py_fields(model)
            var auto = py_contains(fields, "categories") and _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
            var capture = operation == "fit_transform" and auto and (
                (kind == "onehot" and _c08_route(mode, entries, 2)) or (kind == "ordinal" and _c08_route(mode, entries, 4)))
            var fused = False
            if kind == "spline" and operation == "fit_transform":
                # with x_prep_spline_fused one program fits and applies; 'error'
                # extrapolation and order 'F' keep fit + transform
                var extrap = fields["extrapolation"]
                var order = fields["order"]
                fused = not (_is_str(extrap) and String(extrap) == "error") and _is_str(order) and String(order) == "C" and _spline_fused(mode, entries)
            var inverse = PythonObject(None)
            if fused:
                result = _spline_fit(model, values, weights, True, context)
            else:
                inverse = _fit(model, kind, values, weights, capture, context)
                result = model
            if operation == "fit_transform" and not fused:
                # The X converted for the fit is the transform's input.
                result = _transform(model, kind, values, inverse, context)
            _ = values^
        elif operation == "transform" or operation == "inverse_transform":
            _fitted(model)
            var values = xprep_values(args[0], _entries(py_fields(model)["numeric_mode_"], context), context)
            if operation == "transform":
                result = _transform(model, kind, values, PythonObject(None), context)
            elif kind == "kbins":
                result = _kbins_inverse(model, values, context)
            elif kind == "ordinal":
                result = _ordinal_inverse(model, values, context)
            elif kind == "onehot":
                result = _onehot_inverse(model, values, context)
            else:
                raise Error("TypeError: the native encode owner does not invert this class")
            _ = values^
        else:
            raise Error("unknown native encode prep operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
