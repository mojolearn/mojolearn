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
from bindings.prep_stateless_native import PrepLayout, _place_scratch, _fam, _int_like, _repr, _name

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
comptime OP_ONEHOT = 9
comptime OP_MARK_MISSING = 23
comptime OP_FILL = 24
comptime OP_KBINS_EDGES = 25
comptime OP_KBINS_CODES = 26
comptime OP_NAN_MASK = 60
comptime OP_BLOCK_ARGMAX = 76
comptime OP_ORD_INVERSE = 77
comptime OP_KBINS_INVERSE = 80
comptime OP_KBINS_GW = 96
comptime OP_KBINS_WQ = 97
comptime OP_KBINS_WKM = 98
comptime OP_COLB_PART = 124
comptime OP_COLB_FOLD = 125
comptime OP_COLB_SS = 126
comptime OP_COLB_VAR = 127
comptime OP_HCAT = 140
comptime OP_RGATHER = 204
comptime OP_SMROWS = 205
comptime OP_SEL_COUNT = 206
comptime OP_SEL_SCAN = 207
comptime OP_SEL_WRITE = 208
comptime OP_ROWFLAG = 210
comptime OP_WBLK = 230
comptime OP_WSCAN = 231
comptime OP_WDRAW = 232
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
comptime INT32_MAX = 2147483647


# ---------------------------------------------------------------- programs
struct EncProg(Movable):
    """One x_prep program: its arena layout (`PrepLayout`: host words, then
    device scratch), the inputs copied into it, its stages and at most one
    device output region (`_Prog.output`)."""
    var layout: PrepLayout
    var program: List[Int32]
    var inputs: PythonObject
    var out_words: Int

    def __init__(out self) raises:
        self.layout = PrepLayout(0)
        self.program = List[Int32]()
        self.inputs = Python.list()
        self.out_words = 0

    def alloc(mut self, words: Int) -> Int:
        return self.layout.alloc(words)

    def work(mut self, words: Int) -> Int:
        return self.layout.work_words(words)

    def scratch(mut self, words: Int) -> Int:
        return self.layout.scratch_words(words)

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
        var host = self.layout.host
        var scratch = self.layout.scratch
        if host + scratch + self.out_words > INT32_MAX:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        var program = self.program.copy()
        for i in range(len(program)):  # driver-loop: over the program's stage words
            if Int(program[i]) == OUT_MARK:
                program[i] = Int32(host + scratch)
        _place_scratch(program, host)
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


def _bad_rows_stages(mut p: EncProg, codes: Int, n: Int, d: Int, bad: Int) -> Tuple[Int, Int]:
    # `_bad_rows_stages`: (ROWS, TOT) of the rows holding a `bad` code.
    var flags = p.alloc(n)
    var rows = p.alloc(n)
    _stage(p.program, OP_ROWFLAG, n, codes, n, d, bad, NONE, flags)
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
            fills.append(0.0 if fill_value is None else _float(fill_value))
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
    var staged = _bad_rows_stages(p, codes, n, d, bad_code)
    var ran = p.run(entries, context)
    var bad = _bad_rows(ran, staged, 10)
    if len(bad) > 0:
        raise Error("mojolearn: samples " + _list_text(bad) + " " + why)
    return _take2(ran, out, n, d, context)


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
    elif kind == "kbins":
        var encode = fields["encode"]
        var text = String(encode) if _is_str(encode) else String("")
        if text != "onehot" and text != "onehot-dense" and text != "ordinal":
            raise Error("mojolearn: invalid encode " + _repr(encode))
        _ = _kbins_strategy(fields)


def _fit(model: PythonObject, kind: String, values: PythonObject, weights: PythonObject, context: PythonObject) raises:
    if kind == "simple_imputer":
        _imputer_fit(model, values, context)
    elif kind == "kbins":
        _kbins_fit(model, values, weights, context)
    else:
        raise Error("TypeError: the native encode owner does not fit this class")


def _transform(model: PythonObject, kind: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "simple_imputer":
        return _imputer_transform(model, values, context)
    if kind == "kbins":
        return _kbins_transform(model, values, context)
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
            elif operation == "fit_transform":
                _no_fit_params(model, args)
            _precheck(model, kind, context)
            var values = xprep_values(args[0], _entries(_default_mode(context), context), context)
            _fit(model, kind, values, weights, context)
            result = model
            if operation == "fit_transform":
                # The X converted for the fit is the transform's input.
                result = _transform(model, kind, values, context)
            _ = values^
        elif operation == "transform" or operation == "inverse_transform":
            _fitted(model)
            var values = xprep_values(args[0], _entries(py_fields(model)["numeric_mode_"], context), context)
            if operation == "transform":
                result = _transform(model, kind, values, context)
            elif kind == "kbins":
                result = _kbins_inverse(model, values, context)
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
