# SPDX-License-Identifier: Apache-2.0
# The native owner of the stateless x_prep transformers: Binarizer, Normalizer
# and PolynomialFeatures. Each public fit, transform and fit_transform is one
# call into prep_stateless_c, which checks the parameters, records the fitted
# metadata, builds the x_prep program (the same stages, operands and order the
# Python `_Prog` built) and runs it through the shared transport
# (bindings/x_prep_native_owner.mojo `xprep_run`). fit_transform converts X once
# and runs the one transform program.
from std.memory import memcpy, bitcast
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_wrap_array
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.scaler_native import _stage, _ptr, _input
from bindings.scaler_lifecycle_native import _entries
from bindings.x_prep_native_owner import xprep_storage, xprep_take, xprep_values, xprep_run, xprep_r3
from bindings.py_typed_call import py_call_typed
from bindings.prep_support_native import prep_support

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_BINARIZE = 12
comptime OP_NORMALIZE = 47
comptime OP_POLY = 48
comptime OP_TRANSPOSE = 209
comptime OP_COL_STATS = 1
comptime OP_GATHER_COLS = 61
comptime OP_VAR_PTP = 62
comptime OP_COLB_PART = 124
comptime OP_COLB_FOLD = 125
comptime OP_COLB_SS = 126
comptime OP_COLB_VAR = 127
comptime OP_KEY64 = 253
comptime OP_GT = 255
comptime OP_SORT_COLS = 0
comptime OP_QUANTILE = 2
comptime OP_QT_APPLY = 43
comptime OP_QT_INVERSE = 74
comptime OP_RGATHER = 204
comptime OP_PERM_ROWS = 233
comptime OP_GRID = 263
comptime OP_PT_FIT = 44
comptime OP_PT_APPLY = 45
comptime OP_STD_PARAMS = 46
comptime OP_PT_INVERSE = 75
comptime OP_PT_INIT = 104
comptime OP_PT_MAP = 105
comptime OP_PT_FOLD = 106
comptime OP_PT_LOG = 109
comptime OP_PT_SPTS = 110
comptime OP_PT_SMAP = 111
comptime OP_PT_SFOLD = 112
comptime OP_PT_SRES = 113
comptime OP_PTB_PART1 = 172
comptime OP_PTB_MEAN = 173
comptime OP_PTB_PART2 = 174
comptime OP_PTB_FIN = 175
comptime OP_PTB_STEP = 176
# PowerTransformer's fixed golden-section evaluations and per-column state words.
comptime PT_EVALS = 50
comptime PT_STATE = 10
# x_prep_idn_fam bit 8: IDN_PT_BLOCKED.
comptime IDN_PT_BLOCKED = 8
# Row blocks of the blocked column statistics (_expansion_prep._XB).
comptime ROW_BLOCK = 2048
# x_prep_idn_fam bit 1: IDN_STATS_BLOCKED.
comptime IDN_STATS_BLOCKED = 1
# `_Prog.output`: below this many words the output is plain arena words.
comptime OUT_MIN_WORDS = 134217728


def _kind(model: PythonObject, context: PythonObject) raises -> String:
    # The admitted class this model is (or subclasses), by its MRO.
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native preprocessing refuses Python metaclass hooks")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var kinds = context["prep_stateless_kinds"]
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native prep owner does not serve this class")


def _name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _repr(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").repr, value))


def _int_like(value: PythonObject, context: PythonObject) raises -> Bool:
    # numbers.Integral without its Python __instancecheck__: Python ints
    # (bool included) and NumPy integer scalars when NumPy is connected.
    var b = Python.import_module("builtins")
    if Bool(py=py_call(b.isinstance, value, b.int)):
        return True
    var integer = py_get(context, "integer")
    return integer is not None and Bool(py=py_call(b.isinstance, value, integer))


def _norm_code(fields: PythonObject) raises -> Int:
    var b = Python.import_module("builtins")
    var norm = fields["norm"]
    if py_type(norm) is b.str:
        var text = String(norm)
        if text == "l1":
            return 0
        if text == "l2":
            return 1
        if text == "max":
            return 2
    raise Error("mojolearn: invalid norm " + _repr(norm))


def _degrees(fields: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    var b = Python.import_module("builtins")
    var degree = fields["degree"]
    var lo = 0
    var hi = 0
    if _int_like(degree, context):
        hi = Int(py=py_call(b.int, degree))
    else:
        var pair = py_call(b.tuple, degree)
        if len(pair) != 2:
            raise Error("mojolearn: invalid degree " + _repr(degree))
        lo = Int(py=py_call(b.int, pair[0]))
        hi = Int(py=py_call(b.int, pair[1]))
    if hi < 0 or lo < 0 or lo > hi:
        raise Error("mojolearn: invalid degree " + _repr(degree))
    return (lo, hi)


def _emit(mut index: List[Int], mut start: List[Int], terms: List[Int]):
    for i in range(len(terms)):  # driver-loop: over one polynomial term's feature indices
        index.append(terms[i])
    start.append(len(index))


def _combinations(d: Int, k: Int, replacement: Bool, mut index: List[Int], mut start: List[Int]):
    # itertools.combinations(_with_replacement)(range(d), k), in its order.
    if k == 0:
        _emit(index, start, List[Int]())
        return
    if not replacement and k > d:
        return
    var c = List[Int](length=k, fill=0)
    if not replacement:
        for i in range(k):  # driver-loop: over one polynomial term's positions
            c[i] = i
    while True:  # driver-loop: over polynomial terms of feature indices
        _emit(index, start, c)
        var i = k - 1
        while i >= 0 and c[i] >= (d - 1 if replacement else d - k + i):  # driver-loop: over one term's positions
            i -= 1
        if i < 0:
            return
        c[i] += 1
        for j in range(i + 1, k):  # driver-loop: over one term's positions
            c[j] = c[i] if replacement else c[j - 1] + 1


def _words(values: List[Int], context: PythonObject) raises -> PythonObject:
    # Index words as the float32 values `_Prog.put_list` packed (exact).
    var count = max(len(values), 1)
    var out = classical_typed(Python.tuple(count), "<f4", context["output_context"])
    var dest = _ptr(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over polynomial term table words
        dest[i] = Float32(values[i])
        _ = out  # keep the Python object alive through the native call above
    return out


def _fit(model: PythonObject, kind: String, values: PythonObject, context: PythonObject) raises:
    var fields = py_fields(model)
    var d = Int(py=values[4])
    if kind == "normalizer":
        _ = _norm_code(fields)
    elif kind == "polynomial":
        var order = fields["order"]
        if py_type(order) is not Python.import_module("builtins").str or (String(order) != "C" and String(order) != "F"):
            raise Error("mojolearn: PolynomialFeatures order must be 'C' or 'F'")
        var degrees = _degrees(fields, context)
        var replacement = not Bool(py=fields["interaction_only"])
        var index = List[Int]()
        var start = List[Int]()
        start.append(0)
        if Bool(py=fields["include_bias"]):
            _combinations(d, 0, replacement, index, start)
        for k in range(max(1, degrees[0]), degrees[1] + 1):  # driver-loop: over polynomial degrees
            _combinations(d, k, replacement, index, start)
        var nout = len(start) - 1
        var powers = classical_typed(Python.tuple(max(nout, 1), d), "<i8", context["output_context"])
        var table = I64P(unsafe_from_address=Int(py=classical_address(powers)))
        _ = powers  # keep the Python object alive through the native call above
        for t in range(nout):  # driver-loop: over polynomial terms
            for i in range(start[t], start[t + 1]):  # driver-loop: over one term's feature indices
                table[t * d + index[i]] += 1
                _ = powers  # keep the Python object alive through the native call above
        py_setitem(fields, "_poly_index", _words(index, context))
        _ = index  # keep the Python object alive through the native call above
        py_setitem(fields, "_poly_start", _words(start, context))
        _ = start  # keep the Python object alive through the native call above
        py_setitem(fields, "n_output_features_", nout)
        py_setitem(fields, "powers_", powers)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])


def _take_f(storage: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # The (rows, cols) column-major block p2m_transpose wrote, as an F-order Array.
    var b = Python.import_module("builtins")
    var part = py_call(b.memoryview, storage)[py_call(b.slice, offset, offset + rows * cols)]
    var store = py_call(context["array_type"], "f")
    _ = py_call(store.frombytes, py_call(part.cast, "B"))
    return classical_wrap_array(store, Python.tuple(rows, cols), "<f4", "F", context["output_context"])


def _transform(model: PythonObject, kind: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not Python.import_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + String(py_call(Python.import_module("builtins").str, fitted)))
    var entries = _entries(fields["numeric_mode_"], context)
    var inputs = Python.list(Python.tuple(0, values))
    var program = List[Int32]()
    if kind == "binarizer":
        # X, the threshold word, then the output as arena words.
        var threshold = classical_typed(Python.tuple(1), "<f4", context["output_context"])
        _ptr(threshold)[0] = Float32(Float64(py=py_call(Python.import_module("builtins").float, fields["threshold"])))
        _ = threshold  # keep the Python object alive through the native call above
        _ = py_call(inputs.append, Python.tuple(n * d, Python.tuple(0, threshold, PythonObject(None), 1, 1)))
        var out = n * d + 1
        var host = out + n * d
        if host > 2147483647:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        _stage(program, OP_BINARIZE, n * d, 0, n * d, n * d, out)
        var arena = xprep_storage(host, context)
        xprep_run(arena, host, 0, PythonObject(None), 0, program^, inputs, entries, context)
        var result = xprep_take(arena[0], out, n, d, context["output_context"])
        _ = threshold^
        _ = arena^
        return result
    if kind == "normalizer":
        var code = _norm_code(fields)
        var words = n * d
        var dev_out = words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
        var host = words + (0 if dev_out else words)
        if host + (words if dev_out else 0) > 2147483647:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        var arena = xprep_storage(host, context)
        var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
        _stage(program, OP_NORMALIZE, n, 0, n, d, code, host if dev_out else words)
        xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
        var result = xprep_take(output_store[0] if dev_out else arena[0], 0 if dev_out else words, n, d, context["output_context"])
        _ = arena^
        return result
    # PolynomialFeatures: X, the term indices, the term starts, the products.
    var index = fields["_poly_index"]
    var start = fields["_poly_start"]
    var nout = Int(py=fields["n_output_features_"])
    if Int(py=start.size) != nout + 1:
        raise Error("RuntimeError: PolynomialFeatures term table does not match n_output_features_")
    var io = n * d
    var so = io + Int(py=index.size)
    var out = so + Int(py=start.size)
    var words = n * nout
    var fortran = String(fields["order"]) == "F"
    var transpose = fortran and words > 0
    var dev_out = transpose and words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
    var host = out + words + (words if transpose and not dev_out else 0)
    if host + (words if dev_out else 0) > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _ = py_call(inputs.append, Python.tuple(io, Python.tuple(0, index, PythonObject(None), Int(py=index.size), 1)))
    _ = py_call(inputs.append, Python.tuple(so, Python.tuple(0, start, PythonObject(None), Int(py=start.size), 1)))
    _stage(program, OP_POLY, words, 0, n, d, io, so, nout, out)
    var fo = out + words
    if transpose:
        _stage(program, OP_TRANSPOSE, words, out, n, nout, host if dev_out else fo)
    var arena = xprep_storage(host, context)
    var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
    xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
    var result = PythonObject(None)
    if transpose:
        result = _take_f(output_store[0] if dev_out else arena[0], 0 if dev_out else fo, n, nout, context)
    elif fortran:
        # an empty block
        result = classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(n, nout), "<f4", "F", context["output_context"])
    elif words == 0:
        result = classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(n, nout), "<f4", "C", context["output_context"])
    else:
        result = xprep_take(arena[0], out, n, nout, context["output_context"])
    _ = arena^
    return result


struct PrepLayout(Movable):
    """A program's arena words and device scratch words, in allocation order
    (`_Prog.alloc` / `_Prog.work`): scratch offsets are taken past the host
    words when the program runs, so they are resolved with `scratch_at`."""
    var host: Int
    var scratch: Int
    var work: Bool

    def __init__(out self, host: Int):
        self.host = host
        self.scratch = 0
        self.work = xprep_r3("work")

    def alloc(mut self, words: Int) -> Int:
        var offset = self.host
        self.host += max(words, 0)
        return offset

    def scratch_words(mut self, words: Int) -> Int:
        # `_Prog.scratch`: device-only words, whatever the `work` switch.
        var offset = self.scratch
        self.scratch += max(words, 0)
        return -(offset + 2)

    def work_words(mut self, words: Int) -> Int:
        # A stage word -(k + 2) marks scratch word k (-1 is the stages' "none").
        if not self.work:
            return self.alloc(words)
        var offset = self.scratch
        self.scratch += max(words, 0)
        return -(offset + 2)


def _col_stats(mut program: List[Int32], mut layout: PrepLayout, mode: PythonObject, entries: PythonObject,
               xo: Int, n: Int, d: Int, out_buf: Int, variance: Bool = True) raises:
    """`_expansion_prep._cs`: the blocked order under IDN_STATS_BLOCKED on
    the IDENTICAL tier, else one thread per column. `variance` False skips
    the blocked variance pass (its callers read the other rows)."""
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"
    var fam = 0
    if String(mode) == "identical":
        var entry = py_get(entries, "x_prep_idn_fam")
        if entry is not None:
            fam = Int(py=py_call_typed(entry))
    if blocked and (fam & IDN_STATS_BLOCKED) != 0:
        var nb = (n + ROW_BLOCK - 1) // ROW_BLOCK
        var part = layout.work_words(5 * nb * d)
        _stage(program, OP_COLB_PART, nb * d, xo, n, d, part, nb)
        _stage(program, OP_COLB_FOLD, d, part, nb, d, out_buf)
        if variance:
            var ss = layout.work_words(nb * d)
            _stage(program, OP_COLB_SS, nb * d, xo, n, d, out_buf, ss, nb)
            _stage(program, OP_COLB_VAR, d, ss, nb, d, out_buf)
    else:
        _stage(program, OP_COL_STATS, d, xo, n, d, out_buf)


def _place_scratch(mut program: List[Int32], host: Int):
    # A stage word -(k + 2) is scratch word k, at host + k.
    for i in range(len(program) // 16):  # driver-loop: over the program's stages
        for k in range(2, 16):  # small-loop(params: the fixed fourteen-word stage ABI)
            var word = Int(program[i * 16 + k])
            if word < -1:
                program[i * 16 + k] = Int32(host - word - 2)


def _f64_input(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # A Python float as its binary64 word, two 4-byte words (`_f64_words`).
    var words = classical_typed(Python.tuple(2), "<f4", context["output_context"])
    var bits = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(words)))
    _ = words  # keep the Python object alive through the native call above
    bits[0] = bitcast[DType.uint64](Float64(py=py_call(Python.import_module("builtins").float, value)))
    _ = words  # keep the Python object alive through the native call above
    return words


def _variance_fit(model: PythonObject, values: PythonObject, context: PythonObject) raises:
    """VarianceThreshold.fit: column statistics, var_ptp, then variance >
    threshold compared in binary64 on the device (c2_key64, c2_gt): the
    stages, operands and order of the Python program."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode, context)
    var threshold = fields["threshold"]
    var layout = PrepLayout(n * d)
    var st = layout.alloc(6 * d)
    var variance = layout.alloc(d)
    var program = List[Int32]()
    _col_stats(program, layout, mode, entries, 0, n, d, st)
    _stage(program, OP_VAR_PTP, d, st, d, variance, 1 if Bool(py=threshold == PythonObject(0)) else 0)
    var key = layout.work_words(2 * d)
    var mask = layout.alloc(d)
    var limit = layout.alloc(2)
    _stage(program, OP_KEY64, d, variance, 0, key)
    _stage(program, OP_GT, d, key, limit, mask)
    var words = _f64_input(threshold, context)
    var inputs = Python.list(Python.tuple(0, values), Python.tuple(limit, Python.tuple(0, words, PythonObject(None), 2, 1)))
    var arena = _run_layout(program^, layout, inputs, entries, context)
    py_setitem(fields, "variances_", xprep_take(arena[0], variance, d, 0, context["output_context"]))
    var base = F32P(unsafe_from_address=Int(py=arena[1]))
    var support = Python.list()
    var any_kept = False
    for j in range(d):  # driver-loop: over the fitted support mask (feature metadata)
        var kept = base[mask + j] != 0
        any_kept = any_kept or kept
        _ = py_call(support.append, PythonObject(kept))
    py_setitem(fields, "_mask", support)
    _ = words^
    _ = arena^
    if not any_kept:
        # scikit-learn's wording, with its one-sample note
        var message = "No feature in X meets the variance threshold " + String(py_call(b.format, threshold, ".5f"))
        if n == 1:
            message += " (X contains only one sample)"
        raise Error(message)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)


def _select_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """_SelectorMixin.transform: the fitted support's columns of X, by one
    gather_cols stage."""
    var fields = py_fields(model)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not Python.import_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + String(py_call(Python.import_module("builtins").str, fitted)))
    var mask = py_call(Python.import_module("builtins").list, fields["_mask"])
    var keep = List[Int]()
    for j in range(len(mask)):  # driver-loop: over the fitted support mask (feature metadata)
        if Bool(py=mask[j]):
            keep.append(j)
    if len(keep) == 0:
        raise Error("mojolearn: no features were selected")
    var k = len(keep)
    var entries = _entries(fields["numeric_mode_"], context)
    var columns = _words(keep, context)
    _ = keep  # keep the Python object alive through the native call above
    var ko = n * d
    var out = ko + k
    var host = out + n * k
    if host > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var program = List[Int32]()
    _stage(program, OP_GATHER_COLS, n * k, 0, n, d, ko, k, out)
    var inputs = Python.list(Python.tuple(0, values), Python.tuple(ko, Python.tuple(0, columns, PythonObject(None), k, 1)))
    _ = keep  # keep the Python object alive through the native call above
    var arena = xprep_storage(host, context)
    xprep_run(arena, host, 0, PythonObject(None), 0, program^, inputs, entries, context)
    var result = xprep_take(arena[0], out, n, k, context["output_context"])
    _ = columns^
    _ = keep  # keep the Python object alive through the native call above
    _ = arena^
    return result


def _run_layout(var program: List[Int32], layout: PrepLayout, inputs: PythonObject, entries: PythonObject,
                context: PythonObject) raises -> PythonObject:
    # One run of a program laid out by `PrepLayout`; returns the arena.
    if layout.host + layout.scratch > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _place_scratch(program, layout.host)
    var device_scratch = layout.scratch > 0 and (py_get(entries, "x_prep_run_out") is not None or py_get(entries, "x_prep_run_scratch") is not None)
    var arena = xprep_storage(layout.host + (0 if device_scratch else layout.scratch), context)
    xprep_run(arena, layout.host, layout.scratch, PythonObject(None), 0, program^, inputs, entries, context)
    return arena


def _vector(value: Float64, context: PythonObject) raises -> PythonObject:
    var out = classical_typed(Python.tuple(1), "<f4", context["output_context"])
    _ptr(out)[0] = Float32(value)
    _ = out  # keep the Python object alive through the native call above
    return out


def _quantile_fit(model: PythonObject, values: PythonObject, context: PythonObject) raises:
    """QuantileTransformer.fit: the optional keyed row subsample (f2_perm_rows,
    p2m_rgather), the reference grid (c2_grid), sort_cols, column statistics,
    quantile and the p2m_transpose of quantiles_: the stages and order of the
    Python program."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var distribution = fields["output_distribution"]
    if py_type(distribution) is not b.str or (String(distribution) != "uniform" and String(distribution) != "normal"):
        raise Error("mojolearn: invalid output_distribution " + _repr(distribution))
    var n_all = Int(py=values[3])
    var d = Int(py=values[4])
    var n = n_all
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode, context)
    var subsample = fields["subsample"]
    var permute = subsample is not None and n_all > Int(py=py_call(b.int, subsample))
    var seed = PythonObject(None)
    if permute:
        var state = fields["random_state"]
        seed = PythonObject(0) if state is None else py_call(b.int, state)
        n = Int(py=py_call(b.int, subsample))
    var nq = max(1, min(Int(py=py_call(b.int, fields["n_quantiles"])), n))
    var layout = PrepLayout(n_all * d)
    var inputs = Python.list(Python.tuple(0, values))
    var program = List[Int32]()
    var xo = 0
    var seed_words = PythonObject(None)
    if permute:
        var rows = layout.alloc(n)
        var sample = layout.alloc(n * d)
        var so = layout.alloc(2)
        # The uint64 seed's (low, high) 32-bit words (`_seed_words`).
        seed_words = classical_typed(Python.tuple(2), "<f4", context["output_context"])
        var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(seed_words)))
        _ = seed_words  # keep the Python object alive through the native call above
        var mask = PythonObject(0xFFFFFFFF)
        words[0] = UInt32(Int(py=py_call(b.int.__dict__["__and__"], seed, mask)))
        _ = seed_words  # keep the Python object alive through the native call above
        words[1] = UInt32(Int(py=py_call(b.int.__dict__["__and__"], py_call(b.int.__dict__["__rshift__"], seed, PythonObject(32)), mask)))
        _ = seed_words  # keep the Python object alive through the native call above
        _ = py_call(inputs.append, Python.tuple(so, Python.tuple(0, seed_words, PythonObject(None), 2, 1)))
        _stage(program, OP_PERM_ROWS, n, so, n_all, rows)
        _stage(program, OP_RGATHER, n * d, xo, d, rows, sample)
        xo = sample
    var sorted = layout.work_words(n * d)
    var st = layout.alloc(6 * d)
    var grid = layout.alloc(nq)
    var nb = layout.alloc(1)
    var level = _vector(Float64(nq - 1), context)
    _ = py_call(inputs.append, Python.tuple(nb, Python.tuple(0, level, PythonObject(None), 1, 1)))
    _stage(program, OP_GRID, nq, nb, nq, 1, grid)
    var qo = layout.alloc(nq * d)
    _stage(program, OP_SORT_COLS, d, xo, n, d, sorted, 0)
    _col_stats(program, layout, mode, entries, xo, n, d, st, False)
    _stage(program, OP_QUANTILE, nq * d, sorted, n, d, grid, nq, qo, st)
    var qt = layout.alloc(nq * d)
    _stage(program, OP_TRANSPOSE, nq * d, qo, d, nq, qt)
    var arena = _run_layout(program^, layout, inputs, entries, context)
    py_setitem(fields, "_q", xprep_take(arena[0], qo, nq * d, 0, context["output_context"]))
    py_setitem(fields, "quantiles_", xprep_take(arena[0], qt, nq, d, context["output_context"]))
    py_setitem(fields, "references_", xprep_take(arena[0], grid, nq, 0, context["output_context"]))
    py_setitem(fields, "n_quantiles_", nq)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = seed_words^
    _ = level^
    _ = arena^


def _quantile_apply(model: PythonObject, values: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    """QuantileTransformer.transform / inverse_transform: one qt_apply or
    qt_inverse stage over the fitted quantiles and references."""
    var fields = py_fields(model)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not Python.import_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + String(py_call(Python.import_module("builtins").str, fitted)))
    var entries = _entries(fields["numeric_mode_"], context)
    var nq = Int(py=fields["n_quantiles_"])
    var quantiles = _input(fields["_q"], 1, False, context)
    var references = _input(fields["references_"], 1, False, context)
    if Int(py=quantiles.size) != nq * d or Int(py=references.size) != nq:
        raise Error("RuntimeError: QuantileTransformer fitted quantiles do not match n_quantiles_")
    var normal = String(fields["output_distribution"]) == "normal"
    var words = n * d
    var qo = words
    var ro = qo + nq * d
    var out = ro + nq
    var dev_out = words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
    var host = out + (0 if dev_out else words)
    if host + (words if dev_out else 0) > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var inputs = Python.list(Python.tuple(0, values),
                             Python.tuple(qo, Python.tuple(0, quantiles, PythonObject(None), nq * d, 1)),
                             Python.tuple(ro, Python.tuple(0, references, PythonObject(None), nq, 1)))
    var program = List[Int32]()
    _stage(program, OP_QT_INVERSE if inverse else OP_QT_APPLY, words, 0, n, d, qo, nq, ro, 1 if normal else 0, out)
    var arena = xprep_storage(host, context)
    var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
    xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
    var result = xprep_take(output_store[0] if dev_out else arena[0], 0 if dev_out else out, n, d, context["output_context"])
    _ = quantiles^
    _ = references^
    _ = arena^
    return result


def _fam(mode: PythonObject, entries: PythonObject) raises -> Int:
    # `_idn_fam`: the IDENTICAL binding's family switches, 0 on another tier.
    if String(mode) != "identical":
        return 0
    var entry = py_get(entries, "x_prep_idn_fam")
    return Int(py=py_call_typed(entry)) if entry is not None else 0


def _pt_flags(mode: PythonObject, entries: PythonObject) raises -> Int:
    # `_ptimpute_flags`: the FAST binding's x_prep_ptimpute_flags, else 0.
    if String(mode) != "fast":
        return 0
    var entry = py_get(entries, "x_prep_ptimpute_flags")
    return Int(py=py_call_typed(entry)) if entry is not None else 0


def _pt_spec_depth(n: Int, d: Int) -> Int:
    """`_expansion_prep._pt_spec_depth`: MOJOLEARN_XPREP_PT_SPEC (default 3,
    clamped to [0, 6]), lowered until the candidates' transforms fit in 2^28
    words (a size range, not a shape); 0 when even two do not."""
    var s = 3
    try:
        s = Int(String(getenv("MOJOLEARN_XPREP_PT_SPEC", "3")).strip())
    except:
        s = 3
    s = max(0, min(s, 6))
    while s > 1 and ((1 << s) - 1) * n * d > 268435456:  # driver-loop: over speculation depths
        s -= 1
    if s == 1 and 2 * n * d > 268435456:
        s = 0
    return s


def _power_fit(model: PythonObject, values: PythonObject, context: PythonObject) raises:
    """PowerTransformer.fit: the host column's pt_fit, or the device's
    golden-section search as stages (staged, or speculated `spec` deep, with
    the blocked folds under IDN_PT_BLOCKED), then the standardization
    statistics: the stages, operands and order of the Python program."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var method_name = fields["method"]
    if py_type(method_name) is not b.str or (String(method_name) != "yeo-johnson" and String(method_name) != "box-cox"):
        raise Error("mojolearn: invalid method " + _repr(method_name))
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var method = 1 if String(method_name) == "box-cox" else 0
    var standardize = Bool(py=fields["standardize"])
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode, context)
    var layout = PrepLayout(n * d)
    var program = List[Int32]()
    var xo = 0
    var st = layout.alloc(6 * d)
    var lam = layout.alloc(d)
    var host = py_get(entries, "x_prep_host_column") is not None
    var flags = _pt_flags(mode, entries)
    # PT_SCORE_STABLE (bit 32): the device's centered coordinates; never on the host binding
    var centered = (flags & 32) != 0 and not host
    var anchor = layout.alloc(d) if centered else -1
    var anchor_kind = layout.alloc(d) if centered else -1
    _col_stats(program, layout, mode, entries, xo, n, d, st)
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"
    var fam_pt = blocked and (_fam(mode, entries) & IDN_PT_BLOCKED) != 0
    var nb = (n + ROW_BLOCK - 1) // ROW_BLOCK
    var unit = 1 if standardize else 0
    if not fam_pt and host:
        _stage(program, OP_PT_FIT, d, xo, n, d, method, st, lam)
    else:
        var tiled = (flags & 1) != 0
        var spec = _pt_spec_depth(n, d) if String(mode) == "identical" else 0
        if spec:
            var mmax = max((1 << spec) - 1, 2)
            var il = 0
            if not tiled and String(getenv("MOJOLEARN_XPREP_PT_INTERLEAVE", "0")) == "1":
                il = 1
            var state = layout.alloc(PT_STATE * d)
            var leval = layout.alloc(d)
            var spl = layout.alloc(d * mmax)
            var vals = layout.alloc(d * mmax)
            var lg = layout.scratch_words(1 if tiled else n * d)
            var tv = layout.scratch_words(1 if tiled else n * d * mmax)
            _stage(program, OP_PT_INIT, d, method, st, d, lam, state, leval, anchor, anchor_kind, unit)
            if not tiled:
                _stage(program, OP_PT_LOG, n * d, xo, n, d, method, lg)
            var bps = 0
            var bpc = 0
            var bss = 0
            var bpj = 0
            var bmean = 0
            var bcnt = 0
            if fam_pt:
                bps = layout.work_words(nb * d * mmax)
                bpc = layout.work_words(nb * d * mmax)
                bss = layout.work_words(nb * d * mmax)
                bpj = layout.work_words(nb * d)
                bmean = layout.work_words(d * mmax)
                bcnt = layout.work_words(d * mmax)
            var k0 = 0
            while k0 <= PT_EVALS - 2:  # driver-loop: over the fixed golden-section evaluations
                var steps = 2 if k0 == 0 else min(spec, PT_EVALS - 1 - k0)
                var m = 2 if k0 == 0 else (1 << steps) - 1
                _stage(program, OP_PT_SPTS, d, state, leval, spl, m, k0)
                _stage(program, OP_PT_SMAP, m * n * d, xo, n, d, method, spl, m, tv, lg, il)
                if fam_pt:
                    var first = 1 if k0 == 0 else 0
                    _stage(program, OP_PTB_PART1, nb * d * m, xo, n, d, method, tv, m, state, bps, bpc, bpj, nb, first, il)
                    _stage(program, OP_PTB_MEAN, d * m, bps, bpc, bpj, nb, d, m, state, bmean, bcnt, first)
                    _stage(program, OP_PTB_PART2, nb * d * m, tv, n, d, m, state, bmean, bcnt, bss, nb, il)
                    _stage(program, OP_PTB_FIN, d * m, bss, nb, d, m, state, spl, vals, bcnt)
                else:
                    _stage(program, OP_PT_SFOLD, d * m, xo, n, d, method, tv, m, state, spl, vals, 1 if k0 == 0 else 0, il)
                _stage(program, OP_PT_SRES, d, state, leval, m, vals, k0, steps, lam)
                k0 += steps
        else:
            var state = layout.alloc(PT_STATE * d)
            var leval = layout.alloc(d)
            var tv = layout.alloc(1 if tiled else n * d)
            var lg = layout.alloc(1 if tiled else n * d)
            _stage(program, OP_PT_INIT, d, method, st, d, lam, state, leval, anchor, anchor_kind, unit)
            if not tiled:
                _stage(program, OP_PT_LOG, n * d, xo, n, d, method, lg)
            var bps = 0
            var bpc = 0
            var bss = 0
            var bpj = 0
            var bmean = 0
            var bcnt = 0
            if fam_pt:
                bps = layout.work_words(nb * d)
                bpc = layout.work_words(nb * d)
                bss = layout.work_words(nb * d)
                bpj = layout.work_words(nb * d)
                bmean = layout.work_words(d)
                _ = d  # keep the Python object alive through the native call above
                bcnt = layout.work_words(d)
                _ = d  # keep the Python object alive through the native call above
            for k in range(PT_EVALS):  # driver-loop: over the fixed golden-section evaluations
                # tiled: the device fuses pt_map into pt_fold (LG1 = 0 either way)
                _stage(program, OP_PT_MAP, n * d, xo, n, d, method, leval, tv, 0 if tiled else lg + 1)
                if fam_pt:
                    var first = 1 if k == 0 else 0
                    _stage(program, OP_PTB_PART1, nb * d, xo, n, d, method, tv, 1, state, bps, bpc, bpj, nb, first, 0)
                    _stage(program, OP_PTB_MEAN, d, bps, bpc, bpj, nb, d, 1, state, bmean, bcnt, first)
                    _stage(program, OP_PTB_PART2, nb * d, tv, n, d, 1, state, bmean, bcnt, bss, nb, 0)
                    _stage(program, OP_PTB_STEP, d, xo, n, d, method, tv, k, state, leval, lam, bss, nb, bcnt)
                else:
                    _stage(program, OP_PT_FOLD, d, xo, n, d, method, tv, k, state, leval, lam)
    var mean = layout.alloc(d)
    var scale = layout.alloc(d)
    if standardize:
        var tx = layout.alloc(n * d)
        var st2 = layout.alloc(6 * d)
        _stage(program, OP_PT_APPLY, n * d, xo, n, d, lam, method, -1, -1, tx, anchor, anchor_kind)
        _col_stats(program, layout, mode, entries, tx, n, d, st2)
        _stage(program, OP_STD_PARAMS, d, st2, d, mean, scale)
    var arena = _run_layout(program^, layout, Python.list(Python.tuple(0, values)), entries, context)
    var base = F32P(unsafe_from_address=Int(py=arena[1]))
    if method == 1:
        for j in range(d):  # driver-loop: over the per-column minima (feature metadata)
            if base[st + 3 * d + j] <= 0:
                raise Error("mojolearn: The Box-Cox transformation can only be applied to strictly positive data")
    var output = context["output_context"]
    py_setitem(fields, "lambdas_", xprep_take(arena[0], lam, d, 0, output))
    var keep_anchor = centered and standardize
    py_setitem(fields, "_pt_anchor", xprep_take(arena[0], anchor, d, 0, output) if keep_anchor else PythonObject(None))
    py_setitem(fields, "_pt_anchor_kind", xprep_take(arena[0], anchor_kind, d, 0, output) if keep_anchor else PythonObject(None))
    py_setitem(fields, "_mean", xprep_take(arena[0], mean, d, 0, output) if standardize else PythonObject(None))
    py_setitem(fields, "_scale", xprep_take(arena[0], scale, d, 0, output) if standardize else PythonObject(None))
    py_setitem(fields, "_method", method)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = arena^


def _power_apply(model: PythonObject, values: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    """PowerTransformer.transform / inverse_transform: one pt_apply or
    pt_inverse stage over the fitted lambdas (and standardization)."""
    var fields = py_fields(model)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not Python.import_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + String(py_call(Python.import_module("builtins").str, fitted)))
    var entries = _entries(fields["numeric_mode_"], context)
    var standardize = Bool(py=fields["standardize"])
    var words = n * d
    var inputs = Python.list(Python.tuple(0, values))
    var host = words
    var lambdas = _input(fields["lambdas_"], 1, False, context)
    var lo = host
    host += d
    _ = py_call(inputs.append, Python.tuple(lo, Python.tuple(0, lambdas, PythonObject(None), d, 1)))
    var mo = -1
    var so = -1
    var mean = PythonObject(None)
    var scale = PythonObject(None)
    if standardize:
        mean = _input(fields["_mean"], 1, False, context)
        scale = _input(fields["_scale"], 1, False, context)
        mo = host
        host += d
        so = host
        host += d
        _ = py_call(inputs.append, Python.tuple(mo, Python.tuple(0, mean, PythonObject(None), d, 1)))
        _ = py_call(inputs.append, Python.tuple(so, Python.tuple(0, scale, PythonObject(None), d, 1)))
    var dev_out = words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
    var out = host
    if not dev_out:
        host += words
    var anchor = py_get(fields, "_pt_anchor")
    var ao = -1
    var ko = -1
    var anchor_values = PythonObject(None)
    var kind_values = PythonObject(None)
    if anchor is not None:
        anchor_values = _input(anchor, 1, False, context)
        kind_values = _input(fields["_pt_anchor_kind"], 1, False, context)
        ao = host
        host += d
        ko = host
        host += d
        _ = py_call(inputs.append, Python.tuple(ao, Python.tuple(0, anchor_values, PythonObject(None), d, 1)))
        _ = py_call(inputs.append, Python.tuple(ko, Python.tuple(0, kind_values, PythonObject(None), d, 1)))
    if host + (words if dev_out else 0) > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    if dev_out:
        out = host
    var program = List[Int32]()
    _stage(program, OP_PT_INVERSE if inverse else OP_PT_APPLY, words, 0, n, d, lo, Int(py=fields["_method"]), mo, so, out, ao, ko)
    var arena = xprep_storage(host, context)
    var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
    xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
    var result = xprep_take(output_store[0] if dev_out else arena[0], 0 if dev_out else out, n, d, context["output_context"])
    _ = lambdas^
    _ = mean^
    _ = scale^
    _ = anchor_values^
    _ = kind_values^
    _ = arena^
    return result


def prep_stateless_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var kind = _kind(model, context)
        var result = PythonObject(None)
        if operation == "get_support" and (kind == "variance_threshold" or kind == "selector"):
            result = prep_support(model, args[0], context)
        elif operation == "inverse_transform" and (kind == "quantile" or kind == "power"):
            var fields = py_fields(model)
            if not py_contains(fields, "n_features_in_"):
                raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
            var values = xprep_values(args[0], _entries(fields["numeric_mode_"], context), context)
            result = _quantile_apply(model, values, True, context) if kind == "quantile" else _power_apply(model, values, True, context)
            _ = values^
        elif operation == "fit" or operation == "fit_transform" or operation == "transform":
            var mode = context["backend_fields"]["_DEFAULT_MODE"]
            if operation == "transform":
                var fields = py_fields(model)
                if not py_contains(fields, "n_features_in_"):
                    raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
                mode = fields["numeric_mode_"]
            var values = xprep_values(args[0], _entries(mode, context), context)
            var selector = kind == "variance_threshold" or kind == "selector"
            if operation == "transform" and kind == "quantile":
                result = _quantile_apply(model, values, False, context)
            elif operation == "transform" and kind == "power":
                result = _power_apply(model, values, False, context)
            elif operation == "transform":
                result = _select_transform(model, values, context) if selector else _transform(model, kind, values, context)
            elif kind == "selector":
                raise Error("TypeError: this selector's fit is not a native stateless operation")
            else:
                if kind == "variance_threshold":
                    _variance_fit(model, values, context)
                elif kind == "quantile":
                    _quantile_fit(model, values, context)
                elif kind == "power":
                    _power_fit(model, values, context)
                else:
                    _fit(model, kind, values, context)
                result = model
                if operation == "fit_transform":
                    # The X converted for the fit is the transform's input.
                    if kind == "quantile":
                        result = _quantile_apply(model, values, False, context)
                    elif kind == "power":
                        result = _power_apply(model, values, False, context)
                    else:
                        result = _select_transform(model, values, context) if selector else _transform(model, kind, values, context)
            _ = values^
        else:
            raise Error("unknown native stateless prep operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
