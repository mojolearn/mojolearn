# SPDX-License-Identifier: Apache-2.0
# Shared steps of the native classifier owner (bindings/prep_classify_native.mojo):
# the model's kind and numeric mode, the connected optional x_prep exports, the
# X admission of `_expansion_prep._x2d`, the CSR admission, the user parameter
# lists, the statistics stage groups (`_col_stats`, `_cs`, `_cls`,
# `_class_stats`, `_gram`, `_qda_cov`, `_nb_counts`) and the partial_fit class
# codes. Every stage group stages the words, operands and order the Python
# helper of the same name staged.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32, classical_dtype_str, classical_c_order, classical_c_bytes, classical_wrap_array, classical_is_array
from bindings.classical_validate import classical_view
from bindings.x_prep_native_inputs import _rows_fields, _values
from bindings.classical_labels import ClassicalLabels, classical_encode_labels
from bindings.prep_classify_prog import ClsProg, NONE, INT32_MAX, off_add, f32_array, i32_array

comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS) used by the statistics groups.
comptime OP_LOOKUP = 7
comptime OP_COUNT_NEG = 8
comptime OP_MATMUL = 13
comptime OP_COL_STATS = 1
comptime OP_CLASS_STATS = 16
comptime OP_QDA_COV = 40
comptime OP_CLASS_STATS_W = 86
comptime OP_LAB_LOAD = 119
comptime OP_COLB_PART = 124
comptime OP_COLB_FOLD = 125
comptime OP_COLB_SS = 126
comptime OP_COLB_VAR = 127
comptime OP_CSB_PART = 129
comptime OP_CSB_FOLD = 130
comptime OP_CSB_SS = 131
comptime OP_CSB_VAR = 132
comptime OP_CSB1_PART = 162
comptime OP_CSB1_NEG = 163
comptime OP_CSR_DENSE = 164
comptime OP_CSB1_SS = 165
comptime OP_P2M_CCOUNT = 200
comptime OP_P2M_CSCAN = 201
comptime OP_P2M_CSTART = 202
comptime OP_P2M_CWRITE = 203
comptime OP_F2_CLAMP0 = 235
comptime OP_GB_PART = 236
comptime OP_GB_PART_ROW = 237
comptime OP_GB_FOLD = 238
comptime OP_QCB_PART = 239
comptime OP_QCB_FOLD = 240

# x_prep/blocked.mojo XB: the rows a block unit folds (_expansion_prep._XB).
comptime XB = 2048
# `_cls`: the most partial words a blocked class_stats keeps (_CLS_BLOCK_WORDS).
comptime CLS_BLOCK_WORDS = 67108864
# `_gram_rows`: the most partial words a blocked Gram keeps (_GRAM_WORDS).
comptime GRAM_WORDS = 67108864
# _IDN_FAM bits.
comptime IDN_STATS_BLOCKED = 1
comptime IDN_CLASS_ONEPASS = 2
# _IDN_INT bits.
comptime IDN_NB_ONEPASS = 2
comptime IDN_NB_CSR = 4
comptime IDN_NB_CSR_DENSE = 8
# _IDN_FAM2 bits.
comptime F2_GRAM = 16
comptime F2_GRAM_ROWTILE = 32
# `_p2m_chunks`: at most this many chunks (_II_CH).
comptime II_CH = 1024


def cls_builtins() raises -> PythonObject:
    return Python.import_module("builtins")


def cls_name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def cls_repr(value: PythonObject) raises -> String:
    return String(py_call(cls_builtins().repr, value))


def cls_str(value: PythonObject) raises -> String:
    return String(py_call(cls_builtins().str, value))


def cls_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The admitted classifier this model is (or subclasses), by its MRO."""
    var b = cls_builtins()
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var kinds = context["cls_kinds"]
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native classifier owner does not serve this class")


def cls_mode(context: PythonObject) raises -> PythonObject:
    """`_expansion_prep._mode()`: the process numeric mode."""
    return context["backend_fields"]["_DEFAULT_MODE"]


def cls_entry(mode: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_optional_prep_entry(_prep_binding(mode), name)`: the connected export,
    or None (a binding that did not connect fails in the run itself)."""
    var entries = py_get(context["cls_bindings"], mode)
    if entries is None or py_type(entries) is cls_builtins().str:
        return PythonObject(None)
    return py_get(entries, PythonObject(name))


def _bits(mode: PythonObject, name: String, context: PythonObject) raises -> Int:
    if String(mode) != "identical":
        return 0
    var entry = cls_entry(mode, name, context)
    if entry is None:
        return 0
    return Int(py=py_call_typed(entry))


def idn_int(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_idn_int", context)


def idn_fam(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_idn_fam", context)


def idn_fam2(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_idn_fam2", context)


def classical_shared(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_classical_shared", context)


def blocked() -> Bool:
    """`_expansion_prep._blocked()`: MOJOLEARN_XPREP_BLOCKED unset or not "0"."""
    return String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"


def check_fitted(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + cls_name(model) + " instance is not fitted yet")


def check_width(model: PythonObject, fields: PythonObject, d: Int) raises:
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not cls_builtins().int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + cls_name(model) + " was fitted with " + cls_str(fitted))


struct XIn(Movable):
    """An admitted X: a float32 C-ordered array, or a fold's float32 2-D
    DeviceRows kept as a handle (the runner gathers it on the device)."""
    var value: PythonObject
    var n: Int
    var d: Int

    def __init__(out self, value: PythonObject, n: Int, d: Int):
        self.value = value
        self.n = n
        self.d = d


def cls_x2d(source: PythonObject, context: PythonObject) raises -> XIn:
    """`_expansion_prep._x2d(X)`."""
    var b = cls_builtins()
    var value = source
    var n = 0
    var d = 0
    if py_type(source) is context["maxabs_rows_class"]:
        var fields = _rows_fields(source, context)
        var dtype = fields["dtype"]
        var shape = fields["shape"]
        if py_type(dtype) is b.str and String(dtype) == "<f4" and py_type(shape) is b.tuple and len(shape) == 2:
            n = Int(py=shape[0])
            d = Int(py=shape[1])
        else:
            # the host rows and the usual conversion (DeviceRows.materialize)
            value = _values(source, context, True)
            n = Int(py=value.shape[0])
            d = Int(py=value.shape[1])
    else:
        value = classical_host_f32(source, 2, "X", context)
        n = Int(py=value.shape[0])
        d = Int(py=value.shape[1])
    if n == 0 or d == 0:
        raise Error("mojolearn: X must be a nonempty two-dimensional array")
    if n > INT32_MAX // d:
        raise Error("mojolearn: X exceeds the native Int32 indexing bound")
    return XIn(value, n, d)


def is_sparse(source: PythonObject) raises -> Bool:
    """A scipy.sparse container, recognized by its defining module."""
    var module = py_type(source).__module__
    return py_type(module) is cls_builtins().str and Bool(py=py_call(module.startswith, "scipy.sparse"))


def _i32_vector(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_as_typed(source, "<i4", "C", 1, name)` for a CSR index array."""
    var view = classical_view(source, context, name)
    if Int(py=view.ndim) != 1:
        raise Error(name + " must be 1-D")
    var n = Int(py=view.shape[0])
    var typestr = classical_dtype_str(view, context)
    var out = i32_array(n, context)
    var dest = I32P(unsafe_from_address=Int(py=classical_address(out)))
    var raw = py_call(context["array_type"], PythonObject(String("q") if typestr == "<i8" else String("i")))
    if typestr == "<i4" or typestr == "<i8":
        _ = py_call(raw.frombytes, classical_c_bytes(view, name, context))
    else:
        raise Error("TypeError: " + name + " must hold int32 or int64 indices")
    var src = Int(py=py_call(raw.buffer_info)[0])
    for i in range(n):  # cpu-route: the CSR index words narrowed to int32 once at input, as _as_typed did
        var v = Int(I64P(unsafe_from_address=src)[i]) if typestr == "<i8" else Int(I32P(unsafe_from_address=src)[i])
        if v > INT32_MAX or v < -INT32_MAX - 1:
            raise Error(name + " holds an index beyond the int32 range")
        dest[i] = Int32(v)
    _ = raw^
    if n == 0:
        return classical_wrap_array(py_call(context["array_type"], "i"), Python.tuple(0), "<i4", "C", context["output_context"])
    return out


struct CsrIn(Movable):
    """`_csr_input(X)`: (indptr, indices, data, n, d) of X.tocsr()."""
    var indptr: PythonObject
    var indices: PythonObject
    var data: PythonObject
    var n: Int
    var d: Int
    var nnz: Int

    def __init__(out self, indptr: PythonObject, indices: PythonObject, data: PythonObject, n: Int, d: Int, nnz: Int):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.n = n
        self.d = d
        self.nnz = nnz


def csr_input(source: PythonObject, context: PythonObject) raises -> CsrIn:
    var csr = py_call(source.tocsr)
    var shape = csr.shape
    var n = Int(py=py_call(cls_builtins().int, shape[0]))
    var d = Int(py=py_call(cls_builtins().int, shape[1]))
    var nnz = Int(py=py_call(cls_builtins().int, csr.nnz))
    var data = classical_host_f32(csr.data, 1, "data", context) if nnz > 0 else f32_array(0, context)
    return CsrIn(_i32_vector(csr.indptr, "indptr", context), _i32_vector(csr.indices, "indices", context), data, n, d, nnz)


def csr_dense_x(mut pr: ClsProg, csr: CsrIn, mode: PythonObject, context: PythonObject) raises -> Int:
    """`_csr_dense_x`: the offset of the X block the device builds from the
    CSR arrays, or -1 when the binding has no csr_dense (the caller densifies;
    -1 is never an arena or scratch offset)."""
    if (idn_int(mode, context) & IDN_NB_CSR_DENSE) == 0:
        return -1
    if csr.n * csr.d > INT32_MAX:
        raise Error("mojolearn: X exceeds the native Int32 indexing bound")
    var xo = pr.work(csr.n * csr.d)
    var ip = pr.put_words(csr.indptr)
    var ix = pr.put_words(csr.indices)
    var dv = pr.put(csr.data)
    pr.stage(OP_CSR_DENSE, csr.n, ip, ix, dv, csr.d, xo)
    return xo


def float_list(values: PythonObject, context: PythonObject) raises -> List[Float64]:
    """`[float(v) for v in (values.tolist() if hasattr(values, "tolist") else values)]`."""
    var b = cls_builtins()
    var items = values
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        var view = classical_view(values, context, "priors")
        var typestr = classical_dtype_str(view, context)
        var code = String("d")
        if typestr == "<f4":
            code = "f"
        elif typestr == "<i8":
            code = "q"
        elif typestr == "<i4":
            code = "i"
        elif typestr == "<u1" or typestr == "|u1" or typestr == "|b1":
            code = "B"
        elif typestr != "<f8":
            raise Error("TypeError: the values must be a sequence of real numbers")
        var store = py_call(context["array_type"], PythonObject(code))
        _ = py_call(store.frombytes, classical_c_bytes(view, "priors", context))
        items = py_call(store.tolist)
    var out = List[Float64]()
    for i in range(len(items)):  # driver-loop: over a user parameter list (class- or feature-count sized)
        out.append(Float64(py=py_call(b.float, items[i])))
    return out^


def given_priors(values: PythonObject, K: Int, who: String, check_sum: Bool, context: PythonObject) raises -> List[Float64]:
    """`_given_priors`: the user priors as floats, checked as the reference checks them."""
    var vals = float_list(values, context)
    if len(vals) != K:
        raise Error("mojolearn: " + who + ": number of priors must match number of classes")
    for i in range(len(vals)):  # driver-loop: over the user prior list (class-count sized)
        if vals[i] < 0:
            raise Error("mojolearn: " + who + ": priors must be non-negative")
    if check_sum:
        var total = prior_sum(vals)
        if abs(total - 1.0) > 1e-8 * max(1.0, abs(total)) and abs(total - 1.0) > 1e-5:
            raise Error("mojolearn: " + who + ": the sum of the priors should be 1")
    return vals^


def prior_sum(vals: List[Float64]) -> Float64:
    # Python's sum(): left to right, binary64.
    var total = 0.0
    for i in range(len(vals)):  # driver-loop: over the user prior list (class-count sized)
        total += vals[i]
    return total


def is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    return Bool(py=py_call(cls_builtins().isinstance, value, context["real_type"]))


def cls_float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(cls_builtins().float, value))


# ---------------------------------------------------------------- statistics stage groups


def col_stats(mut pr: ClsProg, xo: Int, n: Int, d: Int, out: Int, variance: Bool = True):
    """`_col_stats`: six rows of d (count, mean, variance, min, max, max |x|),
    in the blocked order when `_blocked()`."""
    if not blocked():
        pr.stage(OP_COL_STATS, d, xo, n, d, out)
        return
    var nb = (n + XB - 1) // XB
    var part = pr.work(5 * nb * d)
    pr.stage(OP_COLB_PART, nb * d, xo, n, d, part, nb)
    pr.stage(OP_COLB_FOLD, d, part, nb, d, out)
    if variance:
        var ss = pr.work(nb * d)
        pr.stage(OP_COLB_SS, nb * d, xo, n, d, out, ss, nb)
        pr.stage(OP_COLB_VAR, d, ss, nb, d, out)


def cs(mut pr: ClsProg, mode: PythonObject, xo: Int, n: Int, d: Int, out: Int, context: PythonObject, variance: Bool = True) raises:
    """`_cs`: the blocked order under IDN_STATS_BLOCKED, else one col_stats stage."""
    if blocked() and (idn_fam(mode, context) & IDN_STATS_BLOCKED) != 0:
        col_stats(pr, xo, n, d, out, variance)
    else:
        pr.stage(OP_COL_STATS, d, xo, n, d, out)


def class_stats(mut pr: ClsProg, wo: Int, total: Int, xo: Int, n: Int, d: Int, yo: Int, K: Int,
                cnt: Int, mean_in: Int, var_: Int, sums: Int, mode: PythonObject, context: PythonObject,
                with_mode: Bool) raises:
    """`_class_stats` (wo NONE: unweighted; with_mode: its `mode=` was given)."""
    var mean = mean_in
    if blocked():
        var nb = (n + XB - 1) // XB
        var ps = pr.work(nb * K * d)
        var pc = pr.work(nb * K * d)
        var cn = pr.work(K * d)
        if var_ != NONE and mean == NONE:
            mean = pr.work(K * d)
        var onepass = with_mode and (idn_fam(mode, context) & IDN_CLASS_ONEPASS) != 0
        if onepass:
            pr.stage(OP_CSB1_PART, nb * d, xo, n, d, yo, K, ps, pc, nb, wo, NONE, NONE)
        else:
            pr.stage(OP_CSB_PART, nb * K * d, xo, n, d, yo, K, ps, pc, nb, wo)
        pr.stage(OP_CSB_FOLD, K * d, ps, pc, nb, K, d, cn, cnt, mean, sums, wo)
        if var_ != NONE and onepass:
            pr.stage(OP_CSB1_SS, nb * d, xo, n, d, yo, K, mean, cn, ps, nb, wo)
            pr.stage(OP_CSB_VAR, K * d, ps, nb, K, d, cn, var_)
        elif var_ != NONE:
            pr.stage(OP_CSB_SS, nb * K * d, xo, n, d, yo, K, mean, cn, ps, nb, wo)
            pr.stage(OP_CSB_VAR, K * d, ps, nb, K, d, cn, var_)
        return
    if wo == NONE:
        pr.stage(OP_CLASS_STATS, total, xo, n, d, yo, K, cnt, mean, var_, sums)
    else:
        pr.stage(OP_CLASS_STATS_W, total, xo, n, d, yo, K, cnt, mean, var_, sums, wo)


def cls_stats(mut pr: ClsProg, mode: PythonObject, xo: Int, n: Int, d: Int, yo: Int, K: Int,
              cnt: Int, mean: Int, var_: Int, sums: Int, context: PythonObject, tail: Int = -2) raises:
    """`_cls`: the blocked order under IDN_STATS_BLOCKED while its partials fit,
    else one class_stats stage (tail -2: no trailing parameter; else the one
    trailing word QuadraticDiscriminantAnalysis passes)."""
    var nb = (n + XB - 1) // XB
    if blocked() and (idn_fam(mode, context) & IDN_STATS_BLOCKED) != 0 and nb * K * d <= CLS_BLOCK_WORDS:
        class_stats(pr, NONE, K * d, xo, n, d, yo, K, cnt, mean, var_, sums, mode, context, True)
    elif tail == -2:
        pr.stage(OP_CLASS_STATS, K * d, xo, n, d, yo, K, cnt, mean, var_, sums)
    else:
        pr.stage(OP_CLASS_STATS, K * d, xo, n, d, yo, K, cnt, mean, var_, sums, tail)


def gram_rows(v: Int, n: Int, cells: Int) -> Int:
    """`_gram_rows`."""
    var rows = (v >> 16) if (v >> 16) != 0 else XB
    while ((n + rows - 1) // rows) * cells > GRAM_WORDS and rows < 4194304:  # driver-loop: the block length doubles (a function of the shape only)
        rows *= 2
    return rows if ((n + rows - 1) // rows) * cells <= GRAM_WORDS else 0


def gram(mut pr: ClsProg, mode: PythonObject, z: Int, n: Int, d: Int, g: Int, context: PythonObject) raises:
    """`_gram`: G = Z'Z."""
    var v = idn_fam2(mode, context)
    var rows = gram_rows(v, n, d * d) if (blocked() and (v & F2_GRAM) != 0) else 0
    if rows == 0:
        pr.stage(OP_MATMUL, d * d, z, 1, d, z, d, 1, g, d, n, NONE, NONE)
        return
    var nb = (n + rows - 1) // rows
    var part = pr.work(nb * d * d)
    if (v & F2_GRAM_ROWTILE) != 0:
        pr.stage(OP_GB_PART_ROW, nb * d, z, n, d, part, nb, rows)
    else:
        pr.stage(OP_GB_PART, nb * d * d, z, n, d, part, nb, rows)
    pr.stage(OP_GB_FOLD, d * d, part, nb, d, g)


def qda_cov(mut pr: ClsProg, mode: PythonObject, xo: Int, n: Int, d: Int, yo: Int, K: Int, mean: Int, cnt: Int,
            cov: Int, context: PythonObject) raises:
    """`_qda_cov`: the K class covariances (divisor the class count)."""
    var v = idn_fam2(mode, context)
    var rows = gram_rows(v, n, K * d * d) if (blocked() and (v & F2_GRAM) != 0) else 0
    if rows == 0:
        pr.stage(OP_QDA_COV, K * d * d, xo, n, d, yo, mean, cnt, cov)
        return
    var nb = (n + rows - 1) // rows
    var part = pr.work(nb * K * d * d)
    pr.stage(OP_QCB_PART, nb * d * d, xo, n, d, yo, K, mean, part, nb, rows)
    pr.stage(OP_QCB_FOLD, K * d * d, part, nb, K, d, cnt, cov)


def nb_onepass(mode: PythonObject, context: PythonObject) raises -> Bool:
    return blocked() and (idn_int(mode, context) & IDN_NB_ONEPASS) != 0


def nb_counts(mut pr: ClsProg, wo: Int, xo: Int, n: Int, d: Int, yo: Int, K: Int, cnt: Int, sums: Int,
              thr: Int, neg: Int):
    """`_nb_counts`: the discrete naive Bayes count pass, one unit per (block, column)."""
    var nb = (n + XB - 1) // XB
    var ps = pr.work(nb * K * d)
    var pc = pr.work(nb * K * d)
    var cn = pr.work(K * d)
    var ng = pr.work(nb * d) if neg != NONE else NONE
    pr.stage(OP_CSB1_PART, nb * d, xo, n, d, yo, K, ps, pc, nb, wo, thr, ng)
    pr.stage(OP_CSB_FOLD, K * d, ps, pc, nb, K, d, cn, cnt, NONE, sums, wo)
    if neg != NONE:
        pr.stage(OP_CSB1_NEG, d, ng, nb, d, neg)


def p2m_class_counts(mut pr: ClsProg, co: Int, n: Int, K: Int) -> Int:
    """`_p2m_class_rows(pr, co, n, K, rows=False)`: the class counts' offset."""
    var ch = max(II_CH, (n + II_CH - 1) // II_CH, (n * K + 16777215) // 16777216, 1)
    var nch = max(1, (n + ch - 1) // ch)
    var cnt = pr.alloc(nch * K)
    var tot = pr.alloc(K)
    pr.stage(OP_P2M_CCOUNT, nch, co, n, K, ch, cnt)
    pr.stage(OP_P2M_CSCAN, K, cnt, nch, K, tot)
    return tot


# ---------------------------------------------------------------- labels


def encode_y(model: PythonObject, fields: PythonObject, y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_Classifier._encode_y`: classes_ set, the int32 codes returned."""
    var labels = classical_encode_labels(y, context)
    if labels.n != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    py_setitem(fields, "classes_", labels.classes)
    return labels.codes


struct PartialCodes(Movable):
    """`_partial_codes`' answer: the batch's class codes as int32 words
    (`codes`), or, for a numeric label buffer whose classes are exact
    ascending float32 words (`_DevCodes`), the raw label words the program
    looks up among the classes on the device."""
    var first: Bool
    var dev: Bool
    var codes: PythonObject
    var words: PythonObject
    var kind: Int
    var n: Int
    var cats: PythonObject
    var y: PythonObject

    def __init__(out self, first: Bool, dev: Bool, codes: PythonObject, words: PythonObject, kind: Int, n: Int,
                 cats: PythonObject, y: PythonObject):
        self.first = first
        self.dev = dev
        self.codes = codes
        self.words = words
        self.kind = kind
        self.n = n
        self.cats = cats
        self.y = y


def partial_walk(fields: PythonObject, y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_partial_walk`: the batch's codes among classes_, refusing (and naming)
    labels outside them. The batch is encoded once; its classes are looked up
    in classes_ (Python equality, as the dict did) and the row codes remapped."""
    var b = cls_builtins()
    var labels = classical_encode_labels(y, context)
    if labels.n != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    var classes = fields["classes_"]
    var index = Python.dict()
    for i in range(len(classes)):  # driver-loop: over the class index table (classes-sized)
        py_setitem(index, classes[i], PythonObject(i))
    var found = labels.classes
    var remap = List[Int]()
    var bad = Python.list()
    for k in range(len(found)):  # driver-loop: over the batch's distinct labels (classes-sized)
        var at = py_get(index, found[k])
        if at is None:
            _ = py_call(bad.append, py_call(b.repr, found[k]))
            remap.append(0)
        else:
            remap.append(Int(py=at))
    if len(bad) > 0:
        var named = py_call(b.sorted, py_call(b.set, bad))
        raise Error("mojolearn: The target label(s) " + cls_str(named) + " in y do not exist in the initial classes " + cls_str(classes))
    var out = i32_array(n, context)
    var dest = I32P(unsafe_from_address=Int(py=classical_address(out)))
    var src = I32P(unsafe_from_address=Int(py=classical_address(labels.codes)))
    for i in range(n):  # cpu-route: Python object labels of any kind, the explicit label input step
        dest[i] = Int32(remap[Int(src[i])])
    return out


def _label_kind(typestr: String) -> Tuple[Int, Int]:
    """`_LABEL_KIND`: (lab_load kind, words per label), or (-1, 0)."""
    if typestr == "<f4":
        return (0, 1)
    if typestr == "<i4":
        return (1, 1)
    if typestr == "<u4":
        return (2, 1)
    if typestr == "<i8":
        return (3, 2)
    if typestr == "<f8":
        return (4, 2)
    return (-1, 0)


def partial_codes(model: PythonObject, fields: PythonObject, y: PythonObject, classes: PythonObject, n: Int,
                  context: PythonObject) raises -> PartialCodes:
    """`_partial_codes(est, y, classes, n, defer=True)`."""
    var b = cls_builtins()
    var first = py_get(fields, "classes_") is None
    if first and classes is None:
        raise Error("mojolearn: classes must be passed on the first call to partial_fit.")
    if classes is not None:
        var given = classical_encode_labels(classes, context).classes
        if not first and Bool(py=py_call(b.list, given) != py_call(b.list, fields["classes_"])):
            raise Error("mojolearn: `classes=" + cls_str(given) + "` is not the same as on last call to partial_fit, was: " + cls_str(fields["classes_"]))
        if first:
            py_setitem(fields, "classes_", given)
    # `_partial_dev`: every class an exact int or float with an exact,
    # strictly ascending float32 word, and y a numeric label vector of n labels
    var cl = py_call(b.list, fields["classes_"])
    var K = len(cl)
    var dev = K > 0
    var cats = f32_array(K, context)
    var cp = F32P(unsafe_from_address=Int(py=classical_address(cats)))
    for k in range(K):  # driver-loop: over the classes (classes-sized)
        var t = py_type(cl[k])
        if t is not b.int and t is not b.float:
            dev = False
            break
        var v = Float64(py=py_call(b.float, cl[k]))
        cp[k] = Float32(v)
        if Float64(cp[k]) != v or (k > 0 and not (Float64(cp[k - 1]) < Float64(cp[k]))):
            dev = False
            break
    if dev:
        var words = label_words(y, n, context)
        if words[0] >= 0:
            return PartialCodes(first, True, PythonObject(None), words[2], words[0], n, cats, y)
    return PartialCodes(first, False, partial_walk(fields, y, n, context), PythonObject(None), 0, n, cats, y)


def label_words(y: PythonObject, n: Int, context: PythonObject) raises -> Tuple[Int, Int, PythonObject]:
    """`_label_buffer(y)` when it holds exactly n labels: (kind, n, the raw
    int32 words), or (-1, 0, None) for the walk."""
    var b = cls_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes:
        return (-1, 0, PythonObject(None))
    var view = PythonObject(None)
    try:
        view = classical_view(y, context, "y")
    except:
        return (-1, 0, PythonObject(None))
    var spec = _label_kind(classical_dtype_str(view, context))
    var size = Int(py=view.size)
    if spec[0] < 0 or size == 0 or Int(py=view.ndim) < 1 or size != n:
        return (-1, 0, PythonObject(None))
    var largest = 0
    for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
        largest = max(largest, Int(py=view.shape[k]))
    if size != largest or not classical_c_order(view, context):
        return (-1, 0, PythonObject(None))
    var nw = size * spec[1]
    if nw > 1073741824:
        return (-1, 0, PythonObject(None))
    var store = py_call(context["array_type"], "i")
    _ = py_call(store.frombytes, classical_c_bytes(view, "y", context))
    return (spec[0], size, classical_wrap_array(store, Python.tuple(nw), "<i4", "C", context["output_context"]))


def stage_partial_codes(mut pr: ClsProg, pc: PartialCodes, context: PythonObject) raises -> Tuple[Int, Int]:
    """`_stage_partial_codes`: (the float class codes' offset, the unknown
    label count's offset, NONE for the walk's codes)."""
    if not pc.dev:
        return (pr.put_codes(pc.codes), NONE)
    var n = pc.n
    var K = Int(py=pc.cats.size)
    var x = pr.work(n)
    var w = pr.put_words(pc.words)
    pr.stage(OP_LAB_LOAD, n, w, pc.kind, x)
    var c = pr.work(n)
    var neg = pr.alloc(1)
    var yo = pr.alloc(n)
    var co = pr.put(pc.cats)
    var ko = pr.put_scalar(Float64(K), context)
    pr.stage(OP_LOOKUP, n, x, n, 1, co, K, ko, c)
    pr.stage(OP_COUNT_NEG, 1, c, n, 1, neg)
    pr.stage(OP_F2_CLAMP0, n, c, yo)
    return (yo, neg)


def partial_refuse(pr: ClsProg, neg: Int, fields: PythonObject, pc: PartialCodes, context: PythonObject) raises:
    """The program's post-run refusal of a device-coded batch: the walk names
    the unknown labels."""
    if neg == NONE:
        return
    if pr.value(neg) > 0:
        _ = partial_walk(fields, pc.y, pc.n, context)
        raise Error("mojolearn: a target label in y does not exist in the initial classes")
