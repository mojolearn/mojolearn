# SPDX-License-Identifier: Apache-2.0
# The native owner of LabelEncoder, LabelBinarizer and MultiLabelBinarizer
# (dispatched by bindings/prep_owner_native.mojo). Each public fit,
# fit_transform, transform and inverse_transform is one call here, which
# builds the x_prep programs the Python `_Prog` built (the same stages,
# operands and allocation order, so the same bits): a numeric label buffer
# crosses as its raw words (`_label_buffer`: lab_load, the device sort and a
# chunked run scan, or the small-integer presence route when the binding
# exports x_prep_label_present), str labels take the native str-label lookup,
# other Python label objects the explicit label input step. The fitted
# attributes keep the names, types and dtypes the Python set (`_classes`,
# `_cats`, `classes_`, `y_type_`, `numeric_mode_`, `_given`) and the refusals
# keep their classes and wording.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_array_storage import native_array_zeros, native_array_empty, native_array_from_bytes
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.classical_labels import (
    LabelsView, labels_vector, labels_empty, labels_address, labels_label_kind, labels_view, labels_cast,
    labels_helper, _numeric_label_view, _flatten_labels, _sorted_classes)
from bindings.scaler_native import _stage
from bindings.scaler_lifecycle_native import _entries
from bindings.x_prep_native_owner import xprep_values
from bindings.prep_encode_native import (
    EncProg, OUT_MARK, NONE, _take1, _word, _iword, _label_chunk, _fit_categories, _codes, _default_mode, _truth)
from bindings.str_labels_native import str_lookup
from bindings.user_objects_native import _Env, _f32_exact, _indicator

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_SORT_COLS = 0
comptime OP_LOOKUP = 7
comptime OP_COUNT_NEG = 8
comptime OP_F2I = 11
comptime OP_LABEL_BINARIZE = 51
comptime OP_SCATTER_ONES = 52
comptime OP_BLOCK_ARGMAX = 76
comptime OP_INDICATOR = 87
comptime OP_LAB_LOAD = 119
comptime OP_UNIQ_COUNT = 120
comptime OP_UNIQ_SCAN = 121
comptime OP_UNIQ_WRITE = 122
comptime OP_CHUNK_NEG = 123
comptime OP_ROW_ONES = 135
comptime OP_CAT_ZERO = 157
comptime OP_CAT_PRESENT = 158
comptime OP_PRES_COUNT = 159
comptime OP_PRES_WRITE = 160
comptime OP_F2_CODE_GATHER = 241
comptime OP_C2_BIN_CODE = 242

# `_CAT_R`, `_CAT_CH`: the presence route's value range and scan chunk.
comptime CAT_R = 4096
comptime CAT_CH = 64
# A label buffer of more int32 words takes the label input step.
comptime LABEL_WORDS_MAX = 1073741824

comptime MULTILABEL = "multilabel-indicator"
comptime REFUSE_MULTIOUTPUT = "mojolearn: Multioutput target data is not supported with label binarization"
comptime UNSEEN = "mojolearn: y contains previously unseen labels"


# ---------------------------------------------------------------- shared
def _builtins() raises -> PythonObject:
    return Python.import_module("builtins")


def _raise_python(cls: PythonObject, message: String) raises:
    # A Python exception of exactly `cls` (OverflowError, IndexError), which
    # the door restores as it is.
    ref cpy = Python().cpython()
    cpy.PyErr_SetString(cls._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    var failure = stash_python_error()
    raise failure^


def _check_fitted(fields: PythonObject, name: String) raises:
    # `_check_fitted`: hasattr(self, "_classes").
    if not py_contains(fields, "_classes"):
        raise Error("RuntimeError: mojolearn: this " + name + " instance is not fitted yet")


def _text(value: PythonObject) raises -> String:
    return String(py_call(_builtins().str, value))


def _out(context: PythonObject) raises -> PythonObject:
    return context["output_context"]


def _take_words(ran: PythonObject, offset: Int, count: Int, shape: PythonObject, code: String, typestr: String,
                width: Int, context: PythonObject) raises -> PythonObject:
    """`_Prog.get_i32(off, shape)` (code "i", 4-byte words) and `_class_words`
    (code "q" / "d", 8-byte words): a copy of the program's words at
    `offset` (the output region at OUT_MARK) as an Array of `typestr`."""
    if count == 0:
        return classical_wrap_array(native_array_empty(code), shape, typestr, "C", _out(context))
    var b = _builtins()
    var store = ran[0][0]
    var start = offset
    if offset == OUT_MARK:
        store = ran[1][0]
        start = 0
    var words = count * width // 4
    var part = py_call(b.memoryview, store)[py_call(b.slice, start, start + words)]
    var data = native_array_from_bytes(code, py_call(part.cast, "B"))
    return classical_wrap_array(data, shape, typestr, "C", _out(context))


def _take_i32(ran: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_Prog.get_i32(off, rows)` (cols < 0) or `get_i32(off, (rows, cols))`.
    if cols < 0:
        return _take_words(ran, offset, rows, Python.tuple(rows), "i", "<i4", 4, context)
    return _take_words(ran, offset, rows * cols, Python.tuple(rows, cols), "i", "<i4", 4, context)


def _f32_matrix(values: List[Float64], rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # A float32 C-order Array of the values rounded once to float32.
    var out = labels_empty(Python.tuple(rows, cols), "<f4", _out(context))
    var dest = F32P(unsafe_from_address=labels_address(out))
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over a fitted category or label word list
        dest[i] = Float32(values[i])
        _ = out  # keep the Python object alive through the native call above
    return out


def _classes_array(classes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_classes_array`: int classes an int64 Array, real classes a float64
    Array (`Array.from_list`), anything else a list."""
    var b = _builtins()
    var kind = labels_label_kind(classes, context)
    var k = len(classes)
    if kind == "int":
        var out = labels_vector(k, "<i8", _out(context))
        var dest = I64P(unsafe_from_address=labels_address(out))
        _ = out  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct classes
            dest[i] = Int64(Int(py=py_call(b.int, classes[i])))
            _ = out  # keep the Python object alive through the native call above
        return out
    if kind == "float":
        var out = labels_vector(k, "<f8", _out(context))
        var dest = F64P(unsafe_from_address=labels_address(out))
        _ = out  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct classes
            dest[i] = Float64(py=py_call(b.float, classes[i]))
            _ = out  # keep the Python object alive through the native call above
        return out
    return py_call(b.list, classes)


def _range_classes(k: Int) raises -> PythonObject:
    # list(range(k)): a multilabel indicator's column indices.
    var out = Python.list()
    for j in range(k):  # driver-loop: over the indicator's columns
        _ = py_call(out.append, PythonObject(j))
    return out


# ---------------------------------------------------------------- label buffers
@fieldwise_init
struct LabelBuf(Copyable, Movable):
    """`_LabelBuf`: a numeric label vector's raw int32 words, its lab_load
    kind, its label count, its word count and whether it is a float dtype."""
    var kind: Int
    var words: PythonObject
    var n: Int
    var nw: Int
    var is_float: Bool


def _lab_spec(typestr: String) -> Tuple[Int, Int]:
    # `_LABEL_KIND`: (lab_load kind, words per label), or (-1, 0).
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


def _label_buffer(y: PythonObject, context: PythonObject) raises -> Optional[LabelBuf]:
    """`_label_buffer(y)`: a numeric label vector (a buffer exporter or an
    Array, not a list, str, bool or pandas object) of a lab_load dtype as its
    raw int32 words, or None (the label input step)."""
    var bool_source = False
    var found = _numeric_label_view(y, context, bool_source)
    if not found or bool_source:
        return None
    var view = found.value().copy()
    var spec = _lab_spec(view.typestr)
    var n = view.size()
    if spec[0] < 0 or n == 0:
        return None
    var nw = n * spec[1]
    if nw > LABEL_WORDS_MAX:
        return None
    var words = classical_wrap_array(native_array_zeros("i", nw), Python.tuple(nw), "<i4", "C", _out(context))
    memcpy(dest=I32P(unsafe_from_address=Int(py=classical_address(words))), src=I32P(unsafe_from_address=view.addr), count=nw)
    _ = view  # keep the Python object alive through the native call above
    _ = words  # keep the Python object alive through the native call above
    return LabelBuf(spec[0], words, n, nw, spec[0] == 0 or spec[0] == 4)


def _label_load(mut p: EncProg, lb: LabelBuf) raises -> Int:
    # `_label_load`: the float32 word of each label (NaN when float32 cannot hold it).
    var x = p.work(lb.n)
    var wo = p.put(lb.words, lb.nw)
    _stage(p.program, OP_LAB_LOAD, lb.n, wo, lb.kind, x)
    return x


def _label_unique(mut p: EncProg, x: Int, n: Int) -> Tuple[Int, Int]:
    """`_label_unique`: the sorted distinct words of x (sort_cols and a
    chunked run scan). Returns (U offset, count offset)."""
    var s = p.work(n)
    _stage(p.program, OP_SORT_COLS, 1, x, n, 1, s, 1)
    var ch = _label_chunk(n)
    var nch = (n + ch - 1) // ch
    var cnt = p.work(nch)
    var off = p.work(nch)
    var u = p.alloc(n)
    var k = p.alloc(1)
    _stage(p.program, OP_UNIQ_COUNT, nch, s, n, ch, cnt)
    _stage(p.program, OP_UNIQ_SCAN, 1, cnt, nch, off, k)
    _stage(p.program, OP_UNIQ_WRITE, nch, s, n, ch, off, u)
    return (u, k)


@fieldwise_init
struct LabelFit(Copyable, Movable):
    """(classes list, device categories Array or None, int32 codes Array or None)."""
    var classes: PythonObject
    var cats: PythonObject
    var codes: PythonObject


def _label_classes_of(ran: PythonObject, lb: LabelBuf, u: Int, k: Int, context: PythonObject) raises -> Optional[LabelFit]:
    """`_label_classes_of`: (classes, cats) read back after the run, or None
    when a label has no exact float32 word (NaN sorts last)."""
    var K = Int(_word(ran, k))
    if K == 0:
        return None
    var last = _word(ran, u + K - 1)
    if last != last:
        return None
    var cats = _take1(ran, u, K, context)
    var classes = Python.list()
    for i in range(K):  # driver-loop: over the distinct class values the device found
        var w = _word(ran, u + i)
        if lb.is_float:
            _ = py_call(classes.append, PythonObject(Float64(w)))
        else:
            _ = py_call(classes.append, PythonObject(Int(w)))
    return LabelFit(classes, cats, PythonObject(None))


def _label_present_enabled(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_label_present_enabled`: the binding's x_prep_label_present export is the switch.
    var m = String(mode)
    if m != "fast" and m != "identical":
        return False
    var entry = py_get(entries, "x_prep_label_present")
    if entry is None:
        return False
    return _truth(py_call_typed(entry))


def _label_fit_present(entries: PythonObject, lb: LabelBuf, codes: Bool, context: PythonObject) raises -> Optional[LabelFit]:
    """`_label_fit_present`: exact small-integer distinct labels by presence
    flags; None (the general sort) when the device's BAD flag is set."""
    var n = lb.n
    var p = EncProg()
    var x = _label_load(p, lb)
    var fl = p.work(CAT_R)
    var bad = p.alloc(1)
    var u = p.alloc(CAT_R)
    var k = p.alloc(1)
    var nch = (CAT_R + CAT_CH - 1) // CAT_CH
    var cnt = p.work(nch)
    var off = p.work(nch)
    _stage(p.program, OP_CAT_ZERO, CAT_R, fl)
    _stage(p.program, OP_CAT_PRESENT, n, x, n, 1, CAT_R, fl, bad)
    _stage(p.program, OP_PRES_COUNT, nch, fl, CAT_R, CAT_CH, cnt)
    _stage(p.program, OP_UNIQ_SCAN, 1, cnt, nch, off, k)
    _stage(p.program, OP_PRES_WRITE, nch, fl, CAT_R, CAT_CH, off, u)
    var out = NONE
    if codes:
        var c = p.work(n)
        _stage(p.program, OP_LOOKUP, n, x, n, 1, u, CAT_R, k, c)
        out = p.output(n, entries)
        _stage(p.program, OP_F2I, n, c, out)
    var ran = p.run(entries, context)
    if _iword(ran, bad) != 0:
        return None
    var got = _label_classes_of(ran, lb, u, k, context)
    if not got:
        return None
    var fit = got.value().copy()
    if codes:
        fit.codes = _take_i32(ran, out, n, -1, context)
    return fit^


def _label_fit_device(mode: PythonObject, lb: LabelBuf, codes: Bool, context: PythonObject) raises -> Optional[LabelFit]:
    """`_label_fit_device`: one program, (classes, cats, int32 codes or None),
    or None for the label input step."""
    var entries = _entries(mode, context)
    if _label_present_enabled(mode, entries):
        var got = _label_fit_present(entries, lb, codes, context)
        if got:
            return got^
    var n = lb.n
    var p = EncProg()
    var x = _label_load(p, lb)
    var uk = _label_unique(p, x, n)
    var out = NONE
    if codes:
        var c = p.work(n)
        _stage(p.program, OP_LOOKUP, n, x, n, 1, uk[0], n, uk[1], c)
        out = p.output(n, entries)
        _stage(p.program, OP_F2I, n, c, out)
    var ran = p.run(entries, context)
    var got = _label_classes_of(ran, lb, uk[0], uk[1], context)
    if not got:
        return None
    var fit = got.value().copy()
    if codes:
        fit.codes = _take_i32(ran, out, n, -1, context)
    return fit^


def _label_codes_device(mut p: EncProg, lb: LabelBuf, cats: PythonObject, context: PythonObject) raises -> Int:
    """`_label_codes_device`: each label's index among `cats` (-1 when
    absent). Returns the float codes offset."""
    var x = _label_load(p, lb)
    var K = Int(py=cats.size)
    var c = p.work(lb.n)
    var uo = p.put_array(cats, context)
    var size = List[Float64]()
    size.append(Float64(K))
    var ko = p.put_floats(size, context)
    _stage(p.program, OP_LOOKUP, lb.n, x, lb.n, 1, uo, K, ko, c)
    return c


# ---------------------------------------------------------------- the label input step
def _numeric_labels(values: PythonObject) raises -> PythonObject:
    # `_numeric_labels`: the labels as floats when every one is a real
    # number float32 holds exactly, else None (user_objects f32_exact).
    var env = _Env()
    return _f32_exact(env, values)


def _label_classes(mode: PythonObject, values: PythonObject, context: PythonObject) raises -> LabelFit:
    """`_label_classes`: numeric labels by a device sort and run scan
    (`_fit_categories`), anything else by the ORDER RULE (sorted_classes)."""
    var nums = _numeric_labels(values)
    if nums is None or len(nums) == 0:
        var ordered = _sorted_classes(values, context)
        return LabelFit(ordered.classes, PythonObject(None), PythonObject(None))
    var m = len(nums)
    var words = List[Float64]()
    for i in range(m):  # cpu-route: Python label objects, the explicit label input step
        words.append(Float64(py=nums[i]))
    var arr = _f32_matrix(words, m, 1, context)
    var cats = _fit_categories(Python.tuple(0, arr, PythonObject(None), m, 1), mode, _entries(mode, context), context)[0]
    var b = _builtins()
    var integral = context["labels_integral_type"]
    var ints = 0
    for i in range(len(values)):  # cpu-route: Python label objects, the explicit label input step
        if Bool(py=py_call(b.isinstance, values[i], integral)):
            ints += 1
    var all_ints = ints == len(values)
    var K = Int(py=cats.size)
    var source = F32P(unsafe_from_address=Int(py=classical_address(cats)))
    var classes = Python.list()
    _ = cats  # keep the Python object alive through the native call above
    for i in range(K):  # driver-loop: over the distinct class values the device found
        var w = source[i]
        _ = cats  # keep the Python object alive through the native call above
        if all_ints:
            _ = py_call(classes.append, PythonObject(Int(w)))
        else:
            _ = py_call(classes.append, PythonObject(Float64(w)))
    return LabelFit(classes, cats, PythonObject(None))


def _label_codes(mut p: EncProg, values: PythonObject, cats: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    """`_label_codes`: each label's index among `cats` (or -1) from its
    float64 value rounded to float32. Returns (codes, unknown count)."""
    var b = _builtins()
    var n = len(values)
    var words = List[Float64]()
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        words.append(Float64(py=py_call(b.float, values[i])))
    var arr = _f32_matrix(words, n, 1, context)
    return _codes(p, Python.tuple(0, arr, PythonObject(None), n, 1), Python.list(cats), context)


def _object_index(classes: PythonObject) raises -> PythonObject:
    # {c: i for i, c in enumerate(classes)}: a duplicated class keeps its last index.
    var index = Python.dict()
    for i in range(len(classes)):  # cpu-route: non-str object labels, the explicit label input step
        py_setitem(index, classes[i], PythonObject(i))
    return index


def _object_codes(mut p: EncProg, classes: PythonObject, values: PythonObject, context: PythonObject) raises -> Int:
    # pr.put_list([index.get(v, -1) for v in values]).
    var index = _object_index(classes)
    var codes = List[Float64]()
    for i in range(len(values)):  # cpu-route: non-str object labels, the explicit label input step
        codes.append(Float64(Int(py=py_call(index.get, values[i], PythonObject(-1)))))
    return p.put_floats(codes, context)


def _str_codes(classes: PythonObject, values: PythonObject, as_float: Bool, context: PythonObject) raises -> PythonObject:
    """`_labels.str_labels("lookup", list(classes), values, as_float)`: (codes
    Array, first missing labels, missing count), or None when a class is not a str."""
    var native = str_lookup(py_call(_builtins().list, classes), values, as_float)
    if native is None:
        return native
    var n = len(values)
    var codes = classical_wrap_array(native[0], Python.tuple(n), "<f4" if as_float else "<i4", "C", _out(context))
    return Python.tuple(codes, native[1], native[2])


def _stage_class_gather(mut p: EncProg, codes: Int, n: Int, cats: PythonObject, ints: Bool, context: PythonObject) raises -> Tuple[Int, Int]:
    """`_stage_class_gather`: the n float codes to their classes as 64-bit
    words (f2_code_gather) and the count of codes that are not an integer in
    [0, cats.size). Returns (words offset, count offset)."""
    var out = p.alloc(2 * n)
    var bad = p.work(n)
    var neg = p.alloc(1)
    var co = p.put_array(cats, context)
    _stage(p.program, OP_F2_CODE_GATHER, n, codes, Int(py=cats.size), co, 0 if ints else 1, out, bad)
    _stage(p.program, OP_COUNT_NEG, 1, bad, n, 1, neg)
    return (out, neg)


def _class_words(ran: PythonObject, out: Int, n: Int, ints: Bool, context: PythonObject) raises -> PythonObject:
    # `_class_words`: the n 64-bit class words as an int64 or float64 Array.
    if ints:
        return _take_words(ran, out, n, Python.tuple(n), "q", "<i8", 8, context)
    return _take_words(ran, out, n, Python.tuple(n), "d", "<f8", 8, context)


def _int64_codes(values: PythonObject) raises -> List[Int]:
    """`_user_objects.call("ints", values)`: int(v) of each value, refused
    with OverflowError outside int64."""
    var b = _builtins()
    var lo = PythonObject(-9223372036854775807 - 1)
    var hi = PythonObject(9223372036854775807)
    var out = List[Int]()
    for i in range(len(values)):  # cpu-route: Python code objects read once, the explicit label input step
        var x = py_call(b.int, values[i])
        if Bool(py=x < lo) or Bool(py=hi < x):
            _raise_python(b.OverflowError, "Python int too large to convert to C long")
        out.append(Int(py=x))
    return out^


def _take_classes(classes: PythonObject, codes: List[Int]) raises -> PythonObject:
    # `_user_objects.take(classes, codes)`: [classes[i] for i in codes], a
    # negative index from the end, IndexError out of range.
    var k = len(classes)
    var out = Python.list()
    for t in range(len(codes)):  # cpu-route: one class object per code, the explicit label output step
        var i = codes[t]
        var j = i + k if i < 0 else i
        if j < 0 or j >= k:
            _raise_python(_builtins().IndexError, "list index out of range")
        _ = py_call(out.append, classes[j])
    return out


# ---------------------------------------------------------------- multilabel indicators
def _multilabel_buffer(y: PythonObject, shape: PythonObject, context: PythonObject) raises -> Optional[PythonObject]:
    """`_multilabel_buffer`: a 2-D buffer of finite integral values with at
    most two distinct values as a float32 C-order (n, K) Array; None for an
    empty or one-column y; the multioutput refusal otherwise."""
    var b = _builtins()
    if Int(py=shape[0]) == 0 or Int(py=shape[1]) < 2:
        return None
    if "bool" in _text(py_call(b.getattr, y, "dtype", "")):
        raise Error(REFUSE_MULTIOUTPUT)
    var view = labels_view(y, "y", context)
    var size = view.size()
    var rows = view.shape[0]
    var cols = view.shape[1]
    # `as_f64_c(y, ndim=2)`: float64 in the source's storage order (the
    # tests below do not depend on the order)
    var wide = labels_vector(size, "<f8", _out(context))
    labels_cast(view.addr, view.typestr, labels_address(wide), "<f8", size, context)
    _ = view  # keep the Python object alive through the native call above
    var reduce = labels_helper(context, "reduce_stat")
    var addr = labels_address(wide)
    if not _truth(py_call_typed(reduce, PythonObject(addr), PythonObject(1), PythonObject(size), PythonObject(4))):
        raise Error(REFUSE_MULTIOUTPUT)
    var lo = Float64(py=py_call_typed(reduce, PythonObject(addr), PythonObject(1), PythonObject(size), PythonObject(0)))
    var hi = Float64(py=py_call_typed(reduce, PythonObject(addr), PythonObject(1), PythonObject(size), PythonObject(1)))
    var src = F64P(unsafe_from_address=addr)
    var hits = 0
    for i in range(size):  # driver-loop: over the indicator's values (the native equality counts)
        if src[i] == lo:
            hits += 1
        if hi != lo and src[i] == hi:
            hits += 1
        _ = wide  # keep the Python object alive through the native call above
    if hits != size:
        raise Error(REFUSE_MULTIOUTPUT)
    # `_x2d(Y.astype("<f4"), "y")`: float32 in C order
    var out = labels_empty(Python.tuple(rows, cols), "<f4", _out(context))
    if view.c_order() or view.both_orders():
        labels_cast(addr, "<f8", labels_address(out), "<f4", size, context)
    else:
        var staged = labels_vector(size, "<f4", _out(context))
        labels_cast(addr, "<f8", labels_address(staged), "<f4", size, context)
        _ = py_call_typed(labels_helper(context, "transpose_f32"), PythonObject(labels_address(staged)),
                          PythonObject(labels_address(out)), PythonObject(cols), PythonObject(rows))
        _ = staged  # keep the Python object alive through the native call above
    _ = wide  # keep the Python object alive through the native call above
    return out


def _multilabel_indicator(y: PythonObject, context: PythonObject) raises -> Optional[PythonObject]:
    """`_multilabel_indicator`: y as a float32 (n, K) Array when it is a
    multilabel indicator (2-D, more than one column, at most two distinct
    integral values); None otherwise; refuses a multioutput y."""
    var b = _builtins()
    var shape = py_call(b.getattr, y, "shape", PythonObject(None))
    if shape is not None:
        if len(shape) != 2:
            return None
        return _multilabel_buffer(y, shape, context)
    if not Bool(py=py_call(b.isinstance, y, Python.tuple(b.list, b.tuple))) or len(y) == 0:
        return None
    var seq = Python.tuple(b.list, b.tuple)
    for i in range(len(y)):  # cpu-route: Python list-of-lists y, the explicit label input step
        if not Bool(py=py_call(b.isinstance, y[i], seq)):
            return None
    if len(y[0]) < 2:
        return None
    var env = _Env()
    var got = _indicator(env, y)
    var status = Int(py=got[0])
    if status == 1:
        raise Error("mojolearn: y rows have different lengths")
    if status == 2:
        raise Error(REFUSE_MULTIOUTPUT)
    return classical_wrap_array(got[1], Python.tuple(len(y), Int(py=got[2])), "<f4", "C", _out(context))


# ---------------------------------------------------------------- LabelEncoder
def _set_fit(fields: PythonObject, fit: LabelFit, context: PythonObject) raises:
    py_setitem(fields, "_classes", fit.classes)
    py_setitem(fields, "_cats", fit.cats)


def _fit_labels(fields: PythonObject, mode: PythonObject, y: PythonObject, context: PythonObject) raises:
    # `_label_fit_device` of a numeric buffer, else `_label_classes(flatten_labels(y))`.
    var lb = _label_buffer(y, context)
    if lb:
        var got = _label_fit_device(mode, lb.value(), False, context)
        if got:
            _set_fit(fields, got.value(), context)
            return
    _set_fit(fields, _label_classes(mode, _flatten_labels(y, context), context), context)


def _le_fit(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var mode = _default_mode(context)
    py_setitem(fields, "numeric_mode_", mode)
    _fit_labels(fields, mode, y, context)
    py_setitem(fields, "classes_", _classes_array(fields["_classes"], context))
    return model


def _le_fit_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    # A numeric buffer: classes and codes in ONE program.
    var lb = _label_buffer(y, context)
    if lb:
        var fields = py_fields(model)
        var mode = _default_mode(context)
        py_setitem(fields, "numeric_mode_", mode)
        var got = _label_fit_device(mode, lb.value(), True, context)
        if got:
            var fit = got.value().copy()
            _set_fit(fields, fit, context)
            py_setitem(fields, "classes_", _classes_array(fit.classes, context))
            return fit.codes
    _ = _le_fit(model, y, context)
    return _le_transform(model, y, context)


def _le_transform_device(mode: PythonObject, lb: LabelBuf, cats: PythonObject, context: PythonObject) raises -> Optional[PythonObject]:
    """LabelEncoder.transform of a numeric label buffer: the codes and the
    per-chunk unknown flags in one program; None when a label is unknown."""
    var entries = _entries(mode, context)
    var n = lb.n
    var p = EncProg()
    var codes = _label_codes_device(p, lb, cats, context)
    var out = p.output(n, entries)
    _stage(p.program, OP_F2I, n, codes, out)
    var ch = _label_chunk(n)
    var nch = (n + ch - 1) // ch
    var neg = p.alloc(nch)
    _stage(p.program, OP_CHUNK_NEG, nch, codes, n, ch, neg)
    var ran = p.run(entries, context)
    for i in range(nch):  # driver-loop: over the per-chunk unknown flags
        if _iword(ran, neg + i) != 0:
            return None
    return _take_i32(ran, out, n, -1, context)


def _le_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _builtins()
    var fields = py_fields(model)
    _check_fitted(fields, "LabelEncoder")
    var cats = fields["_cats"]
    var classes = fields["_classes"]
    var mode = fields["numeric_mode_"]
    if cats is not None:
        var lb = _label_buffer(y, context)
        if lb:
            var done = _le_transform_device(mode, lb.value(), cats, context)
            if done:
                return done.value()
            # an unknown label: the label input step names the refusal
    var values = _flatten_labels(y, context)
    var n = len(values)
    if n == 0:
        return classical_wrap_array(native_array_empty("i"), Python.tuple(0), "<i4", "C", _out(context))
    if cats is None or _numeric_labels(values) is None:
        var native = _str_codes(classes, values, False, context)
        if native is not None:
            if Int(py=native[2]) != 0:
                raise Error(UNSEEN + ": " + _text(native[1]))
            return native[0]
        var index = _object_index(classes)
        var missing = Python.list()
        for i in range(n):  # cpu-route: non-str object labels, the explicit label input step
            if not py_contains(index, values[i]):
                _ = py_call(missing.append, values[i])
        if len(missing) > 0:
            var first = Python.list()
            for i in range(min(5, len(missing))):  # driver-loop: over the first five unseen labels
                _ = py_call(first.append, missing[i])
            raise Error(UNSEEN + ": " + _text(first))
        var store = native_array_zeros("i", n)
        var codes = classical_wrap_array(store, Python.tuple(n), "<i4", "C", _out(context))
        var dest = I32P(unsafe_from_address=Int(py=classical_address(codes)))
        _ = codes  # keep the Python object alive through the native call above
        for i in range(n):  # cpu-route: non-str object labels, the explicit label input step
            dest[i] = Int32(Int(py=index[values[i]]))
            _ = codes  # keep the Python object alive through the native call above
        return codes
    var entries = _entries(mode, context)
    var p = EncProg()
    var lc = _label_codes(p, values, cats, context)
    var out = p.alloc(n)
    _stage(p.program, OP_F2I, n, lc[0], out)
    var ran = p.run(entries, context)
    if _word(ran, lc[1]) > 0:
        raise Error(UNSEEN)
    return _take_i32(ran, out, n, -1, context)


def _le_inverse_device(fields: PythonObject, lb: LabelBuf, cats: PythonObject, classes: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`_inverse_device`: each code checked (an integer in [0, K)) and its
    class written as the int64 / float64 word returned, one program."""
    var entries = _entries(fields["numeric_mode_"], context)
    var n = lb.n
    var ints = labels_label_kind(classes, context) == "int"
    var p = EncProg()
    var x = _label_load(p, lb)
    var og = _stage_class_gather(p, x, n, cats, ints, context)
    var ran = p.run(entries, context)
    if _word(ran, og[1]) > 0:
        raise Error(UNSEEN)
    return _class_words(ran, og[0], n, ints, context)


def _le_inverse(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _check_fitted(fields, "LabelEncoder")
    var cats = fields["_cats"]
    var classes = fields["_classes"]
    if cats is not None:
        # `_inverse_device`: an integer or float code buffer over numeric classes
        var lb = _label_buffer(y, context)
        if lb:
            return _le_inverse_device(fields, lb.value(), cats, classes, context)
    # str classes or a Python code list: the codes read once, range-checked
    var codes = _int64_codes(_flatten_labels(y, context))
    var k = len(classes)
    for i in range(len(codes)):  # cpu-route: Python code objects read once, the explicit label input step
        if codes[i] < 0 or codes[i] >= k:
            raise Error(UNSEEN)
    return _classes_array(_take_classes(classes, codes), context)


# ---------------------------------------------------------------- LabelBinarizer
def _lb_fit(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _builtins()
    var fields = py_fields(model)
    if _truth(fields["sparse_output"]):
        raise Error("NotImplementedError: mojolearn: LabelBinarizer(sparse_output=True) is not implemented")
    var integral = context["labels_integral_type"]
    var neg = fields["neg_label"]
    var pos = fields["pos_label"]
    if not (Bool(py=py_call(b.isinstance, neg, integral)) and Bool(py=py_call(b.isinstance, pos, integral))
            and _truth(py_call(Python.import_module("operator").lt, neg, pos))):
        raise Error("mojolearn: neg_label must be an integer below pos_label")
    var mode = _default_mode(context)
    py_setitem(fields, "numeric_mode_", mode)
    var ind = _multilabel_indicator(y, context)
    if ind:
        var columns = _range_classes(Int(py=ind.value().shape[1]))
        py_setitem(fields, "_classes", columns)
        py_setitem(fields, "_cats", PythonObject(None))
        py_setitem(fields, "classes_", _classes_array(columns, context))
        py_setitem(fields, "y_type_", PythonObject(MULTILABEL))
        return model
    _fit_labels(fields, mode, y, context)
    var classes = fields["_classes"]
    py_setitem(fields, "classes_", _classes_array(classes, context))
    py_setitem(fields, "y_type_", PythonObject("binary" if len(classes) <= 2 else "multiclass"))
    return model


def _lb_transform_indicator(mode: PythonObject, arr: PythonObject, classes: PythonObject, neg: Int, pos: Int,
                            context: PythonObject) raises -> PythonObject:
    # A multilabel indicator: each nonzero entry pos_label, each zero neg_label.
    var n = Int(py=arr.shape[0])
    var K = Int(py=arr.shape[1])
    if K != len(classes):
        raise Error("mojolearn: classes " + _text(classes) + " mismatch with the " + String(K) + " label columns in y")
    var entries = _entries(mode, context)
    var p = EncProg()
    var out = p.output(n * K, entries)
    var io = p.put(arr, n * K)
    _stage(p.program, OP_INDICATOR, n * K, io, n * K, 0, NONE, neg, pos, out)
    var ran = p.run(entries, context)
    return _take_i32(ran, out, n, K, context)


def _lb_transform_device(mode: PythonObject, lb: LabelBuf, cats: PythonObject, K: Int, W: Int, neg: Int, pos: Int,
                         context: PythonObject) raises -> PythonObject:
    # A numeric label buffer: the codes and the dense output in one program.
    var entries = _entries(mode, context)
    var n = lb.n
    var p = EncProg()
    var codes = _label_codes_device(p, lb, cats, context)
    var out = p.output(n * W, entries)
    _stage(p.program, OP_LABEL_BINARIZE, n * W, codes, n, K, 1 if W == 1 else 0, neg, pos, W, out)
    var ran = p.run(entries, context)
    return _take_i32(ran, out, n, W, context)


def _lb_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _builtins()
    var fields = py_fields(model)
    _check_fitted(fields, "LabelBinarizer")
    var classes = fields["_classes"]
    var cats = fields["_cats"]
    var mode = fields["numeric_mode_"]
    var y_type = String(fields["y_type_"])
    var neg = Int(py=py_call(b.int, fields["neg_label"]))
    var pos = Int(py=py_call(b.int, fields["pos_label"]))
    var ind = _multilabel_indicator(y, context)
    if ind or y_type == MULTILABEL:
        if y_type != MULTILABEL:
            raise Error("mojolearn: The object was not fitted with multilabel input.")
        if not ind:
            raise Error("mojolearn: y is not a multilabel indicator; the binarizer was fitted with one")
        return _lb_transform_indicator(mode, ind.value(), classes, neg, pos, context)
    var K = len(classes)
    var binary = K <= 2
    var W = 1 if binary else K
    if cats is not None and K > 1:
        var lb = _label_buffer(y, context)
        if lb:
            # the raw words up, the codes and the dense output on the device
            return _lb_transform_device(mode, lb.value(), cats, K, W, neg, pos, context)
    var values = _flatten_labels(y, context)
    var n = len(values)
    var entries = _entries(mode, context)
    var p = EncProg()
    var codes = NONE
    if cats is None or _numeric_labels(values) is None:
        var native = _str_codes(classes, values, True, context) if n > 0 else PythonObject(None)
        if native is not None:
            codes = p.put(native[0], n)
        else:
            codes = _object_codes(p, classes, values, context)
    else:
        codes = _label_codes(p, values, cats, context)[0]
    if K == 1:
        var none = List[Float64]()
        for _ in range(n):  # driver-loop: one absent code per label (one class)
            none.append(-1.0)
        codes = p.put_floats(none, context)
    var out = p.output(n * W, entries)
    _stage(p.program, OP_LABEL_BINARIZE, n * W, codes, n, K, 1 if binary else 0, neg, pos, W, out)
    var ran = p.run(entries, context)
    return _take_i32(ran, out, n, W, context)


def _block_argmax(mut p: EncProg, values: PythonObject, K: Int, context: PythonObject) raises -> Int:
    # `_block_argmax(pr, arr, [K], None, False)`: one code per row.
    var n = Int(py=values[3])
    var W = Int(py=values[4])
    var xo = p.put_x(values)
    var starts = List[Float64]()
    starts.append(0.0)
    var so = p.put_floats(starts, context)
    var widths = List[Float64]()
    widths.append(Float64(K))
    var wo = p.put_floats(widths, context)
    var codes = p.alloc(n)
    _stage(p.program, OP_BLOCK_ARGMAX, n, xo, n, W, 1, so, wo, NONE, 0, codes)
    return codes


def _threshold(fields: PythonObject, threshold: PythonObject) raises -> Float64:
    # threshold, or halfway between neg_label and pos_label.
    if threshold is None:
        var b = _builtins()
        return Float64(Int(py=py_call(b.int, fields["pos_label"])) + Int(py=py_call(b.int, fields["neg_label"]))) / 2.0
    return Float64(py=py_call(_builtins().float, threshold))


def _one_float(value: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.append(value)
    return out^


def _lb_inverse(model: PythonObject, y: PythonObject, threshold: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _check_fitted(fields, "LabelBinarizer")
    var classes = fields["_classes"]
    var cats = fields["_cats"]
    var entries = _entries(fields["numeric_mode_"], context)
    var values = xprep_values(y, entries, context)
    var n = Int(py=values[3])
    var W = Int(py=values[4])
    var K = len(classes)
    var y_type = String(fields["y_type_"])
    var p = EncProg()
    if y_type == MULTILABEL:
        if W != K:
            raise Error("mojolearn: Y has " + String(W) + " columns, expected " + String(K))
        var t = _threshold(fields, threshold)
        var out = p.alloc(n * W)
        var xo = p.put_x(values)
        var to = p.put_floats(_one_float(t), context)
        _stage(p.program, OP_INDICATOR, n * W, xo, n * W, 1, to, 0, 1, out)
        var ran_ml = p.run(entries, context)
        return _take_i32(ran_ml, out, n, W, context)
    var codes = NONE
    if y_type == "multiclass":
        if W != K:
            raise Error("mojolearn: Y has " + String(W) + " columns, expected " + String(K))
        codes = _block_argmax(p, values, K, context)
    else:
        if W > 2:
            raise Error("mojolearn: output_type='binary', but y.shape = (" + String(n) + ", " + String(W) + ")")
        var t = _threshold(fields, threshold)
        # the last column against the threshold, one code per row (c2_bin_code)
        codes = p.alloc(n)
        var xo = p.put_x(values)
        var to = p.put_floats(_one_float(t), context)
        _stage(p.program, OP_C2_BIN_CODE, n, xo, n, W, to, codes)
    if cats is None:
        # str classes: the codes come back for the explicit object-label decode
        var ran_obj = p.run(entries, context)
        var idx = List[Int]()
        for i in range(n):  # driver-loop: over one code per row read back
            idx.append(Int(_word(ran_obj, codes + i)) if K != 1 else 0)
        return _classes_array(_take_classes(classes, idx), context)
    # numeric classes: each code's class gathered on the device; one class: both codes name it
    var ints = labels_label_kind(classes, context) == "int"
    var table = cats
    if K <= 1:
        var both = List[Float64]()
        var source = F32P(unsafe_from_address=Int(py=classical_address(cats)))
        _ = cats  # keep the Python object alive through the native call above
        for _ in range(2):  # small-loop(2: the one class named by both codes)
            for i in range(Int(py=cats.size)):  # driver-loop: over the fitted category words
                both.append(Float64(source[i]))
                _ = cats  # keep the Python object alive through the native call above
        table = classical_wrap_array(native_array_zeros("f", len(both)), Python.tuple(len(both)), "<f4", "C", _out(context))
        var dest = F32P(unsafe_from_address=Int(py=classical_address(table)))
        _ = table  # keep the Python object alive through the native call above
        for i in range(len(both)):  # driver-loop: over the doubled category words
            dest[i] = Float32(both[i])
            _ = table  # keep the Python object alive through the native call above
    var og = _stage_class_gather(p, codes, n, table, ints, context)
    var ran = p.run(entries, context)
    return _class_words(ran, og[0], n, ints, context)


# ---------------------------------------------------------------- MultiLabelBinarizer
@fieldwise_init
struct MLBFlat(Copyable, Movable):
    """`_MLBFlat`: the label sets as one int64 label buffer and int32 row offsets."""
    var n: Int
    var lb: LabelBuf
    var offs: PythonObject


def _mlb_flat(y: PythonObject, context: PythonObject) raises -> Optional[MLBFlat]:
    """`_mlb_flat(y)`: list or tuple rows that are lists, tuples or sets of
    plain ints as one int64 label buffer and the row offsets; None (the label
    input step) for anything else, an int outside int64 or no label at all."""
    var b = _builtins()
    if not Bool(py=py_call(b.isinstance, y, Python.tuple(b.list, b.tuple))):
        return None
    var nrows = len(y)
    if nrows == 0:
        return None
    var rows = Python.list()
    for i in range(nrows):  # cpu-route: the user's Python label rows, the explicit label input step
        var row = y[i]
        var t = py_type(row)
        if t is not b.list and t is not b.tuple and t is not b.set and t is not b.frozenset:
            return None
        _ = py_call(rows.append, row if (t is b.list or t is b.tuple) else py_call(b.list, row))
    var total = 0
    for i in range(nrows):  # cpu-route: the user's Python label rows, the explicit label input step
        var row = rows[i]
        for j in range(len(row)):  # cpu-route: Python label objects, the explicit label input step
            if py_type(row[j]) is not b.int:
                return None
        total += len(row)
    if total == 0 or total > LABEL_WORDS_MAX or 2 * total > LABEL_WORDS_MAX:
        return None
    var lo = PythonObject(-9223372036854775807 - 1)
    var hi = PythonObject(9223372036854775807)
    var nw = 2 * total
    var words = classical_wrap_array(native_array_zeros("i", nw), Python.tuple(nw), "<i4", "C", _out(context))
    var dest = I64P(unsafe_from_address=Int(py=classical_address(words)))
    var offs = classical_wrap_array(native_array_zeros("i", nrows + 1), Python.tuple(nrows + 1), "<i4", "C", _out(context))
    var od = I32P(unsafe_from_address=Int(py=classical_address(offs)))
    _ = words  # keep the Python object alive through the native call above
    _ = offs  # keep the Python object alive through the native call above
    var at = 0
    for i in range(nrows):  # cpu-route: the user's Python label rows, the explicit label input step
        var row = rows[i]
        for j in range(len(row)):  # cpu-route: Python label objects, the explicit label input step
            var v = row[j]
            if Bool(py=v < lo) or Bool(py=hi < v):
                return None
            dest[at] = Int64(Int(py=v))
            _ = words  # keep the Python object alive through the native call above
            at += 1
        od[i + 1] = Int32(at)
        _ = offs  # keep the Python object alive through the native call above
    return MLBFlat(nrows, LabelBuf(3, words, total, nw, False), offs)


def _mlb_transform_flat(model: PythonObject, fl: MLBFlat, context: PythonObject) raises -> PythonObject:
    # `_transform_flat`: the int32 indicator from the device route, one program.
    var fields = py_fields(model)
    var entries = _entries(fields["numeric_mode_"], context)
    var n = fl.n
    var K = len(fields["_classes"])
    var p = EncProg()
    var codes = _label_codes_device(p, fl.lb, fields["_cats"], context)
    var offs = p.put(fl.offs, n + 1)
    var out = p.output(n * max(K, 1), entries)
    _stage(p.program, OP_ROW_ONES, n, codes, offs, K, out)
    var ran = p.run(entries, context)
    return _take_i32(ran, out, n, K, context)


def _fit_flat(fields: PythonObject, mode: PythonObject, fl: Optional[MLBFlat], context: PythonObject) raises -> Bool:
    # `_fit_flat`: classes_ from the device route; False when it does not apply.
    if not fl:
        return False
    var got = _label_fit_device(mode, fl.value().lb, False, context)
    if not got:
        return False
    _set_fit(fields, got.value(), context)
    py_setitem(fields, "_given", PythonObject(False))
    return True


def _rows_lists(y: PythonObject) raises -> PythonObject:
    # [list(row) for row in y]
    var b = _builtins()
    var out = Python.list()
    var rows = py_call(b.list, y)
    for i in range(len(rows)):  # cpu-route: Python iterables of labels, the explicit label input step
        _ = py_call(out.append, py_call(b.list, rows[i]))
    return out


def _mlb_fit(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _builtins()
    var fields = py_fields(model)
    if _truth(fields["sparse_output"]):
        raise Error("NotImplementedError: mojolearn: MultiLabelBinarizer(sparse_output=True) is not implemented")
    var mode = _default_mode(context)
    py_setitem(fields, "numeric_mode_", mode)
    if fields["classes"] is not None:
        py_setitem(fields, "_classes", py_call(b.list, fields["classes"]))
        py_setitem(fields, "_given", PythonObject(True))
        py_setitem(fields, "_cats", PythonObject(None))
    elif not _fit_flat(fields, mode, _mlb_flat(y, context), context):
        var rows = _rows_lists(y)
        var flat = Python.list()
        for i in range(len(rows)):  # cpu-route: Python iterables of labels, the explicit label input step
            _ = py_call(flat.extend, rows[i])
        if len(flat) > 0:
            _set_fit(fields, _label_classes(mode, flat, context), context)
        else:
            py_setitem(fields, "_classes", Python.list())
            py_setitem(fields, "_cats", PythonObject(None))
        py_setitem(fields, "_given", PythonObject(False))
    py_setitem(fields, "classes_", _classes_array(fields["_classes"], context))
    return model


def _mlb_fit_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if fields["classes"] is None and not _truth(fields["sparse_output"]):
        var fl = _mlb_flat(y, context)
        if fl:
            var mode = _default_mode(context)
            py_setitem(fields, "numeric_mode_", mode)
            if _fit_flat(fields, mode, fl, context):
                py_setitem(fields, "classes_", _classes_array(fields["_classes"], context))
                return _mlb_transform_flat(model, fl.value(), context)
    var rows = _rows_lists(y)
    _ = _mlb_fit(model, rows, context)
    return _mlb_transform(model, rows, context)


def _mlb_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _check_fitted(fields, "MultiLabelBinarizer")
    var classes = fields["_classes"]
    var cats = fields["_cats"]
    if cats is not None and len(classes) > 0:
        var fl = _mlb_flat(y, context)
        if fl:
            return _mlb_transform_flat(model, fl.value(), context)
    var rows = _rows_lists(y)
    var n = len(rows)
    var K = len(classes)
    var flat = Python.list()
    var owner = List[Float64]()
    for i in range(n):  # cpu-route: Python iterables of labels, the explicit label input step
        var row = rows[i]
        for j in range(len(row)):  # cpu-route: Python label objects, the explicit label input step
            _ = py_call(flat.append, row[j])
            owner.append(Float64(i))
    var entries = _entries(fields["numeric_mode_"], context)
    var p = EncProg()
    var m = len(flat)
    if m == 0:
        var empty = p.alloc(n * max(K, 1))
        var ran = p.run(entries, context)
        return _take_i32(ran, empty, n, K, context)
    var codes = NONE
    if cats is None or _numeric_labels(flat) is None:
        var native = _str_codes(classes, flat, True, context)
        if native is not None:
            codes = p.put(native[0], m)
        else:
            codes = _object_codes(p, classes, flat, context)
    else:
        codes = _label_codes(p, flat, cats, context)[0]
    var ro = p.put_floats(owner, context)
    var out = p.output(n * max(K, 1), entries)
    _stage(p.program, OP_SCATTER_ONES, m, codes, ro, K, out)
    var ran = p.run(entries, context)
    return _take_i32(ran, out, n, K, context)


def _mlb_inverse(model: PythonObject, yt: PythonObject, context: PythonObject) raises -> PythonObject:
    # One tuple of labels per row: the user list of Python label tuples.
    var b = _builtins()
    var fields = py_fields(model)
    _check_fitted(fields, "MultiLabelBinarizer")
    var classes = fields["_classes"]
    var k = len(classes)
    var rows = py_call(yt.tolist) if Bool(py=py_call(b.hasattr, yt, "tolist")) else py_call(b.list, yt)
    var out = Python.list()
    for i in range(len(rows)):  # cpu-route: builds the user list of Python label tuples (object output)
        var row = py_call(b.list, rows[i])
        var picked = Python.list()
        for j in range(len(row)):  # cpu-route: builds the user list of Python label tuples (object output)
            if _truth(row[j]):
                if j >= k:
                    _raise_python(b.IndexError, "list index out of range")
                _ = py_call(picked.append, classes[j])
        _ = py_call(out.append, py_call(b.tuple, picked))
    return out


# ---------------------------------------------------------------- the dispatch
def prep_labels_call(model: PythonObject, kind: String, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """LabelEncoder / LabelBinarizer / MultiLabelBinarizer fit(y),
    fit_transform(y), transform(y), inverse_transform(y) (LabelBinarizer:
    inverse_transform(Y, threshold))."""
    if kind == "label_encoder":
        if operation == "fit":
            return _le_fit(model, args[0], context)
        if operation == "fit_transform":
            return _le_fit_transform(model, args[0], context)
        if operation == "transform":
            return _le_transform(model, args[0], context)
        if operation == "inverse_transform":
            return _le_inverse(model, args[0], context)
    elif kind == "label_binarizer":
        if operation == "fit":
            return _lb_fit(model, args[0], context)
        if operation == "fit_transform":
            _ = _lb_fit(model, args[0], context)
            return _lb_transform(model, args[0], context)
        if operation == "transform":
            return _lb_transform(model, args[0], context)
        if operation == "inverse_transform":
            return _lb_inverse(model, args[0], args[1], context)
    elif kind == "multilabel_binarizer":
        if operation == "fit":
            return _mlb_fit(model, args[0], context)
        if operation == "fit_transform":
            return _mlb_fit_transform(model, args[0], context)
        if operation == "transform":
            return _mlb_transform(model, args[0], context)
        if operation == "inverse_transform":
            return _mlb_inverse(model, args[0], context)
    raise Error("TypeError: the native label owner does not serve " + operation)
