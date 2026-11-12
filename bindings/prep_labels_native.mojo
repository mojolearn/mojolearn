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
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_array_storage import native_array_zeros, native_array_empty, native_array_from_bytes
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.classical_labels import (
    labels_address, labels_label_kind, labels_cast,
    labels_reduce_stat, _numeric_label_view, _flatten_labels, _sorted_classes)
from bindings.admitted_input import admitted_array, admitted_vector, admit_c_view, admitted_dtype
from core.preprocessing_program import prep_program_stage as _stage
from bindings.scaler_lifecycle_native import _entries
from bindings.prep_entry_cache import prep_entry_word
from bindings.x_prep_native_owner import xprep_values
from bindings.prep_encode_native import (
    EncProg, OUT_MARK, NONE, _take1, _word, _iword, _label_chunk, _fit_categories, _codes, _default_mode, _truth)
from bindings.str_labels_native import str_lookup, StrTable, _read, str_unique_into, str_classes, str_codes_into
from bindings.boundary_read import text_buffer_view, read_text_rows_into, pack_ucs4_str, admitted_buffer_view
from core.boundary_layout import BOUNDARY_KIND_UCS4, BOUNDARY_KIND_UINT, gather_i64
from bindings.classical_pandas import classical_admit_value
from bindings.label_admission_native import LabelTypeKinds
>>>

from bindings.user_objects_native import _Env, _f32_exact, _indicator
from bindings.cpython_table import cpython_table
from bindings.p3_labels_metrics_native import p3_item, p3_type_word, p3_dict_codes, p3_decode_list
from core.label_words_impl import label_codes_i32_to_f64

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
    return native_loaded_module("builtins")


def _raise_python(cls: PythonObject, message: String) raises:
    # A Python exception of exactly `cls` (OverflowError, IndexError), which
    # the door restores as it is.
    var text = message
    ref cpy = Python().cpython()
    cpy.PyErr_SetString(cls._obj_ptr, text.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
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
    var out = admitted_array(Python.tuple(rows, cols), "<f4", _out(context))
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
        var out = admitted_vector(k, "<i8", _out(context))
        var dest = I64P(unsafe_from_address=labels_address(out))
        _ = out  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct classes
            dest[i] = Int64(Int(py=py_call(b.int, classes[i])))
            _ = out  # keep the Python object alive through the native call above
        return out
    if kind == "float":
        var out = admitted_vector(k, "<f8", _out(context))
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
    var view = found.value()
    var spec = _lab_spec(admitted_dtype(view, context))
    var n = Int(py=view.size)
    if spec[0] < 0 or n == 0:
        return None
    var nw = n * spec[1]
    if nw > LABEL_WORDS_MAX:
        return None
    var words = classical_wrap_array(native_array_zeros("i", nw), Python.tuple(nw), "<i4", "C", _out(context))
    unsafe_memcpy(dest=I32P(unsafe_from_address=Int(py=classical_address(words))), src=I32P(unsafe_from_address=Int(py=classical_address(view))), count=nw)
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
    return prep_entry_word(entries, "x_prep_label_present") != 0  # read once per binding


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
def _numeric_label_words(values: PythonObject, mut words: List[Float64], mut all_ints: Bool,
                         count_ints: Bool, context: PythonObject) raises -> Bool:
    """`_numeric_labels` (user_objects f32_exact) in one pass that keeps the
    words: True when every label is a real number, not a bool, that float32
    holds exactly and none is NaN, with `words` the labels' float values
    (what float(v) returns, the value the codes are looked up by). With
    `count_ints`, `all_ints` tells whether every label is integral. This
    replaces the Python float list f32_exact built and the one or two further
    walks over the labels that read it back."""
    var env = _Env()
    var integral = context["labels_integral_type"] if count_ints else PythonObject(None)
    # Exact int and float labels (the common case) are read by type pointer
    # and PyFloat_AsDouble (the same double float(v) returns, the same
    # OverflowError for an int too large); an exact int is integral, an
    # exact float is not. Other classes keep the ABC checks (p3-prep).
    var int_type = env.b.int
    var float_type = env.b.float
    all_ints = count_ints
    var n = len(values)
    words.reserve(n)
    # Lane p2-labels-lists-arrays: an exact list or tuple is read borrowed
    # from its C storage and an exact int / float converted by
    # PyFloat_AsDouble (no subscript, no PythonObject per label).
    var exact = py_type(values) is env.b.list or py_type(values) is env.b.tuple
    var is_tuple = py_type(values) is env.b.tuple
    var int_word = Int(int_type._obj_ptr)
    var float_word = Int(float_type._obj_ptr)
    ref cpy = Python().cpython()
    var t = cpython_table()
    # One pass over the caller's label objects (Python objects a kernel file
    # cannot read): main's recorded input-step debt
    # (tools/hooks/compute_in_shells_baseline.txt), its body made native
    # for exact int / float labels by lane P2.
    for i in range(n):  # list-input: one borrowed read per label of the flattened Python label list (one pass; 'U' vectors take _text_labels)
        var x: Float64
        var is_int: Bool
        if exact:
            var raw = p3_item(values, i, is_tuple)
            var word = p3_type_word(raw)
            if word == int_word or word == float_word:
                x = t[].PyFloat_AsDouble(raw)
                if x == -1.0:
                    if t[].PyErr_Occurred():
                        raise cpy.unsafe_get_error()
                if x != x or Float64(Float32(x)) != x:
                    return False
                words.append(x)
                if all_ints and word != int_word:
                    all_ints = False
                continue
        var v = values[i]
        var cls = py_type(v)
        if cls is int_type:
            x = Float64(py=v)
            is_int = True
        elif cls is float_type:
            x = Float64(py=v)
            is_int = False
        else:
            if env.isinstance(v, env.b.bool) or not env.isinstance(v, env.real):
                return False
            x = Float64(py=py_call(env.b.float, v))
            is_int = all_ints and Bool(py=py_call(env.b.isinstance, v, integral))
        if x != x or Float64(Float32(x)) != x:
            return False
        words.append(x)
        if all_ints and not is_int:
            all_ints = False
    return True


def _label_classes(mode: PythonObject, values: PythonObject, context: PythonObject) raises -> LabelFit:
    """`_label_classes`: numeric labels by a device sort and run scan
    (`_fit_categories`), anything else by the ORDER RULE (sorted_classes)."""
    var words = List[Float64]()
    var all_ints = False
    if not _numeric_label_words(values, words, all_ints, True, context) or len(words) == 0:
        var ordered = _sorted_classes(values, context)
        return LabelFit(ordered.classes, PythonObject(None), PythonObject(None))
    var m = len(words)
    var arr = _f32_matrix(words, m, 1, context)
    var cats = _fit_categories(Python.tuple(0, arr, PythonObject(None), m, 1), mode, _entries(mode, context), context)[0]
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


def _label_codes(mut p: EncProg, words: List[Float64], cats: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    """`_label_codes`: each label's index among `cats` (or -1) from its
    float64 value (the words `_numeric_label_words` read) rounded to
    float32. Returns (codes, unknown count)."""
    var n = len(words)
    var arr = _f32_matrix(words, n, 1, context)
    return _codes(p, Python.tuple(0, arr, PythonObject(None), n, 1), Python.list(cats), context, False,
                  PythonObject(None))


def _object_index(classes: PythonObject) raises -> PythonObject:
    # {c: i for i, c in enumerate(classes)}: a duplicated class keeps its last index.
    var index = Python.dict()
    for i in range(len(classes)):  # driver-loop: over the fitted classes (one dict entry each)
        py_setitem(index, classes[i], PythonObject(i))
    return index


def _object_codes(mut p: EncProg, classes: PythonObject, values: PythonObject, context: PythonObject) raises -> Int:
    # pr.put_list([index.get(v, -1) for v in values]). Lane
    # p2-labels-lists-arrays: one borrowed read and one C dict probe per
    # label (p3_dict_codes) into int32 words, widened natively; the Python
    # `index.get` call per label is gone.
    var index = _object_index(classes)
    var listed = values
    if py_type(listed) is not _builtins().list and py_type(listed) is not _builtins().tuple:
        listed = py_call(_builtins().list, values)
    var n = len(listed)
    var words = List[Int32](unsafe_uninit_length=n)
    p3_dict_codes(index, listed, Int(words.unsafe_ptr()), -1)
    var codes = label_codes_i32_to_f64(Int(words.unsafe_ptr()), n)
    _ = words^  # keep the code words alive through the widening above
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


def _class_words(ran: PythonObject, offset: Int, n: Int, ints: Bool, context: PythonObject) raises -> PythonObject:
    # `_class_words`: the n 64-bit class words as an int64 or float64 Array.
    if ints:
        return _take_words(ran, offset, n, Python.tuple(n), "q", "<i8", 8, context)
    return _take_words(ran, offset, n, Python.tuple(n), "d", "<f8", 8, context)


def _int64_codes(values: PythonObject) raises -> List[Int]:
    """`_user_objects.call("ints", values)`: int(v) of each value, refused
    with OverflowError outside int64."""
    var b = _builtins()
    var lo = PythonObject(-9223372036854775807 - 1)
    var hi = PythonObject(9223372036854775807)
    var out = List[Int]()
    for i in range(len(values)):  # list-input: one int() per entry of the caller's Python code list (an integer code array takes _code_buffer)
        var x = py_call(b.int, values[i])
        if Bool(py=x < lo) or Bool(py=hi < x):
            _raise_python(b.OverflowError, "Python int too large to convert to C long")
        out.append(Int(py=x))
    return out^


def _class_seq(classes: PythonObject, list_only: Bool) raises -> PythonObject:
    """The fitted classes as an exact list (a tuple too unless `list_only`),
    for the native readers that index an exact list or tuple. `_classes` is a
    list from `_set_fit`; a state restored from an archive may hold another
    sequence (an Array), which the former `_take_classes` indexed as is: one
    `list()` of the k classes, once per call."""
    var b = _builtins()
    var cls = py_type(classes)
    if cls is b.list or (not list_only and cls is b.tuple):
        return classes
    return py_call(b.list, classes)


def _class_list(classes: PythonObject, codes: List[Int], context: PythonObject) raises -> PythonObject:
    """`_classes_array([classes[i] for i in codes])` with no Python call per
    label: the codes range-checked natively (a negative index from the end,
    IndexError out of range, as the list indexing did), the result kind
    (`labels_label_kind` of the taken labels) read from the USED classes,
    each class read once; int / real classes gathered as int64 / float64
    words into ONE Array, any other class objects placed by ONE presized
    list (`p3_decode_list`, new references filled in C)."""
    var b = _builtins()
    var k = len(classes)
    var n = len(codes)
    if py_type(classes) is not b.list and py_type(classes) is not b.tuple:
        raise Error("TypeError: class labels require an exact native list or tuple")
    var is_tuple = py_type(classes) is b.tuple
    var slots = List[Int64](length=n, fill=0)
    var used = List[Bool](length=k, fill=False)
    for t in range(n):  # native data loop (not a driver loop): native range check of the code words (no CPython call)
        var i = codes[t]
        var j = i + k if i < 0 else i
        if j < 0 or j >= k:
            _raise_python(b.IndexError, "list index out of range")
        slots[t] = Int64(j)
        used[j] = True
    var all_int = True
    var all_real = True
    var kinds = LabelTypeKinds()
    for j in range(k):  # driver-loop: over the fitted classes (one kind read each)
        if used[j]:
            var kind = kinds.admit_raw(p3_item(classes, j, is_tuple), context)
            all_int = all_int and kind == 3
            all_real = all_real and (kind == 3 or kind == 4)
    _ = kinds^
    if all_int:
        var words = List[Int64](length=k, fill=0)
        for j in range(k):  # driver-loop: over the fitted classes (int(c) of each used one)
            if used[j]:
                words[j] = Int64(Int(py=py_call(b.int, classes[j])))
        var out = admitted_vector(n, "<i8", _out(context))
        var dest = I64P(unsafe_from_address=labels_address(out))
        for t in range(n):  # native data loop (not a driver loop): native gather of the class words (no CPython call)
            dest[t] = words[Int(slots[t])]
        _ = out  # keep the Python object alive through the native writes above
        return out
    if all_real:
        var reals = List[Float64](length=k, fill=0.0)
        for j in range(k):  # driver-loop: over the fitted classes (float(c) of each used one)
            if used[j]:
                reals[j] = Float64(py=py_call(b.float, classes[j]))
        var out = admitted_vector(n, "<f8", _out(context))
        var dest = F64P(unsafe_from_address=labels_address(out))
        for t in range(n):  # native data loop (not a driver loop): native gather of the class words (no CPython call)
            dest[t] = reals[Int(slots[t])]
        _ = out  # keep the Python object alive through the native writes above
        return out
    var placed = p3_decode_list(classes, Int(slots.unsafe_ptr()), 8, n)
    _ = slots  # read by the native fill above
    return placed


# ---------------------------------------------------------------- 'U' label vectors
# A NumPy str label vector crosses as ONE buffer export (bindings/
# boundary_read.mojo `text_buffer_view`): its code points are read natively
# and only the distinct classes (fit) or the first five unseen labels (a
# refusal) become Python str objects. `tolist()` made one str per row and the
# lookups then read each back.

>>>

def _text_labels(y: PythonObject, context: PythonObject, mut table: StrTable) raises -> Bool:
    """A one-dimensional, non-empty NumPy 'U' label vector read ONCE into
    `table` (every label's code points, trailing NULs dropped, which is the
    str `tolist()` gives). False for anything else (a list, another dtype, an
    empty or multi-dimensional array), which keeps the label input step."""
    var ndarray = py_get(context, "ndarray")
    if ndarray is None or not (py_type(y) is ndarray):
        return False
    if String(py=y.dtype.kind) != "U" or Int(py=y.ndim) != 1 or Int(py=y.size) == 0:
        return False
    if String(py=y.dtype.byteorder) == ">":
        return False  # big-endian UCS4: the label input step reads it
    # The label input step's admission of y, before its read.
    classical_admit_value(y, context)
    var view = text_buffer_view(y)
    if view.kind != BOUNDARY_KIND_UCS4:
        return False
    _ = read_text_rows_into(view, table.points, table.starts, table.lengths)
    view.release()
    return True


def _text_fit(table: StrTable, context: PythonObject) raises -> LabelFit:
    """`_label_classes` of a 'U' label vector and its codes: the ORDER RULE
    for str labels (the distinct values in code-point order), ONE str per
    distinct class, each label's int32 code written natively. The labels are
    hashed across the host pool and only the distinct classes are sorted
    (bindings/str_labels_native.mojo `str_unique_into`); the former route
    merge-sorted all n labels on one thread."""
    var n = len(table.starts)
    var codes = classical_wrap_array(native_array_zeros("i", n), Python.tuple(n), "<i4", "C", _out(context))
    var firsts = str_unique_into(table, n, Int(py=classical_address(codes)))
    var k = len(firsts)
    _ = codes  # keep the Python object alive through the native writes above
    var classes = Python.list()
    var base = Int(table.points.unsafe_ptr())
    for c in range(k):  # driver-loop: over the distinct classes (one str each)
        var first = firsts[c]
        _ = py_call(classes.append, pack_ucs4_str(base + table.starts[first] * 4, table.lengths[first]))
    _ = table  # the code points are read by the packs above
    return LabelFit(classes, PythonObject(None), codes)


def _text_codes(classes: PythonObject, table: StrTable, as_float: Bool, context: PythonObject) raises -> PythonObject:
    """`_str_codes(classes, y.tolist(), as_float)` of a 'U' label vector:
    (codes Array, the first five unseen labels, the unseen count), each
    label's int32 (float32 with `as_float`) index among the str classes or
    -1 (a duplicated class keeps its LAST index, as the dict comprehension
    does); None when a class is not a str (the label input step answers).
    The classes are read once each; the labels never become Python objects
    (only the first five unseen ones, for the refusal text). The lookups run
    across the host pool."""
    var k = len(classes)
    if k == 0:
        return PythonObject(None)
    # round2-fixes-2: the classes read by the one str-label reader and the
    # labels looked up by the one shared lookup (bindings/str_labels_native.mojo
    # `_read`, `str_codes_into`, host pool with the GIL released); this file
    # kept its own copy of both (perf audits 2-b and 3).
    var known = StrTable()
    if not _read(_class_seq(classes, True), known):
        return PythonObject(None)
    var lookup_classes = str_classes(known, k)
    var n = len(table.starts)
    var codes = classical_wrap_array(native_array_zeros("f" if as_float else "i", n), Python.tuple(n),
                                     "<f4" if as_float else "<i4", "C", _out(context))
    var unseen = str_codes_into(table, known, lookup_classes, Int(py=classical_address(codes)), as_float)
    _ = codes  # keep the Python object alive through the native writes above
    _ = known
    _ = lookup_classes
    var first = Python.list()
    var base = Int(table.points.unsafe_ptr())
    for u in range(len(unseen.rows)):  # driver-loop: over the first five unseen labels
        var row = unseen.rows[u]
        _ = py_call(first.append, pack_ucs4_str(base + table.starts[row] * 4, table.lengths[row]))
    _ = table
    return Python.tuple(codes, first, unseen.count)


def _code_buffer(y: PythonObject, context: PythonObject, mut codes: List[Int]) raises -> Bool:
    """An integer or bool NumPy code array (any rank, C order) read ONCE
    into `codes` (the `int(v)` of each entry `_int64_codes` read from its
    `tolist()`). False for anything else, including uint64 storage (whose
    words past int64 the list path refuses with OverflowError)."""
    var ndarray = py_get(context, "ndarray")
    if ndarray is None or not (py_type(y) is ndarray):
        return False
    var kind = String(py=y.dtype.kind)
    if kind != "i" and kind != "u" and kind != "b":
        return False
    if String(py=y.dtype.byteorder) == ">" or Int(py=y.ndim) == 0:
        return False  # big-endian words or a 0-d array: the list path reads (or refuses) them
    classical_admit_value(y, context)
    var view = admitted_buffer_view(y)
    if view.kind == BOUNDARY_KIND_UINT and view.itemsize == 8:
        return False
    var n = view.count()
    # PERF FIX (perf-audit-3, 2026-11-11): gathered straight into `codes`
    # (one memcpy for C-order int64, one widening pass otherwise), not
    # `as_i64_c` + a zero-filled list + memcpy (up to three data passes).
    codes = List[Int](unsafe_uninit_length=n)
    gather_i64(view.layout, view.kind, Int(codes.unsafe_ptr()))
    view.release()
    return True


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
    # y in its own dtype, C order (admit_c_view; lane a1b: was labels_view,
    # whose storage-order read of an F-ordered y needed a transpose below).
    var view = admit_c_view(y, "y", context)
    var size = Int(py=view.size)
    var rows = Int(py=view.shape[0])
    var cols = Int(py=view.shape[1])
    # `as_f64_c(y, ndim=2)`: float64 in C order
    var wide = admitted_vector(size, "<f8", _out(context))
    labels_cast(Int(py=classical_address(view)), admitted_dtype(view, context), labels_address(wide), "<f8", size,
                context)
    _ = view  # keep the Python object alive through the native call above
    var addr = labels_address(wide)
    if labels_reduce_stat(context, addr, 1, size, 4).i == 0:
        raise Error(REFUSE_MULTIOUTPUT)
    var lo = labels_reduce_stat(context, addr, 1, size, 0).f
    var hi = labels_reduce_stat(context, addr, 1, size, 1).f
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
    var out = admitted_array(Python.tuple(rows, cols), "<f4", _out(context))
    # The words are in C order (admit_c_view): one narrowing, no transpose.
    # TOMBSTONE: prep_labels _multilabel_buffer py_call into the base binding's transpose_f32 (the F-order y arm) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
    labels_cast(addr, "<f8", labels_address(out), "<f4", size, context)
    _ = view
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
    for i in range(len(y)):  # list-input: one type check per row of the caller's Python list-of-lists y (no buffer form)
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
    var text = StrTable()
    if _text_labels(y, context, text):
        _set_fit(fields, _text_fit(text, context), context)
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
    var text = StrTable()
    if _text_labels(y, context, text):
        # A 'U' label vector: the classes and the codes from ONE read.
        var fields = py_fields(model)
        py_setitem(fields, "numeric_mode_", _default_mode(context))
        var fit = _text_fit(text, context)
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
    if cats is None:
        var text = StrTable()
        if _text_labels(y, context, text):
            var looked = _text_codes(classes, text, False, context)
            if looked is not None:
                if Int(py=looked[2]) != 0:
                    raise Error(String(UNSEEN) + ": " + _text(looked[1]))
                return looked[0]
    var values = _flatten_labels(y, context)
    var n = len(values)
    if n == 0:
        return classical_wrap_array(native_array_empty("i"), Python.tuple(0), "<i4", "C", _out(context))
    var words = List[Float64]()
    var unused_ints = False
    if cats is None or not _numeric_label_words(values, words, unused_ints, False, context):
        var native = _str_codes(classes, values, False, context)
        if native is not None:
            if Int(py=native[2]) != 0:
                raise Error(String(UNSEEN) + ": " + _text(native[1]))
            return native[0]
        # One borrowed read and one C dict probe per label into the result
        # (p3_dict_codes, as `_object_codes`), -1 for an unseen label; the
        # refusal names the first five unseen labels in label order. No
        # Python `in` / subscript call per label.
        var index = _object_index(classes)
        var store = native_array_zeros("i", n)
        var codes = classical_wrap_array(store, Python.tuple(n), "<i4", "C", _out(context))
        var address = Int(py=classical_address(codes))
        p3_dict_codes(index, values, address, -1)
        var dest = I32P(unsafe_from_address=address)
        var unseen = List[Int](length=5, fill=0)
        var count = 0
        for i in range(n):  # native data loop (not a driver loop): native scan for unseen labels (no CPython call)
            if dest[i] < 0:
                if count < 5:
                    unseen[count] = i
                count += 1
        _ = codes  # keep the Python object alive through the native reads above
        if count > 0:
            var first = Python.list()
            for u in range(min(count, 5)):  # driver-loop: over the first five unseen labels
                _ = py_call(first.append, values[unseen[u]])
            raise Error(String(UNSEEN) + ": " + _text(first))
        return codes
    var entries = _entries(mode, context)
    var p = EncProg()
    var lc = _label_codes(p, words, cats, context)
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
    # (an integer code array by ONE buffer read, a list by the label input step)
    var codes = List[Int]()
    if not _code_buffer(y, context, codes):
        codes = _int64_codes(_flatten_labels(y, context))
    var k = len(classes)
    for i in range(len(codes)):  # native data loop (not a driver loop): native range check of the code words (no CPython call)
        if codes[i] < 0 or codes[i] >= k:
            raise Error(UNSEEN)
    return _class_list(classes, codes, context)


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
            and _truth(py_call(native_loaded_module("operator").lt, neg, pos))):
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
    var entries = _entries(mode, context)
    if cats is None:
        var text = StrTable()
        if _text_labels(y, context, text):
            # A 'U' label vector: one buffer read, the float codes natively.
            var looked = _text_codes(classes, text, True, context)
            if looked is not None:
                var tn = len(text.starts)
                var tp = EncProg()
                var tc = tp.put(looked[0], tn)
                return _lb_binarize(tp, tc, tn, K, W, neg, pos, entries, context)
    var values = _flatten_labels(y, context)
    var n = len(values)
    var p = EncProg()
    var codes = NONE
    var words = List[Float64]()
    var unused_ints = False
    if cats is None or not _numeric_label_words(values, words, unused_ints, False, context):
        var native = _str_codes(classes, values, True, context) if n > 0 else PythonObject(None)
        if native is not None:
            codes = p.put(native[0], n)
        else:
            codes = _object_codes(p, classes, values, context)
    else:
        codes = _label_codes(p, words, cats, context)[0]
    return _lb_binarize(p, codes, n, K, W, neg, pos, entries, context)


def _lb_binarize(mut p: EncProg, var codes: Int, n: Int, K: Int, W: Int, neg: Int, pos: Int, entries: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    # The dense label_binarize output of n float codes (one class: every code absent).
    if K == 1:
        codes = p.put_floats(List[Float64](length=n, fill=-1.0), context)
    var out = p.output(n * W, entries)
    _stage(p.program, OP_LABEL_BINARIZE, n * W, codes, n, K, 1 if K <= 2 else 0, neg, pos, W, out)
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
        # The code words read natively from the program's float32 result
        # (its address read once, not one `ran[0][1]` lookup per row).
        var idx = List[Int](length=n, fill=0)
        if K != 1:
            var words = F32P(unsafe_from_address=Int(py=ran_obj[0][1]))
            for i in range(n):  # native data loop (not a driver loop): native read of one code per row (no CPython call)
                idx[i] = Int(words[codes + i])
            _ = ran_obj  # keep the result alive through the native reads above
        return _class_list(classes, idx, context)
    # numeric classes: each code's class gathered on the device; one class: both codes name it
    var ints = labels_label_kind(classes, context) == "int"
    var table = cats
    if K <= 1:
        var both = List[Float64]()
        var source = F32P(unsafe_from_address=Int(py=classical_address(cats)))
        _ = cats  # keep the Python object alive through the native call above
        for _ in range(2):  # driver-loop: over the two codes that name the one class
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
    for i in range(nrows):  # list-input: one read per row of the caller's Python label rows (ragged, no buffer form)
        var row = y[i]
        var t = py_type(row)
        if t is not b.list and t is not b.tuple and t is not b.set and t is not b.frozenset:
            return None
        _ = py_call(rows.append, row if (t is b.list or t is b.tuple) else py_call(b.list, row))
    var total = 0
    for i in range(nrows):  # list-input: one read per row of the caller's Python label rows (ragged, no buffer form)
        var row = rows[i]
        for j in range(len(row)):  # list-input: one read per label of a ragged Python label set
            if py_type(row[j]) is not b.int:
                return None
        total += len(row)
    if total == 0 or total > LABEL_WORDS_MAX or 2 * total > LABEL_WORDS_MAX:
        return None
    # Each exact int is read with PyLong_AsLongLong (an int outside int64
    # sets OverflowError, which is cleared and takes the label input step):
    # no Python rich comparison or Int conversion per label (p3-prep).
    ref cpy = Python().cpython()
    var table = cpython_table()
    var nw = 2 * total
    var words = classical_wrap_array(native_array_zeros("i", nw), Python.tuple(nw), "<i4", "C", _out(context))
    var dest = I64P(unsafe_from_address=Int(py=classical_address(words)))
    var offs = classical_wrap_array(native_array_zeros("i", nrows + 1), Python.tuple(nrows + 1), "<i4", "C", _out(context))
    var od = I32P(unsafe_from_address=Int(py=classical_address(offs)))
    _ = words  # keep the Python object alive through the native call above
    _ = offs  # keep the Python object alive through the native call above
    var at = 0
    for i in range(nrows):  # list-input: one read per row of the caller's Python label rows (ragged, no buffer form)
        var row = rows[i]
        for j in range(len(row)):  # list-input: one read per label of a ragged Python label set
            var v = row[j]
            var word = table[].PyLong_AsLongLong(v._obj_ptr)
            if word == -1 and cpy.PyErr_Occurred():
                cpython_table()[].PyErr_Clear()
                return None
            dest[at] = word
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
    for i in range(len(rows)):  # list-input: one read per row of the caller's ragged label sets (no buffer form)
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
        for i in range(len(rows)):  # list-input: one read per row of the caller's ragged label sets (no buffer form)
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
    for i in range(n):  # list-input: one read per row of the caller's ragged label sets (no buffer form)
        var row = rows[i]
        for j in range(len(row)):  # list-input: one read per label of a ragged Python label set
            _ = py_call(flat.append, row[j])
            owner.append(Float64(i))
    var entries = _entries(fields["numeric_mode_"], context)
    var p = EncProg()
    var m = len(flat)
    if m == 0:
        var empty = p.alloc(n * max(K, 1))
        var ran_empty = p.run(entries, context)
        return _take_i32(ran_empty, empty, n, K, context)
    var codes = NONE
    var words = List[Float64]()
    var unused_ints = False
    if cats is None or not _numeric_label_words(flat, words, unused_ints, False, context):
        var native = _str_codes(classes, flat, True, context)
        if native is not None:
            codes = p.put(native[0], m)
        else:
            codes = _object_codes(p, classes, flat, context)
    else:
        codes = _label_codes(p, words, cats, context)[0]
    var ro = p.put_floats(owner, context)
    var out = p.output(n * max(K, 1), entries)
    _stage(p.program, OP_SCATTER_ONES, m, codes, ro, K, out)
    var ran = p.run(entries, context)
    return _take_i32(ran, out, n, K, context)


def _mlb_inverse_direct(classes: PythonObject, k: Int, yt: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_mlb_inverse` of a C-ordered 2-D numeric indicator (a mojolearn Array
    or an ndarray, int32 / int64 / float32 / float64, (n, k) wide): the set
    cells are found by one native scan of the buffer and only the picked
    classes and the row tuples become Python objects (p3-prep). None for any
    other input (the general walk below). A nonzero or NaN cell is set, as
    the Python truth test reads it."""
    var b = _builtins()
    var ndarray = py_get(context, "ndarray")
    var is_array = py_type(yt) is context["array_class"]
    if not is_array and (ndarray is None or py_type(yt) is not ndarray):
        return None
    if Int(py=yt.ndim) != 2 or Int(py=yt.shape[1]) != k:
        return None
    var dtype = String(yt.dtype) if is_array else String(yt.dtype.str)
    var c_order = (String(yt.order) == "C") if is_array else Bool(py=yt.flags.c_contiguous)
    if not c_order:
        return None
    var code = 0
    if dtype == "<i4":
        code = 1
    elif dtype == "<i8":
        code = 2
    elif dtype == "<f4":
        code = 3
    elif dtype == "<f8":
        code = 4
    else:
        return None
    var n = Int(py=yt.shape[0])
    var address = Int(py=classical_address(yt))
    var out = Python.list()
    for i in range(n):  # list-output: one tuple per row of the public list-of-tuples result
        var picked = Python.list()
        for j in range(k):  # list-output: one native read per indicator cell; Python objects only for the hits
            var at = i * k + j
            var hit: Bool
            if code == 1:
                hit = I32P(unsafe_from_address=address)[at] != 0
            elif code == 2:
                hit = I64P(unsafe_from_address=address)[at] != 0
            elif code == 3:
                hit = F32P(unsafe_from_address=address)[at] != 0
            else:
                hit = F64P(unsafe_from_address=address)[at] != 0
            if hit:
                _ = py_call(picked.append, classes[j])
        _ = yt  # keep the Python object alive through the native reads above
        _ = py_call(out.append, py_call(b.tuple, picked))
    return out


def _mlb_inverse(model: PythonObject, yt: PythonObject, context: PythonObject) raises -> PythonObject:
    # One tuple of labels per row: the user list of Python label tuples.
    var b = _builtins()
    var fields = py_fields(model)
    _check_fitted(fields, "MultiLabelBinarizer")
    var classes = fields["_classes"]
    var k = len(classes)
    var direct = _mlb_inverse_direct(classes, k, yt, context)
    if direct is not None:
        return direct
    var rows = py_call(yt.tolist) if Bool(py=py_call(b.hasattr, yt, "tolist")) else py_call(b.list, yt)
    var out = Python.list()
    for i in range(len(rows)):  # list-output: one tuple per row of the public list-of-tuples result
        var row = py_call(b.list, rows[i])
        var picked = Python.list()
        for j in range(len(row)):  # list-output: one tuple per row of the public list-of-tuples result
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
