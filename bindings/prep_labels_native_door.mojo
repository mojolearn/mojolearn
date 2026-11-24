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
#
# Door split (lane dw-prep-encode, 2026-11-21): every program is UNPACK here
# (the fields, the route's image and capabilities, the caller's labels: a
# buffer becomes an owner of the door's `EncProg`, list labels become float
# words or codes through the label table and the typed forms), RUN in
# bindings/prep_labels_prog_run.mojo (typed requests, the stages on the typed
# `EncProgram`, no Python object) and the typed run, then PACK here (arena
# words to the result). No builder reads a field while it stages. The
# list-route fit of numeric labels runs the encode owner's typed category
# fit over program-owned label words (`_fit_category_words`).
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_array_storage_door import native_array_zeros, native_array_empty, native_array_from_bytes, native_array_unfilled
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains, py_setattr
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.classical_arrays_door import classical_address, classical_wrap_array
from bindings.classical_labels_door import (
    labels_address, labels_label_kind, labels_cast,
    labels_reduce_stat, _numeric_label_view, _flatten_labels, _sorted_classes)
from bindings.admitted_input_unpack import admitted_array, admitted_vector, admit_c_view, admitted_dtype
from bindings.scaler_lifecycle_native_door import scaler_entries
from bindings.prep_entry_cache_unpack import prep_entry_word
from bindings.x_prep_native_owner_door import xprep_values, xprep_address
from bindings.prep_encode_native_door import (
    EncProg, NONE, enc_caps, _take1, _word, _iword, _fit_category_words, _default_mode, _truth)
from bindings.prep_encode_prog_run import OUT_MARK
from bindings.prep_labels_prog_run import (
    LabelWords, LabelCats, LabelCodes, LabelUnbinarize, LAB_CODES_BUFFER, LAB_CODES_WORDS, LAB_CODES_LOOKUP,
    LAB_Y_MULTILABEL, LAB_Y_MULTICLASS, LAB_Y_BINARY,
    lab_fit_stage, lab_transform_stage, lab_list_transform_stage, lab_inverse_stage, lab_indicator_stage,
    lab_binarize_buffer_stage, lab_binarize_codes_stage, lab_unbinarize_stage, lab_row_ones_stage,
    lab_scatter_ones_stage)
from bindings.x_prep_native_owner_run import XprepCaps
from bindings.scaler_native_door import _input
from bindings.str_labels_native_door import str_lookup, _read, str_unique_into, str_codes_into
from bindings.str_labels_native_run import StrTable, str_classes
from bindings.boundary_read_door import text_buffer_view, read_text_rows_into, pack_ucs4_str, admitted_buffer_view
from core.boundary_layout import BOUNDARY_KIND_UCS4, BOUNDARY_KIND_UINT, gather_i64
from bindings.classical_pandas_door import classical_admit_value
from bindings.label_admission_native_unpack import LabelTypeKinds
from bindings.user_objects_native_door import _Env, _f32_exact, _indicator, _new_list, _list_put
from bindings.py_cpython_table import cpython_table
from bindings.p3_labels_metrics_native_door import p3_item, p3_type_word, p3_dict_codes, p3_decode_list
from core.label_words_impl import label_codes_i32_to_f64
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_INVERSE_TRANSFORM, PREP_OPN_TRANSFORM,
)
from bindings.py_balanced import py_item

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

# The stage codes and the presence route's table size are the typed
# builders' (bindings/prep_labels_prog_run.mojo).
# A label buffer of more int32 words takes the label input step.
comptime LABEL_WORDS_MAX = 1073741824
# `_take_words`: a mapped program output of at least this many words is
# returned as a view of the mapping, a smaller one is copied (0.8.55's
# _VIEW_MIN_WORDS = 2^20: a view retains the whole arena mapping, so a small
# result never pins a large arena). The arena is a lazily mapped anonymous
# mapping and the inputs of 2^20+ words go up direct from the caller's buffer
# (never touching it), so the pages a view retains are mostly the downloaded
# output itself.
# TOMBSTONE: MOJOLEARN_PREP_FAST_TAKE_VIEW_ALL (NOISE: views from 16384 words, 12 cells median 1.00, 0.92..1.04; A/B nvc-ab-20261124T095727Z-92fd9f NVIDIA L40S) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_PREP_FAST_TAKE_VIEW_ALL.ab-outcome-2.patch; record in docs/TOMBSTONES.md.
comptime TAKE_VIEW_MIN_WORDS = 1048576

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
    var nk = prep_keys()
    # `_check_fitted`: hasattr(self, "_classes").
    if not py_contains(fields, nk[].k__classes):
        raise Error("RuntimeError: mojolearn: this " + name + " instance is not fitted yet")


def _text(value: PythonObject) raises -> String:
    return String(py_call(_builtins().str, value))


def _out(context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    return py_item(context, nk[].k_output_context)


def _take_words(ran: PythonObject, offset: Int, count: Int, shape: PythonObject, code: String, typestr: String,
                width: Int, context: PythonObject) raises -> PythonObject:
    """`_Prog.get_i32(off, shape)` (code "i", 4-byte words) and `_class_words`
    (code "q" / "d", 8-byte words): a copy of the program's words at
    `offset` (the output region at OUT_MARK) as an Array of `typestr`."""
    var nk = prep_keys()
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
    # TOMBSTONE: MOJOLEARN_XPREP_R3_OFF=view (label owner door; LOSER: the copy 1.24-1.67x slower on 6 of 8 binarizer cells, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts; code recoverable at 0c65d61fed.
    # Restore: git apply experiments/removed/MOJOLEARN_XPREP_R3_OFF-view-labels.patch; record in docs/TOMBSTONES.md.
    if py_type(store) is b.memoryview and words >= TAKE_VIEW_MIN_WORDS:
        # fb-metrics (2026-11-17; full board d95f805cce vs 0.8.55: LabelBinarizer
        # and MultiLabelBinarizer fit/transform 1.6-2.2x): a large mapped
        # output is returned as a view that retains its slice and mapping, as
        # 0.8.55's _Prog._take and the encoder owner's xprep_take
        # (bindings/x_prep_native_owner_door.mojo) do, not copied word by
        # word into a fresh store (a full n x K copy and its first touch per
        # call). Same bytes. VERDICT 2026-11-18 (A/B ab2, owed row
        # xprep-r3-off-view-binarizers, NVIDIA L40S host c6dd9ebea3e3, wheel
        # d652a4b0a1, base and arm back to back, one run per arm): the copy
        # (MOJOLEARN_XPREP_R3_OFF=view) was slower on every cell, arm/base
        # geomean 1.40: label-binarizer taxi fit 815 -> 1362 ms (1.67), infer
        # 78.3 -> 129 (1.65), istella fit 56.3 -> 92.4 (1.64), istella infer
        # 1.06; multilabel-binarizer taxi fit 430 -> 665 (1.55), infer 38.7 ->
        # 59.0 (1.53), istella fit 220 -> 274 (1.24), istella infer 1.07.
        # Digests and quality equal on all 8 cells. The view is the one form
        # here and the switch read is gone (one getenv per take); the
        # encoder owner's `view` choice (xprep_take) was not on these rows
        # and stays. Limits: NVIDIA only (AGENTS.md part 8).
        var typed = py_call(py_call(part.cast, "B").cast, code)
        var ndim = Int(py=py_call(b.len, shape))
        var result = py_call(b.object.__new__, py_item(context, nk[].k_array_class))
        py_setattr(result, "_store", PythonObject(None))
        py_setattr(result, "_base", store)
        py_setattr(result, "_pin", PythonObject(None))
        py_setattr(result, "_mv", typed)
        py_setattr(result, "_addr", PythonObject(xprep_address(typed)))
        py_setattr(result, "_readonly", PythonObject(False))
        py_setattr(result, "shape", shape)
        py_setattr(result, "dtype", PythonObject(typestr))
        py_setattr(result, "ndim", PythonObject(ndim))
        py_setattr(result, "size", PythonObject(count))
        py_setattr(result, "nbytes", PythonObject(count * width))
        py_setattr(result, "itemsize", PythonObject(width))
        py_setattr(result, "strides", Python.tuple(width) if ndim == 1 else Python.tuple(Int(py=shape[1]) * width, width))
        py_setattr(result, "order", PythonObject("C"))
        return result
    var data = native_array_from_bytes(code, py_call(part.cast, "B"))
    return classical_wrap_array(data, shape, typestr, "C", _out(context))


def _take_i32(ran: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_Prog.get_i32(off, rows)` (cols < 0) or `get_i32(off, (rows, cols))`.
    if cols < 0:
        return _take_words(ran, offset, rows, Python.tuple(rows), "i", "<i4", 4, context)
    return _take_words(ran, offset, rows * cols, Python.tuple(rows, cols), "i", "<i4", 4, context)


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
    """`_LabelBuf`: a numeric label vector (its bytes read as `nw` raw int32
    words: the caller's admitted view, or the flat MultiLabelBinarizer words),
    its lab_load kind, its label count, its word count and whether it is a
    float dtype."""
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
    # fb12-rows (2026-11-17): the admitted C-order view itself is the input
    # record (its bytes read as nw int32 words), as 0.8.55's `_label_buffer`
    # (a no-copy int32 view of the label buffer). The x_prep run reads a host
    # record's extent from the record's word count (nw), not `view.size`
    # (bindings/x_prep_native_owner_door.mojo `xprep_run_records`), so an
    # 8-byte view is safe. Was: a fresh nw-word store and a memcpy of the
    # labels per fit, transform and inverse call.
    return LabelBuf(spec[0], view, n, nw, spec[0] == 0 or spec[0] == 4)


def _own_labels(mut p: EncProg, lb: LabelBuf) raises -> LabelWords:
    # UNPACK: the label words' buffer becomes an owner of the program; the
    # typed builders read the vector as these plain values.
    return LabelWords(lb.kind, lb.n, lb.nw, p.own(lb.words, lb.nw))


def _no_cats() -> LabelCats:
    return LabelCats(NONE, 0, 0)


def _own_cats(mut p: EncProg, cats: PythonObject, context: PythonObject) raises -> LabelCats:
    # UNPACK of the fitted numeric classes (`EncProg.put_array`): the
    # admitted flat float32 vector becomes an owner of the program.
    var flat = _input(cats, 1, False, context)
    var words = Int(py=flat.size)
    return LabelCats(p.own(flat, words), Int(py=cats.size), words)


@fieldwise_init
struct LabelFit(Copyable, Movable):
    """(classes list, device categories Array or None, int32 codes Array or
    None, `classes_` Array or None). `array` is set by the device route,
    which writes `classes_` from the class words natively (`_fit_array`);
    None makes `_classes_array` convert the classes list."""
    var classes: PythonObject
    var cats: PythonObject
    var codes: PythonObject
    var array: PythonObject


def _label_classes_of(ran: PythonObject, lb: LabelBuf, u: Int, k: Int, context: PythonObject) raises -> Optional[LabelFit]:
    """`_label_classes_of`: (classes, cats) read back after the run, or None
    when a label has no exact float32 word (NaN sorts last)."""
    # fb15-sync-counts (2026-11-18): the arena address is read once (was two
    # Python subscripts per class word in `_word`), and `classes_` is written
    # here from the same words (int64 / float64, the values the list holds),
    # not rebuilt by `_classes_array` (a kind scan of the list and one
    # int() / float() Python call per class). Same values, same dtypes.
    var words = F32P(unsafe_from_address=Int(py=ran[0][1]))
    var K = Int(words[k])
    if K == 0:
        return None
    var last = words[u + K - 1]
    if last != last:
        return None
    var cats = _take1(ran, u, K, context)
    var classes = Python.list()
    var array = admitted_vector(K, "<f8" if lb.is_float else "<i8", _out(context))
    var dest = labels_address(array)
    var dest_f = F64P(unsafe_from_address=dest)
    var dest_i = I64P(unsafe_from_address=dest)
    for i in range(K):  # driver-loop: over the distinct class values the device found
        var w = words[u + i]
        if lb.is_float:
            dest_f[i] = Float64(w)
            _ = py_call(classes.append, PythonObject(Float64(w)))
        else:
            dest_i[i] = Int64(Int(w))
            _ = py_call(classes.append, PythonObject(Int(w)))
    _ = ran  # keep the arena alive through the reads above
    _ = array  # keep the Python object alive through the native writes above
    return LabelFit(classes, cats, PythonObject(None), array)


def _fit_array(fit: LabelFit, context: PythonObject) raises -> PythonObject:
    # `classes_`: the device route's native array, else `_classes_array`.
    if fit.array is not None:
        return fit.array
    return _classes_array(fit.classes, context)


def _label_present_enabled(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_label_present_enabled`: the binding's x_prep_label_present export is the switch.
    var m = String(mode)
    if m != "fast" and m != "identical":
        return False
    return prep_entry_word(entries, "x_prep_label_present") != 0  # read once per binding


def _label_fit_run(entries: PythonObject, caps: XprepCaps, lb: LabelBuf, present: Bool, codes: Bool,
                   context: PythonObject) raises -> Optional[LabelFit]:
    """One fit program of a numeric label buffer. `present`
    (`_label_fit_present`): exact small-integer distinct labels by presence
    flags, None (the general sort) when the device's BAD flag is set.
    Otherwise the sort and run scan; None when a label has no exact float32
    word (the label input step)."""
    # UNPACK: the label buffer.
    var p = EncProg()
    var w = _own_labels(p, lb)
    # RUN: the typed stages (bindings/prep_labels_prog_run.mojo), then the typed run.
    var plan = lab_fit_stage(p.prog, w, present, codes, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the classes and the codes from the arena.
    if present and _iword(ran, plan.bad) != 0:
        return None
    var got = _label_classes_of(ran, lb, plan.u, plan.k, context)
    if not got:
        return None
    var fit = got.value().copy()
    if codes:
        fit.codes = _take_i32(ran, plan.codes, lb.n, -1, context)
    return fit^


def _label_fit_device(model: PythonObject, mode: PythonObject, lb: LabelBuf, codes: Bool, context: PythonObject) raises -> Optional[LabelFit]:
    """`_label_fit_device`: one program, (classes, cats, int32 codes or None),
    or None for the label input step."""
    # UNPACK: the image of this estimator's route, its capabilities (once for
    # both programs; each run read them before) and the presence switch.
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    if _label_present_enabled(mode, entries):
        var got = _label_fit_run(entries, caps, lb, True, codes, context)
        if got:
            return got^
    return _label_fit_run(entries, caps, lb, False, codes, context)


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
    var nk = prep_keys()
    var env = _Env()
    var integral = py_item(context, nk[].k_labels_integral_type) if count_ints else PythonObject(None)
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


def _label_classes(model: PythonObject, mode: PythonObject, values: PythonObject, context: PythonObject) raises -> LabelFit:
    """`_label_classes`: numeric labels by a device sort and run scan
    (`_fit_category_words`), anything else by the ORDER RULE (sorted_classes)."""
    var words = List[Float64]()
    var all_ints = False
    if not _numeric_label_words(values, words, all_ints, True, context) or len(words) == 0:
        var ordered = _sorted_classes(values, context)
        return LabelFit(ordered.classes, PythonObject(None), PythonObject(None), PythonObject(None))
    # The label words are program-owned float32 words of the category fit
    # (no Array is built for the call), run on the estimator's image.
    var entries = scaler_entries(model, mode, context)
    var cats = _fit_category_words(words, mode, entries, enc_caps(model, entries), context)
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
    return LabelFit(classes, cats, PythonObject(None), PythonObject(None))


def _object_index(classes: PythonObject) raises -> PythonObject:
    # {c: i for i, c in enumerate(classes)}: a duplicated class keeps its last index.
    var index = Python.dict()
    for i in range(len(classes)):  # driver-loop: over the fitted classes (one dict entry each)
        py_setitem(index, classes[i], PythonObject(i))
    return index


def _object_code_words(classes: PythonObject, values: PythonObject) raises -> List[Float64]:
    # UNPACK of object labels: [index.get(v, -1) for v in values] as float
    # code words. Lane p2-labels-lists-arrays: one borrowed read and one C
    # dict probe per label (p3_dict_codes) into int32 words, widened
    # natively; no Python `index.get` call per label.
    var index = _object_index(classes)
    var listed = values
    if py_type(listed) is not _builtins().list and py_type(listed) is not _builtins().tuple:
        listed = py_call(_builtins().list, values)
    var n = len(listed)
    var words = List[Int32](unsafe_uninit_length=n)
    p3_dict_codes(index, listed, Int(words.unsafe_ptr()), -1)
    var codes = label_codes_i32_to_f64(Int(words.unsafe_ptr()), n)
    _ = words^  # keep the code words alive through the widening above
    return codes^


def _list_codes(mut p: EncProg, classes: PythonObject, cats: PythonObject, values: PythonObject, n: Int,
                context: PythonObject) raises -> LabelCodes:
    """UNPACK of the labels of a Python list (`values`, n of them) for a
    program that reads their float codes: numeric labels among numeric
    classes are looked up by the program (their float values and the fitted
    class words); str labels among str classes are a float32 code buffer
    (`_str_codes`, the label table); anything else is one C dict probe per
    label (`_object_code_words`)."""
    var words = List[Float64]()
    var unused_ints = False
    if cats is not None and _numeric_label_words(values, words, unused_ints, False, context):
        var known = _own_cats(p, cats, context)
        return LabelCodes(LAB_CODES_LOOKUP, n, NONE, words^, known^)
    var native = _str_codes(classes, values, True, context) if n > 0 else PythonObject(None)
    if native is not None:
        var owner = p.own(native[0], n)
        return LabelCodes(LAB_CODES_BUFFER, n, owner, List[Float64](), _no_cats())
    return LabelCodes(LAB_CODES_WORDS, n, NONE, _object_code_words(classes, values), _no_cats())


def _str_codes(classes: PythonObject, values: PythonObject, as_float: Bool, context: PythonObject) raises -> PythonObject:
    """`_labels.str_labels("lookup", list(classes), values, as_float)`: (codes
    Array, first missing labels, missing count), or None when a class is not a str."""
    var native = str_lookup(py_call(_builtins().list, classes), values, as_float)
    if native is None:
        return native
    var n = len(values)
    var codes = classical_wrap_array(native[0], Python.tuple(n), "<f4" if as_float else "<i4", "C", _out(context))
    return Python.tuple(codes, native[1], native[2])


def _class_words(ran: PythonObject, offset: Int, n: Int, ints: Bool, context: PythonObject) raises -> PythonObject:
    # `_class_words`: the n 64-bit class words as an int64 or float64 Array.
    if ints:
        return _take_words(ran, offset, n, Python.tuple(n), "q", "<i8", 8, context)
    return _take_words(ran, offset, n, Python.tuple(n), "d", "<f8", 8, context)


def _int64_codes(values: PythonObject) raises -> List[Int]:
    """`_user_objects.call("ints", values)`: int(v) of each value, refused
    with OverflowError outside int64."""
    var b = _builtins()
    # Lane list-input-loops (2026-11-12): an exact list or tuple is read
    # borrowed, an exact int by one PyLong_AsLongLongAndOverflow (the int64
    # range check the two Python `<` comparisons made); any other class is
    # int()-converted first, as before.
    var t = cpython_table()
    ref cpy = Python().cpython()
    var cls = py_type(values)
    var is_tuple = cls is b.tuple
    var exact = is_tuple or cls is b.list
    var int_word = Int(b.int._obj_ptr)
    var int_fn = b.int
    var overflow = List[Int32](length=1, fill=0)
    var n = len(values)
    var out = List[Int](capacity=n)
    for i in range(n):  # list-input: one borrowed read per entry of the caller's Python code list (an integer code array takes _code_buffer)
        var v = PythonObject(from_borrowed=p3_item(values, i, is_tuple)) if exact else values[i]
        var x = v if p3_type_word(v._obj_ptr) == int_word else py_call(int_fn, v)
        overflow[0] = 0
        var word = t[].PyLong_AsLongLongAndOverflow(x._obj_ptr, Int(overflow.unsafe_ptr()))
        if overflow[0] != 0:
            _raise_python(b.OverflowError, "Python int too large to convert to C long")
        if word == -1 and t[].PyErr_Occurred():
            var pending = cpy.unsafe_get_error()
            raise pending^
        out.append(Int(word))
    _ = overflow
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


def _class_list(fitted: PythonObject, codes: List[Int], context: PythonObject) raises -> PythonObject:
    """`_classes_array([classes[i] for i in codes])` with no Python call per
    label: the codes range-checked natively (a negative index from the end,
    IndexError out of range, as the list indexing did), the result kind
    (`labels_label_kind` of the taken labels) read from the USED classes,
    each class read once; int / real classes gathered as int64 / float64
    words into ONE Array, any other class objects placed by ONE presized
    list (`p3_decode_list`, new references filled in C). Classes that are
    not an exact list or tuple (an Array restored from an archive) are
    listed once (`_class_seq`), as the former `_take_classes` indexed them."""
    var b = _builtins()
    # round2-fixes-2: a restored non-list `_classes` was refused here with
    # "TypeError: class labels require an exact native list or tuple"
    # (perf audit 2-b, refusal 6).
    var classes = _class_seq(fitted, False)
    var k = len(classes)
    var n = len(codes)
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


def _text_labels(y: PythonObject, context: PythonObject, mut table: StrTable) raises -> Bool:
    """A one-dimensional, non-empty NumPy 'U' label vector read ONCE into
    `table` (every label's code points, trailing NULs dropped, which is the
    str `tolist()` gives). False for anything else (a list, another dtype, an
    empty or multi-dimensional array), which keeps the label input step."""
    var nk = prep_keys()
    var ndarray = py_get(context, nk[].k_ndarray)
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
    # lane fix-output-zero-fill (2026-11-14): unfilled; str_unique_into writes every label's code.
    var codes = classical_wrap_array(native_array_unfilled("i", n), Python.tuple(n), "<i4", "C", _out(context))
    var firsts = str_unique_into(table, n, Int(py=classical_address(codes)))
    var k = len(firsts)
    _ = codes  # keep the Python object alive through the native writes above
    var classes = Python.list()
    var base = Int(table.points.unsafe_ptr())
    for c in range(k):  # driver-loop: over the distinct classes (one str each)
        var first = firsts[c]
        _ = py_call(classes.append, pack_ucs4_str(base + table.starts[first] * 4, table.lengths[first]))
    _ = table  # the code points are read by the packs above
    return LabelFit(classes, PythonObject(None), codes, PythonObject(None))


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
    # lane fix-output-zero-fill (2026-11-14): unfilled; str_codes_into writes every label's index (-1 unseen).
    var codes = classical_wrap_array(native_array_unfilled("f" if as_float else "i", n), Python.tuple(n),
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
    var nk = prep_keys()
    var ndarray = py_get(context, nk[].k_ndarray)
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
    var nk = prep_keys()
    var b = _builtins()
    if Int(py=shape[0]) == 0 or Int(py=shape[1]) < 2:
        return None
    var dtype = py_call(b.getattr, y, "dtype", "")
    var ndarray = py_get(context, nk[].k_ndarray)
    if ndarray is not None and Bool(py=py_call(b.isinstance, y, ndarray)):
        # An ndarray's dtype kind is a C attribute: str(dtype) ran NumPy's
        # Python _dtype.__str__ / _name_get / issubdtype chain, 7 frames per
        # call (census d95f805cce, LabelBinarizer fit 14, transform 161).
        if String(py=dtype.kind) == "b":
            raise Error(REFUSE_MULTIOUTPUT)
    elif "bool" in _text(dtype):
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
    # Exact list / tuple rows by class word, read borrowed from an exact y;
    # any other row class takes isinstance (lane list-input-loops).
    var list_word = Int(b.list._obj_ptr)
    var tuple_word = Int(b.tuple._obj_ptr)
    var y_word = p3_type_word(y._obj_ptr)
    var y_exact = y_word == list_word or y_word == tuple_word
    for i in range(len(y)):  # list-input: one borrowed row read and class-word check per row of the caller's Python list-of-lists y (no buffer form)
        var row = PythonObject(from_borrowed=p3_item(y, i, y_word == tuple_word)) if y_exact else y[i]
        var w = p3_type_word(row._obj_ptr)
        if w != list_word and w != tuple_word and not Bool(py=py_call(b.isinstance, row, seq)):
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
    var nk = prep_keys()
    py_setitem(fields, nk[].k__classes, fit.classes)
    py_setitem(fields, nk[].k__cats, fit.cats)


def _fit_labels(model: PythonObject, fields: PythonObject, mode: PythonObject, y: PythonObject, context: PythonObject) raises -> LabelFit:
    # `_label_fit_device` of a numeric buffer, else `_label_classes(flatten_labels(y))`.
    # Returns the fit set on `fields` (its `classes_` by `_fit_array`).
    var lb = _label_buffer(y, context)
    if lb:
        var got = _label_fit_device(model, mode, lb.value(), False, context)
        if got:
            var fit = got.value().copy()
            _set_fit(fields, fit, context)
            return fit^
    var text = StrTable()
    if _text_labels(y, context, text):
        var fit = _text_fit(text, context)
        _set_fit(fields, fit, context)
        return fit^
    var fit = _label_classes(model, mode, _flatten_labels(y, context), context)
    _set_fit(fields, fit, context)
    return fit^


def _le_fit(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var fields = py_fields(model)
    var mode = _default_mode(context)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    var fit = _fit_labels(model, fields, mode, y, context)
    py_setitem(fields, nk[].k_classes_, _fit_array(fit, context))
    return model


def _le_fit_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # A numeric buffer: classes and codes in ONE program.
    var lb = _label_buffer(y, context)
    if lb:
        var fields = py_fields(model)
        var mode = _default_mode(context)
        py_setitem(fields, nk[].k_numeric_mode_, mode)
        var got = _label_fit_device(model, mode, lb.value(), True, context)
        if got:
            var fit = got.value().copy()
            _set_fit(fields, fit, context)
            py_setitem(fields, nk[].k_classes_, _fit_array(fit, context))
            return fit.codes
    var text = StrTable()
    if _text_labels(y, context, text):
        # A 'U' label vector: the classes and the codes from ONE read.
        var fields = py_fields(model)
        py_setitem(fields, nk[].k_numeric_mode_, _default_mode(context))
        var fit = _text_fit(text, context)
        _set_fit(fields, fit, context)
        py_setitem(fields, nk[].k_classes_, _classes_array(fit.classes, context))
        return fit.codes
    _ = _le_fit(model, y, context)
    return _le_transform(model, y, context)


def _le_transform_device(model: PythonObject, mode: PythonObject, lb: LabelBuf, cats: PythonObject, context: PythonObject) raises -> Optional[PythonObject]:
    """LabelEncoder.transform of a numeric label buffer: the codes and the
    per-chunk unknown flags in one program; None when a label is unknown."""
    # UNPACK
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var n = lb.n
    var p = EncProg()
    var w = _own_labels(p, lb)
    var known = _own_cats(p, cats, context)
    # RUN
    var plan = lab_transform_stage(p.prog, w, known, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK. The arena address read once (was two Python subscripts per flag in `_iword`).
    var flags = I32P(unsafe_from_address=Int(py=ran[0][1]))
    for i in range(plan.chunks):  # native data loop (not a driver loop): the per-chunk unknown flags, read natively
        if flags[plan.neg + i] != 0:
            return None
    _ = ran  # keep the arena alive through the reads above
    return _take_i32(ran, plan.codes, n, -1, context)


def _le_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var fields = py_fields(model)
    _check_fitted(fields, "LabelEncoder")
    var cats = py_item(fields, nk[].k__cats)
    var classes = py_item(fields, nk[].k__classes)
    var mode = py_item(fields, nk[].k_numeric_mode_)
    if cats is not None:
        var lb = _label_buffer(y, context)
        if lb:
            var done = _le_transform_device(model, mode, lb.value(), cats, context)
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
        # lane fix-output-zero-fill (2026-11-14): unfilled; p3_dict_codes writes every label's code (-1 unseen).
        var store = native_array_unfilled("i", n)
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
    # Numeric labels from a Python list among numeric classes. UNPACK: the
    # label words read above (`words`), the fitted class words, the route.
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var p = EncProg()
    var known = _own_cats(p, cats, context)
    # RUN
    var plan = lab_list_transform_stage(p.prog, words, known)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    if _word(ran, plan.neg) > 0:
        raise Error(UNSEEN)
    return _take_i32(ran, plan.codes, n, -1, context)


def _le_inverse_device(model: PythonObject, fields: PythonObject, lb: LabelBuf, cats: PythonObject, classes: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`_inverse_device`: each code checked (an integer in [0, K)) and its
    class written as the int64 / float64 word returned, one program."""
    var nk = prep_keys()
    # UNPACK
    var entries = scaler_entries(model, py_item(fields, nk[].k_numeric_mode_), context)
    var caps = enc_caps(model, entries)
    var n = lb.n
    var ints = labels_label_kind(classes, context) == "int"
    var p = EncProg()
    var w = _own_labels(p, lb)
    var known = _own_cats(p, cats, context)
    # RUN
    var plan = lab_inverse_stage(p.prog, w, known, ints)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    if _word(ran, plan.bad) > 0:
        raise Error(UNSEEN)
    return _class_words(ran, plan.words, n, ints, context)


def _le_inverse(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var fields = py_fields(model)
    _check_fitted(fields, "LabelEncoder")
    var cats = py_item(fields, nk[].k__cats)
    var classes = py_item(fields, nk[].k__classes)
    if cats is not None:
        # `_inverse_device`: an integer or float code buffer over numeric classes
        var lb = _label_buffer(y, context)
        if lb:
            return _le_inverse_device(model, fields, lb.value(), cats, classes, context)
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
    var nk = prep_keys()
    var b = _builtins()
    var fields = py_fields(model)
    if _truth(py_item(fields, nk[].k_sparse_output)):
        raise Error("NotImplementedError: mojolearn: LabelBinarizer(sparse_output=True) is not implemented")
    var integral = py_item(context, nk[].k_labels_integral_type)
    var neg = py_item(fields, nk[].k_neg_label)
    var pos = py_item(fields, nk[].k_pos_label)
    if not (Bool(py=py_call(b.isinstance, neg, integral)) and Bool(py=py_call(b.isinstance, pos, integral))
            and _truth(py_call(native_loaded_module("operator").lt, neg, pos))):
        raise Error("mojolearn: neg_label must be an integer below pos_label")
    var mode = _default_mode(context)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    var ind = _multilabel_indicator(y, context)
    if ind:
        var columns = _range_classes(Int(py=ind.value().shape[1]))
        py_setitem(fields, nk[].k__classes, columns)
        py_setitem(fields, nk[].k__cats, PythonObject(None))
        py_setitem(fields, nk[].k_classes_, _classes_array(columns, context))
        py_setitem(fields, nk[].k_y_type_, PythonObject(MULTILABEL))
        return model
    var fit = _fit_labels(model, fields, mode, y, context)
    var classes = fit.classes
    py_setitem(fields, nk[].k_classes_, _fit_array(fit, context))
    py_setitem(fields, nk[].k_y_type_, PythonObject("binary" if len(classes) <= 2 else "multiclass"))
    return model


def _lb_transform_indicator(model: PythonObject, mode: PythonObject, arr: PythonObject, classes: PythonObject, neg: Int, pos: Int,
                            context: PythonObject) raises -> PythonObject:
    # A multilabel indicator: each nonzero entry pos_label, each zero neg_label.
    var n = Int(py=arr.shape[0])
    var K = Int(py=arr.shape[1])
    if K != len(classes):
        raise Error("mojolearn: classes " + _text(classes) + " mismatch with the " + String(K) + " label columns in y")
    # UNPACK
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var p = EncProg()
    var owner = p.own(arr, n * K)
    # RUN
    var out = lab_indicator_stage(p.prog, owner, n * K, neg, pos, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take_i32(ran, out, n, K, context)


def _lb_transform_device(model: PythonObject, mode: PythonObject, lb: LabelBuf, cats: PythonObject, K: Int, W: Int, neg: Int, pos: Int,
                         context: PythonObject) raises -> PythonObject:
    # A numeric label buffer: the codes and the dense output in one program.
    # UNPACK
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var p = EncProg()
    var w = _own_labels(p, lb)
    var known = _own_cats(p, cats, context)
    # RUN
    var out = lab_binarize_buffer_stage(p.prog, w, known, K, W, neg, pos, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take_i32(ran, out, lb.n, W, context)


def _lb_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var b = _builtins()
    # UNPACK: the fitted state and the parameters, read once.
    var fields = py_fields(model)
    _check_fitted(fields, "LabelBinarizer")
    var classes = py_item(fields, nk[].k__classes)
    var cats = py_item(fields, nk[].k__cats)
    var mode = py_item(fields, nk[].k_numeric_mode_)
    var y_type = String(py_item(fields, nk[].k_y_type_))
    var neg = Int(py=py_call(b.int, py_item(fields, nk[].k_neg_label)))
    var pos = Int(py=py_call(b.int, py_item(fields, nk[].k_pos_label)))
    var ind = _multilabel_indicator(y, context)
    if ind or y_type == MULTILABEL:
        if y_type != MULTILABEL:
            raise Error("mojolearn: The object was not fitted with multilabel input.")
        if not ind:
            raise Error("mojolearn: y is not a multilabel indicator; the binarizer was fitted with one")
        return _lb_transform_indicator(model, mode, ind.value(), classes, neg, pos, context)
    var K = len(classes)
    var binary = K <= 2
    var W = 1 if binary else K
    if cats is not None and K > 1:
        var lb = _label_buffer(y, context)
        if lb:
            # the raw words up, the codes and the dense output on the device
            return _lb_transform_device(model, mode, lb.value(), cats, K, W, neg, pos, context)
    # str, object or list labels: their float codes are read here (UNPACK),
    # the dense output is one typed program. The capabilities come from the
    # estimator's route (this run read them from the image by name before).
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var p = EncProg()
    if cats is None:
        var text = StrTable()
        if _text_labels(y, context, text):
            # A 'U' label vector: one buffer read, the float codes natively.
            var looked = _text_codes(classes, text, True, context)
            if looked is not None:
                var tn = len(text.starts)
                var owner = p.own(looked[0], tn)
                var codes = LabelCodes(LAB_CODES_BUFFER, tn, owner, List[Float64](), _no_cats())
                return _lb_binarize_run(p, codes, K, W, neg, pos, entries, caps, context)
    var values = _flatten_labels(y, context)
    var n = len(values)
    var source = _list_codes(p, classes, cats, values, n, context)
    return _lb_binarize_run(p, source, K, W, neg, pos, entries, caps, context)


def _lb_binarize_run(mut p: EncProg, source: LabelCodes, K: Int, W: Int, neg: Int, pos: Int, entries: PythonObject,
                     caps: XprepCaps, context: PythonObject) raises -> PythonObject:
    # RUN and PACK of the dense label_binarize output of the float codes
    # `source` (one class: every code absent).
    var out = lab_binarize_codes_stage(p.prog, source, K, W, neg, pos, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    return _take_i32(ran, out, source.n, W, context)


def _threshold(fields: PythonObject, threshold: PythonObject) raises -> Float64:
    var nk = prep_keys()
    # threshold, or halfway between neg_label and pos_label.
    if threshold is None:
        var b = _builtins()
        return Float64(Int(py=py_call(b.int, py_item(fields, nk[].k_pos_label))) + Int(py=py_call(b.int, py_item(fields, nk[].k_neg_label)))) / 2.0
    return Float64(py=py_call(_builtins().float, threshold))


def _lb_inverse(model: PythonObject, y: PythonObject, threshold: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # UNPACK: the fitted state, Y, the target kind, the threshold and (for
    # numeric classes) the class words, read once into the typed request.
    var fields = py_fields(model)
    _check_fitted(fields, "LabelBinarizer")
    var classes = py_item(fields, nk[].k__classes)
    var cats = py_item(fields, nk[].k__cats)
    var entries = scaler_entries(model, py_item(fields, nk[].k_numeric_mode_), context)
    var caps = enc_caps(model, entries)
    var values = xprep_values(y, entries, context)
    var n = Int(py=values[3])
    var W = Int(py=values[4])
    var K = len(classes)
    var y_type = String(py_item(fields, nk[].k_y_type_))
    var target = LAB_Y_BINARY
    var t = 0.0
    if y_type == MULTILABEL:
        target = LAB_Y_MULTILABEL
        if W != K:
            raise Error("mojolearn: Y has " + String(W) + " columns, expected " + String(K))
        t = _threshold(fields, threshold)
    elif y_type == "multiclass":
        target = LAB_Y_MULTICLASS
        if W != K:
            raise Error("mojolearn: Y has " + String(W) + " columns, expected " + String(K))
    else:
        if W > 2:
            raise Error("mojolearn: output_type='binary', but y.shape = (" + String(n) + ", " + String(W) + ")")
        # the last column against the threshold, one code per row (c2_bin_code)
        t = _threshold(fields, threshold)
    var p = EncProg()
    # numeric classes: each code's class gathered on the device; one class: both codes name it
    var gather = target != LAB_Y_MULTILABEL and cats is not None
    var ints = False
    var known = _no_cats()
    if gather:
        ints = labels_label_kind(classes, context) == "int"
        var table = cats
        if K <= 1:
            var both = List[Float64]()
            var source = F32P(unsafe_from_address=Int(py=classical_address(cats)))
            _ = cats  # keep the Python object alive through the native call above
            for _ in range(2):  # driver-loop: over the two codes that name the one class
                for i in range(Int(py=cats.size)):  # driver-loop: over the fitted category words
                    both.append(Float64(source[i]))
                    _ = cats  # keep the Python object alive through the native call above
            # lane fix-output-zero-fill (2026-11-14): unfilled; the loop below writes all len(both) words.
            table = classical_wrap_array(native_array_unfilled("f", len(both)), Python.tuple(len(both)), "<f4", "C", _out(context))
            var dest = F32P(unsafe_from_address=Int(py=classical_address(table)))
            _ = table  # keep the Python object alive through the native call above
            for i in range(len(both)):  # driver-loop: over the doubled category words
                dest[i] = Float32(both[i])
                _ = table  # keep the Python object alive through the native call above
        known = _own_cats(p, table, context)
    var x = p.own_x(values)
    # RUN: the typed stages, then the typed run.
    var plan = lab_unbinarize_stage(p.prog, LabelUnbinarize(target, n, W, K, x, t, gather, known^, ints))
    var ran = p.run_caps(entries, context, caps)
    # PACK
    if target == LAB_Y_MULTILABEL:
        return _take_i32(ran, plan.words, n, W, context)
    if not gather:
        # str classes: the codes come back for the explicit object-label
        # decode. The code words read natively from the program's float32
        # result (its address read once, not one `ran[0][1]` lookup per row).
        var idx = List[Int](length=n, fill=0)
        if K != 1:
            var words = F32P(unsafe_from_address=Int(py=ran[0][1]))
            for i in range(n):  # native data loop (not a driver loop): native read of one code per row (no CPython call)
                idx[i] = Int(words[plan.codes + i])
            _ = ran  # keep the result alive through the native reads above
        return _class_list(classes, idx, context)
    return _class_words(ran, plan.words, n, ints, context)


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
    # Lane list-input-loops (2026-11-12): the rows are held natively (a set
    # row listed once) with their kind, every label read borrowed from the
    # row's C storage and its class compared by word: no Python list of
    # rows, no boxed subscript, no PythonObject per label.
    var list_word = Int(b.list._obj_ptr)
    var tuple_word = Int(b.tuple._obj_ptr)
    var set_word = Int(b.set._obj_ptr)
    var frozenset_word = Int(b.frozenset._obj_ptr)
    var int_word = Int(b.int._obj_ptr)
    var y_word = p3_type_word(y._obj_ptr)
    var y_exact = y_word == list_word or y_word == tuple_word
    # fb-metrics (2026-11-17; MultiLabelBinarizer fit 2.0-2.2x vs 0.8.55):
    # a set row is walked by its C iterator in both passes (0.8.55 chained
    # the rows in C), not copied into a fresh Python list per row first.
    ref cpy = Python().cpython()
    var table = cpython_table()
    var rows = List[PythonObject](capacity=nrows)
    var row_kind = List[Int](capacity=nrows)  # 0 list, 1 tuple, 2 set or frozenset
    for i in range(nrows):  # list-input: one borrowed read per row of the caller's Python label rows (ragged, no buffer form)
        var row = PythonObject(from_borrowed=p3_item(y, i, y_word == tuple_word)) if y_exact else y[i]
        var w = p3_type_word(row._obj_ptr)
        if w == list_word or w == tuple_word:
            rows.append(row)
            row_kind.append(1 if w == tuple_word else 0)
        elif w == set_word or w == frozenset_word:
            rows.append(row)
            row_kind.append(2)
        else:
            return None
    var total = 0
    for i in range(nrows):  # list-input: one read per row of the caller's Python label rows (ragged, no buffer form)
        var m = len(rows[i])
        if row_kind[i] == 2:
            var raw_it = table[].PyObject_GetIter(rows[i]._obj_ptr)
            if not raw_it:
                raise cpy.unsafe_get_error()
            var it = PythonObject(from_owned=raw_it)
            for _ in range(m):  # list-input: one class-word check per label of a Python label set, by its C iterator
                var raw = table[].PyIter_Next(it._obj_ptr)
                if not raw:
                    if cpy.PyErr_Occurred():
                        raise cpy.unsafe_get_error()
                    raise Error("RuntimeError: a label set changed size during iteration")
                var label = PythonObject(from_owned=raw)
                if p3_type_word(label._obj_ptr) != int_word:
                    return None
        else:
            for j in range(m):  # list-input: one borrowed class-word check per label of a ragged Python label set
                if p3_type_word(p3_item(rows[i], j, row_kind[i] == 1)) != int_word:
                    return None
        total += m
    if total == 0 or total > LABEL_WORDS_MAX or 2 * total > LABEL_WORDS_MAX:
        return None
    # Each exact int is read with PyLong_AsLongLong (an int outside int64
    # sets OverflowError, which is cleared and takes the label input step):
    # no Python rich comparison or Int conversion per label (p3-prep).
    var nw = 2 * total
    var words = classical_wrap_array(native_array_zeros("i", nw), Python.tuple(nw), "<i4", "C", _out(context))
    var dest = I64P(unsafe_from_address=Int(py=classical_address(words)))
    var offs = classical_wrap_array(native_array_zeros("i", nrows + 1), Python.tuple(nrows + 1), "<i4", "C", _out(context))
    var od = I32P(unsafe_from_address=Int(py=classical_address(offs)))
    _ = words  # keep the Python object alive through the native call above
    _ = offs  # keep the Python object alive through the native call above
    var at = 0
    for i in range(nrows):  # list-input: one read per row of the caller's Python label rows (ragged, no buffer form)
        var m = len(rows[i])
        if at + m > total:
            raise Error("RuntimeError: a label row changed size during iteration")
        if row_kind[i] == 2:
            # the same unmodified set iterates in the same order as the check pass
            var raw_it = table[].PyObject_GetIter(rows[i]._obj_ptr)
            if not raw_it:
                raise cpy.unsafe_get_error()
            var it = PythonObject(from_owned=raw_it)
            for _ in range(m):  # list-input: one PyLong_AsLongLong per label of a Python label set, by its C iterator
                var raw = table[].PyIter_Next(it._obj_ptr)
                if not raw:
                    if cpy.PyErr_Occurred():
                        raise cpy.unsafe_get_error()
                    raise Error("RuntimeError: a label set changed size during iteration")
                var label = PythonObject(from_owned=raw)
                var word = table[].PyLong_AsLongLong(label._obj_ptr)
                if word == -1 and cpy.PyErr_Occurred():
                    cpython_table()[].PyErr_Clear()
                    return None
                dest[at] = word
                _ = words  # keep the Python object alive through the native call above
                at += 1
        else:
            for j in range(m):  # list-input: one borrowed PyLong_AsLongLong per label of a ragged Python label set
                var word = table[].PyLong_AsLongLong(p3_item(rows[i], j, row_kind[i] == 1))
                if word == -1 and cpy.PyErr_Occurred():
                    cpython_table()[].PyErr_Clear()
                    return None
                dest[at] = word
                _ = words  # keep the Python object alive through the native call above
                at += 1
        od[i + 1] = Int32(at)
        _ = offs  # keep the Python object alive through the native call above
    _ = len(rows)  # keep the rows alive through the borrowed label reads above
    return MLBFlat(nrows, LabelBuf(3, words, total, nw, False), offs)


def _mlb_transform_flat(model: PythonObject, fl: MLBFlat, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # `_transform_flat`: the int32 indicator from the device route, one program.
    # UNPACK
    var fields = py_fields(model)
    var entries = scaler_entries(model, py_item(fields, nk[].k_numeric_mode_), context)
    var caps = enc_caps(model, entries)
    var n = fl.n
    var K = len(py_item(fields, nk[].k__classes))
    var p = EncProg()
    var w = _own_labels(p, fl.lb)
    var known = _own_cats(p, py_item(fields, nk[].k__cats), context)
    var offsets = p.own(fl.offs, n + 1)
    # RUN
    var out = lab_row_ones_stage(p.prog, w, known, n, offsets, K, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take_i32(ran, out, n, K, context)


def _fit_flat(model: PythonObject, fields: PythonObject, mode: PythonObject, fl: Optional[MLBFlat], context: PythonObject) raises -> Bool:
    var nk = prep_keys()
    # `_fit_flat`: classes_ from the device route; False when it does not apply.
    if not fl:
        return False
    var got = _label_fit_device(model, mode, fl.value().lb, False, context)
    if not got:
        return False
    _set_fit(fields, got.value(), context)
    py_setitem(fields, nk[].k__given, PythonObject(False))
    return True


def _rows_lists(y: PythonObject) raises -> PythonObject:
    # [list(row) for row in y]
    var b = _builtins()
    var rows = py_call(b.list, y)
    var n = len(rows)
    var list_fn = b.list
    var result = _new_list(n)
    for i in range(n):  # list-input: one borrowed row read and one list() per row of the caller's ragged label sets (no buffer form), presized result
        _list_put(result, i, py_call(list_fn, PythonObject(from_borrowed=p3_item(rows, i, False))))
    return result


def _mlb_fit(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var b = _builtins()
    var fields = py_fields(model)
    if _truth(py_item(fields, nk[].k_sparse_output)):
        raise Error("NotImplementedError: mojolearn: MultiLabelBinarizer(sparse_output=True) is not implemented")
    var mode = _default_mode(context)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    if py_item(fields, nk[].k_classes) is not None:
        py_setitem(fields, nk[].k__classes, py_call(b.list, py_item(fields, nk[].k_classes)))
        py_setitem(fields, nk[].k__given, PythonObject(True))
        py_setitem(fields, nk[].k__cats, PythonObject(None))
    elif not _fit_flat(model, fields, mode, _mlb_flat(y, context), context):
        var rows = _rows_lists(y)
        var flat = Python.list()
        var extend_fn = flat.extend
        for i in range(len(rows)):  # list-input: one borrowed row read and one list.extend per row of the caller's ragged label sets (no buffer form)
            _ = py_call(extend_fn, PythonObject(from_borrowed=p3_item(rows, i, False)))
        if len(flat) > 0:
            _set_fit(fields, _label_classes(model, mode, flat, context), context)
        else:
            py_setitem(fields, nk[].k__classes, Python.list())
            py_setitem(fields, nk[].k__cats, PythonObject(None))
        py_setitem(fields, nk[].k__given, PythonObject(False))
    py_setitem(fields, nk[].k_classes_, _classes_array(py_item(fields, nk[].k__classes), context))
    return model


def _mlb_fit_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var fields = py_fields(model)
    if py_item(fields, nk[].k_classes) is None and not _truth(py_item(fields, nk[].k_sparse_output)):
        var fl = _mlb_flat(y, context)
        if fl:
            var mode = _default_mode(context)
            py_setitem(fields, nk[].k_numeric_mode_, mode)
            if _fit_flat(model, fields, mode, fl, context):
                py_setitem(fields, nk[].k_classes_, _classes_array(py_item(fields, nk[].k__classes), context))
                return _mlb_transform_flat(model, fl.value(), context)
    var rows = _rows_lists(y)
    _ = _mlb_fit(model, rows, context)
    return _mlb_transform(model, rows, context)


def _mlb_transform(model: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var fields = py_fields(model)
    _check_fitted(fields, "MultiLabelBinarizer")
    var classes = py_item(fields, nk[].k__classes)
    var cats = py_item(fields, nk[].k__cats)
    if cats is not None and len(classes) > 0:
        var fl = _mlb_flat(y, context)
        if fl:
            return _mlb_transform_flat(model, fl.value(), context)
    var rows = _rows_lists(y)
    var n = len(rows)
    var K = len(classes)
    var flat = Python.list()
    var owner = List[Float64]()
    var extend_fn = flat.extend
    for i in range(n):  # list-input: one borrowed row read and one list.extend per row of the caller's ragged label sets (no buffer form), not one append per label
        var row = PythonObject(from_borrowed=p3_item(rows, i, False))
        var m = len(row)
        _ = py_call(extend_fn, row)
        for _ in range(m):  # native data loop (not a driver loop): the row's owner words (no CPython call)
            owner.append(Float64(i))
    # UNPACK: the labels' float codes (`_list_codes`; none when no row holds
    # a label) and each label's row; the route.
    var entries = scaler_entries(model, py_item(fields, nk[].k_numeric_mode_), context)
    var caps = enc_caps(model, entries)
    var p = EncProg()
    var m = len(flat)
    var source = LabelCodes(LAB_CODES_WORDS, 0, NONE, List[Float64](), _no_cats())
    if m > 0:
        source = _list_codes(p, classes, cats, flat, m, context)
    # RUN
    var out = lab_scatter_ones_stage(p.prog, source, owner, n, K, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take_i32(ran, out, n, K, context)


def _mlb_inverse_direct(classes: PythonObject, k: Int, yt: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_mlb_inverse` of a C-ordered 2-D numeric indicator (a mojolearn Array
    or an ndarray, int32 / int64 / float32 / float64, (n, k) wide): the set
    cells are found by one native scan of the buffer and only the picked
    classes and the row tuples become Python objects (p3-prep). None for any
    other input (the general walk below). A nonzero or NaN cell is set, as
    the Python truth test reads it."""
    var nk = prep_keys()
    var b = _builtins()
    var ndarray = py_get(context, nk[].k_ndarray)
    var is_array = py_type(yt) is py_item(context, nk[].k_array_class)
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
    var nk = prep_keys()
    # One tuple of labels per row: the user list of Python label tuples.
    var b = _builtins()
    var fields = py_fields(model)
    _check_fitted(fields, "MultiLabelBinarizer")
    var classes = py_item(fields, nk[].k__classes)
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
def prep_labels_call(model: PythonObject, kind: String, op: Int, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """LabelEncoder / LabelBinarizer / MultiLabelBinarizer fit(y),
    fit_transform(y), transform(y), inverse_transform(y) (LabelBinarizer:
    inverse_transform(Y, threshold))."""
    if kind == "label_encoder":
        if op == PREP_OPN_FIT:
            return _le_fit(model, args[0], context)
        if op == PREP_OPN_FIT_TRANSFORM:
            return _le_fit_transform(model, args[0], context)
        if op == PREP_OPN_TRANSFORM:
            return _le_transform(model, args[0], context)
        if op == PREP_OPN_INVERSE_TRANSFORM:
            return _le_inverse(model, args[0], context)
    elif kind == "label_binarizer":
        if op == PREP_OPN_FIT:
            return _lb_fit(model, args[0], context)
        if op == PREP_OPN_FIT_TRANSFORM:
            _ = _lb_fit(model, args[0], context)
            return _lb_transform(model, args[0], context)
        if op == PREP_OPN_TRANSFORM:
            return _lb_transform(model, args[0], context)
        if op == PREP_OPN_INVERSE_TRANSFORM:
            return _lb_inverse(model, args[0], args[1], context)
    elif kind == "multilabel_binarizer":
        if op == PREP_OPN_FIT:
            return _mlb_fit(model, args[0], context)
        if op == PREP_OPN_FIT_TRANSFORM:
            return _mlb_fit_transform(model, args[0], context)
        if op == PREP_OPN_TRANSFORM:
            return _mlb_transform(model, args[0], context)
        if op == PREP_OPN_INVERSE_TRANSFORM:
            return _mlb_inverse(model, args[0], context)
    raise Error("TypeError: the native label owner does not serve " + operation)
