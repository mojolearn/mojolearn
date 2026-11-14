# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The label, target and X refusals the Python shells `_labels` and
# `_sk_validate` still made in Python (lane py-admission, 2026-11-14, plan
# docs/plans/PYTHON_REMAINING_2026-11-14.md section 3, lane `py-admission`).
# Each is one operation of the compiled label entry `labels_native_call`
# (bindings/labels_native_entry_pack.mojo); the Python side is the one call
# that hands its arguments over.
#
#   labels_prepare_targets_op(labels, weights, n, flags)
#                     `_labels.prepare_targets`: the shared argument-and-label
#                     step (the store allocation, the dtype codes, the
#                     weight refusals with scikit-learn's wording, the cut
#                     of the class table to its k classes)
#   labels_encode_numeric_op(y)
#                     `_labels._encode_labels_native`: one numeric label
#                     vector through prepare_targets, or None
#   labels_unique_inverse_op(y)
#                     `_labels.unique_inverse`: (classes Array, int32 codes)
#                     for numeric labels; the str encoder or the ORDER RULE
#                     (classes list) for any other labels
#   labels_argmax_op(scores)
#                     `_labels.argmax_rows`: first-max column per row, int64
#   labels_decode_any_op(classes, codes)
#                     `_labels.decode_labels`: codes of any integer form (an
#                     Array, a buffer, a list, any rank) made one 1-D int
#                     Array, then the one decode
#   labels_class_index_op(classes, key)
#                     `_labels.ClassLabels.__getitem__` with an integer array
#                     key: its refusals, the widening and the decode
#   admission_check_x_op(X, est, shape, complex, scan)
#                     `_sk_validate._check_X`'s refusals (complex, rank,
#                     empty) and its finiteness scan, scikit-learn's wording
#   admission_check_y_op(y, est, flags, x_rows, y_rows)
#                     `_sk_validate._check_y` / `_check_lengths`' refusals
#                     (y missing, complex, non-finite, continuous labels for
#                     a classifier, inconsistent lengths)
#
# The words read are the ones the Python steps read (admit_c_view: the
# input's own dtype, C order, a strided export copied once), so every class
# table, code and verdict is unchanged: no bit change. The scans are the host
# finiteness scan (core/host_finite_scan.mojo, split across the host's cores)
# and the base binding's reduce_stat / prepare_targets / argmax helpers the
# Python steps called, each run once per call with the GIL released.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.memory import unsafe_memcpy
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays_door import classical_address, classical_is_array
from bindings.classical_labels_run import _encode_code, _native_code
from bindings.admitted_input_unpack import admit_c_view, admitted_dtype, admitted_vector
from bindings.classical_labels_door import (
    ClassicalLabels, labels_builtins, labels_cast, labels_address, _flatten_labels, _text_list_labels,
    _sorted_classes, _native_labels, _numeric_label_view, classical_decode_labels,
)
from bindings.labels_descriptor_boundary_door import (
    labels_prepare_targets, labels_reduce_stat, labels_argmax_rows as labels_argmax_run,
)
from core.admitted_f32_plan import admitted_x_shape_refuse, admitted_target_present
from core.admitted_refusals import admitted_finite_wording
from core.hotpath_impl import HpStat
from core.host_finite_scan import host_all_finite_f32, host_all_finite_f64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from bindings.py_balanced import py_get
from bindings.classical_arrays_door import classical_wrap_array
from bindings.native_array_storage_door import native_array_from_values
from bindings.classical_labels_door import labels_real_type
from bindings.label_input_sources_unpack import (
    labels_list_scalar_kind, LABELS_LIST_INT, LABELS_LIST_FLOAT, LABELS_LIST_BOOL,
)

comptime _LA_I64P = MutPointer[Int64, MutUntrackedOrigin]

#: Exception types the Python label shells raised before lane py-admission,
#: kept: a base-binding helper's own refusal crossed as a bare `Exception`
#: (its direct call through `_buffer._native`), a dtype missing from
#: `_NATIVE_ENCODE` as a `KeyError`. bindings/labels_native_entry_pack.mojo
#: raises these two prefixes as those types; every other door is untouched.
comptime LABELS_BARE_EXCEPTION_KIND = "LabelsBareException"
comptime LABELS_BARE_EXCEPTION = "LabelsBareException:"
comptime LABELS_KEY_ERROR_KIND = "LabelsKeyError"
comptime LABELS_KEY_ERROR = "LabelsKeyError:"


def _la_helper_error(var error: Error) -> Error:
    """A base-binding helper's refusal with the type the shell raised: a
    NaN label as a ValueError, a missing binding as its ImportError, any
    other as the bare Exception the direct call raised."""
    var message = String(error)
    if "NaN label" in message:
        if message.startswith("ValueError: "):
            return error^
        return Error("ValueError: " + message)
    if message.startswith("ImportError:") or message.startswith(LABELS_BARE_EXCEPTION):
        return error^
    return Error(LABELS_BARE_EXCEPTION + " " + message)


def _la_int_width(typestr: String) -> Int:
    """The item width of an integer typestr (any width NumPy hands over), 0
    for any other dtype."""
    if typestr == "<i1" or typestr == "<u1":
        return 1
    if typestr == "<i2" or typestr == "<u2":
        return 2
    if typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<i8" or typestr == "<u8":
        return 8
    return 0


def _la_widen_ints(src: Int, typestr: String, dst: Int, count: Int) raises:
    """`count` integers of `typestr` (int8 / uint8 / int16 / uint16 / int32 /
    uint32 / int64 / uint64) at `src` as int64 words at `dst`: the exact
    values `Array.astype("<i8")` gave, in element ranges across the host's
    cores. A uint64 value above int64's range is refused (OverflowError)."""
    var width = _la_int_width(typestr)
    if width == 0:
        raise Error("TypeError: mojolearn: " + typestr + " is not an integer dtype")
    if count <= 0:
        return
    var signed = typestr.startswith("<i")
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _range(t: Int) {imm src, imm dst, imm count, imm chunk, imm width, imm signed, imm flag_ptr}:
        var lo = t * chunk
        if lo >= count:
            return
        var hi = min(lo + chunk, count)
        var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst)
        for i in range(lo, hi):
            if width == 1:
                if signed:
                    out[i] = MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i).cast[DType.int64]()
                else:
                    out[i] = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i).cast[DType.int64]()
            elif width == 2:
                if signed:
                    out[i] = MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i).cast[DType.int64]()
                else:
                    out[i] = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i).cast[DType.int64]()
            elif width == 4:
                if signed:
                    out[i] = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i).cast[DType.int64]()
                else:
                    out[i] = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i).cast[DType.int64]()
            elif signed:
                out[i] = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i)
            else:
                var v = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=src).unsafe_load(i)
                if v > UInt64(9223372036854775807):
                    flag_ptr.unsafe_store(t, UInt8(1))
                out[i] = v.cast[DType.int64]()

    host_parallelize(_range, tasks)
    var fits = True
    for t in range(tasks):  # driver-loop: over the widening tasks' flags (at most one per core)
        if flags[t] != UInt8(0):
            fits = False
    _ = flags^
    if not fits:
        raise Error("OverflowError: mojolearn: a uint64 label does not fit int64")


def _la_to_i64(src: Int, typestr: String, dst: Int, count: Int, context: PythonObject) raises:
    """Any integer or float words as int64 (floats by the native cast,
    `Array.astype("<i8")`'s rule)."""
    if _la_int_width(typestr) > 0:
        _la_widen_ints(src, typestr, dst, count)
    else:
        labels_cast(src, typestr, dst, "<i8", count, context)

#: `prepare_targets`' weight statuses (`PT_W_*`) and scikit-learn's wording.
comptime LABELS_WEIGHT_NONFINITE = "Input sample_weight contains NaN or infinity."
comptime LABELS_WEIGHT_NEGATIVE = "Negative values in data passed to `sample_weight`"
comptime LABELS_WEIGHT_ZERO = "Sample weights must contain at least one non-zero number."

#: `reduce_stat`'s integrality reduction (`_array._REDUCE_INTEGRAL`).
comptime _LA_REDUCE_INTEGRAL = 4

#: admission_check_y_op flags.
comptime ADMISSION_Y_COMPLEX = 1
comptime ADMISSION_Y_SCAN = 2
comptime ADMISSION_Y_CLASSIFIER = 4
comptime ADMISSION_Y_MISSING = 8


# ---------------------------------------------------------------- targets


def labels_prepare_arrays(keep: PythonObject, y_addr: Int, typestr: String, n: Int, w_addr: Int, w_code: Int,
                          flags: Int, context: PythonObject) raises -> PythonObject:
    """One prepare_targets call over `n` C-contiguous labels of `typestr` at
    `y_addr` (`keep` owns them; `y_addr` 0: no labels) and optional weights:
    `(classes, codes)` Arrays (float64 classes for float labels, int64 for
    integer and bool labels, one int32 code per row), `(None, None)`
    without labels. Refuses a NaN label and the weights scikit-learn
    refuses, in its words."""
    var labelled = y_addr != 0
    var wide_float = typestr == "<f4" or typestr == "<f8"
    var wide = String("<f8") if wide_float else String("<i8")
    var classes_store = PythonObject(None)
    var codes = PythonObject(None)
    var y_code = 0
    var classes_addr = 0
    var codes_addr = 0
    if labelled:
        y_code = _encode_code(typestr)
        if y_code < 0:
            raise Error(LABELS_KEY_ERROR + " " + typestr)
        classes_store = admitted_vector(n, wide, context)
        codes = admitted_vector(n, "<i4", context)
        classes_addr = labels_address(classes_store)
        codes_addr = labels_address(codes)
    var k = 0
    try:
        k = labels_prepare_targets(context, y_addr, y_code, n, classes_addr, codes_addr, w_addr, w_code, flags)
    except error:
        raise _la_helper_error(error^)
    _ = keep  # the labels' owner, alive through the native call above
    if k == -3:
        raise Error(LABELS_WEIGHT_NONFINITE)
    if k == -4:
        raise Error(LABELS_WEIGHT_NEGATIVE)
    if k == -5:
        raise Error(LABELS_WEIGHT_ZERO)
    if k < 0:
        raise Error("RuntimeError: prepare_targets returned the unknown status " + String(k))
    if not labelled:
        return Python.tuple(PythonObject(None), PythonObject(None))
    var classes = classes_store
    if k < n:
        # the first k slots hold the classes: one copy into a k-slot store
        classes = admitted_vector(k, wide, context)
        if k > 0:
            unsafe_memcpy(dest=_LA_I64P(unsafe_from_address=labels_address(classes)),
                          src=_LA_I64P(unsafe_from_address=labels_address(classes_store)), count=k)
        _ = classes_store  # the full table, alive through the copy above
    var result = Python.tuple(classes, codes)
    _ = codes
    return result


def labels_prepare_targets_op(labels: PythonObject, weights: PythonObject, n: Int, flags: Int,
                              context: PythonObject) raises -> PythonObject:
    """`_labels.prepare_targets(labels, weights, n, ...)`: `labels` a
    C-contiguous vector Array of a native encode dtype or None, `weights` a
    float32 / float64 Array of `n` entries or None, `flags` the
    allow-negative (1) and allow-all-zero (2) bits."""
    var y_addr = 0
    var typestr = String("")
    if labels is not None:
        typestr = String(labels.dtype)
        y_addr = labels_address(labels)
    var w_addr = 0
    var w_code = 0
    if weights is not None:
        var w_dtype = String(weights.dtype)
        w_code = _encode_code(w_dtype)
        if w_code < 0:
            raise Error(LABELS_KEY_ERROR + " " + w_dtype)
        w_addr = labels_address(weights)
    var result = labels_prepare_arrays(labels, y_addr, typestr, n, w_addr, w_code, flags, context)
    _ = weights  # alive through the native call above
    return result


def labels_encode_numeric_op(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels._encode_labels_native(y)`: `(classes, codes)` (the class
    values as Python numbers, bools for a bool buffer) for one numeric label
    vector, or None for what only the label input step reads (lists, str or
    object buffers, a pandas Series, a matrix, an empty vector)."""
    var bool_source = False
    var view = _numeric_label_view(y, context, bool_source)
    if not view:
        return PythonObject(None)
    var encoded = _native_labels(view.value(), bool_source, context)
    var result = Python.tuple(encoded.classes, encoded.codes)
    _ = view^
    return result


def labels_unique_inverse_op(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.unique_inverse(y)`: for numeric labels the sorted distinct
    values (a float64 Array for real labels, int64 for integer and bool
    labels) and one int32 code per row; for str and object labels the str
    encoder's or the ORDER RULE's classes (a list) and codes. Refuses an
    empty y, a y that is not a vector and a NaN label."""
    var objects = _la_object_numeric(y, context)
    if objects:
        return _la_unique_numeric(objects.value(), objects.value(), context)
    var view = PythonObject(None)
    var typestr = String("")
    var numeric = False
    try:
        view = admit_c_view(y, "y", context)
        typestr = admitted_dtype(view, context)
        var letter = String(typestr[byte=1])
        numeric = letter == "f" or letter == "i" or letter == "u" or letter == "b"
    except:
        numeric = False
    if not numeric:
        # str or object labels: no device sort orders text, so these take
        # the ORDER RULE's own encoding (the label input step)
        var flat = _flatten_labels(y, context)
        var text = _text_list_labels(flat, context)
        if text:
            var named = text.value().copy()
            return Python.tuple(named.classes, named.codes)
        var ordered = _sorted_classes(flat, context)
        return Python.tuple(ordered.classes, ordered.codes)
    return _la_unique_numeric(view, view, context)


def _la_unique_numeric(view: PythonObject, owner: PythonObject, context: PythonObject) raises -> PythonObject:
    """unique_inverse's numeric arm over a C-ordered numeric view (or
    Array) `view`: refusals, the exact widening of a dtype the encoder has
    no arm for, one prepare_targets call."""
    var typestr = admitted_dtype(view, context)
    var n = Int(py=view.size)
    if n == 0:
        raise Error("mojolearn: y is empty")
    var big = 0
    for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
        if Int(py=view.shape[k]) > big:
            big = Int(py=view.shape[k])
    if n != big:
        raise Error("mojolearn: y must be a vector of labels")
    var keep = view
    var address = Int(py=classical_address(view))
    if _encode_code(typestr) < 0:
        # float16, int16, int8, ...: widened exactly, as the 64-bit sort read them
        var target = String("<f8") if String(typestr[byte=1]) == "f" else String("<i8")
        keep = admitted_vector(n, target, context)
        if target == "<i8":
            _la_widen_ints(address, typestr, labels_address(keep), n)
        else:
            labels_cast(address, typestr, labels_address(keep), target, n, context)
        _ = view
        typestr = target
        address = labels_address(keep)
    var result = labels_prepare_arrays(keep, address, typestr, n, 0, 0, 0, context)
    _ = keep
    _ = view
    _ = owner
    return result


def _la_object_numeric(y: PythonObject, context: PythonObject) raises -> Optional[PythonObject]:
    """An object-dtype ndarray whose every leaf is a real number, as the
    Array `_buffer._materialize` made of it (scikit-learn's check_array
    reads such arrays as numbers): int64 when every leaf is exactly an int
    or bool, float64 when a float is among them or any leaf is another real
    scalar (a NumPy scalar). None for any other input, and for an object
    array holding str or other objects (the label input step reads it)."""
    var ndarray = py_get(context, "ndarray")
    if ndarray is None or py_type(y) is not ndarray:
        return None
    if String(y.dtype.kind) != "O":
        return None
    var b = labels_builtins()
    var flat = _flatten_labels(y, context)
    var n = len(flat)
    var any_float = False
    var real_other = False
    var kind = labels_list_scalar_kind(flat)
    if kind == LABELS_LIST_FLOAT:
        any_float = True
    elif kind != LABELS_LIST_INT and kind != LABELS_LIST_BOOL:
        for i in range(n):  # cpu-route: the leaf types of a user object array, the explicit input step
            var t = py_type(flat[i])
            if t is b.float:
                any_float = True
            elif t is b.int or t is b.bool:
                continue
            elif labels_real_type(t, context):
                real_other = True
            else:
                return None
    var wide = any_float or real_other
    var store = native_array_from_values(String("d" if wide else "q"), flat)
    var shape = py_call(b.tuple, y.shape)
    var array = classical_wrap_array(store, shape, "<f8" if wide else "<i8", "C", context)
    _ = flat
    return array


def labels_argmax_op(scores: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows(scores)`: the row-wise first-max column of an
    `[n_rows, n_cols]` block (strict `>` from column 0: ties keep the lowest
    column, a NaN never replaces) as an int64 Array. Integer scores are
    widened to float64 (exact below 2**53)."""
    var view = admit_c_view(scores, "scores", context)
    var typestr = admitted_dtype(view, context)
    var nd = Int(py=view.ndim)
    if nd < 2:
        raise Error("not enough values to unpack (expected 2, got " + String(nd) + ")")
    if nd > 2:
        raise Error("too many values to unpack (expected 2)")
    var rows = Int(py=view.shape[0])
    var cols = Int(py=view.shape[1])
    var out = admitted_vector(rows, "<i8", context)
    if rows == 0:
        _ = view
        return out
    if cols == 0:
        raise Error("argmax_rows: scores have no columns")
    var keep = view
    var address = Int(py=classical_address(view))
    if typestr != "<f4" and typestr != "<f8":
        keep = admitted_vector(rows * cols, "<f8", context)
        if _la_int_width(typestr) > 0:
            # any integer width: exact int64 first, then the native float64 cast
            var exact = admitted_vector(rows * cols, "<i8", context)
            _la_widen_ints(address, typestr, labels_address(exact), rows * cols)
            labels_cast(labels_address(exact), "<i8", labels_address(keep), "<f8", rows * cols, context)
            _ = exact
        else:
            labels_cast(address, typestr, labels_address(keep), "<f8", rows * cols, context)
        _ = view
        typestr = "<f8"
        address = labels_address(keep)
    try:
        labels_argmax_run(context, typestr == "<f8", address, rows, cols, labels_address(out))
    except error:
        raise _la_helper_error(error^)
    _ = keep
    _ = view
    return out


#: The integer index dtypes `ClassLabels.__getitem__` takes.
def _labels_index_dtype(typestr: String) -> Bool:
    return (typestr == "<i8" or typestr == "<i4" or typestr == "<u4" or typestr == "<u2"
            or typestr == "<u1" or typestr == "<i1")


def labels_class_index_op(classes: PythonObject, key: PythonObject, context: PythonObject) raises -> PythonObject:
    """`classes_[codes]` with an integer array `key`: the label of each
    code (`decode_labels`), as an ndarray `classes_` is indexed."""
    var b = labels_builtins()
    var view = admit_c_view(key, "index", context)
    var typestr = admitted_dtype(view, context)
    if not _labels_index_dtype(typestr):
        raise Error("TypeError: classes_ takes an integer array as an index, not an array of " + typestr)
    if Int(py=view.ndim) != 1:
        var shape = String(py_call(b.repr, py_call(b.tuple, view.shape)))
        raise Error("TypeError: classes_ takes a 1-D integer array as an index, not one of shape " + shape)
    var codes = view
    if not classical_is_array(view, context) or (typestr != "<i8" and typestr != "<i4"):
        # one int64 Array the decode reads (a NumPy view, or a narrower dtype)
        var n = Int(py=view.size)
        codes = admitted_vector(n, "<i8", context)
        _la_to_i64(Int(py=classical_address(view)), typestr, labels_address(codes), n, context)
    var listed = py_call(b.list, classes)
    var result = classical_decode_labels(listed, codes, context)
    _ = codes
    _ = view
    return result


def labels_decode_any_op(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)`: a 1-D int32 / int64 Array
    of codes is decoded as it is; any other codes (a list or tuple, a
    buffer, another integer dtype, a matrix read in C order) are first one
    int64 Array of the same values."""
    if classical_is_array(codes, context) and Int(py=codes.ndim) == 1:
        var dtype = String(codes.dtype)
        if dtype == "<i4" or dtype == "<i8":
            return classical_decode_labels(classes, codes, context)
    var b = labels_builtins()
    if (py_type(codes) is b.list or py_type(codes) is b.tuple) and len(codes) == 0:
        var none = admitted_vector(0, "<i8", context)
        return classical_decode_labels(classes, none, context)
    var view = admit_c_view(codes, "codes", context)
    var typestr = admitted_dtype(view, context)
    var n = Int(py=view.size)
    var wide = admitted_vector(n, "<i8", context)
    _la_to_i64(Int(py=classical_address(view)), typestr, labels_address(wide), n, context)
    _ = view
    var result = classical_decode_labels(classes, wide, context)
    _ = wide
    return result


# ---------------------------------------------------------------- X and y


def _admission_all_finite(view: PythonObject, typestr: String) raises -> Bool:
    """The host finiteness scan of a C-ordered float32 / float64 view; True
    for any other dtype (only float data can hold NaN or infinity)."""
    var n = Int(py=view.size)
    if n == 0 or (typestr != "<f4" and typestr != "<f8"):
        return True
    var address = Int(py=classical_address(view))
    var finite = True
    with GILReleased(Python()):
        if typestr == "<f4":
            finite = host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address), n)
        else:
            finite = host_all_finite_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address), n)
    _ = view
    return finite


def _admission_integral(view: PythonObject, typestr: String, context: PythonObject) raises -> Bool:
    """True unless a float label vector holds a value that is not a whole
    number (scikit-learn's "continuous" target). A matrix or an empty
    vector is not judged here."""
    if typestr != "<f4" and typestr != "<f8":
        return True
    var n = Int(py=view.size)
    var big = 0
    for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
        if Int(py=view.shape[k]) > big:
            big = Int(py=view.shape[k])
    if n == 0 or n != big:
        return True
    # Every value integral exactly when every distinct value is (the
    # Python step reduced the classes of unique_inverse): one reduction.
    var stat = HpStat(0, 0, 0.0, 0, 0, 0)
    try:
        stat = labels_reduce_stat(context, Int(py=classical_address(view)), _native_code(typestr), n,
                                  _LA_REDUCE_INTEGRAL)
    except error:
        raise _la_helper_error(error^)
    _ = view
    return stat.i != 0


def admission_check_x_op(x: PythonObject, est: String, shape: PythonObject, complex: Bool, scan: Bool,
                         context: PythonObject) raises:
    """`_sk_validate._check_X`'s refusals, in scikit-learn's words: a
    complex X, a 0-D or 1-D X, an X with no samples or no features; and,
    with `scan`, NaN or infinity in a float X. `shape` is the tuple the
    shell read from the input's export."""
    if complex:
        admitted_x_shape_refuse(1, 0, "", "", est)
    var b = labels_builtins()
    var rank = len(shape)
    var text = String(py_call(b.repr, py_call(b.tuple, shape)))
    if rank < 2:
        admitted_x_shape_refuse(2, rank, text, "", est)
    if rank == 2:
        if Int(py=shape[0]) == 0:
            admitted_x_shape_refuse(3, rank, text, "", est)
        if Int(py=shape[1]) == 0:
            admitted_x_shape_refuse(4, rank, text, "", est)
    if scan:
        var view = admit_c_view(x, "X", context)
        var typestr = admitted_dtype(view, context)
        if not _admission_all_finite(view, typestr):
            raise Error(admitted_finite_wording(est))
        _ = view


def admission_check_y_op(y: PythonObject, est: String, flags: Int, x_rows: Int, y_rows: Int,
                         context: PythonObject) raises:
    """`_sk_validate._check_y` / `_check_lengths`' refusals, in
    scikit-learn's words: a required y that is None, a complex y, NaN or
    infinity in a float y (`ADMISSION_Y_SCAN`), a continuous float y for a
    classifier, and X and y of different lengths (both >= 0)."""
    if (flags & ADMISSION_Y_MISSING) != 0:
        admitted_target_present(False, est)
    if (flags & ADMISSION_Y_COMPLEX) != 0:
        raise Error("Complex data not supported: " + est + " got a complex y.")
    if (flags & ADMISSION_Y_SCAN) != 0:
        var view = admit_c_view(y, "y", context)
        var typestr = admitted_dtype(view, context)
        if not _admission_all_finite(view, typestr):
            raise Error("Input y contains NaN or infinity (" + est + ").")
        if (flags & ADMISSION_Y_CLASSIFIER) != 0 and not _admission_integral(view, typestr, context):
            raise Error(
                "Unknown label type: continuous. Maybe you are trying to fit a classifier, "
                + "which expects discrete classes on a regression target with continuous values.")
        _ = view
    if x_rows >= 0 and y_rows >= 0 and x_rows != y_rows:
        raise Error("Found input variables with inconsistent numbers of samples: ["
                    + String(x_rows) + ", " + String(y_rows) + "]")
