# SPDX-License-Identifier: Apache-2.0
# `_labels.encode_labels` and `_labels.decode_labels` for the Gaussian process
# classifier's owner (bindings/gp_classifier_native.mojo), under the ORDER RULE
# python/mojolearn/_labels.py states. A numeric label vector (a C-ordered Array
# or ndarray of a native dtype, or a flat list of exact ints or exact floats) is
# encoded by ONE call of the base binding's `prepare_targets` (the device encoder
# on a GPU install, the core host twin on a CPU-only one), the call `encode_labels`
# made; str and other Python label objects take the label input step with the
# native text/integer index or the admitted mixed-numeric order. Integer and
# float classes decode through the base binding's `gather_i64` / `gather_f64`;
# str and bool classes decode to the label objects. The shell hands over the
# base binding (`_backend.binding("_mojolearn", "identical")`, `_buffer._native`'s).
from std.ffi import c_int
from std.python import Python, PythonObject
from core.native_label_codes import NativeLabelCodes
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import NativeModules, native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address, classical_is_array
from bindings.family_common import (
    family_call, family_is_str, family_array, FAMILY_F64P, FAMILY_I64P,
)
from bindings.label_integer_storage import labels_float_storage, _ndarray_buffer_address
from core.gp_label_gather_impl import label_float_gather_f64
from bindings.classical_pandas import classical_admit_value
from bindings.classical_labels import (
    _flatten_labels,
    ClassicalLabels, NativeTextLabels, _TextCodeStorage, _TextLabels, _PythonTextCodes, _NativeTextCodes,
    _text_buffer_labels_into, _text_list_labels_into,
)
from bindings.label_input_sources import labels_storage_source, labels_list_scalar_kind, LABELS_LIST_OTHER, LABELS_LIST_INT, LABELS_LIST_FLOAT, LABELS_LIST_BOOL
from bindings.perf3_label_doors_text import labels_all_exact_str
from bindings.gp_label_snapshot_native import gp_label_list
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.gp_svm_records import gpsvm_side_get, gpsvm_side_put, _label_same, _labels_recordable
from bindings.gp_class_table_boundary import GPPreparedClassTable
from bindings.gp_score_label_storage import GPDecodedScoreLabels
from core.gp_native_label_table import GPLabelTable, GP_LABEL_I64, GP_LABEL_F64
from bindings.gp_labels_boundary import GPLabelsConnection, gp_labels_prepare, gp_labels_gather
from bindings.gp_label_input_boundary import gp_label_input
from bindings.gp_label_order_boundary import gp_integer_classes
from bindings.gp_mixed_label_boundary import gp_mixed_classes, gp_mixed_index, gp_mixed_codes
from core.labels_native_api import LabelsPrepareRequest, LabelsGatherRequest

comptime _NAN_LABEL = "mojolearn: y contains a NaN label; NaN is not a class"


def _gp_native_code(typestr: String) -> Int:
    """`_labels._NATIVE_ENCODE` (bool storage reads as uint8), -1 otherwise."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1" or typestr == "|u1" or typestr == "|b1":
        return 5
    return -1


def _gp_prepare[S: _TextCodeStorage](address: Int, code: Int, n: Int, as_bool: Bool, base: PythonObject,
                context: PythonObject) raises -> _TextLabels[S]:
    var wide = code == 0 or code == 1
    var classes = List[UInt64](length=n, fill=0)
    var codes = S(n, context)
    var connection = GPLabelsConnection(base)
    var request = LabelsPrepareRequest(address, Int(classes.unsafe_ptr()), codes.address(), 0, code, n, 0, 0)
    var k: Int
    try:
        k = gp_labels_prepare(connection, request)
        _ = codes
        _ = classes
    except error:
        if String(error).find("NaN label") != -1:
            raise Error(String(_NAN_LABEL))
        raise Error(String(error))
    var values = Python.list()
    if wide:
        var fp = FAMILY_F64P(unsafe_from_address=Int(classes.unsafe_ptr()))
        for i in range(k):  # driver-loop: PACK only the distinct observable class objects
            _ = py_call(values.append, PythonObject(fp[i]))
    else:
        var ip = FAMILY_I64P(unsafe_from_address=Int(classes.unsafe_ptr()))
        for i in range(k):  # driver-loop: PACK distinct classes in the original native sort order
            _ = py_call(values.append, PythonObject(Bool(ip[i] != 0)) if as_bool else PythonObject(Int(ip[i])))
    _ = classes
    return _TextLabels[S](values, codes^, n)


def _gp_buffer_labels[S: _TextCodeStorage](y: PythonObject, base: PythonObject, context: PythonObject) raises -> Optional[_TextLabels[S]]:
    """`_encode_labels_native(y)`: a C-ordered numeric vector Array or ndarray
    through `prepare_targets`, or None for what the label input step reads
    (a list, a strided or non-native buffer, a matrix, an empty vector: every
    one of them yields the same classes and codes there)."""
    var ty = py_type(y)
    var ndarray = py_get(context, "ndarray")
    var is_array = classical_is_array(y, context)
    if not is_array and (ndarray is None or not (ty is ndarray)):
        return None
    var typestr = String(y.dtype) if is_array else String(y.dtype.str)
    var code = _gp_native_code(typestr)
    var size = Int(py=y.size)
    if code < 0 or size == 0:
        return None
    var contiguous = (String(y.order) == "C" or Int(py=y.ndim) < 2) if is_array else Bool(py=y.flags.c_contiguous)
    if not contiguous:
        return None
    var longest = 0
    for k in range(len(y.shape)):  # driver-loop: over array rank metadata
        longest = max(longest, Int(py=y.shape[k]))
    if size != longest:
        return None
    return _gp_prepare[S](Int(py=classical_address(y)), code, size, typestr == "|b1", base, context)


def _gp_float_storage_labels[S: _TextCodeStorage](y: PythonObject, base: PythonObject, context: PythonObject) raises -> Optional[_TextLabels[S]]:
    """A float32/float64 ndarray or Array of any rank and strides whose every
    label is finite (`labels_float_storage`, one native pass): the labels read
    straight from the buffer in C order into one float64 vector and encoded by
    the same `prepare_targets` call the list path makes. This is the result
    the label input step gives (`tolist` yields exact Python floats in C order,
    flattened depth first, then `_gp_list_labels` stores them as "d"), without
    one Python float per label. None for anything else (an empty target, a
    NaN or infinity, another dtype or class), which keeps the label input
    step and its refusals."""
    if labels_float_storage(y, context, integral=False) != 0:
        return None
    var ndarray = py_get(context, "ndarray")
    var wide: Bool
    var address: Int
    if ndarray is not None and py_type(y) is ndarray:
        wide = String(y.dtype.str) == "<f8"
        address = _ndarray_buffer_address(y)
    else:
        wide = String(y.dtype) == "<f8"
        address = Int(py=y._addr)
    var rank = Int(py=y.ndim)
    var pyshape = y.shape
    var pystrides = y.strides
    var shape = List[Int]()
    var strides = List[Int]()
    var total = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        shape.append(Int(py=pyshape[k]))
        strides.append(Int(py=pystrides[k]))
        total *= shape[k]
    if total == 0:
        return None
    var store = List[Float64](length=total, fill=0.0)
    # The caller's target owns the buffer read here; `store` is new storage.
    label_float_gather_f64(address, wide, shape, strides, Int(store.unsafe_ptr()))
    _ = y  # the caller's target owns the buffer read above
    var result = _gp_prepare[S](Int(store.unsafe_ptr()), 1, total, False, base, context)
    _ = store^
    return result^


def _gp_list_labels[S: _TextCodeStorage](flat: PythonObject, base: PythonObject, context: PythonObject) raises -> Optional[_TextLabels[S]]:
    if len(flat) == 0:
        return None
    var kind = labels_list_scalar_kind(flat)
    if kind != LABELS_LIST_INT and kind != LABELS_LIST_FLOAT:
        return None
    var captured = gp_label_input(flat, kind)
    if not captured:
        return None
    var values = captured.take()
    var result = _gp_prepare[S](values.address(), values.code(), values.count(), False, base, context)
    _ = values
    return result^


def _gp_sorted_classes[S: _TextCodeStorage](labels: PythonObject, context: PythonObject) raises -> _TextLabels[S]:
    """First equal label retained; native keys where their order is equivalent."""
    var b = native_loaded_module("builtins")
    if len(labels) == 0:
        raise Error("mojolearn: y is empty")
    var seen_number = False
    var seen_string = False
    var builtin_integers = True
    for i in range(len(labels)):  # driver-loop: admitted native scalar labels only
        var value = labels[i]
        var kind = _gp_number_kind(value, context)
        var cls = py_type(value)
        builtin_integers = builtin_integers and (cls is b.int or cls is b.bool)
        # Input admission permits exact NumPy str_ as well as builtin str.
        # Their native Unicode slots preserve the original label identities.
        if family_is_str(value):
            seen_string = True
        elif kind < 3:
            seen_number = True
            if Bool(py=value != value):
                raise Error(String(_NAN_LABEL))
        else:
            raise Error("mojolearn: labels must be native numbers or strings without Python protocols")
    if seen_number and seen_string:
        raise Error("mojolearn: y mixes numeric and str labels; encode one kind before fitting")
    if seen_string:
        # The shared text plan retains the first original exact str/NumPy str_
        # object, including surrogate strings, and Python code-point ordering.
        var named = _text_list_labels_into[S](labels, context)
        if named:
            return named.take()
    if builtin_integers:
        return gp_integer_classes[S](labels, context)
    # Exact NumPy/mixed-real comparisons retain their existing native scalar
    # semantics; a mathematical integer/Float64 key would change those rules.
    var classes = gp_mixed_classes(labels)
    var table = gp_mixed_index(classes)
    if len(classes) > 2147483648:
        raise Error("TypeError: mojolearn: cannot build a <i4 Array from these values: signed integer is greater than maximum")
    var codes = S(len(labels), context)
    gp_mixed_codes(table, labels, codes.address())
    _ = labels
    return _TextLabels[S](classes, codes^, len(labels))


def gp_encode_labels_into[S: _TextCodeStorage](y: PythonObject, base: PythonObject, context: PythonObject) raises -> _TextLabels[S]:
    """`_labels.encode_labels(y)`: (classes list, int32 codes Array)."""
    classical_admit_value(y, context)
    var fast = _gp_buffer_labels[S](y, base, context)
    if fast:
        return fast.take()
    # A strided or matrix-shaped float target: one native buffer read in
    # place of the per-label snapshot and list walk below.
    fast = _gp_float_storage_labels[S](y, base, context)
    if fast:
        return fast.take()
    # perf3-label-doors (2026-11-06): the inputs below skip the snapshot,
    # flatten and per-label admission, each with the classes, codes and
    # refusals the walk gives it, so no bits change.
    # A numpy-backed pandas Series of numeric or bool storage: its stored
    # exact ndarray lists the same Python numbers in the same order (other
    # storage, e.g. datetime, lists differently and keeps the walk).
    var stored = labels_storage_source(y, context)
    if stored is not None and stored is not y and py_type(stored) is py_get(context, "ndarray"):
        var kind = String(stored.dtype.kind)
        if kind == "i" or kind == "u" or kind == "b" or kind == "f":
            return gp_encode_labels_into[S](stored, base, context)
    # An exact flat list / tuple of exact ints, floats or bools.
    fast = _gp_flat_list_labels[S](y, base, context)
    if fast:
        return fast.take()
    # An exact '<U' ndarray: UCS4 read from its buffer, native stable sort
    # by code point (Python's str order), the first of equal labels kept.
    var text = _text_buffer_labels_into[S](y, context)
    if text:
        # a local copy first, as trees_public_score.mojo:352 (pass 2, C7)
        return text.take()
    var source = gp_label_list(y, context) if py_type(y) is context["array_class"] else y
    var flat = _flatten_labels(source, context)
    classical_admit_value(flat, context)
    fast = _gp_list_labels[S](flat, base, context)
    if fast:
        return fast.take()
    # A flat list of exact str labels (subclasses and NumPy str_ keep the
    # ORDER RULE below): the native str reader's code-point order and
    # first-of-run classes are the ORDER RULE's.
    if labels_all_exact_str(flat):
        var named = _text_list_labels_into[S](flat, context)
        if named:
            return named.take()
    return _gp_sorted_classes[S](flat, context)


def _gp_flat_list_labels[S: _TextCodeStorage](y: PythonObject, base: PythonObject, context: PythonObject) raises -> Optional[_TextLabels[S]]:
    var kind = labels_list_scalar_kind(y)
    if kind == LABELS_LIST_OTHER:
        return None
    var captured = gp_label_input(y, kind)
    if not captured:
        return None
    var values = captured.take()
    var result = _gp_prepare[S](values.address(), values.code(), values.count(), values.as_bool, base, context)
    _ = values
    return result^


def gp_encode_labels(y: PythonObject, base: PythonObject, context: PythonObject) raises -> NativeTextLabels:
    var result = gp_encode_labels_into[_NativeTextCodes](y, base, context)
    var codes = NativeLabelCodes(0)
    swap(codes, result.codes.values)
    return NativeTextLabels(result.classes, codes^)


def gp_encode_labels_transport(y: PythonObject, base: PythonObject, context: PythonObject) raises -> ClassicalLabels:
    # Worker transport needs an owning exporter; write directly into it with
    # the same encoder, without a native-to-Array staging copy.
    var result = gp_encode_labels_into[_PythonTextCodes](y, base, context)
    return ClassicalLabels(result.classes, result.codes.values, result.count)


def _gp_number_kind(value: PythonObject, context: PythonObject) raises -> Int:
    """0 bool, 1 integer, 2 real, 3 refused: exact native scalar types only.
    Unknown types never reach a Python numeric ABC or conversion hook."""
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.bool:
        return 0
    if cls is b.int:
        return 1
    if cls is b.float:
        return 2
    if cls is b.str or cls is b.bytes:
        return 3
    var scalars = py_get(context, "scalar_types")
    if scalars is not None:
        for i in range(len(scalars)):  # driver-loop: over the connected NumPy scalar types, by identity
            if cls is scalars[i]:
                var kind = String(value.dtype.kind)
                if kind == "b":
                    return 0
                if kind == "i" or kind == "u":
                    return 1
                if kind == "f":
                    return 2
                return 3
    return 3


def _gp_label_kind(classes: PythonObject, context: PythonObject) raises -> String:
    """`_labels.label_kind`: "int", "float" or "object"."""
    var ints = True
    var reals = True
    for i in range(len(classes)):  # driver-loop: over the distinct classes
        var kind = _gp_number_kind(classes[i], context)
        ints = ints and kind == 1
        reals = reals and (kind == 1 or kind == 2)
    if ints:
        return "int"
    if reals:
        return "float"
    return "object"


def _gp_class_table(classes: PythonObject, context: PythonObject) raises -> GPLabelTable:
    var b = native_loaded_module("builtins")
    var kind = _gp_label_kind(classes, context)
    var native_kind = GP_LABEL_I64 if kind == "int" else GP_LABEL_F64 if kind == "float" else 0
    var table = GPLabelTable(native_kind, len(classes))
    if table.available:
        for i in range(len(classes)):  # driver-loop: UNPACK the distinct admitted class values
            # Keep the old builtin conversion before storage. An out-of-range
            # integer changes only the table route, never the public class.
            var value = family_call(b.int if native_kind == GP_LABEL_I64 else b.float, classes[i])
            try:
                if native_kind == GP_LABEL_I64:
                    table.integers[i] = Int64(Int(py=value))
                else:
                    table.reals[i] = Float64(py=value)
            except:
                table.refuse_numeric()
    return table^


def _gp_prepared_class_table(owner: PythonObject, classes: PythonObject, context: PythonObject) raises -> PythonObject:
    var mods = NativeModules()
    if owner is not None:
        var held = gpsvm_side_get(_GP_CLASS_TABLE_SLOT, owner)
        if held is not None:
            var typed = False
            try:
                _ = held.downcast_value_ptr[GPPreparedClassTable]()
                typed = True
            except:
                pass
            if typed:
                var cached = held.downcast_value_ptr[GPPreparedClassTable]()
                var same = len(cached[].labels) == len(classes)
                if same:
                    for i in range(len(classes)):  # driver-loop: exact original class identity/scalar equality policy
                        if not _label_same(mods, cached[].labels[i], classes[i]):
                            same = False
                            break
                if same:
                    return held
    classical_admit_value(classes, context)
    var table = _gp_class_table(classes, context)
    var pins = List[PythonObject]()
    for i in range(len(classes)):  # driver-loop: retain public class identities across RUN/PACK
        pins.append(classes[i])
    var prepared = GPPreparedClassTable(pins^, table^)
    var packed = PythonObject(alloc=prepared^)
    if owner is not None and _labels_recordable(mods, classes):
        _ = gpsvm_side_put(_GP_CLASS_TABLE_SLOT, owner, packed)
    return packed


def _gp_decode_with(classes: PythonObject, table: GPLabelTable, codes: Int, n: Int,
                    base: PythonObject, context: PythonObject) raises -> PythonObject:
    var k = len(classes)
    if table.available and k > 0:
        var dst = family_array(Python.tuple(n), String("<i8") if table.kind == GP_LABEL_I64 else String("<f8"), context)
        var connection = GPLabelsConnection(base)
        gp_labels_gather(connection, table.kind, LabelsGatherRequest(table.address(), codes, Int(py=classical_address(dst)), k, n))
        _ = dst
        _ = table
        return dst
    var pins = List[PythonObject]()
    return _gp_decode_objects[False](classes, codes, n, pins)


def _gp_decode_objects[native: Bool](classes: PythonObject, codes: Int, n: Int,
                                    mut pins: List[PythonObject]) raises -> PythonObject:
    var k = len(classes)
    var rows = FAMILY_I64P(unsafe_from_address=codes)
    # One PyList_New(n) and direct slot stores of the class objects (lane
    # p3-svm-gp-kernel, row 79): the k class pointers are read once, every
    # row stores a new reference to its class, no Python call per row.
    var class_items = List[PythonObject](capacity=k)
    for j in range(k):  # driver-loop: over the distinct classes (k pointers read once)
        class_items.append(classes[j])
    ref cpy = Python().cpython()
    var result = PythonObject(None)
    comptime if not native:
        var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(n)
        if not raw:
            raise cpy.unsafe_get_error()
        # The private NULL-filled list owns initialized slots on failure.
        result = PythonObject(from_owned=raw)
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(n):  # driver-loop: native code-to-label gather stores retained class objects
        var index = Int(rows[i])
        if index < 0 or index >= k:
            raise Error("IndexError: GP class code is outside its retained class table")
        comptime if native:
            pins.append(class_items[index])
        else:
            var item = class_items[index]._obj_ptr
            cpy.Py_IncRef(item)
            # SetItem steals the new reference on BOTH success and failure.
            if set_item(result._obj_ptr, i, item) != 0:
                raise cpy.unsafe_get_error()
    _ = codes
    _ = class_items^
    return result


comptime _GP_CLASS_TABLE_SLOT = "gp_class_table_v1"


def _gp_decode_prepared(owner: PythonObject, var classes: PythonObject, codes: Int, n: Int,
                        base: PythonObject, context: PythonObject) raises -> PythonObject:
    classes = classifier_label_list(classes, context)
    var packed = _gp_prepared_class_table(owner, classes, context)
    var prepared = packed.downcast_value_ptr[GPPreparedClassTable]()
    var result = _gp_decode_with(classes, prepared[].table, codes, n, base, context)
    _ = packed
    return result


def _gp_decode_prepared(owner: PythonObject, var classes: PythonObject, codes: PythonObject,
                        base: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve class admission/table lookup before reading transport metadata.
    classes = classifier_label_list(classes, context)
    var packed = _gp_prepared_class_table(owner, classes, context)
    var prepared = packed.downcast_value_ptr[GPPreparedClassTable]()
    var result = _gp_decode_with(classes, prepared[].table, Int(py=classical_address(codes)), Int(py=codes.size), base, context)
    _ = codes
    _ = packed
    return result


def gp_decode_labels(classes: PythonObject, codes: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var result = _gp_decode_prepared(PythonObject(None), classes, codes, base, context)
    _ = codes
    return result


def gp_decode_labels(owner: PythonObject, classes: PythonObject, codes: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var result = _gp_decode_prepared(owner, classes, codes, base, context)
    _ = codes
    return result


def gp_decode_labels(classes: PythonObject, codes: List[Int64], base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var result = _gp_decode_prepared(PythonObject(None), classes, Int(codes.unsafe_ptr()), len(codes), base, context)
    _ = codes
    return result


def gp_decode_labels(owner: PythonObject, classes: PythonObject, codes: List[Int64], base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var result = _gp_decode_prepared(owner, classes, Int(codes.unsafe_ptr()), len(codes), base, context)
    _ = codes
    return result


def gp_decode_score_labels(owner: PythonObject, var classes: PythonObject, codes: List[Int64],
                           base: PythonObject, context: PythonObject) raises -> GPDecodedScoreLabels:
    # Match public decode admission and cached table selection before truth
    # shape/refusal. Only predicted scalar identities enter the metric later.
    classes = classifier_label_list(classes, context)
    var packed = _gp_prepared_class_table(owner, classes, context)
    var prepared = packed.downcast_value_ptr[GPPreparedClassTable]()
    var numeric = prepared[].table.available and len(classes) > 0
    var result = GPDecodedScoreLabels(len(codes), prepared[].table.kind if numeric else 0)
    if numeric:
        var connection = GPLabelsConnection(base)
        gp_labels_gather(connection, prepared[].table.kind,
            LabelsGatherRequest(prepared[].table.address(), Int(codes.unsafe_ptr()),
                Int(result.words.unsafe_ptr()), len(classes), len(codes)))
        _ = connection
    else:
        _ = _gp_decode_objects[True](classes, Int(codes.unsafe_ptr()), len(codes), result.pins)
    _ = packed
    _ = classes
    _ = codes
    return result^
