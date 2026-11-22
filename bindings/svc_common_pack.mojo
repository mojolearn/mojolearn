# SPDX-License-Identifier: Apache-2.0
# PACK pieces shared by the SVM family doors: the compact public arrays of a
# fit (`svc_prefix`, `svc_scalar_array`), the decode of predicted codes into
# the fitted label types (an int64 Array for int classes, a float64 Array
# for real classes, the list of label objects otherwise; the class table is
# built once per class list), and the int64 codes of a decision (threshold,
# row argmax). Every compiled step is the one the Python called: the base
# binding's host helpers (`gather_i64`, `threshold_labels_i64`,
# `argmax_rows_f32`, ...), resolved through `_buffer._native` (the module
# registers it as `family_svc_native`).
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_item
from bindings.py_attr import py_attr
from bindings.classical_arrays_door import classical_address, classical_wrap_array
from bindings.label_admission_native_pack import classifier_label_list
from bindings.gp_svm_records_door import gpsvm_class_table, gpsvm_class_table_put
from bindings.p3_labels_metrics_native_door import p3_decode_list
from bindings.labels_descriptor_boundary_door import labels_gather
from bindings.family_common_unpack import (
    FAMILY_F32P,
    FAMILY_F64P,
    FAMILY_I64P,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from bindings.svc_common_unpack import svc_builtins, svc_base, svc_label_kind
from bindings.svm_keys_unpack import svm_keys


# ---------------------------------------------------------------- arrays


def svc_prefix(source: PythonObject, count: Int, shape: PythonObject, typestr: String,
               context: PythonObject) raises -> PythonObject:
    """`source[:count].reshape(shape)`: a new compact C-order Array holding
    the first `count` items of the C-order `source` (an exact copy)."""
    var out = family_array(shape, typestr, context)
    var width = 8 if typestr == "<f8" or typestr == "<i8" else 4
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source))),
               count=count * width)
        _ = out  # keep the Python object alive through the native call above
    return out


def svc_prefix(source: Int, count: Int, shape: PythonObject, typestr: String,
               context: PythonObject) raises -> PythonObject:
    """PACK a native owner with the existing one-copy compact Array layout."""
    var result = family_array(shape, typestr, context)
    var width = 8 if typestr == "<f8" or typestr == "<i8" else 4
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))),
            src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=source), count=count * width)
        _ = result
    return result


def svc_scalar_array(value: Float64, typestr: String, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([value], typestr)` for "<f4" or "<f8"."""
    var out = family_array(Python.tuple(1), typestr, context)
    if typestr == "<f4":
        FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out)))[0] = Float32(value)
        _ = out  # keep the Python object alive through the native call above
    else:
        FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out)))[0] = value
        _ = out  # keep the Python object alive through the native call above
    return out


def _svc_class_table(classes: PythonObject, kind: Int, context: PythonObject) raises -> PythonObject:
    """The int64 (kind 0) / float64 (kind 1) gather table of the label list
    `classes`, or None when a class does not pack (or kind 2)."""
    var b = svc_builtins()
    var k = len(classes)
    if kind >= 2 or k == 0:
        return PythonObject(None)
    var table = family_array(Python.tuple(PythonObject(k)), "<i8" if kind == 0 else "<f8", context)
    for i in range(k):  # driver-loop: over the fitted classes
        try:
            if kind == 0:
                var value = py_call(b.int, classes[i])
                if Bool(py=value < PythonObject(-9223372036854775807 - 1)) or Bool(py=value > PythonObject(9223372036854775807)):
                    return PythonObject(None)
                FAMILY_I64P(unsafe_from_address=Int(py=classical_address(table)))[i] = Int64(Int(py=value))
                _ = table  # keep the Python object alive through the native call above
            else:
                FAMILY_F64P(unsafe_from_address=Int(py=classical_address(table)))[i] = Float64(py=py_call(b.float, classes[i]))
                _ = table  # keep the Python object alive through the native call above
        except:
            return PythonObject(None)
    return table


def _svc_decode_with(classes: PythonObject, kind: Int, table: PythonObject, codes: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var b = svc_builtins()
    var n = Int(py=codes.size)
    var k = len(classes)
    if table is not None:
        var out = family_array(Python.tuple(PythonObject(n)), "<i8" if kind == 0 else "<f8", context)
        if n > 0:
            _ = family_call(svc_base("gather_i64" if kind == 0 else "gather_f64", context), classical_address(table),
                            PythonObject(k), classical_address(codes), PythonObject(n), classical_address(out))
            _ = table  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
        return out
    # the label output step: the class objects, one per code (C-level map
    # over the int64 store)
    var values = family_call(b.list, family_call(b.map, py_attr(classes, "__getitem__"), codes._store))
    if kind == 2:
        return values
    var nk = svm_keys()
    try:
        var store = py_call(py_item(context, nk[].k_array_type), PythonObject("q" if kind == 0 else "d"),
                            py_call(b.map, b.int if kind == 0 else b.float, values))
        return classical_wrap_array(store, Python.tuple(PythonObject(n)), "<i8" if kind == 0 else "<f8", "C", context)
    except:
        return values


def svc_decode(var classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for an int64 code Array this
    owner made: an int64 Array for int classes, a float64 Array for real
    classes (the base binding's `gather_i64` / `gather_f64`), the list of
    label objects otherwise."""
    classes = classifier_label_list(classes, context)
    var kind = svc_label_kind(classes, context)
    return _svc_decode_with(classes, kind, _svc_class_table(classes, kind, context), codes, context)


comptime _SVC_CLASS_TABLE_SLOT = "svc_class_table_v1"


def svc_cached_class_table(owner: PythonObject, classes: PythonObject, context: PythonObject) raises -> PythonObject:
    var built = gpsvm_class_table(owner, _SVC_CLASS_TABLE_SLOT, classes)
    if built is None:
        var kind = svc_label_kind(classes, context)
        built = Python.tuple(PythonObject(kind), _svc_class_table(classes, kind, context))
        gpsvm_class_table_put(owner, _SVC_CLASS_TABLE_SLOT, classes, built)
    return built


def svc_decode(owner: PythonObject, var classes: PythonObject, codes: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`svc_decode(classes, codes)` for the fitted `owner`: the label kind and
    the gather table are built once per class list (the same label objects),
    kept in a weak-keyed native record (bindings/gp_svm_records.mojo)
    instead of being rebuilt with a per-class conversion on every predict."""
    classes = classifier_label_list(classes, context)
    var built = svc_cached_class_table(owner, classes, context)
    return _svc_decode_with(classes, Int(py=built[0]), built[1], codes, context)


def svc_decode(owner: PythonObject, var classes: PythonObject, codes: List[Int64],
               context: PythonObject) raises -> PythonObject:
    """PACK native code storage with the same weak class table and label types."""
    classes = classifier_label_list(classes, context)
    var built = svc_cached_class_table(owner, classes, context)
    var kind = Int(py=built[0])
    var table = built[1]
    var n = len(codes)
    var address = Int(codes.unsafe_ptr()) if n else 0
    if table is not None:
        var result = family_array(Python.tuple(n), "<i8" if kind == 0 else "<f8", context)
        if n > 0:
            labels_gather(context, 1 if kind == 0 else 2,
                Int(py=classical_address(table)), len(classes), address, n,
                Int(py=classical_address(result)))
        _ = codes
        _ = table
        return result
    var values = p3_decode_list(classes, address, 8, n)
    _ = codes
    if values is None:
        raise Error("IndexError: list index out of range")
    if kind == 2:
        return values
    # Preserve the old numeric-label PACK attempt and overflow behavior:
    # an integer outside int64 remains a list of its original Python labels.
    var b = svc_builtins()
    var nk = svm_keys()
    try:
        var store = py_call(py_item(context, nk[].k_array_type), PythonObject("q" if kind == 0 else "d"),
            py_call(b.map, b.int if kind == 0 else b.float, values))
        return classical_wrap_array(store, Python.tuple(n), "<i8" if kind == 0 else "<f8", "C", context)
    except:
        return values


def svc_argmax_rows(scores: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order (n, c) float32 or float64 Array."""
    var n = Int(py=scores.shape[0])
    var c = Int(py=scores.shape[1])
    var out = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    if n == 0:
        return out
    if c == 0:
        raise Error("argmax_rows: scores have no columns")
    var name = "argmax_rows_f32" if String(scores.dtype) == "<f4" else "argmax_rows_f64"
    _ = family_call(svc_base(name, context), classical_address(scores), PythonObject(n), PythonObject(c),
                    classical_address(out))
    _ = out  # keep the Python object alive through the native call above
    return out

