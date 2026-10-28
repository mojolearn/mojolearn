# SPDX-License-Identifier: Apache-2.0
# `_labels.encode_labels` and `_labels.decode_labels` for the Gaussian process
# classifier's owner (bindings/gp_classifier_native.mojo), under the ORDER RULE
# python/mojolearn/_labels.py states. A numeric label vector (a C-ordered Array
# or ndarray of a native dtype, or a flat list of exact ints or exact floats) is
# encoded by ONE call of the base binding's `prepare_targets` (the device encoder
# on a GPU install, the core host twin on a CPU-only one), the call `encode_labels`
# made; str and other Python label objects take the label input step with the
# same builtins `sorted_classes` used (dict.fromkeys, sorted, map). Integer and
# float classes decode through the base binding's `gather_i64` / `gather_f64`;
# str and bool classes decode to the label objects. The shell hands over the
# base binding (`_backend.binding("_mojolearn", "identical")`, `_buffer._native`'s).
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address, classical_is_array
from bindings.family_common import (
    family_call, family_is_str, family_array, FAMILY_F64P, FAMILY_I64P,
)
from bindings.archive_schema_native import _schema_label_store
from bindings.classical_pandas import classical_admit_value
from bindings.classical_labels import _flatten_labels
from bindings.classifier_public_lifecycle import classifier_label_list
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry

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


def _gp_prepare(address: Int, code: Int, n: Int, as_bool: Bool, base: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_labels.prepare_targets(labels, None, n)` plus `classes.tolist()`:
    (the k sorted classes as a Python list, the n int32 codes)."""
    var wide = code == 0 or code == 1
    var classes = family_array(Python.tuple(n), String("<f8") if wide else String("<i8"), context)
    var codes = family_array(Python.tuple(n), "<i4", context)
    var entry = neural_descriptor_entry(base, "prepare_targets")
    var k: Int
    try:
        k = Int(py=py_call(entry, PythonObject(address), PythonObject(code), PythonObject(n),
                           classical_address(classes), classical_address(codes), PythonObject(0),
                           PythonObject(0), PythonObject(0)))
        _ = codes  # keep the Python object alive through the native call above
        _ = classes  # keep the Python object alive through the native call above
    except error:
        if String(error).find("NaN label") != -1:
            raise Error(String(_NAN_LABEL))
        raise Error(String(error))
    var values = Python.list()
    if wide:
        var fp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(classes)))
        _ = classes  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct classes
            _ = py_call(values.append, PythonObject(fp[i]))
            _ = classes  # keep the Python object alive through the native call above
    else:
        var ip = FAMILY_I64P(unsafe_from_address=Int(py=classical_address(classes)))
        _ = classes  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct classes
            if as_bool:
                _ = py_call(values.append, PythonObject(Bool(ip[i] != 0)))
                _ = classes  # keep the Python object alive through the native call above
            else:
                _ = py_call(values.append, PythonObject(Int(ip[i])))
                _ = classes  # keep the Python object alive through the native call above
    _ = classes^
    return Python.tuple(values, codes)


def _gp_buffer_labels(y: PythonObject, base: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_encode_labels_native(y)`: a C-ordered numeric vector Array or ndarray
    through `prepare_targets`, or None for what the label input step reads
    (a list, a strided or non-native buffer, a matrix, an empty vector: every
    one of them yields the same classes and codes there)."""
    var ty = py_type(y)
    var ndarray = py_get(context, "ndarray")
    var is_array = classical_is_array(y, context)
    if not is_array and (ndarray is None or not (ty is ndarray)):
        return PythonObject(None)
    var typestr = String(y.dtype) if is_array else String(y.dtype.str)
    var code = _gp_native_code(typestr)
    var size = Int(py=y.size)
    if code < 0 or size == 0:
        return PythonObject(None)
    var contiguous = (String(y.order) == "C" or Int(py=y.ndim) < 2) if is_array else Bool(py=y.flags.c_contiguous)
    if not contiguous:
        return PythonObject(None)
    var longest = 0
    for k in range(len(y.shape)):  # driver-loop: over array rank metadata
        longest = max(longest, Int(py=y.shape[k]))
    if size != longest:
        return PythonObject(None)
    return _gp_prepare(Int(py=classical_address(y)), code, size, typestr == "|b1", base, context)


def _gp_list_labels(flat: PythonObject, base: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_encode_label_list_native`: a flat list of exact ints (within int64)
    or exact floats through `prepare_targets`, else None."""
    var b = native_loaded_module("builtins")
    if len(flat) == 0:
        return PythonObject(None)
    var kinds = py_call(b.set, py_call(b.map, b.type, flat))
    if len(kinds) != 1:
        return PythonObject(None)
    var code = String("")
    if py_contains(kinds, b.int):
        code = "q"
    elif py_contains(kinds, b.float):
        code = "d"
    else:
        return PythonObject(None)
    var store = PythonObject(None)
    try:
        store = py_call(context["array_type"], PythonObject(code), flat)
    except:
        return PythonObject(None)  # an int outside int64: a class only Python can hold
    var address = Int(py=py_call(store.buffer_info)[0])
    var result = _gp_prepare(address, 3 if code == "q" else 1, len(flat), False, base, context)
    _ = store^
    return result


def _gp_sorted_classes(labels: PythonObject, context: PythonObject) raises -> PythonObject:
    """`sorted_classes(labels)` with codes as an int32 Array: the first of
    equal labels kept, the classes sorted by Python's own comparison."""
    var b = native_loaded_module("builtins")
    if len(labels) == 0:
        raise Error("mojolearn: y is empty")
    var seen_number = False
    var seen_string = False
    for i in range(len(labels)):  # driver-loop: admitted native scalar labels only
        var value = labels[i]
        var kind = _gp_number_kind(value, context)
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
    var classes = family_call(b.sorted, py_call(b.dict.fromkeys, labels))
    var table = py_call(b.dict, py_call(b.zip, classes, py_call(b.range, PythonObject(len(classes)))))
    var codes = family_call(b.list, py_call(b.map, py_attr(table, "__getitem__"), labels))
    return Python.tuple(classes, _schema_label_store(codes, "i", "<i4", context))


def gp_encode_labels(y: PythonObject, base: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.encode_labels(y)`: (classes list, int32 codes Array)."""
    classical_admit_value(y, context)
    var fast = _gp_buffer_labels(y, base, context)
    if fast is not None:
        return fast
    var flat = _flatten_labels(y, context)
    classical_admit_value(flat, context)
    fast = _gp_list_labels(flat, base, context)
    if fast is not None:
        return fast
    return _gp_sorted_classes(flat, context)


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


def gp_decode_labels(var classes: PythonObject, codes: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for int64 codes (an Array of
    shape (n,)): an int64 / float64 Array through the base binding's gather
    for int / float classes, else the list of label objects."""
    var b = native_loaded_module("builtins")
    classes = classifier_label_list(classes, context)
    classical_admit_value(classes, context)
    var kind = _gp_label_kind(classes, context)
    var k = len(classes)
    var n = Int(py=codes.size)
    if (kind == "int" or kind == "float") and k > 0:
        var wide = Python.list()
        for i in range(k):  # driver-loop: over the distinct classes
            _ = py_call(wide.append, family_call(b.int if kind == "int" else b.float, classes[i]))
        var table = PythonObject(None)
        try:
            table = py_call(context["array_type"], PythonObject(String("q") if kind == "int" else String("d")), wide)
        except:
            table = PythonObject(None)  # a class outside int64: the label objects come back
        if table is not None:
            var out = family_array(Python.tuple(n), String("<i8") if kind == "int" else String("<f8"), context)
            _ = family_call(neural_descriptor_entry(base, String("gather_i64") if kind == "int" else String("gather_f64")),
                            py_call(table.buffer_info)[0], PythonObject(k), classical_address(codes),
                            PythonObject(n), classical_address(out))
            _ = out  # keep the Python object alive through the native call above
            _ = table^
            return out
    var rows = FAMILY_I64P(unsafe_from_address=Int(py=classical_address(codes)))
    var result = Python.list()
    for i in range(n):  # driver-loop: native code-to-label gather retains output objects
        var index = Int(rows[i])
        if index < 0 or index >= k:
            raise Error("IndexError: GP class code is outside its retained class table")
        _ = py_call(result.append, classes[index])
        _ = codes
    return result
