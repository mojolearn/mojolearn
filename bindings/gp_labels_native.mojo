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
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_arrays import classical_address, classical_is_array
from bindings.family_common import (
    family_repr, family_isinstance, family_call, family_entry, family_array, FAMILY_F64P, FAMILY_I64P,
)
from bindings.archive_schema_native import _schema_label_store

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
    var classes = family_array(Python.tuple(n), "<f8" if wide else "<i8", context)
    var codes = family_array(Python.tuple(n), "<i4", context)
    var entry = family_entry(base, "prepare_targets", context)
    var k: Int
    try:
        k = Int(py=py_call(entry, PythonObject(address), PythonObject(code), PythonObject(n),
                           classical_address(classes), classical_address(codes), PythonObject(0),
                           PythonObject(0), PythonObject(0)))
    except error:
        if String(error).find("NaN label") != -1:
            raise Error(String(_NAN_LABEL))
        raise error^
    var values = Python.list()
    if wide:
        var fp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(classes)))
        for i in range(k):  # driver-loop: over the distinct classes
            _ = py_call(values.append, PythonObject(fp[i]))
    else:
        var ip = FAMILY_I64P(unsafe_from_address=Int(py=classical_address(classes)))
        for i in range(k):  # driver-loop: over the distinct classes
            if as_bool:
                _ = py_call(values.append, PythonObject(ip[i] != 0))
            else:
                _ = py_call(values.append, PythonObject(Int(ip[i])))
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


def _gp_is_nested(item: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return family_isinstance(item, Python.tuple(b.list, b.tuple))


def _gp_any_nested(items: PythonObject) raises -> Bool:
    """`any(map(_is_nested, items))`."""
    for i in range(len(items)):  # cpu-route: Python label objects, the explicit label input step
        if _gp_is_nested(items[i]):
            return True
    return False


def _gp_plain(labels: PythonObject) raises -> String:
    """`_plain_labels`: "number", "str" or "" (the type set)."""
    var b = Python.import_module("builtins")
    if len(labels) == 0:
        return ""
    var kinds = py_call(b.set, py_call(b.map, b.type, labels))
    if Bool(py=py_call(kinds.issubset, Python.tuple(b.int, b.float, b.bool))):
        return "number"
    if len(kinds) == 1 and py_contains(kinds, b.str):
        return "str"
    return ""


def _gp_flatten(y: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels(y)`: a flat Python list of label objects."""
    var b = Python.import_module("builtins")
    var items = y
    if Bool(py=py_call(b.hasattr, y, "tolist")) and not _gp_is_nested(y):
        items = family_call(py_call(b.getattr, y, PythonObject("tolist")))
    if family_isinstance(items, Python.tuple(b.str, b.bytes)):
        raise Error("mojolearn: y must be a sequence of labels")
    var ty = py_type(items)
    if (ty is b.list or ty is b.tuple) and _gp_plain(items) != "":
        return py_call(b.list, items)
    var level = Python.list(items)
    while _gp_any_nested(level):  # driver-loop: over the label nesting levels (depth-sized)
        var flat = Python.list()
        for i in range(len(level)):  # cpu-route: Python label objects, the explicit label input step
            var item = level[i]
            if _gp_is_nested(item):
                _ = py_call(flat.extend, item)
            else:
                _ = py_call(flat.append, item)
        level = flat
    return level


def _gp_list_labels(flat: PythonObject, base: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_encode_label_list_native`: a flat list of exact ints (within int64)
    or exact floats through `prepare_targets`, else None."""
    var b = Python.import_module("builtins")
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


def _gp_no_nan(labels: PythonObject) raises:
    """`any(map(operator.ne, labels, labels))` refused: a NaN label."""
    var b = Python.import_module("builtins")
    var ne = Python.import_module("operator").ne
    if Bool(py=family_call(b.any, py_call(b.map, ne, labels, labels))):
        raise Error(String(_NAN_LABEL))


def _gp_check_kinds(labels: PythonObject) raises:
    """`_check_label_kinds`: the first label that is neither a number nor a
    str (or a NaN before it), then a mix of the two kinds."""
    var b = Python.import_module("builtins")
    var real = Python.import_module("numbers").Real
    var types = py_call(b.list, py_call(b.set, py_call(b.map, b.type, labels)))
    var refused = py_call(b.set)
    var seen_str = False
    var seen_number = False
    for i in range(len(types)):  # driver-loop: over the distinct label types (types-sized)
        var t = types[i]
        if Bool(py=family_call(b.issubclass, t, b.str)):
            seen_str = True
        elif Bool(py=family_call(b.issubclass, t, real)) or String(t.__name__) == "bool_":
            seen_number = True
        else:
            _ = py_call(refused.add, t)
    var stop = -1
    if len(refused) > 0:
        try:
            stop = Int(py=py_call(Python.import_module("operator").indexOf,
                                  py_call(b.map, refused.__contains__, py_call(b.map, b.type, labels)), PythonObject(True)))
        except:
            stop = -1
    var head = labels if stop < 0 else py_call(labels.__getitem__, py_call(b.slice, PythonObject(0), PythonObject(stop)))
    _gp_no_nan(head)
    if stop >= 0:
        raise Error("mojolearn: label " + family_repr(labels[stop]) + " is neither a number nor a str")
    if seen_str and seen_number:
        raise Error("mojolearn: y mixes numeric and str labels; encode one kind before fitting")


def _gp_sorted_classes(labels: PythonObject, context: PythonObject) raises -> PythonObject:
    """`sorted_classes(labels)` with codes as an int32 Array: the first of
    equal labels kept, the classes sorted by Python's own comparison."""
    var b = Python.import_module("builtins")
    if len(labels) == 0:
        raise Error("mojolearn: y is empty")
    var kind = _gp_plain(labels)
    if kind == "number":
        _gp_no_nan(labels)
    elif kind == "":
        _gp_check_kinds(labels)
    var classes = family_call(b.sorted, py_call(b.dict.fromkeys, labels))
    var table = py_call(b.dict, py_call(b.zip, classes, py_call(b.range, PythonObject(len(classes)))))
    var codes = family_call(b.list, py_call(b.map, table.__getitem__, labels))
    return Python.tuple(classes, _schema_label_store(codes, "i", "<i4", context))


def gp_encode_labels(y: PythonObject, base: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.encode_labels(y)`: (classes list, int32 codes Array)."""
    var fast = _gp_buffer_labels(y, base, context)
    if fast is not None:
        return fast
    var flat = _gp_flatten(y)
    fast = _gp_list_labels(flat, base, context)
    if fast is not None:
        return fast
    return _gp_sorted_classes(flat, context)


def _gp_is_bool(value: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return family_isinstance(value, b.bool) or String(py_type(value).__name__) == "bool_"


def _gp_label_kind(classes: PythonObject) raises -> String:
    """`_labels.label_kind`: "int", "float" or "object"."""
    var numbers = Python.import_module("numbers")
    var ints = True
    var reals = True
    for i in range(len(classes)):  # driver-loop: over the distinct classes
        var c = classes[i]
        var plain = not _gp_is_bool(c)
        ints = ints and plain and family_isinstance(c, numbers.Integral)
        reals = reals and plain and family_isinstance(c, numbers.Real)
    if ints:
        return "int"
    if reals:
        return "float"
    return "object"


def gp_decode_labels(classes: PythonObject, codes: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for int64 codes (an Array of
    shape (n,)): an int64 / float64 Array through the base binding's gather
    for int / float classes, else the list of label objects."""
    var b = Python.import_module("builtins")
    var kind = _gp_label_kind(classes)
    var k = len(classes)
    var n = Int(py=codes.size)
    if (kind == "int" or kind == "float") and k > 0:
        var wide = Python.list()
        for i in range(k):  # driver-loop: over the distinct classes
            _ = py_call(wide.append, family_call(b.int if kind == "int" else b.float, classes[i]))
        var table = PythonObject(None)
        try:
            table = py_call(context["array_type"], PythonObject("q" if kind == "int" else "d"), wide)
        except:
            table = PythonObject(None)  # a class outside int64: the label objects come back
        if table is not None:
            var out = family_array(Python.tuple(n), "<i8" if kind == "int" else "<f8", context)
            _ = family_call(family_entry(base, "gather_i64" if kind == "int" else "gather_f64", context),
                            py_call(table.buffer_info)[0], PythonObject(k), classical_address(codes),
                            PythonObject(n), classical_address(out))
            _ = table^
            return out
    var rows = family_call(py_call(b.memoryview, codes._store).tolist)
    return family_call(b.list, py_call(b.map, classes.__getitem__, rows))
