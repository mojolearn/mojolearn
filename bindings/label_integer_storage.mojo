# SPDX-License-Identifier: Apache-2.0
# Integer label storage: the one input kind a target admission can accept
# without reading a single element.
#
# The public target checks (complex refusal, NaN/infinity, "Unknown label type:
# continuous", exact scalar admission) only ever refuse an element that is a
# complex, a non-finite or non-integral real, or a Python object. An exact
# NumPy ndarray or a native Array whose dtype is a fixed-width integer (or, when
# allowed, bool) holds none of those: `tolist()` of it yields exact Python ints
# (bools), each of which every one of those checks admits. Such a target is
# therefore admitted from its dtype metadata alone, without the per-element walk
# (tolist, one Python object and one admission per label) that cost about 2.3 s
# per 20000-label classifier fit through the GP label snapshot. Everything else
# (lists, tuples, float, complex, object, str, datetime, pandas, torch, sparse,
# subclasses) keeps its caller's existing element-by-element path.
from std.python import PythonObject
from bindings.py_balanced import py_type, py_get


def labels_integer_storage(value: PythonObject, context: PythonObject, allow_bool: Bool = True) raises -> Bool:
    """True for an exact ndarray or exact native Array of rank >= 1 whose
    dtype is a fixed-width signed or unsigned integer (or bool when
    `allow_bool`). Reads class identity and dtype metadata only; callers
    still admit the container itself (`classical_admit_value`) first."""
    var cls = py_type(value)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        var kind = String(value.dtype.kind)
        if Int(py=value.ndim) < 1:
            return False
        return kind == "i" or kind == "u" or (allow_bool and kind == "b")
    var array_class = py_get(context, "array_class")
    if array_class is not None and cls is array_class:
        if Int(py=value.ndim) < 1:
            return False
        var typestr = String(value.dtype)
        if typestr.startswith("<i") or typestr.startswith("<u") or typestr.startswith("|i") or typestr.startswith("|u"):
            return True
        return allow_bool and typestr == "|b1"
    return False
