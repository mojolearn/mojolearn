# SPDX-License-Identifier: Apache-2.0
# One-read label sources for the target admission doors (lane perf2-labels,
# 2026-11-06).
#
# `bindings/label_integer_storage.mojo` admits an exact ndarray or native Array
# from its dtype metadata (`labels_integer_storage`) or one native buffer pass
# (`labels_float_storage`). Users also pass y as a Python list or tuple, a
# pandas Series or a str ndarray; those still took the door's per-label walk
# (flatten, one Python object and several scalar checks per label). The
# helpers below answer, without that walk, for the inputs whose every element
# the walk is known to admit:
#
# - `labels_storage_source`: the exact ndarray a numpy-backed pandas Series
#   lists (its single block's values, what `tolist()` reads), or the exact
#   ndarray / Array itself; None for anything else. A door hands the result to
#   `labels_integer_storage` / `labels_float_storage`: a Series' labels are
#   exactly its values' labels, in the same order.
# - `labels_list_integer_storage`: an exact flat list or tuple whose every
#   item is an exact int (or bool when allowed), read through the container's
#   C storage with one type-pointer compare per item: no item is boxed into a
#   Python call, no Python hook runs.
# - `labels_text_storage`: an exact ndarray of native little-endian unicode
#   ('<U'): every element lists as an exact str.
#
# These are new functions; the existing signatures in
# label_integer_storage.mojo are unchanged. Callers still admit the container
# itself (`classical_admit_value`) first, exactly as before.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get


# Kinds returned by `labels_list_scalar_kind`.
comptime LABELS_LIST_OTHER = 0
comptime LABELS_LIST_BOOL = 2
comptime LABELS_LIST_INT = 3
comptime LABELS_LIST_FLOAT = 4


def labels_list_scalar_kind(value: PythonObject) raises -> Int:
    """LABELS_LIST_INT when `value` is an exact nonempty list or tuple whose
    every item is exactly an int, LABELS_LIST_FLOAT when every item is
    exactly a float, LABELS_LIST_BOOL when every item is exactly a bool;
    LABELS_LIST_OTHER otherwise (empty, nested, mixed, subclasses, any other
    object). Items are read borrowed from the container's C storage and their
    classes compared by pointer, stopping at the first other item."""
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var list_type = b.list
    var tuple_type = b.tuple
    var is_tuple = cls is tuple_type
    if cls is not list_type and not is_tuple:
        return LABELS_LIST_OTHER
    var n = len(value)
    if n == 0:
        return LABELS_LIST_OTHER
    ref cpy = Python().cpython()
    var get_item = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem" if is_tuple else "PyList_GetItem")
    var int_type = b.int
    var float_type = b.float
    var bool_type = b.bool
    var kind = LABELS_LIST_OTHER
    for i in range(n):  # driver-loop: exact list/tuple item classes by pointer, one C read per label
        var raw = get_item(value._obj_ptr, i)
        if not raw:
            raise cpy.unsafe_get_error()
        var item_type = cpy.PyObject_Type(raw)
        var item_kind = LABELS_LIST_OTHER
        if item_type == int_type._obj_ptr:
            item_kind = LABELS_LIST_INT
        elif item_type == float_type._obj_ptr:
            item_kind = LABELS_LIST_FLOAT
        elif item_type == bool_type._obj_ptr:
            item_kind = LABELS_LIST_BOOL
        cpy.Py_DecRef(item_type)
        if item_kind == LABELS_LIST_OTHER or (i > 0 and item_kind != kind):
            kind = LABELS_LIST_OTHER
            break
        kind = item_kind
    _ = value  # keep the container alive through the borrowed item reads above
    _ = int_type
    _ = float_type
    _ = bool_type
    return kind


def labels_list_integer_storage(value: PythonObject, allow_bool: Bool = True) raises -> Bool:
    """True for an exact nonempty flat list or tuple of exact ints (or, when
    `allow_bool`, of exact bools). Such a target holds no complex, non-finite
    or non-integral element and no Python object with protocols, so every
    target check of the label walk admits each element: the door admits it
    without the walk. A list mixing ints and bools is not taken (it keeps the
    walk, whose class rules differ for mixed bool/int labels)."""
    var kind = labels_list_scalar_kind(value)
    return kind == LABELS_LIST_INT or (allow_bool and kind == LABELS_LIST_BOOL)


def labels_series_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The exact ndarray an exact single-block pandas Series stores (what its
    `tolist()` lists, read through pandas' native slots), or None: not an
    exact Series, extension storage (nullable, categorical, string, arrow),
    or no NumPy connected. A DataFrame is never taken (its block is stored
    transposed)."""
    var ndarray = py_get(context, "ndarray")
    if ndarray is None:
        return PythonObject(None)
    var series = _pd_class("pandas.core.series", "Series")
    if series is None or py_type(value) is not series:
        return PythonObject(None)
    var values = PythonObject(None)
    try:
        var manager = py_get(_pd_fields(value), "_mgr")
        var blocks = py_get(_pd_fields(manager), "blocks")
        if py_type(blocks) is not native_loaded_module("builtins").tuple or len(blocks) != 1:
            return PythonObject(None)
        values = _pd_get("pandas._libs.internals", "Block", "values", blocks[0])
    except:
        return PythonObject(None)
    if py_type(values) is not ndarray:
        return PythonObject(None)
    return values


def labels_storage_source(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The exact ndarray or native Array whose labels are `value`'s labels in
    the same C order: `value` itself when it is one, the stored ndarray of a
    numpy-backed pandas Series, else None. Pass the result (when not None) to
    `labels_integer_storage` / `labels_float_storage`; the caller holds
    `value` across those reads."""
    var cls = py_type(value)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        return value
    var array_class = py_get(context, "array_class")
    if array_class is not None and cls is array_class:
        return value
    return labels_series_values(value, context)


def labels_text_storage(value: PythonObject, context: PythonObject) raises -> Bool:
    """True for an exact ndarray of rank >= 1 whose dtype is native
    little-endian unicode ('<U'): `tolist()` of it yields exact Python str
    only, so a door whose walk admits every str label admits it from its
    dtype metadata. Reads class identity and dtype metadata only."""
    var ndarray = py_get(context, "ndarray")
    if ndarray is None or py_type(value) is not ndarray:
        return False
    if Int(py=value.ndim) < 1:
        return False
    return String(value.dtype.str).startswith("<U")
