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
# The PACK half of the former `bindings/label_input_sources.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/label_input_sources_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.label_sequence_source_unpack import label_source_count, label_source_raw
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get
from bindings.p3_labels_metrics_native_door import p3_item


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
