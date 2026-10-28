# SPDX-License-Identifier: Apache-2.0
# A categorical target's exported dtype/finiteness without invoking pandas
# Categorical.__array__ or converting ignored labels. Both original buffers
# remain pinned by the returned code snapshot's owner pack.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get
from bindings.classical_arrays import classical_input
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.prep_metadata_pyapi import _array_snapshot, _description, prep_numeric_vector
from core.prep_boundary_impl import prep_category_flags, prep_category_export


def prep_ndarray_backed(value: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    var admitted = cls is _pd_class("pandas.core.arrays.string_", "StringArray") or cls is _pd_class("pandas.core.arrays.datetimes", "DatetimeArray") or cls is _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray") or cls is _pd_class("pandas.core.arrays.period", "PeriodArray")
    if not admitted:
        return value
    return _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", value)


def _category_capture(value: PythonObject, context: PythonObject) raises -> Tuple[ArrayMetadata, ArrayMetadata, Int, Int, Bool]:
    var raw_codes = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", value)
    if py_type(raw_codes) is not context["ndarray"]:
        raise Error("TypeError: pandas categorical codes require native array storage")
    var codes = _array_snapshot(raw_codes, context)
    if len(codes.shape) != 1 or not codes.castable or codes.source.kind != 1:
        raise Error("TypeError: pandas categorical codes require a signed native vector")
    var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", value)
    if py_type(dtype) is not _pd_class("pandas.core.dtypes.dtypes", "CategoricalDtype"):
        raise Error("TypeError: pandas categorical dtype requires native metadata")
    var categories = py_get(_pd_fields(dtype), "_categories")
    var cls = py_type(categories)
    var raw = PythonObject(None)
    if cls is _pd_class("pandas.core.indexes.range", "RangeIndex"):
        raw = py_get(_pd_fields(categories), "_range")
        if py_type(raw) is not native_loaded_module("builtins").range:
            raise Error("TypeError: pandas category range requires native range storage")
        raw = classical_input(raw, context)
    elif cls is _pd_class("pandas.core.indexes.base", "Index") or cls is _pd_class("pandas.core.indexes.datetimes", "DatetimeIndex") or cls is _pd_class("pandas.core.indexes.timedeltas", "TimedeltaIndex") or cls is _pd_class("pandas.core.indexes.period", "PeriodIndex"):
        raw = prep_ndarray_backed(py_get(_pd_fields(categories), "_data"))
    else:
        raise Error("TypeError: pandas categories require native index storage")
    if py_type(raw) is not context["ndarray"]:
        raise Error("TypeError: pandas categories require native array storage")
    var labels = _array_snapshot(raw, context)
    if len(labels.shape) != 1:
        raise Error("TypeError: pandas categories require a one-dimensional native buffer")
    var number = labels.dtype_number
    var width = labels.source.itemsize
    if (number == 11 and width != 4) or (number == 12 and width != 8):
        raise Error("TypeError: pandas categorical float dtype has inconsistent itemsize")
    var flags = prep_category_flags(codes.source, codes.count, codes.native_byte_order,
        labels.source, labels.count, labels.native_byte_order, number)
    var finite = flags[1]
    if flags[0]:
        # pandas take fills missing numeric categories with NaN. Bool needs
        # object promotion; half retains its old outer no-finite-scan rule.
        if number >= 1 and number <= 10:
            number = 12
            width = 8
        elif number == 0:
            number = 17
            width = 8
        if number == 11 or number == 12:
            finite = False
    # Period categories export object values, not their physical int64 codes.
    if cls is _pd_class("pandas.core.indexes.period", "PeriodIndex"):
        number = 17
        width = 8
    codes.owner = Python.tuple(codes.owner, value, dtype, categories, labels.keep, labels.owner)
    return (codes^, labels^, number, width, finite)


def prep_category_target(value: PythonObject, context: PythonObject) raises -> Tuple[ArrayMetadata, Int, Int, Bool]:
    var captured = _category_capture(value, context)
    return (captured[0].copy(), captured[2], captured[3], captured[4])


def prep_category_array(value: PythonObject, context: PythonObject) raises -> ArrayMetadata:
    var captured = _category_capture(value, context)
    ref codes = captured[0]
    ref labels = captured[1]
    var number = captured[2]
    var width = captured[3]
    var owners = Python.tuple(value, codes.keep, codes.owner, labels.keep, labels.owner)
    if number > 12 and number != 23:
        var result = _description(value, codes.shape.copy(), number, owners)
        result.source.itemsize = width
        result.dtype_name = labels.dtype_name
        return result^
    if not labels.castable:
        raise Error("TypeError: categorical numeric storage cannot be cast")
    var result = prep_numeric_vector(codes.count, number, width, owners, context)
    prep_category_export(codes.source, codes.count, codes.native_byte_order,
        labels.source, labels.count, labels.native_byte_order, result.source, number)
    return result^
