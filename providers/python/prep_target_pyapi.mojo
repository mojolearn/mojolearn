# SPDX-License-Identifier: Apache-2.0
# Native validation of the optional target that Normalizer accepts but ignores.
# Keep original target owners until its fit/fit_transform operation finishes.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_arrays import classical_input, classical_is_array
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get
from bindings.classical_warnings import COLUMN_WARNING, _warning_category
from bindings.metrics_native_support import metric_warn
from bindings.host_sparse_native import host_sparse_format, _sparse_fields
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_admission import _ordinary_torch, _one_of
from bindings.sequential_inputs import _promote
from bindings.x_prep_native_inputs import _rows_fields
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.prep_transform_pyapi import _array_snapshot, _pandas_empty_shape
from providers.python.prep_category_pyapi import prep_ndarray_backed, prep_category_target
from providers.python.prep_arrow_target_pyapi import prep_arrow_target
from core.prep_boundary_impl import prep_float_finite, prep_mask_any


def _length(rows: Int, target: Int) raises:
    if rows >= 0 and target >= 0 and rows != target:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(target) + "]")


def _finite(snapshot: ArrayMetadata, name: String) raises:
    if snapshot.dtype_number == 13:
        raise Error("TypeError: mojolearn: unsupported buffer format '" + ("g" if snapshot.native_byte_order else "=g") + "'")
    if snapshot.dtype_number != 11 and snapshot.dtype_number != 12:
        return
    if snapshot.source.itemsize != (4 if snapshot.dtype_number == 11 else 8):
        raise Error("TypeError: target floating dtype has inconsistent itemsize")
    if not prep_float_finite(snapshot.source, snapshot.shape, snapshot.strides,
                            snapshot.count, snapshot.native_byte_order, snapshot.dtype_number):
        raise Error("Input y contains NaN or infinity (" + name + ").")


def _complex(code: Int) -> Bool:
    return code == 14 or code == 15 or code == 16


def _pandas_target(source: PythonObject, rows: Int, multi: Bool, name: String,
                   context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var series = py_type(source) is _pd_class("pandas.core.series", "Series")
    var manager = py_get(_pd_fields(source), "_mgr")
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: pandas target requires native single-block storage")
        blocks = py_get(_pd_fields(manager), "blocks")
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: pandas target requires native block storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not b.tuple or (series and len(blocks) != 1):
        raise Error("TypeError: pandas target requires native block metadata")
    if len(blocks) == 0:
        var shape = _pandas_empty_shape(manager, context)
        _length(rows, shape[0])
        return Python.tuple(source, manager, blocks)
    var snapshots = List[ArrayMetadata]()
    var categorical = List[Bool]()
    var category_finite = List[Bool]()
    var owners = Python.list()
    var count = -1
    var columns = 0
    var objects = False
    var complex = False
    var booleans = False
    var numbers = False
    var common_kind = String()
    var common_size = 0
    var long_double = False
    for k in range(len(blocks)):  # driver-loop: target block metadata and retained raw owners
        var block = blocks[k]
        var cls = py_type(block)
        if cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock") and cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock"):
            raise Error("TypeError: pandas targets require ordinary native blocks")
        var raw = _pd_get("pandas._libs.internals", "Block", "values", block)
        var mask_rows = -1
        var is_category = py_type(raw) is _pd_class("pandas.core.arrays.categorical", "Categorical")
        var is_arrow = py_type(raw) is _pd_class("pandas.core.arrays.arrow.array", "ArrowExtensionArray") or py_type(raw) is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray")
        var finite = True
        var number = -1
        var width = 0
        var data: ArrayMetadata
        if is_category:
            var captured = prep_category_target(raw, context)
            data = captured[0].copy()
            number = captured[1]
            width = captured[2]
            finite = captured[3]
        elif is_arrow:
            var captured = prep_arrow_target(raw, series, context)
            data = captured[0].copy()
            number = captured[1]
            width = captured[2]
            finite = captured[3]
        else:
            var period = py_type(raw) is _pd_class("pandas.core.arrays.period", "PeriodArray")
            raw = prep_ndarray_backed(raw)
            if py_type(raw) is not context["ndarray"]:
                var kind = py_type(raw)
                if kind is not _pd_class("pandas.core.arrays.integer", "IntegerArray") and kind is not _pd_class("pandas.core.arrays.floating", "FloatingArray") and kind is not _pd_class("pandas.core.arrays.boolean", "BooleanArray"):
                    raise Error("TypeError: pandas target requires native numeric or object storage")
                var fields = _pd_fields(raw)
                var mask_value = py_get(fields, "_mask")
                raw = py_get(fields, "_data")
                if py_type(raw) is not context["ndarray"] or py_type(mask_value) is not context["ndarray"]:
                    raise Error("TypeError: pandas nullable target requires native arrays")
                var mask = _array_snapshot(mask_value, context)
                if len(mask.shape) != 1 or mask.dtype_number != 0 or mask.source.itemsize != 1 or not mask.castable:
                    raise Error("TypeError: pandas nullable target mask metadata differs")
                mask_rows = mask.count
                # Nullable DataFrame export is object dtype. A nullable Series
                # with a missing value also exports passive object labels.
                objects = objects or not series or prep_mask_any(mask.source, mask.count)
                _ = py_call(owners.append, Python.tuple(mask.keep, mask.owner))
            data = _array_snapshot(raw, context)
            number = 17 if period else data.dtype_number
            width = data.source.itemsize
        var rank = len(data.shape)
        if rank != 1 and rank != 2:
            raise Error("TypeError: pandas target block rank must be one or two")
        var n = data.shape[rank - 1]
        if mask_rows >= 0 and (rank != 1 or mask_rows != n):
            raise Error("TypeError: pandas nullable target mask metadata differs")
        if count >= 0 and n != count:
            raise Error("pandas target blocks have different row counts")
        count = n
        columns += 1 if rank == 1 else data.shape[0]
        complex = complex or _complex(number)
        var numeric = (number >= 0 and number <= 13) or number == 23
        objects = objects or (not numeric and not _complex(number))
        booleans = booleans or number == 0
        numbers = numbers or number != 0
        if number == 13:
            long_double = True
        elif numeric:
            var kind = String("b") if number == 0 else (String("f") if number >= 11 else (String("i") if number % 2 == 1 else String("u")))
            _promote(common_kind, common_size, kind, width)
        _ = py_call(owners.append, Python.tuple(data.keep, data.owner))
        snapshots.append(data^)
        categorical.append(is_category or is_arrow)
        category_finite.append(finite)
    objects = objects or (booleans and numbers)
    if complex and not objects:
        raise Error("Complex data not supported: " + name + " got a complex y.")
    if not series and columns == 1 and not multi:
        metric_warn(COLUMN_WARNING, _warning_category(context), context)
    if not objects and long_double:
        raise Error("TypeError: mojolearn: unsupported buffer format 'g'")
    if not objects and common_kind == "f" and (common_size == 4 or common_size == 8):
        for k in range(len(snapshots)):  # driver-loop: original target float words; never narrowed
            if categorical[k]:
                if not category_finite[k]:
                    raise Error("Input y contains NaN or infinity (" + name + ").")
            else:
                _finite(snapshots[k], name)
    _length(rows, count)
    return Python.tuple(source, manager, blocks, owners)


def prep_ignored_target(y: PythonObject, rows: Int, required: Bool, multi: Bool,
                        name: String, context: PythonObject) raises -> PythonObject:
    if y is None:
        if required:
            raise Error(name + " requires y to be passed, but the target y is None.")
        return y
    var b = native_loaded_module("builtins")
    var cls = py_type(y)
    # describe() has no shape for these passive scalars/mappings; even a
    # builtin complex scalar y was ignored. A NumPy complex buffer is not.
    if cls is b.int or cls is b.float or cls is b.bool or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.dict or cls is b.range:
        return y
    if cls is context["maxabs_rows_class"]:
        var shape = py_get(_rows_fields(y, context), "shape")
        if py_type(shape) is not b.tuple:
            raise Error("TypeError: target row selection requires native shape metadata")
        for i in range(len(shape)):  # small-loop(rank: passive selected-target shape metadata)
            if py_type(shape[i]) is not b.int:
                raise Error("TypeError: target row selection requires native shape metadata")
        if len(shape) > 0:
            _length(rows, Int(py=shape[0]))
        return Python.tuple(y, shape)
    if host_sparse_format(y).byte_length() > 0:
        var shape = py_get(_sparse_fields(y), "_shape")
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
            raise Error("TypeError: sparse target storage requires native shape metadata")
        _length(rows, Int(py=shape[0]))
        return y
    if native_pandas_exact(y):
        return _pandas_target(y, rows, multi, name, context)
    if _one_of(cls, context["scalar_types"]):
        var scalar_kind = String(y.dtype.kind)
        if scalar_kind == "c":
            raise Error("Complex data not supported: " + name + " got a complex y.")
        if scalar_kind != "f" and scalar_kind != "b" and scalar_kind != "i" and scalar_kind != "u":
            return y
    var sequence = cls is b.list or cls is b.tuple
    var scan = True
    if sequence:
        var first = y
        while py_type(first) is b.list or py_type(first) is b.tuple:  # driver-loop: describe's first-element target category
            if len(first) == 0:
                break
            first = first[0]
        var kind = py_type(first)
        if kind is b.complex:
            raise Error("Complex data not supported: " + name + " got a complex y.")
        scan = kind is b.float
        if _one_of(kind, context["scalar_types"]):
            var code = String(first.dtype.kind)
            if code == "c":
                raise Error("Complex data not supported: " + name + " got a complex y.")
            scan = code == "f"
        if not scan:
            _length(rows, len(y))
            return y
    classical_admit_value(y, context)
    var prepared = native_torch_numpy(y) if _ordinary_torch(cls, context) else y
    var view = prepared if classical_is_array(prepared, context) else classical_input(prepared, context, "y")
    var snapshot = _array_snapshot(view, context)
    if _complex(snapshot.dtype_number):
        raise Error("Complex data not supported: " + name + " got a complex y.")
    var reshape = cls is py_get(context, "ndarray") or cls is context["array_class"] or _ordinary_torch(cls, context)
    if not multi and reshape and len(snapshot.shape) == 2 and snapshot.shape[1] == 1:
        metric_warn(COLUMN_WARNING, _warning_category(context), context)
    if scan:
        _finite(snapshot, name)
    if len(snapshot.shape) > 0:
        _length(rows, snapshot.shape[0])
    return Python.tuple(y, prepared, snapshot.keep, snapshot.owner)
