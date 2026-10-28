# SPDX-License-Identifier: Apache-2.0
# Dedicated preprocessing snapshots and native pandas storage descriptions.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_arrays import classical_address
from bindings.classical_pandas import classical_admit_value, _pd_class, _pd_fields, _pd_get
from bindings.sequential_admission import sequential_admit_value
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_create
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.cast_in import CastIn, cast_in_kind


def prep_check_info(number: Int) -> Bool:
    # Numeric PEP3118 buffers and native ndarray types whose buffer export
    # fails but whose exact shape remains visible to describe() as a frame.
    # Datetime/timedelta do not export PEP3118; object/string/record buffers do.
    return (number >= 0 and number <= 13) or number == 23 or number == 21 or number == 22 or number == 2056 or number == -103


def prep_scalar_dtype(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_get(context, "ndarray") is None:
        return PythonObject(None)
    var cls = py_type(value)
    var classes = sequence_dtype_classes(context)
    for i in range(len(classes)):  # small-loop(native scalar ABI class identities)
        if cls is classes[i][1]:
            return value.dtype
    # NumPy's long/longlong aliases need not share their scalar class even
    # when both occupy eight bytes. Resolve through the verified dtype owner.
    for code in ["q", "Q"]:  # small-loop(native integer aliases)
        var dtype = sequence_dtype_create(PythonObject(code), context)
        if cls is dtype.type:
            return value.dtype
    return PythonObject(None)


def _array_snapshot(value: PythonObject, context: PythonObject) raises -> ArrayMetadata:
    classical_admit_value(value, context)
    var shape = array_shape_once(value, context)
    var result = array_metadata_once(value, shape^, "input", context, allow_non_native=True, allow_unsupported=True)
    if result.is_array:
        var view = result.owner[3]
        if py_type(view) is not native_loaded_module("builtins").memoryview:
            raise Error("TypeError: preprocessing Array storage requires an exact native memoryview")
        sequential_admit_value(view, context)
    return result^


def _pandas_empty_shape(manager: PythonObject, context: PythonObject) raises -> List[Int]:
    var b = native_loaded_module("builtins")
    var axes = _pd_get("pandas._libs.internals", "BlockManager", "axes", manager)
    if py_type(axes) is not b.list or len(axes) != 2:
        raise Error("TypeError: pandas native axes are unavailable")
    var shape = List[Int]()
    for axis in [1, 0]:  # small-loop(metadata: dataframe rows and columns)
        var index = axes[axis]
        if py_type(index) is _pd_class("pandas.core.indexes.range", "RangeIndex"):
            var span = py_get(_pd_fields(index), "_range")
            if py_type(span) is not b.range:
                raise Error("TypeError: pandas range index requires native range storage")
            shape.append(len(span))
        elif py_type(index) is _pd_class("pandas.core.indexes.base", "Index"):
            var storage = py_get(_pd_fields(index), "_data")
            if py_type(storage) is not context["ndarray"]:
                raise Error("TypeError: empty pandas axes require native array storage")
            var dims = array_shape_once(storage, context)
            if len(dims) != 1:
                raise Error("TypeError: pandas index must be one-dimensional")
            shape.append(dims[0])
        else:
            raise Error("TypeError: empty pandas axes require ordinary native index metadata")
    return shape^


def _description(source: PythonObject, var shape: List[Int], dtype: Int,
                 owner: PythonObject) -> ArrayMetadata:
    # No executable pointer: this only carries the already captured public
    # description until target validation finishes and inner preparation
    # raises its original sparse/object/empty refusal.
    var rows = shape[0] if len(shape) > 0 else 1
    var cols = shape[1] if len(shape) > 1 else 1
    return ArrayMetadata(source, shape^, CastIn(0, 0, 8, rows, cols, 0, 0),
                         0, dtype, True, False, False, owner, False, castable=False)


def prep_numeric_vector(count: Int, number: Int, width: Int, owner: PythonObject,
                        context: PythonObject) raises -> ArrayMetadata:
    # pandas is already loaded, so this exact ndarray constructor is the
    # existing provider allocation boundary. Its metadata is known, not read
    # back from mutable caller slots after allocation.
    var kind = cast_in_kind(number, width, "input")
    var code = String("|b1")
    if number != 0:
        code = "<" + ("f" if kind == 2 else ("i" if kind == 1 else "u")) + String(width)
    var output = py_call(context["ndarray"], Python.tuple(count), dtype=PythonObject(code))
    var shape = List[Int]()
    shape.append(count)
    return ArrayMetadata(output, shape^,
        CastIn(Int(py=classical_address(output)), kind, width, count, 1, width, 0),
        count, number, True, True, False, Python.tuple(output, owner), True,
        dtype_name=code)


