# SPDX-License-Identifier: Apache-2.0
# Metric label/shape preparation without Array methods or caller protocols.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import classical_is_array, classical_address
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from bindings.classical_labels import _pandas_values
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_scalar_decode import NativeScalarBuffer
from core.native_scalar_decode import NativeRealBits, native_scalar_real
from bindings.p3_labels_metrics_native import p3_array_numbers, p3_item, p3_type_word


def metric_shape(value: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(value, context)
    return metric_shape_admitted(value, context)


def metric_shape_admitted(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`metric_shape` of a value the caller already admitted
    (`classical_admit_value`) in the same native call, so a label list is not
    walked again. A pandas value's stored block is a new object and is
    still admitted."""
    var cls = py_type(value)
    if cls is py_type(Python.list()) or cls is py_type(Python.tuple()):
        var count = len(value)
        if count > 0:
            var first = py_type(value[0])
            if first is py_type(Python.list()) or first is py_type(Python.tuple()):
                return Python.tuple(count, len(value[0]))
        return Python.tuple(count)
    if classical_is_array(value, context):
        return value.shape
    if native_pandas_exact(value):
        var stored = metric_shape(_pandas_values(value), context)
        # A Series block is one-dimensional. A DataFrame's single native
        # block stores (columns, rows), opposite to its public shape.
        return Python.tuple(stored[1], stored[0]) if len(stored) == 2 else stored
    try:
        var view = native_buffer_view(value)
        return view.shape
    except:
        # Admitted scalars have no shape. Do not ask arbitrary objects for
        # shape/__len__ attributes or invoke an Array's Python __len__.
        return Python.tuple()


def _metric_array_flat(value: PythonObject) raises -> PythonObject:
    var out = Python.list()
    var shape = value.shape
    var strides = value.strides
    var rank = len(shape)
    var count = Int(py=value.size)
    var width = Int(py=value.itemsize)
    var dtype = String(value.dtype)
    var kind = String(dtype[byte=1:2])
    var code = String("i") if kind == "i" else String("I")
    if kind == "f":
        code = "e" if width == 2 else "f" if width == 4 else "d"
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value)))
    # The shape and strides read once as native words, not per element per axis.
    var sizes = List[Int](capacity=rank)
    var steps = List[Int](capacity=rank)
    for axis in range(rank):  # driver-loop: over the array's axes
        sizes.append(Int(py=shape[axis]))
        steps.append(Int(py=strides[axis]))
    # p3-labels-metrics: integer (at most 32-bit unsigned, any signed) and
    # float32 / float64 words become exact Python ints / floats by C calls
    # into one presized list (the same numbers `to_python_number` made, which
    # parsed each int from text and appended it by a Python call). Other
    # dtypes (float16, uint64) keep the general decoder below.
    var fast = p3_array_numbers(Int(py=classical_address(value)), sizes, steps, count, kind, width)
    if fast is not None:
        _ = value  # keep the storage owner alive through every raw word read
        return fast
    for index in range(count):  # driver-loop: materialize explicit label scalars in logical C order
        var residual = index
        var offset = 0
        for reverse_axis in range(rank):  # driver-loop: actual array rank/strides, including Fortran and reversed views
            var axis = rank - 1 - reverse_axis
            var size = sizes[axis]
            offset += (residual % size) * steps[axis]
            residual //= size
        var decoded = NativeScalarBuffer(native_scalar_real(data, offset, code, width, True),
                                         NativeRealBits(), False, False)
        py_call(out.append, decoded.to_python_number(kind != "f"))
    _ = value  # keep the storage owner alive through every raw scalar read
    return out


def metric_flatten(value: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(value, context)
    return metric_flatten_admitted(value, context)


def metric_flatten_admitted(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`metric_flatten` of a value the caller already admitted in the same
    native call. The list a buffer's `tolist` makes is new and is still
    admitted below."""
    var item = value
    var list_type = py_type(Python.list())
    var tuple_type = py_type(Python.tuple())
    if classical_is_array(item, context):
        return _metric_array_flat(item)
    if native_pandas_exact(item):
        item = _pandas_values(item)
    var cls = py_type(item)
    if cls is not list_type and cls is not tuple_type:
        var ndarray = context.get("ndarray")
        if ndarray is not None and cls is ndarray:
            # Exact ndarray.tolist is native storage boxing, not NumPy
            # computation. Object leaves are admitted before later use.
            item = py_call(ndarray.tolist, item)
        else:
            try:
                item = py_call(native_buffer_view(item).tolist)
            except:
                return Python.list(item)
        classical_admit_value(item, context)
    if py_type(item) is not list_type and py_type(item) is not tuple_type:
        return Python.list(item)
    var items = py_call(list_type, item)
    var list_word = Int(list_type._obj_ptr)
    var tuple_word = Int(tuple_type._obj_ptr)
    for _ in range(64):  # driver-loop: preserve the existing homogeneous-level flattening rule
        if len(items) == 0:
            break
        # p3-labels-metrics: item classes read by pointer from the list's C
        # storage, stopping at the first label that is not a container (the
        # level is nested only when every item is one).
        var nested = True
        for index in range(len(items)):  # driver-loop: exact native sequence element classes, early exit
            var child = p3_type_word(p3_item(items, index, False))
            if child != list_word and child != tuple_word:
                nested = False
                break
        if not nested:
            break
        # One C-level list.extend per exact list/tuple row (it was one
        # Python append call per element).
        var flattened = Python.list()
        for index in range(len(items)):  # driver-loop: flatten one admitted container level, one extend per row
            py_call(flattened.extend, items[index])
        items = flattened
    return items
