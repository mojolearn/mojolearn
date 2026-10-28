# SPDX-License-Identifier: Apache-2.0
# One passive capture for ARIMA's series and up-to-three-dimensional exog.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type
from bindings.classical_arrays import classical_is_array, classical_input, classical_address
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32
from providers.python.array_metadata_pyapi import array_shape_once, array_slot_once, array_owner_once
from core.cast_in import cast_in_kind, NPY_FLOAT, NPY_HALF, NPY_DOUBLE


struct ArimaBuffer(Copyable, Movable):
    var keep: PythonObject
    var owner: PythonObject
    var shape: List[Int]
    var strides: List[Int]
    var address: Int
    var count: Int
    var itemsize: Int
    var kind: Int
    var dtype_number: Int
    var c_order: Bool
    # True when the capture is a native float32 copy of pandas storage: the
    # caller's own object was converted, whatever the copy's layout.
    var converted: Bool

    def __init__(out self, keep: PythonObject, owner: PythonObject, var shape: List[Int], var strides: List[Int],
                 address: Int, count: Int, itemsize: Int, kind: Int, dtype_number: Int, c_order: Bool):
        self.keep = keep
        self.owner = owner
        self.shape = shape^
        self.strides = strides^
        self.address = address
        self.count = count
        self.itemsize = itemsize
        self.kind = kind
        self.dtype_number = dtype_number
        self.c_order = c_order
        self.converted = False

    def f32_c(self) -> Bool:
        return self.dtype_number == NPY_FLOAT and self.itemsize == 4 and self.c_order


def arima_buffer(value: PythonObject, name: String, context: PythonObject, packed: Bool = False,
                 refuse_double: String = "") raises -> ArimaBuffer:
    """`refuse_double`, when nonempty, is raised for pandas storage whose
    common dtype promotes beyond float32 (see native_pandas_f32); ndarray and
    Array callers apply their own dtype rule to the returned dtype number."""
    classical_admit_value(value, context)
    var pandas = native_pandas_exact(value)
    var source = value
    if pandas:
        # Native pandas block storage, never DataFrame.__array__; nonfinite
        # values pass through so each operation keeps its own finite rule.
        source = native_pandas_f32(value, context, False, True, refuse_double)
    var prepared = classical_input(source, context, name)
    var shape = array_shape_once(prepared, context)
    if len(shape) < 1 or len(shape) > 3:
        raise Error("ValueError: ARIMA " + name + " must have one to three dimensions")
    var array = classical_is_array(prepared, context)
    var dtype = array_slot_once(prepared, "dtype", context) if array else prepared.dtype
    var item_value = array_slot_once(prepared, "itemsize", context) if array else prepared.itemsize
    var int_type = py_type(PythonObject(0))
    var str_type = py_type(PythonObject(""))
    if py_type(item_value) is not int_type:
        raise Error("TypeError: ARIMA itemsize must be an exact integer")
    var item = Int(py=item_value)
    var number = 0
    var contiguous = False
    if array:
        if py_type(dtype) is not str_type:
            raise Error("TypeError: ARIMA Array dtype must be an exact string")
        var letter = String(dtype[1])
        number = (NPY_HALF if item == 2 else NPY_FLOAT if item == 4 else NPY_DOUBLE) if letter == "f" else 1 if letter == "i" else 2
        var order = array_slot_once(prepared, "order", context)
        if py_type(order) is not str_type:
            raise Error("TypeError: ARIMA Array order must be an exact string")
        contiguous = len(shape) < 2 or String(order) == "C"
    else:
        if not Bool(py=dtype.isnative):
            raise Error("TypeError: ARIMA " + name + " must use native byte order")
        number = Int(py=dtype.num)
        contiguous = Bool(py=prepared.flags.c_contiguous)
    var kind = cast_in_kind(number, item, name)
    var steps = array_slot_once(prepared, "strides", context) if array else prepared.strides
    if py_type(steps) is not py_type(Python.tuple()) or len(steps) != len(shape):
        raise Error("TypeError: ARIMA strides require an exact tuple matching rank")
    var strides = List[Int]()
    var count = PythonObject(1)
    for axis in range(len(shape)):  # driver-loop: captured rank/stride metadata, at most three words
        if shape[axis] < 0 or py_type(steps[axis]) is not int_type:
            raise Error("TypeError: ARIMA shape and strides require native integers")
        strides.append(Int(py=steps[axis]))
        count = count * shape[axis]
    var address = array_slot_once(prepared, "_addr", context) if array else classical_address(prepared)
    if py_type(address) is not int_type:
        raise Error("TypeError: ARIMA address must be a native integer")
    if packed and (not contiguous or item != 4 or number != NPY_FLOAT):
        raise Error("TypeError: ARIMA fitted buffers require contiguous float32 storage")
    var owner = array_owner_once(prepared, context)
    var captured = ArimaBuffer(prepared, owner, shape^, strides^, Int(py=address), Int(py=count), item, kind, number, contiguous)
    captured.converted = pandas
    return captured^
