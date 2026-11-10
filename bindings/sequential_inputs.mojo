# SPDX-License-Identifier: Apache-2.0
# Native array-like admission preserving ndarray, standard buffer, and
# rectangular scalar/array sequence inputs. Container traversal and scalar
# serialization are host ABI work; existing kernels still own model math.
from bindings.native_array_storage import native_array_zeros, native_scratch_bytes
from bindings.native_buffer_address import native_buffer_address
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_array_view import native_array_ndarray_view
from bindings.py_balanced import py_call, py_type, py_get
from bindings.cpython_borrowed import borrowed_item as p3_item, borrowed_type_word as p3_type_word
from bindings.cpython_table import cpython_table
from core.sequential_input_plan import sequence_promote as _promote, SequenceInputPlan, sequence_write, SequenceBufferFormat, sequence_buffer_shape
from bindings.sequential_admission import _one_of
from std.python._cpython import GILReleased, PyObjectPtr
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from providers.python.array_layout_boundary import native_shape_pack
from core.native_array_layout import NativeArrayLayout


def _buffer(mut mods: NativeModules, value: PythonObject, context: PythonObject, samples: Bool) raises -> PythonObject:
    var view = native_buffer_view(value)
    sequence_buffer_shape(Int(py=view.ndim), Bool(py=view.c_contiguous), samples)
    var format = SequenceBufferFormat(String(view.format), Int(py=view.itemsize))
    var dtype = context["numeric_dtypes"][Python.tuple(format.kind, format.itemsize)]
    if format.prefix.byte_length() > 0 and format.kind != "b":
        dtype = PythonObject(format.prefix + format.kind + String(format.itemsize))
    if Bool(py=view.contiguous):
        # Both C and Fortran contiguous buffers are borrowed without a copy.
        # ndarray's C type owns the memoryview and preserves read-only status.
        var windowed_result17 = py_call(context["ndarray"], view.shape, dtype=dtype, buffer=view, strides=view.strides)
        mods.released()
        return windowed_result17^
    # CPython's native buffer implementation packs signed/zero strides in
    # logical C order, including read-only views. No NumPy conversion runs.
    var packed = py_call(view.tobytes, "C")
    var windowed_result16 = py_call(context["ndarray"], view.shape, dtype=dtype, buffer=packed)
    mods.released()
    return windowed_result16^


struct SequenceBoundaryFrame(Movable):
    var owner: PythonObject
    var depth: Int
    var count: Int
    var index: Int
    var is_tuple: Bool
    var is_range: Bool

    def __init__(out self, owner: PythonObject, depth: Int, count: Int, is_tuple: Bool, is_range: Bool):
        self.owner = owner
        self.depth = depth
        self.count = count
        self.index = 0
        self.is_tuple = is_tuple
        self.is_range = is_range


struct SequenceBoundaryCapture(Movable):
    var plan: SequenceInputPlan
    var pins: List[AdmittedBorrowedBuffer]

    def __init__(out self):
        self.plan = SequenceInputPlan()
        self.pins = List[AdmittedBorrowedBuffer]()


struct SequenceBoundaryResult(Movable):
    var shape: List[Int]
    var kind: String
    var itemsize: Int
    var storage: PythonObject

    def __init__(out self, var shape: List[Int], var kind: String, itemsize: Int, storage: PythonObject):
        self.shape = shape^
        self.kind = kind^
        self.itemsize = itemsize
        self.storage = storage


def _sequence_array_capture(mut capture: SequenceBoundaryCapture, value: PythonObject, depth: Int) raises:
    var pin = AdmittedBorrowedBuffer(value)
    var dtype = value.dtype
    capture.plan.array(depth, pin.layout.value().copy(), String(dtype.kind), Bool(py=dtype.isnative))
    capture.pins.append(pin^)


def _sequence_buffer_capture(mut capture: SequenceBoundaryCapture, value: PythonObject, depth: Int) raises:
    var pin = AdmittedBorrowedBuffer(value)
    var format = SequenceBufferFormat(String(pin.format), pin.layout.value().itemsize)
    capture.plan.array(depth, pin.layout.value().copy(), format.kind, format.native)
    capture.pins.append(pin^)


def _sequence_scalar_capture(mut plan: SequenceInputPlan, raw: PyObjectPtr, depth: Int,
    kind: Int, float_target: Bool) raises:
    ref cpy = Python().cpython()
    var table = cpython_table()
    if kind == 3:
        plan.scalar(depth, UInt64(table[].PyLong_AsLongLong(raw)), "b", 1)
    elif kind == 2 or float_target:
        var number = table[].PyFloat_AsDouble(raw)
        if number == -1.0 and table[].PyErr_Occurred():
            raise cpy.unsafe_get_error()
        plan.scalar(depth, bitcast[DType.uint64](number), "f", 8)
    else:
        var signed = table[].PyLong_AsLongLong(raw)
        if signed != -1 or not table[].PyErr_Occurred():
            plan.scalar(depth, bitcast[DType.uint64](signed), "i", 8)
            return
        table[].PyErr_Clear()
        var unsigned = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLong")(raw)
        if table[].PyErr_Occurred():
            table[].PyErr_Clear()
            raise Error("TypeError: input integers exceed the supported numeric array range")
        plan.scalar(depth, unsigned, "u", 8)


def _sequence_capture(mut mods: NativeModules, source: PythonObject, context: PythonObject,
    float_target: Bool) raises -> SequenceBoundaryCapture:
    # Scalar/container unpack must hold GIL. Exact type checks precede reads;
    # no subclass indexing, scalar conversion hook or recursive dispatch runs.
    # Stack depth is at most eight, and scalar storage uses compact native
    # word/tag vectors rather than one descriptor or Python object per value.
    var b = mods.module("builtins")
    var numpy = py_get(context, "ndarray") is not None
    var scalar_types = py_get(context, "scalar_types", Python.tuple())
    var list_type = b.list
    var tuple_type = b.tuple
    var range_type = b.range
    var float_word = Int(b.float._obj_ptr)
    var int_word = Int(b.int._obj_ptr)
    var bool_word = Int(b.bool._obj_ptr)
    var string_type = b.str
    var bytes_type = b.bytes
    var complex_type = b.complex
    var none_type = py_type(PythonObject(None))
    var capture = SequenceBoundaryCapture()
    var stack = List[SequenceBoundaryFrame]()
    var current = source
    var depth = 0
    while True:  # cpu-route: exact external containers are decoded iteratively under the GIL
        var cls = py_type(current)
        var word = Int(cls._obj_ptr)
        if cls is list_type or cls is tuple_type or cls is range_type:
            var count = len(current)
            capture.plan.container(depth, count)
            stack.append(SequenceBoundaryFrame(current, depth, count, cls is tuple_type, cls is range_type))
        elif word == bool_word or word == int_word or word == float_word:
            _sequence_scalar_capture(capture.plan, current._obj_ptr, depth,
                3 if word == bool_word else (2 if word == float_word else 1), float_target)
        elif cls is string_type or cls is bytes_type or cls is complex_type or cls is none_type:
            raise Error("TypeError: inputs must contain numeric scalars")
        elif numpy and (cls is context["ndarray"] or _one_of(cls, scalar_types)):
            _sequence_array_capture(capture, current, depth)
        elif numpy and cls is context["array_class"]:
            var array = native_array_ndarray_view(current, context)
            _sequence_array_capture(capture, array, depth)
        elif numpy and (cls is b.memoryview or cls is context["array_type"] or cls is b.bytearray):
            _sequence_buffer_capture(capture, current, depth)
        else:
            raise Error("TypeError: inputs must contain numeric scalars")
        var found = False
        while len(stack) > 0:  # driver-loop: advance the bounded native traversal stack
            var top = len(stack) - 1
            if stack[top].index == stack[top].count:
                _ = stack.pop()
                continue
            var index = stack[top].index
            stack[top].index += 1
            depth = stack[top].depth + 1
            if stack[top].is_range:
                # An exact range's index operation is a CPython C slot.
                current = stack[top].owner[index]
            else:
                var raw = p3_item(stack[top].owner, index, stack[top].is_tuple)
                var child_type = p3_type_word(raw)
                if child_type == bool_word or child_type == int_word or child_type == float_word:
                    _sequence_scalar_capture(capture.plan, raw, depth,
                        3 if child_type == bool_word else (2 if child_type == float_word else 1), float_target)
                    continue
                current = PythonObject(from_borrowed=raw)
            found = True
            break
        if not found:
            break
    _ = source
    return capture^


def sequential_sequence_native(mut mods: NativeModules, value: PythonObject, context: PythonObject,
    float32_target: Bool = False, array_storage: Bool = True) raises -> SequenceBoundaryResult:
    var capture = _sequence_capture(mods, value, context, float32_target)
    var layout = NativeArrayLayout(capture.plan.shape.copy(), 4 if float32_target else capture.plan.itemsize)
    var storage = PythonObject(None)
    if array_storage:
        var code = String("f") if float32_target else String(context["buffer_codes"][Python.tuple(capture.plan.kind, capture.plan.itemsize)])
        storage = native_array_zeros(code, layout.count)
    else:
        # ndarray consumes any native byte owner: no zero-fill pass before the
        # parallel writer initializes every byte of this newly allocated store.
        storage = native_scratch_bytes(layout.nbytes)
    var destination = native_buffer_address(storage)
    try:
        with GILReleased(Python()):
            sequence_write(capture.plan, destination, float32_target)
    finally:
        _ = capture
        _ = storage
        _ = value
    return SequenceBoundaryResult(capture.plan.shape.copy(), String(capture.plan.kind), capture.plan.itemsize, storage)


def sequential_sequence_native(value: PythonObject, context: PythonObject,
    float32_target: Bool = False, array_storage: Bool = True) raises -> SequenceBoundaryResult:
    var mods = NativeModules()
    return sequential_sequence_native(mods, value, context, float32_target, array_storage)


def sequential_input(mut mods: NativeModules, value: PythonObject, context: PythonObject, samples: Bool = False) raises -> PythonObject:
    var b = mods.module("builtins")
    if py_type(value) is context["ndarray"]:
        return value
    if py_type(value) is context["array_class"]:
        # Views returned from input preparation may escape into fitted state
        # or public outputs. Array._mv can be ownerless; retain its exporter.
        return native_array_ndarray_view(value, context)
    if py_type(value) is b.memoryview or py_type(value) is context["array_type"] or py_type(value) is b.bytearray:
        return _buffer(mods, value, context, samples)
    var packed = sequential_sequence_native(mods, value, context, array_storage=False)
    var dtype = context["numeric_dtypes"][Python.tuple(packed.kind, packed.itemsize)]
    var windowed_result15 = py_call(context["ndarray"], native_shape_pack(packed.shape), dtype=dtype, buffer=packed.storage)
    mods.released()
    return windowed_result15^


def sequential_input(value: PythonObject, context: PythonObject, samples: Bool = False) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return sequential_input(mods, value, context, samples)


def sequential_sequence(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Compatibility metadata PACK for callers not yet using typed descriptors.
    var result = sequential_sequence_native(mods, value, context)
    return Python.tuple(native_shape_pack(result.shape), result.kind, result.itemsize, result.storage)


def sequential_sequence(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var mods = NativeModules()
    return sequential_sequence(mods, value, context)


def sequential_sequence_f32(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = sequential_sequence_native(mods, value, context, True)
    return Python.tuple(native_shape_pack(result.shape), result.storage)


def sequential_sequence_f32(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var mods = NativeModules()
    return sequential_sequence_f32(mods, value, context)
