# SPDX-License-Identifier: Apache-2.0
# Common operation-local buffer metadata. Python ownership stays at this edge;
# numerical owners receive the existing native CastIn and native dimensions.
# Callers first admit supported native types and capture shape once, preserving
# their API-specific rank/emptiness refusals before dtype admission.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_is_array, classical_address
from core.cast_in import CastIn, cast_in_kind, NPY_HALF, NPY_FLOAT, NPY_DOUBLE, CAST_KIND_FLOAT
# The typed view lives in core/ (no Python); doors import it from here or there.
from core.array_metadata import ArrayMetadata


struct ArrayPin(Copyable, Movable):
    """The door-side storage pin of one admitted array.

    `keep` owns the buffer the typed view's `source.addr` points into (the
    caller's object, or the converted copy after `replace_f32/replace_f64`);
    `owner` is the immutable original exporter/storage snapshot. RUN code
    never sees this pin: the door holds it for the whole call and names it
    again after the last RUN read.
    """
    var keep: PythonObject
    var owner: PythonObject

    def __init__(out self):
        self.keep = PythonObject(None)
        self.owner = PythonObject(None)

    def __init__(out self, keep: PythonObject, owner: PythonObject):
        self.keep = keep
        self.owner = owner


def array_python_shape(view: ArrayMetadata) raises -> PythonObject:
    var result = Python.list()
    for dimension in view.shape:  # driver-loop: over captured native shape words
        _ = py_call(result.append, dimension)
    return py_call(native_loaded_module("builtins").tuple, result)


def array_slot_once(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return py_call(context["array_fields"][name], value, context["array_class"])


def array_owner_once(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if not classical_is_array(value, context):
        return value
    # Retain storage owners independently of mutable Array slots.
    return Python.tuple(array_slot_once(value, "_store", context), array_slot_once(value, "_base", context),
                        array_slot_once(value, "_pin", context), array_slot_once(value, "_mv", context))


def array_shape_once(value: PythonObject, context: PythonObject) raises -> List[Int]:
    var shape = array_slot_once(value, "shape", context) if classical_is_array(value, context) else value.shape
    if py_type(shape) is not py_type(Python.tuple()):
        raise Error("TypeError: array shape must be an exact native tuple")
    var result = List[Int]()
    for i in range(len(shape)):  # driver-loop: over admitted native array rank metadata
        if py_type(shape[i]) is not py_type(PythonObject(0)):
            raise Error("TypeError: array dimensions must be native integers")
        result.append(Int(py=shape[i]))
    return result^


def array_metadata_once(value: PythonObject, var shape: List[Int], name: String,
                        context: PythonObject, mut pin: ArrayPin, allow_non_native: Bool = False,
                        complex_error: String = "", f32_error: String = "",
                        allow_unsupported: Bool = False) raises -> ArrayMetadata:
    var nd = len(shape)
    if nd > 2 and complex_error.byte_length() == 0 and not allow_unsupported:
        raise Error(name + " metadata requires an admitted scalar/vector/matrix")
    var array = classical_is_array(value, context)
    var dtype = array_slot_once(value, "dtype", context) if array else value.dtype
    var item_value = array_slot_once(value, "itemsize", context) if array else value.itemsize
    if py_type(item_value) is not py_type(PythonObject(0)):
        raise Error("TypeError: array itemsize must be a native integer")
    var item = Int(py=item_value)
    var native_order = True
    var code = 0
    var dtype_name = String()
    var contiguous = False
    var readonly = False
    if array:
        if py_type(dtype) is not py_type(PythonObject("")):
            raise Error("TypeError: Array dtype must be an exact native string")
        dtype_name = String(dtype)
        var letter = String(dtype[1])
        if letter == "f":
            code = NPY_HALF if item == 2 else (NPY_FLOAT if item == 4 else NPY_DOUBLE)
        else:
            code = 1 if letter == "i" else 2
        var order = array_slot_once(value, "order", context)
        if py_type(order) is not py_type(PythonObject("")):
            raise Error("TypeError: Array order must be an exact native string")
        contiguous = nd < 2 or String(order) == "C"
    else:
        dtype_name = String(dtype.str)
        native_order = Bool(py=dtype.isnative)
        if not native_order and not allow_non_native:
            raise Error("TypeError: " + name + " must be in native byte order")
        code = Int(py=dtype.num)
        var flags = value.flags
        contiguous = Bool(py=flags.c_contiguous)
        readonly = not Bool(py=flags.writeable)
    # Preserve each owner's public dtype refusal without rereading dtype.
    # Reject before cast_in_kind: complex128/longcomplex have wider items
    # than CastIn supports and must never reach a numerical descriptor.
    if complex_error.byte_length() > 0 and (code == 14 or code == 15 or code == 16):
        raise Error(complex_error)
    if nd > 2 and not allow_unsupported:
        raise Error(name + " metadata requires an admitted scalar/vector/matrix")
    if f32_error.byte_length() > 0 and (code != NPY_FLOAT or item != 4):
        raise Error(f32_error)
    var kind = 0
    var castable = nd <= 2
    try:
        kind = cast_in_kind(code, item, name)
    except error:
        if not allow_unsupported:
            raise error
        castable = False
    if not castable:
        # Unsigned kind 0 is only a placeholder; a caller opting into this
        # metadata-only path MUST reject !castable before using source.
        kind = 0
    var strides = array_slot_once(value, "strides", context) if array else value.strides
    if py_type(strides) is not py_type(Python.tuple()) or len(strides) != nd:
        raise Error("TypeError: array strides must be a native tuple matching its rank")
    var captured_strides = List[Int]()
    for axis in range(nd):  # small-loop(admitted stride words)
        if py_type(strides[axis]) is not py_type(PythonObject(0)):
            raise Error("TypeError: array strides must be native integers")
        captured_strides.append(Int(py=strides[axis]))
    var rows = shape[0] if nd > 0 else 1
    var cols = shape[1] if nd > 1 else 1
    var s0 = captured_strides[0] if nd > 0 else 0
    var s1 = captured_strides[1] if nd > 1 else 0
    var address = array_slot_once(value, "_addr", context) if array else classical_address(value)
    if py_type(address) is not py_type(PythonObject(0)):
        raise Error("TypeError: array address must be a native integer")
    var source = CastIn(Int(py=address), kind, item, rows, cols, s0, s1)
    var owner = array_owner_once(value, context)
    var owned_writable = False
    if array:
        var readonly_value = array_slot_once(value, "_readonly", context)
        if py_type(readonly_value) is not py_type(PythonObject(True)):
            raise Error("TypeError: Array readonly flag must be a native boolean")
        readonly = Bool(py=readonly_value)
        owned_writable = owner[1] is None and not readonly
    var count = 1
    for dimension in shape:  # small-loop(all captured shape metadata, including admission-only higher ranks)
        count *= dimension
    # The door holds `pin` through every RUN read of `source.addr`.
    pin = ArrayPin(value, owner)
    return ArrayMetadata(shape^, source, count, code, native_order, contiguous, array, owned_writable,
                         readonly, castable, captured_strides^, dtype_name)
