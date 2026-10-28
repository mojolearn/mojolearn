# SPDX-License-Identifier: Apache-2.0
# Common operation-local buffer metadata. Python ownership stays at this edge;
# numerical owners receive the existing native CastIn and native dimensions.
# Callers first admit supported native types and capture shape once, preserving
# their API-specific rank/emptiness refusals before dtype admission.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_is_array, classical_address
from core.cast_in import CastIn, cast_in_kind, NPY_HALF, NPY_FLOAT, NPY_DOUBLE, CAST_KIND_FLOAT


struct ArrayMetadata(Copyable, Movable):
    var keep: PythonObject
    var shape: List[Int]
    var source: CastIn
    var count: Int
    var dtype_number: Int
    var native_byte_order: Bool
    var c_order: Bool
    var is_array: Bool
    # Immutable original exporter/storage snapshot. `keep` owns the current
    # buffer after a conversion; both must survive until this snapshot dies.
    var owner: PythonObject
    var copied: Bool
    var owned_writable: Bool
    # Writability is independent of exclusive ownership: a borrowed Array
    # view may be writable, while copy=False must respect a readonly view.
    var readonly: Bool
    # Admission-only snapshots may retain higher rank/unsupported dtype
    # metadata for public error ordering. Their CastIn is NOT executable.
    var castable: Bool
    var strides: List[Int]
    # Native descriptor spelling, including datetime/timedelta units. Owners
    # may inspect it without rereading the caller's dtype after admission.
    var dtype_name: String

    def __init__(out self, keep: PythonObject, var shape: List[Int], source: CastIn,
                 count: Int, dtype_number: Int, native_byte_order: Bool,
                 c_order: Bool, is_array: Bool, owner: PythonObject, owned_writable: Bool,
                 readonly: Bool = False, castable: Bool = True,
                 var strides: List[Int] = List[Int](), dtype_name: String = ""):
        self.keep = keep
        self.shape = shape^
        self.source = source
        self.count = count
        self.dtype_number = dtype_number
        self.native_byte_order = native_byte_order
        self.c_order = c_order
        self.is_array = is_array
        self.owner = owner
        self.copied = False
        self.owned_writable = owned_writable
        self.readonly = readonly
        self.castable = castable
        self.strides = strides^
        self.dtype_name = dtype_name
        if len(self.strides) == 0:
            for axis in range(len(self.shape)):  # small-loop(native rank metadata for direct constructors)
                self.strides.append(source.s0 if axis == 0 else source.s1)

    def python_shape(self) raises -> PythonObject:
        var result = Python.list()
        for dimension in self.shape:  # driver-loop: over captured native shape words
            _ = py_call(result.append, dimension)
        return py_call(native_loaded_module("builtins").tuple, result)

    def f32_c(self) -> Bool:
        return self.castable and self.dtype_number == NPY_FLOAT and self.source.itemsize == 4 and self.c_order

    def replace_f32(mut self, keep: PythonObject, address: Int):
        # A new packed buffer's metadata is known from the allocation. Never
        # fetch it back from Python to rediscover the shape/dtype we supplied.
        self.keep = keep
        # Preserve the independently captured original Array slot owners,
        # even if its mutable slots are later changed. `keep` owns this copy.
        self.copied = True
        self.owned_writable = True
        self.readonly = False
        self.source = CastIn(address, CAST_KIND_FLOAT, 4, self.source.rows, self.source.cols,
                             self.source.cols * 4 if len(self.shape) > 0 else 0,
                             4 if len(self.shape) > 1 else 0)
        self.strides = List[Int]()
        for axis in range(len(self.shape)):  # small-loop(replacement matrix/vector stride metadata)
            self.strides.append(self.source.s0 if axis == 0 else self.source.s1)
        self.dtype_number = NPY_FLOAT
        self.dtype_name = "<f4"
        self.native_byte_order = True
        self.c_order = True
        self.is_array = True

    def replace_f64(mut self, keep: PythonObject, address: Int):
        self.keep = keep
        # Same ownership rule as replace_f32; no metadata/owner recapture.
        self.copied = True
        self.owned_writable = True
        self.readonly = False
        self.source = CastIn(address, CAST_KIND_FLOAT, 8, self.source.rows, self.source.cols,
                             self.source.cols * 8 if len(self.shape) > 0 else 0,
                             8 if len(self.shape) > 1 else 0)
        self.strides = List[Int]()
        for axis in range(len(self.shape)):  # small-loop(replacement matrix/vector stride metadata)
            self.strides.append(self.source.s0 if axis == 0 else self.source.s1)
        self.dtype_number = NPY_DOUBLE
        self.dtype_name = "<f8"
        self.native_byte_order = True
        self.c_order = True
        self.is_array = True


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
                        context: PythonObject, allow_non_native: Bool = False,
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
    return ArrayMetadata(value, shape^, source, count, code, native_order, contiguous, array, owner, owned_writable,
                         readonly, castable, captured_strides^, dtype_name)
