# SPDX-License-Identifier: Apache-2.0
# Integer label storage: the one input kind a target admission can accept
# without reading a single element.
#
# The public target checks (complex refusal, NaN/infinity, "Unknown label type:
# continuous", exact scalar admission) only ever refuse an element that is a
# complex, a non-finite or non-integral real, or a Python object. An exact
# NumPy ndarray or a native Array whose dtype is a fixed-width integer (or, when
# allowed, bool) holds none of those: `tolist()` of it yields exact Python ints
# (bools), each of which every one of those checks admits. Such a target is
# therefore admitted from its dtype metadata alone, without the per-element walk
# (tolist, one Python object and one admission per label) that cost about 2.3 s
# per 20000-label classifier fit through the GP label snapshot. Everything else
# (lists, tuples, float, complex, object, str, datetime, pandas, torch, sparse,
# subclasses) keeps its caller's existing element-by-element path.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get
from core.label_storage_impl import label_float_target_refusal


def labels_integer_storage(value: PythonObject, context: PythonObject, allow_bool: Bool = True) raises -> Bool:
    """True for an exact ndarray or exact native Array of rank >= 1 whose
    dtype is a fixed-width signed or unsigned integer (or bool when
    `allow_bool`). Reads class identity and dtype metadata only; callers
    still admit the container itself (`classical_admit_value`) first."""
    var cls = py_type(value)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        var kind = String(value.dtype.kind)
        if Int(py=value.ndim) < 1:
            return False
        return kind == "i" or kind == "u" or (allow_bool and kind == "b")
    var array_class = py_get(context, "array_class")
    if array_class is not None and cls is array_class:
        if Int(py=value.ndim) < 1:
            return False
        var typestr = String(value.dtype)
        if typestr.startswith("<i") or typestr.startswith("<u") or typestr.startswith("|i") or typestr.startswith("|u"):
            return True
        return allow_bool and typestr == "|b1"
    return False


def _ndarray_buffer_address(value: PythonObject) raises -> Int:
    """The first element's address of an ndarray's strided export
    (PyBUF_STRIDES | PyBUF_FORMAT); the export is released at once, the
    caller keeps the owner alive."""
    ref cpy = Python().cpython()
    var storage = List[Int](length=10, fill=0)
    var pointer = storage.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0x1C)) != 0:
        raise cpy.unsafe_get_error()
    var address = storage[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = storage^
    return address


def labels_float_storage(value: PythonObject, context: PythonObject) raises -> Int:
    """-1 unless `value` is an exact ndarray or exact native Array of rank
    >= 1 with a native little-endian float32 or float64 dtype. For such a
    target, the refusal code of `label_float_target_refusal`
    (core/label_storage_impl.mojo): 0 when every label is finite and
    integral, 1 at a NaN/infinity, 2 at a finite non-integral label, judged
    at the first refused label in C order, read from the buffer in one native
    pass. Such a target holds no element any other target check can refuse
    (`tolist()` yields exact Python floats), so callers admit the container
    itself (`classical_admit_value`) first and skip the per-label walk."""
    var cls = py_type(value)
    var typestr: String
    var address: Int
    var ndarray = py_get(context, "ndarray")
    var array_class = py_get(context, "array_class")
    if ndarray is not None and cls is ndarray:
        typestr = String(value.dtype.str)
        if (typestr != "<f4" and typestr != "<f8") or Int(py=value.ndim) < 1:
            return -1
        address = _ndarray_buffer_address(value)
    elif array_class is not None and cls is array_class:
        typestr = String(value.dtype)
        if (typestr != "<f4" and typestr != "<f8") or Int(py=value.ndim) < 1:
            return -1
        address = Int(py=value._addr)
    else:
        return -1
    var rank = Int(py=value.ndim)
    var pyshape = value.shape
    var pystrides = value.strides
    var shape = List[Int]()
    var strides = List[Int]()
    for k in range(rank):  # driver-loop: over array rank metadata
        shape.append(Int(py=pyshape[k]))
        strides.append(Int(py=pystrides[k]))
    # The caller holds `value` (the buffer's owner) across this native read.
    return label_float_target_refusal(address, typestr == "<f8", shape, strides)
