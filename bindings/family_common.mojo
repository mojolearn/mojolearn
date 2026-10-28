# SPDX-License-Identifier: Apache-2.0
# Shared door helpers of the classical family owners (bindings/family_native.mojo,
# lane migrate-classical-b): the `_buffer.as_f32_c` and
# `_sample_weight.check_sample_weight` refusals with their wording, mojolearn
# Array outputs, and calls into the compiled entries of the binding a public
# shell hands over. A Python exception a callee raises (a kernel's refusal,
# a CPU-only install's by-name refusal, a user callable's own exception) is
# stashed and restored unchanged at the door (`stash_python_error`), so the
# caller sees the class and message it saw before the migration.
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from bindings.native_array_storage import native_array_zeros
from bindings.native_loaded_module import native_loaded_module
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_arrays import (
    classical_address, classical_host_f32, classical_wrap_array, _store_code, classical_is_array,
    classical_cast_kind, classical_is_f32, classical_c_order, classical_dtype_str,
)
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from core.family_cast_impl import family_cast_rows, family_cast_rows_f64, family_min_f32
from core.cast_in import CastIn, CAST_KIND_FLOAT, NPY_FLOAT, NPY_DOUBLE
from bindings.classical_validate import classical_view, RESHAPE_HINT
from bindings.optimizer_public_words import stash_python_error
from core.family_weights_impl import (
    family_fill_f32, family_fill_f64, family_weight_status, family_weight_status_f64,
    FAMILY_WEIGHTS_NONFINITE, FAMILY_WEIGHTS_NEGATIVE,
    FAMILY_WEIGHTS_ALL_ZERO,
)

comptime FAMILY_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime FAMILY_F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime FAMILY_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime FAMILY_I64P = MutPointer[Int64, MutUntrackedOrigin]

comptime FAMILY_ZERO_WEIGHTS = "Sample weights must contain at least one non-zero number."
comptime FAMILY_NEGATIVE_WEIGHTS = "Negative values in data passed to `sample_weight`"
comptime FAMILY_NONFINITE_WEIGHTS = "Input sample_weight contains NaN or infinity."


def family_repr(value: PythonObject) raises -> String:
    """`repr(value)`."""
    return String(py_call(native_loaded_module("builtins").repr, value))


def family_str(value: PythonObject) raises -> String:
    """`str(value)`."""
    return String(py_call(native_loaded_module("builtins").str, value))


def family_isinstance(value: PythonObject, classes: PythonObject) raises -> Bool:
    """`isinstance(value, classes)`."""
    return Bool(py=py_call(native_loaded_module("builtins").isinstance, value, classes))


def family_is_str(value: PythonObject) raises -> Bool:
    return family_isinstance(value, native_loaded_module("builtins").str)


def family_is_int(value: PythonObject) raises -> Bool:
    """An int that is not a bool (`isinstance(v, int) and not isinstance(v, bool)`)."""
    var b = native_loaded_module("builtins")
    return family_isinstance(value, b.int) and not family_isinstance(value, b.bool)


def family_is_real(value: PythonObject) raises -> Bool:
    """An int or float that is not a bool."""
    var b = native_loaded_module("builtins")
    return family_isinstance(value, Python.tuple(b.int, b.float)) and not family_isinstance(value, b.bool)


def family_type_name(value: PythonObject) raises -> String:
    """`type(value).__name__`."""
    return String(py_type(value).__name__)


def family_call(callable: PythonObject, *args: PythonObject) raises -> PythonObject:
    """`callable(*args)` without retaining the arguments; a Python exception
    it raises is restored unchanged at the door."""
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_call_kw(callable: PythonObject, keywords: PythonObject, *args: PythonObject) raises -> PythonObject:
    """`callable(*args, **keywords)` (`keywords` a dict) without retaining
    the arguments; a Python exception it raises is restored unchanged."""
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, keywords._obj_ptr)
    cpy.Py_DecRef(envelope)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The compiled entry `name` of the binding the shell resolved for the
    model's numeric mode. A by-name refusal of the binding (a tier that is
    not built, an entry a CPU-only install does not have) is raised
    unchanged; anything other than a compiled entry is refused."""
    var entry = family_call(native_loaded_module("builtins").getattr, binding, PythonObject(name))
    if py_type(entry) is not context["family_builtin_type"]:
        raise Error("TypeError: mojolearn: " + name + " must be a compiled entry")
    return entry


def family_array(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """A new zero-filled C-order mojolearn Array (`_buffer.empty`/`zeros`),
    with or without NumPy: the outputs the Python shells always returned."""
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var width = 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        width = 4
    elif typestr == "<u1" or typestr == "<i1":
        width = 1
    var store = native_array_zeros(String(_store_code(typestr)), count)
    return classical_wrap_array(store, shape, typestr, "C", context)


def family_shape_repr(view: PythonObject) raises -> String:
    """`repr(tuple(a.shape))`."""
    var b = native_loaded_module("builtins")
    return family_repr(py_call(b.tuple, view.shape))


def family_view(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """The admitted input with `_buffer._as_typed`'s rank and emptiness
    refusals, in its wording; `ndim` < 0 accepts any rank."""
    var view = classical_view(source, context, name)
    var nd = Int(py=view.ndim)
    if ndim >= 0 and nd != ndim:
        var hint = String("")
        if ndim == 2 and nd < 2:
            hint = ". " + String(RESHAPE_HINT)
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(nd)
                    + "-D shape " + family_shape_repr(view) + hint)
    if Int(py=view.size) == 0:
        var what = String("sample(s)")
        if nd == 2 and Int(py=view.shape[0]) != 0:
            what = "feature(s)"
        var shape = family_shape_repr(view)
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    return view


def family_cast_f32(view: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """A checked view as C-ordered float32 (the input itself when it already
    is one), the cast `_buffer.as_f32_c` makes. A NaN or an infinity raises
    `Input <name> contains NaN or infinity`, which a caller may reword."""
    return classical_host_f32(view, Int(py=view.ndim), name, context)


def family_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f32_c(source, ndim=ndim, name=name)` with its refusals."""
    var view = family_view(source, ndim, name, context)
    var result = family_cast_f32(view, name, context)
    _ = view^
    return result


def family_cast_view(view: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f32_c`'s cast of a checked view of rank 0, 1 or 2, with
    no finiteness refusal (as_f32_c made none): the view itself when it is
    already float32 in C order, otherwise a new mojolearn Array of its shape.
    """
    var b = native_loaded_module("builtins")
    var nd = Int(py=view.ndim)
    if nd > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(nd) + "-D")
    if not classical_is_array(view, context) and not Bool(py=view.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    if classical_is_f32(view, context) and classical_c_order(view, context):
        return view
    var rows = 1 if nd == 0 else Int(py=view.shape[0])
    var cols = 1 if nd < 2 else Int(py=view.shape[1])
    var item = Int(py=view.itemsize)
    var kind = classical_cast_kind(view, item, name, context)
    var s0 = 0 if nd == 0 else Int(py=view.strides[0])
    var s1 = 0 if nd < 2 else Int(py=view.strides[1])
    var result = family_array(py_call(b.tuple, view.shape), "<f4", context)
    family_cast_rows(Int(py=classical_address(view)), s0, s1, rows, cols, item, kind,
                     FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))))
    _ = result  # keep the Python object alive through the native call above
    return result


def family_cast_view_f64(view: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f64_c`'s cast of a checked view of rank 0, 1 or 2: the
    view itself when it is already float64 in C order, otherwise a new
    mojolearn Array of its shape."""
    var b = native_loaded_module("builtins")
    var nd = Int(py=view.ndim)
    if nd > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(nd) + "-D")
    if not classical_is_array(view, context) and not Bool(py=view.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    if classical_dtype_str(view, context) == "<f8" and classical_c_order(view, context):
        return view
    var rows = 1 if nd == 0 else Int(py=view.shape[0])
    var cols = 1 if nd < 2 else Int(py=view.shape[1])
    var item = Int(py=view.itemsize)
    var kind = classical_cast_kind(view, item, name, context)
    var s0 = 0 if nd == 0 else Int(py=view.strides[0])
    var s1 = 0 if nd < 2 else Int(py=view.strides[1])
    var result = family_array(py_call(b.tuple, view.shape), "<f8", context)
    family_cast_rows_f64(Int(py=classical_address(view)), s0, s1, rows, cols, item, kind,
                         FAMILY_F64P(unsafe_from_address=Int(py=classical_address(result))))
    _ = result  # keep the Python object alive through the native call above
    return result


def family_f32_snapshot(source: PythonObject, ndim: Int, name: String,
                        context: PythonObject, dense: Bool = False) raises -> ArrayMetadata:
    # Common provider metadata admission for classical algorithms. Preserve
    # family_view's rank/empty refusals before casting or numerical execution.
    var prepared = source
    if dense:
        var sparse = host_sparse_format(source)
        if sparse.byte_length() > 0:
            if not py_contains(context, PythonObject("output_context")):
                py_setitem(context, "output_context", context)
            prepared = host_sparse_dense(source, sparse, context)
    var view = classical_view(prepared, context, name)
    var shape = array_shape_once(view, context)
    var nd = len(shape)
    var shape_words = Python.list()
    var count = 1
    for word in shape:  # driver-loop: over captured native array rank metadata
        _ = py_call(shape_words.append, word)
        count *= word
    var pyshape = py_call(native_loaded_module("builtins").tuple, shape_words)
    if ndim >= 0 and nd != ndim:
        var hint = String("")
        if ndim == 2 and nd < 2:
            hint = ". " + String(RESHAPE_HINT)
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(nd)
                    + "-D shape " + family_repr(pyshape) + hint)
    if count == 0:
        var what = String("sample(s)")
        if nd == 2 and shape[0] != 0:
            what = "feature(s)"
        var text = family_repr(pyshape)
        raise Error("mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
                    + " (shape=" + text + ") while a minimum of 1 is required.")
    if nd > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(nd) + "-D")
    var snapshot = array_metadata_once(view, shape^, name, context)
    if not snapshot.f32_c():
        var result = family_array(pyshape, "<f4", context)
        var address = Int(py=classical_address(result))
        var src = snapshot.source
        family_cast_rows(src.addr, src.s0, src.s1, src.rows, src.cols, src.itemsize, src.kind,
                         FAMILY_F32P(unsafe_from_address=address))
        snapshot.replace_f32(result, address)
    _ = view^
    _ = prepared^
    return snapshot^


def family_as_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    var snapshot = family_f32_snapshot(source, ndim, name, context)
    return snapshot.keep


def family_as_f32_dense(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    var snapshot = family_f32_snapshot(source, ndim, name, context, True)
    return snapshot.keep


def family_owned_snapshot(value: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    # Ownership is inspected once at the boundary; shape, count and address
    # are already admitted and must not be fetched again during the copy.
    if value.owned_writable:
        return value.keep
    var result = family_array(value.python_shape(), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
           src=FAMILY_F32P(unsafe_from_address=value.source.addr), count=value.count)
    _ = result
    return result


def family_snapshot_min(value: ArrayMetadata) raises -> Float32:
    return family_min_f32(FAMILY_F32P(unsafe_from_address=value.source.addr), value.count)


def family_owned(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value` (a C-order float32 array) as an owned mojolearn Array: the
    array itself when it is one the owner made, else one copy."""
    if classical_is_array(value, context) and value._base is None and not Bool(py=value._readonly):
        return value
    var b = native_loaded_module("builtins")
    var result = family_array(py_call(b.tuple, value.shape), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), count=Int(py=value.size))
    _ = result  # keep the Python object alive through the native call above
    return result


def family_min(value: PythonObject) raises -> Float32:
    """The minimum of a non-empty C-order float32 array."""
    return family_min_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), Int(py=value.size))


def family_nonfinite(error: Error, name: String) -> Bool:
    """True for the cast's NaN-or-infinity refusal of `name`."""
    return String(error) == "Input " + name + " contains NaN or infinity"


def family_sample_weight(source: PythonObject, n: Int, allow_negative: Bool, context: PythonObject,
                         wide: Bool = False) raises -> PythonObject:
    var snapshot = family_sample_weight_snapshot(source, n, allow_negative, context, wide)
    return snapshot.keep


def _family_weight_scalar(value: Float64, n: Int, wide: Bool,
                          context: PythonObject) raises -> ArrayMetadata:
    var item = 8 if wide else 4
    var weights = family_array(Python.tuple(n), "<f8" if wide else "<f4", context)
    var address = Int(py=classical_address(weights))
    if wide:
        family_fill_f64(FAMILY_F64P(unsafe_from_address=address), n, value)
    else:
        family_fill_f32(FAMILY_F32P(unsafe_from_address=address), n, Float32(value))
    var descriptor = CastIn(address, CAST_KIND_FLOAT, item, n, 1, item, 0)
    return ArrayMetadata(weights, List[Int]([n]), descriptor, n, NPY_DOUBLE if wide else NPY_FLOAT,
                         True, True, True, weights, True)


def _family_weight_buffer(source: PythonObject, n: Int, wide: Bool,
                          context: PythonObject) raises -> ArrayMetadata:
    # Admit supported native storage before reading any user attributes. One
    # shape and dtype/stride snapshot is retained through validation and use.
    var view = classical_view(source, context, "sample_weight")
    var shape = array_shape_once(view, context)
    var count = 1
    for dimension in shape:  # small-loop(admitted sample-weight rank)
        count *= dimension
    if count == 0:
        raise Error("sample_weight.shape == (0,), expected (" + String(n) + ",)!")
    var nd = len(shape)
    if nd == 0:
        # Existing zero-dimensional buffer semantics are a broadcast scalar.
        # Read it with the existing native widening cast; never float(Array).
        if py_type(source) is py_type(PythonObject(True)):
            raise Error("Sample weights must be 1D array or scalar")
        var scalar = array_metadata_once(view, shape^, "sample_weight", context, True)
        var one = family_array(Python.tuple(1), "<f8", context)
        var ptr = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(one)))
        var raw = scalar.source
        var scalar_address = raw.addr
        var reversed = family_array(Python.tuple(raw.itemsize), "<u1", context)
        if not scalar.native_byte_order:
            # float(zero_dim_ndarray) previously accepted swapped scalars.
            # Preserve that scalar-only behavior before the unchanged cast.
            var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=raw.addr)
            var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(reversed)))
            for byte in range(raw.itemsize):  # small-loop(one admitted scalar word)
                dst[byte] = src[raw.itemsize - 1 - byte]
            scalar_address = Int(py=classical_address(reversed))
        family_cast_rows_f64(scalar_address, 0, 0, 1, 1, raw.itemsize, raw.kind, ptr)
        var result = _family_weight_scalar(ptr[0], n, wide, context)
        _ = scalar
        _ = one
        _ = reversed
        return result^
    if nd != 1:
        raise Error("Sample weights must be 1D array or scalar")
    if shape[0] != n:
        var words = Python.list()
        for dimension in shape:  # small-loop(admitted rank for original error)
            _ = py_call(words.append, dimension)
        raise Error("sample_weight.shape == " + family_repr(py_call(py_type(Python.tuple()), words))
                    + ", expected (" + String(n) + ",)!")
    var snapshot = array_metadata_once(view, shape^, "sample_weight", context)
    var ready = snapshot.c_order and snapshot.dtype_number == (NPY_DOUBLE if wide else NPY_FLOAT)
    if not ready:
        var weights = family_array(Python.tuple(n), "<f8" if wide else "<f4", context)
        var address = Int(py=classical_address(weights))
        var raw = snapshot.source
        if wide:
            family_cast_rows_f64(raw.addr, raw.s0, raw.s1, raw.rows, raw.cols, raw.itemsize, raw.kind,
                                 FAMILY_F64P(unsafe_from_address=address))
            snapshot.replace_f64(weights, address)
        else:
            family_cast_rows(raw.addr, raw.s0, raw.s1, raw.rows, raw.cols, raw.itemsize, raw.kind,
                             FAMILY_F32P(unsafe_from_address=address))
            snapshot.replace_f32(weights, address)
    return snapshot^


def family_sample_weight_snapshot(source: PythonObject, n: Int, allow_negative: Bool,
                                  context: PythonObject, wide: Bool = False) raises -> ArrayMetadata:
    var b = native_loaded_module("builtins")
    var snapshot: ArrayMetadata
    if py_type(source) is b.int or py_type(source) is b.float:
        snapshot = _family_weight_scalar(Float64(py=source), n, wide, context)
    else:
        snapshot = _family_weight_buffer(source, n, wide, context)
    var address = snapshot.source.addr
    var status: Int
    if wide:
        status = family_weight_status_f64(FAMILY_F64P(unsafe_from_address=address), n, allow_negative)
    else:
        status = family_weight_status(FAMILY_F32P(unsafe_from_address=address), n, allow_negative)
    if status == FAMILY_WEIGHTS_NONFINITE:
        raise Error(FAMILY_NONFINITE_WEIGHTS)
    if status == FAMILY_WEIGHTS_NEGATIVE:
        raise Error(FAMILY_NEGATIVE_WEIGHTS)
    if status == FAMILY_WEIGHTS_ALL_ZERO:
        raise Error(FAMILY_ZERO_WEIGHTS)
    return snapshot^


def family_field(fields: PythonObject, name: String) raises -> PythonObject:
    """`fields[name]`, refused as the attribute lookup refused it."""
    var value = py_get(fields, PythonObject(name), PythonObject(None))
    if value is None and not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value
