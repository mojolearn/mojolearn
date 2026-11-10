# SPDX-License-Identifier: Apache-2.0
"""Archive PACK: wrap typed results and emit admitted native warning outputs.

Python container creation requires the calling thread's GIL and stays ordered.
Parsing, layout, decompression and text decoding have already completed in the
native host pool. Arrays receive one writable owned copy of native payload bytes.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_setitem
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_arrays import classical_numpy, classical_wrap_layout
from bindings.classical_archive import _archive_file_call
from bindings.classical_warnings import classical_warning
from providers.python.array_layout_boundary import native_shape_pack
from core.native_array_layout import NativeArrayLayout, native_array_code
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_npy import ArchiveLiteralWarning


def archive_warning_pack(warning: ArchiveLiteralWarning, context: PythonObject) raises:
    var message = String("invalid escape sequence '\\")
    if warning.kind == 2:
        message = "invalid octal escape sequence '\\"
        message += chr(48 + ((warning.value >> 6) & 7))
        message += chr(48 + ((warning.value >> 3) & 7))
        message += chr(48 + (warning.value & 7))
    else:
        message += chr(warning.value)
    message += "'"
    # Native sys._getframe sees the same public Python caller as the former
    # C codec. No Python stack-walk helper or application warning hook runs.
    var frame = _archive_file_call(context["warning_getframe"])
    var location = Python.tuple(frame.f_code.co_filename, frame.f_lineno, frame.f_globals["__name__"])
    # The incumbent Mojo codec door surfaced warning-as-error text as a
    # ValueError. The shared emitter admits exact filters/recording/output.
    classical_warning(message, context["warning_deprecation_type"], context, location, "ValueError")


def archive_events_pack(members: List[NativeArchiveDecoded], context: PythonObject) raises:
    # Restore member/token event order after independent native tasks joined.
    for i in range(len(members)):
        for warning in members[i].warnings:
            archive_warning_pack(warning, context)
        if members[i].error.byte_length():
            raise Error(members[i].error)


def archive_numeric_pack(member: NativeArchiveDecoded, context: PythonObject) raises -> PythonObject:
    var shape = native_shape_pack(member.layout.shape)
    var order = String("F" if member.layout.fortran else "C")
    var payload = member.layout.payload.copy()
    if classical_numpy(context):
        var raw = cpython_table()[].PyByteArray_FromStringAndSize(payload.address, payload.size)
        if not raw:
            var failure = stash_python_error()
            raise failure^
        var storage = PythonObject(from_owned=raw)
        return py_call(context["ndarray"], shape, dtype=context["archive_npy_dtypes"][PythonObject(member.layout.dtype)], buffer=storage, order=order)
    var layout = NativeArrayLayout(member.layout.shape.copy(), member.layout.width, order)
    var store = py_call(context["array_type"], native_array_code(member.layout.dtype))
    var raw = Python().cpython().lib.get_function[PyObjectPtr]("PyMemoryView_FromMemory")(
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=payload.address), payload.size, Int32(0x100))
    if not raw:
        var failure = stash_python_error()
        raise failure^
    var view = PythonObject(from_owned=raw)
    # frombytes is the immutable array.array C descriptor. Passing a memoryview
    # to the constructor would instead iterate/cast its byte elements.
    _ = py_call(store.frombytes, view)
    var result = classical_wrap_layout(store, layout, member.layout.dtype, context)
    _ = view
    _ = member
    return result^


def archive_text_cell_pack(member: NativeArchiveDecoded, index: Int) raises -> PythonObject:
    var result = PyObjectPtr()
    if member.layout.kind == 1:
        result = cpython_table()[].PyUnicode_FromKindAndData(Int32(4), Int(member.texts[index].points.unsafe_ptr()), len(member.texts[index].points))
    else:
        result = cpython_table()[].PyBytes_FromStringAndSize(member.texts[index].bytes.address, member.texts[index].bytes.size)
    if not result:
        var failure = stash_python_error()
        raise failure^
    _ = member
    return PythonObject(from_owned=result)


def archive_text_nest_pack(member: NativeArchiveDecoded, axis: Int, offset: Int) raises -> PythonObject:
    ref cpy = Python().cpython()
    var count = member.layout.shape[axis]
    var raw = cpy.PyList_New(count)
    if not raw:
        var failure = stash_python_error()
        raise failure^
    var result = PythonObject(from_owned=raw)
    var stride = 1
    for k in range(axis + 1, len(member.layout.shape)):
        stride *= member.layout.shape[k]
    for i in range(count):
        var item = PythonObject(None)
        if axis + 1 == len(member.layout.shape):
            item = archive_text_cell_pack(member, offset + i)
        else:
            item = archive_text_nest_pack(member, axis + 1, offset + i * stride)
        _ = cpy.PyList_SetItem(raw, i, cpy.Py_NewRef(item._obj_ptr))
    return result^


def archive_member_pack(member: NativeArchiveDecoded, context: PythonObject) raises -> PythonObject:
    if member.layout.kind == 0:
        return archive_numeric_pack(member, context)
    if member.layout.count == 1 or len(member.layout.shape) == 0:
        return archive_text_cell_pack(member, 0)
    return archive_text_nest_pack(member, 0, 0)


def archive_members_pack(members: List[NativeArchiveDecoded], context: PythonObject,
                         emit_events: Bool = True) raises -> PythonObject:
    if emit_events:
        archive_events_pack(members, context)
    var result = Python.dict()
    for i in range(len(members)):
        py_setitem(result, PythonObject(members[i].name), archive_member_pack(members[i], context))
    return result
