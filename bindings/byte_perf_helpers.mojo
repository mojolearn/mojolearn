# SPDX-License-Identifier: Apache-2.0
# Shared result builders for the byte LM, Mamba and Samba bindings: a Python
# list of ints published in one pass through the CPython C API (PyList_New,
# PyLong_FromLongLong, PyList_SetItem), with no per-item `list.append` call.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.ffi import c_int
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from bindings.cpython_table import cpython_table


def byte_int_list_from_i32(address: Int, count: Int) raises -> PythonObject:
    """A new list of `count` Python ints read from native int32 storage."""
    ref cpy = Python().cpython()
    var table = cpython_table()
    var raw = table[].PyList_New(count)
    if not raw:
        raise cpy.unsafe_get_error()
    # The list owns itself at once; unset NULL slots are safe to destroy.
    var out = PythonObject(from_owned=raw)
    # Entries resolved once per process (no by-name lookup per call).
    var box_int = table[].PyLong_FromLongLong
    var set_item = table[].PyList_SetItem
    var source = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # cpu-route: publish completed native token ids as Python integers
        var item = box_int(Int64(source[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    return out


def byte_int_list_from_i64(address: Int, count: Int) raises -> PythonObject:
    """A new list of `count` Python ints read from native int64 storage."""
    ref cpy = Python().cpython()
    var table = cpython_table()
    var raw = table[].PyList_New(count)
    if not raw:
        raise cpy.unsafe_get_error()
    var out = PythonObject(from_owned=raw)
    # Entries resolved once per process (no by-name lookup per call).
    var box_int = table[].PyLong_FromLongLong
    var set_item = table[].PyList_SetItem
    var source = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # cpu-route: publish completed native token ids as Python integers
        var item = box_int(source[i])
        if not item:
            raise cpy.unsafe_get_error()
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    return out


def byte_view_bytes(view: PythonObject, minimum: Int) raises -> List[UInt8]:
    """The C-order bytes of an admitted memoryview (what `view.tobytes("C")`
    returns), copied once by native code under one buffer pin, instead of a
    Python bytes copy followed by a second `array.frombytes` copy. Refuses a
    buffer holding fewer than `minimum` bytes, which a reader would overrun."""
    var pin = AdmittedBorrowedBuffer(view)
    var count = pin.layout.value().count * pin.layout.value().itemsize
    if count < minimum:
        pin.release()
        raise Error("Samba: token id buffer is smaller than its (B, L) shape")
    var out = List[UInt8](length=count, fill=0)
    try:
        pin.layout.value().copy_to(Int(out.unsafe_ptr()))
    except error:
        pin.release()
        raise error
    pin.release()
    _ = view
    return out^
