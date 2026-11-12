# SPDX-License-Identifier: Apache-2.0
# Shared result builders for the byte LM, Mamba and Samba bindings: a Python
# list of ints published in one pass through the CPython C API (PyList_New,
# PyLong_FromLongLong, PyList_SetItem), with no per-item `list.append` call.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.ffi import c_int
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer
from bindings.py_cpython_table import cpython_table


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
