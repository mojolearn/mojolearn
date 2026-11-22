# SPDX-License-Identifier: Apache-2.0
"""`cpython_utf8`: PyUnicode_AsUTF8AndSize with its size read back safely
(lane sr-gbdt, 2026-11-22). Its own module so the shared CPython table
(bindings/py_cpython_table.mojo) is unchanged."""
from std.memory import stack_allocation
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import CPythonTable


def cpython_utf8(table: UnsafePointer[CPythonTable, MutAnyOrigin], obj: PyObjectPtr, mut nbytes: Int) -> Int:
    """`PyUnicode_AsUTF8AndSize(obj, &size)`: the address of the str's UTF-8
    bytes (0 with the Python error pending) and their count in `nbytes`.
    The bytes live as long as `obj`. `table` is the caller's
    `cpython_table()`, fetched once outside its loops.

    THE ONE FORM for this call. The size travels through a stack slot read
    back with a volatile load. The former spelling
    `var size = 0; ...(obj, Int(MutPointer(to=size))); nbytes = size` let the
    compiler forward the 0 it stored past the C call, which it does not see
    writing through an address Int: in the 9e49098bca wheel
    `gbdt_apply_unpack` passed the constant 0 as the model text's byte count
    (gdb: CPython wrote 17110 into the slot, the next instruction was
    `xor %ecx,%ecx`), so every GBDT predict parsed an empty text ("the model
    text is incomplete"). A volatile load is never forwarded."""
    var slot = stack_allocation[1, Int]()
    slot.store[volatile=True](0)
    var address = table[].PyUnicode_AsUTF8AndSize(obj, Int(slot))
    nbytes = slot.load[volatile=True]()
    return address
