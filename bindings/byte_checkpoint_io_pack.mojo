# SPDX-License-Identifier: Apache-2.0
"""Preserve native file exceptions across the checkpoint's Mojo unwind."""
# The PACK half of the former `bindings/byte_checkpoint_io.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/byte_checkpoint_io_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


from bindings.byte_checkpoint_io_unpack import (
    ByteCheckpointIOError,
)


def byte_checkpoint_io(mut failure: ByteCheckpointIOError, var callable: PythonObject,
                      var args: List[PythonObject]) raises -> PythonObject:
    ref cpy = Python().cpython()
    var argv = cpy.PyTuple_New(len(args))
    if not argv:
        failure.capture()
        raise Error("Byte-LM checkpoint native I/O argument allocation failed")
    for i in range(len(args)):  # driver-loop: over native file-call arguments
        # SetItem steals each argument reference; CallObject borrows the tuple
        # and callable. Release our owned call envelope once, on both outcomes.
        if cpy.PyTuple_SetItem(argv, i, args[i].steal_data()) != 0:
            failure.capture()
            cpy.Py_DecRef(argv)
            raise Error("Byte-LM checkpoint native I/O argument construction failed")
    var entry = callable.steal_data()
    var result = cpy.PyObject_CallObject(entry, argv)
    cpy.Py_DecRef(entry)
    cpy.Py_DecRef(argv)
    if not result:
        failure.capture()
        raise Error("Byte-LM checkpoint native I/O failed")
    return PythonObject(from_owned=result)
