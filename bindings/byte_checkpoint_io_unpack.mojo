# SPDX-License-Identifier: Apache-2.0
"""Preserve native file exceptions across the checkpoint's Mojo unwind."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.ffi import c_int, external_call
from core.native_string import unsafe_string_from_utf8_cstr
from bindings.py_balanced import py_call, py_type


struct ByteCheckpointIOError(Movable):
    var pending: Bool
    var has_value: Bool
    var has_traceback: Bool
    var kind: PythonObject
    var value: PythonObject
    var traceback: PythonObject

    def __init__(out self):
        self.pending = False
        self.has_value = False
        self.has_traceback = False
        self.kind = PythonObject(None)
        self.value = PythonObject(None)
        self.traceback = PythonObject(None)

    def capture(mut self):
        ref cpy = Python().cpython()
        var triple = cpy.PyErr_FetchTriple()
        # FetchTriple transfers its three references. PythonObject owns each
        # non-null reference, including cleanup errors that are discarded while
        # preserving the first failure. This is ordinary C API ownership, not
        # compensation for std.python's separate reference-retention defect.
        var kind = PythonObject(from_owned=triple[0]) if triple[0] else PythonObject(None)
        var value = PythonObject(from_owned=triple[1]) if triple[1] else PythonObject(None)
        var traceback = PythonObject(from_owned=triple[2]) if triple[2] else PythonObject(None)
        if not self.pending:
            self.pending = True
            self.has_value = Bool(triple[1])
            self.has_traceback = Bool(triple[2])
            self.kind = kind^
            self.value = value^
            self.traceback = traceback^

    def restore(var self) -> PyObjectPtr:
        ref cpy = Python().cpython()
        var has_value = self.has_value
        var has_traceback = self.has_traceback
        # Keep this aggregate fully initialized until its ordinary destruction.
        # PythonObject's supported copy ownership gives Restore separate owned
        # references; moving individual fields leaves a partially consumed self.
        var kind = self.kind
        var value = self.value
        var traceback = self.traceback
        var value_ptr = value.steal_data() if has_value else PyObjectPtr()
        var traceback_ptr = traceback.steal_data() if has_traceback else PyObjectPtr()
        # Restore steals the owned copies of the references from FetchTriple.
        # It preserves the original subclass, errno and filename attributes.
        # https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Restore
        cpy.PyErr_Restore(kind.steal_data(), value_ptr, traceback_ptr)
        return PyObjectPtr()


def byte_io_file_error(mut io_error: ByteCheckpointIOError, os_error: PythonObject, number: Int, operation: String,
                       path: String, second: String) raises:
    """PACK a failed typed file call (bindings/byte_checkpoint_native_run.mojo
    `ByteCheckpointWriteFailure`: errno, the call's name, its file names) as
    the exact OSError subclass of its errno, for the C entry to raise.
    `os_error` is the builtin `OSError`. The words the Python file calls
    gave: `open` names the file (the temporary one of a write; a directory
    given to a read); `replace` names both files; a failed read, write,
    fsync or close carries no file name."""
    var message = unsafe_string_from_utf8_cstr(external_call["strerror", Pointer[Int8, ImmUntrackedOrigin]](c_int(number)))
    var value = py_call(os_error, number, PythonObject(message))
    if operation == "open":
        value = py_call(os_error, number, PythonObject(message), PythonObject(path))
    elif operation == "replace":
        value = py_call(os_error, number, PythonObject(message), PythonObject(path),
                        PythonObject(None), PythonObject(second))
    io_error.pending = True
    io_error.has_value = True
    io_error.has_traceback = False
    io_error.kind = py_type(value)
    io_error.value = value
