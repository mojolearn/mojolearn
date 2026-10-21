# SPDX-License-Identifier: Apache-2.0
"""Preserve native file exceptions across the checkpoint's Mojo unwind."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


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
