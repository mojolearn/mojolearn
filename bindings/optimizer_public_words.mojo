# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The error boundary of the public optimizers' entries, shared by the GPU
door (bindings/optimizer_public_pyapi.mojo), the CPU door
(bindings/optimizer_public_host_pyapi.mojo), the core binding's doors
(bindings/optimizer_public_door.mojo) and the schedules
(bindings/lr_table_helpers.mojo): unpacking and raising only, no device code
(lane neural-optimizers, 2026-10-21; lane optimizers-thin-glue)."""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception

#: The message of an Error that stands for a Python exception a caller's own
#: object raised (a user `lr_schedule.lr_at`): `optimizer_python_error`
#: restores that exception unchanged instead of raising a new one.
comptime PYTHON_ERROR_PENDING = "mojolearn-optimizer: a Python exception is pending"


struct PendingPythonError(Defaultable, Movable):
    """The (type, value, traceback) of a caller's exception, fetched with
    `PyErr_Fetch` and restored once with `PyErr_Restore` (both own the three
    references), as addresses."""
    var words: List[Int]

    def __init__(out self):
        self.words = List[Int](length=3, fill=0)


comptime PENDING_PYTHON_ERROR = _Global[
    StorageType=PendingPythonError, name="MojoOptimizerPendingPythonError",
    init_fn=PendingPythonError.__init__,
]


def stash_python_error() raises -> Error:
    """Fetch the exception the last CPython call set (clearing the
    indicator, keeping its references) and return the Error that stands for
    it; `optimizer_python_error` puts it back."""
    ref cpy = Python().cpython()
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    var p = pending[].words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    cpy.lib.get_function[NoneType]("PyErr_Fetch")(p, p + 1, p + 2)
    return Error(PYTHON_ERROR_PENDING)


def _restore_python_error() -> PyObjectPtr:
    try:
        ref cpy = Python().cpython()
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        var w = pending[].words.copy()
        pending[].words = List[Int](length=3, fill=0)
        cpy.lib.get_function[NoneType]("PyErr_Restore")(w[0], w[1], w[2])
        return PyObjectPtr()
    except error:
        return raise_python_exception(error)


def optimizer_python_error(error: Error) -> PyObjectPtr:
    """Raise the Python class a kernel or door error names
    (`ValueError: ...`), once without its prefix; restore a caller's own
    pending exception; any other error raises `Exception`."""
    var message = String(error)
    if message == PYTHON_ERROR_PENDING:
        return _restore_python_error()
    if message.startswith("ValueError: "):
        return raise_python_exception(Error(message[byte=12:]), ExceptionType.ValueError)
    if message.startswith("TypeError: "):
        return raise_python_exception(Error(message[byte=11:]), ExceptionType.TypeError)
    if message.startswith("RuntimeError: "):
        return raise_python_exception(Error(message[byte=14:]), ExceptionType("PyExc_RuntimeError"))
    if message.startswith("NotImplementedError: "):
        return raise_python_exception(Error(message[byte=21:]), ExceptionType("PyExc_NotImplementedError"))
    if message.startswith("OverflowError: "):
        return raise_python_exception(Error(message[byte=15:]), ExceptionType("PyExc_OverflowError"))
    if message.startswith("ArithmeticError: "):
        return raise_python_exception(Error(message[byte=17:]), ExceptionType("PyExc_ArithmeticError"))
    if message.startswith("KeyError: "):
        return raise_python_exception(Error(message[byte=10:]), ExceptionType("PyExc_KeyError"))
    return raise_python_exception(error)


def int_words(words: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(words)):  # small-loop(argument words: a fixed count, or one per tensor)
        out.append(Int(py=words[i]))
    return out^


def float_words(words: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(words)):  # small-loop(argument words: a fixed count)
        out.append(Float64(py=words[i]))
    return out^
