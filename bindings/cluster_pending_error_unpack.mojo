# SPDX-License-Identifier: Apache-2.0
# The Python exception `family_call` stashed (bindings/optimizer_public_words_unpack.mojo
# PENDING_PYTHON_ERROR), read at the boundary: its type, its text, its
# release. Moved whole from bindings/cluster_fit_a_native_door.mojo when
# that door was split; the lifecycle doors that turn a stashed exception into
# their own refusal import these.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, PENDING_PYTHON_ERROR


def _pending(index: Int) raises -> PythonObject:
    """A new reference to word `index` (0 type, 1 value) of the Python
    exception `family_call` stashed, or None. The stash is kept."""
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    var address = pending[].words[index]
    if address == 0:
        return PythonObject(None)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("Py_NewRef")(address)
    return PythonObject(from_owned=pointer)


def _drop_pending() raises:
    """Release the stashed exception: the caller raises its own instead."""
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[NoneType]("Py_DecRef")
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    for t in range(3):  # small-loop(the stashed exception's three references)
        var address = pending[].words[t]
        if address:
            release(address)
            pending[].words[t] = 0


def _pending_text(error: Error) raises -> String:
    """`str(e)` of the stashed exception (`error` itself otherwise)."""
    if String(error) != PYTHON_ERROR_PENDING:
        return String(error)
    var value = _pending(1)
    if value is None:
        return String("")
    return String(py=py_call(native_loaded_module("builtins").str, value))
