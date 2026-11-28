# SPDX-License-Identifier: Apache-2.0
# Native truth and reduction configuration for the public training loss.
# The PACK half of the former `bindings/training_loss_options.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/training_loss_options_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.optimizer_public_words_unpack import stash_python_error


def loss_native_item(value: PythonObject, key: PythonObject) raises -> PythonObject:
    # Callers first admit the exact native memoryview or ndarray/void dtype
    # and key; no user indexing protocol reaches this public CPython slot.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, key._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)
