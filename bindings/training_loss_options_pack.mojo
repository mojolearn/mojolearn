# SPDX-License-Identifier: Apache-2.0
# Native truth and reduction configuration for the public training loss.
# The PACK half of the former `bindings/training_loss_options.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/training_loss_options_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get
from bindings.optimizer_admission import opt_exact, opt_value
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.native_numpy_metadata_pack import native_numpy_type
from bindings.sequence_dtype_native_door import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native_unpack import sequence_array_release
from bindings.training_scalar_native_door import training_scalar_warning


def loss_native_item(value: PythonObject, key: PythonObject) raises -> PythonObject:
    # Callers first admit the exact native memoryview or ndarray/void dtype
    # and key; no user indexing protocol reaches this public CPython slot.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, key._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)
