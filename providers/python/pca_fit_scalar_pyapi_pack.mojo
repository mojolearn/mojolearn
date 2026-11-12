# SPDX-License-Identifier: Apache-2.0
# Mutable scalar-array settings are captured before input preparation can
# release the GIL. Conversion, warnings and failures stay at their historical
# validation point. These native records never escape through the Python ABI.
# The PACK half of the former `providers/python/pca_fit_scalar_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/pca_fit_scalar_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_get, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.sequential_admission import sequential_admit_value
from bindings.native_scalar_decode_unpack import NativeScalarBuffer, native_scalar_buffer_dtype
from bindings.native_numpy_metadata_pack import native_numpy_module_fields
from bindings.sequence_dtype_native_door import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native_unpack import sequence_array_release
from bindings.training_scalar_native_door import training_scalar_warning
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, stash_python_error, _restore_python_error
from bindings.manifold_temporal_native import temporal_object_kind
from core.native_scalar_decode import NativeRealBits
from x_decomp.pca_fit_scalar_run import PCAFitScalarArray


def _pca_scalar_bytes(value: PythonObject) raises -> PythonObject:
    # Public native buffer transport without requesting PEP3118 format:
    # datetime/timedelta and Unicode need not expose a format spelling.
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var words = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, words, Int32(0x118)) != 0:
        var error = stash_python_error()
        raise error^
    var result = PythonObject(None)
    try:
        var bytes = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(Int(0), view[2])
        if not bytes:
            var error = stash_python_error()
            raise error^
        result = PythonObject(from_owned=bytes)
        var address = cpy.lib.get_function[Int]("PyBytes_AsString")(result._obj_ptr)
        if not address or cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(address, words, view[2], Int8(67)) != 0:
            var error = stash_python_error()
            raise error^
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(words)
        # Keep the Py_buffer storage alive past the release on every path
        # (ASAP destruction could free it before PyBuffer_Release reads it).
        _ = len(view)
    return result


def _pca_builtin_number(value: PythonObject, integer: Bool, context: PythonObject, error_repr: String = "") raises -> PythonObject:
    # Only already-admitted native builtins/buffer wrappers reach this
    # parser. NumPy temporal/numeric conversion slots never do.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is not None and cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.complex and cls is not b.str and cls is not b.bytes and cls is not b.bytearray and cls is not b.memoryview and cls is not b.list and cls is not b.tuple and cls is not b.dict and cls is not b.set and cls is not b.frozenset and cls is not context["array_type"]:
        raise Error("TypeError: PCA scalar conversion requires native numbers or text without Python conversion hooks")
    var value_error = b.ValueError
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Long" if integer else "PyNumber_Float")(value._obj_ptr)
    if not pointer:
        if not integer and error_repr.byte_length() > 0 and cpython_table()[].PyErr_ExceptionMatches(value_error._obj_ptr) != 0:
            # CPython floatobject.c uses the original object's repr for a
            # text parse refusal. A private memoryview has a different address;
            # preserve the exact native original repr captured before copying.
            var message = "could not convert string to float: " + error_repr
            cpy.PyErr_SetString(value_error._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
        var error = stash_python_error()
        raise error^
    return PythonObject(from_owned=pointer)
