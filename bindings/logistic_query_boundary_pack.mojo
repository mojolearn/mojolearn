# SPDX-License-Identifier: Apache-2.0
"""Exact public feature-field UNPACK for native Logistic query decisions."""
# The PACK half of the former `bindings/logistic_query_boundary.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/logistic_query_boundary_unpack.mojo`.
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from core.native_scalar_decode import NativeRealBits, native_scalar_real, _scalar_ieee
from core.native_array_layout import native_array_itemsize
from core.logistic_query_state import LogisticQueryWidth, logistic_width_integer
from core.logistic_query_boundary_run import logistic_torch_width_run
from core.sequence_array_cast import SEQ_CAST_OVERFLOW
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_scalar_decode_unpack import native_scalar_buffer_dtype, native_scalar_buffer_bytes
from bindings.native_scalar_promotion_unpack import native_scalar_weak_promotion
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_torch_admission import native_torch_exact, native_torch_scalar
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.sequence_array_native_unpack import sequence_array_release
from bindings.sequence_array_warnings_unpack import sequence_array_warn, sequence_array_report
from bindings.sequence_array_warnings_pack import sequence_array_errstate

def _logistic_width_numpy_dtype(value: PythonObject, context: PythonObject,
                                array: Bool) raises -> PythonObject:
    # Descriptor identities, not user dtype attributes, admit the storage.
    # Ordinary fitted integer fields never enter this exceptional metadata
    # path. The existing native dtype constructor supplies immutable types.
    var cls = py_type(value)
    var scalar = False
    var classes = context["scalar_types"]
    for i in range(len(classes)):  # driver-loop: fixed connected primitive scalar classes
        scalar = scalar or cls is classes[i]
    if scalar:
        # An exact connected primitive scalar has its immutable native dtype;
        # do not reconstruct a dtype or rescan its type before each query.
        return value.dtype
    var constructor = native_numpy_dtype(context)
    if not array and not scalar:
        for code in ["q", "Q", "g", "F", "D", "G", "M8", "m8", "S", "U", "V"]:  # driver-loop: native scalar aliases exposed by object-array UNPACK
            scalar = scalar or cls is py_call(constructor, code).type
    if not array and not scalar:
        return PythonObject(None)
    var dtype = value.dtype
    if scalar:
        return dtype
    var numeric = context["numeric_dtypes"]
    if py_type(numeric) is not py_type(Python.dict()):
        raise Error("TypeError: LogisticRegression requires connected native dtype metadata")
    var position = Int(0)
    var key = PyObjectPtr()
    var item = PyObjectPtr()
    while cpython_table()[].PyDict_Next(numeric._obj_ptr, Int(MutPointer(to=position)), Int(MutPointer(to=key)), Int(MutPointer(to=item))) != 0:  # driver-loop: retained primitive dtype owners, no list or dtype allocation
        var known = PythonObject(from_borrowed=item)
        if py_type(dtype) is py_type(known):
            return dtype
    for code in ["q", "Q", "g", "F", "D", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: remaining native dtype identities, never element data
        if py_type(dtype) is py_type(py_call(constructor, code)):
            return dtype
    if sequence_array_release(context) >= 200:
        if py_type(dtype) is py_type(py_call(constructor, "T")):
            return dtype
    raise Error("TypeError: LogisticRegression feature metadata requires a built-in native dtype")


def _logistic_width_item(value: PythonObject) raises -> PythonObject:
    # Exact ndarray native indexing unpacks the one element. No array method
    # or user scalar comparison executes; the returned owner is admitted on
    # the next bounded walk iteration before any conversion can use it.
    var rank = Int(py=value.ndim)
    ref cpy = Python().cpython()
    var index = cpy.PyTuple_New(rank)
    if not index:
        raise cpy.unsafe_get_error()
    var owner = PythonObject(from_owned=index)
    for i in range(rank):  # driver-loop: one all-zero index, bounded by ndarray rank
        var zero = PythonObject(0)
        if cpy.PyTuple_SetItem(index, i, zero.steal_data()) != 0:
            raise cpy.unsafe_get_error()
    var pointer = cpython_table()[].PyObject_GetItem(value._obj_ptr, owner._obj_ptr)
    _ = owner^
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)
