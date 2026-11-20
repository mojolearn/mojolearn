# SPDX-License-Identifier: Apache-2.0
# Shared native label preparation/publication for the neural classifiers. Numeric
# sorting stays in the connected neural provider; text uses native code points.
# The PACK half of the former `bindings/neural_labels_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/neural_labels_native_unpack.mojo`.
from std.memory import unsafe_memcpy
from std.sys import size_of
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, ExternalFunction
from std.ffi import c_int
from bindings.py_balanced import py_call, py_type
from bindings.py_cpython_table import cpython_table
from core.borrowed_buffer import BorrowedBufferLayout
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.sequential_admission_door import sequential_admit_value, SequentialValueAdmitter
from bindings.sequential_inputs_door import sequential_input
from bindings.sequential_arrays_door import sequential_array
from bindings.xseq_native_common_door import xseq_neural_provider
from bindings.str_labels_native_door import _read, str_unique_into
from bindings.str_labels_native_run import StrTable
from bindings.native_buffer_address_unpack import native_buffer_address
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from sequence.neural_model_dispatch import MODEL_LABELS
from core.cast_in import cast_in_kind
from bindings.admitted_input_unpack import AdmittedTypedPin, admit_view
from bindings.admitted_input_run import admitted_view_kind
from bindings.tensor_view import TensorView
from bindings.tensor_view import base_format




# TOMBSTONE: _numeric's own capture (sequential_input + AdmittedBorrowedBuffer of the materialized array; superseded by
# the one admission, bindings/admitted_input_unpack.mojo admit_view) deleted 2026-11-09 by lane/m1-followups-misc; code
# recoverable at 5b5490960d.
# Restore: git apply experiments/removed/NEURAL_LABELS_OWN_ADMISSION.patch
def _numeric(v: TensorView, var captured: AdmittedTypedPin, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """The sorted distinct classes and the int32 codes of a numeric label
    vector the one admission captured (any numeric dtype, its own strides)."""
    var rank = len(v.shape)
    if rank > 2 or (rank == 2 and v.shape[1] != 1):
        raise Error("ValueError: y must be a label vector")
    var n = v.count()
    if n < 1:
        raise Error("ValueError: y is empty")
    var t = admitted_view_kind(v, "y")
    # the classes keep the labels' own dtype: an ndarray's dtype, an Array's dtype string, or the buffer format
    var dtype = PythonObject(None)
    if py_type(captured.keep) is context["ndarray"]:
        dtype = captured.keep.dtype
    else:
        dtype = py_call(native_numpy_dtype(context), v.dtype if v.is_array else base_format(v.fmt))
    if not Bool(py=dtype.isnative):
        raise Error("TypeError: y must use native byte order")
    var classes = sequential_array(Python.tuple(n), dtype, context)
    var codes = sequential_array(Python.tuple(n), context["int32"], context)
    var stride = v.strides[0] if rank > 0 else t[1]
    var addresses: List[Int] = [native_buffer_address(classes), native_buffer_address(codes)]
    var words: List[Int] = [v.addr, t[0], t[1], n, 1, stride, t[1]]
    var count = neural_descriptor_execute(xseq_neural_provider(entries), MODEL_LABELS, addresses, words, List[Float64]())
    var selected = py_call(context["ndarray"], Python.tuple(count), dtype=dtype, buffer=classes)
    _ = captured^
    return Python.tuple(selected, codes)


def _object_array(count: Int, context: PythonObject) raises -> PythonObject:
    # Object arrays must own their allocation so NumPy owns/decrefs every slot.
    # A byte-buffer-backed ndarray would not own the referenced label objects.
    var dtype = py_call(native_numpy_dtype(context), "O")
    return py_call(context["ndarray"], Python.tuple(count), dtype=dtype)
