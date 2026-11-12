# SPDX-License-Identifier: Apache-2.0
# Shared native label preparation/publication for the neural classifiers. Numeric
# sorting stays in the connected neural provider; text uses native code points.
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
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute
from sequence.neural_model_dispatch import MODEL_LABELS
from core.cast_in import cast_in_kind
from bindings.admitted_input import AdmittedTypedPin, admit_view, admitted_view_kind
from bindings.tensor_view import TensorView
from bindings.tensor_view import base_format




from bindings.neural_labels_native_pack import (
    _numeric,
    _object_array,
)


def _item(layout: BorrowedBufferLayout, i: Int) -> Int:
    # The address of element i of a one-dimensional buffer: one multiply,
    # not a coordinate List per element (lane p3-neural-blocks); an indirect
    # (suboffset) buffer keeps the general walk.
    if layout.suboffsets[0] < 0:
        return layout.address + i * layout.strides[0]
    return layout.item_address([i])


def _flat(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = Python.list()
    if py_type(source) is b.list or py_type(source) is b.tuple:
        # One native list copy of the exact sequence (its items are the same
        # objects), not one bound append call per label.
        return py_call(b.list, source)
    if py_type(source) is not context["ndarray"]:
        raise Error("TypeError: labels require a native numeric buffer or a text sequence")
    var dtype = source.dtype
    var kind = String(dtype.kind)
    if kind != "U" and kind != "S" and kind != "O":
        raise Error("TypeError: labels require numeric, string or bytes values")
    admit_native_buffer_exporter(source)
    var borrowed = AdmittedBorrowedBuffer(source)
    var layout = borrowed.layout.value().copy()
    if len(layout.shape) != 1:
        raise Error("ValueError: text and object labels must be one-dimensional")
    ref cpy = Python().cpython()
    var admitter = SequentialValueAdmitter(context)
    # The CPython constructors from the one table (resolved once per process).
    var table = cpython_table()
    var bytes_from = table[].PyBytes_FromStringAndSize
    var unicode_from = table[].PyUnicode_FromKindAndData
    var native = Bool(py=dtype.isnative)
    var words = List[UInt32](length=layout.itemsize // 4, fill=0)
    for i in range(layout.count):  # driver-loop: native label schema decoding, never input hooks
        var address = _item(layout, i)
        var raw = PyObjectPtr()
        if kind == "O":
            raw = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=address)[0]
            if not raw:
                raise Error("TypeError: labels contain an uninitialized object slot")
            raw = cpy.Py_NewRef(raw)
        elif kind == "S":
            var ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
            var count = layout.itemsize
            while count > 0 and ptr[count - 1] == 0:  # driver-loop: remove fixed-width byte-label padding
                count -= 1
            raw = bytes_from(Int(ptr), count)
        else:
            var ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
            for j in range(len(words)):  # driver-loop: decode fixed-width unicode metadata
                var word = UInt32(0)
                for k in range(4):  # small-loop(one unicode code point)
                    word |= UInt32(ptr[4 * j + k]) << UInt32(8 * (k if native else 3 - k))
                words[j] = word
            var count = len(words)
            while count > 0 and words[count - 1] == 0:  # driver-loop: remove fixed-width unicode padding
                count -= 1
            raw = unicode_from(c_int(4), Int(words.unsafe_ptr()), count)
        if not raw:
            raise cpy.unsafe_get_error()
        var value = PythonObject(from_owned=raw)
        admitter.admit(value, context)
        _ = py_call(result.append, value)
    return result


def _object_store(address: Int, value: PythonObject) raises:
    ref cpy = Python().cpython()
    var slot = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=address)
    var previous = slot[0]
    slot[0] = cpy.Py_NewRef(value._obj_ptr)
    if previous:
        cpy.Py_DecRef(previous)


def neural_encode_labels(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    sequential_admit_value(source, context)
    # Numeric labels go through the one admission (admit_view: an Array, a
    # native buffer exporter, or a list / frame it materializes); text and
    # object labels are decoded below. No foreign protocol is admitted.
    var numeric = False
    var captured = AdmittedTypedPin()
    var view = TensorView()
    try:
        view = admit_view(source, "y", context, captured)
        _ = admitted_view_kind(view, "y")  # raises for a text or object buffer
        numeric = True
    except:
        captured = AdmittedTypedPin()  # release a partial capture now, as before
    if numeric:
        return _numeric(view, captured^, entries, context)
    var labels = _flat(source, context)
    var count = len(labels)
    if count == 0:
        raise Error("ValueError: y is empty")
    ref cpy = Python().cpython()
    var all_text = True
    var all_bytes = True
    # Every label is already admitted: a list/tuple source by the walk of
    # `source` above (it visits each element), an ndarray's decoded values by
    # `_flat`. The per-label re-admission repeated that walk.
    for i in range(count):  # driver-loop: native label kind admission
        var flags = cpython_table()[].PyType_GetFlags(py_type(labels[i])._obj_ptr)
        all_text = all_text and (flags & UInt64(1 << 28)) != 0
        all_bytes = all_bytes and (flags & UInt64(1 << 27)) != 0
    if not all_text and not all_bytes:
        var labels_pin = AdmittedTypedPin()
        var labels_view = admit_view(labels, "y", context, labels_pin)
        return _numeric(labels_view, labels_pin^, entries, context)
    # Text labels are read as they are; only bytes need the Latin-1 key.
    var normalized = labels if all_text else Python.list()
    for i in range(0 if all_text else count):  # driver-loop: lossless native bytes-to-codepoint key normalization
        if not all_text:
            var pointer = Int(0)
            var size = 0
            if cpy.lib.get_function[Int32]("PyBytes_AsStringAndSize")(labels[i]._obj_ptr, Pointer(to=pointer), Pointer(to=size)) != 0:
                raise cpy.unsafe_get_error()
            var raw = cpy.lib.get_function[PyObjectPtr]("PyUnicode_DecodeLatin1")(pointer, size, Int(0))
            if not raw:
                raise cpy.unsafe_get_error()
            _ = py_call(normalized.append, PythonObject(from_owned=raw))
    var table = StrTable()
    if not _read(normalized, table):
        raise Error("TypeError: labels require native text")
    var codes = sequential_array(Python.tuple(count), context["int32"], context)
    # perf-audit-3: the distinct classes (each its FIRST label, in code-point
    # order) and every label's int32 code from the parallel hash unique
    # (`str_unique_into`), which sorts only the k distinct classes. The former
    # stable merge sort of all `count` labels ran on one thread; the classes,
    # their order and the codes are the same.
    var unique = str_unique_into(table, count, native_buffer_address(codes))
    var width = 1
    var needs_objects = False
    for c in range(len(unique)):  # driver-loop: over the distinct classes (native metadata)
        var index = unique[c]
        width = max(width, table.lengths[index])
        # Fixed-width U/S rows discard trailing NULs on scalar publication.
        # Preserve those distinct admitted labels in owned object slots.
        if table.lengths[index] > 0 and table.points[table.starts[index] + table.lengths[index] - 1] == 0:
            needs_objects = True
    if needs_objects:
        var classes = _object_array(len(unique), context)
        var base = native_buffer_address(classes)
        for i in range(len(unique)):  # driver-loop: retain exact text values with their explicit lengths
            _object_store(base + i * size_of[PyObjectPtr](), labels[unique[i]])
        return Python.tuple(classes, codes)
    var dtype = py_call(native_numpy_dtype(context), ("S" if all_bytes else "U") + String(width))
    var classes = sequential_array(Python.tuple(len(unique)), dtype, context)
    var target = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(classes))
    for i in range(len(unique)):  # driver-loop: native class-label result serialization
        var index = unique[i]
        for j in range(table.lengths[index]):  # driver-loop: each native class code point
            var point = table.points[table.starts[index] + j]
            if all_bytes:
                target[i * width + j] = UInt8(point)
            else:
                for k in range(4):  # small-loop(one native unicode result code point)
                    target[4 * (i * width + j) + k] = UInt8(point >> UInt32(8 * k))
    _ = codes
    _ = classes
    return Python.tuple(classes, codes)


def neural_decode_labels(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(classes) is not context["ndarray"]:
        raise Error("TypeError: classes_ requires its native ndarray storage")
    var kind = String(classes.dtype.kind)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "S" and kind != "U" and kind != "O":
        raise Error("TypeError: classes_ requires numeric or text label storage")
    var borrowed = AdmittedBorrowedBuffer(classes)
    var layout = borrowed.layout.value().copy()
    if len(layout.shape) != 1:
        raise Error("ValueError: classes_ must be one-dimensional")
    if py_type(codes) is not context["ndarray"]:
        raise Error("TypeError: predicted class indices require native ndarray storage")
    var code_dtype = codes.dtype
    if String(code_dtype.kind) != "i" or not Bool(py=code_dtype.isnative):
        raise Error("TypeError: predicted class indices require native signed integers")
    var code_owner = AdmittedBorrowedBuffer(codes)
    var code_layout = code_owner.layout.value().copy()
    if len(code_layout.shape) != 1 or (code_layout.itemsize != 4 and code_layout.itemsize != 8):
        raise Error("TypeError: predicted class indices require an int32 or int64 vector")
    var count = code_layout.count
    var result = _object_array(count, context) if kind == "O" else sequential_array(Python.tuple(count), classes.dtype, context)
    var base = native_buffer_address(result)
    var admitter = SequentialValueAdmitter(context)
    # Object classes are admitted once per distinct class on first use, not
    # once per predicted row (lane p3-neural-blocks); the same classes are
    # refused, at the first row that predicts them.
    var checked = List[Bool](length=layout.count if kind == "O" else 0, fill=False)
    for i in range(count):  # driver-loop: publish one predicted label from its native class words
        var code_address = _item(code_layout, i)
        var index = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=code_address)[0]) if code_layout.itemsize == 4 else Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=code_address)[0])
        if index < 0 or index >= layout.count:
            raise Error("ValueError: predicted class index is outside classes_")
        if kind == "O":
            ref cpy = Python().cpython()
            var raw = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=_item(layout, index))[0]
            if not raw:
                raise Error("TypeError: classes_ contains an uninitialized object slot")
            var value = PythonObject(from_owned=cpy.Py_NewRef(raw))
            if not checked[index]:
                admitter.admit(value, context)
                var flags = cpython_table()[].PyType_GetFlags(py_type(value)._obj_ptr)
                if (flags & UInt64((1 << 28) | (1 << 27))) == 0:
                    raise Error("TypeError: object classes_ requires native string or bytes values")
                checked[index] = True
            _object_store(base + i * size_of[PyObjectPtr](), value)
            continue
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base + i * layout.itemsize),
               src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=_item(layout, index)), count=layout.itemsize)
    _ = classes
    _ = codes
    return result
