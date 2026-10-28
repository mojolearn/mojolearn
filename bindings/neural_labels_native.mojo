# SPDX-License-Identifier: Apache-2.0
# Shared native label preparation/publication for the neural classifiers. Numeric
# sorting stays in the connected neural provider; text uses native code points.
from std.memory import memcpy
from std.sys import size_of
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.sequential_arrays import sequential_array
from bindings.xseq_native_common import xseq_neural_provider
from bindings.str_labels_native import StrTable, _read, _sorted
from bindings.native_buffer_address import native_buffer_address
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute
from sequence.neural_model_dispatch import MODEL_LABELS
from core.cast_in import cast_in_kind


def _numeric(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    admit_native_buffer_exporter(source)
    var borrowed = AdmittedBorrowedBuffer(source)
    var layout = borrowed.layout.value().copy()
    var rank = len(layout.shape)
    if rank > 2 or (rank == 2 and layout.shape[1] != 1):
        raise Error("ValueError: y must be a label vector")
    var n = layout.count
    if n < 1:
        raise Error("ValueError: y is empty")
    var dtype = source.dtype
    if not Bool(py=dtype.isnative):
        raise Error("TypeError: y must use native byte order")
    var classes = sequential_array(Python.tuple(n), dtype, context)
    var codes = sequential_array(Python.tuple(n), context["int32"], context)
    var kind = cast_in_kind(Int(py=dtype.num), layout.itemsize, "y")
    var stride = layout.strides[0] if rank > 0 else layout.itemsize
    var addresses: List[Int] = [native_buffer_address(classes), native_buffer_address(codes)]
    var words: List[Int] = [layout.address, kind, layout.itemsize, n, 1, stride, layout.itemsize]
    var count = neural_descriptor_execute(xseq_neural_provider(entries), MODEL_LABELS, addresses, words, List[Float64]())
    var selected = py_call(context["ndarray"], Python.tuple(count), dtype=dtype, buffer=classes)
    _ = source
    return Python.tuple(selected, codes)


def _flat(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = Python.list()
    if py_type(source) is b.list or py_type(source) is b.tuple:
        for i in range(len(source)):  # driver-loop: exact label sequence owners
            _ = py_call(result.append, source[i])
        return result
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
    for i in range(layout.count):  # driver-loop: native label schema decoding, never input hooks
        var address = layout.item_address([i])
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
            raw = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(ptr, count)
        else:
            var ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
            var words = List[UInt32](length=layout.itemsize // 4, fill=0)
            var native = Bool(py=dtype.isnative)
            for j in range(len(words)):  # driver-loop: decode fixed-width unicode metadata
                var word = UInt32(0)
                for k in range(4):  # small-loop(one unicode code point)
                    word |= UInt32(ptr[4 * j + k]) << UInt32(8 * (k if native else 3 - k))
                words[j] = word
            var count = len(words)
            while count > 0 and words[count - 1] == 0:  # driver-loop: remove fixed-width unicode padding
                count -= 1
            raw = cpy.lib.get_function[PyObjectPtr]("PyUnicode_FromKindAndData")(Int32(4), words.unsafe_ptr(), count)
        if not raw:
            raise cpy.unsafe_get_error()
        var value = PythonObject(from_owned=raw)
        sequential_admit_value(value, context)
        _ = py_call(result.append, value)
    return result


def _object_array(count: Int, context: PythonObject) raises -> PythonObject:
    # Object arrays must own their allocation so NumPy owns/decrefs every slot.
    # A byte-buffer-backed ndarray would not own the referenced label objects.
    var dtype = py_call(native_numpy_dtype(context), "O")
    return py_call(context["ndarray"], Python.tuple(count), dtype=dtype)


def _object_store(address: Int, value: PythonObject) raises:
    ref cpy = Python().cpython()
    var slot = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=address)
    var previous = slot[0]
    slot[0] = cpy.Py_NewRef(value._obj_ptr)
    if previous:
        cpy.Py_DecRef(previous)


def neural_encode_labels(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    sequential_admit_value(source, context)
    var array = PythonObject(None)
    try:
        array = sequential_input(source, context)
    except:
        pass  # native text sequences are decoded below; no foreign protocols admitted
    if array is not None:
        var kind = String(array.dtype.kind)
        if kind == "b" or kind == "i" or kind == "u" or kind == "f":
            return _numeric(array, entries, context)
    var labels = _flat(source, context)
    var count = len(labels)
    if count == 0:
        raise Error("ValueError: y is empty")
    ref cpy = Python().cpython()
    var all_text = True
    var all_bytes = True
    for i in range(count):  # driver-loop: native label kind admission
        sequential_admit_value(labels[i], context)
        var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(py_type(labels[i])._obj_ptr)
        all_text = all_text and (flags & UInt64(1 << 28)) != 0
        all_bytes = all_bytes and (flags & UInt64(1 << 27)) != 0
    if not all_text and not all_bytes:
        return _numeric(sequential_input(labels, context), entries, context)
    var normalized = Python.list()
    for i in range(count):  # driver-loop: lossless native bytes-to-codepoint key normalization
        if all_text:
            _ = py_call(normalized.append, labels[i])
        else:
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
    var order = _sorted(table, count)
    var codes = sequential_array(Python.tuple(count), context["int32"], context)
    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(codes))
    var unique = List[Int]()
    var width = 1
    var needs_objects = False
    for i in range(count):  # driver-loop: distinct text-label metadata and inverse codes
        var index = order[i]
        if i == 0 or table.compare(order[i - 1], index) != 0:
            unique.append(index)
            width = max(width, table.lengths[index])
            # Fixed-width U/S rows discard trailing NULs on scalar publication.
            # Preserve those distinct admitted labels in owned object slots.
            if table.lengths[index] > 0 and table.points[table.starts[index] + table.lengths[index] - 1] == 0:
                needs_objects = True
        cp[index] = Int32(len(unique) - 1)
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
    for i in range(count):  # driver-loop: publish one predicted label from its native class words
        var code_address = code_layout.item_address([i])
        var index = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=code_address)[0]) if code_layout.itemsize == 4 else Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=code_address)[0])
        if index < 0 or index >= layout.count:
            raise Error("ValueError: predicted class index is outside classes_")
        if kind == "O":
            ref cpy = Python().cpython()
            var raw = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=layout.item_address([index]))[0]
            if not raw:
                raise Error("TypeError: classes_ contains an uninitialized object slot")
            var value = PythonObject(from_owned=cpy.Py_NewRef(raw))
            sequential_admit_value(value, context)
            var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(py_type(value)._obj_ptr)
            if (flags & UInt64((1 << 28) | (1 << 27))) == 0:
                raise Error("TypeError: object classes_ requires native string or bytes values")
            _object_store(base + i * size_of[PyObjectPtr](), value)
            continue
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base + i * layout.itemsize),
               src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=layout.item_address([index])), count=layout.itemsize)
    _ = classes
    _ = codes
    return result
