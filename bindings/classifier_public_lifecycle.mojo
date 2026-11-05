# SPDX-License-Identifier: Apache-2.0
# Native equivalents of the classifier mixin's completed-fit metadata. Keep
# class labels in ordinary native storage while the solver/score consumes them;
# expose the established ClassLabels object only after the operation completes.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.py_attr import py_attr
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get
from bindings.classical_labels import _flatten_labels
from bindings.native_buffer_owner import native_buffer_view
from core.classical_pandas_impl import pd_raw_at
from bindings.label_admission_native import _labels_class as native_labels_class, classifier_label_list as native_classifier_label_list


def _labels_class(context: PythonObject) raises -> PythonObject:
    return native_labels_class(context)


def classifier_label_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    return native_classifier_label_list(value, context)


def _public_labels(values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.list:
        return values
    var cls = _labels_class(context)
    var result = py_call(b.list.__new__, cls)
    _ = py_call(b.list.extend, result, values)
    return result


def _cpl_buffer_address(view: PythonObject) raises -> Int:
    # Same Py_buffer read as arena_materialize_native._address, kept local so
    # this closure does not pull the arena materializer in.
    ref cpy = Python().cpython()
    var storage = List[Int](length=10, fill=0)
    var pointer = storage.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(view._obj_ptr, pointer, Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = storage[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = storage^
    return address


def _arrow_string_names(storage: PythonObject) raises -> PythonObject:
    # Minimal UTF-8 reader copied from the text branch of
    # providers/python/resample_pandas_pyapi.mojo `_rp_arrow` (sharing it would
    # pull the whole resample provider into every classifier binding). Arrow
    # offsets and bytes are read natively; each name is decoded by CPython's
    # PyUnicode_DecodeUTF8. Nulls become None, which the caller treats as a
    # non-string name (no feature_names_in_), exactly as the other storages.
    var refusal = "TypeError: native classifier feature names require native pandas string-index storage"
    var chunked = py_get(_pd_fields(storage), "_pa_array")
    if py_type(chunked) is not _pd_class("pyarrow.lib", "ChunkedArray"):
        raise Error(refusal)
    var arrow_type = chunked.type
    if py_type(arrow_type) is not _pd_class("pyarrow.lib", "DataType"):
        raise Error(refusal)
    # Arrow's public Type enum: 13 = string, 34 = large_string.
    var type_id = Int(py=arrow_type.id)
    if type_id != 13 and type_id != 34:
        raise Error(refusal)
    var offset_size = 8 if type_id == 34 else 4
    var expected = _pd_class("pyarrow.lib", "LargeStringArray" if type_id == 34 else "StringArray")
    var buffer_type = _pd_class("pyarrow.lib", "Buffer")
    var list_type = py_type(Python.list())
    var names = Python.list()
    var count = Int(py=chunked.num_chunks)
    for c in range(count):  # driver-loop: over Arrow chunks of the column-name index
        var chunk = py_call(chunked.chunk, c)
        if py_type(chunk) is not expected:
            raise Error(refusal)
        var length = len(chunk)
        var offset = Int(py=chunk.offset)
        var buffers = py_call(chunk.buffers)
        if py_type(buffers) is not list_type or len(buffers) != 3:
            raise Error(refusal)
        var views = Python.list()
        var addresses = List[Int](length=3, fill=0)
        var sizes = List[Int](length=3, fill=0)
        for i in range(3):  # driver-loop: native buffer admission and owner retention
            if buffers[i] is not None:
                if py_type(buffers[i]) is not buffer_type:
                    raise Error(refusal)
                var view = native_buffer_view(buffers[i])
                addresses[i] = _cpl_buffer_address(view)
                sizes[i] = Int(py=view.nbytes)
                _ = py_call(views.append, view)
        if offset < 0 or length < 0:
            raise Error("ValueError: pandas Arrow chunk offsets exceed output bounds")
        if addresses[0] != 0 and (offset + length + 7) // 8 > sizes[0]:
            raise Error("ValueError: pandas Arrow validity buffer is too small")
        ref cpy = Python().cpython()
        for row in range(length):  # driver-loop: over column names, one stored Arrow string each
            var index = offset + row
            if addresses[0] != 0 and ((MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addresses[0])[index >> 3] >> UInt8(index & 7)) & UInt8(1)) == 0:
                _ = py_call(names.append, PythonObject(None))
                continue
            if addresses[1] == 0 or (index + 2) * offset_size > sizes[1]:
                raise Error("ValueError: pandas Arrow string offsets exceed their buffer")
            var first = Int(pd_raw_at(addresses[1], index * offset_size, offset_size, True))
            var last = Int(pd_raw_at(addresses[1], (index + 1) * offset_size, offset_size, True))
            if first < 0 or last < first or last > sizes[2]:
                raise Error("ValueError: pandas Arrow string span exceeds its native data buffer")
            var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addresses[2] + first)
            var raw = cpy.lib.get_function[PyObjectPtr]("PyUnicode_DecodeUTF8")(data, last - first, Optional[MutPointer[UInt8, MutUntrackedOrigin]](None))
            if not raw:
                raise cpy.unsafe_get_error()
            _ = py_call(names.append, PythonObject(from_owned=raw))
        _ = views^
        _ = buffers^
        _ = chunk^
    if len(names) != len(chunked):
        raise Error("ValueError: pandas Arrow chunks do not fill the declared row count")
    _ = chunked^
    return names


def _feature_names(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # Only an admitted exact DataFrame has column-name semantics. Reading its
    # manager/index storage avoids DataFrame.columns and Index iteration.
    if py_type(source) is not _pd_class("pandas.core.frame", "DataFrame"):
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    var manager = py_get(_pd_fields(source), "_mgr")
    if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
        raise Error("TypeError: native classifier requires ordinary pandas manager storage")
    var axes = _pd_get("pandas._libs.internals", "BlockManager", "axes", manager)
    if py_type(axes) is not b.list or len(axes) != 2:
        raise Error("TypeError: native pandas column axes are unavailable")
    var columns = axes[0]
    # Range/numeric/multi-index names never set feature_names_in_. Ordinary
    # string Index owns either a native ndarray or StringArray's native slot.
    if py_type(columns) is not _pd_class("pandas.core.indexes.base", "Index"):
        return PythonObject(None)
    var storage = py_get(_pd_fields(columns), "_data")
    var storage_class = py_type(storage)
    var names: PythonObject
    # pandas 3 with pyarrow stores string column names as ArrowStringArray
    # (pandas 2 also has its NumpySemantics subclass; _pd_class yields None
    # when a class is absent, so the identity test is simply false).
    if storage_class is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray") or storage_class is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArrayNumpySemantics") or storage_class is _pd_class("pandas.core.arrays.arrow.array", "ArrowExtensionArray"):
        names = _arrow_string_names(storage)
    else:
        if storage_class is _pd_class("pandas.core.arrays.string_", "StringArray"):
            storage = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", storage)
        if py_type(storage) is not context["ndarray"]:
            raise Error("TypeError: native classifier feature names require native pandas string-index storage")
        names = _flatten_labels(storage, context)
    if len(names) == 0:
        return PythonObject(None)
    for i in range(len(names)):  # driver-loop: feature-name type metadata, no conversion hooks
        if py_type(names[i]) is not b.str:
            return PythonObject(None)
    var result = py_call(context["ndarray"], Python.tuple(len(names)), dtype="O")
    for i in range(len(names)):  # driver-loop: copy admitted string references into native object storage
        py_setitem(result, i, names[i])
    return result


def classifier_after_fit(fields: PythonObject, source: PythonObject, context: PythonObject) raises:
    var names = _feature_names(source, context)
    if names is None:
        _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
    else:
        py_setitem(fields, "feature_names_in_", names)
    py_setitem(fields, "classes_", _public_labels(fields["classes_"], context))
    py_setitem(fields, "_mojolearn_fitted", True)
