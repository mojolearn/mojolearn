# SPDX-License-Identifier: Apache-2.0
# Native reads of ordinary Arrow-backed pandas storage (`float64[pyarrow]`,
# `int64[pyarrow]`, `bool[pyarrow]`, ... columns and Arrow string indexes).
# Only exact pyarrow.lib (Cython) types are admitted, and only their compiled
# properties, buffer protocol and length slots are used: no
# ArrowExtensionArray.to_numpy/__array__, no pandas or pyarrow Python code.
# The values are copied into a fresh NumPy-layout block, so no Arrow buffer
# is retained past the read and Arrow ownership stays with the caller's
# pandas object.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_pandas import _pd_class, _pd_fields
from bindings.native_buffer_owner import native_buffer_view
from bindings.sequential_arrays import sequential_address
from bindings.arena_materialize_native import _address


def _arrow_extension_class() raises -> PythonObject:
    return _pd_class("pandas.core.arrays.arrow.array", "ArrowExtensionArray")


def pandas_arrow_exact(values: PythonObject) raises -> Bool:
    """True for an exact ArrowExtensionArray (pandas' `<type>[pyarrow]` storage)."""
    var cls = _arrow_extension_class()
    return cls is not None and py_type(values) is cls


def _arrow_string_exact(values: PythonObject) raises -> Bool:
    var cls = _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray")
    return cls is not None and py_type(values) is cls


def _chunked(values: PythonObject) raises -> PythonObject:
    var chunked = py_get(_pd_fields(values), "_pa_array")
    var cls = _pd_class("pyarrow.lib", "ChunkedArray")
    if chunked is None or cls is None or py_type(chunked) is not cls:
        raise Error("TypeError: Arrow-backed pandas storage requires an exact pyarrow ChunkedArray")
    return chunked


def pandas_arrow_length(values: PythonObject) raises -> Int:
    """The row count of exact Arrow-backed pandas storage (index or column),
    or -1 when `values` is not Arrow-backed."""
    if not pandas_arrow_exact(values) and not _arrow_string_exact(values):
        return -1
    return len(_chunked(values))


def _arrow_typestr(chunk: PythonObject) raises -> String:
    # Exact Cython array classes only; a subclass could carry Python hooks.
    var cls = py_type(chunk)
    var names = Python.list("DoubleArray", "FloatArray", "Int64Array", "Int32Array", "Int16Array",
                            "Int8Array", "UInt64Array", "UInt32Array", "UInt16Array", "UInt8Array",
                            "BooleanArray")
    var codes = List[String]()
    codes.append("<f8")
    codes.append("<f4")
    codes.append("<i8")
    codes.append("<i4")
    codes.append("<i2")
    codes.append("|i1")
    codes.append("<u8")
    codes.append("<u4")
    codes.append("<u2")
    codes.append("|u1")
    codes.append("|b1")
    for i in range(len(names)):  # driver-loop: over the fixed admitted Arrow numeric array classes
        var admitted = _pd_class("pyarrow.lib", String(names[i]))
        if admitted is not None and cls is admitted:
            return codes[i]
    raise Error("TypeError: model selection requires numeric Arrow storage (float, integer or boolean)")


def _itemsize(typestr: String) -> Int:
    if typestr == "<f8" or typestr == "<i8" or typestr == "<u8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<i2" or typestr == "<u2":
        return 2
    return 1


def pandas_arrow_ndarray(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """A one-dimensional NumPy copy of exact numeric Arrow-backed pandas
    storage, with the column's own dtype. Missing floats read as NaN (as
    pandas exports them); missing integers or booleans are refused, as the
    old object export's NA leaves were."""
    var chunked = _chunked(values)
    var b = native_loaded_module("builtins")
    var count = Int(py=chunked.num_chunks)
    var rows = len(chunked)
    var typestr = String()
    for c in range(count):  # driver-loop: over Arrow chunks, admitting one dtype before any copy
        var code = _arrow_typestr(py_call(chunked.chunk, c))
        if c > 0 and code != typestr:
            raise Error("TypeError: Arrow chunks of one column disagree on their type")
        typestr = code
    if count == 0:
        # An empty column: pandas' Arrow dtype still names the width, but no
        # buffer exists to admit; float64 is pandas' empty numeric export.
        typestr = "<f8"
    var output = py_call(context["ndarray"], Python.tuple(rows), dtype=PythonObject(typestr))
    var out = Int(py=sequential_address(output))
    var item = _itemsize(typestr)
    var floating = typestr.startswith("<f")
    var boolean = typestr == "|b1"
    var position = 0
    for c in range(count):  # driver-loop: over Arrow chunks, one native copy each
        var chunk = py_call(chunked.chunk, c)
        var length = len(chunk)
        var offset = Int(py=chunk.offset)
        var nulls = Int(py=chunk.null_count)
        if nulls > 0 and not floating:
            raise Error("TypeError: mojolearn: fold data holds missing values in Arrow integer or boolean storage, not numbers")
        var buffers = py_call(chunk.buffers)
        if py_type(buffers) is not b.list or len(buffers) != 2:
            raise Error("TypeError: numeric Arrow storage requires one validity and one data buffer")
        var buffer_class = _pd_class("pyarrow.lib", "Buffer")
        if length > 0 and (buffer_class is None or py_type(buffers[1]) is not buffer_class):
            raise Error("TypeError: numeric Arrow storage requires an exact pyarrow Buffer")
        if length > 0:
            var data_view = native_buffer_view(buffers[1])
            var data = _address(data_view)
            var dst = out + position * item
            if boolean:
                var bits = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=data)
                var target = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst)
                for r in range(length):  # driver-loop: storage unpack only, one Arrow bit per output byte
                    var bit = offset + r
                    target[r] = (bits[bit >> 3] >> UInt8(bit & 7)) & UInt8(1)
            else:
                memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
                       src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=data + offset * item),
                       count=length * item)
            if nulls > 0:
                if buffer_class is None or py_type(buffers[0]) is not buffer_class:
                    raise Error("TypeError: Arrow storage with missing values requires an exact validity Buffer")
                var valid_view = native_buffer_view(buffers[0])
                var valid = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=_address(valid_view))
                for r in range(length):  # driver-loop: storage only, missing float slots become NaN
                    var bit = offset + r
                    if ((valid[bit >> 3] >> UInt8(bit & 7)) & UInt8(1)) == UInt8(0):
                        # NumPy's canonical quiet NaN, the value pandas exports.
                        if item == 8:
                            MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=dst)[r] = UInt64(0x7FF8000000000000)
                        else:
                            MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)[r] = UInt32(0x7FC00000)
                _ = valid_view^
            _ = data_view^
        _ = buffers^
        _ = chunk^
        position += length
    _ = chunked^
    return output
