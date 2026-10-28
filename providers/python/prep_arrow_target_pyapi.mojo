# SPDX-License-Identifier: Apache-2.0
# Native Arrow target description only: Normalizer does not consume labels.
# Preserve original chunk/buffer owners; do not invoke pandas __array__.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.classical_pandas import _pd_class
from bindings.pandas_arrow_native import _chunked, _arrow_typestr, _itemsize
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.prep_metadata_pyapi import _description
from core.prep_boundary_impl import prep_arrow_finite


def prep_arrow_target(raw: PythonObject, series: Bool, context: PythonObject) raises -> Tuple[ArrayMetadata, Int, Int, Bool]:
    var b = native_loaded_module("builtins")
    var chunked = _chunked(raw)
    var rows = len(chunked)
    var count = Int(py=chunked.num_chunks)
    var number = 12
    var width = 8
    var finite = True
    var objects = not series or py_type(raw) is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray")
    var owners = Python.list()
    var common = String()
    for c in range(count):  # driver-loop: one admitted native Arrow chunk description
        var chunk = py_call(chunked.chunk, c)
        var cls = py_type(chunk)
        # Primitive string/date/time/object Arrow labels are passive here.
        # Only numeric chunks can trigger the old floating-label check.
        var numeric = False
        for word in ["DoubleArray", "FloatArray", "Int64Array", "Int32Array", "Int16Array", "Int8Array", "UInt64Array", "UInt32Array", "UInt16Array", "UInt8Array", "BooleanArray"]:  # small-loop(Arrow primitive native classes)
            numeric = numeric or cls is _pd_class("pyarrow.lib", word)
        if not numeric:
            objects = True
            _ = py_call(owners.append, chunk)
            continue
        var code = _arrow_typestr(chunk)
        if common.byte_length() > 0 and code != common:
            raise Error("TypeError: Arrow target chunks disagree on their native dtype")
        common = code
        var length = len(chunk)
        var offset = Int(py=chunk.offset)
        var nulls = Int(py=chunk.null_count)
        width = _itemsize(code)
        number = 0
        if code.startswith("<f"):
            number = 11 if width == 4 else 12
        elif code.startswith("<i"):
            number = 1 if width == 1 else (3 if width == 2 else (5 if width == 4 else 9))
        elif code.startswith("<u"):
            number = 2 if width == 1 else (4 if width == 2 else (6 if width == 4 else 10))
        if nulls > 0:
            if number == 11 or number == 12:
                finite = False
            else:
                objects = True
        if number == 11 or number == 12:
            var buffers = py_call(chunk.buffers)
            if py_type(buffers) is not b.list or len(buffers) != 2:
                raise Error("TypeError: Arrow floating target requires validity and data buffers")
            var buffer = buffers[1]
            if length > 0:
                if py_type(buffer) is not _pd_class("pyarrow.lib", "Buffer"):
                    raise Error("TypeError: Arrow target data requires native buffer storage")
                var address = Int(py=buffer.address)
                var size = Int(py=buffer.size)
                if offset < 0 or length < 0 or offset > size // width or length > size // width - offset:
                    raise Error("TypeError: Arrow target data buffer is shorter than its metadata")
                if nulls == 0:
                    finite = finite and prep_arrow_finite(address, length, offset, width)
            _ = py_call(owners.append, buffers)
        _ = py_call(owners.append, chunk)
    if objects:
        number = 17
        width = 8
    var shape = List[Int]()
    shape.append(rows)
    var result = _description(raw, shape^, number, Python.tuple(raw, chunked, owners))
    return (result^, number, width, finite)
