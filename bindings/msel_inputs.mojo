# SPDX-License-Identifier: Apache-2.0
# Model-selection input preparation owns admission, byte copies and metadata.
# Never ask caller objects for shape, len, __array__, or DLPack protocols.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_input
from bindings.classical_pandas import (
    classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get,
    _pd_values, _pd_locations,
)
from bindings.sequential_inputs import sequential_sequence, _promote
from bindings.native_torch_admission import native_torch_exact, native_torch_numpy
from bindings.native_buffer_owner import native_buffer_view
from bindings.arena_array_metadata import arena_array_metadata
from bindings.arena_materialize_native import _prepare, _owned, _empty, _cast, _slot


def msel_admit(value: PythonObject, context: PythonObject) raises:
    classical_connect_numpy(context)
    classical_admit_value(value, context)


def _rows_context(context: PythonObject) raises -> PythonObject:
    var rows = context["family_msel_resident_context"]
    classical_connect_numpy(rows)
    return rows


def _pandas_blocks(value: PythonObject) raises -> PythonObject:
    var series = py_type(value) is _pd_class("pandas.core.series", "Series")
    var manager = py_get(_pd_fields(value), "_mgr")
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: model selection requires ordinary pandas Series storage")
        blocks = py_get(_pd_fields(manager), "blocks")
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: model selection requires ordinary pandas DataFrame storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not py_type(Python.tuple()) or len(blocks) == 0 or (series and len(blocks) != 1):
        raise Error("ValueError: pandas input has no supported block storage")
    for i in range(len(blocks)):  # driver-loop: exact storage classes before native descriptor reads
        var cls = py_type(blocks[i])
        if (cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock")
                and cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock")
                and cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock")
                and cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock")):
            raise Error("TypeError: model selection cannot invoke pandas block subclass hooks")
    return blocks


def _pandas_materialize(value: PythonObject, context: PythonObject, rows_context: PythonObject) raises -> PythonObject:
    # Preserve pandas' numeric promotion and column placement. The shared
    # estimator pandas_f32 reader narrows every column, which would change
    # float64 split outputs; this reader copies the original promoted width.
    var b = Python.import_module("builtins")
    var series = py_type(value) is _pd_class("pandas.core.series", "Series")
    var blocks = _pandas_blocks(value)
    var owned = Python.list()
    var positions = Python.list()
    var kind = String()
    var item = 0
    var nrows = -1
    var ncols = 0
    var objects = False
    var booleans = False
    var numbers = False
    for k in range(len(blocks)):  # driver-loop: native block ownership, promotion and placement metadata
        var pair = _pd_values(_pd_get("pandas._libs.internals", "Block", "values", blocks[k]), series, context)
        var data = pair[0]
        if py_type(data) is not context["ndarray"]:
            raise Error("TypeError: model selection requires native pandas numeric blocks")
        var rank = Int(py=data.ndim)
        if rank != 1 and rank != 2:
            raise Error("ValueError: pandas block rank must be one or two")
        var count = Int(py=data.shape[rank - 1])
        var width = 1 if rank == 1 else Int(py=data.shape[0])
        if (nrows >= 0 and count != nrows) or (series and rank != 1):
            raise Error("ValueError: pandas block dimensions disagree")
        nrows = count
        var placement = py_call(b.range, 1) if series else _pd_locations(blocks[k], context)
        if len(placement) != width:
            raise Error("ValueError: pandas block placement and width disagree")
        var prepared = _prepare(data, rows_context)
        var dtype = String(_slot(prepared, "dtype", rows_context))
        var original_kind = String(data.dtype.kind)
        if original_kind == "O":
            _promote(kind, item, "f" if dtype.startswith("<f") else "i", 8)
        else:
            # Promote the source dtype before Array's ABI widening. In
            # particular int16 + float32 remains float32, not float64.
            _promote(kind, item, original_kind, Int(py=data.dtype.itemsize))
        objects = objects or String(data.dtype.kind) == "O" or Bool(py=pair[1])
        booleans = booleans or String(data.dtype.kind) == "b"
        numbers = numbers or String(data.dtype.kind) != "b"
        ncols += width
        _ = py_call(owned.append, prepared)
        _ = py_call(positions.append, placement)
    if objects or (booleans and numbers):
        # The old materializer reads pandas' object export as int64 when
        # every scalar is integral, float64 otherwise.
        item = 8
        kind = "f" if kind == "f" else "i"
    if kind == "i" and item == 2:
        item = 4  # the established Array ABI widens signed 16-bit buffers to int32
    if kind == "b":
        kind = "u"
    if kind == "u" and item == 8:
        kind = "i"
    var dtype = "<" + kind + String(item)
    var output = _empty(Python.tuple(nrows) if series else Python.tuple(nrows, ncols), dtype, rows_context)
    var seen = List[Bool](length=ncols, fill=False)
    var out_address = Int(py=_slot(output, "_addr", rows_context))
    for k in range(len(owned)):  # driver-loop: native typed copies, one source owner kept through its final read
        var block = owned[k]
        if String(_slot(block, "dtype", rows_context)) != dtype:
            block = _cast(block, dtype, rows_context)
        var source = Int(py=_slot(block, "_addr", rows_context))
        for col in range(len(positions[k])):  # driver-loop: the block's native column placement
            var dest = Int(py=positions[k][col])
            if dest < 0 or dest >= ncols or seen[dest]:
                raise Error("ValueError: pandas columns overlap or exceed their matrix")
            seen[dest] = True
            for row in range(nrows):  # driver-loop: storage copy only, preserving every promoted element's bits
                var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + (col * nrows + row) * item)
                var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=out_address + (row * ncols + dest) * item)
                memcpy(dst, src, item)
                _ = block
                _ = output
        _ = block^
    return output


def msel_input(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    msel_admit(value, context)
    var rows = _rows_context(context)
    var cls = py_type(value)
    var b = rows["arena_native_types"]
    if cls is context["array_class"] or cls is py_get(context, "ndarray"):
        return _prepare(value, rows)
    if native_pandas_exact(value):
        return _pandas_materialize(value, context, rows)
    if native_torch_exact(cls):
        return _prepare(native_torch_numpy(value), rows)
    var sequence = value
    if cls is Python.import_module("builtins").range:
        sequence = py_call(b["list"], value)
        cls = py_type(sequence)
    if cls is b["list"] or cls is b["tuple"]:
        var packed = sequential_sequence(sequence, context)
        var kind = String(packed[1])
        if kind != "f" and kind != "i" and kind != "u" and kind != "b":
            raise Error("TypeError: " + name + " requires a rectangular sequence of native numbers")
        var dtype = String("<f8") if kind == "f" else String("<i8")
        var shape = py_call(b["tuple"], packed[0])
        if kind == "f" and Int(py=packed[2]) == 2:
            return _cast(_owned(packed[3], shape, "<f2", rows), dtype, rows)
        # array.array's C iteration preserves the old list materializer's
        # float64/int64 promotion, including native scalar leaves.
        var store = py_call(context["array_type"], "d" if kind == "f" else "q", packed[3])
        return _owned(store, shape, dtype, rows)
    if cls is b["str"] or cls is b["bytes"]:
        raise Error("TypeError: " + name + " must be an array, not a string")
    if py_get(context, "ndarray") is None and (cls is Python.import_module("builtins").memoryview or cls is context["array_type"]):
        var view = native_buffer_view(value)
        var format = String(view.format)
        if format == "h" or format == "Q":
            var packed = py_call(context["array_type"], format, py_call(view.tobytes, "C"))
            var wide = py_call(context["array_type"], "i" if format == "h" else "q", packed)
            return _owned(wide, view.shape, "<i4" if format == "h" else "<i8", rows)
    return _prepare(classical_input(value, context, name), rows)


def msel_input_count(value: PythonObject, context: PythonObject) raises -> Int:
    msel_admit(value, context)
    var cls = py_type(value)
    var b = Python.import_module("builtins")
    if cls is b.list or cls is b.tuple or cls is b.range or cls is context["array_type"]:
        return len(value)
    if cls is context["array_class"] or cls is py_get(context, "ndarray"):
        var shape = arena_array_metadata(value, _rows_context(context))["shape"]
        if len(shape) == 0:
            raise Error("TypeError: a scalar array has no samples")
        return Int(py=shape[0])
    if cls is b.memoryview or cls is b.bytearray:
        return len(value)
    if native_pandas_exact(value):
        var blocks = _pandas_blocks(value)
        var series = cls is _pd_class("pandas.core.series", "Series")
        var data = _pd_values(_pd_get("pandas._libs.internals", "Block", "values", blocks[0]), series, context)[0]
        if py_type(data) is not context["ndarray"] or Int(py=data.ndim) < 1:
            raise Error("TypeError: pandas samples require ordinary native storage")
        return Int(py=data.shape[Int(py=data.ndim) - 1])
    if native_torch_exact(cls):
        return msel_input_count(native_torch_numpy(value), context)
    raise Error("TypeError: model selection requires an ordinary sample container; Python protocols cannot execute")
