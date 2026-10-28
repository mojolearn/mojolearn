# SPDX-License-Identifier: Apache-2.0
# One public preprocessing input snapshot, before dtype conversion or execution.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_buffer_view
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_arrays import classical_input, classical_is_array, classical_typed, classical_address
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get, _pd_locations
from bindings.classical_validate import RESHAPE_HINT
from bindings.host_sparse_native import host_sparse_format, _sparse_fields
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_admission import _ordinary_torch, _one_of, sequential_admit_value
from bindings.sequential_inputs import _promote
from bindings.x_prep_native_inputs import _rows_fields
from providers.python.array_metadata_pyapi import ArrayMetadata, array_slot_once
from providers.python.prep_metadata_pyapi import _array_snapshot, _pandas_empty_shape, _description, prep_numeric_vector
from providers.python.prep_category_pyapi import prep_ndarray_backed, prep_category_array
from providers.python.preprocessing_input_pyapi import _frozen_values
from core.cast_in import CastIn, cast_in_kind, CAST_KIND_FLOAT
from core.scaler_fit_impl import scaler_cast
from core.prep_boundary_impl import prep_float_finite, prep_gather_rows, prep_mask_any, prep_mask_export
from core.classical_pandas_impl import pd_scatter_numeric

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def prep_shape(shape: List[Int], numeric: Bool, name: String) raises:
    var b = native_loaded_module("builtins")
    var items = Python.list()
    for dimension in shape:  # small-loop(rank: captured shape words)
        _ = py_call(items.append, dimension)
    var text = String(py_call(b.repr, py_call(b.tuple, items)))
    var rank = len(shape)
    if numeric and rank < 2:
        raise Error("Expected 2D array, got " + ("scalar" if rank == 0 else "1D") + " array instead (shape=" + text + "). " + RESHAPE_HINT)
    if numeric and rank == 2 and shape[0] == 0:
        raise Error("Found array with 0 sample(s) (shape=" + text + ") while a minimum of 1 is required by " + name + ".")
    if numeric and rank == 2 and shape[1] == 0:
        raise Error("Found array with 0 feature(s) (shape=" + text + ") while a minimum of 1 is required by " + name + ".")


def prep_finite(snapshot: ArrayMetadata, name: String) raises:
    if snapshot.dtype_number == 13:
        # describe recognizes PEP3118 long-double 'g' as floating, but the
        # former _all_finite materializer refused that format before casting.
        raise Error("TypeError: mojolearn: unsupported buffer format '" + ("g" if snapshot.native_byte_order else "=g") + "'")
    if snapshot.dtype_number != 11 and snapshot.dtype_number != 12:
        return
    if snapshot.source.itemsize != (4 if snapshot.dtype_number == 11 else 8):
        raise Error("TypeError: preprocessing floating dtype has inconsistent itemsize")
    if not prep_float_finite(snapshot.source, snapshot.shape, snapshot.strides,
                             snapshot.count, snapshot.native_byte_order, snapshot.dtype_number):
        raise Error("Input X contains NaN or infinity (" + name + ").")


def _pandas_block(value: PythonObject, single: Bool, context: PythonObject) raises -> Tuple[ArrayMetadata, Bool]:
    if py_type(value) is context["ndarray"]:
        return (_array_snapshot(value, context), False)
    var cls = py_type(value)
    if cls is _pd_class("pandas.core.arrays.categorical", "Categorical"):
        return (prep_category_array(value, context), not single)
    var backed = prep_ndarray_backed(value)
    if backed is not value:
        var opaque = cls is _pd_class("pandas.core.arrays.period", "PeriodArray")
        if cls is _pd_class("pandas.core.arrays.datetimes", "DatetimeArray"):
            var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", value)
            opaque = py_type(dtype) is _pd_class("pandas.core.dtypes.dtypes", "DatetimeTZDtype")
        if py_type(backed) is not context["ndarray"]:
            raise Error("TypeError: pandas extension requires native ndarray storage")
        var data = _array_snapshot(backed, context)
        data.owner = Python.tuple(value, data.owner)
        return (data^, opaque)
    if cls is not _pd_class("pandas.core.arrays.integer", "IntegerArray") and cls is not _pd_class("pandas.core.arrays.floating", "FloatingArray") and cls is not _pd_class("pandas.core.arrays.boolean", "BooleanArray"):
        raise Error("TypeError: pandas input requires ordinary numeric storage without extension callbacks")
    var fields = _pd_fields(value)
    var data_value = py_get(fields, "_data")
    var mask_value = py_get(fields, "_mask")
    if py_type(data_value) is not context["ndarray"] or py_type(mask_value) is not context["ndarray"]:
        raise Error("TypeError: pandas nullable storage requires exact native arrays")
    var data = _array_snapshot(data_value, context)
    var mask = _array_snapshot(mask_value, context)
    if len(data.shape) != 1 or len(mask.shape) != 1 or data.count != mask.count or mask.dtype_number != 0 or mask.source.itemsize != 1 or not mask.castable:
        raise Error("TypeError: pandas nullable storage metadata differs")
    var missing = prep_mask_any(mask.source, mask.count)
    data.owner = Python.tuple(data.owner, mask.keep, mask.owner, value)
    # A single extension block (also Series) exports its inferred native
    # numeric dtype. Mixed-block extension promotion remains object dtype.
    # pandas/core/arrays/_utils.py::to_numpy_dtype_inference is the source
    # contract: missing integers become float64, missing bool stays object.
    if not single or (missing and data.dtype_number == 0):
        return (data^, True)
    if not missing:
        return (data^, False)
    if not data.castable:
        raise Error("TypeError: pandas nullable storage requires a native numeric dtype")
    var number = 12 if data.dtype_number >= 1 and data.dtype_number <= 10 else data.dtype_number
    var width = 8 if number == 12 else data.source.itemsize
    var output = prep_numeric_vector(data.count, number, width, Python.tuple(data.keep, data.owner), context)
    prep_mask_export(data.source, mask.source, data.count, data.native_byte_order, output.source, number)
    _ = mask^
    _ = data^
    return (output^, False)


def _pandas_admit_block(block: PythonObject) raises:
    var cls = py_type(block)
    var admitted = False
    for name in ["NumpyBlock", "NumericBlock", "ObjectBlock", "ExtensionBlock", "DatetimeLikeBlock", "DatetimeTZBlock", "CategoricalBlock"]:  # small-loop(pandas native block layouts)
        admitted = admitted or cls is _pd_class("pandas.core.internals.blocks", name)
    if not admitted:
        raise Error("TypeError: pandas requires ordinary native blocks without subclass hooks")


def _pandas_snapshot(source: PythonObject, name: String, validate: Bool, scan: Bool,
                     context: PythonObject) raises -> ArrayMetadata:
    var b = native_loaded_module("builtins")
    var series = py_type(source) is _pd_class("pandas.core.series", "Series")
    var manager = py_get(_pd_fields(source), "_mgr")
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: pandas Series requires native single-block storage")
        blocks = py_get(_pd_fields(manager), "blocks")
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: pandas DataFrame requires native block storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not b.tuple or (series and len(blocks) != 1):
        raise Error("pandas input is empty or has invalid block metadata")
    if len(blocks) == 0:
        var empty_shape = _pandas_empty_shape(manager, context)
        if validate:
            prep_shape(empty_shape, True, name)
        return _description(source, empty_shape^, 12, Python.tuple(source, manager, blocks))
    var snapshots = List[ArrayMetadata]()
    var placements = List[PythonObject]()
    var original_owners = Python.list()
    var rows = -1
    var cols = 0
    var common_kind = String()
    var common_size = 0
    var objects = False
    var booleans = False
    var numbers = False
    var complex = False
    var long_width = 0
    for k in range(len(blocks)):  # driver-loop: admitted pandas block owners and placement metadata
        var block = blocks[k]
        _pandas_admit_block(block)
        var pair = _pandas_block(_pd_get("pandas._libs.internals", "Block", "values", block), len(blocks) == 1, context)
        var data = pair[0].copy()
        var rank = len(data.shape)
        if rank != 1 and rank != 2:
            raise Error("pandas numeric block rank must be one or two")
        var n = data.shape[rank - 1]
        var d = 1 if rank == 1 else data.shape[0]
        if rows >= 0 and rows != n:
            raise Error("pandas numeric block dimensions differ or are empty")
        rows = n
        cols += d
        var positions = py_call(b.range, 1) if series else _pd_locations(block, context)
        if len(positions) != d:
            raise Error("pandas block column positions differ from numeric storage")
        placements.append(positions)
        var complex_block = data.dtype_number == 14 or data.dtype_number == 15 or data.dtype_number == 16
        complex = complex or complex_block
        objects = objects or pair[1] or (not data.castable and not complex_block and data.dtype_number != 13)
        booleans = booleans or data.dtype_number == 0
        numbers = numbers or data.dtype_number != 0
        if data.dtype_number == 13:
            long_width = data.source.itemsize
        if data.castable:
            var kind = String("b") if data.dtype_number == 0 else (String("f") if data.source.kind == CAST_KIND_FLOAT else (String("i") if data.source.kind == 1 else String("u")))
            _promote(common_kind, common_size, kind, data.source.itemsize)
        _ = py_call(original_owners.append, Python.tuple(data.keep, data.owner))
        snapshots.append(data^)
    var shape = List[Int]()
    shape.append(rows)
    if not series:
        shape.append(cols)
    objects = objects or (booleans and numbers)
    if complex and not objects:
        raise Error("Complex data not supported: " + name + " got a complex X.")
    if validate:
        prep_shape(shape, not objects, name)
        if scan and not objects and long_width > 0:
            raise Error("TypeError: mojolearn: unsupported buffer format 'g'")
        if scan and not objects and common_kind == "f" and (common_size == 4 or common_size == 8):
            for k in range(len(snapshots)):  # driver-loop: validate original block words before narrowing
                prep_finite(snapshots[k], name)
    if objects or long_width > 0 or len(shape) != 2 or rows < 1 or cols < 1 or rows > 2147483647 // cols:
        var description = _description(source, shape^, 17 if objects else (13 if long_width > 0 else 12),
                                       Python.tuple(source, manager, blocks, original_owners))
        if not objects and long_width > 0:
            description.source.itemsize = long_width
        return description^
    var output = classical_typed(Python.tuple(rows) if series else Python.tuple(rows, cols), "<f4", context["output_context"])
    var address = Int(py=classical_address(output))
    var dest = F32P(unsafe_from_address=address)
    var seen = List[Bool](length=cols, fill=False)
    for k in range(len(snapshots)):  # driver-loop: native numeric block scatter, metadata already captured
        ref data = snapshots[k]
        if not data.castable:
            raise Error("TypeError: pandas block cannot be cast")
        var rank = len(data.shape)
        var positions = placements[k]
        for col in range(len(positions)):  # driver-loop: native block column placements
            var column = Int(py=positions[col])
            if column < 0 or column >= cols or seen[column]:
                raise Error("pandas block column positions overlap or exceed the matrix")
            seen[column] = True
            pd_scatter_numeric(data.source.addr, data.source.itemsize, data.native_byte_order, data.source.kind,
                data.strides[rank - 1], 0 if rank == 1 else data.strides[0], col, column,
                rows, cols, common_kind == "f" and common_size == 8, dest, True)
    var owner = Python.tuple(array_slot_once(output, "_store", context), PythonObject(None),
                            Python.tuple(source, manager, blocks, original_owners), array_slot_once(output, "_mv", context))
    return ArrayMetadata(output, shape^, CastIn(address, CAST_KIND_FLOAT, 4, rows, 1 if series else cols,
        4 if series else cols * 4, 0 if series else 4), rows * cols, 11, True, True, True,
        owner, True)


def _rows_snapshot(source: PythonObject, context: PythonObject) raises -> ArrayMetadata:
    var b = native_loaded_module("builtins")
    var fields = _rows_fields(source, context)
    var dtype = fields["dtype"]
    if py_type(dtype) is not b.str:
        raise Error("TypeError: native row selection requires native dtype metadata")
    var host_rows = String(context["vendor"]) == "cpu" or String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) == "0" or String(dtype) != "<f4"
    var prior = py_get(fields, "_host")
    if host_rows and prior is not None:
        classical_admit_value(prior, context)
        return _array_snapshot(prior, context)
    var base = fields["base"]
    var indices = fields["indices"]
    classical_admit_value(base, context)
    classical_admit_value(indices, context)
    if not classical_is_array(base, context) or not classical_is_array(indices, context):
        raise Error("TypeError: native row selection requires ordinary native Array storage")
    var data = _array_snapshot(base, context)
    var index = _array_snapshot(indices, context)
    if len(index.shape) != 1 or index.source.kind != 1 or index.source.itemsize != 8 or not index.c_order or not index.castable:
        raise Error("TypeError: native row selection requires a contiguous int64 index buffer")
    if len(data.shape) != 2 or not data.castable:
        raise Error("TypeError: native row selection requires native numeric matrix metadata")
    var nr = data.source.rows
    var d = data.source.cols
    var n = index.count
    if nr < 1 or d < 1 or n < 1 or nr > 2147483647 // d or n > 2147483647 // d:
        raise Error("MaxAbsScaler requires a nonempty row selection within native indexing bounds")
    var code = String("<f") if data.source.kind == CAST_KIND_FLOAT else (String("<i") if data.source.kind == 1 else String("<u"))
    code += String(data.source.itemsize)
    var output = classical_typed(Python.tuple(n, d), code, context["output_context"])
    var address = Int(py=classical_address(output))
    prep_gather_rows(data.source, index.source.addr, n, address)
    if host_rows:
        py_setitem(fields, "_host", output)
    var shape = List[Int]()
    shape.append(n)
    shape.append(d)
    var owner = Python.tuple(array_slot_once(output, "_store", context), PythonObject(None),
                            Python.tuple(source, data.owner, index.owner), array_slot_once(output, "_mv", context))
    var result = ArrayMetadata(output, shape^,
        CastIn(address, data.source.kind, data.source.itemsize, n, d, d * data.source.itemsize, data.source.itemsize),
        n * d, data.dtype_number, True, True, True,
        owner, True)
    _ = data^
    _ = index^
    return result^


def prep_transform_snapshot(source: PythonObject, name: String, validate: Bool,
                            scan: Bool, context: PythonObject,
                            allow_sparse: Bool = False,
                            check_sparse: Bool = True) raises -> ArrayMetadata:
    var b = native_loaded_module("builtins")
    if py_type(source) is context["maxabs_rows_class"]:
        # DeviceRows is a frame-like native handle in _sk_validate.describe;
        # it has width semantics but no outer float-buffer finite scan.
        var snapshot = _rows_snapshot(source, context)
        if validate:
            prep_shape(snapshot.shape, True, name)
        return snapshot^
    # Match loaded native sparse class identities before general admission.
    # Reading an unadmitted class's __module__ could invoke metaclass or
    # descriptor hooks even on this refusal path.
    if host_sparse_format(source).byte_length() > 0:
        if check_sparse and not allow_sparse:
            raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + name + " takes dense X only.)")
        var raw = py_get(_sparse_fields(source), "_shape")
        if py_type(raw) is not b.tuple or len(raw) != 2 or py_type(raw[0]) is not b.int or py_type(raw[1]) is not b.int:
            raise Error("TypeError: sparse input requires native shape metadata")
        var shape = List[Int]()
        shape.append(Int(py=raw[0]))
        shape.append(Int(py=raw[1]))
        return _description(source, shape^, -101, Python.tuple(source, raw))
    classical_admit_value(source, context)
    if native_pandas_exact(source):
        return _pandas_snapshot(source, name, validate, scan, context)
    var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
    var sequence = py_type(prepared) is b.list or py_type(prepared) is b.tuple
    var cls = py_type(prepared)
    var described = cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.str and cls is not b.bytes and cls is not b.dict and cls is not b.range and prepared is not None
    if _one_of(cls, context["scalar_types"]) and String(prepared.dtype.kind) == "c":
        raise Error("Complex data not supported: " + name + " got a complex X.")
    var scan_sequence = True
    var dimensions = List[Int]()
    var numeric_sequence = False
    if sequence:
        var first = prepared
        while py_type(first) is b.list or py_type(first) is b.tuple:  # driver-loop: native sequence rank, matching describe's first branch
            dimensions.append(len(first))
            if len(first) == 0:
                break
            first = first[0]
        var cls = py_type(first)
        if cls is b.complex:
            raise Error("Complex data not supported: " + name + " got a complex X.")
        scan_sequence = cls is b.float
        numeric_sequence = cls is b.int or cls is b.bool or cls is b.float or cls is b.list or cls is b.tuple
        if _one_of(cls, context["scalar_types"]):
            var kind = String(first.dtype.kind)
            if kind == "c":
                raise Error("Complex data not supported: " + name + " got a complex X.")
            scan_sequence = kind == "f"
            numeric_sequence = kind == "f" or kind == "i" or kind == "u" or kind == "b"
        if validate:
            prep_shape(dimensions, numeric_sequence, name)
    var value = PythonObject(None)
    try:
        value = prepared if classical_is_array(prepared, context) else classical_input(prepared, context)
    except error:
        # _check_X only materialized floating lists for its finite scan.
        # Other sequence/scalar conversion failures belong to the inner fit,
        # after y validation and inside the old small-fit error context.
        if (sequence and not (validate and scan and scan_sequence)) or not described:
            var refused = _description(prepared, dimensions^, -103 if sequence and numeric_sequence else -102,
                                       Python.tuple(source, prepared))
            refused.dtype_name = String(error)
            return refused^
        raise error^
    var snapshot = _array_snapshot(value, context)
    if snapshot.dtype_number == 14 or snapshot.dtype_number == 15 or snapshot.dtype_number == 16:
        raise Error("Complex data not supported: " + name + " got a complex X.")
    if validate and described:
        prep_shape(snapshot.shape, (snapshot.dtype_number >= 0 and snapshot.dtype_number <= 13) or snapshot.dtype_number == 23, name)
        if scan and (not sequence or scan_sequence):
            prep_finite(snapshot, name)
    return snapshot^


def prep_transform_values(mut snapshot: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    if snapshot.dtype_number == -102 or snapshot.dtype_number == -103:
        raise Error(snapshot.dtype_name)
    if snapshot.dtype_number == -101:
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (MaxAbsScaler takes dense X only.)")
    if len(snapshot.shape) != 2:
        raise Error("MaxAbsScaler requires a nonempty two-dimensional input")
    var n = snapshot.source.rows
    var d = snapshot.source.cols
    if n < 1 or d < 1:
        raise Error("Scaler requires a nonempty two-dimensional input")
    if n > 2147483647 // d:
        raise Error("Scaler exceeds the native Int32 indexing bound")
    if not snapshot.castable:
        _ = cast_in_kind(snapshot.dtype_number, snapshot.source.itemsize, "input")
        raise Error("TypeError: preprocessing input cannot be cast")
    if snapshot.f32_c() and snapshot.native_byte_order:
        var view = snapshot.owner[3] if snapshot.is_array else native_buffer_view(snapshot.keep)
        if py_type(view) is not native_loaded_module("builtins").memoryview:
            raise Error("TypeError: preprocessing input storage requires an exact native memoryview")
        sequential_admit_value(view, context)
        view = py_call(py_call(view.cast, "B").cast, "f")
        return _frozen_values(snapshot, view, context)
    # Allocate a private packed view, so the transport cannot reread any
    # original mutable Array slot after the operation snapshot was captured.
    var output = classical_typed(Python.tuple(n, d), "<f4", context["output_context"])
    var address = Int(py=classical_address(output))
    scaler_cast(snapshot.source.addr, snapshot.source.s0, snapshot.source.s1, n, d,
                snapshot.source.itemsize, snapshot.source.kind, snapshot.native_byte_order,
                F32P(unsafe_from_address=address))
    snapshot.replace_f32(output, address)
    return output
