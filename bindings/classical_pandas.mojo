# SPDX-License-Identifier: Apache-2.0
# Native ingestion of ordinary pandas storage, without DataFrame.__array__,
# manager properties, extension conversion methods or any Python callbacks.
# A caller imports pandas; runtime admission only reads its loaded modules.
from std.memory import bitcast
from std.python import Python, PythonObject
from core.cast_in import cast_in_kind, cast_w64_bits, f64_to_f32_bits
from core.classical_pandas_impl import pd_raw_at, _pd_double, pd_mask_check, pd_scatter_numeric
from bindings.sequential_inputs import _promote
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_admission import _one_of, sequential_admit_value
from bindings.py_balanced import py_call, py_type, py_get, py_fields


def _pd_class(module_name: String, name: String) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var sys = Python.import_module("sys")
    var module = py_get(sys.modules, module_name)
    if module is None or py_type(module) is not py_type(sys):
        return PythonObject(None)
    var cls = py_get(py_fields(module), name)
    return cls if py_type(cls) is b.type else PythonObject(None)


def native_pandas_exact(value: PythonObject) raises -> Bool:
    var cls = py_type(value)
    return cls is _pd_class("pandas.core.frame", "DataFrame") or cls is _pd_class("pandas.core.series", "Series")


def classical_admit_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: classical input nesting exceeds the supported depth")
    var b = Python.import_module("builtins")
    if py_type(value) is b.tuple or py_type(value) is b.list:
        for k in range(len(value)):  # driver-loop: over native operation argument containers
            classical_admit_value(value[k], context, depth + 1)
    elif not native_pandas_exact(value):
        sequential_admit_value(value, context, depth)


def _pd_type_fields(cls: PythonObject) raises -> PythonObject:
    return py_call(Python.import_module("builtins").type.__dict__["__dict__"].__get__, cls)


def _pd_fields(value: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if py_type(cls) is not b.type:
        raise Error("TypeError: pandas storage requires ordinary native class metadata")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for k in range(len(mro)):  # driver-loop: over the admitted class's native dictionary descriptors
        var descriptor = py_get(_pd_type_fields(mro[k]), "__dict__")
        if descriptor is not None and py_type(descriptor) is py_type(b.memoryview.obj):
            var fields = py_call(descriptor.__get__, value)
            if py_type(fields) is not b.dict:
                raise Error("TypeError: pandas storage must use its native instance dictionary")
            var keys = py_call(b.list, fields.keys())
            for j in range(len(keys)):  # driver-loop: over stored field names before any keyed lookup
                if py_type(keys[j]) is not b.str:
                    raise Error("TypeError: pandas stored field names must be exact native strings")
            return fields
    raise Error("TypeError: pandas storage has no native dictionary descriptor")


def _pd_get(module: String, cls: String, field: String, value: PythonObject) raises -> PythonObject:
    var owner = _pd_class(module, cls)
    if owner is None:
        raise Error("TypeError: pandas native storage metadata is unavailable")
    var descriptor = py_get(_pd_type_fields(owner), field)
    var b = Python.import_module("builtins")
    if descriptor is None or py_type(descriptor) is not py_type(b.memoryview.obj):
        raise Error("TypeError: pandas storage requires native field " + field)
    return py_call(descriptor.__get__, value)


def _pd_raw(value: PythonObject, offset: Int) raises -> UInt64:
    var size = Int(py=value.dtype.itemsize)
    var base = Int(py=sequential_address(value))
    var native = Bool(py=value.dtype.isnative)
    return pd_raw_at(base, offset, size, native)


def _pd_object_bits(value: PythonObject, context: PythonObject) raises -> UInt32:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var number = Float64(0.0)
    if cls is b.int or cls is b.float or cls is b.bool:
        number = Float64(py=value)
    elif _one_of(cls, context["scalar_types"]):
        # Scalar buffer metadata is native. Never call an object's __float__
        # or NumPy array/scalar conversion routine to obtain numeric storage.
        var view = py_call(b.memoryview, value)
        var array = py_call(context["ndarray"], Python.tuple(), dtype=value.dtype, buffer=view)
        var size = Int(py=array.dtype.itemsize)
        var kind = cast_in_kind(Int(py=array.dtype.num), size, "pandas scalar")
        number = _pd_double(_pd_raw(array, 0), kind, size)
        _ = array^
        _ = view^
    else:
        raise Error("TypeError: pandas object storage must contain native numeric scalars without Python conversion hooks")
    return f64_to_f32_bits(bitcast[DType.uint64](number))


def _pd_values(values: PythonObject, series: Bool, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(values) is context["ndarray"]:
        return Python.tuple(values, False)
    var cls = py_type(values)
    if cls is not _pd_class("pandas.core.arrays.integer", "IntegerArray") and cls is not _pd_class("pandas.core.arrays.floating", "FloatingArray") and cls is not _pd_class("pandas.core.arrays.boolean", "BooleanArray"):
        raise Error("TypeError: pandas input requires ordinary numeric storage without extension callbacks")
    var fields = _pd_fields(values)
    var data = py_get(fields, "_data")
    var mask = py_get(fields, "_mask")
    if py_type(data) is not context["ndarray"] or py_type(mask) is not context["ndarray"]:
        raise Error("TypeError: pandas nullable storage requires exact native arrays")
    if Int(py=data.ndim) != 1 or Int(py=mask.ndim) != 1 or Int(py=data.size) != Int(py=mask.size) or Int(py=mask.dtype.num) != 0:
        raise Error("TypeError: pandas nullable storage metadata differs")
    var rows = Int(py=mask.size)
    var stride = Int(py=mask.strides[0])
    pd_mask_check(Int(py=sequential_address(mask)), rows, stride, Int(py=mask.dtype.itemsize), Bool(py=mask.dtype.isnative))
    # DataFrame.to_numpy's default nullable blocks form object storage; Series
    # with no missing values exports its underlying numeric storage instead.
    return Python.tuple(data, not series)


def _pd_locations(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var location = _pd_get("pandas._libs.internals", "Block", "_mgr_locs", block)
    if py_type(location) is not _pd_class("pandas._libs.internals", "BlockPlacement"):
        raise Error("TypeError: pandas block requires native placement metadata")
    var indexer = _pd_get("pandas._libs.internals", "BlockPlacement", "indexer", location)
    if py_type(indexer) is b.slice:
        if py_type(indexer.start) is not b.int or py_type(indexer.stop) is not b.int or (indexer.step is not None and py_type(indexer.step) is not b.int):
            raise Error("TypeError: pandas block placement slice is not native")
        return py_call(b.range, indexer.start, indexer.stop, 1 if indexer.step is None else Int(py=indexer.step))
    if py_type(indexer) is not context["ndarray"] or Int(py=indexer.ndim) != 1 or String(indexer.dtype.kind) != "i":
        raise Error("TypeError: pandas block placement requires native integer storage")
    var result = Python.list()
    var size = Int(py=indexer.dtype.itemsize)
    var kind = cast_in_kind(Int(py=indexer.dtype.num), size, "pandas placement")
    var stride = Int(py=indexer.strides[0])
    for k in range(Int(py=indexer.size)):  # driver-loop: over block column positions
        py_call(result.append, PythonObject(Int(bitcast[DType.int64](cast_w64_bits(_pd_raw(indexer, k * stride), kind, size)))))
    return result


def native_pandas_f32(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if not native_pandas_exact(value):
        raise Error("TypeError: pandas input requires an exact DataFrame or Series")
    var series = py_type(value) is _pd_class("pandas.core.series", "Series")
    var manager = py_get(_pd_fields(value), "_mgr")
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: pandas Series requires native single-block storage")
        blocks = py_get(_pd_fields(manager), "blocks")
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: pandas DataFrame requires native block storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not b.tuple or len(blocks) == 0 or (series and len(blocks) != 1):
        raise Error("pandas input is empty or has invalid block metadata")
    var owned = Python.list()
    var placements = Python.list()
    var rows = -1
    var cols = 0
    var common_kind = String()
    var common_size = 0
    var objects = False
    var booleans = False
    var numbers = False
    for k in range(len(blocks)):  # driver-loop: over native pandas storage blocks
        var block = blocks[k]
        var cls = py_type(block)
        if cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock") and cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock"):
            raise Error("TypeError: pandas requires ordinary native blocks without subclass hooks")
        var pair = _pd_values(_pd_get("pandas._libs.internals", "Block", "values", block), series, context)
        var data = pair[0]
        var rank = Int(py=data.ndim)
        if rank != 1 and rank != 2:
            raise Error("pandas numeric block rank must be one or two")
        var nrows = Int(py=data.shape[rank - 1])
        var ncols = 1 if rank == 1 else Int(py=data.shape[0])
        if nrows < 1 or ncols < 1 or (rows >= 0 and nrows != rows) or (series and rank != 1):
            raise Error("pandas numeric block dimensions differ or are empty")
        rows = nrows
        var positions = py_call(b.range, 1) if series else _pd_locations(block, context)
        if len(positions) != ncols:
            raise Error("pandas block column positions differ from numeric storage")
        cols += ncols
        py_call(owned.append, data)
        py_call(placements.append, positions)
        var kind = String(data.dtype.kind)
        if kind == "O" or Bool(py=pair[1]):
            objects = True
        if kind != "O":
            if kind == "c":
                raise Error("Complex data not supported")
            _promote(common_kind, common_size, kind, Int(py=data.dtype.itemsize))
            booleans = booleans or kind == "b"
            numbers = numbers or kind != "b"
    # pandas deliberately uses object promotion for bool mixed with numbers.
    var double_round = objects or (booleans and numbers) or (common_kind == "f" and common_size == 8)
    var result = sequential_array(Python.tuple(rows) if series else Python.tuple(rows, cols), context["float32"], context)
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    var seen = List[Bool](length=cols, fill=False)
    for k in range(len(owned)):  # driver-loop: over admitted native block owners
        var data = owned[k]
        var positions = placements[k]
        var rank = Int(py=data.ndim)
        var size = Int(py=data.dtype.itemsize)
        var object_dtype = String(data.dtype.kind) == "O"
        var kind = 0 if object_dtype else cast_in_kind(Int(py=data.dtype.num), size, "pandas input")
        var row_stride = Int(py=data.strides[rank - 1])
        var col_stride = 0 if rank == 1 else Int(py=data.strides[0])
        for col in range(len(positions)):  # driver-loop: over pandas block columns, one typed column scatter each
            var dest = Int(py=positions[col])
            if dest < 0 or dest >= cols or seen[dest]:
                raise Error("pandas block column positions overlap or exceed the matrix")
            seen[dest] = True
            if object_dtype:
                # cpu-route: pandas object-dtype cells are Python objects; each one is read through
                # its native buffer metadata, which only the Python door can reach.
                for row in range(rows):  # cpu-route: Python objects marshaled one element at a time before the runtime
                    var item = data[row] if rank == 1 else data[Python.tuple(col, row)]
                    var bits = _pd_object_bits(item, context)
                    if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                        raise Error("Input contains NaN or infinity")
                    output[row * cols + dest] = bitcast[DType.float32](bits)
            else:
                pd_scatter_numeric(Int(py=sequential_address(data)), size, Bool(py=data.dtype.isnative), kind,
                                   row_stride, col_stride, col, dest, rows, cols, double_round, output)
        _ = data^
    _ = owned^
    _ = placements^
    _ = manager^
    return result
