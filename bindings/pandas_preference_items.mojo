# SPDX-License-Identifier: Apache-2.0
"""Passive pandas preference storage UNPACK, without Series/Index iteration.

Ordinary Series/Index iteration yields ndarray.item's Python scalar; nullable
numeric iteration yields NumPy scalars. Preserve that distinction before the
shared sequence dtype inference. Item reads require the GIL and are inherently
ordered CPython admission, not native numerical computation. Conversion runs
later through the existing parallel SequenceInputPlan writer.
"""
from std.python import Python, PythonObject
from std.memory import bitcast
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get, _pd_type_fields
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter
from providers.python.resample_pandas_pyapi import _rp_sparse_field
from bindings.pandas_arrow_native import _chunked
from bindings.sequential_admission import _one_of
from core.sequential_input_plan import sequence_promote


def _array_length(value: PythonObject, context: PythonObject) raises -> Int:
    if py_type(value) is not context["ndarray"]:
        raise Error("TypeError: pandas preference storage requires an exact native array")
    admit_native_buffer_exporter(value)
    if Int(py=value.ndim) != 1:
        raise Error("TypeError: pandas preference storage requires a native vector")
    return Int(py=value.size)


def _item_method(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var descriptor = py_get(_pd_type_fields(cls), "item")
    if (descriptor is None or py_type(descriptor) is not py_type(b.list.append)
            or descriptor.__objclass__ is not cls):
        raise Error("TypeError: pandas preference scalars require the native ndarray item descriptor")
    return descriptor


struct PandasPreferenceSource(Movable):
    # Boundary-only passive storage pins; RUN receives the separate native
    # SequenceInputPlan. No tensor is materialized to inspect these scalars.
    var data: PythonObject
    var mask: PythonObject
    var item: PythonObject
    var extra: PythonObject
    var box: PythonObject
    var mode: Int  # 0 ndarray.item; 1 nullable; 2 range; 3 sparse; 4 Arrow
    var count: Int
    var stored: Int
    var force_float: Bool
    var force_invalid: Bool
    var temporal: Bool
    var box_kind: String

    def __init__(out self, data: PythonObject, mask: PythonObject, item: PythonObject, mode: Int, count: Int):
        self.data = data
        self.mask = mask
        self.item = item
        self.extra = PythonObject(None)
        self.box = PythonObject(None)
        self.mode = mode
        self.count = count
        self.stored = 0
        self.force_float = False
        self.force_invalid = False
        self.temporal = False
        self.box_kind = String()

    def get(self, index: Int, box_nullable: Bool = False) raises -> PythonObject:
        if index < 0 or index >= self.count:
            raise Error("IndexError: pandas preference code lies outside its native categories")
        if self.mode == 5:
            return PythonObject(None)
        if self.mode == 3:
            var position = Int(py=py_call(self.item, self.mask, PythonObject(index)))
            if position < -1 or position >= self.stored:
                raise Error("IndexError: pandas sparse preference index exceeds its retained values")
            var value = self.extra
            if position >= 0 and self.temporal:
                return PythonObject(None)
            if position >= 0:
                value = py_call(self.box, self.data, PythonObject(position)) if box_nullable else self.data[position]
            if box_nullable and self.force_invalid:
                return PythonObject(None)
            if box_nullable and self.force_float:
                return PythonObject(Float64(py=value))
            return value
        if self.mode == 4:
            var scalar = self.data[index]
            if py_type(scalar) is not self.extra:
                raise Error("TypeError: pandas Arrow preference requires its native primitive scalar type")
            if self.stored == 0 or not Bool(py=_pd_get("pyarrow.lib", "Scalar", "is_valid", scalar)):
                return PythonObject(None)
            var value = py_call(self.item, scalar)
            if box_nullable and self.force_float:
                return PythonObject(Float64(py=value))
            return value
        if self.mask is not None and Bool(py=self.mask[index]):
            # Same numeric-scalar refusal as pandas.NA, at the same position.
            return PythonObject(None)
        if self.mode == 2 or (self.mode == 1 and not box_nullable):
            return self.data[index]
        return py_call(self.item, self.data, PythonObject(index))

    def prepare_boxing(mut self, context: PythonObject) raises:
        if self.mode == 2:
            # Category/MultiIndex uses RangeIndex's int64 array representation;
            # ordinary RangeIndex iteration continues to use Python integers.
            # Honor a pre-existing native cached representation without making
            # a new private ndarray just to read the scalar values.
            var b = native_loaded_module("builtins")
            var cache = py_get(_pd_fields(self.extra), "_cache")
            if cache is not None:
                if py_type(cache) is not b.dict or not native_namespace_string_keys(cache._obj_ptr, b.str):
                    raise Error("TypeError: pandas RangeIndex cache requires native string keys")
                if py_contains(cache, "_data"):
                    var data = py_get(cache, "_data")
                    self.count = _array_length(data, context)
                    self.data = data
                    self.item = _item_method(data)
                    self.mode = 0
                    return
            if self.count:
                _ = Int(py=self.data[0])
                _ = Int(py=self.data[self.count - 1])
            return
        if self.mode != 3 or self.stored == self.count:
            return
        # SparseArray.__array__ converted its fill before gathering categories,
        # even when a category selection later skipped every gap. Preserve that
        # error stage without materializing a dense tensor or invoking NumPy.
        if self.force_float:
            self.extra = PythonObject(Float64(py=self.extra))
        elif self.box_kind == "i" or self.box_kind == "u":
            var raw = cpython_table()[].PyNumber_Index(self.extra._obj_ptr)
            if not raw:
                raise Python().cpython().unsafe_get_error()
            self.extra = PythonObject(from_owned=raw)
            if self.box_kind == "i":
                var word = cpython_table()[].PyLong_AsLongLong(self.extra._obj_ptr)
                if word == -1 and Python().cpython().PyErr_Occurred():
                    raise Python().cpython().unsafe_get_error()
        elif self.box_kind == "b":
            self.extra = PythonObject(Bool(py=self.extra))


def _numeric_source(value: PythonObject, context: PythonObject) raises -> PandasPreferenceSource:
    var cls = py_type(value)
    if cls is _pd_class("pandas.core.arrays.sparse.array", "SparseArray"):
        return _sparse_source(value, context)
    if (cls is _pd_class("pandas.core.arrays.arrow.array", "ArrowExtensionArray")
            or cls is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray")):
        return _arrow_source(value, context)
    if (cls is _pd_class("pandas.core.arrays.datetimes", "DatetimeArray")
            or cls is _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray")
            or cls is _pd_class("pandas.core.arrays.period", "PeriodArray")):
        var data = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", value)
        var source = PandasPreferenceSource(data, PythonObject(None), PythonObject(None), 5, _array_length(data, context))
        source.temporal = True
        return source^
    if (cls is _pd_class("pandas.core.arrays.numpy_", "NumpyExtensionArray")
            or cls is _pd_class("pandas.core.arrays.numpy_", "PandasArray")):
        var data = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", value)
        return PandasPreferenceSource(data, PythonObject(None), _item_method(data), 1, _array_length(data, context))
    if cls is _pd_class("pandas.core.arrays.interval", "IntervalArray"):
        var data = py_get(_pd_fields(value), "_left")
        if py_type(data) is not context["ndarray"]:
            if (py_type(data) is not _pd_class("pandas.core.arrays.datetimes", "DatetimeArray")
                    and py_type(data) is not _pd_class("pandas.core.arrays.timedeltas", "TimedeltaArray")):
                raise Error("TypeError: pandas interval preference requires native endpoint storage")
            data = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", data)
        return PandasPreferenceSource(data, PythonObject(None), PythonObject(None), 5, _array_length(data, context))
    if (cls is _pd_class("pandas.core.arrays.string_", "StringArray")
            or cls is _pd_class("pandas.core.arrays.string_", "StringArrayNumpySemantics")):
        var data = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", value)
        return PandasPreferenceSource(data, PythonObject(None), _item_method(data), 0, _array_length(data, context))
    if cls is context["ndarray"]:
        var count = _array_length(value, context)
        return PandasPreferenceSource(value, PythonObject(None), _item_method(value), 0, count)
    if (cls is _pd_class("pandas.core.arrays.integer", "IntegerArray")
            or cls is _pd_class("pandas.core.arrays.floating", "FloatingArray")
            or cls is _pd_class("pandas.core.arrays.boolean", "BooleanArray")):
        var fields = _pd_fields(value)
        var data = py_get(fields, "_data")
        var mask = py_get(fields, "_mask")
        var count = _array_length(data, context)
        if _array_length(mask, context) != count or Int(py=mask.dtype.num) != 0:
            raise Error("TypeError: pandas nullable preference requires matching native boolean mask storage")
        return PandasPreferenceSource(data, mask, _item_method(data), 1, count)
    raise Error("TypeError: pandas preference requires native numeric storage without extension callbacks")


def _plain_index_source(value: PythonObject, context: PythonObject) raises -> PandasPreferenceSource:
    var cls = py_type(value)
    if cls is _pd_class("pandas.core.indexes.range", "RangeIndex"):
        var b = native_loaded_module("builtins")
        var source = py_get(_pd_fields(value), "_range")
        if py_type(source) is not b.range:
            raise Error("TypeError: pandas RangeIndex preference requires native range storage")
        var result = PandasPreferenceSource(source, PythonObject(None), PythonObject(None), 2, len(source))
        result.extra = value
        return result^
    if (cls is _pd_class("pandas.core.indexes.base", "Index")
            or cls is _pd_class("pandas.core.indexes.numeric", "Int64Index")
            or cls is _pd_class("pandas.core.indexes.numeric", "UInt64Index")
            or cls is _pd_class("pandas.core.indexes.numeric", "Float64Index")
            or cls is _pd_class("pandas.core.indexes.datetimes", "DatetimeIndex")
            or cls is _pd_class("pandas.core.indexes.timedeltas", "TimedeltaIndex")
            or cls is _pd_class("pandas.core.indexes.period", "PeriodIndex")
            or cls is _pd_class("pandas.core.indexes.interval", "IntervalIndex")):
        return _numeric_source(py_get(_pd_fields(value), "_data"), context)
    raise Error("TypeError: pandas preference column labels require native numeric Index storage")


def _collect(source: PandasPreferenceSource) raises -> List[PythonObject]:
    var result = List[PythonObject](capacity=source.count)
    for i in range(source.count):  # admission: ordered exact C storage reads preserve Python-versus-NumPy scalar kinds
        result.append(source.get(i))
    _ = source
    return result^


@fieldwise_init
struct PandasColumnSource(Movable):
    var source: PandasPreferenceSource
    var codes: PythonObject
    var count: Int

    def get(self, index: Int) raises -> PythonObject:
        if self.codes is None:
            return self.source.get(index, True)
        if index < 0 or index >= self.count:
            raise Error("IndexError: pandas preference level code lies outside its native categories")
        var code = Int(py=self.codes[index])
        if code == -1:
            return self.absent()
        return self.source.get(code, True)

    def absent(self) -> PythonObject:
        if self.source.temporal:
            return PythonObject(None)
        return PythonObject(bitcast[DType.float64](UInt64(0x7ff8000000000000)))


def _categorical_source(value: PythonObject, context: PythonObject) raises -> PandasColumnSource:
    if py_type(value) is not _pd_class("pandas.core.arrays.categorical", "Categorical"):
        raise Error("TypeError: pandas categorical preference requires exact native categorical storage")
    var codes = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", value)
    var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", value)
    if py_type(dtype) is not _pd_class("pandas.core.dtypes.dtypes", "CategoricalDtype"):
        raise Error("TypeError: pandas categorical preference requires native category metadata")
    var count = _array_length(codes, context)
    if String(codes.dtype.kind) != "i":
        raise Error("TypeError: pandas categorical preference requires native signed category codes")
    var categories = py_get(_pd_fields(dtype), "_categories")
    var source = _plain_index_source(categories, context)
    source.prepare_boxing(context)
    return PandasColumnSource(source^, codes, count)


def _column_source(value: PythonObject, context: PythonObject) raises -> PandasColumnSource:
    if py_type(value) is _pd_class("pandas.core.indexes.category", "CategoricalIndex"):
        return _categorical_source(py_get(_pd_fields(value), "_data"), context)
    var source = _plain_index_source(value, context)
    source.prepare_boxing(context)
    var count = source.count
    return PandasColumnSource(source^, PythonObject(None), count)


def _categorical_items(value: PythonObject, context: PythonObject) raises -> List[PythonObject]:
    var source = _categorical_source(value, context)
    var result = List[PythonObject](capacity=source.count)
    for i in range(source.count):  # admission: native categorical code selects its retained scalar owner; no pandas take/iterator
        result.append(source.get(i))
    _ = source^
    return result^


def _static_descriptor(cls: PythonObject, name: String) raises -> PythonObject:
    # These callers first admit exact known pandas/Arrow C extension types.
    # Their immutable type namespaces cannot be rebound to Python callbacks.
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var function_type = native_loaded_module("types").FunctionType
    for i in range(len(ancestry)):  # metadata: a native C extension's fixed ancestry
        var owner = ancestry[i]
        var descriptor = py_get(_pd_type_fields(owner), name)
        if descriptor is None:
            continue
        if ((cpython_table()[].PyType_GetFlags(owner._obj_ptr) & UInt64(1 << 8)) == 0
                or py_type(descriptor) is function_type):
            raise Error("TypeError: pandas preference storage methods must be immutable native descriptors")
        return descriptor
    raise Error("TypeError: pandas preference native storage method is unavailable")


def _sparse_source(value: PythonObject, context: PythonObject) raises -> PandasPreferenceSource:
    # Preserve SparseArray._get_val_at's exact native lookup primitive and
    # sparse representation. No dense staging or replacement search algorithm.
    var fields = _pd_fields(value)
    var dtype = py_get(fields, "_dtype")
    if py_type(dtype) is not _pd_class("pandas.core.dtypes.dtypes", "SparseDtype"):
        raise Error("TypeError: pandas sparse preference requires native dtype metadata")
    var dtype_fields = _pd_fields(dtype)
    var fill = py_get(dtype_fields, "_fill_value")
    var index = py_get(fields, "_sparse_index")
    if (py_type(index) is not _pd_class("pandas._libs.sparse", "IntIndex")
            and py_type(index) is not _pd_class("pandas._libs.sparse", "BlockIndex")):
        raise Error("TypeError: pandas sparse preference requires a native sparse index")
    var lookup = _static_descriptor(py_type(index), "lookup")
    var rows = Int(py=_rp_sparse_field(index, "length"))
    var points = Int(py=_rp_sparse_field(index, "npoints"))
    var data = py_get(fields, "_sparse_values")
    if rows < 0 or points != _array_length(data, context) or points > rows:
        raise Error("TypeError: pandas sparse preference storage lengths disagree")
    var subtype = py_get(dtype_fields, "_dtype")
    if py_type(subtype) is not py_type(data.dtype) or String(subtype.str) != String(data.dtype.str):
        raise Error("TypeError: pandas sparse preference subtype and storage disagree")
    var source = PandasPreferenceSource(data, index, lookup, 3, rows)
    source.extra = fill
    source.box = _item_method(data)
    source.stored = points
    # A categorical/MultiIndex uses SparseArray's dense scalar representation.
    # Its common dtype depends on fill TYPE only when there are gaps; ordinary
    # Series iteration retains the original fill object and NumPy scalars.
    var kind = String(data.dtype.kind)
    source.temporal = kind == "M" or kind == "m"
    var width = Int(py=data.dtype.itemsize)
    if points != rows and kind != "O":
        var b = native_loaded_module("builtins")
        var fill_cls = py_type(fill)
        var incoming = String()
        var incoming_width = 0
        if fill_cls is b.bool:
            incoming = "b"
            incoming_width = 1
        elif fill_cls is b.int:
            incoming = "i"
            incoming_width = 8
        elif fill_cls is b.float:
            incoming = "f"
            incoming_width = 8
        elif (cpython_table()[].PyType_GetFlags(fill_cls._obj_ptr) & UInt64(512)) == 0 and _one_of(fill_cls, context["scalar_types"]):
            incoming = String(fill.dtype.kind)
            incoming_width = Int(py=fill.dtype.itemsize)
        elif fill_cls is b.str or fill_cls is b.bytes or fill_cls is b.complex:
            incoming = "unsupported"
        else:
            # np.result_type(dtype, an unknown fill class) refused; SparseArray
            # used object storage. Objects stay pinned and are never coerced.
            kind = "O"
        if incoming.byte_length() > 0:
            if ((kind == "b" or kind == "i" or kind == "u" or kind == "f")
                    and (incoming == "b" or incoming == "i" or incoming == "u" or incoming == "f")):
                sequence_promote(kind, width, incoming, incoming_width)
            else:
                kind = "unsupported"
    source.box_kind = kind
    source.force_float = kind == "f" and (width == 2 or width == 4 or width == 8)
    source.force_invalid = (kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "O") or (kind == "f" and width != 2 and width != 4 and width != 8)
    return source^


def _arrow_source(value: PythonObject, context: PythonObject) raises -> PandasPreferenceSource:
    var chunked = _chunked(value)
    _ = _static_descriptor(py_type(chunked), "__getitem__")
    var dtype = _pd_get("pyarrow.lib", "ChunkedArray", "type", chunked)
    if py_type(dtype) is not _pd_class("pyarrow.lib", "DataType"):
        raise Error("TypeError: pandas Arrow preference requires a primitive native type")
    var type_id = Int(py=_pd_get("pyarrow.lib", "DataType", "id", dtype))
    var names: List[String] = ["NullScalar", "BooleanScalar", "UInt8Scalar", "Int8Scalar",
        "UInt16Scalar", "Int16Scalar", "UInt32Scalar", "Int32Scalar", "UInt64Scalar", "Int64Scalar",
        "HalfFloatScalar", "FloatScalar", "DoubleScalar", "StringScalar", "BinaryScalar"]
    var name = String()
    if type_id >= 0 and type_id < len(names):
        name = names[type_id]
    elif type_id == 34:
        name = "LargeStringScalar"
    elif type_id == 35:
        name = "LargeBinaryScalar"
    var rows = len(chunked)
    if name.byte_length() == 0:
        raise Error("TypeError: pandas Arrow preference requires native numeric, string, binary or null storage")
    var scalar_class = _pd_class("pyarrow.lib", name)
    if scalar_class is None:
        raise Error("TypeError: pandas Arrow native scalar type is unavailable")
    var as_py = _static_descriptor(scalar_class, "as_py")
    var source = PandasPreferenceSource(chunked, PythonObject(None), as_py, 4, rows)
    source.extra = scalar_class
    source.stored = type_id
    source.force_float = type_id >= 10 and type_id <= 12
    return source^


def _native_numeric_storage_items(value: PythonObject, context: PythonObject) raises -> List[PythonObject]:
    if py_type(value) is _pd_class("pandas.core.arrays.categorical", "Categorical"):
        return _categorical_items(value, context)
    var source = _numeric_source(value, context)
    return _collect(source)


def _index_items(value: PythonObject, context: PythonObject) raises -> List[PythonObject]:
    var cls = py_type(value)
    if cls is _pd_class("pandas.core.indexes.category", "CategoricalIndex"):
        return _categorical_items(py_get(_pd_fields(value), "_data"), context)
    if cls is _pd_class("pandas.core.indexes.multi", "MultiIndex"):
        var fields = _pd_fields(value)
        var levels = py_get(fields, "_levels")
        var codes = py_get(fields, "_codes")
        var frozen = _pd_class("pandas.core.indexes.frozen", "FrozenList")
        if py_type(levels) is not frozen or py_type(codes) is not frozen:
            raise Error("TypeError: pandas MultiIndex preference requires native level/code storage")
        ref cpy = Python().cpython()
        # FrozenList is a list subtype; read its C list storage, bypassing all
        # Python list-subclass methods. PyList_GetItem validates the storage.
        var level_count = cpython_table()[].PyList_Size(levels._obj_ptr)
        if level_count < 1 or cpython_table()[].PyList_Size(codes._obj_ptr) != level_count:
            raise Error("TypeError: pandas MultiIndex preference requires matching levels and codes")
        var sources = List[PandasColumnSource]()
        var code_pins = List[PythonObject]()
        var count = -1
        for axis in range(level_count):  # metadata: Index levels, each retained once
            var raw_level = cpy.PyList_GetItem(levels._obj_ptr, axis)
            var raw_code = cpy.PyList_GetItem(codes._obj_ptr, axis)
            if not raw_level or not raw_code:
                raise cpy.unsafe_get_error()
            var level = PythonObject(from_borrowed=raw_level)
            var code = PythonObject(from_borrowed=raw_code)
            var rows = _array_length(code, context)
            if (count >= 0 and count != rows) or String(code.dtype.kind) != "i":
                raise Error("TypeError: pandas MultiIndex preference requires matching native signed codes")
            count = rows
            sources.append(_column_source(level, context))
            code_pins.append(code)
        var result = List[PythonObject](capacity=max(count, 0))
        for row in range(count):  # admission: construct each original tuple-valued column label with the CPython tuple API
            var raw = cpy.PyTuple_New(level_count)
            if not raw:
                raise cpy.unsafe_get_error()
            var label = PythonObject(from_owned=raw)
            for axis in range(level_count):  # metadata: components of this admitted column label
                var code = Int(py=code_pins[axis][row])
                var item = sources[axis].absent() if code == -1 else sources[axis].get(code)
                if cpy.PyTuple_SetItem(label._obj_ptr, axis, cpy.Py_NewRef(item._obj_ptr)) != 0:
                    raise cpy.unsafe_get_error()
            result.append(label)
        _ = sources^
        _ = code_pins^
        _ = levels
        _ = codes
        _ = value
        return result^
    if cls is _pd_class("pandas.core.indexes.base", "Index"):
        return _native_numeric_storage_items(py_get(_pd_fields(value), "_data"), context)
    var source = _plain_index_source(value, context)
    return _collect(source)


def pandas_preference_items(value: PythonObject, context: PythonObject) raises -> List[PythonObject]:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var manager = py_get(_pd_fields(value), "_mgr")
    if cls is _pd_class("pandas.core.series", "Series"):
        if py_type(manager) is _pd_class("pandas.core.internals.array_manager", "SingleArrayManager"):
            var arrays = _rp_sparse_field(manager, "arrays")
            if py_type(arrays) is not b.list or len(arrays) != 1:
                raise Error("TypeError: pandas Series preference requires one native storage array")
            return _native_numeric_storage_items(arrays[0], context)
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: pandas Series preference requires native single-block storage")
        var blocks = py_get(_pd_fields(manager), "blocks")
        if py_type(blocks) is not b.tuple or len(blocks) != 1:
            raise Error("TypeError: pandas Series preference requires one native storage block")
        var block = blocks[0]
        var block_cls = py_type(block)
        if (block_cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock")
                and block_cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock")
                and block_cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock")
                and block_cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock")
                and block_cls is not _pd_class("pandas.core.internals.blocks", "CategoricalBlock")
                and block_cls is not _pd_class("pandas.core.internals.blocks", "DatetimeLikeBlock")
                and block_cls is not _pd_class("pandas.core.internals.blocks", "DatetimeTZBlock")):
            raise Error("TypeError: pandas preference requires an ordinary native storage block")
        var storage = _pd_get("pandas._libs.internals", "Block", "values", block)
        return _native_numeric_storage_items(storage, context)
    if cls is not _pd_class("pandas.core.frame", "DataFrame"):
        raise Error("TypeError: pandas preference requires an exact DataFrame or Series")
    if py_type(manager) is _pd_class("pandas.core.internals.array_manager", "ArrayManager"):
        var axes = _rp_sparse_field(manager, "_axes")
        if py_type(axes) is not b.list or len(axes) != 2:
            raise Error("TypeError: pandas DataFrame preference requires native axis metadata")
        return _index_items(axes[1], context)
    if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
        raise Error("TypeError: pandas DataFrame preference requires native block storage")
    # DataFrame iteration yields column labels, never its cell values.
    var axes = _pd_get("pandas._libs.internals", "BlockManager", "axes", manager)
    if py_type(axes) is not b.list or len(axes) != 2:
        raise Error("TypeError: pandas DataFrame preference requires native axis metadata")
    return _index_items(axes[0], context)
