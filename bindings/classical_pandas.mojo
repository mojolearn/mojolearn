# SPDX-License-Identifier: Apache-2.0
# Native ingestion of ordinary pandas storage, without DataFrame.__array__,
# manager properties, extension conversion methods or any Python callbacks.
# A caller imports pandas; runtime admission only reads its loaded modules.
from std.memory import bitcast
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.device_rows_door import door_is_device_rows
from bindings.native_buffer_owner import native_buffer_view
from core.cast_in import cast_in_kind, cast_w64_bits, f64_to_f32_bits
from core.classical_pandas_impl import pd_raw_at, _pd_double, pd_mask_check, pd_scatter_numeric
from bindings.sequential_inputs import _promote
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_admission import _one_of, sequential_admit_value, _SequentialValueTypes, _sequential_admit_value, _cached_value_types, _native_one_of
from bindings.native_loaded_module import NativeModules
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_contains
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.p3_labels_metrics_native import p3_type_word, p3_item


def _pd_native_metaclass(cls: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    if py_type(cls) is b.type:
        return True
    # pandas' ordinary DatetimeIndex/TimedeltaIndex classes use ABCMeta.
    # Accept its exact loaded identity for native metadata reads only;
    # never invoke abstract checks, metaclass hooks or class constructors.
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "abc")
    if module is None or py_type(module) is not py_type(sys):
        return False
    var meta = py_get(py_fields(module), "ABCMeta")
    return py_type(meta) is b.type and py_type(cls) is meta


def _pd_class(module_name: String, name: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, module_name)
    if module is None or py_type(module) is not py_type(sys):
        return PythonObject(None)
    var cls = py_get(py_fields(module), name)
    return cls if _pd_native_metaclass(cls) else PythonObject(None)


def _pd_class_word(modules: PyObjectPtr, module_name: String, name: String) raises -> Int:
    """The word of `sys.modules[module_name].<name>` read by C dict lookups
    (no attribute hook), or 0 when absent."""
    ref cpy = Python().cpython()
    var t = cpython_table()
    var module_key = PythonObject(module_name)
    var module = t[].PyDict_GetItem(modules, module_key._obj_ptr)
    if not module:
        return 0
    var fields = cpython_table()[].PyModule_GetDict(module)
    if not fields:
        cpython_table()[].PyErr_Clear()  # not a module: no pandas class here
        return 0
    var name_key = PythonObject(name)
    var cls = t[].PyDict_GetItem(fields, name_key._obj_ptr)
    if not cls:
        return 0
    return Int(cls)


def native_pandas_exact(value: PythonObject) raises -> Bool:
    # p3-labels-metrics: every input that is not a pandas object (lists,
    # arrays: nearly every call) is answered by two C dict lookups per name
    # and one class-pointer compare; only a class that IS the loaded
    # DataFrame / Series takes the full admitted check below (metaclass and
    # module identity), so the answer is unchanged.
    var modules = cpython_table()[].PyImport_GetModuleDict()
    if modules:
        var word = p3_type_word(value._obj_ptr)
        if word != _pd_class_word(modules, "pandas.core.frame", "DataFrame") and word != _pd_class_word(modules, "pandas.core.series", "Series"):
            return False
    var cls = py_type(value)
    return cls is _pd_class("pandas.core.frame", "DataFrame") or cls is _pd_class("pandas.core.series", "Series")


struct _ClassicalAdmitTypes(Movable):
    # The exact builtin type identities and the loaded pandas classes, read
    # once per admitted value. Every nested list element used to rebuild an
    # empty list and tuple for its type test, a fresh module handle and the
    # sequential admission's type vectors (five context reads and copies),
    # and two pandas class lookups (four module validations): per element of
    # a list X. The values visited and the refusals raised are unchanged.
    var list_type: PythonObject
    var tuple_type: PythonObject
    var int_type: PythonObject
    var float_type: PythonObject
    var bool_type: PythonObject
    var str_type: PythonObject
    var none_type: PythonObject
    var pandas_read: Bool
    var frame: PythonObject
    var series: PythonObject

    def __init__(out self) raises:
        self.list_type = py_type(Python.list())
        self.tuple_type = py_type(Python.tuple())
        self.int_type = py_type(PythonObject(0))
        self.float_type = py_type(PythonObject(0.0))
        self.bool_type = py_type(PythonObject(True))
        self.str_type = py_type(PythonObject(""))
        self.none_type = py_type(PythonObject(None))
        self.pandas_read = False
        self.frame = PythonObject(None)
        self.series = PythonObject(None)

    def __init__(out self, seq: _SequentialValueTypes) raises:
        # The list and tuple identities the admission type vectors already
        # hold (the builtins' exact types): no temporary containers.
        self.list_type = seq.list_type
        self.tuple_type = seq.tuple_type
        self.int_type = py_type(PythonObject(0))
        self.float_type = py_type(PythonObject(0.0))
        self.bool_type = py_type(PythonObject(True))
        self.str_type = py_type(PythonObject(""))
        self.none_type = py_type(PythonObject(None))
        self.pandas_read = False
        self.frame = PythonObject(None)
        self.series = PythonObject(None)

    def builtin_scalar(self, cls: PythonObject) -> Bool:
        return (cls is self.int_type or cls is self.float_type or cls is self.bool_type
                or cls is self.str_type or cls is self.none_type)

    def pandas_exact(mut self, cls: PythonObject) raises -> Bool:
        # The pandas classes are looked up on the first non-scalar leaf only:
        # a list of numbers never reads them.
        if not self.pandas_read:
            self.frame = _pd_class("pandas.core.frame", "DataFrame")
            self.series = _pd_class("pandas.core.series", "Series")
            self.pandas_read = True
        return cls is self.frame or cls is self.series


def _admitted_word(cls: PythonObject, seq: _SequentialValueTypes) -> Int:
    """The class word of `cls` when the sequential admission accepts it as a
    leaf (`_sequential_admit_value` returns at once for it), else 0."""
    if _native_one_of(cls, seq.native_types) or _native_one_of(cls, seq.scalar_types):
        return Int(cls._obj_ptr)
    return 0


def _classical_admit(mut mods: NativeModules, seq: _SequentialValueTypes, mut kinds: _ClassicalAdmitTypes,
                     value: PythonObject, context: PythonObject, depth: Int) raises:
    if depth > 64:
        raise Error("TypeError: classical input nesting exceeds the supported depth")
    var cls = py_type(value)
    if cls is kinds.tuple_type or cls is kinds.list_type:
        # Lane p2-labels-lists-arrays: this walk visits every element of a
        # list input (it was tagged as a loop over argument containers). Each
        # element is read borrowed from the exact container's C storage and an
        # exact builtin scalar leaf (the common X / y list) stops at one class
        # pointer compare: no boxed index, no subscript, no recursion per leaf.
        var is_tuple = cls is kinds.tuple_type
        # A leaf skips only when the admission would accept its class as is
        # (a connected native or scalar type); 0 never matches a class word.
        var int_word = _admitted_word(kinds.int_type, seq)
        var float_word = _admitted_word(kinds.float_type, seq)
        var bool_word = _admitted_word(kinds.bool_type, seq)
        var str_word = _admitted_word(kinds.str_type, seq)
        var none_word = _admitted_word(kinds.none_type, seq)
        for k in range(len(value)):  # driver-loop: per ELEMENT of a list input (data-sized), one borrowed class-pointer read each from the C storage
            var raw = p3_item(value, k, is_tuple)
            var word = p3_type_word(raw)
            if word == int_word or word == float_word or word == bool_word or word == str_word or word == none_word:
                continue
            _classical_admit(mods, seq, kinds, PythonObject(from_borrowed=raw), context, depth + 1)
        _ = value  # keep the container alive through the borrowed reads above
    elif kinds.builtin_scalar(cls):
        # An exact builtin scalar is never pandas storage: skip the pandas
        # class test.
        _sequential_admit_value(mods, value, context, seq, depth)
    elif door_is_device_rows(value, context):
        # A model-selection fold (exact `_arena_io.DeviceRows`): its owner
        # admitted the base and indices natively when it was made; the X
        # check resolves it (bindings/device_rows_door.mojo).
        return
    elif not kinds.pandas_exact(cls):
        _sequential_admit_value(mods, value, context, seq, depth)


def classical_admit_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    var mods = NativeModules()
    classical_admit_value(mods, value, context, depth)


def classical_admit_value(mut mods: NativeModules, value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: classical input nesting exceeds the supported depth")
    # Once per admitted value, not per element: one module handle, one set of
    # sequential admission type vectors, one set of builtin/pandas types.
    # The type vectors come from the per-context cache sequential admission
    # keeps (an identity check of the six connected context entries) instead
    # of five context reads and vector rebuilds per call; the builtins
    # connection check runs first inside the cache, as before.
    var seq = _cached_value_types(mods, context)
    var kinds = _ClassicalAdmitTypes(seq)
    _classical_admit(mods, seq, kinds, value, context, depth)


def classical_admit_fields(fields: PythonObject, names: PythonObject, context: PythonObject) raises:
    """`sequential_admit_value` of every connected metadata field the model
    holds, with one module handle and one set of admission type vectors for
    the whole pass (they were rebuilt for every field on every call)."""
    var mods = NativeModules()
    classical_admit_fields(mods, fields, names, context)


def classical_admit_fields(mut mods: NativeModules, fields: PythonObject, names: PythonObject, context: PythonObject) raises:
    """The same pass with the caller's module handle."""
    var seq = _cached_value_types(mods, context)
    for k in range(len(names)):  # driver-loop: over connected estimator metadata fields
        if py_contains(fields, names[k]):
            _sequential_admit_value(mods, fields[names[k]], context, seq, 0)


def _pd_type_fields(cls: PythonObject) raises -> PythonObject:
    return py_call(native_loaded_module("builtins").type.__dict__["__dict__"].__get__, cls)


def _pd_fields(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if not _pd_native_metaclass(cls):
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
    var b = native_loaded_module("builtins")
    if descriptor is None or py_type(descriptor) is not py_type(b.memoryview.obj):
        raise Error("TypeError: pandas storage requires native field " + field)
    return py_call(descriptor.__get__, value)


def _pd_raw(value: PythonObject, offset: Int) raises -> UInt64:
    var size = Int(py=value.dtype.itemsize)
    var base = Int(py=sequential_address(value))
    var native = Bool(py=value.dtype.isnative)
    return pd_raw_at(base, offset, size, native)


def _pd_object_bits(value: PythonObject, context: PythonObject) raises -> UInt32:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var number = Float64(0.0)
    if cls is b.int or cls is b.float or cls is b.bool:
        number = Float64(py=value)
    elif _one_of(cls, context["scalar_types"]):
        # Scalar buffer metadata is native. Never call an object's __float__
        # or NumPy array/scalar conversion routine to obtain numeric storage.
        var view = native_buffer_view(value)
        var array = py_call(context["ndarray"], Python.tuple(), dtype=value.dtype, buffer=view)
        var size = Int(py=array.dtype.itemsize)
        var kind = cast_in_kind(Int(py=array.dtype.num), size, "pandas scalar")
        number = _pd_double(_pd_raw(array, 0), kind, size)
        _ = array^
        _ = view^
    else:
        raise Error("TypeError: pandas object storage must contain native numeric scalars without Python conversion hooks")
    return f64_to_f32_bits(bitcast[DType.uint64](number))


def _pd_cell_bits(raw: PyObjectPtr, int_word: Int, float_word: Int, bool_word: Int,
                  context: PythonObject) raises -> UInt32:
    """`_pd_object_bits` of one borrowed object-block cell: exact int, float
    and bool through PyFloat_AsDouble (the value `Float64(py=v)` read),
    anything else through `_pd_object_bits` (same refusal)."""
    ref cpy = Python().cpython()
    if not raw:
        raise Error("TypeError: pandas object storage must contain native numeric scalars without Python conversion hooks")
    var word = p3_type_word(raw)
    if word == int_word or word == float_word or word == bool_word:
        var number = cpython_table()[].PyFloat_AsDouble(raw)
        if number == -1.0:
            if cpython_table()[].PyErr_Occurred():
                raise cpy.unsafe_get_error()
        return f64_to_f32_bits(bitcast[DType.uint64](number))
    return _pd_object_bits(PythonObject(from_borrowed=raw), context)


def _pd_values(values: PythonObject, series: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
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
    var b = native_loaded_module("builtins")
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


def native_pandas_f32(value: PythonObject, context: PythonObject, strict_f32: Bool = False, allow_nonfinite: Bool = False,
                      refuse_double: String = "") raises -> PythonObject:
    """One C-ordered float32 copy of an exact pandas DataFrame or Series.
    A nonempty `refuse_double` is the refusal raised, before any conversion,
    when the storage's common dtype is not float32-or-narrower (float64, or
    object promotion of object/nullable-frame/bool-with-number blocks): the
    dtype `np.asarray` would give, so float64-refusing callers keep their
    rule while integer, half and float32 storage is still admitted."""
    var b = native_loaded_module("builtins")
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
    # The four admitted block classes, read once per conversion (they were
    # read per block; p3-labels-metrics).
    var numpy_block = _pd_class("pandas.core.internals.blocks", "NumpyBlock")
    var numeric_block = _pd_class("pandas.core.internals.blocks", "NumericBlock")
    var object_block = _pd_class("pandas.core.internals.blocks", "ObjectBlock")
    var extension_block = _pd_class("pandas.core.internals.blocks", "ExtensionBlock")
    for k in range(len(blocks)):  # driver-loop: over native pandas storage blocks
        var block = blocks[k]
        var cls = py_type(block)
        if cls is not numpy_block and cls is not numeric_block and cls is not object_block and cls is not extension_block:
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
    if strict_f32 and (objects or booleans or common_kind != "f" or common_size != 4):
        raise Error("TypeError: Scaler input must have dtype float32")
    # pandas deliberately uses object promotion for bool mixed with numbers.
    var double_round = objects or (booleans and numbers) or (common_kind == "f" and common_size == 8)
    if double_round and refuse_double.byte_length() > 0:
        raise Error(refuse_double)
    var result = sequential_array(Python.tuple(rows) if series else Python.tuple(rows, cols), context["float32"], context)
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    var int_word = Int(b.int._obj_ptr)
    var float_word = Int(b.float._obj_ptr)
    var bool_word = Int(b.bool._obj_ptr)
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
                # cpu-route: pandas object-dtype cells are Python objects. p3-labels-metrics: each
                # cell's PyObject pointer is read from the object block's storage (no boxed index
                # or index tuple per cell); exact int, float and bool convert by PyFloat_AsDouble
                # (their C slot, as `float(v)` ran it); NumPy scalars keep their buffer read.
                var cells = Int(py=sequential_address(data))
                var cell_col = col * col_stride if rank == 2 else 0
                for row in range(rows):  # cpu-route: Python objects marshaled one element at a time before the runtime
                    var raw = MutPointer[PyObjectPtr, MutUntrackedOrigin](unsafe_from_address=cells + row * row_stride + cell_col)[0]
                    var bits = _pd_cell_bits(raw, int_word, float_word, bool_word, context)
                    if not allow_nonfinite and (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                        raise Error("Input contains NaN or infinity")
                    output[row * cols + dest] = bitcast[DType.float32](bits)
            else:
                pd_scatter_numeric(Int(py=sequential_address(data)), size, Bool(py=data.dtype.isnative), kind,
                                   row_stride, col_stride, col, dest, rows, cols, double_round, output, allow_nonfinite)
        _ = data^
    _ = owned^
    _ = placements^
    _ = manager^
    return result
