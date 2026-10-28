# SPDX-License-Identifier: Apache-2.0
# Complete AutoARIMA construction and forecast operation boundaries. Search and
# fitting remain separate owners; this provider never calls their Python methods.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_typed, classical_address
from bindings.classical_pandas import classical_admit_value
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.xseq_marshal import xseq_cast_nd
from core.cast_in import NPY_DOUBLE, NPY_FLOAT, CAST_KIND_SIGNED
from bindings.sequential_admission import _one_of

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]


def _fields(model: PythonObject, cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(model) is not cls or py_type(cls) is not b.type:
        raise Error("TypeError: AutoARIMA requires its connected native model classes")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # driver-loop: over native instance dictionary descriptors
        var own = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        var descriptor = py_get(own, "__dict__")
        if descriptor is not None and py_type(descriptor) is py_type(b.memoryview.obj):
            var fields = py_call(descriptor.__get__, model)
            if py_type(fields) is not b.dict:
                raise Error("TypeError: AutoARIMA state requires an exact dictionary")
            var keys = py_call(b.list, fields)
            for j in range(len(keys)):  # driver-loop: over passive instance keys before hash lookup
                if py_type(keys[j]) is not b.str:
                    raise Error("TypeError: AutoARIMA state keys require exact strings")
            return fields
    raise Error("TypeError: AutoARIMA requires native instance storage")


def _integer(value: PythonObject, context: PythonObject) raises -> Int:
    var cls = py_type(value)
    if cls is not py_type(PythonObject(0)) and cls is not py_type(PythonObject(0.0)) and cls is not py_type(PythonObject(True)) and not _one_of(cls, context["scalar_types"]):
        raise Error("TypeError: AutoARIMA numeric settings require native scalar values")
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _buffer(value: PythonObject, name: String, context: PythonObject) raises -> ArrayMetadata:
    classical_admit_value(value, context)
    var prepared = classical_input(value, context, name)
    return array_metadata_once(prepared, array_shape_once(prepared, context), name, context)


def autoarima_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model, context["autoarima_class"])
    var source = _buffer(args[0], "endog", context)
    var rank = len(source.shape)
    if rank != 1 and rank != 2 or source.count == 0:
        raise Error("ValueError: AutoARIMA: endog must be (batch_size, n_obs)")
    if source.dtype_number == NPY_DOUBLE:
        raise Error("TypeError: AutoARIMA: float64 is refused; pass float32 (no float64 on the device)")
    var rows = 1 if rank == 1 else source.shape[0]
    var cols = source.shape[0] if rank == 1 else source.shape[1]
    var out = source.keep
    if source.f32_c() and rank == 2:
        pass  # preserve the existing contiguous ndarray/Array storage owner
    elif source.f32_c() and not source.is_array:
        # Exact ndarray was admitted before its native reshape descriptor runs.
        out = py_call(source.keep.reshape, Python.tuple(rows, cols))
    else:
        out = classical_typed(Python.tuple(rows, cols), "<f4", context)
        var strides = List[Int]()
        strides.append(source.source.s0)
        if rank == 2:
            strides.append(source.source.s1)
        xseq_cast_nd(source.source.addr, source.shape, strides, source.source.itemsize, source.source.kind, True,
                     _F32(unsafe_from_address=Int(py=classical_address(out))))
    py_setitem(fields, "endog", args[0])
    py_setitem(fields, "_endog", out)
    py_setitem(fields, "batch_size", rows)
    py_setitem(fields, "n_obs", cols)
    py_setitem(fields, "models", Python.list())
    _ = source
    return PythonObject(None)


def _packed(value: PythonObject, name: String, context: PythonObject, integer: Bool = False) raises -> ArrayMetadata:
    var result = _buffer(value, name, context)
    if not result.c_order or (result.source.itemsize != (8 if integer else 4)):
        raise Error("TypeError: AutoARIMA fitted storage must be contiguous with its native dtype")
    if (not integer and result.dtype_number != NPY_FLOAT) or (integer and result.source.kind != CAST_KIND_SIGNED):
        raise Error("TypeError: AutoARIMA fitted storage has the wrong native dtype")
    return result^


def autoarima_forecast(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model, context["autoarima_class"])
    if args[1] is not None:
        raise Error("NotImplementedError: AutoARIMA.forecast: prediction intervals (level) are not implemented")
    var steps = _integer(args[0], context)
    if steps < 1:
        raise Error("ValueError: mojolearn ARIMA: steps must be >= 1")
    var fitted = py_get(fields, "_fitted")
    var ids_all = py_get(fields, "_ids")
    var list_type = py_type(Python.list())
    if fitted is None:
        raise Error("RuntimeError: AutoARIMA: call fit() first")
    if py_type(fitted) is not list_type or py_type(ids_all) is not list_type or len(fitted) != len(ids_all):
        raise Error("TypeError: AutoARIMA group state requires matching native lists")
    var batch = _integer(fields["batch_size"], context)
    if batch <= 0:
        raise Error("ValueError: AutoARIMA batch_size must be positive")
    var out = classical_typed(Python.tuple(batch, steps), "<f4", context)
    var output_address = Int(py=classical_address(out))
    for group in range(len(fitted)):  # driver-loop: over selected order groups, one compiled forecast each
        if fitted[group] is None:
            raise Error("RuntimeError: AutoARIMA: call fit() first")
        var state = _fields(fitted[group], context["arima_class"])
        var order = state["order"]
        var seasonal = state["seasonal_order"]
        classical_admit_value(Python.tuple(order, seasonal), context)
        if py_type(order) is not py_type(Python.tuple()) or len(order) != 3 or py_type(seasonal) is not py_type(Python.tuple()) or len(seasonal) != 4:
            raise Error("TypeError: AutoARIMA fitted orders require native tuples")
        var rows = _integer(state["batch_size_"], context)
        var n_obs = _integer(state["n_obs_"], context)
        var k = _integer(state["k_"], context)
        var n_exog = _integer(state["n_exog_"], context)
        if rows <= 0 or n_obs <= 0 or n_exog != 0 or k < 0 or k > 1:
            raise Error("ValueError: AutoARIMA fitted dimensions are invalid")
        var y = _packed(state["_y"], "endog", context)
        var params = _packed(state["params_"], "params", context)
        var ids = _packed(ids_all[group], "group indices", context, True)
        var words = Python.list(rows, n_obs, steps)
        var complexity = k + 1
        for axis in range(3):  # small-loop(ARIMA order tuple)
            var word = _integer(order[axis], context)
            if word < 0:
                raise Error("ValueError: AutoARIMA order must be nonnegative")
            if axis != 1:
                complexity += word
            _ = py_call(words.append, word)
        for axis in range(4):  # small-loop(seasonal order tuple)
            var word = _integer(seasonal[axis], context)
            if word < 0:
                raise Error("ValueError: AutoARIMA seasonal order must be nonnegative")
            if axis == 0 or axis == 2:
                complexity += word
            _ = py_call(words.append, word)
        if y.count != rows * n_obs or params.count != rows * complexity or ids.count != rows:
            raise Error("ValueError: AutoARIMA fitted buffer dimensions disagree")
        _ = py_call(words.extend, Python.tuple(k, 0, 0))
        var mode = py_get(state, "numeric_mode")
        if mode is None:
            mode = context["backend_fields"]["_DEFAULT_MODE"]
        if py_type(mode) is not py_type(PythonObject("")):
            raise Error("TypeError: numeric_mode must be an exact string")
        mode = py_call(py_call(mode.strip).lower)
        if String(mode) != "identical" and String(mode) != "fast":
            raise Error("ValueError: numeric_mode must be identical or fast")
        var entries = context["arima_bindings"][mode]
        if py_type(entries) is py_type(PythonObject("")):
            raise Error(String(entries))
        var got = Int(py=py_call_typed(entries["arima_numeric_mode"]))
        if got != (1 if String(mode) == "identical" else 0):
            raise Error("RuntimeError: ARIMA binary numeric mode disagrees with its selected tier")
        var part = classical_typed(Python.tuple(rows, steps), "<f4", context)
        var part_address = Int(py=classical_address(part))
        # n_exog is zero: y supplies a live unused pointer for the two exog slots.
        _ = py_call_typed(entries["arima_forecast"], y.source.addr, y.source.addr, y.source.addr,
                          params.source.addr, part_address, words)
        # The connected base scatter owns bounds validation and finished-result
        # assembly on its compiled target; no Python indexing or data loops.
        _ = py_call_typed(context["base_scatter_rows_bytes"], part_address, ids.source.addr, rows,
                          4 * steps, output_address, batch)
        _ = y
        _ = params
        _ = ids
        _ = part
    return out
