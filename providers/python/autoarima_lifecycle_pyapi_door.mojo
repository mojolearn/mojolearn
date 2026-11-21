# SPDX-License-Identifier: Apache-2.0
# Complete AutoARIMA construction and forecast operation boundaries. Search and
# fitting remain separate owners; this provider never calls their Python methods.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit, dict_keys_admitted, dict_keys_admit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_input, classical_typed, classical_address
from bindings.classical_pandas_door import classical_admit_value, native_pandas_exact, native_pandas_f32
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once
from core.admitted_input_nd import admitted_cast_nd_f32
from core.cast_in import NPY_DOUBLE, NPY_FLOAT, CAST_KIND_SIGNED
from bindings.sequential_admission_door import _one_of
from providers.python.forecast_owner_pyapi_unpack import forecast_commit
from bindings.native_prepared_owner_unpack import prepared_get
from bindings.tsa_arima_resident_unpack import (
    arima_resident_hit,
    arima_resident_call,
)
from bindings.tsa_arima_resident_pack import (
    ARIMA_RESIDENT_SLOT,
    arima_resident_mint,
)
from arima.tsa_arima_resident_run import arima_resident_words
from bindings.tsa_keys_unpack import tsa_keys
from bindings.py_balanced import py_item
from bindings.tsa_arima_route_unpack import arima_device_admit, autoarima_route
from bindings.tsa_hw_route_unpack import TsaHwCall, tsa_hw_resident_handle
from core.arima_run import arima_forecast_run, arima_predict_run
from core.device_route import DeviceRoute, DEVICE_INSTALL
from providers.python.device_route_boundary_door import device_route_clear

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _REFUSE_DOUBLE = "TypeError: AutoARIMA: float64 is refused; pass float32 (no float64 on the device)"


comptime _FIELDS_ROW_SLOT = "xseq_fields_row_v1"  # p3-cnn-sequence-tsa class record


def _fields(model: PythonObject, cls: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var b = native_loaded_module("builtins")
    if py_type(model) is not cls or py_type(cls) is not b.type:
        raise Error("TypeError: AutoARIMA requires its connected native model classes")
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    # p3-cnn-sequence-tsa: the MRO row whose `__dict__` descriptor serves the
    # instance storage is kept per class under the type watcher, and the
    # instance keys are re-checked only after the dict changed (dictionary
    # watcher). Python 3.10/3.11 keep the full walk on every call.
    var mods = NativeModules()
    var kept = class_record(mods, _FIELDS_ROW_SLOT, cls)
    var start = 0
    var stop = len(mro)
    if kept is not None:
        start = Int(py=kept)
        stop = start + 1
    for i in range(start, stop):  # driver-loop: over native instance dictionary descriptors
        var own = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, mro[i])
        var descriptor = py_get(own, nk[].k___dict__)
        if descriptor is not None and py_type(descriptor) is py_type(b.memoryview.obj):
            var fields = py_call(descriptor.__get__, model)
            if py_type(fields) is not b.dict:
                raise Error("TypeError: AutoARIMA state requires an exact dictionary")
            if kept is None:
                class_admit(mods, _FIELDS_ROW_SLOT, cls, PythonObject(i))
            if dict_keys_admitted(fields):
                return fields
            var keys = py_call(b.list, fields)
            for j in range(len(keys)):  # driver-loop: over passive instance keys before hash lookup
                if py_type(keys[j]) is not b.str:
                    raise Error("TypeError: AutoARIMA state keys require exact strings")
            dict_keys_admit(fields)
            return fields
    raise Error("TypeError: AutoARIMA requires native instance storage")


def _integer_object(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var cls = py_type(value)
    var b = native_loaded_module("builtins")
    if cls is not b.int and cls is not b.float and cls is not b.bool and cls is not b.str and cls is not b.bytes and not _one_of(cls, py_item(context, nk[].k_scalar_types)):
        raise Error("TypeError: AutoARIMA numeric settings require native scalar values")
    return py_call(b.int, value)


def _integer(value: PythonObject, context: PythonObject) raises -> Int:
    return Int(py=_integer_object(value, context))


def _number(value: PythonObject, context: PythonObject) raises -> Float64:
    var nk = tsa_keys()
    var cls = py_type(value)
    if cls is not py_type(PythonObject(0)) and cls is not py_type(PythonObject(0.0)) and cls is not py_type(PythonObject(True)) and not _one_of(cls, py_item(context, nk[].k_scalar_types)):
        raise Error("TypeError: AutoARIMA numeric settings require native scalar values")
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def _buffer(value: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin) raises -> ArrayMetadata:
    classical_admit_value(value, context)
    var prepared = classical_input(value, context, name)
    return array_metadata_once(prepared, array_shape_once(prepared, context), name, context, pin)


def _series_buffer(value: PythonObject, name: String, refusal: String, context: PythonObject,
                   mut pin: ArrayPin) raises -> ArrayMetadata:
    """`_buffer` of a public series input that may be a pandas DataFrame or
    Series: its native block storage becomes one float32 copy, refused with
    `refusal` when pandas promotes it beyond float32 (float64 or object)."""
    classical_admit_value(value, context)
    if not native_pandas_exact(value):
        return _buffer(value, name, context, pin)
    var converted = native_pandas_f32(value, context, False, True, refusal)
    return _buffer(converted, name, context, pin)


def autoarima_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _fields(model, py_item(context, nk[].k_autoarima_class))
    # `device` is checked before any field changes (an unknown value, or an
    # explicit "gpu" on a CPU-only installation, is refused in words); it is
    # read into the route owner at the estimator's first operation.
    arima_device_admit(args[1], context, native_loaded_module("builtins"))
    var source_pin = ArrayPin()
    var source = _series_buffer(args[0], "endog", _REFUSE_DOUBLE, context, source_pin)
    var rank = len(source.shape)
    if rank != 1 and rank != 2 or source.count == 0:
        raise Error("ValueError: AutoARIMA: endog must be (batch_size, n_obs)")
    if source.dtype_number == NPY_DOUBLE:
        raise Error(_REFUSE_DOUBLE)
    var rows = 1 if rank == 1 else source.shape[0]
    var cols = source.shape[0] if rank == 1 else source.shape[1]
    var out = source_pin.keep
    if source.f32_c() and rank == 2:
        pass  # preserve the existing contiguous ndarray/Array storage owner
    elif source.f32_c() and not source.is_array:
        # Exact ndarray was admitted before its native reshape descriptor runs.
        out = py_call(source_pin.keep.reshape, Python.tuple(rows, cols))
    else:
        out = classical_typed(Python.tuple(rows, cols), "<f4", context)
        var strides = List[Int]()
        strides.append(source.source.s0)
        if rank == 2:
            strides.append(source.source.s1)
        _ = admitted_cast_nd_f32(source.source.addr, source.shape, strides, source.source.itemsize, source.source.kind,
                     _F32(unsafe_from_address=Int(py=classical_address(out))))
    var updates = Python.dict()
    py_setitem(updates, nk[].k_endog, args[0])
    py_setitem(updates, nk[].k__endog, out)
    py_setitem(updates, nk[].k_batch_size, rows)
    py_setitem(updates, nk[].k_n_obs, cols)
    py_setitem(updates, nk[].k_models, Python.list())
    py_setitem(updates, nk[].k_device, args[1])
    # The route of an estimator constructed again is forgotten.
    device_route_clear(model)
    forecast_commit(fields, updates, py_item(context, nk[].k_tsa_owner_context))
    _ = source
    _ = source_pin^
    return PythonObject(None)


def _packed(value: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin,
            integer: Bool = False) raises -> ArrayMetadata:
    var result = _buffer(value, name, context, pin)
    if not result.c_order or (result.source.itemsize != (8 if integer else 4)):
        raise Error("TypeError: AutoARIMA fitted storage must be contiguous with its native dtype")
    if (not integer and result.dtype_number != NPY_FLOAT) or (integer and result.source.kind != CAST_KIND_SIGNED):
        raise Error("TypeError: AutoARIMA fitted storage has the wrong native dtype")
    return result^


def autoarima_forecast(model: PythonObject, args: PythonObject, context: PythonObject, prediction: Bool = False) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _fields(model, py_item(context, nk[].k_autoarima_class))
    if (args[2] if prediction else args[1]) is not None:
        raise Error("NotImplementedError: AutoARIMA: prediction intervals (level) are not implemented")
    var observations = _integer(py_item(fields, nk[].k_n_obs), context)
    var start = _integer(args[0], context) if prediction else observations
    var end = (observations if args[1] is None else _integer(args[1], context)) if prediction else observations + _integer(args[0], context)
    var steps = end - start
    if steps < 1:
        raise Error("ValueError: AutoARIMA needs start < end and forecast steps >= 1")
    var fitted = py_get(fields, nk[].k__fitted)
    var ids_all = py_get(fields, nk[].k__ids)
    var list_type = py_type(Python.list())
    if fitted is None:
        raise Error("RuntimeError: AutoARIMA: call fit() first")
    if py_type(fitted) is not list_type or py_type(ids_all) is not list_type or len(fitted) != len(ids_all):
        raise Error("TypeError: AutoARIMA group state requires matching native lists")
    var batch = _integer(py_item(fields, nk[].k_batch_size), context)
    if batch <= 0:
        raise Error("ValueError: AutoARIMA batch_size must be positive")
    var out = classical_typed(Python.tuple(batch, steps), "<f4", context)
    var output_address = Int(py=classical_address(out))
    # The AutoARIMA's one route (its device, the default numeric mode its
    # per-group models carry), resolved where the former per-group entry
    # selection first ran and reused by every group.
    var call = TsaHwCall(DeviceRoute(DEVICE_INSTALL), PythonObject(None))
    var routed = False
    for group in range(len(fitted)):  # driver-loop: over selected order groups, one compiled forecast each
        if fitted[group] is None:
            raise Error("RuntimeError: AutoARIMA: call fit() first")
        var state = _fields(fitted[group], py_item(context, nk[].k_arima_class))
        var order = py_item(state, nk[].k_order)
        var seasonal = py_item(state, nk[].k_seasonal_order)
        classical_admit_value(Python.tuple(order, seasonal), context)
        if py_type(order) is not py_type(Python.tuple()) or len(order) != 3 or py_type(seasonal) is not py_type(Python.tuple()) or len(seasonal) != 4:
            raise Error("TypeError: AutoARIMA fitted orders require native tuples")
        var rows = _integer(py_item(state, nk[].k_batch_size_), context)
        var n_obs = _integer(py_item(state, nk[].k_n_obs_), context)
        var k = _integer(py_item(state, nk[].k_k_), context)
        var n_exog = _integer(py_item(state, nk[].k_n_exog_), context)
        if rows <= 0 or n_obs <= 0 or n_exog != 0 or k < 0 or k > 1:
            raise Error("ValueError: AutoARIMA fitted dimensions are invalid")
        var y_pin = ArrayPin()
        var params_pin = ArrayPin()
        var ids_pin = ArrayPin()
        var y = _packed(py_item(state, nk[].k__y), "endog", context, y_pin)
        var params = _packed(py_item(state, nk[].k_params_), "params", context, params_pin)
        var ids = _packed(ids_all[group], "group indices", context, ids_pin, True)
        var complexity = k + 1
        var order_words = List[Int]()
        for axis in range(3):  # small-loop(ARIMA order tuple)
            var word = _integer(order[axis], context)
            if word < 0:
                raise Error("ValueError: AutoARIMA order must be nonnegative")
            if axis != 1:
                complexity += word
            order_words.append(word)
        for axis in range(4):  # small-loop(seasonal order tuple)
            var word = _integer(seasonal[axis], context)
            if word < 0:
                raise Error("ValueError: AutoARIMA seasonal order must be nonnegative")
            if axis == 0 or axis == 2:
                complexity += word
            order_words.append(word)
        if y.count != rows * n_obs or params.count != rows * complexity or ids.count != rows:
            raise Error("ValueError: AutoARIMA fitted buffer dimensions disagree")
        if not routed:
            call = autoarima_route(model, fields, context, native_loaded_module("builtins"))
            routed = True
        var part = classical_typed(Python.tuple(rows, steps), "<f4", context)
        var part_address = Int(py=classical_address(part))
        # The group model's resident fitted state (lane ne1-a-decomp-tsa,
        # N029): its y and parameters are uploaded once per fit, the record
        # shared with ARIMA.predict (bindings/tsa_arima_resident.mojo); a
        # binding without the resident entries keeps the per-call upload.
        var member = fitted[group]
        var resident_words = arima_resident_words(rows, n_obs, order_words, k, 0)
        var resident_mods = NativeModules()
        var record = prepared_get(resident_mods, ARIMA_RESIDENT_SLOT, member)
        var handle = -1
        if record is not None:
            handle = arima_resident_hit(record, state, call, resident_words)
        if handle < 0:
            record = arima_resident_mint(member, state, call, resident_words, y.source.addr, y.source.addr,
                                         params.source.addr)
            if record is not None:
                handle = tsa_hw_resident_handle(record[3])
        if handle >= 0:
            # n_exog is zero: the output supplies a live unused future pointer.
            arima_resident_call(call, handle, part_address, part_address, start, end, steps, not prediction)
        else:
            # n_exog is zero: y supplies a live unused pointer for the two exog slots.
            var route = call.route
            var y_address = y.source.addr
            var params_address = params.source.addr
            var py = Python()
            try:
                with GILReleased(py):
                    if prediction:
                        _ = arima_predict_run(route, y_address, y_address, y_address, params_address, part_address,
                                              rows, n_obs, start, end, order_words, k, 0)
                    else:
                        _ = arima_forecast_run(route, y_address, y_address, y_address, params_address, part_address,
                                               rows, n_obs, steps, order_words, k, 0)
            finally:
                _ = part
                _ = call.owner  # the route owner pins the image through the call
        _ = record  # the record (and its lifetime owner) outlive the query
        # The connected base scatter owns bounds validation and finished-result
        # assembly on its compiled target; no Python indexing or data loops.
        _ = py_call_typed(py_item(context, nk[].k_base_scatter_rows_bytes), part_address, ids.source.addr, rows,
                          4 * steps, output_address, batch)
        _ = y
        _ = params
        _ = ids
        _ = y_pin^  # the door's pins hold the fitted words through every RUN read
        _ = params_pin^
        _ = ids_pin^
        _ = part
    return out


def autoarima_predict_native(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return autoarima_forecast(model, args, context, True)
