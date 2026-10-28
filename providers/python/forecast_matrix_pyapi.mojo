# SPDX-License-Identifier: Apache-2.0
# Complete VAR/STL public owners. Numerical buffers are captured once; fixed
# argument/result metadata stays here and existing compiled arithmetic owns data.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields, _class_fields, _truth
from bindings.sequential_admission import _one_of
from providers.python.arima_buffer_pyapi import ArimaBuffer, arima_buffer
from providers.python.autoarima_lifecycle_pyapi import _integer, _number
from providers.python.forecast_initialize_pyapi import _scalar
from providers.python.forecast_owner_pyapi import forecast_commit
from bindings.xseq_native_common import xseq_zeros, xseq_addr, xseq_f32p, xseq_entries, xseq_first
from core.xseq_marshal import xseq_cast_nd, xseq_var_blocks
from core.cast_in import NPY_DOUBLE, NPY_FLOAT, CAST_KIND_FLOAT


def _state(model: PythonObject, key: String, context: PythonObject) raises -> PythonObject:
    if py_type(model) is not context[key]:
        raise Error("TypeError: forecast operation requires its exact connected native class")
    return _fields(model)


def _new_result(key: String, context: PythonObject) raises -> PythonObject:
    var cls = context[key]
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: forecast results require an ordinary native type")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var storage = False
    for i in range(len(ancestry)):  # driver-loop: result class lifecycle and descriptor metadata before allocation
        var own = _class_fields(ancestry[i])
        if py_contains(own, "__del__"):
            raise Error("TypeError: native forecast results cannot invoke Python finalizers")
        if not storage and py_contains(own, "__dict__"):
            var descriptor = own["__dict__"]
            if py_type(descriptor) is not py_type(b.type.__dict__["__dict__"]) or descriptor.__objclass__ is not ancestry[i]:
                raise Error("TypeError: forecast results require ordinary instance storage")
            storage = True
    return py_call(b.object.__new__, cls)


def _matrix(value: PythonObject, name: String, context: PythonObject, reject_double: Bool = True) raises -> ArimaBuffer:
    var refusal = String("TypeError: ") + name + ": float64 is refused; pass float32"
    var source = arima_buffer(value, name, context["xseq_lifecycle_context"], False, refusal if reject_double else String())
    if len(source.shape) > 2:
        raise Error("ValueError: " + name + " requires a vector or matrix")
    if reject_double and source.dtype_number == NPY_DOUBLE:
        raise Error("TypeError: " + name + ": float64 is refused; pass float32")
    if not source.f32_c() or py_type(source.keep) is not context["ndarray"]:
        var shape = Python.tuple(source.shape[0]) if len(source.shape) == 1 else Python.tuple(source.shape[0], source.shape[1])
        var result = xseq_zeros(shape, context)
        xseq_cast_nd(source.address, source.shape, source.strides, source.itemsize, source.kind, False, xseq_f32p(result))
        source.keep = result
        source.address = Int(py=xseq_addr(result))
        source.itemsize = 4
        source.kind = CAST_KIND_FLOAT
        source.dtype_number = NPY_FLOAT
        source.c_order = True
    return source^


def _mode(value: PythonObject) raises:
    if value is not None and py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode requires an exact string or None")


def _var_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_var_class", context)
    if args[1] is not None:
        raise Error("NotImplementedError: VAR: exog is not implemented")
    _mode(args[2])
    var y = _matrix(args[0], "VAR", context)
    if len(y.shape) != 2 or y.shape[1] < 1:
        raise Error("ValueError: VAR: endog must be (n_obs, n_variables)")
    var updates = Python.dict()
    py_setitem(updates, "endog", args[0])
    py_setitem(updates, "exog", args[1])
    py_setitem(updates, "_endog", y.keep)
    py_setitem(updates, "neqs", y.shape[1])
    py_setitem(updates, "numeric_mode", args[2])
    forecast_commit(fields, updates, context["xseq_forecast_ownership"])
    return PythonObject(None)


def _var_publish(fields: PythonObject, y: ArimaBuffer, params: ArimaBuffer, sigma: PythonObject,
                 resid: ArimaBuffer, p: Int, kt: Int, mode: PythonObject, context: PythonObject, originals: PythonObject) raises:
    if len(y.shape) != 2 or len(params.shape) != 2 or len(resid.shape) != 2:
        raise Error("ValueError: VARResults requires endog, params and resid matrices")
    var n = y.shape[0]
    var k = y.shape[1]
    var m = Int(py=PythonObject(k) * p + kt)
    var rows = n - p
    if p < 0 or kt < 0 or kt > 1 or k < 1 or rows < 0 or params.shape[0] != m or params.shape[1] != k or resid.shape[0] != rows or resid.shape[1] != k:
        raise Error("ValueError: VARResults dimensions disagree with lag/trend metadata")
    var intercept = xseq_zeros(Python.tuple(k), context)
    var coefs = xseq_zeros(Python.tuple(p, k, k), context)
    var fitted = xseq_zeros(Python.tuple(rows, k), context)
    xseq_var_blocks(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=params.address),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y.address),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=resid.address),
                    xseq_f32p(intercept), xseq_f32p(coefs), xseq_f32p(fitted), p, kt, k, rows)
    var updates = Python.dict()
    py_setitem(updates, "endog", y.keep if originals is None else originals[0])
    py_setitem(updates, "params", params.keep if originals is None else originals[1])
    py_setitem(updates, "sigma_u", sigma)
    py_setitem(updates, "resid", resid.keep if originals is None else originals[3])
    py_setitem(updates, "k_ar", p)
    py_setitem(updates, "k_trend", kt)
    py_setitem(updates, "neqs", k)
    py_setitem(updates, "nobs", rows)
    py_setitem(updates, "df_resid", rows - m)
    py_setitem(updates, "intercept", intercept)
    py_setitem(updates, "coefs", coefs)
    py_setitem(updates, "fittedvalues", fitted)
    py_setitem(updates, "_numeric_mode", mode)
    forecast_commit(fields, updates, context["xseq_forecast_ownership"])


def _var_results_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_var_results", context)
    var lifecycle = context["xseq_lifecycle_context"]
    var y = _matrix(args[0], "VARResults endog", context, False)
    var params = _matrix(args[1], "VARResults params", context, False)
    var resid = _matrix(args[3], "VARResults resid", context, False)
    var p = _integer(args[4], lifecycle)
    var kt = _integer(args[5], lifecycle)
    _mode(args[6])
    _var_publish(fields, y, params, args[2], resid, p, kt, args[6], context, args)
    return PythonObject(None)


def _var_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_var_class", context)
    var lifecycle = context["xseq_lifecycle_context"]
    var string_type = py_type(PythonObject(""))
    if args[2] is not None:
        raise Error("NotImplementedError: VAR.fit: information-criterion order selection is not implemented")
    if py_type(args[1]) is not string_type or String(args[1]) != "ols":
        raise Error("ValueError: VAR.fit: method must be 'ols'")
    if py_type(args[3]) is not string_type or (String(args[3]) != "c" and String(args[3]) != "n"):
        raise Error("NotImplementedError: VAR.fit: only trends 'c' and 'n' are implemented")
    var p = _integer(args[0], lifecycle)
    var kt = 1 if String(args[3]) == "c" else 0
    var y = _matrix(fields["_endog"], "VAR", context)
    if len(y.shape) != 2 or y.shape[1] < 1 or p < 0 or p >= y.shape[0]:
        raise Error("ValueError: VAR.fit needs a matrix with 0 <= maxlags < observations")
    var mode = py_get(fields, "numeric_mode")
    _mode(mode)
    var route = Python.dict()
    py_setitem(route, "numeric_mode", mode)
    var n = y.shape[0]
    var k = y.shape[1]
    var m = Int(py=PythonObject(k) * p + kt)
    var params = xseq_zeros(Python.tuple(m, k), context)
    var sigma = xseq_zeros(Python.tuple(k, k), context)
    var resid = xseq_zeros(Python.tuple(n - p, k), context)
    var entries = xseq_entries(route, context)
    var code = Int(py=py_call_typed(entries["var_fit"], Python.list(y.address, xseq_addr(params), xseq_addr(sigma), xseq_addr(resid)), Python.list(n, k, p, kt)))
    if code:
        raise Error("LinAlgError: VAR.fit: the lagged design is rank deficient (Cholesky pivot of column " + String(code - 1) + ")")
    var result = _new_result("xseq_var_results", context)
    var packed_params = _matrix(params, "native params", context)
    var packed_resid = _matrix(resid, "native resid", context)
    _var_publish(_fields(result), y, packed_params, sigma, packed_resid, p, kt, mode, context, PythonObject(None))
    return result


def _var_forecast(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_var_results", context)
    var lifecycle = context["xseq_lifecycle_context"]
    var p = _integer(fields["k_ar"], lifecycle)
    var k = _integer(fields["neqs"], lifecycle)
    var kt = _integer(fields["k_trend"], lifecycle)
    var steps = _integer(args[1], lifecycle)
    var y = _matrix(args[0], "VARResults.forecast", context)
    var params = _matrix(fields["params"], "VARResults.params", context, False)
    if p < 0 or k < 1 or kt < 0 or kt > 1 or steps < 0 or len(y.shape) != 2 or y.shape[1] != k or y.shape[0] < p:
        raise Error("ValueError: VAR forecast requires nonnegative steps and sufficient history columns/rows")
    if len(params.shape) != 2 or params.shape[0] != Int(py=PythonObject(k) * p + kt) or params.shape[1] != k:
        raise Error("ValueError: VAR fitted parameter dimensions disagree")
    var state = Python.dict()
    var mode = fields["_numeric_mode"]
    _mode(mode)
    py_setitem(state, "numeric_mode", mode)
    var entries = xseq_entries(state, context)
    var result = xseq_zeros(Python.tuple(steps, k), context)
    var history = y.address + (y.shape[0] - p) * k * 4
    _ = py_call_typed(entries["var_forecast"], Python.list(history, params.address, xseq_addr(result)), Python.list(k, p, kt, steps))
    _ = y
    _ = params
    return result


def _stl_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_stl_class", context)
    var lifecycle = context["xseq_lifecycle_context"]
    var y = _matrix(args[0], "STL", context)
    var one = len(y.shape) == 1
    var rows = 1 if one else y.shape[0]
    var n = y.shape[0] if one else y.shape[1]
    if n < 1:
        raise Error("ValueError: STL: endog must be (n_obs,) or (batch_size, n_obs)")
    if args[1] is None:
        raise Error("ValueError: STL: period is required")
    for i in range(1, len(args)):  # driver-loop: thirteen fixed constructor setting words
        _scalar(args[i], lifecycle)
    _mode(args[12])
    var period = _integer(args[1], lifecycle)
    if period < 2:
        raise Error("ValueError: period must be a positive integer >= 2")
    var trend = args[3]
    if trend is None:
        var seasonal = _number(args[2], lifecycle)
        if seasonal == 0 or seasonal == 1.5:
            raise Error("ZeroDivisionError: STL default trend smoother has a zero denominator")
        var value = 1.5 * Float64(period) / (1.0 - 1.5 / seasonal)
        var calculated = Int(py=py_call(native_loaded_module("math").ceil, value))
        trend = PythonObject(calculated + (1 if calculated % 2 == 0 else 0))
    var low = args[4]
    if low is None:
        low = PythonObject(period + 1 + (1 if (period + 1) % 2 == 0 else 0))
    for i in range(5, 8):  # driver-loop: three fixed Loess degree metadata words
        var degree = _number(args[i], lifecycle)
        if degree != 0 and degree != 1:
            raise Error("ValueError: STL smoother degrees must be 0 or 1")
    var b = native_loaded_module("builtins")
    var updates = Python.dict()
    py_setitem(updates, "endog", args[0])
    py_setitem(updates, "_endog", py_call(y.keep.reshape, Python.tuple(rows, n)) if one else y.keep)
    py_setitem(updates, "_one", one)
    var names = Python.tuple("period", "seasonal", "trend", "low_pass", "seasonal_deg", "trend_deg", "low_pass_deg")
    var values = Python.tuple(period, _integer(args[2], lifecycle), _integer(trend, lifecycle), _integer(low, lifecycle), _integer(args[5], lifecycle), _integer(args[6], lifecycle), _integer(args[7], lifecycle))
    for i in range(len(names)):  # driver-loop: seven normalized smoother metadata fields
        py_setitem(updates, names[i], values[i])
    py_setitem(updates, "robust", Bool(py=args[8]))
    py_setitem(updates, "seasonal_jump", _integer(args[9], lifecycle))
    py_setitem(updates, "trend_jump", _integer(args[10], lifecycle))
    py_setitem(updates, "low_pass_jump", _integer(args[11], lifecycle))
    py_setitem(updates, "numeric_mode", args[12])
    forecast_commit(fields, updates, context["xseq_forecast_ownership"])
    return PythonObject(None)


def _decompose_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_decompose_result", context)
    var updates = Python.dict()
    var names = Python.tuple("observed", "seasonal", "trend", "resid", "weights")
    for i in range(len(names)):  # driver-loop: five retained result references, with no input protocols
        py_setitem(updates, names[i], args[i])
    forecast_commit(fields, updates, context["xseq_forecast_ownership"])
    return PythonObject(None)


def _stl_config(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_stl_class", context)
    var result = Python.dict()
    var names = Python.tuple("period", "seasonal", "seasonal_deg", "seasonal_jump", "trend", "trend_deg", "trend_jump", "low_pass", "low_pass_deg", "low_pass_jump", "robust")
    for i in range(len(names)):  # driver-loop: native configuration references, never descriptor callbacks
        py_setitem(result, names[i], fields[names[i]])
    return result


def _stl_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _state(model, "xseq_stl_class", context)
    var lifecycle = context["xseq_lifecycle_context"]
    _scalar(fields["robust"], lifecycle)
    _scalar(fields["_one"], lifecycle)
    var mode = py_get(fields, "numeric_mode")
    _mode(mode)
    var route = Python.dict()
    py_setitem(route, "numeric_mode", mode)
    var one = Bool(py=fields["_one"])
    var robust = Bool(py=fields["robust"])
    var inner = (2 if robust else 5) if args[0] is None else _integer(args[0], lifecycle)
    var outer = (15 if robust else 0) if args[1] is None else _integer(args[1], lifecycle)
    var y = _matrix(fields["_endog"], "STL", context)
    if len(y.shape) != 2 or y.shape[1] < 1:
        raise Error("ValueError: STL fitted input must be a nonempty series matrix")
    var batch = y.shape[0]
    var n = y.shape[1]
    var shape = Python.tuple(batch, n)
    var season = xseq_zeros(shape, context)
    var trend = xseq_zeros(shape, context)
    var weights = xseq_zeros(shape, context)
    var resid = xseq_zeros(shape, context)
    var plan = Python.list(batch, n)
    var names = Python.tuple("period", "seasonal", "trend", "low_pass")
    for i in range(len(names)):  # driver-loop: four fixed smoother length words
        _ = py_call(plan.append, _integer(fields[names[i]], lifecycle))
    var degree = _integer(fields["seasonal_deg"], lifecycle) + 2 * _integer(fields["trend_deg"], lifecycle) + 4 * _integer(fields["low_pass_deg"], lifecycle)
    _ = py_call(plan.extend, Python.tuple(degree, _integer(fields["seasonal_jump"], lifecycle), _integer(fields["trend_jump"], lifecycle), _integer(fields["low_pass_jump"], lifecycle), inner, outer))
    _ = py_call_typed(xseq_entries(route, context)["stl"], Python.list(y.address, xseq_addr(season), xseq_addr(trend), xseq_addr(weights), xseq_addr(resid)), plan)
    var result = _new_result("xseq_decompose_result", context)
    _ = _decompose_initialize(result, Python.tuple(xseq_first(y.keep, one), xseq_first(season, one), xseq_first(trend, one), xseq_first(resid, one), xseq_first(weights, one)), context)
    _ = y
    return result


def forecast_matrix(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "var_initialize":
        return _var_initialize(model, args, context)
    if operation == "var_results_initialize":
        return _var_results_initialize(model, args, context)
    if operation == "var_forecast":
        return _var_forecast(model, args, context)
    if operation == "var_fit":
        return _var_fit(model, args, context)
    if operation == "stl_initialize":
        return _stl_initialize(model, args, context)
    if operation == "stl_config":
        return _stl_config(model, context)
    if operation == "stl_fit":
        return _stl_fit(model, args, context)
    if operation == "decompose_initialize":
        return _decompose_initialize(model, args, context)
    raise Error("RuntimeError: unknown VAR/STL operation")
