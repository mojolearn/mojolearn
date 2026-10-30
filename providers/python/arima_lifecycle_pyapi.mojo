# SPDX-License-Identifier: Apache-2.0
# ARIMA construction and parameter publication. Only fixed order/shape metadata
# is traversed here; parameter blocks move through the compiled strided copy.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields, _truth
from bindings.sequential_admission import _one_of
from providers.python.autoarima_lifecycle_pyapi import _integer_object, _integer, _buffer, _packed
from providers.python.forecast_owner_pyapi import forecast_commit
from bindings.classical_arrays import classical_address
from bindings.tsa_hw_native import tsa_array
from core.native_scalar_decode import native_scalar_real
from core.cast_in import CAST_KIND_BOOL, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED


def arima_order(value: PythonObject, width: Int, name: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var values = Python.list()
    var normalized = True
    if cls is b.tuple or cls is b.list or cls is b.range or cls is context["array_type"]:
        if len(value) != width:
            raise Error("ValueError: mojolearn ARIMA: " + name + " must have " + String(width) + " entries")
        for i in range(width):  # driver-loop: the three or four order metadata words
            var item = value[i]
            var kind = py_type(item)
            if kind is not b.int and kind is not b.bool and kind is not b.float and not _one_of(kind, context["scalar_types"]):
                raise Error("TypeError: ARIMA orders require native numeric scalars")
            var word = _integer_object(item, context)
            if Bool(py=word != item) or Bool(py=word < 0):
                raise Error("ValueError: mojolearn ARIMA: " + name + " entries must be non-negative integers")
            normalized = normalized and kind is b.int
            _ = py_call(values.append, word)
        if cls is b.tuple and normalized:
            return value
    else:
        var source = _buffer(value, name, context)
        if len(source.shape) != 1 or source.count != width:
            raise Error("ValueError: mojolearn ARIMA: " + name + " must have " + String(width) + " entries")
        var code = String("?") if source.source.kind == CAST_KIND_BOOL else String("q") if source.source.kind == CAST_KIND_SIGNED else String("Q") if source.source.kind == CAST_KIND_UNSIGNED else String("e") if source.source.itemsize == 2 else String("f") if source.source.itemsize == 4 else String("d")
        for i in range(width):  # driver-loop: exact decoding of bounded order metadata, never series data
            var decoded = native_scalar_real(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.source.addr), i * source.source.s0, code, source.source.itemsize, True)
            var integral = decoded.special == 0 and (not decoded.negative or not decoded.truth())
            for bit in range(128):  # driver-loop: at most two scalar significand limbs
                if bit + decoded.scale < 0 and decoded.bit(bit) != 0:
                    integral = False
            if not integral:
                raise Error("ValueError: mojolearn ARIMA: " + name + " entries must be non-negative integers")
            _ = py_call(values.append, py_call(b.int, decoded.integer_text(), 2))
        _ = source
    return py_call(b.tuple, values)


def arima_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var b = native_loaded_module("builtins")
    var order = arima_order(args[0], 3, "order", context)
    var seasonal = arima_order(args[1], 4, "seasonal_order", context)
    var trend = args[2]
    var intercept = 0
    if trend is None:
        intercept = 0 if Bool(py=order[1] + seasonal[1]) else 1
    elif py_type(trend) is b.str:
        var name = String(py=py_call(trend.lower))
        if name == "c":
            intercept = 1
        elif name != "n":
            raise Error("NotImplementedError: mojolearn ARIMA: trend is refused; use None, 'n' or 'c'. Pass time or polynomial trends as exog regressors")
    else:
        raise Error("NotImplementedError: mojolearn ARIMA: trend must be None, 'n' or 'c'; pass polynomial trends as exog regressors")
    var method = args[3]
    if py_type(method) is not b.str:
        raise Error("ValueError: mojolearn ARIMA: method is a name")
    var lowered = py_call(method.lower)
    if not Bool(py=method == lowered):
        method = lowered
    var method_name = String(py=method)
    if method_name != "ml" and method_name != "css" and method_name != "css-ml":
        raise Error("ValueError: mojolearn ARIMA: unknown method; cuML's three are ['css', 'css-ml', 'ml'] and this implementation offers 'ml'")
    var maxiter = _integer_object(args[4], context)
    if (Bool(py=args[5]) if _one_of(py_type(args[5]), context["scalar_types"]) else _truth(args[5])):
        raise Error("NotImplementedError: mojolearn ARIMA: verbose is refused; this implementation prints no log lines")
    if args[6] is not None:
        raise Error("NotImplementedError: mojolearn ARIMA: output_type is a cuML-internal selector; this package returns mojolearn.Array")
    var updates = Python.dict()
    py_setitem(updates, "order", order)
    py_setitem(updates, "seasonal_order", seasonal)
    py_setitem(updates, "trend", trend)
    py_setitem(updates, "k_", intercept)
    py_setitem(updates, "method", method)
    py_setitem(updates, "maxiter", maxiter)
    py_setitem(updates, "verbose", False)
    py_setitem(updates, "output_type", PythonObject(None))
    py_setitem(updates, "n_exog_", 0)
    # Kernel order bounds remain the kernel owner's decision. Arbitrary-size
    # builtin integers here preserve constructor/clone metadata unchanged.
    py_setitem(updates, "complexity_", order[0] + order[2] + seasonal[0] + seasonal[2] + intercept + 1)
    # The constructor's own numeric_mode (formerly added by the `_mode`
    # NumericModeMixin wrapper): stored as given, checked when fit selects
    # the binary (bindings/tsa_arima_native.mojo).
    py_setitem(updates, "numeric_mode", args[7])
    forecast_commit(fields, updates, context["tsa_owner_context"])
    return PythonObject(None)


def arima_parameter(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var b = native_loaded_module("builtins")
    if py_type(args[0]) is not b.str:
        raise Error("TypeError: ARIMA parameter name must be an exact string")
    var name = String(py=args[0])
    var intercept = _integer(fields["k_"], context)
    var regressors = _integer(py_get(fields, "n_exog_", PythonObject(0)), context)
    if name == "mu_" and intercept == 0:
        raise Error("AttributeError: mojolearn ARIMA: there is no mu_ because this model has no intercept; construct with trend='c'")
    if name == "beta_" and regressors == 0:
        raise Error("AttributeError: mojolearn ARIMA: there is no beta_ because this model was fit without exogenous regressors; pass exog to fit")
    if not py_contains(fields, "params_"):
        raise Error("AttributeError: mojolearn ARIMA: call fit() before " + name)
    var order = arima_order(fields["order"], 3, "order", context)
    var seasonal = arima_order(fields["seasonal_order"], 4, "seasonal_order", context)
    var offset = intercept + regressors
    var width = 0
    var flatten = False
    if name == "mu_":
        offset = 0
        width = 1
        flatten = True
    elif name == "beta_":
        offset = intercept
        width = regressors
    elif name == "ar_":
        width = Int(py=order[0])
    elif name == "ma_":
        offset += Int(py=order[0])
        width = Int(py=order[2])
    elif name == "sar_":
        offset += Int(py=order[0]) + Int(py=order[2])
        width = Int(py=seasonal[0])
    elif name == "sma_":
        offset += Int(py=order[0]) + Int(py=order[2]) + Int(py=seasonal[0])
        width = Int(py=seasonal[2])
    elif name == "sigma2_":
        offset = _integer(fields["complexity_"], context) - 1
        width = 1
        flatten = True
    else:
        raise Error("AttributeError: unknown ARIMA parameter " + name)
    var source = _packed(fields["params_"], "params", context)
    if len(source.shape) != 2:
        raise Error("ValueError: ARIMA params must be a fitted parameter matrix")
    var rows = source.shape[0]
    var stride = source.shape[1]
    if offset < 0 or width < 0 or offset + width > stride:
        raise Error("ValueError: ARIMA parameter metadata exceeds fitted storage")
    var result = tsa_array("f", "<f4", Python.tuple(rows) if flatten else Python.tuple(rows, width), rows * width, 4, context)
    if rows * width > 0:
        var dims = List[Int64]()
        dims.append(Int64(rows))
        dims.append(Int64(width))
        dims.append(Int64(stride))
        dims.append(1)
        dims.append(Int64(width))
        dims.append(1)
        dims.append(Int64(offset))
        dims.append(0)
        _ = py_call_typed(context["base_strided_copy_bytes"], source.source.addr, classical_address(result), Int(dims.unsafe_ptr()), 2, 4)
        # The integer address does not retain the native dimension buffer.
        # Keep its owner through the indirect compiled copy, as array_native's
        # strided-copy boundary does for its metadata buffer. The installed
        # 47508 CPU ARIMA fit returned successfully, but parameter extraction
        # crashed in this path; retained public/direct-entry reproductions:
        # ~/mojolearn-evidence/arima-cpu-repair-20261029/.
        _ = dims
    _ = source
    return result


def arima_repr(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var b = native_loaded_module("builtins")
    var order = arima_order(fields["order"], 3, "order", context)
    var seasonal = arima_order(fields["seasonal_order"], 4, "seasonal_order", context)
    var trend = fields["trend"]
    var method = fields["method"]
    if (trend is not None and py_type(trend) is not b.str) or py_type(method) is not b.str:
        raise Error("TypeError: ARIMA repr requires passive native trend/method metadata")
    return PythonObject("ARIMA(order=" + String(py=py_call(b.repr, order)) + ", seasonal_order=" + String(py=py_call(b.repr, seasonal)) + ", trend=" + String(py=py_call(b.repr, trend)) + ", method=" + String(py=py_call(b.repr, method)) + ")")
