# SPDX-License-Identifier: Apache-2.0
# The native owner of QNRegressor's public fit, predict and score. Each is one
# call into qn_c, which converts and checks the inputs, selects the numeric-mode
# binding, runs the existing qn_fit / qn_decision_function entries (the same
# arguments the Python shell passed, so the same bits) and stores the fitted
# attributes. score predicts and scores in the same call: the prediction stays
# a native buffer handed straight to the R^2 program.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_host_f32
from bindings.classical_validate import classical_view, classical_check_weights_input
from bindings.classical_score import classical_r2_score

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def _qn_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The estimator's numeric mode (None: the process mode), its connected
    # estimators entries, checked against the selected vendor and mode.
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["qn_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    if String(entries["estimators_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: QN native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: QN native mode differs from selected execution")
    return entries


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(Python.import_module("builtins").int, value))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(Python.import_module("builtins").float, value))


def qn_regressor_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var entries = _qn_entries(fields, context)
    var x = classical_host_f32(args[0], 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var view = classical_view(args[1], context, "y")
    if Int(py=view.ndim) != 1:
        raise Error("mojolearn QNRegressor requires a 1-D y")
    var target = classical_host_f32(view, 1, "y", context)
    if Int(py=target.shape[0]) != rows:
        raise Error("mojolearn QNRegressor X and y lengths differ")
    var weights = classical_check_weights_input(args[2], rows, context)
    var tol = _float(fields["tol"])
    var delta = fields["delta"]
    var change_tol = _float(delta) if delta is not None else tol * 0.01
    var loss = py_get(context["qn_losses"], fields["loss"])
    if loss is None:
        raise Error("mojolearn QNRegressor: unknown loss " + String(py_call(Python.import_module("builtins").repr, fields["loss"])))
    var intercept = Bool(py=fields["fit_intercept"])
    var w = classical_typed(Python.tuple(cols + (1 if intercept else 0)), "<f4", context["output_context"])
    var info = classical_typed(Python.tuple(2), "<f4", context["output_context"])
    var has_sw = weights is not None
    var params = Python.list(rows, cols, 1, _float(fields["l1_strength"]), _float(fields["l2_strength"]),
                             tol, change_tol, _int(fields["max_iter"]), _int(fields["linesearch_max_iter"]),
                             _int(fields["lbfgs_memory"]), 1 if intercept else 0,
                             1 if Bool(py=fields["penalty_normalized"]) else 0, 1 if has_sw else 0,
                             _int(loss), Float64(0.0))
    if has_sw:
        _ = py_call(params.append, classical_address(weights))
    var n_iter = py_call_typed(entries["qn_fit"], classical_address(x), classical_address(target),
                               classical_address(w), classical_address(info), params)
    var coef = classical_typed(Python.tuple(cols), "<f4", context["output_context"])
    var source = F32P(unsafe_from_address=Int(py=classical_address(w)))
    var dest = F32P(unsafe_from_address=Int(py=classical_address(coef)))
    for j in range(cols):  # driver-loop: over the fitted coefficients (model structure)
        dest[j] = source[j]
    var iterations = classical_typed(Python.tuple(1), "<i8", context["output_context"])
    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(iterations)))[0] = Int64(_int(n_iter))
    var words = F32P(unsafe_from_address=Int(py=classical_address(info)))
    py_setitem(fields, "input_copied_", PythonObject(not (x is args[0])))
    py_setitem(fields, "_w", w)
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", PythonObject(Float64(source[cols])) if intercept else PythonObject(0.0))
    py_setitem(fields, "n_iter_", iterations)
    py_setitem(fields, "objective_", PythonObject(Float64(words[0])))
    py_setitem(fields, "retcode_", PythonObject(Int(words[1])))
    py_setitem(fields, "n_features_in_", cols)
    _ = x^
    _ = target^
    _ = weights^
    _ = info^
    return model


def qn_regressor_predict(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "_w"):
        raise Error("mojolearn QNRegressor: call fit before predict")
    var entries = _qn_entries(fields, context)
    var x = classical_host_f32(source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if cols != Int(py=fields["n_features_in_"]):
        raise Error("mojolearn QNRegressor feature count differs from fit")
    var out = classical_typed(Python.tuple(rows), "<f4", context["output_context"])
    _ = py_call_typed(entries["qn_decision_function"], classical_address(x), classical_address(fields["_w"]),
                      classical_address(out), Python.list(rows, cols, 1 if Bool(py=fields["fit_intercept"]) else 0))
    _ = x^
    return out


def qn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if operation == "fit":
            result = qn_regressor_fit(model, args, context)
        elif operation == "predict":
            result = qn_regressor_predict(model, args[0], context)
        elif operation == "score":
            # The prediction is the R^2 program's input as it is: no
            # publication and no second conversion between the two.
            result = classical_r2_score(qn_regressor_predict(model, args[0], context), args[1], context)
        else:
            raise Error("unknown native QN operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
