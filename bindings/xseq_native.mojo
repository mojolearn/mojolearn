# SPDX-License-Identifier: Apache-2.0
"""`xseq_native_call(model, operation, args, context)`: the one native call
of every migrated public operation of the sequence-expansion estimators
(python/mojolearn/_native_xseq_bridge.py connects it at import).

Operations: rnn_fit, rnn_predict, rnn_predict_proba, rnn_hidden_sequence,
rnn_score and mlp_score (the regressors' R^2), rnn_classifier_score and
mlp_classifier_score (the classifiers' accuracy; bindings/xseq_score_native.mojo)
(bindings/xseq_neural_native.mojo, the recurrent models); mlp_fit,
mlp_predict, mlp_predict_proba (the perceptrons); theta_*, croston_*, ets_*,
garch_fit, var_fit, stl_fit (bindings/xseq_forecast_native.mojo).
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.xseq_neural_native import xseq_rnn, xseq_mlp
from bindings.xseq_forecast_native import xseq_forecast
from bindings.xseq_score_native import xseq_regressor_score, xseq_classifier_score
from bindings.xseq_prophet_native import prophet_fit, prophet_predict
from providers.python.prophet_lifecycle_pyapi import prophet_initialize, prophet_add_seasonality
from bindings.py_balanced import py_fields
from providers.python.xseq_public_lifecycle_pyapi import XseqPublicLifecycle
from providers.python.recurrent_state_pyapi import recurrent_state
from providers.python.estimator_lifecycle_pyapi import estimator_fit_begin, estimator_fit_end


def _xseq_raise(error: Error, args_ptr: PyObjectPtr) -> PyObjectPtr:
    """The Python exception a native refusal names in its prefix."""
    var text = String(error)
    if text.startswith("NotFittedError:"):
        try:
            ref cpy = Python().cpython()
            var lifecycle = causal_borrowed_item(cpy.PyTuple_GetItem(args_ptr, 3), "xseq_lifecycle_context")
            var cls = causal_borrowed_item(lifecycle, "active_not_fitted_class")
            return causal_set_exception(error_without_class(error, "NotFittedError"), cls)
        except nested:
            return raise_python_exception(nested)
    if text.startswith("LinAlgError:"):
        try:
            ref cpy = Python().cpython()
            # The active ABI args tuple owns the context and the class
            # throughout PyErr_SetString; borrowed without transfer.
            var cls = causal_borrowed_item(cpy.PyTuple_GetItem(args_ptr, 3), "xseq_linalg_error")
            return causal_set_exception(error_without_class(error, "LinAlgError"), cls)
        except nested:
            return raise_python_exception(nested)
    if text.startswith("TypeError:"):
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
    if text.startswith("NotImplementedError:"):
        return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
    if text.startswith("RuntimeError:"):
        return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
    if text.startswith("ImportError:"):
        return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
    if text.startswith("IndexError:"):
        return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
    if text.startswith("AttributeError:"):
        return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
    if text.startswith("OverflowError:"):
        return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
    if text.startswith("ZeroDivisionError:"):
        return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
    return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


def _xseq_execute(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var result: PythonObject
    if operation == "rnn_score" or operation == "mlp_score":
        # predict, then R^2 against y, in this one call
        var predict_args = Python.tuple(args[0])
        var prediction = (xseq_rnn(model, "rnn_predict", predict_args, context) if operation == "rnn_score"
                          else xseq_mlp(model, "mlp_predict", predict_args, context))
        result = xseq_regressor_score(model, prediction, args, context)
    elif operation == "rnn_classifier_score" or operation == "mlp_classifier_score":
        # `ClassifierScoreMixin.score`: predict, then the accuracy of it
        # against y, in this one call
        var predict_args = Python.tuple(args[0])
        var prediction = (xseq_rnn(model, "rnn_predict", predict_args, context) if operation == "rnn_classifier_score"
                          else xseq_mlp(model, "mlp_predict", predict_args, context))
        result = xseq_classifier_score(model, prediction, args, context)
    elif operation == "prophet_initialize":
        result = prophet_initialize(model, args, context)
    elif operation == "prophet_add_seasonality":
        result = prophet_add_seasonality(model, args, context)
    elif operation == "prophet_fit":
        result = prophet_fit(model, args, context)
    elif operation == "prophet_predict":
        result = prophet_predict(model, args, context)
    elif operation.startswith("rnn_"):
        result = xseq_rnn(model, operation, args, context)
    elif operation.startswith("mlp_"):
        result = xseq_mlp(model, operation, args, context)
    else:
        result = xseq_forecast(model, operation, args, context)
    return result


def xseq_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(py=call[1])
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        if operation == "rnn_state_dict" or operation == "rnn_load_state_dict":
            result = recurrent_state(model, operation, args, context)
        elif (operation.startswith("rnn_") or operation.startswith("mlp_")) and operation != "rnn_hidden_sequence":
            var lifecycle = XseqPublicLifecycle(model, operation, args, context)
            if lifecycle.fitting:
                var depth = estimator_fit_begin(py_fields(model))
                try:
                    result = _xseq_execute(model, operation, lifecycle.arguments, context)
                except problem:
                    raise lifecycle.refusal(problem)
                finally:
                    estimator_fit_end(model, depth)
            else:
                result = _xseq_execute(model, operation, lifecycle.arguments, context)
            result = lifecycle.finish(model, result, context)
        else:
            result = _xseq_execute(model, operation, args, context)
        return result.steal_data()
    except error:
        return _xseq_raise(error, args_ptr)
