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
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.xseq_neural_native_door import xseq_rnn, xseq_mlp
from bindings.xseq_forecast_native_door import xseq_forecast
from bindings.xseq_score_native_door import xseq_regressor_score, xseq_classifier_score
from bindings.xseq_prophet_native_door import prophet_fit, prophet_predict
from providers.python.prophet_lifecycle_pyapi_pack import prophet_initialize, prophet_add_seasonality
from bindings.py_balanced import py_fields
from providers.python.xseq_public_lifecycle_pyapi_door import XseqPublicLifecycle
from providers.python.recurrent_state_pyapi_door import recurrent_state
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_fit_begin, estimator_fit_end
from bindings.py_keys import py_op_code
from bindings.xseq_keys_unpack import (
    xseq_keys, XSEQ_OPN_RNN_STATE_DICT, XSEQ_OPN_RNN_LOAD_STATE_DICT, XSEQ_OPN_RNN_FIT, XSEQ_OPN_RNN_PREDICT,
    XSEQ_OPN_RNN_SCORE, XSEQ_OPN_RNN_CLASSIFIER_SCORE, XSEQ_OPN_MLP_FIT, XSEQ_OPN_MLP_PREDICT, XSEQ_OPN_MLP_SCORE,
    XSEQ_OPN_MLP_CLASSIFIER_SCORE, XSEQ_OPN_PROPHET_INITIALIZE, XSEQ_OPN_PROPHET_ADD_SEASONALITY,
    XSEQ_OPN_PROPHET_FIT, XSEQ_OPN_PROPHET_PREDICT,
)


def _xseq_raise(error: Error, args_ptr: PyObjectPtr) -> PyObjectPtr:
    """The Python exception a native refusal names in its prefix."""
    var text = String(error)
    if text == PYTHON_ERROR_PENDING:
        # A Python exception the natively called metrics owner left set (its warnings, its admission).
        return optimizer_python_error(error)
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


# bw2-sequence (2026-11-20): the owner of an operation code is a range test
# (codes are numbered by owner: bindings/xseq_keys_unpack.mojo).
comptime _XSEQ_STATE = 0
comptime _XSEQ_RNN = 1
comptime _XSEQ_MLP = 2
comptime _XSEQ_PROPHET = 3
comptime _XSEQ_FORECAST = 4


def _xseq_owner(operation: Int, name: PythonObject) raises -> Int:
    """The owner of an operation code. Only a name that is not an operation
    (-1) builds its String, for the prefix route it took before: an unknown
    `rnn_`/`mlp_` name runs that owner's public lifecycle, any other goes to
    the forecasters' unknown-operation path (a name that is not a str raises
    here, where its String copy raised)."""
    if operation >= XSEQ_OPN_RNN_STATE_DICT and operation <= XSEQ_OPN_RNN_LOAD_STATE_DICT:
        return _XSEQ_STATE
    if operation >= XSEQ_OPN_RNN_FIT and operation <= XSEQ_OPN_RNN_CLASSIFIER_SCORE:
        return _XSEQ_RNN
    if operation >= XSEQ_OPN_MLP_FIT and operation <= XSEQ_OPN_MLP_CLASSIFIER_SCORE:
        return _XSEQ_MLP
    if operation >= XSEQ_OPN_PROPHET_INITIALIZE and operation <= XSEQ_OPN_PROPHET_PREDICT:
        return _XSEQ_PROPHET
    if operation >= 0:
        return _XSEQ_FORECAST
    var text = String(py=name)
    if text.startswith("rnn_"):
        return _XSEQ_RNN
    if text.startswith("mlp_"):
        return _XSEQ_MLP
    return _XSEQ_FORECAST


def _xseq_execute(model: PythonObject, operation: Int, owner: Int, name: PythonObject, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    var result: PythonObject
    if operation == XSEQ_OPN_RNN_SCORE or operation == XSEQ_OPN_MLP_SCORE:
        # predict, then R^2 against y, in this one call
        var predict_args = Python.tuple(args[0])
        var prediction = (xseq_rnn(model, XSEQ_OPN_RNN_PREDICT, predict_args, context) if operation == XSEQ_OPN_RNN_SCORE
                          else xseq_mlp(model, XSEQ_OPN_MLP_PREDICT, predict_args, context))
        result = xseq_regressor_score(model, prediction, args, context)
    elif operation == XSEQ_OPN_RNN_CLASSIFIER_SCORE or operation == XSEQ_OPN_MLP_CLASSIFIER_SCORE:
        # `ClassifierScoreMixin.score`: predict, then the accuracy of it
        # against y, in this one call
        var predict_args = Python.tuple(args[0])
        var prediction = (xseq_rnn(model, XSEQ_OPN_RNN_PREDICT, predict_args, context) if operation == XSEQ_OPN_RNN_CLASSIFIER_SCORE
                          else xseq_mlp(model, XSEQ_OPN_MLP_PREDICT, predict_args, context))
        result = xseq_classifier_score(model, prediction, args, context)
    elif operation == XSEQ_OPN_PROPHET_INITIALIZE:
        result = prophet_initialize(model, args, context)
    elif operation == XSEQ_OPN_PROPHET_ADD_SEASONALITY:
        result = prophet_add_seasonality(model, args, context)
    elif operation == XSEQ_OPN_PROPHET_FIT:
        result = prophet_fit(model, args, context)
    elif operation == XSEQ_OPN_PROPHET_PREDICT:
        result = prophet_predict(model, args, context)
    elif owner == _XSEQ_RNN:
        result = xseq_rnn(model, operation, args, context)
    elif owner == _XSEQ_MLP:
        result = xseq_mlp(model, operation, args, context)
    else:
        result = xseq_forecast(model, operation, name, args, context)
    return result


def xseq_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        # bw2-sequence (2026-11-20): the operation is one lookup of the
        # caller's str in the held operation table (it was copied into a
        # String and routed by up to sixteen compares and prefix tests per
        # call, then compared again by each owner's chain). Owners take the
        # code; only a name that is not an operation builds its String.
        var name = call[1]
        var nk = xseq_keys()
        var operation = py_op_code(nk[].ops, name)
        var owner = _xseq_owner(operation, name)
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        if owner == _XSEQ_STATE:
            # providers/python/recurrent_state_pyapi_door.mojo (another family) takes the name.
            result = recurrent_state(model, String(py=name), args, context)
        elif owner == _XSEQ_RNN or owner == _XSEQ_MLP:
            var lifecycle = XseqPublicLifecycle(model, operation, name, args, context)
            if lifecycle.fitting:
                var depth = estimator_fit_begin(py_fields(model))
                try:
                    result = _xseq_execute(model, operation, owner, name, lifecycle.arguments, context)
                except problem:
                    raise lifecycle.refusal(problem)
                finally:
                    estimator_fit_end(model, depth)
            else:
                result = _xseq_execute(model, operation, owner, name, lifecycle.arguments, context)
            result = lifecycle.finish(model, result, context)
        else:
            result = _xseq_execute(model, operation, owner, name, args, context)
        return result.steal_data()
    except error:
        return _xseq_raise(error, args_ptr)
