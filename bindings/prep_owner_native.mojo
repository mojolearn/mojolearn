# SPDX-License-Identifier: Apache-2.0
# The public door of the x_prep family (registered as prep_encode_native_call).
# Every public preprocessing operation is one call here. The classes the
# encode owner serves (bindings/prep_encode_native.mojo) go on to
# prep_encode_c unchanged; the label encoders (bindings/prep_labels_native.mojo),
# IterativeImputer (bindings/prep_iterimp_native.mojo), RFE and the
# mutual-information functions (bindings/prep_select_native.mojo) and the
# univariate score functions (below) are dispatched by the model's class
# (`prep_owner_kinds`) or, for a module-level function (model None), by the
# operation name. Each owner builds the x_prep programs the Python built (the
# same stages, operands and order, so the same bits).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes import classical_connect_numpy
from bindings.prep_encode_native import prep_encode_c, prep_encode_scores_call
from bindings.prep_labels_native import prep_labels_call
from bindings.prep_iterimp_native import prep_iterimp_call
from bindings.prep_select_native import prep_rfe_call, prep_mutual_info_call
from bindings.prep_composition_lifecycle import composition_before, composition_after_fit
from bindings.parallel_public_result import parallel_public_result
from bindings.public_lifecycle_owner import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result


def _owner_kind(model: PythonObject, context: PythonObject) raises -> String:
    # The owner kind of this model's class (or a base, by its MRO); "" when
    # the encode owner serves it (or model is None: a module function).
    if model is None:
        return ""
    var b = Python.import_module("builtins")
    if py_type(py_type(model)) is not b.type:
        raise Error("TypeError: native preprocessing refuses Python metaclass hooks")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var kinds = context["prep_owner_kinds"]
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    return ""


def prep_owner_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var kind = _owner_kind(model, context)
        if kind == "" and model is not None:
            # SimpleImputer, KBinsDiscretizer, the encoders, SplineTransformer, SelectKBest
            return prep_encode_c(py_self, args_ptr)
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if kind == "label_encoder" or kind == "label_binarizer" or kind == "multilabel_binarizer":
            # The label encoders' public lifecycle (the former `_mode`
            # wrappers): NotFittedError, fit bookkeeping and NumPy results;
            # their input is y, so no X table checks.
            var public = (operation == "fit" or operation == "fit_transform" or operation == "transform"
                          or operation == "inverse_transform")
            var depth = lifecycle_begin(model, operation, args[0], False, False, True, context) if public else 0
            try:
                result = prep_labels_call(model, kind, operation, args, context)
            finally:
                if public:
                    lifecycle_end(model, operation, depth)
            if public:
                lifecycle_complete(model, operation, PythonObject(None), False, context)
                result = lifecycle_result(model, operation, result, "", context)
        elif kind == "iterative_imputer":
            var prepared = composition_before(model, kind, operation, args, context)
            result = prep_iterimp_call(model, operation, prepared, context)
        elif kind == "rfe":
            var prepared = composition_before(model, kind, operation, args, context)
            result = prep_rfe_call(model, operation, prepared, context)
        elif kind != "":
            raise Error("TypeError: the native prep owner does not serve this class")
        elif operation == "mutual_info_classif" or operation == "mutual_info_regression":
            result = prep_mutual_info_call(operation, args, context)
        elif operation == "f_classif" or operation == "chi2" or operation == "f_regression" or operation == "r_regression":
            result = prep_encode_scores_call(operation, args, context)
        else:
            raise Error("unknown native prep operation")
        if kind == "rfe" or kind == "iterative_imputer":
            if operation == "fit" or operation == "fit_transform":
                composition_after_fit(model, args[0], context)
            result = parallel_public_result(result, operation, PythonObject(None), context)
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
        if String(error).startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
