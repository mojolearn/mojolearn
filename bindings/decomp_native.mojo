# SPDX-License-Identifier: Apache-2.0
# The native owner door of the decomposition expansion's estimators
# (FastICA, ...): every public operation of those classes is one call into
# decomp_c, which admits the model by its connected class and dispatches the
# operation to the class's native file (bindings/ica_native.mojo, ...).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type, py_fields
from bindings.classical_attributes import classical_connect_numpy
from bindings.ica_native import ica_fit, ica_project
from bindings.fa_native import fa_fit, fa_query
from bindings.kpca_native import kpca_fit, kpca_fit_transform, kpca_transform
from bindings.classical_error_boundary import classical_python_error
from providers.python.family_public_lifecycle_pyapi import kpca_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal
from bindings.rp_native import rp_components, rp_fit, rp_getstate, rp_project, rp_query, rp_set_components
from bindings.lda_native import lda_fit, lda_partial_fit, lda_project, lda_query
from bindings.decomp_public_lifecycle import decomp_public_method, decomp_public_begin, decomp_public_finish
from bindings.public_lifecycle_owner import lifecycle_end
from bindings.nmf_native import nmf_fit, nmf_fit_transform, nmf_inverse_transform, nmf_transform


def _family(model: PythonObject, context: PythonObject) raises -> String:
    """The connected class the model's MRO holds ('' for none)."""
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return ""
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var classes = context["decomp_classes"]
    var names = py_call(b.list, classes.keys())
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        for j in range(len(names)):  # driver-loop: over the connected classes
            if mro[k] is classes[names[j]]:
                return String(names[j])
    return ""


def _kpca_public(model: PythonObject, operation: String, args: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    # The whole public lifecycle (fitted check, X validation, width and
    # feature names, fit bookkeeping, NumPy results) around the owner
    # (providers/python/family_public_lifecycle_pyapi.mojo).
    var result = PythonObject(None)
    var lifecycle = kpca_lifecycle_spec(operation)
    if lifecycle.method.byte_length() == 0:
        raise Error("unknown private KernelPCA operation")
    var public = family_lifecycle_begin(model, operation, lifecycle, args, 0, context)
    try:
        var x = public.args[0]
        if operation == "fit":
            result = kpca_fit(model, x, context)
        elif operation == "fit_transform":
            result = kpca_fit_transform(model, x, context)
        else:
            result = kpca_transform(model, x, context)
    except error:
        var failure = neighbors_public_refusal(error, public, model)
        raise failure^
    finally:
        neighbors_public_end(model, public)
    return family_lifecycle_finish(model, lifecycle, result, public, context)


def _decomp_dispatch(model: PythonObject, family: String, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var result = PythonObject(None)
    if family == "FastICA":
        if operation == "fit":
            result = ica_fit(model, args[0], False, context)
        elif operation == "fit_transform":
            result = ica_fit(model, args[0], True, context)
        elif operation == "transform" or operation == "inverse_transform":
            result = ica_project(model, args[0], operation == "inverse_transform", context)
        else:
            raise Error("unknown private FastICA operation")
    elif family == "NMF":
        if operation == "fit":
            result = nmf_fit(model, args[0], args[1], context)
        elif operation == "fit_transform":
            result = nmf_fit_transform(model, args[0], args[1], args[2], context)
        elif operation == "transform":
            result = nmf_transform(model, args[0], context)
        elif operation == "inverse_transform":
            result = nmf_inverse_transform(model, args[0], context)
        else:
            raise Error("unknown private NMF operation")
    elif family == "FactorAnalysis":
        if operation == "fit":
            result = fa_fit(model, args[0], context)
        elif operation == "fit_transform":
            # `_Base.fit_transform`: fit, then transform(X), the same two
            # steps in this one call
            _ = fa_fit(model, args[0], context)
            result = fa_query(model, args[0], 0, context)
        elif operation == "transform":
            result = fa_query(model, args[0], 0, context)
        elif operation == "get_covariance":
            result = fa_query(model, PythonObject(None), 1, context)
        elif operation == "get_precision":
            result = fa_query(model, PythonObject(None), 2, context)
        elif operation == "score_samples":
            result = fa_query(model, args[0], 3, context)
        elif operation == "score":
            result = fa_query(model, args[0], 4, context)
        else:
            raise Error("unknown private FactorAnalysis operation")
    elif family == "LatentDirichletAllocation":
        if operation == "fit":
            _ = lda_fit(model, args[0], context)
            result = model
        elif operation == "fit_transform":
            # the transform projects the X the fit converted: one conversion
            var x = lda_fit(model, args[0], context)
            result = lda_project(model, py_fields(model), x, 2, Bool(py=args[1]), context)
        elif operation == "partial_fit":
            result = lda_partial_fit(model, args[0], context)
        elif operation == "transform":
            result = lda_query(model, args[0], 2, Bool(py=args[1]), context)
        elif operation == "score":
            result = lda_query(model, args[0], 3, False, context)
        elif operation == "perplexity":
            result = lda_query(model, args[0], 4, Bool(py=args[1]), context)
        else:
            raise Error("unknown private LatentDirichletAllocation operation")
    elif family == "GaussianRandomProjection" or family == "SparseRandomProjection":
        var sparse = family == "SparseRandomProjection"
        if operation == "fit":
            _ = rp_fit(model, args[0], sparse, context)
            result = model
        elif operation == "fit_transform":
            # the transform projects the X the fit converted: one conversion
            var x = rp_fit(model, args[0], sparse, context)
            result = rp_project(model, py_fields(model), args[0], x, False, context)
        elif operation == "transform" or operation == "inverse_transform":
            result = rp_query(model, args[0], operation == "inverse_transform", context)
        elif operation == "components":
            result = rp_components(model)
        elif operation == "set_components":
            rp_set_components(model, args[0])
        elif operation == "getstate":
            result = rp_getstate(model)
        else:
            raise Error("unknown private random projection operation")
    else:
        raise Error("unknown private decomposition estimator")
    return result


def decomp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: decomp_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple:
            raise Error("TypeError: native decomposition operation and argument tuple are required")
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if py_type(context) is not b.dict:
            raise Error("TypeError: native decomposition requires connected context")
        var family = _family(model, context)
        if family == "":
            raise Error("TypeError: the native decomposition owner requires a connected estimator")
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if family == "KernelPCA":
            # KernelPCA runs its own public lifecycle (family_public_lifecycle_pyapi);
            # every other family runs the decomposition lifecycle here.
            result = _kpca_public(model, operation, args, context)
        else:
            var method = decomp_public_method(operation)
            var depth = decomp_public_begin(model, method, args, True, True, context)
            try:
                result = _decomp_dispatch(model, family, operation, args, context)
            finally:
                lifecycle_end(model, method, depth)
            result = decomp_public_finish(model, method, args, result, "", context)
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:"):
            return classical_python_error(error^, args_ptr)
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
