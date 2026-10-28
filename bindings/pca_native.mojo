# SPDX-License-Identifier: Apache-2.0
# The native PCA owner: every public PCA operation is one call into pca_c.
from std.python import Python, PythonObject
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type
from bindings.sequential_admission import sequential_admit_value
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_attributes import classical_connect_numpy
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.pca_metadata import pca_dictionary, pca_fields, pca_initialize, pca_get_params, pca_set_params, pca_entries
from bindings.pca_fit_native import pca_fit_operation
from bindings.pca_query_native import pca_query_operation
from bindings.pca_archive import pca_archive_save, pca_archive_load


def _pca_admitted(model: PythonObject, loading: Bool, context: PythonObject) raises -> Bool:
    # PCA itself or an ordinary subclass of it (HostPCA, a user's class):
    # the class's MRO, read through type's own descriptor, holds PCA.
    var b = Python.import_module("builtins")
    var cls = model if loading else py_type(model)
    if py_type(cls) is not b.type:
        return False
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        if mro[k] is context["pca_class"]:
            return True
    return False


def pca_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: pca_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple:
            raise Error("TypeError: native PCA operation and argument tuple are required")
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if py_type(context) is not b.dict:
            raise Error("TypeError: native PCA requires connected context")
        var loading = operation == "load"
        if not _pca_admitted(model, loading, context):
            raise Error("TypeError: the native PCA owner requires a PCA instance")
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if loading:
            _ = pca_dictionary(context)
            result = pca_archive_load(model, args[0], context)
            return result.steal_data()
        if operation == "transform" or operation == "inverse_transform":
            # The query's native PCA provider owns sparse/pandas/Torch and
            # ordinary buffer admission. Sequential's narrower generic
            # admission must not preempt those supported native paths.
            result = pca_query_operation(model, args[0], operation == "inverse_transform", context)
            return result.steal_data()
        if operation == "fit" or operation == "fit_transform":
            result = pca_fit_operation(model, args, operation == "fit_transform", context)
            return result.steal_data()
        # Archive paths use the strict shared native path/file admission;
        # other legacy operations retain their existing generic admission.
        if operation != "save":
            sequential_admit_value(args, context)
        var fields = pca_fields(model, context)
        if operation == "initialize":
            result = pca_initialize(model, args)
        elif operation == "get_params":
            result = pca_get_params(model, context)
        elif operation == "set_params":
            result = pca_set_params(model, args[0], context)
        elif operation == "is_fitted":
            result = classical_is_fitted(model)
        elif operation == "save":
            result = pca_archive_save(model, args[0], context)
        else:
            if py_get(context, "ndarray") is None:
                raise Error("ImportError: mojolearn: PCA requires optional NumPy support. Install it with: "
                            + 'python -m pip install "mojolearn[numpy]" ("mojolearn[verify]" also includes NumPy).')
            var entries = pca_entries(fields, context)
            if operation == "numeric_mode_used":
                result = PythonObject("identical" if Int(py=entries["estimators_numeric_mode"]()) == 1 else "fast")
            elif operation == "vendor_used":
                result = entries["estimators_vendor"]()
            else:
                raise Error("unknown private PCA operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("NotFittedError:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 3)
                var cls = causal_borrowed_item(context, "active_not_fitted_class")
                var message = String(error)
                return causal_set_exception(Error(String(message[byte=16:])), cls)
            except nested:
                return raise_python_exception(nested)
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
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
