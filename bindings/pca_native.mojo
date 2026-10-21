# SPDX-License-Identifier: Apache-2.0
# Incremental private PCA operation owner. No public route is activated here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.sequential_admission import sequential_admit_value
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.pca_metadata import pca_dictionary, pca_fields, pca_initialize, pca_get_params, pca_set_params, pca_entries
from bindings.pca_exact import pca_exact_fit, pca_transform
from bindings.pca_archive import pca_archive_save, pca_archive_load


def pca_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: pca_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = Python.import_module("builtins")
        if b.type(call[1]) is not b.str or b.type(call[2]) is not b.tuple:
            raise Error("TypeError: native PCA operation and argument tuple are required")
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if b.type(context) is not b.dict:
            raise Error("TypeError: native PCA requires connected context")
        var loading = operation == "load"
        if (model is not context["pca_class"] if loading else b.type(model) is not context["pca_class"]):
            raise Error("TypeError: the private PCA owner requires an exact PCA instance")
        var result = PythonObject(None)
        if loading:
            _ = pca_dictionary(context)
            result = pca_archive_load(model, args[0], context)
            return result.steal_data()
        # Archive paths use the strict shared native path/file admission;
        # all other arguments must be admitted before any conversion or hook.
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
            if operation == "transform" or operation == "inverse_transform":
                if not Bool(py=classical_is_fitted(model)):
                    _ = classical_not_fitted_class(context)
                    raise Error("NotFittedError: This PCA instance is not fitted yet. Call 'fit' with appropriate arguments first.")
            var entries = pca_entries(fields, context)
            if operation == "fit" or operation == "fit_transform":
                result = pca_exact_fit(model, args, entries, context)
                if operation == "fit_transform":
                    result = pca_transform(model, args[0], False, entries, context)
            elif operation == "transform" or operation == "inverse_transform":
                result = pca_transform(model, args[0], operation == "inverse_transform", entries, context)
            elif operation == "numeric_mode_used":
                result = PythonObject("identical" if Int(py=entries["estimators_numeric_mode"]()) == 1 else "fast")
            elif operation == "vendor_used":
                result = entries["estimators_vendor"]()
            else:
                raise Error("unknown private PCA operation")
        return result.steal_data()
    except error:
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
            return raise_python_exception(error, ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error, ExceptionType.ValueError)
