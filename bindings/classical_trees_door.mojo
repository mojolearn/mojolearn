# SPDX-License-Identifier: Apache-2.0
# The tree, density, saved-model inference and host archive operations of
# `classical_native_call`, as their own compiled door (2026-11-07 host split;
# docs/build/HOT_GROUPS.md section 6). bindings/classical_native.mojo
# forwards these operations here with the caller's own argument tuple
# (PyObject_Call of `context["classical_trees_door"]`, no Python frame), so
# the steps, their order, the contexts they read and the error mapping are
# the ones the classical door ran before: moved unchanged.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_error_boundary import classical_python_error
from bindings.classical_caller import classical_caller_outside
from bindings.host_archive_native import host_archive_build
from bindings.host_archive_predict_native import host_archive_predict
from bindings.host_knn_predict_native import host_knn_predict
from bindings.native_operation_codes import operation_table, OP_ROUTE_TREES, OP_ROUTE_HOST_TREE
from bindings.host_tree_native import host_tree_operation
from bindings.trees_public_native import trees_public_operation
from bindings.host_inference_native import host_inference_operation
from bindings.host_kde_fit_native import host_kde_fit
from bindings.host_iforest_fit_native import host_iforest_fit
from bindings.host_density_constructor_native import host_density_initialize
from bindings.host_density_archive_native import host_density_save
from bindings.host_density_hash_native import host_density_hash
from bindings.py_balanced import py_type, py_get
from bindings.admitted_call import AdmittedCall
from bindings.classical_door_routes import classical_trees_operation


def _trees_coded(model: PythonObject, code: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A `trees_*` or `host_tree_*` operation by its integer code."""
    var table = operation_table()
    if code < 0 or code >= len(table[].names):
        raise Error("TypeError: unknown native operation code")
    var route = table[].routes[code]
    var name = table[].names[code].copy()
    if route == OP_ROUTE_TREES:
        return trees_public_operation(model, name, args, context)
    if route == OP_ROUTE_HOST_TREE:
        return host_tree_operation(model, name, args, context["host_tree_context"])
    raise Error("TypeError: native operation code is not a tree operation")


def _trees_named(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "host_kde_fit":
        return host_kde_fit(model, args, context["host_inference_context"])
    if operation == "host_iforest_fit":
        return host_iforest_fit(model, args, context["host_inference_context"])
    if operation == "host_density_save":
        return host_density_save(model, args, context["host_inference_context"])
    if operation == "host_density_hash":
        return host_density_hash(model, args, context["host_inference_context"])
    if operation == "host_density_initialize":
        return host_density_initialize(model, args, context["host_inference_context"])
    if operation.startswith("host_infer_"):
        return host_inference_operation(model, operation, args, context["host_inference_context"], context, classical_caller_outside(context))
    if operation == "archive_infer_predict" or operation == "archive_infer_predict_proba":
        var route = "host_infer_predict" if operation == "archive_infer_predict" else "host_infer_predict_proba"
        return host_inference_operation(model, route, args, context["host_inference_context"], context, False)
    if operation == "archive_knn_classifier" or operation == "archive_knn_regressor":
        return host_knn_predict(model, args[0], operation == "archive_knn_classifier", Bool(py=args[1]),
                                context["host_archive_predict_context"])
    if operation.startswith("trees_"):
        # The public tree ensembles' one-call methods
        # (bindings/trees_public_native.mojo).
        return trees_public_operation(model, operation, args, context)
    if operation.startswith("host_tree_"):
        return host_tree_operation(model, operation, args, context["host_tree_context"])
    if operation == "host_archive_predict" or operation == "host_archive_predict_proba":
        var predict_context = py_get(context, "host_archive_predict_context")
        if predict_context is None:
            raise Error("RuntimeError: native host archive schemas are not connected")
        return host_archive_predict(args[0], args[1], operation == "host_archive_predict_proba", predict_context)
    if operation == "host_archive_model" or operation == "host_archive_load" or operation == "host_archive_classical_model":
        var archive_context = py_get(context, "host_archive_context")
        if archive_context is None:
            raise Error("RuntimeError: native host archive schemas are not connected")
        # Checkpoint input admission belongs to the archive reader: its
        # exact native file descriptors and standard path layouts include
        # BytesIO/FileIO/buffered streams, which numeric-array admission
        # correctly refuses for unrelated estimator operations.
        return host_archive_build(args[0], archive_context, model if operation == "host_archive_load" else PythonObject(None), operation == "host_archive_classical_model")
    raise Error("RuntimeError: unknown native tree operation " + operation)


def classical_trees_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    """`trees_native_call(model, operation, args, context)`: the same four
    arguments `classical_native_call` received."""
    try:
        var request = PythonObject(from_borrowed=args_ptr)
        if len(request) != 4:
            raise Error("TypeError: trees_native_call requires model, operation, arguments and context")
        var call = AdmittedCall()
        var b = call.builtins()
        if py_type(request[1]) is b.int and py_type(request[3]) is b.dict:
            var coded = _trees_coded(request[0], Int(py=request[1]), request[2], request[3])
            return coded.steal_data()
        if py_type(request[1]) is not b.str or py_type(request[3]) is not b.dict:
            raise Error("TypeError: native classical operations require an exact operation string and context dictionary")
        var operation = String(request[1])
        if not classical_trees_operation(operation):
            raise Error("RuntimeError: unknown native tree operation " + operation)
        var result = _trees_named(request[0], operation, request[2], request[3])
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
