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
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.classical_caller_unpack import classical_caller_outside
from bindings.host_archive_native_door import host_archive_build
from bindings.host_archive_predict_native_pack import host_archive_predict
from bindings.host_knn_predict_native_pack import host_knn_predict
from bindings.native_operation_codes import operation_table, OP_ROUTE_TREES, OP_ROUTE_HOST_TREE, OP_ROUTE_CLASSICAL
from bindings.native_operation_names_unpack import operation_lookup, operation_code
from bindings.host_tree_native_door import host_tree_operation
from bindings.trees_public_native_door import trees_public_operation
from bindings.host_inference_native_unpack import host_inference_operation
from bindings.host_kde_fit_native_door import host_kde_fit
from bindings.host_iforest_fit_native_door import host_iforest_fit
from bindings.host_density_constructor_native_pack import host_density_initialize
from bindings.host_density_archive_native_door import host_density_save
from bindings.host_density_hash_native_unpack import host_density_hash, host_bound_hash
from bindings.py_balanced import py_type, py_get
from bindings.admitted_call_door import AdmittedCall
from bindings.classical_door_routes import classical_trees_operation
from bindings.classical_keys_unpack import classical_keys
from bindings.operation_names_classical import (
    CLASSICAL_TREES_FIRST, CLASSICAL_INFER_FIRST,
    CLASSICAL_OPN_ARCHIVE_INFER_PREDICT, CLASSICAL_OPN_ARCHIVE_INFER_PREDICT_PROBA,
    CLASSICAL_OPN_ARCHIVE_KNN_CLASSIFIER, CLASSICAL_OPN_ARCHIVE_KNN_REGRESSOR,
    CLASSICAL_OPN_HOST_ARCHIVE_CLASSICAL_MODEL, CLASSICAL_OPN_HOST_ARCHIVE_LOAD, CLASSICAL_OPN_HOST_ARCHIVE_MODEL,
    CLASSICAL_OPN_HOST_ARCHIVE_PREDICT, CLASSICAL_OPN_HOST_ARCHIVE_PREDICT_PROBA, CLASSICAL_OPN_HOST_BOUND_HASH,
    CLASSICAL_OPN_HOST_DENSITY_HASH, CLASSICAL_OPN_HOST_DENSITY_INITIALIZE, CLASSICAL_OPN_HOST_DENSITY_SAVE,
    CLASSICAL_OPN_HOST_IFOREST_FIT, CLASSICAL_OPN_HOST_KDE_FIT,
)
from bindings.py_balanced import py_item


def _trees_routed(model: PythonObject, route: Int, name: String, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """A `trees_*` or `host_tree_*` operation of the table, by its route; `name`
    is the table's own String, borrowed (bw2-classical: it was copied per
    call)."""
    if route == OP_ROUTE_TREES:
        # The public tree ensembles' one-call methods
        # (bindings/trees_public_native.mojo).
        return trees_public_operation(model, name, args, context)
    if route == OP_ROUTE_HOST_TREE:
        var nk = classical_keys()
        return host_tree_operation(model, name, args, py_item(context, nk[].k_host_tree_context))
    raise Error("TypeError: native operation code is not a tree operation")


def _trees_coded(model: PythonObject, code: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A `trees_*` or `host_tree_*` operation by its integer code."""
    var table = operation_table()
    if code < 0 or code >= len(table[].names):
        raise Error("TypeError: unknown native operation code")
    return _trees_routed(model, table[].routes[code], table[].names[code], args, context)


def _trees_leaf(model: PythonObject, operation: Int, name: String, args: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """The tree door's exact operations by their leaf
    (bindings/operation_names_classical.mojo); `name` is the table's own
    String, borrowed: the `host_infer_*` owner still takes the name."""
    var nk = classical_keys()
    if operation == CLASSICAL_OPN_HOST_KDE_FIT:
        return host_kde_fit(model, args, py_item(context, nk[].k_host_inference_context))
    if operation == CLASSICAL_OPN_HOST_IFOREST_FIT:
        return host_iforest_fit(model, args, py_item(context, nk[].k_host_inference_context))
    if operation == CLASSICAL_OPN_HOST_DENSITY_SAVE:
        return host_density_save(model, args, py_item(context, nk[].k_host_inference_context))
    if operation == CLASSICAL_OPN_HOST_DENSITY_HASH:
        return host_density_hash(model, args, py_item(context, nk[].k_host_inference_context))
    if operation == CLASSICAL_OPN_HOST_BOUND_HASH:
        return host_bound_hash(model, args, py_item(context, nk[].k_host_inference_context))
    if operation == CLASSICAL_OPN_HOST_DENSITY_INITIALIZE:
        return host_density_initialize(model, args, py_item(context, nk[].k_host_inference_context))
    if operation >= CLASSICAL_INFER_FIRST:
        # The `host_infer_*` names (bindings/operation_names_classical.mojo
        # numbers them last).
        return host_inference_operation(model, name, args, py_item(context, nk[].k_host_inference_context), context, classical_caller_outside(context))
    if operation == CLASSICAL_OPN_ARCHIVE_INFER_PREDICT or operation == CLASSICAL_OPN_ARCHIVE_INFER_PREDICT_PROBA:
        var route = "host_infer_predict" if operation == CLASSICAL_OPN_ARCHIVE_INFER_PREDICT else "host_infer_predict_proba"
        return host_inference_operation(model, route, args, py_item(context, nk[].k_host_inference_context), context, False)
    if operation == CLASSICAL_OPN_ARCHIVE_KNN_CLASSIFIER or operation == CLASSICAL_OPN_ARCHIVE_KNN_REGRESSOR:
        return host_knn_predict(model, args[0], operation == CLASSICAL_OPN_ARCHIVE_KNN_CLASSIFIER, Bool(py=args[1]),
                                py_item(context, nk[].k_host_archive_predict_context))
    if operation == CLASSICAL_OPN_HOST_ARCHIVE_PREDICT or operation == CLASSICAL_OPN_HOST_ARCHIVE_PREDICT_PROBA:
        var predict_context = py_get(context, nk[].k_host_archive_predict_context)
        if predict_context is None:
            raise Error("RuntimeError: native host archive schemas are not connected")
        return host_archive_predict(args[0], args[1], operation == CLASSICAL_OPN_HOST_ARCHIVE_PREDICT_PROBA, predict_context)
    if (operation == CLASSICAL_OPN_HOST_ARCHIVE_MODEL or operation == CLASSICAL_OPN_HOST_ARCHIVE_LOAD
            or operation == CLASSICAL_OPN_HOST_ARCHIVE_CLASSICAL_MODEL):
        var archive_context = py_get(context, nk[].k_host_archive_context)
        if archive_context is None:
            raise Error("RuntimeError: native host archive schemas are not connected")
        # Checkpoint input admission belongs to the archive reader: its
        # exact native file descriptors and standard path layouts include
        # BytesIO/FileIO/buffered streams, which numeric-array admission
        # correctly refuses for unrelated estimator operations.
        return host_archive_build(args[0], archive_context, model if operation == CLASSICAL_OPN_HOST_ARCHIVE_LOAD else PythonObject(None), operation == CLASSICAL_OPN_HOST_ARCHIVE_CLASSICAL_MODEL)
    raise Error("RuntimeError: unknown native tree operation " + name)


def _trees_prefixed(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A tree-door name the operation table does not hold: the prefixes the
    door forwards by name, in the old order."""
    if operation.startswith("host_infer_"):
        var nk = classical_keys()
        return host_inference_operation(model, operation, args, py_item(context, nk[].k_host_inference_context), context, classical_caller_outside(context))
    if operation.startswith("trees_"):
        return _trees_routed(model, OP_ROUTE_TREES, operation, args, context)
    if operation.startswith("host_tree_"):
        return _trees_routed(model, OP_ROUTE_HOST_TREE, operation, args, context)
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
        # bw2-classical: one dictionary lookup of the caller's own str in the
        # operation table (it was copied into a String and walked through the
        # door's 23 name and prefix tests twice, here and in the route test).
        # Only a name the table does not hold builds its String.
        var names = operation_lookup()
        var code = operation_code(names, request[1])
        if code >= 0:
            var route = names[].table.routes[code]
            if route == OP_ROUTE_TREES or route == OP_ROUTE_HOST_TREE:
                var routed = _trees_routed(request[0], route, names[].table.names[code], request[2], request[3])
                return routed.steal_data()
            if route == OP_ROUTE_CLASSICAL and names[].table.leaves[code] >= CLASSICAL_TREES_FIRST:
                var leaf = _trees_leaf(request[0], names[].table.leaves[code], names[].table.names[code],
                                       request[2], request[3])
                return leaf.steal_data()
            raise Error("RuntimeError: unknown native tree operation " + String(request[1]))
        var operation = String(request[1])
        if not classical_trees_operation(operation):
            raise Error("RuntimeError: unknown native tree operation " + operation)
        var result = _trees_prefixed(request[0], operation, request[2], request[3])
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
