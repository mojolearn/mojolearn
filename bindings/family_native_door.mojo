# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classical families' public methods (lane
# migrate-classical-b): one dispatcher, reached from the classical owner's
# door (`classical_native_call`, bindings/classical_native.mojo) for every
# `family_*` operation. Each family's file holds its operations; the door
# maps their `<Class>: ` error prefixes and restores a stashed Python
# exception unchanged.
from providers.python.estimator_binding_pyapi_door import estimator_arguments
from providers.python.cluster_public_lifecycle_pyapi_door import cluster_public_operation
from std.python import PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.family_common_pack import family_call
from bindings.kmeans_native_pack import kmeans_operation, kmeans_unknown_operation
from bindings.kernel_methods_native_door import kernel_methods_operation, kernel_methods_unknown_operation
from bindings.linalg_native_door import linalg_operation, linalg_unknown_operation
from bindings.neighbors_native_pack import neighbors_operation
from bindings.metrics_reg_door import metrics_reg_operation, metrics_reg_unknown_operation
from bindings.cluster_native_door import cluster_operation, cluster_unknown_operation
from bindings.svm_mixture_native_door import svm_mixture_operation, svm_mixture_unknown_operation
from bindings.svc_native_pack import svc_operation, svc_unknown_operation
from bindings.gp_native_pack import gp_operation, gp_unknown_operation
from bindings.metrics_cls_native_unpack import metrics_cls_operation, metrics_cls_unknown_operation
from bindings.metrics_exp_native_unpack import metrics_exp_operation, metrics_exp_unknown_operation
from bindings.cluster_fit_a_native_door import cluster_fit_a_operation, cluster_fit_a_unknown_operation
from providers.python.hdbscan_public_lifecycle_pyapi_pack import hdbscan_public_operation, hdbscan_unknown_operation
from providers.python.density_public_lifecycle_pyapi_pack import density_public_operation, density_unknown_operation
from bindings.cluster_fit_b_native_unpack import cluster_fit_b_operation, cluster_fit_b_unknown_operation
from bindings.operation_names_xc import XC_OPN_FAMILY_XC_DBSCAN_PREDICT
from bindings.operation_names_cfb import CFB_OPN_FAMILY_XC_DBSCAN_PREDICT
from bindings.mixture_public_native_pack import mixture_operation
from mixture.mixture_door_run import mixture_unknown_operation
from bindings.linalg_ops_native_unpack import linalg_ops_operation, linalg_ops_unknown_operation
from providers.python.lowbit_weight_pyapi_door import lowbit_weight_operation, lowbit_weight_unknown_operation
from bindings.parallel_public_native_door import parallel_public_operation
from bindings.parallel_plan_door import plan_devices, plan_enter_all, plan_exit_all
from bindings.parallel_plan_specs import par_spec_families, par_spec_rebuild, par_spec_admits, par_spec_driver_name
from bindings.parallel_keys_unpack import parallel_keys
from bindings.py_keys import py_op_code
from bindings.native_operation_names_unpack import operation_lookup, operation_code
from bindings.native_operation_codes import (
    family_route, family_route_owner, family_route_pre, family_route_cluster, OP_LEAF_NONE,
    FR_PRE_ESTIMATORS, FR_PRE_SVM, FR_PRE_BASE, FR_PRE_X_NEIGHBORS,
    FR_DENSITY, FR_PAR_PUBLIC, FR_PAR_RUN, FR_KMEANS, FR_KM, FR_SVM, FR_XC, FR_METRIC, FR_XN, FR_CHOL,
    FR_MSEL, FR_SVC, FR_GP, FR_MCLS, FR_MEXP, FR_HDBSCAN_PUBLIC, FR_CFA, FR_CFB, FR_MIX, FR_MANI, FR_LOW,
    FR_LINALG_OPS)


def _family_par_run(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A cooperative multi-GPU driver as one native call
    (lane/native-parallel-driver-20261024,
    docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md): args (driver, devices,
    owner operation, owner args, owner entry, owner context, plan bindings).
    The driver's shell made its type and mode refusals and resolved the
    plain method's entry, context and bindings. Here: the devices and route
    are admitted as the old pool admitted them, the owner operation must be
    one the driver's spec names (bindings/parallel_plan_specs.mojo), the plan
    binds the devices to this thread in every binding the plain owner
    reaches, for the families the old worker's environment enabled, and the
    plain owner runs: a family operation here (owner entry None), any other
    through the plain method's own compiled entry and context. The plans are
    released in a `finally`."""
    # bw2-parallel: the driver and its owner operation are one lookup each of
    # the caller's own str in the parallel key table (it built two Strings
    # and walked the specs' name chains per call). Only a refusal builds the
    # names' text.
    var nk = parallel_keys()
    var driver = py_op_code(nk[].ops_par_driver, args[0])
    if not par_spec_admits(driver, py_op_code(nk[].ops_par_owner, args[2])):
        raise Error("RuntimeError: parallel driver " + String(args[0]) + " does not run " + String(args[2]))
    var devices = plan_devices(args[1], String(par_spec_driver_name(driver)), True, context)
    var owner_args = args[3]
    var owner_entry = args[4]
    var owner_context = args[5]
    var bindings = args[6]
    if owner_entry is not None and py_type(owner_entry) is not context["family_builtin_type"]:
        raise Error("TypeError: a parallel driver's owner must be a compiled entry")
    if len(bindings) == 0:
        raise Error("ImportError: " + String(par_spec_rebuild(driver)))
    var result = PythonObject(None)
    plan_enter_all(bindings, devices, String(par_spec_rebuild(driver)), par_spec_families(driver))
    try:
        if owner_entry is None:
            # No recursion back into the dispatcher (lane ne-fix-hang3): the
            # owner runs through `_family_owner_call`, which never reaches
            # this driver. A recursion through the dispatcher (first three
            # functions, then two) stopped the Mojo compiler with no CPU use
            # and no diagnostic on _mojolearn_runtime_classical_host (lesson
            # 7 of docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md section 4).
            # Same steps: no driver spec in bindings/parallel_plan_specs.mojo
            # admits `family_par_run` as its owner, so the owner is never the
            # driver itself.
            # bw2-core: the owner's route word and leaf code are one lookup
            # of the driver's own str in the operation table (it ran the
            # prefix chain of `family_route` on the String per call). A name
            # the table does not hold keeps the chain.
            var table = operation_lookup()
            var code = operation_code(table, args[2])
            if code >= 0:
                result = _family_owner_call(model, table[].table.family_routes[code], table[].table.leaves[code],
                                            table[].table.names[code], owner_args, context)
            else:
                var owner = String(args[2])
                result = _family_owner_call(model, family_route(owner), OP_LEAF_NONE, owner, owner_args, context)
        else:
            # The owner name is the caller's own str (it was a str built
            # from the String copy of it).
            result = family_call(owner_entry, model, args[2], owner_args, owner_context)
    finally:
        plan_exit_all(bindings)
    return result



def family_operation(model: PythonObject, operation: String, var args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """A `family_*` operation by a name the operation table does not hold
    (the door's str path resolves a table name to its code and calls
    `family_operation_routed`): the prefix chain of `family_route` on the
    name, and no leaf code, so a converted owner refuses it as unknown."""
    return family_operation_routed(model, family_route(operation), OP_LEAF_NONE, operation, args^, context)


def family_operation_routed(model: PythonObject, route: Int, leaf: Int, operation: String,
                            var args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A `family_*` operation by its packed route word and leaf code
    (bindings/native_operation_codes.mojo `family_routes` and `leaves`,
    precomputed per name): one integer switch instead of the former prefix
    chain, same steps in the same order. A converted owner takes `leaf`; an
    owner that still matches the name takes `operation` (the table's own
    String for a table name)."""
    # `family_par_run` has no estimator-argument step and no cluster flag
    # (`family_route("family_par_run")` is FR_PAR_RUN alone), so running the
    # driver before those steps is the former order.
    if family_route_owner(route) == FR_PAR_RUN:
        return _family_par_run(model, args, context)
    return _family_owner_call(model, route, leaf, operation, args^, context)


def _family_owner_call(model: PythonObject, route: Int, leaf: Int, operation: String, var args: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """Every family owner except the parallel driver: the estimator-argument
    step, the cluster refusal, then the owner. Non-recursive: no owner here
    reaches `_family_par_run` or `family_operation_routed`."""
    # These public frontdoors share one import-connected binding provider.
    # Deeper algorithm dispatch is migrated separately; do not equate this
    # boundary repair with completion of every transitive runtime path.
    var pre = family_route_pre(route)
    if pre == FR_PRE_ESTIMATORS:
        args = estimator_arguments(model, "_mojolearn_estimators", args, context)
    elif pre == FR_PRE_SVM:
        args = estimator_arguments(model, "_mojolearn_svm", args, context)
    elif pre == FR_PRE_BASE:
        args = estimator_arguments(model, "_mojolearn", args, context)
    elif pre == FR_PRE_X_NEIGHBORS:
        args = estimator_arguments(model, "_mojolearn_x_neighbors", args, context)
    if family_route_cluster(route):
        if len(args) > 0 and args[0] is None:
            return cluster_public_operation(model, family_route_owner(route), leaf, operation, args, context)
    # Two-level switch (lane ne-fix-hang3, lesson 12): the owner word picks
    # one of two halves of about ten owners each, instead of one chain of
    # twenty-one calls to distinct owners in this function. Same owners,
    # same order (the halves split the former chain at FR_MSEL).
    var owner = family_route_owner(route)
    if owner < FR_MSEL:
        return _family_owner_first(model, owner, leaf, operation, args, context)
    return _family_owner_second(model, owner, leaf, operation, args, context)


def _family_owner_first(model: PythonObject, owner: Int, leaf: Int, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """Owners FR_DENSITY .. FR_CHOL (and FR_UNKNOWN, refused). `leaf` is
    the operation's leaf code for a converted owner (FR_DENSITY, FR_KMEANS,
    FR_KM, FR_SVM, FR_XC, FR_METRIC, FR_CHOL); the others match `operation`."""
    if owner == FR_DENSITY:
        if leaf == OP_LEAF_NONE:
            density_unknown_operation(operation)
        return density_public_operation(model, leaf, args, context)
    if owner == FR_PAR_PUBLIC:
        return parallel_public_operation(model, args, context)
    if owner == FR_KMEANS:
        if leaf == OP_LEAF_NONE:
            kmeans_unknown_operation(operation)
        return kmeans_operation(model, leaf, args, context)
    if owner == FR_KM:
        if leaf == OP_LEAF_NONE:
            kernel_methods_unknown_operation(operation)
        return kernel_methods_operation(model, leaf, args, context)
    if owner == FR_SVM:
        if leaf == OP_LEAF_NONE:
            svm_mixture_unknown_operation(operation)
        return svm_mixture_operation(model, leaf, args, context)
    if owner == FR_XC:
        if leaf == OP_LEAF_NONE:
            cluster_unknown_operation(operation)
        if leaf == XC_OPN_FAMILY_XC_DBSCAN_PREDICT:
            return cluster_fit_b_operation(model, CFB_OPN_FAMILY_XC_DBSCAN_PREDICT, args, context)
        return cluster_operation(model, leaf, args, context)
    if owner == FR_METRIC:
        if leaf == OP_LEAF_NONE:
            metrics_reg_unknown_operation(operation)
        return metrics_reg_operation(leaf, args, context)
    if owner == FR_XN:
        return neighbors_operation(model, operation, args, context)
    if owner == FR_CHOL:
        if leaf == OP_LEAF_NONE:
            linalg_unknown_operation(operation)
        return linalg_operation(model, leaf, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)


def _family_owner_second(model: PythonObject, owner: Int, leaf: Int, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """Owners FR_MSEL .. FR_LINALG_OPS. `leaf` is the operation's leaf code
    for a converted owner (FR_SVC, FR_GP, FR_HDBSCAN_PUBLIC, FR_CFA, FR_CFB, FR_MCLS,
    FR_MEXP, FR_MIX, FR_LOW, FR_LINALG_OPS); the others match `operation`."""
    if owner == FR_MSEL:
        # Preserve the existing generic entry ABI without importing the
        # independently compiled model-selection owner. Import setup retains
        # its verified native entry; Python never chooses an operation route.
        var entry = py_get(context, "family_msel_door")
        if entry is None:
            raise Error("RuntimeError: the model-selection native owner is not connected")
        if py_type(entry) is not context["family_builtin_type"]:
            raise Error("TypeError: model selection requires its compiled native entry")
        return family_call(entry, model, PythonObject(operation), args, context)
    if owner == FR_SVC:
        if leaf == OP_LEAF_NONE:
            svc_unknown_operation(operation)
        return svc_operation(model, leaf, args, context)
    if owner == FR_GP:
        if leaf == OP_LEAF_NONE:
            # Not a name of the table: the unchanged unknown-operation error.
            gp_unknown_operation(operation)
        return gp_operation(model, leaf, args, context)
    if owner == FR_MCLS:
        if leaf == OP_LEAF_NONE:
            metrics_cls_unknown_operation(operation)
        return metrics_cls_operation(model, leaf, args, context)
    if owner == FR_MEXP:
        if leaf == OP_LEAF_NONE:
            metrics_exp_unknown_operation(operation)
        return metrics_exp_operation(model, leaf, args, context)
    if owner == FR_HDBSCAN_PUBLIC:
        if leaf == OP_LEAF_NONE:
            hdbscan_unknown_operation(operation)
        return hdbscan_public_operation(model, leaf, args, context)
    if owner == FR_CFA:
        if leaf == OP_LEAF_NONE:
            cluster_fit_a_unknown_operation(operation)
        return cluster_fit_a_operation(model, leaf, args, context)
    if owner == FR_CFB:
        if leaf == OP_LEAF_NONE:
            cluster_fit_b_unknown_operation(operation)
        return cluster_fit_b_operation(model, leaf, args, context)
    if owner == FR_MIX:
        if leaf == OP_LEAF_NONE:
            mixture_unknown_operation(operation)
        return mixture_operation(model, leaf, args, context)
    if owner == FR_MANI:
        # The shared import-time context retains the independently compiled
        # owner. Forward through its verified C entry without a source import
        # of the manifold implementation or a Python operation router.
        var entry = py_get(context, "family_mani_door")
        if entry is None:
            raise Error("RuntimeError: the manifold native owner is not connected")
        if py_type(entry) is not context["family_builtin_type"]:
            raise Error("TypeError: manifold requires its compiled native entry")
        return family_call(entry, model, PythonObject(operation), args, context)
    if owner == FR_LOW:
        if leaf == OP_LEAF_NONE:
            lowbit_weight_unknown_operation(operation)
        return lowbit_weight_operation(model, leaf, args, context)
    if owner == FR_LINALG_OPS:
        if leaf == OP_LEAF_NONE:
            linalg_ops_unknown_operation(operation)
        return linalg_ops_operation(model, leaf, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
