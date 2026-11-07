# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classical families' public methods (lane
# migrate-classical-b): one dispatcher, reached from the classical owner's
# door (`classical_native_call`, bindings/classical_native.mojo) for every
# `family_*` operation. Each family's file holds its operations; the door
# maps their `<Class>: ` error prefixes and restores a stashed Python
# exception unchanged.
from providers.python.estimator_binding_pyapi import estimator_arguments
from providers.python.cluster_public_lifecycle_pyapi import cluster_public_operation
from std.python import PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.family_common import family_call
from bindings.kmeans_native import kmeans_operation
from bindings.kernel_methods_native import kernel_methods_operation
from bindings.linalg_native import linalg_operation
from bindings.neighbors_native import neighbors_operation
from bindings.metrics_reg_native import metrics_reg_operation
from bindings.cluster_native import cluster_operation
from bindings.svm_mixture_native import svm_mixture_operation
from bindings.svc_native import svc_operation
from bindings.gp_native import gp_operation
from bindings.metrics_cls_native import metrics_cls_operation
from bindings.metrics_exp_native import metrics_exp_operation
from bindings.cluster_fit_a_native import cluster_fit_a_operation
from providers.python.hdbscan_public_lifecycle_pyapi import hdbscan_public_operation
from providers.python.density_public_lifecycle_pyapi import density_public_operation
from bindings.cluster_fit_b_native import cluster_fit_b_operation
from bindings.mixture_native import mixture_operation
from bindings.linalg_ops_native import linalg_ops_operation
from providers.python.lowbit_weight_pyapi import lowbit_weight_operation
from bindings.parallel_public_native import parallel_public_operation
from bindings.parallel_plan_door import plan_devices, plan_enter_all, plan_exit_all
from bindings.parallel_plan_specs import par_spec_families, par_spec_rebuild, par_spec_admits
from bindings.native_operation_codes import (
    family_route, family_route_owner, family_route_pre, family_route_cluster,
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
    var driver = String(args[0])
    var owner = String(args[2])
    if not par_spec_admits(driver, owner):
        raise Error("RuntimeError: parallel driver " + driver + " does not run " + owner)
    var devices = plan_devices(args[1], driver, True, context)
    var owner_args = args[3]
    var owner_entry = args[4]
    var owner_context = args[5]
    var bindings = args[6]
    if owner_entry is not None and py_type(owner_entry) is not context["family_builtin_type"]:
        raise Error("TypeError: a parallel driver's owner must be a compiled entry")
    if len(bindings) == 0:
        raise Error("ImportError: " + par_spec_rebuild(driver))
    var result = PythonObject(None)
    plan_enter_all(bindings, devices, par_spec_rebuild(driver), par_spec_families(driver))
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
            result = _family_owner_call(model, family_route(owner), owner, owner_args, context)
        else:
            result = family_call(owner_entry, model, PythonObject(owner), owner_args, owner_context)
    finally:
        plan_exit_all(bindings)
    return result



def family_operation(model: PythonObject, operation: String, var args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """A `family_*` operation by name (string callers: the parallel driver's
    owner, the door's str path)."""
    return family_operation_routed(model, family_route(operation), operation, args^, context)


def family_operation_routed(model: PythonObject, route: Int, operation: String, var args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """A `family_*` operation by its packed route word
    (bindings/native_operation_codes.mojo `family_route`, precomputed per
    name for coded calls): one integer switch instead of the former prefix
    chain, same steps in the same order."""
    # `family_par_run` has no estimator-argument step and no cluster flag
    # (`family_route("family_par_run")` is FR_PAR_RUN alone), so running the
    # driver before those steps is the former order.
    if family_route_owner(route) == FR_PAR_RUN:
        return _family_par_run(model, args, context)
    return _family_owner_call(model, route, operation, args^, context)


def _family_owner_call(model: PythonObject, route: Int, operation: String, var args: PythonObject,
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
            return cluster_public_operation(model, operation, args, context)
    var owner = family_route_owner(route)
    if owner == FR_DENSITY:
        return density_public_operation(model, operation, args, context)
    if owner == FR_PAR_PUBLIC:
        return parallel_public_operation(model, args, context)
    if owner == FR_KMEANS:
        return kmeans_operation(model, operation, args, context)
    if owner == FR_KM:
        return kernel_methods_operation(model, operation, args, context)
    if owner == FR_SVM:
        return svm_mixture_operation(model, operation, args, context)
    if owner == FR_XC:
        return cluster_operation(model, operation, args, context)
    if owner == FR_METRIC:
        return metrics_reg_operation(operation, args, context)
    if owner == FR_XN:
        return neighbors_operation(model, operation, args, context)
    if owner == FR_CHOL:
        return linalg_operation(model, operation, args, context)
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
        return svc_operation(model, operation, args, context)
    if owner == FR_GP:
        return gp_operation(model, operation, args, context)
    if owner == FR_MCLS:
        return metrics_cls_operation(model, operation, args, context)
    if owner == FR_MEXP:
        return metrics_exp_operation(model, operation, args, context)
    if owner == FR_HDBSCAN_PUBLIC:
        return hdbscan_public_operation(model, operation, args, context)
    if owner == FR_CFA:
        return cluster_fit_a_operation(model, operation, args, context)
    if owner == FR_CFB:
        return cluster_fit_b_operation(model, operation, args, context)
    if owner == FR_MIX:
        return mixture_operation(model, operation, args, context)
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
        return lowbit_weight_operation(model, operation, args, context)
    if owner == FR_LINALG_OPS:
        return linalg_ops_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
