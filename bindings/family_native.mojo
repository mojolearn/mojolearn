# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classical families' public methods (lane
# migrate-classical-b): one dispatcher, reached from the classical owner's
# door (`classical_native_call`, bindings/classical_native.mojo) for every
# `family_*` operation. Each family's file holds its operations; the door
# maps their `<Class>: ` error prefixes and restores a stashed Python
# exception unchanged.
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
from bindings.cluster_fit_b_native import cluster_fit_b_operation
from bindings.mixture_native import mixture_operation
from bindings.linalg_ops_native import linalg_ops_operation
from bindings.parallel_plan_door import plan_devices, plan_enter, plan_exit
from bindings.parallel_plan_specs import par_spec_families, par_spec_rebuild, par_spec_admits


def _family_par_run(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A cooperative multi-GPU driver as one native call
    (lane/native-parallel-driver-20261024,
    docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md): args (driver, devices,
    owner operation, owner args). The driver's shell made its type and mode
    refusals and resolved the binding (owner args[0], the plain method's).
    Here: the devices and route are admitted as the old pool admitted them,
    the owner operation must be one the driver's spec names
    (bindings/parallel_plan_specs.mojo), the plan binds the devices to this
    thread in the binding for the families the old worker's environment
    enabled, and the plain owner runs, the plan released in a `finally`."""
    var driver = String(args[0])
    var owner = String(args[2])
    if not par_spec_admits(driver, owner):
        raise Error("RuntimeError: parallel driver " + driver + " does not run " + owner)
    var devices = plan_devices(args[1], driver, True, context)
    var owner_args = args[3]
    var binding = owner_args[0]
    var result = PythonObject(None)
    plan_enter(binding, devices, par_spec_rebuild(driver), par_spec_families(driver))
    try:
        result = family_operation(model, owner, owner_args, context)
    finally:
        plan_exit(binding)
    return result


def family_operation(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    if operation == "family_par_run":
        return _family_par_run(model, args, context)
    if operation.startswith("family_kmeans_"):
        return kmeans_operation(model, operation, args, context)
    if operation.startswith("family_km_"):
        return kernel_methods_operation(model, operation, args, context)
    if operation.startswith("family_svm_"):
        return svm_mixture_operation(model, operation, args, context)
    if operation.startswith("family_xc_"):
        return cluster_operation(model, operation, args, context)
    if operation.startswith("family_metric_"):
        return metrics_reg_operation(operation, args, context)
    if operation.startswith("family_xn_"):
        return neighbors_operation(model, operation, args, context)
    if operation.startswith("family_chol_"):
        return linalg_operation(model, operation, args, context)
    if operation.startswith("family_msel_"):
        # Preserve the existing generic entry ABI without importing the
        # independently compiled model-selection owner. Import setup retains
        # its verified native entry; Python never chooses an operation route.
        var entry = py_get(context, "family_msel_door")
        if entry is None:
            raise Error("RuntimeError: the model-selection native owner is not connected")
        if py_type(entry) is not context["family_builtin_type"]:
            raise Error("TypeError: model selection requires its compiled native entry")
        return family_call(entry, model, PythonObject(operation), args, context)
    if operation.startswith("family_svc_"):
        return svc_operation(model, operation, args, context)
    if operation.startswith("family_gp_"):
        return gp_operation(model, operation, args, context)
    if operation.startswith("family_mcls_"):
        return metrics_cls_operation(model, operation, args, context)
    if operation.startswith("family_mexp_"):
        return metrics_exp_operation(model, operation, args, context)
    if operation.startswith("family_cfa_"):
        return cluster_fit_a_operation(model, operation, args, context)
    if operation.startswith("family_cfb_"):
        return cluster_fit_b_operation(model, operation, args, context)
    if operation.startswith("family_mix_"):
        return mixture_operation(model, operation, args, context)
    if operation.startswith("family_mani_"):
        # The shared import-time context retains the independently compiled
        # owner. Forward through its verified C entry without a source import
        # of the manifold implementation or a Python operation router.
        var entry = py_get(context, "family_mani_door")
        if entry is None:
            raise Error("RuntimeError: the manifold native owner is not connected")
        if py_type(entry) is not context["family_builtin_type"]:
            raise Error("TypeError: manifold requires its compiled native entry")
        return family_call(entry, model, PythonObject(operation), args, context)
    if operation.startswith("family_lin_"):
        return linalg_ops_operation(model, operation, args, context)
    if operation.startswith("family_rs_"):
        return linalg_ops_operation(model, operation, args, context)
    if operation.startswith("family_pg_"):
        return linalg_ops_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
