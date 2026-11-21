# SPDX-License-Identifier: Apache-2.0
# The compiled owner of GaussianProcessRegressor/Classifier public methods and the parallel_gaussian_process functions (python/mojolearn/_gp_impl.py, _gpc_impl.py, parallel_gaussian_process.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_gp_*` operation.
# The regressor's operations are bindings/gp_regressor_native_door.mojo's, the
# classifier's bindings/gp_classifier_native_door.mojo's.
from std.python import PythonObject
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_mode
from bindings.gp_argument_pins_unpack import GPArgumentPins
from bindings.gp_regressor_native_door import gp_regressor_operation
from bindings.gp_classifier_native_door import gp_classifier_operation
from bindings.gp_kernel_public_native_door import gp_kernel_operation
from providers.python.gp_public_lifecycle_pyapi_door import gp_public_operation, gp_public_vendor_used


from bindings.operation_names_gp import (
    GP_OPN_FAMILY_GP_GPR_FIT, GP_OPN_FAMILY_GP_GPC_LML, GP_OPN_FAMILY_GP_GPC_FIT, GP_OPN_FAMILY_GP_GPC_PREDICT,
    GP_OPN_FAMILY_GP_GPC_SCORE, GP_OPN_FAMILY_GP_GPR_VENDOR_USED)


def gp_unknown_operation(operation: String) raises:
    """The error of a `family_gp_*` name that is not an operation (it has no
    leaf code): the text of the owner its prefix selected, as before. Always
    raises."""
    if operation.startswith("family_gp_kernel_"):
        raise Error("RuntimeError: unknown GP kernel metadata operation")
    if operation.startswith("family_gp_gpr_"):
        raise Error("RuntimeError: unknown gaussian process regressor operation " + operation)
    if operation.startswith("family_gp_gpc_"):
        raise Error("RuntimeError: unknown gaussian process classifier operation " + operation)
    raise Error("RuntimeError: unknown family operation " + operation)


def gp_operation(model: PythonObject, operation: Int, var args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    # bw2-core: `operation` is the leaf code of bindings/operation_names_gp.mojo
    # (kernel codes, then regressor, then classifier), resolved once by the
    # family door; the owner is a range test (it was three prefix tests on the
    # name String, then each owner's exact-name chain).
    if operation < GP_OPN_FAMILY_GP_GPR_FIT:
        return gp_kernel_operation(model, operation, args, context)
    if operation >= GP_OPN_FAMILY_GP_GPR_VENDOR_USED:
        # `vendor_used()` of either estimator: the route's device read-back.
        return gp_public_vendor_used(model, context)
    if operation != GP_OPN_FAMILY_GP_GPC_LML and len(args) > 0 and args[0] is None:
        return gp_public_operation(model, operation, args, context)
    # Internal/archive/worker calls retain their explicit provider pins in
    # the same native envelope used by the public lifecycle door.
    var values = GPArgumentPins(args)
    if operation != GP_OPN_FAMILY_GP_GPC_LML and len(values) > 0 and values[0] is None:
        values.set(0, estimator_binding(model, "_mojolearn_gp", context))
    if operation == GP_OPN_FAMILY_GP_GPC_FIT or operation == GP_OPN_FAMILY_GP_GPC_PREDICT or operation == GP_OPN_FAMILY_GP_GPC_SCORE:
        if values[1] is None:
            values.set(1, estimator_binding_mode(PythonObject("identical"), "_mojolearn", context))
    if operation < GP_OPN_FAMILY_GP_GPC_LML:
        return gp_regressor_operation(model, operation, values, context)
    return gp_classifier_operation(model, operation, values, context)
