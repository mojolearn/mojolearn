# SPDX-License-Identifier: Apache-2.0
# Actual affected public roots, compile only; never execute this artifact.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.robust_native import robust_lifecycle_c
from bindings.maxabs_native import maxabs_lifecycle_c
from bindings.host_svc_native import host_svc_attribute_c
from bindings.svc_linear_native import svc_linear_operation
from bindings.device_plan_binding import device_plan_log_binding


def svc_operation(model: PythonObject, operation: PythonObject, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    return svc_linear_operation(model, String(operation), args, context)


@export
def PyInit__native_import_setup() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native_import_setup")
        module.def_py_c_function(robust_lifecycle_c, "robust")
        module.def_py_c_function(maxabs_lifecycle_c, "maxabs")
        module.def_py_c_function(host_svc_attribute_c, "host_svc_attribute")
        module.def_function[svc_operation]("svc")
        module.def_function[device_plan_log_binding]("device_plan_log")
        return module.finalize()
    except error:
        abort(String(error))
