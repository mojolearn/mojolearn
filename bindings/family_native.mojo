# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classical families' public methods (lane
# migrate-classical-b): one dispatcher, reached from the classical owner's
# door (`classical_native_call`, bindings/classical_native.mojo) for every
# `family_*` operation. Each family's file holds its operations; the door
# maps their `<Class>: ` error prefixes and restores a stashed Python
# exception unchanged.
from std.python import PythonObject
from bindings.kmeans_native import kmeans_operation
from bindings.model_selection_native import msel_operation
from bindings.kernel_methods_native import kernel_methods_operation


def family_operation(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    if operation.startswith("family_kmeans_"):
        return kmeans_operation(model, operation, args, context)
    if operation.startswith("family_km_"):
        return kernel_methods_operation(model, operation, args, context)
    if operation.startswith("family_msel_"):
        return msel_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
