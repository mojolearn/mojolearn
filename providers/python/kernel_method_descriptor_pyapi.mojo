# SPDX-License-Identifier: Apache-2.0
# Kernel-method admission over the common provider. No new provider type or
# numerical implementation: this file only admits fixed private ABI words.
from std.os import getenv
from std.python import PythonObject
from bindings.py_balanced import py_get, py_setitem
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_reuse, neural_descriptor_execute,
    neural_descriptor_info,
)
from kernel_methods.descriptor_schema import KERNEL_METHOD_ABI, KERNEL_METHOD_OPERATIONS, kernel_method_operation, kernel_method_scalar_mask


def kernel_method_provider(fields: PythonObject, binding: PythonObject) raises -> PythonObject:
    if getenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0") != "1":
        return PythonObject(None)
    var factory = neural_descriptor_entry(binding, "kernel_methods_api_v1", False)
    if factory is None:
        raise Error("kernel methods native provider is unavailable for this binding")
    var previous = py_get(fields, "_native_kernel_method_provider")
    var result = neural_descriptor_reuse(previous, factory, String(KERNEL_METHOD_ABI), KERNEL_METHOD_OPERATIONS)
    py_setitem(fields, "_native_kernel_method_provider", result)
    return result


def kernel_method_execute(native: PythonObject, name: String, addrs: PythonObject,
                          params: PythonObject, handle: Int = 0) raises -> PythonObject:
    var operation = kernel_method_operation(name)
    var schema = kernel_method_scalar_mask(operation)
    if len(params) != schema[0] or len(addrs) > 8:
        raise Error("kernel methods provider: packed admission does not match operation")
    var a = List[Int]()
    if operation == 6:
        a.append(handle)
    for i in range(len(addrs)):  # small-loop(at most eight fixed native buffer slots)
        a.append(Int(py=addrs[i]))
    var p = List[Int](length=schema[0], fill=0)
    var f = List[Float64](length=schema[0], fill=0)
    for i in range(schema[0]):  # small-loop(at most ten scalar parameter slots)
        var value = params[i]
        if (schema[1] & (1 << i)) != 0:
            f[i] = Float64(py=value)
        else:
            p[i] = Int(py=value)
    return PythonObject(neural_descriptor_execute(native, operation, a, p, f))


def kernel_method_metadata(native: PythonObject, operation: Int) raises -> Int:
    return neural_descriptor_execute(native, operation, [], [], [])
