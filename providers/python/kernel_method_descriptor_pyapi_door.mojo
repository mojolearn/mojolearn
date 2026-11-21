# SPDX-License-Identifier: Apache-2.0
# Kernel-method admission over the common provider. No new provider type or
# numerical implementation: this file only admits fixed private ABI words.
from std.python import PythonObject
from bindings.py_balanced import py_get, py_setitem
from providers.python.neural_descriptor_pyapi_door import (
    neural_descriptor_entry, neural_descriptor_reuse, neural_descriptor_execute,
)
from kernel_methods.descriptor_schema import KERNEL_METHOD_ABI, KERNEL_METHOD_OPERATIONS
from bindings.svm_keys_unpack import svm_keys


def kernel_method_provider(fields: PythonObject, binding: PythonObject, identical: Bool) raises -> PythonObject:
    # Both supported identical providers expose the same versioned table:
    # native_descriptor_dispatch on devices, descriptor_host on CPU. Fast
    # retains its existing entries; the former candidate environment switch
    # cannot silently route an identical operation through packed CPython math.
    if not identical:
        return PythonObject(None)
    var factory = neural_descriptor_entry(binding, "kernel_methods_api_v1", False)
    if factory is None:
        raise Error("RuntimeError: identical kernel methods require kernel_methods_api_v1; rebuild the kernel methods binding")
    var nk = svm_keys()
    var previous = py_get(fields, nk[].k__native_kernel_method_provider)
    var result = neural_descriptor_reuse(previous, factory, String(KERNEL_METHOD_ABI), KERNEL_METHOD_OPERATIONS)
    py_setitem(fields, nk[].k__native_kernel_method_provider, result)
    return result


# The packed admission `kernel_method_execute` (two Python lists re-read per
# call) is gone: the door builds a typed request and RUN calls the table
# (kernel_methods/native_call_run.mojo `KmCall`, `km_call_run`).


def kernel_method_metadata(native: PythonObject, operation: Int) raises -> Int:
    return neural_descriptor_execute(native, operation, [], [], [])
