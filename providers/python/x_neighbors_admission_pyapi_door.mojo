# SPDX-License-Identifier: Apache-2.0
# Pure XN provider admission, independent of estimator mode/registry selection.
# Shared estimator selection can call this without an import cycle and return
# the retained mode/capability snapshot to the existing operation owner.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute,
)
from providers.python.compiled_binding_snapshot_pyapi import (
    CompiledBindingSnapshot, compiled_binding_retained, compiled_binding_retain,
)
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT, XN_MODE_OPERATION, xn_operation_id


def xn_is_module(value: PythonObject) raises -> Bool:
    return py_type(value) is py_type(native_loaded_module("builtins"))


def xn_module(value: PythonObject) raises -> PythonObject:
    if xn_is_module(value):
        return value
    var owner = value.downcast_value_ptr[CompiledBindingSnapshot]()
    if owner[].abi != String(XN_ABI_NAME):
        raise Error("x_neighbors requires its admitted shared native provider")
    return owner[].module


def xn_capabilities(binding: PythonObject) raises -> Int:
    return binding.downcast_value_ptr[CompiledBindingSnapshot]()[].capabilities[0]


def xn_admit_checked(binding: PythonObject, expected: Int = 1) raises -> PythonObject:
    if expected != 1:
        raise Error("x_neighbors checked native admission currently requires identical mode")
    if not xn_is_module(binding):
        _ = xn_module(binding)
        if binding.downcast_value_ptr[CompiledBindingSnapshot]()[].mode != expected:
            raise Error("RuntimeError: mojolearn: _mojolearn_x_neighbors compiled mode differs from requested identical mode")
        return binding
    var factory = neural_descriptor_entry(binding, "x_neighbors_native_api_v1")
    # The admitted snapshot is retained for the process, keyed by this exact
    # module and the factory entry just read from its live namespace, and
    # holding both strongly (a loaded extension is never unloaded). Mode and
    # capabilities are compile-time constants of the image, so later public
    # operations reuse it instead of connecting a provider and making two
    # descriptor calls each time (lane perf-provider-cache, 2026-11-06). A
    # replaced module or entry misses and runs the admission below again.
    var retained = compiled_binding_retained(binding, factory, String(XN_ABI_NAME))
    if retained is not None:
        return retained
    var provider = neural_descriptor_connect(factory, String(XN_ABI_NAME), XN_OPERATION_COUNT)
    var mode = neural_descriptor_execute(provider, XN_MODE_OPERATION, [], [], [])
    if mode != expected:
        var compiled = String("fast") if mode == 0 else String("unknown")
        raise Error("RuntimeError: mojolearn: _mojolearn_x_neighbors was compiled for " + compiled
                    + ", but this call requested identical; rebuild the identical binding")
    var capabilities = neural_descriptor_execute(provider, xn_operation_id("capabilities"), [], [], [])
    var snapshot = PythonObject(alloc=CompiledBindingSnapshot(provider, binding, String(XN_ABI_NAME), mode, [capabilities]))
    compiled_binding_retain(binding, factory, String(XN_ABI_NAME), snapshot)
    return snapshot
