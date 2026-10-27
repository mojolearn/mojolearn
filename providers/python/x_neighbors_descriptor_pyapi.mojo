# SPDX-License-Identifier: Apache-2.0
# One shared provider admission for the x_neighbors backend's existing packed
# CPU/GPU ABI. This introduces no provider class or estimator-specific transport.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect,
    neural_descriptor_execute,
)
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT, XN_MODE_OPERATION, xn_operation_id


def xn_is_module(value: PythonObject) raises -> Bool:
    return py_type(value) is py_type(native_loaded_module("builtins"))


def xn_module(value: PythonObject) raises -> PythonObject:
    if xn_is_module(value):
        return value
    var owner = value.downcast_value_ptr[NeuralDescriptorProvider]()
    if owner[].name != String(XN_ABI_NAME):
        raise Error("x_neighbors requires its admitted shared native provider")
    return owner[].module


def xn_admit(binding: PythonObject) raises -> PythonObject:
    # Called by the full public native operation, never by item/iteration loops.
    if not xn_is_module(binding):
        _ = xn_module(binding)
        return binding
    if getenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0") != "1":
        return binding
    var factory = neural_descriptor_entry(binding, "x_neighbors_native_api_v1")
    var provider = neural_descriptor_connect(factory, String(XN_ABI_NAME), XN_OPERATION_COUNT)
    if neural_descriptor_execute(provider, XN_MODE_OPERATION, [], [], []) != 1:
        raise Error("x_neighbors native dispatch candidate requires identical mode")
    return provider


def xn_admit_arguments(args: PythonObject) raises -> PythonObject:
    var binding = xn_admit(args[0])
    var result = Python.list(binding)
    for i in range(1, len(args)):  # driver-loop: over fixed public method arguments
        _ = py_call(result.append, args[i])
    return result


def xn_execute(binding: PythonObject, name: String, addresses: PythonObject,
                ints: PythonObject, floats: PythonObject) raises:
    if xn_is_module(binding):
        _ = py_call(neural_descriptor_entry(binding, "xn_" + name), addresses, ints, floats)
        return
    var a = List[Int]()
    var p = List[Int]()
    var f = List[Float64]()
    for i in range(len(addresses)):  # small-loop(named operation's fixed address schema)
        a.append(Int(py=addresses[i]))
    for i in range(len(ints)):  # small-loop(named operation's fixed integer schema)
        p.append(Int(py=ints[i]))
    for i in range(len(floats)):  # small-loop(named operation's fixed scalar schema)
        f.append(Float64(py=floats[i]))
    _ = neural_descriptor_execute(binding, xn_operation_id(name), a, p, f)
