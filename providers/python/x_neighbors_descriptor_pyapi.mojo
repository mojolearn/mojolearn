# SPDX-License-Identifier: Apache-2.0
# One shared provider admission for the x_neighbors backend's existing packed
# CPU/GPU ABI. This introduces no provider class or estimator-specific transport.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.family_common import family_call
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_execute,
)
from providers.python.estimator_binding_pyapi import estimator_binding
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.x_neighbors_admission_pyapi import xn_is_module, xn_module, xn_capabilities, xn_admit_checked
from x_neighbors.descriptor_schema import xn_operation_id


def xn_admit(binding: PythonObject) raises -> PythonObject:
    # Called by the full public native operation, never by item/iteration loops.
    if not xn_is_module(binding):
        _ = xn_module(binding)
        return binding
    if getenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0") != "1":
        return binding
    return xn_admit_checked(binding)


def xn_pcs_draw_entry(binding: PythonObject) raises -> PythonObject:
    # Capture the legacy entry only on that existing route; the native route
    # reads its operation-local capability snapshot and never resolves it.
    var legacy = PythonObject(None)
    var available = False
    if xn_is_module(binding):
        legacy = neural_descriptor_entry(binding, "x_neighbors_kfeat_pcs_draw_idn", False)
        available = legacy is not None
    else:
        available = (xn_capabilities(binding) & 8) != 0
    if not available:
        raise Error("NotImplementedError: mojolearn PolynomialCountSketch: this binary lacks "
                    + "x_neighbors_kfeat_pcs_draw_idn (rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")
    return legacy


def xn_pcs_draw(binding: PythonObject, legacy: PythonObject, seed: Int,
                components: Int, total: Int, index_address: Int, sign_address: Int) raises:
    if legacy is not None:
        # Remaining normal-route migration debt, until shared mode admission
        # promotes the native provider. This is still CPython numerical dispatch.
        _ = family_call(legacy, Python.list(seed, components, total), index_address, sign_address)
        return
    var snapshot = binding.downcast_value_ptr[CompiledBindingSnapshot]()
    _ = neural_descriptor_execute(snapshot[].provider, xn_operation_id("pcs_draw"),
                                  [index_address, sign_address], [seed, components, total], [])


def xn_admit_arguments(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var source = args[0]
    var backend = context["backend_fields"]
    if (py_type(source) is backend["_HostBinding"] or py_type(source) is backend["_NoGpuBinding"]):
        # Public functional callers may still carry an import-connected host
        # facade. Resolve its already-loaded module without invoking __getattr__.
        source = estimator_binding(PythonObject(None), "_mojolearn_x_neighbors", context)
    var binding = xn_admit(source)
    var result = Python.list(binding)
    for i in range(1, len(args)):  # driver-loop: over fixed public method arguments
        _ = py_call(result.append, args[i])
    return result


def xn_execute(binding: PythonObject, name: String, addresses: PythonObject,
                ints: PythonObject, floats: PythonObject) raises:
    if xn_is_module(binding):
        var export_name = "xn_" + name
        if name == "ocsvm_alpha_init" or name == "unit_ff":
            export_name = "x_neighbors_" + name
        elif name == "ocsvm_resident":
            export_name = "x_neighbors_ocsvm_resident_idn"
        _ = py_call(neural_descriptor_entry(binding, export_name), addresses, ints, floats)
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
    var snapshot = binding.downcast_value_ptr[CompiledBindingSnapshot]()
    _ = neural_descriptor_execute(snapshot[].provider, xn_operation_id(name), a, p, f)
