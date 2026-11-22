# SPDX-License-Identifier: Apache-2.0
# One shared provider admission for the x_neighbors backend's existing packed
# CPU/GPU ABI. This introduces no provider class or estimator-specific transport.
from providers.python.classical_native_dispatch import CLASSICAL_NATIVE_DISPATCH
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_common_pack import family_call
from providers.python.neural_descriptor_pyapi_door import (
    neural_descriptor_entry, neural_descriptor_execute,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.compiled_binding_snapshot_pyapi_door import CompiledBindingSnapshot
from providers.python.x_neighbors_admission_pyapi_door import xn_is_module, xn_module, xn_capabilities, xn_admit_checked
from x_neighbors.descriptor_schema import xn_operation_id
from bindings.py_optional import py_or_none, py_present


def xn_admit(binding: PythonObject) raises -> PythonObject:
    # Called by the full public native operation, never by item/iteration loops.
    if not xn_is_module(binding):
        _ = xn_module(binding)
        return binding
    # A module (fast mode, explicit internal callers) keeps its by-name
    # entries unless the build admits it natively (CLASSICAL_NATIVE_DISPATCH,
    # the B arm); identical public calls arrive as the admitted snapshot.
    if not CLASSICAL_NATIVE_DISPATCH:
        return binding
    return xn_admit_checked(binding)


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
        elif name == "ocsvm_labels" or name == "nc_labels" or name == "nc_proba" or name == "lp_dense_proba":
            export_name = "x_neighbors_" + name
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
