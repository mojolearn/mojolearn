# SPDX-License-Identifier: Apache-2.0
# The common native provider retains the exact factory, capsule and module.
# Only capsule admission uses the extension door; linkage executes by the
# native descriptor, with no numerical CPython fallback or runtime mode switch.
from std.python import PythonObject
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute,
)
from hierarchy.solver_descriptor_schema import (
    HIERARCHY_SOLVER_ABI, HIERARCHY_SOLVER_OPERATIONS, HIERARCHY_SOLVER_LINKAGE,
    HIERARCHY_SOLVER_MODE, HIERARCHY_SOLVER_CAPABILITIES, HIERARCHY_SOLVER_CAP_LINKAGE,
    HIERARCHY_SOLVER_CAP_DEVICE,
)


def hierarchy_solver_admit(binding: PythonObject) raises -> PythonObject:
    # cfb_agglo keeps the existing solver selection: the current default-mode
    # module (or an explicit parallel-plan module), not the model's mode. That
    # selection already checks its legacy mode readback. Capture this table's
    # own metadata once and keep the selected image alive through execution.
    var factory = neural_descriptor_entry(binding, "hierarchy_solver_native_api_v1")
    var provider = neural_descriptor_connect(factory, String(HIERARCHY_SOLVER_ABI), HIERARCHY_SOLVER_OPERATIONS)
    var mode = neural_descriptor_execute(provider, HIERARCHY_SOLVER_MODE, [], [], [])
    if mode != 0 and mode != 1:
        raise Error("hierarchy solver provider: unsupported compiled numeric mode")
    var capabilities = neural_descriptor_execute(provider, HIERARCHY_SOLVER_CAPABILITIES, [], [], [])
    if capabilities != HIERARCHY_SOLVER_CAP_LINKAGE and capabilities != (HIERARCHY_SOLVER_CAP_LINKAGE | HIERARCHY_SOLVER_CAP_DEVICE):
        raise Error("hierarchy solver provider: linkage capability is unavailable")
    return PythonObject(alloc=CompiledBindingSnapshot(provider, binding, String(HIERARCHY_SOLVER_ABI), mode, [capabilities]))


def hierarchy_solver_linkage(snapshot: PythonObject, addresses: List[Int],
                             params: List[Int]) raises -> Int:
    var owner = snapshot.downcast_value_ptr[CompiledBindingSnapshot]()
    if owner[].abi != String(HIERARCHY_SOLVER_ABI):
        raise Error("hierarchy linkage requires its admitted solver provider")
    var result = neural_descriptor_execute(owner[].provider, HIERARCHY_SOLVER_LINKAGE,
                                          addresses, params, [])
    _ = snapshot
    return result
