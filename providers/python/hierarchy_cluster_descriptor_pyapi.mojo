# SPDX-License-Identifier: Apache-2.0
# The common provider owns the admitted native factory, capsule and module.
# Only capsule discovery uses a CPython entry. Computation uses the descriptor
# directly, without a packed numerical fallback or a second mode selection.
from std.python import PythonObject
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute,
)
from hierarchy.cluster_descriptor_schema import (
    HIERARCHY_CLUSTER_ABI, HIERARCHY_CLUSTER_OPERATIONS, HIERARCHY_CLUSTER_FIT,
    HIERARCHY_CLUSTER_MODE, HIERARCHY_CLUSTER_CAPABILITIES,
    HIERARCHY_CLUSTER_CAP_FIT, HIERARCHY_CLUSTER_CAP_DEVICE,
)


def hierarchy_cluster_admit(binding: PythonObject) raises -> PythonObject:
    # cfb_agglo retains the model's numeric_mode selection (or explicit parallel
    # graph module). Its existing loader checks mode; snapshot this table too.
    var factory = neural_descriptor_entry(binding, "hierarchy_cluster_native_api_v1")
    var provider = neural_descriptor_connect(factory, String(HIERARCHY_CLUSTER_ABI), HIERARCHY_CLUSTER_OPERATIONS)
    var mode = neural_descriptor_execute(provider, HIERARCHY_CLUSTER_MODE, [], [], [])
    if mode != 0 and mode != 1:
        raise Error("hierarchy cluster provider: unsupported compiled numeric mode")
    var capabilities = neural_descriptor_execute(provider, HIERARCHY_CLUSTER_CAPABILITIES, [], [], [])
    if capabilities != HIERARCHY_CLUSTER_CAP_FIT and capabilities != (HIERARCHY_CLUSTER_CAP_FIT | HIERARCHY_CLUSTER_CAP_DEVICE):
        raise Error("hierarchy cluster provider: fit capability is unavailable")
    return PythonObject(alloc=CompiledBindingSnapshot(provider, binding, String(HIERARCHY_CLUSTER_ABI), mode, [capabilities]))


def hierarchy_cluster_execute(snapshot: PythonObject, addresses: List[Int],
                              params: List[Int], floats: List[Float64]) raises:
    var owner = snapshot.downcast_value_ptr[CompiledBindingSnapshot]()
    if owner[].abi != String(HIERARCHY_CLUSTER_ABI):
        raise Error("hierarchy fit requires its admitted cluster provider")
    _ = neural_descriptor_execute(owner[].provider, HIERARCHY_CLUSTER_FIT, addresses, params, floats)
    _ = snapshot
