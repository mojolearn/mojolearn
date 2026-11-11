# SPDX-License-Identifier: Apache-2.0
# The common provider owns the admitted native factory, capsule and module.
# Only capsule discovery uses a CPython entry. Computation uses the descriptor
# directly, without a packed numerical fallback or a second mode selection.
from std.python import PythonObject
from providers.python.compiled_binding_snapshot_pyapi import (
    CompiledBindingSnapshot, compiled_binding_retained, compiled_binding_retain,
)
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute,
)
from hierarchy.cluster_descriptor_schema import (
    HIERARCHY_CLUSTER_ABI, HIERARCHY_CLUSTER_OPERATIONS,
    HIERARCHY_CLUSTER_MODE, HIERARCHY_CLUSTER_CAPABILITIES,
    HIERARCHY_CLUSTER_CAP_FIT, HIERARCHY_CLUSTER_CAP_DEVICE,
)


def hierarchy_cluster_admit(binding: PythonObject) raises -> PythonObject:
    # cfb_agglo retains the model's numeric_mode selection (or explicit parallel
    # graph module). Its existing loader checks mode; snapshot this table too.
    var factory = neural_descriptor_entry(binding, "hierarchy_cluster_native_api_v1")
    # Mode and capabilities are compile-time constants of the loaded image:
    # reuse the snapshot admitted for this exact module and live factory entry
    # (lane perf-provider-cache); a replaced one runs the admission again.
    var retained = compiled_binding_retained(binding, factory, String(HIERARCHY_CLUSTER_ABI))
    if retained is not None:
        return retained
    var provider = neural_descriptor_connect(factory, String(HIERARCHY_CLUSTER_ABI), HIERARCHY_CLUSTER_OPERATIONS)
    var mode = neural_descriptor_execute(provider, HIERARCHY_CLUSTER_MODE, [], [], [])
    if mode != 0 and mode != 1:
        raise Error("hierarchy cluster provider: unsupported compiled numeric mode")
    var capabilities = neural_descriptor_execute(provider, HIERARCHY_CLUSTER_CAPABILITIES, [], [], [])
    if capabilities != HIERARCHY_CLUSTER_CAP_FIT and capabilities != (HIERARCHY_CLUSTER_CAP_FIT | HIERARCHY_CLUSTER_CAP_DEVICE):
        raise Error("hierarchy cluster provider: fit capability is unavailable")
    var snapshot = PythonObject(alloc=CompiledBindingSnapshot(provider, binding, String(HIERARCHY_CLUSTER_ABI), mode, [capabilities]))
    compiled_binding_retain(binding, factory, String(HIERARCHY_CLUSTER_ABI), snapshot)
    return snapshot
