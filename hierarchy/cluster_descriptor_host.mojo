# SPDX-License-Identifier: Apache-2.0
# Native descriptor over the unchanged x_cluster host numerical owner.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from x_cluster.host.host_ops import HostOps
from hierarchy.cluster_descriptor_impl import hierarchy_cluster_fit
from hierarchy.cluster_descriptor_schema import (
    HIERARCHY_CLUSTER_ABI, HIERARCHY_CLUSTER_MODE, HIERARCHY_CLUSTER_CAPABILITIES,
    HIERARCHY_CLUSTER_CAP_FIT, HIERARCHY_CLUSTER_CAP_DEVICE,
    hierarchy_cluster_validate_call,
)


def hierarchy_cluster_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                                ff: Int, nf: Int, result: Int, context: Int,
                                sink: NativeDescriptorError) abi("C") -> Int:
    try:
        hierarchy_cluster_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var answer = 0
        if operation == HIERARCHY_CLUSTER_MODE:
            answer = GLOBAL_NUMERIC_MODE
        elif operation == HIERARCHY_CLUSTER_CAPABILITIES:
            answer = HIERARCHY_CLUSTER_CAP_FIT
        else:
            # The same operations owner as x_cluster_call, including its
            # context, scratch ownership and synchronization on destruction.
            var ops = HostOps()
            hierarchy_cluster_fit(ops, a, p, ff)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct HierarchyClusterHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HIERARCHY_CLUSTER_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(hierarchy_cluster_host_v1)


# The extension image owns the table and name borrowed by PyCapsule.
comptime HIERARCHY_CLUSTER_HOST_API = _Global[StorageType=HierarchyClusterHostAPI,
    name="MojoHierarchyClusterHostAPIv1", init_fn=HierarchyClusterHostAPI.__init__]
