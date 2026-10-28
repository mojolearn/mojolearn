# SPDX-License-Identifier: Apache-2.0
# Direct native transport to the existing host linkage body. No Python objects,
# callbacks or arithmetic changes belong in this dispatcher.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.hostptr import i32_ptr
from core.mojolearn_solver_host_impl import linkage_fit_host_into
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from hierarchy.solver_descriptor_schema import (
    HIERARCHY_SOLVER_ABI, HIERARCHY_SOLVER_MODE, HIERARCHY_SOLVER_CAPABILITIES,
    HIERARCHY_SOLVER_CAP_LINKAGE, hierarchy_solver_validate_call,
    hierarchy_solver_validate_buffers,
)


def hierarchy_solver_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                             ff: Int, nf: Int, result: Int, context: Int,
                             sink: NativeDescriptorError) abi("C") -> Int:
    try:
        hierarchy_solver_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var answer = 0
        if operation == HIERARCHY_SOLVER_MODE:
            answer = GLOBAL_NUMERIC_MODE
        elif operation == HIERARCHY_SOLVER_CAPABILITIES:
            answer = HIERARCHY_SOLVER_CAP_LINKAGE
        else:
            hierarchy_solver_validate_buffers(a, p)
            linkage_fit_host_into(a[0], i32_ptr(a[1]), i32_ptr(a[2]), i32_ptr(a[3]),
                                  p[0], p[1], p[2], p[3], p[4] != 0)
            # The existing host solver writes -1 to info[0]: it runs Prim's
            # ordered MST, not the GPU Boruvka rounds reported by that slot.
            answer = -1
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct HierarchySolverHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HIERARCHY_SOLVER_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(hierarchy_solver_host_v1)


# PyCapsule borrows both table and name: the module image owns them for life.
comptime HIERARCHY_SOLVER_HOST_API = _Global[StorageType=HierarchySolverHostAPI,
    name="MojoHierarchySolverHostAPIv1", init_fn=HierarchySolverHostAPI.__init__]
