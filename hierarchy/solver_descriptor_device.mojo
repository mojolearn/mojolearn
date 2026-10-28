# SPDX-License-Identifier: Apache-2.0
# Direct native transport to the existing device linkage owner, preserving the
# solver module's process context and completion boundary. No Python arithmetic.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from hierarchy.estimator import linkage_fit_host
from hierarchy.solver_descriptor_schema import (
    HIERARCHY_SOLVER_ABI, HIERARCHY_SOLVER_MODE, HIERARCHY_SOLVER_CAPABILITIES,
    HIERARCHY_SOLVER_CAP_LINKAGE, HIERARCHY_SOLVER_CAP_DEVICE,
    hierarchy_solver_validate_call, hierarchy_solver_validate_buffers,
)


# Exactly bindings/_mojolearn_solver.mojo's slot; no second device context.
comptime _DEVCTX_SLOT = "MojoSolverContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoSolverContextFast"


def hierarchy_solver_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
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
            answer = HIERARCHY_SOLVER_CAP_LINKAGE | HIERARCHY_SOLVER_CAP_DEVICE
        else:
            hierarchy_solver_validate_buffers(a, p)
            var ctx = process_ctx[_DEVCTX_SLOT]()
            answer = linkage_fit_host(ctx, f32_ptr(a[0]), i32_ptr(a[1]), i32_ptr(a[2]),
                                      i32_ptr(a[3]), p[0], p[1], p[2], p[3], p[4] != 0)
            ctx.synchronize()
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct HierarchySolverDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HIERARCHY_SOLVER_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(hierarchy_solver_device_v1)


# PyCapsule borrows both table and name: the module image owns them for life.
comptime HIERARCHY_SOLVER_DEVICE_API = _Global[StorageType=HierarchySolverDeviceAPI,
    name="MojoHierarchySolverDeviceAPIv1", init_fn=HierarchySolverDeviceAPI.__init__]
