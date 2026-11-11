# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the solver operations table
(core/solver_entry_table.mojo): the host coordinate-descent bodies of
core/mojolearn_solver_host_impl.mojo, the ones the CPU binding's Python
exports (bindings/_mojolearn_solver_host.mojo `cd_fit`, `cd_predict`) call.
The table entry is called with the GIL already released by its caller.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.solver_entry_table import SV_OP_CD_FIT, SV_OP_CD_PREDICT, sv_ops_check_spans
from core.mojolearn_solver_host_impl import cd_fit_host_into, cd_predict_host_into


def solver_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        sv_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == SV_OP_CD_FIT:
            var has_sw = p[5] != 0
            if has_sw and a[4] == 0:
                raise Error("cd_fit: has_sample_weight needs the weight address as params[10]")
            var cp = f32_ptr(a[2])
            var ip = f32_ptr(a[3])
            r[] = cd_fit_host_into(a[0], a[1], a[4] if has_sw else 0, cp, ip, p[0], p[1], p[2] != 0, p[3],
                                   Float32(f[0]), Float32(f[1]), Float32(f[2]), p[4] != 0, has_sw, p[6] != 0)
        else:
            var op = f32_ptr(a[2])
            cd_predict_host_into(a[0], a[1], op, p[0], p[1], Float32(f[0]))
            r[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SolverOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(solver_ops_host_v1)


comptime _OPS_SLOT = "MojolearnSolverOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSolverOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime SOLVER_OPS_HOST_API = SharedSlot[SolverOpsHostAPI, _OPS_SLOT]
