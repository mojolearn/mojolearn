# SPDX-License-Identifier: Apache-2.0
"""The device column of the solver operations table
(core/solver_entry_table.mojo) and the coordinate-descent bodies the GPU
binding's Python exports (bindings/_mojolearn_solver.mojo `cd_fit`,
`cd_predict`) share with it: the same checks, the same kernels, the same
words. The table entry is called with the GIL already released by its
caller; the Python exports release it around the same bodies.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.solver_entry_table import SV_OP_CD_FIT, SV_OP_CD_PREDICT, sv_ops_check_spans
from solver.estimator import cd_fit_host, cd_predict_host

#: The solver binding's ONE process-lifetime DeviceContext
#: (core/neural_context.mojo, lane/devctx-lifetime): a context per call
#: exhausts Metal command queues. Moved here from the binding so the table
#: entry and the Python exports share the one context.
comptime SOLVER_DEVCTX_SLOT = "MojolearnSolverContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSolverContextFastV3"


def sv_cd_fit_device_run(x: Int, y: Int, coef: Int, info: Int, sample_weight: Int, n_rows: Int, n_cols: Int,
                         fit_intercept: Bool, epochs: Int, alpha: Float32, l1_ratio: Float32, tol: Float32,
                         shuffle: Bool, has_sw: Bool, row_major: Bool) raises -> Int:
    """`cdFit` for Lasso / ElasticNet: n_cols float32 coefficients at `coef`,
    the intercept at `info[0]`; returns the epochs run. Without weights the
    label words stand in for the weight pointer and are never read as
    weights (the binding's contract)."""
    var xp = f32_ptr(x)
    var yp = f32_ptr(y)
    var cp = f32_ptr(coef)
    var ip = f32_ptr(info)
    var swp = yp
    if has_sw:
        if sample_weight == 0:
            raise Error("cd_fit: has_sample_weight needs the weight address as params[10]")
        swp = f32_ptr(sample_weight)
    var ctx = process_ctx[SOLVER_DEVCTX_SLOT]()
    var n_iter = cd_fit_host(
        ctx, xp, yp, cp, ip, n_rows, n_cols, fit_intercept, epochs, alpha,
        l1_ratio, tol, shuffle, has_sw, swp, row_major,
    )
    ctx.synchronize()
    return n_iter


def sv_cd_predict_device_run(x: Int, coef: Int, dst: Int, n_rows: Int, n_cols: Int, intercept: Float32) raises:
    """`cdPredict` -> `linearRegH`: `pred = X coef + intercept` on the device."""
    var xp = f32_ptr(x)
    var cp = f32_ptr(coef)
    var op = f32_ptr(dst)
    var ctx = process_ctx[SOLVER_DEVCTX_SLOT]()
    cd_predict_host(ctx, xp, cp, op, n_rows, n_cols, intercept)
    ctx.synchronize()


def solver_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        sv_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == SV_OP_CD_FIT:
            r[] = sv_cd_fit_device_run(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2] != 0, p[3],
                                       Float32(f[0]), Float32(f[1]), Float32(f[2]), p[4] != 0, p[5] != 0, p[6] != 0)
        else:
            sv_cd_predict_device_run(a[0], a[1], a[2], p[0], p[1], Float32(f[0]))
            r[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SolverOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(solver_ops_device_v1)


comptime _OPS_SLOT = "MojolearnSolverOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSolverOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime SOLVER_OPS_DEVICE_API = SharedSlot[SolverOpsDeviceAPI, _OPS_SLOT]
