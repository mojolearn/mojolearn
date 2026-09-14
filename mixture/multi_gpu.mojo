# SPDX-License-Identifier: Apache-2.0
"""Row-sharded GaussianMixture E-step. Whole rows move, never partial sums.

Every E-step quantity before the mean log likelihood is a pure function of
one sample row and the parameters: `X . P_k` is a GEMM whose output cell
contracts only row i of `X` (gemm fp32.v1 cell contract), the Mahalanobis
fold is one thread per row, the weighted log probability is one cell, the
logsumexp and its row max are one thread per row and the log responsibility
is one cell. So contiguous row ranges run the original `gmm_e_step` on their
owners, and their five per-row outputs are copied back into the original row
positions. The mean log likelihood (the convergence quantity) is then folded
on the root by the original one-thread ascending `meanll_kernel` over the
complete gathered `lse`. The M-step, the precision Cholesky, the convergence
test and the initialization stay on the root and are unchanged; the KMeans
initialization uses its own row-tile driver when MOJOLEARN_KMEANS_DEVICE_COUNT
is set by the same cooperative worker.

No floating-point value from one shard is ever combined with a value from
another shard. Traced stages are recorded on the root after the gather, in the
original stage order, so a traced multi-device fit writes the same card.
Sabotage probes are refused by the distributed path.
"""
from std.os import getenv
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from core.step_phase import STEP_PHASE_TIMERS
from mixture.checks.estep import (
    GMM_ELEM_TPB,
    GMM_ROW_TPB,
    gmm_e_step,
    gmm_estep_gemm_workspace_floats,
    gmm_estep_scratch_floats,
    meanll_kernel,
)
from mixture.checks.gmm_sabotage import GMM_SAB_NONE


def gmm_device_count() raises -> Int:
    var value = String(getenv("MOJOLEARN_GMM_DEVICE_COUNT"))
    if value == "":
        return 1
    var count = Int(value)
    if count < 1 or count > 64:
        raise Error("GaussianMixture device count must be in [1, 64]")
    if count > 1 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("multi-GPU GaussianMixture requires IDENTICAL numeric mode")
    return count


def gmm_row_shard_begin(n: Int, rank: Int, active: Int) -> Int:
    """First global row owned by `rank`: contiguous, ascending, ragged tail."""
    return n * rank // active


struct GmmEStepShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var means: DeviceBuffer[DType.float32]
    var prec: DeviceBuffer[DType.float32]
    var log_det_chol: DeviceBuffer[DType.float32]
    var log_weights: DeviceBuffer[DType.float32]
    var scratch: DeviceBuffer[DType.float32]
    var gws: DeviceBuffer[DType.float32]
    var mahal: DeviceBuffer[DType.float32]
    var wlp: DeviceBuffer[DType.float32]
    var rowmax: DeviceBuffer[DType.float32]
    var lse: DeviceBuffer[DType.float32]
    var logresp: DeviceBuffer[DType.float32]
    var meanll: DeviceBuffer[DType.float32]
    var begin: Int
    var rows: Int

    def __init__(out self, rank: Int, begin: Int, rows: Int, d: Int, ncomp: Int) raises:
        self.ctx = DeviceContext(device_id=rank)
        self.begin = begin
        self.rows = rows
        self.x = self.ctx.enqueue_create_buffer[DType.float32](rows * d)
        self.means = self.ctx.enqueue_create_buffer[DType.float32](ncomp * d)
        self.prec = self.ctx.enqueue_create_buffer[DType.float32](ncomp * d * d)
        self.log_det_chol = self.ctx.enqueue_create_buffer[DType.float32](ncomp)
        self.log_weights = self.ctx.enqueue_create_buffer[DType.float32](ncomp)
        self.scratch = self.ctx.enqueue_create_buffer[DType.float32](gmm_estep_scratch_floats(rows, d))
        self.gws = self.ctx.enqueue_create_buffer[DType.float32](gmm_estep_gemm_workspace_floats(rows, d))
        self.mahal = self.ctx.enqueue_create_buffer[DType.float32](rows * ncomp)
        self.wlp = self.ctx.enqueue_create_buffer[DType.float32](rows * ncomp)
        self.rowmax = self.ctx.enqueue_create_buffer[DType.float32](rows)
        self.lse = self.ctx.enqueue_create_buffer[DType.float32](rows)
        self.logresp = self.ctx.enqueue_create_buffer[DType.float32](rows * ncomp)
        self.meanll = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.ctx.synchronize()

    def __deinit__(deinit self):
        _ = self.meanll^
        _ = self.logresp^
        _ = self.lse^
        _ = self.rowmax^
        _ = self.wlp^
        _ = self.mahal^
        _ = self.gws^
        _ = self.scratch^
        _ = self.log_weights^
        _ = self.log_det_chol^
        _ = self.prec^
        _ = self.means^
        _ = self.x^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def gmm_e_step_dispatch(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut means: DeviceBuffer[DType.float32],
    prec_input: DeviceBuffer[DType.float32],
    linv_input: DeviceBuffer[DType.float32],
    mut log_det_chol: DeviceBuffer[DType.float32],
    mut log_weights: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    mut gws: DeviceBuffer[DType.float32],
    mut mahal: DeviceBuffer[DType.float32],
    mut wlp: DeviceBuffer[DType.float32],
    mut rowmax: DeviceBuffer[DType.float32],
    mut lse: DeviceBuffer[DType.float32],
    mut logresp: DeviceBuffer[DType.float32],
    mut meanll: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    ncomp: Int,
    mut trace: IdentityTrace,
    tag: String,
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    sabotage: Int = GMM_SAB_NONE,
) raises:
    """`gmm_e_step`, or its row-sharded form when MOJOLEARN_GMM_DEVICE_COUNT > 1.

    Same buffers, same stages, same postconditions. A single row, or a device
    count of one, runs the original entry unchanged.
    """
    var count = gmm_device_count()
    if count == 1 or n < 2:
        gmm_e_step(
            ctx, x, means, prec_input, linv_input, log_det_chol, log_weights,
            scratch, gws, mahal, wlp, rowmax, lse, logresp, meanll, n, d,
            ncomp, trace, tag, elem_tpb, row_tpb, sabotage,
        )
        return
    if sabotage != GMM_SAB_NONE:
        raise Error("parallel GaussianMixture E-step does not execute sabotage probes")
    comptime if STEP_PHASE_TIMERS:
        raise Error("parallel GaussianMixture cannot use process-global GEMM phase counters")
    if n <= 0 or d <= 0 or ncomp <= 0:
        raise Error("gmm_e_step_dispatch: n, d and n_components must all be positive")
    if n > 2147483647 // (ncomp if ncomp > d else d):
        raise Error("parallel GaussianMixture E-step exceeds signed 32-bit row indexing")
    if len(x) < n * d or len(means) < ncomp * d or len(prec_input) < ncomp * d * d:
        raise Error("gmm_e_step_dispatch: an input buffer is shorter than its shape")
    if len(mahal) < n * ncomp or len(wlp) < n * ncomp or len(logresp) < n * ncomp:
        raise Error("gmm_e_step_dispatch: an n x K output buffer is shorter than its shape")
    if len(rowmax) < n or len(lse) < n or len(meanll) < 1:
        raise Error("gmm_e_step_dispatch: a row output buffer is shorter than its shape")
    var active = min(count, n)
    # Root inputs must be complete before any owner reads a copy of them.
    ctx.synchronize()
    var shards = List[GmmEStepShard]()
    for rank in range(active):
        var begin = gmm_row_shard_begin(n, rank, active)
        var rows = gmm_row_shard_begin(n, rank + 1, active) - begin
        var shard = GmmEStepShard(rank, begin, rows, d, ncomp)
        var source_row = begin
        comptime if is_defined["MOJOLEARN_GMM_PARALLEL_SABOTAGE"]():
            # Check-only arm: every later owner reads its rows one row early.
            # The native gate must report DIVERGENT under this build.
            if rank > 0:
                source_row = begin - 1
        var xv = x.create_sub_buffer[DType.float32](source_row * d, rows * d)
        var mv = means.create_sub_buffer[DType.float32](0, ncomp * d)
        var pv = prec_input.create_sub_buffer[DType.float32](0, ncomp * d * d)
        var lv = log_det_chol.create_sub_buffer[DType.float32](0, ncomp)
        var wv = log_weights.create_sub_buffer[DType.float32](0, ncomp)
        xv.enqueue_copy_to(shard.x)
        mv.enqueue_copy_to(shard.means)
        pv.enqueue_copy_to(shard.prec)
        lv.enqueue_copy_to(shard.log_det_chol)
        wv.enqueue_copy_to(shard.log_weights)
        ctx.synchronize()
        _ = xv^
        _ = mv^
        _ = pv^
        _ = lv^
        _ = wv^
        shards.append(shard^)
    # Owners enqueue their original E-steps; no owner waits on another.
    for rank in range(active):
        ref shard = shards[rank]
        var quiet = IdentityTrace.disabled()
        gmm_e_step(
            shard.ctx, shard.x, shard.means, shard.prec, shard.prec,
            shard.log_det_chol, shard.log_weights, shard.scratch, shard.gws,
            shard.mahal, shard.wlp, shard.rowmax, shard.lse, shard.logresp,
            shard.meanll, shard.rows, d, ncomp, quiet, tag, elem_tpb, row_tpb,
            GMM_SAB_NONE,
        )
    # Gather exact bytes into original row positions, owner by owner.
    for rank in range(active):
        ref shard = shards[rank]
        var mh = mahal.create_sub_buffer[DType.float32](shard.begin * ncomp, shard.rows * ncomp)
        var wl = wlp.create_sub_buffer[DType.float32](shard.begin * ncomp, shard.rows * ncomp)
        var rm = rowmax.create_sub_buffer[DType.float32](shard.begin, shard.rows)
        var ls = lse.create_sub_buffer[DType.float32](shard.begin, shard.rows)
        var lr = logresp.create_sub_buffer[DType.float32](shard.begin * ncomp, shard.rows * ncomp)
        shard.mahal.enqueue_copy_to(mh)
        shard.wlp.enqueue_copy_to(wl)
        shard.rowmax.enqueue_copy_to(rm)
        shard.lse.enqueue_copy_to(ls)
        shard.logresp.enqueue_copy_to(lr)
        shard.ctx.synchronize()
        _ = mh^
        _ = wl^
        _ = rm^
        _ = ls^
        _ = lr^
    _ = shards^
    ctx.synchronize()
    # The original stage order: mahal, wlp, rowmax, lse, logresp, meanll.
    trace.record_device(ctx, tag + ".mahal", mahal, n * ncomp)
    trace.record_device(ctx, tag + ".wlp", wlp, n * ncomp)
    trace.record_device(ctx, tag + ".rowmax", rowmax, n)
    trace.record_device(ctx, tag + ".lse", lse, n)
    trace.record_device(ctx, tag + ".logresp", logresp, n * ncomp)
    # THE CONVERGENCE QUANTITY: the original one-block, one-thread ascending
    # fold over the complete gathered lse, on the root.
    ctx.enqueue_function[meanll_kernel](
        lse.unsafe_ptr(),
        meanll.unsafe_ptr(),
        Int32(n),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    trace.record_device(ctx, tag + ".meanll", meanll, 1)
