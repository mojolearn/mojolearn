# SPDX-License-Identifier: Apache-2.0
"""Apple FAST neural experiments for the byte LM (lane afn-lm, 2026-10-03).

`BYTE_LM_FAST_APPLE` is true only on a FAST build (no
`-D MOJOLEARN_NUMERIC_IDENTICAL`) whose host is Apple silicon. Every
experiment below is that AND its own `-D MOJOLEARN_AFN_LM_<NAME>` define
(or `-D MOJOLEARN_AFN_LM_ALL`), default OFF, so an IDENTICAL build, and a
FAST build on NVIDIA or AMD, compiles main's code unchanged.
docs/apple-fast/notes/neural-lm.md lists what each one changes.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import identical_div, identical_exp, identical_fmax, identical_log
from core.device_scan import DeviceScanScratch, NONFINITE_NONE
from core.step_phase import step_count_d2h, step_count_h2d, step_count_launch, step_count_sync
from training.checks.loss_contract import CE_NEG_INF_BITS

comptime BYTE_LM_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)


comptime _AFN_LM_ALL = is_defined["MOJOLEARN_AFN_LM_ALL"]()

#: -D MOJOLEARN_AFN_LM_NOSYNC: ONE host wait per train step. The ids go up
#: from the caller's own List (no staging host buffers, no waits), the loss
#: stays on the device until the end, the update is the shadow copy plus
#: the Adam kernel with no entry scans and no wait, and every validation
#: (loss, gradients, parameters, moments, v >= 0) is ONE status kernel with
#: integer atomic minimums, read back with the loss in one copy and one wait.
# T01 AFN26 alias: not tested in this campaign; preserves the existing arm.
comptime AFN_LM_NOSYNC = BYTE_LM_FAST_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_NOSYNC"]() or _AFN_LM_ALL
    or is_defined["MOJOLEARN_AFN26_LM_NOSYNC"]()
)
#: -D MOJOLEARN_AFN_LM_BWD_NOSYNC: the block backward's RMSNorm stage drops
#: the three host waits the non-IDENTICAL tiers kept there ("old fences",
#: transformer_backward.mojo `_bwd_rms_norm_kernels`): 6 per layer.
# F08/default M3 Ultra 2026-10-06: promote this FAST Apple switch only.
# Whole train-checkpoint-refusal task [2,7,24,3,1,8,40,2,513], 12 steps:
# candidate/baseline trajectory 0.8766, first step 0.9357; heldout loss
# 6.231029987335205 equal, resume_error=0 on both, refusal/recovery captured.
# One excluded warmup and one scored trajectory; one shape, no universal
# hardware claim. Caller67d0efb29; reused artifact hashes/builds in
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F08/default.
# Identity/compile receipts reused, not rerun. Active frozen binaries unchanged.
# Explicit OFF permits rollback and future baseline captures.
comptime AFN_LM_BWD_NOSYNC = BYTE_LM_FAST_APPLE and (
    not is_defined["MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF"]()
)
#: -D MOJOLEARN_AFN_LM_BWD_FUSE: the block backward takes IDENTICAL's fused
#: Apple routes on FAST (fused SiLU-gate VJP, norm2 + residual fused, the
#: shipped norm kernel arm), which FAST had been routed around.
# F08/fused M3 2026-10-06: B/A trajectory0.9494, first-step0.8546 on the
# same 12-step/513-vocab task; heldout/resume unchanged. Small single-shape
# gain remains tentative; retain OFF. Same caller/build provenance and
# evidence root as F08/default above, results/F08/fused; no identity retest.
# T05 AFN26 alias: not tested in this campaign; historical evidence above stands.
comptime AFN_LM_BWD_FUSE = BYTE_LM_FAST_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_BWD_FUSE"]() or _AFN_LM_ALL
    or is_defined["MOJOLEARN_AFN26_LM_BWD_FUSE"]()
)
#: -D MOJOLEARN_AFN_LM_PARAM_VIEWS: every block's nine weights are views of
#: the flat `param` and its nine weight gradients views of the flat `grad`,
#: re-bound each step: no unpack and no pack launch (2 per layer), and the
#: 18 per-layer separately allocated buffers are dropped.
# F08/views M3 2026-10-06: B/A trajectory0.9765 but first-step1.0455;
# heldout/resume unchanged. Mixed small single-sample gain; retain OFF.
# Same 12-step task and provenance, evidence results/F08/views; promoting
# NOSYNC does not claim the unmeasured combination with views or fusion.
# T03 AFN26 alias: not tested in this campaign; historical evidence above stands.
comptime AFN_LM_PARAM_VIEWS = BYTE_LM_FAST_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_PARAM_VIEWS"]() or _AFN_LM_ALL
    or is_defined["MOJOLEARN_AFN26_LM_PARAM_VIEWS"]()
)
#: -D MOJOLEARN_AFN_LM_HEAD_FUSE: softmax, cross entropy, the mean loss and
#: the logits gradient in ONE row kernel after the head GEMM (plus a one-cell
#: reset), replacing the CE forward (refusal scan, targets download, two
#: waits, its kernels) and the CE backward's two kernels.
# T02 AFN26 alias: not tested in this campaign; default remains disabled.
comptime AFN_LM_HEAD_FUSE = BYTE_LM_FAST_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_HEAD_FUSE"]() or _AFN_LM_ALL
    or is_defined["MOJOLEARN_AFN26_LM_HEAD_FUSE"]()
)

#: Status cells of the NOSYNC step (in `DeviceScanScratch.part`).
comptime AFN_ST_LOSS = 0
comptime AFN_ST_GRAD = 1
comptime AFN_ST_PARAM = 2
comptime AFN_ST_M = 3
comptime AFN_ST_V = 4
comptime AFN_ST_VNEG = 5
comptime AFN_ST_CELLS = 8
comptime AFN_TPB = 256
# T11: not tested. Power-of-two blocks preserve the complete reduction tree;
# fewer threads trade vocabulary strides for occupancy, more trade occupancy
# for fewer strides. This CE-only choice leaves status scan geometry unchanged.
# These controls do not enable HEAD_FUSE; use one geometry define per arm.
comptime AFN_LM_CE_TPB = (
    128 if BYTE_LM_FAST_APPLE and is_defined["MOJOLEARN_AFN26_LM_CE_BLOCK128"]()
    else 512 if BYTE_LM_FAST_APPLE and is_defined["MOJOLEARN_AFN26_LM_CE_BLOCK512"]()
    else 256
)
comptime AFN_SCAN_MAX_BLOCKS = 1024


def afn_reset_kernel(
    status: MutPointer[Int32, MutAnyOrigin],
    loss: MutPointer[Float32, MutAnyOrigin],
    reset_status: Int32,
    reset_loss: Int32,
):
    """Fixed size (AFN_ST_CELLS threads, one block): status cells to
    NONFINITE_NONE and/or the loss cell to +0.0."""
    var t = Int(thread_idx.x)
    if reset_status != 0 and t < AFN_ST_CELLS:
        status.unsafe_store(t, NONFINITE_NONE)
    if reset_loss != 0 and t == 0:
        loss.unsafe_store(0, Float32(0.0))


@always_inline
def _afn_nonfinite(x: Float32) -> Bool:
    return (bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000)


@always_inline
def _afn_negative(x: Float32) -> Bool:
    var bits = bitcast[DType.uint32](x)
    return (bits & UInt32(0x80000000)) != UInt32(0) and (bits & UInt32(0x7FFFFFFF)) != UInt32(0)


def afn_step_status_kernel(
    status: MutPointer[Int32, MutAnyOrigin],
    loss: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    param: MutPointer[Float32, MutAnyOrigin],
    m: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Every validation of the step in one grid-stride pass: each thread
    keeps the FIRST failing index of its own stride per predicate and
    folds it with an integer atomic minimum, so each cell ends holding
    the smallest failing index (the index the per-buffer scans report),
    or NONFINITE_NONE. Integer minimum is order-free."""
    var n = Int(n_in)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    if gid == 0 and _afn_nonfinite(loss.unsafe_load(0)):
        _ = Atomic[DType.int32].min(status + AFN_ST_LOSS, Int32(0))
    var bg = NONFINITE_NONE
    var bp = NONFINITE_NONE
    var bm = NONFINITE_NONE
    var bv = NONFINITE_NONE
    var bn = NONFINITE_NONE
    var i = gid
    while i < n:
        if bg == NONFINITE_NONE and _afn_nonfinite(grad.unsafe_load(i)):
            bg = Int32(i)
        if bp == NONFINITE_NONE and _afn_nonfinite(param.unsafe_load(i)):
            bp = Int32(i)
        if bm == NONFINITE_NONE and _afn_nonfinite(m.unsafe_load(i)):
            bm = Int32(i)
        var vi = v.unsafe_load(i)
        if bv == NONFINITE_NONE and _afn_nonfinite(vi):
            bv = Int32(i)
        if bn == NONFINITE_NONE and _afn_negative(vi):
            bn = Int32(i)
        i += stride
    if bg != NONFINITE_NONE:
        _ = Atomic[DType.int32].min(status + AFN_ST_GRAD, bg)
    if bp != NONFINITE_NONE:
        _ = Atomic[DType.int32].min(status + AFN_ST_PARAM, bp)
    if bm != NONFINITE_NONE:
        _ = Atomic[DType.int32].min(status + AFN_ST_M, bm)
    if bv != NONFINITE_NONE:
        _ = Atomic[DType.int32].min(status + AFN_ST_V, bv)
    if bn != NONFINITE_NONE:
        _ = Atomic[DType.int32].min(status + AFN_ST_VNEG, bn)


def afn_ce_fused_kernel(
    loss: MutPointer[Float32, MutAnyOrigin],
    dlogits: MutPointer[Float32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
    divisor: Float32,
):
    """One threadgroup per row: row max, the softmax denominator, the row's
    negative log likelihood folded into the mean loss (f32 atomic add, the
    one reassociated sum), and `dlogits = (softmax - onehot) / divisor`.
    The same identical_exp/identical_log/identical_div as the IDENTICAL CE,
    so only the fold order of the denominator and the loss mean moves.
    `dlogits` must not alias `logits` (the trainer's `ce_dlogits` is a
    view of `ce_expo`, never of `logits`)."""
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var vocab = Int(vocab_in)
    var base = row * vocab
    var red = stack_allocation[
        AFN_LM_CE_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var mx = bitcast[DType.float32](CE_NEG_INF_BITS)
    var j = tid
    while j < vocab:
        mx = identical_fmax(mx, logits.unsafe_load(base + j))
        j += AFN_LM_CE_TPB
    red.unsafe_store(tid, mx)
    barrier()
    var active = AFN_LM_CE_TPB // 2
    while active > 0:
        if tid < active:
            red.unsafe_store(tid, identical_fmax(red.unsafe_load(tid), red.unsafe_load(tid + active)))
        barrier()
        active = active // 2
    var rmax = red.unsafe_load(0)
    barrier()
    var s = Float32(0.0)
    j = tid
    while j < vocab:
        s += identical_exp(logits.unsafe_load(base + j) - rmax)
        j += AFN_LM_CE_TPB
    red.unsafe_store(tid, s)
    barrier()
    active = AFN_LM_CE_TPB // 2
    while active > 0:
        if tid < active:
            red.unsafe_store(tid, red.unsafe_load(tid) + red.unsafe_load(tid + active))
        barrier()
        active = active // 2
    var denom = red.unsafe_load(0)
    var target = Int(targets.unsafe_load(row))
    j = tid
    while j < vocab:
        var p = identical_div(identical_exp(logits.unsafe_load(base + j) - rmax), denom)
        var t = Float32(1.0) if j == target else Float32(0.0)
        dlogits.unsafe_store(base + j, identical_div(p - t, divisor))
        j += AFN_LM_CE_TPB
    if tid == 0:
        var nll = identical_log(denom) - (logits.unsafe_load(base + target) - rmax)
        _ = Atomic.fetch_add(loss, identical_div(nll, divisor))


def afn_reset(ctx: DeviceContext, mut status: DeviceBuffer[DType.int32],
              mut loss: DeviceBuffer[DType.float32], reset_status: Bool,
              reset_loss: Bool) raises:
    step_count_launch()
    ctx.enqueue_function[afn_reset_kernel](
        status.unsafe_ptr(), loss.unsafe_ptr(),
        Int32(1 if reset_status else 0), Int32(1 if reset_loss else 0),
        grid_dim=(1, 1, 1), block_dim=(AFN_ST_CELLS, 1, 1),
    )


def afn_ce_fused(ctx: DeviceContext, mut loss: DeviceBuffer[DType.float32],
                 mut dlogits: DeviceBuffer[DType.float32],
                 mut logits: DeviceBuffer[DType.float32],
                 mut targets: DeviceBuffer[DType.int32],
                 mut status: DeviceBuffer[DType.int32],
                 rows: Int, vocab: Int) raises:
    """HEAD_FUSE: reset the loss cell, then the fused row kernel. Every
    target was validated on the host (`byte_validate_tokens`), so no row is
    ignored and the MEAN divisor is `rows`, as `ce_divisor` gives it."""
    if rows < 1 or vocab < 1:
        raise Error("byte LM: fused CE needs rows and vocab")
    afn_reset(ctx, status, loss, False, True)
    step_count_launch()
    ctx.enqueue_function[afn_ce_fused_kernel](
        loss.unsafe_ptr(), dlogits.unsafe_ptr(), logits.unsafe_ptr(),
        targets.unsafe_ptr(), Int32(vocab), Float32(rows),
        grid_dim=(rows, 1, 1), block_dim=(AFN_LM_CE_TPB, 1, 1),
    )


def afn_upload_ids(ctx: DeviceContext, mut ids_dev: DeviceBuffer[DType.int32],
                   mut targets_dev: DeviceBuffer[DType.int32],
                   ids: List[Int32], batch: Int, length: Int) raises:
    """NOSYNC: inputs `ids[b, 0:L]` and targets `ids[b, 1:L+1]` copied from
    the caller's List straight into the device buffers, two copies per row,
    no staging buffer, no wait. The caller keeps `ids` alive past the step's
    one wait (the List is borrowed for the whole resident step, and every
    failure path waits inside `_byte_recover` before it returns)."""
    if len(ids) != batch * (length + 1):
        raise Error("byte LM: token count differs from row-major [batch,length+1]")
    var src = ids.unsafe_ptr()
    for b in range(batch):
        var vi = ids_dev.create_sub_buffer[DType.int32](b * length, length)
        var vt = targets_dev.create_sub_buffer[DType.int32](b * length, length)
        step_count_h2d()
        ctx.enqueue_copy(dst_buf=vi, src_ptr=src + b * (length + 1))
        step_count_h2d()
        ctx.enqueue_copy(dst_buf=vt, src_ptr=src + b * (length + 1) + 1)


def afn_status_view(mut scan: DeviceScanScratch) raises -> DeviceBuffer[DType.int32]:
    return scan.part.create_sub_buffer[DType.int32](0, AFN_ST_CELLS)


def afn_step_finish(ctx: DeviceContext, mut scan: DeviceScanScratch,
                    mut loss: DeviceBuffer[DType.float32],
                    mut grad: DeviceBuffer[DType.float32],
                    mut param: DeviceBuffer[DType.float32],
                    mut m: DeviceBuffer[DType.float32],
                    mut v: DeviceBuffer[DType.float32],
                    n: Int) raises -> Float32:
    """NOSYNC: the one status launch, ONE readback (8 status cells and the
    loss, into the scratch's pinned host mirror) and the step's ONE wait.
    Raises the message of the FIRST failing check in the order the
    synchronous step checks them (loss, gradients, parameters, first
    moments, second moments, negative second moment); the caller's
    shadow point makes that a rollback. Returns the loss."""
    var status = afn_status_view(scan)
    var blocks = (n + AFN_TPB - 1) // AFN_TPB
    if blocks > AFN_SCAN_MAX_BLOCKS:
        blocks = AFN_SCAN_MAX_BLOCKS
    if blocks < 1:
        blocks = 1
    step_count_launch()
    ctx.enqueue_function[afn_step_status_kernel](
        status.unsafe_ptr(), loss.unsafe_ptr(), grad.unsafe_ptr(),
        param.unsafe_ptr(), m.unsafe_ptr(), v.unsafe_ptr(), Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
    )
    var hp = scan.host.unsafe_ptr()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=hp, src_buf=status)
    var loss_cell = loss.create_sub_buffer[DType.float32](0, 1)
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=hp.bitcast[Float32]() + AFN_ST_CELLS, src_buf=loss_cell)
    step_count_sync()
    ctx.synchronize()
    _ = status^
    _ = loss_cell^
    var value = hp.bitcast[Float32]().unsafe_load(AFN_ST_CELLS)
    if hp.unsafe_load(AFN_ST_LOSS) != NONFINITE_NONE:
        raise Error("byte LM: nonfinite loss at 0")
    var names: List[String] = ["gradients", "parameters", "first moments", "second moments"]
    for c in range(4):
        var bad = hp.unsafe_load(AFN_ST_GRAD + c)
        if bad != NONFINITE_NONE:
            raise Error("byte LM: nonfinite " + names[c] + " at " + String(bad))
    if hp.unsafe_load(AFN_ST_VNEG) != NONFINITE_NONE:
        raise Error("byte LM: negative second moment")
    return value
