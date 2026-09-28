# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE ON THE DEVICE: x_cnn/ops.mojo's element functions, one
thread per element, and the pinned GEMM (`identical_gemm_into`, full FP32, never
the vendor route) for every contraction. The `*_into` entries take the
caller's host addresses in and out (the binding's path, DEVIATION 5716); the
List-returning forms wrap them for the seam check. Every entry runs on the
one process-lifetime context and synchronizes before it returns.

The host twin is x_cnn/host/ops_host.mojo: the same element functions in a
loop and `gemm_oracle` for the contractions."""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.ffi import _Global
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_identical import (
    identical_gemm_with_plan, identical_gemm_workspace_floats, PLAN_SPLIT_32_2X2, PLAN_SPLIT_64_4X4,
    PLAN_SPLIT_16_1X1, PLAN_APPLE_MMA, PLAN_TUNED_32_2X2, PLAN_SPLITK, apple_mma_applies, apple_mma_applies_one_leaf, PLAN_APPLE_MMA_SPLIT, PLAN_APPLE_MMA_SPLIT_BIG,
    identical_gemm_splitk_fits, choose_gemm_plan,
)
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_APPLE
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN
from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from x_cnn.ops import (
    FP, IP, ElemFn, CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW,
    im2col_at, im2col_taps_at, conv_out_at, dout_rows_at, col2im_at, fill_one_at,
    PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW,
    maxpool_fwd_at, maxpool_bwd_at, avgpool_fwd_at, avgpool_bwd_at, relu_maxpool_fwd_at, pool_relu_rows_bwd_at,
    conv_out_val, pool_relu_row_val,
    relu_fwd_at, relu_bwd_at, add_at, bias_rows_at, softmax_xent_row_at, seq_mean, sgd_at,
    bn_stats_at, bn_eval_stats_at, bn_apply_at, bn_running_at, bn_bwd_red_at, bn_bwd_dx_at, bn_bwd_eval_dx_at,
    dropout2d_at, mul_at, spmm_at, gcn_deg_at, gcn_norm_at,
    pad_fwd_at, pad_bwd_at, adapt_avg_fwd_at, adapt_avg_bwd_at, adapt_max_fwd_at, adapt_max_bwd_at,
    sage_max_fwd_at, sage_max_bwd_at, l2norm_fwd_at, l2norm_bwd_at, adam_at, gather_rows_at,
)

comptime TPB = 256
#: lane/cnn-apple2: the FAST tier on Apple measures its GEMM plans (the
#: simdgroup matrix plans among them). `-D MOJOLEARN_XCNN_NO_FAST_TUNE` is
#: the before arm (round 1's FAST: the 4090 split plans and the dispatcher).
#: lane/cnn-apple2: IDENTICAL on Apple also times APPLE_MMA against the
#: dispatcher outside the weight gradients (`-D MOJOLEARN_XCNN_NO_NT_TUNE`
#: is the before arm).
comptime APPLE_NT_TUNE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and TARGET_COLUMN == COLUMN_APPLE
    and not is_defined["MOJOLEARN_XCNN_NO_NT_TUNE"]()
)
comptime APPLE_FAST_TUNE = (
    GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL
    and TARGET_COLUMN == COLUMN_APPLE
    and not is_defined["MOJOLEARN_XCNN_NO_FAST_TUNE"]()
)


struct _CnnContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_cnn entry. A context
    per call exhausted Metal's per-process command queues on the M2 Pro
    ("Failed to create Metal command queue for context") within one trainer
    fit (memory: METAL QUEUE LIMIT IS PER-PROCESS). Storage is
    `std.ffi._Global` (the pattern of the trees, RF and byte LM bindings),
    one slot per numeric tier so a FAST and an IDENTICAL .so in one process
    never share it."""
    var ctx: Optional[DeviceContext]
    #: The workspace (DEVIATION 5718): one cached device buffer per slot,
    #: grown on demand and kept for the process.
    var ws: List[DeviceBuffer[DType.float32]]
    #: The resident arrays (DEVIATION 5718): owning buffers the caller holds
    #: by device address between entries (`res_alloc` / `res_free`).
    var res: List[DeviceBuffer[DType.float32]]
    #: Apple IDENTICAL (DEVIATION 5720): (m, n, k, plan) quads, the measured
    #: fastest weight/bias-gradient plan per shape.
    var tuned: List[Int]
    #: lane/cnn-apple2: freed resident arrays kept for reuse by `res_alloc`
    #: (at most `RES_POOL_MAX_FLOATS` in all).
    var pool: List[DeviceBuffer[DType.float32]]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.ws = List[DeviceBuffer[DType.float32]]()
        self.res = List[DeviceBuffer[DType.float32]]()
        self.tuned = List[Int]()
        self.pool = List[DeviceBuffer[DType.float32]]()


comptime _CTX_NAME = "MojoXCnnContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXCnnContextFast"
comptime X_CNN_CONTEXT = _Global[StorageType=_CnnContext, name=_CTX_NAME, init_fn=_CnnContext.__init__]


def cnn_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_CNN_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


def elem_kernel[f: ElemFn](a: FP, b: FP, c: FP, d: FP, q: IP, p: IP, total: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(total):
        f(i, a, b, c, d, q, p)


@always_inline
def fp(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def ip(mut buf: DeviceBuffer[DType.int32]) -> IP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def launch[f: ElemFn](ctx: DeviceContext, a: FP, b: FP, c: FP, d: FP, q: IP, p: IP, total: Int) raises:
    if total <= 0:
        return
    comptime k = elem_kernel[f]
    ctx.enqueue_function[k](a, b, c, d, q, p, Int32(total), grid_dim=(total + TPB - 1) // TPB, block_dim=TPB)


# lane/cnn-apple2: the two layout changes of the conv block (GEMM rows
# [n*S + s, oc] <-> NCHW [n, oc, s], S = OH*OW) as 32x32 tiles through
# threadgroup memory, so both the reads and the writes are coalesced (the
# one-thread-per-element forms read or write with stride OC or S). Each
# stored word is the element function's (`conv_out_val`,
# `pool_relu_row_val`); only which thread computes it changes. Threadgroup
# memory only between the barrier's two sides (no device-memory ordering is
# assumed). `-D MOJOLEARN_XCNN_NO_TILED_LAYOUT` is the before arm.
comptime TILED_LAYOUT = not is_defined["MOJOLEARN_XCNN_NO_TILED_LAYOUT"]()
#: The pooled backward's rows tiled (`rows_bwd_tiled_kernel`) measured
#: SLOWER on the M4 Pro (block 1 1.44 -> 1.73 ms, block 2 0.75 -> 0.96:
#: each thread runs four max-pool gathers in series); opt-in only,
#: `-D MOJOLEARN_XCNN_TILED_ROWS`.
comptime TILED_ROWS = TILED_LAYOUT and is_defined["MOJOLEARN_XCNN_TILED_ROWS"]()
comptime _LT = 32
comptime _LR = 8


def conv_out_tiled_kernel(y2: FP, bias: FP, dst: FP, p: IP, S: Int32, OC: Int32):
    """dst[n, oc, s] = conv_out_val(y2[n*S + s, oc]); block (32, 8), grid
    (ceil(S/32), ceil(OC/32), N)."""
    var t = stack_allocation[_LT * (_LT + 1), Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var s0 = Int(block_idx.x) * _LT
    var c0 = Int(block_idx.y) * _LT
    var n = Int(block_idx.z)
    var ss = Int(S)
    var cc = Int(OC)
    comptime for j in range(_LT // _LR):
        var s = s0 + ty + j * _LR
        var oc = c0 + tx
        if s < ss and oc < cc:
            t[(ty + j * _LR) * (_LT + 1) + tx] = y2.unsafe_load((n * ss + s) * cc + oc)
    barrier()
    comptime for j in range(_LT // _LR):
        var oc = c0 + ty + j * _LR
        var s = s0 + tx
        if s < ss and oc < cc:
            dst.unsafe_store((n * cc + oc) * ss + s, conv_out_val(t[tx * (_LT + 1) + ty + j * _LR], bias, oc, p))


def rows_bwd_tiled_kernel(dpool: FP, yconv: FP, grow: FP, idx: IP, p: IP, S: Int32, OC: Int32):
    """grow[n*S + s, oc] = pool_relu_row_val(NCHW (n, oc, s)); block (32, 8),
    grid (ceil(S/32), ceil(OC/32), N)."""
    var t = stack_allocation[_LT * (_LT + 1), Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var s0 = Int(block_idx.x) * _LT
    var c0 = Int(block_idx.y) * _LT
    var n = Int(block_idx.z)
    var ss = Int(S)
    var cc = Int(OC)
    comptime for j in range(_LT // _LR):
        var oc = c0 + ty + j * _LR
        var s = s0 + tx
        if s < ss and oc < cc:
            t[(ty + j * _LR) * (_LT + 1) + tx] = pool_relu_row_val((n * cc + oc) * ss + s, dpool, yconv, idx, p)
    barrier()
    comptime for j in range(_LT // _LR):
        var s = s0 + ty + j * _LR
        var oc = c0 + tx
        if s < ss and oc < cc:
            grow.unsafe_store((n * ss + s) * cc + oc, t[tx * (_LT + 1) + ty + j * _LR])


def dout_rows_tiled_kernel(dout: FP, g: FP, S: Int32, OC: Int32):
    """g[n*S + s, oc] = ftz(dout[n, oc, s]) (`dout_rows_at`'s word); block
    (32, 8), grid (ceil(S/32), ceil(OC/32), N)."""
    var t = stack_allocation[_LT * (_LT + 1), Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var s0 = Int(block_idx.x) * _LT
    var c0 = Int(block_idx.y) * _LT
    var n = Int(block_idx.z)
    var ss = Int(S)
    var cc = Int(OC)
    comptime for j in range(_LT // _LR):
        var oc = c0 + ty + j * _LR
        var s = s0 + tx
        if s < ss and oc < cc:
            t[(ty + j * _LR) * (_LT + 1) + tx] = ftz(dout.unsafe_load((n * cc + oc) * ss + s))
    barrier()
    comptime for j in range(_LT // _LR):
        var s = s0 + ty + j * _LR
        var oc = c0 + tx
        if s < ss and oc < cc:
            g.unsafe_store((n * ss + s) * cc + oc, t[tx * (_LT + 1) + ty + j * _LR])


#: lane/cnn-apple2: im2col one thread per (row, channel) (`im2col_taps_at`).
#: `-D MOJOLEARN_XCNN_NO_IM2COL_TAPS` is the before arm.
comptime IM2COL_TAPS = not is_defined["MOJOLEARN_XCNN_NO_IM2COL_TAPS"]()


def _im2col(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], mut cols: DeviceBuffer[DType.float32],
    mut dp: DeviceBuffer[DType.int32], rows: Int, ckk: Int, C: Int,
) raises:
    comptime if IM2COL_TAPS:
        launch[im2col_taps_at](ctx, fp(dx), fp(cols), fp(cols), fp(cols), ip(dp), ip(dp), rows * C)
    else:
        launch[im2col_at](ctx, fp(dx), fp(cols), fp(cols), fp(cols), ip(dp), ip(dp), rows * ckk)


def _tiled_grid(N: Int, S: Int, OC: Int) -> Tuple[Int, Int, Int]:
    return ((S + _LT - 1) // _LT, (OC + _LT - 1) // _LT, N)


def _apple_tn_candidates(m: Int, n: Int, k: Int, default: Int) -> List[Int]:
    """The weight/bias-gradient (OP_TN) plans that were ever competitive on
    an Apple GPU at the x_cnn shapes (lane/cnn-apple sweeps)."""
    var cand = List[Int]()
    cand.append(default)
    if n == 1:
        cand.append(PLAN_SPLIT_16_1X1)
        if m * n <= 4096 and identical_gemm_splitk_fits(m, n, k):
            cand.append(PLAN_SPLITK)
        comptime if not is_defined["MOJOLEARN_XCNN_NO_MMA_SPLIT"]():
            if apple_mma_applies(m, n, k):
                cand.append(PLAN_APPLE_MMA_SPLIT)
    else:
        cand.append(PLAN_SPLIT_16_1X1)
        if n >= 64:
            cand.append(PLAN_TUNED_32_2X2)
            if apple_mma_applies(m, n, k):
                cand.append(PLAN_APPLE_MMA)
        # lane/cnn-apple2: the simdgroup matrix kernel over leaf groups
        # (the weight gradient's long k on grid.y). `-D
        # MOJOLEARN_XCNN_NO_MMA_SPLIT` is the before arm.
        comptime if not is_defined["MOJOLEARN_XCNN_NO_MMA_SPLIT"]():
            if n >= 8 and apple_mma_applies(m, n, k):
                cand.append(PLAN_APPLE_MMA_SPLIT)
                if m >= 64 and n >= 64:
                    cand.append(PLAN_APPLE_MMA_SPLIT_BIG)
    return cand^


def _apple_tuned_plan(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int, op: Int, cand: List[Int],
) raises -> Int:
    """The fastest of `cand` for this shape and orientation on this device,
    timed once (one run each, after a wait for the entry's earlier work) and
    cached for the process. In IDENTICAL every candidate writes the same
    words into `c` (contract 6.1); in FAST the candidates are the same
    leaves and fold on different units (the FAST tier's plan choice)."""
    var s = _slots()
    var i = 0
    while i + 4 < len(s[].tuned):
        if s[].tuned[i] == m and s[].tuned[i + 1] == n and s[].tuned[i + 2] == k and s[].tuned[i + 3] == op:
            return s[].tuned[i + 4]
        i += 5
    var need = 0
    for j in range(len(cand)):
        need = max(need, identical_gemm_workspace_floats(m, n, k, cand[j]))
    var wp = ws(ctx, GEMM_WS_SLOT, need)
    ctx.synchronize()
    var best = cand[0]
    var best_ns = perf_counter_ns()  # replaced by the first candidate
    for j in range(len(cand)):
        # lane/cnn-apple2: one untimed run first, so a candidate's first-use
        # cost (its pipeline) does not decide against it
        identical_gemm_with_plan(ctx, c, a, b, wp, m, n, k, op, cand[j])
        ctx.synchronize()
        var t0 = perf_counter_ns()
        identical_gemm_with_plan(ctx, c, a, b, wp, m, n, k, op, cand[j])
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if j == 0 or dt < best_ns:
            best_ns = dt
            best = cand[j]
    _ = wp^
    s[].tuned.append(m)
    s[].tuned.append(n)
    s[].tuned.append(k)
    s[].tuned.append(op)
    s[].tuned.append(best)
    return best


def device_gemm(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int, op: Int,
) raises:
    """`C = op(A) . op(B)` under mojolearn.identical.gemm.fp32.v1, full FP32
    in both tiers (allow_vendor=False: the NVIDIA vendor route is TF32).
    Asynchronous: the entry's own synchronize ends it."""
    # DEVIATION 5718: the shipped dispatcher (`identical_gemm_into`, the plan
    # `choose_gemm_plan` picks) on a cached workspace, instead of
    # `identical_gemm`'s allocate, run, synchronize and free per call.
    # The weight and bias gradients (OP_TN, a small m x n output over the
    # N*OH*OW rows) name their split plan: the dispatcher's SPLIT 16x16 and,
    # on NVIDIA, the long-k group rule's own workspace and wait were 2x to 7x
    # slower on the RTX 4090 at the CNNClassifier and Conv2d shapes, bits
    # equal on every plan (the forced-plan sweep, progress/cnn.md phase 4).
    # A plan is the EXECUTION plan: the partition and fold come from `k`.
    # No floor on `k`, so the lane checks' small fixtures take this path on
    # every column too.
    if op == OP_TN and m * n <= 65536:
        var plan = PLAN_SPLIT_64_4X4 if (m >= 64 and n >= 64) else PLAN_SPLIT_32_2X2
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN == COLUMN_APPLE:
            # DEVIATION 5720 (lane/cnn-apple, forced-plan sweeps, every plan
            # bit-equal, tools/apple_speed_cnn/gemm_plans.mojo): on Apple the
            # fastest weight/bias-gradient plan depends on the GPU's size.
            # The M4 (10 cores) wants the simdgroup matrix or the 32x32
            # register tile (64 x 576 x 262144: SPLIT 64x64 130.6, MMA 52.4
            # ms; 64 x 288 x 65536: 24.0 vs TUNED 32x32 11.0 ms); the M3
            # Ultra wants the split plans (the same two: 26.6 vs 68.6 ms and
            # 4.4 vs 9.3 ms). So the plan is MEASURED once per shape per
            # process among the plans that were ever competitive and cached
            # (`_apple_tuned_plan`). Execution plan only: the partition and
            # the fold come from `k`, so every candidate stores the same bits.
            plan = _apple_tuned_plan(ctx, c, a, b, m, n, k, op, _apple_tn_candidates(m, n, k, plan))
        comptime if APPLE_FAST_TUNE:
            # lane/cnn-apple2: the FAST tier measures the same candidates,
            # the simdgroup matrix plans among them (APPLE_MMA_FAST)
            plan = _apple_tuned_plan(ctx, c, a, b, m, n, k, op, _apple_tn_candidates(m, n, k, plan))
        var wp = ws(ctx, GEMM_WS_SLOT, identical_gemm_workspace_floats(m, n, k, plan))
        identical_gemm_with_plan(ctx, c, a, b, wp, m, n, k, op, plan)
        _ = wp^
        return
    comptime if APPLE_FAST_TUNE or APPLE_NT_TUNE:
        # lane/cnn-apple2: FAST on Apple has no simdgroup matrix plan in the
        # shipped dispatcher (it is IDENTICAL's), and neither tier's
        # dispatcher takes it for a one-leaf ragged k (the first block's
        # k = 27); where it applies, time it against the dispatcher's pick
        # once per shape.
        if m >= 8 and n >= 8 and apple_mma_applies_one_leaf(m, n, k):
            var fc = List[Int]()
            fc.append(choose_gemm_plan(m, n, k))
            if fc[0] != PLAN_APPLE_MMA:
                fc.append(PLAN_APPLE_MMA)
            var fplan = _apple_tuned_plan(ctx, c, a, b, m, n, k, op, fc)
            var fw = ws(ctx, GEMM_WS_SLOT, identical_gemm_workspace_max_floats(m, n, k))
            identical_gemm_with_plan(ctx, c, a, b, fw, m, n, k, op, fplan)
            _ = fw^
            return
    var w = ws(ctx, GEMM_WS_SLOT, identical_gemm_workspace_max_floats(m, n, k))
    identical_gemm_into[False](ctx, c, a, b, w, m, n, k, op)
    _ = w^


# ------------------------------------------------------------------ host I/O
# DEVIATION 5716 (phase d, 2026-09-27): the binding hands the device entries
# the caller's HOST ADDRESSES and they copy straight between those and the
# device, once each way. Before, every array made five host copies around one
# transfer (read_f32 into a List, a copy inside upload_f32, the pinned
# staging buffer, an element-by-element append out of it, copy_f32 into the
# caller's array): measured on the RTX 4090 at N 256, 64x32x32, the
# forward's two transfers cost 30 ms against 2.9 ms of kernels. Copies only:
# no value is touched, so neither tier's bits move. Every `*_into` entry
# enqueues its uploads, kernels and downloads on the one in-order context and
# synchronizes ONCE before its buffers drop (`[[mojo-buffer-freed-at-last-use]]`:
# the trailing `_ = buf^` lines keep them alive past that wait). The caller's
# arrays stay alive for the whole call (the binding holds them).


def up(ctx: DeviceContext, src: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A device copy of `n` floats at host address `src` (no staging copy)."""
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
    return buf^


def up_i(ctx: DeviceContext, src: IP, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
    return buf^


def down(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], dst: FP, n: Int) raises:
    """Enqueue the first `n` floats of `buf` into host address `dst`; the
    caller synchronizes (a sub-buffer view synchronizes here, since the view
    drops at return)."""
    if n <= 0:
        return
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
        ctx.synchronize()


def down_i(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], dst: IP, n: Int) raises:
    if n <= 0:
        return
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.int32](0, n)
        ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
        ctx.synchronize()


# ------------------------------------------------ workspace + resident arrays
# DEVIATION 5718 (phase 4, IDENTICAL speed, 2026-09-27). Measured on the RTX
# 4090 (the block backward at N 256, 64x64x32x32): freeing an entry's
# eighteen device buffers took 12.3 ms and allocating them 1.2 ms, against
# 9.2 ms of kernels; in CNNClassifier's step every activation also crossed
# PCIe two or three times. Two changes, both plumbing:
# - THE WORKSPACE: an entry's device buffers are views of per-slot buffers
#   cached for the process (`ws`), grown when a call needs more. Every entry
#   synchronizes before it returns and uses each slot once, so no two live
#   views of one entry share a slot and no entry sees another's.
# - RESIDENT ARRAYS: `res_alloc` hands the caller a device address it keeps
#   between entries; the `[resident=True]` form of an entry reads and writes
#   those addresses in place of uploading and downloading host arrays. The
#   CPU twin's resident arrays are host allocations and its resident entries
#   are its ordinary ones.
# Neither changes a kernel, a GEMM plan choice, an operand, or the order of
# anything: the same launches on the same values. Neither tier's bits move.


def _slots() raises -> UnsafePointer[_CnnContext, MutAnyOrigin]:
    return X_CNN_CONTEXT.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def view(ctx: DeviceContext, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A non-owning device buffer over `n` floats at device address `p`."""
    return DeviceBuffer[DType.float32](ctx, p, n if n > 0 else 1, owning=False)


def view_i(ctx: DeviceContext, p: IP, n: Int) raises -> DeviceBuffer[DType.int32]:
    return DeviceBuffer[DType.int32](ctx, p, n if n > 0 else 1, owning=False)


#: The pinned GEMM's workspace; every entry's other slots are below it.
comptime GEMM_WS_SLOT = 31


def ws(ctx: DeviceContext, slot: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """Workspace slot `slot` as `n` floats (at least 1)."""
    var need = n if n > 0 else 1
    var s = _slots()
    while len(s[].ws) <= slot:
        s[].ws.append(ctx.enqueue_create_buffer[DType.float32](1))
    if len(s[].ws[slot]) < need:
        # the old buffer may still be read by work enqueued earlier in this
        # entry (the GEMM workspace slot serves every GEMM of an entry)
        ctx.synchronize()
        s[].ws[slot] = ctx.enqueue_create_buffer[DType.float32](need)
    return view(ctx, s[].ws[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), need)


def ws_i(ctx: DeviceContext, slot: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    """Workspace slot `slot` as `n` Int32 words (the same 4-byte storage)."""
    var b = ws(ctx, slot, n)
    return view_i(ctx, b.unsafe_ptr().bitcast[Int32]().unsafe_origin_cast[MutAnyOrigin](), n)


def put[resident: Bool](ctx: DeviceContext, slot: Int, src: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """An entry's input: the resident array itself, or host `src` copied into slot `slot`."""
    comptime if resident:
        return view(ctx, src, n)
    var b = ws(ctx, slot, n)
    if n > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=src)
    return b^


def put_i[resident: Bool](ctx: DeviceContext, slot: Int, src: IP, n: Int) raises -> DeviceBuffer[DType.int32]:
    comptime if resident:
        return view_i(ctx, src, n)
    var b = ws_i(ctx, slot, n)
    if n > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=src)
    return b^


def put_prm(ctx: DeviceContext, slot: Int, prm: List[Int32]) raises -> DeviceBuffer[DType.int32]:
    """A parameter block into slot `slot` (the caller keeps `prm` alive past its synchronize)."""
    var b = ws_i(ctx, slot, len(prm))
    if len(prm) > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=prm.unsafe_ptr())
    return b^


def put_hyper(ctx: DeviceContext, slot: Int, h: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var b = ws(ctx, slot, len(h))
    if len(h) > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=h.unsafe_ptr())
    return b^


def outb[resident: Bool](ctx: DeviceContext, slot: Int, dst: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """An entry's output: the resident array itself, or slot `slot` (downloaded by `fetch`)."""
    comptime if resident:
        return view(ctx, dst, n)
    return ws(ctx, slot, n)


def outb_i[resident: Bool](ctx: DeviceContext, slot: Int, dst: IP, n: Int) raises -> DeviceBuffer[DType.int32]:
    comptime if resident:
        return view_i(ctx, dst, n)
    return ws_i(ctx, slot, n)


def fetch[resident: Bool](ctx: DeviceContext, buf: DeviceBuffer[DType.float32], dst: FP, n: Int) raises:
    comptime if not resident:
        down(ctx, buf, dst, n)


def fetch_i[resident: Bool](ctx: DeviceContext, buf: DeviceBuffer[DType.int32], dst: IP, n: Int) raises:
    comptime if not resident:
        down_i(ctx, buf, dst, n)


#: lane/cnn-apple2: the most freed resident storage kept for reuse (floats;
#: 256 MB). `-D MOJOLEARN_XCNN_NO_RES_POOL` is the before arm (no pool).
comptime RES_POOL_MAX_FLOATS = 64 * 1024 * 1024
comptime RES_POOL = not is_defined["MOJOLEARN_XCNN_NO_RES_POOL"]()


def res_alloc(n: Int) raises -> Int:
    """A resident array of `n` floats (4-byte words), zero filled; its device address."""
    var ctx = cnn_ctx()
    var need = n if n > 0 else 1
    # lane/cnn-apple2: a freed array of at least `need` and at most twice
    # it (the smallest such) instead of a new allocation; zero filled the
    # same way, so the caller sees the same words.
    comptime if RES_POOL:
        var s = _slots()
        var pick = -1
        for j in range(len(s[].pool)):
            var ln = len(s[].pool[j])
            if ln >= need and ln <= 2 * need and (pick < 0 or ln < len(s[].pool[pick])):
                pick = j
        if pick >= 0:
            var pb = s[].pool.pop(pick)
            pb.enqueue_fill(Float32(0))
            var paddr = Int(pb.unsafe_ptr())
            s[].res.append(pb^)
            _ = ctx^
            return paddr
    var b = ctx.enqueue_create_buffer[DType.float32](need)
    b.enqueue_fill(Float32(0))
    # No wait (lane/cnn-apple): the fill is ordered before every later use
    # on the one in-order context, and every host read (res_download) waits.
    var addr = Int(b.unsafe_ptr())
    _slots()[].res.append(b^)
    _ = ctx^
    return addr


def res_free(addr: Int) raises:
    # the array may still be read or written by enqueued work
    var ctx = cnn_ctx()
    ctx.synchronize()
    _ = ctx^
    var s = _slots()
    for k in range(len(s[].res)):
        if Int(s[].res[k].unsafe_ptr()) == addr:
            var b = s[].res.pop(k)
            comptime if RES_POOL:
                # keep it for reuse while the pool stays under its cap
                # (the wait above ended every use of it)
                var held = len(b)
                for j in range(len(s[].pool)):
                    held += len(s[].pool[j])
                if held <= RES_POOL_MAX_FLOATS:
                    s[].pool.append(b^)
                    return
            _ = b^
            return
    raise Error("x_cnn: res_free of an address res_alloc did not return")


def res_upload(addr: Int, src: FP, n: Int) raises:
    """n 4-byte words from host `src` into the resident array at `addr`."""
    if n <= 0:
        return
    var ctx = cnn_ctx()
    var b = view(ctx, FP(unsafe_from_address=addr), n)
    ctx.enqueue_copy(dst_buf=b, src_ptr=src)
    ctx.synchronize()
    _ = b^
    _ = ctx^


def res_gather(dst_addr: Int, src_addr: Int, rows: IP, n: Int, row: Int) raises:
    """Resident dst[r] = resident src[rows[r]] for r < n, `row` 4-byte words
    each (a word copy; host `rows` are the indices)."""
    if n <= 0 or row <= 0:
        return
    var ctx = cnn_ctx()
    var di = put_i[False](ctx, 0, rows, n)
    var prm: List[Int32] = [Int32(row)]
    var dp = put_prm(ctx, 1, prm)
    var src = view(ctx, FP(unsafe_from_address=src_addr), 1)
    var dst = view(ctx, FP(unsafe_from_address=dst_addr), n * row)
    launch[gather_rows_at](ctx, fp(src), fp(dst), fp(dst), fp(dst), ip(di), ip(dp), n * row)
    ctx.synchronize()
    _ = prm^
    _ = di^
    _ = dp^
    _ = src^
    _ = dst^
    _ = ctx^


def res_download(addr: Int, dst: FP, n: Int) raises:
    if n <= 0:
        return
    var ctx = cnn_ctx()
    var b = view(ctx, FP(unsafe_from_address=addr), n)
    ctx.enqueue_copy(dst_ptr=dst, src_buf=b)
    ctx.synchronize()
    _ = b^
    _ = ctx^


@always_inline
def lp(mut l: List[Float32]) -> FP:
    """A List's storage as an entry's host address (the List must outlive the call)."""
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def lpi(mut l: List[Int32]) -> IP:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


# ------------------------------------------------------------------ entries


def gemm_into(a: FP, b: FP, c: FP, m: Int, n: Int, k: Int, op: Int) raises:
    var ctx = cnn_ctx()
    var da = put[False](ctx, 0, a, m * k)
    var db = put[False](ctx, 1, b, n * k)
    var dc = ws(ctx, 2, m * n)
    device_gemm(ctx, dc, da, db, m, n, k, op)
    down(ctx, dc, c, m * n)
    ctx.synchronize()
    _ = da^
    _ = db^
    _ = dc^
    _ = ctx^


def gemm_device(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int, op: Int) raises -> List[Float32]:
    var sa = a.copy()
    var sb = b.copy()
    var out = List[Float32](length=m * n, fill=Float32(0))
    gemm_into(lp(sa), lp(sb), lp(out), m, n, k, op)
    _ = sa^
    _ = sb^
    return out^


def conv2d_forward_into(x: FP, w: FP, bias: FP, prm: List[Int32], dst: FP) raises:
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var ctx = cnn_ctx()
    var dx = put[False](ctx, 0, x, N * C * Int(prm[CP_H]) * Int(prm[CP_W]))
    var dw = put[False](ctx, 1, w, OC * ckk)
    var dbias = put[False](ctx, 2, bias, OC)
    var dp = put_prm(ctx, 3, prm)
    var cols = ws(ctx, 4, rows * ckk)
    var y2 = ws(ctx, 5, rows * OC)
    var dout = ws(ctx, 6, rows * OC)
    _im2col(ctx, dx, cols, dp, rows, ckk, C)
    device_gemm(ctx, y2, cols, dw, rows, OC, ckk, OP_NT)
    _conv_out(ctx, y2, dbias, dout, dp, N, rows // N, OC)
    down(ctx, dout, dst, rows * OC)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = dout^
    _ = ctx^


def conv2d_backward_into(x: FP, w: FP, dout: FP, prm: List[Int32], gx_out: FP, gw_out: FP, gb_out: FP) raises:
    """dx (N*C*H*W), dW (OC*C*KH*KW), db (OC)."""
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var ctx = cnn_ctx()
    var dxin = put[False](ctx, 0, x, nx)
    var dw = put[False](ctx, 1, w, OC * ckk)
    var ddout = put[False](ctx, 2, dout, rows * OC)
    var dp = put_prm(ctx, 3, prm)
    var cols = ws(ctx, 4, rows * ckk)
    var g = ws(ctx, 5, rows * OC)
    var ones = ws(ctx, 6, rows)
    var gw = ws(ctx, 7, OC * ckk)
    var gb = ws(ctx, 8, OC)
    var dcols = ws(ctx, 9, rows * ckk)
    var gx = ws(ctx, 10, nx)
    _im2col(ctx, dxin, cols, dp, rows, ckk, C)
    comptime if TILED_LAYOUT:
        var tg = _tiled_grid(N, rows // N, OC)
        ctx.enqueue_function[dout_rows_tiled_kernel](
            fp(ddout), fp(g), Int32(rows // N), Int32(OC),
            grid_dim=(tg[0], tg[1], tg[2]), block_dim=(_LT, _LR, 1),
        )
    else:
        launch[dout_rows_at](ctx, fp(ddout), fp(g), fp(g), fp(g), ip(dp), ip(dp), rows * OC)
    launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
    # DEVIATION 5701: the weight gradient's reduction over the N*OH*OW rows is
    # the pinned GEMM's (leaves + balanced fold), never an atomic accumulation.
    device_gemm(ctx, gw, g, cols, OC, ckk, rows, OP_TN)
    device_gemm(ctx, gb, g, ones, OC, 1, rows, OP_TN)
    device_gemm(ctx, dcols, g, dw, rows, ckk, OC, OP_NN)
    launch[col2im_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    down(ctx, gx, gx_out, nx)
    down(ctx, gw, gw_out, OC * ckk)
    down(ctx, gb, gb_out, OC)
    ctx.synchronize()
    _ = dxin^
    _ = dw^
    _ = ddout^
    _ = dp^
    _ = cols^
    _ = g^
    _ = ones^
    _ = gw^
    _ = gb^
    _ = dcols^
    _ = gx^
    _ = ctx^


def conv2d_backward_device(
    x: List[Float32], w: List[Float32], dout: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    """[dx | dW | db], concatenated (the List form, for the seam check)."""
    var OC = Int(prm[CP_OC])
    var ckk = Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var nx = Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W])
    var sx = x.copy()
    var sw = w.copy()
    var sd = dout.copy()
    var out = List[Float32](length=nx + OC * ckk + OC, fill=Float32(0))
    var base = lp(out)
    conv2d_backward_into(lp(sx), lp(sw), lp(sd), prm, base, base + nx, base + nx + OC * ckk)
    _ = sx^
    _ = sw^
    _ = sd^
    return out^


def _pool_sizes(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def maxpool2d_forward_into(x: FP, prm: List[Int32], dst: FP, idx_out: IP) raises:
    var sizes = _pool_sizes(prm)
    var no = sizes[1]
    var ctx = cnn_ctx()
    var dx = put[False](ctx, 0, x, sizes[0])
    var dp = put_prm(ctx, 1, prm)
    var dout = ws(ctx, 2, no)
    var di = ws_i(ctx, 3, no)
    launch[maxpool_fwd_at](ctx, fp(dx), fp(dout), fp(dout), fp(dout), ip(di), ip(dp), no)
    down(ctx, dout, dst, no)
    down_i(ctx, di, idx_out, no)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dout^
    _ = di^
    _ = ctx^


def maxpool2d_forward_device(x: List[Float32], prm: List[Int32], mut idx: List[Int32]) raises -> List[Float32]:
    var no = _pool_sizes(prm)[1]
    var sx = x.copy()
    var out = List[Float32](length=no, fill=Float32(0))
    idx = List[Int32](length=no, fill=Int32(0))
    maxpool2d_forward_into(lp(sx), prm, lp(out), lpi(idx))
    _ = sx^
    return out^


def maxpool2d_backward_into(dout: FP, idx: IP, prm: List[Int32], gx_out: FP) raises:
    var sizes = _pool_sizes(prm)
    var nx = sizes[0]
    var ctx = cnn_ctx()
    var dd = put[False](ctx, 0, dout, sizes[1])
    var di = put_i[False](ctx, 1, idx, sizes[1])
    var dp = put_prm(ctx, 2, prm)
    var gx = ws(ctx, 3, nx)
    launch[maxpool_bwd_at](ctx, fp(dd), fp(gx), fp(gx), fp(gx), ip(di), ip(dp), nx)
    down(ctx, gx, gx_out, nx)
    ctx.synchronize()
    _ = dd^
    _ = di^
    _ = dp^
    _ = gx^
    _ = ctx^


def avgpool2d_forward_into(x: FP, prm: List[Int32], dst: FP) raises:
    var sizes = _pool_sizes(prm)
    var no = sizes[1]
    var ctx = cnn_ctx()
    var dx = put[False](ctx, 0, x, sizes[0])
    var dp = put_prm(ctx, 1, prm)
    var dout = ws(ctx, 2, no)
    launch[avgpool_fwd_at](ctx, fp(dx), fp(dout), fp(dout), fp(dout), ip(dp), ip(dp), no)
    down(ctx, dout, dst, no)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dout^
    _ = ctx^


def avgpool2d_forward_device(x: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var sx = x.copy()
    var out = List[Float32](length=_pool_sizes(prm)[1], fill=Float32(0))
    avgpool2d_forward_into(lp(sx), prm, lp(out))
    _ = sx^
    return out^


def avgpool2d_backward_into(dout: FP, prm: List[Int32], gx_out: FP) raises:
    var sizes = _pool_sizes(prm)
    var nx = sizes[0]
    var ctx = cnn_ctx()
    var dd = put[False](ctx, 0, dout, sizes[1])
    var dp = put_prm(ctx, 1, prm)
    var gx = ws(ctx, 2, nx)
    launch[avgpool_bwd_at](ctx, fp(dd), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    down(ctx, gx, gx_out, nx)
    ctx.synchronize()
    _ = dd^
    _ = dp^
    _ = gx^
    _ = ctx^


def map2_into[f: ElemFn](a: FP, na: Int, b: FP, nb: Int, n_out: Int, prm: List[Int32], dst: FP) raises:
    """dst[i] = f(a, b) for i < n_out (slots a, b, dst); `nb == 0` reuses a."""
    var ctx = cnn_ctx()
    var da = put[False](ctx, 0, a, na)
    var db = put[False](ctx, 1, b, nb)
    var dp = put_prm(ctx, 2, prm)
    var dout = ws(ctx, 3, n_out)
    if nb > 0:
        launch[f](ctx, fp(da), fp(db), fp(dout), fp(dout), ip(dp), ip(dp), n_out)
    else:
        launch[f](ctx, fp(da), fp(da), fp(dout), fp(dout), ip(dp), ip(dp), n_out)
    down(ctx, dout, dst, n_out)
    ctx.synchronize()
    _ = da^
    _ = db^
    _ = dp^
    _ = dout^
    _ = ctx^


def relu_forward_into(x: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[relu_fwd_at](x, n, x, 0, n, prm, dst)


def relu_backward_into(x: FP, g: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[relu_bwd_at](x, n, g, n, n, prm, dst)


def add_into(a: FP, b: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[add_at](a, n, b, n, n, prm, dst)


def mul_into(a: FP, b: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[mul_at](a, n, b, n, n, prm, dst)


def linear_forward_into[resident: Bool = False](x: FP, w: FP, bias: FP, n: Int, d_in: Int, d_out: Int, dst: FP) raises:
    """y = x W^T + b: the pinned GEMM NT, then one add per element."""
    var ctx = cnn_ctx()
    var dx = put[resident](ctx, 0, x, n * d_in)
    var dw = put[resident](ctx, 1, w, d_out * d_in)
    var db = put[resident](ctx, 2, bias, d_out)
    var prm: List[Int32] = [Int32(n), Int32(d_in), Int32(d_out)]
    var dp = put_prm(ctx, 3, prm)
    var y = ws(ctx, 4, n * d_out)
    var dout = outb[resident](ctx, 5, dst, n * d_out)
    device_gemm(ctx, y, dx, dw, n, d_out, d_in, OP_NT)
    launch[bias_rows_at](ctx, fp(y), fp(db), fp(dout), fp(dout), ip(dp), ip(dp), n * d_out)
    fetch[resident](ctx, dout, dst, n * d_out)
    ctx.synchronize()
    _ = prm^
    _ = dx^
    _ = dw^
    _ = db^
    _ = dp^
    _ = y^
    _ = dout^
    _ = ctx^


def linear_backward_into[resident: Bool = False](
    x: FP, w: FP, g: FP, n: Int, d_in: Int, d_out: Int, gx_out: FP, gw_out: FP, gb_out: FP
) raises:
    """dW = G^T X and db = G^T 1 (GEMM TN over the rows, the pinned fold),
    dx = G W (GEMM NN)."""
    var ctx = cnn_ctx()
    var dx = put[resident](ctx, 0, x, n * d_in)
    var dw = put[resident](ctx, 1, w, d_out * d_in)
    var dg = put[resident](ctx, 2, g, n * d_out)
    var dones = ws(ctx, 3, n)
    dones.enqueue_fill(Float32(1))
    var gx = outb[resident](ctx, 4, gx_out, n * d_in)
    var gw = outb[resident](ctx, 5, gw_out, d_out * d_in)
    var gb = outb[resident](ctx, 6, gb_out, d_out)
    device_gemm(ctx, gw, dg, dx, d_out, d_in, n, OP_TN)
    device_gemm(ctx, gb, dg, dones, d_out, 1, n, OP_TN)
    device_gemm(ctx, gx, dg, dw, n, d_in, d_out, OP_NN)
    fetch[resident](ctx, gx, gx_out, n * d_in)
    fetch[resident](ctx, gw, gw_out, d_out * d_in)
    fetch[resident](ctx, gb, gb_out, d_out)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dg^
    _ = dones^
    _ = gx^
    _ = gw^
    _ = gb^
    _ = ctx^


def softmax_xent_into[resident: Bool = False](
    logits: FP, labels: IP, n: Int, k: Int, grad_out: FP, proba_out: FP
) raises -> Float32:
    """grad and proba (n*k each) into the caller's arrays; returns the mean loss."""
    var ctx = cnn_ctx()
    var dl = put[resident](ctx, 0, logits, n * k)
    var dy = put_i[resident](ctx, 1, labels, n)
    var prm: List[Int32] = [Int32(n), Int32(k)]
    var dp = put_prm(ctx, 2, prm)
    var grad = outb[resident](ctx, 3, grad_out, n * k)
    var proba = outb[resident](ctx, 4, proba_out, n * k)
    var rl = ws(ctx, 5, n)
    grad.enqueue_fill(Float32(0))
    rl.enqueue_fill(Float32(0))
    launch[softmax_xent_row_at](ctx, fp(dl), fp(grad), fp(proba), fp(rl), ip(dy), ip(dp), n)
    var rows = List[Float32](length=n, fill=Float32(0))
    fetch[resident](ctx, grad, grad_out, n * k)
    fetch[resident](ctx, proba, proba_out, n * k)
    down(ctx, rl, lp(rows), n)
    ctx.synchronize()
    _ = prm^
    _ = dl^
    _ = dy^
    _ = dp^
    _ = grad^
    _ = proba^
    _ = rl^
    _ = ctx^
    return seq_mean(rows, n)


def softmax_xent_device(logits: List[Float32], labels: List[Int32], n: Int, k: Int) raises -> List[Float32]:
    """[grad (n*k) | proba (n*k) | mean loss (1)] (the List form, for the seam check)."""
    var sl = logits.copy()
    var sy = labels.copy()
    var out = List[Float32](length=2 * n * k + 1, fill=Float32(0))
    var base = lp(out)
    var loss = softmax_xent_into(lp(sl), lpi(sy), n, k, base, base + n * k)
    out[2 * n * k] = loss
    _ = sl^
    _ = sy^
    return out^


def sgd_into[resident: Bool = False](w: FP, g: FP, v: FP, hyper: List[Float32], n: Int) raises:
    """In place: w and the momentum buffer v."""
    var ctx = cnn_ctx()
    var dw = put[resident](ctx, 0, w, n)
    var dg = put[resident](ctx, 1, g, n)
    var dv = put[resident](ctx, 2, v, n)
    var dh = put_hyper(ctx, 3, hyper)
    var prm: List[Int32] = [Int32(n)]
    var dp = put_prm(ctx, 4, prm)
    launch[sgd_at](ctx, fp(dw), fp(dg), fp(dv), fp(dh), ip(dp), ip(dp), n)
    fetch[resident](ctx, dw, w, n)
    fetch[resident](ctx, dv, v, n)
    ctx.synchronize()
    _ = prm^
    _ = dw^
    _ = dg^
    _ = dv^
    _ = dh^
    _ = dp^
    _ = ctx^


def sgd_device(w: List[Float32], g: List[Float32], v: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    """[w' | v'] (the List form, for the seam check)."""
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sv = v.copy()
    sgd_into(lp(sw), lp(sg), lp(sv), hyper, n)
    sw.extend(sv^)
    _ = sg^
    return sw^


def batchnorm_forward_into(x: FP, running: FP, aux: FP, prm: List[Int32], training: Bool, y_out: FP) raises:
    """y into y_out; running (2C) and aux (2 + 7C, the statistics the backward reads) in place."""
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var nr = 2 * C
    var na = 2 + 7 * C
    var ctx = cnn_ctx()
    var dx = up(ctx, x, total)
    var dr = up(ctx, running, nr)
    var da = up(ctx, aux, na)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](total)
    if training:
        launch[bn_stats_at](ctx, fp(dx), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    else:
        launch[bn_eval_stats_at](ctx, fp(dr), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    launch[bn_apply_at](ctx, fp(dx), fp(da), fp(dout), fp(dout), ip(dp), ip(dp), total)
    if training:
        launch[bn_running_at](ctx, fp(dr), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    down(ctx, dout, y_out, total)
    down(ctx, dr, running, nr)
    down(ctx, da, aux, na)
    ctx.synchronize()
    _ = dx^
    _ = dr^
    _ = da^
    _ = dp^
    _ = dout^
    _ = ctx^


def batchnorm_forward_device(
    x: List[Float32], running: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    """[y | running' | aux'] (the List form, for the seam check)."""
    var sx = x.copy()
    var sr = running.copy()
    var sa = aux.copy()
    var out = List[Float32](length=len(x), fill=Float32(0))
    batchnorm_forward_into(lp(sx), lp(sr), lp(sa), prm, training, lp(out))
    out.extend(sr^)
    out.extend(sa^)
    _ = sx^
    return out^


def batchnorm_backward_into(x: FP, g: FP, aux: FP, prm: List[Int32], training: Bool, dx_out: FP) raises:
    """dx into dx_out; aux in place (sum_g = dbeta and sum_gx = dgamma)."""
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var na = 2 + 7 * C
    var ctx = cnn_ctx()
    var dx = up(ctx, x, total)
    var dg = up(ctx, g, total)
    var da = up(ctx, aux, na)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](total)
    launch[bn_bwd_red_at](ctx, fp(dx), fp(dg), fp(da), fp(da), ip(dp), ip(dp), C)
    if training:
        launch[bn_bwd_dx_at](ctx, fp(dx), fp(dg), fp(da), fp(dout), ip(dp), ip(dp), total)
    else:
        launch[bn_bwd_eval_dx_at](ctx, fp(dx), fp(dg), fp(da), fp(dout), ip(dp), ip(dp), total)
    down(ctx, dout, dx_out, total)
    down(ctx, da, aux, na)
    ctx.synchronize()
    _ = dx^
    _ = dg^
    _ = da^
    _ = dp^
    _ = dout^
    _ = ctx^


def dropout2d_into(x: FP, n: Int, prm: List[Int32], hyper: List[Float32], y_out: FP, mask_out: FP) raises:
    var ctx = cnn_ctx()
    var dx = up(ctx, x, n)
    var dp = upload_i32(ctx, prm)
    var dh = upload_f32(ctx, hyper)
    var mask = ctx.enqueue_create_buffer[DType.float32](n)
    var dout = ctx.enqueue_create_buffer[DType.float32](n)
    launch[dropout2d_at](ctx, fp(dx), fp(mask), fp(dout), fp(dh), ip(dp), ip(dp), n)
    down(ctx, dout, y_out, n)
    down(ctx, mask, mask_out, n)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dh^
    _ = mask^
    _ = dout^
    _ = ctx^


def dropout2d_device(x: List[Float32], prm: List[Int32], hyper: List[Float32]) raises -> List[Float32]:
    """[y | mask] (the List form, for the seam check)."""
    var n = len(x)
    var sx = x.copy()
    var out = List[Float32](length=2 * n, fill=Float32(0))
    var base = lp(out)
    dropout2d_into(lp(sx), n, prm, hyper, base, base + n)
    _ = sx^
    return out^


def spmm_into(vals: FP, nvals: Int, h: FP, csr: List[Int32], prm: List[Int32], dst: FP) raises:
    var total = Int(prm[0]) * Int(prm[1])
    var ctx = cnn_ctx()
    var dv = up(ctx, vals, nvals)
    var dh = up(ctx, h, total)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](total)
    launch[spmm_at](ctx, fp(dv), fp(dh), fp(dout), fp(dout), ip(dq), ip(dp), total)
    down(ctx, dout, dst, total)
    ctx.synchronize()
    _ = dv^
    _ = dh^
    _ = dq^
    _ = dp^
    _ = dout^
    _ = ctx^


def spmm_device(vals: List[Float32], h: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var sv = vals.copy()
    var sh = h.copy()
    var out = List[Float32](length=Int(prm[0]) * Int(prm[1]), fill=Float32(0))
    spmm_into(lp(sv), len(sv), lp(sh), csr, prm, lp(out))
    _ = sv^
    _ = sh^
    return out^


def gcn_norm_device(w: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var n = Int(prm[0])
    var nnz = Int(prm[2])
    var ctx = cnn_ctx()
    var dw = upload_f32(ctx, w)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var dis = ctx.enqueue_create_buffer[DType.float32](n)
    var vals = ctx.enqueue_create_buffer[DType.float32](nnz)
    launch[gcn_deg_at](ctx, fp(dw), fp(dis), fp(dis), fp(dis), ip(dq), ip(dp), n)
    launch[gcn_norm_at](ctx, fp(dw), fp(dis), fp(vals), fp(vals), ip(dq), ip(dp), nnz)
    var result = download_f32(ctx, vals, nnz)
    _ = dw^
    _ = dq^
    _ = dp^
    _ = dis^
    _ = vals^
    _ = ctx^
    return result^


def pad2d_forward_into(x: FP, prm: List[Int32], dst: FP) raises:
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var n_out = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    map2_into[pad_fwd_at](x, nx, x, 0, n_out, prm, dst)


def pad2d_backward_into(g: FP, prm: List[Int32], dst: FP) raises:
    var nc = Int(prm[0]) * Int(prm[1])
    var ng = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var n_out = nc * Int(prm[2]) * Int(prm[3])
    map2_into[pad_bwd_at](g, ng, g, 0, n_out, prm, dst)


def pad2d_backward_device(g: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var sg = g.copy()
    var out = List[Float32](length=Int(prm[0]) * Int(prm[1]) * Int(prm[2]) * Int(prm[3]), fill=Float32(0))
    pad2d_backward_into(lp(sg), prm, lp(out))
    _ = sg^
    return out^


def adaptive_device[f: ElemFn](a: List[Float32], idx: List[Int32], n_out: Int, prm: List[Int32], mut idx_out: List[Int32]) raises -> List[Float32]:
    """One adaptive-pool element function over n_out outputs; `idx` in, `idx_out` out (max pooling)."""
    var ctx = cnn_ctx()
    var da = upload_f32(ctx, a)
    var di = upload_i32(ctx, idx)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](n_out)
    launch[f](ctx, fp(da), fp(da), fp(out), fp(out), ip(di), ip(dp), n_out)
    var result = download_f32(ctx, out, n_out)
    idx_out = download_i32(ctx, di, len(idx))
    _ = da^
    _ = di^
    _ = dp^
    _ = out^
    _ = ctx^
    return result^


def adaptive_pool_device(x: List[Float32], idx: List[Int32], prm: List[Int32], kind: Int, mut idx_out: List[Int32]) raises -> List[Float32]:
    """kind 0 avg forward, 1 avg backward, 2 max forward (idx_out = winners), 3 max backward (idx = winners)."""
    var nc = Int(prm[0]) * Int(prm[1])
    var nin = nc * Int(prm[2]) * Int(prm[3])
    var nout = nc * Int(prm[4]) * Int(prm[5])
    if kind == 0:
        return adaptive_device[adapt_avg_fwd_at](x, idx, nout, prm, idx_out)
    if kind == 1:
        return adaptive_device[adapt_avg_bwd_at](x, idx, nin, prm, idx_out)
    if kind == 2:
        var slots = List[Int32](length=nout, fill=Int32(0))
        return adaptive_device[adapt_max_fwd_at](x, slots, nout, prm, idx_out)
    return adaptive_device[adapt_max_bwd_at](x, idx, nin, prm, idx_out)


def graph4_device[f: ElemFn](a: List[Float32], b: List[Float32], aux: List[Float32], csr: List[Int32], prm: List[Int32], total: Int, n_out: Int) raises -> List[Float32]:
    """[dst (n_out) | aux'] of one element function over `total` items (slots a, b, aux, dst)."""
    var ctx = cnn_ctx()
    var da = upload_f32(ctx, a)
    var db = upload_f32(ctx, b)
    var dx = upload_f32(ctx, aux)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](n_out)
    launch[f](ctx, fp(da), fp(db), fp(dx), fp(out), ip(dq), ip(dp), total)
    var result = download_f32(ctx, out, n_out)
    var raux = download_f32(ctx, dx, len(aux))
    _ = da^
    _ = db^
    _ = dx^
    _ = dq^
    _ = dp^
    _ = out^
    _ = ctx^
    result.extend(raux^)
    return result^


def graph_op_device(a: List[Float32], b: List[Float32], aux: List[Float32], csr: List[Int32], prm: List[Int32], kind: Int) raises -> List[Float32]:
    """kind 0 sage max forward, 1 sage max backward, 2 l2 normalize forward, 3 its backward; [dst | aux']."""
    var n = Int(prm[0])
    var F = Int(prm[1])
    if kind == 0:
        return graph4_device[sage_max_fwd_at](a, b, aux, csr, prm, n * F, n * F)
    if kind == 1:
        return graph4_device[sage_max_bwd_at](a, b, aux, csr, prm, n * F, n * F)
    if kind == 2:
        return graph4_device[l2norm_fwd_at](a, b, aux, csr, prm, n, n * F)
    return graph4_device[l2norm_bwd_at](a, b, aux, csr, prm, n, n * F)


def adam_into[resident: Bool = False](w: FP, g: FP, mv: FP, hyper: List[Float32], n: Int) raises:
    """In place: w and mv = [m (n) | v (n)]."""
    var ctx = cnn_ctx()
    var dw = put[resident](ctx, 0, w, n)
    var dg = put[resident](ctx, 1, g, n)
    var dm = put[resident](ctx, 2, mv, 2 * n)
    var dh = put_hyper(ctx, 3, hyper)
    var prm: List[Int32] = [Int32(n)]
    var dp = put_prm(ctx, 4, prm)
    launch[adam_at](ctx, fp(dw), fp(dg), fp(dm), fp(dh), ip(dp), ip(dp), n)
    fetch[resident](ctx, dw, w, n)
    fetch[resident](ctx, dm, mv, 2 * n)
    ctx.synchronize()
    _ = prm^
    _ = dw^
    _ = dg^
    _ = dm^
    _ = dh^
    _ = dp^
    _ = ctx^


def adam_device(w: List[Float32], g: List[Float32], mv: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    """[w' | mv'] (the List form, for the seam check)."""
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sm = mv.copy()
    adam_into(lp(sw), lp(sg), lp(sm), hyper, n)
    sw.extend(sm^)
    _ = sg^
    return sw^


def opt_many_resident[adam: Bool](
    ws_: List[Int], gs: List[Int], bs: List[Int], ns: List[Int], hyper: List[Float32]
) raises:
    """lane/cnn-apple2: one optimizer step over EVERY parameter of a trainer
    step in one entry (one wait instead of one per parameter): per parameter
    `j` the same launch `sgd_into` / `adam_into` makes (`sgd_at` / `adam_at`
    on resident w, g and buffer, the step's one hyper block, p[0] = n_j), in
    the caller's order. Each launch touches only its own parameter's arrays,
    so the words are the per-parameter entries' words."""
    var ctx = cnn_ctx()
    var dh = put_hyper(ctx, 3, hyper)
    var prm = List[Int32]()
    for j in range(len(ns)):
        prm.append(Int32(ns[j]))
    var dp = put_prm(ctx, 4, prm)
    var pp = ip(dp)
    for j in range(len(ns)):
        var pw = FP(unsafe_from_address=ws_[j])
        var pg = FP(unsafe_from_address=gs[j])
        var pb = FP(unsafe_from_address=bs[j])
        comptime if adam:
            launch[adam_at](ctx, pw, pg, pb, fp(dh), pp + j, pp + j, ns[j])
        else:
            launch[sgd_at](ctx, pw, pg, pb, fp(dh), pp + j, pp + j, ns[j])
    ctx.synchronize()
    _ = prm^
    _ = dh^
    _ = dp^
    _ = ctx^


def res_gather_pair(
    dst_addr: Int, src_addr: Int, row: Int, dst2_addr: Int, src2_addr: Int, row2: Int, rows: IP, n: Int
) raises:
    """lane/cnn-apple2: `res_gather` twice on the same host rows (the batch
    and its labels) in one entry, one upload of the rows and one wait; the
    same `gather_rows_at` launches on the same words."""
    if n <= 0:
        return
    var ctx = cnn_ctx()
    var di = put_i[False](ctx, 0, rows, n)
    var prm: List[Int32] = [Int32(row), Int32(row2)]
    var dp = put_prm(ctx, 1, prm)
    var pp = ip(dp)
    if row > 0:
        launch[gather_rows_at](
            ctx, FP(unsafe_from_address=src_addr), FP(unsafe_from_address=dst_addr),
            FP(unsafe_from_address=dst_addr), FP(unsafe_from_address=dst_addr), ip(di), pp, n * row,
        )
    if row2 > 0:
        launch[gather_rows_at](
            ctx, FP(unsafe_from_address=src2_addr), FP(unsafe_from_address=dst2_addr),
            FP(unsafe_from_address=dst2_addr), FP(unsafe_from_address=dst2_addr), ip(di), pp + 1, n * row2,
        )
    ctx.synchronize()
    _ = prm^
    _ = di^
    _ = dp^
    _ = ctx^


# ------------------------------------------------------------ the conv block
# DEVIATION 5717 (phase d, 2026-09-27): CNNClassifier's block, Conv2d ->
# ReLU -> MaxPool2d, in ONE entry each way, so the activations between the
# three stay on the device. The same element functions and the same GEMMs,
# launched in the same order on the same values as the three separate layer
# entries: neither tier's bits move. The backward RECOMPUTES the conv output
# (im2col, GEMM NT, conv_out) from the input it uploads anyway for the
# weight gradient, instead of carrying it through the host; the recompute is
# the forward's own kernels on the forward's own inputs, so its bits are the
# forward's. `pool` False is Conv2d -> ReLU (the map is smaller than the
# window). `need_dx` False skips col2im and the NN GEMM (the first block's
# input gradient, which the trainer never reads).


def _conv_relu_on_device(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.float32],
    mut dbias: DeviceBuffer[DType.float32], mut dp: DeviceBuffer[DType.int32], mut cols: DeviceBuffer[DType.float32],
    mut y2: DeviceBuffer[DType.float32], mut yconv: DeviceBuffer[DType.float32], rows: Int, OC: Int, ckk: Int,
    N: Int, C: Int,
) raises:
    """cols = im2col(x); y2 = cols . W^T; yconv = the NCHW conv output (+ bias)."""
    _im2col(ctx, dx, cols, dp, rows, ckk, C)
    device_gemm(ctx, y2, cols, dw, rows, OC, ckk, OP_NT)
    _conv_out(ctx, y2, dbias, yconv, dp, N, rows // N, OC)


def _conv_out(
    ctx: DeviceContext, mut y2: DeviceBuffer[DType.float32], mut dbias: DeviceBuffer[DType.float32],
    mut yconv: DeviceBuffer[DType.float32], mut dp: DeviceBuffer[DType.int32], N: Int, S: Int, OC: Int,
) raises:
    """The GEMM rows to NCHW (+ bias): the tiled kernel, or `conv_out_at`."""
    comptime if TILED_LAYOUT:
        if N > 0 and S > 0 and OC > 0:
            var g = _tiled_grid(N, S, OC)
            ctx.enqueue_function[conv_out_tiled_kernel](
                fp(y2), fp(dbias), fp(yconv), ip(dp), Int32(S), Int32(OC),
                grid_dim=(g[0], g[1], g[2]), block_dim=(_LT, _LR, 1),
            )
        return
    launch[conv_out_at](ctx, fp(y2), fp(dbias), fp(yconv), fp(yconv), ip(dp), ip(dp), N * S * OC)


def conv_block_forward_into[resident: Bool = False](
    x: FP, w: FP, bias: FP, cprm: List[Int32], pprm: List[Int32], pool: Bool, dst: FP, idx_out: IP,
    save_cols: Int = 0, save_y: Int = 0,
) raises:
    """dst = maxpool(relu(conv(x))) (idx_out its winners), or relu(conv(x)) when `pool` is False.
    Resident with `save_cols` and `save_y` (device addresses, rows*C*KH*KW and
    N*OC*OH*OW floats): the im2col matrix and the conv output are written
    there for the backward to read instead of recomputing them."""
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = N * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ny = rows * OC
    var no = _pool_sizes(pprm)[1] if pool else ny
    var ctx = cnn_ctx()
    var dx = put[resident](ctx, 0, x, N * C * Int(cprm[CP_H]) * Int(cprm[CP_W]))
    var dw = put[resident](ctx, 1, w, OC * ckk)
    var dbias = put[resident](ctx, 2, bias, OC)
    var dp = put_prm(ctx, 3, cprm)
    var saved = resident and save_cols != 0 and save_y != 0
    var cols = view(ctx, FP(unsafe_from_address=save_cols), rows * ckk) if saved else ws(ctx, 4, rows * ckk)
    var y2 = ws(ctx, 5, ny)
    var yconv = view(ctx, FP(unsafe_from_address=save_y), ny) if saved else ws(ctx, 6, ny)
    _conv_relu_on_device(ctx, dx, dw, dbias, dp, cols, y2, yconv, rows, OC, ckk, N, C)
    # the block's output: the pool's, or the ReLU's when there is no pool
    var pout = outb[resident](ctx, 7, dst, no)
    if pool:
        var dpp = put_prm(ctx, 9, pprm)
        var di = outb_i[resident](ctx, 10, idx_out, no)
        # DEVIATION 5720: ReLU and the max pool in one launch (the same values)
        launch[relu_maxpool_fwd_at](ctx, fp(yconv), fp(pout), fp(pout), fp(pout), ip(di), ip(dpp), no)
        fetch[resident](ctx, pout, dst, no)
        fetch_i[resident](ctx, di, idx_out, no)
        ctx.synchronize()
        _ = dpp^
        _ = di^
    else:
        launch[relu_fwd_at](ctx, fp(yconv), fp(yconv), fp(pout), fp(pout), ip(dp), ip(dp), ny)
        fetch[resident](ctx, pout, dst, ny)
        ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = yconv^
    _ = pout^
    _ = ctx^


def conv_block_backward_into[resident: Bool = False](
    x: FP, w: FP, bias: FP, g: FP, idx: IP, cprm: List[Int32], pprm: List[Int32], pool: Bool, need_dx: Bool,
    gx_out: FP, gw_out: FP, gb_out: FP, save_cols: Int = 0, save_y: Int = 0,
) raises:
    """From the gradient of the block's output `g`: dx (when `need_dx`), dW, db.
    Resident with `save_cols` and `save_y`: the forward's im2col matrix and
    conv output, read instead of recomputed (the same kernels on the same
    inputs produced them, so the values are the recomputation's)."""
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = N * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ny = rows * OC
    var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
    var no = _pool_sizes(pprm)[1] if pool else ny
    var ctx = cnn_ctx()
    var dx = put[resident](ctx, 0, x, nx)
    var dw = put[resident](ctx, 1, w, OC * ckk)
    var dbias = put[resident](ctx, 2, bias, OC)
    var dgo = put[resident](ctx, 3, g, no)
    var dp = put_prm(ctx, 5, cprm)
    var saved = resident and save_cols != 0 and save_y != 0
    var cols = view(ctx, FP(unsafe_from_address=save_cols), rows * ckk) if saved else ws(ctx, 7, rows * ckk)
    var y2 = ws(ctx, 8, ny)
    var yconv = view(ctx, FP(unsafe_from_address=save_y), ny) if saved else ws(ctx, 9, ny)
    var gy = ws(ctx, 11, ny)
    if not saved:
        _conv_relu_on_device(ctx, dx, dw, dbias, dp, cols, y2, yconv, rows, OC, ckk, N, C)
    # conv2d_backward_into from here, on the device-resident gy and cols
    var grow = ws(ctx, 12, ny)
    # the conv block followed by the pool block: a host array the upload
    # reads, kept alive past the entry's synchronize
    var both = cprm.copy()
    both.extend(pprm.copy())
    if pool:
        # DEVIATION 5720: the max pool's backward, the ReLU's and the row
        # layout in one launch (the same values)
        var di = put_i[resident](ctx, 4, idx, no)
        var dpb = put_prm(ctx, 6, both)
        comptime if TILED_ROWS:
            var S = rows // N
            var g = _tiled_grid(N, S, OC)
            ctx.enqueue_function[rows_bwd_tiled_kernel](
                fp(dgo), fp(yconv), fp(grow), ip(di), ip(dpb), Int32(S), Int32(OC),
                grid_dim=(g[0], g[1], g[2]), block_dim=(_LT, _LR, 1),
            )
        else:
            launch[pool_relu_rows_bwd_at](ctx, fp(dgo), fp(yconv), fp(grow), fp(grow), ip(di), ip(dpb), ny)
        _ = di^
        _ = dpb^
    else:
        launch[relu_bwd_at](ctx, fp(yconv), fp(dgo), fp(gy), fp(gy), ip(dp), ip(dp), ny)
        launch[dout_rows_at](ctx, fp(gy), fp(grow), fp(grow), fp(grow), ip(dp), ip(dp), ny)
    var ones = ws(ctx, 13, rows)
    var gw = outb[resident](ctx, 14, gw_out, OC * ckk)
    var gb = outb[resident](ctx, 15, gb_out, OC)
    launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
    # DEVIATION 5701: the pinned GEMM's fold over the rows, never an atomic.
    device_gemm(ctx, gw, grow, cols, OC, ckk, rows, OP_TN)
    device_gemm(ctx, gb, grow, ones, OC, 1, rows, OP_TN)
    if need_dx:
        var dcols = ws(ctx, 16, rows * ckk)
        var gx = outb[resident](ctx, 17, gx_out, nx)
        device_gemm(ctx, dcols, grow, dw, rows, ckk, OC, OP_NN)
        launch[col2im_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
        fetch[resident](ctx, gx, gx_out, nx)
        _ = dcols^
        _ = gx^
    fetch[resident](ctx, gw, gw_out, OC * ckk)
    fetch[resident](ctx, gb, gb_out, OC)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dgo^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = yconv^
    _ = gy^
    _ = both^
    _ = grow^
    _ = ones^
    _ = gw^
    _ = gb^
    _ = ctx^
