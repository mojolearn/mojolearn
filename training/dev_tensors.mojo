# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane fam2-neural (2026-10-04): DEVICE TENSORS FOR THE TRAINING OPS.

The training binding's ops take host addresses and return host arrays, so a
training loop written with them moves every operand across the bus on every
call: `linear_forward` uploads the input and the weight and downloads the
output, `cross_entropy` uploads the logits it was just handed and downloads
their gradient (256 MB each way at 8,192 x 8,192), `linear_backward` uploads
that gradient again, and the optimizer uploads the weight gradient.

This file is a pool of float32 device arrays addressed by integer handles
(the binding's own context, `_Global` storage, one pool per tier) and the
forms of the linear and cross-entropy ops that read and write them. A chain
linear -> cross-entropy -> linear backward -> optimizer then keeps every
N * V and parameter-sized array on the device; what crosses is the targets,
the row losses and the loss scalar.

Each form runs the host entry's own calls in the host entry's order on the
same values (the certified GEMM, `identical_ce_loss_resident`), with the
non-finite refusals as device scans: no bit moves. IDENTICAL, GPU columns;
`-D MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF` (or MOJOLEARN_IDN_ALL_OFF) leaves
the entries unregistered and the Python functions then refuse a device
array by name."""
from std.ffi import _Global
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_scan import device_first_nonfinite
from gemm.checks.gemm_backward import (
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
)
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_NT
from training.estimator import identical_ce_loss_dev
from gemm.experiments.neural_grouped_backward import neural_linear_backward_into

comptime NI_G03_GROUP_LINEAR = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; G03 A, default OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_G03_GROUP_LINEAR"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NI_G03_SEPARATE_LINEAR = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; G03 matched B, default OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_G03_SEPARATE_LINEAR"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NI_G04_RETAIN_SCRATCH = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; G04 A, default OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_G04_RETAIN_SCRATCH"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
# A bounded 64 MiB retention budget, independent of benchmark dimensions.
# Oversized calls retain the existing temporary-allocation/completion path.
comptime NI_G04_MAX_SCRATCH_FLOATS = (64 * 1024 * 1024) // 4

comptime IDN_TRAIN_DEV_TENSORS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and not (is_defined["MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]
comptime _DP = MutPointer[Float32, MutAnyOrigin]


struct _DevPool(Defaultable, Movable):
    var b: List[DeviceBuffer[DType.float32]]
    #: floats held by each slot; 0 marks a free slot
    var n: List[Int]
    var gemm_scratch: List[DeviceBuffer[DType.float32]]

    def __init__(out self):
        self.b = List[DeviceBuffer[DType.float32]]()
        self.n = List[Int]()
        self.gemm_scratch = List[DeviceBuffer[DType.float32]]()


comptime _DEV_POOL_NAME = "MojoTrainingDevTensorsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoTrainingDevTensorsFast"
comptime _DEV_POOL = _Global[StorageType=_DevPool, name=_DEV_POOL_NAME, init_fn=_DevPool.__init__]


def train_dev_alloc(ctx: DeviceContext, n: Int) raises -> Int:
    """A device array of `n` floats (uninitialized); its handle."""
    if n < 1:
        raise Error("train_dev_alloc: n must be >= 1, got " + String(n))
    var pool = _DEV_POOL.get_or_create_ptr()
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var h = -1
    for j in range(len(pool[].n)):  # small-loop(pool: live device-array handles): free handle slot search, no data
        if pool[].n[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].b.append(buf^)
        pool[].n.append(n)
        h = len(pool[].n) - 1
    else:
        pool[].b[h] = buf^
        pool[].n[h] = n
    return h


def train_dev_size(h: Int) raises -> Int:
    """The floats handle `h` holds; refuses a handle that is not open."""
    var pool = _DEV_POOL.get_or_create_ptr()
    if h < 0 or h >= len(pool[].n) or pool[].n[h] == 0:
        raise Error("training device array: handle " + String(h) + " is not open")
    return pool[].n[h]


def train_dev_free(ctx: DeviceContext, h: Int) raises:
    """Free handle `h` (after the queue drains)."""
    _ = train_dev_size(h)
    ctx.synchronize()
    var pool = _DEV_POOL.get_or_create_ptr()
    pool[].b[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].n[h] = 0
    comptime if NI_G04_RETAIN_SCRATCH:
        var any_live = False
        for j in range(len(pool[].n)):  # handle metadata only
            any_live = any_live or pool[].n[j] != 0
        if not any_live:
            # This entry already drained the queue before freeing handles.
            pool[].gemm_scratch = List[DeviceBuffer[DType.float32]]()


def _linear_scratch(ctx: DeviceContext, required: Int) raises -> DeviceBuffer[DType.float32]:
    """Same binding/context owner as device handles; at most one retained slab.

    Callers synchronize before releasing a temporary or reusing the pool.
    This optimization removes allocations, not required completion/refusals.
    """
    var size = max(1, required)
    comptime if NI_G04_RETAIN_SCRATCH:
        if size <= NI_G04_MAX_SCRATCH_FLOATS:
            var pool = _DEV_POOL.get_or_create_ptr()
            if len(pool[].gemm_scratch) == 0:
                pool[].gemm_scratch.append(ctx.enqueue_create_buffer[DType.float32](size))
            elif len(pool[].gemm_scratch[0]) < size:
                ctx.synchronize()
                pool[].gemm_scratch[0] = ctx.enqueue_create_buffer[DType.float32](size)
            var ptr = pool[].gemm_scratch[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            return DeviceBuffer[DType.float32](ctx, ptr, size, owning=False)
    return ctx.enqueue_create_buffer[DType.float32](size)


def train_dev_view(ctx: DeviceContext, h: Int, n: Int, name: String) raises -> DeviceBuffer[DType.float32]:
    """A non-owning buffer over the first `n` floats of handle `h` (the
    pool keeps the storage alive; the handle must hold at least `n`)."""
    var have = train_dev_size(h)
    if n < 1 or have < n:
        raise Error(
            "training device array: " + name + " needs " + String(n) + " floats, handle "
            + String(h) + " holds " + String(have)
        )
    var pool = _DEV_POOL.get_or_create_ptr()
    var p = pool[].b[h].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    return DeviceBuffer[DType.float32](ctx, p, n, owning=False)


def train_dev_put(ctx: DeviceContext, h: Int, src: _FP, n: Int) raises:
    """Host `n` floats into handle `h`."""
    var v = train_dev_view(ctx, h, n, "the upload")
    ctx.enqueue_copy(dst_buf=v, src_ptr=src)
    ctx.synchronize()
    _ = v^


def train_dev_get(ctx: DeviceContext, h: Int, dst: _FP, n: Int) raises:
    """Handle `h`'s first `n` floats into host memory."""
    var v = train_dev_view(ctx, h, n, "the download")
    ctx.enqueue_copy(dst_ptr=dst, src_buf=v)
    ctx.synchronize()
    _ = v^


def _refuse_dev(ctx: DeviceContext, name: String, mut buf: DeviceBuffer[DType.float32], n: Int) raises:
    """`samba_ops._refuse_nonfinite`'s refusal, scanned on the device."""
    var idx = device_first_nonfinite(ctx, buf, n)
    if idx >= 0:
        raise Error("mojolearn samba ops: non-finite " + name + " at flat index " + String(idx))


def train_linear_forward_dev(ctx: DeviceContext, c_h: Int, a_h: Int, w_h: Int, m: Int, n: Int, k: Int) raises -> Int:
    """`samba_linear_forward_host` on device arrays: C[m, n] (handle c_h,
    written) = A[m, k] (a_h) . W[n, k]^T (w_h). Returns `m * n`."""
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    var a = train_dev_view(ctx, a_h, m * k, "linear input")
    var w = train_dev_view(ctx, w_h, n * k, "linear weight")
    var c = train_dev_view(ctx, c_h, m * n, "linear output")
    _refuse_dev(ctx, "linear input", a, m * k)
    _refuse_dev(ctx, "linear weight", w, n * k)
    var ws = _linear_scratch(ctx, identical_gemm_workspace_max_floats(m, n, k))
    identical_gemm_into(ctx, c, a, w, ws, m, n, k, OP_NT)
    ctx.synchronize()
    _ = a^
    _ = w^
    _ = c^
    _ = ws^
    return m * n


def train_linear_backward_dev(
    ctx: DeviceContext, da_h: Int, dw_h: Int, dc_h: Int, a_h: Int, w_h: Int, m: Int, n: Int, k: Int
) raises -> Int:
    """`samba_linear_backward_host` on device arrays: da[m, k] (da_h,
    written) = dc . W and dw[n, k] (dw_h, written) = dc^T . A. Returns
    `m * k`."""
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    var a = train_dev_view(ctx, a_h, m * k, "linear input")
    var w = train_dev_view(ctx, w_h, n * k, "linear weight")
    var dc = train_dev_view(ctx, dc_h, m * n, "linear upstream gradient")
    var da = train_dev_view(ctx, da_h, m * k, "linear input gradient")
    var dw = train_dev_view(ctx, dw_h, n * k, "linear weight gradient")
    _refuse_dev(ctx, "linear input", a, m * k)
    _refuse_dev(ctx, "linear weight", w, n * k)
    _refuse_dev(ctx, "linear upstream gradient", dc, m * n)
    comptime if NI_G03_GROUP_LINEAR or NI_G03_SEPARATE_LINEAR:
        # Independent output ownership is an API/lifetime condition, never
        # a matrix-size dispatch rule. Keep the old route for aliased handles.
        if (da_h != dw_h and da_h != dc_h and da_h != a_h and da_h != w_h
                and dw_h != dc_h and dw_h != a_h and dw_h != w_h):
            neural_linear_backward_into[NI_G03_GROUP_LINEAR](ctx, da, dw, dc, a, w, m, n, k)
            ctx.synchronize()
            _ = a^
            _ = w^
            _ = dc^
            _ = da^
            _ = dw^
            return m * k
    comptime if NI_G04_RETAIN_SCRATCH:
        # Both products are enqueued serially on the same context, so one
        # slab safely serves the two unchanged contraction/fold schedules.
        var shared_ws = _linear_scratch(ctx, max(
            identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k),
            identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)))
        identical_gemm_backward_a_into(ctx, da, dc, w, shared_ws, m, n, k, OP_NT)
        identical_gemm_backward_b_into(ctx, dw, dc, a, shared_ws, m, n, k, OP_NT)
        ctx.synchronize()
        _ = a^
        _ = w^
        _ = dc^
        _ = da^
        _ = dw^
        _ = shared_ws^
        return m * k
    var ws_a = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k)
    )
    var ws_b = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)
    )
    identical_gemm_backward_a_into(ctx, da, dc, w, ws_a, m, n, k, OP_NT)
    identical_gemm_backward_b_into(ctx, dw, dc, a, ws_b, m, n, k, OP_NT)
    ctx.synchronize()
    _ = a^
    _ = w^
    _ = dc^
    _ = da^
    _ = dw^
    _ = ws_a^
    _ = ws_b^
    return m * k


def train_ce_loss_dev(
    ctx: DeviceContext,
    loss_ptr: _FP,
    row_ptr: _FP,
    dlogits_h: Int,
    logits_h: Int,
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_rows: Int,
    vocab: Int,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    want_grad: Int,
    label_smoothing: Float32,
) raises -> Int:
    """`identical_ce_loss_host` on device arrays: the logits are handle
    `logits_h` (N * V floats), the gradient is written into handle
    `dlogits_h` (N * V floats; ignored, pass any open handle, when
    `want_grad == 0`). Returns `count`."""
    if n_rows < 1 or vocab < 1:
        raise Error("mojolearn training: cross entropy needs N >= 1 and V >= 1")
    var cells = n_rows * vocab
    var logits = train_dev_view(ctx, logits_h, cells, "logits")
    var count = 0
    if want_grad != 0:
        var dlogits = train_dev_view(ctx, dlogits_h, cells, "the logits gradient")
        count = identical_ce_loss_dev(
            ctx, loss_ptr, row_ptr, dlogits, logits, targets_ptr, n_rows, vocab, ignore_index, reduction,
            num_items, want_grad, label_smoothing,
        )
        _ = dlogits^
    else:
        var one = ctx.enqueue_create_buffer[DType.float32](1)
        count = identical_ce_loss_dev(
            ctx, loss_ptr, row_ptr, one, logits, targets_ptr, n_rows, vocab, ignore_index, reduction,
            num_items, want_grad, label_smoothing,
        )
        _ = one^
    _ = logits^
    return count


def train_dev_copy(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], dst_off: Int, mut src: DeviceBuffer[DType.float32],
    src_off: Int, n: Int,
) raises:
    """dst[dst_off : dst_off + n] = src[src_off : src_off + n], on the device."""
    if n < 1 or dst_off < 0 or src_off < 0 or dst_off + n > len(dst) or src_off + n > len(src):
        raise Error("training device array: a device copy is outside its buffers")
    ctx.enqueue_copy(
        dst_buf=dst.create_sub_buffer[DType.float32](dst_off, n),
        src_buf=src.create_sub_buffer[DType.float32](src_off, n),
    )
    ctx.synchronize()
