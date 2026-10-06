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
from gemm.neural_backward import (
    BWD_DC_LEFT,gemm_backward_a_call,gemm_backward_b_call,
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
)
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats, GemmWorkspace, NEURAL_WORKSPACE_REUSE_ENABLED
from gemm.experiments.neural_plans import NN12
from gemm.contract import OP_NN, OP_NT, OP_TN
from training.estimator import identical_ce_loss_dev
from training.neural_identical_experiments import IDN_TRAIN_BACKWARD_SCRATCH
from training.neural_gemm_workspace import (
    IDN_TRAINING_GEMM_WORKSPACE, training_gemm_cached_into, training_gemm_cached_close,
)

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
    # NN12 scratch belongs to an output handle, and dies with that handle.
    # The existing binding pool already owns one in-order context; this is
    # not a process-wide cache indexed by raw addresses or shapes.
    var workspace: List[GemmWorkspace]

    def __init__(out self):
        self.b = List[DeviceBuffer[DType.float32]]()
        self.n = List[Int]()
        self.workspace = List[GemmWorkspace]()


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
        comptime if NEURAL_WORKSPACE_REUSE_ENABLED:
            # Finish context allocations before publishing the new handle.
            pool[].workspace.append(GemmWorkspace(ctx))
        pool[].b.append(buf^)
        pool[].n.append(n)
        h = len(pool[].n) - 1
    else:
        comptime if NEURAL_WORKSPACE_REUSE_ENABLED:
            pool[].workspace[h] = GemmWorkspace(ctx)
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
    comptime if NEURAL_WORKSPACE_REUSE_ENABLED:
        pool[].workspace[h].close(ctx)
    pool[].b[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].n[h] = 0
    comptime if IDN_TRAINING_GEMM_WORKSPACE:
        var any_live = False
        for i in range(len(pool[].n)):  # small-loop(pool: device-array handles): inspect allocation metadata, never tensor elements
            any_live = any_live or pool[].n[i] > 0
        if not any_live:
            training_gemm_cached_close(ctx)


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
    comptime if NEURAL_WORKSPACE_REUSE_ENABLED:
        var pool = _DEV_POOL.get_or_create_ptr()
        pool[].workspace[c_h].run(ctx,c,a,w,m,n,k,OP_NT)
    elif IDN_TRAINING_GEMM_WORKSPACE:
        training_gemm_cached_into(ctx, c, a, w, m, n, k, OP_NT)
        ctx.synchronize()
    else:
        var ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, n, k))
        identical_gemm_into(ctx, c, a, w, ws, m, n, k, OP_NT)
        ctx.synchronize()
        _ = ws^
    _ = a^
    _ = w^
    _ = c^
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
    comptime if NEURAL_WORKSPACE_REUSE_ENABLED:
        var pool = _DEV_POOL.get_or_create_ptr()
        var ac = gemm_backward_a_call(OP_NT,m,n,k)
        var bc = gemm_backward_b_call(OP_NT,m,n,k)
        if ac[4]==BWD_DC_LEFT:
            pool[].workspace[da_h].run(ctx,da,dc,w,ac[1],ac[2],ac[3],ac[0])
        else:
            pool[].workspace[da_h].run(ctx,da,w,dc,ac[1],ac[2],ac[3],ac[0])
        if bc[4]==BWD_DC_LEFT:
            pool[].workspace[dw_h].run(ctx,dw,dc,a,bc[1],bc[2],bc[3],bc[0])
        else:
            pool[].workspace[dw_h].run(ctx,dw,a,dc,bc[1],bc[2],bc[3],bc[0])
        ctx.synchronize()
    else:
        var na = identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k)
        var nb = identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)
        var ws_a = ctx.enqueue_create_buffer[DType.float32](max(na, nb) if IDN_TRAIN_BACKWARD_SCRATCH else na)
        var ws_b: DeviceBuffer[DType.float32]
        comptime if IDN_TRAIN_BACKWARD_SCRATCH:
            # NI36: both GEMMs enqueue onto this SAME in-order context and the
            # final wait precedes release; dA/dW themselves remain distinct.
            ws_b = ws_a.create_sub_buffer[DType.float32](0, nb)
        else:
            ws_b = ctx.enqueue_create_buffer[DType.float32](nb)
        identical_gemm_backward_a_into(ctx, da, dc, w, ws_a, m, n, k, OP_NT)
        identical_gemm_backward_b_into(ctx, dw, dc, a, ws_b, m, n, k, OP_NT)
        ctx.synchronize()
        _ = ws_a^
        _ = ws_b^
    _ = a^
    _ = w^
    _ = dc^
    _ = da^
    _ = dw^
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
