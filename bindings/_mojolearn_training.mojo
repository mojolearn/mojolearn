# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the neural-training lane, which is the optimizer
step, the global-norm gradient clip and the cross-entropy loss.

A TWELFTH EXTENSION MODULE, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here: an independently changing binding must not become a merge
point. `training/` is the newest lane in the tree and the one most likely to
gain entry points, so it gets its own `.so` rather than riding a sibling's.
All of them land in one wheel.

THE ORIGINAL THREE ENTRY POINTS expose functions over
`training/estimator.mojo`, which is itself pointer-shaped transport over
`training/checks/optimizer.mojo` and `training/checks/loss.mojo`. **NO NEW
ARITHMETIC LANDED IN THOSE WRAPPERS.** A paper draft said "neural
training is internal, not a public API"; that sentence was true and this
module is the only thing that was wrong with it.

The small-MLP additions expose bias/ReLU, ReLU backward and ascending-row
sum from `training/mlp_ops.mojo`. They require IDENTICAL and bounded shapes;
their new arithmetic needs separate root-run qualification. Existing loss
and optimizer arithmetic is unchanged.

WHERE THE MEASUREMENT STOPS, AND IT IS UNEVEN. The loss contract card (md5
`a87615d9`) and the optimizer contract card (md5 `97d160b0`) are
BYTE-IDENTICAL on an Apple M4 and an AMD MI325X at the 2026-08-28 legs, over
24 cases and 61,925 cells and 33 cases and 382,822 cells respectively.
**NO NVIDIA LEG HAS RUN**, for either card or for the training loop's
checkpoint comparison. This is a TWO-VENDOR result and nothing here may be
read as a three-vendor one.

Arrays cross as borrowed NumPy addresses; all device buffers and contexts
live for one call and no pointer is retained. The Python wrapper owns the
arrays and keeps them alive for the duration of the call
(`python/mojolearn/_arrays.py` is where that contract is written down).

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and stops
working above roughly nine arguments, so buffer addresses go positionally
and every scalar goes in one `params` list. THE ORDER OF THAT LIST IS
WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent
reordering is a wrong answer, not a failure -- swap `beta1` and `beta2` and
every step still returns a full buffer of plausible floats -- and the length
check at the top of each function is the only thing standing between a
swapped pair and a number nobody can tell is wrong.

`_mojolearn_training` IS registered in `python/mojolearn/_backend.py`'s
`_MODULES` and `_build_script`, and in `packaging/macos/build_release_wheel.
sh`'s `BUILD_SCRIPTS` and `EXT_NAMES`. DEVIATION 869 is why both halves are
named here: an extension absent from `_MODULES` is never re-pointed, so
under `MOJOLEARN_NUMERIC_MODE=identical` a plain import resolves to the FAST
binary sitting beside it and returns fast arithmetic under the identical
label. An extension absent from `EXT_NAMES` ships STALE rather than absent,
which is worse, because the wheel then carries a binary from an earlier
build with no sign that it did.

DEVIATIONS 1590 through 1599 are this surface's. 1590 is
`training/estimator.mojo` and this file; 1591 through 1599 are unassigned.
"""

from training.neural_arithmetic_profile import neural_training_profile
from bindings.residual_dropout_boundary import residual_dropout_binding, residual_dropout_backward_binding
from bindings.neural_gemm_boundary import neural_gemm_binding
from training.neural_session_mlp import nn_mlp_sessions_device


# DEVIATION 2486: shared byte-preserving host copies.
from bindings.hostptr import f32_ptr, i32_ptr
from std.os import abort, getenv
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"
from checks.vendor import COMPILED_VENDOR
from max.gpu.host import DeviceContext, HostBuffer

from training.clip_multi_gpu import parallel_clip_grad_norm_host, clip_pool_fault_available
from training.accumulate_multi_gpu import parallel_accumulate_host, accumulate_pool_fault_available
from training.optimizer_multi_gpu import parallel_optimizer_step_host
from training.estimator import (
    identical_clip_grad_norm_addrs_host,
    identical_ce_loss_host,
    identical_clip_grad_norm_host,
    identical_optimizer_step_host,
    identical_optimizer_step_resident_host, opt_download_staged, opt_pool_buffers,
)
# lane fam2-neural (2026-10-04): device tensors for the training ops
from training.dev_tensors import (
    IDN_TRAIN_DEV_TENSORS, train_dev_alloc, train_dev_free, train_dev_put, train_dev_get, train_dev_view,
    train_dev_copy, train_linear_forward_dev, train_linear_backward_dev, train_ce_loss_dev,
)
# lane fam2-neural (2026-10-04): parameters and gradients resident across steps
from training.estimator import (
    identical_optimizer_step_resident_io, IDN_OPT_PARAMS_RESIDENT, OPT_IO_ALL, IDN_MAXIMIZE_DEV,
    maximize_negate_device,
)
from std.ffi import _Global
from max.gpu.host import DeviceBuffer
from training.mlp_ops import (
    mlp_bias_activation_host, mlp_relu_backward_host, mlp_sum_rows_host,
    mlp_train_step_host, mlp_validate_shape,
)
#: lane afn-mlp (2026-10-03): the Apple FAST resident session, registered
#: only under FAST + Apple + `-D MOJOLEARN_AFN_MLP_RESIDENT` / `_MULTISTEP`
#: / `_ALL` (training/mlp_fast.mojo); absent from every other build.
from training.mlp_fast import (
    MLP_MULTISTEP, MLP_RESIDENT,
    mlp_resident_close_host, mlp_resident_download_host, mlp_resident_open_host,
    mlp_resident_step_host, mlp_resident_upload_host,
)
from training.checks.optimizer_contract import microbatch_split_is_identical
from training.chunked_lm_head_v2 import (
    chunked_lm_head_v2_loss_host, chunked_lm_head_v2_train_host,
)
from training.samba_ops import (
    samba_accumulate_host,
    samba_embedding_backward_host,
    samba_embedding_forward_host,
    samba_head_loss_host,
    samba_linear_backward_host,
    samba_linear_forward_host,
    samba_rms_norm_backward_host,
    samba_rms_norm_forward_host,
)
# lane afn-samba (2026-10-03): the fused Samba entries exist only in an
# Apple FAST build with MOJOLEARN_AFN_SAMBA_FUSE (or _ALL); AFN_SAMBA_FUSE
# is false everywhere else and nothing below registers.
from training.samba_afn import (
    AFN_SAMBA_FUSE,
    samba_afn_embedding_backward_tied_binding,
    samba_afn_norm_head_forward_binding,
    samba_afn_tail_train_binding,
)
from core.philox_neural import neural_rng_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _negate_through_device(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises:
    """dst[0:n] = `maximize_negate(src[0:n])` ON THE DEVICE: one bulk upload,
    one sign-flip launch (`maximize_negate_device`), one bulk download
    (cpu3-bindings; the host loops it replaces flipped n floats on the CPU).
    The flip is exact, so the bits are the host loop's on every column.
    `src` and `dst` may be the same buffer."""
    if n <= 0:
        return
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=src)
    maximize_negate_device(ctx, d, n)
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d)
    ctx.synchronize()
    _ = d^


def training_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `gbdt_numeric_mode`, and
    for the same reason: the wrapper reads it once and refuses to run if the
    binary it loaded disagrees with the mode the package asked for. A
    wrong-arm measurement that is correctly labelled by accident is the
    failure this prevents, and a boolean could not do that job once a third
    tier existed, because DETERMINISTIC answered 0 and read back as "fast".

    **UNDER FAST THIS LANE HAS NO CONTRACT AT ALL.** `checks/numerics.mojo`
    compiles the pinned helpers away, so the loss and the optimizer are the
    same loops spelled in whatever arithmetic the vendor's compiler chose.
    They still run and they still train; they promise nothing about bits.
    """
    return PythonObject(GLOBAL_NUMERIC_MODE)


def training_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back: the answer
    comes from the binary that actually loaded, never from the directory it
    sat in or from the environment. `python/mojolearn/_backend.py` refuses at
    import when this disagrees with the vendor directory the set was loaded
    from."""
    return PythonObject(String(COMPILED_VENDOR))


def clip_pool_fault_available_binding() raises -> PythonObject:
    return PythonObject(clip_pool_fault_available())


def clip_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)


def accumulate_pool_fault_available_binding() raises -> PythonObject:
    return PythonObject(accumulate_pool_fault_available())


def accumulate_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)


def optimizer_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)


def optimizer_step_binding(
    param_addr: PythonObject,
    grad_addr: PythonObject,
    m_addr: PythonObject,
    v_addr: PythonObject,
    offsets_addr: PythonObject,
    init_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One step of `mojolearn.identical.optimizer.fp32.v1`. Returns `N`, the
    total element count `offsets[J]`.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_training_impl.py`):

        0   n_tensors       J; `offsets` holds J + 1 int32
        1   kind            0 = SGD, 1 = Adam, 2 = AdamW
        2   t               the step number, ONE-BASED; first step is 1
        3   nesterov        0 or 1
        4   lr              (float)
        5   beta1           (float; Adam and AdamW only)
        6   beta2           (float; Adam and AdamW only)
        7   eps             (float; Adam and AdamW only)
        8   weight_decay    (float)
        9   momentum        (float; SGD only)
        10  dampening       (float; SGD only)
        11  max_norm        (float; <= 0 turns the gradient-norm clip OFF)
        12  maximize        0 or 1, OPTIONAL (a 12-value list is 0). 1 runs
                            the step on the sign-flipped gradient,
                            training/maximize.mojo (DEVIATION 6200); the
                            caller's gradient is never left negated

    SLOT 2 IS THE TRAP IN THIS LIST. `t` is the OPTIMIZER's step counter and
    it is one-based, so a caller looping `for t in range(n)` and passing `t`
    directly divides by `1 - beta^0`, which is exactly zero.
    `training/estimator.mojo` refuses `t < 1` by name rather than returning
    the infinities that would follow.

    SLOTS 5 THROUGH 7 ARE READ ONLY BY ADAM AND ADAMW; slots 9 and 10 only
    by SGD. They are all present in every call because one params list for
    three algorithms is one order to keep in step instead of three.

    THE BUFFERS, and every size is computable by the caller before the call.
    Let `N = offsets[J]`.

        `param_addr`    N float32, read and WRITTEN IN PLACE
        `grad_addr`     N float32, read; WRITTEN IN PLACE when the clip runs
        `m_addr`        N float32, read and WRITTEN. Adam's `exp_avg`,
                        SGD's `momentum_buffer`
        `v_addr`        N float32, read and WRITTEN by Adam and AdamW.
                        **SGD NEVER TOUCHES IT AND IT MUST STILL BE N FLOATS
                        LONG**: the certified entry takes one signature for
                        both algorithms
        `offsets_addr`  J + 1 int32, ascending, `offsets[0] == 0`
        `init_addr`     J int32, read and WRITTEN. SGD's per-tensor
                        `buf_initialized` flag (contract 7.3b); CARRIED
                        STATE that belongs in a checkpoint beside `m`
        `info_addr`     3 float32, written on every call:

                            0  clip_ran    1.0 or +0.0
                            1  total_norm  the pre-clip global norm, or +0.0
                            2  coef        the clamped coefficient, or +0.0

    Slot 0 of `info` is what a caller branches on. A coefficient of 1.0 says
    the clip RAN and found nothing to do, which is a different fact from the
    clip not running, and the oracle draws the same distinction by leaving
    its `clip.*` stages empty rather than filling them with a 1.
    """
    if len(params) != 12 and len(params) != 13:
        raise Error(
            "optimizer_step: params must contain 12 or 13 values, got "
            + String(len(params))
        )
    var pp = _f32_ptr(Int(py=param_addr))
    var gp = _f32_ptr(Int(py=grad_addr))
    var maximize = len(params) == 13 and Int(py=params[12]) != 0
    var mp = _f32_ptr(Int(py=m_addr))
    var vp = _f32_ptr(Int(py=v_addr))
    var op = _i32_ptr(Int(py=offsets_addr))
    var ip = _i32_ptr(Int(py=init_addr))
    var fp = _f32_ptr(Int(py=info_addr))
    var n_tensors = Int(py=params[0])
    var kind = Int(py=params[1])
    var t = Int(py=params[2])
    var nesterov = Int(py=params[3])
    var lr = Float32(Float64(py=params[4]))
    var beta1 = Float32(Float64(py=params[5]))
    var beta2 = Float32(Float64(py=params[6]))
    var eps = Float32(Float64(py=params[7]))
    var weight_decay = Float32(Float64(py=params[8]))
    var momentum = Float32(Float64(py=params[9]))
    var dampening = Float32(Float64(py=params[10]))
    var max_norm = Float32(Float64(py=params[11]))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        if maximize:
            # The step reads a negated COPY, so the caller's gradient is never
            # negated, not even for the length of the call; a clipped
            # gradient is written back through the same sign flip.
            if n_tensors < 1 or op[n_tensors] < Int32(0):
                raise Error("optimizer_step: maximize needs a registry with offsets[J] >= 0")
            var n_flat = Int(op[n_tensors])
            var neg = List[Float32](length=max(n_flat, 1), fill=Float32(0.0))
            var np_ = neg.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            _negate_through_device(ctx, gp, np_, n_flat)
            n_total = parallel_optimizer_step_host(
                ctx, pp, np_, mp, vp, op, ip, fp, n_tensors, kind, t, nesterov,
                lr, beta1, beta2, eps, weight_decay, momentum, dampening,
                max_norm,
            )
            if max_norm > Float32(0.0):
                _negate_through_device(ctx, np_, gp, n_flat)
            _ = neg^  # `np_` is untracked: keep the copy alive past its last use
        else:
            n_total = parallel_optimizer_step_host(
                ctx, pp, gp, mp, vp, op, ip, fp, n_tensors, kind, t, nesterov,
                lr, beta1, beta2, eps, weight_decay, momentum, dampening,
                max_norm,
            )
    return PythonObject(n_total)


# ===========================================================================
# THE RESIDENT OPTIMIZER MOMENTS (lane/neural-pass4, 2026-09-30).
#
# `optimizer_step` moves seven registry-sized buffers a step: parameters,
# gradient, `m` and `v` up, parameters, `m` and `v` down. On the Samba step at
# the board shape (5.7 M parameters, L4, PCIe) that transport is 55.7 of
# 147.6 ms, more than the Mamba-3 backward. `m` and `v` are the optimizer's
# own state and nothing else reads them between steps, so a Python optimizer
# may keep them here: `optimizer_resident_open` allocates a pair on the
# binding's context (zero filled, as `zeros` filled them), `optimizer_resident_step`
# runs the SAME `identical_optimizer_step` on that pair (four transfers fewer
# a step), `optimizer_resident_download` / `_upload` move them for a
# `state_dict` / `load_state_dict`, `optimizer_resident_close` frees them.
# Storage: `std.ffi._Global` (the pattern of core/neural_context.mojo), one
# slot per tier. A handle is an index into the pool; a closed handle's slot
# is reused by the next open.
# ===========================================================================


struct _OptPool(Defaultable, Movable):
    var m: List[DeviceBuffer[DType.float32]]
    var v: List[DeviceBuffer[DType.float32]]
    #: the handle's device copies of the parameters and the gradients (lane
    #: neural-pass26, 2026-10-01): the resident step uploads into these and
    #: downloads the parameters from them, where it created two fresh
    #: buffers of n floats every step (two 64 MB allocations a step at the
    #: board's 16.8M parameters: 72 to 81 ms a step on the M4, 1.1 s for the
    #: board's ten steps on every Mac, against a few ms of copies)
    var p: List[DeviceBuffer[DType.float32]]
    var g: List[DeviceBuffer[DType.float32]]
    #: pinned host staging for the two uploads: the caller's arrays are
    #: memcpy'd here and DMA'd to the device, where a raw host pointer handed
    #: to the device paid a cold first-use mapping (about 20 ms per 64 MB on
    #: the M4 for every new gradient array) and ran at a few GB/s
    var sp: List[HostBuffer[DType.float32]]
    var sg: List[HostBuffer[DType.float32]]
    #: floats held by each slot; 0 marks a free slot
    var n: List[Int]

    def __init__(out self):
        self.m = List[DeviceBuffer[DType.float32]]()
        self.v = List[DeviceBuffer[DType.float32]]()
        self.p = List[DeviceBuffer[DType.float32]]()
        self.g = List[DeviceBuffer[DType.float32]]()
        self.sp = List[HostBuffer[DType.float32]]()
        self.sg = List[HostBuffer[DType.float32]]()
        self.n = List[Int]()


comptime _OPT_POOL_NAME = "MojoNeuralTrainingOptPoolIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingOptPoolFast"
comptime _OPT_POOL = _Global[StorageType=_OptPool, name=_OPT_POOL_NAME, init_fn=_OptPool.__init__]


def _opt_pool_handle(handle: PythonObject, want: Int) raises -> Int:
    var h = Int(py=handle)
    var pool = _OPT_POOL.get_or_create_ptr()
    if h < 0 or h >= len(pool[].n) or pool[].n[h] == 0:
        raise Error("optimizer_resident: handle " + String(h) + " is not open")
    if want > 0 and pool[].n[h] != want:
        raise Error(
            "optimizer_resident: handle " + String(h) + " holds "
            + String(pool[].n[h]) + " floats, the call names " + String(want)
        )
    return h


def optimizer_resident_open_binding(n_total: PythonObject) raises -> PythonObject:
    """Allocate a resident `(m, v)` pair of `n_total` floats each, zero
    filled. Returns the handle."""
    var n = Int(py=n_total)
    if n < 1:
        raise Error("optimizer_resident_open: n_total must be >= 1, got " + String(n))
    var h = -1
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        var pool = _OPT_POOL.get_or_create_ptr()
        var m = ctx.enqueue_create_buffer[DType.float32](n)
        var v = ctx.enqueue_create_buffer[DType.float32](n)
        m.enqueue_fill(Float32(0.0))
        v.enqueue_fill(Float32(0.0))
        # The fills are ordered before every later use on the one in-order
        # context, and every host read (the download) waits.
        var n_pool = n if opt_pool_buffers() else 1
        var pbuf = ctx.enqueue_create_buffer[DType.float32](n_pool)
        var gbuf = ctx.enqueue_create_buffer[DType.float32](n_pool)
        var n_stage = n if opt_download_staged() else 1
        var spin = ctx.enqueue_create_host_buffer[DType.float32](n_stage)
        var sgin = ctx.enqueue_create_host_buffer[DType.float32](n_stage)
        for j in range(len(pool[].n)):  # small-loop(n: optimizer handle slots in the pool): finds a free handle, not data
            if pool[].n[j] == 0 and h < 0:
                h = j
        if h < 0:
            pool[].m.append(m^)
            pool[].v.append(v^)
            pool[].p.append(pbuf^)
            pool[].g.append(gbuf^)
            pool[].sp.append(spin^)
            pool[].sg.append(sgin^)
            pool[].n.append(n)
            h = len(pool[].n) - 1
        else:
            pool[].m[h] = m^
            pool[].v[h] = v^
            pool[].p[h] = pbuf^
            pool[].g[h] = gbuf^
            pool[].sp[h] = spin^
            pool[].sg[h] = sgin^
            pool[].n[h] = n
    return PythonObject(h)


def optimizer_resident_close_binding(handle: PythonObject) raises -> PythonObject:
    """Free a handle's pair (after the queue drains). Returns the handle."""
    var h = _opt_pool_handle(handle, 0)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        ctx.synchronize()
        var pool = _OPT_POOL.get_or_create_ptr()
        pool[].m[h] = ctx.enqueue_create_buffer[DType.float32](1)
        pool[].v[h] = ctx.enqueue_create_buffer[DType.float32](1)
        pool[].p[h] = ctx.enqueue_create_buffer[DType.float32](1)
        pool[].g[h] = ctx.enqueue_create_buffer[DType.float32](1)
        pool[].sp[h] = ctx.enqueue_create_host_buffer[DType.float32](1)
        pool[].sg[h] = ctx.enqueue_create_host_buffer[DType.float32](1)
        pool[].n[h] = 0
    return PythonObject(h)


def optimizer_resident_download_binding(
    handle: PythonObject, m_addr: PythonObject, v_addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """The pair to host memory (`n_total` floats each). Returns `n_total`."""
    var n = Int(py=n_total)
    var h = _opt_pool_handle(handle, n)
    var mp = _f32_ptr(Int(py=m_addr))
    var vp = _f32_ptr(Int(py=v_addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        var pool = _OPT_POOL.get_or_create_ptr()
        ctx.enqueue_copy(dst_ptr=mp, src_buf=pool[].m[h])
        ctx.enqueue_copy(dst_ptr=vp, src_buf=pool[].v[h])
        ctx.synchronize()
    return PythonObject(n)


def optimizer_resident_upload_binding(
    handle: PythonObject, m_addr: PythonObject, v_addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """Host memory (`n_total` floats each) into the pair. Returns `n_total`."""
    var n = Int(py=n_total)
    var h = _opt_pool_handle(handle, n)
    var mp = _f32_ptr(Int(py=m_addr))
    var vp = _f32_ptr(Int(py=v_addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        var pool = _OPT_POOL.get_or_create_ptr()
        ctx.enqueue_copy(dst_buf=pool[].m[h], src_ptr=mp)
        ctx.enqueue_copy(dst_buf=pool[].v[h], src_ptr=vp)
        ctx.synchronize()
    return PythonObject(n)


def optimizer_resident_step_binding(
    param_addr: PythonObject,
    grad_addr: PythonObject,
    offsets_addr: PythonObject,
    init_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
    handle: PythonObject,
) raises -> PythonObject:
    """`optimizer_step` with `m` and `v` taken from the handle's resident
    pair instead of two host buffers: the same `params` list (its 12 or 13
    slots, word for word), the same buffers otherwise, the same step. The
    single-device route only (`MOJOLEARN_OPTIMIZER_DEVICE_COUNT` is the
    per-call optimizer's; the Python side keeps that route there). Returns
    `N`."""
    if len(params) != 12 and len(params) != 13:
        raise Error(
            "optimizer_resident_step: params must contain 12 or 13 values, got "
            + String(len(params))
        )
    var h = _opt_pool_handle(handle, 0)
    var pp = _f32_ptr(Int(py=param_addr))
    var gp = _f32_ptr(Int(py=grad_addr))
    var maximize = len(params) == 13 and Int(py=params[12]) != 0
    var op = _i32_ptr(Int(py=offsets_addr))
    var ip = _i32_ptr(Int(py=init_addr))
    var fp = _f32_ptr(Int(py=info_addr))
    var n_tensors = Int(py=params[0])
    var kind = Int(py=params[1])
    var t = Int(py=params[2])
    var nesterov = Int(py=params[3])
    var lr = Float32(Float64(py=params[4]))
    var beta1 = Float32(Float64(py=params[5]))
    var beta2 = Float32(Float64(py=params[6]))
    var eps = Float32(Float64(py=params[7]))
    var weight_decay = Float32(Float64(py=params[8]))
    var momentum = Float32(Float64(py=params[9]))
    var dampening = Float32(Float64(py=params[10]))
    var max_norm = Float32(Float64(py=params[11]))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        var pool = _OPT_POOL.get_or_create_ptr()
        if maximize:
            if n_tensors < 1 or op[n_tensors] < Int32(0):
                raise Error("optimizer_resident_step: maximize needs a registry with offsets[J] >= 0")
            var n_flat = Int(op[n_tensors])
            var neg = List[Float32](length=max(n_flat, 1), fill=Float32(0.0))
            var np_ = neg.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            _negate_through_device(ctx, gp, np_, n_flat)
            # the parameter/gradient device buffers: the handle's pooled pair on
            # Apple, a fresh pair per step elsewhere (lane neural-pass34)
            var pooled = opt_pool_buffers()
            var pb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].n[h])
            var gb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].n[h])
            if pooled:
                n_total = identical_optimizer_step_resident_host(
                    ctx, pp, np_, pool[].m[h], pool[].v[h], pool[].p[h], pool[].g[h], pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                    kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                    momentum, dampening, max_norm,
                )
            else:
                n_total = identical_optimizer_step_resident_host(
                    ctx, pp, np_, pool[].m[h], pool[].v[h], pb, gb, pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                    kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                    momentum, dampening, max_norm,
                )
            if max_norm > Float32(0.0):
                _negate_through_device(ctx, np_, gp, n_flat)
            _ = neg^  # `np_` is untracked: keep the copy alive past its last use
        else:
            # the parameter/gradient device buffers: the handle's pooled pair on
            # Apple, a fresh pair per step elsewhere (lane neural-pass34)
            var pooled = opt_pool_buffers()
            var pb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].n[h])
            var gb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].n[h])
            if pooled:
                n_total = identical_optimizer_step_resident_host(
                    ctx, pp, gp, pool[].m[h], pool[].v[h], pool[].p[h], pool[].g[h], pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                    kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                    momentum, dampening, max_norm,
                )
            else:
                n_total = identical_optimizer_step_resident_host(
                    ctx, pp, gp, pool[].m[h], pool[].v[h], pb, gb, pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                    kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                    momentum, dampening, max_norm,
                )
    return PythonObject(n_total)


# ---------------------------------------------------------------------------
# lane fam2-neural (2026-10-04): THE HANDLE'S PARAMETERS AND GRADIENT, RESIDENT.
# `optimizer_resident_step` still moved three registry-sized buffers a step
# (parameters up, gradient up, parameters down). These entries let the
# caller keep the parameters (and, when its gradient is already there, the
# gradient) in the handle's device buffers between steps:
#   optimizer_resident_put(handle, which, addr, n)   host -> device (0 = parameters, 1 = gradient)
#   optimizer_resident_get(handle, which, addr, n)   device -> host
#   optimizer_resident_step_io(..., [handle, io])    the step, moving only the transfers `io` names
# The step is `identical_optimizer_step` on the same buffers: no bit moves.
# Registered only under IDN_OPT_PARAMS_RESIDENT.


def _opt_dev_pair(ctx: DeviceContext, h: Int) raises:
    """The handle's parameter and gradient device buffers at full size
    (off Apple they are one float until first needed here)."""
    var pool = _OPT_POOL.get_or_create_ptr()
    var n = pool[].n[h]
    if len(pool[].p[h]) < n:
        pool[].p[h] = ctx.enqueue_create_buffer[DType.float32](n)
    if len(pool[].g[h]) < n:
        pool[].g[h] = ctx.enqueue_create_buffer[DType.float32](n)


def optimizer_resident_put_binding(
    handle: PythonObject, which: PythonObject, addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """Host `n_total` floats into the handle's parameter (which = 0) or
    gradient (which = 1) device buffer. Returns `n_total`."""
    var n = Int(py=n_total)
    var h = _opt_pool_handle(handle, n)
    var w = Int(py=which)
    if w != 0 and w != 1:
        raise Error("optimizer_resident_put: which is 0 (parameters) or 1 (gradient)")
    var src = _f32_ptr(Int(py=addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        _opt_dev_pair(ctx, h)
        var pool = _OPT_POOL.get_or_create_ptr()
        if w == 0:
            ctx.enqueue_copy(dst_buf=pool[].p[h], src_ptr=src)
        else:
            ctx.enqueue_copy(dst_buf=pool[].g[h], src_ptr=src)
        ctx.synchronize()
    return PythonObject(n)


def optimizer_resident_get_binding(
    handle: PythonObject, which: PythonObject, addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """The handle's parameter (which = 0) or gradient (which = 1) device
    buffer into host memory (`n_total` floats). Returns `n_total`."""
    var n = Int(py=n_total)
    var h = _opt_pool_handle(handle, n)
    var w = Int(py=which)
    if w != 0 and w != 1:
        raise Error("optimizer_resident_get: which is 0 (parameters) or 1 (gradient)")
    var dst = _f32_ptr(Int(py=addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        _opt_dev_pair(ctx, h)
        var pool = _OPT_POOL.get_or_create_ptr()
        if w == 0:
            ctx.enqueue_copy(dst_ptr=dst, src_buf=pool[].p[h])
        else:
            ctx.enqueue_copy(dst_ptr=dst, src_buf=pool[].g[h])
        ctx.synchronize()
    return PythonObject(n)


def maximize_dev_available_binding() raises -> PythonObject:
    """Present when `Optimizer.step` should run maximize through
    `optimizer_resident_step_io` (the sign flip on the device)."""
    return PythonObject(1)


def optimizer_resident_step_io_binding(
    param_addr: PythonObject,
    grad_addr: PythonObject,
    offsets_addr: PythonObject,
    init_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
    handle_io: PythonObject,
) raises -> PythonObject:
    """`optimizer_resident_step` moving only the transfers `io` names
    (handle_io = [handle, io]; io bits: 1 parameters up, 2 gradient up, 4
    parameters down, 8 clipped gradient down). A transfer left out means
    that buffer is already in the handle's device buffer
    (`optimizer_resident_put`, or the previous step's result) and its
    address here is not read (pass any valid float address, e.g. `info`).
    `params` is `optimizer_step`'s list, word for word. Maximize negates
    the gradient on the device (`maximize_negate_device`), wherever it came
    from. Returns `N`."""
    if len(params) != 12 and len(params) != 13:
        raise Error(
            "optimizer_resident_step_io: params must contain 12 or 13 values, got "
            + String(len(params))
        )
    if len(handle_io) != 2:
        raise Error("optimizer_resident_step_io: the last argument is [handle, io]")
    var h = _opt_pool_handle(handle_io[0], 0)
    var io = Int(py=handle_io[1])
    if io < 0 or io > OPT_IO_ALL:
        raise Error("optimizer_resident_step_io: io is a mask of 1, 2, 4 and 8")
    var pp = _f32_ptr(Int(py=param_addr))
    var gp = _f32_ptr(Int(py=grad_addr))
    var maximize = len(params) == 13 and Int(py=params[12]) != 0
    var op = _i32_ptr(Int(py=offsets_addr))
    var ip = _i32_ptr(Int(py=init_addr))
    var fp = _f32_ptr(Int(py=info_addr))
    var n_tensors = Int(py=params[0])
    var kind = Int(py=params[1])
    var t = Int(py=params[2])
    var nesterov = Int(py=params[3])
    var lr = Float32(Float64(py=params[4]))
    var beta1 = Float32(Float64(py=params[5]))
    var beta2 = Float32(Float64(py=params[6]))
    var eps = Float32(Float64(py=params[7]))
    var weight_decay = Float32(Float64(py=params[8]))
    var momentum = Float32(Float64(py=params[9]))
    var dampening = Float32(Float64(py=params[10]))
    var max_norm = Float32(Float64(py=params[11]))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        _opt_dev_pair(ctx, h)
        var pool = _OPT_POOL.get_or_create_ptr()
        n_total = identical_optimizer_step_resident_io(
            ctx, pp, gp, pool[].m[h], pool[].v[h], pool[].p[h], pool[].g[h], pool[].sp[h], pool[].sg[h], op, ip,
            fp, n_tensors, kind, t, nesterov, lr, beta1, beta2, eps, weight_decay, momentum, dampening,
            max_norm, io, maximize,
        )
    return PythonObject(n_total)


# ---------------------------------------------------------------------------
# lane fam2-neural (2026-10-04): DEVICE TENSORS (training/dev_tensors.mojo).
# Float32 device arrays by handle, and the linear / cross-entropy ops and
# the resident optimizer reading and writing them, so a training loop's
# N * V and parameter-sized arrays never cross the bus. Registered only
# under IDN_TRAIN_DEV_TENSORS.


def _handles(handles: PythonObject, want: Int, name: String) raises -> List[Int]:
    if len(handles) != want:
        raise Error(name + ": handles must contain " + String(want) + " entries, got " + String(len(handles)))
    var out = List[Int]()
    for i in range(want):  # small-loop(want: device array handles of one op call): reads handle list, not data
        out.append(Int(py=handles[i]))
    return out^


def train_dev_alloc_binding(n: PythonObject) raises -> PythonObject:
    """A device array of `n` floats; returns its handle."""
    var count = Int(py=n)
    var h = -1
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        h = train_dev_alloc(ctx, count)
    return PythonObject(h)


def train_dev_free_binding(handle: PythonObject) raises -> PythonObject:
    var h = Int(py=handle)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        train_dev_free(ctx, h)
    return PythonObject(h)


def train_dev_put_binding(handle: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Host `n` floats at `addr` into the handle."""
    var h = Int(py=handle)
    var count = Int(py=n)
    var src = _f32_ptr(Int(py=addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        train_dev_put(ctx, h, src, count)
    return PythonObject(count)


def train_dev_get_binding(handle: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The handle's first `n` floats into host memory at `addr`."""
    var h = Int(py=handle)
    var count = Int(py=n)
    var dst = _f32_ptr(Int(py=addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        train_dev_get(ctx, h, dst, count)
    return PythonObject(count)


def linear_forward_dev_binding(handles: PythonObject, params: PythonObject) raises -> PythonObject:
    """`linear_forward` on device arrays. handles = [c (written), a, w];
    params = [m, n, k]."""
    var hs = _handles(handles, 3, "linear_forward_dev")
    _params(params, 3, "linear_forward_dev")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = train_linear_forward_dev(ctx, hs[0], hs[1], hs[2], m, n, k)
    return PythonObject(count)


def linear_backward_dev_binding(handles: PythonObject, params: PythonObject) raises -> PythonObject:
    """`linear_backward` on device arrays. handles = [da (written), dw
    (written), dc, a, w]; params = [m, n, k]."""
    var hs = _handles(handles, 5, "linear_backward_dev")
    _params(params, 3, "linear_backward_dev")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = train_linear_backward_dev(ctx, hs[0], hs[1], hs[2], hs[3], hs[4], m, n, k)
    return PythonObject(count)


def ce_loss_dev_binding(
    loss_addr: PythonObject,
    row_addr: PythonObject,
    targets_addr: PythonObject,
    handles: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ce_loss` with the logits and their gradient as device arrays.
    handles = [dlogits (written when want_grad != 0; any open handle
    otherwise), logits]; `params` is `ce_loss`'s list, word for word (7
    values); loss, row and targets are host addresses as there. Returns
    `count`."""
    if len(params) != 7:
        raise Error("ce_loss_dev: params must contain 7 values, got " + String(len(params)))
    var hs = _handles(handles, 2, "ce_loss_dev")
    var lp = _f32_ptr(Int(py=loss_addr))
    var rp = _f32_ptr(Int(py=row_addr))
    var tp = _i32_ptr(Int(py=targets_addr))
    var n_rows = Int(py=params[0])
    var vocab = Int(py=params[1])
    var ignore_index = Int(py=params[2])
    var reduction = Int(py=params[3])
    var num_items = Int(py=params[4])
    var want_grad = Int(py=params[5])
    var label_smoothing = Float32(Float64(py=params[6]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = train_ce_loss_dev(
            ctx, lp, rp, hs[0], hs[1], tp, n_rows, vocab, ignore_index, reduction, num_items, want_grad,
            label_smoothing,
        )
    return PythonObject(count)


def optimizer_resident_copy_dev_binding(handle: PythonObject, params: PythonObject) raises -> PythonObject:
    """A device-to-device copy between an optimizer handle's parameter or
    gradient buffer and a device array. params = [which (0 parameters, 1
    gradient), tensor handle, offset (floats, into the optimizer's flat
    registry), n, direction (0: tensor -> optimizer, 1: optimizer ->
    tensor)]. Returns `n`."""
    _params(params, 5, "optimizer_resident_copy_dev")
    var h = _opt_pool_handle(handle, 0)
    var which = Int(py=params[0])
    var th = Int(py=params[1])
    var off = Int(py=params[2])
    var n = Int(py=params[3])
    var direction = Int(py=params[4])
    if (which != 0 and which != 1) or (direction != 0 and direction != 1):
        raise Error("optimizer_resident_copy_dev: which and direction are 0 or 1")
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        _opt_dev_pair(ctx, h)
        var pool = _OPT_POOL.get_or_create_ptr()
        var tv = train_dev_view(ctx, th, n, "the optimizer copy")
        if which == 0:
            if direction == 0:
                train_dev_copy(ctx, pool[].p[h], off, tv, 0, n)
            else:
                train_dev_copy(ctx, tv, 0, pool[].p[h], off, n)
        else:
            if direction == 0:
                train_dev_copy(ctx, pool[].g[h], off, tv, 0, n)
            else:
                train_dev_copy(ctx, tv, 0, pool[].g[h], off, n)
        _ = tv^
    return PythonObject(n)


def clip_grad_norm_binding(
    grad_addr: PythonObject,
    offsets_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`torch.nn.utils.clip_grad_norm_` at `norm_type = 2`, section 3 of the
    optimizer contract. Scales the gradient IN PLACE. Returns `N`.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_training_impl.py`):

        0  n_tensors    J; `offsets` holds J + 1 int32
        1  max_norm     (float; must be > 0, refused otherwise)

    THE BUFFERS. Let `N = offsets[J]`.

        `grad_addr`     N float32, read and WRITTEN IN PLACE
        `offsets_addr`  J + 1 int32, ascending, `offsets[0] == 0`
        `info_addr`     2 float32: `0` the pre-clip global norm, `1` the
                        clamped coefficient

    `max_norm <= 0` is REFUSED here rather than treated as "no clipping":
    reaching this entry point is the clip running. The optimizer step takes
    `max_norm <= 0` as OFF because it has another job to do either way.

    THE PER-TENSOR ORDER IS THE CALLER'S `offsets` ORDER AND NOTHING ELSE.
    The reference folds twice, a norm per tensor and then a norm over those
    (contract 3.1), and `j` is the `param_id` whose ascending order fixes the
    cross-tensor summation. A registry that is stable across runs is the only
    thing that makes the number reproducible; reordering the tensors is a
    different, equally valid, different-bits answer.
    """
    if len(params) != 2:
        raise Error(
            "clip_grad_norm: params must contain 2 values, got "
            + String(len(params))
        )
    var gp = _f32_ptr(Int(py=grad_addr))
    var op = _i32_ptr(Int(py=offsets_addr))
    var fp = _f32_ptr(Int(py=info_addr))
    var n_tensors = Int(py=params[0])
    var max_norm = Float32(Float64(py=params[1]))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n_total = parallel_clip_grad_norm_host(
            ctx, gp, op, fp, n_tensors, max_norm,
        )
    return PythonObject(n_total)


def clip_grad_norm_multi_binding(
    grad_addrs: PythonObject,
    offsets_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`clip_grad_norm_binding` from the caller's J tensors in place
    (`grad_addrs` a list of J float32 buffer addresses, tensor j's
    `offsets[j] .. offsets[j+1]` values), so no flat host copy is packed
    or unpacked; the same `identical_clip_grad_norm` over the same device
    buffer (lane/neural-net-experiment). Single device only: a
    MOJOLEARN_OPTIMIZER_DEVICE_COUNT above 1 is refused here and takes
    the packed entry."""
    if len(params) != 2:
        raise Error(
            "clip_grad_norm_multi: params must contain 2 values, got "
            + String(len(params))
        )
    if String(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1")) != "1":
        raise Error("clip_grad_norm_multi: one device only; the packed clip_grad_norm serves several")
    var n_tensors = Int(py=params[0])
    if len(grad_addrs) != n_tensors:
        raise Error(
            "clip_grad_norm_multi: " + String(len(grad_addrs)) + " addresses for "
            + String(n_tensors) + " tensors"
        )
    var addrs = List[Int]()
    for j in range(n_tensors):  # small-loop(n_tensors: gradient tensors of the model): one pointer per tensor, not data
        var a = Int(py=grad_addrs[j])
        if a == 0:
            raise Error("clip_grad_norm_multi: null gradient address at " + String(j))
        addrs.append(a)
    var op = _i32_ptr(Int(py=offsets_addr))
    var fp = _f32_ptr(Int(py=info_addr))
    var max_norm = Float32(Float64(py=params[1]))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n_total = identical_clip_grad_norm_addrs_host(
            ctx, addrs, op, fp, n_tensors, max_norm,
        )
    return PythonObject(n_total)


def ce_loss_binding(
    loss_addr: PythonObject,
    row_addr: PythonObject,
    dlogits_addr: PythonObject,
    logits_addr: PythonObject,
    targets_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`mojolearn.identical.loss.ce.fp32.v1`, forward and optionally
    backward, in ONE call. Returns `count`, the number of rows whose target
    is not `ignore_index`.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_training_impl.py`):

        0  n_rows
        1  vocab
        2  ignore_index      (torch's default is -100)
        3  reduction         0 = none, 1 = sum, 2 = mean
        4  num_items         < 1 means "not supplied", the MEAN arm's own
                             default
        5  want_grad         0 = forward only, 1 = also write dlogits
        6  label_smoothing   (float; the contract's `eps`)

    SLOT 6 IS THE TRAP IN THIS LIST. `eps == 0` selects a DIFFERENT KERNEL
    rather than a bit-inert branch (contract 6.2(c), DEVIATION 1155), so it
    is not a knob that quietly does nothing at its default. Passing a tiny
    nonzero value where zero was meant changes which code runs.

    THE BUFFERS. Let `N = n_rows` and `V = vocab`.

        `loss_addr`     1 float32, written when `reduction != 0`; left +0.0
                        under `REDUCTION_NONE`, where the certified forward
                        returns before seam L13
        `row_addr`      N float32, the per-row loss, written on every call.
                        The only output under `REDUCTION_NONE`
        `dlogits_addr`  N * V float32, written only when `want_grad != 0`.
                        May be a one-element buffer otherwise, but NEVER 0:
                        `_f32_ptr` refuses a null address
        `logits_addr`   N * V float32, read, row-major
        `targets_addr`  N int32, read

    **FORWARD AND BACKWARD ARE ONE CALL AND THAT IS NOT PACKAGING.** The
    backward reads `expo` and `denom`, the buffers the forward wrote, and
    recomputes nothing; a second spelling of the softmax is a second thing
    that can be wrong. Two entry points would have to recompute the forward
    or hand its intermediates out to Python, and this surface does neither.

    `REDUCTION_NONE` HAS NO BACKWARD and `want_grad` is refused with it
    (contract section 11).
    """
    if len(params) != 7:
        raise Error(
            "ce_loss: params must contain 7 values, got " + String(len(params))
        )
    var lp = _f32_ptr(Int(py=loss_addr))
    var rp = _f32_ptr(Int(py=row_addr))
    var dp = _f32_ptr(Int(py=dlogits_addr))
    var xp = _f32_ptr(Int(py=logits_addr))
    var tp = _i32_ptr(Int(py=targets_addr))
    var n_rows = Int(py=params[0])
    var vocab = Int(py=params[1])
    var ignore_index = Int(py=params[2])
    var reduction = Int(py=params[3])
    var num_items = Int(py=params[4])
    var want_grad = Int(py=params[5])
    var label_smoothing = Float32(Float64(py=params[6]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = identical_ce_loss_host(
            ctx, lp, rp, dp, xp, tp, n_rows, vocab, ignore_index, reduction,
            num_items, want_grad, label_smoothing,
        )
    return PythonObject(count)


def mlp_bias_activation_binding(
    input_addr: PythonObject, bias_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """C-row-major f32 bias + optional ReLU; params=[rows,cols,relu_flag].

    Borrow input rows*cols, bias cols and output rows*cols floats.
    Returns rows*cols after synchronous copyback; IDENTICAL only.
    """
    if len(params) != 3:
        raise Error("mlp_bias_activation params must be [rows,cols,relu_flag]")
    var rows = Int(py=params[0])
    var cols = Int(py=params[1])
    var relu_flag = Int(py=params[2])
    mlp_validate_shape(rows, cols)
    if relu_flag != 0 and relu_flag != 1:
        raise Error("mlp_bias_activation relu_flag must be 0 or 1")
    var xp = _f32_ptr(Int(py=input_addr))
    var bp = _f32_ptr(Int(py=bias_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = mlp_bias_activation_host(ctx, xp, bp, op, rows, cols, relu_flag)
    return PythonObject(count)


def mlp_relu_backward_binding(
    activation_addr: PythonObject, incoming_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params=[rows,cols]; all three borrowed buffers hold rows*cols f32.

    Derivative is incoming where activation>0, otherwise +0 (including zero).
    Returns rows*cols after synchronous copyback; IDENTICAL only.
    """
    if len(params) != 2:
        raise Error("mlp_relu_backward params must be [rows,cols]")
    var rows = Int(py=params[0])
    var cols = Int(py=params[1])
    mlp_validate_shape(rows, cols)
    var ap = _f32_ptr(Int(py=activation_addr))
    var gp = _f32_ptr(Int(py=incoming_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = mlp_relu_backward_host(ctx, ap, gp, op, rows, cols)
    return PythonObject(count)


def mlp_sum_rows_binding(
    input_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params=[rows,cols]; borrow rows*cols input and cols output f32.

    Returns cols after synchronous ascending-row sum; IDENTICAL only.
    """
    if len(params) != 2:
        raise Error("mlp_sum_rows params must be [rows,cols]")
    var rows = Int(py=params[0])
    var cols = Int(py=params[1])
    mlp_validate_shape(rows, cols)
    var xp = _f32_ptr(Int(py=input_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = mlp_sum_rows_host(ctx, xp, op, rows, cols)
    return PythonObject(count)


def mlp_train_step_binding(
    addresses: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The small MLP's whole step as ONE call (lane/apple-mlp-fused,
    2026-09-30): the same GEMMs, bias/ReLU/row-sum launches, cross-entropy
    and AdamW step `SmallMLPTrainer` made through twelve binding calls, the
    intermediates kept on the device. Returns `rows * 3`, the logits written.

    addresses, in this exact order (mirrored word for word in
    `python/mojolearn/_mlp_impl.py::_fused`):

        0  x        rows x 8 f32, read
        1  y        rows i32 classes 0..2, read (unread under mode 0)
        2  w1       16 x 8 f32   read; WRITTEN IN PLACE under mode 2
        3  b1       16 f32       "
        4  w2       3 x 16 f32   "
        5  b2       3 f32        "
        6  m        195 f32      read and written under mode 2 (any one
                                 float otherwise, never 0)
        7  v        195 f32      "
        8  flags    4 i32        read and written back under mode 2
        9  loss     1 f32        written (modes 1 and 2)
        10 logits   rows x 3 f32 written
        11 dw1      16 x 8 f32   written (modes 1 and 2)
        12 db1      16 f32       "
        13 dw2      3 x 16 f32   "
        14 db2      3 f32        "
        15 dx       rows x 8 f32 written when want_input_grad (else any
                                 one float, never 0)
        16 info     3 f32        the optimizer's info, +0.0 (no clip)

    params, in this exact order:

        0  rows              1..256
        1  mode              0 forward only, 1 forward and backward,
                             2 the whole step
        2  t                 the optimizer's ONE-BASED step (mode 2)
        3  lr                (float)
        4  beta1             (float)
        5  beta2             (float)
        6  eps               (float)
        7  weight_decay      (float)
        8  want_input_grad   0 or 1

    A silent reorder here is a WRONG ANSWER and not a crash. If you change
    either list, change `_fused` in the same edit.
    """
    var a = _addrs(addresses, 17, "mlp_train_step")
    _params(params, 9, "mlp_train_step")
    var rows = Int(py=params[0])
    var mode = Int(py=params[1])
    var t = Int(py=params[2])
    var lr = Float32(Float64(py=params[3]))
    var beta1 = Float32(Float64(py=params[4]))
    var beta2 = Float32(Float64(py=params[5]))
    var eps = Float32(Float64(py=params[6]))
    var weight_decay = Float32(Float64(py=params[7]))
    var want_input_grad = Int(py=params[8])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = mlp_train_step_host(
            ctx,
            _f32_ptr(a[0]), _i32_ptr(a[1]),
            _f32_ptr(a[2]), _f32_ptr(a[3]), _f32_ptr(a[4]), _f32_ptr(a[5]),
            _f32_ptr(a[6]), _f32_ptr(a[7]), _i32_ptr(a[8]),
            _f32_ptr(a[9]), _f32_ptr(a[10]),
            _f32_ptr(a[11]), _f32_ptr(a[12]), _f32_ptr(a[13]), _f32_ptr(a[14]),
            _f32_ptr(a[15]), _f32_ptr(a[16]),
            rows, mode, t, lr, beta1, beta2, eps, weight_decay, want_input_grad,
        )
    return PythonObject(count)


# ---------------------------------------------------------------------------
# lane afn-mlp (2026-10-03): the small MLP's RESIDENT SESSION on Apple FAST.
# `python/mojolearn/_mlp_impl.py::SmallMLPTrainer` opens one per trainer when
# the binding exposes `mlp_resident_open`; weights and moments live on the
# device between steps and come down only for `state_dict`.
# ---------------------------------------------------------------------------


def mlp_resident_open_binding(
    addresses: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """addresses = [w1, b1, w2, b2, m, v] (read); params = [cap_k], the most
    minibatches one `mlp_resident_steps` call may run. Returns the handle."""
    var a = _addrs(addresses, 6, "mlp_resident_open")
    _params(params, 1, "mlp_resident_open")
    var cap_k = Int(py=params[0])
    var h = -1
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        h = mlp_resident_open_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]), _f32_ptr(a[3]),
            _f32_ptr(a[4]), _f32_ptr(a[5]), cap_k,
        )
    return PythonObject(h)


def mlp_resident_upload_binding(
    handle: PythonObject, addresses: PythonObject,
) raises -> PythonObject:
    """addresses = [w1, b1, w2, b2, m, v] (read): replace the session's state."""
    var a = _addrs(addresses, 6, "mlp_resident_upload")
    var h = Int(py=handle)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        mlp_resident_upload_host(
            ctx, h, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]), _f32_ptr(a[3]),
            _f32_ptr(a[4]), _f32_ptr(a[5]),
        )
    return PythonObject(h)


def mlp_resident_download_binding(
    handle: PythonObject, addresses: PythonObject,
) raises -> PythonObject:
    """addresses = [w1, b1, w2, b2, m, v] (written): the session's state."""
    var a = _addrs(addresses, 6, "mlp_resident_download")
    var h = Int(py=handle)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        mlp_resident_download_host(
            ctx, h, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]), _f32_ptr(a[3]),
            _f32_ptr(a[4]), _f32_ptr(a[5]),
        )
    return PythonObject(h)


def mlp_resident_close_binding(handle: PythonObject) raises -> PythonObject:
    var h = Int(py=handle)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        mlp_resident_close_host(ctx, h)
    return PythonObject(h)


def mlp_resident_steps_binding(
    handle: PythonObject, addresses: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`k` steps on the session (mirrored word for word in
    `python/mojolearn/_mlp_impl.py::SmallMLPTrainer._resident_steps`):

        addresses: x (k*rows x 8), y (k*rows i32), losses (k f32, written),
                   logits (rows x 3, the last step's), dw1, db1, dw2, db2
                   (the last step's), dx (rows x 8 when want_input_grad,
                   else any one float)
        params:    rows, k, mode, t, lr, beta1, beta2, eps, weight_decay,
                   want_input_grad

    `k > 1` is compiled under MOJOLEARN_AFN_MLP_MULTISTEP only. Returns
    `rows * 3`."""
    var a = _addrs(addresses, 9, "mlp_resident_steps")
    _params(params, 10, "mlp_resident_steps")
    var h = Int(py=handle)
    var rows = Int(py=params[0])
    var k = Int(py=params[1])
    var mode = Int(py=params[2])
    var t = Int(py=params[3])
    var lr = Float32(Float64(py=params[4]))
    var beta1 = Float32(Float64(py=params[5]))
    var beta2 = Float32(Float64(py=params[6]))
    var eps = Float32(Float64(py=params[7]))
    var weight_decay = Float32(Float64(py=params[8]))
    var want_input_grad = Int(py=params[9])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = mlp_resident_step_host(
            ctx, h, _f32_ptr(a[0]), _i32_ptr(a[1]), _f32_ptr(a[2]), _f32_ptr(a[3]),
            _f32_ptr(a[4]), _f32_ptr(a[5]), _f32_ptr(a[6]), _f32_ptr(a[7]), _f32_ptr(a[8]),
            rows, k, mode, t, lr, beta1, beta2, eps, weight_decay, want_input_grad,
        )
    return PythonObject(count)


# ===========================================================================
# THE SAMBA STACK'S OPS: embedding, RMSNorm, LM head, accumulate, RNG.
# Every one takes (addresses, params) as two Python lists, the byte-LM
# binding's shape, and the order of each list is written out once here and
# once in python/mojolearn/_training_impl.py in the same words.
# ===========================================================================


def _addrs(addresses: PythonObject, want: Int, name: String) raises -> List[Int]:
    if len(addresses) != want:
        raise Error(
            name + ": addresses must contain " + String(want) + " entries, got "
            + String(len(addresses))
        )
    var out = List[Int]()
    for i in range(want):  # small-loop(want: buffer addresses of one op call): reads pointer list, not data
        var a = Int(py=addresses[i])
        if a == 0:
            raise Error(name + ": null buffer address at slot " + String(i))
        out.append(a)
    return out^


def _params(params: PythonObject, want: Int, name: String) raises:
    if len(params) != want:
        raise Error(
            name + ": params must contain " + String(want) + " values, got "
            + String(len(params))
        )


def embedding_forward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [y (n_positions*width f32, written), w (vocab*width f32),
    ids (n_positions i32)]; params = [n_positions, vocab, width]."""
    var a = _addrs(addresses, 3, "embedding_forward")
    _params(params, 3, "embedding_forward")
    var n_positions = Int(py=params[0])
    var vocab = Int(py=params[1])
    var width = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_embedding_forward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _i32_ptr(a[2]),
            n_positions, vocab, width,
        )
    return PythonObject(count)


def embedding_backward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [dw (vocab*width f32, written), dy (n_positions*width f32),
    ids (n_positions i32)]; params = [n_positions, vocab, width]."""
    var a = _addrs(addresses, 3, "embedding_backward")
    _params(params, 3, "embedding_backward")
    var n_positions = Int(py=params[0])
    var vocab = Int(py=params[1])
    var width = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_embedding_backward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _i32_ptr(a[2]),
            n_positions, vocab, width,
        )
    return PythonObject(count)


def rms_norm_forward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [y (m*dm f32, written), x (m*dm f32), w (dm f32)];
    params = [m, dm, eps (float)]."""
    var a = _addrs(addresses, 3, "rms_norm_forward")
    _params(params, 3, "rms_norm_forward")
    var m = Int(py=params[0])
    var dm = Int(py=params[1])
    var eps = Float32(Float64(py=params[2]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_rms_norm_forward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]), m, dm, eps,
        )
    return PythonObject(count)


def rms_norm_backward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [dx (m*dm f32, written), dw (dm f32, written), dy (m*dm
    f32), x (m*dm f32), w (dm f32)]; params = [m, dm, eps (float)]."""
    var a = _addrs(addresses, 5, "rms_norm_backward")
    _params(params, 3, "rms_norm_backward")
    var m = Int(py=params[0])
    var dm = Int(py=params[1])
    var eps = Float32(Float64(py=params[2]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_rms_norm_backward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]),
            _f32_ptr(a[3]), _f32_ptr(a[4]), m, dm, eps,
        )
    return PythonObject(count)


def mlp_sessions_binding(addresses: PythonObject, input_addresses: PythonObject,
                         row_counts: PythonObject, dims: PythonObject) raises -> PythonObject:
    """Native whole-model A/B: shared weights, independent session row batches."""
    var a = _addrs(addresses, 5, "mlp_sessions")
    _params(dims, 3, "mlp_sessions")
    if len(input_addresses) != len(row_counts):
        raise Error("neural MLP sessions: metadata length mismatch")
    var inputs = List[Int]()
    var rows = List[Int]()
    for i in range(len(row_counts)):
        inputs.append(Int(py=input_addresses[i]))
        rows.append(Int(py=row_counts[i]))
    var in_width = Int(py=dims[0])
    var hidden = Int(py=dims[1])
    var out_width = Int(py=dims[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = nn_mlp_sessions_device(ctx, inputs, rows, _f32_ptr(a[1]), _f32_ptr(a[2]), _f32_ptr(a[3]), _f32_ptr(a[4]), _f32_ptr(a[0]), in_width, hidden, out_width)
    return PythonObject(count)


def linear_forward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`C[m, n] = A[m, k] . W[n, k]^T`. addresses = [c (written), a, w];
    params = [m, n, k]."""
    var a = _addrs(addresses, 3, "linear_forward")
    _params(params, 3, "linear_forward")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_linear_forward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]), m, n, k,
        )
    return PythonObject(count)


def linear_backward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [da (m*k, written), dw (n*k, written), dc (m*n), a (m*k),
    w (n*k)]; params = [m, n, k]."""
    var a = _addrs(addresses, 5, "linear_backward")
    _params(params, 3, "linear_backward")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_linear_backward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]),
            _f32_ptr(a[3]), _f32_ptr(a[4]), m, n, k,
        )
    return PythonObject(count)


def samba_head_loss_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`linear_forward`, `ce_loss` with a gradient and `linear_backward` in
    ONE call (lane/py-lm), the logits and their gradient kept on the device.
    addresses = [loss (1 f32, written), row_loss (m f32, written), da (m*k,
    written), dw (n*k, written), a (m*k), w (n*k), targets (m i32)];
    params = [m, n, k, ignore_index, reduction (1 sum, 2 mean), num_items,
    label_smoothing (float)]. Returns `count`, as `ce_loss` does."""
    var a = _addrs(addresses, 7, "samba_head_loss")
    _params(params, 7, "samba_head_loss")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var ignore_index = Int(py=params[3])
    var reduction = Int(py=params[4])
    var num_items = Int(py=params[5])
    var label_smoothing = Float32(Float64(py=params[6]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = samba_head_loss_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]), _f32_ptr(a[3]),
            _f32_ptr(a[4]), _f32_ptr(a[5]), _i32_ptr(a[6]),
            m, n, k, ignore_index, reduction, num_items, label_smoothing,
        )
    return PythonObject(count)


def accumulate_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The clause 9.2 balanced tree. addresses = [out (n f32, written),
    parts (a*n f32, ascending microbatch index)]; params = [n, a, t_tokens]
    where `t_tokens >= 1` asks for the alignment predicate and `-1` makes
    no alignment claim."""
    var a = _addrs(addresses, 2, "accumulate")
    _params(params, 3, "accumulate")
    var n = Int(py=params[0])
    var steps = Int(py=params[1])
    var t_tokens = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = parallel_accumulate_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), n, steps, t_tokens,
        )
    return PythonObject(count)


def accumulation_is_aligned_binding(params: PythonObject) raises -> PythonObject:
    """`microbatch_split_is_identical(t_tokens, a)`, contract clause 9.2,
    as 1 or 0. params = [t_tokens, a]. Host only, no device."""
    _params(params, 2, "accumulation_is_aligned")
    var t_tokens = Int(py=params[0])
    var a = Int(py=params[1])
    if microbatch_split_is_identical(t_tokens, a):
        return PythonObject(1)
    return PythonObject(0)


def neural_rng_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [out (n f32, written), inp (n f32 for the dropout kinds,
    else a one-float placeholder)]; params = [n, offset, seed_lo, seed_hi,
    stream_id, kind (0 uniform, 1 normal, 2 dropout forward, 3 dropout
    backward), a (float), b (float)] where (a, b) is (lo, span), (mean, sd)
    or (p, scale)."""
    var a = _addrs(addresses, 2, "neural_rng")
    _params(params, 8, "neural_rng")
    var n = Int(py=params[0])
    var offset = Int(py=params[1])
    var seed_lo = Int(py=params[2])
    var seed_hi = Int(py=params[3])
    var stream_id = Int(py=params[4])
    var kind = Int(py=params[5])
    var pa = Float32(Float64(py=params[6]))
    var pb = Float32(Float64(py=params[7]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = neural_rng_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), n, offset, seed_lo, seed_hi,
            stream_id, kind, pa, pb,
        )
    return PythonObject(count)


from training.neural_identical_experiments import IDN_CHUNKED_LM_HEAD_V2, IDN_LOSS_TOKEN_TREE_V2, IDN_ATTENTION_V2
from gemm.contract import CONTRACT_K_LEAF_MIN


def training_experiment_profile_binding() raises -> PythonObject:
    """Checkpoint identity for optional NI34/35/08 numerical contracts."""
    var profile = String("baseline")
    if IDN_ATTENTION_V2:
        profile += "+attention-online-tile32-v2"
    if IDN_CHUNKED_LM_HEAD_V2:
        profile += "+head-serial-logit-chunked-v2"
    if IDN_LOSS_TOKEN_TREE_V2:
        profile += "+ce-token-tree256-v2"
    if CONTRACT_K_LEAF_MIN != 128:
        profile += "+gemm-leaf" + String(CONTRACT_K_LEAF_MIN)
    return PythonObject(profile)


def training_chunked_lm_head_enabled_binding() raises -> PythonObject:
    """NI34 profile selector shared by native-host and GPU Samba wrappers."""
    return PythonObject(IDN_CHUNKED_LM_HEAD_V2)


def chunked_lm_head_v2_loss_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Opt-in v2 forward/loss. addresses=[loss,max,denom,h,w,target],
    params=[rows,vocab,width]. V1 remains the default everywhere else."""
    var a = _addrs(addresses, 6, "chunked_lm_head_v2_loss")
    _params(params, 3, "chunked_lm_head_v2_loss")
    var rows = Int(py=params[0])
    var vocab = Int(py=params[1])
    var width = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = chunked_lm_head_v2_loss_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]),
            _f32_ptr(a[3]), _f32_ptr(a[4]), _i32_ptr(a[5]),
            rows, vocab, width,
        )
    return PythonObject(count)


def chunked_lm_head_v2_train_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Explicit opt-in v2 train stage. addresses=[loss,max,denom,dh,dw,h,w,target]."""
    var a = _addrs(addresses, 8, "chunked_lm_head_v2_train")
    _params(params, 3, "chunked_lm_head_v2_train")
    var rows = Int(py=params[0])
    var vocab = Int(py=params[1])
    var width = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        count = chunked_lm_head_v2_train_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]),
            _f32_ptr(a[3]), _f32_ptr(a[4]), _f32_ptr(a[5]),
            _f32_ptr(a[6]), _i32_ptr(a[7]), rows, vocab, width,
        )
    return PythonObject(count)


def neural_arithmetic_profile_binding() raises -> PythonObject:
    return PythonObject(neural_training_profile())


@export
def PyInit__mojolearn_training() abi("C") -> PythonObject:
    # FAST AND IDENTICAL (lane neural, 2026-09-27). This lane was
    # IDENTICAL-only from 2026-09-10 because its fused kernels were once gated
    # on IDENTICAL and the lower tiers fell back to slower unfused arms. Those
    # fallbacks are gone: FAST runs the same kernels and the same launches with
    # the pins in checks/numerics.mojo compiled to the free schedule. FAST
    # promises quality, never bits. The
    # DETERMINISTIC tier stays tree-only: refuse to exist under it.
    comptime if GLOBAL_NUMERIC_MODE > NUMERIC_IDENTICAL:  # NUMERIC_DETERMINISTIC (2)
        abort(
            String(
                "_mojolearn_training: refusing to initialize -- this lane builds"
                " FAST and IDENTICAL only. Rebuild with"
                " MOJOLEARN_NUMERIC_MODE=identical (or fast) bash bindings/build_training.sh"
            )
        )
    try:
        var m = PythonModuleBuilder("_mojolearn_training")
        m.def_function[neural_arithmetic_profile_binding]("neural_arithmetic_profile")
        m.def_function[training_vendor_binding]("training_vendor")
        m.def_function[training_numeric_mode_binding]("training_numeric_mode")
        m.def_function[clip_pool_fault_available_binding]("clip_pool_fault_available")
        m.def_function[clip_parallel_available_binding]("clip_parallel_available")
        m.def_function[accumulate_pool_fault_available_binding]("accumulate_pool_fault_available")
        m.def_function[accumulate_parallel_available_binding]("accumulate_parallel_available")
        m.def_function[optimizer_parallel_available_binding]("optimizer_parallel_available")
        m.def_function[optimizer_step_binding]("optimizer_step")
        m.def_function[optimizer_resident_open_binding]("optimizer_resident_open")
        m.def_function[optimizer_resident_close_binding]("optimizer_resident_close")
        m.def_function[optimizer_resident_download_binding]("optimizer_resident_download")
        m.def_function[optimizer_resident_upload_binding]("optimizer_resident_upload")
        m.def_function[optimizer_resident_step_binding]("optimizer_resident_step")
        comptime if IDN_OPT_PARAMS_RESIDENT:
            m.def_function[optimizer_resident_put_binding]("optimizer_resident_put")
            m.def_function[optimizer_resident_get_binding]("optimizer_resident_get")
            m.def_function[optimizer_resident_step_io_binding]("optimizer_resident_step_io")
            comptime if IDN_MAXIMIZE_DEV:
                m.def_function[maximize_dev_available_binding]("optimizer_maximize_dev")
        comptime if IDN_TRAIN_DEV_TENSORS:
            m.def_function[train_dev_alloc_binding]("train_dev_alloc")
            m.def_function[train_dev_free_binding]("train_dev_free")
            m.def_function[train_dev_put_binding]("train_dev_put")
            m.def_function[train_dev_get_binding]("train_dev_get")
            m.def_function[linear_forward_dev_binding]("linear_forward_dev")
            m.def_function[linear_backward_dev_binding]("linear_backward_dev")
            m.def_function[ce_loss_dev_binding]("ce_loss_dev")
            comptime if IDN_OPT_PARAMS_RESIDENT:
                m.def_function[optimizer_resident_copy_dev_binding]("optimizer_resident_copy_dev")
        m.def_function[clip_grad_norm_binding]("clip_grad_norm")
        m.def_function[clip_grad_norm_multi_binding]("clip_grad_norm_multi")
        m.def_function[ce_loss_binding]("ce_loss")
        m.def_function[mlp_bias_activation_binding]("mlp_bias_activation")
        m.def_function[mlp_relu_backward_binding]("mlp_relu_backward")
        m.def_function[mlp_sum_rows_binding]("mlp_sum_rows")
        m.def_function[mlp_train_step_binding]("mlp_train_step")
        comptime if MLP_RESIDENT:
            # lane afn-mlp: Apple FAST only, behind its define
            m.def_function[mlp_resident_open_binding]("mlp_resident_open")
            m.def_function[mlp_resident_upload_binding]("mlp_resident_upload")
            m.def_function[mlp_resident_download_binding]("mlp_resident_download")
            m.def_function[mlp_resident_close_binding]("mlp_resident_close")
            m.def_function[mlp_resident_steps_binding]("mlp_resident_step")
        comptime if MLP_MULTISTEP:
            m.def_function[mlp_resident_steps_binding]("mlp_resident_steps")
        m.def_function[embedding_forward_binding]("embedding_forward")
        m.def_function[embedding_backward_binding]("embedding_backward")
        m.def_function[rms_norm_forward_binding]("rms_norm_forward")
        m.def_function[rms_norm_backward_binding]("rms_norm_backward")
        m.def_function[linear_forward_binding]("linear_forward")
        m.def_function[neural_gemm_binding]("neural_gemm")
        m.def_function[mlp_sessions_binding]("mlp_sessions")
        m.def_function[residual_dropout_binding]("residual_dropout")
        m.def_function[residual_dropout_backward_binding]("residual_dropout_backward")
        m.def_function[linear_backward_binding]("linear_backward")
        m.def_function[samba_head_loss_binding]("samba_head_loss")
        comptime if AFN_SAMBA_FUSE:
            m.def_function[samba_afn_norm_head_forward_binding]("samba_afn_norm_head_forward")
            m.def_function[samba_afn_tail_train_binding]("samba_afn_tail_train")
            m.def_function[samba_afn_embedding_backward_tied_binding]("samba_afn_embedding_backward_tied")
        m.def_function[accumulate_binding]("accumulate")
        m.def_function[accumulation_is_aligned_binding]("accumulation_is_aligned")
        m.def_function[neural_rng_binding]("neural_rng")
        m.def_function[chunked_lm_head_v2_loss_binding]("chunked_lm_head_v2_loss")
        m.def_function[chunked_lm_head_v2_train_binding]("chunked_lm_head_v2_train")
        m.def_function[training_chunked_lm_head_enabled_binding]("training_chunked_lm_head_enabled")
        m.def_function[training_experiment_profile_binding]("training_experiment_profile")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_training: ", e))
