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

from core.experiment_guards_neural import NEURAL_GUARDS_OK
from training.neural_arithmetic_profile import neural_training_profile
from bindings.residual_dropout_boundary import (
    residual_dropout_binding, residual_dropout_backward_binding,
    residual_dropout_dev_binding, residual_dropout_backward_dev_binding,
)
from bindings.neural_gemm_boundary import neural_gemm_binding


# DEVIATION 2486: shared byte-preserving host copies.
from core.hostptr import f32_ptr, i32_ptr
from std.os import abort, getenv
from std.python import Python, PythonObject
from bindings.sha256_door import sha256_hexdigest
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from training.native_primitive_dispatch import NATIVE_PRIMITIVE_API, native_primitive_operation
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojolearnNeuralTrainingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTrainingContextFastV3"
from checks.vendor import COMPILED_VENDOR
from max.gpu.host import DeviceContext, HostBuffer

from training.accumulate_multi_gpu import parallel_accumulate_host, accumulate_pool_fault_available
from training.optimizer_public import OPT_POOL, opt_pool_handle, per_call_step
from bindings.optimizer_public_door import entry_zero
from bindings.optimizer_public_pyapi import (
    optimizer_door_accumulate_c, optimizer_door_clip_c, optimizer_door_io_c, optimizer_door_load_state_c,
    optimizer_door_state_c, optimizer_door_state_set_c, optimizer_door_step_c, optimizer_resident_close_binding,
    optimizer_resident_download_binding, optimizer_resident_open_binding,
    optimizer_resident_step_binding, optimizer_resident_step_io_binding,
    optimizer_resident_upload_binding, maximize_dev_available_binding,
)
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
    identical_optimizer_step_resident_io, IDN_OPT_PARAMS_RESIDENT, OPT_IO_ALL, OPT_IO_UP_G, IDN_MAXIMIZE_DEV,
    maximize_negate_device,
)
from std.ffi import _Global
from max.gpu.host import DeviceBuffer
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
from training.samba_afn import AFN_SAMBA_FUSE
from providers.python.training_samba_afn_pyapi import (
    samba_afn_embedding_backward_tied_binding,
    samba_afn_tail_train_binding,
)
# lane S1 samba-resident (2026-10-07): the device-resident Samba forward and
# train step, registered only when IDN_SAMBA_RESIDENT_STEP holds (IDENTICAL
# default since 2026-10-08, lane grid-act-4; off under
# -D MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2); `samba_resident_enabled` is registered always so the Python
# shell can ask (training/samba_resident.mojo).
from training.neural_identical_experiments import IDN_SAMBA_RESIDENT_STEP
from training.samba_resident import (
    SambaResidentSession,
    samba_resident_train_step,
    samba_resident_accumulated_step,
)
from core.philox_neural import neural_rng_host
# lane x6-samba-on-executor (2026-11-09): the Samba stack program, the one
# device path of SambaStack.forward / step and SambaDecodeSession
# (training/samba_program.mojo).
from training.samba_program import SambaStackProgram, SAMBA_PROGRAM_ON


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def training_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL. The same shape as `gbdt_numeric_mode`, and for
    the same reason: the wrapper reads it once and refuses to run if the
    binary it loaded disagrees with the mode the package asked for. A
    wrong-arm measurement that is correctly labelled by accident is the
    failure this prevents.

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


# TOMBSTONE: clip_grad_norm, clip_pool_fault_available, optimizer_resident_copy_dev exports (dead native entries: no live caller; clip_grad_norm_ runs optimizer_door_clip) deleted 2026-10-26 by c76141c5c; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_training.patch


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
        n_total = per_call_step(
            ctx, pp, gp, mp, vp, op, ip, fp, n_tensors, kind, t, nesterov,
            lr, beta1, beta2, eps, weight_decay, momentum, dampening, max_norm, maximize,
        )
    return PythonObject(n_total)


# ===========================================================================
# THE RESIDENT OPTIMIZER STATE (lane/neural-pass4, 2026-09-30; lane
# fam2-neural, 2026-10-04): the pool and its `optimizer_resident_*` entries
# moved to training/optimizer_public.mojo and bindings/optimizer_public_pyapi.mojo
# (lane neural-optimizers, 2026-10-21), the same words and the same steps.
# ===========================================================================


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


def _ce_row_ptr(row_addr: PythonObject, reduction: Int) raises -> Optional[MutPointer[Float32, MutUntrackedOrigin]]:
    """The per-row loss buffer. Address 0 is accepted when `reduction` is
    SUM or MEAN (lane ne3-c-blocks-embedding, row N118): the caller has no
    use for the rows, so the drivers skip their download (None: a `Pointer`
    is never null); under NONE the rows are the result and a null address is
    refused as before."""
    var address = Int(py=row_addr)
    if address == 0 and reduction != 0:
        return None
    return _f32_ptr(address)


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
    var reduction = Int(py=params[3])
    var rp = _ce_row_ptr(row_addr, reduction)
    var tp = _i32_ptr(Int(py=targets_addr))
    var n_rows = Int(py=params[0])
    var vocab = Int(py=params[1])
    var ignore_index = Int(py=params[2])
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
        `row_addr`      N float32, the per-row loss, written on every call
                        that names one. The only output under
                        `REDUCTION_NONE`; under SUM or MEAN it may be 0
                        (no rows are downloaded)
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
    var reduction = Int(py=params[3])
    var rp = _ce_row_ptr(row_addr, reduction)
    var dp = _f32_ptr(Int(py=dlogits_addr))
    var xp = _f32_ptr(Int(py=logits_addr))
    var tp = _i32_ptr(Int(py=targets_addr))
    var n_rows = Int(py=params[0])
    var vocab = Int(py=params[1])
    var ignore_index = Int(py=params[2])
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


# TOMBSTONE: the small-MLP primitives mlp_bias_activation, mlp_relu_backward, mlp_sum_rows (training/mlp_ops.mojo _mlp_kernel and its *_host functions; on the CPU column training/host/mlp_oracle.mojo host_mlp_relu_backward / host_mlp_sum_rows and core/mojolearn_training_host_impl.mojo *_host_into) and the mlp_sessions / mlp_sessions_rows entries (MOJOLEARN_NN64_SESSION_PACK) deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3. Their callers (ParallelNeuralTrainer's shard gradient, SmallMLPTrainer's inference sessions) run the trainer's captured program (autograd/small_mlp_program.mojo).
# Restore: git apply experiments/removed/M2_SMALL_MLP_PRIMITIVES.patch


# TOMBSTONE: mlp_train_step, the mlp_resident_* session and small_mlp_api_v1 entries (MOJOLEARN_AFN_MLP_RESIDENT / _MULTISTEP / _FUSED_STEP / _ALL, merged-unmeasured; superseded by SmallMLPTrainer's captured program, autograd/small_mlp_program.mojo, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch


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


def neural_primitives_api_binding() raises -> PythonObject:
    ref cpy = Python().cpython()
    var api = NATIVE_PRIMITIVE_API.get_or_create_ptr()
    var capsule = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(
        api[].entries.unsafe_ptr(), api[].name.as_c_string_slice().unsafe_ptr(), PyObjectPtr())
    if not capsule:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=capsule)


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
        count = native_primitive_operation(0, a, [n_positions, vocab, width], [])
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
        count = native_primitive_operation(1, a, [n_positions, vocab, width], [])
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
    var eps = Float64(py=params[2])
    var count = 0
    with GILReleased(Python()):
        count = native_primitive_operation(2, a, [m, dm], [Float64(eps)])
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
    var eps = Float64(py=params[2])
    var count = 0
    with GILReleased(Python()):
        count = native_primitive_operation(3, a, [m, dm], [Float64(eps)])
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
        count = native_primitive_operation(4, a, [m, n, k], [])
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
        count = native_primitive_operation(5, a, [m, n, k], [])
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


# ===========================================================================
# THE DEVICE-RESIDENT SAMBA FORWARD AND TRAIN STEP (lane S1 samba-resident,
# 2026-10-07; IDENTICAL default since 2026-10-08 (lane grid-act-4, grid
# ge123e6f9); -D MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF restores the per-op route).
# One `_SambaResidentSession` per Python `SambaStack` (samba_resident_open),
# then one call per optimizer step (the forward runs SambaStackProgram since lane x6)
# (samba_resident_step): the registry, the gradient, the activations and the
# block stages stay on the device; the Python side keeps no per-layer loop.
# The optimizer's moments are the Python optimizer's resident pair (an
# `optimizer_resident_open` handle). Same kernels, operands and order as the
# per-op route, which stays as arm B (training/samba_resident.mojo).
# ===========================================================================


def samba_resident_native_handle_binding(session: PythonObject) raises -> PythonObject:
    # Checked ownership admission for sessions created by the public binding.
    return PythonObject(Int(session.downcast_value_ptr[SambaResidentSession]()))


def samba_resident_enabled_binding() raises -> PythonObject:
    """Whether this build carries the resident Samba entries."""
    return PythonObject(IDN_SAMBA_RESIDENT_STEP)


def samba_resident_open_binding(
    layers: PythonObject, params: PythonObject, param_addr: PythonObject
) raises -> PythonObject:
    """Open a stack's residency. `layers` = one code per layer in stack order
    (0 mamba3, 1 attention); `params` = [vocab, d_model, n_heads, n_kv_heads,
    head_dim, intermediate, tie_embeddings (0/1), norm_eps (float), n_total]
    (the attention shape slots are 0 when the stack has no attention layer);
    `param_addr` = the flat float32 registry (n_total floats, read once here).
    Returns the session object; pass it to the two entries below."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("samba_resident_open: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
    else:
        _params(params, 9, "samba_resident_open")
        var kinds = List[Int]()
        for i in range(len(layers)):  # small-loop(layers: one kind code per layer): reads codes, not data
            kinds.append(Int(py=layers[i]))
        var vocab = Int(py=params[0])
        var dm = Int(py=params[1])
        var nh = Int(py=params[2])
        var nkv = Int(py=params[3])
        var hd = Int(py=params[4])
        var it = Int(py=params[5])
        var tie = Int(py=params[6]) != 0
        var eps = Float32(Float64(py=params[7]))
        var n_total = Int(py=params[8])
        var pp = _f32_ptr(Int(py=param_addr))
        var ctx = neural_ctx[_NEURAL_CTX]()
        var session = SambaResidentSession(
            ctx, kinds, vocab, dm, nh, nkv, hd, it, tie, eps, n_total, pp
        )
        return PythonObject(alloc=session^)


# TOMBSTONE: SAMBA_THREE_TIER_FORWARD samba_resident_forward_binding (the resident forward entry; SambaStack.forward runs SambaStackProgram) deleted 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch


def samba_resident_step_binding(
    session: PythonObject, addresses: PythonObject, params: PythonObject,
    handle: PythonObject,
) raises -> PythonObject:
    """`SambaStack.train_step` at accumulation_steps 1, resident.
    addresses = [params (n_total f32, read and WRITTEN IN PLACE), ids (b*l
    i32), targets (b*l i32), offsets (J+1 i32), buf_initialized (J i32,
    written), info (3 f32, written: clip_ran, total_norm, coef), loss (1 f32,
    written)]; params = [b, l] + the `optimizer_step` params list (its 12 or
    13 slots, word for word); `handle` = the optimizer's resident moments
    (`optimizer_resident_open`, n_total floats). Returns `count`, the number
    of targets that are not the ignore index."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("samba_resident_step: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
    else:
        var owner = session.downcast_value_ptr[SambaResidentSession]()
        var a = _addrs(addresses, 7, "samba_resident_step")
        if len(params) != 14 and len(params) != 15:
            raise Error(
                "samba_resident_step: params must contain 14 or 15 values, got "
                + String(len(params))
            )
        var b = Int(py=params[0])
        var l = Int(py=params[1])
        var n_tensors = Int(py=params[2])
        var kind = Int(py=params[3])
        var t = Int(py=params[4])
        var nesterov = Int(py=params[5])
        var lr = Float32(Float64(py=params[6]))
        var beta1 = Float32(Float64(py=params[7]))
        var beta2 = Float32(Float64(py=params[8]))
        var eps = Float32(Float64(py=params[9]))
        var weight_decay = Float32(Float64(py=params[10]))
        var momentum = Float32(Float64(py=params[11]))
        var dampening = Float32(Float64(py=params[12]))
        var max_norm = Float32(Float64(py=params[13]))
        if len(params) == 15 and Int(py=params[14]) != 0:
            raise Error("samba_resident_step: maximize is not carried by the resident step (SambaStack never sets it)")
        var h = opt_pool_handle(Int(py=handle), owner[].n_total)
        if owner[].busy:
            raise Error("samba_resident_step: session busy")
        owner[].busy = True
        var count = 0
        try:
            with GILReleased(Python()):
                var pool = OPT_POOL.get_or_create_ptr()
                count = samba_resident_train_step(
                    owner[], _f32_ptr(a[0]), _i32_ptr(a[1]), _i32_ptr(a[2]), _f32_ptr(a[6]),
                    pool[].m[h], pool[].v[h], pool[].sp[h], pool[].sg[h],
                    _i32_ptr(a[3]), _i32_ptr(a[4]), _f32_ptr(a[5]),
                    b, l, n_tensors, kind, t, nesterov, lr, beta1, beta2, eps,
                    weight_decay, momentum, dampening, max_norm,
                )
        except error:
            owner[].busy = False
            raise error
        owner[].busy = False
        return PythonObject(count)


def samba_resident_accumulated_step_binding(
    session: PythonObject, addresses: PythonObject, params: PythonObject,
    handle: PythonObject,
) raises -> PythonObject:
    """`SambaStack.train_step` at accumulation_steps a > 1 (no dropout), one
    resident program (training/samba_resident.mojo
    samba_resident_accumulated_step). addresses as `samba_resident_step`;
    params = [b, l, a, total target count] + the `optimizer_step` params
    list (its 12 or 13 slots). Returns the target count."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("samba_resident_accumulated_step: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
    else:
        var owner = session.downcast_value_ptr[SambaResidentSession]()
        var a = _addrs(addresses, 7, "samba_resident_accumulated_step")
        if len(params) != 16 and len(params) != 17:
            raise Error("samba_resident_accumulated_step: params must contain 16 or 17 values, got " + String(len(params)))
        var b = Int(py=params[0])
        var l = Int(py=params[1])
        var steps = Int(py=params[2])
        var total = Int(py=params[3])
        var n_tensors = Int(py=params[4])
        var kind = Int(py=params[5])
        var t = Int(py=params[6])
        var nesterov = Int(py=params[7])
        var lr = Float32(Float64(py=params[8]))
        var beta1 = Float32(Float64(py=params[9]))
        var beta2 = Float32(Float64(py=params[10]))
        var eps = Float32(Float64(py=params[11]))
        var weight_decay = Float32(Float64(py=params[12]))
        var momentum = Float32(Float64(py=params[13]))
        var dampening = Float32(Float64(py=params[14]))
        var max_norm = Float32(Float64(py=params[15]))
        if len(params) == 17 and Int(py=params[16]) != 0:
            raise Error("samba_resident_step: maximize is not carried by the resident step (SambaStack never sets it)")
        var h = opt_pool_handle(Int(py=handle), owner[].n_total)
        if owner[].busy:
            raise Error("samba_resident_step: session busy")
        owner[].busy = True
        var count = 0
        try:
            with GILReleased(Python()):
                var pool = OPT_POOL.get_or_create_ptr()
                count = samba_resident_accumulated_step(
                    owner[], _f32_ptr(a[0]), _i32_ptr(a[1]), _i32_ptr(a[2]), _f32_ptr(a[6]),
                    pool[].m[h], pool[].v[h], pool[].sp[h], pool[].sg[h],
                    _i32_ptr(a[3]), _i32_ptr(a[4]), _f32_ptr(a[5]),
                    b, l, steps, total, n_tensors, kind, t, nesterov, lr, beta1, beta2, eps,
                    weight_decay, momentum, dampening, max_norm,
                )
        except error:
            owner[].busy = False
            raise error
        owner[].busy = False
        return PythonObject(count)


# ===========================================================================
# THE SAMBA STACK PROGRAM (lane x6-samba-on-executor, 2026-11-09; plan row X6,
# owed row N094; training/samba_program.mojo). ONE runtime path for every
# Samba inference call on a device build: `SambaStack.forward` (stateless or
# with a state), `SambaStack.step` and `SambaDecodeSession`. Each layer is one
# executor step op; the hidden state never leaves the device. The per-call
# doors open a program per (batch, capacity) and reload the caller's registry
# and state on every call (the registry is the public truth); a decode session
# owns its program, loaded once at open.
# TOMBSTONE: SAMBA_THREE_TIER_FORWARD (DELETED: the resident forward
# samba_resident_forward, the provider forward and the decode residency
# SambaDecodeResident, superseded by SambaStackProgram, lane x6) deleted
# 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch
# ===========================================================================


def _int_words(values: PythonObject, name: String) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(values)):  # small-loop(values: state addresses or cursor words of one program call): argument words, not data
        out.append(Int(py=values[i]))
    return out^


def _nonnull_words(values: PythonObject, name: String) raises -> List[Int]:
    var out = _int_words(values, name)
    for i in range(len(out)):  # small-loop(out: state addresses of one program call): argument words, not data
        if out[i] == 0:
            raise Error(name + ": null state address at slot " + String(i))
    return out^


def _py_words(words: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(words)):  # small-loop(words: two cursor words per layer): result words, not data
        _ = out.append(PythonObject(words[i]))
    return out


def samba_program_enabled_binding() raises -> PythonObject:
    """Whether this build carries the Samba stack program (device builds)."""
    return PythonObject(SAMBA_PROGRAM_ON)


def _samba_program_new(layers: PythonObject, params: PythonObject, param_addr: PythonObject,
                       name: String) raises -> SambaStackProgram:
    _params(params, 11, name)
    var kinds = List[Int]()
    for i in range(len(layers)):  # small-loop(layers: one kind code per layer): reads codes, not data
        kinds.append(Int(py=layers[i]))
    var pa = Int(py=param_addr)
    if pa == 0:
        raise Error(name + ": null registry address")
    var ctx = neural_ctx[_NEURAL_CTX]()
    return SambaStackProgram(
        ctx, kinds, Int(py=params[0]), Int(py=params[1]), Int(py=params[2]), Int(py=params[3]),
        Int(py=params[4]), Int(py=params[5]), Int(py=params[6]) != 0, Float32(Float64(py=params[7])),
        Int(py=params[8]), Int(py=params[9]), Int(py=params[10]), _f32_ptr(pa),
    )


def samba_program_open_binding(
    layers: PythonObject, params: PythonObject, param_addr: PythonObject
) raises -> PythonObject:
    """Open a stack program. `layers` = one code per layer (0 mamba3, 1
    attention); `params` = [vocab, d_model, n_heads, n_kv_heads, head_dim,
    intermediate, tie_embeddings (0/1), norm_eps (float), n_total,
    batch_size, max_tokens]; `param_addr` = the flat float32 registry.
    Returns the program object, zero state."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_program_open: a host build has no device program")
    else:
        var program = _samba_program_new(layers, params, param_addr, "samba_program_open")
        return PythonObject(alloc=program^)


def samba_program_shape_binding(program: PythonObject) raises -> PythonObject:
    """(batch_size, max_tokens, usable) of an open program."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_program_shape: a host build has no device program")
    else:
        var owner = program.downcast_value_ptr[SambaStackProgram]()
        return Python.tuple(owner[].b, owner[].smax, owner[].usable)


def samba_program_forward_binding(
    program: PythonObject, addresses: PythonObject, params: PythonObject,
    pieces: PythonObject, cursors: PythonObject,
) raises -> PythonObject:
    """One per-call Samba run. addresses = [registry (n_total f32, read),
    ids (b * l i32, read), logits (b * l * vocab f32, written)], optionally
    followed by the Mamba-3 layers' report arrays (four f32 addresses per
    Mamba-3 layer in layer order: h_last, k_last, v_last, theta_last;
    written); params =
    [b, l]. The registry is reloaded only when the program's weight key is
    stale (training/samba_program.mojo refresh_weights). With no state
    addresses the run starts from a zero state and the
    state is discarded (`SambaStack.forward`); otherwise `pieces` / `cursors`
    are the caller's state (as `samba_decode_load` takes them), loaded before
    the run and written back after it, and the post-call cursors are
    returned. One wait. A failed call marks the program unusable."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_program_forward: a host build has no device program")
    else:
        var owner = program.downcast_value_ptr[SambaStackProgram]()
        # [registry, ids, logits] then, optionally, the Mamba-3 layers' four
        # report addresses each (h_last, k_last, v_last, theta_last).
        var head = Python.list(addresses[0], addresses[1], addresses[2]) if len(addresses) > 3 else addresses
        var a = _addrs(head, 3, "samba_program_forward")
        var reports = List[Int]()
        for i in range(3, len(addresses)):  # driver-loop: over the Mamba-3 layers' report addresses
            reports.append(Int(py=addresses[i]))
        _params(params, 2, "samba_program_forward")
        var b = Int(py=params[0])
        var l = Int(py=params[1])
        if b != owner[].b:
            raise Error("samba_program_forward: the program holds " + String(owner[].b) + " rows, the call has "
                        + String(b))
        var stateful = len(pieces) > 0
        var sp = _nonnull_words(pieces, "samba_program_forward") if stateful else List[Int]()
        var sc = _int_words(cursors, "samba_program_forward") if stateful else List[Int]()
        if owner[].busy:
            raise Error("samba_program_forward: program busy")
        if not owner[].usable:
            raise Error("samba_program_forward: the program was lost after a failed call")
        owner[].busy = True
        var out = List[Int]()
        try:
            with GILReleased(Python()):
                owner[].refresh_weights(_f32_ptr(a[0]))
                if stateful:
                    owner[].load_state(sp, sc)
                else:
                    owner[].zero_state()
                owner[].run(_i32_ptr(a[1]), _f32_ptr(a[2]), l)
                if len(reports) > 0:
                    owner[].enqueue_reports(reports)
                if stateful:
                    out = owner[].enqueue_export(sp)
                owner[].finish()
        except error:
            owner[].busy = False
            owner[].usable = False
            raise error
        owner[].busy = False
        if not stateful:
            return PythonObject(None)
        return _py_words(out)


def samba_decode_open_binding(
    layers: PythonObject, params: PythonObject, param_addr: PythonObject,
    pieces: PythonObject, cursors: PythonObject,
) raises -> PythonObject:
    """Open a decode session's program: `samba_program_open`'s arguments,
    the registry read once here, then `pieces` / `cursors` (the state, as
    `samba_decode_load` takes them) loaded. Returns the program object."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_decode_open: a host build has no device program")
    else:
        var a = _nonnull_words(pieces, "samba_decode_open")
        var c = _int_words(cursors, "samba_decode_open")
        var program = _samba_program_new(layers, params, param_addr, "samba_decode_open")
        with GILReleased(Python()):
            program.load_state(a, c)
            program.finish()
        return PythonObject(alloc=program^)


def _decode_run(session: PythonObject, addresses: PythonObject, params: PythonObject, n_params: Int,
                name: String) raises -> PythonObject:
    var owner = session.downcast_value_ptr[SambaStackProgram]()
    var a = _addrs(addresses, 2, name)
    _params(params, n_params, name)
    if Int(py=params[0]) != owner[].b:
        raise Error(name + ": the session holds " + String(owner[].b) + " rows, the call has "
                    + String(Int(py=params[0])))
    var n = Int(py=params[1]) if n_params == 2 else 1
    if owner[].busy:
        raise Error(name + ": session busy")
    if not owner[].usable:
        raise Error(name + ": the session was lost after a failed call; close it and open a new one")
    owner[].busy = True
    try:
        with GILReleased(Python()):
            if n_params == 2:
                owner[].decode(_i32_ptr(a[0]), _f32_ptr(a[1]), n)
            else:
                owner[].run(_i32_ptr(a[0]), _f32_ptr(a[1]), 1)
                owner[].finish()
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(owner[].first_position())


def samba_decode_step_binding(
    session: PythonObject, addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """One token per row. addresses = [ids (b i32, read), logits (b * vocab
    f32, written)]; params = [b], refused unless it is the session's batch.
    Returns the first attention layer's cached tokens after the step. A
    failed step marks the session lost."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_decode_step: a host build has no device program")
    else:
        return _decode_run(session, addresses, params, 1, "samba_decode_step")


def samba_decode_tokens_binding(
    session: PythonObject, addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """N tokens per row in one call (decode plan D2). addresses = [ids
    (b * n i32 in (B, N) order, read), logits (b * n * vocab f32 in
    (B, N, vocab) order, written)]; params = [b, n]. Bytes equal n
    `samba_decode_step` calls. A failed call marks the session lost."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_decode_tokens: a host build has no device program")
    else:
        return _decode_run(session, addresses, params, 2, "samba_decode_tokens")


def samba_decode_export_binding(session: PythonObject, pieces: PythonObject) raises -> PythonObject:
    """The session's state into the caller's arrays (`samba_decode_load`'s
    addresses); returns the cursor words as a list of ints."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_decode_export: a host build has no device program")
    else:
        var owner = session.downcast_value_ptr[SambaStackProgram]()
        var a = _nonnull_words(pieces, "samba_decode_export")
        if owner[].busy:
            raise Error("samba_decode_export: session busy")
        owner[].busy = True
        var cursors = List[Int]()
        try:
            with GILReleased(Python()):
                cursors = owner[].export_state(a)
        except error:
            owner[].busy = False
            raise error
        owner[].busy = False
        return _py_words(cursors)


def samba_decode_load_binding(
    session: PythonObject, pieces: PythonObject, cursors: PythonObject
) raises -> PythonObject:
    """The caller's state arrays into the session (after the caller edited
    them). pieces = per layer in stack order, ten Mamba-3 piece addresses
    (theta, h, buffer_qrot, buffer_krot, buffer_v, buffer_dt, buffer_sig,
    buffer_adt, pending_k, pending_v) or two attention addresses (k_cache,
    v_cache); cursors = per layer two words (buffered_tokens, pending) or
    (cached_tokens, 0). A successful load makes a lost session usable."""
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba_decode_load: a host build has no device program")
    else:
        var owner = session.downcast_value_ptr[SambaStackProgram]()
        var a = _nonnull_words(pieces, "samba_decode_load")
        var c = _int_words(cursors, "samba_decode_load")
        if owner[].busy:
            raise Error("samba_decode_load: session busy")
        owner[].busy = True
        try:
            with GILReleased(Python()):
                owner[].load_state(a, c)
                owner[].finish()
        except error:
            owner[].busy = False
            owner[].usable = False
            raise error
        owner[].busy = False
        owner[].usable = True
        return PythonObject(None)


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


from training.neural_identical_experiments import IDN_CHUNKED_LM_HEAD_V2
from gemm.contract import CONTRACT_K_LEAF_MIN


def training_experiment_profile_binding() raises -> PythonObject:
    """Checkpoint identity for optional NI34/35/08 numerical contracts."""
    var profile = String("baseline")
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_SOFTMAX=2 (slower) deleted 2026-10-10 by c1bf9d832; code recoverable at ca25d9321.
    if IDN_CHUNKED_LM_HEAD_V2:
        profile += "+head-serial-logit-chunked-v2"
    # TOMBSTONE: MOJOLEARN_IDN_CE_TOKEN_FOLD=2 (noise) deleted 2026-10-10 by 3034789a2; code recoverable at ca25d9321.
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


def sha256_hex_binding(value: PythonObject) raises -> PythonObject:
    """The SHA-256 of a bytes-like value, 64 lowercase hex digits (core/sha256.mojo)."""
    return sha256_hexdigest(value)


def neural_arithmetic_profile_binding() raises -> PythonObject:
    return PythonObject(neural_training_profile())


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_training.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_training() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    # FAST AND IDENTICAL (lane neural, 2026-09-27). This lane was
    # IDENTICAL-only from 2026-09-10 because its fused kernels were once gated
    # on IDENTICAL and the lower tiers fell back to slower unfused arms. Those
    # fallbacks are gone: FAST runs the same kernels and the same launches with
    # the pins in checks/numerics.mojo compiled to the free schedule. FAST
    # promises quality, never bits.
    try:
        var m = PythonModuleBuilder("_mojolearn_training")
        m.def_function[neural_arithmetic_profile_binding]("neural_arithmetic_profile")
        m.def_function[sha256_hex_binding]("sha256_hex")
        m.def_function[training_vendor_binding]("training_vendor")
        m.def_function[training_numeric_mode_binding]("training_numeric_mode")
        # TOMBSTONE: clip_grad_norm, clip_pool_fault_available, optimizer_resident_copy_dev exports (dead native entries: no live caller; clip_grad_norm_ runs optimizer_door_clip) deleted 2026-10-26 by c76141c5c; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_training.patch
        m.def_function[clip_parallel_available_binding]("clip_parallel_available")
        m.def_function[accumulate_pool_fault_available_binding]("accumulate_pool_fault_available")
        m.def_function[accumulate_parallel_available_binding]("accumulate_parallel_available")
        m.def_function[optimizer_parallel_available_binding]("optimizer_parallel_available")
        m.def_function[optimizer_step_binding]("optimizer_step")
        m.def_py_c_function(optimizer_door_step_c, "optimizer_door_step")
        m.def_py_c_function(optimizer_door_io_c, "optimizer_door_io")
        m.def_py_c_function(optimizer_door_state_c, "optimizer_door_state")
        m.def_py_c_function(optimizer_door_state_set_c, "optimizer_door_state_set")
        m.def_py_c_function(optimizer_door_load_state_c, "optimizer_door_load_state")
        m.def_py_c_function(optimizer_door_clip_c, "optimizer_door_clip")
        m.def_py_c_function(optimizer_door_accumulate_c, "optimizer_door_accumulate")
        m.def_py_c_function(entry_zero, "optimizer_door_zero")
        m.def_function[optimizer_resident_open_binding]("optimizer_resident_open")
        m.def_function[optimizer_resident_close_binding]("optimizer_resident_close")
        m.def_function[optimizer_resident_download_binding]("optimizer_resident_download")
        m.def_function[optimizer_resident_upload_binding]("optimizer_resident_upload")
        m.def_function[optimizer_resident_step_binding]("optimizer_resident_step")
        comptime if IDN_OPT_PARAMS_RESIDENT:
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
            # lane ne3-c-blocks-embedding (row N117): residual dropout on device arrays
            m.def_function[residual_dropout_dev_binding]("residual_dropout_dev")
            m.def_function[residual_dropout_backward_dev_binding]("residual_dropout_backward_dev")
            # TOMBSTONE: clip_grad_norm, clip_pool_fault_available, optimizer_resident_copy_dev exports (dead native entries: no live caller; clip_grad_norm_ runs optimizer_door_clip) deleted 2026-10-26 by c76141c5c; code recoverable at a88745022.
            # Restore: git apply experiments/removed/dead_entries_training.patch
        # TOMBSTONE: clip_grad_norm, clip_pool_fault_available, optimizer_resident_copy_dev exports (dead native entries: no live caller; clip_grad_norm_ runs optimizer_door_clip) deleted 2026-10-26 by c76141c5c; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_training.patch
        m.def_function[ce_loss_binding]("ce_loss")
        # TOMBSTONE: mlp_train_step, the mlp_resident_* session and small_mlp_api_v1 entries (MOJOLEARN_AFN_MLP_RESIDENT / _MULTISTEP / _FUSED_STEP / _ALL, merged-unmeasured; superseded by SmallMLPTrainer's captured program, autograd/small_mlp_program.mojo, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
        # Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch
        m.def_function[neural_primitives_api_binding]("neural_primitives_api_v1")
        m.def_function[samba_resident_native_handle_binding]("samba_resident_native_handle_v1")
        m.def_function[embedding_forward_binding]("embedding_forward")
        m.def_function[embedding_backward_binding]("embedding_backward")
        m.def_function[rms_norm_forward_binding]("rms_norm_forward")
        m.def_function[rms_norm_backward_binding]("rms_norm_backward")
        m.def_function[linear_forward_binding]("linear_forward")
        m.def_function[neural_gemm_binding]("neural_gemm")
        m.def_function[residual_dropout_binding]("residual_dropout")
        m.def_function[residual_dropout_backward_binding]("residual_dropout_backward")
        m.def_function[linear_backward_binding]("linear_backward")
        m.def_function[samba_head_loss_binding]("samba_head_loss")
        m.def_function[samba_resident_enabled_binding]("samba_resident_enabled")
        comptime if IDN_SAMBA_RESIDENT_STEP:
            # lane S1 samba-resident: IDENTICAL only, behind its define
            _ = m.add_type[SambaResidentSession]("_SambaResidentSession")
            m.def_function[samba_resident_open_binding]("samba_resident_open")
            m.def_function[samba_resident_step_binding]("samba_resident_step")
            m.def_function[samba_resident_accumulated_step_binding]("samba_resident_accumulated_step")
        m.def_function[samba_program_enabled_binding]("samba_program_enabled")
        comptime if SAMBA_PROGRAM_ON:
            # lane x6-samba-on-executor: the Samba stack program (device builds)
            _ = m.add_type[SambaStackProgram]("_SambaStackProgram")
            m.def_function[samba_program_open_binding]("samba_program_open")
            m.def_function[samba_program_shape_binding]("samba_program_shape")
            m.def_function[samba_program_forward_binding]("samba_program_forward")
            m.def_function[samba_decode_open_binding]("samba_decode_open")
            m.def_function[samba_decode_step_binding]("samba_decode_step")
            m.def_function[samba_decode_tokens_binding]("samba_decode_tokens")
            m.def_function[samba_decode_export_binding]("samba_decode_export")
            m.def_function[samba_decode_load_binding]("samba_decode_load")
        comptime if AFN_SAMBA_FUSE:
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
