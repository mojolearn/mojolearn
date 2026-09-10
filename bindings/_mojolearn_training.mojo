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

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from max.gpu.host import DeviceContext

from training.estimator import (
    identical_ce_loss_host,
    identical_clip_grad_norm_host,
    identical_optimizer_step_host,
)
from training.mlp_ops import (
    mlp_bias_activation_host, mlp_relu_backward_host, mlp_sum_rows_host,
    mlp_validate_shape,
)
from training.checks.optimizer_oracle import microbatch_split_is_identical
from training.samba_ops import (
    samba_accumulate_host,
    samba_embedding_backward_host,
    samba_embedding_forward_host,
    samba_linear_backward_host,
    samba_linear_forward_host,
    samba_rms_norm_backward_host,
    samba_rms_norm_forward_host,
)
from core.philox_neural import neural_rng_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


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
    if len(params) != 12:
        raise Error(
            "optimizer_step: params must contain 12 values, got "
            + String(len(params))
        )
    var pp = _f32_ptr(Int(py=param_addr))
    var gp = _f32_ptr(Int(py=grad_addr))
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
        var ctx = DeviceContext()
        n_total = identical_optimizer_step_host(
            ctx, pp, gp, mp, vp, op, ip, fp, n_tensors, kind, t, nesterov,
            lr, beta1, beta2, eps, weight_decay, momentum, dampening,
            max_norm,
        )
    return PythonObject(n_total)


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
        var ctx = DeviceContext()
        n_total = identical_clip_grad_norm_host(
            ctx, gp, op, fp, n_tensors, max_norm,
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
        count = mlp_sum_rows_host(ctx, xp, op, rows, cols)
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
    for i in range(want):
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
        count = samba_rms_norm_backward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]),
            _f32_ptr(a[3]), _f32_ptr(a[4]), m, dm, eps,
        )
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
        var ctx = DeviceContext()
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
        var ctx = DeviceContext()
        count = samba_linear_backward_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), _f32_ptr(a[2]),
            _f32_ptr(a[3]), _f32_ptr(a[4]), m, n, k,
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
        var ctx = DeviceContext()
        count = samba_accumulate_host(
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
        var ctx = DeviceContext()
        count = neural_rng_host(
            ctx, _f32_ptr(a[0]), _f32_ptr(a[1]), n, offset, seed_lo, seed_hi,
            stream_id, kind, pa, pb,
        )
    return PythonObject(count)


@export
def PyInit__mojolearn_training() abi("C") -> PythonObject:
    # IDENTICAL-ONLY (2026-09-10). The FAST and DETERMINISTIC builds of this
    # lane were never a faster path: every fused kernel here is gated on
    # `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`, so the lower tiers fell back
    # to the unfused arms and ran SLOWER than the default. They are no longer
    # built (bindings/build_training.sh refuses) and the lane no longer carries
    # the fallbacks. Refuse to exist rather than answer under a tier label
    # whose arithmetic is gone.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        abort(
            String(
                "_mojolearn_training: refusing to initialize -- this lane supports only"
                " the IDENTICAL tier. Rebuild with"
                " MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh"
            )
        )
    try:
        var m = PythonModuleBuilder("_mojolearn_training")
        m.def_function[training_vendor_binding]("training_vendor")
        m.def_function[training_numeric_mode_binding]("training_numeric_mode")
        m.def_function[optimizer_step_binding]("optimizer_step")
        m.def_function[clip_grad_norm_binding]("clip_grad_norm")
        m.def_function[ce_loss_binding]("ce_loss")
        m.def_function[mlp_bias_activation_binding]("mlp_bias_activation")
        m.def_function[mlp_relu_backward_binding]("mlp_relu_backward")
        m.def_function[mlp_sum_rows_binding]("mlp_sum_rows")
        m.def_function[embedding_forward_binding]("embedding_forward")
        m.def_function[embedding_backward_binding]("embedding_backward")
        m.def_function[rms_norm_forward_binding]("rms_norm_forward")
        m.def_function[rms_norm_backward_binding]("rms_norm_backward")
        m.def_function[linear_forward_binding]("linear_forward")
        m.def_function[linear_backward_binding]("linear_backward")
        m.def_function[accumulate_binding]("accumulate")
        m.def_function[accumulation_is_aligned_binding]("accumulation_is_aligned")
        m.def_function[neural_rng_binding]("neural_rng")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_training: ", e))
