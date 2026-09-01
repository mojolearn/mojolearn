# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-pointer surface for the two training profiles,
`mojolearn.identical.optimizer.fp32.v1` and
`mojolearn.identical.loss.ce.fp32.v1`.

DEVIATION 1590. Written 2026-09-01 so the optimizer step, the global-norm
clip and the cross-entropy loss are reachable from Python. Nothing numeric
lands here.

WHY THIS FILE EXISTS, AND WHY IT IS NOT UNDER `checks/`
-------------------------------------------------------
This lane is shaped unlike every other family in the tree. It has no
`impl/` directory and, until this file, no `estimator.mojo`; its
implementations sit under `training/checks/`, which everywhere else in this
repository means gates and oracles rather than shipped code. `loss.mojo`,
`optimizer.mojo` and `train_loop.mojo` are real device implementations
living in the gate directory.

**THE IMPLEMENTATIONS WERE NOT MOVED, DELIBERATELY.** Relocating them to
`training/impl/` would change the import path of every file that gates
them -- `loss_check.mojo`, `optimizer_check.mojo`, `train_step_check.mojo`,
`loss_fixture.mojo`, `optimizer_fixture.mojo` and the two oracles -- and
those gates carry the only measurements this lane has, namely the loss card
md5 `a87615d9` and the optimizer card md5 `97d160b0`, each byte-identical on
an Apple M4 and an AMD MI325X, and the training-loop checkpoint
`h_all = 463245ce6c97e68d` on both boxes. A rename is a cheap edit with an
expensive failure mode, and putting a gated result at risk to tidy a
directory is the wrong trade. So this file is POINTER-SHAPED WRAPPERS THAT
CALL THE FUNCTIONS WHERE THEY ARE. The move is owed and is a separate
change with its own re-run of both gates.

WHAT IS AND IS NOT HERE
-----------------------
**There is no arithmetic in this file, and there must never be any.** It
allocates device buffers, uploads, calls the certified entry points, copies
back and waits. A multiply or an add appearing below would be a second
implementation of a bit-exact contract. That is `gemm/host_entry.mojo`'s
rule verbatim and it is the rule this file is written under; the bits come
from `training/checks/optimizer.mojo` and `training/checks/loss.mojo` and
from nowhere else. Even the ones vector is `ce_ones`, the oracle's own
producer, because a wrong ones vector is a wrong answer with no symptom.

WHERE THE MEASUREMENT STOPS, AND IT IS UNEVEN
---------------------------------------------
Two things are true at once and a caller cannot see either from here.

  * The loss contract card (17 records, md5 `a87615d9`) and the optimizer
    contract card (18 records, md5 `97d160b0`) are BYTE-IDENTICAL on an
    Apple M4 and an AMD MI325X, at the 2026-08-28 legs. Loss clause (a)
    passed 24 cases and all 61,925 compared cells; optimizer clause (a)
    passed 33 cases and all 382,822 compared cells.
  * **NO NVIDIA LEG HAS RUN** for either card, and none for the training
    loop's checkpoint comparison. This is a TWO-VENDOR result. Apple and
    AMD agreeing closes nothing on its own: they agreed through 302 GBDT
    stages while NVIDIA diverged at `tree001.winners.scores`.

Both sentences are repeated on `python/mojolearn/_training_impl.py`'s
classes, because that is the docstring a user actually reads.

THE REFUSAL SPLIT, STATED RATHER THAN LEFT TO BE DISCOVERED
-----------------------------------------------------------
Refusals divide in two here and the division is deliberate.

  * **SHAPE AND CONFIGURATION are refused in this file, before any device
    work.** Buffer counts, the offsets vector, `kind`, `t`, and the
    finiteness of every hyperparameter. The certified entry points have no
    opinion about any of these -- `optimizer_step_oracle` refuses no shape
    at all -- so nothing is restated by checking them here.
  * **BUFFER CONTENTS are refused by the certified entry points
    themselves**, as their first statement, and this file does not repeat
    that scan. `identical_optimizer_step` calls `opt_refuse_device_inputs`
    (DEVIATION 1496) and `identical_ce_forward_into` calls
    `ce_refuse_device_inputs` (DEVIATION 1495); both delegate to the
    oracle's own predicate so that the two sides fail with the same name.
    A second copy here would be a second thing to keep in step, which is
    the mistake both of those deviations exist to undo.

The one exception is the loss, and it is named on
`identical_ce_loss_host` below: `ce_refuse_inputs` is ONE function covering
shape, configuration and contents together and cannot be split, so calling
it before the upload costs a host copy of `logits`. That copy is stated,
not hidden.

`[[mojo-buffer-freed-at-last-use]]`: a `DeviceBuffer` is dead at its last
use, so every entry below ends in explicit `_ =` keep-alives past the final
`ctx.synchronize()`. The hazard is invisible at review time and free at run
time.

`[[mojo-string-float-roundtrip]]`: nothing here prints.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from training.checks.loss import (
    identical_ce_backward_into,
    identical_ce_forward_into,
    identical_ce_ones_floats,
    identical_ce_workspace_max_floats,
)
from training.checks.loss_oracle import (
    REDUCTION_MEAN,
    REDUCTION_NONE,
    REDUCTION_SUM,
    CeConfig,
    ce_count,
    ce_ones,
    ce_refuse_inputs,
)
from training.checks.optimizer import (
    OPT_RECORD_INTERMEDIATES,
    SAB_CHUNKS,
    identical_clip_grad_norm,
    identical_optimizer_step,
    identical_optimizer_workspace_floats,
)
from training.checks.optimizer_oracle import (
    OPT_ADAM,
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    refuse_nonfinite_scalar,
)


# ===========================================================================
# THE SHARED SHAPE REFUSALS
# ===========================================================================


def _offsets_from_ptr(
    offsets_ptr: MutPointer[Int32, MutUntrackedOrigin], n_tensors: Int
) raises -> List[Int]:
    """`offsets[0 .. J]` as host `Int`s, with every structural refusal this
    surface owes, BEFORE any device work.

    `offsets` has `J + 1` entries; `offsets[j] .. offsets[j+1]` is tensor
    `j`'s slice of the flat parameter buffer and **`j` IS the `param_id`**
    (optimizer contract 3.3). Its ascending order is the cross-tensor
    summation order of the clip, so it is a REGISTRY the caller owns and not
    something this file may reorder for convenience.

    `offsets[0]` must be 0. `identical_clip_grad_norm` reads
    `offsets[j] .. offsets[j+1]` as ABSOLUTE indices into `grad` and reads
    `offsets[J]` as the total element count, so a nonzero first entry is a
    buffer that is partly unread and partly out of bounds rather than an
    offset into a larger array.

    An EMPTY tensor is refused rather than skipped. The clip would enqueue a
    GEMM at `k = 0`, which is `gemm` contract section 8's degenerate case,
    and no gate in this tree has run the clip through it.
    """
    if n_tensors < 1:
        raise Error(
            String("mojolearn training: n_tensors must be at least 1, got ")
            + String(n_tensors)
        )
    var offsets = List[Int]()
    for j in range(n_tensors + 1):
        offsets.append(Int(offsets_ptr.unsafe_load(j)))
    if offsets[0] != 0:
        raise Error(
            String("mojolearn training: offsets[0] is ")
            + String(offsets[0])
            + String(", must be 0 (offsets index the flat buffer from its")
            + String(" first element; optimizer contract 3.3)")
        )
    for j in range(n_tensors):
        var count = offsets[j + 1] - offsets[j]
        if count < 1:
            raise Error(
                String("mojolearn training: tensor ")
                + String(j)
                + String(" spans ")
                + String(count)
                + String(" elements (offsets ")
                + String(offsets[j])
                + String(" .. ")
                + String(offsets[j + 1])
                + String("); offsets must be strictly ascending and an empty")
                + String(" tensor is REFUSED, not skipped")
            )
    return offsets^


def _refuse_hyperparameters(cfg: OptimizerConfig) raises:
    """Every scalar the step reads, through the ORACLE'S OWN
    `refuse_nonfinite_scalar`, before any device work.

    **This is not a restatement of contract 8a.** 8a is about the four
    BUFFERS and the certified entry point refuses those itself, as its first
    statement (DEVIATION 1496). Nothing anywhere refuses a NaN `lr`, and a
    NaN `lr` poisons every parameter on the first step with no name attached
    to the failure.

    The bounds below are the profile's own reachability conditions and not
    taste. `step_scalars` computes `identical_div(lr, bc1)` where
    `bc1 = 1 - beta1^t`, so `beta1 = 1.0` divides by zero; `eps` sits under
    a square root's sum; `weight_decay`, `momentum` and `dampening` are
    read as coefficients.
    """
    refuse_nonfinite_scalar(String("lr"), cfg.lr)
    refuse_nonfinite_scalar(String("beta1"), cfg.beta1)
    refuse_nonfinite_scalar(String("beta2"), cfg.beta2)
    refuse_nonfinite_scalar(String("eps"), cfg.eps)
    refuse_nonfinite_scalar(String("weight_decay"), cfg.weight_decay)
    refuse_nonfinite_scalar(String("momentum"), cfg.momentum)
    refuse_nonfinite_scalar(String("dampening"), cfg.dampening)
    refuse_nonfinite_scalar(String("max_norm"), cfg.max_norm)
    if cfg.kind != OPT_SGD:
        if cfg.beta1 < Float32(0.0) or cfg.beta1 >= Float32(1.0):
            raise Error(
                String("mojolearn training: beta1 must be in [0, 1), got ")
                + String(cfg.beta1)
                + String(" (at beta1 = 1 the bias correction 1 - beta1^t is")
                + String(" exactly 0 and step_scalars divides by it)")
            )
        if cfg.beta2 < Float32(0.0) or cfg.beta2 >= Float32(1.0):
            raise Error(
                String("mojolearn training: beta2 must be in [0, 1), got ")
                + String(cfg.beta2)
            )
        if cfg.eps < Float32(0.0):
            raise Error(
                String("mojolearn training: eps must be >= 0, got ")
                + String(cfg.eps)
            )


# ===========================================================================
# THE OPTIMIZER STEP
# ===========================================================================


def identical_optimizer_step_host(
    ctx: DeviceContext,
    param_ptr: MutPointer[Float32, MutUntrackedOrigin],
    grad_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
    offsets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    init_ptr: MutPointer[Int32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_tensors: Int,
    kind: Int,
    t: Int,
    nesterov: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    momentum: Float32,
    dampening: Float32,
    max_norm: Float32,
) raises -> Int:
    """One step of `mojolearn.identical.optimizer.fp32.v1` from host memory.
    Returns the total element count, `offsets[J]`.

    THE BUFFERS, AND EVERY SIZE IS COMPUTABLE BY THE CALLER BEFORE THE CALL.
    Let `N = offsets[J]` and `J = n_tensors`.

        `param_ptr`    N float32, READ AND WRITTEN in place
        `grad_ptr`     N float32, READ, and WRITTEN in place when the clip
                       runs (`max_norm > 0`); untouched otherwise
        `m_ptr`        N float32, READ AND WRITTEN. Adam's `exp_avg`;
                       SGD's `momentum_buffer`. One state pair serves both
                       algorithms so a checkpoint has one shape.
        `v_ptr`        N float32, READ AND WRITTEN by Adam and AdamW.
                       **SGD DOES NOT READ OR WRITE IT and it must still be
                       N floats long**, because the certified entry takes
                       one signature for both algorithms.
        `offsets_ptr`  J + 1 int32, ascending, `offsets[0] == 0`
        `init_ptr`     J int32, READ AND WRITTEN. SGD's per-tensor
                       `buf_initialized` flag, contract 7.3b: 0 means this
                       tensor's momentum buffer has never been written, so
                       the first step COPIES the gradient into it rather
                       than running the recurrence. It is CARRIED STATE and
                       belongs in a checkpoint beside `m`. Adam ignores it
                       and this file leaves it unchanged there.
        `info_ptr`     3 float32, WRITTEN on every call:

                           0  clip_ran     1.0 when the clip ran, +0.0 when
                                           `max_norm <= 0`
                           1  total_norm   the pre-clip global norm, or
                                           +0.0 when the clip did not run
                           2  coef         the clamped coefficient, or +0.0
                                           when the clip did not run

                       **+0.0 AND NOT 1.0 IN THE OFF CASE.** A coefficient
                       of 1.0 is a claim that the clip ran and found nothing
                       to do, which is a different fact from the clip not
                       running, and `optimizer_step_oracle` makes the same
                       distinction by leaving the `clip.*` stages EMPTY
                       rather than filling them. Slot 0 is what a caller
                       branches on.

    `kind` is 0 SGD, 1 Adam, 2 AdamW; `t` is ONE-BASED and the first step of
    a run is `t = 1`. `max_norm <= 0` turns the clip OFF, which is the
    configuration the profile's parameter-count-invariance clause 11(c) runs
    under -- with clipping ON one parameter's update depends on every other
    gradient in the model, by the reference's own semantics and not by a
    defect (contract 3.5).

    WHAT THIS FILE ALLOCATES, so that the memory is not a surprise. Four
    buffers of `N` floats for the state and two more of `N` for the recorded
    intermediates only when the binary was built with
    `-D MOJOLEARN_OPT_RECORD` (one float each otherwise, which is what the
    kernel's own docstring says a non-recording caller may pass). Then `J`
    for `sumsq`, `J` for `norms`, one cell for the total, two for the
    finish, `identical_optimizer_workspace_floats(offsets)` for the GEMM
    workspace and `SAB_CHUNKS` for the sabotage partials, which are unread
    in a clean build.

    **THE WORKSPACE COMES FROM THE CERTIFIED SIZER, NEVER FROM A GUESS.**
    Sizing it for one shape and letting the dispatcher pick another plan is
    an out-of-bounds write that a small shape will not show you; that cost
    the GEMM lane a run, where a one-float workspace still produced right
    answers at 64 x 4 and whole regions of `+0.0` at 64 x 64.

    THE CLIP RUNS TWICE OVER THE GRADIENT AND ONLY ONCE ARITHMETICALLY.
    `identical_optimizer_step` performs the clip internally and discards its
    return value, so this wrapper reads `total_norm` and `coef` back out of
    the `out2` buffer it owns AFTER the step returns. It does not call the
    clip a second time to learn the number.
    """
    if kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
        raise Error(
            String("mojolearn training: kind must be 0 (SGD), 1 (Adam) or 2")
            + String(" (AdamW), got ")
            + String(kind)
        )
    if t < 1:
        raise Error(
            String("mojolearn training: t is ONE-BASED and the first step of")
            + String(" a run is t = 1, got ")
            + String(t)
            + String(" (at t = 0 the bias correction 1 - beta^0 is exactly 0")
            + String(" and step_scalars divides by it)")
        )
    var offsets = _offsets_from_ptr(offsets_ptr, n_tensors)
    var n_total = offsets[n_tensors]

    var cfg = OptimizerConfig(
        kind,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        momentum,
        dampening,
        nesterov != 0,
        max_norm,
    )
    _refuse_hyperparameters(cfg)

    # ---- Transport in. Nothing above this line touched the device.
    var param = ctx.enqueue_create_buffer[DType.float32](n_total)
    var grad = ctx.enqueue_create_buffer[DType.float32](n_total)
    var m_state = ctx.enqueue_create_buffer[DType.float32](n_total)
    var v_state = ctx.enqueue_create_buffer[DType.float32](n_total)
    ctx.enqueue_copy(dst_buf=param, src_ptr=param_ptr)
    ctx.enqueue_copy(dst_buf=grad, src_ptr=grad_ptr)
    ctx.enqueue_copy(dst_buf=m_state, src_ptr=m_ptr)
    ctx.enqueue_copy(dst_buf=v_state, src_ptr=v_ptr)
    ctx.synchronize()

    # `denom_out` and `q_out` are written only under `MOJOLEARN_OPT_RECORD`.
    # The pointers are in the kernel signature either way, so the recording
    # and non-recording builds are ONE kernel with ONE signature; the SIZE
    # is what changes, and it is read off the same comptime flag the kernel
    # reads rather than guessed at here.
    var record_n = 1
    comptime if OPT_RECORD_INTERMEDIATES:
        record_n = n_total
    var denom_out = ctx.enqueue_create_buffer[DType.float32](record_n)
    var q_out = ctx.enqueue_create_buffer[DType.float32](record_n)

    var sumsq = ctx.enqueue_create_buffer[DType.float32](n_tensors)
    var norms = ctx.enqueue_create_buffer[DType.float32](n_tensors)
    var total_cell = ctx.enqueue_create_buffer[DType.float32](1)
    var out2 = ctx.enqueue_create_buffer[DType.float32](2)
    var ws_floats = identical_optimizer_workspace_floats(offsets)
    var ws = ctx.enqueue_create_buffer[DType.float32](ws_floats)
    var sab_partials = ctx.enqueue_create_buffer[DType.float32](SAB_CHUNKS)
    ctx.synchronize()

    var buf_initialized = List[Bool]()
    for j in range(n_tensors):
        buf_initialized.append(init_ptr.unsafe_load(j) != Int32(0))

    # ---- THE ONE CALL THAT COMPUTES ANYTHING. Everything above is
    # transport and everything below is transport.
    identical_optimizer_step(
        ctx,
        param,
        grad,
        m_state,
        v_state,
        denom_out,
        q_out,
        sumsq,
        norms,
        total_cell,
        out2,
        ws,
        sab_partials,
        buf_initialized,
        offsets,
        cfg,
        t,
    )

    # ---- Transport out.
    ctx.enqueue_copy(dst_ptr=param_ptr, src_buf=param)
    ctx.enqueue_copy(dst_ptr=m_ptr, src_buf=m_state)
    ctx.enqueue_copy(dst_ptr=v_ptr, src_buf=v_state)
    if max_norm > Float32(0.0):
        # The gradient is scaled IN PLACE by the clip, so a caller who
        # inspects its own gradient array after the step sees the CLIPPED
        # values -- which is what `torch.nn.utils.clip_grad_norm_` does and
        # is the reason it carries a trailing underscore.
        ctx.enqueue_copy(dst_ptr=grad_ptr, src_buf=grad)
    ctx.synchronize()

    for j in range(n_tensors):
        var flag = Int32(0)
        if buf_initialized[j]:
            flag = Int32(1)
        init_ptr.unsafe_store(j, flag)

    info_ptr.unsafe_store(0, Float32(0.0))
    info_ptr.unsafe_store(1, Float32(0.0))
    info_ptr.unsafe_store(2, Float32(0.0))
    if max_norm > Float32(0.0):
        var h = ctx.enqueue_create_host_buffer[DType.float32](2)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=out2)
        ctx.synchronize()
        info_ptr.unsafe_store(0, Float32(1.0))
        info_ptr.unsafe_store(1, h.unsafe_ptr().unsafe_load(0))
        info_ptr.unsafe_store(2, h.unsafe_ptr().unsafe_load(1))
        _ = h^

    _ = param
    _ = grad
    _ = m_state
    _ = v_state
    _ = denom_out
    _ = q_out
    _ = sumsq
    _ = norms
    _ = total_cell
    _ = out2
    _ = ws
    _ = sab_partials
    return n_total


# ===========================================================================
# THE GLOBAL-NORM CLIP, ON ITS OWN
# ===========================================================================


def identical_clip_grad_norm_host(
    ctx: DeviceContext,
    grad_ptr: MutPointer[Float32, MutUntrackedOrigin],
    offsets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_tensors: Int,
    max_norm: Float32,
) raises -> Int:
    """`torch.nn.utils.clip_grad_norm_` at `norm_type = 2`, section 3 of the
    optimizer contract, from host memory. Returns `offsets[J]`.

    THE BUFFERS. Let `N = offsets[J]` and `J = n_tensors`.

        `grad_ptr`     N float32, READ AND WRITTEN IN PLACE
        `offsets_ptr`  J + 1 int32, ascending, `offsets[0] == 0`
        `info_ptr`     2 float32: `0` the pre-clip global norm, `1` the
                       clamped coefficient

    There is no `clip_ran` slot here, unlike the step's `info`, because
    reaching this function IS the clip running. `max_norm <= 0` is refused
    rather than treated as OFF, for the same reason: a caller who wanted no
    clipping would not call this.

    WHY THE ANSWER IS NOT `sqrt(sum of every square)`. The reference folds
    TWICE -- a per-tensor norm, then a norm over those (contract 3.1) -- and
    the flat spelling is a DIFFERENT number in float32. Both folds are
    delegated to `identical_gemm_into` at `m = n = 1`, `OP_NT`, so they
    inherit the v1 leaf partition, the balanced tree, the odd-tail carry and
    the GEMM's own three-vendor measurement at leg 11 (`144aa5b`). The
    per-tensor ORDER is the caller's `offsets` order and nothing else, which
    is why a registry stable across runs is what makes the number
    reproducible.

    **THE REFUSAL HERE IS PARTIAL AND THAT IS THE CERTIFIED ENTRY'S OWN
    STATEMENT, not something this wrapper introduces.**
    `identical_clip_grad_norm` refuses only the SCALAR `clip.total_norm`; a
    non-finite gradient element does reach that scalar through the sum of
    squares, so the case that matters for clipping is caught, but there is
    no whole-buffer device scan and a non-finite parameter is not this
    function's business at all. A device-side refusal pass is owed.

    **ONE SYNCHRONIZE PER TENSOR.** The certified entry waits after each
    tensor's GEMM to keep its sub-buffer views alive, so a model with many
    small tensors pays `J` round trips per clip. That is a real cost this
    lane has not priced and a batched launcher is owed; it is stated here
    because a caller choosing a parameter registry is the one who decides
    `J`.
    """
    if max_norm <= Float32(0.0):
        raise Error(
            String("mojolearn training: max_norm must be > 0, got ")
            + String(max_norm)
            + String("; this entry point IS the clip, and 'no clipping' is")
            + String(" spelled by not calling it (the optimizer step takes")
            + String(" max_norm <= 0 as OFF because it has another job)")
        )
    refuse_nonfinite_scalar(String("max_norm"), max_norm)
    var offsets = _offsets_from_ptr(offsets_ptr, n_tensors)
    var n_total = offsets[n_tensors]

    var grad = ctx.enqueue_create_buffer[DType.float32](n_total)
    ctx.enqueue_copy(dst_buf=grad, src_ptr=grad_ptr)
    ctx.synchronize()

    var sumsq = ctx.enqueue_create_buffer[DType.float32](n_tensors)
    var norms = ctx.enqueue_create_buffer[DType.float32](n_tensors)
    var total_cell = ctx.enqueue_create_buffer[DType.float32](1)
    var out2 = ctx.enqueue_create_buffer[DType.float32](2)
    var ws = ctx.enqueue_create_buffer[DType.float32](
        identical_optimizer_workspace_floats(offsets)
    )
    var sab_partials = ctx.enqueue_create_buffer[DType.float32](SAB_CHUNKS)
    ctx.synchronize()

    # THE ONE CALL THAT COMPUTES ANYTHING.
    _ = identical_clip_grad_norm(
        ctx,
        grad,
        sumsq,
        norms,
        total_cell,
        out2,
        ws,
        sab_partials,
        offsets,
        max_norm,
    )

    ctx.enqueue_copy(dst_ptr=grad_ptr, src_buf=grad)
    var h = ctx.enqueue_create_host_buffer[DType.float32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=out2)
    ctx.synchronize()
    info_ptr.unsafe_store(0, h.unsafe_ptr().unsafe_load(0))
    info_ptr.unsafe_store(1, h.unsafe_ptr().unsafe_load(1))
    _ = h^

    _ = grad
    _ = sumsq
    _ = norms
    _ = total_cell
    _ = out2
    _ = ws
    _ = sab_partials
    return n_total


# ===========================================================================
# THE CROSS-ENTROPY LOSS
# ===========================================================================


def identical_ce_loss_host(
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dlogits_ptr: MutPointer[Float32, MutUntrackedOrigin],
    logits_ptr: MutPointer[Float32, MutUntrackedOrigin],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_rows: Int,
    vocab: Int,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    want_grad: Int,
    label_smoothing: Float32,
) raises -> Int:
    """`mojolearn.identical.loss.ce.fp32.v1` from host memory. Returns
    `count`, the number of rows whose target is not `ignore_index`.

    THE BUFFERS. Let `N = n_rows` and `V = vocab`.

        `loss_ptr`     1 float32, WRITTEN when `reduction != 0`. Left
                       `+0.0` under `REDUCTION_NONE`, where the certified
                       forward returns before seam L13 and there is no
                       scalar to write.
        `row_ptr`      N float32, WRITTEN on every call: the per-row loss
                       before any reduction. This is the only output under
                       `REDUCTION_NONE`.
        `dlogits_ptr`  N * V float32, WRITTEN only when `want_grad != 0`.
                       May be any one-element buffer otherwise.
        `logits_ptr`   N * V float32, READ, row-major
        `targets_ptr`  N int32, READ

    `reduction` is 0 NONE, 1 SUM, 2 MEAN. `num_items` below 1 means "not
    supplied", which is the MEAN arm's own default. `label_smoothing` is the
    contract's `eps` and **`eps == 0` selects a DIFFERENT KERNEL rather than
    a bit-inert branch** (contract 6.2(c), DEVIATION 1155); it is not a knob
    that quietly does nothing at its default.

    **THE FORWARD AND THE BACKWARD ARE ONE CALL, AND THAT IS NOT PACKAGING.**
    `identical_ce_backward_into` reads `expo` and `denom`, the buffers the
    forward wrote, and RECOMPUTES NOTHING: a second spelling of the softmax
    is a second thing that can be wrong. Two separate host entries would
    have to either recompute the forward or hand its intermediates back to
    Python, and this surface does neither. So `want_grad` is a flag on one
    call.

    **`REDUCTION_NONE` HAS NO BACKWARD** and `want_grad` is refused with it
    (contract section 11): a per-row upstream vector adds a product per cell
    whose placement, before or after seam L16's division, is a real decision
    with two different answers, and this lane has no caller for it.

    `count` IS A HOST INTEGER AND IT IS `ce_count`'s (contract 5.5).
    Counting the unignored rows on device would be exact and order-free and
    perfectly legal, and v1 keeps it on the host so that `ce_divisor` -- the
    ONE producer of the quantity the forward and the backward must agree
    about -- stays testable with no GPU present. This wrapper calls
    `ce_count`, it does not re-derive the number, and it returns it so a
    caller can see the divisor it got.

    WHAT THIS FILE ALLOCATES, and at the shipped Llama-3 vocabulary it is
    the dominant cost of the call. `shift` and `expo` are `N * V` floats
    each; `weights` and `dlogits` add two more when `want_grad` is set; the
    smoothing path adds a third `N * V` for `logp`. At `V = 128256` each of
    those is 513 KB per row. The forward is ROW INDEPENDENT through seam L11
    -- only L12 folds over `N`, and it folds `[N]` floats and not `[N, V]`
    -- so a caller who cannot afford the residency splits the ROWS, calls
    per chunk and concatenates, and under `REDUCTION_NONE` the answer is
    bit-identical to the unsplit call.

    THE REFUSAL COSTS A HOST COPY OF `logits`, AND IT IS NAMED RATHER THAN
    HIDDEN. `ce_refuse_inputs` covers the shape refusals, the configuration
    refusals, the non-finite scan and the target-range scan in ONE function
    and cannot be split, so calling it before any device work means
    materializing `logits` as a host `List`. Restating half of it here would
    be a second copy of a refusal, which is the mistake DEVIATION 1495
    exists to undo, and the certified forward then scans the same values a
    second time after the upload. A pointer-taking overload of
    `ce_refuse_inputs` in `loss_oracle.mojo` would remove both copies; that
    file is this lane's and the change is OWED, not made here, because it
    would edit a file both gates import.
    """
    if reduction != REDUCTION_NONE:
        if reduction != REDUCTION_SUM and reduction != REDUCTION_MEAN:
            raise Error(
                String("mojolearn training: reduction must be 0 (none), 1")
                + String(" (sum) or 2 (mean), got ")
                + String(reduction)
            )
    if want_grad != 0 and reduction == REDUCTION_NONE:
        raise Error(
            String("mojolearn training: REDUCTION_NONE has no backward")
            + String(" (loss contract section 11); ask for a gradient with")
            + String(" reduction 'sum' or 'mean'")
        )
    if n_rows < 1:
        raise Error(
            String("mojolearn training: n_rows must be at least 1, got ")
            + String(n_rows)
        )

    var cfg = CeConfig(vocab, ignore_index, reduction, label_smoothing, num_items)

    # ---- The refusals, BEFORE any device work, through the oracle's own
    # function. The host copy this needs is the cost named in the docstring.
    var h_logits = List[Float32]()
    var h_targets = List[Int32]()
    for i in range(n_rows):
        h_targets.append(targets_ptr.unsafe_load(i))
    for i in range(n_rows * vocab):
        h_logits.append(logits_ptr.unsafe_load(i))
    _ = ce_refuse_inputs(h_logits, h_targets, cfg)
    var count = ce_count(h_targets, ignore_index)
    _ = h_logits^
    _ = h_targets^

    var cells = n_rows * vocab
    var smoothing = cfg.smoothing_is_spelled()
    var smooth_cells = 1
    var smooth_rows = 1
    if smoothing:
        smooth_cells = cells
        smooth_rows = n_rows

    # ---- Transport in.
    var logits = ctx.enqueue_create_buffer[DType.float32](cells)
    var targets = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ctx.enqueue_copy(dst_buf=logits, src_ptr=logits_ptr)
    ctx.enqueue_copy(dst_buf=targets, src_ptr=targets_ptr)
    ctx.synchronize()

    var max_v = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var shift = ctx.enqueue_create_buffer[DType.float32](cells)
    var expo = ctx.enqueue_create_buffer[DType.float32](cells)
    var denom = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var logdenom = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var logp_target = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var nll = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var logp = ctx.enqueue_create_buffer[DType.float32](smooth_cells)
    var logp_sum = ctx.enqueue_create_buffer[DType.float32](smooth_rows)
    var smooth = ctx.enqueue_create_buffer[DType.float32](smooth_rows)
    var row = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var total = ctx.enqueue_create_buffer[DType.float32](1)
    var loss = ctx.enqueue_create_buffer[DType.float32](1)

    # THE ONES VECTOR IS THE ORACLE'S OWN, NOT A LOOP WRITTEN HERE. A wrong
    # value is a wrong answer with no symptom, because any vector produces a
    # plausible weighted sum.
    var ones_n = identical_ce_ones_floats(n_rows, vocab)
    var h_ones = ce_ones(ones_n)
    var ones = ctx.enqueue_create_buffer[DType.float32](ones_n)
    ctx.enqueue_copy(dst_buf=ones, src_ptr=h_ones.unsafe_ptr())

    # THE WORKSPACE COMES FROM THE CERTIFIED SIZER, NEVER FROM A GUESS.
    var ws = ctx.enqueue_create_buffer[DType.float32](
        identical_ce_workspace_max_floats(n_rows, vocab, reduction)
    )
    ctx.synchronize()

    # ---- THE CALLS THAT COMPUTE ANYTHING.
    identical_ce_forward_into(
        ctx,
        max_v,
        shift,
        expo,
        denom,
        logdenom,
        logp_target,
        nll,
        logp,
        logp_sum,
        smooth,
        row,
        total,
        loss,
        logits,
        targets,
        ones,
        ws,
        n_rows,
        count,
        cfg,
    )

    var grad_cells = 1
    if want_grad != 0:
        grad_cells = cells
    var weights = ctx.enqueue_create_buffer[DType.float32](grad_cells)
    var dlogits = ctx.enqueue_create_buffer[DType.float32](grad_cells)
    if want_grad != 0:
        # Enqueued behind the forward on the SAME context, which MAX runs in
        # order, so `expo` and `denom` are the forward's own values by the
        # time the weights kernel reads them. A second context or a second
        # stream would break that, and would be a change to the certified
        # entry's contract rather than to this file.
        identical_ce_backward_into(
            ctx,
            weights,
            dlogits,
            expo,
            denom,
            logp,
            targets,
            n_rows,
            count,
            cfg,
        )
    ctx.synchronize()

    # ---- Transport out.
    ctx.enqueue_copy(dst_ptr=row_ptr, src_buf=row)
    if reduction != REDUCTION_NONE:
        ctx.enqueue_copy(dst_ptr=loss_ptr, src_buf=loss)
    if want_grad != 0:
        ctx.enqueue_copy(dst_ptr=dlogits_ptr, src_buf=dlogits)
    ctx.synchronize()
    if reduction == REDUCTION_NONE:
        loss_ptr.unsafe_store(0, Float32(0.0))

    _ = h_ones^
    _ = max_v
    _ = shift
    _ = expo
    _ = denom
    _ = logdenom
    _ = logp_target
    _ = nll
    _ = logp
    _ = logp_sum
    _ = smooth
    _ = row
    _ = total
    _ = loss
    _ = logits
    _ = targets
    _ = ones
    _ = ws
    _ = weights
    _ = dlogits
    return count
