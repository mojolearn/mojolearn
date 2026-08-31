# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device optimizer step of `mojolearn.identical.optimizer.fp32.v1`.

**THIS BANNER WAS FALSE AND IS CORRECTED. COMPILED, RUN AND GATED ON ONE
DEVICE.** Until 2026-08-31 this header read "NOTHING IN THIS FILE HAS EVER BEEN
COMPILED OR EXECUTED", and added that no compiler had seen it, no GPU had run
it, that it had been to no vendor, that its agreement with the oracle had not
been checked once, and that `training/mojo_only/optimizer_check.mojo` did not
exist. **`optimizer_check.mojo` was written at `ecd1a436` and both gates ran at
`b90f52ab`**, which falsified all of it. The run measured a real defect here:
with clipping OFF, a NaN planted in a PARAMETER reached `param.out`, because
the only device-side refusal lived in `identical_clip_grad_norm`, which does
not run at all when clipping is off, and covered `grad` alone. `param`, `grad`,
`m` and `v` are all inputs and `m` / `v` are CARRIED STATE, so a non-finite
entering `v` is permanent. DEVIATION 1496 added the refusal as the first
statement of `identical_optimizer_step`, calling the ORACLE'S OWN function
rather than restating it. The contract is
`training/IDENTICAL_OPTIMIZER_CONTRACT.md`; the answer is
`training/mojo_only/optimizer_oracle.mojo::optimizer_step_oracle`, bit for
bit. Contract section 16 lists what is owed. **IT HAS STILL BEEN TO ONE
VENDOR AND NOT THREE.**

ONE SOURCE, THREE VENDORS, AND NO INLINE VENDOR TEST
-----------------------------------------------------
There is no `if apple` in this file and there may never be one
(`always-gpu-agnostic`). The only vendor-shaped number an optimizer step
has is a block size, which is a SCHEDULING row and free in both modes
because no float crosses a thread boundary in the update (see below).
`OPT_TPB` is a literal here and **that is an owed defect, not a design** --
`column_max_block_size(COLUMN_SPEC_BASELINE)` is 128, so a literal 256
would be REFUSED on the portable-floor column rather than resolved down to
it. The fix is a `kernel_matrix.mojo` row and that file is outside this
lane's write set (contract 16.6).

THE SEPARATION THIS FILE MAKES STRUCTURAL
------------------------------------------
The NUMERICAL PLAN is the op sequence of contract 7.2 and 7.3, the seam
table of section 6, and the clip's partition and fold. The EXECUTION PLAN
is grid and block geometry. Here the separation is three structural facts
rather than three promises.

1. **THE UPDATE IS ELEMENTWISE AND NO FLOAT CROSSES A THREAD BOUNDARY.**
   One thread owns one element, reads that element's `param`, `grad`, `m`
   and `v`, and writes that element's results. There is no shared memory,
   no cross-thread combination, no atomic and no reduction. So there is
   nothing for a block size to reorder, and launch invariance is a
   property of the KERNEL'S SHAPE rather than of a check that happens to
   pass.
2. **EVERY SCALAR THE UPDATE NEEDS IS COMPUTED ONCE ON THE HOST** by
   `optimizer_oracle.step_scalars` -- the oracle's own function, not a
   second copy -- and passed in as `Float32` kernel arguments. A kernel
   cannot compute `beta1^t`, cannot see `t`, and has no path to a `pow`.
   Contract 7.1.
3. **EVERY REDUCTION IS DELEGATED TO `identical_gemm`.** The clip's sums
   of squares are v1 GEMM calls at `m = n = 1`, `k = N`, `OP_NT`
   (contract 3.2). This file contains no fold, no `pinned_block_sum`, and
   no partition of its own. The one place it would have needed all three
   is the one place it asks a closed, three-vendor-measured artifact
   instead.

A CONSEQUENCE OF 1 AND 2 WORTH STATING. Once clipping is done, the Adam
update is ONE launch over the entire flattened model, and its bits do not
depend on the tensor boundaries at all. `param_id` order matters to the
CLIP (contract 3.3) and to nothing else. SGD's momentum path is launched
per tensor only because the "buffer initialized" flag is per tensor
(contract 7.3b), and a per-element flag would let the first element of a
tensor take the copy arm while the second takes the recurrence arm on the
same step.

WHAT `NUMERIC_FAST` DOES HERE
------------------------------
Everything, with the pins compiled away. `identical_mul_add` becomes
`a*b + c`, `identical_mul` becomes a plain product, `ftz` becomes the
identity, `identical_sqrt` becomes the stdlib's device path (approximate
on NVIDIA, DEVIATION 258) and `identical_div` becomes `/`. The op
sequence, the launch structure, the delegation and the clip's partition
are UNCHANGED, so FAST is a correct optimizer, it is the same code on the
same path rather than dead code or a second kernel, and it makes no
identity claim.

`[[mojo-buffer-freed-at-last-use]]`: a `DeviceBuffer` is dead at its
`.unsafe_ptr()`, so every launcher below either takes buffers the CALLER
owns and keeps alive past the caller's own `synchronize()`, or holds its
own scratch past a `ctx.synchronize()` with an explicit `_ =` use. The
sub-buffer views in `identical_clip_grad_norm` are the delicate case and
carry their own note.

DEVIATIONS 1170 through 1186. Section 12 of the contract is the sabotage
set. The arms that live in a KERNEL are here; the ones that live in a
GATE (`OPT_SAB_MICROBATCH_SERIAL`, `OPT_SAB_RESUME_REINIT`, and the
fixture-shape requirements behind `OPT_SAB_CLIP_PARAM_ORDER`'s `J >= 3`)
are named here and implemented nowhere.

OWED, AND NONE OF IT IS IN THIS LANE'S WRITE SET
------------------------------------------------
The full list is contract section 16. The five that bear on THIS file.

1. `training/mojo_only/optimizer_check.mojo`. **No gate exists**, so
   device-equals-oracle has never been checked, launch invariance has
   never been checked, and every sabotage switch below has been built
   zero times.
2. A `kernel_matrix.mojo` row for `OPT_TPB`. It is a literal 256 and
   `column_max_block_size(COLUMN_SPEC_BASELINE)` is 128.
3. A DEVICE-SIDE non-finite refusal pass. `identical_clip_grad_norm`
   refuses only the scalar `clip.total_norm`; the oracle refuses all four
   buffers. The device path is weaker than the contract and says so at
   the function.
4. A BATCHED clip launcher. This one synchronizes once per tensor to keep
   `create_sub_buffer` views alive past their `.unsafe_ptr()`, which is
   `J` waits per step and an unpriced cost (`IDENTITY IS NOT FREE`).
5. An asynchronous form of `identical_optimizer_step`. The host readback
   inside the clip forces a mid-step wait, so one does not exist yet.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.mojo_only.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.mojo_only.gemm_oracle import OP_NT
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_pow,
    identical_rsqrt,
    identical_sqrt,
)
from training.mojo_only.optimizer_oracle import (
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    StepScalars,
    clip_eps,
    pow_int_f32,
    refuse_nonfinite,
    refuse_nonfinite_scalar,
    step_scalars,
)


# ===========================================================================
# THE SABOTAGE SWITCHES (contract section 12)
# ===========================================================================
# OFF in every build that does not name them. Each is a specific way to
# break the contract that a plausible implementation could reach by
# accident, and each exists so the gate can be shown to FAIL -- "a check
# that has never failed is a check nobody has tested".
#
# Build with, for example --
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_OPT_SABOTAGE_EPS_INSIDE_SQRT=1 \
#         -I . training/mojo_only/optimizer_check.mojo
#
# **EVERY ONE OF THEM HAS A FIXTURE PROPERTY WITHOUT WHICH IT IS INERT**,
# and the property is written beside the switch rather than left for the
# gate author to rediscover. An elementwise update passes almost any
# fixture, so the inert cases here are the rule and not the exception.

#: `sqrt(v_hat + eps)` instead of `sqrt(v_hat) + eps`. Contract 4d.
#: INERT unless the fixture plants `v` in the 1e-20 to 1e-12 band. With
#: `eps = 1e-8`, `eps^2` is 1e-16, and an ordinary gradient gives `v`
#: around 1e-4 where the two spellings agree to the last bit.
comptime SAB_EPS_INSIDE_SQRT = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_EPS_INSIDE_SQRT"
]()

#: `identical_rsqrt(v)` and a multiply instead of `sqrt` then `div`.
#: Contract 4b. INERT on a handful of round `v`: DEVIATION 741 measured
#: the pinned `1/sqrt` off the correctly rounded rsqrt on 134,858 of
#: 520,133 positive-normal lanes, so about three quarters of inputs agree.
#: Needs THOUSANDS of hashed `v`, and the result must be read on all three
#: columns before anything is concluded (the DEVIATION 258 lesson).
comptime SAB_RSQRT = is_defined["MOJOLEARN_OPT_SABOTAGE_RSQRT"]()

#: `m * (1/denom)` instead of `m / denom`. Contract 4c. **INERT on
#: power-of-two denominators, where a reciprocal-multiply is EXACT.**
comptime SAB_RECIP_MUL = is_defined["MOJOLEARN_OPT_SABOTAGE_RECIP_MUL"]()

#: AdamW's decay folded into the GRADIENT instead of applied to the
#: PARAMETER. Contract 7.4. **INERT at `weight_decay == 0`, where the two
#: algorithms are the same arithmetic -- and 0 is the reference's own
#: default. This is the single most likely vacuous gate in the lane.**
comptime SAB_ADAMW_AS_ADAM = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_ADAMW_AS_ADAM"
]()

#: `p - lr*wd*p` instead of `p * (1 - lr*wd)`. Contract 7.4a. INERT when
#: `lr*wd` is a power of two, and weak at tiny `p`.
comptime SAB_DECAY_ADD_FORM = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_DECAY_ADD_FORM"
]()

#: `m + c1*(g - m)`, the `lerp_` spelling, instead of PRODUCT then FMA.
#: Contract 7.2a. INERT when `g` and `m` are in the same binade with no
#: low bits to lose.
comptime SAB_MOMENT_LERP = is_defined["MOJOLEARN_OPT_SABOTAGE_MOMENT_LERP"]()

#: `(c2*g)*g` instead of `c2*(g*g)`. Contract 7.2b. INERT on round `g`.
comptime SAB_SQ_ASSOC = is_defined["MOJOLEARN_OPT_SABOTAGE_SQ_ASSOC"]()

#: Explicit `m_hat` and `v_hat` and `lr *` at the end, the documented-
#: pseudocode shape, instead of the step-size-and-denominator shape.
#: Contract 7.2c. No fixture property is known to make this inert, which
#: is itself a claim that has not been checked -- report the moved-cell
#: count, do not assume it separates.
comptime SAB_MHAT_FORM = is_defined["MOJOLEARN_OPT_SABOTAGE_MHAT_FORM"]()

#: O14 as a rounded product then a rounded add. Contract 7.2d.
#: **THIS IS THE MOST IMPORTANT INERT CASE IN THE FILE.**
#: `check-ieee-arith` scored Metal as UNFUSED over 2^20 HASHED patterns
#: and the verdict was WRONG, because ZERO of those patterns separate a
#: fused `a*b + c` from an unfused one -- random exponents put the product
#: and the addend so far apart that both spellings round identically. The
#: fixture must be BUILT to separate, meaning `step_size * q` and `p`
#: within a few binades with the product's tail nonzero. A random fixture
#: reports this arm INERT and the report is false.
comptime SAB_UNFUSED_UPDATE = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_UNFUSED_UPDATE"
]()

#: Flush only at the store, not at every intermediate. Contract section 6.
#: INERT on any fixture whose gradients are within a few binades of 1.0.
#: **Plant a gradient near 1e-25, where the VALUE is a perfectly ordinary
#: normal and the SQUARE is not representable as one.**
comptime SAB_FTZ_LATE = is_defined["MOJOLEARN_OPT_SABOTAGE_FTZ_LATE"]()

#: Skip the clip rescale when the coefficient is exactly 1.0. Contract
#: 3.4c. INERT on normal gradients -- `fma(1.0, x, -0.0)` returns `x`
#: exactly for every finite `x`. **Plant a SUBNORMAL gradient cell, and
#: compare the `clip.grad` stage, not `param.out`**, because the
#: optimizer's own load flushes and the difference is card visible and
#: downstream inert.
comptime SAB_CLIP_SKIP_AT_ONE = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_CLIP_SKIP_AT_ONE"
]()

#: One flat `sqrt(sum_j sumsq_j)` instead of the reference's two-level
#: `sqrt(sum_j norm_j^2)`. Contract 3.1. INERT at `J == 1`.
comptime SAB_CLIP_FLAT_NORM = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_CLIP_FLAT_NORM"
]()

#: Reverse the `param_id` order of the cross-tensor fold. Contract 3.3.
#: **INERT at `J == 2`: reversing two elements swaps the two children of
#: ONE tree node and `a + b` equals `b + a` bitwise.** Needs `J >= 3`,
#: prefer `J = 5` so the odd-tail carry fires twice.
comptime SAB_CLIP_PARAM_ORDER = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_CLIP_PARAM_ORDER"
]()

#: A hand-written serial fold instead of the v1 GEMM. Contract 3.2.
#: **INERT when every `N <= 128`: there `P == 1`, the tree has no
#: arithmetic node, and the v1 answer IS the serial ascending chain.**
#: Needs `N > 128`, and `N = 300` for the ragged 44-element last leaf.
comptime SAB_CLIP_SERIAL_FOLD = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_CLIP_SERIAL_FOLD"
]()

#: The clip partition reads the LAUNCH instead of `contract_leaf_size(N)`.
#: Contract 3.2 and gemm contract section 6's first sentence. INERT under
#: a single launch geometry; needs the geometry sweep.
comptime SAB_CLIP_BLOCK_PARTITION = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_CLIP_BLOCK_PARTITION"
]()

#: `beta^t` by running product instead of `pow_int_f32`. Contract 5.1.
#: **INERT at `t <= 6`: the two spellings agree exactly through `t = 6`
#: for `beta = 0.9` and `beta = 0.999` alike (PREDICTED, derived
#: off-repository, not measured).** Run to `t >= 8`, ideally 1000.
comptime SAB_POW_RUNNING = is_defined["MOJOLEARN_OPT_SABOTAGE_POW_RUNNING"]()

#: `beta^t` through `identical_pow`, which is `exp(t * log(beta))`.
#: Contract 5.1. Separates at `t = 1`; a cheap arm, worth keeping for
#: exactly that reason.
comptime SAB_POW_EXPLOG = is_defined["MOJOLEARN_OPT_SABOTAGE_POW_EXPLOG"]()

#: Recompute the host scalars INSIDE the kernel, per element. Contract
#: 7.1's ban. **THIS ARM IS EXPECTED TO BE BIT-INERT**, because the
#: recomputation uses the same pinned primitives. It is a REACH probe, not
#: a bit probe -- it proves the kernel could reach the scalars -- and its
#: result must be REPORTED as inert rather than counted as a pass. An arm
#: whose predicted answer is "no bits move" is worth having only when it
#: is labelled that way in advance, which is what this comment does.
comptime SAB_SCALARS_PER_ELEMENT = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_SCALARS_PER_ELEMENT"
]()

#: `b_1 = c_damp * g` instead of the COPY `b_1 = g`. Contract 7.3a.
#: **INERT at `dampening == 0`, which is the default**, because `c_damp`
#: is then exactly 1.0.
comptime SAB_MOMENTUM_FIRST_STEP = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_MOMENTUM_FIRST_STEP"
]()

#: `b + momentum*g` instead of `g + momentum*b`. Contract 7.3c. INERT at
#: `momentum == 0`, and ALSO inert at `t = 1`, where `b == g` makes the
#: two readings one expression.
comptime SAB_NESTEROV_ORDER = is_defined[
    "MOJOLEARN_OPT_SABOTAGE_NESTEROV_ORDER"
]()

comptime ANY_SABOTAGE = (
    SAB_EPS_INSIDE_SQRT
    or SAB_RSQRT
    or SAB_RECIP_MUL
    or SAB_ADAMW_AS_ADAM
    or SAB_DECAY_ADD_FORM
    or SAB_MOMENT_LERP
    or SAB_SQ_ASSOC
    or SAB_MHAT_FORM
    or SAB_UNFUSED_UPDATE
    or SAB_FTZ_LATE
    or SAB_CLIP_SKIP_AT_ONE
    or SAB_CLIP_FLAT_NORM
    or SAB_CLIP_PARAM_ORDER
    or SAB_CLIP_SERIAL_FOLD
    or SAB_CLIP_BLOCK_PARTITION
    or SAB_POW_RUNNING
    or SAB_POW_EXPLOG
    or SAB_SCALARS_PER_ELEMENT
    or SAB_MOMENTUM_FIRST_STEP
    or SAB_NESTEROV_ORDER
)


def optimizer_sabotage_name() -> String:
    """The active sabotage, for a check to print beside every number. Two
    sabotages at once is a build nobody can interpret, so this returns the
    first it finds and a gate should refuse a multi-arm build."""
    if SAB_EPS_INSIDE_SQRT:
        return String("EPS_INSIDE_SQRT")
    if SAB_RSQRT:
        return String("RSQRT")
    if SAB_RECIP_MUL:
        return String("RECIP_MUL")
    if SAB_ADAMW_AS_ADAM:
        return String("ADAMW_AS_ADAM")
    if SAB_DECAY_ADD_FORM:
        return String("DECAY_ADD_FORM")
    if SAB_MOMENT_LERP:
        return String("MOMENT_LERP")
    if SAB_SQ_ASSOC:
        return String("SQ_ASSOC")
    if SAB_MHAT_FORM:
        return String("MHAT_FORM")
    if SAB_UNFUSED_UPDATE:
        return String("UNFUSED_UPDATE")
    if SAB_FTZ_LATE:
        return String("FTZ_LATE")
    if SAB_CLIP_SKIP_AT_ONE:
        return String("CLIP_SKIP_AT_ONE")
    if SAB_CLIP_FLAT_NORM:
        return String("CLIP_FLAT_NORM")
    if SAB_CLIP_PARAM_ORDER:
        return String("CLIP_PARAM_ORDER")
    if SAB_CLIP_SERIAL_FOLD:
        return String("CLIP_SERIAL_FOLD")
    if SAB_CLIP_BLOCK_PARTITION:
        return String("CLIP_BLOCK_PARTITION")
    if SAB_POW_RUNNING:
        return String("POW_RUNNING")
    if SAB_POW_EXPLOG:
        return String("POW_EXPLOG")
    if SAB_SCALARS_PER_ELEMENT:
        return String("SCALARS_PER_ELEMENT")
    if SAB_MOMENTUM_FIRST_STEP:
        return String("MOMENTUM_FIRST_STEP")
    if SAB_NESTEROV_ORDER:
        return String("NESTEROV_ORDER")
    return String("none")


#: Record `adam.denom`, `adam.q` and `sgd.dir` into caller-supplied
#: buffers. OFF by default so the shipped step performs no store it does
#: not need; the CARD build defines it. The pointers are in every kernel
#: signature either way, so the two builds are one kernel with one
#: signature and not two kernels.
comptime OPT_RECORD_INTERMEDIATES = is_defined["MOJOLEARN_OPT_RECORD"]()


# ===========================================================================
# THE EXECUTION PLAN (contract 1184)
# ===========================================================================

#: Threads per block for every elementwise kernel here. A SCHEDULING row.
#: It is free in both modes BECAUSE no float crosses a thread boundary in
#: any kernel below -- change it and a different thread does a different
#: element, and not one addition changes what it is added to.
#:
#: **IT IS A LITERAL AND THAT IS AN OWED DEFECT** (contract 16.6).
#: `column_max_block_size(COLUMN_SPEC_BASELINE)` is 128 -- the intersection
#: of Vulkan's required `maxComputeWorkGroupInvocations` and WebGPU's
#: default -- so a literal 256 is REFUSED on the portable-floor column
#: rather than resolved down to it. The right shape is a
#: `kernel_matrix.mojo` row resolved by `column`, and that file is outside
#: this lane's write set.
comptime OPT_TPB = 256

#: The chunk count the two SABOTAGE-ONLY reduction kernels use when they
#: stand in for the v1 GEMM. Never reached in a clean build.
comptime SAB_CHUNKS = 64


def _grid_for(n: Int) -> Int:
    """Blocks needed to cover `n` elements at `OPT_TPB`. Never 0, because
    a zero grid is a launch some backends reject and others silently
    accept."""
    if n <= 0:
        return 1
    return (n + OPT_TPB - 1) // OPT_TPB


# ===========================================================================
# THE HOST SCALARS (contract 7.1), AND THE TWO POW SABOTAGES
# ===========================================================================


def device_step_scalars(cfg: OptimizerConfig, t: Int) raises -> StepScalars:
    """`optimizer_oracle.step_scalars`, with the two `beta^t` sabotage arms
    wrapped around it.

    **THE CLEAN PATH CALLS THE ORACLE'S OWN FUNCTION.** Not a copy of it.
    A second spelling of the scalars is a second arithmetic wearing the
    same name, and it is exactly the drift `identical_mul`'s docstring
    complains about with `pinned_mul`'s three copies. The sabotage arms
    below rebuild the whole struct rather than patching one field,
    because `bc1`, `step_size` and `rt_bc2` all descend from `b1t` and
    `b2t` and a patched field would leave the rest consistent with the
    clean value, which would make the arm weaker than the defect it
    models.

    `[[mojo-string-float-roundtrip]]`: nothing here prints.
    """
    comptime if SAB_POW_RUNNING or SAB_POW_EXPLOG:
        var b1t = Float32(1.0)
        var b2t = Float32(1.0)
        comptime if SAB_POW_RUNNING:
            # SABOTAGE: the running product, `t - 1` sequential roundings.
            # A real implementation reaches this because it is one
            # multiply per step instead of `log t`. INERT at `t <= 6`.
            for _i in range(t):
                b1t = ftz(identical_mul(b1t, cfg.beta1))
                b2t = ftz(identical_mul(b2t, cfg.beta2))
        else:
            # SABOTAGE: a general pow, `exp(t * log(beta))`. Not exact
            # even at t = 1, so this one separates immediately.
            b1t = ftz(identical_pow(cfg.beta1, Float32(t)))
            b2t = ftz(identical_pow(cfg.beta2, Float32(t)))
        var one = Float32(1.0)
        var bc1 = ftz(one - b1t)
        var bc2 = ftz(one - b2t)
        return StepScalars(
            b1t,
            b2t,
            bc1,
            bc2,
            ftz(identical_div(cfg.lr, bc1)),
            ftz(identical_sqrt(bc2)),
            ftz(one - cfg.beta1),
            ftz(one - cfg.beta2),
            ftz(one - ftz(identical_mul(cfg.lr, cfg.weight_decay))),
            -cfg.lr,
            ftz(one - cfg.dampening),
        )
    return step_scalars(cfg, t)


# ===========================================================================
# THE ELEMENTWISE UPDATE KERNELS (contract 7.2 and 7.3)
# ===========================================================================


def adam_update_kernel(
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    m_state: MutPointer[Float32, MutAnyOrigin],
    v_state: MutPointer[Float32, MutAnyOrigin],
    denom_out: MutPointer[Float32, MutAnyOrigin],
    q_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    is_adamw_in: Int32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    c1: Float32,
    c2: Float32,
    step_size: Float32,
    rt_bc2: Float32,
    decay_mul: Float32,
    lr: Float32,
    bc1: Float32,
    bc2: Float32,
):
    """Contract 7.2, seams O1 through O14, one thread per element.

    Every argument after `n_in` is a HOST scalar (contract 7.1). This
    kernel cannot see `t`, cannot compute `beta1^t` and has no path to a
    `pow`. `lr`, `bc1` and `bc2` are passed only because
    `SAB_MHAT_FORM` needs them to spell the alternative it models; the
    clean path uses `step_size` and `rt_bc2` and never reads them.

    THE ELEMENTWISE PROPERTY IS THE WHOLE LAUNCH-INVARIANCE ARGUMENT.
    This kernel reads element `i` and writes element `i`. No shared
    memory, no `barrier`, no atomic, no cross-lane primitive, no
    reduction. So `block_dim` and `grid_dim` decide WHICH thread does
    element `i` and cannot decide what element `i` is. Contract 11(b) is
    then a property of the shape of this function, and the gate verifies
    what the construction promises rather than being the only thing
    holding the claim up.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return

    var is_adamw = is_adamw_in != Int32(0)
    comptime if SAB_ADAMW_AS_ADAM:
        # SABOTAGE: decoupled decay becomes coupled. Contract 7.4.
        # **INERT at weight_decay == 0**, which is the default.
        is_adamw = False

    var g = ftz(grad.unsafe_load(i))  # O1, seam
    var p = ftz(param.unsafe_load(i))  # O2, seam
    var mp = ftz(m_state.unsafe_load(i))  # O3, seam
    var vp = ftz(v_state.unsafe_load(i))  # O3, seam

    comptime if SAB_FTZ_LATE:
        # SABOTAGE: drop the load seams and flush only at the store.
        # Contract section 6. INERT unless an INTERMEDIATE lands
        # subnormal -- plant `g` near 1e-25 so `g*g` is 1e-50.
        g = grad.unsafe_load(i)
        p = param.unsafe_load(i)
        mp = m_state.unsafe_load(i)
        vp = v_state.unsafe_load(i)

    if weight_decay != Float32(0.0):
        if is_adamw:
            # O4b. DECOUPLED, a PRODUCT on the PARAMETER.
            comptime if SAB_DECAY_ADD_FORM:
                # SABOTAGE: `p - lr*wd*p`, the additive reading.
                var lw = ftz(identical_mul(lr, weight_decay))
                p = ftz(identical_mul_add(-lw, p, p))
            else:
                p = ftz(identical_mul(decay_mul, p))
        else:
            # O4a. COUPLED, ONE fused rounding into the GRADIENT.
            # NOTE the negative-zero caveat the oracle carries. This
            # branch is NOT a free optimization, because `fma(0, p, g)`
            # is not bitwise `g` when `g` is `-0.0`.
            g = ftz(identical_mul_add(weight_decay, p, g))

    # O5 and O6, a PRODUCT then an FMA. Contract 7.2a.
    var m = Float32(0.0)
    comptime if SAB_MOMENT_LERP:
        # SABOTAGE: the `lerp_` spelling, `m + c1*(g - m)`.
        var d = ftz(g - mp)
        m = ftz(identical_mul_add(c1, d, mp))
    else:
        var ms = ftz(identical_mul(beta1, mp))
        m = ftz(identical_mul_add(c1, g, ms))

    # O7, O8, O9. Contract 7.2b, the square is formed FIRST.
    var v = Float32(0.0)
    var vs = ftz(identical_mul(beta2, vp))
    comptime if SAB_SQ_ASSOC:
        # SABOTAGE: `(c2*g)*g` instead of `c2*(g*g)`. INERT on round `g`.
        var cg = ftz(identical_mul(c2, g))
        v = ftz(identical_mul_add(cg, g, vs))
    else:
        var g2 = ftz(identical_mul(g, g))
        v = ftz(identical_mul_add(c2, g2, vs))

    comptime if SAB_FTZ_LATE:
        # Under the late-flush arm the intermediates above were stored
        # unflushed; recompute them that way so the arm is a whole
        # spelling and not a half one.
        var ms2 = identical_mul(beta1, mp)
        m = identical_mul_add(c1, g, ms2)
        var g2b = identical_mul(g, g)
        v = identical_mul_add(c2, g2b, identical_mul(beta2, vp))

    # O10 through O13, the denominator and the quotient.
    var dn = Float32(0.0)
    var q = Float32(0.0)
    comptime if SAB_MHAT_FORM:
        # SABOTAGE: the documented-pseudocode shape. `m_hat` and `v_hat`
        # explicitly, `lr *` at the end. One more rounding.
        var mh = ftz(identical_div(m, bc1))
        var vh = ftz(identical_div(v, bc2))
        dn = ftz(ftz(identical_sqrt(vh)) + eps)
        q = ftz(identical_div(mh, dn))
        var stepped = ftz(identical_mul(lr, q))
        param.unsafe_store(i, ftz(p - stepped))
        m_state.unsafe_store(i, ftz(m))
        v_state.unsafe_store(i, ftz(v))
        comptime if OPT_RECORD_INTERMEDIATES:
            denom_out.unsafe_store(i, dn)
            q_out.unsafe_store(i, q)
        return

    comptime if SAB_EPS_INSIDE_SQRT:
        # SABOTAGE: `eps` inside the root. Contract 4d.
        # INERT unless `v` is planted in the 1e-20 to 1e-12 band.
        var vh2 = ftz(identical_div(v, bc2))
        dn = ftz(identical_sqrt(ftz(vh2 + eps)))
    else:
        comptime if SAB_RSQRT:
            # SABOTAGE: an rsqrt where the contract spells sqrt then div.
            # Contract 4b. `identical_rsqrt` is `1/portable_sqrtf`, which
            # DEVIATION 741 measured off the correctly rounded rsqrt on
            # 134,858 of 520,133 positive normals -- so about three
            # quarters of inputs agree and a small fixture is inert.
            var r = ftz(identical_rsqrt(v))
            var s2 = ftz(identical_div(Float32(1.0), r))
            var inv_rt = ftz(identical_div(Float32(1.0), rt_bc2))
            dn = ftz(ftz(identical_mul(s2, inv_rt)) + eps)
        else:
            var s = ftz(identical_sqrt(v))  # O10
            var sd = ftz(identical_div(s, rt_bc2))  # O11
            dn = ftz(sd + eps)  # O12, eps OUTSIDE the root

    comptime if SAB_RECIP_MUL:
        # SABOTAGE: a reciprocal and a multiply. Contract 4c.
        # **EXACT, and therefore INERT, on a power-of-two denominator.**
        var rd = ftz(identical_div(Float32(1.0), dn))
        q = ftz(identical_mul(m, rd))
    else:
        q = ftz(identical_div(m, dn))  # O13, a TRUE divide

    # O14. ONE fused rounding. Contract 7.2d.
    var p_out = Float32(0.0)
    comptime if SAB_UNFUSED_UPDATE:
        # SABOTAGE: round the product, then round the subtract.
        # **A RANDOM FIXTURE CANNOT SEE THIS.** Zero of 2^20 hashed
        # patterns separated fused from unfused in `check-ieee-arith`.
        var pr = ftz(identical_mul(step_size, q))
        p_out = ftz(p - pr)
    else:
        p_out = ftz(identical_mul_add(-step_size, q, p))

    param.unsafe_store(i, p_out)
    m_state.unsafe_store(i, ftz(m))
    v_state.unsafe_store(i, ftz(v))
    comptime if OPT_RECORD_INTERMEDIATES:
        denom_out.unsafe_store(i, dn)
        q_out.unsafe_store(i, q)


def sgd_update_kernel(
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    dir_out: MutPointer[Float32, MutAnyOrigin],
    begin_in: Int32,
    count_in: Int32,
    buf_initialized_in: Int32,
    nesterov_in: Int32,
    momentum: Float32,
    c_damp: Float32,
    weight_decay: Float32,
    neg_lr: Float32,
):
    """Contract 7.3, seams S1 through S5, one thread per element.

    **LAUNCHED PER TENSOR**, because `buf_initialized` is a PER-TENSOR
    flag (contract 7.3b) and passing it as a scalar is what makes that
    structural. A per-element flag would let the first element of a tensor
    take the copy arm while the second takes the recurrence arm on the
    same step, which is not a spelling of anything.

    `begin_in` is the tensor's offset into the flat buffers, so the
    launcher never has to build a sub-buffer view for this path and
    `[[mojo-buffer-freed-at-last-use]]` has one fewer place to bite.
    """
    var count = Int(count_in)
    var local = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if local >= count:
        return
    var i = Int(begin_in) + local

    var g = ftz(grad.unsafe_load(i))  # S1
    var p = ftz(param.unsafe_load(i))  # S2

    if weight_decay != Float32(0.0):
        g = ftz(identical_mul_add(weight_decay, p, g))  # S3

    var b = ftz(buf.unsafe_load(i))
    if momentum != Float32(0.0):
        if buf_initialized_in == Int32(0):
            comptime if SAB_MOMENTUM_FIRST_STEP:
                # SABOTAGE: apply the dampening on the step that CREATES
                # the buffer. Contract 7.3a.
                # **INERT at dampening == 0, which is the default.**
                b = ftz(identical_mul(c_damp, g))
            else:
                # A COPY. No arithmetic, no seam of its own.
                b = g
        else:
            var bs = ftz(identical_mul(momentum, b))
            b = ftz(identical_mul_add(c_damp, g, bs))
        buf.unsafe_store(i, ftz(b))
        if nesterov_in != Int32(0):
            comptime if SAB_NESTEROV_ORDER:
                # SABOTAGE: the transposed reading. Contract 7.3c.
                # INERT at t = 1, where b == g.
                g = ftz(identical_mul_add(momentum, g, b))
            else:
                g = ftz(identical_mul_add(momentum, b, g))
        else:
            g = b

    param.unsafe_store(i, ftz(identical_mul_add(neg_lr, g, p)))  # S5
    comptime if OPT_RECORD_INTERMEDIATES:
        dir_out.unsafe_store(i, g)


# ===========================================================================
# THE CLIP PASS (contract section 3)
# ===========================================================================


def sqrt_vec_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Seam C3. `norm_j = ftz(identical_sqrt(ftz(sumsq_j)))`, `J` elements.

    `identical_sqrt` and not `std.math.sqrt`. This is not defensive --
    `std.math.sqrt` lowers to an APPROXIMATE PTX sqrt on NVIDIA (DEVIATION
    258, 180,714 of 2^20 patterns off by one ulp with 176,577 on normals),
    and an unrouted `sqrt` was the single NVIDIA miss in an otherwise
    closed lane (DEVIATION 550)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(identical_sqrt(ftz(src.unsafe_load(i)))))


def clip_finish_kernel(
    out2: MutPointer[Float32, MutAnyOrigin],
    total_sumsq: MutPointer[Float32, MutAnyOrigin],
    max_norm: Float32,
):
    """Seams C5, C6 and C7 in one single-threaded kernel, writing
    `[total_norm, coef]`.

    It is a kernel rather than host arithmetic so that `clip.total_norm`
    and `clip.coef` are DEVICE-COMPUTED stages -- a card whose scalars
    were computed on the host would agree across vendors for a reason that
    has nothing to do with the claim.

    The clamp here is contract 8c's ONE compare-select in the whole
    profile. Its operand is a square root and is therefore non-negative,
    and the launcher refuses a non-finite `total_norm` on the host before
    anything downstream reads `coef`."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var tn = ftz(identical_sqrt(ftz(total_sumsq.unsafe_load(0))))
    var denom = ftz(tn + clip_eps())
    var coef = ftz(identical_div(max_norm, denom))
    if not (coef < Float32(1.0)):
        coef = Float32(1.0)
    out2.unsafe_store(0, tn)
    out2.unsafe_store(1, coef)


def clip_scale_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    coef: Float32,
):
    """Seam C8, contract 3.4c. **UNCONDITIONAL**, every element, including
    when `coef` is exactly 1.0.

    Why the obvious optimization is forbidden. `identical_mul(1.0, x)` is
    `fma(1.0, x, -0.0)`, which returns `x` exactly for every finite `x`
    including both signed zeros -- so for a NORMAL gradient the multiply
    and a skip agree. They do NOT agree for a SUBNORMAL gradient, because
    the multiply's operand and result flushes turn it into a signed zero
    and a skip leaves the original pattern in the buffer.

    And the honest half. **That difference is CARD VISIBLE and DOWNSTREAM
    INERT**: `adam_update_kernel`'s O1 flushes the gradient on load, so by
    the time the value reaches `m` and `v` the two spellings have
    converged. This clause protects the `clip.grad` stage and any other
    consumer of the gradient buffer -- a logger, a second optimizer, a
    gradient-statistics pass -- and it does not protect `param.out`. A
    gate for `SAB_CLIP_SKIP_AT_ONE` that compares `param.out` will report
    the sabotage inert and the report will be true and useless."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    comptime if SAB_CLIP_SKIP_AT_ONE:
        if coef == Float32(1.0):
            return
    grad.unsafe_store(i, ftz(identical_mul(coef, ftz(grad.unsafe_load(i)))))


def sab_chunk_sumsq_kernel(
    partials: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    begin_in: Int32,
    count_in: Int32,
    chunks_in: Int32,
):
    """**SABOTAGE ONLY.** A sum of squares whose partition is `chunks_in`
    equal pieces rather than `contract_leaf_size(N)` leaves, each summed
    serially and ascending.

    At `chunks == 1` this is `SAB_CLIP_SERIAL_FOLD` -- the whole-N
    ascending chain, which is what a hand-written fold gives and which
    equals the v1 answer if and only if `P == 1`, that is `N <= 128`.
    At `chunks` derived from the launch it is
    `SAB_CLIP_BLOCK_PARTITION` -- a partition that reads the LAUNCH, which
    gemm contract section 6's first sentence forbids in those words.

    Unreached in a clean build."""
    var chunks = Int(chunks_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= chunks:
        return
    var count = Int(count_in)
    var begin = Int(begin_in)
    var per = (count + chunks - 1) // chunks
    var lo = c * per
    var hi = lo + per
    if hi > count:
        hi = count
    var acc = Float32(0.0)
    for p in range(lo, hi):
        var x = ftz(src.unsafe_load(begin + p))
        acc = ftz(identical_mul_add(x, x, acc))
    partials.unsafe_store(c, ftz(acc))


def sab_combine_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    idx_in: Int32,
    count_in: Int32,
):
    """**SABOTAGE ONLY.** A serial ascending fold of `count_in` partials
    into `dst[idx_in]`. This is the shape
    `core/gram_splitk.mojo::gram_splitk_reduce_kernel` ships and that the
    gemm contract required before its Phase 2 call -- an `O(P)` dependency
    chain where v1's tree has `O(log P)`. Unreached in a clean build."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var acc = Float32(0.0)
    for t in range(Int(count_in)):
        acc = ftz(ftz(acc) + ftz(partials.unsafe_load(t)))
    dst.unsafe_store(Int(idx_in), ftz(acc))


def identical_clip_grad_norm(
    ctx: DeviceContext,
    mut grad: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    mut norms: DeviceBuffer[DType.float32],
    mut total_cell: DeviceBuffer[DType.float32],
    mut out2: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut sab_partials: DeviceBuffer[DType.float32],
    offsets: List[Int],
    max_norm: Float32,
) raises -> Float32:
    """DEVIATIONS 1178, 1179, 1180. The whole clip pass on device, contract
    section 3. Returns the clamped coefficient.

    `offsets` has length `J + 1`; `offsets[j] .. offsets[j+1]` is tensor
    `j`'s slice of the flat `grad` buffer, and **`j` IS the `param_id`.**
    Its ascending order is the cross-tensor summation order (contract
    3.3), and it comes from the CALLER'S registry rather than from
    anything this function can see -- which is the point, because a
    registry that is stable across runs is the only thing that makes the
    order reproducible.

    Buffer sizes the caller must provide.
        `sumsq`        J
        `norms`        J
        `total_cell`   1
        `out2`         2
        `ws`           `identical_optimizer_workspace_floats(offsets)`,
                       which is the max over the per-tensor GEMMs and the
                       cross-tensor one
        `sab_partials` `SAB_CHUNKS`, unread in a clean build

    THE REDUCTION IS DELEGATED AND THAT IS THE CLAUSE. Each `sumsq_j` is
    `identical_gemm_into(..., 1, 1, N_j, OP_NT)` with the gradient as both
    operands. Not a fold written here. It inherits the v1 leaf partition,
    the v1 balanced tree, the odd-tail carry, the no-padding clause, and a
    three-vendor MEASUREMENT (leg 11, commit 144aa5b, Apple M4 against
    H100 against MI325X). See `clip_tensor_sumsq_oracle`'s docstring.

    `[[mojo-buffer-freed-at-last-use]]` IS THE DELICATE PART HERE.
    `create_sub_buffer` produces a view whose LAST USE is the
    `.unsafe_ptr()` inside `identical_gemm_into`, so a view created in a
    loop iteration could be freed before the kernels it was handed to have
    run. This function therefore SYNCHRONIZES ONCE PER TENSOR and keeps
    each view alive with an explicit `_ =` after the wait. **That is `J`
    synchronizations per step and it is a real cost this lane has not
    priced** (`IDENTITY IS NOT FREE`). The fix is a batched launcher that
    holds every view in a list across one wait, and it is owed rather than
    guessed at here.

    THE REFUSAL IS PARTIAL AND SAYING SO IS THE POINT. Contract 8a refuses
    a non-finite value in ANY gradient, parameter or state. This function
    refuses only the SCALAR `clip.total_norm`, because that is one float
    to copy back and a whole-buffer scan is a device pass this lane has
    not written. A non-finite gradient element does reach `total_norm` as
    a non-finite value through the sum of squares, so the scalar refusal
    catches the case that matters for CLIPPING -- but it does NOT catch a
    non-finite PARAMETER or a non-finite `m` or `v`, and it does not run
    at all when clipping is off. `optimizer_step_oracle` refuses all four
    buffers. **The device path is therefore weaker than the contract and
    the gap is a device-side refusal pass, owed.**
    """
    var j_count = len(offsets) - 1
    if j_count <= 0:
        return Float32(0.0)

    for j in range(j_count):
        var begin = offsets[j]
        var count = offsets[j + 1] - begin
        # Where the norm LANDS is `param_id` order. The sabotage writes it
        # backwards and changes nothing else, so what it moves is the
        # cross-tensor fold's order and only that.
        var slot = j
        comptime if SAB_CLIP_PARAM_ORDER:
            slot = j_count - 1 - j

        comptime if SAB_CLIP_SERIAL_FOLD or SAB_CLIP_BLOCK_PARTITION:
            var chunks = 1
            comptime if SAB_CLIP_BLOCK_PARTITION:
                # SABOTAGE: the partition is derived from the LAUNCH --
                # one chunk per BLOCK at `OPT_TPB` -- instead of from
                # `contract_leaf_size(N)`. gemm contract section 6's first
                # sentence forbids exactly this, and the property it
                # breaks is that the partition is a pure function of `N`
                # and the PROFILE. Here it is a function of `N` and
                # `OPT_TPB`, and `OPT_TPB` is an EXECUTION constant.
                chunks = _grid_for(count)
                if chunks > SAB_CHUNKS:
                    chunks = SAB_CHUNKS
            ctx.enqueue_function[sab_chunk_sumsq_kernel](
                sab_partials.unsafe_ptr(),
                grad.unsafe_ptr(),
                Int32(begin),
                Int32(count),
                Int32(chunks),
                grid_dim=(_grid_for(chunks), 1, 1),
                block_dim=(OPT_TPB, 1, 1),
            )
            ctx.enqueue_function[sab_combine_kernel](
                sumsq.unsafe_ptr(),
                sab_partials.unsafe_ptr(),
                Int32(slot),
                Int32(chunks),
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
            )
            ctx.synchronize()
        else:
            # THE CLEAN PATH. Two views of the same range so that the two
            # `mut` operands are distinct locals; the GEMM only reads
            # them.
            var ga = grad.create_sub_buffer[DType.float32](begin, count)
            var gb = grad.create_sub_buffer[DType.float32](begin, count)
            var cv = sumsq.create_sub_buffer[DType.float32](slot, 1)
            identical_gemm_into(ctx, cv, ga, gb, ws, 1, 1, count, OP_NT)
            ctx.synchronize()
            # The keep-alives. Without these three the views are dead at
            # the `.unsafe_ptr()` inside the call above.
            _ = ga
            _ = gb
            _ = cv

    # C3, then C4. Under the flat-norm sabotage the per-tensor sqrt is
    # skipped entirely and the sumsq values are summed directly, which is
    # `sqrt(sum_j sumsq_j)` -- the one-level form the reference does not
    # use. Contract 3.1. INERT at J == 1.
    comptime if SAB_CLIP_FLAT_NORM:
        ctx.enqueue_function[sab_combine_kernel](
            total_cell.unsafe_ptr(),
            sumsq.unsafe_ptr(),
            Int32(0),
            Int32(j_count),
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        ctx.synchronize()
    else:
        ctx.enqueue_function[sqrt_vec_kernel](
            norms.unsafe_ptr(),
            sumsq.unsafe_ptr(),
            Int32(j_count),
            grid_dim=(_grid_for(j_count), 1, 1),
            block_dim=(OPT_TPB, 1, 1),
        )
        ctx.synchronize()
        var na = norms.create_sub_buffer[DType.float32](0, j_count)
        var nb = norms.create_sub_buffer[DType.float32](0, j_count)
        var tv = total_cell.create_sub_buffer[DType.float32](0, 1)
        identical_gemm_into(ctx, tv, na, nb, ws, 1, 1, j_count, OP_NT)
        ctx.synchronize()
        _ = na
        _ = nb
        _ = tv

    ctx.enqueue_function[clip_finish_kernel](
        out2.unsafe_ptr(),
        total_cell.unsafe_ptr(),
        max_norm,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    ctx.synchronize()

    var h = ctx.enqueue_create_host_buffer[DType.float32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=out2)
    ctx.synchronize()
    var total_norm = h.unsafe_ptr().unsafe_load(0)
    var coef = h.unsafe_ptr().unsafe_load(1)
    _ = h^

    # Contract 8a, on the one scalar this path can afford to inspect.
    refuse_nonfinite_scalar(String("clip.total_norm"), total_norm)

    ctx.enqueue_function[clip_scale_kernel](
        grad.unsafe_ptr(),
        Int32(offsets[j_count]),
        coef,
        grid_dim=(_grid_for(offsets[j_count]), 1, 1),
        block_dim=(OPT_TPB, 1, 1),
    )
    ctx.synchronize()
    return coef


# ===========================================================================
# THE HOST-VISIBLE ENTRY POINT
# ===========================================================================


def identical_optimizer_workspace_floats(offsets: List[Int]) -> Int:
    """The `ws` size `identical_clip_grad_norm` needs at this parameter
    shape -- the max over the per-tensor GEMMs and the cross-tensor one.

    **Sizing a workspace for one shape and letting the dispatcher pick
    another plan is an out-of-bounds write a small shape will not show
    you.** That cost the gemm lane a run --- a 1-float workspace passed to a
    SPLITK dispatch at 64 x 4 still produced the right answer because the
    allocation had slack, and only at 64 x 64 did whole regions come back
    `+0.0`. Use this function, never a guess."""
    var j_count = len(offsets) - 1
    var w = 1
    if j_count <= 0:
        return w
    for j in range(j_count):
        var count = offsets[j + 1] - offsets[j]
        var wj = identical_gemm_workspace_max_floats(1, 1, count)
        if wj > w:
            w = wj
    var wt = identical_gemm_workspace_max_floats(1, 1, j_count)
    if wt > w:
        w = wt
    return w


def opt_refuse_device_inputs(
    ctx: DeviceContext,
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    offsets: List[Int],
    cfg: OptimizerConfig,
) raises:
    """Contract 8a ON THE DEVICE ENTRY POINT, which is where it was missing.
    DEVIATION 1496.

    **THE GAP THIS CLOSES WAS MEASURED, NOT SUSPECTED.**
    `optimizer_check.mojo` clause (f), first execution 2026-08-25: with
    clipping OFF, a NaN planted in a PARAMETER reached `param.out`.
    `identical_optimizer_step` had no refusal of its own, and
    `identical_clip_grad_norm` -- the only refusal anywhere on the device
    side -- **does not run at all when clipping is off**. So contract 8a's
    "REFUSED BY NAME before any recorded stage" was a property of
    `optimizer_step_oracle` and not of the profile.

    **FOUR BUFFERS, NOT ONE, AND THAT IS THE POINT.** The loss lane's gap was
    one input; here `param`, `grad`, `m` and `v` are all inputs to a step,
    and `m`/`v` are CARRIED STATE. A non-finite that enters `v` is permanent
    -- every later step divides by `sqrt(v)` -- so an unrefused `v` poisons a
    run rather than a step. The clipping-on path caught `grad` alone, which
    is one of four.

    **IT CALLS THE ORACLE'S OWN `refuse_nonfinite` AND DOES NOT RESTATE IT**,
    so the two sides fail with the same name and the same message and can be
    compared at all. Same reason the loss lane's DEVIATION 1495 does it, and
    the same precedent (`llama_refuse_bad_inputs`).

    **THE COST IS A DOWNLOAD OF ALL FOUR BUFFERS PER STEP.** For a real model
    that is the whole parameter set crossing the bus every step, which is not
    affordable and is stated rather than hidden. A device-side scan writing
    one count per buffer is OWED; it must produce the SAME name and the SAME
    first offending index or it is a different refusal wearing this one's
    name. Until then a caller who needs the speed and accepts the risk can
    build with `-D MOJOLEARN_OPT_TRUST_INPUTS=1`, which is a DELIBERATE
    downgrade of the profile and is named so it appears in the banner.

    **INPUTS, NOT INTERMEDIATES**, the stated gap every lane here carries.
    """
    comptime if is_defined["MOJOLEARN_OPT_TRUST_INPUTS"]():
        return
    var n = offsets[len(offsets) - 1] if len(offsets) > 0 else 0
    if n <= 0:
        return
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hg = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hp.unsafe_ptr(), src_buf=param)
    ctx.enqueue_copy(dst_ptr=hg.unsafe_ptr(), src_buf=grad)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=m_state)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=v_state)
    ctx.synchronize()
    var lp = List[Float32]()
    var lg = List[Float32]()
    var lm = List[Float32]()
    var lv = List[Float32]()
    for i in range(n):
        lp.append(hp.unsafe_ptr().unsafe_load(i))
        lg.append(hg.unsafe_ptr().unsafe_load(i))
        lm.append(hm.unsafe_ptr().unsafe_load(i))
        lv.append(hv.unsafe_ptr().unsafe_load(i))
    # ORDER MATCHES THE ORACLE'S so the two sides name the SAME buffer first
    # on an input that is bad in more than one place.
    refuse_nonfinite(String("param"), lp)
    refuse_nonfinite(String("grad"), lg)
    if cfg.kind != OPT_SGD:
        refuse_nonfinite(String("exp_avg"), lm)
        refuse_nonfinite(String("exp_avg_sq"), lv)
    else:
        refuse_nonfinite(String("momentum_buffer"), lm)
    _ = hp
    _ = hg
    _ = hm
    _ = hv


def identical_optimizer_step(
    ctx: DeviceContext,
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    mut denom_out: DeviceBuffer[DType.float32],
    mut q_out: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    mut norms: DeviceBuffer[DType.float32],
    mut total_cell: DeviceBuffer[DType.float32],
    mut out2: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut sab_partials: DeviceBuffer[DType.float32],
    mut buf_initialized: List[Bool],
    offsets: List[Int],
    cfg: OptimizerConfig,
    t: Int,
) raises:
    """**THE HOST-VISIBLE ENTRY POINT.** One step of profile
    `mojolearn.identical.optimizer.fp32.v1`, and the answer is
    `optimizer_step_oracle`, bit for bit, on Apple, NVIDIA and AMD, at
    every legal launch geometry.

    **THAT AGREEMENT HAS NEVER BEEN CHECKED.** No gate exists.

    THE FOUR PHASES, and their order is part of the contract.

    1. **Clip** (section 3), over the WHOLE model, before anything
       elementwise. Its coefficient is a function of every gradient there
       is, which is why it cannot be an elementwise decision -- and why,
       with clipping ON, one parameter's update is NOT independent of the
       rest of the model. That is the reference's semantics and not a
       defect (contract 3.5), and contract 11(c)'s parameter-count
       invariance gate must therefore run with `max_norm <= 0`.
    2. **Host scalars** (7.1), ONCE, through the ORACLE's own
       `step_scalars`. Not a second copy of it.
    3. **The elementwise update** (7.2 or 7.3).
    4. **The momentum flags** flip once per tensor, AFTER that tensor's
       kernel, never per element (contract 7.3b).

    Phase 1 comes first and phase 3 reads the CLIPPED gradient. A caller
    that wants no clipping passes `max_norm <= 0` and the gradient buffer
    is untouched.

    Adam is ONE LAUNCH over the whole flattened model. Nothing in
    `adam_update_kernel` reads a tensor boundary, so `param_id` order
    matters to the clip and to nothing else. SGD is one launch per tensor
    and only because of the per-tensor flag.

    **THIS FUNCTION SYNCHRONIZES BEFORE IT RETURNS.** Every buffer is the
    CALLER'S and the caller must keep them alive past its own wait; the
    host readback inside `identical_clip_grad_norm` already forces a wait
    mid-step, so an asynchronous form of this entry does not exist yet and
    would need the refusal moved off the critical path.

    `denom_out` and `q_out` are written only when the build defines
    `MOJOLEARN_OPT_RECORD`. A caller that does not record may pass any
    one-element buffer; the pointers are in the signature either way so
    that the recording and non-recording builds are ONE kernel with ONE
    signature.
    """
    # DEVIATION 1496: contract 8a, on the DEVICE path. Measured missing
    # by optimizer_check clause (f) -- with clipping OFF a NaN planted
    # in a PARAMETER reached param.out, because the only device-side
    # refusal lives in identical_clip_grad_norm and does not run.
    # FIRST statement in the body: "before any recorded stage".
    opt_refuse_device_inputs(
        ctx, param, grad, m_state, v_state, offsets, cfg
    )

    var j_count = len(offsets) - 1
    if j_count <= 0:
        return
    var n_total = offsets[j_count]
    if n_total <= 0:
        return

    # PHASE 1.
    if cfg.max_norm > Float32(0.0):
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
            cfg.max_norm,
        )

    # PHASE 2. The oracle's own function (or, under the two pow arms, the
    # sabotage wrapper). Contract 7.1's ban on recomputing these inside a
    # kernel is a DESIGN rule, and `SAB_SCALARS_PER_ELEMENT` is its REACH
    # probe -- expected bit-inert, reported as such, never counted as a
    # pass.
    var sc = device_step_scalars(cfg, t)
    comptime if SAB_SCALARS_PER_ELEMENT:
        # SABOTAGE: prove the scalars are reachable from inside the step
        # by recomputing them here from `t` through the same primitives.
        # The bits do not move, and the arm exists to say so out loud.
        var b1 = pow_int_f32(cfg.beta1, t)
        var b2 = pow_int_f32(cfg.beta2, t)
        var one = Float32(1.0)
        sc = StepScalars(
            b1,
            b2,
            ftz(one - b1),
            ftz(one - b2),
            ftz(identical_div(cfg.lr, ftz(one - b1))),
            ftz(identical_sqrt(ftz(one - b2))),
            ftz(one - cfg.beta1),
            ftz(one - cfg.beta2),
            ftz(one - ftz(identical_mul(cfg.lr, cfg.weight_decay))),
            -cfg.lr,
            ftz(one - cfg.dampening),
        )

    # PHASE 3.
    if cfg.kind == OPT_SGD:
        for j in range(j_count):
            var begin = offsets[j]
            var count = offsets[j + 1] - begin
            if count <= 0:
                continue
            var init_flag = Int32(0)
            if buf_initialized[j]:
                init_flag = Int32(1)
            var nest = Int32(0)
            if cfg.nesterov:
                nest = Int32(1)
            ctx.enqueue_function[sgd_update_kernel](
                param.unsafe_ptr(),
                grad.unsafe_ptr(),
                m_state.unsafe_ptr(),
                denom_out.unsafe_ptr(),
                Int32(begin),
                Int32(count),
                init_flag,
                nest,
                cfg.momentum,
                sc.c_damp,
                cfg.weight_decay,
                sc.neg_lr,
                grid_dim=(_grid_for(count), 1, 1),
                block_dim=(OPT_TPB, 1, 1),
            )
        ctx.synchronize()
        # PHASE 4. Per TENSOR, after the launches, never per element.
        if cfg.momentum != Float32(0.0):
            for j in range(j_count):
                buf_initialized[j] = True
    else:
        var is_adamw = Int32(0)
        if cfg.kind == OPT_ADAMW:
            is_adamw = Int32(1)
        ctx.enqueue_function[adam_update_kernel](
            param.unsafe_ptr(),
            grad.unsafe_ptr(),
            m_state.unsafe_ptr(),
            v_state.unsafe_ptr(),
            denom_out.unsafe_ptr(),
            q_out.unsafe_ptr(),
            Int32(n_total),
            is_adamw,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            sc.c1,
            sc.c2,
            sc.step_size,
            sc.rt_bc2,
            sc.decay_mul,
            cfg.lr,
            sc.bc1,
            sc.bc2,
            grid_dim=(_grid_for(n_total), 1, 1),
            block_dim=(OPT_TPB, 1, 1),
        )
        ctx.synchronize()

    # `[[mojo-buffer-freed-at-last-use]]`: keep every caller buffer alive
    # past the wait above, so that a caller who drops its own handle
    # immediately after this returns cannot free memory a queued kernel
    # still points at.
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
