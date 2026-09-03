# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The NORMATIVE host Float32 oracle of the BACKWARD pass of one Llama-shaped
decoder block, under profile `mojolearn.identical.transformer.fp32.v1` and
the routing document `transformer/IDENTICAL_BACKWARD_PLAN.md`.

**COMPILED AND RUN 2026-09-03, NOT YET GATED.** Written 2026-08-25; this
file compiled cleanly on the FIRST attempt alongside the device file and
`transformer_backward_check.mojo`, whose preflight assertions all passed.
The check then REFUSED TO CERTIFY, because its `d_out` fixture cannot
separate a fused multiply-add chain from an unfused one and three sabotage
arms are therefore unfalsifiable. Every sentence that says what two runs
will AGREE ON is still a PREDICTION. The word "identical" appears here only
as the name of the profile and of the imported functions. It is not a claim
this lane has earned.

WHAT IS OWED, before any sentence in this file is evidence
-----------------------------------------------------------
  1. A `d_out` fixture that separates fused from unfused, so the three
     unfalsifiable sabotage arms can bite and the gate can certify.
  2. The sabotage ledger: no arm has yet been shown able to fail.
  3. A FLOAT64 DIRECTIONAL-DERIVATIVE reference. Bit identity says the
     answer is the same everywhere; it does not say the answer is the RIGHT
     derivative, and a transpose error is bit identical on three vendors.
     The plan's clause (f) covers the ROUTING and the linear seams by an
     exact-integer fixture; it does NOT cover `silu'`, the `rsqrt` tail or
     the softmax closed form, because transcendentals do not preserve exact
     integers. **That reference does not exist and is the largest owed
     item.**
  4. A three-vendor leg. Nothing here has run on one device, let alone
     three, and two backends agreeing closes nothing.

THE SHAPE OF THIS FILE, AND WHY IT IS NOT `gemm_backward.mojo`'s SHAPE
-----------------------------------------------------------------------
`gemm/checks/gemm_backward.mojo` contains NO arithmetic. The gradient of
a matmul is two more matmuls at a different transpose, so that file is pure
routing and inherits an already-certified fold. **This file cannot be that
file**, and `transformer/IDENTICAL_BACKWARD_PLAN.md` section 2 is the table
that says where and why. The short version, because it is the finding:

  Contract 7.2 licenses routing a fold through gemm v1 when the CONTRACTION
  LENGTH is the same in a prefill and in a decode step, and refuses it when
  the length differs, because gemm v1's partition `P` is a pure function of
  `k`. **A DERIVATIVE SWAPS WHICH AXIS IS CONTRACTED**: `dA` contracts over
  the output width and `dB` over the batch. So S11, the QK product, was
  routed BECAUSE it contracts over `head_dim` -- and its `dq` contracts over
  the kv length and its `dk` over the query count, neither of which is
  `head_dim`. **S11's backward may not be routed.** DEVIATIONS 1402, 1403.

  The same rule runs the other way at S19. The value sum was deliberately
  NOT a gemm call (contract DEVIATION 807) because it contracts over the key
  axis; its `dy` derivative contracts over `head_dim`, which IS path
  invariant, so the forward's most hand-written seam has the backward that
  routes most cleanly. DEVIATION 1405.

Eleven of twenty seams route. Six are new folds, four of which are the same
shape: serial ascending over one axis, seeded `+0.0`, one
`identical_mul_add` per term. That is contract S1's shape and contract S19's
shape and this file introduces no seventh.

THERE ARE NO SABOTAGE ARMS IN THIS FILE, ON PURPOSE
-----------------------------------------------------
`transformer_oracle.mojo` has none either. The oracle is the ANSWER; the
device spelling is what gets sabotaged and the check file is what arms it.
All twenty-two arms of plan section 6.3 live in
`transformer/checks/transformer_backward.mojo`, except two which are the
GEMM lane's own (`SAB_BWD_UNTRANSPOSED`, `SAB_BWD_OPERAND_ORDER`) and reach
this file through the routing functions it calls -- which is deliberate, and
is DEVIATION 1425.

DEVIATION 1425: THIS FILE CALLS THE GEMM LANE'S ROUTING TABLE
----------------------------------------------------------------
Every weight gradient and every projection activation gradient goes through
`gemm/checks/gemm_backward.mojo::gemm_backward_a_call` and
`::gemm_backward_b_call` rather than through a re-typed copy of their
six-row table. Two reasons and the second is load bearing. First, a lane
that retypes a transpose table gets a wrong answer that is bit identical on
three vendors. Second, calling them means this lane's routed gradients
INHERIT gemm gates G1 and G2 and inherit both `SAB_BWD_*` arms through a new
entry point -- which is exactly what `gemm/IDENTICAL_BACKWARD_PLAN.md`
section 5 asks for when it says the forward sabotages must be shown to fail
through the backward entry points as well as the forward ones.

THIS LANE'S DEVIATIONS ARE 1400-1449. The ones spelled in this file:
  1401 the organizing rule (route a config-length fold, pin a launch-length one)
  1402 `dq` pinned, gemm v1 REFUSED
  1403 `dk` pinned, gemm v1 REFUSED
  1404 `dv` pinned, the mirror of contract DEVIATION 807
  1405 the attention-weight gradient ROUTED at `k = head_dim`
  1406 the softmax backward as the CLOSED FORM, the decomposed graph REFUSED
  1407 the `z` fold, serial ascending over the ABSOLUTE key index, FUSED
  1408 the RMSNorm backward's `c` fold, contract S1's shape
  1409 the RMSNorm backward's elementwise tail, node by node
  1410 the norm weight gradients as a gemm v1 `OP_NN` against a ones vector
  1411 the SiLU derivative's five roundings, `sigmoid` recomputed
  1412 the RoPE backward as the transposed rotation
  1413 the fan-in accumulations, NO `+0.0` seed, FORWARD-USE order
  1414 the mask backward as an exact identity
  1415 the scale backward on the finished `dS`
  1417 `dk_cache` / `dv_cache` over the full `[0, S)` as a HANDOFF
  1418 the 37-stage backward card
  1420 the recomputation set (`rstd`, `inner`, `sigmoid`)
  1421 the saved-stage set
  1423 the incoming gradient refused nonfinite BY BITS
  1424 the GQA head-group fold order in `dk` and `dv`
  1425 this file calls the GEMM lane's routing table

`pinned_mul` DOES NOT APPEAR BY THAT NAME. `transformer_oracle.mojo` resolved
contract DEVIATION 816 by calling `checks/numerics.mojo::identical_mul`
(DEVIATION 826, the canonical home) and this file follows it exactly, for
the same reason: a fourth copy of one multiply is four chances to be edited
apart. A reader matching the contract's seam table against this code should
read every `identical_mul` as the contract's `pinned_mul`.

RISK: WHAT IS LEAST LIKELY TO COMPILE
---------------------------------------
  * `gemm_backward_a_call`'s FIVE-ELEMENT `Tuple` return and the `call[4]`
    indexing below. That lane's own open-questions section flags the tuple
    as one of the two things most likely to need a syntax adjustment;
    `gemm_backward.mojo` compiles too. If it moves, the two `_gemm_bwd_*`
    helpers here are the only two call sites.
  * `mut` arguments that are struct fields. This file follows
    `transformer_oracle.mojo`'s rule -- LOCALS THEN ASSIGN, never
    `mut st.field` at a call site -- because that is the shape which
    produces the `MutUntrackedOrigin` versus `MutAnyOrigin` unification
    errors this repository has lost time to.
  * `[[mojo-list-copy]]`: `List[Float32]` is not implicitly copyable, so
    every read of a stage into a new binding is `.copy()` and every handoff
    is `^`.
"""

from std.memory import bitcast

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_backward import (
    BWD_DC_LEFT,
    gemm_backward_a_call,
    gemm_backward_b_call,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, gemm_oracle
from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
)
from transformer.checks.transformer_fixture import (
    RMS_EPS,
    TransformerDims,
    TransformerWeights,
    attention_scale,
)
from transformer.checks.transformer_oracle import (
    RopeTable,
    TransformerStages,
    refuse_nonfinite,
)


# ===========================================================================
# THE TWO EXACT SCALINGS OF THE RMSNORM BACKWARD (DEVIATION 1409)
#
# Both are exact powers of two, so neither introduces a rounding. Neither is
# BIT INERT, which is why both are SPELLED rather than elided: contract S8
# drops the `* attention_scaling` because it is exactly `1.0` and inert on
# every input including both zero signs, and that argument does not transfer
# to a value that changes the number.
#
# They also do not cancel. `-0.5 * z` followed by `* 2.0` is `-z` for every
# finite `z` whose halving stays out of the flush region, and the two-step
# spelling is kept because the reference's autograd GRAPH has two steps:
# `rsqrt_backward` contributes the `-0.5` and `pow(2)` backward contributes
# the `2`.
# ===========================================================================

comptime BWD_NEG_HALF: Float32 = -0.5
comptime BWD_TWO: Float32 = 2.0

comptime TRANSFORMER_BACKWARD_STAGE_COUNT = 37


# ===========================================================================
# THE ROUTING DOOR (DEVIATION 1425)
#
# Two functions, and they are the ONLY place in this lane where a backward
# GEMM shape is decided. Every projection in this profile is a forward
# `OP_NT` call, so every route below reads the `OP_NT` row of the plan's
# section 2.2 table -- but the `op` argument is passed through rather than
# assumed, because a router that assumes its own input is a router that
# cannot be sabotaged.
# ===========================================================================


def _gemm_bwd_a(
    dc: List[Float32],
    other: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) raises -> List[Float32]:
    """`dA` for a forward `C = op(A) . op(B)` at `(m, n, k)`.

    `other` is the forward's `B` operand, unchanged and untransposed. The
    op, the backward shape and the operand SIDE all come from
    `gemm_backward_a_call`; nothing here decides any of the three.

    **The side flag is honored rather than assumed.** Two of the GEMM
    lane's six rows put `dC` on the RIGHT, and although neither of them is
    reached by this profile (every forward call here is `OP_NT`, whose `dA`
    and `dB` both put `dC` left), honoring it is what lets
    `SAB_BWD_OPERAND_ORDER` reach this file at all. Plan section 6.3
    predicts that arm is INERT through these entry points for exactly that
    reason, and an inert arm that is WIRED is worth more than one that is
    not, because the next profile may not be all-`OP_NT`.
    """
    var call = gemm_backward_a_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        return gemm_oracle(dc, other, call[0], call[1], call[2], call[3])
    return gemm_oracle(other, dc, call[0], call[1], call[2], call[3])


def _gemm_bwd_b(
    dc: List[Float32],
    other: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) raises -> List[Float32]:
    """`dB` for a forward `C = op(A) . op(B)` at `(m, n, k)`.

    `other` is the forward's `A` operand. **This is the call whose `k'` is
    the TOKEN COUNT** for every projection in this block
    (`gemm_backward_b_call`'s own docstring calls that the GEMM lane's
    finding), so it is the call whose partition, whose plan and whose bits
    move when the batch size moves. Nothing about that threatens
    cross-vendor identity and everything about it means a training run has
    to declare its batch and chunk schedule as part of its numerical
    specification. Plan section 5.2 and 5.4.
    """
    var call = gemm_backward_b_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        return gemm_oracle(dc, other, call[0], call[1], call[2], call[3])
    return gemm_oracle(other, dc, call[0], call[1], call[2], call[3])


def _ones(n: Int) -> List[Float32]:
    """`n` entries of exactly `Float32(1.0)`, for the norm weight gradients.

    DEVIATION 1410 and `gemm/IDENTICAL_BACKWARD_PLAN.md` section 3.3's
    trick. **A wrong value here is a wrong gradient with no symptom**,
    because any vector produces a plausible weighted column sum, so the gate
    for it is not "the buffer was allocated" -- it is the exact-integer arm
    of plan clause (f), which fails if the entries are anything else."""
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


# ===========================================================================
# THE STAGES (plan section 6.1, DEVIATION 1418)
# ===========================================================================


def backward_stage_tag(i: Int) raises -> String:
    """The thirty-seven backward tags, in the plan's order.

    **THE STRINGS AND THE ORDER ARE BOTH PART OF THE INSTRUMENT.**
    `tools/identity_trace_diff.py` aligns two traces by their TAG SEQUENCES
    before it compares a single hash, so a renamed tag or a reordered pair
    does not produce a smaller diff -- it produces a WRONG ALIGNMENT that
    pairs one run's stage against another run's different stage and reports
    a plausible answer. `core/identity_trace.mojo`'s header calls that the
    worst thing the instrument can do.

    A device transcription must emit exactly these, in exactly this order,
    with its own driver prefix. Nothing may be inserted, nothing skipped
    conditionally, and a stage that is empty for a given shape must still be
    recorded.

    Stage 0 is the INCOMING GRADIENT and it is on the card deliberately.
    Two cards whose inputs differ are diffing their fixtures;
    `gemm/IDENTICAL_BACKWARD_PLAN.md`'s G9 makes the same point in the same
    words. **Compare stage 0 before comparing any other stage.**
    """
    if i == 0:
        return String("bwd.in.d_residual2")
    if i == 1:
        return String("bwd.d_down_proj_out")
    if i == 2:
        return String("bwd.d_mlp_gated")
    if i == 3:
        return String("bwd.dW_down")
    if i == 4:
        return String("bwd.d_silu_out")
    if i == 5:
        return String("bwd.d_up_proj_out")
    if i == 6:
        return String("bwd.d_gate_proj_out")
    if i == 7:
        return String("bwd.dW_gate")
    if i == 8:
        return String("bwd.dW_up")
    if i == 9:
        return String("bwd.d_norm2_out")
    if i == 10:
        return String("bwd.norm2.dot")
    if i == 11:
        return String("bwd.dW_norm2")
    if i == 12:
        return String("bwd.norm2.dx")
    if i == 13:
        return String("bwd.d_residual1")
    if i == 14:
        return String("bwd.d_o_proj_out")
    if i == 15:
        return String("bwd.d_attn_ctx")
    if i == 16:
        return String("bwd.dW_o")
    if i == 17:
        return String("bwd.d_attn_weights")
    if i == 18:
        return String("bwd.attn.zdot")
    if i == 19:
        return String("bwd.d_attn_masked")
    if i == 20:
        return String("bwd.d_attn_scores")
    if i == 21:
        return String("bwd.d_qk_cell")
    if i == 22:
        return String("bwd.d_q_rope")
    if i == 23:
        return String("bwd.d_k_cache")
    if i == 24:
        return String("bwd.d_v_cache")
    if i == 25:
        return String("bwd.d_k_rope")
    if i == 26:
        return String("bwd.d_v_proj_out")
    if i == 27:
        return String("bwd.d_q_proj_out")
    if i == 28:
        return String("bwd.d_k_proj_out")
    if i == 29:
        return String("bwd.dW_q")
    if i == 30:
        return String("bwd.dW_k")
    if i == 31:
        return String("bwd.dW_v")
    if i == 32:
        return String("bwd.d_norm1_out")
    if i == 33:
        return String("bwd.norm1.dot")
    if i == 34:
        return String("bwd.dW_norm1")
    if i == 35:
        return String("bwd.norm1.dx")
    if i == 36:
        return String("bwd.d_x")
    raise Error(
        String("transformer backward: no stage ")
        + String(i)
        + " (there are "
        + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
        + ", plan section 6.1)"
    )


struct TransformerBackwardStages(Movable):
    """Every recorded stage of one backward call, in the card's order.

    Layouts, with `M = B * L` token-major, `S = pos0 + L` the kv length,
    `qw = n_heads*head_dim`, `kw = n_kv*head_dim`.

        in_d_residual2   [M, dm]          the incoming gradient, as given
        d_down_proj_out  [M, dm]
        d_mlp_gated      [M, inter]
        dw_down          [dm, inter]      W's own shape, not its transpose
        d_silu_out       [M, inter]
        d_up_proj_out    [M, inter]
        d_gate_proj_out  [M, inter]
        dw_gate          [inter, dm]
        dw_up            [inter, dm]
        d_norm2_out      [M, dm]
        norm2_dot        [M]
        dw_norm2         [dm]
        norm2_dx         [M, dm]
        d_residual1      [M, dm]
        d_o_proj_out     [M, dm]
        d_attn_ctx       [M, qw]
        dw_o             [dm, qw]
        d_attn_weights   [B, nh, L, S]
        attn_zdot        [B, nh, L]
        d_attn_masked    [B, nh, L, S]
        d_attn_scores    [B, nh, L, S]
        d_qk_cell        [B, nh, L, S]
        d_q_rope         [M, qw]
        d_k_cache        [B, nkv, S, hd]  the FULL used prefix, DEVIATION 1417
        d_v_cache        [B, nkv, S, hd]  the FULL used prefix
        d_k_rope         [M, kw]          the [pos0, S) slice
        d_v_proj_out     [M, kw]          the [pos0, S) slice
        d_q_proj_out     [M, qw]
        d_k_proj_out     [M, kw]
        dw_q             [qw, dm]
        dw_k             [kw, dm]
        dw_v             [kw, dm]
        d_norm1_out      [M, dm]
        norm1_dot        [M]
        dw_norm1         [dm]
        norm1_dx         [M, dm]
        d_x              [M, dm]          THE OUTPUT

    **`d_attn_scores` is recorded even though it is bitwise equal to
    `d_attn_masked` by construction** (the mask backward is the exact
    identity, DEVIATION 1414). That is the reason: an unrecorded identity is
    an unchecked one, and `B13_MASK_ZEROES_GRAD` is the arm that breaks it,
    at the masked cells whose `dS` is `-0.0`.

    **`norm1_dot` and `norm2_dot` are recorded** for the reason the forward
    records `norm1.sumsq`: a fold whose only evidence is the value it feeds
    cannot be localized, and the forward lane's scar is that thirteen moved
    stages were absorbed by a residual add while an output-only gate called
    the sabotage inert.

    **`d_k_cache` and `d_v_cache` hold `[0, S)` and NOT the allocation**
    (`core/identity_trace.mojo` rule 3). Their `[0, pos0)` half is the
    gradient owed to tokens from EARLIER calls and it is a HANDOFF, not a
    claim: assembling a multi-call backward is out of this lane's scope.

    FOUR `[B, nh, L, S]` BUFFERS is what the eager path costs on the
    backward, against the forward's six, so a backward call's peak footprint
    is roughly 1.7x a forward call's on the dominant term, plus the saved
    forward stages held for the interval. Plan section 7 prices it. It is
    also what makes every attention seam separately recordable, and a lane
    whose whole instrument is the per-stage card should not begin by fusing
    the stages away.
    """

    var in_d_residual2: List[Float32]
    var d_down_proj_out: List[Float32]
    var d_mlp_gated: List[Float32]
    var dw_down: List[Float32]
    var d_silu_out: List[Float32]
    var d_up_proj_out: List[Float32]
    var d_gate_proj_out: List[Float32]
    var dw_gate: List[Float32]
    var dw_up: List[Float32]
    var d_norm2_out: List[Float32]
    var norm2_dot: List[Float32]
    var dw_norm2: List[Float32]
    var norm2_dx: List[Float32]
    var d_residual1: List[Float32]
    var d_o_proj_out: List[Float32]
    var d_attn_ctx: List[Float32]
    var dw_o: List[Float32]
    var d_attn_weights: List[Float32]
    var attn_zdot: List[Float32]
    var d_attn_masked: List[Float32]
    var d_attn_scores: List[Float32]
    var d_qk_cell: List[Float32]
    var d_q_rope: List[Float32]
    var d_k_cache: List[Float32]
    var d_v_cache: List[Float32]
    var d_k_rope: List[Float32]
    var d_v_proj_out: List[Float32]
    var d_q_proj_out: List[Float32]
    var d_k_proj_out: List[Float32]
    var dw_q: List[Float32]
    var dw_k: List[Float32]
    var dw_v: List[Float32]
    var d_norm1_out: List[Float32]
    var norm1_dot: List[Float32]
    var dw_norm1: List[Float32]
    var norm1_dx: List[Float32]
    var d_x: List[Float32]

    def __init__(out self):
        self.in_d_residual2 = List[Float32]()
        self.d_down_proj_out = List[Float32]()
        self.d_mlp_gated = List[Float32]()
        self.dw_down = List[Float32]()
        self.d_silu_out = List[Float32]()
        self.d_up_proj_out = List[Float32]()
        self.d_gate_proj_out = List[Float32]()
        self.dw_gate = List[Float32]()
        self.dw_up = List[Float32]()
        self.d_norm2_out = List[Float32]()
        self.norm2_dot = List[Float32]()
        self.dw_norm2 = List[Float32]()
        self.norm2_dx = List[Float32]()
        self.d_residual1 = List[Float32]()
        self.d_o_proj_out = List[Float32]()
        self.d_attn_ctx = List[Float32]()
        self.dw_o = List[Float32]()
        self.d_attn_weights = List[Float32]()
        self.attn_zdot = List[Float32]()
        self.d_attn_masked = List[Float32]()
        self.d_attn_scores = List[Float32]()
        self.d_qk_cell = List[Float32]()
        self.d_q_rope = List[Float32]()
        self.d_k_cache = List[Float32]()
        self.d_v_cache = List[Float32]()
        self.d_k_rope = List[Float32]()
        self.d_v_proj_out = List[Float32]()
        self.d_q_proj_out = List[Float32]()
        self.d_k_proj_out = List[Float32]()
        self.dw_q = List[Float32]()
        self.dw_k = List[Float32]()
        self.dw_v = List[Float32]()
        self.d_norm1_out = List[Float32]()
        self.norm1_dot = List[Float32]()
        self.dw_norm1 = List[Float32]()
        self.norm1_dx = List[Float32]()
        self.d_x = List[Float32]()


def backward_oracle_dump(
    st: TransformerBackwardStages,
) -> List[List[Float32]]:
    """The thirty-seven stages in CARD ORDER, index-aligned with
    `backward_stage_tag`.

    One function pairs tags with values so that the pairing exists in ONE
    place. `transformer_oracle.mojo::oracle_dump` is the same shape for the
    same reason, and a device dump must produce a list of the same length in
    the same order -- the check file's FIRST assertion should be that the
    two lengths agree, before any hash is compared."""
    var out = List[List[Float32]]()
    out.append(st.in_d_residual2.copy())
    out.append(st.d_down_proj_out.copy())
    out.append(st.d_mlp_gated.copy())
    out.append(st.dw_down.copy())
    out.append(st.d_silu_out.copy())
    out.append(st.d_up_proj_out.copy())
    out.append(st.d_gate_proj_out.copy())
    out.append(st.dw_gate.copy())
    out.append(st.dw_up.copy())
    out.append(st.d_norm2_out.copy())
    out.append(st.norm2_dot.copy())
    out.append(st.dw_norm2.copy())
    out.append(st.norm2_dx.copy())
    out.append(st.d_residual1.copy())
    out.append(st.d_o_proj_out.copy())
    out.append(st.d_attn_ctx.copy())
    out.append(st.dw_o.copy())
    out.append(st.d_attn_weights.copy())
    out.append(st.attn_zdot.copy())
    out.append(st.d_attn_masked.copy())
    out.append(st.d_attn_scores.copy())
    out.append(st.d_qk_cell.copy())
    out.append(st.d_q_rope.copy())
    out.append(st.d_k_cache.copy())
    out.append(st.d_v_cache.copy())
    out.append(st.d_k_rope.copy())
    out.append(st.d_v_proj_out.copy())
    out.append(st.d_q_proj_out.copy())
    out.append(st.d_k_proj_out.copy())
    out.append(st.dw_q.copy())
    out.append(st.dw_k.copy())
    out.append(st.dw_v.copy())
    out.append(st.d_norm1_out.copy())
    out.append(st.norm1_dot.copy())
    out.append(st.dw_norm1.copy())
    out.append(st.norm1_dx.copy())
    out.append(st.d_x.copy())
    return out^


def record_transformer_backward_card(
    mut trace: IdentityTrace, prefix: String, st: TransformerBackwardStages
) raises:
    """Write all thirty-seven stages to an identity trace, in card order.

    `prefix` becomes `prefix + "." + tag`, or the bare tag when empty. Tags
    must be UNIQUE WITHIN A TRACE and `core/identity_trace.mojo` enforces
    it, so a driver that records a FORWARD card and a BACKWARD card into one
    trace must give them different prefixes. This function does not check
    that, because the trace already does and its error message is better.

    The tags here all begin `bwd.`, so they cannot collide with the
    forward's thirty even under an empty prefix. That is deliberate: a
    single trace holding both cards is the shape a check file wants, because
    the forward stages ARE the backward's saved inputs and a divergence in
    one is diagnosed by looking at the other.

    `[[mojo-len-string]]`: `len(String)` is unsupported in this toolchain;
    `byte_length()` is the spelling."""
    var values = backward_oracle_dump(st)
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        var tag = backward_stage_tag(i)
        if prefix.byte_length() > 0:
            tag = prefix + String(".") + tag
        # A LOCAL COPY rather than `values[i]` at the call site, keeping a
        # mutable-argument borrow off a list element -- the shape that
        # produced this repository's origin-unification errors.
        var one = values[i].copy()
        trace.record_list_f32(tag, one)
        _ = one^


# ===========================================================================
# THE NEW SEAMS
# ===========================================================================


def rms_norm_backward_into(
    dy: List[Float32],
    x: List[Float32],
    wnorm: List[Float32],
    sumsq: List[Float32],
    m: Int,
    dm: Int,
    mut dot: List[Float32],
    mut dx: List[Float32],
    mut dprod: List[Float32],
) raises:
    """The backward of seams S1 through S4. DEVIATIONS 1408, 1409, 1420.

    **THE REFERENCE HAS NO FUSED RMSNORM BACKWARD TO MIRROR.**
    `LlamaRMSNorm.forward` (modeling_llama.py:62-67, re-read in the checkout
    on 2026-08-25) is six lines of ordinary tensor ops, so autograd
    differentiates the ELEMENTWISE GRAPH node by node. This function mirrors
    that graph rather than the algebraically-collapsed formula, for the same
    reason contract S10 refuses an fma: the reference rounds where it
    rounds, and being righter than the reference is not the goal.

    The forward graph, per row:

        v = x.pow(2)  ->  mn = v.mean(-1)  ->  a = mn + eps
        r = rsqrt(a)  ->  h = x * r        ->  y = w * h

    and the pinned backward, per row, in this order:

        rstd RECOMPUTED from the SAVED sumsq, not saved and not re-folded
        dh_j = pinned_mul(dy_j, w_j)                              PRODUCT
        c    = ftz(fma(dh_j, x_j, c)), ASCENDING j from +0.0      FUSED FOLD
        r2   = pinned_mul(rstd, rstd)
        r3   = pinned_mul(r2, rstd)
        cr3  = pinned_mul(c, r3)
        da   = pinned_mul(-0.5, cr3)          rsqrt backward, -0.5 * dr * r^3
        dv   = identical_div(da, d_model)     mean backward, ONE division
        dx1_j = pinned_mul(dh_j, rstd)        h = x*r, the x branch
        tx_j  = pinned_mul(2.0, x_j)          pow(2) backward's 2*x
        dx2_j = pinned_mul(dv, tx_j)
        dx_j  = ftz(ftz(dx1_j) + ftz(dx2_j))                      UNFUSED ADD

    and, for the weight gradient, the per-cell product the caller then feeds
    to a gemm v1 `OP_NN` against a ones vector:

        inner_j  = pinned_mul(x_j, rstd)      a recompute of forward S3
        dprod_j  = pinned_mul(dy_j, inner_j)

    FOUR DECISIONS, EACH OF WHICH COULD HAVE GONE THE OTHER WAY.

    **(a) `c` IS SERIAL ASCENDING, FUSED, FROM `+0.0`** -- contract S1's
    shape unchanged. `d_model` is a CONFIGURATION quantity, so unlike the
    softmax's `z` fold or the `dq` chain there is no path-invariance
    argument forcing this; the reasons are that it gives the block ONE fold
    shape instead of two, that one thread owns one token row so no launch
    geometry can reorder it, and that a reader can put contract S1 beside
    it. `core/pinned_reduce.mojo::pinned_block_sum` is REFUSED here for the
    reasons contract 5.3 refuses it at S17: it is a halving tree, it pairs
    by STRIDE, and it is a different sum that would pass every
    launch-invariance gate because a tree is perfectly launch invariant.
    FUSED because S1 is fused and because both are sums of products.
    Sabotages `B01_DOT_UNFUSED` and `B01_DOT_DESCENDING`, both INERT at
    `d_model == 1` and the second NOT inert at `d_model == 2` -- an fma
    keeps the second product exact, so a two-term fma chain is order
    dependent where a two-term ADD chain is not.

    **(b) `rstd` IS RECOMPUTED FROM THE SAVED `sumsq`.** `identical_div` and
    `identical_rsqrt` are pure functions of bits the card already holds, so
    the recompute is bit exact by construction. Saving `rstd` would add a
    stage the forward does not have. Recomputing the SUM OF SQUARES instead
    of reading it is sabotage `B_RSTD_RECOMPUTE_DESCENDING`, whose ASCENDING
    half must move NOTHING -- and that half is what turns "a recompute is
    exact" from a belief into a checked statement.

    **(c) `-0.5` AND `2.0` ARE SPELLED.** Both are exact and neither is
    inert. See `BWD_NEG_HALF` above.

    **(d) `dprod` EXISTS SO THE WEIGHT GRADIENT CAN BE A GEMM**
    (DEVIATION 1410). `dW[j] = sum_t dy_tj * inner_tj` is a Hadamard then a
    reduce, so the obvious reading is that it needs a new pinned fold. It
    does not, for the reason `gemm/IDENTICAL_BACKWARD_PLAN.md` section 3.3
    gives about `db`: with the product materialized, the reduction is
    `ones . dprod`, an `OP_NN` at `(1, dm, M)` whose leaf is
    `fma(ftz(1.0), ftz(p), acc)`, the contract's ascending flushed chain
    inside a leaf and the contract's balanced tree across leaves, with
    nothing new to certify. **THE COST THIS LANE PAYS THAT THE GEMM LANE DID
    NOT**: two roundings per term instead of one. `db[j] = sum dC[i,j]` had
    no product to round; this does, so the routed form rounds the product
    and then rounds the add where a hand-written `fma(dy, inner, acc)` would
    round once. It is pinned anyway, because one arithmetic under one
    certificate beats one rounding per term, and because the hand form would
    be a SIXTH new fold in a lane that has five. Recorded, not hidden.

    `dot`, `dx` and `dprod` are APPENDED to in row-major order and are
    expected empty on entry."""
    for t in range(m):
        # ---- rstd, recomputed. DEVIATION 1420. -------------------------
        var mean = ftz(identical_div(ftz(sumsq[t]), Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))

        # ---- dh and the c fold, one ascending pass. ---------------------
        var dh = List[Float32]()
        var c = Float32(0.0)
        for j in range(dm):
            var dhj = ftz(identical_mul(ftz(dy[t * dm + j]), ftz(wnorm[j])))
            dh.append(dhj)
            c = ftz(identical_mul_add(dhj, ftz(x[t * dm + j]), c))
        c = ftz(c)
        dot.append(c)

        # ---- the rsqrt / mean tail, once per row. -----------------------
        var r2 = ftz(identical_mul(rstd, rstd))
        var r3 = ftz(identical_mul(r2, rstd))
        var cr3 = ftz(identical_mul(c, r3))
        var da = ftz(identical_mul(BWD_NEG_HALF, cr3))
        var dv = ftz(identical_div(da, Float32(dm)))

        # ---- the two x branches, and the weight-gradient product. -------
        for j in range(dm):
            var xj = ftz(x[t * dm + j])
            var dx1 = ftz(identical_mul(dh[j], rstd))
            var tx = ftz(identical_mul(BWD_TWO, xj))
            var dx2 = ftz(identical_mul(dv, tx))
            dx.append(ftz(ftz(dx1) + ftz(dx2)))
            var inner = ftz(identical_mul(xj, rstd))
            dprod.append(
                ftz(identical_mul(ftz(dy[t * dm + j]), inner))
            )
        _ = dh^


def silu_backward_into(
    dsi: List[Float32], g: List[Float32], n: Int, mut dg: List[Float32]
) raises:
    """The backward of seam S20. DEVIATION 1411.

    `silu(x) = x / (1 + exp(-x))`, ONE division (contract S20), and NOT
    `x * sigmoid(x)`. That forward refusal has a consequence here which is
    easy to miss and which is decision (a) below.

    Pinned, five roundings after the sigmoid, left to right, no fma:

        sg = identical_sigmoid(x)          DEVIATION 743, portable_sigmoidf
        r1 = ftz(1.0 - sg)                 SUBTRACT
        r2 = pinned_mul(x, r1)             PRODUCT
        r3 = ftz(1.0 + r2)                 UNFUSED ADD
        r4 = pinned_mul(sg, r3)            PRODUCT
        dg = pinned_mul(dsi, r4)           PRODUCT

    **(a) `sg` IS RECOMPUTED FROM `gate_proj.out`, NEVER RECONSTRUCTED FROM
    `silu.out`.** `silu(x) = x * sigmoid(x)` is true in the reals and FALSE
    in Float32 under this profile: the forward is one division and `x * sg`
    is a division followed by a product. So `sg = silu_out / x` is a
    different number, it is `0/0` at `x = 0`, and it costs a division to get
    a worse answer. `portable_sigmoidf` and `portable_siluf` share the SAME
    `d = portable_expf(-x) + 1.0` and differ only in the numerator, so the
    recomputed sigmoid is exactly `1/d` against the forward's `x/d` -- the
    cleanest relationship available. Sabotage `B20_SIGMOID_FROM_SILU`,
    **INERT for every `x` at or above about 17**, where `1 + exp(-x)` rounds
    to exactly `1.0`, `silu(x) == x` exactly, and the reconstruction returns
    exactly `1.0`. A fixture whose gate activations are all large is blind
    to that arm.

    **(b) THE ASSOCIATION IS `sg * (1 + x*(1-sg))`**, not
    `sg + x*sg*(1-sg)`. Equal in the reals, different in the last bit on
    ordinary inputs, four operations instead of five. Sabotage
    `B20_SILU_DERIV_ALT_ASSOC`, INERT wherever `sg` is exactly `1.0` and at
    `x == 0`.

    **(c) `1 + x*(1-sg)` IS UNFUSED.** An `fma(x, r1, 1.0)` is one rounding
    where this is two. Sabotage `B20_SILU_DERIV_FUSED`, INERT wherever the
    product is exactly representable.

    **THE REFERENCE'S OWN SPELLING COULD NOT BE VERIFIED.** There is no
    PyTorch checkout in `/Users/andrewhendel/CascadeProjects/upstream/` --
    verified again on 2026-08-25, the directory holds `cccl`, `cuml`,
    `cuvs`, `curand-headers`, `mamba`, `modular`, `raft`, `scikit-learn` and
    `transformers`, and no torch -- so ATen's `silu_backward` was not read.
    **This is a stated gap, not a decision made on evidence**, it is
    contract section 5.4's gap in a second place, and it is the second thing
    to check when a checkout lands."""
    for i in range(n):
        var x = ftz(g[i])
        var sg = ftz(identical_sigmoid(x))
        var r1 = ftz(ftz(Float32(1.0)) - ftz(sg))
        var r2 = ftz(identical_mul(x, r1))
        var r3 = ftz(ftz(Float32(1.0)) + ftz(r2))
        var r4 = ftz(identical_mul(sg, r3))
        dg.append(ftz(identical_mul(ftz(dsi[i]), r4)))


def rope_backward_into(
    dout: List[Float32],
    n_head: Int,
    head_dim: Int,
    b: Int,
    l: Int,
    pos0: Int,
    rope: RopeTable,
    mut out: List[Float32],
) raises:
    """The backward of seams S9 and S10, the TRANSPOSED rotation.
    DEVIATION 1412.

    The forward acts on the pair `(a_i, a_{i+half})` at table column
    `ci = i` as a rotation:

        out_i        = a_i * c - a_{i+half} * s        M = [[c, -s],
        out_{i+half} = a_{i+half} * c + a_i * s             [s,  c]]

    `M` is orthogonal so the exact derivative is `M^T`:

        da_i        = dout_i * c + dout_{i+half} * s
        da_{i+half} = dout_{i+half} * c - dout_i * s

    which, written in the forward's own `rotate_half` idiom, is THE SAME
    CODE WITH THE NEGATION MOVED FROM THE LOWER HALF TO THE UPPER HALF:

        forward   j <  half: rot = -x[j+half]   j >= half: rot = +x[j-half]
        backward  j <  half: rot = +d[j+half]   j >= half: rot = -d[j-half]

    **EVERYTHING ELSE IS IDENTICAL.** The same `ci = j mod half` column, the
    same `cos` and `sin` rows at the same ABSOLUTE position, two
    `pinned_mul` calls, one UNFUSED add. Same three roundings. Contract
    DEVIATION 811's refusal of an fma applies here for the reason it applied
    there: an fma is one rounding where the structure has three, and it is
    the natural thing for a kernel author to write. Sabotage
    `B10_ROPE_BWD_FUSED`.

    So the rounding BUDGET is inherited and the SPELLING is new, and the one
    thing this lane must pin is the SIGN CONVENTION. It is one character in
    two branches and getting it wrong produces a plausible,
    correctly-shaped, wrong gradient that is bit identical on three vendors.
    Sabotage `B09_ROPE_TRANSPOSE_SIGN`, **INERT AT ABSOLUTE POSITION 0**
    where `sin` is exactly `+0.0` and `cos` exactly `1.0` -- per CELL and
    not per fixture, so the gate must COUNT moved cells and RAISE if the
    count equals the position-0 population.

    **ONE SENTENCE THAT SHOULD STOP A READER EXPECTING THE WRONG THING.**
    This is the exact transpose of the EXACT rotation, spelled with the
    forward's rounding budget. It is NOT the numerical adjoint of the
    ROUNDED forward map, because a rounded map has no adjoint.
    `rope_backward(rope(x))` does not return `x` and no clause anywhere in
    this lane says it does.

    THE POSITION IS ABSOLUTE, ALWAYS, and for the same reason the forward's
    is: `pos0 + li` where `pos0` is the cache's `used` count before this
    call's append. An implementation indexing by the local index agrees on
    every single-call fixture and disagrees on every decode step."""
    var half = head_dim // 2
    var width = n_head * head_dim
    var m = b * l
    for t in range(m):
        var li = t % l
        var p = pos0 + li
        if p < 0 or p >= rope.positions:
            raise Error(
                String("transformer backward: absolute position ")
                + String(p)
                + " outside the rotary table of "
                + String(rope.positions)
                + " positions REFUSED (the table is a configuration"
                + " quantity, so a call that overruns it is a"
                + " misconfiguration and not something to grow into)"
            )
        for h in range(n_head):
            var base = t * width + h * head_dim
            for j in range(head_dim):
                var ci: Int
                var rot: Float32
                # THE ONE CHARACTER. The forward negates the UPPER partner
                # for the lower half; the transpose negates the LOWER
                # partner for the upper half.
                if j < half:
                    ci = j
                    rot = ftz(dout[base + j + half])
                else:
                    ci = j - half
                    rot = -ftz(dout[base + j - half])
                var c = ftz(rope.cos[p * half + ci])
                var s = ftz(rope.sin[p * half + ci])
                var pa = ftz(identical_mul(ftz(dout[base + j]), c))
                var pb = ftz(identical_mul(rot, s))
                out.append(ftz(ftz(pa) + ftz(pb)))


def softmax_backward_into(
    dy: List[Float32],
    y: List[Float32],
    b: Int,
    nh: Int,
    l: Int,
    s: Int,
    mut zdot: List[Float32],
    mut ds: List[Float32],
) raises:
    """The backward of seams S14 through S18, as ONE closed form.
    DEVIATIONS 1406 and 1407.

        z_row = sum over j ASCENDING, ABSOLUTE key index, from +0.0
                z = ftz(fma(dy_j, y_j, z))                        FUSED
        dS_j  = pinned_mul(y_j, ftz(dy_j - z))       SUBTRACT then PRODUCT

    **THERE IS NO MAX BACKWARD, NO EXP BACKWARD AND NO DIVISION BACKWARD,
    AND REFUSING TO WRITE THEM IS A DECISION WITH A NAME** (DEVIATION 1406).

    `softmax` is ONE autograd node in the reference, not a graph. The
    evidence is in the checkout:
    `transformers/src/transformers/pytorch_utils.py:50-58` defines
    `softmax_backward_data(parent, grad_output, output)` whose body is
    `from torch import _softmax_backward_data; return _softmax_backward_data(
    grad_output, output, parent.dim, output.dtype)`. A private ATen symbol
    taking `(grad_output, output, dim, dtype)` and NOT taking the input, the
    max, the exponentials or the denominator, is a closed-form derivative of
    the whole op. The reference never differentiates through `max`, `exp` or
    the division.

    **WRITING THE DECOMPOSED GRAPH WOULD BE A DIFFERENT ANSWER, AND WORSE IN
    A SPECIFIC WAY.** `y = softmax(s)` is invariant to the row maximum `m`,
    so `dL/dm` is analytically EXACTLY zero. Autograd does not know that: it
    computes a sum of terms that cancel in the reals and do NOT cancel in
    Float32, and then SCATTERS that nonzero residue onto whichever element
    the max selected. That scatter is unpinnable here.

      1. `identical_fmax` returns a VALUE, not an INDEX. Contract 5.1 pins
         the max as an order-free fold and explicitly leaves its TOPOLOGY
         free, because the operation is exactly associative. An argmax is
         not associative and its answer under ties depends on the fold
         shape, so a free topology and a defined argmax are incompatible.
      2. Ties are reachable: a masked row's tail is a run of identical
         `-FLT_MAX` values.
      3. `max(+0.0, -0.0)` is a MEASURED three-vendor split (IDENTITY_PATHS
         row 39, 2026-08-23: `-0.0` on Apple, `+0.0` on NVIDIA and AMD) and
         contract 5.1 says a row of attention scores reaches both zero signs
         easily.

    So the decomposed graph would move a numerically-nonzero quantity to an
    index no clause in this profile defines. REFUSED, and it happens to also
    be the reference's behavior, which is the only kind of correctness
    improvement this lane accepts. Sabotage `B18_SOFTMAX_DECOMPOSED`, INERT
    at `s == 1` where both forms give zero.

    **THE `z` FOLD FOLLOWS CONTRACT 5.3's ARGUMENT AND IT CARRIES, BUT IT
    NEEDED CHECKING RATHER THAN ASSUMING, BECAUSE THE TERMS ARE DIFFERENT
    TERMS.** At a masked cell `y_j` is exactly `+0.0` (contract 7.1) while
    `dy_j` is an ordinary nonzero number, so the term is
    `fma(dy_j, +0.0, acc) = acc + (+-0.0) = acc`, provided `acc` is not
    `-0.0`. A `+0.0`-seeded fma chain never holds `-0.0`: `fma` returns a
    negative zero only when the exact `a*b + acc` is a zero of negative
    sign, which under round-to-nearest needs BOTH addends to be `-0.0`
    (an exact cancellation of two nonzero opposites gives `+0.0`), and the
    seed forbids it. **So the masked tail is bitwise inert and `z` -- and
    every activation gradient downstream of it -- is independent of the kv
    length.** That is the theorem the backward's length-invariance clause
    rests on, and it is contract 7.1 pointed the other way.

    FUSED, matching S1 and S19, because it is a sum of products.
    `pinned_block_sum` is refused for the third time in this profile and for
    the same reason: it is a halving tree, it is perfectly launch invariant,
    and it is simply a different sum. Sabotages `B18_ZFOLD_UNFUSED` (INERT
    at `s == 1`) and `B18_ZFOLD_DESCENDING` (INERT at `s == 1` ONLY -- NOT
    at `s == 2`, because an fma keeps the second product exact).

    **WHAT CONTRACT 5.4's OPEN QUESTION COSTS HERE: NOTHING.** The closed
    form reads only `y` and `dy`, so if the forward's S18 were ever changed
    from a division to a reciprocal multiply, this function would not change
    at all -- it would be the derivative of a different forward. The gap
    does not compound, which is rare enough to be worth a sentence.

    `zdot` is appended `[B, nh, L]` and `ds` `[B, nh, L, S]`, both in the
    card's layout, both expected empty on entry."""
    for bb in range(b):
        for h in range(nh):
            for qi in range(l):
                var base = ((bb * nh + h) * l + qi) * s
                var z = Float32(0.0)
                for j in range(s):
                    z = ftz(
                        identical_mul_add(
                            ftz(dy[base + j]), ftz(y[base + j]), z
                        )
                    )
                z = ftz(z)
                zdot.append(z)
                for j in range(s):
                    var diff = ftz(ftz(dy[base + j]) - ftz(z))
                    ds.append(ftz(identical_mul(ftz(y[base + j]), diff)))


# ===========================================================================
# THE BLOCK BACKWARD
# ===========================================================================


def transformer_block_backward_oracle(
    w: TransformerWeights,
    fwd: TransformerStages,
    d_out: List[Float32],
    b: Int,
    l: Int,
    pos0: Int,
    rope: RopeTable,
) raises -> TransformerBackwardStages:
    """The gradient of ONE `LlamaDecoderLayer.forward` call, stage by stage.

    `fwd` is the SAVED forward stages of THIS SAME CALL, produced by
    `transformer_oracle.mojo::transformer_block_oracle`. `d_out` is
    `d(residual2.out)` at `[M, d_model]`, arriving from somewhere this lane
    does not specify -- exactly as `dC` arrives in `gemm_backward.mojo`.
    `pos0` is the KV cache's `used` count BEFORE the forward call's append,
    so `S = pos0 + l` and cache slot `j` is absolute position `j`.

    Returns the eleven parameter gradients, the input gradient `d_x`, and
    `d_k_cache` / `d_v_cache` over the FULL `[0, S)` range.

    **WHAT IS NOT READ, AND IT IS A FINDING** (DEVIATION 1421). The
    closed-form softmax backward reads only `attn.weights`. So
    `attn.scores`, `attn.masked`, `attn.max`, `attn.exp` and `attn.denom` --
    five materialized buffers the forward computed, four of them
    `[B, nh, L, S]` -- are never touched here. Neither is `q_proj.out` or
    `k_proj.out`, because the RoPE backward is a linear map that reads only
    the TABLE, so the pre-rotation activations are not on the backward path
    at all.

    **NO LOSS, NO OPTIMIZER, NO ACCUMULATION BUFFER, NO TAPE.** And no
    multi-call assembly: `d_k_cache`'s `[0, pos0)` half is the gradient owed
    to tokens from earlier calls and it is handed over complete rather than
    consumed (DEVIATION 1417).

    THE ORGANIZING RULE, applied mechanically and to nothing else
    (DEVIATION 1401):

    > Route a derivative fold through gemm v1 if and only if its contraction
    > LENGTH is a configuration quantity. Pin it as a serial ascending chain
    > seeded `+0.0` if its length is a launch quantity. Token counts appear
    > as `k'` in every WEIGHT gradient and there the rule does not apply,
    > because a weight gradient IS the sum over the batch and no spelling
    > makes it otherwise.

    THE HONEST CLAUSE ABOUT CHOPPING, because the forward's
    decode-equals-prefill does not transfer unchanged:

      * `dq`, and every activation gradient reachable only through it, is
        INDEPENDENT of the kv length and of how the sequence was chopped.
      * `d_k_cache`, `d_v_cache` and every WEIGHT gradient are SUMS OVER THE
        QUERIES IN THIS CALL, so a per-call backward computes a PARTIAL and
        the caller must add the partials. That is the transformer form of
        `gemm/IDENTICAL_BACKWARD_PLAN.md` section 3.2's finding that `dB`'s
        `k` is the token count.

    Nothing in this function reads `B` except as a loop bound, so batch
    composition invariance of the activation gradients is a property of the
    SHAPE of the loops rather than of a check that happens to pass."""
    var dims = w.dims.copy()
    dims.validate()
    var dm = dims.d_model
    var nh = dims.n_heads
    var nkv = dims.n_kv_heads
    var hd = dims.head_dim
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    var n_rep = dims.n_rep()
    var m = b * l
    var s = pos0 + l
    var cells = b * nh * l * s
    var st = TransformerBackwardStages()

    # ---- refusals, before ANY recorded stage --------------------------
    if l <= 0 or b <= 0 or pos0 < 0:
        raise Error(
            "transformer backward: B and L must be positive and pos0"
            " non-negative"
        )
    if len(d_out) != m * dm:
        raise Error(
            String("transformer backward: d_out has ")
            + String(len(d_out))
            + " elements and the shape [B, L, d_model] wants "
            + String(m * dm)
        )
    if len(fwd.residual2_out) != m * dm:
        raise Error(
            "transformer backward: the saved forward stages do not match"
            " (B, L); residual2.out has the wrong length"
        )
    if len(fwd.kv_v_cache) != b * nkv * s * hd:
        raise Error(
            String("transformer backward: kv.v_cache has ")
            + String(len(fwd.kv_v_cache))
            + " elements and pos0 + L = "
            + String(s)
            + " wants "
            + String(b * nkv * s * hd)
            + " REFUSED. A wrong pos0 is a wrong ABSOLUTE key index"
            + " everywhere, and it produces a plausible gradient."
        )
    if len(fwd.attn_weights) != cells:
        raise Error(
            "transformer backward: attn.weights does not match (B, nh, L, S)"
        )
    if rope.head_dim != hd:
        raise Error(
            "transformer backward: the rotary table does not match head_dim"
        )
    # DEVIATION 1423. The incoming gradient is an INPUT and gets the same
    # treatment contract section 8 gives every other input: refused BY BITS,
    # not by compares, because Metal FLUSHES COMPARE OPERANDS (IDENTITY_PATHS
    # row 49) so `v != v` is a test with two meanings across columns. NaN
    # payloads are vendor-shaped (row 39 measured three payloads for one IEEE
    # answer) and a certified stage may not contain one.
    #
    # **A GRADIENT IS EXACTLY WHERE NaNs APPEAR IN PRACTICE**, which makes
    # this refusal more likely to fire than the forward's and makes the
    # named error worth more.
    refuse_nonfinite("d_residual2", d_out)

    var ones_m = _ones(m)

    # =====================================================================
    # STAGE 0-1. S23, `r2 = r1 + dp`. Both branches take the incoming
    # gradient unchanged: an add's derivative is the identity in both
    # arguments and there is NO ROUNDING here. A copy, not a seam.
    # =====================================================================
    st.in_d_residual2 = d_out.copy()
    st.d_down_proj_out = d_out.copy()

    # =====================================================================
    # STAGE 2-3. `down_proj`: forward `OP_NT` at `(m, dm, inter)`.
    # ROUTED, both halves. `d_mlp_gated`'s k' is `dm` (a config quantity);
    # `dw_down`'s k' is `m` (the TOKEN COUNT), declared rather than defended.
    # =====================================================================
    st.d_mlp_gated = _gemm_bwd_a(
        st.d_down_proj_out, w.w_down, OP_NT, m, dm, inter
    )
    st.dw_down = _gemm_bwd_b(
        st.d_down_proj_out, fwd.mlp_gated, OP_NT, m, dm, inter
    )

    # =====================================================================
    # STAGE 4-5. S21, `gt = si * u`. Two products, ROUTING: `pinned_mul`
    # settles the fusion question by existing (DEVIATION 720), so this lane
    # chooses nothing.
    # =====================================================================
    for i in range(m * inter):
        var dgt = ftz(st.d_mlp_gated[i])
        st.d_silu_out.append(
            ftz(identical_mul(dgt, ftz(fwd.up_proj_out[i])))
        )
        st.d_up_proj_out.append(
            ftz(identical_mul(dgt, ftz(fwd.silu_out[i])))
        )

    # =====================================================================
    # STAGE 6. S20, the SiLU derivative. NEW ARITHMETIC. DEVIATION 1411.
    # =====================================================================
    var dgate = List[Float32]()
    silu_backward_into(
        st.d_silu_out, fwd.gate_proj_out, m * inter, dgate
    )
    st.d_gate_proj_out = dgate^

    # =====================================================================
    # STAGE 7-9. `gate_proj` and `up_proj`: forward `OP_NT` at
    # `(m, inter, dm)`. ROUTED, plus a TWO-TERM fan-in.
    #
    # A two-term fan-in has NO ORDER TO PIN, because IEEE addition is
    # bitwise commutative on every non-NaN input and NaN is refused at the
    # door. The clause is written down anyway, because "it does not matter"
    # is exactly the kind of sentence that turns out to matter.
    #
    # **NO `+0.0` SEED** (DEVIATION 1413). `+0.0 + x` equals `x` for every
    # value except `x = -0.0`, where it gives `+0.0` -- IDENTITY_PATHS row
    # 39 in one line -- so a seed LAUNDERS a negative-zero first term, and a
    # negative zero is reachable in a gradient (every masked attention cell
    # produces one). An autograd engine's `AccumulateGrad` installs the
    # first incoming gradient as the buffer rather than allocating zeros,
    # and this follows it. Sabotage `B_FANIN_ZERO_SEED`, which is PREDICTED
    # TO MOVE ZERO CELLS ON EVERY UNPLANTED FIXTURE and is therefore vacuous
    # without a planted `-0.0` -- said out loud so nobody fires it, sees
    # nothing, and deletes it.
    #
    # The accumulation is in FORWARD-USE ORDER: `gate_proj` is evaluated
    # before `up_proj` at LlamaMLP.forward :175.
    # =====================================================================
    st.dw_gate = _gemm_bwd_b(
        st.d_gate_proj_out, fwd.norm2_out, OP_NT, m, inter, dm
    )
    st.dw_up = _gemm_bwd_b(
        st.d_up_proj_out, fwd.norm2_out, OP_NT, m, inter, dm
    )
    var t_gate = _gemm_bwd_a(
        st.d_gate_proj_out, w.w_gate, OP_NT, m, inter, dm
    )
    var t_up = _gemm_bwd_a(st.d_up_proj_out, w.w_up, OP_NT, m, inter, dm)
    for i in range(m * dm):
        st.d_norm2_out.append(ftz(ftz(t_gate[i]) + ftz(t_up[i])))
    _ = t_gate^
    _ = t_up^

    # =====================================================================
    # STAGE 10-12. `post_attention_layernorm` backward. The norm's forward
    # INPUT is `residual1.out`. NEW FOLD plus a ROUTED weight gradient.
    # =====================================================================
    var dot2 = List[Float32]()
    var dx2 = List[Float32]()
    var prod2 = List[Float32]()
    rms_norm_backward_into(
        st.d_norm2_out,
        fwd.residual1_out,
        w.norm2_w,
        fwd.norm2_sumsq,
        m,
        dm,
        dot2,
        dx2,
        prod2,
    )
    st.norm2_dot = dot2^
    st.dw_norm2 = gemm_oracle(ones_m, prod2, OP_NN, 1, dm, m)
    st.norm2_dx = dx2^
    _ = prod2^

    # =====================================================================
    # STAGE 13-14. S22, `r1 = x + o`. `residual1.out` fans out into the
    # norm (LDL:321) and into the residual add (LDL:323); FORWARD-USE ORDER
    # puts the norm branch first. Two terms, so no order to pin, stated.
    # =====================================================================
    for i in range(m * dm):
        st.d_residual1.append(
            ftz(ftz(st.norm2_dx[i]) + ftz(st.in_d_residual2[i]))
        )
    st.d_o_proj_out = st.d_residual1.copy()

    # =====================================================================
    # STAGE 15-16. `o_proj`: forward `OP_NT` at `(m, dm, qw)`. ROUTED.
    # =====================================================================
    st.d_attn_ctx = _gemm_bwd_a(st.d_o_proj_out, w.w_o, OP_NT, m, dm, qw)
    st.dw_o = _gemm_bwd_b(st.d_o_proj_out, fwd.attn_ctx, OP_NT, m, dm, qw)

    # =====================================================================
    # STAGE 17. The attention-weight gradient. **ROUTED**, DEVIATION 1405,
    # and this is the seam where the organizing rule pays off.
    #
    #     dy[b,h,t,j] = sum over d of dctx[b,h,t,d] * v_cache[b,kv,j,d]
    #
    # contracts over `head_dim`, which is THE SAME INTEGER in a prefill and
    # in a decode step, so gemm v1's `P = f(k)` is the same partition in
    # both paths and routing is decode-safe. It is a gemm v1 `OP_NT` cell at
    # `(l, s, hd)` -- the very shape contract DEVIATION 808 already runs for
    # S11.
    #
    # **THE ASYMMETRY WITH STAGES 22-24 LOOKS LIKE AN INCONSISTENCY AND IS
    # THE OPPOSITE OF ONE.** S19 was hand-written in the forward because it
    # contracts over the key axis; its `dy` derivative contracts over
    # `head_dim` and routes. S11 was routed in the forward because it
    # contracts over `head_dim`; its `dq` and `dk` derivatives contract over
    # the kv length and the query count and must be pinned. Both are the
    # same rule applied to whichever axis the DERIVATIVE contracts.
    #
    # **THIS PIN CANNOT BE FALSIFIED AT THE GATE SHAPE.** Sabotage
    # `B10_DW_VIA_CHAIN` replaces the routed gemm with a hand chain over
    # `d`, and the two agree bit for bit whenever `P(head_dim) == 1`, i.e.
    # `head_dim <= 128`. The profile's fixtures are `head_dim` 16 and 24.
    # So DEVIATION 1405 rests on the argument above and on gemm v1's own
    # certificate, NOT on a fired arm, and that is recorded here rather than
    # discovered later.
    #
    # `repeat_kv` is `kv = h // n_rep`, an INDEX MAP and never a
    # materialized expansion (contract DEVIATION 813). At `n_rep == 1` a
    # broken head-to-kv map is INVISIBLE, so the gates must carry both.
    # =====================================================================
    for bb in range(b):
        for h in range(nh):
            var kv = h // n_rep
            var dctx_head = List[Float32]()
            for qi in range(l):
                for d in range(hd):
                    dctx_head.append(
                        st.d_attn_ctx[(bb * l + qi) * qw + h * hd + d]
                    )
            var v_head = List[Float32]()
            for j in range(s):
                for d in range(hd):
                    v_head.append(
                        fwd.kv_v_cache[(bb * nkv + kv) * s * hd + j * hd + d]
                    )
            var cell = gemm_oracle(dctx_head, v_head, OP_NT, l, s, hd)
            for i in range(l * s):
                st.d_attn_weights.append(cell[i])
            _ = dctx_head^
            _ = v_head^
            _ = cell^

    # =====================================================================
    # STAGE 18-19. The softmax backward, ONE closed form. DEVIATION 1406.
    # =====================================================================
    var zdot = List[Float32]()
    var dsoft = List[Float32]()
    softmax_backward_into(
        st.d_attn_weights, fwd.attn_weights, b, nh, l, s, zdot, dsoft
    )
    st.attn_zdot = zdot^
    st.d_attn_masked = dsoft^

    # =====================================================================
    # STAGE 20. S13's backward, an EXACT IDENTITY. DEVIATION 1414.
    #
    # `masked = scores + mask_value` where the mask value is a CONSTANT, so
    # `d(masked)/d(scores)` is exactly 1 at every cell INCLUDING the masked
    # ones, and the gradient passes through with no rounding.
    #
    # **THE MASKED CELLS ARE NOT ZEROED, AND THAT IS A DECISION.** They are
    # already signed zeros -- `dS_j = pinned_mul(+0.0, dy_j - z)` carries
    # the sign of `dy_j - z` -- so forcing `+0.0` would differ only where
    # that sign is negative, which is roughly half the masked cells and is
    # REACHABLE WITHOUT A PLANT. Sabotage `B13_MASK_ZEROES_GRAD`, and the
    # ORACLE must predict the exact moved count before the device is asked.
    # =====================================================================
    st.d_attn_scores = st.d_attn_masked.copy()

    # =====================================================================
    # STAGE 21. S12's backward. `scores = cell * scale`, so
    # `d_cell = d_scores * scale`, ONE `pinned_mul` per score applied to the
    # FINISHED gradient. DEVIATION 1415.
    #
    # Folding the scale into the `dq` chain instead is sabotage
    # `B12_SCALE_INTO_DQ`, and it is **INERT AT EVERY POWER-OF-FOUR
    # `head_dim`**: at 16 and 64 the scale is exactly `0.25` and `0.125`,
    # exact scaling commutes with an fma chain bitwise, and moving it to the
    # far side changes nothing. The `head_dim = 24` fixture that contract
    # section 3 already requires -- for a different reason, to catch a wrong
    # scale SPELLING -- is what makes this arm fire at all.
    # =====================================================================
    var scale = attention_scale(hd)
    for i in range(cells):
        st.d_qk_cell.append(
            ftz(identical_mul(ftz(st.d_attn_scores[i]), scale))
        )

    # =====================================================================
    # STAGE 22. `dq`. **NEW ARITHMETIC, AND THE LANE'S LARGEST FINDING.**
    # DEVIATION 1402.
    #
    #     dq[b,h,t,d] = sum over j ASCENDING, ABSOLUTE key index, from +0.0
    #                   acc = ftz(fma(d_cell[b,h,t,j], k_cache[b,kv,j,d], acc))
    #
    # **GEMM v1 IS REFUSED HERE AND THE REASON IS NOT THE FORWARD'S
    # REASON.** S11 was ROUTED (contract DEVIATION 808) precisely because it
    # contracts over `head_dim`, the same integer in both paths. Its `dA`
    # contracts over the OUTPUT WIDTH, and S11's output width is the KV
    # AXIS. So the routed `dq` would be an `OP_NN` at `(l, hd, s)` with
    # `k' = S`, and `P = f(S)` builds one tree at `S = 257` and a different
    # one at `S = 200`. The masked `+0.0` tail, bitwise inert in a serial
    # ascending chain, is NOT inert under a tree whose shape changes with
    # the length.
    #
    # Under the chain it IS inert: at a masked `(t, j)` the gradient
    # `d_cell` is a signed zero (stage 20's note), `fma(+-0.0, k, acc)` is
    # `acc + (+-0.0)`, and a `+0.0`-seeded chain never holds `-0.0`. **So
    # `dq` is independent of the kv length and a decode step's `dq` is the
    # prefill's `dq` bit for bit.** That is the backward's clause (c) and
    # clause (d), and it holds by CONSTRUCTION -- the gate exists to catch
    # the construction being violated by an execution plan.
    #
    # Sabotage `B11_DQ_VIA_GEMM`, **INERT AT EVERY `S <= 128`** where
    # `P(S) == 1` and the gemm cell IS the whole-`k` ascending chain from
    # `+0.0`. The `L = 257` fixture is the only one that reaches it. An arm
    # that passes clause (a) by construction and needs clause (d) to bite is
    # exactly the shape contract section 10 warns will look inert and get
    # deleted.
    #
    # The loop is (batch, query, head, depth), which IS the output's
    # token-major `[M, qw]` order, so the cells are appended in place and no
    # write-by-index helper is needed. Loop order is free here because every
    # output cell owns its own chain.
    # =====================================================================
    for bb in range(b):
        for qi in range(l):
            for h in range(nh):
                var kv = h // n_rep
                var wbase = ((bb * nh + h) * l + qi) * s
                var vbase = (bb * nkv + kv) * s * hd
                for d in range(hd):
                    var acc = Float32(0.0)
                    for j in range(s):
                        acc = ftz(
                            identical_mul_add(
                                ftz(st.d_qk_cell[wbase + j]),
                                ftz(fwd.kv_k_cache[vbase + j * hd + d]),
                                acc,
                            )
                        )
                    st.d_q_rope.append(acc)

    # =====================================================================
    # STAGE 23. `dk_cache`. NEW ARITHMETIC. DEVIATIONS 1403, 1424, 1417.
    #
    #     dk[b,kv,j,d] = for h in the kv group ASCENDING
    #                      for t ASCENDING
    #                        acc = ftz(fma(d_cell[b,h,t,j], q_rope[b,h,t,d], acc))
    #
    # **ONE CHAIN, NOT A SUM OF PER-HEAD PARTIALS** (DEVIATION 1424). With
    # `n_rep > 1` several attention heads share one kv head, and folding
    # each head separately and adding the partials is a different
    # association. The head axis is OUTERMOST, matching the forward's own
    # `(batch, head, query)` nesting.
    #
    # **GEMM v1 IS REFUSED**: the routed form is `OP_TN` at `(s, hd, l)`
    # with `k' = L`, the QUERY COUNT, path dependent. Sabotage
    # `B11_DK_VIA_GEMM`, INERT only when `L <= 128` AND `n_rep == 1`
    # together -- at `n_rep == 2` the routed form must accumulate two
    # per-head gemms and fires on shape alone.
    #
    # **THIS IS A PARTIAL AND IT IS NAMED ONE** (DEVIATION 1417). A key at
    # slot `j` is read by every query at absolute position `>= j`, and the
    # queries in LATER calls are not in this call. So the fold over `[0, l)`
    # is the contribution of THIS call's queries and nothing else. The full
    # `[0, S)` range is emitted so a multi-call assembler has a complete
    # partial to add; assembling is out of scope. **THE FORWARD'S
    # DECODE-EQUALS-PREFILL DOES NOT TRANSFER TO THIS OUTPUT AND THIS LANE
    # DOES NOT PRETEND IT DOES.**
    # =====================================================================
    for bb in range(b):
        for kv in range(nkv):
            for j in range(s):
                for d in range(hd):
                    var acc = Float32(0.0)
                    for hh in range(n_rep):
                        var h = kv * n_rep + hh
                        for t in range(l):
                            var wbase = ((bb * nh + h) * l + t) * s
                            # The other operand is the FORWARD's rotated
                            # query, `q_rope.out`, token-major [M, qw]. NOT
                            # `q_proj.out`: the QK product reads the ROTATED
                            # q, so its derivative does too, and reading the
                            # pre-rotation activation here would produce a
                            # plausible gradient that is wrong by one
                            # rotation.
                            acc = ftz(
                                identical_mul_add(
                                    ftz(st.d_qk_cell[wbase + j]),
                                    ftz(
                                        fwd.q_rope_out[
                                            (bb * l + t) * qw + h * hd + d
                                        ]
                                    ),
                                    acc,
                                )
                            )
                    st.d_k_cache.append(acc)

    # =====================================================================
    # STAGE 24. `dv_cache`. NEW ARITHMETIC. DEVIATION 1404, the mirror of
    # contract DEVIATION 807.
    #
    #     dv[b,kv,j,d] = for h in the kv group ASCENDING
    #                      for t ASCENDING
    #                        acc = ftz(fma(y[b,h,t,j], d_ctx[b,h,t,d], acc))
    #
    # Same fold axis as stage 23, same refusal, same partial-gradient
    # caveat. Sabotage `B19_DV_VIA_GEMM`, the backward twin of
    # `S19_VALUE_SUM_VIA_GEMM`, and it inherits that arm's warning verbatim:
    # without clause (d) it looks inert and gets deleted.
    #
    # The masked cells contribute `fma(+0.0, dctx, acc)` because `y` is
    # exactly `+0.0` there (contract 7.1), so they are bitwise inert here
    # too -- which is why `dv` for a key slot does not depend on how many
    # PADDING keys were in the launch, only on how many QUERIES were.
    # =====================================================================
    for bb in range(b):
        for kv in range(nkv):
            for j in range(s):
                for d in range(hd):
                    var acc = Float32(0.0)
                    for hh in range(n_rep):
                        var h = kv * n_rep + hh
                        for t in range(l):
                            var wbase = ((bb * nh + h) * l + t) * s
                            acc = ftz(
                                identical_mul_add(
                                    ftz(fwd.attn_weights[wbase + j]),
                                    ftz(
                                        st.d_attn_ctx[
                                            (bb * l + t) * qw + h * hd + d
                                        ]
                                    ),
                                    acc,
                                )
                            )
                    st.d_v_cache.append(acc)

    # =====================================================================
    # STAGE 25-26. The KV append's backward: a SLICE, no arithmetic. This
    # call's own tokens occupy slots `[pos0, S)` and their gradient is
    # simply read out at the token-major `[M, kw]` layout the projections
    # expect. Slots `[0, pos0)` are the handoff.
    # =====================================================================
    for bb in range(b):
        for li in range(l):
            for kv in range(nkv):
                for d in range(hd):
                    var ix = (bb * nkv + kv) * s * hd + (pos0 + li) * hd + d
                    st.d_k_rope.append(st.d_k_cache[ix])
                    st.d_v_proj_out.append(st.d_v_cache[ix])

    # =====================================================================
    # STAGE 27-28. The RoPE backward, on q and on k. DEVIATION 1412.
    # ONE function serves both, because upstream applies one function to
    # both (:158-159) and a second spelling would be a second place to
    # drift. **RoPE is NOT applied to v** and therefore v's gradient does
    # not pass through here -- easy to get wrong and impossible to see in
    # the output, because a rotated gradient is still a plausible gradient.
    # =====================================================================
    var dqp = List[Float32]()
    var dkp = List[Float32]()
    rope_backward_into(st.d_q_rope, nh, hd, b, l, pos0, rope, dqp)
    rope_backward_into(st.d_k_rope, nkv, hd, b, l, pos0, rope, dkp)
    st.d_q_proj_out = dqp^
    st.d_k_proj_out = dkp^

    # =====================================================================
    # STAGE 29-32. The three input projections: forward `OP_NT` at
    # `(m, qw, dm)` and `(m, kw, dm)`. ROUTED, plus THE ONE THREE-TERM
    # FAN-IN IN THIS BLOCK, which IS order dependent.
    #
    # Pinned FORWARD-USE ORDER, `q` then `k` then `v`
    # (`LlamaAttention.forward` :250, :251, :252), left associative, and
    # with NO `+0.0` SEED. Sabotage `B_FANIN_ORDER_QKV_REVERSED`, INERT
    # wherever any two of the three terms are zero.
    # =====================================================================
    st.dw_q = _gemm_bwd_b(st.d_q_proj_out, fwd.norm1_out, OP_NT, m, qw, dm)
    st.dw_k = _gemm_bwd_b(st.d_k_proj_out, fwd.norm1_out, OP_NT, m, kw, dm)
    st.dw_v = _gemm_bwd_b(st.d_v_proj_out, fwd.norm1_out, OP_NT, m, kw, dm)
    var t_q = _gemm_bwd_a(st.d_q_proj_out, w.w_q, OP_NT, m, qw, dm)
    var t_k = _gemm_bwd_a(st.d_k_proj_out, w.w_k, OP_NT, m, kw, dm)
    var t_v = _gemm_bwd_a(st.d_v_proj_out, w.w_v, OP_NT, m, kw, dm)
    for i in range(m * dm):
        var acc = ftz(ftz(t_q[i]) + ftz(t_k[i]))
        st.d_norm1_out.append(ftz(ftz(acc) + ftz(t_v[i])))
    _ = t_q^
    _ = t_k^
    _ = t_v^

    # =====================================================================
    # STAGE 33-35. `input_layernorm` backward. Its forward INPUT is the
    # block input `x`, which the card saved as `input.x`.
    # =====================================================================
    var dot1 = List[Float32]()
    var dx1 = List[Float32]()
    var prod1 = List[Float32]()
    rms_norm_backward_into(
        st.d_norm1_out,
        fwd.input_x,
        w.norm1_w,
        fwd.norm1_sumsq,
        m,
        dm,
        dot1,
        dx1,
        prod1,
    )
    st.norm1_dot = dot1^
    st.dw_norm1 = gemm_oracle(ones_m, prod1, OP_NN, 1, dm, m)
    st.norm1_dx = dx1^
    _ = prod1^

    # =====================================================================
    # STAGE 36. THE OUTPUT. `x` fans out into the norm (LDL:306) and into
    # the residual add (LDL:317); FORWARD-USE ORDER puts the norm first.
    # Two terms, no order to pin, no seed.
    # =====================================================================
    for i in range(m * dm):
        st.d_x.append(ftz(ftz(st.norm1_dx[i]) + ftz(st.d_residual1[i])))

    _ = ones_m^
    return st^
