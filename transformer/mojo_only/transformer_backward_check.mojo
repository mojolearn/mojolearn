"""The gate file of the BACKWARD pass of profile
`mojolearn.identical.transformer.fp32.v1`, the path
`transformer/IDENTICAL_BACKWARD_PLAN.md` section 6.2 names and section 11
lists as owed item (2).

NOT A PORT. It runs the device backward
(`transformer/mojo_only/transformer_backward.mojo`) against the host oracle
(`transformer/mojo_only/transformer_backward_oracle.mojo`) and compares all
THIRTY-SEVEN recorded stages BY BITS.

**NOTHING IN THIS FILE HAS EVER BEEN COMPILED OR EXECUTED.** Written
2026-08-25, DEVIATIONS 1525 through 1549. No `mojo` process has read it, no
device has run it, no bit produced by it has been observed. Every sentence
below that says a clause "passes", a sabotage "bites", or a stage "moves" is
a PREDICTION about what the source says. **The sabotage ledger below has a
column for every arm and a value in none of them**, because inventing
numbers for a file that has not run would be the single worst thing a gate
could do -- a fabricated ledger reads exactly like evidence.

The FORWARD gate (`transformer_check.mojo`) is green on an M4 and this file
is modelled on it line for line. That is deliberate: the two cards are read
by the same differ, the two expectation tables are held to the same
discipline, and a reader who has audited one has audited most of the other.

WHAT IT CHECKS
---------------
1. Plan section 6.1's THIRTY-SEVEN stages, device against oracle, BITWISE,
   over a case SET, reporting `<tag> on X of Y cells`. "It failed" is not a
   finding.
2. THE CARD ITSELF: the thirty-seven tags in plan order, once each, at the
   path the CALLER chose (DEVIATION 1526).
3. Plan section 6.2's clauses (a) through (g), each with the negative
   control that answers "what would make this pass while gating nothing".
4. Plan section 6.3's TWENTY-TWO sabotage arms, each with the fixture case
   that can distinguish it and its predicted INERT case asserted.

WHY CLAUSE (d) IS THE CHUNK THEOREM AND NOT DECODE-EQUALS-PREFILL
-------------------------------------------------------------------
The forward's clause (d) says a token's output bits do not depend on how the
sequence was chopped up. **The backward cannot say that about every output
and this file does not pretend it can.** Plan 5.2:

    dq, and every activation gradient reachable ONLY through dq
        -> INDEPENDENT of the kv length and of the chopping.
    dk_cache, dv_cache, and every WEIGHT gradient
        -> A SUM OVER THE QUERIES IN THIS CALL, so a per-call backward
           computes a PARTIAL and the caller must add the partials.

> **A backward pass's INPUT gradients can be made chopping-invariant. Its
> PARAMETER gradients cannot, because they ARE the sum over the chopping.**

So clause (d) here has two halves and the second one is honest about a hole.

**(d1), WHICH RUNS ON THE DEVICE.** A length-`L` sequence is run as ONE
forward+backward and then as TWO, chunked over the QUERY axis with the KV
cache carried. For the tokens in the second chunk, every activation gradient
that folds over the KEY axis must be bit-identical to the unchunked run --
`bwd.d_q_rope`, `bwd.attn.zdot`, `bwd.d_attn_masked`, `bwd.d_attn_scores`,
`bwd.d_qk_cell`, `bwd.d_attn_weights`, `bwd.d_attn_ctx`, `bwd.d_q_proj_out`
and the whole MLP path. And the stages that fold over the QUERY axis --
`bwd.d_k_cache`, `bwd.d_v_cache`, `bwd.d_k_proj_out`, `bwd.d_v_proj_out`,
`bwd.d_norm1_out`, `bwd.norm1.dot`, `bwd.norm1.dx`, `bwd.d_x` and every
`bwd.dW_*` -- must MOVE, and the clause ASSERTS that they do, because a gate
that only asserted the positive half would pass on an implementation that
computed constants.

**(d2), WHICH DOES NOT RUN ON THE DEVICE, AND WHY. DEVIATION 1527.** Plan
5.3's actual theorem is about a CARRIED ACCUMULATOR: split a serial
ascending chain at any index, store the accumulator, resume from it, and the
result is the unsplit chain's bit for bit. **THERE IS NO CHUNKED ENTRY
POINT.** `llama_decoder_layer_backward` takes no seed, no carry flag and no
accumulator argument, and neither does
`transformer_block_backward_oracle`; `grep -i chunk` over both files returns
two prose mentions and no parameter. So the theorem the plan calls "the
property this lane gets for free that the GEMM lane cannot have" has NO
IMPLEMENTATION TO GATE. What this file does instead is demonstrate the
theorem on the exact chain shape the plan pins -- a `+0.0`-seeded ascending
`fma` chain -- at MISALIGNED split points, WITH the uncarried control that
must move, and then say in the SCOPE line that the device half is unbuilt.
That is a demonstration, not a gate, and it is labelled as one.

FOUR ARMS ARE PREDICTED INERT AT THE GATE SHAPE AND TWO OF THOSE CANNOT
FIRE WITHOUT NEW FIXTURES. **THEY ARE REPORTED AS SMOKE TESTS BY NAME.**
-------------------------------------------------------------------------
Plan 6.3 says so and this file repeats it where a reader will meet it,
because the forward lane's own scar is that an arm which looks inert gets
deleted.

  * **`B10_DW_VIA_CHAIN` CANNOT FIRE AT ANY FIXTURE THIS PROFILE HAS.** It
    replaces the routed `k' = head_dim` gemm with a hand chain, and the two
    are bit-identical whenever `P(head_dim) == 1`, i.e. `head_dim <= 128`.
    The fixtures are `head_dim` 16 and 24. **DEVIATION 1405 is therefore
    pinned by ARGUMENT and by gemm v1's certificate, not by a fired arm.**
  * **`BWD_OPERAND_ORDER` IS INERT THROUGH THIS LANE ENTIRELY.** Every
    forward call in this profile is `OP_NT`, whose `dA` and `dB` both put
    `dC` on the LEFT, so the arm's whole content -- ignoring the side flag
    -- is a no-op here. Firing it and reporting green would be reporting
    that a gate cannot see something as though it had looked.
  * **`B11_DQ_VIA_GEMM` needs `S >= 129`** and the only fixture that reaches
    it is `long_l257`, which is OFF by default because clause (a) at
    `S = 257` is 4 buffers of `1*2*257*257` cells per stage. Set
    `MOJOLEARN_TFB_CHECK_LONG=1`. Without it the arm is UNGATED and this
    file says UNGATED, not green.
  * **`B_FANIN_ZERO_SEED` HAS NO CONSTRUCTIBLE WITNESS IN THIS LANE AND
    THAT IS A NEW FINDING, DEVIATION 1528.** Plan 6.3 says it is "vacuous
    without a plant" and predicts it moves zero cells on every unplanted
    fixture. It is worse than that: **no plant available to this gate can
    make it fire.** The arm fires only where the FIRST accumulated term of
    the `d(norm1.out)` fan-in is `-0.0`, and that first term is
    `bwd.d_q_proj_out`, which is the RoPE transpose of `bwd.d_q_rope`, which
    is a `+0.0`-seeded fma chain -- and plan 5.1(i) PROVES such a chain
    never holds `-0.0`. The transpose is `ftz(a + ftz(rot * s))` of two
    values reachable only from that chain, so it cannot manufacture a `-0.0`
    either. The only input this gate controls is `d_out`, and driving it to
    all-`-0.0` gives `+0.0` at `d_q_rope` for exactly that reason. **The arm
    needs a planted `-0.0` INSIDE a saved forward stage, which means a
    fixture edit this agent may not make.** Reported as NOT CONSTRUCTED.

ONE DISAGREEMENT WITH PLAN SECTION 6.2's CLAUSE (c), RECORDED RATHER THAN
SILENTLY WORKED AROUND. DEVIATION 1529.
-------------------------------------------------------------------------
Clause (c) as written asks for "BATCH COMPOSITION and SEQUENCE LENGTH
invariance of the activation gradients, `bwd.d_x`, `bwd.d_q_rope`,
`bwd.d_norm1_out`, `bwd.d_attn_ctx`".

**Two of those four are NOT sequence-length invariant, and the plan's own
section 5.2 is what says so.** `bwd.d_norm1_out` is the THREE-TERM fan-in of
the `q`, `k` and `v` projection gradients, and `bwd.d_k_proj_out` /
`bwd.d_v_proj_out` are the `[pos0, S)` slices of `bwd.d_k_cache` and
`bwd.d_v_cache` -- which fold over the QUERY axis. A key at slot `j` is read
by every query at position `>= j`, so lengthening the sequence gives key `j`
MORE contributing queries and its gradient changes. `bwd.d_x` is
`bwd.norm1.dx` plus `bwd.d_residual1`, and `norm1.dx` is downstream of
`d_norm1_out`, so it moves too.

That is not a defect in the profile -- it is what a gradient IS -- it is a
defect in the clause's LIST. So this file gates the corrected split:

    LENGTH-INVARIANT (asserted equal, per token):
        0,1,2,4,5,6,9,10,12,13,14,15,17,18,19,20,21,22,27
    LENGTH-DEPENDENT (asserted to MOVE):
        3,7,8,11,16,23,24,25,26,28,29,30,31,32,33,34,35,36

**BATCH composition is different and the plan is right about it there.**
`bwd.d_k_cache` is indexed `[B, n_kv, S, hd]`, so batch row `bb`'s cache
gradient sums only `bb`'s own queries and IS batch-composition invariant --
while the eleven `bwd.dW_*` are not and must move, which plan 5.4 states and
this file asserts. **The two halves of clause (c) therefore have DIFFERENT
moving sets, and a gate that used one set for both would be wrong twice.**

WHAT WOULD MAKE EACH CLAUSE PASS WHILE GATING NOTHING
-------------------------------------------------------
* **(a)** A device dump and an oracle dump that are the same object, or two
  dumps of different LENGTHS compared to the shorter. `compare_stage` raises
  on a length mismatch and the two dumps are built by two functions in two
  files from two structs that share no code. What clause (a) CANNOT catch is
  our oracle being wrong in the same way as our device; only an independent
  reference can and `transformer/corpus/` does not exist.
* **(b)** Comparing only `bwd.d_x`. The forward lane MEASURED thirteen moved
  stages being absorbed by a residual add while an output-only gate called
  the sabotage inert. All thirty-seven are compared.
* **(c)** A row slicer that returns row 0 whatever row it is asked for, or a
  token viewer that returns token 0 whatever token it is asked for. EACH
  HALF HAS ITS OWN FIRING CONTROL and neither is optional.
* **(d)** A chunked path and an unchunked path that share a buffer, so the
  comparison is a value against itself. The control is a deliberate token
  MISALIGNMENT that MUST differ.
* **(e)** An unconditional refusal. Every plant would be "refused by name"
  and the clause would gate nothing. The control is a CLEAN call that must
  NOT raise and must record all thirty-seven stages.
* **(f)** A tolerance. Plan 6.2(f) is emphatic that a transpose error is bit
  identical on three vendors, so identity is not correctness. What is
  checkable exactly is checked exactly; what needs a float64 directional
  derivative is NAMED and left open.
* **(g)** An arm whose `-D` was misspelled and silently ignored, so a clean
  build reports itself as a bitten sabotage. `MOJOLEARN_TFB_EXPECT_SABOTAGE`
  is DEVIATION 1530 and it is `tools/gemm_ladder.sh:71`'s scar written down.

THE SABOTAGE LEDGER, **NOT MEASURED**
---------------------------------------
Twenty-two arms, plan section 6.3's table. Twenty are this lane's and two
are the GEMM lane's, reaching this file through `_route_a` and `_route_b`.

    arm                          predicted first stage   witness case
    B01_DOT_UNFUSED              bwd.norm2.dot           base_b1_l4_nrep1
    B01_DOT_DESCENDING           bwd.norm2.dot           base_b1_l4_nrep1
    B_RSTD_RECOMPUTE_DESCENDING  bwd.norm2.dx            base_b1_l4_nrep1
    B09_ROPE_TRANSPOSE_SIGN      bwd.d_q_proj_out        base_b1_l4_nrep1
    B09_ROPE_HALVES_ADJACENT     bwd.d_q_proj_out        base_b1_l4_nrep1
    B10_ROPE_BWD_FUSED           bwd.d_q_proj_out        base_b1_l4_nrep1
    B10_DW_VIA_CHAIN             (none; SMOKE TEST)      --
    B11_DQ_VIA_GEMM              clause (c) length       long_l257
    B11_DK_VIA_GEMM              bwd.d_k_cache           base_b2_l4_nrep2
    B19_DV_VIA_GEMM              bwd.d_v_cache           base_b2_l4_nrep2
    B12_SCALE_INTO_DQ            bwd.d_qk_cell           odd_head_dim_24
    B13_MASK_ZEROES_GRAD         bwd.d_attn_scores       base_b1_l4_nrep1
    B18_SOFTMAX_DECOMPOSED       bwd.d_attn_masked       base_b1_l4_nrep1
    B18_ZFOLD_UNFUSED            bwd.attn.zdot           base_b1_l4_nrep1
    B18_ZFOLD_DESCENDING         bwd.attn.zdot           base_b1_l4_nrep1
    B20_SIGMOID_FROM_SILU        bwd.d_gate_proj_out     base_b1_l4_nrep1
    B20_SILU_DERIV_ALT_ASSOC     bwd.d_gate_proj_out     base_b1_l4_nrep1
    B20_SILU_DERIV_FUSED         bwd.d_gate_proj_out     base_b1_l4_nrep1
    B_FANIN_ZERO_SEED            (none; NOT CONSTRUCTED) --
    B_FANIN_ORDER_QKV_REVERSED   bwd.d_norm1_out         base_b1_l4_nrep1
    BWD_UNTRANSPOSED             bwd.d_mlp_gated         base_b1_l4_nrep1
    BWD_OPERAND_ORDER            (none; SMOKE TEST)      --

Six arms carry an INERT case, and the inert case is the half that makes an
arm a reach proof instead of a smoke test:

    B11_DK_VIA_GEMM         inert on base_b1_l4_nrep1  (L<=128 AND n_rep==1)
    B19_DV_VIA_GEMM         inert on base_b1_l4_nrep1
    B12_SCALE_INTO_DQ       inert on base_b1_l4_nrep1  (head_dim 16, exact)
    B18_SOFTMAX_DECOMPOSED  inert on base_b1_l1_nrep2  (S == 1)
    B18_ZFOLD_UNFUSED       inert on base_b1_l1_nrep2
    B18_ZFOLD_DESCENDING    inert on base_b1_l1_nrep2

and the three ROPE arms carry a PER-CELL inert mask instead of a per-case
one: they must move NOTHING at ABSOLUTE POSITION 0, where `sin` is exactly
`+0.0`. Plan 6.3 spells the requirement -- "per CELL, not per fixture, so
the gate must COUNT moved cells and RAISE if the count equals the position-0
population" -- and `moved_cells_at_position_zero` is that count.

**SEVEN OF THE TWENTY-TWO HAVE NO ASSERTED INERT CASE AND THE REASON IS THE
SAME FOR ALL SEVEN: THIS PROFILE HAS NO `d_model == 1` OR `head_dim == 2`
FIXTURE, AND NO FIXTURE WHOSE GATE ACTIVATIONS ARE ALL >= 17.** Plan 6.3
predicts those inert sets and the fixture set cannot express them. They are
reported as MOVED-BUT-UNMASKED rather than as reach proofs, which is the
honest label. Adding `d_model == 1` to `transformer_fixture.mojo` would
close three of the seven in one line and it is a file this agent may not
edit.

RUNNING IT
-----------
    MOJOLEARN_IDENTITY_TRACE=/tmp/tb.card \\
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        transformer/mojo_only/transformer_backward_check.mojo

and one sabotage arm at a time, each of which MUST fail:

    MOJOLEARN_TFB_EXPECT_SABOTAGE=B18_ZFOLD_DESCENDING \\
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_TRANSFORMER_SABOTAGE_B18_ZFOLD_DESCENDING=1 \\
        -I . transformer/mojo_only/transformer_backward_check.mojo

When an arm is armed this file INVERTS its verdict: a clean compare is then
the FAILURE, because it means the sabotage was reached and made no
difference, or was never reached at all. Both are `[[reached-but-inert]]`.

ENVIRONMENT
------------
    MOJOLEARN_IDENTITY_TRACE            where the card goes (DEV 1526)
    MOJOLEARN_TFB_EXPECT_SABOTAGE       guard against a misspelled -D
    MOJOLEARN_TFB_CHECK_LONG            add long_l257 to clause (a)
    MOJOLEARN_TFB_CHECK_CLAUSE_B        run clause (b)
    MOJOLEARN_TFB_CHECK_CLAUSE_C        run clause (c), both halves
    MOJOLEARN_TFB_CHECK_CLAUSE_D        run clause (d), both halves
    MOJOLEARN_TFB_CHECK_CLAUSE_E        run clause (e)
    MOJOLEARN_TFB_CHECK_CLAUSE_F        run clause (f), the exact-integer arm

Clauses (b) through (f) are OFF by default and each is one line to turn on.
That is a COST decision and not a confidence one; a leg that reports only
clause (a) has closed only clause (a) and the printed SCOPE line says so.

OWED, and this file covers none of it
---------------------------------------
* **A COMPILE.** Nothing here, and nothing in either file it drives, has
  been through the front end.
* **A RUN, ON ANY COLUMN.** Zero bits observed.
* **THE 22 SABOTAGE BUILDS.** Twenty-two compiles, one arm each.
* **THE FLOAT64 DIRECTIONAL DERIVATIVE.** Plan section 11's largest item.
  Clause (f) covers the ROUTING and the linear seams by exact integers; the
  nonlinear seams -- `silu'`, the `rsqrt` tail, the softmax closed form --
  need a float64 reference at a stated tolerance and **there is none**.
  `transformer_oracle.mojo` deliberately carries no float64 reference and
  `transformer/corpus/` does not exist. **THE BITS ARE GATED AND THE
  CALCULUS IS NOT.** That sentence is printed in the SCOPE line of every
  run of this file, because it is the single most likely thing for a reader
  to take more of than is offered.
* **A CHUNKED BACKWARD ENTRY POINT**, DEVIATION 1527, without which plan
  5.3's chunk theorem has nothing to gate on a device.
* **AN `x` FIELD ON `LlamaDeviceStages`**, plan section 1's cross-lane
  request. Until it lands this file must keep the forward's `x` upload alive
  itself and pass it to the backward, which is exactly the shape
  `[[mojo-buffer-freed-at-last-use]]` punishes.

`[[mojo-buffer-freed-at-last-use]]`: every device object below is a LOCAL
kept alive past the `ctx.synchronize()` that reads it, with explicit
`_ = x^` moves at the foot of every runner. A `DeviceBuffer` is dead at
`.unsafe_ptr()` and this repository has lost a night to that.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from gemm.mojo_only.gemm_backward import gemm_backward_sabotage_name
from gemm.mojo_only.gemm_identical import gemm_sabotage_name
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul,
    identical_mul_add,
)
from transformer.mojo_only.transformer_fixture import (
    BITS_NEG_ZERO,
    BITS_POS_INF,
    BITS_POS_ZERO,
    BITS_QNAN,
    FIXTURE_CASE_COUNT,
    PLANT_NONE,
    RMS_EPS,
    ROPE_THETA,
    FixtureCase,
    ScorePlant,
    TransformerDims,
    TransformerWeights,
    bits32_hex,
    bits_of,
    f32_from_bits,
    fixture_case,
    fixture_case_by_name,
    fixture_case_seed,
    fixture_dims,
    fixture_score_plant,
    fixture_splitmix64,
    fixture_weights,
    fixture_x,
    mask_fill,
    mode_name,
    profile_constants_are_intact,
)
from transformer.mojo_only.transformer_oracle import (
    TransformerKVCache,
    TransformerStages,
    build_rope_table,
    transformer_block_oracle,
)
from transformer.mojo_only.transformer_backward_oracle import (
    BWD_NEG_HALF,
    BWD_TWO,
    TRANSFORMER_BACKWARD_STAGE_COUNT,
    backward_oracle_dump,
    backward_stage_tag,
    softmax_backward_into,
    transformer_block_backward_oracle,
)
from transformer.ported.transformers.models.llama.modeling_llama import (
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _download,
    _upload,
    llama_block_sabotage_name,
    llama_decoder_layer_forward_planted,
)
from transformer.mojo_only.transformer_backward import (
    BWD_ANY_SABOTAGE,
    LlamaBackwardStages,
    llama_backward_sabotage_name,
    llama_decoder_layer_backward,
)


# ===========================================================================
# THE CARD PATH IS THE CALLER'S WHEN THE CALLER NAMES ONE.
#
# DEVIATION 1526, and it is DEVIATION 970 not repeated. There, a
# `comptime TRACE_PATH` was read DIRECTLY by every write site, so the card
# always landed in /tmp no matter what the caller asked for, and the Apple
# column of 2026-08-24 ran the mamba lane GREEN while phase 8 reported "NO
# CARD written" -- because the card was in /tmp under the alias while the
# judge looked in lanes/. A green check with no card is INDISTINGUISHABLE
# FROM SUCCESS to the gate above it. Read at RUN time. NEVER HARDCODE ONE.
# ===========================================================================

comptime TRACE_PATH = "/tmp/mojolearn_transformer_backward.trace"


def card_path() -> String:
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(TRACE_PATH)


#: The backward driver's prefix. DISJOINT from `transformer_check.mojo`'s
#: `tfx`, so a single trace can hold a forward card and a backward card
#: without `IdentityTrace`'s uniqueness invariant firing -- which is the
#: shape a diagnosing reader wants, because the forward stages ARE the
#: backward's saved inputs and a divergence in one is diagnosed by looking
#: at the other.
comptime TAG_PREFIX = "tbx"

comptime CLAUSE_B_LAUNCHES = 8

#: Stage layout kinds. `M = B * L` token-major; `S = pos0 + L`.
#:
#: K_TOKEN   `[M, W]`        row `bb` owns `L * W` contiguous cells
#: K_ATTNROW `[B,nh,L,S]`    query `qi`'s `S` keys, per head
#: K_ATTNSCL `[B,nh,L]`      query `qi`'s one cell, per head
#: K_KV      `[B,nkv,S,hd]`  slot `j`, per (batch, kv head)
#: K_WEIGHT  no batch axis, no token axis. A SUM OVER EVERY TOKEN IN THE
#:           CALL, which is why it is its own kind and not a "global": it is
#:           the one kind that MUST MOVE under both halves of clause (c).
comptime K_TOKEN = 0
comptime K_ATTNROW = 1
comptime K_ATTNSCL = 2
comptime K_KV = 3
comptime K_WEIGHT = 4


def bwd_stage_kind(i: Int) raises -> Int:
    if i == 3 or i == 7 or i == 8 or i == 11 or i == 16:
        return K_WEIGHT  # dW_down, dW_gate, dW_up, dW_norm2, dW_o
    if i == 29 or i == 30 or i == 31 or i == 34:
        return K_WEIGHT  # dW_q, dW_k, dW_v, dW_norm1
    if i == 17 or i == 19 or i == 20 or i == 21:
        return K_ATTNROW  # d_attn_weights, d_attn_masked, d_attn_scores,
        # d_qk_cell
    if i == 18:
        return K_ATTNSCL  # attn.zdot
    if i == 23 or i == 24:
        return K_KV  # d_k_cache, d_v_cache
    if i < 0 or i >= TRANSFORMER_BACKWARD_STAGE_COUNT:
        raise Error(
            String("transformer_backward_check: no stage ") + String(i)
        )
    return K_TOKEN


def bwd_token_width(i: Int, dims: TransformerDims) raises -> Int:
    """Cells per TOKEN for the token-major stages, plan 6.1's shapes with
    `M = B*L` factored out."""
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    if i == 0 or i == 1 or i == 9 or i == 12 or i == 13 or i == 14:
        return dm
    if i == 32 or i == 35 or i == 36:
        return dm
    if i == 2 or i == 4 or i == 5 or i == 6:
        return it
    if i == 10 or i == 33:
        return 1  # norm2.dot, norm1.dot
    if i == 15 or i == 22 or i == 27:
        return qw  # d_attn_ctx, d_q_rope, d_q_proj_out
    if i == 25 or i == 26 or i == 28:
        return kw  # d_k_rope, d_v_proj_out, d_k_proj_out
    raise Error(
        String("transformer_backward_check: stage ")
        + String(i)
        + " is not token-major and has no per-token width"
    )


def bwd_weight_cells(i: Int, dims: TransformerDims) raises -> Int:
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    if i == 3:
        return dm * it  # dW_down
    if i == 7 or i == 8:
        return it * dm  # dW_gate, dW_up
    if i == 11 or i == 34:
        return dm  # dW_norm2, dW_norm1
    if i == 16:
        return dm * qw  # dW_o
    if i == 29:
        return qw * dm  # dW_q
    if i == 30 or i == 31:
        return kw * dm  # dW_k, dW_v
    raise Error(
        String("transformer_backward_check: stage ")
        + String(i)
        + " is not a weight gradient"
    )


def bwd_stage_cells(
    i: Int, dims: TransformerDims, b: Int, l: Int, s: Int
) raises -> Int:
    """The USED cell count of stage `i` at this launch.

    **NEVER `len(buf)`.** The four `[B, nh, L, S]` buffers are allocated at
    `s_max` and used PACKED at `s`, and `core/identity_trace.mojo` rule 3 is
    emphatic: hashing the tail folds uninitialized memory into a stage,
    which differs run to run on ONE machine and would make the instrument
    report divergence everywhere. Every count below is the same expression
    `llama_decoder_layer_backward`'s own `_rec` call passes, and if the two
    ever disagree the card and the gate are looking at different arrays."""
    var kind = bwd_stage_kind(i)
    if kind == K_WEIGHT:
        return bwd_weight_cells(i, dims)
    if kind == K_ATTNROW:
        return b * dims.n_heads * l * s
    if kind == K_ATTNSCL:
        return b * dims.n_heads * l
    if kind == K_KV:
        return b * dims.n_kv_heads * s * dims.head_dim
    return b * l * bwd_token_width(i, dims)


def env_on(name: String) -> Bool:
    return String(getenv(name)) != ""


def env_str(name: String) -> String:
    return String(getenv(name))


def hexbits(v: Float32) -> String:
    return bits32_hex(v)


def mode_is_identical() -> Bool:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return True
    return False


def llama_dims_of(d: TransformerDims) -> LlamaDims:
    """`TransformerDims` to `LlamaDims`, field by field.

    The two types are deliberately not one type -- `modeling_llama.mojo`'s
    docstring says no fixture type crosses its boundary, because the two
    halves of this lane were written concurrently and a shared type would
    have been a guess by both -- and this three-line function is the whole
    cost of that."""
    return LlamaDims(
        d.d_model, d.n_heads, d.n_kv_heads, d.head_dim, d.intermediate
    )


# ===========================================================================
# THE INCOMING GRADIENT
#
# DEVIATION 1531. `d(residual2.out)` is an INPUT to this lane and the
# fixture file has no generator for it -- `transformer_fixture.mojo`
# predates the backward and its eleven tensor ids stop at the cache tail.
# This gate may not edit that file, so the generator lives here, and it
# lives here under a HARD RULE about what it must not emit.
#
# **THE FIVE WAYS A FIXTURE WENT BLIND ON 2026-08-25 ARE ALL AVAILABLE HERE
# AND THREE OF THEM ARE ABOUT THIS ONE FUNCTION.**
#
#   * EXACTLY-REPRESENTABLE VALUES. A generator emitting integers scaled by
#     `2^-4` made every product exact, so an fma-versus-unfused arm was bit
#     neutral on 16 of 20 rows. `B18_ZFOLD_UNFUSED`, `B01_DOT_UNFUSED` and
#     `B20_SILU_DERIV_FUSED` are exactly that class of arm and all three are
#     aimed at seams this generator feeds. Every value below carries a FULL
#     23-BIT MANTISSA drawn from the hash.
#   * ONE BINADE. Four partials all inside `[1, 2)` made a balanced tree and
#     a serial chain agree, so a fold-order control did not move.
#     `B18_ZFOLD_DESCENDING` and `B01_DOT_DESCENDING` are fold-order arms.
#     The exponent here is DRAWN, over `2^-8` to `2^8`, so two contributors
#     to one fold can differ by `2^16` and the order decides which is
#     absorbed.
#   * ABSORPTION. A control that dropped a tail term did not move because
#     the term rounded away. `guard_d_out_separates` MEASURES that an
#     ascending fma chain over one generated row and a descending one give
#     different bits, and RAISES if they do not.
#
# WHAT IT NEVER EMITS, and each exclusion is load bearing.
#
#   * **No NaN and no infinity.** The exponent is clamped strictly inside
#     the normal range. Plan 6.2(e) refuses both BY NAME, and a fixture that
#     trips the refusal it is not testing reports a refusal as a failure.
#   * **No subnormal and no zero.** Both are PLANTED where they are wanted
#     and never drawn, so an arm whose predicted inert set is "no subnormal
#     intermediate" has a set this gate DECIDED rather than hoped for.
#   * **Nothing past `2^8`.** A gradient is multiplied by weights and summed
#     over `intermediate = 300` at the widest fixture, and a `2^14` input
#     would put `bwd.dW_down` within reach of overflow. An `inf` in a stage
#     is bitwise deterministic and would not break identity -- but plan
#     6.2(e)'s audit would then be measuring an overflow it did not plant,
#     and a fixture that quietly saturates is a fixture whose fold-order
#     arms stop separating.
# ===========================================================================

comptime TID_D_OUT = 21
"""Distinct from `transformer_fixture.mojo`'s eleven tensor ids (1 through
11), so a `d_out` value can never collide with a weight or an input the
forward already generated. A collision would be harmless right up to the
moment somebody compared two hashes and read meaning into the agreement."""

comptime DOUT_PLANT_NONE = 0
comptime DOUT_PLANT_NEG_ZEROS = 1
comptime DOUT_PLANT_ALTERNATING_ZEROS = 2


def bwd_d_out(c: FixtureCase, dims: TransformerDims, plant: Int) raises -> List[
    Float32
]:
    """`d(residual2.out)` at `[B*L, d_model]`, full mantissa, wide exponent.

    Built by ASSEMBLING A BIT PATTERN and not by scaling a fraction, because
    the scaling spellings are exactly the ones that went blind. See the
    block comment above for the three traps this closes and the three values
    it refuses to emit.

    `[[mojo-int-widening-sign-extends]]`: every widening goes through a
    `UInt64` mask and never through a signed intermediate."""
    var n = c.b * c.l * dims.d_model
    var key = fixture_splitmix64(
        fixture_case_seed(0) ^ (UInt64(TID_D_OUT) << 32)
    )
    var out = List[Float32](capacity=n if n > 0 else 1)
    for i in range(n):
        var h = fixture_splitmix64(key + UInt64(i))
        var frac = UInt32(Int((h >> 40) & UInt64(0x007FFFFF)))
        var sign = UInt32(Int((h >> 3) & UInt64(1))) << 31
        # 17 exponent values, 119 through 135, i.e. 2^-8 through 2^8.
        var eb = Int((h >> 8) & UInt64(0x1F))
        var expf = 119 + (eb % 17)
        out.append(f32_from_bits(sign | (UInt32(expf) << 23) | frac))
    if plant == DOUT_PLANT_NEG_ZEROS:
        for i in range(n):
            out[i] = f32_from_bits(BITS_NEG_ZERO)
    elif plant == DOUT_PLANT_ALTERNATING_ZEROS:
        # Both zero signs through the whole incoming gradient. Plan 6.2(e)'s
        # SIGNED-ZERO half of the audit: the masked cells of
        # `bwd.d_attn_scores` have predicted sign `sign(dy_j - z)`, and a
        # fixture with no zeros in it cannot exercise the prediction.
        for i in range(n):
            if i % 2 == 0:
                out[i] = f32_from_bits(BITS_POS_ZERO)
            elif i % 3 == 0:
                out[i] = f32_from_bits(BITS_NEG_ZERO)
    return out^


def guard_d_out_separates(c: FixtureCase, dims: TransformerDims) raises -> String:
    """**THE DEMONSTRATION THAT THE INCOMING GRADIENT CAN SEPARATE A FOLD
    ORDER.** Without it, every `*_DESCENDING` and `*_UNFUSED` verdict this
    file prints is a verdict about a generator nobody measured.

    `training/mojo_only/loss_check.mojo`'s guard 4 is the model and it is
    TWO-SIDED for a reason: a one-sided "they agreed" is indistinguishable
    from a broken comparison. So this arm shows

      1. an ASCENDING `+0.0`-seeded fma chain over one generated row and a
         DESCENDING one giving DIFFERENT bits -- if they agree, every
         fold-order arm in the table is unfalsifiable here;
      2. the FUSED chain and the UNFUSED product-then-add chain giving
         DIFFERENT bits over the same row -- if they agree, the three
         `*_UNFUSED` arms are unfalsifiable;
      3. the exponent SPREAD, printed, because "the values are full
         mantissa" is a claim about the generator and "these `d_model`
         partials span 16 binades" is a claim about the ROW, and it is the
         second that decides whether an order can be seen."""
    var dm = dims.d_model
    var d_out = bwd_d_out(c, dims, DOUT_PLANT_NONE)
    var other = fixture_x(c)
    if len(other) < dm:
        raise Error(
            "transformer_backward_check: the case's x is shorter than one"
            " row, so the guard has no second operand"
        )
    var asc = f32_from_bits(BITS_POS_ZERO)
    for j in range(dm):
        asc = ftz(identical_mul_add(d_out[j], other[j], asc))
    var desc = f32_from_bits(BITS_POS_ZERO)
    var j2 = dm - 1
    while j2 >= 0:
        desc = ftz(identical_mul_add(d_out[j2], other[j2], desc))
        j2 -= 1
    var unfused = f32_from_bits(BITS_POS_ZERO)
    for j in range(dm):
        unfused = ftz(unfused + ftz(identical_mul(d_out[j], other[j])))
    var lo = 255
    var hi = 0
    for j in range(dm):
        var e = Int((bits_of(d_out[j]) >> UInt32(23)) & UInt32(0xFF))
        if e < lo:
            lo = e
        if e > hi:
            hi = e
    if bits_of(asc) == bits_of(desc):
        raise Error(
            String("transformer_backward_check: THE d_out GENERATOR IS")
            + " BLIND TO FOLD ORDER. An ascending fma chain and a descending"
            + " one over one row both gave "
            + hexbits(asc)
            + ". That is the 2026-08-25 failure exactly -- exactly"
            + "-representable values, or one binade -- and"
            + " B18_ZFOLD_DESCENDING and B01_DOT_DESCENDING would both be"
            + " reported inert ([[reached-but-inert]])."
        )
    if bits_of(asc) == bits_of(unfused):
        raise Error(
            String("transformer_backward_check: THE d_out GENERATOR CANNOT")
            + " SEPARATE A FUSED CHAIN FROM AN UNFUSED ONE. Both gave "
            + hexbits(asc)
            + ", so every product in the row is exactly representable and"
            + " B18_ZFOLD_UNFUSED, B01_DOT_UNFUSED and B20_SILU_DERIV_FUSED"
            + " are all unfalsifiable on this fixture"
            + " ([[reached-but-inert]])."
        )
    return (
        String("d_out generator: ascending ")
        + hexbits(asc)
        + " vs descending "
        + hexbits(desc)
        + " (SEPARATES) vs unfused "
        + hexbits(unfused)
        + " (SEPARATES); exponent field spans "
        + String(lo)
        + ".."
        + String(hi)
        + " ("
        + String(hi - lo)
        + " binades, so the partials are NOT all in one)"
    )


# ===========================================================================
# THE DEVICE SIDE
#
# DEVIATION 1532: this file imports the underscore-prefixed `_download` and
# `_upload` out of `modeling_llama.mojo`, exactly as `transformer_check.mojo`
# does. That is the repository's existing habit for gate files
# (`glm/mojo_only/qn_losses_check.mojo:73`, `mojo_only/
# portable_fmax_check.mojo:85-86`) and the alternative -- a second upload and
# download here -- is a second spelling of a device copy that can drift from
# the one the block itself uses. **A gate whose plumbing is not the block's
# plumbing is a gate that can pass because its plumbing is different.**
# ===========================================================================


def backward_device_dump(
    ctx: DeviceContext,
    mut bst: LlamaBackwardStages,
    dims: TransformerDims,
    b: Int,
    l: Int,
    s: Int,
) raises -> List[List[Float32]]:
    """All thirty-seven stage buffers back on the host, CARD ORDER,
    index-aligned with `backward_oracle_dump` and with `backward_stage_tag`.

    Every count is `bwd_stage_cells`, which is the same expression
    `llama_decoder_layer_backward`'s own `_rec` calls pass. **If the two ever
    disagree the card and the gate are looking at different arrays**, and the
    four `[B, nh, L, S]` buffers are the ones that would: they are allocated
    at `s_max` and used packed at `s`."""
    var out = List[List[Float32]]()
    out.append(_download(ctx, bst.in_d_residual2, bwd_stage_cells(0, dims, b, l, s)))
    out.append(_download(ctx, bst.d_down_proj_out, bwd_stage_cells(1, dims, b, l, s)))
    out.append(_download(ctx, bst.d_mlp_gated, bwd_stage_cells(2, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_down, bwd_stage_cells(3, dims, b, l, s)))
    out.append(_download(ctx, bst.d_silu_out, bwd_stage_cells(4, dims, b, l, s)))
    out.append(_download(ctx, bst.d_up_proj_out, bwd_stage_cells(5, dims, b, l, s)))
    out.append(_download(ctx, bst.d_gate_proj_out, bwd_stage_cells(6, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_gate, bwd_stage_cells(7, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_up, bwd_stage_cells(8, dims, b, l, s)))
    out.append(_download(ctx, bst.d_norm2_out, bwd_stage_cells(9, dims, b, l, s)))
    out.append(_download(ctx, bst.norm2_dot, bwd_stage_cells(10, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_norm2, bwd_stage_cells(11, dims, b, l, s)))
    out.append(_download(ctx, bst.norm2_dx, bwd_stage_cells(12, dims, b, l, s)))
    out.append(_download(ctx, bst.d_residual1, bwd_stage_cells(13, dims, b, l, s)))
    out.append(_download(ctx, bst.d_o_proj_out, bwd_stage_cells(14, dims, b, l, s)))
    out.append(_download(ctx, bst.d_attn_ctx, bwd_stage_cells(15, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_o, bwd_stage_cells(16, dims, b, l, s)))
    out.append(_download(ctx, bst.d_attn_weights, bwd_stage_cells(17, dims, b, l, s)))
    out.append(_download(ctx, bst.attn_zdot, bwd_stage_cells(18, dims, b, l, s)))
    out.append(_download(ctx, bst.d_attn_masked, bwd_stage_cells(19, dims, b, l, s)))
    out.append(_download(ctx, bst.d_attn_scores, bwd_stage_cells(20, dims, b, l, s)))
    out.append(_download(ctx, bst.d_qk_cell, bwd_stage_cells(21, dims, b, l, s)))
    out.append(_download(ctx, bst.d_q_rope, bwd_stage_cells(22, dims, b, l, s)))
    out.append(_download(ctx, bst.d_k_cache, bwd_stage_cells(23, dims, b, l, s)))
    out.append(_download(ctx, bst.d_v_cache, bwd_stage_cells(24, dims, b, l, s)))
    out.append(_download(ctx, bst.d_k_rope, bwd_stage_cells(25, dims, b, l, s)))
    out.append(_download(ctx, bst.d_v_proj_out, bwd_stage_cells(26, dims, b, l, s)))
    out.append(_download(ctx, bst.d_q_proj_out, bwd_stage_cells(27, dims, b, l, s)))
    out.append(_download(ctx, bst.d_k_proj_out, bwd_stage_cells(28, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_q, bwd_stage_cells(29, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_k, bwd_stage_cells(30, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_v, bwd_stage_cells(31, dims, b, l, s)))
    out.append(_download(ctx, bst.d_norm1_out, bwd_stage_cells(32, dims, b, l, s)))
    out.append(_download(ctx, bst.norm1_dot, bwd_stage_cells(33, dims, b, l, s)))
    out.append(_download(ctx, bst.dw_norm1, bwd_stage_cells(34, dims, b, l, s)))
    out.append(_download(ctx, bst.norm1_dx, bwd_stage_cells(35, dims, b, l, s)))
    out.append(_download(ctx, bst.d_x, bwd_stage_cells(36, dims, b, l, s)))
    return out^


def run_device_backward(
    ctx: DeviceContext,
    c: FixtureCase,
    dims: TransformerDims,
    w: TransformerWeights,
    x: List[Float32],
    d_out: List[Float32],
    b: Int,
    l: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises -> List[List[Float32]]:
    """ONE forward and ONE backward on the device, from FRESH everything,
    thirty-seven stages returned on the host.

    The forward runs with the trace DISABLED and only the backward records.
    That is deliberate and it is not a cost decision: a single trace holding
    both cards is a fine thing for a human to read, and a card a ROUND JUDGE
    reads must hold the thirty-seven tags of plan section 6.1 and nothing
    else. `tools/e3_round_judge.sh` counts records.

    **`x_dev` IS PASSED TO THE BACKWARD AND THIS IS PLAN SECTION 1's
    CROSS-LANE REQUEST IN ACTION.** `LlamaDeviceStages` holds no `x` field --
    the forward launcher uploads `x`, uses it, and lets it die -- so the
    backward takes the block input as an EXPLICIT ARGUMENT and the caller
    owns it. `[[mojo-buffer-freed-at-last-use]]`: `dx` is a local here and is
    still alive when the backward reads it, because the explicit `_ = dx^` at
    the foot keeps the owner alive past every `.unsafe_ptr()` it hands out.
    Parking `x` in a backward scratch field instead would have been the wrong
    fix and the plan says it nearly was: at the point the first norm's
    backward runs, every scratch buffer still holds a live gradient term, and
    reusing one would have read a gradient where a block input belongs -- a
    plausible, in-bounds, WRONG `dx`."""
    var ldims = llama_dims_of(dims)
    var cap = c.cache_cap
    if cap < l:
        cap = l
    var dw = LlamaDeviceWeights(
        ctx, ldims, RMS_EPS, w.norm1_w, w.norm2_w, w.w_q, w.w_k, w.w_v,
        w.w_o, w.w_gate, w.w_up, w.w_down,
    )
    var kv = LlamaKVCache(ctx, b, ldims, cap)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var stages = LlamaDeviceStages(ctx, b, l, cap, ldims)
    var dx = _upload(ctx, x)
    var off = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, stages, kv, rope, dw, dx, b, l, 0, PLANT_AT_NONE,
        List[Int](), List[UInt32](), off, "fwd",
    )
    var bst = LlamaBackwardStages(ctx, b, l, cap, ldims)
    llama_decoder_layer_backward(
        ctx, bst, stages, dw, rope.cos, rope.sin, dx, d_out, b, l, 0,
        trace, prefix,
    )
    var out = backward_device_dump(ctx, bst, dims, b, l, l)
    _ = bst^
    _ = stages^
    _ = dw^
    _ = kv^
    _ = rope^
    _ = dx^
    return out^


def run_host_backward(
    c: FixtureCase,
    dims: TransformerDims,
    w: TransformerWeights,
    x: List[Float32],
    d_out: List[Float32],
    b: Int,
    l: Int,
) raises -> List[List[Float32]]:
    """The oracle's thirty-seven stages for the same case, card order.

    The forward is run FIRST and its saved stages are handed to the
    backward, because that is what a backward call IS -- plan section 1's
    saved set is `TransformerStages` and nothing else. A gate that fed the
    backward synthetic stages would be gating a function nobody calls."""
    var cache = TransformerKVCache(b, dims, c.cache_cap if c.cache_cap >= l else l)
    var rope = build_rope_table(dims)
    var plant = ScorePlant.none()
    var fwd = transformer_block_oracle(w, x, b, l, cache, rope, plant)
    var st = transformer_block_backward_oracle(w, fwd, d_out, b, l, 0, rope)
    var out = backward_oracle_dump(st)
    _ = st^
    _ = fwd^
    return out^


# ===========================================================================
# COMPARING
# ===========================================================================


@fieldwise_init
struct StageDiff(Copyable, Movable):
    var name: String
    var n_cells: Int
    var n_diff: Int
    var first: Int


def compare_stage(
    name: String, host: List[Float32], dev: List[Float32], loud: Bool
) raises -> StageDiff:
    """Bitwise, cell by cell.

    A LENGTH mismatch is reported as such rather than compared to the
    shorter of the two, because a stage that is the wrong SIZE is a
    different defect from a stage that is the wrong VALUE, and comparing to
    the shorter of two lists is how a truncated stage passes.

    **BY BITS AND NEVER BY COMPARES.** `host[i] == dev[i]` would call `+0.0`
    and `-0.0` equal, which would launder plan 6.2(e)'s whole signed-zero
    half -- the masked cells of `bwd.d_attn_scores` have predicted sign
    `sign(dy_j - z)` -- and Metal flushes compare operands (IDENTITY_PATHS
    row 49) so a compare-written gate has a different meaning on different
    columns."""
    if len(host) != len(dev):
        raise Error(
            String("transformer_backward_check: stage ")
            + name
            + " has "
            + String(len(host))
            + " cells on one side and "
            + String(len(dev))
            + " on the other. A stage of the wrong SHAPE is a different"
            + " defect from a stage of the wrong VALUE and is not compared"
            + " to the shorter of the two."
        )
    var n_diff = 0
    var first = -1
    for i in range(len(host)):
        if bitcast[DType.uint32](host[i]) != bitcast[DType.uint32](dev[i]):
            n_diff += 1
            if first < 0:
                first = i
    if n_diff == 0:
        if loud:
            print("  OK    " + name + "  (" + String(len(host)) + " cells)")
    elif loud:
        print(
            "  MOVED "
            + name
            + "  "
            + String(n_diff)
            + " of "
            + String(len(host))
            + " cells, first cell "
            + String(first)
            + "  a "
            + hexbits(host[first])
            + "  b "
            + hexbits(dev[first])
        )
    return StageDiff(name, len(host), n_diff, first)


def compare_dumps(
    a: List[List[Float32]], b: List[List[Float32]], loud: Bool
) raises -> List[StageDiff]:
    """Thirty-seven stages, in card order, by tag.

    **THE FIRST ASSERTION IS THE LENGTH OF THE DUMPS THEMSELVES**, which is
    what `backward_oracle_dump`'s docstring asks the check file for in as
    many words: "the check file's FIRST assertion should be that the two
    lengths agree, before any hash is compared". Two dumps of different
    lengths mean the two halves of this lane disagree about what a stage
    list IS, and every per-stage comparison after that would be comparing
    misaligned tags -- which does not produce a smaller diff, it produces a
    plausible WRONG answer."""
    if (
        len(a) != TRANSFORMER_BACKWARD_STAGE_COUNT
        or len(b) != TRANSFORMER_BACKWARD_STAGE_COUNT
    ):
        raise Error(
            String("transformer_backward_check: the host dump has ")
            + String(len(a))
            + " stages and the device dump has "
            + String(len(b))
            + ", and plan section 6.1 lists "
            + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
        )
    var out = List[StageDiff]()
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        out.append(compare_stage(backward_stage_tag(i), a[i], b[i], loud))
    return out^


def count_moved(diffs: List[StageDiff]) -> Int:
    var n = 0
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            n += 1
    return n


def total_cells(dump: List[List[Float32]]) -> Int:
    var n = 0
    for i in range(len(dump)):
        n += len(dump[i])
    return n


def first_moved_index(diffs: List[StageDiff]) -> Int:
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return i
    return -1


def first_moved(diffs: List[StageDiff]) -> String:
    """Plan section 6.3's own report shape: `<tag> on X of Y cells`. The
    discipline is that an arm moves the stage its OWN seam writes "and no
    earlier one", so the FIRST moved stage is the finding and the count is
    the evidence that it is not a single-bit coincidence."""
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return (
                diffs[i].name
                + " on "
                + String(diffs[i].n_diff)
                + " of "
                + String(diffs[i].n_cells)
                + " cells"
            )
    return String("")


def stage_index_of(tag: String) raises -> Int:
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if backward_stage_tag(i) == tag:
            return i
    raise Error(
        String("transformer_backward_check: '")
        + tag
        + "' is not one of plan section 6.1's thirty-seven tags"
    )


def check_card_tags(path: String) raises -> Int:
    """The card holds plan 6.1's thirty-seven tags, in plan order, each
    exactly once, each carrying this driver's prefix.

    A CHECK ON THE COMPOSITION AND NOT ON THE ARITHMETIC, **and it is the
    check that catches a card that was never written**, which was DEVIATION
    970's actual damage: `read_trace_lines` on a path nothing wrote RAISES,
    rather than returning an empty list that a lenient gate would call zero
    differences."""
    var lines = read_trace_lines(path)
    print(
        "card: "
        + String(len(lines))
        + " records at "
        + path
        + ", plan section 6.1 wants "
        + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
    )
    if len(lines) != TRANSFORMER_BACKWARD_STAGE_COUNT:
        raise Error(
            String("transformer_backward_check: the card has ")
            + String(len(lines))
            + " records and plan section 6.1 lists "
            + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
        )
    for i in range(len(lines)):
        var fields = lines[i].split("\t")
        if len(fields) < 2:
            raise Error(
                String("transformer_backward_check: malformed trace record: ")
                + lines[i]
            )
        var got = String(fields[1])
        var expect = String(TAG_PREFIX) + "." + backward_stage_tag(i)
        if got != expect:
            raise Error(
                String("transformer_backward_check: card record ")
                + String(i)
                + " is '"
                + got
                + "', plan section 6.1 wants '"
                + expect
                + "'. A renamed or reordered tag does not make"
                + " identity_trace_diff.py's diff smaller, it makes its"
                + " ALIGNMENT wrong."
            )
    print(
        "card: "
        + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
        + "/"
        + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
        + " tags in plan section 6.1's order, all unique. Stage 0 is the"
        " INCOMING GRADIENT and it is on the card deliberately: two cards"
        " whose inputs differ are diffing their fixtures"
    )
    return len(lines)


# ===========================================================================
# THE CUTS: a row slice and a token view, per stage kind
#
# Clauses (c) and (d) are nothing but row slices and token views, and a
# whole-buffer compare cannot even be SPELLED across two different `B` or
# two different `L` -- the buffers are different lengths. That is precisely
# why the batch-major and attention-shaped stages are the ones to watch:
# they are the ones a batch- or length-dependent bug would live in.
# ===========================================================================


def row_slice(
    values: List[Float32],
    i: Int,
    bb: Int,
    b: Int,
    l: Int,
    s: Int,
    dims: TransformerDims,
) raises -> List[Float32]:
    """The cells of stage `i` belonging to batch row `bb`.

    The WEIGHT gradients have no batch axis and are returned WHOLE. That is
    not a convenience: plan 5.4 says they are a sum over `M = B*L` and must
    MOVE under a batch-composition change, so returning them whole is what
    lets clause (c)'s negative half compare them at all.

    `[[mojo-list-copy]]`: `.copy()` and not `var a = b`. A `List[Float32]`
    is not implicitly copyable and `var a = b` fails to compile with "value
    of type 'List[Float32]' cannot be implicitly copied", which cost the
    gemm backward lane a build."""
    var kind = bwd_stage_kind(i)
    if kind == K_WEIGHT:
        return values.copy()
    var w: Int
    if kind == K_TOKEN:
        w = l * bwd_token_width(i, dims)
    elif kind == K_ATTNROW:
        w = dims.n_heads * l * s
    elif kind == K_ATTNSCL:
        w = dims.n_heads * l
    else:
        w = dims.n_kv_heads * s * dims.head_dim
    var out = List[Float32](capacity=w if w > 0 else 1)
    for j in range(w):
        out.append(values[bb * w + j])
    return out^


def token_view(
    values: List[Float32],
    i: Int,
    b: Int,
    l: Int,
    s: Int,
    qi: Int,
    keep: Int,
    dims: TransformerDims,
) raises -> List[Float32]:
    """The cells of stage `i` that belong to query token `qi`, restricted to
    the first `keep` positions of the KEY axis.

    **THE `keep` ARGUMENT IS WHY THIS EXISTS RATHER THAN A SIMPLER TOKEN
    SLICER.** Two runs at different `L` do not agree on the length of the
    key axis, and four of the thirty-seven stages are laid out
    `[B, n_heads, L, S]` with `S` in the innermost stride. Comparing whole
    rows would compare a row of `t+1` cells against a row of `S`, which
    `compare_stage` correctly refuses as a length mismatch. Comparing the
    PREFIX is the comparison plan 5.1 actually licenses: it proves every
    masked cell's contribution is a signed zero at all four folds and
    bitwise inert in a `+0.0`-seeded chain, so the prefix is where the claim
    lives and the tail is where the theorem lives.

    `attn.zdot` needs no `keep` and is compared WHOLE, which is the SHARPER
    test of the two: plan 5.1(ii) says every masked term of the `z` fold is
    exactly `+0.0` because `y_j` is, so a `z` folded over a different number
    of terms shows up there and nowhere else.

    The WEIGHT gradients have no token axis and are returned whole, for the
    same reason `row_slice` returns them whole."""
    var kind = bwd_stage_kind(i)
    var nh = dims.n_heads
    var nkv = dims.n_kv_heads
    var hd = dims.head_dim
    var out = List[Float32]()
    if kind == K_WEIGHT:
        return values.copy()
    if kind == K_TOKEN:
        var w = bwd_token_width(i, dims)
        for bb in range(b):
            var start = (bb * l + qi) * w
            for j in range(w):
                out.append(values[start + j])
        return out^
    if kind == K_ATTNROW:
        for bb in range(b):
            for h in range(nh):
                var base = ((bb * nh + h) * l + qi) * s
                for j in range(keep):
                    out.append(values[base + j])
        return out^
    if kind == K_ATTNSCL:
        for bb in range(b):
            for h in range(nh):
                out.append(values[(bb * nh + h) * l + qi])
        return out^
    # K_KV: the first `keep` SLOTS, per (batch, kv head). A cache-gradient
    # slot is an ABSOLUTE key position, so `keep` here counts positions and
    # not queries.
    for bb in range(b):
        for kvh in range(nkv):
            for j in range(keep):
                for dd in range(hd):
                    out.append(values[((bb * nkv + kvh) * s + j) * hd + dd])
    return out^


def is_query_axis_stage(i: Int) raises -> Bool:
    """Does stage `i` fold over the QUERY axis?

    **DEVIATION 1529 IS THIS FUNCTION.** Plan 6.2's clause (c) lists
    `bwd.d_x` and `bwd.d_norm1_out` among the length-INVARIANT activation
    gradients, and the plan's own section 5.2 says the opposite: a key at
    slot `j` is read by every query at position `>= j`, so `bwd.d_k_cache`
    and `bwd.d_v_cache` gain contributors when the sequence lengthens, and
    `bwd.d_k_proj_out` / `bwd.d_v_proj_out` are their `[pos0, S)` slices,
    and `bwd.d_norm1_out` is the THREE-TERM fan-in that includes both, and
    `bwd.norm1.dot`, `bwd.norm1.dx` and `bwd.d_x` are all downstream of it.

    Every WEIGHT gradient is here too, because a weight gradient IS the sum
    over the tokens in the call and plan 5.2 says no spelling makes it
    otherwise.

    **THE OTHER HALF OF THE PLAN IS RIGHT AND THIS FUNCTION IS NOT USED FOR
    IT.** For BATCH composition `bwd.d_k_cache` is `[B, n_kv, S, hd]`, so
    batch row `bb`'s cache gradient sums only `bb`'s own queries and IS
    invariant; only the WEIGHT gradients move there. The two halves of
    clause (c) have DIFFERENT moving sets and a gate that used one set for
    both would be wrong twice."""
    if bwd_stage_kind(i) == K_WEIGHT:
        return True
    if i == 23 or i == 24 or i == 25 or i == 26 or i == 28:
        return True  # d_k_cache, d_v_cache, d_k_rope, d_v_proj_out,
        # d_k_proj_out
    if i == 32 or i == 33 or i == 35 or i == 36:
        return True  # d_norm1_out, norm1.dot, norm1.dx, d_x
    return False


def moved_cells_at_position_zero(
    host: List[Float32],
    dev: List[Float32],
    i: Int,
    b: Int,
    l: Int,
    pos0: Int,
    dims: TransformerDims,
) raises -> Int:
    """How many differing cells of a TOKEN-major stage belong to ABSOLUTE
    POSITION 0.

    **PLAN 6.3 ASKS FOR EXACTLY THIS AND ASKS FOR IT BY NAME.** The three
    RoPE arms are "INERT AT ABSOLUTE POSITION 0, where `sin` is exactly
    `+0.0` -- **per CELL, not per fixture, so the gate must COUNT moved
    cells and RAISE if the count equals the position-0 population**". An
    arm that moved a position-0 cell has not been caught by a coincidence;
    it has been caught doing something the rotation cannot do, because at
    `sin == +0.0` and `cos == 1.0` the transpose IS the forward and IS the
    identity."""
    if bwd_stage_kind(i) != K_TOKEN:
        raise Error(
            "transformer_backward_check: the position-0 mask is only"
            " defined for a token-major stage"
        )
    if pos0 != 0:
        return 0
    var w = bwd_token_width(i, dims)
    var n = 0
    for bb in range(b):
        var start = (bb * l + 0) * w
        for j in range(w):
            if bitcast[DType.uint32](host[start + j]) != bitcast[
                DType.uint32
            ](dev[start + j]):
                n += 1
    return n


def masked_negative_zero_count(
    d_attn_scores: List[Float32],
    b: Int,
    l: Int,
    s: Int,
    pos0: Int,
    dims: TransformerDims,
) -> Int:
    """The ORACLE'S PREDICTION of how many cells `B13_MASK_ZEROES_GRAD`
    moves, computed BEFORE the device is asked.

    Plan 6.3: the arm "Moves ONLY masked cells whose `dS` is `-0.0`, i.e.
    where `dy_j - z < 0`. Predicted to move roughly half the masked cells,
    and **the oracle must predict the exact count before the device is
    asked**."

    A masked cell is one whose key index `j` exceeds the query's absolute
    position `pos0 + t`. At such a cell the pinned mask backward is the
    exact identity, so `dS` is `pinned_mul(y_j, dy_j - z)` with `y_j`
    exactly `+0.0` -- a signed zero whose sign is `sign(dy_j - z)`. The arm
    writes `+0.0` there instead, so it moves exactly the cells whose pinned
    value is `-0.0` and no others. **This is what makes the arm a
    measurement rather than a smoke test**, and a gate that only asserted
    "it moved something" would pass on an arm that zeroed the whole
    buffer."""
    var nh = dims.n_heads
    var n = 0
    for bb in range(b):
        for h in range(nh):
            for t in range(l):
                var base = ((bb * nh + h) * l + t) * s
                for j in range(s):
                    if j <= pos0 + t:
                        continue
                    if bitcast[DType.uint32](
                        d_attn_scores[base + j]
                    ) == UInt32(0x80000000):
                        n += 1
    return n


# ===========================================================================
# PREFLIGHT
#
# DEVIATION 1533. Everything here is cheap to check and catastrophic to get
# wrong, because a wrong constant does not look like a wrong constant -- it
# looks like a kernel bug on every stage downstream of it. They run BEFORE
# any device call, so a build with a bad constant or a blind fixture fails
# in a second rather than after a case sweep.
# ===========================================================================


def preflight() raises:
    print("preflight: the assertions the plan and the oracles asked for")

    # ---- 1. The FORWARD profile's three frozen scalars ------------------
    # The backward reads the SAVED forward stages, so a wrong forward
    # constant is a wrong backward on every stage with no separate symptom.
    if not profile_constants_are_intact():
        raise Error(
            String("transformer_backward_check: a FROZEN profile constant")
            + " has the wrong bits. rms_eps "
            + hexbits(RMS_EPS)
            + " wants 0x358637bd; rope_theta "
            + hexbits(ROPE_THETA)
            + " wants 0x461c4000; mask_fill "
            + hexbits(mask_fill())
            + " wants 0xff7fffff. This is a v2, not a bug."
        )
    print(
        "  frozen forward constants OK: eps "
        + hexbits(RMS_EPS)
        + ", theta "
        + hexbits(ROPE_THETA)
        + ", mask fill "
        + hexbits(mask_fill())
    )

    # ---- 2. The RMSNorm backward's TWO EXACT SCALINGS -------------------
    # DEVIATION 1409. `-0.5` and `2.0`. Both are exact powers of two so
    # neither introduces a rounding, and NEITHER IS BIT INERT -- which is
    # why both are SPELLED in the oracle rather than elided. Contract S8
    # drops `* attention_scaling` because it is exactly `1.0` and inert on
    # every input including both zero signs, and **that argument does not
    # transfer to a value that changes the number.** The two also do not
    # cancel: `-0.5 * z` then `* 2.0` is `-z` for every finite `z` whose
    # halving stays out of the flush region, and the two-step spelling is
    # kept because the reference's autograd GRAPH has two steps.
    if bits_of(BWD_NEG_HALF) != UInt32(0xBF000000):
        raise Error(
            String("transformer_backward_check: BWD_NEG_HALF is ")
            + hexbits(BWD_NEG_HALF)
            + " and wants 0xbf000000 (-0.5 exactly). DEVIATION 1409 rests"
            + " on it being an exact power of two so that it introduces no"
            + " rounding."
        )
    if bits_of(BWD_TWO) != UInt32(0x40000000):
        raise Error(
            String("transformer_backward_check: BWD_TWO is ")
            + hexbits(BWD_TWO)
            + " and wants 0x40000000 (2.0 exactly)"
        )
    var probe = f32_from_bits(UInt32(0x3FC90FDB))  # pi/2, full mantissa
    var half_then_two = ftz(
        identical_mul(ftz(identical_mul(probe, BWD_NEG_HALF)), BWD_TWO)
    )
    if bits_of(half_then_two) != (bits_of(probe) ^ UInt32(0x80000000)):
        raise Error(
            String("transformer_backward_check: (-0.5 * x) * 2.0 is not")
            + " exactly -x at "
            + hexbits(probe)
            + " (got "
            + hexbits(half_then_two)
            + "). DEVIATION 1409's 'neither introduces a rounding' is then"
            + " false on this toolchain and the two-step spelling is not"
            + " free."
        )
    print(
        "  RMSNorm backward scalings: BWD_NEG_HALF "
        + hexbits(BWD_NEG_HALF)
        + ", BWD_TWO "
        + hexbits(BWD_TWO)
        + ", and (-0.5 * x) * 2.0 == -x EXACTLY at a full-mantissa probe"
    )

    # ---- 3. Plan 5.1(i), the property EVERY inertness claim rests on ----
    # "A `+0.0`-seeded fma chain never holds `-0.0`." `fma(a, b, acc)`
    # returns `-0.0` only when the exact value `a*b + acc` is a zero of
    # negative sign, and under round-to-nearest that happens only if both
    # `a*b` and `acc` are `-0.0` (an exact cancellation of two nonzero
    # opposites gives `+0.0`). The seed is `+0.0`, so the property is
    # preserved at every step.
    #
    # **THIS IS ALSO DEVIATION 1528's PROOF**, which is why it is asserted
    # rather than quoted: it is exactly why `B_FANIN_ZERO_SEED` has no
    # constructible witness in this lane.
    var neg = f32_from_bits(BITS_NEG_ZERO)
    var pos = f32_from_bits(BITS_POS_ZERO)
    var one = f32_from_bits(UInt32(0x3F800000))
    var probes: List[Float32] = [pos, neg, one, -one, probe, -probe]
    var bad = 0
    for i in range(len(probes)):
        for j in range(len(probes)):
            var r = ftz(identical_mul_add(probes[i], probes[j], pos))
            if bits_of(r) == UInt32(0x80000000):
                bad += 1
    if bad != 0:
        raise Error(
            String("transformer_backward_check: a +0.0-seeded fma produced")
            + " -0.0 on "
            + String(bad)
            + " of "
            + String(len(probes) * len(probes))
            + " probes. Plan 5.1(i) says that is impossible, and EVERY"
            + " masked-tail inertness claim in this lane rests on it."
        )
    if bits_of(ftz(identical_mul_add(one, neg, neg))) != UInt32(0x80000000):
        raise Error(
            "transformer_backward_check: fma(1.0, -0.0, -0.0) did NOT give"
            " -0.0, so plan 5.1(i)'s proof has no negative half and the"
            " assertion above is vacuous -- it would pass on a toolchain"
            " that never produces -0.0 at all ([[reached-but-inert]])."
        )
    print(
        "  plan 5.1(i): a +0.0-seeded fma never returns -0.0 on "
        + String(len(probes) * len(probes))
        + " probes INCLUDING both zero signs, and fma(1.0, -0.0, -0.0) DOES"
        " return -0.0, so the assertion is not vacuous"
    )

    # ---- 4. The softmax backward at S == 1, the arms' inert case --------
    # `B18_SOFTMAX_DECOMPOSED`, `B18_ZFOLD_UNFUSED` and
    # `B18_ZFOLD_DESCENDING` are all predicted INERT at `s == 1`, and
    # `base_b1_l1_nrep2` is the fixture that is supposed to deliver it. The
    # MECHANISM is asserted here rather than the mask: at `s == 1` the
    # softmax weight is exactly `1.0`, so `z = dy` and `dS = 1.0 * (dy - z)`
    # is a zero -- and a one-term fold has no order and nothing to fuse.
    # **A control whose mechanism does not exist in the case is one of the
    # five ways a fixture went blind on 2026-08-25**, so the mechanism is
    # what gets checked.
    var dy1: List[Float32] = [probe]
    var y1: List[Float32] = [one]
    var z1 = List[Float32]()
    var ds1 = List[Float32]()
    z1.append(pos)
    ds1.append(pos)
    softmax_backward_into(dy1, y1, 1, 1, 1, 1, z1, ds1)
    if bits_of(ds1[0]) != UInt32(0x00000000):
        raise Error(
            String("transformer_backward_check: the softmax backward at")
            + " s == 1 gave dS = "
            + hexbits(ds1[0])
            + " and not +0.0. The three S18 arms all take"
            + " base_b1_l1_nrep2 as their INERT case on the strength of"
            + " that, and if it is false the masks are wrong"
            + " ([[reached-but-inert]])."
        )
    print(
        "  softmax backward at s == 1: z = dy exactly and dS = "
        + hexbits(ds1[0])
        + ", so the three S18 arms' INERT case has the mechanism it claims"
    )

    # ---- 5. THE INCOMING GRADIENT SEPARATES ----------------------------
    var c0 = fixture_case(0)
    print("  " + guard_d_out_separates(c0, fixture_dims(c0)))


# ===========================================================================
# THE CASE SET
# ===========================================================================


def clause_a_cases() raises -> List[Int]:
    """The default clause-(a) set: every SINGLE-CALL fixture case whose kv
    length is at most gemm v1's `CONTRACT_K_LEAF_MIN` of 128.

    It is `transformer_check.mojo`'s set exactly, and it is the same set for
    the same two reasons, restated because a set copied without its argument
    is a set nobody checked:

      * `pos_offset_129` (case 6) makes TWO calls carrying the cache. That
        is structurally the CHUNKED path, not a prefill, and it is clause
        (d)'s case -- running it through clause (a) would be running clause
        (d)'s fixture under clause (a)'s expectations.
      * `long_l257` (case 8) has `S = 257 > 128`, so gemm v1's `P` is 3
        rather than 1 and `B11_DQ_VIA_GEMM` stops being inert. It is the
        ONLY fixture that can fire that arm, and it is opt-in because clause
        (a) there is four `[B, nh, L, S]` buffers of `1*2*257*257` cells per
        stage. `MOJOLEARN_TFB_CHECK_LONG=1` adds it, and under that arm
        clause (a) is then EXPECTED to move -- which is MORE evidence, not a
        contradiction, and clause (g) says so where the expectation is
        evaluated.

    What the set buys, because a set chosen for what it excludes is a set
    nobody checked for what it includes:

      case 0   base_b1_l4_nrep1    n_rep == 1, repeat_kv is the identity,
                                   and B11_DK_VIA_GEMM's INERT case
      case 1   base_b2_l4_nrep2    n_rep == 2, which fires B11_DK_VIA_GEMM
                                   and B19_DV_VIA_GEMM on SHAPE alone
      case 2   base_b1_l1_nrep2    S == 1, the three S18 arms' inert case
      case 3   base_b3_l16_nrep2   B == 3, clause (c)'s batch half
      case 4   wide_inter300       down_proj at k = 300, the only case that
                                   runs gemm v1's balanced fold tree at all
      case 5   odd_head_dim_24     the INEXACT attention scale, and the only
                                   shape that can fire B12_SCALE_INTO_DQ
      case 7   long_l64            L == 64
      case 9   adv_signed_zeros    -0.0 through the whole forward
      case 10  adv_subnormal_x     the ftz unit at the LOAD seams
      case 11  adv_score_neg_zero  signed zeros in the score buffer
      case 12  adv_score_extreme   a planted extreme score
      case 13  adv_cache_hot_tail  the fold walks [0, used), not [0, cap)
      case 14  adv_masked_zero_row a whole masked row
    """
    var out: List[Int] = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]
    if env_on("MOJOLEARN_TFB_CHECK_LONG"):
        out.append(8)
    return out^


@fieldwise_init
struct CaseVerdict(Copyable, Movable):
    """One case's clause-(a) result, kept so that clause (g)'s expectation
    table can be evaluated ACROSS cases in one binary. That is what makes an
    arm's INERT case expressible at all: it needs two cases in one process,
    and `B12_SCALE_INTO_DQ` firing on `odd_head_dim_24` while staying inert
    on `base_b1_l4_nrep1` is the whole difference between a reach proof and
    a smoke test."""

    var name: String
    var n_moved: Int
    var first_index: Int
    var first: String
    var cells: Int
    var pos0_moved: Int
    """Moved cells of `bwd.d_q_proj_out` at ABSOLUTE POSITION 0, for the
    three RoPE arms' PER-CELL inert mask (plan 6.3)."""
    var mask_moved: Int
    """Moved cells of `bwd.d_attn_scores`, for `B13_MASK_ZEROES_GRAD`."""
    var mask_predicted: Int
    """What the ORACLE predicted that count would be, computed BEFORE the
    device was asked."""


def clause_a_case(
    ctx: DeviceContext, k: Int, mut trace: IdentityTrace, prefix: String
) raises -> CaseVerdict:
    """Plan 6.2 clause (a) at ONE fixture case, all thirty-seven stages.

    **WHAT WOULD MAKE THIS PASS WHILE GATING NOTHING. Three things, and two
    of them are closed here:**

      1. A device dump and an oracle dump that are the same object. They are
         not: they come from two functions in two files reading two structs
         (`LlamaBackwardStages` and `TransformerBackwardStages`) that share
         no code.
      2. Two dumps of different LENGTHS compared to the shorter of the two.
         `compare_dumps` checks the stage COUNT and `compare_stage` raises
         on a per-stage length mismatch.
      3. **Our oracle being wrong in the same way as our device. THIS IS NOT
         CLOSED AND CANNOT BE CLOSED HERE.** Both halves are ours.
         `transformer/corpus/` does not exist and plan 6.2(f) is explicit
         that a TRANSPOSE ERROR IS BIT IDENTICAL ON THREE VENDORS -- so
         clause (a) can be perfectly green on a gradient that is not the
         gradient. Clause (f) is the partial answer and the float64
         directional derivative is the real one, and it does not exist."""
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var d_out = bwd_d_out(c, dims, DOUT_PLANT_NONE)

    var host = run_host_backward(c, dims, w, x, d_out, c.b, c.l)
    var dev = run_device_backward(
        ctx, c, dims, w, x, d_out, c.b, c.l, trace, prefix
    )
    var diffs = compare_dumps(host, dev, False)
    var moved = count_moved(diffs)
    var fi = first_moved_index(diffs)
    var fname = String("")
    if fi >= 0:
        fname = backward_stage_tag(fi)
    var cells = total_cells(host)
    var pz = moved_cells_at_position_zero(
        host[27], dev[27], 27, c.b, c.l, 0, dims
    )
    var predicted = masked_negative_zero_count(
        host[20], c.b, c.l, c.l, 0, dims
    )
    var line = (
        "  case "
        + String(k)
        + " "
        + String(c.name)
        + "  B="
        + String(c.b)
        + " L="
        + String(c.l)
        + " S="
        + String(c.l)
        + " dm="
        + String(c.d_model)
        + " nh="
        + String(c.n_heads)
        + " nkv="
        + String(c.n_kv_heads)
        + " hd="
        + String(c.head_dim)
        + " inter="
        + String(c.intermediate)
        + "  "
        + String(cells)
        + " cells: "
    )
    if moved == 0:
        print(line + "37/37 stages bit-identical")
    else:
        print(
            line
            + String(moved)
            + " of 37 stages MOVED, first at "
            + first_moved(diffs)
        )
        # The per-stage detail, printed only when something moved, because
        # thirty-seven OK lines per case across fourteen cases is 518 lines
        # of nothing and the one line that matters would be lost in it.
        _ = compare_dumps(host, dev, True)
    return CaseVerdict(
        String(c.name),
        moved,
        fi,
        fname,
        cells,
        pz,
        diffs[20].n_diff,
        predicted,
    )


# ===========================================================================
# CLAUSE (b): the same bits on every one of eight repeated launches
# ===========================================================================


def clause_b(ctx: DeviceContext, k: Int) raises:
    """Plan 6.2 clause (b). Eight launches, each with its OWN fresh weights,
    cache, rotary table, forward stage buffers, BACKWARD stage buffers,
    input upload and kernel dispatches, every stage compared to the first on
    every cell.

    Fresh EVERYTHING and not just a fresh call, because the failure this
    clause is for is an execution plan that is not a pure function of the
    input -- a scratch buffer read before it is written, an accumulator that
    survives a call, a launch geometry chosen from a clock. **The backward
    has ELEVEN untagged scratch fields** (`dh`, `dprod`, `rstd`, `dvcoef`,
    `ones`, `tmp0`, `tmp1`, `tmp2`, `head_a`, `head_b`, `head_c`) and every
    one of them is exactly the kind of buffer a re-call would hide, so this
    clause matters more here than it did in the forward.

    **WHAT WOULD MAKE IT PASS WHILE GATING NOTHING: comparing only
    `bwd.d_x`.** The forward lane MEASURED thirteen moved stages being
    absorbed by a residual add. Every one of the thirty-seven is compared on
    every cell.

    **AND THE ONE HOLE NO CONTROL CLOSES:** eight identical runs of a
    deterministically WRONG gradient pass it. Clause (b) is an invariance
    claim and says nothing about correctness."""
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var d_out = bwd_d_out(c, dims, DOUT_PLANT_NONE)
    print(
        "clause (b): "
        + String(CLAUSE_B_LAUNCHES)
        + " repeated launches of "
        + String(c.name)
        + ", every stage, every cell, fresh state each time"
    )
    var off1 = IdentityTrace.disabled()
    var base = run_device_backward(
        ctx, c, dims, w, x, d_out, c.b, c.l, off1, "b1"
    )
    var cells = total_cells(base)
    for run in range(2, CLAUSE_B_LAUNCHES + 1):
        var off = IdentityTrace.disabled()
        var got = run_device_backward(
            ctx, c, dims, w, x, d_out, c.b, c.l, off, "b" + String(run)
        )
        var diffs = compare_dumps(base, got, False)
        if count_moved(diffs) != 0:
            raise Error(
                String("transformer_backward_check: CLAUSE (b) FAILED,")
                + " launch "
                + String(run)
                + " differs from launch 1 at "
                + first_moved(diffs)
            )
    print(
        "clause (b): PASS, launches 2.."
        + String(CLAUSE_B_LAUNCHES)
        + " bit-identical to launch 1 on all "
        + String(cells)
        + " cells of all 37 stages"
    )


# ===========================================================================
# CLAUSE (c), FIRST HALF: BATCH COMPOSITION
# ===========================================================================


def clause_c_batch(ctx: DeviceContext, k: Int) raises:
    """Plan 6.2 clause (c), the batch half, and plan section 5.4.

    **THE POSITIVE HALF.** Nothing in plan section 3 reads `B`, so every
    ACTIVATION gradient is batch-composition invariant by construction and
    the gate exists to catch the construction being violated by an execution
    plan. That includes `bwd.d_k_cache` and `bwd.d_v_cache`, which are
    indexed `[B, n_kv, S, hd]` -- batch row `bb`'s cache gradient sums only
    `bb`'s own queries, so it IS invariant here even though it is NOT
    length-invariant. **The two halves of clause (c) have different moving
    sets and this is where they part.**

    **THE NEGATIVE HALF, WHICH THE PLAN INSISTS ON AND WITHOUT WHICH THE
    CLAUSE IS HALF A CLAUSE.** The eleven weight gradients' `k'` is
    `M = B*L`, so a launch carrying three sequences produces a DIFFERENT
    `dW` from three launches of one, and no spelling makes it otherwise --
    summing over more tokens IS the operation. Plan 6.2(c): "a gate that
    only asserts the positive half would pass on an implementation that
    computed constants". So the nine `bwd.dW_*` stages are asserted to MOVE,
    **and the HOST ORACLE PREDICTS THE MOVED CELL COUNT BEFORE THE DEVICE IS
    ASKED** -- an exact number, not "something moved", because an arm that
    zeroed the whole buffer would also move something.

    `IDENTICAL_SSM_NOTES` records that batch invariance for attention has
    been pursued in public (Thinking Machines' *Defeating Nondeterminism in
    LLM Inference*). **Batch invariance of a WEIGHT GRADIENT is not a thing
    anyone can have**, and this file must not be read as claiming a weaker
    version of a published result.

    **AND THE FIRING CONTROL, BECAUSE WITHOUT ONE THE POSITIVE HALF IS
    WORTHLESS.** If `row_slice` were wrong -- if it returned row 0 whatever
    row it was asked for -- every comparison would be a row against ITSELF
    and would pass for ever, on every vendor, hiding any batch dependence
    there is. So the clause first proves the slicer can tell two rows apart:
    rows 0 and 1 of the `B=2` run have different input tokens and different
    incoming gradients and must differ somewhere. A zero there RAISES and
    calls the clause VACUOUS rather than passing it."""
    var c = fixture_case(k)
    if c.b < 3:
        raise Error(
            String("transformer_backward_check: clause (c)'s batch half")
            + " needs a case with B >= 3 and "
            + String(c.name)
            + " has B="
            + String(c.b)
        )
    if c.plant != PLANT_NONE:
        raise Error(
            String("transformer_backward_check: clause (c)'s batch half")
            + " refuses the planted case "
            + String(c.name)
            + ". A score plant is a FLAT index into [B, n_heads, L, S], so"
            + " the same plant lands in a different cell at every B and the"
            + " comparison would be measuring the plant."
        )
    var dims = fixture_dims(c)
    var w = fixture_weights(c)
    var l = c.l
    var dm = c.d_model
    var x3 = fixture_x(c)
    var g3 = bwd_d_out(c, dims, DOUT_PLANT_NONE)
    var row_len = l * dm

    print(
        "clause (c) batch: "
        + String(c.name)
        + ", the same row at B=1, B=2 and B=3, plus the NEGATIVE half"
    )

    # ONE `x` and ONE `d_out` are generated at the case's B and SLICED, so
    # row 0's inputs are identical across the three compositions BY
    # CONSTRUCTION rather than by coincidence. Generating three cases would
    # give three seeds and nothing about them would be comparable.
    var x1 = List[Float32]()
    var g1 = List[Float32]()
    for i in range(row_len):
        x1.append(x3[i])
        g1.append(g3[i])
    var x2 = List[Float32]()
    var g2 = List[Float32]()
    for i in range(2 * row_len):
        x2.append(x3[i])
        g2.append(g3[i])

    var off1 = IdentityTrace.disabled()
    var d1 = run_device_backward(ctx, c, dims, w, x1, g1, 1, l, off1, "cb1")
    var off2 = IdentityTrace.disabled()
    var d2 = run_device_backward(ctx, c, dims, w, x2, g2, 2, l, off2, "cb2")
    var off3 = IdentityTrace.disabled()
    var d3 = run_device_backward(ctx, c, dims, w, x3, g3, 3, l, off3, "cb3")

    # ---- THE FIRING CONTROL ---------------------------------------------
    var control = 0
    var batched = 0
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if bwd_stage_kind(i) == K_WEIGHT:
            continue
        batched += 1
        var a0 = row_slice(d2[i], i, 0, 2, l, l, dims)
        var a1 = row_slice(d2[i], i, 1, 2, l, l, dims)
        var d = compare_stage("control", a0, a1, False)
        if d.n_diff > 0:
            control += 1
    if control == 0:
        raise Error(
            "transformer_backward_check: CLAUSE (c) BATCH IS VACUOUS. Rows"
            " 0 and 1 of the B=2 run are bit-identical on every activation"
            " stage, which cannot be true of two different input sequences"
            " with two different incoming gradients. `row_slice` is not"
            " cutting distinct rows, so every comparison below is a row"
            " against itself ([[reached-but-inert]])."
        )
    print(
        "clause (c) batch control: rows 0 and 1 of the B=2 run differ on "
        + String(control)
        + " of "
        + String(batched)
        + " activation stages, so the row slicer distinguishes rows"
    )

    # ---- THE POSITIVE HALF ----------------------------------------------
    var cells = 0
    var badp = 0
    var first_bad = String("")
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if bwd_stage_kind(i) == K_WEIGHT:
            continue
        var r0_1 = row_slice(d1[i], i, 0, 1, l, l, dims)
        var r0_2 = row_slice(d2[i], i, 0, 2, l, l, dims)
        var r0_3 = row_slice(d3[i], i, 0, 3, l, l, dims)
        cells += len(r0_1) * 2
        var a = compare_stage("row0 B1vB2", r0_1, r0_2, False)
        var bcmp = compare_stage("row0 B1vB3", r0_1, r0_3, False)
        if a.n_diff > 0 or bcmp.n_diff > 0:
            badp += 1
            if first_bad == "":
                first_bad = backward_stage_tag(i) + " row 0"
        var r1_2 = row_slice(d2[i], i, 1, 2, l, l, dims)
        var r1_3 = row_slice(d3[i], i, 1, 3, l, l, dims)
        cells += len(r1_2)
        var cc = compare_stage("row1 B2vB3", r1_2, r1_3, False)
        if cc.n_diff > 0:
            badp += 1
            if first_bad == "":
                first_bad = backward_stage_tag(i) + " row 1"
    if badp != 0:
        raise Error(
            String("transformer_backward_check: CLAUSE (c) BATCH FAILED on ")
            + String(badp)
            + " stages, first at "
            + first_bad
            + ": an ACTIVATION gradient's bits depend on who shares its"
            + " launch. Plan 5.4 says nothing in section 3 reads B, so this"
            + " is a finding about the execution plan."
        )
    print(
        "clause (c) batch POSITIVE: PASS, every activation stage's row 0 is"
        " identical at B=1, B=2 and B=3 and row 1 at B=2 and B=3, on all "
        + String(cells)
        + " compared cells"
    )

    # ---- THE NEGATIVE HALF, WITH THE ORACLE PREDICTING THE COUNT --------
    var h1 = run_host_backward(c, dims, w, x1, g1, 1, l)
    var h3 = run_host_backward(c, dims, w, x3, g3, 3, l)
    var weight_stages = 0
    var moved_stages = 0
    var predicted_cells = 0
    var device_cells = 0
    var still = String("")
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if bwd_stage_kind(i) != K_WEIGHT:
            continue
        weight_stages += 1
        var ph = compare_stage("predict", h1[i], h3[i], False)
        var pd = compare_stage("device", d1[i], d3[i], False)
        predicted_cells += ph.n_diff
        device_cells += pd.n_diff
        if pd.n_diff > 0:
            moved_stages += 1
        elif still == "":
            still = backward_stage_tag(i)
    if moved_stages != weight_stages:
        raise Error(
            String("transformer_backward_check: CLAUSE (c) BATCH's NEGATIVE")
            + " HALF FAILED. Only "
            + String(moved_stages)
            + " of "
            + String(weight_stages)
            + " weight gradients moved between B=1 and B=3, first unmoved at "
            + still
            + ". A weight gradient IS the sum over the tokens in the call,"
            + " so one that does not move when two more sequences join the"
            + " launch is not a weight gradient -- it is a constant, and a"
            + " gate asserting only the positive half would have passed it."
        )
    if predicted_cells != device_cells:
        raise Error(
            String("transformer_backward_check: CLAUSE (c) BATCH's NEGATIVE")
            + " HALF DISAGREES WITH THE ORACLE. The host oracle predicted "
            + String(predicted_cells)
            + " moved weight-gradient cells between B=1 and B=3 and the"
            + " device produced "
            + String(device_cells)
            + ". Plan 6.2(c) requires the oracle to predict the count BEFORE"
            + " the device is asked, precisely so that 'it moved' cannot be"
            + " satisfied by an arm that zeroed the buffer."
        )
    print(
        "clause (c) batch NEGATIVE: PASS, all "
        + String(weight_stages)
        + " weight gradients MOVE between B=1 and B=3, on exactly "
        + String(device_cells)
        + " cells -- the count the host oracle predicted before the device"
        " was asked"
    )


# ===========================================================================
# CLAUSE (c), SECOND HALF: SEQUENCE LENGTH
# ===========================================================================


def clause_c_length(ctx: DeviceContext, k: Int) raises:
    """Plan 6.2 clause (c), the length half, **with DEVIATION 1529's
    correction applied to the clause's own list.**

    **IT IS ONE WEIGHT SET, ONE `x` AND ONE `d_out`, TRUNCATED, AND IT HAS
    TO BE.** The obvious spelling is to run several fixture CASES at several
    lengths, and it is wrong: `fixture_case_seed` gives every case its own
    seed, so four cases have four different weight sets and nothing about
    them is comparable. So this half takes ONE case, keeps its weights, and
    truncates `x` and `d_out` to 4, 16 and the case's own `L`. Tokens 0..3
    then have bit-identical inputs in all three runs by construction.
    (`bwd_d_out`'s key is index-based and does not depend on `L`, which is
    what makes the truncation a prefix rather than a different draw.)

    **WHAT IS ASSERTED EQUAL** -- the stages that fold over the KEY axis or
    over nothing at all. Plan 5.1(ii) is the proof: at a masked `(t, j)`
    every one of the four folds' contributions is a SIGNED ZERO, and plan
    5.1(i) proves a `+0.0`-seeded chain never holds `-0.0`, so every masked
    term is bitwise inert in every chain. A query token's `dq` is the same
    bits whether the sequence has 4 keys or 257.

    **WHAT IS ASSERTED TO MOVE** -- the stages that fold over the QUERY
    axis, which `is_query_axis_stage` enumerates and DEVIATION 1529 argues.
    Plan 6.2's clause (c) puts `bwd.d_x` and `bwd.d_norm1_out` in the
    INVARIANT list and its own section 5.2 puts them in the moving one; this
    file follows 5.2, because a key at slot `j` is read by every query at
    position `>= j` and lengthening the sequence gives that key more
    contributors. **If the plan's list were followed instead, this clause
    would fail on a correct implementation** -- which is the most expensive
    kind of wrong a gate can be, because somebody would "fix" the kernel.

    **AND THE FIRING CONTROL.** If `token_view` were wrong -- if it returned
    token 0 whatever token it was asked for -- every comparison would be a
    token against itself and would pass for ever. So the clause first
    compares token 0 of the shortest run against token 1 of the next, a
    DELIBERATE MISALIGNMENT that MUST differ. A zero there raises VACUOUS,
    not FAILED."""
    var c = fixture_case(k)
    if c.b != 1:
        raise Error(
            "transformer_backward_check: clause (c)'s length half is"
            " written for B=1"
        )
    if c.plant != PLANT_NONE:
        raise Error(
            String("transformer_backward_check: clause (c)'s length half")
            + " refuses the planted case "
            + String(c.name)
            + ": a flat score-plant index lands in a different cell at every"
            + " L, so the comparison would be measuring the plant"
        )
    if c.split != c.l:
        raise Error(
            "transformer_backward_check: clause (c)'s length half needs a"
            " single-call case (split == l)"
        )
    var dims = fixture_dims(c)
    var w = fixture_weights(c)
    var dm = c.d_model
    var x_full = fixture_x(c)
    var g_full = bwd_d_out(c, dims, DOUT_PLANT_NONE)

    var lens = List[Int]()
    lens.append(4)
    if c.l >= 16:
        lens.append(16)
    if c.l > 16:
        lens.append(c.l)
    var keep = lens[0]
    if len(lens) < 2:
        raise Error(
            "transformer_backward_check: clause (c)'s length half needs at"
            " least two lengths and the case chosen is too short"
        )

    print(
        "clause (c) length: "
        + String(c.name)
        + " truncated to "
        + String(len(lens))
        + " lengths, one weight set and one d_out, tokens 0.."
        + String(keep - 1)
        + " compared (DEVIATION 1529's corrected split)"
    )

    var runs = List[List[List[Float32]]]()
    var hosts = List[List[List[Float32]]]()
    for li in range(len(lens)):
        var ll = lens[li]
        var cl = FixtureCase(
            c.name, 1, ll, ll, c.d_model, c.n_heads, c.n_kv_heads,
            c.head_dim, c.intermediate, ll, c.plant,
        )
        var xs = List[Float32]()
        var gs = List[Float32]()
        for i in range(ll * dm):
            xs.append(x_full[i])
            gs.append(g_full[i])
        var off = IdentityTrace.disabled()
        runs.append(
            run_device_backward(
                ctx, cl, dims, w, xs, gs, 1, ll, off, "cl" + String(ll)
            )
        )
        hosts.append(run_host_backward(cl, dims, w, xs, gs, 1, ll))

    # ---- THE FIRING CONTROL: token 0 against token 1 --------------------
    var control = 0
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if bwd_stage_kind(i) == K_WEIGHT:
            continue
        var a = token_view(runs[0][i], i, 1, lens[0], lens[0], 0, keep, dims)
        var bcmp = token_view(
            runs[1][i], i, 1, lens[1], lens[1], 1, keep, dims
        )
        if len(a) != len(bcmp):
            continue
        var d = compare_stage("control", a, bcmp, False)
        if d.n_diff > 0:
            control += 1
    if control == 0:
        raise Error(
            "transformer_backward_check: CLAUSE (c) LENGTH IS VACUOUS. Token"
            " 0 of the shortest run is bit-identical to token 1 of the next"
            " on every stage, which cannot be true of two different tokens."
            " `token_view` is not cutting distinct tokens, so every"
            " comparison below is a token against itself"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (c) length control: token 0 vs token 1 across two lengths"
        " differs on "
        + String(control)
        + " stages, so the token viewer distinguishes tokens"
    )

    # ---- THE INVARIANT SET ----------------------------------------------
    var cells = 0
    var bad = 0
    var first_bad = String("")
    for li in range(1, len(lens)):
        for t in range(keep):
            for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
                if is_query_axis_stage(i):
                    continue
                var kk = keep
                if bwd_stage_kind(i) == K_ATTNROW:
                    kk = t + 1  # the causal prefix; past it is a signed zero
                var a = token_view(
                    runs[0][i], i, 1, lens[0], lens[0], t, kk, dims
                )
                var bcmp = token_view(
                    runs[li][i], i, 1, lens[li], lens[li], t, kk, dims
                )
                cells += len(a)
                var d = compare_stage("len", a, bcmp, False)
                if d.n_diff > 0:
                    bad += 1
                    if first_bad == "":
                        first_bad = (
                            backward_stage_tag(i)
                            + " at token "
                            + String(t)
                            + ", L="
                            + String(lens[li])
                            + ", "
                            + String(d.n_diff)
                            + " of "
                            + String(d.n_cells)
                            + " cells"
                        )
    if bad != 0:
        raise Error(
            String("transformer_backward_check: CLAUSE (c) LENGTH FAILED on ")
            + String(bad)
            + " stage-tokens, first at "
            + first_bad
            + ". Plan 5.1 proves every masked contribution is a signed zero"
            + " and 5.1(i) proves a +0.0-seeded chain never holds -0.0, so a"
            + " failure here is a finding about the profile's masked-tail"
            + " theorem and not about the gate."
        )
    print(
        "clause (c) length POSITIVE: PASS, the key-axis activation gradients"
        " are bit-identical for tokens 0.."
        + String(keep - 1)
        + " across "
        + String(len(lens))
        + " sequence lengths on all "
        + String(cells)
        + " compared cells"
    )

    # ---- THE MOVING SET, WITH THE ORACLE PREDICTING ---------------------
    var qstages = 0
    var qmoved = 0
    var qpred = 0
    var qdev = 0
    var stuck = String("")
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if not is_query_axis_stage(i):
            continue
        qstages += 1
        var kk = keep
        var ah = token_view(hosts[0][i], i, 1, lens[0], lens[0], 0, kk, dims)
        var bh = token_view(
            hosts[len(lens) - 1][i], i, 1, lens[len(lens) - 1],
            lens[len(lens) - 1], 0, kk, dims,
        )
        var ad = token_view(runs[0][i], i, 1, lens[0], lens[0], 0, kk, dims)
        var bd = token_view(
            runs[len(lens) - 1][i], i, 1, lens[len(lens) - 1],
            lens[len(lens) - 1], 0, kk, dims,
        )
        if len(ah) != len(bh) or len(ad) != len(bd):
            continue
        var ph = compare_stage("predict", ah, bh, False)
        var pd = compare_stage("device", ad, bd, False)
        qpred += ph.n_diff
        qdev += pd.n_diff
        if pd.n_diff > 0:
            qmoved += 1
        elif stuck == "":
            stuck = backward_stage_tag(i)
    if qmoved == 0:
        raise Error(
            String("transformer_backward_check: CLAUSE (c) LENGTH's MOVING")
            + " HALF FAILED. NONE of the "
            + String(qstages)
            + " query-axis stages moved between L="
            + String(lens[0])
            + " and L="
            + String(lens[len(lens) - 1])
            + ", first unmoved at "
            + stuck
            + ". A key at slot j is read by every query at position >= j, so"
            + " lengthening the sequence MUST give it more contributors."
            + " A gate asserting only the invariant half would have passed"
            + " an implementation that computed constants."
        )
    if qpred != qdev:
        raise Error(
            String("transformer_backward_check: CLAUSE (c) LENGTH's MOVING")
            + " HALF DISAGREES WITH THE ORACLE. Predicted "
            + String(qpred)
            + " moved cells, device produced "
            + String(qdev)
            + "."
        )
    print(
        "clause (c) length NEGATIVE: PASS, "
        + String(qmoved)
        + " of "
        + String(qstages)
        + " query-axis stages MOVE with the sequence length, on exactly "
        + String(qdev)
        + " cells -- the count the host oracle predicted. **DEVIATION 1529:"
        " plan 6.2's clause (c) lists bwd.d_x and bwd.d_norm1_out as"
        " length-INVARIANT and they are in this moving set, which is what"
        " the plan's own section 5.2 says.**"
    )


# ===========================================================================
# CLAUSE (d): THE CARRIED-ACCUMULATOR CHUNK THEOREM
# ===========================================================================


@fieldwise_init
struct ChunkVerdict(Copyable, Movable):
    """Clause (d1)'s result as a VALUE rather than a raise, because clause
    (g) has to INVERT it for the arms that are falsifiable only here, and a
    gate that can only raise cannot be inverted."""

    var bad: Int
    var cells: Int
    var first_token: Int
    var first_stage: String
    var control: Int
    var moved_query_axis: Int


def clause_d_device(ctx: DeviceContext, k: Int) raises -> ChunkVerdict:
    """Plan 6.2 clause (d), the half that runs on a device.

    A length-`L` sequence is run ONCE as a single forward+backward and then
    as TWO, CHUNKED OVER THE QUERY AXIS with the KV cache carried across the
    two forwards. Every token of the second chunk is then compared against
    the same absolute token of the unchunked run.

    **WHAT MUST MATCH** -- every stage that folds over the KEY axis or over
    nothing at all. Plan 5.1: `z` and `dq` fold over the key axis and are
    therefore independent of how the sequence was chopped, exactly as
    `attn.denom` and `attn.ctx` are in the forward. A query token's `dq` is
    the same bits whether it arrived in a prefill of 133 or in a chunk of 4
    behind a cache of 129.

    **WHAT MUST MOVE** -- `bwd.d_k_cache`, `bwd.d_v_cache`, their two
    `[pos0, S)` slices, the three-term fan-in downstream of them, and every
    weight gradient. Plan 5.2: a key at slot `j` is read by every query at
    position `>= j`, and the queries of the FIRST chunk are not in the
    second call. **So a per-call backward computes a PARTIAL and the caller
    must add the partials**, and the clause asserts the partial is a partial
    rather than pretending the plan says otherwise.

    **THE FIRING CONTROL, AND THIS CLAUSE IS THE ONE MOST EXPOSED WITHOUT
    ONE.** If the chunked path and the unchunked path shared a buffer or a
    cached card, the comparison would be a value against ITSELF and would
    pass for ever on every vendor. So the clause first compares chunk-2
    token `qi` against unchunked token `pos0 + qi + 1` -- a DELIBERATE
    MISALIGNMENT that MUST differ. If it does not, the comparison cannot
    tell two tokens apart and the clause raises VACUOUS, not FAILED.

    THE COST, stated rather than hidden: three forwards and three backwards
    at `L` up to 133, which is why clause (d) is off by default."""
    var c = fixture_case(k)
    if c.b != 1:
        raise Error(
            String("transformer_backward_check: clause (d) is written for")
            + " B=1 and "
            + String(c.name)
            + " has B="
            + String(c.b)
        )
    if c.split >= c.l:
        raise Error(
            String("transformer_backward_check: clause (d) needs a TWO-CALL")
            + " case (split < l) and "
            + String(c.name)
            + " has split="
            + String(c.split)
            + " l="
            + String(c.l)
            + ". A single-call case cannot be chunked over the query axis"
            + " without inventing a boundary the fixture never declared."
        )
    if c.plant != PLANT_NONE:
        raise Error(
            String("transformer_backward_check: clause (d) refuses the")
            + " planted case "
            + String(c.name)
            + ": a planted score in a two-call fixture makes it impossible"
            + " to say whether a divergence came from the plant or from the"
            + " chunking, and separating those is the clause's whole value."
        )
    var dims = fixture_dims(c)
    var ldims = llama_dims_of(dims)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var d_out = bwd_d_out(c, dims, DOUT_PLANT_NONE)
    var l = c.l
    var dm = c.d_model
    var cut = c.split
    var l2 = l - cut
    var cap = c.cache_cap
    if cap < l:
        cap = l

    print(
        "clause (d) device: "
        + String(c.name)
        + ", L="
        + String(l)
        + " unchunked against "
        + String(cut)
        + " + "
        + String(l2)
        + " chunked over the QUERY axis with the cache carried"
    )

    # ---- the unchunked run ----------------------------------------------
    var off_u = IdentityTrace.disabled()
    var un = run_device_backward(
        ctx, c, dims, w, x, d_out, 1, l, off_u, "unchunked"
    )

    # ---- the chunked run: TWO forwards on ONE cache, TWO backwards ------
    # The weights, the cache and the rotary table are built ONCE and
    # carried, which is the whole point: a fresh cache per chunk would be
    # two prefills and would test nothing.
    var dw = LlamaDeviceWeights(
        ctx, ldims, RMS_EPS, w.norm1_w, w.norm2_w, w.w_q, w.w_k, w.w_v,
        w.w_o, w.w_gate, w.w_up, w.w_down,
    )
    var kv = LlamaKVCache(ctx, 1, ldims, cap)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)

    var x1 = List[Float32]()
    for i in range(cut * dm):
        x1.append(x[i])
    var g1 = List[Float32]()
    for i in range(cut * dm):
        g1.append(d_out[i])
    var st1 = LlamaDeviceStages(ctx, 1, cut, cap, ldims)
    var dx1 = _upload(ctx, x1)
    var off1 = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, st1, kv, rope, dw, dx1, 1, cut, 0, PLANT_AT_NONE,
        List[Int](), List[UInt32](), off1, "chunk1",
    )
    var bst1 = LlamaBackwardStages(ctx, 1, cut, cap, ldims)
    var offb1 = IdentityTrace.disabled()
    llama_decoder_layer_backward(
        ctx, bst1, st1, dw, rope.cos, rope.sin, dx1, g1, 1, cut, 0,
        offb1, "chunk1b",
    )
    _ = bst1^
    _ = st1^
    _ = dx1^

    var x2 = List[Float32]()
    var g2 = List[Float32]()
    for i in range(cut * dm, l * dm):
        x2.append(x[i])
        g2.append(d_out[i])
    var st2 = LlamaDeviceStages(ctx, 1, l2, cap, ldims)
    var dx2 = _upload(ctx, x2)
    var off2 = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, st2, kv, rope, dw, dx2, 1, l2, cut, PLANT_AT_NONE,
        List[Int](), List[UInt32](), off2, "chunk2",
    )
    var bst2 = LlamaBackwardStages(ctx, 1, l2, cap, ldims)
    var offb2 = IdentityTrace.disabled()
    llama_decoder_layer_backward(
        ctx, bst2, st2, dw, rope.cos, rope.sin, dx2, g2, 1, l2, cut,
        offb2, "chunk2b",
    )
    var ch = backward_device_dump(ctx, bst2, dims, 1, l2, l)
    _ = bst2^
    _ = st2^
    _ = dx2^
    _ = dw^
    _ = kv^
    _ = rope^

    # ---- THE FIRING CONTROL: misaligned tokens MUST differ --------------
    var control = 0
    for qi in range(l2 - 1):
        for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
            if bwd_stage_kind(i) != K_TOKEN:
                continue  # the only cut whose length is qi-independent
            var a = token_view(un[i], i, 1, l, l, cut + qi + 1, l, dims)
            var bcmp = token_view(ch[i], i, 1, l2, l, qi, l, dims)
            if len(a) != len(bcmp):
                continue
            var d = compare_stage("control", a, bcmp, False)
            if d.n_diff > 0:
                control += 1
    if l2 > 1 and control == 0:
        raise Error(
            "transformer_backward_check: CLAUSE (d) IS VACUOUS. Chunk-2"
            " token qi is bit-identical to unchunked token pos0+qi+1 on"
            " every token-major stage, which cannot be true of different"
            " tokens. The chunked and unchunked paths are not two"
            " computations here -- they share a buffer or a cached card --"
            " so the aligned comparison is a value against itself"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (d) control: the deliberate misalignment differs on "
        + String(control)
        + " stage-token comparisons, so the comparison distinguishes tokens"
    )

    # ---- THE CLAUSE: aligned tokens must MATCH on the key-axis stages ---
    var cells = 0
    var bad = 0
    var first_token = -1
    var first_stage = String("")
    for qi in range(l2):
        for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
            if is_query_axis_stage(i):
                continue
            var a = token_view(un[i], i, 1, l, l, cut + qi, l, dims)
            var bcmp = token_view(ch[i], i, 1, l2, l, qi, l, dims)
            cells += len(a)
            var d = compare_stage(
                backward_stage_tag(i) + " token " + String(cut + qi),
                a,
                bcmp,
                False,
            )
            if d.n_diff > 0:
                bad += 1
                if first_token < 0:
                    first_token = cut + qi
                    first_stage = (
                        backward_stage_tag(i)
                        + " on "
                        + String(d.n_diff)
                        + " of "
                        + String(d.n_cells)
                        + " cells"
                    )

    # ---- THE NEGATIVE HALF: the query-axis stages MUST move -------------
    var moved_q = 0
    for i in range(TRANSFORMER_BACKWARD_STAGE_COUNT):
        if not is_query_axis_stage(i):
            continue
        if bwd_stage_kind(i) == K_WEIGHT or bwd_stage_kind(i) == K_KV:
            var d = compare_stage("partial", un[i], ch[i], False)
            if d.n_diff > 0:
                moved_q += 1

    if bad == 0:
        print(
            "clause (d) device: PASS, "
            + String(l2)
            + " chunk-2 tokens bit-identical to the unchunked run on every"
            " key-axis stage, on all "
            + String(cells)
            + " compared cells"
        )
    else:
        print(
            "clause (d) device: "
            + String(bad)
            + " stage-tokens DIFFER, first at token "
            + String(first_token)
            + ", "
            + first_stage
        )
    return ChunkVerdict(bad, cells, first_token, first_stage, control, moved_q)


def clause_d_carry_theorem() raises:
    """Plan 5.3, DEVIATION 1416, **demonstrated and NOT gated, and DEVIATION
    1527 is why.**

    The theorem: split a `+0.0`-seeded serial ascending `fma` chain at ANY
    index, STORE the accumulator, and resume the second piece from it. The
    result is the unsplit chain's bit for bit, because it is literally the
    same sequence of operations in the same order, and `ftz` is idempotent
    so the store and the reload change nothing.

    **THERE IS NOTHING IN THIS LANE TO POINT THAT AT.**
    `llama_decoder_layer_backward` takes no seed, no carry flag and no
    accumulator argument, and neither does
    `transformer_block_backward_oracle`. The plan calls this "the property
    this lane gets for free that the GEMM lane cannot have" and it is
    unimplemented on both halves. So what follows is a DEMONSTRATION on the
    chain shape the plan pins, and it is labelled as one in the SCOPE line.

    **THE FIXTURE IS `{2^24, 1, 1, 1, ...}` AND THE CHOICE IS THE WHOLE
    POINT.** One of the ways a fixture went blind on 2026-08-25 was ONE
    BINADE: four partials all inside `[1, 2)` made a balanced tree and a
    serial chain agree, so a fold-order control did not move. Here `2^24`
    sits exactly at the top of Float32's exact-integer range, so `2^24 + 1`
    is the midpoint of `2^24` and `2^24 + 2` and round-half-to-EVEN returns
    `2^24` -- every single one of the small terms is absorbed by the running
    sum. Under the UNCARRIED spelling the small terms are summed FIRST,
    among themselves, and only then added, so they survive as a group. The
    two answers differ by ULPs and the difference is the whole content of
    plan 5.3.

    **AND THE CONTROL IS TWO-SIDED, BECAUSE A ONE-SIDED DEMONSTRATION IS
    INDISTINGUISHABLE FROM A BROKEN COMPARISON.** It reports how many split
    points the uncarried spelling moves at AND how many it does not, and it
    RAISES only if the number that move is zero. It does not raise on the
    ones that do not move: at a split that leaves one term on the far side,
    `2^24 + 1` rounds back to `2^24` and the uncarried answer coincides --
    which is a true fact about the arithmetic and would be a lie to hide.
    Plan 6.2(d) says the uncarried control "MUST MOVE at any chunk of 2 or
    more terms" and that sentence is very slightly too strong; the measured
    per-split table below is what replaces it."""
    print(
        "clause (d) carry theorem: **A DEMONSTRATION, NOT A GATE.** There is"
        " no chunked backward entry point on either half of this lane"
        " (DEVIATION 1527), so plan 5.3's theorem has nothing to be pointed"
        " at. What follows is the theorem on the chain shape plan section 3"
        " pins."
    )
    var big = f32_from_bits(UInt32(0x4B800000))  # 2^24 exactly
    var one = f32_from_bits(UInt32(0x3F800000))
    var pos = f32_from_bits(BITS_POS_ZERO)
    var n = 8
    var a = List[Float32]()
    a.append(big)
    for _ in range(1, n):
        a.append(one)

    var unsplit = pos
    for i in range(n):
        unsplit = ftz(identical_mul_add(a[i], one, unsplit))

    var carry_bad = 0
    for cut in range(1, n):
        var acc = pos
        for i in range(cut):
            acc = ftz(identical_mul_add(a[i], one, acc))
        # THE CARRY: the stored accumulator becomes the second piece's seed.
        var resumed = acc
        for i in range(cut, n):
            resumed = ftz(identical_mul_add(a[i], one, resumed))
        if bits_of(resumed) != bits_of(unsplit):
            carry_bad += 1
    if carry_bad != 0:
        raise Error(
            String("transformer_backward_check: the CARRY theorem is FALSE")
            + " on this toolchain at "
            + String(carry_bad)
            + " of "
            + String(n - 1)
            + " split points. Plan 5.3 says a carried chain is 'literally"
            + " the same sequence of operations in the same order', so a"
            + " failure here is a finding about `ftz` or about `fma` and not"
            + " about the chain."
        )

    var uncarried_moved = 0
    var uncarried_inert = 0
    for cut in range(1, n):
        var p1 = pos
        for i in range(cut):
            p1 = ftz(identical_mul_add(a[i], one, p1))
        var p2 = pos
        for i in range(cut, n):
            p2 = ftz(identical_mul_add(a[i], one, p2))
        if bits_of(ftz(p1 + p2)) != bits_of(unsplit):
            uncarried_moved += 1
        else:
            uncarried_inert += 1
    if uncarried_moved == 0:
        raise Error(
            String("transformer_backward_check: the UNCARRIED CONTROL IS")
            + " DEAD. Adding two independently +0.0-seeded partials"
            + " reproduced the unsplit chain at ALL "
            + String(n - 1)
            + " split points, so this demonstration cannot tell a carry"
            + " from an add and its 'the carry reproduces the chain' half"
            + " proves nothing ([[reached-but-inert]])."
        )
    print(
        "clause (d) carry theorem: unsplit "
        + hexbits(unsplit)
        + " over {2^24, 1 x 7}; the CARRY reproduces it at ALL "
        + String(n - 1)
        + " split points including misaligned ones; the UNCARRIED"
        " two-partial spelling MOVES at "
        + String(uncarried_moved)
        + " of "
        + String(n - 1)
        + " and is inert at "
        + String(uncarried_inert)
        + " (the ones that leave a single absorbed term on the far side --"
        " plan 6.2(d)'s 'MUST MOVE at any chunk of 2 or more terms' is very"
        " slightly too strong and this table is the correction)."
    )


# ===========================================================================
# CLAUSE (e): THE ROW-39 AUDIT AND THE SIGNED-ZERO AUDIT
# ===========================================================================


def nonfinite_cells(values: List[Float32]) -> Int:
    """NaN or infinity cells, BY BITS AND NEVER BY COMPARES.

    Metal FLUSHES COMPARE OPERANDS (IDENTITY_PATHS row 49, DEVIATION 746
    (i)), so `v != v` is a test with two meanings across columns while a
    mask-and-compare on the exponent field has one. Integer operations do
    not flush anywhere. This must return exactly 1 for a single plant on
    every column or the audit is measuring the toolchain rather than the
    refusal."""
    var n = 0
    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au >= UInt32(0x7F800000):
            n += 1
    return n


def clause_e(ctx: DeviceContext, k: Int) raises:
    """Plan 6.2 clause (e), both halves.

    **HALF ONE, THE ROW-39 AUDIT.** A NaN or an infinity in the incoming
    gradient must be REFUSED BY NAME before any recorded stage, because NaN
    payloads are vendor-shaped -- IDENTITY_PATHS row 39 measured three
    payloads for one IEEE answer, `0x7fc00000` on Apple, `0x7fffffff` on
    NVIDIA, `0xffc00000` on AMD -- and a certified stage may never hold one.
    **A GRADIENT IS EXACTLY WHERE NaNs APPEAR IN PRACTICE**, which makes
    this refusal more likely to fire than the forward's and makes the named
    error worth more.

    Each plant goes in at cell `len / 2` and NEVER at cell 0: a plant at
    index 0 is the one a loop that skips its first element would still
    catch, which makes it the weakest possible plant site.

    Three things are checked per plant, and the first is the one the other
    lanes' warnings are about:

      1. **REACH, MEASURED.** The planted list's non-finite cells are
         counted BY BITS and must be exactly 1. If the plant did not survive
         into the list the refusal that follows fired for some other reason
         and the audit proves nothing. Reach is MEASURED, never inferred.
      2. BOTH SIDES raise, and BOTH messages NAME `d_residual2`. The host
         oracle and the device entry point are checked separately, because
         **a refusal that lives in one half and not the other is exactly the
         defect two other lanes shipped on 2026-08-25** -- the contract said
         "refused by name before any recorded stage", the host oracle did it
         and the device entry point did not. **THIS LANE IS CLEAN ON THAT
         POINT and the check is what makes the sentence a measurement.**
      3. The trace holds ZERO records. "Before any recorded stage" is the
         actual clause, and a refusal that fires after `bwd.in.d_residual2`
         was hashed has already put a vendor-shaped payload into a card.

    **AND ITS CONTROL, WHICH IS THE ONE THAT MATTERS.** If the refusal
    raised UNCONDITIONALLY -- a stray raise, a walk over the wrong buffer, a
    mask that matched every finite value -- then every plant would be
    "refused by name" and the clause would pass for ever WHILE GATING
    NOTHING. So the clause first runs a CLEAN backward and requires that it
    does NOT raise and that it records all thirty-seven stages.

    **THE SCOPE GAP, MEASURED AND NAMED. DEVIATION 1534.** The backward's
    refusal covers `d_residual2` AND NOTHING ELSE. It does not look at
    `x_dev`, at the eleven weights, or at any of the saved forward stages
    it reads. Plan 6.2(e) asks only for the incoming gradient, so **the code
    and the plan AGREE and this is a gap in the CLAUSE's scope rather than a
    conformance defect** -- which is a different finding from the embedding
    lane's DEVIATION 1506, where the contract promised a refusal the device
    did not perform. It is measured here rather than argued: a NaN is
    planted into `x_dev` AFTER the forward has run, READ BACK OFF THE DEVICE
    by bits so reach is measured, and the backward is called WITH THE TRACE
    DISABLED so that no vendor-shaped payload can enter a card. The result
    is REPORTED. It is not asserted either way, because asserting that a
    refusal is absent would freeze a gap into a requirement.

    **HALF TWO, THE SIGNED-ZERO AUDIT.** Plan 6.2(e) asks for the masked
    cells of `bwd.d_attn_scores`, "whose predicted sign is
    `sign(dy_j - z)`". Clause (a) already compares those cells bitwise; what
    clause (a) CANNOT say is whether both signs OCCUR. A fixture on which
    every masked cell is `+0.0` tests the prediction on half its range and
    reports full marks, and `B13_MASK_ZEROES_GRAD`'s whole content is the
    `-0.0` half. So this half runs the `DOUT_PLANT_ALTERNATING_ZEROS`
    fixture and requires BOTH signs to appear among the masked cells."""
    var c = fixture_case(k)
    if c.plant != PLANT_NONE:
        raise Error(
            "transformer_backward_check: clause (e) refuses a planted case;"
            " the audit plants its own bits and a second plant would confuse"
            " which one the refusal fired on"
        )
    var dims = fixture_dims(c)
    var ldims = llama_dims_of(dims)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var b = c.b
    var l = c.l
    var dm = dims.d_model
    var m = b * l
    var cap = c.cache_cap
    if cap < l:
        cap = l
    var path = card_path() + ".clause_e"

    print(
        "clause (e): the row-39 audit on the incoming gradient, 2 bit"
        " patterns at cell len/2, plus the signed-zero audit, on "
        + String(c.name)
    )

    # ---- THE CONTROL ----------------------------------------------------
    var cpath = card_path() + ".clause_e_control"
    var clean = bwd_d_out(c, dims, DOUT_PLANT_NONE)
    var ctrace = IdentityTrace.to_path(cpath)
    var ctrl_raised = False
    try:
        _ = run_device_backward(
            ctx, c, dims, w, x, clean, b, l, ctrace, "clause_e_ctrl"
        )
    except e:
        ctrl_raised = True
    if ctrl_raised:
        raise Error(
            "transformer_backward_check: CLAUSE (e) IS VACUOUS. A clean"
            " backward raised, so 'the call raised' below does not"
            " distinguish a refusal from a broken call."
        )
    var ctrl_recs = read_trace_lines(cpath)
    if len(ctrl_recs) != TRANSFORMER_BACKWARD_STAGE_COUNT:
        raise Error(
            String("transformer_backward_check: CLAUSE (e) IS VACUOUS. A")
            + " clean call recorded "
            + String(len(ctrl_recs))
            + " stages instead of "
            + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
            + ", so 'zero stages recorded' below does not distinguish a"
            + " refusal from a call that never ran."
        )
    var host_ctrl_raised = False
    try:
        _ = run_host_backward(c, dims, w, x, clean, b, l)
    except e:
        host_ctrl_raised = True
    if host_ctrl_raised:
        raise Error(
            "transformer_backward_check: CLAUSE (e) IS VACUOUS. The HOST"
            " oracle raised on a clean gradient, so its refusal is"
            " unconditional and every plant below would be 'refused'"
            " whatever it held."
        )
    print(
        "clause (e) control: a clean call raises on NEITHER half and the"
        " device records all "
        + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
        + " stages, so the refusal is not unconditional and 'zero records'"
        " is a real signal"
    )

    # ---- THE PLANTS -----------------------------------------------------
    var patterns: List[UInt32] = [BITS_QNAN, BITS_POS_INF]
    var pat_names: List[String] = [String("NaN"), String("infinity")]
    var checked = 0
    for pk in range(len(patterns)):
        var g = bwd_d_out(c, dims, DOUT_PLANT_NONE)
        g[len(g) // 2] = f32_from_bits(patterns[pk])
        var reached = nonfinite_cells(g)
        if reached != 1:
            raise Error(
                String("transformer_backward_check: CLAUSE (e) IS VACUOUS")
                + " for "
                + pat_names[pk]
                + ": the plant did NOT survive into the gradient list ("
                + String(reached)
                + " non-finite cells, expected exactly 1). Any refusal after"
                + " this fired for another reason"
                + " ([[reached-but-inert]])."
            )

        var hraised = False
        var hmsg = String("")
        try:
            _ = run_host_backward(c, dims, w, x, g, b, l)
        except e:
            hraised = True
            hmsg = String(e)
        if not hraised:
            raise Error(
                String("transformer_backward_check: CLAUSE (e) FAILED. A ")
                + pat_names[pk]
                + " in d_residual2 was NOT refused by the HOST oracle."
            )
        if hmsg.find(String("d_residual2")) < 0:
            raise Error(
                String("transformer_backward_check: CLAUSE (e) FAILED. The")
                + " host refusal does not NAME d_residual2: "
                + hmsg
            )

        var trace = IdentityTrace.to_path(path)
        var draised = False
        var dmsg = String("")
        try:
            _ = run_device_backward(
                ctx, c, dims, w, x, g, b, l, trace, "clause_e"
            )
        except e:
            draised = True
            dmsg = String(e)
        if not draised:
            raise Error(
                String("transformer_backward_check: CLAUSE (e) FAILED. A ")
                + pat_names[pk]
                + " in d_residual2 was NOT refused by the DEVICE entry"
                + " point, though the HOST oracle refused it. That is"
                + " exactly the split two other lanes shipped on"
                + " 2026-08-25: the contract's sentence true of the oracle"
                + " and false of the device."
            )
        if dmsg.find(String("d_residual2")) < 0:
            raise Error(
                String("transformer_backward_check: CLAUSE (e) FAILED. The")
                + " device refusal does not NAME d_residual2: "
                + dmsg
            )
        var recs = read_trace_lines(path)
        if len(recs) != 0:
            raise Error(
                String("transformer_backward_check: CLAUSE (e) FAILED. The")
                + " refusal for "
                + pat_names[pk]
                + " fired, but "
                + String(len(recs))
                + " stage(s) were already recorded. The clause is REFUSED"
                + " BEFORE ANY RECORDED STAGE."
            )
        checked += 1
    print(
        "clause (e) row-39: PASS, "
        + String(checked)
        + " plants, each reach-measured by bits, each refused BY NAME on"
        " BOTH halves, each with 0 stages recorded"
    )

    # ---- DEVIATION 1534: WHAT THE REFUSAL DOES NOT COVER ----------------
    # Measured with the trace DISABLED, so no vendor-shaped payload can
    # enter a card. Reported and NOT asserted: asserting that a refusal is
    # absent would freeze a gap into a requirement.
    var dw = LlamaDeviceWeights(
        ctx, ldims, RMS_EPS, w.norm1_w, w.norm2_w, w.w_q, w.w_k, w.w_v,
        w.w_o, w.w_gate, w.w_up, w.w_down,
    )
    var kv = LlamaKVCache(ctx, b, ldims, cap)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var stages = LlamaDeviceStages(ctx, b, l, cap, ldims)
    var xp = x.copy()
    var dxp = _upload(ctx, xp)
    var offf = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, stages, kv, rope, dw, dxp, b, l, 0, PLANT_AT_NONE,
        List[Int](), List[UInt32](), offf, "gapfwd",
    )
    # Plant AFTER the forward, so the forward's own thirteen-name refusal
    # cannot be what fires.
    xp[len(xp) // 2] = f32_from_bits(BITS_QNAN)
    var dxq = _upload(ctx, xp)
    var back = _download(ctx, dxq, m * dm)
    var xreach = nonfinite_cells(back)
    if xreach != 1:
        raise Error(
            String("transformer_backward_check: the DEVIATION 1534 audit is")
            + " VACUOUS: the planted NaN did not arrive on the device ("
            + String(xreach)
            + " non-finite cells read back, expected exactly 1)"
            + " ([[reached-but-inert]])."
        )
    var bst = LlamaBackwardStages(ctx, b, l, cap, ldims)
    var offg = IdentityTrace.disabled()
    var gap_raised = False
    try:
        llama_decoder_layer_backward(
            ctx, bst, stages, dw, rope.cos, rope.sin, dxq, clean, b, l, 0,
            offg, "gap",
        )
    except e:
        gap_raised = True
    print(
        "DEVIATION 1534 AUDIT: a NaN planted into x_dev AFTER the forward"
        " was READ BACK OFF THE DEVICE ("
        + String(xreach)
        + " non-finite cell, so reach is MEASURED). The backward raised: "
        + String(gap_raised)
        + ". **The backward's refusal covers d_residual2 AND NOTHING ELSE**"
        " -- not x_dev, not the eleven weights, not the saved forward"
        " stages. Plan 6.2(e) asks only for the incoming gradient, so the"
        " CODE AND THE PLAN AGREE and this is a gap in the CLAUSE's scope,"
        " not a conformance defect. It is reported and NOT asserted: an"
        " assertion here would freeze the gap into a requirement."
    )
    _ = bst^
    _ = stages^
    _ = dw^
    _ = kv^
    _ = rope^
    _ = dxp^
    _ = dxq^

    # ---- HALF TWO: THE SIGNED-ZERO AUDIT --------------------------------
    var gz = bwd_d_out(c, dims, DOUT_PLANT_ALTERNATING_ZEROS)
    var hz = run_host_backward(c, dims, w, x, gz, b, l)
    var nh = dims.n_heads
    var s = l
    var pos_zero = 0
    var neg_zero = 0
    var other = 0
    for bb in range(b):
        for h in range(nh):
            for t in range(l):
                var base = ((bb * nh + h) * l + t) * s
                for j in range(s):
                    if j <= t:
                        continue
                    var u = bitcast[DType.uint32](hz[20][base + j])
                    if u == UInt32(0x00000000):
                        pos_zero += 1
                    elif u == UInt32(0x80000000):
                        neg_zero += 1
                    else:
                        other += 1
    if other != 0:
        raise Error(
            String("transformer_backward_check: CLAUSE (e) SIGNED-ZERO")
            + " AUDIT FAILED. "
            + String(other)
            + " masked cells of bwd.d_attn_scores are NOT a signed zero."
            + " Plan 5.1(ii) proves y_j is exactly +0.0 at a masked cell, so"
            + " dS_j = pinned_mul(+0.0, dy_j - z) must be +-0.0 and nothing"
            + " else. A masked cell holding a NORMAL is the masked-tail"
            + " theorem failing, and every length- and chunk-invariance"
            + " claim in this lane rests on it."
        )
    if pos_zero == 0 or neg_zero == 0:
        raise Error(
            String("transformer_backward_check: CLAUSE (e)'s SIGNED-ZERO")
            + " AUDIT IS HALF BLIND. The masked cells hold "
            + String(pos_zero)
            + " +0.0 and "
            + String(neg_zero)
            + " -0.0, and BOTH must occur for the predicted sign"
            + " `sign(dy_j - z)` to be tested on its whole range. A fixture"
            + " on which every masked cell is one sign reports full marks on"
            + " half a prediction, and B13_MASK_ZEROES_GRAD's entire content"
            + " is the -0.0 half ([[reached-but-inert]])."
        )
    print(
        "clause (e) signed zeros: PASS, the masked cells of"
        " bwd.d_attn_scores are ALL signed zeros ("
        + String(pos_zero)
        + " are +0.0 and "
        + String(neg_zero)
        + " are -0.0), so plan 6.2(e)'s predicted sign is exercised on BOTH"
        " halves of its range"
    )


# ===========================================================================
# CLAUSE (f): CORRECTNESS, WHICH IS A DIFFERENT QUESTION FROM IDENTITY
# ===========================================================================


def clause_f() raises:
    """Plan 6.2 clause (f). **A TRANSPOSE ERROR IS BIT IDENTICAL ON THREE
    VENDORS**, so nothing clause (a) through (e) says is evidence that the
    gradient is the gradient.

    What is checkable EXACTLY is checked exactly here, on the HOST, with no
    epsilon anywhere. What is not is NAMED and left open, and the naming is
    the more important half.

    **ARM 1: THE SOFTMAX CLOSED FORM, AGAINST HAND-WRITTEN INTEGER
    ARITHMETIC.** On operands that are small integers the whole closed form
    is exactly representable: `z = sum_j dy_j * y_j` and
    `dS_j = y_j * (dy_j - z)` have products bounded by 64 and sums by 256 at
    the shapes below, all far under `2^24`, so `fma`, `pinned_mul` and the
    subtraction are all EXACT and the answer can be written down. That is
    `gemm/IDENTICAL_BACKWARD_PLAN.md` G2's method, DEVIATION 1051, lifted --
    and it catches the things identity cannot: a transposed sign, a
    subtraction the wrong way round, a missing `y` factor, a `z` folded over
    the wrong axis.

    **AND ITS CONTROL.** An exact-integer arm that agreed with everything
    would be indistinguishable from a broken comparison, so the arm also
    runs a DELIBERATELY WRONG closed form -- `y_j * (z - dy_j)`, the
    subtraction reversed, which is the single most likely transcription
    error in this seam -- and requires it to DISAGREE. `loss_check.mojo`'s
    guard 4 is the pattern.

    **ARM 2: THE MASK BACKWARD IS THE EXACT IDENTITY.** Plan 6.1 records
    `bwd.d_attn_scores` "even though it is bitwise equal to
    `bwd.d_attn_masked` by construction... That is the whole reason: the
    equality is what `B13_MASK_ZEROES_GRAD` breaks, and an unrecorded
    identity is an unchecked one." Here it is checked, on every clause-(a)
    shape, in the HOST oracle -- which is where a broken identity would be a
    calculus error rather than a device one.

    **WHAT THIS CLAUSE DOES NOT COVER, AND IT IS THE LARGEST GAP IN THE
    LANE.** `silu'`, the `rsqrt` tail and the softmax's transcendental
    inputs do not survive exact integers. Gating those needs a FLOAT64
    DIRECTIONAL-DERIVATIVE reference at a stated tolerance and **there is
    none**: `transformer_oracle.mojo` deliberately carries no float64
    reference, `transformer/corpus/` does not exist, and a float64 backward
    written now would be a second unreviewed implementation of a gradient
    whose first implementation has not been compiled. **THE BITS ARE GATED
    AND THE CALCULUS IS NOT.** That sentence is printed at the end of every
    run of this file."""
    print(
        "clause (f): CORRECTNESS. A transpose error is bit identical on"
        " three vendors, so identity is not correctness."
    )

    # ---- ARM 1 -----------------------------------------------------------
    var bq = 1
    var nh = 1
    var l = 2
    var s = 3
    var dy_i: List[Int] = [3, -2, 5, 1, 4, -6]
    var y_i: List[Int] = [2, 1, -3, 4, -1, 2]
    var dy = List[Float32]()
    var y = List[Float32]()
    for i in range(len(dy_i)):
        dy.append(Float32(dy_i[i]))
        y.append(Float32(y_i[i]))
    var zdot = List[Float32]()
    var ds = List[Float32]()
    for _ in range(bq * nh * l):
        zdot.append(Float32(0.0))
    for _ in range(bq * nh * l * s):
        ds.append(Float32(0.0))
    softmax_backward_into(dy, y, bq, nh, l, s, zdot, ds)

    var wrong = 0
    var control_moved = 0
    for t in range(l):
        var z_exact = 0
        for j in range(s):
            z_exact += dy_i[t * s + j] * y_i[t * s + j]
        if bits_of(zdot[t]) != bits_of(Float32(z_exact)):
            raise Error(
                String("transformer_backward_check: CLAUSE (f) ARM 1")
                + " FAILED at the z fold, row "
                + String(t)
                + ". Hand arithmetic says "
                + String(z_exact)
                + " ("
                + hexbits(Float32(z_exact))
                + ") and the oracle produced "
                + hexbits(zdot[t])
                + ". Every operand is a small integer and every partial is"
                + " far under 2^24, so this is EXACT and the disagreement is"
                + " a calculus error, not a rounding."
            )
        for j in range(s):
            var idx = t * s + j
            var want = y_i[idx] * (dy_i[idx] - z_exact)
            if bits_of(ds[idx]) != bits_of(Float32(want)):
                wrong += 1
            var flipped = y_i[idx] * (z_exact - dy_i[idx])
            if bits_of(Float32(want)) != bits_of(Float32(flipped)):
                control_moved += 1
    if wrong != 0:
        raise Error(
            String("transformer_backward_check: CLAUSE (f) ARM 1 FAILED on ")
            + String(wrong)
            + " of "
            + String(l * s)
            + " dS cells against hand-written exact integer arithmetic."
            + " The closed form is dS_j = y_j * (dy_j - z)."
        )
    if control_moved == 0:
        raise Error(
            "transformer_backward_check: CLAUSE (f) ARM 1's CONTROL IS DEAD."
            " The reversed subtraction `y_j * (z - dy_j)` gave the SAME"
            " values as `y_j * (dy_j - z)` on every cell, so this arm cannot"
            " see a transposed sign and its agreement proves nothing"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (f) arm 1: PASS, the softmax closed form matches"
        " hand-written EXACT integer arithmetic on "
        + String(l * s)
        + " dS cells and "
        + String(l)
        + " z folds, with NO EPSILON; and the reversed-subtraction control"
        " disagrees on "
        + String(control_moved)
        + " of "
        + String(l * s)
        + " cells, so the arm can see a sign error"
    )

    # ---- ARM 2 -----------------------------------------------------------
    var cases = clause_a_cases()
    var identity_cells = 0
    for ci in range(len(cases)):
        var c = fixture_case(cases[ci])
        var dims = fixture_dims(c)
        var w = fixture_weights(c)
        var x = fixture_x(c)
        var g = bwd_d_out(c, dims, DOUT_PLANT_NONE)
        var h = run_host_backward(c, dims, w, x, g, c.b, c.l)
        var d = compare_stage("mask identity", h[19], h[20], False)
        if d.n_diff != 0:
            raise Error(
                String("transformer_backward_check: CLAUSE (f) ARM 2 FAILED")
                + " on "
                + String(c.name)
                + ": bwd.d_attn_masked and bwd.d_attn_scores differ on "
                + String(d.n_diff)
                + " of "
                + String(d.n_cells)
                + " cells, first at cell "
                + String(d.first)
                + ". Plan 6.1 and DEVIATION 1414 say the mask backward is"
                + " the EXACT IDENTITY, which is why both stages are on the"
                + " card, and B13_MASK_ZEROES_GRAD is the arm that breaks"
                + " it. If they differ in a CLEAN build the identity is"
                + " false and the second stage is not what it claims."
            )
        identity_cells += d.n_cells
    print(
        "clause (f) arm 2: PASS, bwd.d_attn_masked == bwd.d_attn_scores"
        " bitwise on all "
        + String(identity_cells)
        + " cells of "
        + String(len(cases))
        + " cases -- the identity plan 6.1 records the second stage in order"
        " to check"
    )
    print(
        "clause (f) SCOPE: the ROUTING and the linear seams are covered by"
        " exact integers. **THE NONLINEAR SEAMS ARE NOT.** silu', the rsqrt"
        " tail and the softmax's transcendental inputs do not survive exact"
        " integers, and gating them needs a FLOAT64 DIRECTIONAL-DERIVATIVE"
        " reference at a stated tolerance which DOES NOT EXIST. **The bits"
        " are gated and the calculus is not.**"
    )


# ===========================================================================
# CLAUSE (g): EVERY CLAUSE ABOVE FALSIFIABLE BY A NAMED SABOTAGE
#
# DEVIATION 1535. The expectation table below is plan section 6.3's,
# DUPLICATED into code, and a duplicated table is a table that can drift.
# The alternative was to have the check read the plan, which is a markdown
# file, which would make the gate depend on parsing prose. The duplication
# is accepted and the mitigation is that every row cites 6.3 and that the
# ONE place this file's table knowingly departs from the plan's -- clause
# (c)'s length list, DEVIATION 1529 -- is argued at length in the header.
# ===========================================================================

#: An ordinary arm: it must move a stage on a clause-(a) case.
comptime ARM_ORDINARY = 0
#: An arm that passes clause (a) BY CONSTRUCTION and is falsifiable only in
#: clause (c)'s length half or clause (d). **These are the arms the forward
#: lane's contract warns will look inert and be deleted.**
comptime ARM_NEEDS_LENGTH = 1
#: An arm that CANNOT FIRE at any shape this profile has. Reported as a
#: SMOKE TEST by name; never green.
comptime ARM_SMOKE = 2
#: An arm for which no separating fixture exists that this gate may build.
comptime ARM_NOT_CONSTRUCTED = 3


@fieldwise_init
struct ArmExpectation(Copyable, Movable):
    """What one sabotage arm must do, from plan section 6.3's table.

    `first_stage` is the tag the arm's OWN SEAM writes, and the discipline
    is that the arm must move THAT stage "and no earlier one". An arm that
    moves an earlier stage is not aimed where it says it is; an arm that
    moves a later one has been absorbed on the way and its clause is being
    gated by the wrong stage.

    `inert_case` is a fixture on which it must move NOTHING, and it is the
    half that turns a smoke test into a REACH PROOF. **SEVEN OF THE
    TWENTY-TWO HAVE NONE**, and the reason is the same for all seven: this
    profile has no `d_model == 1` fixture, no `head_dim == 2` fixture and no
    fixture whose gate activations are all at or above 17. Plan 6.3 predicts
    those inert sets and the fixture set cannot express them, so they are
    reported as MOVED-BUT-UNMASKED, which is the honest label.

    `pos0_inert` is the three RoPE arms' clause and nothing else's. Plan 6.3
    requires it PER CELL and not per fixture: the arm must move NOTHING at
    ABSOLUTE POSITION 0, where `sin` is exactly `+0.0` and the transpose is
    the identity, so "the gate must COUNT moved cells and RAISE if the count
    equals the position-0 population".

    `predicts_count` is `B13_MASK_ZEROES_GRAD`'s clause: the oracle must
    predict the exact moved-cell count BEFORE the device is asked, because
    "it moved something" would also be satisfied by an arm that zeroed the
    whole buffer."""

    var arm: String
    var kind: Int
    var first_stage: String
    var witness_case: String
    var inert_case: String
    var pos0_inert: Bool
    var predicts_count: Bool
    var note: String


def arm_expectation(arm: String) raises -> ArmExpectation:
    if arm == "B01_DOT_UNFUSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.norm2.dot"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "the RMSNorm backward's c fold, product-then-add instead of"
                " one fma per term. **NO INERT CASE**: plan 6.3 predicts"
                " inert at d_model == 1 and on exactly-representable rows,"
                " and this profile has no d_model == 1 fixture. The d_out"
                " generator's own guard proves the rows are NOT exactly"
                " representable, which is the other half of the prediction"
            ),
        )
    if arm == "B01_DOT_DESCENDING":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.norm2.dot"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "the same fold walked DESCENDING. **NOT inert at"
                " d_model == 2**: an fma keeps the second product exact, so"
                " a two-term fma chain is order dependent where a two-term"
                " ADD chain is not. Inert at d_model == 1 ONLY, and there is"
                " no such fixture"
            ),
        )
    if arm == "B_RSTD_RECOMPUTE_DESCENDING":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.norm2.dx"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "rstd recomputed by re-folding the sum of squares DESCENDING"
                " instead of reading the saved norm*.sumsq. **Its ASCENDING"
                " half must move NOTHING and that half is what turns 'a"
                " recompute is bit exact' (DEVIATION 1420) from a belief"
                " into a checked statement** -- and the ascending half is"
                " what clause (a) checks on every clean build"
            ),
        )
    if arm == "B09_ROPE_TRANSPOSE_SIGN":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_q_proj_out"),
            String("base_b1_l4_nrep1"), String(""), True, False,
            String(
                "the RoPE backward with the FORWARD's sign convention: one"
                " character in two branches, producing a plausible"
                " correctly-shaped WRONG gradient. **INERT AT ABSOLUTE"
                " POSITION 0 -- per CELL** -- so the gate counts moved cells"
                " there and raises if any moved"
            ),
        )
    if arm == "B09_ROPE_HALVES_ADJACENT":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_q_proj_out"),
            String("base_b1_l4_nrep1"), String(""), True, False,
            String(
                "the rotate-half pairing replaced by the adjacent"
                " (2i, 2i+1) pairing -- the single most commonly"
                " mistranscribed thing about RoPE. Inert at head_dim == 2"
                " (no such fixture) and at position 0 (per cell, asserted)"
            ),
        )
    if arm == "B10_ROPE_BWD_FUSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_q_proj_out"),
            String("base_b1_l4_nrep1"), String(""), True, False,
            String(
                "the two products and the add fused into one fma: one"
                " rounding where the structure has three. Inert at position"
                " 0, where fma(rot, +0.0, a) and ftz(a + ftz(rot * +0.0))"
                " agree for a != -0.0 -- and plan 5.1(i) forbids a == -0.0"
            ),
        )
    if arm == "B10_DW_VIA_CHAIN":
        return ArmExpectation(
            arm, ARM_SMOKE, String(""), String(""), String(""), False, False,
            String(
                "**CANNOT FIRE AT THE GATE SHAPE AT ALL.** It replaces the"
                " routed k' = head_dim gemm with a hand chain, and the two"
                " agree bit for bit whenever P(head_dim) == 1, i.e."
                " head_dim <= 128. The fixtures are head_dim 16 and 24."
                " **DEVIATION 1405 is therefore pinned by ARGUMENT and by"
                " gemm v1's own certificate, NOT by a fired arm**, and that"
                " is recorded rather than discovered later"
            ),
        )
    if arm == "B11_DQ_VIA_GEMM":
        return ArmExpectation(
            arm, ARM_NEEDS_LENGTH, String(""), String("long_l257"),
            String("base_b1_l4_nrep1"), False, False,
            String(
                "dq routed through gemm v1 OP_NN at k' = S. **INERT AT EVERY"
                " S <= 128**, where P(S) == 1 and the gemm cell IS the"
                " whole-k ascending chain from +0.0. It passes clause (a) at"
                " a fixed shape and breaks the LENGTH clause, which is"
                " exactly the shape the forward contract warns will look"
                " inert and be deleted. Needs MOJOLEARN_TFB_CHECK_LONG=1"
            ),
        )
    if arm == "B11_DK_VIA_GEMM":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_k_cache"),
            String("base_b2_l4_nrep2"), String("base_b1_l4_nrep1"),
            False, False,
            String(
                "dk routed through gemm v1 OP_TN with the head group"
                " accumulated as per-head partials. **INERT ONLY WHEN"
                " L <= 128 AND n_rep == 1 TOGETHER**; at n_rep == 2 the"
                " partial accumulation is a different association and it"
                " fires on SHAPE alone, which is what makes"
                " base_b1_l4_nrep1 a real inert case and not a quiet one"
            ),
        )
    if arm == "B19_DV_VIA_GEMM":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_v_cache"),
            String("base_b2_l4_nrep2"), String("base_b1_l4_nrep1"),
            False, False,
            String(
                "dv routed the same way. **The backward twin of the"
                " forward's S19_VALUE_SUM_VIA_GEMM, and it inherits its"
                " warning verbatim**: without a clause that exercises the"
                " chopping it looks inert and gets deleted"
            ),
        )
    if arm == "B12_SCALE_INTO_DQ":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_qk_cell"),
            String("odd_head_dim_24"), String("base_b1_l4_nrep1"),
            False, False,
            String(
                "the attention scale folded into the dq chain instead of"
                " applied to the finished dS. **INERT AT EVERY POWER-OF-FOUR"
                " head_dim**: at 16 and 64 the scale is exactly 0.25 /"
                " 0.125, exact scaling commutes with an fma chain bitwise,"
                " and moving it to the far side changes nothing. The"
                " head_dim = 24 fixture contract section 3 already requires"
                " -- for a different reason -- is what makes this arm fire"
                " at all"
            ),
        )
    if arm == "B13_MASK_ZEROES_GRAD":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_attn_scores"),
            String("base_b1_l4_nrep1"), String(""), False, True,
            String(
                "the mask backward forced to write +0.0 at masked cells"
                " instead of passing the gradient through. It moves ONLY the"
                " masked cells whose dS is -0.0, and **the oracle predicts"
                " the exact count BEFORE the device is asked** -- because"
                " 'it moved something' would also be satisfied by an arm"
                " that zeroed the whole buffer"
            ),
        )
    if arm == "B18_SOFTMAX_DECOMPOSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_attn_masked"),
            String("base_b1_l4_nrep1"), String("base_b1_l1_nrep2"),
            False, False,
            String(
                "the softmax backward as the DECOMPOSED autograd graph."
                " **AND THE ARM IS WEAKER THAN THE CLAUSE IT GUARDS**: it"
                " separates the ASSOCIATION (y*dy - y*z against y*(dy - z))"
                " and NOT the argmax, because a device kernel cannot scatter"
                " an argmax residue without a float ATOMIC and an atomic is"
                " what this whole file refuses. **The argmax half of"
                " DEVIATION 1406 is UNFIRABLE BY CONSTRUCTION**, which is"
                " itself an argument for the refusal and is recorded rather"
                " than papered over"
            ),
        )
    if arm == "B18_ZFOLD_UNFUSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.attn.zdot"),
            String("base_b1_l4_nrep1"), String("base_b1_l1_nrep2"),
            False, False,
            String(
                "the z fold spelled product-then-add. Inert at s == 1 and on"
                " any row whose products are all exactly representable --"
                " and guard_d_out_separates PROVES this fixture's are not"
            ),
        )
    if arm == "B18_ZFOLD_DESCENDING":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.attn.zdot"),
            String("base_b1_l4_nrep1"), String("base_b1_l1_nrep2"),
            False, False,
            String(
                "the z fold walked DESCENDING over the key axis. **Inert at"
                " s == 1 ONLY -- NOT at s == 2**, because an fma keeps the"
                " second product exact and the two orders round different"
                " quantities, where a two-term ADD chain would be order free"
            ),
        )
    if arm == "B20_SIGMOID_FROM_SILU":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_gate_proj_out"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "sigmoid reconstructed as silu_out / gate_proj_out instead"
                " of recomputed. **NO INERT CASE**: plan 6.3 predicts inert"
                " for every gate activation at or above about 17, where"
                " 1 + exp(-x) rounds to exactly 1.0, and no fixture here has"
                " all-large gate activations. It also produces 0/0 at a zero"
                " gate, which the profile's refusals do NOT catch because"
                " they run on inputs and this is an intermediate"
            ),
        )
    if arm == "B20_SILU_DERIV_ALT_ASSOC":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_gate_proj_out"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "sg + x*sg*(1-sg) instead of sg*(1 + x*(1-sg)). Inert"
                " wherever sg is exactly 1.0 and at x == 0; no fixture"
                " isolates either"
            ),
        )
    if arm == "B20_SILU_DERIV_FUSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_gate_proj_out"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "fma(x, 1-sg, 1.0): one rounding where the pinned form has"
                " two. Inert wherever x * (1 - sg) is exactly representable,"
                " which is the class of blindness the d_out generator's own"
                " guard exists to rule out"
            ),
        )
    if arm == "B_FANIN_ZERO_SEED":
        return ArmExpectation(
            arm, ARM_NOT_CONSTRUCTED, String("bwd.d_norm1_out"), String(""),
            String(""), False, False,
            String(
                "**NO CONSTRUCTIBLE WITNESS IN THIS LANE. DEVIATION 1528.**"
                " Plan 6.3 says 'vacuous without a plant'; it is worse. The"
                " arm fires only where the FIRST accumulated term of the"
                " fan-in is -0.0, and that term is bwd.d_q_proj_out, the"
                " RoPE transpose of bwd.d_q_rope, which is a +0.0-seeded fma"
                " chain -- and plan 5.1(i) PROVES such a chain never holds"
                " -0.0 (this file's preflight asserts it). The transpose"
                " cannot manufacture one either. The only input this gate"
                " controls is d_out, and driving it to all -0.0 gives +0.0"
                " at d_q_rope for exactly that reason. **The arm needs a"
                " planted -0.0 INSIDE a saved forward stage, which is a"
                " fixture edit this agent may not make.**"
            ),
        )
    if arm == "B_FANIN_ORDER_QKV_REVERSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_norm1_out"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "the three-term fan-in accumulated v, k, q instead of"
                " q, k, v. **NO INERT CASE**: plan 6.3 predicts inert"
                " wherever any two of the three terms are zero and no"
                " fixture produces that"
            ),
        )
    if arm == "BWD_UNTRANSPOSED":
        return ArmExpectation(
            arm, ARM_ORDINARY, String("bwd.d_mlp_gated"),
            String("base_b1_l4_nrep1"), String(""), False, False,
            String(
                "the GEMM lane's own arm, reaching this file through"
                " _route_a and _route_b. The transpose is dropped and"
                " dA = dC . B / dB = A . dC are computed as written -- the"
                " mistake gemm_backward.mojo exists to prevent, and the one"
                " most likely to survive a careless gate because it computes"
                " a perfectly plausible matrix of the right size. Its first"
                " stage here is bwd.d_mlp_gated, the FIRST routed call the"
                " backward makes"
            ),
        )
    if arm == "BWD_OPERAND_ORDER":
        return ArmExpectation(
            arm, ARM_SMOKE, String(""), String(""), String(""), False, False,
            String(
                "**INERT THROUGH THIS LANE ENTIRELY.** The dC side flag is"
                " ignored and dC is always passed LEFT, which is correct for"
                " three of the six gemm-backward calls and wrong for the"
                " other three (OP_TN's dA, and dB for forward OP_NN and"
                " OP_TN). **Every forward call in this profile is OP_NT**,"
                " whose dA and dB both put dC LEFT, so the arm is a no-op"
                " here. Recorded as a SMOKE TEST rather than run and"
                " reported green"
            ),
        )
    raise Error(
        String("transformer_backward_check: '")
        + arm
        + "' is not one of plan section 6.3's twenty-two sabotage names. If"
        + " an arm was added to transformer_backward.mojo or to"
        + " gemm_backward.mojo, this table and plan section 6.3 both owe it"
        + " a row."
    )


def find_verdict(
    verdicts: List[CaseVerdict], name: String
) raises -> CaseVerdict:
    for i in range(len(verdicts)):
        if verdicts[i].name == name:
            return verdicts[i].copy()
    raise Error(
        String("transformer_backward_check: the sabotage expectation names")
        + " case '"
        + name
        + "' and it was not in the clause-(a) set this build ran. The arm"
        + " CANNOT BE EVALUATED, which is NOT the same as the arm passing"
        + " ([[reached-but-inert]])."
    )


def clause_g(
    ctx: DeviceContext, arm: String, verdicts: List[CaseVerdict]
) raises:
    """The INVERTED verdict of a sabotage build.

    When an arm is armed this file INVERTS: a clean compare is the FAILURE,
    because it means the sabotage was reached and made no difference, or was
    never reached at all. Both are `[[reached-but-inert]]` and both are
    reported as such rather than as a pass.

    **AN INERT-LOOKING ARM HERE IS THE GATE ON THE GATE.** Plan 6.3's last
    paragraph is that four of the twenty-two are predicted inert at the gate
    shape and two of those cannot be made to fire without new fixtures, and
    the forward lane's own experience is that such arms get deleted. So a
    SMOKE arm raises a NAMED error rather than printing a pass, and an arm
    whose witness case was not in this build's set raises rather than
    silently reporting nothing."""
    var exp = arm_expectation(arm)
    print("clause (g): arm " + arm + " -- " + exp.note)

    if exp.kind == ARM_SMOKE:
        raise Error(
            String("transformer_backward_check: SABOTAGE ")
            + arm
            + " IS A SMOKE TEST AND THIS BINARY WAS BUILT WITH IT. It cannot"
            + " fire at any shape this profile has, so whatever this run"
            + " prints is a statement about nothing. "
            + exp.note
            + " Reporting it green would be reporting that a gate cannot see"
            + " something as though it had looked"
            + " ([[reached-but-inert]])."
        )

    if exp.kind == ARM_NOT_CONSTRUCTED:
        raise Error(
            String("transformer_backward_check: SABOTAGE ")
            + arm
            + " HAS NO CONSTRUCTIBLE WITNESS and this binary was built with"
            + " it. "
            + exp.note
        )

    if exp.kind == ARM_NEEDS_LENGTH:
        # ---- clause (a) is EXPECTED to be green, and that is not the pass
        var moved_cases = 0
        var first_case = String("")
        for i in range(len(verdicts)):
            if verdicts[i].n_moved > 0:
                moved_cases += 1
                if first_case == "":
                    first_case = verdicts[i].name + " at " + verdicts[i].first
        if not env_on("MOJOLEARN_TFB_CHECK_LONG"):
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " IS ARMED AND ITS ONLY WITNESS IS NOT IN THIS BUILD'S"
                + " CASE SET. It is INERT at every S <= 128 by construction,"
                + " so the default set can say nothing about it. Set"
                + " MOJOLEARN_TFB_CHECK_LONG=1 to add long_l257. **The arm"
                + " is UNGATED here, which is not the same as the arm"
                + " passing** ([[reached-but-inert]])."
            )
        if moved_cases == 0:
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " IS ARMED, long_l257 IS IN THE SET, AND CLAUSE (a) MOVED"
                + " NO BIT ON ANY CASE. At S = 257 gemm v1's P is 3 where"
                + " the profile's chain has one leaf, so the arm must move"
                + " something. Either its branch was never reached or it is"
                + " inert there ([[reached-but-inert]])."
            )
        print(
            "clause (g): "
            + arm
            + " BIT with long_l257 in the set: "
            + String(moved_cases)
            + " of "
            + String(len(verdicts))
            + " cases moved, first at "
            + first_case
            + ". At S <= 128 it is inert BY CONSTRUCTION, which is why the"
            " default set cannot see it and why plan 6.3 predicts exactly"
            " that."
        )
        var iv = find_verdict(verdicts, exp.inert_case)
        if iv.n_moved != 0:
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " moved "
                + String(iv.n_moved)
                + " stages on "
                + exp.inert_case
                + " (S = 4), and plan 6.3 says it is INERT at every"
                + " S <= 128 because P(S) == 1 there and the gemm cell IS"
                + " the profile's chain. An arm that moves at S = 4 is not"
                + " the arm the plan describes."
            )
        print(
            "clause (g): "
            + arm
            + " is INERT on "
            + exp.inert_case
            + " (S = 4 <= 128), which is the half that makes it a REACH"
            " PROOF rather than a smoke test."
        )
        return

    # ---- the ordinary arms ----------------------------------------------
    var wv = find_verdict(verdicts, exp.witness_case)
    if wv.n_moved == 0:
        raise Error(
            String("transformer_backward_check: SABOTAGE ")
            + arm
            + " IS ARMED AND MOVED NO BIT on its witness case "
            + exp.witness_case
            + ". Either its branch was never reached at this shape or it is"
            + " inert there ([[reached-but-inert]]). It falsifies NOTHING"
            + " and must not be reported as a passing arm."
        )
    var want = stage_index_of(exp.first_stage)
    if wv.first_index != want:
        raise Error(
            String("transformer_backward_check: SABOTAGE ")
            + arm
            + " moved '"
            + wv.first
            + "' FIRST on "
            + exp.witness_case
            + ", and plan section 6.3 says its own seam writes '"
            + exp.first_stage
            + "'. Each arm must move the stage its OWN seam writes and no"
            + " earlier one; an earlier stage means the arm is not aimed"
            + " where it says it is."
        )
    print(
        "clause (g): "
        + arm
        + " BIT on "
        + exp.witness_case
        + ": "
        + String(wv.n_moved)
        + " of 37 stages moved, FIRST at "
        + exp.first_stage
        + ", which is the stage its own seam writes."
    )

    if exp.pos0_inert:
        if wv.pos0_moved != 0:
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " moved "
                + String(wv.pos0_moved)
                + " cells of bwd.d_q_proj_out AT ABSOLUTE POSITION 0, and"
                + " plan 6.3 requires it to move NOTHING there. At position"
                + " 0 `sin` is exactly +0.0 and `cos` exactly 1.0, so the"
                + " transposed rotation IS the forward rotation IS the"
                + " identity, and a sign, a pairing or a fusion cannot"
                + " change it. **The mask is PER CELL, not per fixture**,"
                + " which is why the gate counts rather than compares"
                + " ([[verify-reach-not-output]])."
            )
        print(
            "clause (g): "
            + arm
            + " moved 0 cells at ABSOLUTE POSITION 0, where sin is exactly"
            " +0.0 and the transpose is the identity. That PER-CELL mask is"
            " what makes it a reach proof rather than a smoke test."
        )

    if exp.predicts_count:
        if wv.mask_moved != wv.mask_predicted:
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " moved "
                + String(wv.mask_moved)
                + " cells of bwd.d_attn_scores and the HOST ORACLE PREDICTED "
                + String(wv.mask_predicted)
                + " -- the masked cells whose pinned dS is -0.0, counted"
                + " from the oracle's own dump BEFORE the device was asked."
                + " A different count means the arm is not doing what plan"
                + " 6.3 says it does, and 'it moved something' would have"
                + " been satisfied by an arm that zeroed the whole buffer."
            )
        if wv.mask_predicted == 0:
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " HAS A PREDICTED COUNT OF ZERO on "
                + exp.witness_case
                + ", so no masked cell of the oracle's dS is -0.0 and the"
                + " arm is unfalsifiable on this fixture. Clause (e)'s"
                + " signed-zero audit is the half that catches this before"
                + " the arm is fired ([[reached-but-inert]])."
            )
        print(
            "clause (g): "
            + arm
            + " moved EXACTLY the "
            + String(wv.mask_predicted)
            + " cells the host oracle predicted -- the masked cells whose"
            " pinned dS is -0.0 -- which is a MEASUREMENT and not a smoke"
            " test."
        )

    if exp.inert_case != "":
        var iv = find_verdict(verdicts, exp.inert_case)
        if iv.n_moved != 0:
            raise Error(
                String("transformer_backward_check: SABOTAGE ")
                + arm
                + " moved "
                + String(iv.n_moved)
                + " stages on "
                + exp.inert_case
                + ", first at "
                + iv.first
                + ", and plan section 6.3 requires it to move NOTHING there."
                + " An arm that moves everywhere is a smoke test; the inert"
                + " case is what makes it a REACH PROOF"
                + " ([[verify-reach-not-output]])."
            )
        print(
            "clause (g): "
            + arm
            + " is INERT on "
            + exp.inert_case
            + ", 37/37 stages unmoved, which is the half that makes it a"
            " reach proof rather than a smoke test."
        )
    elif not exp.pos0_inert and not exp.predicts_count:
        print(
            "clause (g): "
            + arm
            + " has **NO ASSERTED INERT CASE** and is therefore"
            " MOVED-BUT-UNMASKED, not a reach proof. "
            + exp.note
        )
    _ = ctx


# ===========================================================================


def main() raises:
    var armed = llama_backward_sabotage_name()
    var gemm_bwd_armed = gemm_backward_sabotage_name()
    var fwd_armed = llama_block_sabotage_name()
    var gemm_fwd_armed = gemm_sabotage_name()

    print(
        "=== transformer BACKWARD identity gate, profile"
        " mojolearn.identical.transformer.fp32.v1"
    )
    print(
        "=== NOTHING IN THIS FILE, NOR IN EITHER FILE IT DRIVES, HAS EVER"
        " BEEN COMPILED OR RUN BEFORE THIS PROCESS. Read the header."
    )
    # ALL FOUR ARE PRINTED and `transformer_backward.mojo::
    # llama_backward_sabotage_name` asks for exactly that in its own
    # docstring: "a backward binary can carry a forward arm, a gemm arm and
    # a gemm-backward arm as well, and a banner naming only one of the four
    # MISLABELS THE RUN."
    print(
        "mode "
        + mode_name()
        + "   backward sabotage: "
        + armed
        + "   gemm-backward: "
        + gemm_bwd_armed
        + "   forward block: "
        + fwd_armed
        + "   gemm forward: "
        + gemm_fwd_armed
    )

    # ---- WHICH ARM THIS BINARY WAS BUILT WITH ---------------------------
    # DEVIATION 1530, and it is `tools/gemm_ladder.sh:71`'s scar written
    # down. A `-D MOJOLEARN_TRANSFORMER_SABOTAGE_...` with a typo in it is
    # SILENTLY IGNORED by the compiler: `is_defined` returns False and the
    # build is clean. The operator then sees a green gate and records it as
    # "arm X did not bite", which is the exact inverse of the truth.
    var effective = armed
    if effective == "none" and gemm_bwd_armed != "none":
        effective = gemm_bwd_armed
    var expect = env_str("MOJOLEARN_TFB_EXPECT_SABOTAGE")
    if expect != "":
        if expect != effective:
            raise Error(
                String("transformer_backward_check: the caller expected")
                + " sabotage '"
                + expect
                + "' and this BINARY carries '"
                + effective
                + "'. A misspelled -D is silently ignored by the compiler"
                + " and produces a clean build that a caller reads as 'the"
                + " arm did not bite'. Fix the -D or the expectation."
            )
        print(
            "ledger: the caller expected '"
            + expect
            + "' and the binary agrees, so the -D was not silently dropped"
        )
    elif effective == "none":
        print(
            "ledger: this binary is CLEAN -- no sabotage arm is compiled in."
            " Set MOJOLEARN_TFB_EXPECT_SABOTAGE to have the binary check its"
            " own -D."
        )
    else:
        print(
            "ledger: this binary carries sabotage '"
            + effective
            + "' and the caller did not say so. Set"
            " MOJOLEARN_TFB_EXPECT_SABOTAGE to close the misspelled -D hole."
        )
    if fwd_armed != "none" or gemm_fwd_armed != "none":
        print(
            "ledger: WARNING -- a FORWARD arm is also compiled in ('"
            + fwd_armed
            + "' / '"
            + gemm_fwd_armed
            + "'). The backward reads the forward's SAVED STAGES, so a"
            " forward arm moves this file's inputs and every backward"
            " verdict below is a verdict about a different fixture. This is"
            " reported and not refused, because a deliberate"
            " forward-plus-backward build is a legitimate experiment -- but"
            " it is not a backward sabotage measurement."
        )

    preflight()

    var ctx = DeviceContext()

    # ---- CLAUSE (a) ------------------------------------------------------
    # The FIRST case in the set writes the card, at the caller's path, under
    # this driver's prefix. The rest run with the trace DISABLED, because
    # `IdentityTrace` enforces tag UNIQUENESS within one trace and fourteen
    # cases would emit `tbx.bwd.d_x` fourteen times and raise. A per-case
    # prefix would be the alternative and it is deliberately not taken: the
    # card `tools/e3_round_judge.sh` reads must have the thirty-seven tags
    # of plan section 6.1 and nothing else, and a 518-record card is not
    # that card.
    var cases = clause_a_cases()
    print(
        "clause (a): "
        + String(len(cases))
        + " fixture cases, all 37 stages, device vs host oracle, BITWISE"
    )
    var verdicts = List[CaseVerdict]()
    var cpath = card_path()
    for ci in range(len(cases)):
        if ci == 0:
            var trace = IdentityTrace.to_path(cpath)
            verdicts.append(clause_a_case(ctx, cases[ci], trace, TAG_PREFIX))
        else:
            var off = IdentityTrace.disabled()
            var pfx = "case" + String(cases[ci])
            verdicts.append(clause_a_case(ctx, cases[ci], off, pfx))

    _ = check_card_tags(cpath)

    var moved_cases = 0
    var first_case = String("")
    var all_cells = 0
    for i in range(len(verdicts)):
        all_cells += verdicts[i].cells
        if verdicts[i].n_moved > 0:
            moved_cases += 1
            if first_case == "":
                first_case = verdicts[i].name + " at " + verdicts[i].first

    var any_armed = effective != "none"
    if any_armed:
        clause_g(ctx, effective, verdicts)
        print(
            "clauses (b), (c) and (d) are NOT run under a sabotage build:"
            " they are INVARIANCE claims and a deterministic sabotage"
            " satisfies them. Clause (e) is not run either: the refusal is"
            " upstream of every sabotaged seam. Clause (f) is not run"
            " because it is a HOST correctness arm and a device sabotage"
            " cannot reach it."
        )
    else:
        if moved_cases != 0:
            var n = verdicts[0].n_moved
            var f = verdicts[0].first
            for i in range(len(verdicts)):
                if verdicts[i].n_moved > 0:
                    n = verdicts[i].n_moved
                    f = verdicts[i].first
                    break
            if mode_is_identical():
                raise Error(
                    String("transformer_backward_check: CLAUSE (a) FAILED, ")
                    + String(n)
                    + " stages differ from the oracle, first at "
                    + f
                    + "  (case "
                    + first_case
                    + ")"
                )
            else:
                # FAST arms of (a) are RECORDED, not asserted, where they
                # are vendor-shaped -- plan 6.2(a), and the metrics lane's
                # leg-11 lesson. Under FAST every `identical_*` compiles
                # away to the platform's own spelling and the two halves of
                # this lane are two different platforms' libm, so a
                # difference here is EXPECTED and means nothing about the
                # profile.
                print(
                    "clause (a) [FAST]: RECORDED, NOT ASSERTED. "
                    + String(moved_cases)
                    + " of "
                    + String(len(verdicts))
                    + " cases differ from the oracle, first at "
                    + first_case
                    + ". FAST is unversioned and makes no identity claim."
                )
        else:
            print(
                "clause (a): PASS, "
                + String(len(verdicts))
                + " cases, 37/37 stages bit-identical to the oracle on all "
                + String(all_cells)
                + " cells, "
                + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
                + "/"
                + String(TRANSFORMER_BACKWARD_STAGE_COUNT)
                + " card tags"
            )

        if env_on("MOJOLEARN_TFB_CHECK_CLAUSE_B"):
            clause_b(ctx, 0)
        else:
            print(
                "clause (b): SKIPPED (set MOJOLEARN_TFB_CHECK_CLAUSE_B=1)"
            )

        if env_on("MOJOLEARN_TFB_CHECK_CLAUSE_C"):
            clause_c_batch(ctx, fixture_case_by_name("base_b3_l16_nrep2"))
            clause_c_length(ctx, fixture_case_by_name("long_l64"))
        else:
            print(
                "clause (c): SKIPPED (set MOJOLEARN_TFB_CHECK_CLAUSE_C=1)."
                " BOTH HALVES ARE SKIPPED TOGETHER and that is deliberate:"
                " batch composition and sequence length are two different"
                " fixtures with two DIFFERENT MOVING SETS (DEVIATION 1529),"
                " and a lane that ran one would have half a clause with a"
                " whole clause's name."
            )

        if env_on("MOJOLEARN_TFB_CHECK_CLAUSE_D"):
            var dv = clause_d_device(
                ctx, fixture_case_by_name("pos_offset_129")
            )
            if dv.bad != 0:
                raise Error(
                    String("transformer_backward_check: CLAUSE (d) FAILED")
                    + " on "
                    + String(dv.bad)
                    + " stage-tokens, first at token "
                    + String(dv.first_token)
                    + ", "
                    + dv.first_stage
                    + ". Plan 5.1 makes the key-axis activation gradients"
                    + " chopping-invariant BY CONSTRUCTION, so a failure"
                    + " here is a finding about the profile and not about"
                    + " the gate."
                )
            if dv.moved_query_axis == 0:
                raise Error(
                    "transformer_backward_check: CLAUSE (d)'s NEGATIVE HALF"
                    " FAILED. NOT ONE query-axis stage moved between the"
                    " chunked and unchunked runs. Plan 5.2 says"
                    " bwd.d_k_cache, bwd.d_v_cache and every weight gradient"
                    " are SUMS OVER THE QUERIES IN THIS CALL, so a chunked"
                    " backward MUST produce a partial. A gate asserting only"
                    " the positive half would pass on an implementation that"
                    " computed constants."
                )
            print(
                "clause (d) device NEGATIVE: PASS, "
                + String(dv.moved_query_axis)
                + " query-axis stages are PARTIALS under chunking, which is"
                " what plan 5.2 says a per-call backward computes"
            )
            clause_d_carry_theorem()
        else:
            print(
                "clause (d): SKIPPED (set MOJOLEARN_TFB_CHECK_CLAUSE_D=1)."
                " NOTE: two sabotage arms (B11_DK_VIA_GEMM's and"
                " B19_DV_VIA_GEMM's chopping half) are sharpest here, and"
                " DEVIATION 1527 means the CARRIED-ACCUMULATOR half is a"
                " DEMONSTRATION and not a gate on either code path."
            )

        if env_on("MOJOLEARN_TFB_CHECK_CLAUSE_E"):
            clause_e(ctx, 0)
        else:
            print(
                "clause (e): SKIPPED (set MOJOLEARN_TFB_CHECK_CLAUSE_E=1)"
            )

        if env_on("MOJOLEARN_TFB_CHECK_CLAUSE_F"):
            clause_f()
        else:
            print(
                "clause (f): SKIPPED (set MOJOLEARN_TFB_CHECK_CLAUSE_F=1)."
                " NOTE: it is the ONLY clause in this file that asks whether"
                " the gradient is the GRADIENT. A transpose error is bit"
                " identical on three vendors, so every other clause here can"
                " be green on an answer that is not the answer."
            )

        print(
            "SCOPE: this build, this column, "
            + mode_name()
            + " only. What is NOT closed by anything printed above:"
            " **THE CALCULUS.** Clause (f) covers the routing and the linear"
            " seams by exact integers; silu', the rsqrt tail and the softmax"
            " closed form need a FLOAT64 DIRECTIONAL-DERIVATIVE reference at"
            " a stated tolerance and THERE IS NONE -- transformer/corpus/"
            " does not exist and transformer_oracle.mojo deliberately"
            " carries no float64 reference. **THE BITS ARE GATED AND THE"
            " CALCULUS IS NOT.** Also not closed: **plan 5.3's carried"
            " chunk theorem on a device** (DEVIATION 1527 -- neither half of"
            " this lane has a chunked entry point, so clause (d2) is a"
            " demonstration and not a gate); **B10_DW_VIA_CHAIN and"
            " BWD_OPERAND_ORDER**, which cannot fire at any shape this"
            " profile has and are SMOKE TESTS by name;"
            " **B_FANIN_ZERO_SEED**, which has no constructible witness"
            " (DEVIATION 1528); **seven arms with no asserted inert case**,"
            " because this profile has no d_model == 1, no head_dim == 2 and"
            " no all-large-gate fixture; **an INDEPENDENT reference** --"
            " every clause here is our device against our oracle and both"
            " are ours; **FAST mode**; **the twenty-one sabotage builds this"
            " binary is not**; and **every column that is not this one** --"
            " plan section 11: nothing cross-vendor until a leg runs, and"
            " Apple and AMD agreed bit for bit through 302 GEMM stages while"
            " NVIDIA diverged at tree001.winners.scores, so two backends"
            " agreeing closes nothing."
        )


# ===========================================================================
# OWED, AND WHY I DID NOT DO IT HERE
#
# This file is the only one this lane's gate agent was permitted to write.
# Everything below is a belief about ANOTHER file, recorded rather than
# acted on, so that the next agent inherits the finding instead of
# rediscovering it. **None of it has been verified by running anything.**
#
# 1. **A CHUNKED BACKWARD ENTRY POINT, DEVIATION 1527.** Plan section 5.3's
#    carried-accumulator chunk theorem is the property the plan calls "the
#    one this lane gets for free that the GEMM lane cannot have", and
#    NEITHER HALF OF THE LANE IMPLEMENTS IT. `llama_decoder_layer_backward`
#    and `transformer_block_backward_oracle` both take no seed, no carry
#    flag and no accumulator argument. The shape it needs is the embedding
#    lane's `accumulate` exactly: one `Bool` that says "do not zero-fill the
#    four accumulating outputs, seed their chains from what you were
#    handed". Until it exists, clause (d2) is a demonstration on a
#    hand-written chain and the theorem is ungated on both halves.
#
# 2. **PLAN SECTION 6.2's CLAUSE (c) LIST IS WRONG, DEVIATION 1529.** It
#    names `bwd.d_x` and `bwd.d_norm1_out` as SEQUENCE-LENGTH invariant and
#    the plan's own section 5.2 says they are not: both are downstream of
#    `bwd.d_k_proj_out` / `bwd.d_v_proj_out`, which are the `[pos0, S)`
#    slices of query-axis sums. `[[fix-docs-on-discovery]]` binds whoever
#    edits that file: **delete the false sentence, do not soften it.** The
#    corrected clause is `is_query_axis_stage` in this file, and the
#    BATCH half of the clause is unaffected -- `bwd.d_k_cache` is
#    `[B, n_kv, S, hd]` and IS batch-composition invariant. **The two halves
#    have different moving sets and the plan uses one list for both.**
#
# 3. **PLAN 6.2(d)'s "MUST MOVE at any chunk of 2 or more terms" IS VERY
#    SLIGHTLY TOO STRONG.** `clause_d_carry_theorem` measures it: over
#    `{2^24, 1 x 7}` the uncarried two-partial spelling is INERT at the
#    splits that leave a single absorbed term on the far side, because
#    `2^24 + 1` rounds back to `2^24`. The sentence should read "must move
#    at SOME split of 2 or more terms, and the gate must report which".
#
# 4. **A `d_out` GENERATOR IN `transformer_fixture.mojo`, DEVIATION 1531.**
#    That file's eleven tensor ids stop at the cache tail because it predates
#    the backward, so this gate carries its own generator and its own
#    separation guard. Two generators in two files is two chances to be
#    edited apart, and the right home is the fixture. It is a file this
#    agent may not edit.
#
# 5. **A `d_model == 1` FIXTURE CASE.** Plan 6.3 predicts that inert set for
#    `B01_DOT_UNFUSED`, `B01_DOT_DESCENDING` and
#    `B_RSTD_RECOMPUTE_DESCENDING`, and the fixture set cannot express it,
#    so all three are reported MOVED-BUT-UNMASKED rather than as reach
#    proofs. One case closes three arms.
#
# 6. **AN `x` FIELD ON `LlamaDeviceStages`**, plan section 1's cross-lane
#    request. A one-line edit that deletes an argument from
#    `llama_decoder_layer_backward` and removes the one place this gate has
#    to keep a forward buffer alive by hand.
#
# 7. **A RUNNER FOR THE TWENTY-TWO SABOTAGE BUILDS.** `is_defined` is a
#    compile-time query, so exercising the set is twenty-two compiles. What
#    is owed is a script that, per arm, builds with the arm's `-D`, exports
#    `MOJOLEARN_TFB_EXPECT_SABOTAGE=<arm>` and `MOJOLEARN_IDENTITY_TRACE=
#    <out>/<arm>.card`, runs this file and requires a NONZERO exit.
#    `B11_DQ_VIA_GEMM` additionally needs `MOJOLEARN_TFB_CHECK_LONG=1`, and
#    `B10_DW_VIA_CHAIN`, `BWD_OPERAND_ORDER` and `B_FANIN_ZERO_SEED` are
#    expected to raise a NAMED "smoke test" or "not constructed" error
#    rather than a bitten-arm one, which a runner must distinguish or it
#    will record three false failures. A shell file under `tools/` is
#    outside this agent's remit.
#
# 8. **`transformer/corpus/` AND THE FLOAT64 DIRECTIONAL DERIVATIVE.** The
#    only two things that can catch our oracle being wrong in the same way
#    as our device. Neither exists. Plan section 11 calls the float64
#    reference the lane's largest owed item and this file agrees: **a
#    transpose error is bit identical on three vendors.**
#
# 9. **`IDENTITY_PATHS.md`, `CARD_GAPS.md` AND `UNWIRED.md`.** No row names
#    the backward profile. DEVIATION 1400 reserves an IDENTITY_PATHS row and
#    plan section 11 says it must not be written until a three-vendor leg
#    runs; `UNWIRED.md` is where "specified, never compiled" belongs and
#    that is now the state of three files in this directory.
#
# 10. **`tools/e1_bootstrap.sh` PHASE 8.** This lane's card is not wired into
#     the leg's judge. What phase 8 needs is one entry that sets
#     `MOJOLEARN_IDENTITY_TRACE` to
#     `<out>/lanes/transformer_backward.identical.card` and runs this file.
#     DEVIATION 1526 is why that will now work and DEVIATION 970 is why it
#     would not have.
# ===========================================================================
