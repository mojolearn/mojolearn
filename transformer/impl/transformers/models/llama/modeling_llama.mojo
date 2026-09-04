# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`transformers/models/llama/modeling_llama.py`: ONE Llama-shaped decoder
block, on the device, under profile
`mojolearn.identical.transformer.fp32.v1`. **COPY, DO NOT IMPROVE.**

PORT OF huggingface/transformers at `d56c55b`,
`src/transformers/models/llama/modeling_llama.py`, read on disk at
`/Users/andrewhendel/CascadeProjects/upstream/transformers/` on 2026-08-24.
Partial, inference only, eager attention only. What is MIRRORED here, symbol
by symbol (line numbers verified against that checkout, not quoted from the
contract):

| theirs | lines | here |
|---|---|---|
| `LlamaRMSNorm.forward` | :62-67 | `llama_rms_norm_kernel`, `llama_rms_norm` |
| `LlamaRotaryEmbedding.compute_default_rope_parameters` | :93-109 | `llama_rope_inv_freq_host` |
| `LlamaRotaryEmbedding.forward` | :113-127 | `llama_rope_table_kernel`, `LlamaRopeTable` |
| `rotate_half` + `apply_rotary_pos_emb` | :130-134, :138-160 | `apply_rotary_pos_emb_kernel`, `apply_rotary_pos_emb` |
| `repeat_kv` | :179-188 | an INDEX MAP, `h // n_rep`; a copy, never materialized |
| `eager_attention_forward` | :191-213 | `eager_attention_forward` and the nine kernels it launches |
| `LlamaMLP.forward` | :174-176 | `llama_mlp_forward` |
| `LlamaAttention.forward` | :243-281 | `llama_attention_forward` |
| `LlamaDecoderLayer.forward` | :295-324 | `llama_decoder_layer_forward` |

THE CONTRACT IS `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` AND IT IS
FROZEN. Section 4's seam table decides every rounding below; section 9's
stage list decides every tag and their order; section 5 decides the softmax.
This file CONSUMES that document and never amends it. The host oracle
`transformer/checks/transformer_oracle.mojo` is the ANSWER, bit for bit;
this file is an independent transcription of the same order into kernels and
the two share only the seam functions themselves (`ftz`,
`identical_mul_add`, `identical_exp`, `identical_div`, `identical_rsqrt`,
`identical_sin`, `identical_cos`, `identical_fmax`, `identical_silu`,
`portable_powf`), by design.

**NOTHING IN THIS FILE HAS BEEN COMPILED OR RUN.** It was written on
2026-08-25 by a lane that is forbidden to execute anything. Every statement
below about what it computes is CONSTRUCTION; every statement about what
that would mean across vendors is PREDICTION. No claim here is measured, and
in particular no sentence in this file says the block is bit-identical
across vendors, because no leg has run.

WHAT THIS FILE DOES NOT REBUILD, PER CONTRACT SECTION 0
---------------------------------------------------------
This is the founding instruction of the lane and it is checked here rather
than assumed.

* **Every linear projection and the QK product** go through
  `gemm/checks/gemm_identical.mojo::identical_gemm`, profile
  `mojolearn.identical.gemm.fp32.v1`. Seven `OP_NT` projection calls and
  `B * n_heads` `OP_NT` score calls. There is no matmul in this file.
* **Every transcendental** is `checks/numerics.mojo`'s. No exp, no
  division, no reciprocal square root, no sine, no cosine, no maximum and no
  power is spelled here.
* **The residual add (S22, S23)** is IMPORTED from the mamba lane's device
  spelling, `residual_add_kernel`. Contract section 0 marks it REUSED and it
  is reused literally, not transcribed.
* **`pinned_mul` (DEVIATION 720)** is IMPORTED from the same file rather
  than copied a fourth time. Contract section 12.3's DEVIATION 816 permits
  either; the import is the one that does not add a place to drift.
* **`identical_silu` (DEVIATION 744)** is the numerics lane's single-quotient
  SiLU. Not spelled here.

The one thing NOT reused is RMSNorm, and it is not reused for a reason that
is a cross-lane BLOCKER, not a preference. See DEVIATION 1025 below.

THE LANE'S OWN NEW ARITHMETIC IS ROPE, SOFTMAX AND THE VALUE SUM
------------------------------------------------------------------
Contract section 0's founding sentence. In this file that is
`llama_rope_inv_freq_host` (S6), `llama_rope_table_kernel` (S7, S8),
`apply_rotary_pos_emb_kernel` (S9, S10), `attn_mask_kernel` (S13),
`attn_max_kernel` (S14), `attn_exp_kernel` (S15, S16),
`attn_denom_kernel` (S17), `attn_weights_kernel` (S18) and
`attn_context_kernel` (S19). Everything else is a call into a closed
profile, a copy, or one flushed elementwise seam.

WHY NO FLOAT CROSSES A THREAD BOUNDARY IN THIS FILE
-----------------------------------------------------
Every kernel below owns its output cell entirely.

* The RMSNorm fold (S1) is ONE THREAD PER TOKEN ROW. Contract S1: "one fold
  per row, no block fold".
* The softmax row maximum (S14) is ONE THREAD PER (batch, head, query). Its
  fold shape is FREE under contract 5.1 because `identical_fmax` is exactly
  associative over all of Float32, and this file still walks it ascending,
  because a free shape is not a reason to pick a second one.
* The softmax denominator (S17) is ONE THREAD PER (batch, head, query),
  walking the ABSOLUTE key index ASCENDING from `+0.0` with plain adds.
  Contract 5.3. This is slow and it is the profile.
* The attention-weighted value sum (S19) is ONE THREAD PER
  (batch, head, query, head_dim), walking the ABSOLUTE key index ASCENDING
  from `+0.0` with `fma`. Contract 4 S19, DEVIATION 807.
* Every other seam is one thread per output cell.

There is no shared memory, no warp primitive, no atomic, no cross-block
reduction and no `pinned_reduce` call anywhere in this file. So contract
section 10's clause (b) (the same bits on 8 repeated launches) and clause
(c) (batch composition and sequence length invariance) are properties of the
SHAPE of these kernels rather than of a check that happens to pass -- which
is the gemm lane's argument at its own header, made here for the same
reason. **A block count is a summation order**, and no fold boundary,
accumulator seed or tap index below is a function of `block_dim`,
`block_idx` or the grid.

`[[mojo-buffer-freed-at-last-use]]`: every buffer this file hands to a
kernel is a FIELD of `LlamaDeviceWeights`, `LlamaRopeTable`, `LlamaKVCache`
or `LlamaDeviceStages`, whose lifetime is the struct's. No launcher below
allocates a scratch buffer and returns without waiting.

THE `OP_NT` TRAP, INHERITED FROM THE MAMBA LANE'S SCAR
--------------------------------------------------------
Two files in this repository number the GEMM orientations DIFFERENTLY.
`bench/gemm_shapes.mojo` is `OP_NT = 0, OP_TN = 1, OP_NN = 2`.
`gemm/checks/gemm_oracle.mojo` is `OP_NN = 0, OP_NT = 1, OP_TN = 2`, and
that is the numbering `identical_gemm` reads. Passing the bench table's
codes once produced a whole card of plausible, in-bounds, WRONG products
that no assertion caught, because a weight at `[out, in]` has exactly as
many elements as one at `[in, out]`. **Every call below goes through
`_gemm_op_nt` or `_gemm_op_nn`, which import from `gemm_oracle` and say
which numbering they are in.** Contract section 10's `S05_OP_NUMBERING` is
that mistake, planted.

WHAT IS GATED SO FAR
----------------------
**NOTHING.** Not one line of this file has been compiled. There is no green
shape, no card, no repeat, no vendor. The whole of contract section 10 is
owed: clause (a) at every shape, clause (b)'s eight launches, clause (c)'s
batch and length sweeps, clause (d)'s decode-equals-prefill, clause (e)'s
row-39 audit, and every one of the thirteen sabotage arms in section 10's
table. **A sabotage nobody has fired is not a gate**, so no clause in
contract section 4 is falsified by anything here.

WHAT `NUMERIC_FAST` DOES HERE
-------------------------------
Everything, with the pins compiled away, on the same code and the same
launches. FAST makes no identity claim (contract section 11).

DEVIATIONS THIS FILE OWNS
---------------------------
The transformer lane owns 800-819 (contract section 12.3) and those numbers
are the CONTRACT's, cited here and never renumbered. The decisions this
file's construction forced, which the contract does not decide, are numbered
in the fresh range **1020-1029**, this file's alone.

**DEVIATION 1020 -- the execution plan.** Upstream is torch ops over whole
tensors; there is no upstream kernel decomposition to mirror for the
elementwise seams, and the two reference SHAPES the contract cites
(`mha_gpu_naive`, `softmax_kernel`) are shapes this profile deliberately
does not take, because both fold across threads. The plan here is one thread
per output cell for every seam except S1 (one thread per token row), S14 and
S17 (one thread per (batch, head, query) row) and S19 (one thread per output
cell, whose fold is over the key axis inside that thread). This is an
EXECUTION plan quantity in `archive/plans/IDENTICAL_GEMM_PLAN.md`'s sense: it decides
which thread computes a cell, never the sequence of values accumulated into
it.

**DEVIATION 1021 -- S11's operands are GATHERED per (batch, head), and
`identical_gemm` is called rather than `identical_gemm_into`.** Contract S11
is one `gemm.fp32.v1` `OP_NT` cell per `(batch, head)` with `k = head_dim`,
and the GEMM profile takes only CONTIGUOUS row-major operands (gemm contract
section 2). `q_rope.out` is token-major `[M, n_heads*head_dim]` (contract
section 9), so one head's `[L, head_dim]` block is strided, not contiguous.
Three copy kernels materialize the operands and scatter the result. Copies
are not seams (contract section 4's preamble). The `_into` form with a
caller-owned workspace would save `B * n_heads` allocations and drains, and
it is NOT used, because sizing a workspace for one plan and letting
`choose_gemm_plan` pick another is an out-of-bounds write that a small shape
does not show you -- it cost the gemm lane a run
(`gemm_identical.mojo:1348-1356`). This profile publishes no timing number
(contract section 11), so the cost buys the safety outright.

**DEVIATION 1022 -- the KV cache is REPACKED OUT OF PLACE at the new
length.** Contract section 9 records `kv.k_cache` as `[B, n_kv, S, head_dim]`
where `S` is the length AFTER the append, so the recorded stage is the packed
used region and the row stride is `S`, which grows every call. The append
kernel therefore writes a NEW packed cache from the OLD packed cache and
`k_rope.out`, into a second buffer, and the result is copied into the cache
afterwards. Upstream is out of place too: `past_key_values.update`
(modeling_llama.py:261-262) is `torch.cat`, which allocates. The in-place
roll is the tempting spelling and at `L >= 1` with a growing stride it reads
its own writes; this is the mamba lane's DEVIATION 726 in a different shape.

**DEVIATION 1023 -- `attn.weights` doubles as the S17 sabotage's scratch.**
`S17_DENOM_HALVING_TREE` needs `S` floats of scratch per row to fold a tree
in one thread, and a kernel cannot allocate. `attn.weights`
`[B, n_heads, L, S]` is written by S18, which runs strictly after S17 and
overwrites every cell, so the sabotage borrows it. The IDENTICAL arm passes
the same pointer and never reads it. The alternative was a second
`[B, n_heads, L, S]` buffer allocated in every build to serve one sabotage.

**DEVIATION 1024 -- the three rope stages are RECORDED every call.**
Contract section 9 says `rope.inv_freq`, `rope.cos` and `rope.sin` are
"recorded once per configuration rather than per call", and also lists them
in a thirty-tag card order that `tools/identity_trace_diff.py` aligns on
before comparing any hash. Those two readings conflict for a driver that
calls the block twice. This file resolves it toward the differ: the table is
COMPUTED once per configuration (which is what the cost sentence is about)
and RECORDED on every call, off the same buffers, so every call's card is
exactly section 9's thirty tags in exactly section 9's order. Three extra
hashes per call. If the orchestrator wants the other reading, the three
`record_device` calls in `llama_attention_forward` are the only site.

**DEVIATION 1025 -- `llama_rms_norm_kernel` is a TEMPORARY transcription and
must be DELETED.** Contract section 0 marks RMSNorm REUSED and points at
`mamba/impl/.../modeling_mamba.mojo::mamba_rms_norm_kernel` (:529-575).
That kernel reads `RMS_EPS`, a module constant `1e-5` at
`mamba/checks/mamba_fixture.mojo:44`, directly. Llama's `rms_norm_eps` is
`1e-6` (`configuration_llama.py:73`, verified in the checkout). The two
values give different bits, so the mamba kernel CANNOT be called from here
until eps is an ARGUMENT -- which is contract DEVIATION 801, a cross-lane
edit in the mamba tree that this lane is forbidden to make. The kernel below
is `mamba_rms_norm_kernel` transcribed with `RMS_EPS` replaced by `eps_in`
and NOTHING else changed, so that the block can be built and gated now. **It
is duplicated arithmetic and that is a defect, not a design.** When 801
lands, delete `llama_rms_norm_kernel` and make `llama_rms_norm`'s body a
call to `mamba_rms_norm(ctx, sumsq, out_buf, x, weight, m, d_model, eps)`.
There is exactly one call site of each, so the swap is two lines.

**DEVIATION 1026 -- `ftz` on EVERY buffer load.** Contract section 4
requires every operand loaded from a buffer to pass `ftz` and every seam
result to pass `ftz`. This file spells the flush at every load including the
ones whose stored value was already flushed: it cannot move a bit and the
contract's rule is per LOAD, not per value. The mamba lane's DEVIATION 731
made the same call for the same reason.

**DEVIATION 1027 -- `refuse_nonfinite` reads the device buffers back.**
Contract section 8 refuses a NaN or an infinity BY NAME before any recorded
stage, because NaN payloads are vendor-shaped (IDENTITY_PATHS row 39
measured `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA, `0xffc00000` on AMD
for one IEEE answer) and a certified stage may not contain one. A kernel
cannot raise, so the entry copies each named input to the host and tests it
there, **BY BITS and not by compares**, because Metal FLUSHES COMPARE
OPERANDS (row 49) and `-subnormal < 0.0` is FALSE there while the sign bit
is set. It costs a drain per input. The host twin is the oracle's
`refuse_nonfinite`; it is not imported, because a refusal policy is not a
seam function.

**DEVIATION 1028 -- `pos0` must EQUAL the pre-append cache length, and a
mismatch RAISES.** Every index in contract sections 5.5 and 7 is by ABSOLUTE
position, and this file makes cache slot `j` mean absolute position `j`.
That is only coherent if the tokens of this call start at absolute position
`kv.s`. Passing a `pos0` that disagrees would silently produce a card that
is internally consistent and positionally wrong, which is exactly the class
of defect clause (d) exists to catch and exactly the class a single-path
gate cannot see. So it is a named refusal instead. The batch shares one
`pos0`: v1 has no ragged batch (contract section 3's gate shape is a
rectangular `B x L`).

**DEVIATION 1029 -- S11's decode-safety is an INHERITED reliance on gemm v1,
and it is wider than the contract's sentence.** Contract 7.2 argues S11 is
decode-safe because "GEMM v1's per-cell arithmetic is a pure function of `k`
and the profile" and `k = head_dim` in both paths. True, and not sufficient
as written: this file calls the GEMM at `n = S`, which is `L` in prefill and
`t + 1` in decode, and `choose_gemm_plan(m, n, k)` READS `n`
(`gemm_identical.mojo:1093-1126`; it can return SPLITK, three tile shapes or
FLAT depending on `m * n`). So clause (d) at S11 rests on the gemm profile's
own claim that all five plans are the same bits, which is its section 6.1
and which `check_device_is_launch_invariant` is the gate for. **That is a
dependency on another profile's gate, not on an argument in this file**, and
if the gemm's plan invariance ever regresses, `attn.scores` is where it
surfaces here.

WHAT THIS FILE IS NOT
-----------------------
Not FlashAttention, not SDPA, not an online softmax, not paged attention,
not chunked prefill (contract section 6). Not a model: no embedding, no
`lm_head`, no logit, no token, no sampling (contract section 11). Not
agreement with HuggingFace, PyTorch or MAX: the fold orders,
transcendentals and division below are OURS.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_identical import identical_gemm

# ORIENTATION NUMBERING: these are `gemm_oracle`'s, where
# `OP_NN = 0, OP_NT = 1, OP_TN = 2`. They are NOT `bench/gemm_shapes.mojo`'s
# `OP_NT = 0`. `identical_gemm` reads these. See the header's trap note.
from gemm.checks.gemm_oracle import OP_NN, OP_NT

# CROSS-LANE REUSE, contract section 0's table. `residual_add_kernel` is
# seam S16 of the mamba contract and seams S22 and S23 of this one, the same
# four lines; `pinned_mul` is DEVIATION 720 and this is the IMPORT arm of
# DEVIATION 816 rather than a fourth copy. The cost of the import is that a
# transformer build now compiles the mamba module and its scan file. If that
# coupling ever becomes a problem, both symbols are four lines each and
# localizing them is mechanical -- but do NOT localize them casually, because
# the whole point of contract section 0 is that this block has one residual
# add and one uncontractible multiply, not two.
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    batchinv_norm_chunk,
    pinned_mul,
    residual_add_kernel,
)

from checks.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_mul_add,
    identical_rsqrt,
    identical_silu,
    identical_sin,
    portable_powf,
)


# ===========================================================================
# PROFILE CONSTANTS, contract section 3. Every one is FROZEN: changing a
# value here is a v2 and not an amendment.
#
# Written BY BITS where a decimal literal would be a round-trip risk
# (`[[mojo-string-float-roundtrip]]`: `String(Float32)` does not round trip,
# so this repository's habit is `<decimal>/<hex bits>` and the hex is the
# authority).
# ===========================================================================

comptime MASK_FILL_BITS: UInt32 = 0xFF7FFFFF
"""`torch.finfo(torch.float32).min`, `-3.4028234663852886e+38`, the value
`masking_utils.py:601-603` writes where the mask is False. ADDED, not
selected (contract 4.1(a)), and finite, not `-inf` (contract 4.1(b)): with
`-inf` a fully masked row would give `-inf - (-inf) = NaN` and a computed
NaN's payload is vendor-shaped."""

comptime NEG_INF_BITS: UInt32 = 0xFF800000
"""The `S13_MASK_NEG_INF` sabotage's value, and nothing else."""

comptime MAX_ABS_POSITION = 8192
"""Contract section 3 and DEVIATION 812: the Cody-Waite domain of
`_cephes_sincosf_core`, shared by `portable_sinf` and `portable_cosf`.
STRICTLY LESS THAN. Refused by name below rather than silently wrapped."""


# ===========================================================================
# THE SCORE PLANT'S INJECTION POINTS.
#
# **THESE TWO INTEGERS ARE A CROSS-FILE CONTRACT AND THEY ARE DUPLICATED ON
# PURPOSE.** The host oracle applies its plant at exactly two points
# (`transformer_oracle.mojo`, its `_apply_plant` calls) and names them with
# `PLANT_AT_SCORES = 0` and `PLANT_AT_MASKED = 1` in
# `transformer_fixture.mojo:925` and `:930`. This file may not import from
# `transformer/checks/` (the two halves of this lane were written
# concurrently by different agents and a shared type would have been a guess
# by both), so the VALUES are restated here and the orchestrator owns
# keeping them equal. If they ever disagree, the device plants at a point
# the oracle does not and clause (a) fails at `attn.scores` on every planted
# case while every unplanted case stays green -- which reads exactly like an
# arithmetic bug and is not one.
#
# WHY A PLANT EXISTS AT ALL. Contract 4.1(a) needs a `-0.0` SCORE to
# separate the mask's ADD from a select, and 4.1(b) needs a score of
# magnitude about 1e35 to separate `-FLT_MAX` from `-inf`, and 5.1 needs a
# `-0.0` beside a `+0.0` in one row to separate `identical_fmax` from a
# plain compare. **No assignment of finite weights produces any of them**
# (a `-0.0` score is reachable only through `ftz` of a negative subnormal
# GEMM accumulator, and gemm contract 9.2(a) says products alone can never
# make one), so the contract's own sentence is that the fixture has to plant
# it BY BITS. Three of the thirteen sabotage arms are unfirable without
# this hook.
# ===========================================================================

comptime PLANT_AT_SCORES = 0
"""After S12 and BEFORE `attn.scores` is recorded, so the planted bits are
IN the recorded stage. Mirrors the oracle's first `_apply_plant`."""

comptime PLANT_AT_MASKED = 1
"""After S13 and BEFORE `attn.masked` is recorded. Mirrors the oracle's
second `_apply_plant`."""

comptime PLANT_AT_NONE = -1
"""No plant. The value a driver passes for an ordinary case; it matches
neither point, so both hooks are no-ops."""


# ===========================================================================
# THE SABOTAGE ARMS (a clause nobody can falsify is not a clause).
#
# Contract section 10's table, one comptime arm each, OFF in every build that
# does not name them:
#
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_TRANSFORMER_SABOTAGE_S17_DENOM_HALVING_TREE=1 \
#         -I . transformer/checks/transformer_check.mojo
#
# The names are disjoint from the mamba lane's, so one driver can arm any
# of them without collision. This comment said the mamba lane had ELEVEN
# until `SAB_BATCHINV_NORM_CHUNK_FROM_M` made it twelve; the count is
# dropped rather than corrected, because a number in a comment beside a
# list that is right there is a number that goes stale.
#
# TWO OF THESE PASS CLAUSE (a) BY CONSTRUCTION and the contract says so in
# section 10's last paragraph: `S07_ROPE_RELATIVE_POSITION` and
# `S19_VALUE_SUM_VIA_GEMM` move NOTHING in a fixed-length prefill. A lane
# that fires them without a clause (d) gate will conclude they are inert and
# delete them. They are not inert. They are the two arms that need decode.
# ===========================================================================

comptime SAB_S1_FOLD_DESCENDING = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S1_FOLD_DESCENDING"
]()
"""S1: fold the sum of squares descending. `mean(-1)` (LRN:64) has no
documented fold order, so the order is a PROFILE decision and this is the arm
that falsifies it. First stage it must move: `norm1.sumsq`."""

comptime SAB_S07_ROPE_RELATIVE_POSITION = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S07_ROPE_RELATIVE_POSITION"
]()
"""S7: index the rotary table from the start of THIS launch's slice instead
of by absolute position. Inert in prefill (where `pos0 == 0`), and it must
break clause (d) at every decode step past the first. Contract 7.2."""

comptime SAB_S09_ROPE_HALVES_SWAPPED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S09_ROPE_HALVES_SWAPPED"
]()
"""S9: put `rotate_half`'s negation on the UPPER half instead of the lower,
i.e. `cat(x2, -x1)` where LRH:130-134 is `cat(-x2, x1)`. Same two numbers,
same magnitudes, one sign each in the wrong place. First stage:
`q_rope.out`."""

comptime SAB_S10_ROPE_FUSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S10_ROPE_FUSED"
]()
"""S10: contract `(q*cos) + (rotate_half(q)*sin)` (LAR:158) into one `fma`.
One rounding where the reference has three, and the natural thing for a
kernel to write. Contract DEVIATION 811. First stage: `q_rope.out`."""

comptime SAB_S12_SCALE_INTO_Q = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S12_SCALE_INTO_Q"
]()
"""S12: pre-scale `q` instead of scaling the finished dot (EAF:204 applies
`* scaling` to the matmul's OUTPUT). One rounding per q element rather than
one per score, and a different answer. First stage: `attn.scores`."""

comptime SAB_S13_MASK_NEG_INF = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S13_MASK_NEG_INF"
]()
"""S13: `-inf` instead of `finfo(float32).min`. `s + (-FLT_MAX)` is
`-FLT_MAX` exactly for `|s| < 2^103`, so only a PLANTED extreme score
(magnitude about 1e35) separates the two. Contract 4.1(b). First stage:
`attn.masked`, and only on the planted row."""

comptime SAB_S13_MASK_SELECT = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S13_MASK_SELECT"
]()
"""S13: select instead of add, i.e. skip the `+0.0` at an unmasked cell.
`x + (+0.0) == x` for every finite x, every infinity and every NaN EXCEPT
`x = -0.0`, where it is `+0.0`. Contract 4.1(a), IDENTITY_PATHS row 39. Only
a score PLANTED at `-0.0` by bits separates them. First stage:
`attn.masked`."""

comptime SAB_S14_MAX_PLAIN_COMPARE = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S14_MAX_PLAIN_COMPARE"
]()
"""S14: `a > b ? a : b` instead of `identical_fmax`. `max(+0.0, -0.0)` was
MEASURED on 2026-08-23 as `-0.0` on Apple and `+0.0` on NVIDIA and AMD -- a
live three-vendor split (contract 5.1, IDENTITY_PATHS rows 13 and 39). It
must move `attn.max` on a row carrying a planted `-0.0` beside a `+0.0` and
must NOT move it on an ordinary row, which is what makes it a reach proof
rather than a smoke test."""

comptime SAB_S17_DENOM_HALVING_TREE = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S17_DENOM_HALVING_TREE"
]()
"""S17: a halving tree instead of the serial ascending chain. **The most
likely way to get this seam wrong**, because the deterministic block fold is
sitting in the tree (`core/pinned_reduce.mojo::pinned_block_sum`) and it is
perfectly launch-invariant. It is simply a DIFFERENT SUM. Contract 5.3. It
must move `attn.denom` at any kv length of 3 or more."""

comptime SAB_S18_RECIPROCAL_MUL = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S18_RECIPROCAL_MUL"
]()
"""S18: `e * (1/denom)`, two roundings, instead of `e / denom`, one. MAX's
own `softmax_kernel` multiplies by a reciprocal, and contract 5.4 records
that the REFERENCE's spelling could not be verified because there is no
PyTorch checkout on this machine. First stage: `attn.weights`."""

comptime SAB_S19_VALUE_SUM_VIA_GEMM = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S19_VALUE_SUM_VIA_GEMM"
]()
"""S19: route the value sum through `identical_gemm` (`OP_NN`, `k = S`).
Contract DEVIATION 807 and section 7.2: `P = f(k)` under the gemm's
leaf-and-tree topology, and `k = S` DIFFERS between prefill and decode, so
the masked `+0.0` tail stops being bitwise inert. It must leave clause (a)
GREEN at a fixed length and break clause (d) past the first 128 keys, which
is exactly the failure a single-path gate cannot see."""

comptime SAB_S20_SILU_MUL_SIGMOID = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S20_SILU_MUL_SIGMOID"
]()
"""S20: `z * sigmoid(z)`, two roundings, instead of the reference's single
quotient `z / (1 + exp(-z))`. It is what MAX itself spells
(`max/kernels/src/nn/activations.mojo:249`). DEVIATION 744. First stage:
`silu.out`."""

comptime SAB_S05_OP_NUMBERING = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_S05_OP_NUMBERING"
]()
"""S5: pass `bench/gemm_shapes.mojo`'s `OP_NT = 0` to `identical_gemm`,
which reads `gemm_oracle`'s numbering and sees `OP_NN`. Every buffer is
still exactly the right SIZE, so nothing raises, no bound is exceeded and
every product is wrong. THE TRAP, PLANTED. First stage: `q_proj.out`."""

comptime SAB_BATCHINV_NORM_CHUNK_FROM_M = is_defined[
    "MOJOLEARN_BATCHINV_SABOTAGE_NORM_CHUNK_FROM_M"
]()
"""S1, AIMED AT THE BATCH AXIS, and NOT one of contract section 10's
thirteen.

It belongs to `checks/batch_invariance_check.mojo` and it is the mamba
lane's arm of the same `-D` name, declared a second time here rather than
imported because every other arm in this file is declared here and a
comptime that arrives by import is one more thing to be wrong about. The
`-D` string is the SAME literal, so one define arms both blocks; the two
`*_batchinv_sabotage_armed()` accessors exist so that the check can prove
it did (`[[verify-reach-not-output]]`).

Under it, S1's per-row sum of squares is folded in chunks of
`batchinv_norm_chunk(d_model, m)` -- a width derived from `m = B * L` --
instead of one serial ascending chain. That is gemm contract section 6.1's
forbidden `L = f(k, m)` moved one layer up. It is BITWISE INERT at m < 64,
so this lane's own clause (c) (B in {1, 2, 3}) cannot see it, and it bites
at B = 17 and B = 64.

**`transformer_check.mojo` ARMED WITH THIS WILL RAISE FROM
`arm_expectation` BY NAME**, because that table holds the thirteen arms of
contract section 10 and this is not one of them. That refusal is correct
and must not be "fixed" by adding a row: a gate that does not own an arm
should refuse to score it. Run `pixi run check-batch-invariance` instead."""


def llama_batchinv_sabotage_armed() -> Bool:
    """Whether THIS FILE compiled with the batch-invariance sabotage. Paired
    with `mamba_batchinv_sabotage_armed()`; a disagreement between the two
    means one of the two `-D` string literals is misspelled and half the run
    is silently clean."""
    comptime if SAB_BATCHINV_NORM_CHUNK_FROM_M:
        return True
    return False


comptime BLOCK_ANY_SABOTAGE = (
    SAB_S1_FOLD_DESCENDING
    or SAB_S07_ROPE_RELATIVE_POSITION
    or SAB_S09_ROPE_HALVES_SWAPPED
    or SAB_S10_ROPE_FUSED
    or SAB_S12_SCALE_INTO_Q
    or SAB_S13_MASK_NEG_INF
    or SAB_S13_MASK_SELECT
    or SAB_S14_MAX_PLAIN_COMPARE
    or SAB_S17_DENOM_HALVING_TREE
    or SAB_S18_RECIPROCAL_MUL
    or SAB_S19_VALUE_SUM_VIA_GEMM
    or SAB_S20_SILU_MUL_SIGMOID
    or SAB_S05_OP_NUMBERING
    or SAB_BATCHINV_NORM_CHUNK_FROM_M
)


def llama_block_sabotage_name() -> String:
    """The armed sabotage, for a driver that must refuse to certify a
    sabotaged build -- and, more importantly, must refuse to report a
    sabotaged build as CLEAN when the `-D` was misspelled and silently
    ignored (`tools/gemm_ladder.sh:71`'s scar). Exactly one arm is expected
    to be on; arming two is the driver's error to catch, not this
    function's."""
    comptime if SAB_S1_FOLD_DESCENDING:
        return String("S1_FOLD_DESCENDING")
    comptime if SAB_S07_ROPE_RELATIVE_POSITION:
        return String("S07_ROPE_RELATIVE_POSITION")
    comptime if SAB_S09_ROPE_HALVES_SWAPPED:
        return String("S09_ROPE_HALVES_SWAPPED")
    comptime if SAB_S10_ROPE_FUSED:
        return String("S10_ROPE_FUSED")
    comptime if SAB_S12_SCALE_INTO_Q:
        return String("S12_SCALE_INTO_Q")
    comptime if SAB_S13_MASK_NEG_INF:
        return String("S13_MASK_NEG_INF")
    comptime if SAB_S13_MASK_SELECT:
        return String("S13_MASK_SELECT")
    comptime if SAB_S14_MAX_PLAIN_COMPARE:
        return String("S14_MAX_PLAIN_COMPARE")
    comptime if SAB_S17_DENOM_HALVING_TREE:
        return String("S17_DENOM_HALVING_TREE")
    comptime if SAB_S18_RECIPROCAL_MUL:
        return String("S18_RECIPROCAL_MUL")
    comptime if SAB_S19_VALUE_SUM_VIA_GEMM:
        return String("S19_VALUE_SUM_VIA_GEMM")
    comptime if SAB_S20_SILU_MUL_SIGMOID:
        return String("S20_SILU_MUL_SIGMOID")
    comptime if SAB_S05_OP_NUMBERING:
        return String("S05_OP_NUMBERING")
    comptime if SAB_BATCHINV_NORM_CHUNK_FROM_M:
        return String("BATCHINV_NORM_CHUNK_FROM_M")
    return String("none")


# ===========================================================================
# LAUNCH GEOMETRY. An EXECUTION plan quantity (DEVIATION 1020): it decides
# which thread owns a cell and nothing else. No kernel below reads
# `block_dim` or `block_idx` in any expression that reaches a fold boundary,
# an accumulator seed, a key index or a table index.
# ===========================================================================

comptime LLAMA_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + LLAMA_TPB - 1) // LLAMA_TPB
    if g < 1:
        return 1
    return g


def _gemm_op_nt() -> Int:
    """`OP_NT` in `gemm_oracle`'s numbering (`OP_NN = 0, OP_NT = 1,
    OP_TN = 2`), which is the numbering `identical_gemm` reads. NOT
    `bench/gemm_shapes.mojo`'s `OP_NT = 0`. See the header's trap note."""
    comptime if SAB_S05_OP_NUMBERING:
        return OP_NN
    return OP_NT


def _gemm_op_nn() -> Int:
    """`OP_NN` in the same numbering. Reached ONLY by the
    `S19_VALUE_SUM_VIA_GEMM` sabotage arm; the profile has no `OP_NN`
    call."""
    return OP_NN


# ===========================================================================
# DEVICE I/O PLUMBING. Not seams, not arithmetic -- these three are
# `modeling_mamba.mojo`'s `mamba_upload`, `mamba_download` and `mamba_zeros`
# transcribed under lane-neutral names, because a llama file calling
# `mamba_upload` reads as a mistake. If a shared `core/` home for them ever
# appears, both lanes should move.
# ===========================================================================


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """One host list onto the device, contiguous, in index order."""
    var n = len(values)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """The first `n` elements of a device buffer, as a host list. The gates
    read stages with this and DEVIATION 1027's refusal reads inputs with
    it."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def _zeros(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    dev.enqueue_fill(Float32(0.0))
    ctx.synchronize()
    return dev^


def _plant_bits(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    count: Int,
    at_here: Int,
    plant_at: Int,
    plant_idx: List[Int],
    plant_bits: List[UInt32],
) raises:
    """The device twin of the oracle's `_apply_plant`. A NO-OP unless the
    plant names THIS point and carries indices.

    Done as a HOST ROUND TRIP rather than a kernel, and the reason is
    honesty about what this lane can verify. A plant kernel would need an
    `Int32` index buffer and a `UInt32` bit buffer, which are two device
    buffer element types nothing else in this file uses and which this lane
    cannot compile to check. A download, a store by index and an upload use
    only the three forms the mamba lane's file already exercises. It costs
    two drains and a full buffer copy per planted stage, on the handful of
    fixture cases that plant anything at all, in a profile that publishes no
    timing number.

    **AN INDEX THAT MISSES IS REFUSED.** A plant that lands outside the used
    region is worse than no plant: the gate goes green and proves nothing.
    The bound is `count`, the USED cell count, not `len(buf)`, because the
    buffers here are allocated at `s_max` and used at `s` (see
    `LlamaDeviceStages`) and a plant into the unused tail would be invisible
    to the recorded stage. The oracle's own refusal says the same thing
    about its lists.
    """
    if plant_at != at_here:
        return
    if len(plant_idx) == 0:
        return
    if len(plant_idx) != len(plant_bits):
        raise Error(
            String("llama: the score plant has ")
            + String(len(plant_idx))
            + " indices and "
            + String(len(plant_bits))
            + " bit patterns"
        )
    var n = len(buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    for i in range(len(plant_idx)):
        var j = plant_idx[i]
        if j < 0 or j >= count:
            raise Error(
                String("llama: score plant index ")
                + String(j)
                + " is outside the "
                + String(count)
                + " used cells REFUSED (a plant that misses is worse than"
                + " no plant: the gate goes green and proves nothing)"
            )
        host.unsafe_ptr().unsafe_store(
            j, bitcast[DType.float32](plant_bits[i])
        )
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^


# ===========================================================================
# SHAPE, PARAMETERS, STATE AND STAGES
# ===========================================================================


@fieldwise_init
struct LlamaDims(Copyable, Movable):
    """One block's shape. Contract section 3's "free, subject to the
    divisibility rules" row: NONE of these five is frozen, because a model
    shape is not an arithmetic. `head_dim` is `LlamaConfig.head_dim`, which
    `__post_init__` defaults to `hidden_size // num_attention_heads`
    (configuration_llama.py:85-89) but which a config may set independently,
    so it is carried rather than derived."""

    var d_model: Int
    var n_heads: Int
    var n_kv: Int
    var head_dim: Int
    var intermediate: Int

    def n_rep(self) -> Int:
        """`num_key_value_groups`, `LlamaAttention.__init__` :224. Head `h`
        reads kv head `h // n_rep`, which is `repeat_kv`'s
        `expand`-then-`reshape` (:186-188) as an INDEX MAP. Contract
        DEVIATION 813 declares that a COPY, so it is never materialized and
        it is not a seam."""
        return self.n_heads // self.n_kv

    def q_width(self) -> Int:
        return self.n_heads * self.head_dim

    def kv_width(self) -> Int:
        return self.n_kv * self.head_dim

    def half(self) -> Int:
        """`head_dim // 2`, the rotary table's width. The table is
        `cat(freqs, freqs)` (LRE:123), so `cos[j]` and `cos[j + half]` are
        the SAME value (contract S9) and only half is stored. Contract
        section 9's `rope.cos [P_max, head_dim/2]` is that decision on the
        card."""
        return self.head_dim // 2

    def validate(self) raises:
        """Contract section 3's divisibility rules, REFUSED BY NAME rather
        than silently truncated."""
        if self.d_model <= 0:
            raise Error("llama: d_model must be positive")
        if self.intermediate <= 0:
            raise Error("llama: intermediate_size must be positive")
        if self.n_heads <= 0 or self.n_kv <= 0 or self.head_dim <= 0:
            raise Error("llama: n_heads, n_kv_heads and head_dim must be > 0")
        if self.d_model != self.n_heads * self.head_dim:
            raise Error(
                String("llama: d_model must equal n_heads*head_dim, got ")
                + String(self.d_model)
                + " vs "
                + String(self.n_heads)
                + "*"
                + String(self.head_dim)
            )
        if self.n_heads % self.n_kv != 0:
            raise Error(
                String("llama: n_heads % n_kv_heads must be 0, got ")
                + String(self.n_heads)
                + " % "
                + String(self.n_kv)
            )
        if self.head_dim % 2 != 0:
            raise Error(
                String("llama: head_dim must be even (RoPE pairs halves),")
                + " got "
                + String(self.head_dim)
            )


def _expect_len(name: String, got: Int, want: Int) raises:
    """A weight of the wrong length is the defect class that produces a card
    of plausible, in-bounds, WRONG numbers -- the same class as the `OP_NT`
    trap. Refused by name at construction, where the message can say which
    weight."""
    if got != want:
        raise Error(
            String("llama: ")
            + name
            + " has "
            + String(got)
            + " floats, expected "
            + String(want)
        )


struct LlamaDeviceWeights(Movable):
    """One block's parameters on the device, in the upstream shapes. Every
    `nn.Linear` weight is `[out_features, in_features]`, which is torch's
    layout, and every one is read by the GEMM as an `OP_NT` right operand.
    Row-major, contiguous, no padding.

    **NO BIAS ANYWHERE.** `attention_bias` and `mlp_bias` are both `False`
    at the config defaults (configuration_llama.py:81, :83) and contract
    section 2 REFUSES a nonzero bias rather than specifying where it would
    round. There is no bias field to pass one through.

    `eps` is `LlamaConfig.rms_norm_eps`, `1e-6` / `0x358637BD`, FROZEN by
    contract section 3. It lives here rather than in `LlamaDims` because it
    is an arithmetic constant, not a shape. It is an ARGUMENT and not a
    module constant precisely because the mamba lane's being a module
    constant is what blocks DEVIATION 801.
    """

    var dims: LlamaDims
    var eps: Float32
    var norm1_w: DeviceBuffer[DType.float32]  # [d_model]
    var norm2_w: DeviceBuffer[DType.float32]  # [d_model]
    var w_q: DeviceBuffer[DType.float32]  # [n_heads*head_dim, d_model]
    var w_k: DeviceBuffer[DType.float32]  # [n_kv*head_dim, d_model]
    var w_v: DeviceBuffer[DType.float32]  # [n_kv*head_dim, d_model]
    var w_o: DeviceBuffer[DType.float32]  # [d_model, n_heads*head_dim]
    var w_gate: DeviceBuffer[DType.float32]  # [intermediate, d_model]
    var w_up: DeviceBuffer[DType.float32]  # [intermediate, d_model]
    var w_down: DeviceBuffer[DType.float32]  # [d_model, intermediate]

    def __init__(
        out self,
        ctx: DeviceContext,
        dims: LlamaDims,
        eps: Float32,
        norm1_w: List[Float32],
        norm2_w: List[Float32],
        w_q: List[Float32],
        w_k: List[Float32],
        w_v: List[Float32],
        w_o: List[Float32],
        w_gate: List[Float32],
        w_up: List[Float32],
        w_down: List[Float32],
    ) raises:
        """Nine host lists and two scalars. **No fixture type crosses this
        boundary on purpose**: the oracle half of this lane is being written
        concurrently and by a different agent, so nothing in
        `transformer/checks/` is imported anywhere in this file."""
        dims.validate()
        self.dims = dims.copy()
        self.eps = eps
        var dm = dims.d_model
        var qw = dims.q_width()
        var kw = dims.kv_width()
        var it = dims.intermediate
        _expect_len("norm1.weight", len(norm1_w), dm)
        _expect_len("norm2.weight", len(norm2_w), dm)
        _expect_len("q_proj.weight", len(w_q), qw * dm)
        _expect_len("k_proj.weight", len(w_k), kw * dm)
        _expect_len("v_proj.weight", len(w_v), kw * dm)
        _expect_len("o_proj.weight", len(w_o), dm * qw)
        _expect_len("gate_proj.weight", len(w_gate), it * dm)
        _expect_len("up_proj.weight", len(w_up), it * dm)
        _expect_len("down_proj.weight", len(w_down), dm * it)
        # DEVIATION 1875 -- THE WEIGHTS ARE REFUSED HERE, ON THE HOST, ONCE.
        #
        # `llama_refuse_bad_inputs` walks all thirteen named inputs and it
        # DOWNLOADS every weight tensor to do it. That is correct and it was
        # being paid on EVERY forward call. Measured on an H100 2026-08-25 at
        # the Llama-3-8B shapes: 218 million floats, about 872 MB per call,
        # 825 to 967 ms -- and the whole block measured 936 to 1010 ms, flat
        # in token count from t1 to t512. A forward pass whose cost does not
        # move with its input is not computing the input.
        #
        # THE WEIGHTS DO NOT CHANGE BETWEEN CALLS, and here they are still on
        # the HOST, so this check costs no device traffic at all. It also
        # fires EARLIER than the old one and under the SAME upstream names,
        # which is what the oracle comparison needs.
        #
        # WHAT THIS DOES NOT COVER, said plainly: a caller that writes into a
        # weight BUFFER after construction. `llama_refuse_bad_inputs` is
        # unchanged and still walks all thirteen names in its original order
        # for anyone who needs that, and it is what
        # `transformer_check.mojo`'s refusal audit calls directly, so the
        # audit's coverage of every name is untouched.
        _refuse_nonfinite_named("input_layernorm.weight", norm1_w)
        _refuse_nonfinite_named("post_attention_layernorm.weight", norm2_w)
        _refuse_nonfinite_named("q_proj.weight", w_q)
        _refuse_nonfinite_named("k_proj.weight", w_k)
        _refuse_nonfinite_named("v_proj.weight", w_v)
        _refuse_nonfinite_named("o_proj.weight", w_o)
        _refuse_nonfinite_named("gate_proj.weight", w_gate)
        _refuse_nonfinite_named("up_proj.weight", w_up)
        _refuse_nonfinite_named("down_proj.weight", w_down)
        self.norm1_w = _upload(ctx, norm1_w)
        self.norm2_w = _upload(ctx, norm2_w)
        self.w_q = _upload(ctx, w_q)
        self.w_k = _upload(ctx, w_k)
        self.w_v = _upload(ctx, w_v)
        self.w_o = _upload(ctx, w_o)
        self.w_gate = _upload(ctx, w_gate)
        self.w_up = _upload(ctx, w_up)
        self.w_down = _upload(ctx, w_down)


struct LlamaKVCache(Movable):
    """The recurrent state between calls (contract section 7.2):
    `k_cache` and `v_cache`, each `[B, n_kv, S, head_dim]`, appended to at
    modeling_llama.py:261-262.

    **CACHE SLOT `j` MEANS ABSOLUTE POSITION `j`.** That identification is
    what makes contract section 5.5 ("every one of S11 through S19 indexes
    the key axis BY ABSOLUTE POSITION, not by an offset into whatever slice
    this launch happens to hold") mechanical here rather than a discipline.
    DEVIATION 1028 refuses a `pos0` that would break it.

    `s` is the USED length and it grows by `L` per call; `s_max` is the
    allocation. The buffers are PACKED at stride `s`, not at stride `s_max`,
    because contract section 9 records the stage as `[B, n_kv, S, head_dim]`
    and a hash of the first `B*n_kv*S*head_dim` elements has to BE that
    array. DEVIATION 1022 is why the repack is out of place.

    Prefill is a fresh `LlamaKVCache` with `s == 0`; the decode step is the
    same block function at `l == 1` carrying this. ONE spelling for both.
    """

    var b: Int
    var n_kv: Int
    var head_dim: Int
    var s_max: Int
    var s: Int
    var k: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, b: Int, dims: LlamaDims, s_max: Int
    ) raises:
        if b <= 0:
            raise Error("llama: KV cache needs B > 0")
        if s_max <= 0:
            raise Error("llama: KV cache needs s_max > 0")
        if s_max > MAX_ABS_POSITION:
            raise Error(
                String("llama: s_max ")
                + String(s_max)
                + " exceeds the absolute-position ceiling "
                + String(MAX_ABS_POSITION)
                + " (DEVIATION 812: the Cody-Waite domain of"
                + " _cephes_sincosf_core, shared by portable_sinf and"
                + " portable_cosf)"
            )
        self.b = b
        self.n_kv = dims.n_kv
        self.head_dim = dims.head_dim
        self.s_max = s_max
        self.s = 0
        self.k = _zeros(ctx, b * dims.n_kv * s_max * dims.head_dim)
        self.v = _zeros(ctx, b * dims.n_kv * s_max * dims.head_dim)


struct LlamaDeviceStages(Movable):
    """Every recorded stage of one block call, contract section 9, in card
    order, in the layouts that section names. `M = B * L` token-major rows.

    The attention buffers are allocated at `s_max` and USED PACKED at the
    call's `s`, so `record_device(..., count = b*n_heads*l*s)` hashes
    exactly the `[B, n_heads, L, S]` array the card lists and never folds
    uninitialized tail memory into a stage. `core/identity_trace.mojo`'s own
    docstring is emphatic about this: "when a buffer is used short, PASS THE
    LENGTH".

    Four fields carry NO TAG. `qbh`, `kbh` and `sbh` are DEVIATION 1021's
    per-(batch, head) gather scratch for the GEMM; they hold copies and no
    seam ever writes them. `k_cache`/`v_cache` here are DEVIATION 1022's
    out-of-place repack targets and they DO carry the `kv.k_cache` and
    `kv.v_cache` tags -- the stage is the cache as COMPUTED, recorded before
    it is copied into `LlamaKVCache`.
    """

    var b: Int
    var l: Int
    var s_max: Int
    var dims: LlamaDims
    var norm1_sumsq: DeviceBuffer[DType.float32]  # [M]
    var norm1_out: DeviceBuffer[DType.float32]  # [M, d_model]
    var q_proj: DeviceBuffer[DType.float32]  # [M, n_heads*head_dim]
    var k_proj: DeviceBuffer[DType.float32]  # [M, n_kv*head_dim]
    var v_proj: DeviceBuffer[DType.float32]  # [M, n_kv*head_dim]
    var q_rope: DeviceBuffer[DType.float32]  # [M, n_heads*head_dim]
    var k_rope: DeviceBuffer[DType.float32]  # [M, n_kv*head_dim]
    var k_cache: DeviceBuffer[DType.float32]  # [B, n_kv, S, head_dim]
    var v_cache: DeviceBuffer[DType.float32]  # [B, n_kv, S, head_dim]
    var scores: DeviceBuffer[DType.float32]  # [B, n_heads, L, S]
    var masked: DeviceBuffer[DType.float32]  # [B, n_heads, L, S]
    var amax: DeviceBuffer[DType.float32]  # [B, n_heads, L]
    var aexp: DeviceBuffer[DType.float32]  # [B, n_heads, L, S]
    var denom: DeviceBuffer[DType.float32]  # [B, n_heads, L]
    var weights: DeviceBuffer[DType.float32]  # [B, n_heads, L, S]
    var ctxv: DeviceBuffer[DType.float32]  # [M, n_heads*head_dim]
    var o_proj: DeviceBuffer[DType.float32]  # [M, d_model]
    var residual1: DeviceBuffer[DType.float32]  # [M, d_model]
    var norm2_sumsq: DeviceBuffer[DType.float32]  # [M]
    var norm2_out: DeviceBuffer[DType.float32]  # [M, d_model]
    var gate_proj: DeviceBuffer[DType.float32]  # [M, intermediate]
    var up_proj: DeviceBuffer[DType.float32]  # [M, intermediate]
    var silu_out: DeviceBuffer[DType.float32]  # [M, intermediate]
    var gated: DeviceBuffer[DType.float32]  # [M, intermediate]
    var down_proj: DeviceBuffer[DType.float32]  # [M, d_model]
    var residual2: DeviceBuffer[DType.float32]  # [M, d_model]
    var qbh: DeviceBuffer[DType.float32]  # [L, head_dim]     (no tag)
    var kbh: DeviceBuffer[DType.float32]  # [s_max, head_dim] (no tag)
    var sbh: DeviceBuffer[DType.float32]  # [L, s_max]        (no tag)

    def __init__(
        out self,
        ctx: DeviceContext,
        b: Int,
        l: Int,
        s_max: Int,
        dims: LlamaDims,
    ) raises:
        dims.validate()
        if b <= 0 or l <= 0:
            raise Error("llama: stages need B > 0 and L > 0")
        if s_max < l:
            raise Error(
                String("llama: s_max ")
                + String(s_max)
                + " is smaller than L "
                + String(l)
                + "; the cache must hold at least one call's tokens"
            )
        self.b = b
        self.l = l
        self.s_max = s_max
        self.dims = dims.copy()
        var m = b * l
        var dm = dims.d_model
        var qw = dims.q_width()
        var kw = dims.kv_width()
        var hd = dims.head_dim
        var it = dims.intermediate
        var nh = dims.n_heads
        var nkv = dims.n_kv
        self.norm1_sumsq = _zeros(ctx, m)
        self.norm1_out = _zeros(ctx, m * dm)
        self.q_proj = _zeros(ctx, m * qw)
        self.k_proj = _zeros(ctx, m * kw)
        self.v_proj = _zeros(ctx, m * kw)
        self.q_rope = _zeros(ctx, m * qw)
        self.k_rope = _zeros(ctx, m * kw)
        self.k_cache = _zeros(ctx, b * nkv * s_max * hd)
        self.v_cache = _zeros(ctx, b * nkv * s_max * hd)
        self.scores = _zeros(ctx, b * nh * l * s_max)
        self.masked = _zeros(ctx, b * nh * l * s_max)
        self.amax = _zeros(ctx, b * nh * l)
        self.aexp = _zeros(ctx, b * nh * l * s_max)
        self.denom = _zeros(ctx, b * nh * l)
        self.weights = _zeros(ctx, b * nh * l * s_max)
        self.ctxv = _zeros(ctx, m * qw)
        self.o_proj = _zeros(ctx, m * dm)
        self.residual1 = _zeros(ctx, m * dm)
        self.norm2_sumsq = _zeros(ctx, m)
        self.norm2_out = _zeros(ctx, m * dm)
        self.gate_proj = _zeros(ctx, m * it)
        self.up_proj = _zeros(ctx, m * it)
        self.silu_out = _zeros(ctx, m * it)
        self.gated = _zeros(ctx, m * it)
        self.down_proj = _zeros(ctx, m * dm)
        self.residual2 = _zeros(ctx, m * dm)
        self.qbh = _zeros(ctx, l * hd)
        self.kbh = _zeros(ctx, s_max * hd)
        self.sbh = _zeros(ctx, l * s_max)


# ===========================================================================
# `LlamaRMSNorm.forward` (:62-67). Seams S1-S4.
#
#     variance = hidden_states.pow(2).mean(-1, keepdim=True)          :64
#     hidden_states = hidden_states * torch.rsqrt(variance + eps)     :65
#     return self.weight * hidden_states.to(input_dtype)              :67
#
# eps is `config.rms_norm_eps`, 1e-6, bits 0x358637BD -- contract section 3.
#
# **DEVIATION 1025: THIS KERNEL IS A DUPLICATE AND MUST BE DELETED.** It is
# `mamba/impl/transformers/models/mamba/modeling_mamba.mojo`'s
# `mamba_rms_norm_kernel` (:529-575) with the module constant `RMS_EPS`
# replaced by the argument `eps_in` and NOTHING else changed. The mamba
# kernel is the REUSED form contract section 0 names; it cannot be called
# from here while its epsilon is a `comptime` at
# `mamba/checks/mamba_fixture.mojo:44` set to 1e-5. Lifting it is contract
# DEVIATION 801, a cross-lane edit this lane is forbidden to make. The
# CROSS-LANE REQUEST is in this lane's report, verbatim, both spellings.
# ===========================================================================


def llama_rms_norm_kernel(
    sumsq: MutPointer[Float32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
    eps_in: Float32,
):
    """One thread per TOKEN ROW. Contract S1: "serial ascending j from +0.0,
    one fold per row, no block fold" -- so the row's fold never leaves this
    thread's registers and no launch geometry can reorder it."""
    var m = Int(m_in)
    var dm = Int(dm_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return

    # S1, FUSED: `acc = ftz(fma(x_j, x_j, acc))`, ascending from +0.0. The
    # +0.0 seed is contract section 8's clause -- an all-zero row sums to
    # +0.0 on every vendor because IEEE says (+0) + (-0) = +0.
    var acc = Float32(0.0)
    comptime if SAB_S1_FOLD_DESCENDING:
        for jj in range(dm):
            var jd = dm - 1 - jj
            var xd = ftz(x.unsafe_load(t * dm + jd))
            acc = ftz(identical_mul_add(xd, xd, acc))
    else:
        for j in range(dm):
            var xj = ftz(x.unsafe_load(t * dm + j))
            acc = ftz(identical_mul_add(xj, xj, acc))

    comptime if SAB_BATCHINV_NORM_CHUNK_FROM_M:
        # SABOTAGE: a chunked fold whose CHUNK WIDTH is derived from
        # `m = B * L`, so the row's summation order is a function of how
        # many other tokens shared the launch. Additive rather than a third
        # branch above, so S1's two profile spellings stay one if/else.
        # `batchinv_norm_chunk` is the mamba module's, imported, so the two
        # blocks cannot hold two opinions about the chunk width.
        var chunk = batchinv_norm_chunk(dm, m)
        var acc_b = Float32(0.0)
        var j0 = 0
        while j0 < dm:
            var j1 = j0 + chunk
            if j1 > dm:
                j1 = dm
            var part = Float32(0.0)
            for j in range(j0, j1):
                var xb = ftz(x.unsafe_load(t * dm + j))
                part = ftz(identical_mul_add(xb, xb, part))
            acc_b = ftz(acc_b + part)
            j0 = j1
        acc = acc_b

    sumsq.unsafe_store(t, acc)

    # S2: the mean through `identical_div` (row 49) and the reciprocal
    # square root through `identical_rsqrt`, which is the reference's
    # `1 / sqrt` (DEVIATION 741) and NEVER the hardware rsqrt intrinsic
    # (DEVIATION 746 RECORDED Metal's intrinsic as correctly rounded where
    # the reference's spelling is not -- being righter than the reference is
    # not the goal).
    var mean = ftz(identical_div(acc, Float32(dm)))
    var rstd = ftz(identical_rsqrt(ftz(mean + eps_in)))

    # S3 and S4, both PRODUCT: `hidden * rstd` then `weight * hidden`, each
    # its own rounding, neither contractible into a neighboring add.
    for j in range(dm):
        var inner = ftz(pinned_mul(ftz(x.unsafe_load(t * dm + j)), rstd))
        out_buf.unsafe_store(
            t * dm + j, ftz(pinned_mul(ftz(weight.unsafe_load(j)), inner))
        )


def llama_rms_norm(
    ctx: DeviceContext,
    mut sumsq: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    m: Int,
    d_model: Int,
    eps: Float32,
) raises:
    """`LlamaRMSNorm.forward(hidden_states)` (:62-67) over `M = B * L` token
    rows. ASYNCHRONOUS: the caller synchronizes.

    **AFTER DEVIATION 801 LANDS THIS BODY BECOMES ONE CALL** to
    `mamba_rms_norm(ctx, sumsq, out_buf, x, weight, m, d_model, eps)` and
    `llama_rms_norm_kernel` is deleted. This launcher exists so that the
    swap touches one function and no call site.
    """
    ctx.enqueue_function[llama_rms_norm_kernel](
        sumsq.unsafe_ptr(),
        out_buf.unsafe_ptr(),
        x.unsafe_ptr(),
        weight.unsafe_ptr(),
        Int32(m),
        Int32(d_model),
        eps,
        grid_dim=(_grid(m), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )


# ===========================================================================
# `LlamaRotaryEmbedding` (:73-127). Seams S6, S7, S8.
#
#     inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))    :108
#     freqs = (inv_freq_expanded @ position_ids_expanded).transpose   :120
#     emb = torch.cat((freqs, freqs), dim=-1)                         :123
#     cos = emb.cos() * self.attention_scaling                        :124
#     sin = emb.sin() * self.attention_scaling                        :125
#
# `attention_scaling` is EXACTLY 1.0 for `rope_type = "default"`
# (`compute_default_rope_parameters` :105 `attention_factor = 1.0`), so the
# multiply at :124-125 is bit-inert on every finite input and both zero signs
# and IS NOT SPELLED. Contract section 3, DEVIATION 810.
#
# `cat(freqs, freqs)` is a COPY and is not materialized: only `[p_max, half]`
# is stored and column `j` of a head reads table column `j % half`.
# ===========================================================================


def llama_rope_inv_freq_host(
    theta: Float32, head_dim: Int
) raises -> List[Float32]:
    """S6, on the HOST, once per `(theta, head_dim)`. Contract section 4's
    S6 row.

        e   = ftz(identical_div(Float32(2*i), Float32(head_dim)))
        inv = ftz(identical_div(1.0, portable_powf(theta, e)))

    **`portable_powf` and NOT the host libm `pow`.** A constant computed by
    the host's libm is IDENTITY_PATHS row 18 applied to a table: cross-vendor
    is cross-HOST for a constant, and three build machines with three libms
    would seed three different rotary tables that no device stage could
    explain. Contract DEVIATION 809.

    `portable_powf` is `portable_expf(p * portable_logf(x))`. The syntactic
    `*` in it is the numerics lane's spelling, not this lane's; it feeds a
    function ARGUMENT and has no neighboring add to be contracted into, so
    it is not an FMA hazard here. That is an argument, not a measurement.

    NOT FLUSHED-AND-DONE: the result list is uploaded verbatim and the
    device flushes on load like every other operand (DEVIATION 1026).
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise Error("llama: rope needs an even positive head_dim")
    var half = head_dim // 2
    var out = List[Float32]()
    for i in range(half):
        var e = ftz(identical_div(Float32(2 * i), Float32(head_dim)))
        var p = portable_powf(theta, e)
        out.append(ftz(identical_div(Float32(1.0), p)))
    return out^


def llama_rope_table_kernel(
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    inv_freq: MutPointer[Float32, MutAnyOrigin],
    p_max_in: Int32,
    half_in: Int32,
):
    """S7 and S8. One thread per `(position, frequency)` cell.

    S7, PRODUCT: `pinned_mul(Float32(abs_pos), inv_freq[i])`. Upstream this
    is a `k = 1` matmul in FP32 with autocast explicitly disabled
    (:118-122). At `k = 1` the gemm leaf is `ftz(fma(a, b, +0.0))`, which is
    bit-equal to this product because both operands are NON-NEGATIVE (the
    position is an absolute index and `inv_freq` is a reciprocal of a
    positive power), so the two spellings coincide and contract S7 pins the
    cheaper one.

    S8: `identical_cos` (DEVIATION 258) and `identical_sin` (DEVIATION 820).
    They share `_cephes_sincosf_core`, so BOTH HALVES OF ONE ROTATION COME
    FROM ONE REDUCTION rather than from two functions wearing one name.

    THE TRIG PAIR'S FLUSH ASYMMETRY IS NOT A BUG HERE. `portable_sinf`
    flushes its input and `portable_cosf` does not (DEVIATION 820), because
    near zero `sin` returns its argument and a subnormal survives to the
    output while `cos` returns 1.0 for every subnormal on every column. The
    angle is flushed once, before both calls, so the asymmetry cannot be
    reached from this site at all.

    `portable_sinf(-0.0)` is `-0.0`, a knowing departure from Cephes. It is
    UNREACHABLE here: the angle is `position * inv_freq` with
    `position >= 0` and `inv_freq > 0`. Contract section 8's last clause
    exists so nobody plants a fixture expecting Cephes's answer.
    """
    var p_max = Int(p_max_in)
    var half = Int(half_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= p_max * half:
        return
    var pos = i // half
    var f = i - pos * half
    var angle = ftz(
        pinned_mul(Float32(pos), ftz(inv_freq.unsafe_load(f)))
    )
    cos_tab.unsafe_store(i, ftz(identical_cos(angle)))
    sin_tab.unsafe_store(i, ftz(identical_sin(angle)))


struct LlamaRopeTable(Movable):
    """`LlamaRotaryEmbedding` (:73-127), computed ONCE per
    `(theta, head_dim, p_max)` and indexed by ABSOLUTE POSITION. Contract
    DEVIATION 810, and MAX's own `rope.mojo` does the same thing (it takes
    `freqs_cis` as a tensor and computes no angle in any kernel).

    `inv_freq` `[half]` is S6 and it is computed on the HOST, once. `cos`
    and `sin` are `[p_max, half]` and they are S7 and S8, computed on the
    DEVICE, because the transcendentals are where the cross-vendor claim
    lives and a host-computed table would move that claim onto the host's
    libm -- which is IDENTITY_PATHS row 18's hazard applied to a constant.

    S6 stays on the host anyway, and the asymmetry is deliberate: it is one
    `portable_powf` per table column, its inputs are integers, and contract
    section 4's S6 row says HOST in capitals.

    DEFINED HERE, AFTER ITS TWO FUNCTIONS, ON PURPOSE. This file avoids
    forward references at module scope throughout, because it was written
    by a lane that cannot compile and a resolution order it guessed wrong
    would be a whole build cycle spent on nothing.
    """

    var half: Int
    var p_max: Int
    var theta: Float32
    var inv_freq: DeviceBuffer[DType.float32]  # [half]
    var cos: DeviceBuffer[DType.float32]  # [p_max, half]
    var sin: DeviceBuffer[DType.float32]  # [p_max, half]

    def __init__(
        out self,
        ctx: DeviceContext,
        dims: LlamaDims,
        theta: Float32,
        p_max: Int,
    ) raises:
        dims.validate()
        if p_max <= 0:
            raise Error("llama: the rotary table needs p_max > 0")
        if p_max > MAX_ABS_POSITION:
            raise Error(
                String("llama: p_max ")
                + String(p_max)
                + " exceeds the absolute-position ceiling "
                + String(MAX_ABS_POSITION)
                + " (contract section 3, DEVIATION 812)"
            )
        self.half = dims.half()
        self.p_max = p_max
        self.theta = theta
        self.inv_freq = _upload(
            ctx, llama_rope_inv_freq_host(theta, dims.head_dim)
        )
        self.cos = _zeros(ctx, p_max * self.half)
        self.sin = _zeros(ctx, p_max * self.half)
        ctx.enqueue_function[llama_rope_table_kernel](
            self.cos.unsafe_ptr(),
            self.sin.unsafe_ptr(),
            self.inv_freq.unsafe_ptr(),
            Int32(p_max),
            Int32(self.half),
            grid_dim=(_grid(p_max * self.half), 1, 1),
            block_dim=(LLAMA_TPB, 1, 1),
        )
        ctx.synchronize()


# ===========================================================================
# `rotate_half` (:130-134) and `apply_rotary_pos_emb` (:138-160).
# Seams S9 and S10.
#
#     x1 = x[..., : x.shape[-1] // 2]                                 :132
#     x2 = x[..., x.shape[-1] // 2 :]                                 :133
#     return torch.cat((-x2, x1), dim=-1)                             :134
#     q_embed = (q * cos) + (rotate_half(q) * sin)                    :158
#
# ONE kernel serves q and k, because upstream applies the same function to
# both (:158-159) and a second spelling would be a second place to drift.
# ===========================================================================


def apply_rotary_pos_emb_kernel(
    out_buf: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    hd_in: Int32,
    pos0_in: Int32,
):
    """One thread per output cell of `[M, n_h*head_dim]`, token-major.

    THE POSITION IS THE ABSOLUTE POSITION, ALWAYS (contract S7, 5.5, 7.2).
    Token `t` of this launch is at absolute position `pos0 + t`, so a decode
    step reads the SAME table rows the prefill read for that token. An
    implementation that indexes from the start of the slice is the
    `S07_ROPE_RELATIVE_POSITION` sabotage and it must break clause (d) and
    nothing else.

    S9, PRODUCT twice: `q * cos` and `rotate_half(q) * sin`, each its own
    rounding (LAR:158 writes two separate products).

    S10, UNFUSED ADD: `ftz(ftz(S9a) + ftz(S9b))`. **An fma here is one
    rounding where the reference has three**, and it is the natural thing
    for a kernel to write. Contract DEVIATION 811.

    THE NEGATION IS EXACT. `-x` flips one bit and is not a seam, so it
    carries no flush of its own (the mamba lane's S15 made the same call).
    """
    var m = Int(m_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var hd = Int(hd_in)
    var pos0 = Int(pos0_in)
    var half = hd // 2
    var width = nh * hd
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * width:
        return

    var tok = i // width
    var rem = i - tok * width
    var h = rem // hd
    var d = rem - h * hd
    var t = tok - (tok // l) * l

    var pos = pos0 + t
    comptime if SAB_S07_ROPE_RELATIVE_POSITION:
        pos = t

    # `cat(freqs, freqs)` (:123): column `d` and column `d + half` read the
    # SAME table entry. A COPY, never materialized.
    var f = d
    if f >= half:
        f = f - half
    var cos_v = ftz(cos_tab.unsafe_load(pos * half + f))
    var sin_v = ftz(sin_tab.unsafe_load(pos * half + f))

    var xj = ftz(x.unsafe_load(i))

    # `rotate_half`: `cat(-x2, x1)`. The lower half takes MINUS the upper
    # half's partner; the upper half takes the lower half's partner as is.
    var rh: Float32
    comptime if SAB_S09_ROPE_HALVES_SWAPPED:
        if d < half:
            rh = ftz(x.unsafe_load(i + half))
        else:
            rh = -ftz(x.unsafe_load(i - half))
    else:
        if d < half:
            rh = -ftz(x.unsafe_load(i + half))
        else:
            rh = ftz(x.unsafe_load(i - half))

    var a = ftz(pinned_mul(xj, cos_v))
    comptime if SAB_S10_ROPE_FUSED:
        out_buf.unsafe_store(i, ftz(identical_mul_add(rh, sin_v, a)))
    else:
        var bterm = ftz(pinned_mul(rh, sin_v))
        out_buf.unsafe_store(i, ftz(ftz(a) + ftz(bterm)))


def apply_rotary_pos_emb(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut cos_tab: DeviceBuffer[DType.float32],
    mut sin_tab: DeviceBuffer[DType.float32],
    m: Int,
    l: Int,
    n_h: Int,
    head_dim: Int,
    pos0: Int,
) raises:
    """`apply_rotary_pos_emb(q, k, cos, sin)` (:138-160), one tensor per
    call. `n_h` is `n_heads` for q and `n_kv` for k. ASYNCHRONOUS."""
    ctx.enqueue_function[apply_rotary_pos_emb_kernel](
        out_buf.unsafe_ptr(),
        x.unsafe_ptr(),
        cos_tab.unsafe_ptr(),
        sin_tab.unsafe_ptr(),
        Int32(m),
        Int32(l),
        Int32(n_h),
        Int32(head_dim),
        Int32(pos0),
        grid_dim=(_grid(m * n_h * head_dim), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )


# ===========================================================================
# `past_key_values.update(key_states, value_states, self.layer_idx)`
# (:261-262), which for the default `DynamicCache` is a `torch.cat` along
# the sequence axis.
#
# COPIES, NOT A SEAM (contract section 4's preamble names the KV cache
# append explicitly). DEVIATION 1022: out of place, at the NEW stride.
# ===========================================================================


def kv_append_kernel(
    new_cache: MutPointer[Float32, MutAnyOrigin],
    old_cache: MutPointer[Float32, MutAnyOrigin],
    fresh: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_old_in: Int32,
):
    """One thread per cell of the NEW `[B, n_kv, S_new, head_dim]` cache.

    Slot `j < s_old` copies the old cache (which is packed at stride
    `s_old`); slot `j >= s_old` copies `fresh`, which is token-major
    `[M, n_kv*head_dim]`, at token `j - s_old`.

    **CACHE SLOT `j` IS ABSOLUTE POSITION `j`**, which holds because
    DEVIATION 1028 refuses a call whose `pos0` is not `s_old`.
    """
    var b = Int(b_in)
    var l = Int(l_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var s_old = Int(s_old_in)
    var s_new = s_old + l
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * nkv * s_new * hd:
        return
    var d = i % hd
    var rest = i // hd
    var j = rest % s_new
    var rest2 = rest // s_new
    var kvh = rest2 % nkv
    var bb = rest2 // nkv
    var v: Float32
    if j < s_old:
        v = old_cache.unsafe_load(((bb * nkv + kvh) * s_old + j) * hd + d)
    else:
        var t = j - s_old
        v = fresh.unsafe_load((bb * l + t) * nkv * hd + kvh * hd + d)
    new_cache.unsafe_store(i, v)


# ===========================================================================
# `eager_attention_forward` (:191-213), the pinned path. Contract section 6
# excludes FlashAttention, SDPA, paged attention and chunked prefill, and
# the reason is that an ONLINE softmax's rescale count is the KV TILE COUNT,
# an execution-plan quantity, which is IDENTITY_PATHS rows 3 and 7 moved
# inside the softmax where no contract can reach it by pinning a topology.
#
#     attn_weights = torch.matmul(q, k.transpose(2,3)) * scaling      :204
#     attn_weights = attn_weights + attention_mask                    :206
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1)      :208
#     attn_output  = torch.matmul(attn_weights, value_states)         :210
#
# Nine kernels and `B * n_heads` GEMM calls below, one recorded stage per
# clause, because a lane whose whole instrument is the per-stage card should
# not begin by fusing the stages away (contract section 6's last paragraph).
# ===========================================================================


def gather_q_head_kernel(
    qbh: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    nh_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    h_in: Int32,
    scale_in: Float32,
):
    """DEVIATION 1021: one head's `[L, head_dim]` block, contiguous, for the
    GEMM. A COPY.

    `scale_in` is read ONLY by the `S12_SCALE_INTO_Q` sabotage arm. The
    profile scales the FINISHED dot (S12), never `q`."""
    var l = Int(l_in)
    var nh = Int(nh_in)
    var hd = Int(hd_in)
    var bb = Int(bb_in)
    var h = Int(h_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= l * hd:
        return
    var t = i // hd
    var d = i - t * hd
    var v = q_rope.unsafe_load((bb * l + t) * nh * hd + h * hd + d)
    comptime if SAB_S12_SCALE_INTO_Q:
        # SABOTAGE: one rounding per q element rather than one per score.
        v = ftz(pinned_mul(ftz(v), scale_in))
    qbh.unsafe_store(i, v)


def gather_kv_head_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    cache: MutPointer[Float32, MutAnyOrigin],
    s_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    kvh_in: Int32,
):
    """DEVIATION 1021: one kv head's `[S, head_dim]` block for the GEMM. A
    COPY. `repeat_kv` (:179-188) is the `kvh = h // n_rep` the CALLER passes
    -- an index map, never a materialized expansion (contract DEVIATION
    813)."""
    var s = Int(s_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var bb = Int(bb_in)
    var kvh = Int(kvh_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= s * hd:
        return
    dst.unsafe_store(i, cache.unsafe_load((bb * nkv + kvh) * s * hd + i))


def scatter_scores_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    sbh: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    s_in: Int32,
    nh_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """DEVIATION 1021: the GEMM's `[L, S]` result into
    `[B, n_heads, L, S]`. A COPY, no arithmetic. The scale (S12) is a
    SEPARATE kernel on purpose: a copy that also rounds is a copy nobody
    reviews as a seam."""
    var l = Int(l_in)
    var s = Int(s_in)
    var nh = Int(nh_in)
    var bb = Int(bb_in)
    var h = Int(h_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= l * s:
        return
    var t = i // s
    var j = i - t * s
    scores.unsafe_store(((bb * nh + h) * l + t) * s + j, sbh.unsafe_load(i))


def gather_scores_head_kernel(
    sbh: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    s_in: Int32,
    nh_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """`scatter_scores_kernel` in reverse. SABOTAGE-ONLY: it exists so
    `S19_VALUE_SUM_VIA_GEMM` has a body. Nothing in the profile calls it."""
    var l = Int(l_in)
    var s = Int(s_in)
    var nh = Int(nh_in)
    var bb = Int(bb_in)
    var h = Int(h_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= l * s:
        return
    var t = i // s
    var j = i - t * s
    sbh.unsafe_store(i, src.unsafe_load(((bb * nh + h) * l + t) * s + j))


def scatter_ctx_head_kernel(
    ctxv: MutPointer[Float32, MutAnyOrigin],
    qbh: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    nh_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """One head's `[L, head_dim]` context into `[M, n_heads*head_dim]`.
    SABOTAGE-ONLY, the other half of `S19_VALUE_SUM_VIA_GEMM`."""
    var l = Int(l_in)
    var nh = Int(nh_in)
    var hd = Int(hd_in)
    var bb = Int(bb_in)
    var h = Int(h_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= l * hd:
        return
    var t = i // hd
    var d = i - t * hd
    ctxv.unsafe_store(
        (bb * l + t) * nh * hd + h * hd + d, qbh.unsafe_load(i)
    )


def attn_scale_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale_in: Float32,
):
    """S12, PRODUCT: `* scaling` applied to the FINISHED dot (:204 scales
    the matmul's output). `pinned_mul(score, scale)`, one rounding per
    score.

    `scale` is `identical_rsqrt(Float32(head_dim))` computed ONCE on the
    host (contract DEVIATION 802). **A wrong scale SPELLING is invisible at
    a power-of-four head_dim**: at 16 the scale is 0.25 (0x3E800000) and at
    64 it is 0.125 (0x3E000000), both exact, so contract section 3 requires
    a second fixture at a non-power-of-four `head_dim` such as 24.

    IN PLACE, and safe: one thread owns one cell and reads only that cell.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    comptime if SAB_S12_SCALE_INTO_Q:
        # SABOTAGE: the scale already went into q, so this seam is skipped
        # entirely and the GEMM's own output stands.
        return
    scores.unsafe_store(i, ftz(pinned_mul(ftz(scores.unsafe_load(i)), scale_in)))


def attn_mask_kernel(
    masked: MutPointer[Float32, MutAnyOrigin],
    scores: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    l_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
):
    """S13: `attn_weights + attention_mask` (:206), ONE ADD of
    `finfo(float32).min` where masked and of `+0.0` where not
    (masking_utils.py:601-603). One thread per cell.

    **THE `+0.0` ADD AT AN UNMASKED CELL MAY NOT BE ELIDED** (contract
    4.1(a)). `x + (+0.0) == x` for every finite x, every infinity and every
    NaN EXCEPT `x = -0.0`, where it is `+0.0`. An implementation that skips
    the add keeps a `-0.0` score that the reference LAUNDERS, and it passes
    every fixture that does not PLANT a negative zero in a score by bits.
    `S13_MASK_SELECT` is that mistake.

    **`-FLT_MAX`, NOT `-inf`** (contract 4.1(b)). The causal mask never
    produces a fully masked row, because position t always attends to
    itself, so the two values differ only at a PLANTED extreme score:
    `s + (-FLT_MAX)` is `-FLT_MAX` exactly for `|s| < 2^103` and overflows
    to `-inf` below roughly `-1e31`. `S13_MASK_NEG_INF` is the other value.

    THE MASK IS CAUSAL AND NOTHING ELSE (contract section 11): key at
    absolute position `j` is visible to query token `t` iff
    `j <= pos0 + t`. No sliding window, no prefix mask, no attention sink.
    """
    var b = Int(b_in)
    var nh = Int(nh_in)
    var l = Int(l_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * nh * l * s:
        return
    var j = i % s
    var rest = i // s
    var t = rest % l

    var fill: Float32
    comptime if SAB_S13_MASK_NEG_INF:
        fill = bitcast[DType.float32](UInt32(NEG_INF_BITS))
    else:
        fill = bitcast[DType.float32](UInt32(MASK_FILL_BITS))

    var sv = ftz(scores.unsafe_load(i))
    if j <= pos0 + t:
        comptime if SAB_S13_MASK_SELECT:
            # SABOTAGE: the unmasked cell passes through unchanged, so a
            # planted `-0.0` score survives instead of being laundered.
            masked.unsafe_store(i, sv)
        else:
            masked.unsafe_store(i, ftz(sv + Float32(0.0)))
    else:
        comptime if SAB_S13_MASK_SELECT:
            masked.unsafe_store(i, fill)
        else:
            masked.unsafe_store(i, ftz(sv + fill))


def attn_max_kernel(
    amax: MutPointer[Float32, MutAnyOrigin],
    masked: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    l_in: Int32,
    s_in: Int32,
):
    """S14: the softmax row maximum, `identical_fmax` (DEVIATION 825) folded
    over EVERY element of the row, MASKED CELLS INCLUDED (contract 5.2),
    with no seed. One thread per `(batch, head, query)` row.

    **THE FOLD SHAPE IS FREE AND THIS IS THE ONLY SEAM IN THE PROFILE WHERE
    IT IS** (contract 5.1). `identical_fmax` is exactly commutative and
    associative over all of Float32 including both zeros and NaN, so an
    execution plan may choose its own tree here -- because the operation is
    exactly associative, not because the difference is thought to be small.
    This kernel still walks ascending, because a free shape is not a reason
    to introduce a second one.

    **`pinned_block_max` MAY NOT BE USED** and this is the trap, because it
    is the deterministic-looking helper already in the tree. Its fold is a
    plain `other > red[tid]` compare (`core/pinned_reduce.mojo:159-190`),
    which is precisely the spelling IDENTITY_PATHS row 13 closed everywhere
    else, and its own block comment says a caller whose inputs can carry
    `+-0.0` or NaN must say why before using it. **This caller cannot say
    why.** `max(+0.0, -0.0)` was MEASURED on 2026-08-23 as `-0.0` on Apple
    and `+0.0` on NVIDIA and AMD, and a row of attention scores reaches both
    zero signs easily -- a masked lane one way, a flushed subnormal the
    other.

    WHAT THE CLAUSE COSTS AND WHY IT IS KEPT. Downstream it costs nothing:
    the only consumer is S15, and `s - (+0.0)` and `s - (-0.0)` agree bit
    for bit for every finite s except `s = -0.0`, and `exp` of either
    difference is exactly 1.0. **On the card it costs the clause**, because
    `attn.max` is a recorded stage and clause (a) requires every stage to
    agree bitwise. A laundered divergence is still a divergence at the stage
    that produced it, and the card is the ONLY instrument that can see this
    clause at all.
    """
    var b = Int(b_in)
    var nh = Int(nh_in)
    var l = Int(l_in)
    var s = Int(s_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= b * nh * l:
        return
    var base = r * s
    # No seed: the fold starts at element 0, which is always a real element
    # because `s >= 1` and, being at absolute position 0, is always visible.
    var m = ftz(masked.unsafe_load(base))
    for j in range(1, s):
        var v = ftz(masked.unsafe_load(base + j))
        comptime if SAB_S14_MAX_PLAIN_COMPARE:
            # SABOTAGE: the plain float compare. Identical on every ordinary
            # row and vendor-shaped at +-0.0.
            if v > m:
                m = v
        else:
            m = identical_fmax(m, v)
    amax.unsafe_store(r, m)


def attn_exp_kernel(
    aexp: MutPointer[Float32, MutAnyOrigin],
    masked: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    s_in: Int32,
):
    """S15 and S16: `exp(s - m)`, one thread per cell.

    S15 is a plain subtract, `ftz(ftz(s) - ftz(m))`. S16 is `identical_exp`,
    IDENTITY_PATHS row 12's polynomial, NOT the hardware exp and NOT
    `exp2` with the scale folded into it (which is what MAX's own
    FlashAttention kernels do -- contract section 6's third consequence).

    **THIS IS WHERE THE MASKED TAIL BECOMES EXACTLY `+0.0`** and everything
    in contract section 7 rests on it. For a masked cell the difference is
    about `-3.4e38`; `portable_expf` returns exactly `+0.0` below
    `-87.33655`, and `+0.0` rather than `-0.0`. The denominator is then at
    least `exp(0) = 1.0` (the maximal element contributes exactly 1.0), so
    `identical_div(+0.0, denom)` is exactly `+0.0` too, and both the S17
    chain and the S19 chain are bitwise inert on that tail. That is the
    theorem decode-equals-prefill and length-invariance are built on.
    """
    var n = Int(n_in)
    var s = Int(s_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var r = i // s
    var d = ftz(ftz(masked.unsafe_load(i)) - ftz(amax.unsafe_load(r)))
    aexp.unsafe_store(i, ftz(identical_exp(d)))


def attn_denom_kernel(
    denom: MutPointer[Float32, MutAnyOrigin],
    aexp: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    l_in: Int32,
    s_in: Int32,
):
    """S17: the softmax denominator, a SERIAL ASCENDING CHAIN over the
    ABSOLUTE key index from `+0.0`, plain adds. Contract 5.3. One thread per
    `(batch, head, query)` row.

        acc = +0.0
        for j = 0 .. s-1:  acc = ftz(ftz(acc) + ftz(e[j]))

    Nothing to fuse: `e[j]` is not a product.

    **`core/pinned_reduce.mojo::pinned_block_sum` MAY NOT BE USED FOR THIS**,
    and saying so is the point of the clause. It is a halving tree, its own
    docstring says a halving tree and CUB's warp-then-block shape combine
    different partials, and a halving tree is not a serial ascending chain
    either. Reaching for the deterministic block fold BECAUSE it is the
    deterministic block fold is the single most likely way to get this
    wrong, and it would be wrong in a way that passes every
    launch-invariance gate, because the tree is perfectly launch-invariant.
    It is simply a different sum.

    TWO REASONS FOR THE CHAIN, AND THE SECOND IS THE LOAD-BEARING ONE.
    First, it is the same spelling as S1 and as the mamba contract's S10, so
    the block has ONE fold shape rather than two. Second, **it is what makes
    decode equal prefill and makes the answer independent of sequence
    length**: under the GEMM's topology `P` is a pure function of the
    contraction length, so a row folded over 257 keys and the same row
    folded over 5 keys have different trees and different bits, while under
    a serial ascending chain seeded `+0.0` a tail of exactly-`+0.0` terms is
    bitwise inert.

    THE PRICE, STATED RATHER THAN HIDDEN. A row's fold may not be split
    across threads, so v1's kv length is bounded by what one thread will
    walk, and this profile is reference quality and SLOW BY CONSTRUCTION. A
    v2 that wants a tree must fix the fold LENGTH so prefill and decode fold
    the same number of terms, which means folding over the ALLOCATED cache
    length, which makes `P` a function of an allocation quantity and puts
    batch invariance back at risk. That is a research question and contract
    5.3 names it as one.

    `scratch` is DEVIATION 1023: `attn.weights`, borrowed by the sabotage
    arm and untouched by the profile.
    """
    var b = Int(b_in)
    var nh = Int(nh_in)
    var l = Int(l_in)
    var s = Int(s_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= b * nh * l:
        return
    var base = r * s

    comptime if SAB_S17_DENOM_HALVING_TREE:
        # SABOTAGE: a halving tree over the same terms. Perfectly
        # deterministic, perfectly launch-invariant, and a DIFFERENT SUM.
        # At s = 3 it folds (0+2) then (+1) where the chain folds
        # ((0+1)+2).
        for j in range(s):
            scratch.unsafe_store(base + j, ftz(aexp.unsafe_load(base + j)))
        var n = s
        while n > 1:
            var halfn = (n + 1) // 2
            for j in range(n - halfn):
                var a = ftz(scratch.unsafe_load(base + j))
                var c = ftz(scratch.unsafe_load(base + halfn + j))
                scratch.unsafe_store(base + j, ftz(a + c))
            n = halfn
        denom.unsafe_store(r, ftz(scratch.unsafe_load(base)))
    else:
        var acc = Float32(0.0)
        for j in range(s):
            acc = ftz(ftz(acc) + ftz(aexp.unsafe_load(base + j)))
        denom.unsafe_store(r, ftz(acc))


def attn_weights_kernel(
    weights: MutPointer[Float32, MutAnyOrigin],
    aexp: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    s_in: Int32,
):
    """S18: `e / denom`, ONE DIVISION per weight through `identical_div`
    (IDENTITY_PATHS row 49's `portable_divf`), never a reciprocal
    multiplied in. One thread per cell.

    **A RECIPROCAL MULTIPLIED IN IS A DIFFERENT ANSWER.** `e * (1/denom)`
    rounds twice where `e / denom` rounds once and they differ in the last
    bit on ordinary inputs.

    **THE REFERENCE'S OWN SPELLING COULD NOT BE VERIFIED.**
    `nn.functional.softmax` dispatches into ATen and there is no PyTorch
    checkout in `/Users/andrewhendel/CascadeProjects/upstream/`, so whether
    ATen's CUDA softmax divides or multiplies by a reciprocal is NOT KNOWN
    to this lane. MAX's `softmax_kernel` multiplies by a reciprocal, which
    is evidence about MAX and not about the reference. The profile pins the
    DIVISION because it is the spelling with one rounding and because
    `identical_div` is already characterized per class on Apple. **This is a
    stated gap, not a decision made on evidence** (contract 5.4), and it is
    the first thing to check when a PyTorch checkout lands.
    """
    var n = Int(n_in)
    var s = Int(s_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var r = i // s
    var e = ftz(aexp.unsafe_load(i))
    var dv = ftz(denom.unsafe_load(r))
    comptime if SAB_S18_RECIPROCAL_MUL:
        var recip = ftz(identical_div(Float32(1.0), dv))
        weights.unsafe_store(i, ftz(pinned_mul(e, recip)))
    else:
        weights.unsafe_store(i, ftz(identical_div(e, dv)))


def attn_context_kernel(
    ctxv: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_in: Int32,
):
    """S19: the attention-weighted value sum, SERIAL ASCENDING over the
    ABSOLUTE key index from `+0.0`, `acc = ftz(fma(w[j], v[j][d], acc))`.
    One thread per output cell of `[M, n_heads*head_dim]`.

    **DELIBERATELY NOT A `gemm.fp32.v1` CALL** (contract DEVIATION 807), and
    this is the least obvious decision in the document, so it is argued
    here rather than cited.

    S11, the QK product, CONTRACTS OVER `head_dim`, which is the same
    integer in prefill and in decode. GEMM v1's per-cell arithmetic is a
    pure function of the contraction length `k` and the profile, so a
    contraction axis whose LENGTH is the same in both paths is safe to hand
    to it. S19 contracts over the KEY AXIS, whose length is `t + 1` in
    decode and `L` in prefill. Under the GEMM's leaf-and-tree topology
    `P = f(k)`, so those are different trees over the same numbers and the
    extra `L - t - 1` masked terms would NOT be bitwise inert. Under a
    serial ascending chain seeded `+0.0` they ARE inert, because
    `fma(+0.0, v, acc)` is `acc + (+-0.0)`, which is `acc` for every `acc`
    except `acc = -0.0`, and the `+0.0` seed forbids that.

    So the same asymmetry that makes S11 safe to reuse makes S19 unsafe to
    reuse, and `S19_VALUE_SUM_VIA_GEMM` is the arm that proves it: it must
    leave clause (a) GREEN at a fixed length and break clause (d) past the
    first 128 keys, which is exactly the failure a single-path gate cannot
    see.

    `repeat_kv` is `kvh = h // n_rep` (contract DEVIATION 813). **At
    `n_rep == 1` a broken head-to-kv-head map is INVISIBLE**, so the gates
    must carry both `n_rep == 1` and `n_rep == 2`.
    """
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var s = Int(s_in)
    var n_rep = nh // nkv
    var width = nh * hd
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * l * width:
        return
    var tok = i // width
    var rem = i - tok * width
    var h = rem // hd
    var d = rem - h * hd
    var bb = tok // l
    var t = tok - bb * l
    var kvh = h // n_rep
    var wbase = ((bb * nh + h) * l + t) * s
    var vbase = (bb * nkv + kvh) * s * hd

    var acc = Float32(0.0)
    for j in range(s):
        var w = ftz(weights.unsafe_load(wbase + j))
        var v = ftz(v_cache.unsafe_load(vbase + j * hd + d))
        acc = ftz(identical_mul_add(w, v, acc))
    ctxv.unsafe_store(i, acc)


# ===========================================================================
# `LlamaMLP.forward` (:174-176). Seams S20 and S21.
#
#     down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))     :175
# ===========================================================================


def silu_kernel(
    silu_out: MutPointer[Float32, MutAnyOrigin],
    gate: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """S20: `ACT2FN["silu"]`, which is ATen's `F.silu`, the SINGLE QUOTIENT
    `z / (1 + exp(-z))` -- ONE division and ONE rounding.

    **NOT `z * sigmoid(z)`**, which is two roundings and which is what MAX
    itself spells (`max/kernels/src/nn/activations.mojo:249`). DEVIATION 744,
    IDENTITY_PATHS row 53. `S20_SILU_MUL_SIGMOID` is the other spelling."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var z = ftz(gate.unsafe_load(i))
    comptime if SAB_S20_SILU_MUL_SIGMOID:
        var sg = ftz(
            identical_div(
                Float32(1.0), ftz(Float32(1.0) + ftz(identical_exp(-z)))
            )
        )
        silu_out.unsafe_store(i, ftz(pinned_mul(z, sg)))
    else:
        silu_out.unsafe_store(i, ftz(identical_silu(z)))


def mlp_gated_kernel(
    gated: MutPointer[Float32, MutAnyOrigin],
    silu_out: MutPointer[Float32, MutAnyOrigin],
    up: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """S21, PRODUCT: `silu(gate_proj(x)) * up_proj(x)` (:175), one
    rounding, uncontractible into the `down_proj` GEMM that consumes it."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    gated.unsafe_store(
        i,
        ftz(
            pinned_mul(ftz(silu_out.unsafe_load(i)), ftz(up.unsafe_load(i)))
        ),
    )


# ===========================================================================
# THE REFUSAL, contract section 8. Before ANY recorded stage.
# ===========================================================================


def _refuse_nonfinite_named(name: String, values: List[Float32]) raises:
    """Contract section 8 / IDENTITY_PATHS row 39, the device path's copy of
    the oracle's `refuse_nonfinite`. Tested BY BITS, not by compares: **Metal
    FLUSHES COMPARE OPERANDS** (row 49), so `-subnormal < 0.0` is FALSE
    there while the sign bit is set, and a bit test is the only spelling
    with one meaning on every column. DEVIATION 1027 explains why this is
    not the oracle's function imported."""
    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > UInt32(0x7F800000):
            raise Error(
                String("llama: NaN in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (row 39: NaN payloads are vendor-shaped;"
                + " no stage may record one)"
            )
        if au == UInt32(0x7F800000):
            raise Error(
                String("llama: infinity in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (row 39)"
            )


def llama_refuse_bad_call(
    ctx: DeviceContext,
    mut w: LlamaDeviceWeights,
    mut rope: LlamaRopeTable,
    mut x: DeviceBuffer[DType.float32],
    mut kv: LlamaKVCache,
    b: Int,
    l: Int,
) raises:
    """The inputs that ACTUALLY CHANGE from one block call to the next.

    DEVIATION 1875. `llama_refuse_bad_inputs` walks thirteen names, ten of
    which are weights that are identical on every call and are now refused
    once, on the host, in `LlamaDeviceWeights.__init__`. What is left here is
    what a second call can genuinely differ in: the hidden states, the rotary
    table, and the carried key and value caches.

    The names and their ORDER are the same as the corresponding entries in
    `llama_refuse_bad_inputs`, because the point of using the upstream names
    at all is that this side and the oracle's side fail identically.

    The KV entries are the reason this cannot be hoisted with the weights.
    They are WRITTEN BY THE BLOCK, so a decode step's input is the previous
    step's output, and two of the refusal audit's plants land there
    specifically to exercise a refusal on a SECOND call.
    """
    var dims = w.dims.copy()
    var dm = dims.d_model
    var hd = dims.head_dim
    _refuse_nonfinite_named("hidden_states", _download(ctx, x, b * l * dm))
    _refuse_nonfinite_named(
        "rotary_emb.inv_freq", _download(ctx, rope.inv_freq, dims.half())
    )
    if kv.s > 0:
        _refuse_nonfinite_named(
            "past_key_values.key_cache",
            _download(ctx, kv.k, b * dims.n_kv * kv.s * hd),
        )
        _refuse_nonfinite_named(
            "past_key_values.value_cache",
            _download(ctx, kv.v, b * dims.n_kv * kv.s * hd),
        )


def llama_refuse_bad_inputs(
    ctx: DeviceContext,
    mut w: LlamaDeviceWeights,
    mut rope: LlamaRopeTable,
    mut x: DeviceBuffer[DType.float32],
    mut kv: LlamaKVCache,
    b: Int,
    l: Int,
) raises:
    """Every named input and parameter of one block call, refused if it
    holds a NaN or an infinity. The names are the upstream parameter names
    so that this side and the oracle's side fail IDENTICALLY, which is the
    only way the two refusals can be compared at all.

    **THE REFUSAL COVERS INPUTS AND NOT INTERMEDIATES**, and that is a
    STATED GAP, contract DEVIATION 815. A score can overflow to `-inf`
    through S13 at a planted extreme, and at extreme weights `exp` can
    saturate. Those are deterministic and the same on every vendor, so they
    do not break identity; a COMPUTED NaN would, and this profile does not
    check for one at every stage. What catches it is the card, since a stage
    hash containing a vendor-shaped payload cannot match.
    """
    var dims = w.dims.copy()
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var hd = dims.head_dim
    _refuse_nonfinite_named("hidden_states", _download(ctx, x, b * l * dm))
    _refuse_nonfinite_named(
        "input_layernorm.weight", _download(ctx, w.norm1_w, dm)
    )
    _refuse_nonfinite_named(
        "post_attention_layernorm.weight", _download(ctx, w.norm2_w, dm)
    )
    _refuse_nonfinite_named("q_proj.weight", _download(ctx, w.w_q, qw * dm))
    _refuse_nonfinite_named("k_proj.weight", _download(ctx, w.w_k, kw * dm))
    _refuse_nonfinite_named("v_proj.weight", _download(ctx, w.w_v, kw * dm))
    _refuse_nonfinite_named("o_proj.weight", _download(ctx, w.w_o, dm * qw))
    _refuse_nonfinite_named(
        "gate_proj.weight", _download(ctx, w.w_gate, it * dm)
    )
    _refuse_nonfinite_named("up_proj.weight", _download(ctx, w.w_up, it * dm))
    _refuse_nonfinite_named(
        "down_proj.weight", _download(ctx, w.w_down, dm * it)
    )
    _refuse_nonfinite_named(
        "rotary_emb.inv_freq", _download(ctx, rope.inv_freq, dims.half())
    )
    if kv.s > 0:
        _refuse_nonfinite_named(
            "past_key_values.key_cache",
            _download(ctx, kv.k, b * dims.n_kv * kv.s * hd),
        )
        _refuse_nonfinite_named(
            "past_key_values.value_cache",
            _download(ctx, kv.v, b * dims.n_kv * kv.s * hd),
        )


# ===========================================================================
# THE ATTENTION SUBMODULE
# ===========================================================================


def llama_attention_scale(head_dim: Int) -> Float32:
    """`self.scaling = self.head_dim ** -0.5` (`LlamaAttention.__init__`
    :226), computed ONCE on the host and stored FP32. Contract DEVIATION
    802.

    The SPELLING is pinned (`identical_rsqrt`, DEVIATION 741) so that the
    value does not depend on the host's libm, which is IDENTITY_PATHS row
    18's hazard applied to a constant. Contract section 3 records that
    `head_dim ** -0.5` in float64 rounded once to FP32 equals
    `f32div(1, f32sqrt(head_dim))` at head_dim in
    {8, 16, 32, 64, 80, 128, 256} -- measured on the HOST on 2026-08-24 and
    NOT on a device, which is agreement at seven points and not a proof."""
    return ftz(identical_rsqrt(Float32(head_dim)))


def eager_attention_forward(
    ctx: DeviceContext,
    mut stages: LlamaDeviceStages,
    b: Int,
    l: Int,
    s: Int,
    pos0: Int,
    dims: LlamaDims,
    plant_at: Int,
    plant_idx: List[Int],
    plant_bits: List[UInt32],
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`eager_attention_forward(module, query, key, value, attention_mask,
    scaling, dropout)` (:191-213), inference only, `dropout = 0.0` and
    REFUSED as a config (contract section 2).

    Their order, and this function's:

        key_states   = repeat_kv(key, n_rep)                          :202
        value_states = repeat_kv(value, n_rep)                        :203
        attn_weights = matmul(query, key_states.transpose(2,3))*scale :204
        attn_weights = attn_weights + attention_mask                  :206
        attn_weights = softmax(attn_weights, dim=-1, float32)         :208
        attn_output  = matmul(attn_weights, value_states)             :210

    `repeat_kv` is not spelled: it is `h // n_rep` wherever a kv head is
    indexed, which is contract DEVIATION 813's declared COPY.

    Records `attn.scores`, `attn.masked`, `attn.max`, `attn.exp`,
    `attn.denom`, `attn.weights` and `attn.ctx`, in contract section 9's
    order. Every one of those is a separate materialized buffer, which the
    eager path costs and which contract section 6 argues is the point: a
    lane whose whole instrument is the per-stage card should not begin by
    fusing the stages away.
    """
    var nh = dims.n_heads
    var nkv = dims.n_kv
    var hd = dims.head_dim
    var n_rep = dims.n_rep()
    var cells = b * nh * l * s
    var scale = llama_attention_scale(hd)

    # ---- S11 (:204's matmul). `gemm.fp32.v1` OP_NT with `k = head_dim`,
    #      ONE CALL PER (batch, head). Contract DEVIATION 808.
    #      C[L, S] = q_head[L, hd] . k_head[S, hd]^T.
    #      `k = head_dim` is the SAME integer in prefill and decode, which
    #      is what makes this reuse decode-safe (contract 7.2). See
    #      DEVIATION 1029 for the part of that argument the contract leaves
    #      to the gemm profile's own plan-invariance gate.
    for bb in range(b):
        for h in range(nh):
            var kvh = h // n_rep
            ctx.enqueue_function[gather_q_head_kernel](
                stages.qbh.unsafe_ptr(),
                stages.q_rope.unsafe_ptr(),
                Int32(l),
                Int32(nh),
                Int32(hd),
                Int32(bb),
                Int32(h),
                scale,
                grid_dim=(_grid(l * hd), 1, 1),
                block_dim=(LLAMA_TPB, 1, 1),
            )
            ctx.enqueue_function[gather_kv_head_kernel](
                stages.kbh.unsafe_ptr(),
                stages.k_cache.unsafe_ptr(),
                Int32(s),
                Int32(nkv),
                Int32(hd),
                Int32(bb),
                Int32(kvh),
                grid_dim=(_grid(s * hd), 1, 1),
                block_dim=(LLAMA_TPB, 1, 1),
            )
            ctx.synchronize()
            identical_gemm(
                ctx,
                stages.sbh,
                stages.qbh,
                stages.kbh,
                l,
                s,
                hd,
                _gemm_op_nt(),
            )
            ctx.enqueue_function[scatter_scores_kernel](
                stages.scores.unsafe_ptr(),
                stages.sbh.unsafe_ptr(),
                Int32(l),
                Int32(s),
                Int32(nh),
                Int32(bb),
                Int32(h),
                grid_dim=(_grid(l * s), 1, 1),
                block_dim=(LLAMA_TPB, 1, 1),
            )
            ctx.synchronize()

    # ---- S12 (:204's `* scaling`), applied to the FINISHED dot.
    ctx.enqueue_function[attn_scale_kernel](
        stages.scores.unsafe_ptr(),
        Int32(cells),
        scale,
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    # The score plant, injection point 1. Applied AFTER S12 and BEFORE the
    # stage is recorded, so the planted bits are IN `attn.scores` -- which
    # is the oracle's order and the only order under which clause (a) can
    # compare a planted case at all.
    _plant_bits(
        ctx,
        stages.scores,
        cells,
        PLANT_AT_SCORES,
        plant_at,
        plant_idx,
        plant_bits,
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.scores", stages.scores, cells
    )

    # ---- S13 (:206). An ADD, of -FLT_MAX or of +0.0. Not a select.
    ctx.enqueue_function[attn_mask_kernel](
        stages.masked.unsafe_ptr(),
        stages.scores.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(l),
        Int32(s),
        Int32(pos0),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    # The score plant, injection point 2. `PLANT_MASKED_ZERO_ROW` and any
    # other case whose separating value has to survive the mask lands here.
    _plant_bits(
        ctx,
        stages.masked,
        cells,
        PLANT_AT_MASKED,
        plant_at,
        plant_idx,
        plant_bits,
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.masked", stages.masked, cells
    )

    # ---- S14 (:208's row max). `identical_fmax`, fold shape free.
    ctx.enqueue_function[attn_max_kernel](
        stages.amax.unsafe_ptr(),
        stages.masked.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(l),
        Int32(s),
        grid_dim=(_grid(b * nh * l), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.max", stages.amax, b * nh * l
    )

    # ---- S15 and S16 (:208). `exp(s - m)`.
    ctx.enqueue_function[attn_exp_kernel](
        stages.aexp.unsafe_ptr(),
        stages.masked.unsafe_ptr(),
        stages.amax.unsafe_ptr(),
        Int32(cells),
        Int32(s),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.exp", stages.aexp, cells
    )

    # ---- S17 (:208's denominator). SERIAL ASCENDING, ABSOLUTE index.
    ctx.enqueue_function[attn_denom_kernel](
        stages.denom.unsafe_ptr(),
        stages.aexp.unsafe_ptr(),
        stages.weights.unsafe_ptr(),  # DEVIATION 1023, sabotage scratch
        Int32(b),
        Int32(nh),
        Int32(l),
        Int32(s),
        grid_dim=(_grid(b * nh * l), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.denom", stages.denom, b * nh * l
    )

    # ---- S18 (:208's normalize). ONE DIVISION per weight.
    ctx.enqueue_function[attn_weights_kernel](
        stages.weights.unsafe_ptr(),
        stages.aexp.unsafe_ptr(),
        stages.denom.unsafe_ptr(),
        Int32(cells),
        Int32(s),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.weights", stages.weights, cells
    )

    # ---- S19 (:210). SERIAL ASCENDING over the ABSOLUTE key index, fma.
    #      NOT a gemm call. Contract DEVIATION 807 and section 7.2.
    comptime if SAB_S19_VALUE_SUM_VIA_GEMM:
        # SABOTAGE: `C[L, hd] = W[L, S] . V[S, hd]`, an OP_NN cell with
        # `k = S`. `P = f(S)` and `S` differs between prefill and decode, so
        # the masked `+0.0` tail stops being bitwise inert. Clause (a) stays
        # GREEN at a fixed length; clause (d) is what falls over.
        for bb2 in range(b):
            for h2 in range(nh):
                var kvh2 = h2 // n_rep
                ctx.enqueue_function[gather_scores_head_kernel](
                    stages.sbh.unsafe_ptr(),
                    stages.weights.unsafe_ptr(),
                    Int32(l),
                    Int32(s),
                    Int32(nh),
                    Int32(bb2),
                    Int32(h2),
                    grid_dim=(_grid(l * s), 1, 1),
                    block_dim=(LLAMA_TPB, 1, 1),
                )
                ctx.enqueue_function[gather_kv_head_kernel](
                    stages.kbh.unsafe_ptr(),
                    stages.v_cache.unsafe_ptr(),
                    Int32(s),
                    Int32(nkv),
                    Int32(hd),
                    Int32(bb2),
                    Int32(kvh2),
                    grid_dim=(_grid(s * hd), 1, 1),
                    block_dim=(LLAMA_TPB, 1, 1),
                )
                ctx.synchronize()
                identical_gemm(
                    ctx,
                    stages.qbh,
                    stages.sbh,
                    stages.kbh,
                    l,
                    hd,
                    s,
                    _gemm_op_nn(),
                )
                ctx.enqueue_function[scatter_ctx_head_kernel](
                    stages.ctxv.unsafe_ptr(),
                    stages.qbh.unsafe_ptr(),
                    Int32(l),
                    Int32(nh),
                    Int32(hd),
                    Int32(bb2),
                    Int32(h2),
                    grid_dim=(_grid(l * hd), 1, 1),
                    block_dim=(LLAMA_TPB, 1, 1),
                )
                ctx.synchronize()
    else:
        ctx.enqueue_function[attn_context_kernel](
            stages.ctxv.unsafe_ptr(),
            stages.weights.unsafe_ptr(),
            stages.v_cache.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(nh),
            Int32(nkv),
            Int32(hd),
            Int32(s),
            grid_dim=(_grid(b * l * nh * hd), 1, 1),
            block_dim=(LLAMA_TPB, 1, 1),
        )
        ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".attn.ctx", stages.ctxv, b * l * nh * hd
    )


def llama_attention_forward(
    ctx: DeviceContext,
    mut stages: LlamaDeviceStages,
    mut kv: LlamaKVCache,
    mut rope: LlamaRopeTable,
    mut w: LlamaDeviceWeights,
    b: Int,
    l: Int,
    pos0: Int,
    plant_at: Int,
    plant_idx: List[Int],
    plant_bits: List[UInt32],
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`LlamaAttention.forward(hidden_states, position_embeddings,
    attention_mask, past_key_values)` (:243-281), eager path, inference
    only.

    Their order, and this function's:

        query_states = q_proj(h).view(hidden_shape).transpose(1, 2)   :252
        key_states   = k_proj(h).view(hidden_shape).transpose(1, 2)   :253
        value_states = v_proj(h).view(hidden_shape).transpose(1, 2)   :254
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                   :257
        k, v = past_key_values.update(k, v, self.layer_idx)           :261
        attn_output, _ = attention_interface(...)                     :270
        attn_output = attn_output.reshape(*input_shape, -1)           :279
        attn_output = self.o_proj(attn_output)                        :280

    `view` and `transpose` are COPIES that move no bits, and the head
    structure lives in the INDEX ARITHMETIC of the kernels rather than in a
    materialized transpose. So does the `reshape` at :279, which is why
    `attn.ctx` is already `[M, n_heads*head_dim]` and feeds `o_proj`
    directly.

    `attention_interface` is `eager_attention_forward` and nothing else.
    :264-266 lets `ALL_ATTENTION_FUNCTIONS` choose it at run time; **v1 pins
    the eager path** (contract section 6) because SDPA's arithmetic is
    chosen by whichever backend the runtime picks, which is the same class
    of defect as a vendor BLAS dispatch.

    `norm1.out` arrives as this function's input from `stages`: the decoder
    layer normalizes first and this reads `stages.norm1_out`, rather than
    taking a tensor argument.
    """
    var dims = stages.dims.copy()
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var nkv = dims.n_kv
    var hd = dims.head_dim
    var m = b * l
    var s_old = kv.s
    var s = s_old + l

    # ---- q_proj, k_proj, v_proj (:252-254). `nn.Linear(d_model, *,
    #      bias=attention_bias)` with `attention_bias` False, so weight
    #      only. GEMM v1 OP_NT: C[M, out] = norm1_out[M, dm] . W[out, dm]^T.
    identical_gemm(
        ctx, stages.q_proj, stages.norm1_out, w.w_q, m, qw, dm, _gemm_op_nt()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".q_proj.out", stages.q_proj, m * qw
    )
    identical_gemm(
        ctx, stages.k_proj, stages.norm1_out, w.w_k, m, kw, dm, _gemm_op_nt()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".k_proj.out", stages.k_proj, m * kw
    )
    identical_gemm(
        ctx, stages.v_proj, stages.norm1_out, w.w_v, m, kw, dm, _gemm_op_nt()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".v_proj.out", stages.v_proj, m * kw
    )

    # ---- the rotary table (:113-127). COMPUTED once per configuration,
    #      RECORDED every call. DEVIATION 1024: contract section 9 says
    #      "once" and also lists these three inside a thirty-tag card order
    #      the differ aligns on; this file resolves toward the differ.
    trace.record_device[DType.float32](
        ctx, prefix + ".rope.inv_freq", rope.inv_freq, dims.half()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".rope.cos", rope.cos, rope.p_max * rope.half
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".rope.sin", rope.sin, rope.p_max * rope.half
    )

    # ---- apply_rotary_pos_emb (:257). S9 and S10, q then k. V IS NOT
    #      ROTATED (:257 takes q and k only).
    apply_rotary_pos_emb(
        ctx,
        stages.q_rope,
        stages.q_proj,
        rope.cos,
        rope.sin,
        m,
        l,
        dims.n_heads,
        hd,
        pos0,
    )
    apply_rotary_pos_emb(
        ctx,
        stages.k_rope,
        stages.k_proj,
        rope.cos,
        rope.sin,
        m,
        l,
        nkv,
        hd,
        pos0,
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".q_rope.out", stages.q_rope, m * qw
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".k_rope.out", stages.k_rope, m * kw
    )

    # ---- past_key_values.update (:261-262). Copies. DEVIATION 1022: the
    #      repack is OUT OF PLACE, at the NEW stride S.
    ctx.enqueue_function[kv_append_kernel](
        stages.k_cache.unsafe_ptr(),
        kv.k.unsafe_ptr(),
        stages.k_rope.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(nkv),
        Int32(hd),
        Int32(s_old),
        grid_dim=(_grid(b * nkv * s * hd), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.enqueue_function[kv_append_kernel](
        stages.v_cache.unsafe_ptr(),
        kv.v.unsafe_ptr(),
        stages.v_proj.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(nkv),
        Int32(hd),
        Int32(s_old),
        grid_dim=(_grid(b * nkv * s * hd), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".kv.k_cache", stages.k_cache, b * nkv * s * hd
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".kv.v_cache", stages.v_cache, b * nkv * s * hd
    )
    # The cache carries the NEW packed contents into the next call. A copy,
    # taken AFTER the stages are recorded so the stage is the cache as
    # computed. DEVIATION 1022 is why the two buffers are distinct; the rest
    # of THIS call reads `stages.k_cache` and `stages.v_cache`, which hold
    # the same bits.
    ctx.enqueue_copy(dst_buf=kv.k, src_buf=stages.k_cache)
    ctx.enqueue_copy(dst_buf=kv.v, src_buf=stages.v_cache)
    ctx.synchronize()
    kv.s = s

    # ---- the attention interface (:264-277). Eager, pinned.
    eager_attention_forward(
        ctx,
        stages,
        b,
        l,
        s,
        pos0,
        dims,
        plant_at,
        plant_idx,
        plant_bits,
        trace,
        prefix,
    )

    # ---- o_proj (:280). `nn.Linear(n_heads*head_dim, d_model,
    #      bias=attention_bias)`, no bias.
    #      C[M, dm] = ctx[M, qw] . w_o[dm, qw]^T, `k = n_heads*head_dim`.
    identical_gemm(
        ctx, stages.o_proj, stages.ctxv, w.w_o, m, dm, qw, _gemm_op_nt()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".o_proj.out", stages.o_proj, m * dm
    )


# ===========================================================================
# THE MLP SUBMODULE
# ===========================================================================


def llama_mlp_forward(
    ctx: DeviceContext,
    mut stages: LlamaDeviceStages,
    mut w: LlamaDeviceWeights,
    m: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`LlamaMLP.forward(x)` (:174-176).

        down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))   :175

    Three `OP_NT` GEMM cells and two elementwise seams. `mlp_bias` is False
    (configuration_llama.py:83), so no projection carries a bias.

    Reads `stages.norm2_out` and writes `stages.down_proj`.
    """
    var dims = stages.dims.copy()
    var dm = dims.d_model
    var it = dims.intermediate

    # ---- gate_proj and up_proj. C[M, it] = norm2_out[M, dm] . W[it, dm]^T.
    identical_gemm(
        ctx,
        stages.gate_proj,
        stages.norm2_out,
        w.w_gate,
        m,
        it,
        dm,
        _gemm_op_nt(),
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".gate_proj.out", stages.gate_proj, m * it
    )
    identical_gemm(
        ctx,
        stages.up_proj,
        stages.norm2_out,
        w.w_up,
        m,
        it,
        dm,
        _gemm_op_nt(),
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".up_proj.out", stages.up_proj, m * it
    )

    # ---- act_fn (:175). S20.
    ctx.enqueue_function[silu_kernel](
        stages.silu_out.unsafe_ptr(),
        stages.gate_proj.unsafe_ptr(),
        Int32(m * it),
        grid_dim=(_grid(m * it), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".silu.out", stages.silu_out, m * it
    )

    # ---- the gate product (:175). S21.
    ctx.enqueue_function[mlp_gated_kernel](
        stages.gated.unsafe_ptr(),
        stages.silu_out.unsafe_ptr(),
        stages.up_proj.unsafe_ptr(),
        Int32(m * it),
        grid_dim=(_grid(m * it), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".mlp.gated", stages.gated, m * it
    )

    # ---- down_proj (:175). C[M, dm] = gated[M, it] . w_down[dm, it]^T,
    #      `k = intermediate_size`. **THIS IS THE ONE CELL IN A SMALL
    #      CONFIG WHERE THE GEMM'S BALANCED FOLD TREE RUNS AT ALL.** Every
    #      other projection has `k <= 128`, which is `P == 1` under the gemm
    #      profile; at `intermediate_size = 300` this one has `P = 3` with a
    #      ragged 44-element last leaf and one carry (contract section 3).
    #      Without that fixture the tree sits unexercised inside the block.
    identical_gemm(
        ctx,
        stages.down_proj,
        stages.gated,
        w.w_down,
        m,
        dm,
        it,
        _gemm_op_nt(),
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".down_proj.out", stages.down_proj, m * dm
    )


# ===========================================================================
# `LlamaDecoderLayer.forward` (:295-324). THE ENTRY POINT.
# ===========================================================================


def llama_decoder_layer_forward_planted(
    ctx: DeviceContext,
    mut stages: LlamaDeviceStages,
    mut kv: LlamaKVCache,
    mut rope: LlamaRopeTable,
    mut w: LlamaDeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    pos0: Int,
    plant_at: Int,
    plant_idx: List[Int],
    plant_bits: List[UInt32],
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`LlamaDecoderLayer.forward(hidden_states, ...)` (:295-324).

        residual = hidden_states                                      :305
        hidden_states = self.input_layernorm(hidden_states)           :306
        hidden_states, _ = self.self_attn(...)                        :308
        hidden_states = residual + hidden_states                      :317
        residual = hidden_states                                      :320
        hidden_states = self.post_attention_layernorm(hidden_states)  :321
        hidden_states = self.mlp(hidden_states)                       :322
        hidden_states = residual + hidden_states                      :323

    **THE ENTRY POINT of profile
    `mojolearn.identical.transformer.fp32.v1`.**

    Prefill is a fresh zero `LlamaKVCache` at `pos0 = 0`; the decode step is
    this same function at `l == 1` and `pos0 = kv.s`, carrying the cache.
    ONE SPELLING FOR BOTH PATHS -- which is what makes contract section 10's
    clause (d) (decode == prefill, bitwise, at every position) a theorem the
    gate then verifies, rather than a coincidence the gate hopes for. The
    theorem's two halves are in `attn_denom_kernel` and
    `attn_context_kernel`, and the fact underneath both is in
    `attn_exp_kernel`.

    Emits exactly the thirty stages of contract section 9, in section 9's
    order, under `prefix`. Tags must be UNIQUE within a trace
    (`core/identity_trace.mojo`'s tag invariant), so the DRIVER supplies the
    prefix -- one per block call -- and it carries no machine property.

    **NOTHING BELOW HAS BEEN COMPILED OR RUN.** Read the header.
    """
    var dims = stages.dims.copy()
    dims.validate()
    var dm = dims.d_model
    var m = b * l

    if b <= 0 or l <= 0:
        raise Error(
            String("llama_decoder_layer_forward: B and L must be positive,")
            + " got B="
            + String(b)
            + " L="
            + String(l)
        )
    if stages.b != b or stages.l != l:
        raise Error(
            String("llama_decoder_layer_forward: stages were built for B=")
            + String(stages.b)
            + " L="
            + String(stages.l)
            + " but the call is B="
            + String(b)
            + " L="
            + String(l)
        )
    if kv.b != b or kv.n_kv != dims.n_kv or kv.head_dim != dims.head_dim:
        raise Error(
            "llama_decoder_layer_forward: the KV cache's shape is not the"
            " call's"
        )
    if kv.s_max != stages.s_max:
        raise Error(
            String("llama_decoder_layer_forward: the cache holds s_max=")
            + String(kv.s_max)
            + " but the stages were sized for s_max="
            + String(stages.s_max)
            + "; a stage recorded past the cache is uninitialized memory"
        )
    if w.dims.d_model != dims.d_model:
        raise Error(
            "llama_decoder_layer_forward: the weights' d_model is not the"
            " stages'"
        )
    # DEVIATION 1028. Cache slot j IS absolute position j, so this call's
    # tokens must start exactly where the cache ends.
    if pos0 != kv.s:
        raise Error(
            String("llama_decoder_layer_forward: pos0=")
            + String(pos0)
            + " but the cache already holds "
            + String(kv.s)
            + " positions. Cache slot j IS absolute position j (DEVIATION"
            + " 1028); a mismatch gives a card that is internally"
            + " consistent and positionally wrong"
        )
    if kv.s + l > kv.s_max:
        raise Error(
            String("llama_decoder_layer_forward: the cache would grow to ")
            + String(kv.s + l)
            + " past its capacity "
            + String(kv.s_max)
        )
    if pos0 + l > rope.p_max:
        raise Error(
            String("llama_decoder_layer_forward: absolute position ")
            + String(pos0 + l - 1)
            + " is past the rotary table's p_max "
            + String(rope.p_max)
        )
    if pos0 + l > MAX_ABS_POSITION:
        raise Error(
            String("llama_decoder_layer_forward: absolute position ")
            + String(pos0 + l - 1)
            + " reaches the Cody-Waite ceiling "
            + String(MAX_ABS_POSITION)
            + " (contract section 3, DEVIATION 812)"
        )
    if rope.half != dims.half():
        raise Error(
            "llama_decoder_layer_forward: the rotary table was built for a"
            " different head_dim"
        )

    # Contract section 8, before ANY recorded stage.
    # DEVIATION 1875: the WEIGHT half of this refusal moved to
    # `LlamaDeviceWeights.__init__`, where the tensors are still on the host
    # and cost nothing to scan. What is left is what a call can actually
    # differ in. `llama_refuse_bad_inputs` is unchanged and still walks all
    # thirteen names for anyone who wants them in one call.
    llama_refuse_bad_call(ctx, w, rope, x, kv, b, l)

    trace.record_device[DType.float32](ctx, prefix + ".input.x", x, m * dm)

    # ---- residual = hidden_states (:305): a NAME, not a copy. `x` is read
    #      again at S22 and nothing writes it.
    # ---- self.input_layernorm(...) (:306) == LlamaRMSNorm.forward
    #      (:62-67). S1-S4.
    llama_rms_norm(
        ctx,
        stages.norm1_sumsq,
        stages.norm1_out,
        x,
        w.norm1_w,
        m,
        dm,
        w.eps,
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".norm1.sumsq", stages.norm1_sumsq, m
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".norm1.out", stages.norm1_out, m * dm
    )

    # ---- self.self_attn(...) (:308-316).
    llama_attention_forward(
        ctx,
        stages,
        kv,
        rope,
        w,
        b,
        l,
        pos0,
        plant_at,
        plant_idx,
        plant_bits,
        trace,
        prefix,
    )

    # ---- residual + hidden_states (:317). S22. The mamba lane's S16
    #      kernel, IMPORTED (contract section 0).
    ctx.enqueue_function[residual_add_kernel](
        stages.residual1.unsafe_ptr(),
        x.unsafe_ptr(),
        stages.o_proj.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".residual1.out", stages.residual1, m * dm
    )

    # ---- residual = hidden_states (:320), then
    #      self.post_attention_layernorm(...) (:321). S1-S4 again, the SAME
    #      kernel with the SAME eps and a different weight.
    llama_rms_norm(
        ctx,
        stages.norm2_sumsq,
        stages.norm2_out,
        stages.residual1,
        w.norm2_w,
        m,
        dm,
        w.eps,
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".norm2.sumsq", stages.norm2_sumsq, m
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".norm2.out", stages.norm2_out, m * dm
    )

    # ---- self.mlp(...) (:322). S5, S20, S21.
    llama_mlp_forward(ctx, stages, w, m, trace, prefix)

    # ---- residual + hidden_states (:323). S23, the same imported kernel.
    ctx.enqueue_function[residual_add_kernel](
        stages.residual2.unsafe_ptr(),
        stages.residual1.unsafe_ptr(),
        stages.down_proj.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".residual2.out", stages.residual2, m * dm
    )


def llama_decoder_layer_forward(
    ctx: DeviceContext,
    mut stages: LlamaDeviceStages,
    mut kv: LlamaKVCache,
    mut rope: LlamaRopeTable,
    mut w: LlamaDeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    pos0: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """THE ORDINARY ENTRY POINT. One block call with NO score plant.

    Every argument is a scalar or a buffer or one of this file's own
    structs; nothing from `transformer/checks/` crosses this boundary,
    on purpose (the oracle half of this lane was written concurrently by a
    different agent). The planted form is
    `llama_decoder_layer_forward_planted` and it is what the three
    bit-planting fixture cases need.
    """
    llama_decoder_layer_forward_planted(
        ctx,
        stages,
        kv,
        rope,
        w,
        x,
        b,
        l,
        pos0,
        PLANT_AT_NONE,
        List[Int](),
        List[UInt32](),
        trace,
        prefix,
    )
