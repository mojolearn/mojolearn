# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The DEVICE spelling of the backward pass of one Llama-shaped decoder
block, ONE SOURCE FOR THREE VENDORS, under profile
`mojolearn.identical.transformer.fp32.v1` and the routing document
`transformer/IDENTICAL_BACKWARD_PLAN.md`.

**NOTHING IN THIS FILE HAS BEEN COMPILED OR RUN.** Written 2026-08-25 by a
lane that is forbidden to execute anything. Every statement below about what
a kernel computes is CONSTRUCTION; every statement about what that would
mean across vendors is PREDICTION. **No sentence here says the backward is
bit identical across vendors, because no leg has run and no gate exists.**

WHAT IS OWED
--------------
  1. That it compiles. The RISK list is at the bottom of this docstring.
  2. `transformer/original/transformer_backward_check.mojo`, which DOES NOT
     EXIST. No gate, no fixture, no card, and **not one of the twenty
     sabotage arms below has ever been fired.** Twenty arms that have never
     been shown able to fail are twenty arms nobody has tested, which is the
     condition this project treats as unproven rather than as evidence.
  3. A FLOAT64 directional-derivative reference. Bit identity does not say
     the answer is the RIGHT derivative, and a transpose error is bit
     identical on three vendors.
  4. A three-vendor leg. Apple and AMD agreed bit for bit through 302 stages
     on the GEMM lane while NVIDIA diverged, so two backends agreeing closes
     nothing and zero backends agreeing closes less.

WHY THIS FILE IS IN `original/` AND NOT IN `derived/` (DEVIATION 1426)
-----------------------------------------------------------------------
A `derived/` path in this repository MIRRORS AN UPSTREAM PATH exactly, and
there is no upstream path to mirror: **HuggingFace ships no backward.**
`LlamaDecoderLayer` is forward-only Python and autograd GENERATES the
gradient from the elementwise graph, so there is no
`modeling_llama_backward.py` to place this beside. Putting it under
`derived/` would claim a correspondence that does not exist. Where the
reference's own derivative IS a named symbol -- `torch._softmax_backward_data`
and ATen's `silu_backward` -- this file says so and says that neither could
be read, because there is no PyTorch checkout.

WHAT THIS FILE DOES NOT REBUILD
---------------------------------
* **Every routed gradient** goes through
  `gemm/original/gemm_identical.mojo::identical_gemm` at a shape decided by
  `gemm/original/gemm_backward.mojo::gemm_backward_a_call` and
  `::gemm_backward_b_call`. **This file does not restate the six-row
  transpose table** (DEVIATION 1425): a lane that retypes it gets a wrong
  answer that is bit identical on three vendors, and calling them makes this
  lane's routed gradients inherit gemm gates G1 and G2 and inherit
  `SAB_BWD_UNTRANSPOSED` and `SAB_BWD_OPERAND_ORDER` as live arms through a
  new entry point.
* **Every transcendental** is `original/numerics.mojo`'s. No exp, no
  division, no reciprocal square root and no sigmoid is spelled here.
* **`pinned_mul` (DEVIATION 720)** is IMPORTED from the mamba lane, exactly
  as the forward device file imports it, rather than copied a fifth time.
* **The shape, weight, cache and stage types** are the forward device
  module's (DEVIATION 1427). The dependency arrow points backward-onto-
  forward, which is the right direction.

THE LANE'S OWN NEW ARITHMETIC (plan section 2)
------------------------------------------------
Six folds and one new spelling, and nothing else:

    bwd_softmax_zdot_kernel     the softmax's z fold, kv axis        NEW
    bwd_norm_dot_kernel         the RMSNorm backward's c fold        NEW
    bwd_dq_kernel               dq, key axis                         NEW
    bwd_dk_kernel               dk, query axis                       NEW
    bwd_dv_kernel               dv, query axis                       NEW
    bwd_silu_backward_kernel    silu', five roundings, no fold       NEW
    bwd_rope_kernel             the TRANSPOSED rotation              NEW SPELLING

Four of the folds are ONE shape: serial ascending over one axis, seeded
`+0.0`, `acc = ftz(fma(a, b, acc))`. That is contract S1's shape and
contract S19's shape and this file introduces no seventh.

WHY NO FLOAT CROSSES A THREAD BOUNDARY (DEVIATION 1422)
---------------------------------------------------------
Every kernel below owns its output cell entirely.

* The softmax `z` fold is ONE THREAD PER `(batch, head, query)`.
* The RMSNorm backward `c` fold is ONE THREAD PER TOKEN ROW.
* `dq` is ONE THREAD PER `(batch, query, head, depth)` walking the ABSOLUTE
  key index ascending.
* `dk` and `dv` are ONE THREAD PER `(batch, kv head, key slot, depth)`
  walking `(head in the kv group, query)` ascending -- **one chain, not a
  sum of per-head partials** (DEVIATION 1424).
* Everything else is one thread per output cell.

There is **no shared memory, no warp primitive, NO FLOAT ATOMIC, no
cross-block reduction and no `core/pinned_reduce.mojo` call anywhere in this
file.** That matters more here than it did in the forward, because
`gemm/IDENTICAL_BACKWARD_PLAN.md` section 4.5 named the attention backward
the one row with "a genuine structural obstacle", on the grounds that the
standard fused attention backward accumulates `dK` and `dV` across thread
blocks with float atomics. **That obstacle is gone, and not by cleverness:**
it is the direct consequence of refusing the fused shape. One thread per
output cell means no two threads ever write the same address, and with GQA
the whole head group is walked inside that one thread. The price is the
eager path's memory and the chains' latency, and plan section 7 prices both
rather than hiding them.

So contract clause (b) (the same bits on repeated launches) and clause (c)
(batch composition and length invariance of the activation gradients) are
properties of the SHAPE of these kernels rather than of a check that happens
to pass. **A block count is a summation order**, and no fold boundary,
accumulator seed or tap index below is a function of `block_dim`,
`block_idx` or the grid.

THE ONE PLACE THE FORWARD'S DECODE CLAUSE DOES NOT TRANSFER
--------------------------------------------------------------
`dq` is independent of the kv length, by the same masked-tail argument the
forward uses (plan section 5.1). `dk`, `dv` and every WEIGHT gradient are
SUMS OVER THE QUERIES IN THIS CALL and are therefore PARTIAL gradients: a
key at slot `j` is read by every query at absolute position `>= j`, and
queries in later calls are not in this call. Both are emitted complete over
`[0, S)` as a HANDOFF (DEVIATION 1417) and assembling a multi-call backward
is out of scope. **This file does not pretend that a decode-step backward
equals a prefill backward, and no clause anywhere in this lane says so.**

`[[mojo-buffer-freed-at-last-use]]`: every buffer this file hands to a
kernel is a FIELD of `LlamaBackwardStages` or of the forward's own structs,
whose lifetime is the struct's. No launcher below allocates a scratch buffer
and returns without waiting.

RISK: WHAT IS LEAST LIKELY TO COMPILE
---------------------------------------
  * `gemm_backward_a_call`'s FIVE-ELEMENT `Tuple` return and the `call[4]`
    indexing. The GEMM lane's own open questions flag it, and that file has
    never been compiled either. If it moves, `_route_a` and `_route_b` below
    are the only two call sites.
  * The per-head gather / `identical_gemm` / scatter loops. `identical_gemm`
    SYNCHRONIZES internally, and four buffers must stay alive across it.
  * **TWO OR MORE DISTINCT FIELDS OF ONE STRUCT PASSED AS SEPARATE `mut`
    ARGUMENTS.** `_route_a(ctx, bst.d_mlp_gated, bst.d_down_proj_out, ...)`
    and the twelve-argument `bwd_rms_norm` call are the shape in question,
    and this file is full of them because the alternative -- passing
    `mut bst` AND one of its fields at the same call site -- is a real alias
    that no borrow checker should accept. If the compiler refuses distinct
    fields too, the fix is a free function taking raw
    `MutPointer[Float32, MutAnyOrigin]`s, which is mechanical and touches
    only the launchers.
  * `mut` arguments that are struct fields at all, the shape that produces
    the `MutUntrackedOrigin` versus `MutAnyOrigin` unification errors this
    repository has lost time to.
  * The `comptime if` arms inside kernels. The forward device file uses the
    same construct throughout and it has not been compiled either.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.original.gemm_backward import (
    BWD_DC_LEFT,
    gemm_backward_a_call,
    gemm_backward_b_call,
)
from gemm.original.gemm_identical import identical_gemm
from gemm.original.gemm_oracle import OP_NN, OP_NT, OP_TN
from mamba.derived.transformers.models.mamba.modeling_mamba import pinned_mul
from original.numerics import (
    ftz,
    identical_div,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
)
from transformer.derived.transformers.models.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    llama_attention_scale,
)


# ===========================================================================
# THE TWO EXACT SCALINGS OF THE RMSNORM BACKWARD (DEVIATION 1409).
#
# Both are exact powers of two and neither introduces a rounding. Neither is
# BIT INERT, which is why both are SPELLED: contract S8 drops the
# `* attention_scaling` because it is exactly `1.0` and inert on every input
# including both zero signs, and that argument does not transfer to a value
# that changes the number.
# ===========================================================================

comptime BWD_NEG_HALF: Float32 = -0.5
comptime BWD_TWO: Float32 = 2.0

comptime BWD_STAGE_COUNT = 37


# ===========================================================================
# LAUNCH GEOMETRY. An EXECUTION plan quantity: it decides which thread owns
# a cell and nothing else. Declared here rather than imported from the
# forward module, so that a reader can see in one place that this file's
# geometry is its own and reaches no fold boundary.
# ===========================================================================

comptime BWD_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + BWD_TPB - 1) // BWD_TPB
    if g < 1:
        return 1
    return g


# ===========================================================================
# DEVICE I/O PLUMBING (DEVIATION 1427). Three helpers, restated here rather
# than imported from the forward module's underscore-prefixed versions,
# because reaching across a module boundary for a private symbol is a
# coupling nobody declared. The duplication is named as a cost: if a shared
# `core/` home for them ever appears, three lanes should move.
# ===========================================================================


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
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
    """The first `n` elements of a device buffer as a host list. NOTHING IN
    THIS FILE CALLS IT: it is here for the gate file that does not exist
    yet, which has to read stages back off the device to compare them
    against the host oracle and to count moved cells for the sabotage arms.
    If the gate file grows its own, delete this one."""
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


def _fill_ones(
    ctx: DeviceContext, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """`n` entries of exactly `Float32(1.0)`, for the two RMSNorm weight
    gradients (DEVIATION 1410).

    **A wrong value here is a wrong gradient with no symptom**, because any
    vector produces a plausible weighted column sum. The gate for it is not
    "the buffer was allocated"; it is the exact-integer arm of plan clause
    (f), which fails if the entries are anything else."""
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    dev.enqueue_fill(Float32(1.0))
    ctx.synchronize()
    return dev^


# ===========================================================================
# THE SABOTAGE ARMS. A clause nobody can falsify is not a clause.
#
# Plan section 6.3's twenty lane-owned arms, one `comptime` each, OFF in
# every build that does not name them. The other two arms of the
# twenty-two are the GEMM lane's own (`SAB_BWD_UNTRANSPOSED`,
# `SAB_BWD_OPERAND_ORDER`) and they reach this file through `_route_a` and
# `_route_b`.
#
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_TRANSFORMER_SABOTAGE_B11_DQ_VIA_GEMM=1 \
#         -I . transformer/original/transformer_backward_check.mojo
#
# The names are disjoint from the forward lane's thirteen and from the mamba
# lane's eleven, so one driver can arm any of them without collision.
#
# **FOUR OF THESE ARE PREDICTED INERT AT THE CURRENT GATE SHAPE AND TWO OF
# THE FOUR CANNOT BE MADE TO FIRE WITHOUT CHANGING THE FIXTURES.** Each one
# says so at its own definition, because the forward lane's scar is that an
# arm which looks inert gets deleted, and two of its thirteen only bite
# under clause (d).
# ===========================================================================

#: The RMSNorm backward's `c` fold spelled product-then-add instead of one
#: `fma` per term. INERT at `d_model == 1` and on exactly-representable rows.
comptime SAB_B01_DOT_UNFUSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B01_DOT_UNFUSED"
]()
#: The same fold walked DESCENDING. INERT at `d_model == 1` ONLY. **NOT
#: inert at `d_model == 2`**: an fma keeps the second product exact, so a
#: two-term fma chain is order dependent where a two-term ADD chain is not.
comptime SAB_B01_DOT_DESCENDING = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B01_DOT_DESCENDING"
]()
#: `rstd` recomputed by re-folding the sum of squares DESCENDING instead of
#: reading the saved `norm*.sumsq`. Its ASCENDING half must move NOTHING,
#: and that half is what turns "a recompute is bit exact" (DEVIATION 1420)
#: from a belief into a checked statement. INERT at `d_model == 1`.
comptime SAB_B_RSTD_RECOMPUTE_DESCENDING = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B_RSTD_RECOMPUTE_DESCENDING"
]()
#: The RoPE backward spelled with the FORWARD's sign convention, i.e. the
#: rotation run forwards instead of transposed. One character in two
#: branches, producing a plausible correctly-shaped wrong gradient.
#: **INERT AT ABSOLUTE POSITION 0**, where `sin` is exactly `+0.0` -- per
#: CELL, not per fixture, so the gate must COUNT moved cells and RAISE if
#: the count equals the position-0 population.
comptime SAB_B09_ROPE_TRANSPOSE_SIGN = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B09_ROPE_TRANSPOSE_SIGN"
]()
#: The rotate-half pairing replaced by the adjacent `(2i, 2i+1)` pairing --
#: the single most commonly mistranscribed thing about RoPE. INERT at
#: `head_dim == 2` and at position 0.
comptime SAB_B09_ROPE_HALVES_ADJACENT = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B09_ROPE_HALVES_ADJACENT"
]()
#: The two products and the add fused into one `fma`: one rounding where the
#: structure has three. INERT at position 0.
comptime SAB_B10_ROPE_BWD_FUSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B10_ROPE_BWD_FUSED"
]()
#: The attention-weight gradient spelled as a hand chain over `head_dim`
#: instead of the ROUTED gemm v1 cell. **THIS ARM CANNOT FIRE AT THE GATE
#: SHAPE AT ALL**: the two agree bit for bit whenever `P(head_dim) == 1`,
#: i.e. `head_dim <= 128`, and the profile's fixtures are 16 and 24. So
#: DEVIATION 1405 rests on plan section 2.1's argument and on gemm v1's own
#: certificate, NOT on a fired arm, and that is recorded here rather than
#: discovered later.
comptime SAB_B10_DW_VIA_CHAIN = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B10_DW_VIA_CHAIN"
]()
#: `dq` routed through gemm v1 `OP_NN` at `(l, hd, s)`, `k' = S`. **INERT AT
#: EVERY `S <= 128`**, where `P(S) == 1` and the gemm cell IS the whole-`k`
#: ascending chain from `+0.0`. Needs `S >= 129`; the `L = 257` fixture is
#: the only one that reaches it. Passes clause (a) at a fixed shape and
#: breaks the LENGTH clause, which is the shape contract section 10 warns
#: will look inert and get deleted.
comptime SAB_B11_DQ_VIA_GEMM = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B11_DQ_VIA_GEMM"
]()
#: `dk` routed through gemm v1 `OP_TN` at `(s, hd, l)`, `k' = L`, with the
#: head group accumulated as per-head partials. INERT only when `L <= 128`
#: AND `n_rep == 1` together; at `n_rep == 2` the partial accumulation is a
#: different association and it fires on shape alone.
comptime SAB_B11_DK_VIA_GEMM = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B11_DK_VIA_GEMM"
]()
#: `dv` routed the same way. The backward twin of the forward's
#: `S19_VALUE_SUM_VIA_GEMM`, inheriting its warning verbatim.
comptime SAB_B19_DV_VIA_GEMM = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B19_DV_VIA_GEMM"
]()
#: The attention scale folded into the `dq` chain instead of applied to the
#: finished `dS`. **INERT AT EVERY POWER-OF-FOUR `head_dim`**: at 16 and 64
#: the scale is exactly `0.25` / `0.125`, exact scaling commutes with an fma
#: chain bitwise, and moving it to the far side changes nothing. The
#: `head_dim = 24` fixture contract section 3 already requires -- for a
#: different reason -- is what makes this arm fire at all.
comptime SAB_B12_SCALE_INTO_DQ = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B12_SCALE_INTO_DQ"
]()
#: The mask backward forced to write `+0.0` at masked cells instead of
#: passing the gradient through. Moves ONLY masked cells whose `dS` is
#: `-0.0`, i.e. where `dy_j - z < 0`: predicted to move roughly half the
#: masked cells, and **the oracle must predict the exact count before the
#: device is asked**.
comptime SAB_B13_MASK_ZEROES_GRAD = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B13_MASK_ZEROES_GRAD"
]()
#: The softmax backward spelled as the DECOMPOSED autograd graph, including
#: the analytically-zero max gradient scattered through an argmax. INERT at
#: `s == 1`, where both forms give zero.
comptime SAB_B18_SOFTMAX_DECOMPOSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B18_SOFTMAX_DECOMPOSED"
]()
#: The `z` fold spelled product-then-add. INERT at `s == 1`.
comptime SAB_B18_ZFOLD_UNFUSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B18_ZFOLD_UNFUSED"
]()
#: The `z` fold walked DESCENDING over the key axis. INERT at `s == 1` only.
comptime SAB_B18_ZFOLD_DESCENDING = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B18_ZFOLD_DESCENDING"
]()
#: `sigmoid` reconstructed as `silu_out / gate_proj_out` instead of
#: recomputed. **INERT for every gate activation at or above about 17**,
#: where `1 + exp(-x)` rounds to exactly `1.0` and the reconstruction
#: returns exactly `1.0`. A fixture whose gate activations are all large is
#: blind to it, and it produces `0/0` at a zero gate.
comptime SAB_B20_SIGMOID_FROM_SILU = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B20_SIGMOID_FROM_SILU"
]()
#: `sg + x*sg*(1-sg)` instead of `sg*(1 + x*(1-sg))`. INERT wherever `sg` is
#: exactly `1.0` and at `x == 0`.
comptime SAB_B20_SILU_DERIV_ALT_ASSOC = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B20_SILU_DERIV_ALT_ASSOC"
]()
#: `fma(x, 1-sg, 1.0)`: one rounding where the pinned form has two. INERT
#: wherever `x * (1 - sg)` is exactly representable.
comptime SAB_B20_SILU_DERIV_FUSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B20_SILU_DERIV_FUSED"
]()
#: The fan-in accumulations seeded with `+0.0`. **PREDICTED TO MOVE ZERO
#: CELLS ON EVERY UNPLANTED FIXTURE**: it fires only where the FIRST
#: accumulated term is `-0.0`, which needs a planted `-0.0`. **Vacuous
#: without a plant**, said out loud so that nobody fires it, sees nothing
#: and deletes it.
comptime SAB_B_FANIN_ZERO_SEED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B_FANIN_ZERO_SEED"
]()
#: The three-term fan-in at `d(norm1.out)` accumulated `v, k, q` instead of
#: `q, k, v`. INERT wherever any two of the three terms are zero.
comptime SAB_B_FANIN_ORDER_QKV_REVERSED = is_defined[
    "MOJOLEARN_TRANSFORMER_SABOTAGE_B_FANIN_ORDER_QKV_REVERSED"
]()

comptime BWD_ANY_SABOTAGE = (
    SAB_B01_DOT_UNFUSED
    or SAB_B01_DOT_DESCENDING
    or SAB_B_RSTD_RECOMPUTE_DESCENDING
    or SAB_B09_ROPE_TRANSPOSE_SIGN
    or SAB_B09_ROPE_HALVES_ADJACENT
    or SAB_B10_ROPE_BWD_FUSED
    or SAB_B10_DW_VIA_CHAIN
    or SAB_B11_DQ_VIA_GEMM
    or SAB_B11_DK_VIA_GEMM
    or SAB_B19_DV_VIA_GEMM
    or SAB_B12_SCALE_INTO_DQ
    or SAB_B13_MASK_ZEROES_GRAD
    or SAB_B18_SOFTMAX_DECOMPOSED
    or SAB_B18_ZFOLD_UNFUSED
    or SAB_B18_ZFOLD_DESCENDING
    or SAB_B20_SIGMOID_FROM_SILU
    or SAB_B20_SILU_DERIV_ALT_ASSOC
    or SAB_B20_SILU_DERIV_FUSED
    or SAB_B_FANIN_ZERO_SEED
    or SAB_B_FANIN_ORDER_QKV_REVERSED
)


def llama_backward_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner.

    A check must print this AND `llama_block_sabotage_name()` AND
    `gemm_sabotage_name()` AND `gemm_backward_sabotage_name()`, because a
    backward binary can carry a forward arm, a gemm arm and a gemm-backward
    arm as well, and a banner naming only one of the four MISLABELS THE RUN.
    """
    comptime if SAB_B01_DOT_UNFUSED:
        return String("B01_DOT_UNFUSED")
    comptime if SAB_B01_DOT_DESCENDING:
        return String("B01_DOT_DESCENDING")
    comptime if SAB_B_RSTD_RECOMPUTE_DESCENDING:
        return String("B_RSTD_RECOMPUTE_DESCENDING")
    comptime if SAB_B09_ROPE_TRANSPOSE_SIGN:
        return String("B09_ROPE_TRANSPOSE_SIGN")
    comptime if SAB_B09_ROPE_HALVES_ADJACENT:
        return String("B09_ROPE_HALVES_ADJACENT")
    comptime if SAB_B10_ROPE_BWD_FUSED:
        return String("B10_ROPE_BWD_FUSED")
    comptime if SAB_B10_DW_VIA_CHAIN:
        return String("B10_DW_VIA_CHAIN")
    comptime if SAB_B11_DQ_VIA_GEMM:
        return String("B11_DQ_VIA_GEMM")
    comptime if SAB_B11_DK_VIA_GEMM:
        return String("B11_DK_VIA_GEMM")
    comptime if SAB_B19_DV_VIA_GEMM:
        return String("B19_DV_VIA_GEMM")
    comptime if SAB_B12_SCALE_INTO_DQ:
        return String("B12_SCALE_INTO_DQ")
    comptime if SAB_B13_MASK_ZEROES_GRAD:
        return String("B13_MASK_ZEROES_GRAD")
    comptime if SAB_B18_SOFTMAX_DECOMPOSED:
        return String("B18_SOFTMAX_DECOMPOSED")
    comptime if SAB_B18_ZFOLD_UNFUSED:
        return String("B18_ZFOLD_UNFUSED")
    comptime if SAB_B18_ZFOLD_DESCENDING:
        return String("B18_ZFOLD_DESCENDING")
    comptime if SAB_B20_SIGMOID_FROM_SILU:
        return String("B20_SIGMOID_FROM_SILU")
    comptime if SAB_B20_SILU_DERIV_ALT_ASSOC:
        return String("B20_SILU_DERIV_ALT_ASSOC")
    comptime if SAB_B20_SILU_DERIV_FUSED:
        return String("B20_SILU_DERIV_FUSED")
    comptime if SAB_B_FANIN_ZERO_SEED:
        return String("B_FANIN_ZERO_SEED")
    comptime if SAB_B_FANIN_ORDER_QKV_REVERSED:
        return String("B_FANIN_ORDER_QKV_REVERSED")
    return String("none")


# ===========================================================================
# THE ELEMENTWISE PLUMBING KERNELS. None of these is a seam of its own; each
# spells exactly one rounding (or none) that the plan's table names.
# ===========================================================================


def bwd_copy_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """A bit copy. **NO ROUNDING, NO `ftz`.**

    Used at the three places the derivative is the exact identity: S23's two
    branches (an add's derivative is 1 in both arguments), S22's `d o_proj`
    branch, and S13's mask backward (the mask value is a CONSTANT, so
    `d(masked)/d(scores)` is exactly 1 at every cell INCLUDING the masked
    ones). A copy that also flushed would be a seam nobody reviewed as one.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    dst.unsafe_store(i, src.unsafe_load(i))


def bwd_mask_grad_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    l_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
):
    """S13's backward as its own kernel so that `B13_MASK_ZEROES_GRAD` has a
    body. In a clean build this is `bwd_copy_kernel` with a causal test it
    does not use. DEVIATION 1414.

    **THE MASKED CELLS ARE NOT ZEROED, AND THAT IS A DECISION.** They are
    already SIGNED zeros: `dS_j = pinned_mul(y_j, dy_j - z)` with `y_j`
    exactly `+0.0` carries the sign of `dy_j - z`. Forcing `+0.0` therefore
    differs exactly where that sign is negative, which is roughly half the
    masked cells and **is reachable without a plant** -- so unlike
    `B_FANIN_ZERO_SEED`, this arm is not vacuous on an ordinary fixture, and
    the oracle can predict its count.
    """
    var b = Int(b_in)
    var nh = Int(nh_in)
    var l = Int(l_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * nh * l * s:
        return
    comptime if SAB_B13_MASK_ZEROES_GRAD:
        var j = i % s
        var rest = i // s
        var t = rest % l
        if j > pos0 + t:
            # SABOTAGE: a positive zero where the profile passes a signed one.
            dst.unsafe_store(i, Float32(0.0))
            return
    dst.unsafe_store(i, src.unsafe_load(i))


def bwd_mul_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    bb: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """One `pinned_mul` per cell: `dst = a * b`, ONE rounding, and no codegen
    may contract it into a neighboring add (DEVIATION 720).

    S21's two products and the RMSNorm weight gradient's `dprod` are both
    this kernel. ROUTING under the plan's definitions: `pinned_mul` settles
    the fusion question by existing, so this lane chooses nothing here."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    dst.unsafe_store(
        i, ftz(pinned_mul(ftz(a.unsafe_load(i)), ftz(bb.unsafe_load(i))))
    )


def bwd_scale_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale_in: Float32,
):
    """S12's backward, DEVIATION 1415: `d_cell = d_scores * scale`, ONE
    `pinned_mul` per score applied to the FINISHED gradient.

    Folding the scale into the `dq` chain instead is `B12_SCALE_INTO_DQ`,
    which skips this kernel entirely. Read that arm's note at its
    definition: it is INERT at every power-of-four `head_dim`, so the
    `head_dim = 24` fixture is what makes it fire."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    comptime if SAB_B12_SCALE_INTO_DQ:
        # SABOTAGE: the scale moves into the dq chain, so this seam is
        # skipped and the softmax's own gradient stands.
        dst.unsafe_store(i, src.unsafe_load(i))
        return
    dst.unsafe_store(i, ftz(pinned_mul(ftz(src.unsafe_load(i)), scale_in)))


def bwd_add2_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    bb: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """A TWO-TERM fan-in, `ftz(ftz(a) + ftz(b))`, UNFUSED.

    **NO ORDER TO PIN**: IEEE addition is bitwise commutative on every
    non-NaN input and NaN is refused at the door. The clause is written down
    anyway, because "it does not matter" is exactly the kind of sentence
    that turns out to matter.

    **NO `+0.0` SEED** (DEVIATION 1413). `+0.0 + x` equals `x` for every
    value except `x = -0.0`, where it gives `+0.0` -- IDENTITY_PATHS row 39
    in one line -- so a seed LAUNDERS a negative-zero first term, and a
    negative zero is reachable in a gradient: every masked attention cell
    produces one. An autograd engine's `AccumulateGrad` installs the first
    incoming gradient as the buffer rather than allocating zeros, and this
    follows it."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    comptime if SAB_B_FANIN_ZERO_SEED:
        # SABOTAGE: the seed that launders a negative zero. Predicted to
        # move ZERO cells unless a `-0.0` is PLANTED into the first term.
        var z = ftz(Float32(0.0) + ftz(a.unsafe_load(i)))
        dst.unsafe_store(i, ftz(ftz(z) + ftz(bb.unsafe_load(i))))
        return
    dst.unsafe_store(
        i, ftz(ftz(a.unsafe_load(i)) + ftz(bb.unsafe_load(i)))
    )


def bwd_add3_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    bb: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """THE ONE THREE-TERM FAN-IN IN THIS BLOCK, at `d(norm1.out)`, and it IS
    order dependent because `(a+b)+c` and `a+(b+c)` are different numbers.

    Pinned FORWARD-USE order, `q` then `k` then `v`
    (`LlamaAttention.forward` :250, :251, :252), left associative, no seed.
    Sabotage `B_FANIN_ORDER_QKV_REVERSED`, INERT wherever any two of the
    three terms are zero."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var va = ftz(a.unsafe_load(i))
    var vb = ftz(bb.unsafe_load(i))
    var vc = ftz(c.unsafe_load(i))
    comptime if SAB_B_FANIN_ORDER_QKV_REVERSED:
        # SABOTAGE: v, k, q.
        var acc_s = ftz(ftz(vc) + ftz(vb))
        dst.unsafe_store(i, ftz(ftz(acc_s) + ftz(va)))
        return
    comptime if SAB_B_FANIN_ZERO_SEED:
        var z = ftz(Float32(0.0) + va)
        var acc_z = ftz(ftz(z) + ftz(vb))
        dst.unsafe_store(i, ftz(ftz(acc_z) + ftz(vc)))
        return
    var acc = ftz(ftz(va) + ftz(vb))
    dst.unsafe_store(i, ftz(ftz(acc) + ftz(vc)))


# ===========================================================================
# S20's BACKWARD. DEVIATION 1411.
# ===========================================================================


def bwd_silu_backward_kernel(
    dg: MutPointer[Float32, MutAnyOrigin],
    dsi: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    silu_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`d g = d(silu) * silu'(g)`, one thread per cell, five roundings after
    the sigmoid, left to right, no fma.

        sg = identical_sigmoid(x)         DEVIATION 743, portable_sigmoidf
        r1 = ftz(1.0 - sg)                SUBTRACT
        r2 = pinned_mul(x, r1)            PRODUCT
        r3 = ftz(1.0 + r2)                UNFUSED ADD
        r4 = pinned_mul(sg, r3)           PRODUCT
        dg = pinned_mul(dsi, r4)          PRODUCT

    **`sg` IS RECOMPUTED FROM `gate_proj.out`, NEVER RECONSTRUCTED FROM
    `silu.out`.** `silu(x) = x * sigmoid(x)` is true in the reals and FALSE
    in Float32 under this profile: contract S20's forward is ONE division
    and `x * sg` is a division followed by a product. So `silu_out / x` is a
    different number, it is `0/0` at `x = 0`, and it costs a division to get
    a worse answer. `portable_sigmoidf` and `portable_siluf` share the SAME
    `d = portable_expf(-x) + 1.0` and differ only in the numerator, so the
    recomputed sigmoid is exactly `1/d` against the forward's `x/d`.

    `silu_out` is passed in ONLY so `B20_SIGMOID_FROM_SILU` has a body. The
    profile never reads it, which is the same relationship the forward's
    `attn_denom_kernel` has with its `scratch` argument (DEVIATION 1023).

    **THE REFERENCE'S OWN SPELLING COULD NOT BE VERIFIED.** There is no
    PyTorch checkout in `/Users/andrewhendel/CascadeProjects/upstream/` --
    verified again on 2026-08-25 -- so ATen's `silu_backward` was not read.
    This is a STATED GAP, not a decision made on evidence, it is contract
    section 5.4's gap in a second place, and it is the second thing to check
    when a checkout lands."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var x = ftz(g.unsafe_load(i))

    var sg: Float32
    comptime if SAB_B20_SIGMOID_FROM_SILU:
        # SABOTAGE: sigmoid reconstructed from the forward's SiLU output.
        # INERT for every x at or above about 17, where 1 + exp(-x) rounds
        # to exactly 1.0 and this returns exactly 1.0.
        sg = ftz(identical_div(ftz(silu_out.unsafe_load(i)), x))
    else:
        sg = ftz(identical_sigmoid(x))

    var r1 = ftz(ftz(Float32(1.0)) - ftz(sg))

    comptime if SAB_B20_SILU_DERIV_ALT_ASSOC:
        # SABOTAGE: sg + x*sg*(1-sg). Equal in the reals, different in the
        # last bit, five operations instead of four.
        var p1 = ftz(pinned_mul(x, sg))
        var p2 = ftz(pinned_mul(p1, r1))
        var d_alt = ftz(ftz(sg) + ftz(p2))
        dg.unsafe_store(
            i, ftz(pinned_mul(ftz(dsi.unsafe_load(i)), d_alt))
        )
        return

    var r3: Float32
    comptime if SAB_B20_SILU_DERIV_FUSED:
        # SABOTAGE: one rounding where the pinned form has two.
        r3 = ftz(identical_mul_add(x, r1, Float32(1.0)))
    else:
        var r2 = ftz(pinned_mul(x, r1))
        r3 = ftz(ftz(Float32(1.0)) + ftz(r2))

    var r4 = ftz(pinned_mul(sg, r3))
    dg.unsafe_store(i, ftz(pinned_mul(ftz(dsi.unsafe_load(i)), r4)))


# ===========================================================================
# S1-S4's BACKWARD. DEVIATIONS 1408, 1409, 1410, 1420.
#
# Three kernels, because the row-level coefficients (`rstd`, `dv`) must be
# finished before any cell of `dx` can be written, and because splitting
# them keeps `bwd.norm*.dot` separately recordable. A fold whose only
# evidence is the value it feeds cannot be localized, and the forward lane's
# scar is that thirteen moved stages were absorbed by a residual add while
# an output-only gate called the sabotage inert.
# ===========================================================================


def bwd_norm_dh_kernel(
    dh: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
):
    """`dh_j = pinned_mul(dy_j, w_j)`, the `y = w * h` node's backward
    (LRN:67). One thread per cell, one rounding, ROUTING."""
    var m = Int(m_in)
    var dm = Int(dm_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * dm:
        return
    var j = i - (i // dm) * dm
    dh.unsafe_store(
        i,
        ftz(
            pinned_mul(ftz(dy.unsafe_load(i)), ftz(weight.unsafe_load(j)))
        ),
    )


def bwd_norm_dot_kernel(
    dot: MutPointer[Float32, MutAnyOrigin],
    rstd_out: MutPointer[Float32, MutAnyOrigin],
    dv_out: MutPointer[Float32, MutAnyOrigin],
    dh: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
    eps_in: Float32,
):
    """The `c` fold and the row coefficients. ONE THREAD PER TOKEN ROW, so
    the row's fold never leaves this thread's registers and no launch
    geometry can reorder it.

        c    = ftz(fma(dh_j, x_j, c)), ASCENDING j from +0.0     FUSED
        rstd = ftz(rsqrt(ftz(div(sumsq, dm) + eps)))             RECOMPUTED
        r2   = pinned_mul(rstd, rstd)
        r3   = pinned_mul(r2, rstd)
        cr3  = pinned_mul(c, r3)
        da   = pinned_mul(-0.5, cr3)     rsqrt backward, -0.5 * dr * r^3
        dv   = identical_div(da, dm)     mean backward, ONE division

    **THE `c` FOLD IS CONTRACT S1's SHAPE UNCHANGED** (DEVIATION 1408).
    `d_model` is a CONFIGURATION quantity, so unlike the softmax's `z` fold
    or the `dq` chain there is no path-invariance argument forcing this; the
    reasons are that it gives the block ONE fold shape instead of two, that
    one thread owns one token row, and that a reader can put contract S1
    beside it. `core/pinned_reduce.mojo::pinned_block_sum` is REFUSED for
    the reasons contract 5.3 refuses it at S17: it is a halving tree, it
    pairs by STRIDE, and it is a different sum that would pass every
    launch-invariance gate because a tree is perfectly launch invariant.

    **`rstd` IS RECOMPUTED FROM THE SAVED `sumsq`** (DEVIATION 1420).
    `identical_div` and `identical_rsqrt` are pure functions of bits the
    card already holds, so the recompute is bit exact BY CONSTRUCTION.
    Sabotage `B_RSTD_RECOMPUTE_DESCENDING` re-folds the sum of squares
    instead of reading it; its ASCENDING half must move NOTHING and that
    half is what turns "a recompute is exact" from a belief into a checked
    statement.

    **`-0.5` AND `2.0` ARE SPELLED, NOT ELIDED.** Both are exact scalings by
    a power of two and neither is BIT INERT, so unlike contract S8's
    `* attention_scaling` (exactly `1.0`, inert on every input including
    both zero signs) they change the number and must appear."""
    var m = Int(m_in)
    var dm = Int(dm_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return

    # ---- the c fold ---------------------------------------------------
    var c = Float32(0.0)
    comptime if SAB_B01_DOT_DESCENDING:
        for jj in range(dm):
            var jd = dm - 1 - jj
            c = ftz(
                identical_mul_add(
                    ftz(dh.unsafe_load(t * dm + jd)),
                    ftz(x.unsafe_load(t * dm + jd)),
                    c,
                )
            )
    elif SAB_B01_DOT_UNFUSED:
        for j in range(dm):
            var p = ftz(
                pinned_mul(
                    ftz(dh.unsafe_load(t * dm + j)),
                    ftz(x.unsafe_load(t * dm + j)),
                )
            )
            c = ftz(ftz(c) + ftz(p))
    else:
        for j in range(dm):
            c = ftz(
                identical_mul_add(
                    ftz(dh.unsafe_load(t * dm + j)),
                    ftz(x.unsafe_load(t * dm + j)),
                    c,
                )
            )
    c = ftz(c)
    dot.unsafe_store(t, c)

    # ---- rstd, recomputed ---------------------------------------------
    var ss: Float32
    comptime if SAB_B_RSTD_RECOMPUTE_DESCENDING:
        # SABOTAGE: the sum of squares re-folded DESCENDING rather than the
        # saved `sumsq` read. INERT at d_model == 1.
        ss = Float32(0.0)
        for jj2 in range(dm):
            var jd2 = dm - 1 - jj2
            var xd = ftz(x.unsafe_load(t * dm + jd2))
            ss = ftz(identical_mul_add(xd, xd, ss))
    else:
        ss = ftz(sumsq.unsafe_load(t))
    var mean = ftz(identical_div(ss, Float32(dm)))
    var rstd = ftz(identical_rsqrt(ftz(mean + eps_in)))
    rstd_out.unsafe_store(t, rstd)

    # ---- the rsqrt and mean nodes -------------------------------------
    var r2 = ftz(pinned_mul(rstd, rstd))
    var r3 = ftz(pinned_mul(r2, rstd))
    var cr3 = ftz(pinned_mul(c, r3))
    var da = ftz(pinned_mul(BWD_NEG_HALF, cr3))
    dv_out.unsafe_store(t, ftz(identical_div(da, Float32(dm))))


def bwd_norm_dx_kernel(
    dx: MutPointer[Float32, MutAnyOrigin],
    dprod: MutPointer[Float32, MutAnyOrigin],
    dh: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rstd_in: MutPointer[Float32, MutAnyOrigin],
    dv_in: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
):
    """The two `x` branches and the weight-gradient product. One thread per
    cell.

        dx1_j = pinned_mul(dh_j, rstd)     h = x*r, the x branch
        tx_j  = pinned_mul(2.0, x_j)       pow(2) backward's 2*x
        dx2_j = pinned_mul(dv, tx_j)
        dx_j  = ftz(ftz(dx1_j) + ftz(dx2_j))               UNFUSED ADD

        inner_j = pinned_mul(x_j, rstd)    a recompute of forward S3
        dprod_j = pinned_mul(dy_j, inner_j)

    `dprod` exists so the weight gradient can be a GEMM (DEVIATION 1410).
    `dW[j] = sum_t dy_tj * inner_tj` is a Hadamard then a reduce, so the
    obvious reading is that it needs a new pinned fold. It does not, for the
    reason `gemm/IDENTICAL_BACKWARD_PLAN.md` section 3.3 gives about `db`:
    with the product materialized, the reduction is `ones . dprod`, an
    `OP_NN` at `(1, dm, M)` whose leaf is `fma(ftz(1.0), ftz(p), acc)` --
    the contract's ascending flushed chain inside a leaf and the contract's
    balanced tree across leaves, with nothing new to certify.

    **THE COST THIS LANE PAYS THAT THE GEMM LANE DID NOT**: two roundings
    per term instead of one. `db[j] = sum dC[i,j]` had no product to round;
    this does, so the routed form rounds the product and THEN rounds the add
    where a hand-written `fma(dy, inner, acc)` would round once. Pinned
    anyway, because one arithmetic under one certificate beats one rounding
    per term, and because the hand form would be a SIXTH new fold in a lane
    that has five. Recorded, not hidden."""
    var m = Int(m_in)
    var dm = Int(dm_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * dm:
        return
    var t = i // dm
    var xj = ftz(x.unsafe_load(i))
    var rstd = ftz(rstd_in.unsafe_load(t))
    var dv = ftz(dv_in.unsafe_load(t))

    var dx1 = ftz(pinned_mul(ftz(dh.unsafe_load(i)), rstd))
    var tx = ftz(pinned_mul(BWD_TWO, xj))
    var dx2 = ftz(pinned_mul(dv, tx))
    dx.unsafe_store(i, ftz(ftz(dx1) + ftz(dx2)))

    var inner = ftz(pinned_mul(xj, rstd))
    dprod.unsafe_store(
        i, ftz(pinned_mul(ftz(dy.unsafe_load(i)), inner))
    )


# ===========================================================================
# S9 AND S10's BACKWARD: THE TRANSPOSED ROTATION. DEVIATION 1412.
# ===========================================================================


def bwd_rope_kernel(
    out_buf: MutPointer[Float32, MutAnyOrigin],
    dout: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    hd_in: Int32,
    pos0_in: Int32,
):
    """One thread per output cell of `[M, n_h*head_dim]`, token-major. ONE
    kernel serves q and k, because upstream applies one function to both
    (:158-159) and a second spelling would be a second place to drift.

    The forward acts on the pair `(a_i, a_{i+half})` at table column
    `ci = i` as a rotation `M = [[c, -s], [s, c]]`. `M` is orthogonal, so
    the exact derivative is `M^T = [[c, s], [-s, c]]`, which written in the
    forward's own `rotate_half` idiom is THE SAME CODE WITH THE NEGATION
    MOVED FROM THE LOWER HALF TO THE UPPER HALF:

        forward   j <  half: rot = -x[j+half]   j >= half: rot = +x[j-half]
        backward  j <  half: rot = +d[j+half]   j >= half: rot = -d[j-half]

    **EVERYTHING ELSE IS IDENTICAL.** Same `ci = j mod half` column, same
    `cos` and `sin` rows at the same ABSOLUTE position, two `pinned_mul`
    calls, one UNFUSED add. Same three roundings, and contract DEVIATION
    811's refusal of an fma applies for the reason it applied there.

    So the rounding BUDGET is inherited and the SPELLING is new, and the one
    thing this lane pins is the SIGN CONVENTION -- one character in two
    branches, producing a plausible correctly-shaped wrong gradient that is
    bit identical on three vendors. Sabotage `B09_ROPE_TRANSPOSE_SIGN`,
    **INERT AT ABSOLUTE POSITION 0** where `sin` is exactly `+0.0`.

    THE POSITION IS THE ABSOLUTE POSITION, ALWAYS, for the same reason the
    forward's is. THE NEGATION IS EXACT: `-x` flips one bit and is not a
    seam, so it carries no flush of its own.

    **ONE SENTENCE SO NOBODY EXPECTS THE WRONG THING.** This is the exact
    transpose of the EXACT rotation, spelled with the forward's rounding
    budget. It is NOT the numerical adjoint of the ROUNDED forward map,
    because a rounded map has no adjoint. `bwd_rope(rope(x))` does not
    return `x` and no clause in this lane says it does."""
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

    var f = d
    if f >= half:
        f = f - half
    var cos_v = ftz(cos_tab.unsafe_load(pos * half + f))
    var sin_v = ftz(sin_tab.unsafe_load(pos * half + f))
    var dj = ftz(dout.unsafe_load(i))

    var rh: Float32
    comptime if SAB_B09_ROPE_HALVES_ADJACENT:
        # SABOTAGE: the (2i, 2i+1) pairing several kernels and the original
        # RoFormer paper use, where HuggingFace's checkpoint layout makes
        # the HALVES spelling the right one.
        var e = d - (d // 2) * 2
        if e == 0:
            rh = ftz(dout.unsafe_load(i + 1))
        else:
            rh = -ftz(dout.unsafe_load(i - 1))
    elif SAB_B09_ROPE_TRANSPOSE_SIGN:
        # SABOTAGE: the FORWARD's sign convention, i.e. the rotation run
        # forwards where the derivative needs it transposed.
        if d < half:
            rh = -ftz(dout.unsafe_load(i + half))
        else:
            rh = ftz(dout.unsafe_load(i - half))
    else:
        if d < half:
            rh = ftz(dout.unsafe_load(i + half))
        else:
            rh = -ftz(dout.unsafe_load(i - half))

    var pa = ftz(pinned_mul(dj, cos_v))
    comptime if SAB_B10_ROPE_BWD_FUSED:
        out_buf.unsafe_store(i, ftz(identical_mul_add(rh, sin_v, pa)))
    else:
        var pb = ftz(pinned_mul(rh, sin_v))
        out_buf.unsafe_store(i, ftz(ftz(pa) + ftz(pb)))


# ===========================================================================
# S14-S18's BACKWARD, AS ONE CLOSED FORM. DEVIATIONS 1406, 1407.
# ===========================================================================


def bwd_softmax_zdot_kernel(
    zdot: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    l_in: Int32,
    s_in: Int32,
):
    """`z = sum_j dy_j * y_j`, a SERIAL ASCENDING CHAIN over the ABSOLUTE
    key index from `+0.0`, FUSED. One thread per `(batch, head, query)`.

    **THE `z` FOLD FOLLOWS CONTRACT 5.3's ARGUMENT AND IT CARRIES, BUT IT
    NEEDED CHECKING RATHER THAN ASSUMING, BECAUSE THE TERMS ARE DIFFERENT
    TERMS.** At a masked cell `y_j` is exactly `+0.0` (contract 7.1) while
    `dy_j` is an ordinary nonzero number, so the term is
    `fma(dy_j, +0.0, acc) = acc + (+-0.0) = acc`, provided `acc` is not
    `-0.0`. A `+0.0`-seeded fma chain never holds `-0.0`: `fma` returns a
    negative zero only when the exact `a*b + acc` is a zero of NEGATIVE
    SIGN, which under round-to-nearest needs BOTH addends to be `-0.0` (an
    exact cancellation of two nonzero opposites gives `+0.0`), and the seed
    forbids it. **So the masked tail is bitwise inert and `z` -- and every
    activation gradient downstream of it -- is independent of the kv
    length.** That is the theorem the backward's length clause rests on and
    it is contract 7.1 pointed the other way.

    **`core/pinned_reduce.mojo::pinned_block_sum` MAY NOT BE USED**, for the
    third time in this profile and for the same reason: it is a halving
    tree, it is perfectly launch invariant, and it is simply a DIFFERENT
    SUM. Reaching for the deterministic block fold BECAUSE it is the
    deterministic block fold is the single most likely way to get this
    wrong.

    FUSED, matching contract S1 and S19, because it is a sum of products.

    Sabotages `B18_ZFOLD_UNFUSED` (INERT at `s == 1`) and
    `B18_ZFOLD_DESCENDING` (INERT at `s == 1` ONLY -- **not at `s == 2`**,
    because an fma keeps the second product exact, so a two-term fma chain
    is order dependent where a two-term ADD chain is not)."""
    var b = Int(b_in)
    var nh = Int(nh_in)
    var l = Int(l_in)
    var s = Int(s_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= b * nh * l:
        return
    var base = r * s
    var z = Float32(0.0)
    comptime if SAB_B18_ZFOLD_DESCENDING:
        for jj in range(s):
            var jd = s - 1 - jj
            z = ftz(
                identical_mul_add(
                    ftz(dy.unsafe_load(base + jd)),
                    ftz(y.unsafe_load(base + jd)),
                    z,
                )
            )
    elif SAB_B18_ZFOLD_UNFUSED:
        for j in range(s):
            var p = ftz(
                pinned_mul(
                    ftz(dy.unsafe_load(base + j)),
                    ftz(y.unsafe_load(base + j)),
                )
            )
            z = ftz(ftz(z) + ftz(p))
    else:
        for j in range(s):
            z = ftz(
                identical_mul_add(
                    ftz(dy.unsafe_load(base + j)),
                    ftz(y.unsafe_load(base + j)),
                    z,
                )
            )
    zdot.unsafe_store(r, ftz(z))


def bwd_softmax_ds_kernel(
    ds: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    s_in: Int32,
):
    """`dS_j = pinned_mul(y_j, ftz(dy_j - z))`. One SUBTRACT, one PRODUCT,
    UNFUSED. One thread per cell.

    **THERE IS NO MAX BACKWARD, NO EXP BACKWARD AND NO DIVISION BACKWARD,
    AND REFUSING TO WRITE THEM IS A DECISION WITH A NAME** (DEVIATION 1406).

    `softmax` is ONE autograd node in the reference, not a graph, and the
    evidence is in the checkout:
    `transformers/src/transformers/pytorch_utils.py:50-58` defines
    `softmax_backward_data(parent, grad_output, output)` whose body is
    `from torch import _softmax_backward_data; return
    _softmax_backward_data(grad_output, output, parent.dim, output.dtype)`.
    A private ATen symbol taking `(grad_output, output, dim, dtype)` and NOT
    taking the input, the max, the exponentials or the denominator is a
    closed-form derivative of the whole op.

    **THE DECOMPOSED GRAPH WOULD BE A DIFFERENT ANSWER, AND WORSE IN A
    SPECIFIC WAY.** `y = softmax(s)` is invariant to the row maximum `m`, so
    `dL/dm` is analytically EXACTLY zero -- and autograd does not know that.
    It computes a sum of terms that cancel in the reals and do NOT cancel in
    Float32, then SCATTERS that residue onto whichever element the max
    selected. That scatter is unpinnable here:

      1. `identical_fmax` returns a VALUE, not an INDEX. Contract 5.1 pins
         the max as an order-free fold and leaves its TOPOLOGY free because
         the operation is exactly associative; an argmax is NOT associative
         and its answer under ties depends on the fold shape, so a free
         topology and a defined argmax are incompatible.
      2. Ties are reachable: a masked row's tail is a run of identical
         `-FLT_MAX` values.
      3. `max(+0.0, -0.0)` is a MEASURED three-vendor split (IDENTITY_PATHS
         row 39, 2026-08-23: `-0.0` on Apple, `+0.0` on NVIDIA and AMD).

    REFUSED, and it happens to also be the reference's behavior -- the only
    kind of correctness improvement this lane accepts. Sabotage
    `B18_SOFTMAX_DECOMPOSED`, INERT at `s == 1`.

    **WHAT CONTRACT 5.4's OPEN QUESTION COSTS HERE: NOTHING.** This form
    reads only `y` and `dy`, so if S18 were ever changed from a division to
    a reciprocal multiply this kernel would not change -- it would be the
    derivative of a different forward. The gap does not compound, which is
    rare enough to be worth a sentence."""
    var n = Int(n_in)
    var s = Int(s_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var r = i // s
    var yv = ftz(y.unsafe_load(i))
    var dv = ftz(dy.unsafe_load(i))
    var z = ftz(zdot.unsafe_load(r))

    comptime if SAB_B18_SOFTMAX_DECOMPOSED:
        # SABOTAGE: the product DISTRIBUTED across the subtraction, which is
        # what differentiating the division node and the exp node separately
        # produces: `y*dy - y*z` instead of `y*(dy - z)`. Two products and a
        # subtract where the pinned form has a subtract and a product, so a
        # different rounding path on every ordinary cell.
        #
        # **THE MAX NODE'S CONTRIBUTION IS NOT SPELLED EVEN HERE**, because
        # a device kernel cannot scatter an argmax residue without a float
        # ATOMIC and an atomic is precisely what this file refuses. So this
        # arm is a WEAKER sabotage than the full decomposed graph: it
        # separates the association and not the argmax. The argmax half is
        # unfirable by construction, which is itself the argument for
        # DEVIATION 1406 and is recorded rather than papered over.
        #
        # INERT at s == 1, where y == 1 and z == dy and both forms give
        # exactly zero.
        var p1 = ftz(pinned_mul(yv, ftz(dv)))
        var p2 = ftz(pinned_mul(yv, ftz(z)))
        ds.unsafe_store(i, ftz(ftz(p1) - ftz(p2)))
        return

    ds.unsafe_store(i, ftz(pinned_mul(yv, ftz(ftz(dv) - ftz(z)))))


# ===========================================================================
# THE THREE ATTENTION CHAINS. DEVIATIONS 1402, 1403, 1404, 1424.
#
# ONE THREAD PER OUTPUT CELL, no float atomic, no cross-thread reduction.
# `gemm/IDENTICAL_BACKWARD_PLAN.md` section 4.5 named the attention backward
# the one row with "a genuine structural obstacle", because the standard
# FUSED attention backward accumulates `dK` and `dV` across thread blocks
# with float atomics. That obstacle is gone here, and not by cleverness: it
# is the direct consequence of refusing the fused shape.
# ===========================================================================


def bwd_dq_kernel(
    dq: MutPointer[Float32, MutAnyOrigin],
    dcell: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_in: Int32,
    scale_in: Float32,
):
    """`dq[b,h,t,d] = sum_j dcell[b,h,t,j] * k_cache[b,kv,j,d]`, SERIAL
    ASCENDING over the ABSOLUTE key index from `+0.0`, one `fma` per term.
    One thread per output cell of `[M, n_heads*head_dim]`.

    **GEMM v1 IS REFUSED HERE AND THE REASON IS NOT THE FORWARD'S REASON.
    THIS IS THE LANE'S LARGEST FINDING** (DEVIATION 1402).

    S11, the QK product, was ROUTED through gemm v1 (contract DEVIATION 808)
    precisely because it contracts over `head_dim`, the same integer in a
    prefill and in a decode step. **A DERIVATIVE SWAPS WHICH AXIS IS
    CONTRACTED**: `dA` contracts over the OUTPUT WIDTH, and S11's output
    width is the KV AXIS. So the routed `dq` would be an `OP_NN` at
    `(l, hd, s)` with `k' = S`, and `P = f(S)` builds one tree at `S = 257`
    and a different one at `S = 200`. The masked `+0.0` tail -- bitwise
    inert in a serial ascending chain -- is NOT inert under a tree whose
    shape changes with the length.

    Under the chain it IS inert. At a masked `(t, j)` the gradient `dcell`
    is a signed zero (the softmax's `y_j` is exactly `+0.0` there and the
    mask backward is the identity), `fma(+-0.0, k, acc)` is `acc + (+-0.0)`,
    and a `+0.0`-seeded chain never holds `-0.0`. **So `dq` is independent
    of the kv length and a decode step's `dq` is the prefill's `dq` bit for
    bit** -- the backward's clause (c) and clause (d), holding by
    CONSTRUCTION, with the gate there to catch an execution plan violating
    the construction.

    Sabotage `B11_DQ_VIA_GEMM` lives in the launcher rather than here,
    because it is a different call graph and not a different branch. It is
    **INERT AT EVERY `S <= 128`**.

    `scale_in` is read ONLY by `B12_SCALE_INTO_DQ`. The profile scales the
    finished `dS` (DEVIATION 1415), never folds the scale into this chain,
    and that arm is INERT at every power-of-four `head_dim`.

    `repeat_kv` is `kv = h // n_rep`, an INDEX MAP (contract DEVIATION 813).
    At `n_rep == 1` a broken head-to-kv map is INVISIBLE, so the gates must
    carry both `n_rep == 1` and `n_rep == 2`."""
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
        var g = ftz(dcell.unsafe_load(wbase + j))
        comptime if SAB_B12_SCALE_INTO_DQ:
            # SABOTAGE: the scale folded in per term rather than applied to
            # the finished dS. Bitwise inert whenever the scale is an exact
            # power of two, which it is at head_dim 16 and 64.
            g = ftz(pinned_mul(g, scale_in))
        var kv = ftz(k_cache.unsafe_load(vbase + j * hd + d))
        acc = ftz(identical_mul_add(g, kv, acc))
    dq.unsafe_store(i, acc)


def bwd_dk_kernel(
    dk: MutPointer[Float32, MutAnyOrigin],
    dcell: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_in: Int32,
):
    """`dk[b,kv,j,d] = sum over (h in the kv group ASCENDING, t ASCENDING)
    of dcell[b,h,t,j] * q_rope[b,h,t,d]`, from `+0.0`, FUSED. One thread per
    output cell of `[B, n_kv, S, head_dim]`.

    **ONE CHAIN, NOT A SUM OF PER-HEAD PARTIALS** (DEVIATION 1424). With
    `n_rep > 1` several attention heads share one kv head, and folding each
    head separately and adding the partials is a DIFFERENT ASSOCIATION. The
    head axis is OUTERMOST, matching the forward's own `(batch, head,
    query)` nesting.

    **THE OTHER OPERAND IS `q_rope.out`, NOT `q_proj.out`.** The QK product
    reads the ROTATED query, so its derivative does too, and reading the
    pre-rotation activation here would produce a plausible gradient that is
    wrong by exactly one rotation -- invisible in a magnitude check.

    **GEMM v1 IS REFUSED**: the routed form is `OP_TN` at `(s, hd, l)` with
    `k' = L`, the QUERY COUNT, path dependent. Sabotage `B11_DK_VIA_GEMM` in
    the launcher, INERT only when `L <= 128` AND `n_rep == 1` together.

    **THIS IS A PARTIAL GRADIENT AND IT IS NAMED ONE** (DEVIATION 1417). A
    key at slot `j` is read by every query at absolute position `>= j`, and
    the queries in LATER calls are not in this call, so the fold over
    `[0, l)` is the contribution of THIS call's queries and nothing else.
    The full `[0, S)` range is written so a multi-call assembler has a
    complete partial to add; assembling is out of scope. **THE FORWARD'S
    DECODE-EQUALS-PREFILL DOES NOT TRANSFER TO THIS OUTPUT AND THIS FILE
    DOES NOT PRETEND IT DOES.**"""
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var s = Int(s_in)
    var n_rep = nh // nkv
    var qw = nh * hd
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * nkv * s * hd:
        return
    var d = i % hd
    var rest = i // hd
    var j = rest % s
    var rest2 = rest // s
    var kvh = rest2 % nkv
    var bb = rest2 // nkv

    var acc = Float32(0.0)
    for hh in range(n_rep):
        var h = kvh * n_rep + hh
        for t in range(l):
            var wbase = ((bb * nh + h) * l + t) * s
            acc = ftz(
                identical_mul_add(
                    ftz(dcell.unsafe_load(wbase + j)),
                    ftz(q_rope.unsafe_load((bb * l + t) * qw + h * hd + d)),
                    acc,
                )
            )
    dk.unsafe_store(i, acc)


def bwd_dv_kernel(
    dv: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_in: Int32,
):
    """`dv[b,kv,j,d] = sum over (h in the kv group ASCENDING, t ASCENDING)
    of y[b,h,t,j] * dctx[b,h,t,d]`, from `+0.0`, FUSED. One thread per
    output cell.

    DEVIATION 1404, the mirror of contract DEVIATION 807. Same fold axis as
    `bwd_dk_kernel`, same refusal of gemm v1, same partial-gradient caveat.
    Sabotage `B19_DV_VIA_GEMM` is the backward twin of the forward's
    `S19_VALUE_SUM_VIA_GEMM` and inherits its warning verbatim: without
    clause (d) it looks inert and gets deleted.

    The masked cells contribute `fma(+0.0, dctx, acc)` because `y` is
    exactly `+0.0` there (contract 7.1), so they are bitwise inert here too
    -- which is why `dv` for a key slot does not depend on how many PADDING
    keys were in the launch, only on how many QUERIES were."""
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var s = Int(s_in)
    var n_rep = nh // nkv
    var qw = nh * hd
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * nkv * s * hd:
        return
    var d = i % hd
    var rest = i // hd
    var j = rest % s
    var rest2 = rest // s
    var kvh = rest2 % nkv
    var bb = rest2 // nkv

    var acc = Float32(0.0)
    for hh in range(n_rep):
        var h = kvh * n_rep + hh
        for t in range(l):
            var wbase = ((bb * nh + h) * l + t) * s
            acc = ftz(
                identical_mul_add(
                    ftz(weights.unsafe_load(wbase + j)),
                    ftz(dctx.unsafe_load((bb * l + t) * qw + h * hd + d)),
                    acc,
                )
            )
    dv.unsafe_store(i, acc)


def bwd_dw_chain_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_in: Int32,
):
    """SABOTAGE-ONLY (`B10_DW_VIA_CHAIN`). The attention-weight gradient
    spelled as a hand chain over `head_dim` instead of the ROUTED gemm v1
    cell that the profile uses.

    **THIS ARM CANNOT FIRE AT THE GATE SHAPE.** The chain and the gemm agree
    bit for bit whenever `P(head_dim) == 1`, i.e. `head_dim <= 128`, and the
    profile's fixtures are 16 and 24. So DEVIATION 1405 rests on the
    argument in `bwd_attention_weight_grad`'s docstring and on gemm v1's own
    certificate, NOT on a fired arm. It is written and wired anyway, because
    a profile that ever admits `head_dim > 128` needs it, and because an arm
    that exists and is INERT is honest where a missing arm is a gap nobody
    recorded."""
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var s = Int(s_in)
    var n_rep = nh // nkv
    var qw = nh * hd
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * nh * l * s:
        return
    var j = i % s
    var rest = i // s
    var t = rest % l
    var rest2 = rest // l
    var h = rest2 % nh
    var bb = rest2 // nh
    var kvh = h // n_rep
    var vbase = (bb * nkv + kvh) * s * hd
    var cbase = (bb * l + t) * qw + h * hd

    var acc = Float32(0.0)
    for d in range(hd):
        acc = ftz(
            identical_mul_add(
                ftz(dctx.unsafe_load(cbase + d)),
                ftz(v_cache.unsafe_load(vbase + j * hd + d)),
                acc,
            )
        )
    dw.unsafe_store(i, acc)


# ===========================================================================
# THE GATHER / SCATTER COPIES for the per-(batch, head) GEMM calls. None is
# a seam: each moves bits untouched, and the arithmetic they surround is
# gemm v1's. Contract section 4's preamble is explicit that a copy is not a
# seam, and the forward device file makes the same split for the same
# reason: a copy that also rounds is a copy nobody reviews as a seam.
# ===========================================================================


def bwd_gather_ctx_head_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    nh_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """One head's `[L, head_dim]` block of `d(attn.ctx)`, contiguous."""
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
    dst.unsafe_store(i, dctx.unsafe_load((bb * l + t) * nh * hd + h * hd + d))


def bwd_gather_kv_head_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    cache: MutPointer[Float32, MutAnyOrigin],
    s_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    kvh_in: Int32,
):
    """One kv head's `[S, head_dim]` block. `repeat_kv` is the
    `kvh = h // n_rep` the CALLER passes: an index map, never a materialized
    expansion (contract DEVIATION 813)."""
    var s = Int(s_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var bb = Int(bb_in)
    var kvh = Int(kvh_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= s * hd:
        return
    dst.unsafe_store(i, cache.unsafe_load((bb * nkv + kvh) * s * hd + i))


def bwd_scatter_head_ls_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    s_in: Int32,
    nh_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """A `[L, S]` per-head block into `[B, n_heads, L, S]`."""
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
    dst.unsafe_store(((bb * nh + h) * l + t) * s + j, src.unsafe_load(i))


def bwd_gather_head_ls_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    s_in: Int32,
    nh_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """`bwd_scatter_head_ls_kernel` in reverse. SABOTAGE-ONLY: it exists so
    the three `*_VIA_GEMM` arms have a body. Nothing in the profile calls
    it."""
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
    dst.unsafe_store(i, src.unsafe_load(((bb * nh + h) * l + t) * s + j))


def bwd_scatter_q_head_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    l_in: Int32,
    nh_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    h_in: Int32,
):
    """A `[L, head_dim]` per-head block into `[M, n_heads*head_dim]`.
    SABOTAGE-ONLY, the other half of `B11_DQ_VIA_GEMM`."""
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
    dst.unsafe_store((bb * l + t) * nh * hd + h * hd + d, src.unsafe_load(i))


def bwd_accum_kv_head_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    s_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    bb_in: Int32,
    kvh_in: Int32,
    first_in: Int32,
):
    """Add a `[S, head_dim]` per-head partial into one kv head's slice.
    SABOTAGE-ONLY, the accumulation half of `B11_DK_VIA_GEMM` and
    `B19_DV_VIA_GEMM`.

    `first_in` non-zero STORES rather than adds, so the per-head partials
    are accumulated with no `+0.0` seed -- which is what makes the arm a
    faithful spelling of "route it and add the groups" rather than a
    straw man carrying a second defect."""
    var s = Int(s_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var bb = Int(bb_in)
    var kvh = Int(kvh_in)
    var first = Int(first_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= s * hd:
        return
    var ix = (bb * nkv + kvh) * s * hd + i
    if first != 0:
        dst.unsafe_store(ix, src.unsafe_load(i))
        return
    dst.unsafe_store(
        ix, ftz(ftz(dst.unsafe_load(ix)) + ftz(src.unsafe_load(i)))
    )


def bwd_kv_slice_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
):
    """The KV append's backward: a SLICE, no arithmetic. This call's own
    tokens occupy slots `[pos0, S)` and their gradient is read out at the
    token-major `[M, n_kv*head_dim]` layout the projections expect. Slots
    `[0, pos0)` are the HANDOFF of DEVIATION 1417 and are not consumed
    here."""
    var b = Int(b_in)
    var l = Int(l_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= b * l * nkv * hd:
        return
    var d = i % hd
    var rest = i // hd
    var kvh = rest % nkv
    var rest2 = rest // nkv
    var li = rest2 % l
    var bb = rest2 // l
    dst.unsafe_store(
        i, src.unsafe_load((bb * nkv + kvh) * s * hd + (pos0 + li) * hd + d)
    )


# ===========================================================================
# THE ROUTING DOOR (DEVIATION 1425). Two launchers, and they are the ONLY
# place in this file where a backward GEMM shape is decided.
# ===========================================================================


def _route_a(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut other: DeviceBuffer[DType.float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`dA` for a forward `C = op(A) . op(B)` at `(m, n, k)`, into
    `out_buf`. `other` is the forward's `B` operand, unchanged and
    untransposed.

    **THE SIDE FLAG IS HONORED RATHER THAN ASSUMED.** Two of the GEMM lane's
    six rows put `dC` on the RIGHT. Neither is reached by this profile --
    every forward call here is `OP_NT`, whose `dA` and `dB` both put `dC`
    left -- so plan section 6.3 predicts `SAB_BWD_OPERAND_ORDER` is INERT
    through these entry points. An inert arm that is WIRED is worth more
    than one that is not, because the next profile may not be all-`OP_NT`.

    **DEVIATION 1428: this calls the SYNCHRONIZING `identical_gemm`, not
    `identical_gemm_backward_a_into`.** The `_into` form takes a
    caller-owned workspace, and the forward device file uses the
    synchronizing form throughout; mixing would be a second discipline in
    one lane. **The cost is that gemm gate G7's workspace-sizing coverage is
    NOT inherited**, and that cost is stated rather than hidden."""
    var call = gemm_backward_a_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        identical_gemm(
            ctx, out_buf, dc, other, call[1], call[2], call[3], call[0]
        )
        return
    identical_gemm(
        ctx, out_buf, other, dc, call[1], call[2], call[3], call[0]
    )


def _route_b(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut other: DeviceBuffer[DType.float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`dB` for a forward `C = op(A) . op(B)` at `(m, n, k)`. `other` is the
    forward's `A` operand.

    **THIS IS THE CALL WHOSE `k'` IS THE TOKEN COUNT** for every projection
    in this block, which is `gemm_backward_b_call`'s own headline finding.
    So it is the call whose partition, whose plan and whose BITS move when
    the batch size moves. Nothing about that threatens cross-vendor
    identity and everything about it means a training run has to declare its
    batch and chunk schedule as part of its numerical specification --
    plan sections 5.2 and 5.4, and the gate asserts the NEGATIVE property
    that these outputs MOVE under a change of batch composition."""
    var call = gemm_backward_b_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        identical_gemm(
            ctx, out_buf, dc, other, call[1], call[2], call[3], call[0]
        )
        return
    identical_gemm(
        ctx, out_buf, other, dc, call[1], call[2], call[3], call[0]
    )


# ===========================================================================
# THE STAGES AND THE SCRATCH
# ===========================================================================


struct LlamaBackwardStages(Movable):
    """The thirty-seven recorded backward stages plus the scratch the
    launchers need. Plan section 6.1 lists the tags and the layouts.

    **FOUR `[B, n_heads, L, S]` BUFFERS is what the eager path costs on the
    backward**, against the forward's six, so a backward call's peak
    footprint is roughly 1.7x a forward call's on the dominant term, plus
    the saved forward stages held for the whole interval. Plan section 7
    prices it. It is also what makes every attention seam separately
    recordable, and a lane whose whole instrument is the per-stage card
    should not begin by fusing the stages away.

    The attention buffers are allocated at `s_max` and USED PACKED at the
    call's `s`, so `record_device(..., count = b*nh*l*s)` hashes exactly the
    array the card lists and never folds uninitialized tail memory into a
    stage. `core/identity_trace.mojo` is emphatic: when a buffer is used
    short, PASS THE LENGTH.

    **THE SCRATCH FIELDS CARRY NO TAG.** `dh`, `dprod`, `rstd`, `dvcoef`,
    `ones`, `tmp0`, `tmp1`, `tmp2`, `head_a`, `head_b`, `head_c` hold
    intermediates and per-head gathers, and no seam records them. `dprod`
    and `dh` are the two a reader might expect to see on the card: `dprod`
    is the input to a gemm whose OUTPUT is `bwd.dW_norm*`, and `dh` is the
    input to a fold whose output is `bwd.norm*.dot`, so both are one step
    from a recorded stage and neither adds localization."""

    var b: Int
    var l: Int
    var s_max: Int
    var dims: LlamaDims

    var in_d_residual2: DeviceBuffer[DType.float32]
    var d_down_proj_out: DeviceBuffer[DType.float32]
    var d_mlp_gated: DeviceBuffer[DType.float32]
    var dw_down: DeviceBuffer[DType.float32]
    var d_silu_out: DeviceBuffer[DType.float32]
    var d_up_proj_out: DeviceBuffer[DType.float32]
    var d_gate_proj_out: DeviceBuffer[DType.float32]
    var dw_gate: DeviceBuffer[DType.float32]
    var dw_up: DeviceBuffer[DType.float32]
    var d_norm2_out: DeviceBuffer[DType.float32]
    var norm2_dot: DeviceBuffer[DType.float32]
    var dw_norm2: DeviceBuffer[DType.float32]
    var norm2_dx: DeviceBuffer[DType.float32]
    var d_residual1: DeviceBuffer[DType.float32]
    var d_o_proj_out: DeviceBuffer[DType.float32]
    var d_attn_ctx: DeviceBuffer[DType.float32]
    var dw_o: DeviceBuffer[DType.float32]
    var d_attn_weights: DeviceBuffer[DType.float32]
    var attn_zdot: DeviceBuffer[DType.float32]
    var d_attn_masked: DeviceBuffer[DType.float32]
    var d_attn_scores: DeviceBuffer[DType.float32]
    var d_qk_cell: DeviceBuffer[DType.float32]
    var d_q_rope: DeviceBuffer[DType.float32]
    var d_k_cache: DeviceBuffer[DType.float32]
    var d_v_cache: DeviceBuffer[DType.float32]
    var d_k_rope: DeviceBuffer[DType.float32]
    var d_v_proj_out: DeviceBuffer[DType.float32]
    var d_q_proj_out: DeviceBuffer[DType.float32]
    var d_k_proj_out: DeviceBuffer[DType.float32]
    var dw_q: DeviceBuffer[DType.float32]
    var dw_k: DeviceBuffer[DType.float32]
    var dw_v: DeviceBuffer[DType.float32]
    var d_norm1_out: DeviceBuffer[DType.float32]
    var norm1_dot: DeviceBuffer[DType.float32]
    var dw_norm1: DeviceBuffer[DType.float32]
    var norm1_dx: DeviceBuffer[DType.float32]
    var d_x: DeviceBuffer[DType.float32]

    # ---- scratch, no tag ------------------------------------------------
    var dh: DeviceBuffer[DType.float32]  # [M, d_model]
    var dprod: DeviceBuffer[DType.float32]  # [M, d_model]
    var rstd: DeviceBuffer[DType.float32]  # [M]
    var dvcoef: DeviceBuffer[DType.float32]  # [M]
    var ones: DeviceBuffer[DType.float32]  # [M], exactly 1.0
    var tmp0: DeviceBuffer[DType.float32]  # [M, max(d_model, intermediate)]
    var tmp1: DeviceBuffer[DType.float32]
    var tmp2: DeviceBuffer[DType.float32]
    var head_a: DeviceBuffer[DType.float32]  # [L, head_dim]
    var head_b: DeviceBuffer[DType.float32]  # [s_max, head_dim]
    var head_c: DeviceBuffer[DType.float32]  # [L, s_max]

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
            raise Error("llama backward: stages need B > 0 and L > 0")
        if s_max < l:
            raise Error(
                String("llama backward: s_max ")
                + String(s_max)
                + " is smaller than L "
                + String(l)
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
        var wide = dm
        if it > wide:
            wide = it

        self.in_d_residual2 = _zeros(ctx, m * dm)
        self.d_down_proj_out = _zeros(ctx, m * dm)
        self.d_mlp_gated = _zeros(ctx, m * it)
        self.dw_down = _zeros(ctx, dm * it)
        self.d_silu_out = _zeros(ctx, m * it)
        self.d_up_proj_out = _zeros(ctx, m * it)
        self.d_gate_proj_out = _zeros(ctx, m * it)
        self.dw_gate = _zeros(ctx, it * dm)
        self.dw_up = _zeros(ctx, it * dm)
        self.d_norm2_out = _zeros(ctx, m * dm)
        self.norm2_dot = _zeros(ctx, m)
        self.dw_norm2 = _zeros(ctx, dm)
        self.norm2_dx = _zeros(ctx, m * dm)
        self.d_residual1 = _zeros(ctx, m * dm)
        self.d_o_proj_out = _zeros(ctx, m * dm)
        self.d_attn_ctx = _zeros(ctx, m * qw)
        self.dw_o = _zeros(ctx, dm * qw)
        self.d_attn_weights = _zeros(ctx, b * nh * l * s_max)
        self.attn_zdot = _zeros(ctx, b * nh * l)
        self.d_attn_masked = _zeros(ctx, b * nh * l * s_max)
        self.d_attn_scores = _zeros(ctx, b * nh * l * s_max)
        self.d_qk_cell = _zeros(ctx, b * nh * l * s_max)
        self.d_q_rope = _zeros(ctx, m * qw)
        self.d_k_cache = _zeros(ctx, b * nkv * s_max * hd)
        self.d_v_cache = _zeros(ctx, b * nkv * s_max * hd)
        self.d_k_rope = _zeros(ctx, m * kw)
        self.d_v_proj_out = _zeros(ctx, m * kw)
        self.d_q_proj_out = _zeros(ctx, m * qw)
        self.d_k_proj_out = _zeros(ctx, m * kw)
        self.dw_q = _zeros(ctx, qw * dm)
        self.dw_k = _zeros(ctx, kw * dm)
        self.dw_v = _zeros(ctx, kw * dm)
        self.d_norm1_out = _zeros(ctx, m * dm)
        self.norm1_dot = _zeros(ctx, m)
        self.dw_norm1 = _zeros(ctx, dm)
        self.norm1_dx = _zeros(ctx, m * dm)
        self.d_x = _zeros(ctx, m * dm)

        self.dh = _zeros(ctx, m * dm)
        self.dprod = _zeros(ctx, m * dm)
        self.rstd = _zeros(ctx, m)
        self.dvcoef = _zeros(ctx, m)
        self.ones = _fill_ones(ctx, m)
        self.tmp0 = _zeros(ctx, m * wide)
        self.tmp1 = _zeros(ctx, m * wide)
        self.tmp2 = _zeros(ctx, m * wide)
        self.head_a = _zeros(ctx, l * hd)
        self.head_b = _zeros(ctx, s_max * hd)
        self.head_c = _zeros(ctx, l * s_max)


def backward_stage_tag(i: Int) raises -> String:
    """The thirty-seven backward tags, in the plan's order, DUPLICATED from
    `transformer_backward_oracle.mojo::backward_stage_tag` ON PURPOSE.

    This file may not import from `transformer/original/`'s oracle half in
    the direction that would make the two halves one module -- the forward
    lane made the same split and its `PLANT_AT_*` constants carry the same
    note. **The ORCHESTRATOR owns keeping these two lists equal.** If they
    ever disagree, `tools/identity_trace_diff.py` aligns two tag SEQUENCES
    and produces a WRONG ALIGNMENT that pairs one run's stage against
    another run's different stage and reports a plausible answer, which
    reads exactly like an arithmetic bug and is not one."""
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
        String("llama backward: no stage ")
        + String(i)
        + " (there are "
        + String(BWD_STAGE_COUNT)
        + ", plan section 6.1)"
    )


def _rec(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    prefix: String,
    idx: Int,
    mut buf: DeviceBuffer[DType.float32],
    count: Int,
) raises:
    """One card record, with the tag looked up by INDEX so that the tag and
    the order come from one place. `count` is the USED length, never
    `len(buf)`: the attention buffers are allocated at `s_max` and used at
    `s`, and hashing the tail would report a difference where there is
    none."""
    var tag = backward_stage_tag(idx)
    if prefix.byte_length() > 0:
        tag = prefix + String(".") + tag
    trace.record_device[DType.float32](ctx, tag, buf, count)


# ===========================================================================
# THE ATTENTION-WEIGHT GRADIENT: THE ONE ATTENTION SEAM THAT ROUTES.
# ===========================================================================


def bwd_attention_weight_grad(
    ctx: DeviceContext,
    mut bst: LlamaBackwardStages,
    mut fwd: LlamaDeviceStages,
    b: Int,
    l: Int,
    s: Int,
    dims: LlamaDims,
) raises:
    """`dy[b,h,t,j] = sum_d dctx[b,h,t,d] * v_cache[b,kv,j,d]`, ROUTED
    through a gemm v1 `OP_NT` cell at `(l, s, head_dim)`, one call per
    `(batch, head)`. DEVIATION 1405.

    **THIS IS THE SEAM WHERE THE ORGANIZING RULE PAYS OFF.** The fold
    contracts over `head_dim`, which is THE SAME INTEGER in a prefill and in
    a decode step, so gemm v1's `P = f(k)` is the same partition in both
    paths and routing is decode-safe. It is the very shape contract
    DEVIATION 808 already runs for S11.

    **THE ASYMMETRY WITH `bwd_dq_kernel` LOOKS LIKE AN INCONSISTENCY AND IS
    THE OPPOSITE OF ONE.** S19 was hand-written in the forward because it
    contracts over the key axis; its `dy` derivative contracts over
    `head_dim` and routes. S11 was routed in the forward because it
    contracts over `head_dim`; its `dq` and `dk` derivatives contract over
    the kv length and the query count and must be pinned. Both are the same
    rule applied to whichever axis the DERIVATIVE contracts.

    **THIS PIN CANNOT BE FALSIFIED AT THE GATE SHAPE.** `B10_DW_VIA_CHAIN`
    is bit identical to this whenever `P(head_dim) == 1`, i.e.
    `head_dim <= 128`, and the fixtures are 16 and 24. So DEVIATION 1405
    rests on the argument above and on gemm v1's own certificate, not on a
    fired arm.

    `identical_gemm` SYNCHRONIZES internally, so `head_a`, `head_b` and
    `head_c` must stay alive across it; they are `bst`'s fields and their
    lifetime is the struct's."""
    var nh = dims.n_heads
    var nkv = dims.n_kv
    var hd = dims.head_dim
    var n_rep = dims.n_rep()

    comptime if SAB_B10_DW_VIA_CHAIN:
        # SABOTAGE: the hand chain over head_dim. INERT at every
        # head_dim <= 128, which is every fixture in this profile.
        ctx.enqueue_function[bwd_dw_chain_kernel](
            bst.d_attn_weights.unsafe_ptr(),
            bst.d_attn_ctx.unsafe_ptr(),
            fwd.v_cache.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(nh),
            Int32(nkv),
            Int32(hd),
            Int32(s),
            grid_dim=(_grid(b * nh * l * s), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )
        ctx.synchronize()
        return

    for bb in range(b):
        for h in range(nh):
            var kvh = h // n_rep
            ctx.enqueue_function[bwd_gather_ctx_head_kernel](
                bst.head_a.unsafe_ptr(),
                bst.d_attn_ctx.unsafe_ptr(),
                Int32(l),
                Int32(nh),
                Int32(hd),
                Int32(bb),
                Int32(h),
                grid_dim=(_grid(l * hd), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
            ctx.enqueue_function[bwd_gather_kv_head_kernel](
                bst.head_b.unsafe_ptr(),
                fwd.v_cache.unsafe_ptr(),
                Int32(s),
                Int32(nkv),
                Int32(hd),
                Int32(bb),
                Int32(kvh),
                grid_dim=(_grid(s * hd), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
            ctx.synchronize()
            identical_gemm(
                ctx, bst.head_c, bst.head_a, bst.head_b, l, s, hd, OP_NT
            )
            ctx.enqueue_function[bwd_scatter_head_ls_kernel](
                bst.d_attn_weights.unsafe_ptr(),
                bst.head_c.unsafe_ptr(),
                Int32(l),
                Int32(s),
                Int32(nh),
                Int32(bb),
                Int32(h),
                grid_dim=(_grid(l * s), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
            ctx.synchronize()


def bwd_attention_grads(
    ctx: DeviceContext,
    mut bst: LlamaBackwardStages,
    mut fwd: LlamaDeviceStages,
    b: Int,
    l: Int,
    s: Int,
    dims: LlamaDims,
    scale: Float32,
) raises:
    """`dq`, `dk_cache` and `dv_cache`. The three PINNED chains, plus the
    three `*_VIA_GEMM` sabotage arms which are a different CALL GRAPH rather
    than a different branch and therefore live here.

    Read `bwd_dq_kernel`'s docstring for why gemm v1 is refused; it is the
    lane's largest finding and it is argued there rather than repeated."""
    var nh = dims.n_heads
    var nkv = dims.n_kv
    var hd = dims.head_dim
    var qw = dims.q_width()
    var n_rep = dims.n_rep()

    # ---- dq ------------------------------------------------------------
    comptime if SAB_B11_DQ_VIA_GEMM:
        # SABOTAGE: `OP_NN` at `(l, hd, s)`, `k' = S`. INERT at every
        # S <= 128, where P(S) == 1 and the gemm cell IS the whole-k
        # ascending chain from +0.0. Passes clause (a) at a fixed shape and
        # breaks the LENGTH clause.
        for bb in range(b):
            for h in range(nh):
                var kvh = h // n_rep
                ctx.enqueue_function[bwd_gather_head_ls_kernel](
                    bst.head_c.unsafe_ptr(),
                    bst.d_qk_cell.unsafe_ptr(),
                    Int32(l),
                    Int32(s),
                    Int32(nh),
                    Int32(bb),
                    Int32(h),
                    grid_dim=(_grid(l * s), 1, 1),
                    block_dim=(BWD_TPB, 1, 1),
                )
                ctx.enqueue_function[bwd_gather_kv_head_kernel](
                    bst.head_b.unsafe_ptr(),
                    fwd.k_cache.unsafe_ptr(),
                    Int32(s),
                    Int32(nkv),
                    Int32(hd),
                    Int32(bb),
                    Int32(kvh),
                    grid_dim=(_grid(s * hd), 1, 1),
                    block_dim=(BWD_TPB, 1, 1),
                )
                ctx.synchronize()
                identical_gemm(
                    ctx, bst.head_a, bst.head_c, bst.head_b, l, hd, s, OP_NN
                )
                ctx.enqueue_function[bwd_scatter_q_head_kernel](
                    bst.d_q_rope.unsafe_ptr(),
                    bst.head_a.unsafe_ptr(),
                    Int32(l),
                    Int32(nh),
                    Int32(hd),
                    Int32(bb),
                    Int32(h),
                    grid_dim=(_grid(l * hd), 1, 1),
                    block_dim=(BWD_TPB, 1, 1),
                )
                ctx.synchronize()
    else:
        ctx.enqueue_function[bwd_dq_kernel](
            bst.d_q_rope.unsafe_ptr(),
            bst.d_qk_cell.unsafe_ptr(),
            fwd.k_cache.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(nh),
            Int32(nkv),
            Int32(hd),
            Int32(s),
            scale,
            grid_dim=(_grid(b * l * qw), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )
        ctx.synchronize()

    # ---- dk ------------------------------------------------------------
    comptime if SAB_B11_DK_VIA_GEMM:
        # SABOTAGE: `OP_TN` at `(s, hd, l)`, `k' = L`, with the head group
        # accumulated as per-head PARTIALS -- a different association from
        # the profile's single chain. INERT only when L <= 128 AND
        # n_rep == 1 together.
        for bb2 in range(b):
            for kv2 in range(nkv):
                for hh in range(n_rep):
                    var h2 = kv2 * n_rep + hh
                    ctx.enqueue_function[bwd_gather_head_ls_kernel](
                        bst.head_c.unsafe_ptr(),
                        bst.d_qk_cell.unsafe_ptr(),
                        Int32(l),
                        Int32(s),
                        Int32(nh),
                        Int32(bb2),
                        Int32(h2),
                        grid_dim=(_grid(l * s), 1, 1),
                        block_dim=(BWD_TPB, 1, 1),
                    )
                    ctx.enqueue_function[bwd_gather_ctx_head_kernel](
                        bst.head_a.unsafe_ptr(),
                        bst.d_q_rope.unsafe_ptr(),
                        Int32(l),
                        Int32(nh),
                        Int32(hd),
                        Int32(bb2),
                        Int32(h2),
                        grid_dim=(_grid(l * hd), 1, 1),
                        block_dim=(BWD_TPB, 1, 1),
                    )
                    ctx.synchronize()
                    identical_gemm(
                        ctx,
                        bst.head_b,
                        bst.head_c,
                        bst.head_a,
                        s,
                        hd,
                        l,
                        OP_TN,
                    )
                    var first = 0
                    if hh == 0:
                        first = 1
                    ctx.enqueue_function[bwd_accum_kv_head_kernel](
                        bst.d_k_cache.unsafe_ptr(),
                        bst.head_b.unsafe_ptr(),
                        Int32(s),
                        Int32(nkv),
                        Int32(hd),
                        Int32(bb2),
                        Int32(kv2),
                        Int32(first),
                        grid_dim=(_grid(s * hd), 1, 1),
                        block_dim=(BWD_TPB, 1, 1),
                    )
                    ctx.synchronize()
    else:
        ctx.enqueue_function[bwd_dk_kernel](
            bst.d_k_cache.unsafe_ptr(),
            bst.d_qk_cell.unsafe_ptr(),
            fwd.q_rope.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(nh),
            Int32(nkv),
            Int32(hd),
            Int32(s),
            grid_dim=(_grid(b * nkv * s * hd), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )
        ctx.synchronize()

    # ---- dv ------------------------------------------------------------
    comptime if SAB_B19_DV_VIA_GEMM:
        # SABOTAGE: the backward twin of the forward's
        # S19_VALUE_SUM_VIA_GEMM, same shape, same warning.
        for bb3 in range(b):
            for kv3 in range(nkv):
                for hh3 in range(n_rep):
                    var h3 = kv3 * n_rep + hh3
                    ctx.enqueue_function[bwd_gather_head_ls_kernel](
                        bst.head_c.unsafe_ptr(),
                        fwd.weights.unsafe_ptr(),
                        Int32(l),
                        Int32(s),
                        Int32(nh),
                        Int32(bb3),
                        Int32(h3),
                        grid_dim=(_grid(l * s), 1, 1),
                        block_dim=(BWD_TPB, 1, 1),
                    )
                    ctx.enqueue_function[bwd_gather_ctx_head_kernel](
                        bst.head_a.unsafe_ptr(),
                        bst.d_attn_ctx.unsafe_ptr(),
                        Int32(l),
                        Int32(nh),
                        Int32(hd),
                        Int32(bb3),
                        Int32(h3),
                        grid_dim=(_grid(l * hd), 1, 1),
                        block_dim=(BWD_TPB, 1, 1),
                    )
                    ctx.synchronize()
                    identical_gemm(
                        ctx,
                        bst.head_b,
                        bst.head_c,
                        bst.head_a,
                        s,
                        hd,
                        l,
                        OP_TN,
                    )
                    var first3 = 0
                    if hh3 == 0:
                        first3 = 1
                    ctx.enqueue_function[bwd_accum_kv_head_kernel](
                        bst.d_v_cache.unsafe_ptr(),
                        bst.head_b.unsafe_ptr(),
                        Int32(s),
                        Int32(nkv),
                        Int32(hd),
                        Int32(bb3),
                        Int32(kv3),
                        Int32(first3),
                        grid_dim=(_grid(s * hd), 1, 1),
                        block_dim=(BWD_TPB, 1, 1),
                    )
                    ctx.synchronize()
    else:
        ctx.enqueue_function[bwd_dv_kernel](
            bst.d_v_cache.unsafe_ptr(),
            fwd.weights.unsafe_ptr(),
            bst.d_attn_ctx.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(nh),
            Int32(nkv),
            Int32(hd),
            Int32(s),
            grid_dim=(_grid(b * nkv * s * hd), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )
        ctx.synchronize()


# ===========================================================================
# THE RMSNORM BACKWARD LAUNCHER
# ===========================================================================


def bwd_rms_norm(
    ctx: DeviceContext,
    mut dot_out: DeviceBuffer[DType.float32],
    mut dx_out: DeviceBuffer[DType.float32],
    mut dw_out: DeviceBuffer[DType.float32],
    mut dh: DeviceBuffer[DType.float32],
    mut dprod: DeviceBuffer[DType.float32],
    mut rstd: DeviceBuffer[DType.float32],
    mut dvcoef: DeviceBuffer[DType.float32],
    mut ones: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    m: Int,
    dm: Int,
    eps: Float32,
) raises:
    """Three kernels then one gemm. `dot_out`, `dx_out` and `dw_out` are the
    three recorded stages; `dh`, `dprod`, `rstd`, `dvcoef` and `ones` are
    the untagged scratch.

    **THE BUFFERS ARE PASSED ONE BY ONE RATHER THAN AS `mut bst`**, and that
    is not style. Taking `mut bst: LlamaBackwardStages` here and calling it
    as `bwd_rms_norm(ctx, bst, bst.norm2_dot, ...)` would alias the whole
    struct against one of its own fields at the call site, which no borrow
    checker should accept and which this lane cannot test. Twelve arguments
    is the cost of not writing an aliasing call. Distinct FIELDS of one
    struct as separate `mut` arguments is a milder version of the same
    question and it is on this file's RISK list.

    The weight gradient is `ones[1 x M] . dprod[M x dm]`, an `OP_NN` at
    `(1, dm, M)` -- **the GEMM lane's own bias-gradient trick lifted intact**
    (DEVIATION 851 becoming DEVIATION 1410). Its `k'` is `M`, THE TOKEN
    COUNT, so this output is not batch-composition invariant and the gate
    asserts that it MOVES."""
    ctx.enqueue_function[bwd_norm_dh_kernel](
        dh.unsafe_ptr(),
        dy.unsafe_ptr(),
        weight.unsafe_ptr(),
        Int32(m),
        Int32(dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[bwd_norm_dot_kernel](
        dot_out.unsafe_ptr(),
        rstd.unsafe_ptr(),
        dvcoef.unsafe_ptr(),
        dh.unsafe_ptr(),
        x.unsafe_ptr(),
        sumsq.unsafe_ptr(),
        Int32(m),
        Int32(dm),
        eps,
        grid_dim=(_grid(m), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[bwd_norm_dx_kernel](
        dx_out.unsafe_ptr(),
        dprod.unsafe_ptr(),
        dh.unsafe_ptr(),
        dy.unsafe_ptr(),
        x.unsafe_ptr(),
        rstd.unsafe_ptr(),
        dvcoef.unsafe_ptr(),
        Int32(m),
        Int32(dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    identical_gemm(ctx, dw_out, ones, dprod, 1, dm, m, OP_NN)


# ===========================================================================
# THE BLOCK BACKWARD
# ===========================================================================


def llama_decoder_layer_backward(
    ctx: DeviceContext,
    mut bst: LlamaBackwardStages,
    mut fwd: LlamaDeviceStages,
    mut w: LlamaDeviceWeights,
    mut cos_tab: DeviceBuffer[DType.float32],
    mut sin_tab: DeviceBuffer[DType.float32],
    mut x_dev: DeviceBuffer[DType.float32],
    d_out: List[Float32],
    b: Int,
    l: Int,
    pos0: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """The gradient of ONE `LlamaDecoderLayer.forward` call, stage by stage,
    recorded onto `trace` in the plan's card order.

    `fwd` is the SAVED forward stages of THIS SAME CALL. `d_out` is
    `d(residual2.out)` at `[M, d_model]`, arriving from somewhere this lane
    does not specify -- exactly as `dC` arrives in `gemm_backward.mojo`.
    `pos0` is the KV cache's used count BEFORE the forward call's append, so
    `S = pos0 + l` and cache slot `j` is absolute position `j`.

    `cos_tab` and `sin_tab` are `LlamaRopeTable`'s two buffers, passed
    directly rather than as the struct, so that this entry point does not
    need a `mut` binding on a struct whose other fields it never reads.

    `x_dev` is the BLOCK INPUT, `[M, d_model]`, and it is an argument
    because `LlamaDeviceStages` does not hold one: the forward launcher
    uploads `x`, uses it and lets it die. **THIS BACKWARD'S ONE MISSING
    SAVED TENSOR**, named rather than worked around -- see stage 33's
    comment and this lane's cross-lane request.
    `[[mojo-buffer-freed-at-last-use]]`: the CALLER owns it and must keep it
    alive past its own `ctx.synchronize()`.

    **WHAT IS NOT READ, AND IT IS A FINDING** (DEVIATION 1421). The
    closed-form softmax backward reads only `fwd.weights`. So `fwd.scores`,
    `fwd.masked`, `fwd.amax`, `fwd.aexp` and `fwd.denom` -- five buffers the
    forward computed, four of them `[B, nh, L, S]` -- are never touched
    here. Neither is `fwd.q_proj` or `fwd.k_proj`, because the RoPE backward
    is a linear map that reads only the TABLE, so the pre-rotation
    activations are not on the backward path at all.

    **NO LOSS, NO OPTIMIZER, NO ACCUMULATION BUFFER, NO TAPE**, and no
    multi-call assembly.

    Nothing below reads `B` except as a grid bound, so batch composition
    invariance of the ACTIVATION gradients is a property of the SHAPE of
    these kernels. The WEIGHT gradients are the opposite and the gate says
    so."""
    var dims = fwd.dims.copy()
    dims.validate()
    var dm = dims.d_model
    var nh = dims.n_heads
    var nkv = dims.n_kv
    var hd = dims.head_dim
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var m = b * l
    var s = pos0 + l
    var cells = b * nh * l * s

    # ---- refusals, before ANY recorded stage --------------------------
    if l <= 0 or b <= 0 or pos0 < 0:
        raise Error(
            "llama backward: B and L must be positive and pos0 non-negative"
        )
    if b != bst.b or l != bst.l:
        raise Error("llama backward: the stage buffers do not match (B, L)")
    if s > bst.s_max:
        raise Error(
            String("llama backward: pos0 + L = ")
            + String(s)
            + " exceeds the allocated s_max "
            + String(bst.s_max)
        )
    if len(d_out) != m * dm:
        raise Error(
            String("llama backward: d_out has ")
            + String(len(d_out))
            + " elements and [B, L, d_model] wants "
            + String(m * dm)
        )
    if len(x_dev) < m * dm:
        raise Error(
            String("llama backward: x_dev holds ")
            + String(len(x_dev))
            + " floats and the block input [B, L, d_model] wants "
            + String(m * dm)
            + " REFUSED. A short block input reads past its end into"
            + " whatever the allocator handed back, and the resulting dx is"
            + " plausible and wrong."
        )
    # DEVIATION 1423. The incoming gradient is an INPUT and gets contract
    # section 8's treatment: refused BY BITS, not by compares, because Metal
    # FLUSHES COMPARE OPERANDS (IDENTITY_PATHS row 49) so `v != v` is a test
    # with two meanings across columns. NaN payloads are vendor-shaped (row
    # 39 measured three payloads for one IEEE answer) and a certified stage
    # may not contain one.
    #
    # **A GRADIENT IS EXACTLY WHERE NaNs APPEAR IN PRACTICE**, which makes
    # this refusal more likely to fire than the forward's and makes the
    # named error worth more.
    for i in range(len(d_out)):
        var au = bitcast[DType.uint32](d_out[i]) & UInt32(0x7FFFFFFF)
        if au > UInt32(0x7F800000):
            raise Error(
                String("llama backward: NaN in d_residual2 at flat index ")
                + String(i)
                + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                + " stage may record one)"
            )
        if au == UInt32(0x7F800000):
            raise Error(
                String(
                    "llama backward: infinity in d_residual2 at flat index "
                )
                + String(i)
                + " REFUSED (row 39)"
            )

    var scale = llama_attention_scale(hd)

    # =====================================================================
    # STAGE 0-1. S23's backward. An add's derivative is the identity in both
    # arguments, so both branches take the incoming gradient unchanged. NO
    # ROUNDING: a copy, not a seam.
    # =====================================================================
    var d_in = _upload(ctx, d_out)
    ctx.enqueue_function[bwd_copy_kernel](
        bst.in_d_residual2.unsafe_ptr(),
        d_in.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[bwd_copy_kernel](
        bst.d_down_proj_out.unsafe_ptr(),
        d_in.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = d_in^
    _rec(ctx, trace, prefix, 0, bst.in_d_residual2, m * dm)
    _rec(ctx, trace, prefix, 1, bst.d_down_proj_out, m * dm)

    # =====================================================================
    # STAGE 2-3. `down_proj`: forward `OP_NT` at `(m, dm, it)`. ROUTED.
    # =====================================================================
    _route_a(
        ctx, bst.d_mlp_gated, bst.d_down_proj_out, w.w_down, OP_NT, m, dm, it
    )
    _rec(ctx, trace, prefix, 2, bst.d_mlp_gated, m * it)
    _route_b(
        ctx, bst.dw_down, bst.d_down_proj_out, fwd.gated, OP_NT, m, dm, it
    )
    _rec(ctx, trace, prefix, 3, bst.dw_down, dm * it)

    # =====================================================================
    # STAGE 4-5. S21's backward, two `pinned_mul`s. ROUTING.
    # =====================================================================
    ctx.enqueue_function[bwd_mul_kernel](
        bst.d_silu_out.unsafe_ptr(),
        bst.d_mlp_gated.unsafe_ptr(),
        fwd.up_proj.unsafe_ptr(),
        Int32(m * it),
        grid_dim=(_grid(m * it), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[bwd_mul_kernel](
        bst.d_up_proj_out.unsafe_ptr(),
        bst.d_mlp_gated.unsafe_ptr(),
        fwd.silu_out.unsafe_ptr(),
        Int32(m * it),
        grid_dim=(_grid(m * it), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 4, bst.d_silu_out, m * it)
    _rec(ctx, trace, prefix, 5, bst.d_up_proj_out, m * it)

    # =====================================================================
    # STAGE 6. S20's backward. NEW ARITHMETIC. DEVIATION 1411.
    # =====================================================================
    ctx.enqueue_function[bwd_silu_backward_kernel](
        bst.d_gate_proj_out.unsafe_ptr(),
        bst.d_silu_out.unsafe_ptr(),
        fwd.gate_proj.unsafe_ptr(),
        fwd.silu_out.unsafe_ptr(),
        Int32(m * it),
        grid_dim=(_grid(m * it), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 6, bst.d_gate_proj_out, m * it)

    # =====================================================================
    # STAGE 7-9. `gate_proj` and `up_proj`: forward `OP_NT` at
    # `(m, it, dm)`. ROUTED, plus a TWO-TERM fan-in in FORWARD-USE order
    # (gate is evaluated before up at LMLP:175).
    # =====================================================================
    _route_b(
        ctx, bst.dw_gate, bst.d_gate_proj_out, fwd.norm2_out, OP_NT, m, it, dm
    )
    _rec(ctx, trace, prefix, 7, bst.dw_gate, it * dm)
    _route_b(
        ctx, bst.dw_up, bst.d_up_proj_out, fwd.norm2_out, OP_NT, m, it, dm
    )
    _rec(ctx, trace, prefix, 8, bst.dw_up, it * dm)
    _route_a(
        ctx, bst.tmp0, bst.d_gate_proj_out, w.w_gate, OP_NT, m, it, dm
    )
    _route_a(ctx, bst.tmp1, bst.d_up_proj_out, w.w_up, OP_NT, m, it, dm)
    ctx.enqueue_function[bwd_add2_kernel](
        bst.d_norm2_out.unsafe_ptr(),
        bst.tmp0.unsafe_ptr(),
        bst.tmp1.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 9, bst.d_norm2_out, m * dm)

    # =====================================================================
    # STAGE 10-12. `post_attention_layernorm` backward. Its forward INPUT is
    # `residual1.out`.
    # =====================================================================
    bwd_rms_norm(
        ctx,
        bst.norm2_dot,
        bst.norm2_dx,
        bst.dw_norm2,
        bst.dh,
        bst.dprod,
        bst.rstd,
        bst.dvcoef,
        bst.ones,
        bst.d_norm2_out,
        fwd.residual1,
        w.norm2_w,
        fwd.norm2_sumsq,
        m,
        dm,
        w.eps,
    )
    _rec(ctx, trace, prefix, 10, bst.norm2_dot, m)
    _rec(ctx, trace, prefix, 11, bst.dw_norm2, dm)
    _rec(ctx, trace, prefix, 12, bst.norm2_dx, m * dm)

    # =====================================================================
    # STAGE 13-14. S22's backward. `residual1.out` fans out into the norm
    # (LDL:321) and into the residual add (LDL:323); FORWARD-USE order puts
    # the norm branch first. Two terms, so no order to pin -- stated rather
    # than assumed.
    # =====================================================================
    ctx.enqueue_function[bwd_add2_kernel](
        bst.d_residual1.unsafe_ptr(),
        bst.norm2_dx.unsafe_ptr(),
        bst.in_d_residual2.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 13, bst.d_residual1, m * dm)
    ctx.enqueue_function[bwd_copy_kernel](
        bst.d_o_proj_out.unsafe_ptr(),
        bst.d_residual1.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 14, bst.d_o_proj_out, m * dm)

    # =====================================================================
    # STAGE 15-16. `o_proj`: forward `OP_NT` at `(m, dm, qw)`. ROUTED.
    # =====================================================================
    _route_a(ctx, bst.d_attn_ctx, bst.d_o_proj_out, w.w_o, OP_NT, m, dm, qw)
    _rec(ctx, trace, prefix, 15, bst.d_attn_ctx, m * qw)
    _route_b(ctx, bst.dw_o, bst.d_o_proj_out, fwd.ctxv, OP_NT, m, dm, qw)
    _rec(ctx, trace, prefix, 16, bst.dw_o, dm * qw)

    # =====================================================================
    # STAGE 17. The attention-weight gradient, the ONE attention seam that
    # ROUTES. DEVIATION 1405.
    # =====================================================================
    bwd_attention_weight_grad(ctx, bst, fwd, b, l, s, dims)
    _rec(ctx, trace, prefix, 17, bst.d_attn_weights, cells)

    # =====================================================================
    # STAGE 18-19. The softmax backward, ONE closed form. DEVIATION 1406.
    # =====================================================================
    ctx.enqueue_function[bwd_softmax_zdot_kernel](
        bst.attn_zdot.unsafe_ptr(),
        bst.d_attn_weights.unsafe_ptr(),
        fwd.weights.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(l),
        Int32(s),
        grid_dim=(_grid(b * nh * l), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 18, bst.attn_zdot, b * nh * l)
    ctx.enqueue_function[bwd_softmax_ds_kernel](
        bst.d_attn_masked.unsafe_ptr(),
        bst.d_attn_weights.unsafe_ptr(),
        fwd.weights.unsafe_ptr(),
        bst.attn_zdot.unsafe_ptr(),
        Int32(cells),
        Int32(s),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 19, bst.d_attn_masked, cells)

    # =====================================================================
    # STAGE 20. S13's backward, an EXACT IDENTITY. DEVIATION 1414.
    # =====================================================================
    ctx.enqueue_function[bwd_mask_grad_kernel](
        bst.d_attn_scores.unsafe_ptr(),
        bst.d_attn_masked.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(l),
        Int32(s),
        Int32(pos0),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 20, bst.d_attn_scores, cells)

    # =====================================================================
    # STAGE 21. S12's backward. DEVIATION 1415.
    # =====================================================================
    ctx.enqueue_function[bwd_scale_kernel](
        bst.d_qk_cell.unsafe_ptr(),
        bst.d_attn_scores.unsafe_ptr(),
        Int32(cells),
        scale,
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 21, bst.d_qk_cell, cells)

    # =====================================================================
    # STAGE 22-24. The three PINNED attention chains. DEVIATIONS 1402, 1403,
    # 1404, 1424, and the lane's largest finding is argued at
    # `bwd_dq_kernel`.
    # =====================================================================
    bwd_attention_grads(ctx, bst, fwd, b, l, s, dims, scale)
    _rec(ctx, trace, prefix, 22, bst.d_q_rope, m * qw)
    _rec(ctx, trace, prefix, 23, bst.d_k_cache, b * nkv * s * hd)
    _rec(ctx, trace, prefix, 24, bst.d_v_cache, b * nkv * s * hd)

    # =====================================================================
    # STAGE 25-26. The KV append's backward: a SLICE, no arithmetic.
    # =====================================================================
    ctx.enqueue_function[bwd_kv_slice_kernel](
        bst.d_k_rope.unsafe_ptr(),
        bst.d_k_cache.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(nkv),
        Int32(hd),
        Int32(s),
        Int32(pos0),
        grid_dim=(_grid(m * kw), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[bwd_kv_slice_kernel](
        bst.d_v_proj_out.unsafe_ptr(),
        bst.d_v_cache.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(nkv),
        Int32(hd),
        Int32(s),
        Int32(pos0),
        grid_dim=(_grid(m * kw), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 25, bst.d_k_rope, m * kw)
    _rec(ctx, trace, prefix, 26, bst.d_v_proj_out, m * kw)

    # =====================================================================
    # STAGE 27-28. The RoPE backward, on q and on k. DEVIATION 1412. ONE
    # kernel serves both. **RoPE is NOT applied to v**, so v's gradient does
    # not pass through here -- easy to get wrong and impossible to see in
    # the output, because a rotated gradient is still a plausible gradient.
    # =====================================================================
    ctx.enqueue_function[bwd_rope_kernel](
        bst.d_q_proj_out.unsafe_ptr(),
        bst.d_q_rope.unsafe_ptr(),
        cos_tab.unsafe_ptr(),
        sin_tab.unsafe_ptr(),
        Int32(m),
        Int32(l),
        Int32(nh),
        Int32(hd),
        Int32(pos0),
        grid_dim=(_grid(m * qw), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[bwd_rope_kernel](
        bst.d_k_proj_out.unsafe_ptr(),
        bst.d_k_rope.unsafe_ptr(),
        cos_tab.unsafe_ptr(),
        sin_tab.unsafe_ptr(),
        Int32(m),
        Int32(l),
        Int32(nkv),
        Int32(hd),
        Int32(pos0),
        grid_dim=(_grid(m * kw), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 27, bst.d_q_proj_out, m * qw)
    _rec(ctx, trace, prefix, 28, bst.d_k_proj_out, m * kw)

    # =====================================================================
    # STAGE 29-32. The three input projections. ROUTED, plus THE ONE
    # THREE-TERM FAN-IN in this block, which IS order dependent.
    # =====================================================================
    _route_b(ctx, bst.dw_q, bst.d_q_proj_out, fwd.norm1_out, OP_NT, m, qw, dm)
    _rec(ctx, trace, prefix, 29, bst.dw_q, qw * dm)
    _route_b(ctx, bst.dw_k, bst.d_k_proj_out, fwd.norm1_out, OP_NT, m, kw, dm)
    _rec(ctx, trace, prefix, 30, bst.dw_k, kw * dm)
    _route_b(ctx, bst.dw_v, bst.d_v_proj_out, fwd.norm1_out, OP_NT, m, kw, dm)
    _rec(ctx, trace, prefix, 31, bst.dw_v, kw * dm)
    _route_a(ctx, bst.tmp0, bst.d_q_proj_out, w.w_q, OP_NT, m, qw, dm)
    _route_a(ctx, bst.tmp1, bst.d_k_proj_out, w.w_k, OP_NT, m, kw, dm)
    _route_a(ctx, bst.tmp2, bst.d_v_proj_out, w.w_v, OP_NT, m, kw, dm)
    ctx.enqueue_function[bwd_add3_kernel](
        bst.d_norm1_out.unsafe_ptr(),
        bst.tmp0.unsafe_ptr(),
        bst.tmp1.unsafe_ptr(),
        bst.tmp2.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 32, bst.d_norm1_out, m * dm)

    # =====================================================================
    # STAGE 33-35. `input_layernorm` backward. Its forward INPUT is the
    # BLOCK INPUT `x`, which the forward card records as `input.x` but which
    # `LlamaDeviceStages` DOES NOT HOLD as a buffer -- the forward launcher
    # uploads `x`, uses it, and lets it die.
    #
    # **THAT IS THIS BACKWARD'S ONE MISSING SAVED TENSOR AND IT IS NAMED
    # RATHER THAN WORKED AROUND.** `x_dev` is an explicit argument here, the
    # caller uploads it, and the caller owns it past its own synchronize. It
    # is NOT parked in a scratch field: `bst.tmp2` still holds the `dW_v`
    # route's `dA` term at this point in the call, and reusing it would have
    # silently read a gradient where a block input belongs -- a plausible,
    # in-bounds, wrong `dx`.
    #
    # CROSS-LANE REQUEST, in this lane's report: an `x` field on
    # `LlamaDeviceStages`, written by the forward launcher, would delete
    # this argument. That is a one-line edit to a file this lane may not
    # touch.
    # =====================================================================
    bwd_rms_norm(
        ctx,
        bst.norm1_dot,
        bst.norm1_dx,
        bst.dw_norm1,
        bst.dh,
        bst.dprod,
        bst.rstd,
        bst.dvcoef,
        bst.ones,
        bst.d_norm1_out,
        x_dev,
        w.norm1_w,
        fwd.norm1_sumsq,
        m,
        dm,
        w.eps,
    )
    _rec(ctx, trace, prefix, 33, bst.norm1_dot, m)
    _rec(ctx, trace, prefix, 34, bst.dw_norm1, dm)
    _rec(ctx, trace, prefix, 35, bst.norm1_dx, m * dm)

    # =====================================================================
    # STAGE 36. THE OUTPUT. `x` fans out into the norm (LDL:306) and into
    # the residual add (LDL:317); FORWARD-USE order puts the norm first.
    # =====================================================================
    ctx.enqueue_function[bwd_add2_kernel](
        bst.d_x.unsafe_ptr(),
        bst.norm1_dx.unsafe_ptr(),
        bst.d_residual1.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    _rec(ctx, trace, prefix, 36, bst.d_x, m * dm)
