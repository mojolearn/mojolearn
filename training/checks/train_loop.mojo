# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The composed training step of `mojolearn.identical.train.step.fp32.v1`.

**THIS BANNER WAS FALSE AND IS CORRECTED. COMPILED AND RUN, GREEN ON ONE
DEVICE, NOT ON THREE.** Until 2026-08-31 this header read "THIS FILE HAS NEVER
BEEN COMPILED AND HAS NEVER BEEN EXECUTED", and added that no compiler had
read it, no GPU had run a step from it, no checkpoint digest had ever been
computed by it, and therefore no two digests had ever been compared.
**Commit `5ce6eb17` falsified all four clauses in the same commit that added
this file.** It compiles; `train_step_check.mojo` drove a whole step through
it with all clauses green on one device, thirteen stages bitwise against the
host oracle; a digest was computed, computed twice with the same result, and a
checkpoint taken at step 4 and resumed through step 8 reproduced eight
continuous steps exactly. Written 2026-08-25 by the training-loop lane,
DEVIATIONS 1550 through 1589. The design is
`training/TRAINING_LOOP_PLAN.md` and the gate is
`training/checks/train_step_check.mojo`, which has run.

**WHAT IS STILL UNPAID: THE OTHER TWO VENDORS.** No two digests from DIFFERENT
VENDORS have been compared, and that comparison is the claim this file exists
to support. One device is one device.

WHAT THIS IS
------------
Every identical op in this repository is gated IN ISOLATION. This file is the
first thing that COMPOSES them, and composition is where the claim actually
lives: two ops that are each bitwise identical can still produce a divergent
run, through absorption into a range no fixture chose, through `m` and `v`
integrating a one-ULP gradient difference across steps, and through the
optimizer's `beta^t` spellings, which agree exactly through `t = 6` and first
differ at `t = 7`. A per-op gate cannot see any of the three.

WHAT IS OWED, and the first two are not optional
-------------------------------------------------
  1. **`transformer/checks/transformer_backward.mojo` HAS NO GATE** and it
     is stage 8 of twelve. Nothing has ever compared it to its oracle.
  2. **`embedding/checks/embedding_identical.mojo` HAS NO GATE** and it is
     stages 2 and 9. Its `PLAN_SORT` is additionally not written, so the one
     clause that would show its arithmetic does not read the execution plan
     cannot run at all.
     Consequence, stated in the plan's section 1.1 and repeated here because
     it is the sentence most likely to be skipped: **three matching
     checkpoint digests on a step containing two ungated ops show that those
     two ops agree across vendors. They do not show that either is CORRECT.**
     Three machines computing the same wrong gradient agree perfectly.
     Correctness lives in `train_step_check.mojo` clause (a), on ONE device,
     and it must be green BEFORE any GPU is rented.
  3. ~~This file has never been compiled.~~ **It compiles as of `5ce6eb17`.**
     Its header's section at the bottom is kept as the record of what the
     lane was least confident about before that compile; it is answered, not
     pending. What replaces this item is the cross-vendor run.
  4. `llama_decoder_layer_backward` takes its incoming gradient as a HOST
     `List[Float32]` (DEVIATION 1577), so the step downloads `d_h` and
     re-uploads it in the middle of every step. At this model size that is
     512 floats and irrelevant; at any real size it is a bus round trip in
     the hot path. A device-buffer form of that entry point is owed and is
     not in this lane's write set.
  5. `identical_optimizer_step` downloads `param`, `grad`, `m` and `v` every
     step for its non-finite refusal. 214 KB per step here, unaffordable at
     any real size, and `MOJOLEARN_OPT_TRUST_INPUTS=1` is a DELIBERATE
     downgrade of the profile rather than an optimization.
  6. A `checks/kernel_matrix.mojo` row for `TRAIN_TPB`. It is a literal
     128, matching `LLAMA_TPB`, and the portable baseline column's cap is
     also 128 -- so it is legal by coincidence rather than by construction.

ONE SOURCE, THREE VENDORS, AND NOTHING TO BRANCH ON
----------------------------------------------------
`[[always-gpu-agnostic]]`. There is no `if apple` here and there is nowhere
one could go. **The two kernels this file adds contain no arithmetic** --
`train_copy_range_kernel` moves bytes and `train_ulp_perturb_kernel` adds one
to an integer bit pattern. Every rounding decision in a step belongs to an op
that already has a contract, a card and (for four of them) a three-vendor
measurement. That is deliberate: a harness that introduced its own arithmetic
would be a harness that could fail the identity claim on its own account.

WHAT `NUMERIC_FAST` DOES HERE
------------------------------
Nothing, directly. This file contains no pinned primitive to compile away.
FAST changes what the ELEVEN composed ops do and leaves the loop, the digest
and the data generator unchanged, so a FAST run is a correct training run
that makes NO identity claim. `mode_banner()` reads the mode back out of the
binary rather than trusting the build command
(`[[mojolearn-shared-checkout-mode-flip]]`).

WHAT IS NOT CLAIMED
--------------------
"Bitwise identical" means GIVEN THE SAME DRAWS. Here the draws are a pure
function of one integer seed and the step index, so the condition holds by
construction -- and it stops holding the moment anything stochastic enters.
FP32 only. ONE decoder layer, one architecture, one shape. No claim about
batch splitting (the weight gradient's contraction length IS the token count,
so a split legitimately moves bits -- plan section 4.2 N7). No performance
number: a traced step drains the queue four times and the optimizer's refusal
downloads four buffers, so **any timing taken from this harness is fiction.**

WHAT THIS FILE IS LEAST CONFIDENT COMPILES
-------------------------------------------
  1. Passing a STRUCT FIELD as a `mut` argument (`tb.logits` into
     `identical_ce_forward_into`). `optimizer_oracle.mojo` avoids this
     deliberately and calls it a place-expression borrow it cannot check
     without a compiler. This file does it, following
     `transformer_backward.mojo`'s precedent (`_route_a(ctx,
     bst.d_mlp_gated, ...)`), because the alternative is a `train_step` with
     thirty-five parameters. **If it does not compile, that is where.**
  2. Pointer arithmetic on a host pointer (`hp.unsafe_ptr() + offsets[j]`)
     before `.bitcast[UInt8]()`, in `checkpoint_digest`.
  3. `bitcast[DType.uint32](Float32)` inside a device kernel.
     `transformer_backward.mojo` uses that spelling on the host;
     `loss_oracle.mojo` uses `rebind[UInt32](x.to_bits())` instead.
  4. `DeviceBuffer` lifetimes across the pack step, which reads
     `LlamaBackwardStages`' eleven gradient fields AFTER the backward
     returned.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import FNV_OFFSET, IdentityTrace, fnv1a64_bytes
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_backward import (
    identical_gemm_backward_a_into,
    identical_gemm_backward_b_into,
    identical_gemm_backward_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT

from embedding.checks.embedding_identical import (
    emb_run_scratch_ints,
    identical_embedding_backward_into,
    identical_embedding_forward_into,
)
from embedding.checks.embedding_oracle import EmbConfig

from training.checks.loss import (
    identical_ce_backward_into,
    identical_ce_forward_into,
    identical_ce_ones_floats,
    identical_ce_workspace_max_floats,
)
from training.checks.loss_oracle import REDUCTION_MEAN, CeConfig
from training.checks.optimizer import (
    SAB_CHUNKS,
    identical_optimizer_step,
    identical_optimizer_workspace_floats,
)
from training.checks.optimizer_oracle import OPT_ADAMW, OptimizerConfig

from transformer.impl.transformers.models.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    llama_decoder_layer_forward,
)
from transformer.checks.transformer_backward import (
    LlamaBackwardStages,
    llama_decoder_layer_backward,
)


# ===========================================================================
# THE FROZEN SHAPE (plan section 2)
# ===========================================================================
# None of these may change without a new profile version, because the
# checkpoint digest is a fold over a buffer whose LAYOUT they define.

comptime TRAIN_VOCAB = 64
comptime TRAIN_D_MODEL = 32
comptime TRAIN_N_HEADS = 4
comptime TRAIN_N_KV = 2
comptime TRAIN_HEAD_DIM = 8
comptime TRAIN_INTERMEDIATE = 64
comptime TRAIN_B = 2
comptime TRAIN_L = 8
comptime TRAIN_ROPE_POSITIONS = 64

comptime TRAIN_RMS_EPS: Float32 = 1e-6
"""`0x358637BD`. FROZEN by the transformer contract, section 3."""

comptime TRAIN_ROPE_THETA: Float32 = 10000.0
"""`0x461C4000`. FROZEN by the transformer contract, section 3."""

comptime TRAIN_TPB = 128
"""SCHEDULING. `LLAMA_TPB`'s value, and the only place this file reads a
block size. Both kernels below are elementwise byte moves, so nothing here
can reorder a float and this number cannot reach an arithmetic result. It is
a literal and that is OWED item 6, not a design."""

comptime TRAIN_J = 11
"""Parameter tensors. `identical_optimizer_step`'s `offsets` has `J + 1`
entries and `j` IS the `param_id`."""

comptime SEED_BASE: UInt64 = 0x547261696E4C6F70
"""ASCII-ish `TrainLop`. Distinct from `transformer_fixture`'s
`FIXTURE_SEED_BASE` and from `mamba_fixture`'s `CORPUS_SEED_BASE` on purpose:
two lanes sharing a seed base makes two different fixtures correlate, which
is harmless right up to the moment somebody compares a digest across lanes
and reads meaning into it."""


# ---- the arms (plan section 4.2). Harness arms, not kernel arms. ----------
# There is no `comptime` sabotage in this file and there is nothing here to
# sabotage: neither kernel moves a float. Every arm below describes a defect
# in the CALLER'S behavior, which is why it is an environment selection
# rather than a `-D`. `optimizer_check.mojo` made the same call for
# `OPT_SAB_RESUME_REINIT` and `OPT_SAB_MICROBATCH_SERIAL` (DEVIATION 1473).

comptime ARM_NONE = 0
comptime ARM_SEED_PLUS_ONE = 1
comptime ARM_DATA_REVERSE = 2
comptime ARM_ULP = 3
comptime ARM_ZERO_LR = 4
comptime ARM_CLIP_ON = 5


def arm_name(arm: Int) -> String:
    if arm == ARM_SEED_PLUS_ONE:
        return String("seed_plus_one")
    if arm == ARM_DATA_REVERSE:
        return String("data_reverse")
    if arm == ARM_ULP:
        return String("ulp")
    if arm == ARM_ZERO_LR:
        return String("zero_lr")
    if arm == ARM_CLIP_ON:
        return String("clip_on")
    return String("none")


def arm_by_name(name: String) raises -> Int:
    if name == "" or name == "none":
        return ARM_NONE
    if name == "seed_plus_one":
        return ARM_SEED_PLUS_ONE
    if name == "data_reverse":
        return ARM_DATA_REVERSE
    if name == "ulp":
        return ARM_ULP
    if name == "zero_lr":
        return ARM_ZERO_LR
    if name == "clip_on":
        return ARM_CLIP_ON
    raise Error(
        String("train_loop: unknown MOJOLEARN_TRAIN_ARM '")
        + name
        + "'. Known arms: none, seed_plus_one, data_reverse, ulp, zero_lr,"
        + " clip_on. **A misspelled arm name must NEVER be treated as"
        + " 'none'**: the operator would then record a clean run as 'the"
        + " arm did not bite', which is the exact inverse of the truth."
        + " That is tools/gemm_ladder.sh:71's scar, ported to an env var."
    )


# ===========================================================================
# ENVIRONMENT, HAND PARSED
# ===========================================================================
# `[[mojo-string-float-roundtrip]]` and the string traps: `s[:n]` is refused
# and `len(String)` is unsupported. The integer parser below walks bytes
# rather than calling anything this lane has not seen compiled.


def env_str(name: String) -> String:
    return String(getenv(name))


def env_int(name: String, fallback: Int) raises -> Int:
    """A non-negative decimal integer, or `fallback` when unset.

    REFUSES a non-numeric value rather than falling back to the default. A
    typo in `MOJOLEARN_TRAIN_STEPS` that silently ran the default step count
    would produce a green run at the wrong `N`, and `N` is what makes the
    step-count clauses non-vacuous.
    """
    var s = env_str(name)
    var b = s.as_bytes()
    if len(b) == 0:
        return fallback
    var acc = 0
    for i in range(len(b)):
        var c = Int(b[i])
        if c < 48 or c > 57:
            raise Error(
                String("train_loop: ")
                + name
                + " must be a non-negative decimal integer, got '"
                + s
                + "'"
            )
        acc = acc * 10 + (c - 48)
    return acc


def env_u64(name: String, fallback: UInt64) raises -> UInt64:
    var s = env_str(name)
    var b = s.as_bytes()
    if len(b) == 0:
        return fallback
    var acc = UInt64(0)
    for i in range(len(b)):
        var c = Int(b[i])
        if c < 48 or c > 57:
            raise Error(
                String("train_loop: ")
                + name
                + " must be a non-negative decimal integer, got '"
                + s
                + "'"
            )
        # `+` and `*` here are REAL arithmetic on UInt64 and wrap. See
        # `train_splitmix64` for why that sentence is written down.
        acc = acc * UInt64(10) + UInt64(c - 48)
    return acc


def mode_banner() -> String:
    """Read the numeric mode BACK OUT of the binary.

    `[[mojolearn-shared-checkout-mode-flip]]`: a mode flip with no lock gives
    correctly-labelled measurements of the WRONG arm. The build command is
    not evidence; this is.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("NUMERIC_IDENTICAL")
    return String("NUMERIC_FAST")


def hex16(v: UInt64) -> String:
    """Sixteen lowercase hex digits, zero padded, most significant first.

    `core/identity_trace.mojo::_hex16`'s spelling, restated rather than
    imported because that name is private to that module. The two must agree
    and `train_step_check.mojo` clause (e) asserts they do on a fixed value.
    """
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out^


def step_tag(step: Int) -> String:
    """`s0007`. FOUR DIGITS, ZERO PADDED (DEVIATION 1575).

    `tools/identity_trace_diff.py` aligns two traces by their TAG SEQUENCES,
    so the tags must sort the same way they run. `s10` sorting before `s2` is
    how a differ pairs one run's step 10 against another run's step 2 and
    reports a plausible, wrong answer.

    Refuses above 9999 rather than emitting a five-digit tag that breaks the
    alignment silently.
    """
    var s = step
    if s < 0:
        s = 0
    var out = String("s")
    comptime DIGITS = "0123456789"
    out += String(DIGITS[byte=(s // 1000) % 10])
    out += String(DIGITS[byte=(s // 100) % 10])
    out += String(DIGITS[byte=(s // 10) % 10])
    out += String(DIGITS[byte=s % 10])
    return out^


# ===========================================================================
# THE DATA GENERATOR (plan section 6, DEVIATIONS 1556 and 1565)
# ===========================================================================


def train_splitmix64(z_in: UInt64) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim.

    **EVERY `+` AND `*` HERE IS A REAL OPERATION AND MUST STAY ONE.**
    `[[mojo-amp-plus-is-bitwise-and]]`: Mojo's `x &+ k` computes `x & k` with
    NO COMPILE ERROR, and it produced wrong hashes in this repository twice
    on 2026-08-25 alone. Do not "fix" any of these into `&+`. DEVIATION 1565.

    COPIED rather than imported from
    `transformer/checks/transformer_fixture.mojo::fixture_splitmix64`,
    which is DEVIATION 1000's own decision made again for the same reason --
    the alternative points a `training/ -> transformer/checks/` arrow at a
    lane under concurrent edit, for a function that is exact integer
    arithmetic and therefore cannot drift the way a float seam can. The cost
    if wrong is duplication, and `train_step_check.mojo` clause (e) owes one
    assertion that the two copies agree on a handful of inputs.
    """
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def train_batch_ids(seed: UInt64, step_index: Int) -> List[Int32]:
    """`B * (L + 1)` token ids for one step. NO CLOCK, NO ADDRESS, NO PID.

    Each of the `B` rows draws `L + 1` ids. The first `L` of a row are the
    inputs and the last `L` are the next-token targets, so **no row is ever
    ignored** and `count == M` exactly. That keeps `ce_divisor` on its simple
    arm and keeps `ignore_index` off the measured path (DEVIATION 1556).

    `>> 33` takes the HIGH bits before the modulus. splitmix64's low bits are
    the weaker half, and `% 64` reads exactly six of them.
    """
    var key = train_splitmix64(
        seed ^ (UInt64(step_index) * UInt64(0x9E3779B97F4A7C15))
    )
    var n = TRAIN_B * (TRAIN_L + 1)
    var out = List[Int32]()
    for i in range(n):
        var h = train_splitmix64(key + UInt64(i))
        out.append(Int32(Int((h >> 33) % UInt64(TRAIN_VOCAB))))
    return out^


def batch_inputs(ids: List[Int32]) -> List[Int32]:
    """The first `L` of each row, flattened token-major. `[M]`."""
    var out = List[Int32]()
    for r in range(TRAIN_B):
        for t in range(TRAIN_L):
            out.append(ids[r * (TRAIN_L + 1) + t])
    return out^


def batch_targets(ids: List[Int32]) -> List[Int32]:
    """The last `L` of each row, flattened token-major. `[M]`."""
    var out = List[Int32]()
    for r in range(TRAIN_B):
        for t in range(TRAIN_L):
            out.append(ids[r * (TRAIN_L + 1) + t + 1])
    return out^


def effective_step(step: Int, n_steps: Int, arm: Int) -> Int:
    """Which step's batch this step trains on.

    The `data_reverse` arm presents the same `N` batches in reverse ORDER
    (plan N2). **At `N == 1` reversing is the identity and the arm cannot
    fire**; the caller refuses that combination rather than reporting a pass
    (DEVIATION 1560), and this function does not know about it.
    """
    if arm == ARM_DATA_REVERSE:
        return n_steps + 1 - step
    return step


# ===========================================================================
# THE PARAMETER LAYOUT (plan section 2, DEVIATION 1550)
# ===========================================================================
# **THIS ORDER IS PART OF THE CHECKPOINT DIGEST SPECIFICATION.** `j` IS the
# `param_id` and its ascending order is the optimizer clip's cross-tensor
# summation order (optimizer contract 3.3). Changing it invalidates every
# digest this profile has ever produced.

comptime PID_EMBED = 0
comptime PID_NORM1 = 1
comptime PID_W_Q = 2
comptime PID_W_K = 3
comptime PID_W_V = 4
comptime PID_W_O = 5
comptime PID_NORM2 = 6
comptime PID_W_GATE = 7
comptime PID_W_UP = 8
comptime PID_W_DOWN = 9
comptime PID_LM_HEAD = 10


def param_id_name(j: Int) -> String:
    if j == PID_EMBED:
        return String("embed")
    if j == PID_NORM1:
        return String("norm1_w")
    if j == PID_W_Q:
        return String("w_q")
    if j == PID_W_K:
        return String("w_k")
    if j == PID_W_V:
        return String("w_v")
    if j == PID_W_O:
        return String("w_o")
    if j == PID_NORM2:
        return String("norm2_w")
    if j == PID_W_GATE:
        return String("w_gate")
    if j == PID_W_UP:
        return String("w_up")
    if j == PID_W_DOWN:
        return String("w_down")
    if j == PID_LM_HEAD:
        return String("lm_head")
    return String("?")


def param_id_count(j: Int) -> Int:
    """Element count of tensor `j` at the frozen shape."""
    comptime DM = TRAIN_D_MODEL
    comptime QW = TRAIN_N_HEADS * TRAIN_HEAD_DIM
    comptime KW = TRAIN_N_KV * TRAIN_HEAD_DIM
    comptime IT = TRAIN_INTERMEDIATE
    if j == PID_EMBED:
        return TRAIN_VOCAB * DM
    if j == PID_NORM1:
        return DM
    if j == PID_W_Q:
        return QW * DM
    if j == PID_W_K:
        return KW * DM
    if j == PID_W_V:
        return KW * DM
    if j == PID_W_O:
        return DM * QW
    if j == PID_NORM2:
        return DM
    if j == PID_W_GATE:
        return IT * DM
    if j == PID_W_UP:
        return IT * DM
    if j == PID_W_DOWN:
        return DM * IT
    if j == PID_LM_HEAD:
        return TRAIN_VOCAB * DM
    return 0


def train_offsets() -> List[Int]:
    """`offsets[j] .. offsets[j+1]` is tensor `j`. Length `J + 1`."""
    var out = List[Int]()
    var acc = 0
    out.append(0)
    for j in range(TRAIN_J):
        acc += param_id_count(j)
        out.append(acc)
    return out^


def train_n_total() -> Int:
    var acc = 0
    for j in range(TRAIN_J):
        acc += param_id_count(j)
    return acc


def train_param_range(seed: UInt64, j: Int) -> Tuple[Float64, Float64]:
    """`(lo, hi)` for tensor `j`'s initialization.

    **EVERY RANGE IS DYADIC**, so `lo + (hi - lo) * top24 * 2^-24` is exact
    in float64 and the only rounding is the final cast to Float32. That is
    `transformer_fixture.fixture_tensor`'s stated property, restated here
    because it becomes FALSE SILENTLY if anyone changes a range to something
    like 0.1. `train_step_check.mojo` owes that assertion.

    The ranges differ per tensor so that a weight read out of the wrong
    buffer is usually out of RANGE as well as out of place
    (`[[uniform-test-data-hides-permutation]]`).
    """
    _ = seed
    if j == PID_NORM1 or j == PID_NORM2:
        return (0.5, 1.5)
    if j == PID_EMBED:
        return (-1.0, 1.0)
    if j == PID_W_Q or j == PID_W_K or j == PID_W_V:
        return (-0.5, 0.5)
    if j == PID_W_O or j == PID_W_GATE or j == PID_W_UP:
        return (-0.25, 0.25)
    if j == PID_W_DOWN:
        return (-0.125, 0.125)
    return (-0.25, 0.25)


def train_init_tensor(seed: UInt64, j: Int) -> List[Float32]:
    """Tensor `j`'s initial values, in FLAT ROW-MAJOR index order.

    Distinct per `(tensor, index)` by construction, so a transposed or
    mis-strided read of any weight lands on a different number and shows up
    in the loss. A uniform initialization destroys that property and would
    make clause (f)'s offset round trip pass with every offset wrong.
    """
    var key = train_splitmix64(seed ^ (UInt64(j + 1) << 32))
    var r = train_param_range(seed, j)
    var span = r[1] - r[0]
    var n = param_id_count(j)
    var out = List[Float32]()
    for i in range(n):
        var h = train_splitmix64(key + UInt64(i))
        # 2^-24, written as its exact decimal so no rounding hides here.
        var unit = Float64(Int(h >> 40)) * 0.000000059604644775390625
        out.append(Float32(r[0] + span * unit))
    return out^


def train_init_params(seed: UInt64) -> List[Float32]:
    """The whole flat parameter vector at step 0, in `param_id` order."""
    var out = List[Float32]()
    for j in range(TRAIN_J):
        var t = train_init_tensor(seed, j)
        for i in range(len(t)):
            out.append(t[i])
    return out^


# ===========================================================================
# THE TWO KERNELS. NEITHER CONTAINS ARITHMETIC.
# ===========================================================================


def _grid(n: Int) -> Int:
    """Blocks to cover `n` items at `TRAIN_TPB`. SCHEDULING, and the only
    place this file reads a block size."""
    if n < 1:
        return 1
    return (n + TRAIN_TPB - 1) // TRAIN_TPB


def train_copy_range_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    dst_off: Int32,
    src_off: Int32,
    count_in: Int32,
):
    """`dst[dst_off + i] = src[src_off + i]`, one thread per element.

    **NO ARITHMETIC. NOT ONE FLOAT OPERATION.** The value read is the value
    stored, bit for bit, with no flush, no cast and no combine. A block size
    cannot reorder a copy and there is no accumulator to seed, so this kernel
    is launch invariant by its SHAPE rather than by a check that happens to
    pass. It is also why this file adds no sabotage arm: an arm that moves no
    float is a comment.

    **A WRONG OFFSET IS THE DANGEROUS DEFECT HERE**, and it produces
    plausible, in-bounds, wrong numbers that are IDENTICAL on all three
    vendors -- so no cross-vendor comparison can see it. `train_step_check`
    clause (f) is the assertion that does.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count_in):
        return
    dst.unsafe_store(Int(dst_off) + i, src.unsafe_load(Int(src_off) + i))


def train_ulp_perturb_kernel(
    buf: MutPointer[Float32, MutAnyOrigin],
    index: Int32,
    ulps: Int32,
):
    """Replace `buf[index]` by its one-ULP neighbor AWAY FROM ZERO.

    ON THE BITS and never by adding a small float. `bits + 1` increases the
    magnitude for both signs, because the sign lives above the twenty-three
    mantissa bits and the eight exponent bits that `+1` walks. Adding
    `1.2e-7` instead would be a float operation whose result depends on the
    value's binade, would be a no-op on a large enough value, and would put
    an arithmetic decision inside a control.

    Plan N3. One thread, deliberately -- the perturbation must hit exactly
    one cell and a grid that covered more would be a different control.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var i = Int(index)
    var u = bitcast[DType.uint32](buf.unsafe_load(i))
    # DEVIATION 1579: THE MAGNITUDE IS A PARAMETER BECAUSE ONE ULP WAS NOT
    # ENOUGH, AND WHICH REASON IT WAS NOT IS THE WHOLE QUESTION.
    #
    # First run, 2026-08-25: perturbing the last float of `w_v` by one ULP
    # moved the CHECKPOINT and did NOT move step 1's loss (both
    # 4.7812595 / 0x40990014). N3 raised, correctly. Two causes are possible
    # and they are opposite in severity:
    #
    #   (a) that element is never READ -- an off-by-one that drops a
    #       tensor's tail, which is exactly what N3 was positioned to catch;
    #   (b) one ULP of one weight is ABSORBED on the way to the loss,
    #       through v, attention, o_proj, the residual, the norm, the MLP,
    #       the head and a softmax. Absorption has already produced four
    #       false negatives in this repository tonight.
    #
    # A control that cannot distinguish them is not a control. `ulps` makes
    # the arm a SWEEP: if the loss moves at a larger magnitude, it is (b) and
    # the perturbation was too small; if it never moves at any magnitude, it
    # is (a) and the weight is dead. The DEFAULT STAYS 1, so the strict
    # form is what a normal run asserts.
    var d = UInt32(1)
    if ulps > Int32(0):
        d = UInt32(Int(ulps))
    buf.unsafe_store(i, bitcast[DType.float32](u + d))


# ===========================================================================
# THE CHECKPOINT DIGEST (plan section 3, DEVIATION 1551)
# spec `mojolearn.identical.train.ckpt.v1`
# ===========================================================================


struct TrainDigest(Movable):
    """One checkpoint's five digests.

    **A PURE FUNCTION OF THE PARAMETER AND OPTIMIZER-STATE BITS AND OF
    NOTHING ELSE.** Not of `t`, not of the clock, not of the device, not of
    the block size, not of the plan `choose_gemm_plan` picked, not of the
    learning rate, not of the seed, and NOT OF ANY BUFFER CAPACITY -- every
    count below is `n_total` or a tensor extent, never `len(buf)`. That last
    one is `core/identity_trace.mojo` rule 3 and it is not a hypothetical: a
    buffer allocated with slack and hashed to its capacity folds
    uninitialized memory into the digest, which differs run to run on ONE
    machine and would make the instrument report divergence everywhere.
    """

    var h_param: UInt64
    var h_m: UInt64
    var h_v: UInt64
    var h_all: UInt64
    var h_tensor: List[UInt64]

    def __init__(out self):
        self.h_param = UInt64(0)
        self.h_m = UInt64(0)
        self.h_v = UInt64(0)
        self.h_all = UInt64(0)
        self.h_tensor = List[UInt64]()

    def words(self) -> List[UInt32]:
        """`[h_param, h_m, h_v, h_all]` as eight `UInt32`, high word first.

        RECORDED AS `u32` AND NOT AS `u64` (DEVIATION 1567), for a reason
        found by reading `core/identity_trace.mojo::_dtype_name`: it handles
        `int64` and NOT `uint64`, so a `u64` record would emit the dtype `?`
        and `tools/identity_trace_diff.py`'s parser accepts exactly six
        names. Splitting into high and low `u32` words needs no conversion
        that could trap and no dtype the differ does not know.

        And never as TEXT. `String(Float32)` does not round trip in this
        toolchain (`[[mojo-string-float-roundtrip]]`), and although these are
        integers, a digest that went through a decimal round trip anywhere
        could report agreement across a real difference -- the worst failure
        an instrument of this kind can have.
        """
        var out = List[UInt32]()
        out.append(UInt32((self.h_param >> UInt64(32)) & UInt64(0xFFFFFFFF)))
        out.append(UInt32(self.h_param & UInt64(0xFFFFFFFF)))
        out.append(UInt32((self.h_m >> UInt64(32)) & UInt64(0xFFFFFFFF)))
        out.append(UInt32(self.h_m & UInt64(0xFFFFFFFF)))
        out.append(UInt32((self.h_v >> UInt64(32)) & UInt64(0xFFFFFFFF)))
        out.append(UInt32(self.h_v & UInt64(0xFFFFFFFF)))
        out.append(UInt32((self.h_all >> UInt64(32)) & UInt64(0xFFFFFFFF)))
        out.append(UInt32(self.h_all & UInt64(0xFFFFFFFF)))
        return out^

    def tensor_words(self) -> List[UInt32]:
        """`h_tensor[0..J-1]` as `2*J` `UInt32`, high word first."""
        var out = List[UInt32]()
        for j in range(len(self.h_tensor)):
            var h = self.h_tensor[j]
            out.append(UInt32((h >> UInt64(32)) & UInt64(0xFFFFFFFF)))
            out.append(UInt32(h & UInt64(0xFFFFFFFF)))
        return out^


def download_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """`n` elements of a device buffer, on the host.

    `n` IS PASSED and is never `len(buf)`. Same rule as the digest and the
    same reason.
    """
    var out = List[Float32]()
    if n < 1:
        return out^
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view
    ctx.synchronize()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    # `[[mojo-buffer-freed-at-last-use]]`: `h` is dead at its
    # `.unsafe_ptr()`, and a freed host buffer under an in-flight read reads
    # garbage. Keep it alive past the loop.
    _ = h
    return out^


def digest_of_lists(
    param: List[Float32],
    m_state: List[Float32],
    v_state: List[Float32],
    offsets: List[Int],
) raises -> TrainDigest:
    """The digest, computed from HOST lists.

    THE FUNCTION IS FNV-1a64 -- offset basis `0xCBF29CE484222325`, prime
    `0x100000001B3`, folded ONE BYTE AT A TIME over the LITTLE-ENDIAN bytes
    of the elements IN INDEX ORDER. It is
    `core/identity_trace.mojo::fnv1a64_bytes`, CALLED and not restated, so
    that a digest and a trace record mean the same kind of thing and so that
    `tools/identity_trace_diff.py` recomputing from a `.bin` dump agrees.
    Byte at a time on purpose: a word-at-a-time variant is faster and is a
    DIFFERENT FUNCTION.

    THE ORDER, and it is the specification.

        h_param  = h(FNV_OFFSET, param,   n_total)
        h_m      = h(FNV_OFFSET, m_state, n_total)
        h_v      = h(FNV_OFFSET, v_state, n_total)
        h_all    = h(h(h(FNV_OFFSET, param, n), m_state, n), v_state, n)
        h_tensor[j] = h(FNV_OFFSET, param + offsets[j], count_j)

    `h_all` is the CHAINED fold -- one continuous byte stream, param then `m`
    then `v` -- and it is the single number two machines compare. The other
    four exist so a mismatch has an ADDRESS instead of a verdict. That is
    `core/identity_trace.mojo`'s whole argument ("a claim that can only be
    checked at the END is a claim nobody can debug") applied to a run rather
    than to a fit.

    Optimizer state is IN the digest because a checkpoint you can resume from
    IS param plus `m` plus `v`, and because an `m`/`v` divergence is a real
    composition failure that the parameters can mask for a step or two
    (DEVIATION 1571). It is ALSO separately addressed, for the same reason.
    """
    var n = len(param)
    if len(m_state) != n or len(v_state) != n:
        raise Error(
            String("train_loop: digest needs param, m and v the same")
            + " length, got "
            + String(n)
            + "/"
            + String(len(m_state))
            + "/"
            + String(len(v_state))
        )
    var d = TrainDigest()
    # DEVIATION 1578. `fnv1a64_bytes` takes `o: MutOrigin`, and these three
    # arrive BORROWED, so `param.unsafe_ptr()` carries an immutable origin
    # and does not unify. The repository's existing answer is
    # `bench/gemm_ladder_main.mojo::_hash_f32`, which copies into a local
    # `var` for exactly this reason; a local binding is what gives a mutable
    # origin. The copy is real -- three lists of `n_total` floats per digest
    # per step -- and it is paid here rather than making these `mut`, because
    # `mut` on a function that only READS would say something false about
    # what it does. The alternative is an immutable-origin overload of
    # `fnv1a64_bytes` in `core/identity_trace.mojo`, which is OWED and would
    # delete the copy at every call site in the tree.
    var pcopy = param.copy()
    var mcopy = m_state.copy()
    var vcopy = v_state.copy()
    var pp = pcopy.unsafe_ptr().bitcast[UInt8]()
    var mp = mcopy.unsafe_ptr().bitcast[UInt8]()
    var vp = vcopy.unsafe_ptr().bitcast[UInt8]()
    var nb = n * 4

    d.h_param = fnv1a64_bytes(FNV_OFFSET, pp, nb)
    d.h_m = fnv1a64_bytes(FNV_OFFSET, mp, nb)
    d.h_v = fnv1a64_bytes(FNV_OFFSET, vp, nb)
    var chained = fnv1a64_bytes(FNV_OFFSET, pp, nb)
    chained = fnv1a64_bytes(chained, mp, nb)
    chained = fnv1a64_bytes(chained, vp, nb)
    d.h_all = chained

    var jc = len(offsets) - 1
    for j in range(jc):
        var begin = offsets[j]
        var count = offsets[j + 1] - begin
        # DEVIATION 1578 again: a borrowed `param` gives an immutable origin.
        # `pcopy` is the mutable local the whole-parameter digest above
        # already hashes, so the per-tensor digests read THE SAME BYTES by
        # construction rather than by two pointers happening to agree -- which
        # is what makes a per-tensor mismatch a real address into the whole.
        var base = (pcopy.unsafe_ptr() + begin).bitcast[UInt8]()
        d.h_tensor.append(fnv1a64_bytes(FNV_OFFSET, base, count * 4))

    # Keep the owners alive past the last pointer use. The COPIES are the
    # ones the pointers came from, so they are the ones that must survive.
    _ = pcopy
    _ = mcopy
    _ = vcopy
    _ = param
    _ = m_state
    _ = v_state
    return d^


def record_checkpoint(
    mut trace: IdentityTrace,
    step: Int,
    d: TrainDigest,
) raises:
    """Two records. Tags name a POSITION IN THE ALGORITHM and carry no
    property of the machine (`core/identity_trace.mojo` rule 2)."""
    var pre = String("train.") + step_tag(step) + "."
    var tw = d.tensor_words()
    trace.record_host(
        pre + "ckpt.tensors", tw.unsafe_ptr(), len(tw)
    )
    _ = tw
    var dw = d.words()
    trace.record_host(
        pre + "ckpt.digest", dw.unsafe_ptr(), len(dw)
    )
    _ = dw


# ===========================================================================
# THE RUN CONFIGURATION
# ===========================================================================


struct TrainConfig(Copyable, Movable):
    var steps: Int
    var seed: UInt64
    var arm: Int
    var lr: Float32
    var weight_decay: Float32
    var max_norm: Float32

    def __init__(out self):
        self.steps = 8
        self.seed = SEED_BASE
        self.arm = ARM_NONE
        self.lr = Float32(1e-3)
        self.weight_decay = Float32(0.01)
        self.max_norm = Float32(0.0)

    @staticmethod
    def for_arm(steps: Int, seed: UInt64, arm: Int) raises -> Self:
        """The clean configuration, then the arm's departure from it.

        `weight_decay` is NONZERO on the clean arm on purpose (plan N5.4).
        With `weight_decay = 0` the embedding rows for tokens absent from the
        batch would be EXACTLY frozen, and the per-tensor movement clause
        would be satisfied by the present rows alone -- so a completely inert
        embedding gradient would pass it.

        Clipping is OFF on the clean arm (DEVIATION 1558). With clipping ON
        one parameter's update is a function of every gradient in the model
        (optimizer contract 3.5), which is the reference's own semantics and
        not a defect, but it couples every tensor to every other and destroys
        the per-tensor localization `h_tensor` exists to give.
        """
        var c = Self()
        c.steps = steps
        c.seed = seed
        c.arm = arm
        if arm == ARM_SEED_PLUS_ONE:
            c.seed = seed + UInt64(1)
        if arm == ARM_ZERO_LR:
            c.lr = Float32(0.0)
            c.weight_decay = Float32(0.0)
        if arm == ARM_CLIP_ON:
            c.max_norm = Float32(1.0)
        if arm == ARM_DATA_REVERSE and steps < 2:
            raise Error(
                String("train_loop: the data_reverse arm is VACUOUS at N = ")
                + String(steps)
                + ". Reversing a one-element sequence is the identity, so"
                + " the arm cannot fire and a green result would mean"
                + " nothing. REFUSED rather than reported as a pass"
                + " (DEVIATION 1560). Run it at N >= 2."
            )
        return c^

    def optimizer(self) -> OptimizerConfig:
        """AdamW. `t` is ONE-BASED; the first step of a run is `t = 1`."""
        return OptimizerConfig(
            OPT_ADAMW,
            self.lr,
            Float32(0.9),
            Float32(0.999),
            Float32(1e-8),
            self.weight_decay,
            Float32(0.0),
            Float32(0.0),
            False,
            self.max_norm,
        )


def train_dims() -> LlamaDims:
    return LlamaDims(
        TRAIN_D_MODEL,
        TRAIN_N_HEADS,
        TRAIN_N_KV,
        TRAIN_HEAD_DIM,
        TRAIN_INTERMEDIATE,
    )


# ===========================================================================
# THE BUFFERS
# ===========================================================================


def _zeros(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var k = n
    if k < 1:
        k = 1
    var b = ctx.enqueue_create_buffer[DType.float32](k)
    ctx.enqueue_memset(b, Float32(0.0))
    ctx.synchronize()
    return b^


def _zeros_i32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.int32]:
    var k = n
    if k < 1:
        k = 1
    var b = ctx.enqueue_create_buffer[DType.int32](k)
    ctx.enqueue_memset(b, Int32(0))
    ctx.synchronize()
    return b^


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    if n < 1:
        return _zeros(ctx, 1)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, values[i])
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h
    return d^


def _upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    if n < 1:
        return _zeros_i32(ctx, 1)
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, values[i])
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h
    return d^


def _ones(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """EXACTLY `Float32(1.0)` in every entry.

    **A WRONG VALUE HERE IS A WRONG ANSWER WITH NO SYMPTOM**, because any
    vector produces a plausible weighted sum. `identical_ce_ones_floats`'s
    own warning, and the reason the ones are built here rather than assumed.
    """
    var vals = List[Float32]()
    var k = n
    if k < 1:
        k = 1
    for _ in range(k):
        vals.append(Float32(1.0))
    return _upload(ctx, vals)


struct TrainBuffers(Movable):
    """Everything the step touches that is not owned by a lane's own struct.

    **PASSING THESE FIELDS AS `mut` ARGUMENTS is this file's single largest
    compile risk** and the header says so. `optimizer_oracle.mojo` avoids the
    place-expression borrow deliberately; `transformer_backward.mojo` relies
    on it (`_route_a(ctx, bst.d_mlp_gated, ...)`). This file follows the
    latter, because the alternative is a `train_step` with thirty-five
    parameters, and it is named as the first thing to look at if the build
    fails.
    """

    var n_total: Int
    var offsets: List[Int]

    # ---- optimizer state and scratch -----------------------------------
    var param: DeviceBuffer[DType.float32]
    var grad: DeviceBuffer[DType.float32]
    var m_state: DeviceBuffer[DType.float32]
    var v_state: DeviceBuffer[DType.float32]
    var denom_out: DeviceBuffer[DType.float32]
    var q_out: DeviceBuffer[DType.float32]
    var sumsq: DeviceBuffer[DType.float32]
    var norms: DeviceBuffer[DType.float32]
    var total_cell: DeviceBuffer[DType.float32]
    var out2: DeviceBuffer[DType.float32]
    var opt_ws: DeviceBuffer[DType.float32]
    var sab_partials: DeviceBuffer[DType.float32]

    # ---- the two tensors the llama structs do not own -------------------
    var emb_w: DeviceBuffer[DType.float32]  # [V, d_model]
    var lm_w: DeviceBuffer[DType.float32]  # [V, d_model]
    var dw_emb: DeviceBuffer[DType.float32]  # [V, d_model]
    var dw_lm: DeviceBuffer[DType.float32]  # [V, d_model]

    # ---- data ------------------------------------------------------------
    var ids: DeviceBuffer[DType.int32]  # [M]
    var targets: DeviceBuffer[DType.int32]  # [M]

    # ---- activations ------------------------------------------------------
    var x: DeviceBuffer[DType.float32]  # [M, d_model]  block input
    var logits: DeviceBuffer[DType.float32]  # [M, V]
    var d_h: DeviceBuffer[DType.float32]  # [M, d_model]

    # ---- cross entropy ----------------------------------------------------
    var ce_max: DeviceBuffer[DType.float32]
    var ce_shift: DeviceBuffer[DType.float32]
    var ce_expo: DeviceBuffer[DType.float32]
    var ce_denom: DeviceBuffer[DType.float32]
    var ce_logdenom: DeviceBuffer[DType.float32]
    var ce_logp_target: DeviceBuffer[DType.float32]
    var ce_nll: DeviceBuffer[DType.float32]
    var ce_logp: DeviceBuffer[DType.float32]
    var ce_logp_sum: DeviceBuffer[DType.float32]
    var ce_smooth: DeviceBuffer[DType.float32]
    var ce_row: DeviceBuffer[DType.float32]
    var ce_total: DeviceBuffer[DType.float32]
    var ce_loss: DeviceBuffer[DType.float32]
    var ce_weights: DeviceBuffer[DType.float32]
    var ce_dlogits: DeviceBuffer[DType.float32]
    var ce_ones: DeviceBuffer[DType.float32]
    var ce_ws: DeviceBuffer[DType.float32]

    # ---- gemm workspaces ---------------------------------------------------
    var head_ws: DeviceBuffer[DType.float32]
    var head_bwd_ws: DeviceBuffer[DType.float32]

    # ---- embedding backward run scratch ------------------------------------
    var emb_counts: DeviceBuffer[DType.int32]
    var emb_run_begin: DeviceBuffer[DType.int32]
    var emb_perm: DeviceBuffer[DType.int32]

    var buf_initialized: List[Bool]

    def __init__(out self, ctx: DeviceContext, seed: UInt64) raises:
        comptime M = TRAIN_B * TRAIN_L
        comptime DM = TRAIN_D_MODEL
        comptime V = TRAIN_VOCAB

        self.offsets = train_offsets()
        self.n_total = train_n_total()
        var n = self.n_total

        self.param = _upload(ctx, train_init_params(seed))
        self.grad = _zeros(ctx, n)
        self.m_state = _zeros(ctx, n)
        self.v_state = _zeros(ctx, n)
        # `denom_out` and `q_out` are written only under
        # `MOJOLEARN_OPT_RECORD`, and the pointers are in the signature
        # either way so that recording and non-recording builds are ONE
        # kernel with ONE signature. One element each here.
        self.denom_out = _zeros(ctx, 1)
        self.q_out = _zeros(ctx, 1)
        self.sumsq = _zeros(ctx, TRAIN_J)
        self.norms = _zeros(ctx, TRAIN_J)
        self.total_cell = _zeros(ctx, 1)
        self.out2 = _zeros(ctx, 2)
        # **SIZED WITH THE HELPER AND NEVER GUESSED.** A workspace sized for
        # one plan while the dispatcher picks another is an out-of-bounds
        # write a small shape will not show you; it cost the gemm lane a run.
        self.opt_ws = _zeros(
            ctx, identical_optimizer_workspace_floats(self.offsets)
        )
        self.sab_partials = _zeros(ctx, SAB_CHUNKS)

        self.emb_w = _zeros(ctx, V * DM)
        self.lm_w = _zeros(ctx, V * DM)
        self.dw_emb = _zeros(ctx, V * DM)
        self.dw_lm = _zeros(ctx, V * DM)

        self.ids = _zeros_i32(ctx, M)
        self.targets = _zeros_i32(ctx, M)

        self.x = _zeros(ctx, M * DM)
        self.logits = _zeros(ctx, M * V)
        self.d_h = _zeros(ctx, M * DM)

        self.ce_max = _zeros(ctx, M)
        self.ce_shift = _zeros(ctx, M * V)
        self.ce_expo = _zeros(ctx, M * V)
        self.ce_denom = _zeros(ctx, M)
        self.ce_logdenom = _zeros(ctx, M)
        self.ce_logp_target = _zeros(ctx, M)
        self.ce_nll = _zeros(ctx, M)
        # `eps == 0` selects a DIFFERENT CODE PATH rather than a bit-inert
        # arm (loss contract 6.2(c)), so these three are never touched and
        # are one-element placeholders, which the launcher permits by name.
        self.ce_logp = _zeros(ctx, 1)
        self.ce_logp_sum = _zeros(ctx, 1)
        self.ce_smooth = _zeros(ctx, 1)
        self.ce_row = _zeros(ctx, M)
        self.ce_total = _zeros(ctx, 1)
        self.ce_loss = _zeros(ctx, 1)
        self.ce_weights = _zeros(ctx, M * V)
        self.ce_dlogits = _zeros(ctx, M * V)
        self.ce_ones = _ones(ctx, identical_ce_ones_floats(M, V))
        # REDUCTION_MEAN, named and not spelled as the literal 2. A wrong
        # reduction here would size the workspace for a shape the dispatcher
        # does not pick, which is an out-of-bounds write rather than a wrong
        # answer.
        self.ce_ws = _zeros(
            ctx, identical_ce_workspace_max_floats(M, V, REDUCTION_MEAN)
        )

        self.head_ws = _zeros(
            ctx, identical_gemm_workspace_max_floats(M, V, DM)
        )
        self.head_bwd_ws = _zeros(
            ctx,
            identical_gemm_backward_workspace_max_floats(
                OP_NT, M, V, DM, False
            ),
        )

        # `emb_run_scratch_ints` is the embedding lane's own total, and the
        # three buffers below are its three pieces. They are allocated
        # SEPARATELY because `identical_embedding_backward_into` takes three
        # buffers, and the total is checked against the pieces so that a
        # change to that helper cannot silently leave one of them short --
        # which is the out-of-bounds class its own docstring warns about.
        var scratch = emb_run_scratch_ints(V, M)
        if scratch != V + (V + 1) + M:
            raise Error(
                String("train_loop: emb_run_scratch_ints says ")
                + String(scratch)
                + " ints and counts+run_begin+perm is "
                + String(V + (V + 1) + M)
                + ". The embedding lane changed its run structure and this"
                + " harness would hand it three buffers of the wrong size."
            )
        self.emb_counts = _zeros_i32(ctx, V)
        self.emb_run_begin = _zeros_i32(ctx, V + 1)
        self.emb_perm = _zeros_i32(ctx, M)

        self.buf_initialized = List[Bool]()
        for _ in range(TRAIN_J):
            self.buf_initialized.append(False)


# ===========================================================================
# PACK AND UNPACK (plan section 2.1, DEVIATION 1553)
# ===========================================================================
# Pure element copies. No arithmetic, so neither can move a bit -- BUT A
# WRONG OFFSET GIVES PLAUSIBLE, IN-BOUNDS, WRONG NUMBERS THAT ARE IDENTICAL
# ON ALL THREE VENDORS. `train_step_check.mojo` clause (f) is the assertion.


def _copy_into(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    dst_off: Int,
    src_off: Int,
    count: Int,
) raises:
    if count < 1:
        return
    ctx.enqueue_function[train_copy_range_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        Int32(dst_off),
        Int32(src_off),
        Int32(count),
        grid_dim=(_grid(count), 1, 1),
        block_dim=(TRAIN_TPB, 1, 1),
    )


def unpack_params(
    ctx: DeviceContext,
    mut tb: TrainBuffers,
    mut w: LlamaDeviceWeights,
) raises:
    """Flat parameter state -> the eleven per-tensor buffers.

    **THE FLAT BUFFER IS THE STATE.** The per-tensor buffers are a VIEW
    refreshed at the top of every step, and the digest reads the flat buffer
    and only the flat buffer. A digest taken over a per-tensor view would be
    a digest of something the optimizer never wrote, which is failure mode V3
    in the plan's table.
    """
    var o = tb.offsets.copy()
    _copy_into(ctx, tb.emb_w, tb.param, 0, o[PID_EMBED], param_id_count(PID_EMBED))
    _copy_into(ctx, w.norm1_w, tb.param, 0, o[PID_NORM1], param_id_count(PID_NORM1))
    _copy_into(ctx, w.w_q, tb.param, 0, o[PID_W_Q], param_id_count(PID_W_Q))
    _copy_into(ctx, w.w_k, tb.param, 0, o[PID_W_K], param_id_count(PID_W_K))
    _copy_into(ctx, w.w_v, tb.param, 0, o[PID_W_V], param_id_count(PID_W_V))
    _copy_into(ctx, w.w_o, tb.param, 0, o[PID_W_O], param_id_count(PID_W_O))
    _copy_into(ctx, w.norm2_w, tb.param, 0, o[PID_NORM2], param_id_count(PID_NORM2))
    _copy_into(ctx, w.w_gate, tb.param, 0, o[PID_W_GATE], param_id_count(PID_W_GATE))
    _copy_into(ctx, w.w_up, tb.param, 0, o[PID_W_UP], param_id_count(PID_W_UP))
    _copy_into(ctx, w.w_down, tb.param, 0, o[PID_W_DOWN], param_id_count(PID_W_DOWN))
    _copy_into(ctx, tb.lm_w, tb.param, 0, o[PID_LM_HEAD], param_id_count(PID_LM_HEAD))
    ctx.synchronize()


def pack_grads(
    ctx: DeviceContext,
    mut tb: TrainBuffers,
    mut bst: LlamaBackwardStages,
) raises:
    """The eleven gradient buffers -> the flat `grad`, in `param_id` order.

    The order here MUST match `unpack_params` and `train_offsets` exactly.
    Three spellings of one layout is three chances to get it wrong, and that
    is precisely why clause (f) round-trips the pair rather than reading
    them.
    """
    var o = tb.offsets.copy()
    _copy_into(ctx, tb.grad, tb.dw_emb, o[PID_EMBED], 0, param_id_count(PID_EMBED))
    _copy_into(ctx, tb.grad, bst.dw_norm1, o[PID_NORM1], 0, param_id_count(PID_NORM1))
    _copy_into(ctx, tb.grad, bst.dw_q, o[PID_W_Q], 0, param_id_count(PID_W_Q))
    _copy_into(ctx, tb.grad, bst.dw_k, o[PID_W_K], 0, param_id_count(PID_W_K))
    _copy_into(ctx, tb.grad, bst.dw_v, o[PID_W_V], 0, param_id_count(PID_W_V))
    _copy_into(ctx, tb.grad, bst.dw_o, o[PID_W_O], 0, param_id_count(PID_W_O))
    _copy_into(ctx, tb.grad, bst.dw_norm2, o[PID_NORM2], 0, param_id_count(PID_NORM2))
    _copy_into(ctx, tb.grad, bst.dw_gate, o[PID_W_GATE], 0, param_id_count(PID_W_GATE))
    _copy_into(ctx, tb.grad, bst.dw_up, o[PID_W_UP], 0, param_id_count(PID_W_UP))
    _copy_into(ctx, tb.grad, bst.dw_down, o[PID_W_DOWN], 0, param_id_count(PID_W_DOWN))
    _copy_into(ctx, tb.grad, tb.dw_lm, o[PID_LM_HEAD], 0, param_id_count(PID_LM_HEAD))
    ctx.synchronize()


def upload_batch(
    ctx: DeviceContext, mut tb: TrainBuffers, ids: List[Int32]
) raises:
    """The step's inputs and targets onto the device."""
    comptime M = TRAIN_B * TRAIN_L
    var inp = batch_inputs(ids)
    var tgt = batch_targets(ids)
    var hi = ctx.enqueue_create_host_buffer[DType.int32](M)
    var ht = ctx.enqueue_create_host_buffer[DType.int32](M)
    ctx.synchronize()
    for i in range(M):
        hi.unsafe_ptr().unsafe_store(i, inp[i])
        ht.unsafe_ptr().unsafe_store(i, tgt[i])
    ctx.enqueue_copy(dst_buf=tb.ids, src_ptr=hi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=tb.targets, src_ptr=ht.unsafe_ptr())
    ctx.synchronize()
    _ = hi
    _ = ht


# ===========================================================================
# THE STEP
# ===========================================================================


def train_step(
    ctx: DeviceContext,
    mut tb: TrainBuffers,
    mut w: LlamaDeviceWeights,
    mut rope: LlamaRopeTable,
    mut stages: LlamaDeviceStages,
    mut bst: LlamaBackwardStages,
    mut op_trace: IdentityTrace,
    cfg: TrainConfig,
    ids: List[Int32],
    t: Int,
) raises -> Float32:
    """ONE step. Returns the loss.

    `t` IS ONE-BASED. `optimizer_step_oracle`'s convention, and the first
    step of a run is `t = 1`.

    THE TWELVE STAGES, in order, with their status. Four are GATED, four are
    GATED elsewhere, two are WRITTEN WITH NO GATE, and two are this file's.

        1  data        this lane
        2  embed       WRITTEN, NO GATE
        3  block fwd   GATED (5 clauses, 13 arms)
        4  lm_head     GATED, three vendors
        5  loss        GATED (6 clauses)
        6  loss bwd    GATED (6 clauses)
        7  head bwd    GATED
        8  block bwd   WRITTEN, NO GATE
        9  embed bwd   WRITTEN, NO GATE
       10  pack        this lane
       11  optimizer   GATED (6 clauses)
       12  digest      this lane (caller's)

    A FRESH KV CACHE EVERY STEP (DEVIATION 1554). Prefill only, `pos0 = 0`. A
    carried cache would make a step depend on history through a second
    channel, and separating a cache divergence from a parameter divergence
    inside one digest is not possible.
    """
    comptime M = TRAIN_B * TRAIN_L
    comptime DM = TRAIN_D_MODEL
    comptime V = TRAIN_VOCAB

    var emb_cfg = EmbConfig.llama(V, DM)
    var ce_cfg = CeConfig.causal_lm(V)

    # ---- 1. data -------------------------------------------------------
    upload_batch(ctx, tb, ids)

    # ---- 2. embed ------------------------------------------------------
    # WRITTEN, NO GATE. Stage 2 of twelve.
    identical_embedding_forward_into(ctx, tb.x, tb.emb_w, tb.ids, M, emb_cfg)
    ctx.synchronize()

    # ---- 3. block forward ----------------------------------------------
    var kv = LlamaKVCache(ctx, TRAIN_B, train_dims(), TRAIN_L)
    llama_decoder_layer_forward(
        ctx,
        stages,
        kv,
        rope,
        w,
        tb.x,
        TRAIN_B,
        TRAIN_L,
        0,
        op_trace,
        String("fwd.") + step_tag(t),
    )
    ctx.synchronize()

    # ---- 4. lm_head. `logits[M, V] = residual2[M, DM] . lm_w[V, DM]^T` --
    identical_gemm_into(
        ctx, tb.logits, stages.residual2, tb.lm_w, tb.head_ws, M, V, DM, OP_NT
    )
    ctx.synchronize()

    # ---- 5 and 6. loss and its backward, enqueued back to back ----------
    # This is the shape `loss.mojo`'s launchers exist for: `_into` forms that
    # enqueue and return, one wait for the pair.
    identical_ce_forward_into(
        ctx,
        tb.ce_max,
        tb.ce_shift,
        tb.ce_expo,
        tb.ce_denom,
        tb.ce_logdenom,
        tb.ce_logp_target,
        tb.ce_nll,
        tb.ce_logp,
        tb.ce_logp_sum,
        tb.ce_smooth,
        tb.ce_row,
        tb.ce_total,
        tb.ce_loss,
        tb.logits,
        tb.targets,
        tb.ce_ones,
        tb.ce_ws,
        M,
        M,
        ce_cfg,
    )
    identical_ce_backward_into(
        ctx,
        tb.ce_weights,
        tb.ce_dlogits,
        tb.ce_expo,
        tb.ce_denom,
        tb.ce_logp,
        tb.targets,
        M,
        M,
        ce_cfg,
    )
    ctx.synchronize()

    # `count` is the HOST integer `M`, and it is `M` because no row is ever
    # ignored -- every target came from position `t + 1` of a real row
    # (DEVIATION 1556). `ce_divisor` is the ONE producer of the quantity the
    # forward and the backward must agree about, and both calls above pass it
    # the same three arguments.

    # ---- 7. lm_head backward --------------------------------------------
    # Forward was `OP_NT` at `(M, V, DM)`. `dA` takes `A`'s shape `[M, DM]`
    # and `dB` takes `B`'s shape `[V, DM]`. The two write DISJOINT buffers
    # and read the same `dC`, so they need no barrier between them and may
    # share one workspace on one context.
    identical_gemm_backward_a_into(
        ctx, tb.d_h, tb.ce_dlogits, tb.lm_w, tb.head_bwd_ws, M, V, DM, OP_NT
    )
    identical_gemm_backward_b_into(
        ctx,
        tb.dw_lm,
        tb.ce_dlogits,
        stages.residual2,
        tb.head_bwd_ws,
        M,
        V,
        DM,
        OP_NT,
    )
    ctx.synchronize()

    # ---- 8. block backward ----------------------------------------------
    # WRITTEN, NO GATE. Stage 8 of twelve, and the largest of the two
    # ungated ones.
    #
    # DEVIATION 1577: this entry point takes its incoming gradient as a HOST
    # `List[Float32]`, so the step downloads `d_h` and the callee re-uploads
    # it. 512 floats here and irrelevant; a bus round trip in the hot path at
    # any real size. A device-buffer form is OWED and is not in this lane's
    # write set.
    var d_out = download_f32(ctx, tb.d_h, M * DM)
    llama_decoder_layer_backward(
        ctx,
        bst,
        stages,
        w,
        rope.cos,
        rope.sin,
        tb.x,
        d_out,
        TRAIN_B,
        TRAIN_L,
        0,
        op_trace,
        String("bwd.") + step_tag(t),
    )
    ctx.synchronize()

    # ---- 9. embedding backward -------------------------------------------
    # WRITTEN, NO GATE. `accumulate` is FALSE, so `dW` is `+0.0`-filled here
    # every step and there is no carry to get wrong. The microbatch carry
    # (contract 7.4) is deliberately unused: this profile does not split a
    # batch, and the plan's N7 says why a split is not an invariance claim.
    identical_embedding_backward_into(
        ctx,
        tb.dw_emb,
        bst.d_x,
        tb.ids,
        tb.emb_counts,
        tb.emb_run_begin,
        tb.emb_perm,
        M,
        emb_cfg,
    )
    ctx.synchronize()

    # ---- 10. pack ---------------------------------------------------------
    pack_grads(ctx, tb, bst)

    # ---- 11. optimizer ----------------------------------------------------
    identical_optimizer_step(
        ctx,
        tb.param,
        tb.grad,
        tb.m_state,
        tb.v_state,
        tb.denom_out,
        tb.q_out,
        tb.sumsq,
        tb.norms,
        tb.total_cell,
        tb.out2,
        tb.opt_ws,
        tb.sab_partials,
        tb.buf_initialized,
        tb.offsets,
        cfg.optimizer(),
        t,
    )

    var loss_l = download_f32(ctx, tb.ce_loss, 1)
    # `[[mojo-buffer-freed-at-last-use]]`: the KV cache and the downloaded
    # gradient must outlive every kernel that was handed their pointers.
    _ = kv^
    _ = d_out
    return loss_l[0]


# ===========================================================================
# THE RUN
# ===========================================================================


struct TrainRun(Movable):
    """One run's result. The digests, the losses, and the step count.

    `steps_run` is a COUNTER and not the requested `N`. A loop that silently
    ran zero steps is failure mode V1 in the plan's table, and the only thing
    that can catch it is a number the loop itself increments.
    """

    var steps_run: Int
    var losses: List[Float32]
    var initial: TrainDigest
    var final: TrainDigest
    var param0: List[Float32]
    var paramN: List[Float32]

    def __init__(out self):
        self.steps_run = 0
        self.losses = List[Float32]()
        self.initial = TrainDigest()
        self.final = TrainDigest()
        self.param0 = List[Float32]()
        self.paramN = List[Float32]()


def snapshot(
    ctx: DeviceContext, mut tb: TrainBuffers
) raises -> TrainDigest:
    """A checkpoint digest of the LIVE flat buffers.

    Counts are `n_total`, never `len(buf)`. See `TrainDigest`.
    """
    var p = download_f32(ctx, tb.param, tb.n_total)
    var m = download_f32(ctx, tb.m_state, tb.n_total)
    var v = download_f32(ctx, tb.v_state, tb.n_total)
    return digest_of_lists(p, m, v, tb.offsets)


def run_training(
    ctx: DeviceContext,
    cfg: TrainConfig,
    mut trace: IdentityTrace,
    mut op_trace: IdentityTrace,
) raises -> TrainRun:
    """`cfg.steps` steps. Emits four records per step plus a step-zero pair.

    THE RECORD COUNT IS EXACTLY `2 + 4*N` and `train_step_check.mojo` clause
    (e) asserts it (DEVIATION 1563). A trace with zero records compared
    against a trace with zero records is three machines agreeing about
    nothing, and that failure is otherwise completely silent.

        train.sNNNN.data.ids       i32  B*(L+1)
        train.sNNNN.loss           f32  1
        train.sNNNN.ckpt.tensors   u32  2*J
        train.sNNNN.ckpt.digest    u32  8

    `op_trace` is a SEPARATE trace for the composed ops' own thirty-seven
    backward stages and twenty-six forward ones. It is disabled by default,
    because those cards belong to those lanes' gates and folding them into
    this file's would make the record count depend on which ops were traced.
    """
    var run = TrainRun()
    var tb = TrainBuffers(ctx, cfg.seed)
    var dims = train_dims()
    var w = LlamaDeviceWeights(
        ctx,
        dims,
        TRAIN_RMS_EPS,
        train_init_tensor(cfg.seed, PID_NORM1),
        train_init_tensor(cfg.seed, PID_NORM2),
        train_init_tensor(cfg.seed, PID_W_Q),
        train_init_tensor(cfg.seed, PID_W_K),
        train_init_tensor(cfg.seed, PID_W_V),
        train_init_tensor(cfg.seed, PID_W_O),
        train_init_tensor(cfg.seed, PID_W_GATE),
        train_init_tensor(cfg.seed, PID_W_UP),
        train_init_tensor(cfg.seed, PID_W_DOWN),
    )
    var rope = LlamaRopeTable(ctx, dims, TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS)
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, dims)
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, dims)

    # ---- the ULP arm, BEFORE step 1 (plan N3) ---------------------------
    # The LAST float of `w_v`. `w_v` because every token's forward reads it,
    # so the perturbation cannot land on a row the batch never touched; the
    # LAST float because an off-by-one that drops a tensor's tail is a defect
    # this positions the arm to catch.
    if cfg.arm == ARM_ULP:
        var idx = tb.offsets[PID_W_V + 1] - 1
        # DEVIATION 1579: MOJOLEARN_TRAIN_ULPS sweeps the magnitude so the
        # N3 failure can be attributed. Default 1 keeps the strict form.
        var nulps = 1
        var us = String(getenv("MOJOLEARN_TRAIN_ULPS"))
        if us != "":
            nulps = Int(atol(us))
        ctx.enqueue_function[train_ulp_perturb_kernel](
            tb.param.unsafe_ptr(),
            Int32(idx),
            Int32(nulps),
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        ctx.synchronize()

    run.param0 = download_f32(ctx, tb.param, tb.n_total)
    run.initial = snapshot(ctx, tb)
    record_checkpoint(trace, 0, run.initial)

    for step in range(1, cfg.steps + 1):
        var es = effective_step(step, cfg.steps, cfg.arm)
        var ids = train_batch_ids(cfg.seed, es)

        var pre = String("train.") + step_tag(step) + "."
        trace.record_host(
            pre + "data.ids", ids.unsafe_ptr(), len(ids)
        )

        unpack_params(ctx, tb, w)
        var loss = train_step(
            ctx, tb, w, rope, stages, bst, op_trace, cfg, ids, step
        )
        run.losses.append(loss)
        run.steps_run += 1

        trace.record_scalar_f32(pre + "loss", loss)
        var d = snapshot(ctx, tb)
        record_checkpoint(trace, step, d)
        if step == cfg.steps:
            run.final = d^
        _ = ids

    run.paramN = download_f32(ctx, tb.param, tb.n_total)

    # `[[mojo-buffer-freed-at-last-use]]`: every owner stays alive past the
    # last kernel that was handed one of its pointers.
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    return run^


def resume_training(
    ctx: DeviceContext,
    cfg: TrainConfig,
    first: Int,
    mut trace: IdentityTrace,
    mut op_trace: IdentityTrace,
) raises -> TrainDigest:
    """`first` steps, then `cfg.steps - first` more from the CARRIED state.

    Plan N6. `t` continues across the boundary (it is one-based over the
    whole run, not over each half) and the data generator continues its
    stream, so the digest after both halves must EQUAL the digest after one
    uninterrupted run of `cfg.steps`.

    **THIS IS A RESUME WITHIN ONE PROCESS AND NOT FROM A FILE.** There is no
    checkpoint serialization in v1 (DEVIATION 1569), so
    `OPT_SAB_RESUME_REINIT`'s real form -- a checkpoint that drops the
    momentum flags on the way to disk -- is not reachable from here and this
    control does not test it. Saying so is the point.
    """
    if first < 1 or first >= cfg.steps:
        raise Error(
            String("train_loop: resume split must satisfy 1 <= first < N,")
            + " got first="
            + String(first)
            + " N="
            + String(cfg.steps)
        )
    var tb = TrainBuffers(ctx, cfg.seed)
    var dims = train_dims()
    var w = LlamaDeviceWeights(
        ctx,
        dims,
        TRAIN_RMS_EPS,
        train_init_tensor(cfg.seed, PID_NORM1),
        train_init_tensor(cfg.seed, PID_NORM2),
        train_init_tensor(cfg.seed, PID_W_Q),
        train_init_tensor(cfg.seed, PID_W_K),
        train_init_tensor(cfg.seed, PID_W_V),
        train_init_tensor(cfg.seed, PID_W_O),
        train_init_tensor(cfg.seed, PID_W_GATE),
        train_init_tensor(cfg.seed, PID_W_UP),
        train_init_tensor(cfg.seed, PID_W_DOWN),
    )
    var rope = LlamaRopeTable(ctx, dims, TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS)
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, dims)
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, dims)

    var out = TrainDigest()
    for step in range(1, cfg.steps + 1):
        var es = effective_step(step, cfg.steps, cfg.arm)
        var ids = train_batch_ids(cfg.seed, es)
        unpack_params(ctx, tb, w)
        var loss = train_step(
            ctx, tb, w, rope, stages, bst, op_trace, cfg, ids, step
        )
        _ = loss
        if step == first:
            # The mid-run checkpoint. Taken and DISCARDED here on purpose:
            # what N6 compares is the END state, and a mid-run digest that
            # matched would say nothing about whether `t` continued.
            var mid = snapshot(ctx, tb)
            _ = mid^
        if step == cfg.steps:
            out = snapshot(ctx, tb)
        _ = ids
    _ = trace
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    return out^


# ===========================================================================
# THE ENTRY POINT
# ===========================================================================


def write_ckpt_file(path: String, d: TrainDigest, cfg: TrainConfig) raises:
    """The sidecar a three-vendor comparison `diff`s.

    One line per digest, in a fixed order, so `diff a.ckpt b.ckpt` is the
    whole verdict and the trace differ is only needed for the address.
    """
    if path == "":
        return
    var body = String("")
    body += "profile mojolearn.identical.train.ckpt.v1\n"
    body += "mode " + mode_banner() + "\n"
    body += "arm " + arm_name(cfg.arm) + "\n"
    body += "steps " + String(cfg.steps) + "\n"
    body += "param " + hex16(d.h_param) + "\n"
    body += "m " + hex16(d.h_m) + "\n"
    body += "v " + hex16(d.h_v) + "\n"
    body += "all " + hex16(d.h_all) + "\n"
    for j in range(len(d.h_tensor)):
        body += (
            "p"
            + String(j)
            + " "
            + param_id_name(j)
            + " "
            + hex16(d.h_tensor[j])
            + "\n"
        )
    with open(path, "w") as fh:
        fh.write(body)


def main() raises:
    var steps = env_int("MOJOLEARN_TRAIN_STEPS", 8)
    var seed = env_u64("MOJOLEARN_TRAIN_SEED", SEED_BASE)
    var arm = arm_by_name(env_str("MOJOLEARN_TRAIN_ARM"))

    print(
        "=== training step, profile mojolearn.identical.train.step.fp32.v1"
    )
    print(
        "=== NOTHING IN train_loop.mojo, train_step_check.mojo OR"
        " TRAINING_LOOP_PLAN.md HAD EVER BEEN COMPILED OR RUN BEFORE THIS"
        " PROCESS. Read the header."
    )
    print(
        "=== TWO OF THE TWELVE STAGES HAVE NO GATE:"
        " transformer/checks/transformer_backward.mojo and"
        " embedding/checks/embedding_identical.mojo. Three matching"
        " digests would show those two AGREE across vendors. They would NOT"
        " show either is CORRECT. Run train_step_check clause (a) first."
    )
    print("mode " + mode_banner() + "   arm " + arm_name(arm))

    # V10: a misspelled selector must never read as a clean run.
    var expect = env_str("MOJOLEARN_TRAIN_EXPECT_ARM")
    if expect != "":
        if expect != arm_name(arm):
            raise Error(
                String("train_loop: the caller expected arm '")
                + expect
                + "' and this run carries '"
                + arm_name(arm)
                + "'. A misspelled selector that silently ran the clean"
                + " configuration would be recorded as 'the arm did not"
                + " bite', which is the exact inverse of the truth"
                + " (tools/gemm_ladder.sh:71's scar). Fix one or the other."
            )
        print("ledger: the caller named '" + expect + "' and the run agrees")
    elif arm == ARM_NONE:
        print(
            "ledger: CLEAN run, no arm. Set MOJOLEARN_TRAIN_EXPECT_ARM to"
            " have the binary check its own selector."
        )
    else:
        print(
            "ledger: this run carries arm '"
            + arm_name(arm)
            + "' and the caller did not say so. Set"
            " MOJOLEARN_TRAIN_EXPECT_ARM to close the misspelled-selector"
            " hole."
        )

    if steps < 1:
        raise Error(
            String("train_loop: MOJOLEARN_TRAIN_STEPS must be >= 1, got ")
            + String(steps)
            + ". A run of zero steps produces the INITIAL digest, which is"
            + " a pure function of the seed and matches on every machine."
            + " That is plan failure mode V1 and it is REFUSED here rather"
            + " than reported."
        )

    var cfg = TrainConfig.for_arm(steps, seed, arm)
    var ctx = DeviceContext()

    var trace_path = env_str("MOJOLEARN_IDENTITY_TRACE")
    var trace = IdentityTrace()
    if trace_path != "":
        trace = IdentityTrace.to_path(trace_path, "", False)
        trace.header(
            String("mojolearn.identical.train.step.fp32.v1 mode=")
            + mode_banner()
            + " arm="
            + arm_name(cfg.arm)
            + " steps="
            + String(cfg.steps)
            + " J="
            + String(TRAIN_J)
            + " n_total="
            + String(train_n_total())
        )
    var op_path = env_str("MOJOLEARN_TRAIN_OP_TRACE")
    var op_trace = IdentityTrace.disabled()
    if op_path != "":
        op_trace = IdentityTrace.to_path(op_path)

    var run = run_training(ctx, cfg, trace, op_trace)

    if run.steps_run != cfg.steps:
        raise Error(
            String("train_loop: asked for ")
            + String(cfg.steps)
            + " steps and the loop counted "
            + String(run.steps_run)
            + ". A loop that silently runs a different number of steps"
            + " produces a deterministic digest that matches on every"
            + " machine (plan failure mode V1)."
        )

    print("steps_run " + String(run.steps_run))
    for i in range(len(run.losses)):
        print(
            "  step "
            + String(i + 1)
            + " loss "
            + String(run.losses[i])
            + " / "
            + hex16(UInt64(bitcast[DType.uint32](run.losses[i])))
        )
    print("ckpt.param " + hex16(run.final.h_param))
    print("ckpt.m     " + hex16(run.final.h_m))
    print("ckpt.v     " + hex16(run.final.h_v))
    print("ckpt.all   " + hex16(run.final.h_all))
    for j in range(len(run.final.h_tensor)):
        print(
            "  p"
            + String(j)
            + " "
            + param_id_name(j)
            + " "
            + hex16(run.final.h_tensor[j])
        )

    # The movement report (plan section 5.3). PRINTED ALWAYS, because the
    # first run must be checked against the paper arithmetic before any
    # digest is believed, and because a digest printed on its own is not a
    # result.
    var offs = train_offsets()
    var changed = 0
    for i in range(len(run.param0)):
        var a = bitcast[DType.uint32](run.param0[i])
        var b = bitcast[DType.uint32](run.paramN[i])
        if a != b:
            changed += 1
    print(
        "moved "
        + String(changed)
        + "/"
        + String(len(run.param0))
        + " parameters changed bits over "
        + String(run.steps_run)
        + " steps"
    )
    for j in range(TRAIN_J):
        var lo = offs[j]
        var hi = offs[j + 1]
        var worst = Float32(0.0)
        var moved = 0
        for i in range(lo, hi):
            var dv = run.paramN[i] - run.param0[i]
            if dv < Float32(0.0):
                dv = -dv
            if dv > worst:
                worst = dv
            if bitcast[DType.uint32](run.param0[i]) != bitcast[
                DType.uint32
            ](run.paramN[i]):
                moved += 1
        print(
            "  p"
            + String(j)
            + " "
            + param_id_name(j)
            + " moved "
            + String(moved)
            + "/"
            + String(hi - lo)
            + " max|dp| "
            + String(worst)
        )

    write_ckpt_file(env_str("MOJOLEARN_TRAIN_CKPT"), run.final, cfg)
    print(
        "NOTE: no timing may be quoted from this binary. A traced step"
        " drains the queue four times and the optimizer's refusal downloads"
        " four buffers every step (HOST_AND_DEVICE.md)."
    )
