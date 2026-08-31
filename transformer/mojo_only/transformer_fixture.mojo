# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures and bit plants for the IDENTICAL FP32 transformer block,
profile `mojolearn.identical.transformer.fp32.v1`. NOT A PORT.

**NOTHING IN THIS FILE HAS BEEN COMPILED OR RUN.** Every claim below is
CONSTRUCTION or PREDICTION. Written 2026-08-25 against
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`, which is FROZEN and which
this file consumes rather than amends. Where a sentence here says what a
fixture "separates" or "must move", read it as a prediction that
`transformer/mojo_only/transformer_check.mojo` owes a gate for; no gate has
been written and no bit has been observed.

## What is here, and why each thing is here

1. **A hashed generator, spec `mojolearn.transformer.fixture.hash.v1`.**
   Every tensor element is
   `f32(lo + (hi - lo) * top24(splitmix64(key + i)) * 2^-24)` with
   `key = splitmix64(seed ^ (tensor_id << 32))`. It is the mamba corpus
   generator's map (`mamba/corpus/gen_corpus.py`, spec
   `mojolearn.mamba.corpus.hash.v1`, reimplemented at
   `mamba/mojo_only/mamba_fixture.mojo:112-135`) with this lane's own
   SEED_BASE and its own tensor ids.

   DEVIATION 1000: the four lines of splitmix64 are COPIED here rather than
   imported from `mamba/mojo_only/mamba_fixture.mojo`. Chosen because the
   alternative points a `transformer/ -> mamba/` arrow at a lane that is
   under concurrent edit, for a function that is exact integer arithmetic
   with no rounding and therefore cannot drift the way a float seam can.
   The cost if wrong is duplication: two copies of a hash have two chances
   to be edited apart. The mitigation the check file owes is one assert
   that this generator and `mamba_fixture.corpus_splitmix64` agree on a
   handful of inputs, which is cheap and which catches the only failure
   the copy can have.

2. **Bit plants.** `uniform-test-data-hides-permutation` is the standing
   rule and this file obeys it twice over. The hashed values are all
   DISTINCT per (tensor, index), so a transposed or mis-strided read of any
   weight lands on a different number and shows up; and the values a hash
   can never produce (`-0.0`, a subnormal, `1e35`) are PLANTED by bits,
   because contract 4.1(a) says in as many words that a `-0.0` score has to
   be planted or the mask-elision defect is invisible.

3. **A case table.** Fifteen cases, every one small enough to run on a
   laptop GPU inside a short lease. There is deliberately no large case
   "for later"; the largest thing here is a `[1, 2, 257, 257]` score buffer
   at 132k floats.

## The plant scheme, stated so a reader can audit a fixture by eye

Nothing in this file is uniform and nothing is all-ones. Concretely:

- Each tensor gets its own `key`, so `w_q` and `w_k` never share a value at
  the same index and a q/k swap is visible.
- Within a tensor the values walk `key + i` in FLAT ROW-MAJOR index order,
  so a row/column transposition permutes them and a transposed read of
  `w_q` produces a different `q_proj.out`. This is the property a uniform
  fixture destroys.
- The RANGES differ per tensor (`x` in [-2, 2], the norm weights in
  [0.5, 1.5], `w_q/w_k/w_v` in [-0.5, 0.5], `w_o/w_gate/w_up` in
  [-0.25, 0.25], `w_down` in [-0.125, 0.125]), so a weight read from the
  wrong buffer is usually out of range as well as out of place.
- The planted values (`-0.0`, `+-1e35`, subnormals) are placed at named,
  computable indices, never scattered, so a check can say WHICH cell it
  expects to move.

## Scale note, and it is a PREDICTION, not a measurement

The ranges were chosen so the softmax has something to do. With
`d_model = 32`, `head_dim = 16` and the ranges above, a rough
propagation-of-variance estimate puts the pre-scale `q . k^T` dot near
standard deviation 8 and the post-scale score near 2, so a row of scores
spans a few units, `exp` neither saturates nor collapses, and the softmax
weights are genuinely non-uniform. THIS ARITHMETIC WAS DONE ON PAPER AND
NOT RUN. If the observed scores turn out to be within a few ulps of uniform
the ranges are wrong and the fixture is weak, and the first run should
print a score histogram before any gate is believed.

## The one thing no INPUT to this block can do

No choice of x, of weights, of shape or of planted score reaches contract
section 5.1's hazard, and this lane believes that is a defect in the
contract rather than in the fixture. Section 5.1 pins the softmax row
maximum to `identical_fmax` on the ground that "a row of attention scores
reaches both zero signs easily", but S14 does not read scores, it reads
`attn.masked`, and contract 4.1(a) is itself the proof that `attn.masked`
contains no negative zero: at an unmasked cell the value is
`ftz(ftz(s) + (+0.0))`, and `x + (+0.0)` is `+0.0` for `x = -0.0` and for
every negative subnormal the inner `ftz` has already flushed to `-0.0`; at a
masked cell the value is `-FLT_MAX` or `-inf`.

So the pin is right and its stated REACH is wrong, and a sabotage arm built
on the stated reach will be inert and will be deleted by whoever writes the
gates. `PLANT_MASKED_ZERO_ROW` and case `adv_masked_zero_row` are this
file's answer: a plant applied AFTER the mask, which is the only way to put
a negative zero in front of S14 without also sabotaging S13, and a gate that
sabotages two seams at once cannot say which one it proved. DEVIATION 1005,
argued at the plant.
"""

from std.memory import bitcast

from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_rsqrt,
    numeric_mode_name,
)


# ===========================================================================
# PROFILE CONSTANTS (contract section 3). The FROZEN ones. Changing a value
# in this block is a v2 of `mojolearn.identical.transformer.fp32.v1`; it does
# not amend v1.
# ===========================================================================

comptime RMS_EPS: Float32 = 1e-6
"""`LlamaConfig.rms_norm_eps`, configuration_llama.py:73, verified in the
pinned checkout on 2026-08-25. Bits `0x358637BD`.

**NOT `mamba/mojo_only/mamba_fixture.mojo:44`'s `RMS_EPS`, which is `1e-5`.**
The two lanes normalize with different epsilons and the mamba constant is a
module-level `comptime` read directly by `mamba_rms_norm_kernel`, so the
device half of this lane cannot reuse that kernel until eps is an ARGUMENT.
That lift is contract DEVIATION 801, it is a cross-lane edit, and it is NOT
made here because this lane owns two files and that is not one of them. The
host oracle needs nothing from the mamba lane and spells RMSNorm itself."""

comptime ROPE_THETA: Float32 = 10000.0
"""The RoPE base. Bits `0x461C4000`, which is exact (verified by a host
float32 round trip on 2026-08-25, not by running Mojo).

**THE CONTRACT'S CITATION FOR THIS IS SLIGHTLY WRONG AND THE CORRECT ONE IS
HERE.** Contract section 3 cites "`RopeParameters` default base,
modeling_rope_utils.py:177". Line 177 in the pinned checkout is
`base = rope_parameters_dict["rope_theta"]`, which is where the base is
READ, not where its default is DEFINED. The default lives at
`src/transformers/modeling_rope_utils.py:739`,
`RotaryEmbeddingConfigMixin.default_theta = 10_000.0`, and reaches
`rope_parameters_dict` through :752. `LlamaConfig` does not override it (a
grep for `default_theta` across `src/transformers/models/` shows overrides
in llama4, olmo3, cohere, gpt_oss and others, and NOT in llama), so 10000.0
is right for this profile.

The contract is FROZEN and this lane does not amend it. The correction is
recorded here and reported to the orchestrator, per
`[[fix-docs-on-discovery]]`, because the sentence in the contract is one a
future reader would follow to a line that does not say what they were told
it says."""

comptime BITS_RMS_EPS: UInt32 = 0x358637BD
comptime BITS_ROPE_THETA: UInt32 = 0x461C4000
comptime BITS_MASK_FILL: UInt32 = 0xFF7FFFFF
"""`torch.finfo(float32).min`, `-3.4028234663852886e+38`, masking_utils.py:601
in the pinned checkout. **ADDED, never selected** (contract S13 and 4.1(a)),
and **finite, never `-inf`** (contract 4.1(b))."""

comptime MAX_ABS_POSITION = 8192
"""Contract DEVIATION 812. `_cephes_sincosf_core`'s Cody-Waite reduction is
valid on |x| < 8192 and does not guard above it, and RoPE's angle at
`inv_freq[0] == 1.0` is the position itself, so the position IS the angle at
the worst index. The oracle REFUSES BY NAME at or above this rather than
computing a number the reduction has lost. `portable_sinf`'s own docstring
(DEVIATION 820) addresses this lane by name and asks for the choice to be
made; contract section 3 made it and this constant is where it lives."""


def mask_fill() -> Float32:
    """The additive mask value, by BITS.

    Spelled from bits rather than from the decimal literal
    `-3.4028234663852886e+38` because `[[mojo-string-float-roundtrip]]` says
    a Float32 does not round trip through this toolchain's decimal
    formatting, and a mask value that is one ulp off is a mask value that
    still looks like a mask and produces different bits at every masked
    cell in every row."""
    return bitcast[DType.float32](BITS_MASK_FILL)


def unmasked_fill() -> Float32:
    """`+0.0`, contract section 3's "unmasked fill", masking_utils.py:603.

    A separate named function so that no reader can mistake the S13 add for
    an optional one. Contract 4.1(a): adding this is NOT a no-op, because
    `(-0.0) + (+0.0)` is `+0.0`, and an implementation that skips the add
    where the mask is true keeps a `-0.0` that the reference launders."""
    return Float32(0.0)


def profile_constants_are_intact() -> Bool:
    """PREDICTION, and the check file owes the assertion: the three frozen
    scalars have the bits contract section 3 pins.

    Exists because a `comptime Float32 = 1e-6` is the compiler's parse of a
    decimal literal and this repository has been bitten by decimal float
    text before. Cheap to check, catastrophic to get wrong (a wrong eps
    moves every stage from `norm1.out` onward and looks exactly like a
    kernel bug)."""
    if bitcast[DType.uint32](RMS_EPS) != BITS_RMS_EPS:
        return False
    if bitcast[DType.uint32](ROPE_THETA) != BITS_ROPE_THETA:
        return False
    if bitcast[DType.uint32](mask_fill()) != BITS_MASK_FILL:
        return False
    return True


# ===========================================================================
# SHAPE
# ===========================================================================


@fieldwise_init
struct TransformerDims(Copyable, Movable):
    """One Llama-shaped decoder block's shape, plus the ONE configuration
    quantity RoPE needs.

    `rope_positions` is the size of the precomputed rotary table and it is a
    CONFIGURATION quantity in the sense of `max_position_embeddings`, NOT a
    launch quantity. DEVIATION 1002, and the reason is a clause rather than
    a preference: `rope.cos` and `rope.sin` are recorded card stages
    (contract section 9), and contract clause (c) requires a row's bits to
    be identical at L in {4, 16, 64, 257}. A table sized from L would give
    `rope.cos` a different length and a different hash at every L, so
    clause (c) would fail at the table stage while every activation stage
    agreed. Sizing it from the config makes the table launch-independent by
    construction. The cost is that a call is REFUSED if it would touch a
    position at or beyond `rope_positions`, which is the right failure."""

    var d_model: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var rope_positions: Int

    def n_rep(self) -> Int:
        """`LlamaAttention.num_key_value_groups`, modeling_llama.py:223.
        `repeat_kv` (:179-188) maps attention head `h` to kv head
        `h // n_rep`, which is what `expand` then `reshape` does. A COPY,
        not a seam (contract section 4 preamble, DEVIATION 813)."""
        return self.n_heads // self.n_kv_heads

    def q_width(self) -> Int:
        return self.n_heads * self.head_dim

    def kv_width(self) -> Int:
        return self.n_kv_heads * self.head_dim

    def half_head(self) -> Int:
        """RoPE pairs index `j` with `j + head_dim/2` (`rotate_half`,
        modeling_llama.py:130-134) and the table is `cat(freqs, freqs)`
        (:123), so the DEDUPLICATED table has this many columns."""
        return self.head_dim // 2

    def validate(self) raises:
        """Contract section 3's divisibility rules, REFUSED BY NAME rather
        than silently truncated.

        Silent truncation is the failure this exists to prevent: an odd
        `head_dim` would make `rotate_half` pair the middle element with
        itself and an `n_heads % n_kv_heads != 0` would make `repeat_kv`'s
        reshape drop heads, and both produce a plausible number."""
        if self.d_model <= 0:
            raise Error("transformer: d_model must be positive")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise Error("transformer: head counts must be positive")
        if self.head_dim <= 0:
            raise Error("transformer: head_dim must be positive")
        if self.intermediate <= 0:
            raise Error("transformer: intermediate_size must be positive")
        if self.d_model != self.n_heads * self.head_dim:
            raise Error(
                String("transformer: d_model ")
                + String(self.d_model)
                + " != n_heads * head_dim = "
                + String(self.n_heads * self.head_dim)
                + " REFUSED (contract section 3)"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise Error(
                String("transformer: n_heads ")
                + String(self.n_heads)
                + " not divisible by n_kv_heads "
                + String(self.n_kv_heads)
                + " REFUSED (contract section 3; repeat_kv's reshape needs"
                + " it)"
            )
        if self.head_dim % 2 != 0:
            raise Error(
                String("transformer: head_dim ")
                + String(self.head_dim)
                + " is odd; RoPE pairs j with j + head_dim/2 REFUSED"
                + " (contract section 3)"
            )
        if self.rope_positions <= 0 or self.rope_positions > MAX_ABS_POSITION:
            raise Error(
                String("transformer: rope_positions ")
                + String(self.rope_positions)
                + " outside (0, "
                + String(MAX_ABS_POSITION)
                + "] REFUSED (DEVIATION 812: the Cody-Waite domain of"
                + " _cephes_sincosf_core)"
            )


def attention_scale(head_dim: Int) -> Float32:
    """Contract S-constant / DEVIATION 802: `identical_rsqrt(Float32(head_dim))`,
    computed ONCE on the host and stored FP32.

    The reference writes `self.head_dim ** -0.5` (modeling_llama.py:226),
    which is the HOST's libm `pow` and therefore IDENTITY_PATHS row 18
    applied to a constant: cross-vendor is cross-HOST when the value is
    baked at configuration time. `identical_rsqrt` (DEVIATION 741, over
    `portable_sqrtf`) is the spelling that does not depend on whose libm
    built the model.

    Contract section 3 records a host measurement from 2026-08-24 that the
    two agree at head_dim in {8, 16, 32, 64, 80, 128, 256}. **That is
    agreement at seven points and not a proof, and this lane did not re-take
    it.** It also records why a fixture at a non-power-of-four head_dim is
    required: at 16 the scale is exactly 0.25 and at 64 exactly 0.125, so
    every power-of-four shape is blind to a wrong spelling. Case
    `odd_head_dim_24` exists for that and for nothing else."""
    return ftz(identical_rsqrt(Float32(head_dim)))


# ===========================================================================
# WEIGHTS
# ===========================================================================


struct TransformerWeights(Copyable, Movable):
    """One decoder block's parameters, host side, in the UPSTREAM shapes.

    Every one is `nn.Linear.weight`, which torch stores `[out_features,
    in_features]` and applies as `y = x @ W^T`. That is exactly a
    `gemm.fp32.v1` `OP_NT` cell with the weight as the right operand and no
    transpose anywhere, which is why contract S5 can say "the GEMM refuses
    no shape" and mean it.

        norm1_w  [d_model]                       input_layernorm.weight
        norm2_w  [d_model]                       post_attention_layernorm.weight
        w_q      [n_heads*head_dim, d_model]     self_attn.q_proj.weight
        w_k      [n_kv*head_dim,    d_model]     self_attn.k_proj.weight
        w_v      [n_kv*head_dim,    d_model]     self_attn.v_proj.weight
        w_o      [d_model, n_heads*head_dim]     self_attn.o_proj.weight
        w_gate   [intermediate, d_model]         mlp.gate_proj.weight
        w_up     [intermediate, d_model]         mlp.up_proj.weight
        w_down   [d_model, intermediate]         mlp.down_proj.weight

    **NO BIASES ANYWHERE, and their absence is a refusal rather than an
    omission.** `attention_bias` and `mlp_bias` are both `False` at the
    config defaults (configuration_llama.py:81, :83) and contract section 2
    REFUSES a nonzero bias rather than specifying where it would round.
    There is no bias field to set, so there is no way to set one wrongly."""

    var dims: TransformerDims
    var norm1_w: List[Float32]
    var norm2_w: List[Float32]
    var w_q: List[Float32]
    var w_k: List[Float32]
    var w_v: List[Float32]
    var w_o: List[Float32]
    var w_gate: List[Float32]
    var w_up: List[Float32]
    var w_down: List[Float32]

    def __init__(out self, dims: TransformerDims):
        self.dims = dims.copy()
        self.norm1_w = List[Float32]()
        self.norm2_w = List[Float32]()
        self.w_q = List[Float32]()
        self.w_k = List[Float32]()
        self.w_v = List[Float32]()
        self.w_o = List[Float32]()
        self.w_gate = List[Float32]()
        self.w_up = List[Float32]()
        self.w_down = List[Float32]()


# ===========================================================================
# THE HASH, spec `mojolearn.transformer.fixture.hash.v1` (DEVIATION 1000)
# ===========================================================================


def fixture_splitmix64(z_in: UInt64) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim.

    **`+` HERE IS A REAL ADD AND MUST STAY ONE.** `[[mojo-amp-plus-is-bitwise-and]]`
    records that Mojo's `&+` computes `x & k` with no compile error, which
    silently produced wrong hashes in this repository once already. Every
    `+` below is the plain operator on UInt64, which wraps, which is what
    splitmix64 wants. Do not "fix" one into a `&+`."""
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def fixture_tensor(
    seed: UInt64, tensor_id: Int, n: Int, lo: Float64, hi: Float64
) -> List[Float32]:
    """`n` values `f32(lo + (hi - lo) * top24 * 2^-24)`, in flat row-major
    index order.

    ONE rounding, the final cast, because every (lo, hi) this file uses is
    dyadic and `top24 * 2^-24` is dyadic, so the float64 evaluation is
    exact. That is a PREDICTION of exactness resting on the ranges being
    dyadic; the check file owes the assertion, and if a range is ever
    changed to something like 0.1 the sentence becomes false silently."""
    var key = fixture_splitmix64(seed ^ (UInt64(tensor_id) << 32))
    var span = hi - lo
    var out = List[Float32]()
    for i in range(n):
        var h = fixture_splitmix64(key + UInt64(i))
        var unit = Float64(Int(h >> 40)) * 0.000000059604644775390625  # 2^-24
        out.append(Float32(lo + span * unit))
    return out^


comptime TID_X = 1
comptime TID_NORM1_W = 2
comptime TID_NORM2_W = 3
comptime TID_W_Q = 4
comptime TID_W_K = 5
comptime TID_W_V = 6
comptime TID_W_O = 7
comptime TID_W_GATE = 8
comptime TID_W_UP = 9
comptime TID_W_DOWN = 10
comptime TID_CACHE_TAIL = 11
"""The hot-tail plant's own id, so a tail value can never collide with a
real key or value the block computed."""

comptime FIXTURE_SEED_BASE: UInt64 = 0x546672666D724C6C
"""ASCII-ish, and distinct from `mamba_fixture.CORPUS_SEED_BASE`
(`0x4D616D6261436F72`) on purpose: two lanes sharing a seed base would make
two different blocks' fixtures correlate, which is harmless right up to the
moment somebody compares a hash across lanes and reads meaning into it."""


def fixture_case_seed(k: Int) -> UInt64:
    """Case k's seed. The `0x1000` stride is the mamba corpus's, kept so the
    two lanes' seed schedules look the same to a reader."""
    return FIXTURE_SEED_BASE + UInt64(0x1000) * UInt64(k)


# ===========================================================================
# BIT PLANTS
# ===========================================================================


def f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


comptime BITS_POS_ZERO: UInt32 = 0x00000000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_POS_INF: UInt32 = 0x7F800000
comptime BITS_NEG_INF: UInt32 = 0xFF800000
comptime BITS_QNAN: UInt32 = 0x7FC00000
comptime BITS_MIN_SUBNORMAL: UInt32 = 0x00000001
comptime BITS_BIG_SUBNORMAL: UInt32 = 0x007FFFFF
comptime BITS_NEG_SUBNORMAL: UInt32 = 0x80000101
comptime BITS_POS_1E35: UInt32 = 0x799A130C
comptime BITS_NEG_1E35: UInt32 = 0xF99A130C
"""`+-1e35`, contract 4.1(b)'s separating magnitude.

The clause: `s + (-FLT_MAX)` equals `-FLT_MAX` exactly for |s| < 2^103 and
overflows to `-inf` below roughly `-1e31`. So at `+1e35` the ADD gives a
value one step in from `-FLT_MAX` while a SELECT gives `-FLT_MAX` exactly
(`S13_MASK_SELECT` separates), and at `-1e35` the ADD overflows to `-inf`
while the `-inf` mask arm would have given `-inf` anyway
(`S13_MASK_NEG_INF` separates in the other direction).

PROVENANCE OF THESE FOUR BIT PATTERNS, because the standing rule is to say
where a number came from. They were computed on 2026-08-25 with an IEEE-754
float32 round trip ON THE HOST, OUTSIDE THIS REPOSITORY, and NOT by running
any Mojo. The same computation gave `1e35 -> 0x799A130C`,
`1e35 + (-FLT_MAX) -> 0xFF7FECBD` (which is NOT `0xFF7FFFFF`, so the add and
the select separate) and `-1e35 + (-FLT_MAX) -> -inf` (so the finite mask
and the `-inf` mask separate in the other direction). Contract 4.1(b)'s
threshold, `|s| < 2^103` for the add to be exactly `-FLT_MAX`, is about
1.01e31, and 1e35 is four orders past it, which is why it works.

**The check file must still print these back**, because a bit pattern that
was right when it was typed is not the same thing as a bit pattern the
toolchain agrees with. If they ever disagree, the separation works for any
magnitude in [1e32, 1e37] and the constant is a convenience, not a clause."""


def plant_every_kth(mut values: List[Float32], k: Int, phase: Int, bits: UInt32):
    """Overwrite every flat index `i` with `i % k == phase`.

    `mamba_fixture.plant_every_kth`'s shape, deliberately: a plant that is
    spelled the same way in two lanes is a plant a reviewer reads once."""
    for i in range(len(values)):
        if i % k == phase:
            values[i] = f32_from_bits(bits)


# ===========================================================================
# THE CASE TABLE
#
# Contract section 3's gate shape is d_model 32, n_heads 2, head_dim 16,
# n_kv_heads in {1, 2}, intermediate in {64, 300}, B in {1, 2, 3}, L in
# {1, 4, 16, 64, 257}, absolute positions in [0, 512). Every case below sits
# inside that box. The two that leave the box's CENTER do so for a reason
# named in the case's own comment.
# ===========================================================================

comptime FIXTURE_CASE_COUNT = 15

comptime PLANT_NONE = 0
comptime PLANT_X_SIGNED_ZEROS = 1
comptime PLANT_X_SUBNORMAL = 2
comptime PLANT_SCORE_NEG_ZERO = 3
comptime PLANT_SCORE_EXTREME = 4
comptime PLANT_MASKED_ZERO_ROW = 5
comptime PLANT_CACHE_HOT_TAIL = 6


@fieldwise_init
struct FixtureCase(Copyable, Movable):
    """One runnable case. Small by construction; see the file docstring.

    `split` is how many of the `l` tokens go in the FIRST block call. When
    `split == l` the case is a single prefill. When `split < l` the driver
    makes two calls, `split` tokens and then `l - split`, carrying the KV
    cache, which is contract section 7.2's decode-equals-prefill shape and
    is also the only way this file produces a nonzero starting absolute
    position without hand-seeding a cache with values no block ever
    computed.

    `cache_cap` is the ALLOCATED capacity of the KV cache. It is normally
    `l`. Where it is larger the case is testing that the folds read the
    USED length and not the allocated one, which is the hazard contract
    section 5.3 names when it says a v2 tree would have to fold over the
    allocation."""

    var name: StaticString
    var b: Int
    var l: Int
    var split: Int
    var d_model: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var cache_cap: Int
    var plant: Int


def fixture_case(k: Int) raises -> FixtureCase:
    # name                      b   l  split  dm  nh nkv hd  inter cap  plant
    if k == 0:
        # The centre of the box. n_rep == 1, so `repeat_kv` is the identity
        # and a broken head-to-kv-head map is INVISIBLE here. That is the
        # cost contract DEVIATION 813 states out loud, and case 1 is the
        # answer to it.
        return FixtureCase(
            "base_b1_l4_nrep1", 1, 4, 4, 32, 2, 2, 16, 64, 4, PLANT_NONE
        )
    if k == 1:
        # n_rep == 2. The gates must carry BOTH this and case 0 or GQA is
        # admitted untested (contract section 3).
        return FixtureCase(
            "base_b2_l4_nrep2", 2, 4, 4, 32, 2, 1, 16, 64, 4, PLANT_NONE
        )
    if k == 2:
        # L == 1. The degenerate prefill: one token, one key, a softmax over
        # a row of length one whose weight is exactly 1.0. Worth having
        # because a row of length one hides every fold-order clause, so a
        # gate that passes ONLY here has proved nothing.
        return FixtureCase(
            "base_b1_l1_nrep2", 1, 1, 1, 32, 2, 1, 16, 64, 1, PLANT_NONE
        )
    if k == 3:
        # B == 3, the third arm of clause (c)'s batch-composition half.
        return FixtureCase(
            "base_b3_l16_nrep2", 3, 16, 16, 32, 2, 1, 16, 64, 16, PLANT_NONE
        )
    if k == 4:
        # intermediate_size 300, and it is NOT decoration (contract section
        # 3). Every projection at d_model 32 has k <= 128, which is P == 1
        # under gemm v1, so the whole balanced fold tree sits unexercised
        # inside the block. At 300 the down_proj has k = 300, giving P = 3
        # with a ragged 44-element last leaf and one carry, which is the
        # GEMM contract's own clause-5 shape lifted into this block.
        return FixtureCase(
            "wide_inter300", 1, 4, 4, 32, 2, 1, 16, 300, 4, PLANT_NONE
        )
    if k == 5:
        # head_dim 24, so the attention scale is 1/sqrt(24), which is NOT
        # exact. At head_dim 16 the scale is exactly 0.25 and at 64 exactly
        # 0.125, so every power-of-four shape agrees no matter how the
        # scale was spelled. This is the only case that can see a wrong
        # spelling of DEVIATION 802.
        return FixtureCase(
            "odd_head_dim_24", 1, 4, 4, 48, 2, 1, 24, 64, 4, PLANT_NONE
        )
    if k == 6:
        # A nonzero starting absolute position: 129 tokens, then 4 more.
        # RoPE must read the table at 129..132 on the second call. An
        # implementation that indexes the table from the start of the
        # current slice reads 0..3 instead, which is sabotage
        # `S07_ROPE_RELATIVE_POSITION`, and 129 is chosen rather than a
        # small offset because it is past a plausible tile boundary.
        return FixtureCase(
            "pos_offset_129", 1, 133, 129, 32, 2, 1, 16, 64, 133, PLANT_NONE
        )
    if k == 7:
        # L == 64, and the first case whose kv length exceeds 128 is case 8.
        return FixtureCase(
            "long_l64", 1, 64, 64, 32, 2, 1, 16, 64, 64, PLANT_NONE
        )
    if k == 8:
        # L == 257, clause (c)'s longest sequence, and the only case where
        # the kv length passes gemm v1's CONTRACT_K_LEAF_MIN of 128. It is
        # the case sabotage `S19_VALUE_SUM_VIA_GEMM` needs: routing S19
        # through the GEMM changes P from 1 to 3 somewhere past key 128, so
        # a fixture that never gets there cannot see the sabotage at all.
        # The score buffer here is [1, 2, 257, 257], about 132k floats, and
        # that is the largest allocation in this file.
        return FixtureCase(
            "long_l257", 1, 257, 257, 32, 2, 1, 16, 64, 257, PLANT_NONE
        )
    if k == 9:
        # Signed zeros planted into x by bits. A hashed value cannot land
        # on -0.0, so if the fixture does not plant it the whole signed-zero
        # half of contract section 8 is untested. Whole tokens of each sign
        # plus a scattered rule, the mamba corpus's `adv_signed_zeros`
        # shape.
        return FixtureCase(
            "adv_signed_zeros", 2, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_X_SIGNED_ZEROS,
        )
    if k == 10:
        # Subnormals planted into x. Their only job is to make the `ftz`
        # unit at the LOAD seams reachable: contract section 4's preamble
        # says every operand loaded from a buffer passes ftz, and on Apple
        # that is bit-inert, so the clause is invisible on the box this
        # will first run on. It is NOT invisible on CUDA.
        return FixtureCase(
            "adv_subnormal_x", 1, 4, 4, 32, 2, 1, 16, 64, 4, PLANT_X_SUBNORMAL
        )
    if k == 11:
        # A `-0.0` planted into `attn.scores` at an UNMASKED cell, contract
        # 4.1(a). The mask ADD launders it to `+0.0`; an implementation that
        # elides the add at unmasked cells keeps the `-0.0`. Sabotage
        # `S13_MASK_SELECT` moves `attn.masked` by ONE SIGN BIT here and by
        # nothing anywhere else, which is what makes it a reach proof and
        # not a smoke test.
        return FixtureCase(
            "adv_score_neg_zero", 1, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_SCORE_NEG_ZERO,
        )
    if k == 12:
        # `+-1e35` planted into `attn.scores` at MASKED cells, contract
        # 4.1(b). Separates the add from a select AND `-FLT_MAX` from
        # `-inf`. Both sabotages `S13_MASK_SELECT` and `S13_MASK_NEG_INF`
        # need this case; neither is reachable through the causal mask
        # alone, because position t always attends to itself so no row is
        # ever fully masked.
        return FixtureCase(
            "adv_score_extreme", 1, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_SCORE_EXTREME,
        )
    if k == 13:
        # A KV cache allocated to 16 and used to 4, with hashed NONZERO
        # values in slots [4, 16). See `PLANT_CACHE_HOT_TAIL` below for
        # exactly what this can and cannot see; the short version is that
        # it is a gate on the STAGE SHAPES and on the mask reaching the
        # tail, and it is NOT expected to move the block output.
        return FixtureCase(
            "adv_cache_hot_tail", 1, 4, 4, 32, 2, 1, 16, 64, 16,
            PLANT_CACHE_HOT_TAIL,
        )
    if k == 14:
        # Contract section 5.1's signed-zero row maximum, reached the ONLY
        # way it can be reached: a plant applied AFTER the mask. See
        # `PLANT_AT_MASKED` and DEVIATION 1005 for the argument that the
        # contract's own S13 makes this unreachable through S12, and that
        # sabotage `S14_MAX_PLAIN_COMPARE` built on the contract's stated
        # reach will be INERT and get deleted.
        #
        # The row is (b=0, h=0, q=3), keys 0..3, all causal for query 3 and
        # therefore all unmasked. Rewritten to `-0.0, -1.0, -2.0, +0.0` so
        # the maximum IS the ambiguous comparison. `identical_fmax` returns
        # `+0.0` in every fold order (key(+0) is 0x80000000, key(-0) is
        # 0x7FFFFFFF); a plain `a > b ? a : b` returns whichever the
        # topology reached last, which row 39 measured as `-0.0` on Apple
        # and `+0.0` on NVIDIA and AMD. A LIVE THREE-VENDOR SPLIT AT A
        # STAGE THE CARD RECORDS.
        #
        # Contract 5.1's own honest note applies and is worth repeating:
        # downstream this costs NOTHING, because `s - (+0.0)` and
        # `s - (-0.0)` agree for every finite s and `exp` of either zero is
        # exactly 1.0, so the divergence is laundered before it reaches an
        # output. It costs the CLAUSE, at `attn.max`. The card is the only
        # instrument that can see it, which is the argument for recording
        # `attn.max` rather than treating it as an internal.
        #
        # The rest of the block's rows are UNTOUCHED and are the negative
        # control: the sabotage must move `attn.max` here and NOT on an
        # ordinary row.
        return FixtureCase(
            "adv_masked_zero_row", 1, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_MASKED_ZERO_ROW,
        )
    raise Error(
        String("transformer_fixture: no case ")
        + String(k)
        + " (there are "
        + String(FIXTURE_CASE_COUNT)
        + ")"
    )


def fixture_case_by_name(name: String) raises -> Int:
    for k in range(FIXTURE_CASE_COUNT):
        var c = fixture_case(k)
        if String(c.name) == name:
            return k
    raise Error(String("transformer_fixture: no case named '") + name + "'")


def fixture_dims(c: FixtureCase) raises -> TransformerDims:
    """The case's shape, with the rotary table sized at 512 positions for
    every case.

    512 is contract section 3's "absolute positions in [0, 512)". ONE table
    size across the whole case table is deliberate: it makes `rope.cos` and
    `rope.sin` identical stages at every L and every B, which is what
    clause (c) needs (DEVIATION 1002). The longest case reaches position
    256 (`long_l257`) and the offset case reaches 132, so 512 has room and
    is well inside DEVIATION 812's 8192 ceiling."""
    var d = TransformerDims(
        c.d_model, c.n_heads, c.n_kv_heads, c.head_dim, c.intermediate, 512
    )
    d.validate()
    return d^


# ===========================================================================
# THE TENSORS
# ===========================================================================


def fixture_weights(c: FixtureCase) raises -> TransformerWeights:
    """The case's parameters at the default ranges.

    The ranges are stated in the file docstring's plant scheme and repeated
    here as literals rather than hidden behind a fan-in formula, because a
    formula is one more thing that can be wrong in a way that looks
    principled. Every one is dyadic."""
    var dims = fixture_dims(c)
    var seed = fixture_case_seed(k_of(c))
    var w = TransformerWeights(dims)
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate

    w.norm1_w = fixture_tensor(seed, TID_NORM1_W, dm, 0.5, 1.5)
    w.norm2_w = fixture_tensor(seed, TID_NORM2_W, dm, 0.5, 1.5)
    w.w_q = fixture_tensor(seed, TID_W_Q, qw * dm, -0.5, 0.5)
    w.w_k = fixture_tensor(seed, TID_W_K, kw * dm, -0.5, 0.5)
    w.w_v = fixture_tensor(seed, TID_W_V, kw * dm, -0.5, 0.5)
    w.w_o = fixture_tensor(seed, TID_W_O, dm * qw, -0.25, 0.25)
    w.w_gate = fixture_tensor(seed, TID_W_GATE, inter * dm, -0.25, 0.25)
    w.w_up = fixture_tensor(seed, TID_W_UP, inter * dm, -0.25, 0.25)
    w.w_down = fixture_tensor(seed, TID_W_DOWN, dm * inter, -0.125, 0.125)
    return w^


def k_of(c: FixtureCase) raises -> Int:
    """The case's index, recovered from its name.

    A `FixtureCase` does not carry its own index because a struct that
    carries an index can be built with the wrong one; recovering it from
    the table means the table is the single source. Linear over fourteen
    string compares, on a path that runs once per case."""
    return fixture_case_by_name(String(c.name))


def fixture_x(c: FixtureCase) raises -> List[Float32]:
    """The case's block input, `[B, L, d_model]` row-major (token-major),
    with the case's plant applied.

    Plants that touch x are applied HERE and nowhere else, so a reader who
    wants to know what a case feeds the block reads one function."""
    var seed = fixture_case_seed(k_of(c))
    var x = fixture_tensor(seed, TID_X, c.b * c.l * c.d_model, -2.0, 2.0)
    var d = c.d_model

    if c.plant == PLANT_X_SIGNED_ZEROS:
        # Whole `+0.0` tokens at t % 4 == 0, whole `-0.0` tokens at
        # t % 4 == 2, and of what is left every flat i % 7 == 3 is `-0.0`.
        # `mamba_fixture.corpus_case_x`'s zero rule, reused so the two
        # lanes' signed-zero cases have the same shape.
        #
        # A WHOLE ZERO TOKEN IS THE INTERESTING ONE: its RMSNorm sum of
        # squares is `+0.0` from the `+0.0` seed regardless of the zero
        # signs in the row (IEEE `(+0) + (-0) = +0`), so `norm1.sumsq` is
        # `+0.0`, `mean` is `+0.0`, `rstd` is `identical_rsqrt(1e-6)`
        # which is finite, and `norm1.out` is the SIGNED zero times the
        # weight. The sign survives to `q_proj.out` through gemm v1's
        # `+0.0`-seeded leaf, where 9.2(a) launders it. That chain is a
        # PREDICTION and the check file should print it.
        for bb in range(c.b):
            for t in range(c.l):
                if t % 4 == 0 or t % 4 == 2:
                    var base = (bb * c.l + t) * d
                    for j in range(d):
                        if t % 4 == 0:
                            x[base + j] = f32_from_bits(BITS_POS_ZERO)
                        else:
                            x[base + j] = f32_from_bits(BITS_NEG_ZERO)
        for i in range(len(x)):
            var t = (i // d) % c.l
            if t % 4 != 0 and t % 4 != 2 and i % 7 == 3:
                x[i] = f32_from_bits(BITS_NEG_ZERO)
        return x^

    if c.plant == PLANT_X_SUBNORMAL:
        # Three subnormal classes, at three different residues so no cell
        # gets two of them: the smallest positive, the largest positive,
        # and a negative one. The negative one is the load-bearing plant,
        # because `[[metal-hardware-gaps]]` and DEVIATION 746 (i) record
        # that METAL FLUSHES COMPARE OPERANDS, so any implementation that
        # tests a sign with `x < 0.0` instead of by bits gets a different
        # answer here than the host does.
        plant_every_kth(x, 11, 1, BITS_MIN_SUBNORMAL)
        plant_every_kth(x, 11, 5, BITS_BIG_SUBNORMAL)
        plant_every_kth(x, 11, 9, BITS_NEG_SUBNORMAL)
        return x^

    return x^


def fixture_cache_tail(c: FixtureCase) raises -> List[Float32]:
    """The hot-tail plant's values, `2 * B * n_kv * (cap - l) * head_dim` of
    them: the k tail first, then the v tail. Empty for every case but 13.

    **WHAT THIS CAN AND CANNOT SEE, stated plainly because a fixture whose
    bite is overstated is worse than no fixture.**

    It CANNOT move the block output of a correct implementation, and it
    cannot move the output of most wrong ones either. Contract 7.1 proves
    the masked tail is bitwise inert: a key at slot j > the query's absolute
    position gets `-FLT_MAX` added, `exp` of `-3.4e38 - m` is exactly
    `+0.0`, the denominator is at least 1.0 so the weight is exactly
    `+0.0`, and both the S17 chain and the S19 fma chain are seeded `+0.0`
    where a `+0.0` term is inert. All of that holds whether the tail holds
    zeros or holds 1e30.

    What it CAN see is three things, and they are the things a zero tail
    hides:

      (a) A stage recorded at the ALLOCATED length. `kv.k_cache` is
          `[B, n_kv, S, head_dim]` with S the USED length (contract section
          9). An implementation that hashes the whole allocation folds the
          tail into the hash, and with a zero tail that is invisible
          because zeros hash the same on both sides at the sizes that
          matter. This is `core/identity_trace.mojo`'s own rule 3 -- hash
          the logical buffer, not a machine-sized scratch -- with the
          scratch being the cache tail.
      (b) A fold that walks the allocation with the mask NOT applied to the
          tail. With a zero tail those cells score `+0.0`, `exp(0 - m)` is
          a real weight and the answer is wrong, so a zero tail catches
          this too; a hot tail catches it LOUDER and, more usefully,
          catches the variant where the tail is skipped by the mask but
          still folded into `attn.max`.
      (c) Uninitialized memory. A device buffer allocated and not written
          holds whatever was there, which differs run to run on ONE machine
          and would make clause (b) fail for a reason that has nothing to
          do with vendors. Writing the tail with known bits turns that from
          a flake into a value.

    So: this case is a gate on the STAGE SHAPES and on the mask's reach,
    not on the arithmetic, and if it ever moves `residual2.out` something
    is wrong that this fixture did not predict."""
    if c.plant != PLANT_CACHE_HOT_TAIL:
        var empty = List[Float32]()
        return empty^
    var dims = fixture_dims(c)
    var tail_slots = c.cache_cap - c.l
    if tail_slots <= 0:
        raise Error(
            "transformer_fixture: PLANT_CACHE_HOT_TAIL with no tail; set"
            " cache_cap > l"
        )
    var n = 2 * c.b * dims.n_kv_heads * tail_slots * dims.head_dim
    # Range chosen large enough that a tail cell folded into a real sum
    # would be obvious in the first printed digit, and small enough that
    # `q . k^T` over it cannot overflow FP32.
    var tail = fixture_tensor(
        fixture_case_seed(k_of(c)), TID_CACHE_TAIL, n, -8.0, 8.0
    )
    return tail^


# ===========================================================================
# SCORE PLANTS (DEVIATION 1001 and DEVIATION 1005)
# ===========================================================================
#
# DEVIATION 1001: the oracle accepts a list of (flat index, bit pattern)
# overrides applied to the attention score buffer.
#
# WHAT WAS CHOSEN. A plant hook inside the attention path, taking effect at
# one of two named points, empty on every ordinary case.
#
# WHAT THE ALTERNATIVE WAS. Reaching the same score values through the
# WEIGHTS, so that the oracle has no hook at all and the fixture is pure
# input. That is strictly better where it works, and it does not work here:
# contract 4.1(a) says a `-0.0` score is reachable only through `ftz` of a
# negative subnormal gemm accumulator and that GEMM contract 9.2(a) rules
# out products alone producing one, so no assignment of finite weights makes
# a score exactly `-0.0`. The contract's own sentence is "the fixture has to
# plant it by bits".
#
# WHAT IT COSTS IF WRONG. The device spelling must accept the same override
# buffer or clause (a) cannot be checked on the planted cases, and an
# override buffer in a kernel is a branch in the hot path that exists only
# for tests. If the branch is compiled in unconditionally it is a permanent
# tax on a profile whose whole point is that its arithmetic is pinned; if it
# is compiled out under a flag, then the arm that runs the gates is not the
# arm that ships, which is the trap `mojotrees-switches-must-flip` and
# `[[the shared checkout's mode flip]]` both name. The recommendation to the
# device lane is a comptime parameter defaulting to no-plant, so the shipped
# kernel has no branch and the gate build has one, and the gate build must
# print which arm it is.
#
# DEVIATION 1005: the plant has TWO application points, and the second one
# exists because of a defect this lane believes it found in contract section
# 5.1. See `PLANT_AT_MASKED` below. Reported to the orchestrator rather than
# fixed here, because the contract is FROZEN and this lane does not amend it.

comptime PLANT_AT_SCORES = 0
"""Applied to `attn.scores`, that is AFTER S12's scale and BEFORE S13's mask
add. This is where contract 4.1(a) and 4.1(b) want their plants, because
both clauses are about what S13 does to a value."""

comptime PLANT_AT_MASKED = 1
"""Applied to `attn.masked`, that is AFTER S13 and BEFORE S14's row maximum.

**THIS EXISTS BECAUSE SECTION 5.1's HAZARD IS UNREACHABLE THROUGH SECTION
4.1(a)'s S13, AND THE TWO CLAUSES CONTRADICT EACH OTHER ON REACHABILITY.**
The argument, which is this lane's and which the orchestrator should have
checked before it is believed:

  - Section 5.1 justifies pinning the row maximum to `identical_fmax` on the
    ground that "a row of attention scores reaches both zero signs easily, a
    masked lane one way and a flushed subnormal the other", and requires
    sabotage `S14_MAX_PLAIN_COMPARE` to move `attn.max` "on a row carrying a
    planted `-0.0` beside a `+0.0`".
  - But S14 does not read scores. It reads `attn.masked`, and section 4.1(a)
    is the proof that `attn.masked` has no negative zero in it. At an
    unmasked cell the value is `ftz(ftz(s) + (+0.0))`, and `x + (+0.0)` is
    `+0.0` for `x = -0.0` and for `x` any negative subnormal that the inner
    `ftz` has already flushed to `-0.0`. At a masked cell the value is
    `-FLT_MAX` or `-inf`. Neither is a negative zero.
  - So under v1's own S13 the only selection in the profile never sees the
    case its pin was written for. `identical_fmax` is still the right
    spelling and the clause still costs nothing, but the REACH ARGUMENT is
    wrong, and a sabotage arm built on it will be INERT and will be deleted
    by whoever writes the gates, which is precisely the failure contract
    section 10 warns about for `S07_ROPE_RELATIVE_POSITION` and
    `S19_VALUE_SUM_VIA_GEMM`.

`[[reached-but-inert]]` is the standing rule this is an instance of, and
`[[verify reach, not output]]` is why the plant point had to move rather
than the plant value. Applying the plant after the mask is the only way to
put a `-0.0` in front of S14 without also sabotaging S13, and a gate that
sabotages two seams at once cannot say which one it proved."""


@fieldwise_init
struct ScorePlant(Copyable, Movable):
    """A set of bit overrides into one attention buffer.

    Flat indices into `[B, n_heads, L, S]`, which is the shape contract
    section 9 gives `attn.scores`, `attn.masked`, `attn.exp` and
    `attn.weights`. Use `score_index` to build one; do not compute the
    stride by hand at a call site, because getting `n_heads` and `S` the
    wrong way round produces a valid index into a different cell and a
    fixture that plants somewhere other than where it says it does is worse
    than one that crashes."""

    var at: Int
    var idx: List[Int]
    var bits: List[UInt32]

    @staticmethod
    def none() -> Self:
        return Self(PLANT_AT_SCORES, List[Int](), List[UInt32]())

    def is_empty(self) -> Bool:
        return len(self.idx) == 0

    def add(mut self, i: Int, b: UInt32):
        self.idx.append(i)
        self.bits.append(b)


def score_index(
    dims: TransformerDims, b: Int, h: Int, qi: Int, key: Int, l: Int, s: Int
) -> Int:
    """The flat index of `[b, h, qi, key]` in a `[B, n_heads, L, S]` buffer.

    `l` and `s` are passed rather than read off `dims` because neither is a
    model shape; L is the launch's token count and S is the kv length after
    the append, and contract section 3 is explicit that those are launch
    quantities and not part of the arithmetic."""
    return ((b * dims.n_heads + h) * l + qi) * s + key


def fixture_score_plant(
    c: FixtureCase, dims: TransformerDims, l: Int, s: Int
) raises -> ScorePlant:
    """The case's score plant, or an empty one.

    Every planted cell is at a NAMED, COMPUTABLE index so a gate can assert
    which cell moved. Nothing is scattered by a hash."""
    var p = ScorePlant.none()

    if c.plant == PLANT_SCORE_NEG_ZERO:
        # An UNMASKED cell: query 3 attending to key 1, which is causal
        # (1 <= 3). Contract 4.1(a): the mask add turns this into `+0.0`
        # and an elided add leaves it `-0.0`. Head 1 rather than head 0 so
        # that a head-index bug shows up as "the wrong head moved" instead
        # of "nothing moved".
        if l < 4 or s < 4:
            raise Error(
                "transformer_fixture: PLANT_SCORE_NEG_ZERO needs l >= 4 and"
                " s >= 4"
            )
        p.at = PLANT_AT_SCORES
        p.add(score_index(dims, 0, 1, 3, 1, l, s), BITS_NEG_ZERO)
        # And the control: the SAME row, a `+0.0` at an unmasked cell that
        # the add leaves alone. Without this the case cannot tell "the add
        # ran" from "the add ran and also broke every positive zero".
        p.add(score_index(dims, 0, 1, 3, 2, l, s), BITS_POS_ZERO)
        return p^

    if c.plant == PLANT_SCORE_EXTREME:
        # MASKED cells: query 0 attending to keys 1, 2 and 3, all of which
        # are strictly after position 0 and therefore masked under the
        # causal rule. Contract 4.1(b) wants a magnitude near 1e35 in a
        # masked cell; both signs are planted because they separate
        # different things (see BITS_POS_1E35's docstring).
        if l < 4 or s < 4:
            raise Error(
                "transformer_fixture: PLANT_SCORE_EXTREME needs l >= 4 and"
                " s >= 4"
            )
        p.at = PLANT_AT_SCORES
        p.add(score_index(dims, 0, 0, 0, 1, l, s), BITS_POS_1E35)
        p.add(score_index(dims, 0, 0, 0, 2, l, s), BITS_NEG_1E35)
        # An ordinary masked cell left alone, as the negative control: it
        # must stay exactly `mask_fill()` under every arm.
        return p^

    if c.plant == PLANT_MASKED_ZERO_ROW:
        # Contract section 5.1's hazard, reached the only way it can be
        # reached (DEVIATION 1005). Row (b=0, h=0, q=3) of `attn.masked` is
        # rewritten so that every UNMASKED cell is at or below zero and the
        # two largest are `+0.0` and `-0.0`. Then the row maximum IS the
        # ambiguous comparison, `identical_fmax` returns `+0.0` in every
        # fold order by the total-order key, and a plain `a > b ? a : b`
        # returns whichever the topology reached last.
        #
        # Keys 0..3 are all causal for query 3, so all four are unmasked.
        if l < 4 or s < 4:
            raise Error(
                "transformer_fixture: PLANT_MASKED_ZERO_ROW needs l >= 4 and"
                " s >= 4"
            )
        p.at = PLANT_AT_MASKED
        p.add(score_index(dims, 0, 0, 3, 0, l, s), BITS_NEG_ZERO)
        p.add(score_index(dims, 0, 0, 3, 1, l, s), 0xBF800000)  # -1.0
        p.add(score_index(dims, 0, 0, 3, 2, l, s), 0xC0000000)  # -2.0
        p.add(score_index(dims, 0, 0, 3, 3, l, s), BITS_POS_ZERO)
        return p^

    return p^


# ===========================================================================
# SLICING, for clause (c)'s batch-composition half
# ===========================================================================


def slice_batch_row(
    x: List[Float32], b: Int, l: Int, d_model: Int, row: Int
) raises -> List[Float32]:
    """One sequence out of a `[B, L, d_model]` batch, as a `[1, L, d_model]`
    batch.

    Clause (c) compares a row's bits when its sequence shares the launch
    with zero, one or two others, so the driver needs the SAME numbers in a
    B=1 launch and in a B=3 one. Slicing is a copy and copies are not seams,
    so the two launches see bit-identical inputs by construction and any
    difference in the output is the execution plan's.

    THE NEGATIVE CONTROL MATTERS AS MUCH AS THE GATE, and this function
    does not provide it. A clause (c) gate that only ever compares row 0
    against row 0 passes for ever if the slice is broken and returns row 0
    both times. The gate must also show that row 0 and row 1 DIFFER. That is
    the mamba lane's clause (c) lesson and contract section 7.2 restates it
    for clause (d)."""
    if row < 0 or row >= b:
        raise Error(
            String("transformer_fixture: batch row ")
            + String(row)
            + " out of range for B = "
            + String(b)
        )
    var row_len = l * d_model
    var out = List[Float32]()
    for i in range(row_len):
        out.append(x[row * row_len + i])
    return out^


# ===========================================================================
# REPORTING
# ===========================================================================


def bits32_hex(v: Float32) -> String:
    """`<hex bits>` for a Float32.

    Every float this lane prints goes out as `<decimal>/<hex bits>` and the
    hex is the one a reader may trust: `[[mojo-string-float-roundtrip]]`
    records that `String(Float32)` does not round trip in this toolchain, so
    a decimal in a log is a lossy summary and a bit pattern is the value."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def f32_report(v: Float32) -> String:
    return String(v) + "/" + bits32_hex(v)


def mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()
