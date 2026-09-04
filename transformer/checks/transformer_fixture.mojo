# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures and adversarial bit plants for the IDENTICAL Transformer gate."""

from std.memory import bitcast

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_rsqrt,
    numeric_mode_name,
)



comptime RMS_EPS: Float32 = 1e-6
"""`LlamaConfig.rms_norm_eps`, configuration_llama.py:73, verified in the pinned checkout on 2026-08-25."""

comptime ROPE_THETA: Float32 = 10000.0
"""The RoPE base."""

comptime BITS_RMS_EPS: UInt32 = 0x358637BD
comptime BITS_ROPE_THETA: UInt32 = 0x461C4000
comptime BITS_MASK_FILL: UInt32 = 0xFF7FFFFF
"""`torch.finfo(float32).min`, `-3.4028234663852886e+38`, masking_utils.py:601 in the pinned checkout."""

comptime MAX_ABS_POSITION = 8192
"""Contract DEVIATION 812. The oracle REFUSES BY NAME at or above this rather than computing a number the reduction has lost."""


def mask_fill() -> Float32:
    """The additive mask value, by BITS."""
    return bitcast[DType.float32](BITS_MASK_FILL)


def unmasked_fill() -> Float32:
    """`+0.0`, contract section 3's "unmasked fill", masking_utils.py:603."""
    return Float32(0.0)


def profile_constants_are_intact() -> Bool:
    """PREDICTION, and the check file owes the assertion: the three frozen scalars have the bits contract section 3 pins."""
    if bitcast[DType.uint32](RMS_EPS) != BITS_RMS_EPS:
        return False
    if bitcast[DType.uint32](ROPE_THETA) != BITS_ROPE_THETA:
        return False
    if bitcast[DType.uint32](mask_fill()) != BITS_MASK_FILL:
        return False
    return True




@fieldwise_init
struct TransformerDims(Copyable, Movable):
    """One Llama-shaped decoder block's shape, plus the ONE configuration quantity RoPE needs. The cost is that a call is REFUSED if it would touch a position at or beyond `rope_positions`, which is the right failure."""

    var d_model: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var rope_positions: Int

    def n_rep(self) -> Int:
        """`LlamaAttention.num_key_value_groups`, modeling_llama.py:223."""
        return self.n_heads // self.n_kv_heads

    def q_width(self) -> Int:
        return self.n_heads * self.head_dim

    def kv_width(self) -> Int:
        return self.n_kv_heads * self.head_dim

    def half_head(self) -> Int:
        """RoPE pairs index `j` with `j + head_dim/2` (`rotate_half`, modeling_llama.py:130-134) and the table is `cat(freqs, freqs)` (:123), so the DEDUPLICATED table has this many columns."""
        return self.head_dim // 2

    def validate(self) raises:
        """Contract section 3's divisibility rules, REFUSED BY NAME rather than silently truncated."""
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
    """Contract S-constant / DEVIATION 802: `identical_rsqrt(Float32(head_dim))`, computed ONCE on the host and stored FP32."""
    return ftz(identical_rsqrt(Float32(head_dim)))




struct TransformerWeights(Copyable, Movable):
    """One decoder block's parameters, host side, in the UPSTREAM shapes. That is exactly a `gemm.fp32.v1` `OP_NT` cell with the weight as the right operand and no transpose anywhere, which is why contract S5 can say "the GEMM refuses no shape" and mean it."""

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




def fixture_splitmix64(z_in: UInt64) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim. **`+` HERE IS A REAL ADD AND MUST STAY ONE.** `[[mojo-amp-plus-is-bitwise-and]]` records that Mojo's `&+` computes `x & k` with no compile error, which silently produced wrong hashes in this repository once already."""
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def fixture_tensor(
    seed: UInt64, tensor_id: Int, n: Int, lo: Float64, hi: Float64
) -> List[Float32]:
    """`n` values `f32(lo + (hi - lo) * top24 * 2^-24)`, in flat row-major index order."""
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
"""The hot-tail plant's own id, so a tail value can never collide with a real key or value the block computed."""

comptime FIXTURE_SEED_BASE: UInt64 = 0x546672666D724C6C
"""ASCII-ish, and distinct from `mamba_fixture.CORPUS_SEED_BASE` (`0x4D616D6261436F72`) on purpose: two lanes sharing a seed base would make two different blocks' fixtures correlate, which is harmless right up to the moment somebody compares a hash across lanes and reads meaning into it."""


def fixture_case_seed(k: Int) -> UInt64:
    """Case k's seed."""
    return FIXTURE_SEED_BASE + UInt64(0x1000) * UInt64(k)




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
"""`+-1e35`, contract 4.1(b)'s separating magnitude. **The check file must still print these back**, because a bit pattern that was right when it was typed is not the same thing as a bit pattern the toolchain agrees with."""


def plant_every_kth(mut values: List[Float32], k: Int, phase: Int, bits: UInt32):
    """Overwrite every flat index `i` with `i % k == phase`."""
    for i in range(len(values)):
        if i % k == phase:
            values[i] = f32_from_bits(bits)



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
    """One runnable case."""

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
    if k == 0:
        return FixtureCase(
            "base_b1_l4_nrep1", 1, 4, 4, 32, 2, 2, 16, 64, 4, PLANT_NONE
        )
    if k == 1:
        return FixtureCase(
            "base_b2_l4_nrep2", 2, 4, 4, 32, 2, 1, 16, 64, 4, PLANT_NONE
        )
    if k == 2:
        return FixtureCase(
            "base_b1_l1_nrep2", 1, 1, 1, 32, 2, 1, 16, 64, 1, PLANT_NONE
        )
    if k == 3:
        return FixtureCase(
            "base_b3_l16_nrep2", 3, 16, 16, 32, 2, 1, 16, 64, 16, PLANT_NONE
        )
    if k == 4:
        return FixtureCase(
            "wide_inter300", 1, 4, 4, 32, 2, 1, 16, 300, 4, PLANT_NONE
        )
    if k == 5:
        return FixtureCase(
            "odd_head_dim_24", 1, 4, 4, 48, 2, 1, 24, 64, 4, PLANT_NONE
        )
    if k == 6:
        return FixtureCase(
            "pos_offset_129", 1, 133, 129, 32, 2, 1, 16, 64, 133, PLANT_NONE
        )
    if k == 7:
        return FixtureCase(
            "long_l64", 1, 64, 64, 32, 2, 1, 16, 64, 64, PLANT_NONE
        )
    if k == 8:
        return FixtureCase(
            "long_l257", 1, 257, 257, 32, 2, 1, 16, 64, 257, PLANT_NONE
        )
    if k == 9:
        return FixtureCase(
            "adv_signed_zeros", 2, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_X_SIGNED_ZEROS,
        )
    if k == 10:
        return FixtureCase(
            "adv_subnormal_x", 1, 4, 4, 32, 2, 1, 16, 64, 4, PLANT_X_SUBNORMAL
        )
    if k == 11:
        return FixtureCase(
            "adv_score_neg_zero", 1, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_SCORE_NEG_ZERO,
        )
    if k == 12:
        return FixtureCase(
            "adv_score_extreme", 1, 4, 4, 32, 2, 1, 16, 64, 4,
            PLANT_SCORE_EXTREME,
        )
    if k == 13:
        return FixtureCase(
            "adv_cache_hot_tail", 1, 4, 4, 32, 2, 1, 16, 64, 16,
            PLANT_CACHE_HOT_TAIL,
        )
    if k == 14:
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
    """The case's shape, with the rotary table sized at 512 positions for every case."""
    var d = TransformerDims(
        c.d_model, c.n_heads, c.n_kv_heads, c.head_dim, c.intermediate, 512
    )
    d.validate()
    return d^




def fixture_weights(c: FixtureCase) raises -> TransformerWeights:
    """The case's parameters at the default ranges."""
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
    """The case's index, recovered from its name."""
    return fixture_case_by_name(String(c.name))


def fixture_x(c: FixtureCase) raises -> List[Float32]:
    """The case's block input, `[B, L, d_model]` row-major (token-major), with the case's plant applied."""
    var seed = fixture_case_seed(k_of(c))
    var x = fixture_tensor(seed, TID_X, c.b * c.l * c.d_model, -2.0, 2.0)
    var d = c.d_model

    if c.plant == PLANT_X_SIGNED_ZEROS:
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
        plant_every_kth(x, 11, 1, BITS_MIN_SUBNORMAL)
        plant_every_kth(x, 11, 5, BITS_BIG_SUBNORMAL)
        plant_every_kth(x, 11, 9, BITS_NEG_SUBNORMAL)
        return x^

    return x^


def fixture_cache_tail(c: FixtureCase) raises -> List[Float32]:
    """The hot-tail plant's values, `2 * B * n_kv * (cap - l) * head_dim` of them: the k tail first, then the v tail."""
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
    var tail = fixture_tensor(
        fixture_case_seed(k_of(c)), TID_CACHE_TAIL, n, -8.0, 8.0
    )
    return tail^



comptime PLANT_AT_SCORES = 0
"""Applied to `attn.scores`, that is AFTER S12's scale and BEFORE S13's mask add."""

comptime PLANT_AT_MASKED = 1
"""Applied to `attn.masked`, that is AFTER S13 and BEFORE S14's row maximum."""


@fieldwise_init
struct ScorePlant(Copyable, Movable):
    """A set of bit overrides into one attention buffer."""

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
    """The flat index of `[b, h, qi, key]` in a `[B, n_heads, L, S]` buffer."""
    return ((b * dims.n_heads + h) * l + qi) * s + key


def fixture_score_plant(
    c: FixtureCase, dims: TransformerDims, l: Int, s: Int
) raises -> ScorePlant:
    """The case's score plant, or an empty one."""
    var p = ScorePlant.none()

    if c.plant == PLANT_SCORE_NEG_ZERO:
        if l < 4 or s < 4:
            raise Error(
                "transformer_fixture: PLANT_SCORE_NEG_ZERO needs l >= 4 and"
                " s >= 4"
            )
        p.at = PLANT_AT_SCORES
        p.add(score_index(dims, 0, 1, 3, 1, l, s), BITS_NEG_ZERO)
        p.add(score_index(dims, 0, 1, 3, 2, l, s), BITS_POS_ZERO)
        return p^

    if c.plant == PLANT_SCORE_EXTREME:
        if l < 4 or s < 4:
            raise Error(
                "transformer_fixture: PLANT_SCORE_EXTREME needs l >= 4 and"
                " s >= 4"
            )
        p.at = PLANT_AT_SCORES
        p.add(score_index(dims, 0, 0, 0, 1, l, s), BITS_POS_1E35)
        p.add(score_index(dims, 0, 0, 0, 2, l, s), BITS_NEG_1E35)
        return p^

    if c.plant == PLANT_MASKED_ZERO_ROW:
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




def slice_batch_row(
    x: List[Float32], b: Int, l: Int, d_model: Int, row: Int
) raises -> List[Float32]:
    """One sequence out of a `[B, L, d_model]` batch, as a `[1, L, d_model]` batch. The gate must also show that row 0 and row 1 DIFFER."""
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




def bits32_hex(v: Float32) -> String:
    """`<hex bits>` for a Float32."""
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
    """The build's tier, from the ONE definition of it. This used to be a local two-way `IDENTICAL`-or-`FAST`, written when there were two tiers, and it answered "FAST" for a DETERMINISTIC build -- so a driver run under the middle tier printed the wrong arm onto every line it produced."""
    return numeric_mode_name()
