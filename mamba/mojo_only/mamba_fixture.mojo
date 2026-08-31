# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures for the Mamba-1 identity block. NOT A PORT.

Two generators live here and each exists for a stated reason.

1. **The corpus generator, reimplemented bit for bit.** `mamba/corpus/` is
   written by the corpus lane (`gen_corpus.py`, spec
   `mojolearn.mamba.corpus.hash.v1`): every tensor element is
   `f32(lo + (hi - lo) * top24(splitmix64(key + i)) * 2^-24)` with
   `key = splitmix64(seed ^ (tensor_id << 32))`, evaluated exactly in
   float64 and rounded ONCE. That is a pure function of five integers and
   two dyadic rationals, so it is reimplemented here rather than read from
   disk: the check that loads a corpus case FIRST asserts the file's bytes
   equal this generator's bytes (the two lanes agreeing on the fixture is
   itself a gate), then compares stages. Fixtures used off-corpus (the
   device gates, the card driver) call this generator directly so the lane
   does not need the corpus directory to run.

2. **Bit plants.** Values assembled from bit patterns for the row-39 audit
   (-0.0, NaN, infinity) and the denormal seams, per
   `uniform-test-data-hides-permutation`: a hashed value cannot land on
   -0.0, so the fixture must PLANT it.

Scale note (why the ranges are what they are): with weights at the corpus
ranges, `softplus(dt_proj + bias)` spans roughly [1e-3, 0.13] and
`exp(delta * A)` sits in (0, 1), so the scan recurrence actually recurses
instead of saturating at 0 or 1. The adversarial cases push each boundary
on purpose.
"""

from std.memory import bitcast

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


# ===========================================================================
# Dimensions. d_state and d_conv are Mamba-1 constants (MambaConfig defaults
# state_size=16, conv_kernel=4; transformers d56c55b
# src/transformers/models/mamba/configuration_mamba.py).
# ===========================================================================

comptime D_STATE = 16
comptime D_CONV = 4
comptime EXPAND = 2
comptime RMS_EPS: Float32 = 1e-5


@fieldwise_init
struct MambaDims(Copyable, Movable):
    """One block's shape. `dt_rank = ceil(d_model / 16)` mirrors
    `configuration_mamba.py` (`time_step_rank = math.ceil(hidden_size / 16)`)."""

    var d_model: Int
    var d_inner: Int
    var dt_rank: Int

    @staticmethod
    def of(d_model: Int) -> Self:
        return Self(d_model, EXPAND * d_model, (d_model + 15) // 16)

    def x_proj_rows(self) -> Int:
        return self.dt_rank + 2 * D_STATE


struct MambaWeights(Copyable, Movable):
    """One block's parameters, host side, in the upstream shapes
    (`shapes_for` in `mamba/corpus/gen_corpus.py`), all row-major:

        norm_w   [d_model]
        w_in     [2*d_inner, d_model]     in_proj.weight   (bias: none, use_bias False)
        conv_w   [d_inner, d_conv]        conv1d.weight (their [d_inner,1,d_conv], the 1 dropped)
        conv_b   [d_inner]                conv1d.bias      (use_conv_bias True)
        w_x      [dt_rank+2*d_state, d_inner]  x_proj.weight (bias: none)
        w_dt     [d_inner, dt_rank]       dt_proj.weight
        b_dt     [d_inner]                dt_proj.bias
        a_log    [d_inner, d_state]       A_log
        d_skip   [d_inner]                D
        w_out    [d_model, d_inner]       out_proj.weight  (bias: none)
    """

    var dims: MambaDims
    var norm_w: List[Float32]
    var w_in: List[Float32]
    var conv_w: List[Float32]
    var conv_b: List[Float32]
    var w_x: List[Float32]
    var w_dt: List[Float32]
    var b_dt: List[Float32]
    var a_log: List[Float32]
    var d_skip: List[Float32]
    var w_out: List[Float32]

    def __init__(out self, dims: MambaDims):
        self.dims = dims.copy()
        self.norm_w = List[Float32]()
        self.w_in = List[Float32]()
        self.conv_w = List[Float32]()
        self.conv_b = List[Float32]()
        self.w_x = List[Float32]()
        self.w_dt = List[Float32]()
        self.b_dt = List[Float32]()
        self.a_log = List[Float32]()
        self.d_skip = List[Float32]()
        self.w_out = List[Float32]()


# ===========================================================================
# The corpus hash, spec `mojolearn.mamba.corpus.hash.v1`
# (mamba/corpus/gen_corpus.py::splitmix64_scalar / hashed_unit / map_range).
# ===========================================================================


def corpus_splitmix64(z_in: UInt64) -> UInt64:
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def corpus_tensor(
    seed: UInt64, tensor_id: Int, n: Int, lo: Float64, hi: Float64
) -> List[Float32]:
    """`n` values `f32(lo + (hi-lo) * top24 * 2^-24)`. The float64
    evaluation is exact for the corpus's dyadic (lo, hi) -- gen_corpus.py
    asserts that element by element with exact rationals -- so the ONE
    rounding is the final cast, and this reimplementation is bit-equal to
    the files by construction (and gated, not assumed:
    `check_corpus_generator_agrees`)."""
    var key = corpus_splitmix64(seed ^ (UInt64(tensor_id) << 32))
    var span = hi - lo
    var out = List[Float32]()
    for i in range(n):
        var h = corpus_splitmix64(key + UInt64(i))
        var unit = Float64(Int(h >> 40)) * 0.000000059604644775390625  # 2^-24
        out.append(Float32(lo + span * unit))
    return out^


# Tensor ids, gen_corpus.py::TENSOR_IDS.
comptime TID_X = 1
comptime TID_W_IN = 2
comptime TID_CONV_W = 3
comptime TID_CONV_B = 4
comptime TID_W_X = 5
comptime TID_W_DT = 6
comptime TID_B_DT = 7
comptime TID_A_LOG = 8
comptime TID_D = 9
comptime TID_W_OUT = 10
comptime TID_NORM_W = 11

comptime CORPUS_SEED_BASE: UInt64 = 0x4D616D6261436F72
"""gen_corpus.py::SEED_BASE; case k's seed is BASE + 0x1000 * k."""


def corpus_case_seed(k: Int) -> UInt64:
    return CORPUS_SEED_BASE + UInt64(0x1000) * UInt64(k)


def fan_in_scale(fan_in: Int) -> Float64:
    """gen_corpus.py::fan_in_scale: 0.5 / 2^ceil(log2(fan_in)/2), a dyadic
    stand-in for 0.5/sqrt(fan_in). fan_in here is always a power of four or
    twice one (8, 16, 32), so ceil(log2/2) is 2 for 8..16 and 3 for 17..64."""
    var log2 = 0
    var v = 1
    while v < fan_in:
        v *= 2
        log2 += 1
    var half_up = (log2 + 1) // 2
    var denom = 1
    for _ in range(half_up):
        denom *= 2
    return 0.5 / Float64(denom)


def corpus_weights(seed: UInt64, dims: MambaDims) -> MambaWeights:
    """The DEFAULT ranges (gen_corpus.py::default_ranges). Adversarial
    overrides are separate functions below, one per corpus case, so a
    range can never silently drift from the case that motivated it."""
    var w = MambaWeights(dims)
    var s_in = fan_in_scale(dims.d_model)
    var s_x = fan_in_scale(dims.d_inner)
    w.norm_w = corpus_tensor(seed, TID_NORM_W, dims.d_model, 0.5, 1.5)
    w.w_in = corpus_tensor(
        seed, TID_W_IN, 2 * dims.d_inner * dims.d_model, -s_in, s_in
    )
    w.conv_w = corpus_tensor(seed, TID_CONV_W, dims.d_inner * D_CONV, -0.5, 0.5)
    w.conv_b = corpus_tensor(seed, TID_CONV_B, dims.d_inner, -0.125, 0.125)
    w.w_x = corpus_tensor(
        seed, TID_W_X, dims.x_proj_rows() * dims.d_inner, -s_x, s_x
    )
    w.w_dt = corpus_tensor(seed, TID_W_DT, dims.d_inner * dims.dt_rank, -1.0, 1.0)
    w.b_dt = corpus_tensor(seed, TID_B_DT, dims.d_inner, -7.0, -2.0)
    w.a_log = corpus_tensor(seed, TID_A_LOG, dims.d_inner * D_STATE, 0.0, 2.75)
    w.d_skip = corpus_tensor(seed, TID_D, dims.d_inner, 0.5, 1.5)
    w.w_out = corpus_tensor(
        seed, TID_W_OUT, dims.d_model * dims.d_inner, -s_out_scale(dims), s_out_scale(dims)
    )
    return w^


def s_out_scale(dims: MambaDims) -> Float64:
    return fan_in_scale(dims.d_inner)


def corpus_x(seed: UInt64, b: Int, l: Int, d_model: Int) -> List[Float32]:
    return corpus_tensor(seed, TID_X, b * l * d_model, -2.0, 2.0)


# ===========================================================================
# The corpus CASE TABLE (gen_corpus.py::CASES, 16 cases), reproduced so the
# lane can regenerate any case without the directory, and so
# `check_corpus_generator_agrees` can catch either side drifting.
# ===========================================================================

comptime CORPUS_CASE_COUNT = 16


@fieldwise_init
struct CorpusCase(Copyable, Movable):
    var name: StaticString
    var b: Int
    var l: Int
    var d_model: Int
    var seed_index: Int
    """The index whose seed generates the tensors (differs from the case's
    own index only for the comp_row slices, which use the parent's)."""
    var x_row: Int
    """-1: x is the case's own tensor. >= 0: x is that batch row of the
    parent case's x (comp_row0/comp_row1)."""


def corpus_case(k: Int) raises -> CorpusCase:
    if k == 0:
        return CorpusCase("base_b1_l1_d8", 1, 1, 8, 0, -1)
    if k == 1:
        return CorpusCase("base_b2_l4_d8", 2, 4, 8, 1, -1)
    if k == 2:
        return CorpusCase("base_b3_l16_d8", 3, 16, 8, 2, -1)
    if k == 3:
        return CorpusCase("base_b1_l64_d8", 1, 64, 8, 3, -1)
    if k == 4:
        return CorpusCase("base_b1_l1_d16", 1, 1, 16, 4, -1)
    if k == 5:
        return CorpusCase("base_b3_l4_d16", 3, 4, 16, 5, -1)
    if k == 6:
        return CorpusCase("base_b2_l16_d16", 2, 16, 16, 6, -1)
    if k == 7:
        return CorpusCase("base_b1_l64_d16", 1, 64, 16, 7, -1)
    if k == 8:
        return CorpusCase("adv_softplus_guard_b2_l8_d8", 2, 8, 8, 8, -1)
    if k == 9:
        return CorpusCase("adv_a_very_negative_b1_l16_d16", 1, 16, 16, 9, -1)
    if k == 10:
        return CorpusCase("adv_a_near_zero_b3_l8_d8", 3, 8, 8, 10, -1)
    if k == 11:
        return CorpusCase("adv_signed_zeros_b2_l8_d8", 2, 8, 8, 11, -1)
    if k == 12:
        return CorpusCase("adv_gate_saturation_b1_l8_d16", 1, 8, 16, 12, -1)
    if k == 13:
        return CorpusCase("comp_b2_l257_d8", 2, 257, 8, 13, -1)
    if k == 14:
        return CorpusCase("comp_row0_b1_l257_d8", 1, 257, 8, 13, 0)
    if k == 15:
        return CorpusCase("comp_row1_b1_l257_d8", 1, 257, 8, 13, 1)
    raise Error("corpus_case: no case " + String(k))


def corpus_case_weights(k: Int) raises -> MambaWeights:
    """The case's parameters, overrides applied (gen_corpus.py's
    `overrides` / `segments` entries, one branch per adversarial case)."""
    var c = corpus_case(k)
    var dims = MambaDims.of(c.d_model)
    var seed = corpus_case_seed(c.seed_index)
    var w = corpus_weights(seed, dims)
    if k == 8:  # dt_proj.bias straddles the softplus threshold 20
        w.b_dt = corpus_tensor(seed, TID_B_DT, dims.d_inner, 19.875, 20.125)
    if k == 9:  # A in [-22026, -403]: exp(delta*A) crosses the denormal band
        w.a_log = corpus_tensor(
            seed, TID_A_LOG, dims.d_inner * D_STATE, 6.0, 10.0
        )
    if k == 10:  # A within ulps of 0: exp(delta*A) rounds to exactly 1
        w.a_log = corpus_tensor(
            seed, TID_A_LOG, dims.d_inner * D_STATE, -18.0, -12.0
        )
    if k == 12:  # the gate half of in_proj.weight in [-2^30, 2^30]
        var s_in = fan_in_scale(dims.d_model)
        var n = 2 * dims.d_inner * dims.d_model
        var half = dims.d_inner * dims.d_model
        var key = corpus_splitmix64(seed ^ (UInt64(TID_W_IN) << 32))
        var out = List[Float32]()
        for i in range(n):
            var h = corpus_splitmix64(key + UInt64(i))
            var unit = Float64(Int(h >> 40)) * 0.000000059604644775390625
            if i < half:
                out.append(Float32(-s_in + (2.0 * s_in) * unit))
            else:
                out.append(
                    Float32(-1073741824.0 + 2147483648.0 * unit)
                )
        w.w_in = out^
    return w^


def corpus_case_x(k: Int) raises -> List[Float32]:
    """The case's input, zero rule / parent slice applied."""
    var c = corpus_case(k)
    var seed = corpus_case_seed(c.seed_index)
    if c.x_row >= 0:
        var parent = corpus_case(c.seed_index)
        var px = corpus_x(seed, parent.b, parent.l, parent.d_model)
        var out = List[Float32]()
        var row_len = parent.l * parent.d_model
        for i in range(row_len):
            out.append(px[c.x_row * row_len + i])
        return out^
    var x = corpus_x(seed, c.b, c.l, c.d_model)
    if k == 11:
        # gen_corpus.py::apply_zero_rule: whole +0.0 tokens (t%4==0), whole
        # -0.0 tokens (t%4==2), and of the remaining elements flat i%7==3
        # is -0.0
        var d = c.d_model
        for bb in range(c.b):
            for t in range(c.l):
                if t % 4 == 0 or t % 4 == 2:
                    var base = (bb * c.l + t) * d
                    for j in range(d):
                        if t % 4 == 0:
                            x[base + j] = f32_from_bits(UInt32(0))
                        else:
                            x[base + j] = f32_from_bits(BITS_NEG_ZERO)
        for i in range(len(x)):
            var t = (i // d) % c.l
            if t % 4 != 0 and t % 4 != 2 and i % 7 == 3:
                x[i] = f32_from_bits(BITS_NEG_ZERO)
        return x^
    return x^


# ===========================================================================
# Bit plants (row 39, denormal seams)
# ===========================================================================


def f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_POS_INF: UInt32 = 0x7F800000
comptime BITS_QNAN: UInt32 = 0x7FC00000
comptime BITS_MIN_SUBNORMAL: UInt32 = 0x00000001
comptime BITS_BIG_SUBNORMAL: UInt32 = 0x007FFFFF  # largest subnormal
comptime BITS_NEG_SUBNORMAL: UInt32 = 0x80000101


def plant_every_kth(
    mut values: List[Float32], k: Int, phase: Int, bits: UInt32
):
    """Overwrite every k-th element (flat index i with i % k == phase)."""
    for i in range(len(values)):
        if i % k == phase:
            values[i] = f32_from_bits(bits)


def bits32_hex(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


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
